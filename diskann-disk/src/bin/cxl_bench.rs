//! CXL Smoke Test & Benchmark
//!
//! Two modes:
//!   --mode disk   (default) Option 3: mmap disk index, test CxlMmapReader + CxlVertexProvider
//!   --mode numa              Option 2: extract from disk index, place vectors+graph on CXL NUMA,
//!                                      PQ on local DRAM, compare latency
//!
//! Usage:
//!   # Option 3 — DRAM baseline
//!   ./cxl_bench --mode disk --index <disk.index> --queries <queries.fbin>
//!
//!   # Option 3 — on CXL devdax
//!   sudo ./cxl_bench --mode disk --device /dev/dax1.0 --skip-copy --index <disk.index> --queries <queries.fbin>
//!
//!   # Option 2 — NUMA split, single CXL node
//!   ./cxl_bench --mode numa --local-node 0 --cxl-nodes 3 --index <disk.index> --queries <queries.fbin>
//!
//!   # Option 2 — NUMA split, multiple CXL nodes (interleaved)
//!   ./cxl_bench --mode numa --local-node 0 --cxl-nodes 3,4 --index <disk.index> --queries <queries.fbin>
//!   ./cxl_bench --mode numa --local-node 0 --cxl-nodes 3-5 --index <disk.index> --queries <queries.fbin>
//!   ./cxl_bench --mode numa --local-node 0 --cxl-nodes 3-4,6 --index <disk.index> --queries <queries.fbin>

#[cfg(feature = "cxl")]
mod bench {
    use std::path::Path;
    use std::time::Instant;

    use diskann_disk::utils::aligned_file_reader::cxl_mmap_reader::{CxlMmapConfig, CxlMmapReader};
    use diskann_disk::search::provider::cxl_sector_graph::NodeOffsetCalculator;
    use diskann_disk::search::provider::cxl_vertex_provider::{
        CxlVertexProvider, CxlVertexProviderConfig,
    };

    // ========================================================================
    // NUMA helpers
    // ========================================================================

    const MPOL_BIND: i32 = 2;
    const MPOL_INTERLEAVE: i32 = 3;

    /// Parse node spec: "2,3,4" or "2-4" or "2-4,6" or "2,4"
    fn parse_node_spec(spec: &str) -> Vec<u32> {
        let mut nodes = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.contains('-') {
                let bounds: Vec<&str> = part.split('-').collect();
                if bounds.len() == 2 {
                    let lo: u32 = bounds[0].trim().parse().expect("bad node range start");
                    let hi: u32 = bounds[1].trim().parse().expect("bad node range end");
                    for n in lo..=hi { nodes.push(n); }
                }
            } else {
                nodes.push(part.parse().expect("bad node number"));
            }
        }
        nodes.sort();
        nodes.dedup();
        assert!(!nodes.is_empty(), "empty node spec");
        nodes
    }

    /// Build nodemask bitmask from node list.
    fn nodes_to_mask(nodes: &[u32]) -> u64 {
        let mut mask: u64 = 0;
        for &n in nodes {
            assert!(n < 64, "NUMA node {} too high for 64-bit mask", n);
            mask |= 1 << n;
        }
        mask
    }

    struct NumaBuf {
        ptr: *mut u8,
        size: usize,
    }

    impl NumaBuf {
        /// Allocate on specific NUMA node(s) via mmap + mbind.
        /// Single node: MPOL_BIND. Multiple nodes: MPOL_INTERLEAVE.
        fn alloc(size: usize, nodes: &[u32]) -> Self {
            let size = size.max(4096);
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(), size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1, 0,
                )
            };
            assert!(ptr != libc::MAP_FAILED, "mmap of {} bytes failed", size);

            let nodemask = nodes_to_mask(nodes);
            let policy = if nodes.len() == 1 { MPOL_BIND } else { MPOL_INTERLEAVE };

            let ret = unsafe {
                libc::syscall(
                    libc::SYS_mbind,
                    ptr, size, policy,
                    &nodemask as *const u64, 64 as libc::c_ulong, 0 as libc::c_ulong,
                ) as libc::c_int
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                unsafe { libc::munmap(ptr, size); }
                panic!("mbind to NUMA node(s) {:?} failed: {} (errno {}). Run as root.",
                    nodes, err, err.raw_os_error().unwrap_or(-1));
            }

            // Touch pages to fault onto target node(s)
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }
            Self { ptr: ptr as *mut u8, size }
        }

        /// Allocate without NUMA binding — default policy (local DRAM).
        fn alloc_local(size: usize) -> Self {
            let size = size.max(4096);
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(), size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1, 0,
                )
            };
            assert!(ptr != libc::MAP_FAILED, "mmap of {} bytes failed", size);
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }
            Self { ptr: ptr as *mut u8, size }
        }

        fn as_ptr(&self) -> *const u8 { self.ptr }
        fn as_mut_ptr(&self) -> *mut u8 { self.ptr }
        #[allow(dead_code)]
        fn len(&self) -> usize { self.size }
    }

    impl Drop for NumaBuf {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size); }
        }
    }

    // ========================================================================
    // Header parsing
    // ========================================================================

    struct ParsedHeader {
        num_pts: u64,
        dims: u64,
        medoid: u64,
        node_len: u64,
        block_size: u64,
        associated_data_length: u64,
    }

    impl ParsedHeader {
        /// Compute max_degree given element size (1=int8, 4=float32).
        fn max_degree(&self, elem_size: usize) -> usize {
            let vector_len = self.dims as usize * elem_size;
            let remaining = (self.node_len as usize)
                .saturating_sub(vector_len)
                .saturating_sub(self.associated_data_length as usize);
            (remaining / std::mem::size_of::<u32>()).saturating_sub(1)
        }

        /// Auto-detect element size from node_len and dims.
        /// Tries int8 (1) and float32 (4), picks whichever gives a sane max_degree.
        fn detect_elem_size(&self) -> usize {
            let deg_i8 = self.max_degree(1);
            let deg_f32 = self.max_degree(4);
            if deg_f32 > 0 && deg_f32 < 1024 {
                4
            } else if deg_i8 > 0 && deg_i8 < 1024 {
                1
            } else {
                eprintln!("WARNING: cannot auto-detect elem_size. node_len={}, dims={}. Assuming float32.",
                    self.node_len, self.dims);
                4
            }
        }
    }

    fn parse_disk_index_header(path: &str) -> ParsedHeader {
        use std::io::Read;
        let mut f = std::fs::File::open(path).expect("Cannot open index file");
        let mut buf = [0u8; 104];
        f.read_exact(&mut buf).expect("Cannot read index header");
        let r64 = |off: usize| u64::from_le_bytes(buf[off..off+8].try_into().unwrap());
        ParsedHeader {
            num_pts: r64(8), dims: r64(16), medoid: r64(24),
            node_len: r64(32), associated_data_length: r64(80), block_size: r64(88),
        }
    }

    fn read_fbin_queries(path: &Path) -> Vec<Vec<f32>> {
        use std::io::Read;
        let file_size = std::fs::metadata(path).expect("Cannot stat query file").len() as usize;
        let mut file = std::fs::File::open(path).expect("Cannot open query file");
        let mut hdr = [0u8; 8];
        file.read_exact(&mut hdr).unwrap();
        let npts = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let ndims = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;

        // Auto-detect element size from file size
        let data_bytes = file_size - 8;
        let bytes_per_vec = data_bytes / npts;
        let is_float32 = bytes_per_vec == ndims * 4;
        let is_int8 = bytes_per_vec == ndims;

        if is_float32 {
            println!("  Queries: {} points x {} dims (float32)", npts, ndims);
            let mut queries = Vec::with_capacity(npts);
            for _ in 0..npts {
                let mut buf = vec![0u8; ndims * 4];
                file.read_exact(&mut buf).unwrap();
                queries.push(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
            }
            queries
        } else if is_int8 {
            println!("  Queries: {} points x {} dims (int8 → float32)", npts, ndims);
            let mut queries = Vec::with_capacity(npts);
            for _ in 0..npts {
                let mut buf = vec![0u8; ndims];
                file.read_exact(&mut buf).unwrap();
                queries.push(buf.iter().map(|&b| b as i8 as f32).collect());
            }
            queries
        } else {
            panic!("Unknown query format: {} bytes for {} points x {} dims (expected {}B for f32 or {}B for i8)",
                data_bytes, npts, ndims, npts * ndims * 4, npts * ndims);
        }
    }

    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    // ========================================================================
    // Search helpers
    // ========================================================================

    struct SearchableIndex {
        vectors_ptr: *const u8,
        graph_ptr: *const u8,
        dims: usize,
        max_degree: usize,
        graph_node_stride: usize,
    }

    impl SearchableIndex {
        fn get_vector(&self, id: u32) -> &[f32] {
            unsafe {
                std::slice::from_raw_parts(
                    self.vectors_ptr.add(id as usize * self.dims * 4) as *const f32,
                    self.dims,
                )
            }
        }
        fn get_neighbors(&self, id: u32) -> &[u32] {
            unsafe {
                let base = self.graph_ptr.add(id as usize * self.graph_node_stride);
                let num_nbrs = (*(base as *const u32)) as usize;
                std::slice::from_raw_parts(base.add(4) as *const u32, num_nbrs.min(self.max_degree))
            }
        }
    }

    fn greedy_search_generic(
        index: &SearchableIndex, query: &[f32],
        k: usize, search_l: usize, num_points: usize,
    ) -> (Vec<u32>, u32) {
        use std::collections::{BinaryHeap, HashSet};
        use std::cmp::Ordering;
        #[derive(Clone)]
        struct Cand { id: u32, dist: f32 }
        impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
        impl Eq for Cand {}
        impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for Cand { fn cmp(&self, o: &Self) -> Ordering { o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) } }

        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut results: Vec<Cand> = Vec::new();
        heap.push(Cand { id: 0, dist: l2_distance(query, index.get_vector(0)) });
        let mut io_ops = 0u32;

        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            io_ops += 1;
            for &nbr in index.get_neighbors(cur.id) {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                io_ops += 1;
                heap.push(Cand { id: nbr, dist: l2_distance(query, index.get_vector(nbr)) });
            }
            results.push(cur);
        }
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        (results.iter().map(|c| c.id).collect(), io_ops)
    }

    // ========================================================================
    // Mode: disk (Option 3)
    // ========================================================================

    fn run_disk_mode(args: &BenchArgs, header: &ParsedHeader, num_points: usize, dims: usize, max_degree: usize, elem_size: usize) {
        if elem_size != 4 {
            eprintln!("WARNING: disk mode uses CxlVertexProvider which reads raw bytes as float32.");
            eprintln!("         For int8 data, use --mode numa instead (converts to float32 during extraction).");
        }
        let block_size = if args.block_size > 0 { args.block_size } else { header.block_size as usize };
        let data_start = if args.data_start > 0 { args.data_start } else { block_size };
        let src_vec_bytes = dims * elem_size;

        let cxl_path = if args.device.is_empty() {
            println!("\n[2] DRAM baseline (no --device)");
            args.index_path.clone()
        } else if args.skip_copy {
            println!("\n[2] Using device: {}", args.device);
            args.device.clone()
        } else {
            println!("\n[2] Copying index to {}", args.device);
            let sz = std::fs::metadata(&args.index_path).unwrap().len();
            std::process::Command::new("sudo")
                .args(["dd", &format!("if={}", args.index_path), &format!("of={}", args.device), "bs=4M",
                    &format!("count={}", (sz + 4*1024*1024 - 1) / (4*1024*1024))])
                .status().expect("dd failed");
            args.device.clone()
        };

        println!("\n[3] Opening mmap reader...");
        let t0 = Instant::now();
        let reader = CxlMmapReader::with_config(&cxl_path, CxlMmapConfig {
            prefault: true, advise_random: true, use_huge_pages: true,
        }).expect("mmap failed");
        let mapped_len = reader.mapped_len();
        println!("  Mapped {} bytes ({:.1} GB) in {:.1} ms",
            mapped_len, mapped_len as f64 / 1e9, t0.elapsed().as_secs_f64() * 1000.0);

        let offsets = NodeOffsetCalculator::new(block_size, src_vec_bytes, max_degree, data_start);
        println!("\n[4] Layout: raw={} eff={} efficiency={:.1}%",
            offsets.raw_node_len(), offsets.effective_node_len(), offsets.efficiency() * 100.0);

        let last = offsets.node_offset((num_points - 1) as u32) + offsets.raw_node_len();
        if last > mapped_len {
            eprintln!("  WARNING: last node at {} > file size {}", last, mapped_len);
        }

        let queries = load_queries(args, dims);
        let k = 10.min(num_points);
        let search_l = 50.min(num_points);

        println!("\n[5] Search benchmark (k={}, L={}, reps={}, queries={})",
            k, search_l, args.reps, queries.len());
        println!("\n  {:>10} {:>10} {:>10} {:>10} {:>10}", "prefetch", "avg_us", "p50_us", "p99_us", "avg_io");
        println!("  {}", "-".repeat(54));

        for &pf in &[0, 1, 2, 3, 4] {
            let provider = CxlVertexProvider::<f32>::new(
                CxlMmapReader::with_config(&cxl_path, CxlMmapConfig {
                    prefault: true, advise_random: true, use_huge_pages: true,
                }).unwrap(),
                NodeOffsetCalculator::new(block_size, src_vec_bytes, max_degree, data_start),
                dims, max_degree,
                CxlVertexProviderConfig { prefetch_ahead: pf, cache_enabled: pf > 0, cache_size: if pf > 0 { 64 } else { 0 } },
            );
            let (avg, p50, p99, avg_io) = bench_search_provider(&provider, &queries, k, search_l, num_points, args.reps);
            println!("  {:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1}", pf, avg, p50, p99, avg_io);
        }

        println!("\n[6] Prefetch: {} nodes", 1000.min(num_points));
        let n = 1000.min(num_points);
        let ns = offsets.raw_node_len();
        let t = Instant::now();
        for i in 0..n { reader.prefetch_node(offsets.node_offset(i as u32), ns); }
        println!("  {:.0} ns/node", t.elapsed().as_nanos() as f64 / n as f64);

        println!("\n[7] Random read latency");
        let nr = 10000.min(num_points * 100);
        let mut lats: Vec<f64> = (0..nr).filter_map(|i| {
            let off = offsets.node_offset((i % num_points) as u32);
            if off >= mapped_len { return None; }
            let t = Instant::now();
            unsafe { std::ptr::read_volatile(reader.ptr_at(off)); }
            Some(t.elapsed().as_nanos() as f64)
        }).collect();
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        if n > 0 {
            println!("  {} reads: avg={:.0}ns p50={:.0}ns p99={:.0}ns",
                n, lats.iter().sum::<f64>() / n as f64, lats[n/2], lats[((n as f64*0.99) as usize).min(n-1)]);
        }
    }

    fn bench_search_provider(
        provider: &CxlVertexProvider<f32>, queries: &[Vec<f32>],
        k: usize, search_l: usize, num_points: usize, reps: usize,
    ) -> (f64, f64, f64, f64) {
        let mut lats = Vec::new();
        let mut total_io = 0u64;
        for _ in 0..reps {
            for q in queries {
                let t = Instant::now();
                let (_, io) = greedy_search_via_provider(provider, q, k, search_l, num_points);
                lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                total_io += io as u64;
            }
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        (lats.iter().sum::<f64>() / n as f64, lats[n/2],
         lats[((n as f64*0.99) as usize).min(n-1)], total_io as f64 / n as f64)
    }

    fn greedy_search_via_provider(
        provider: &CxlVertexProvider<f32>, query: &[f32],
        k: usize, search_l: usize, num_points: usize,
    ) -> (Vec<u32>, u32) {
        use std::collections::{BinaryHeap, HashSet};
        use std::cmp::Ordering;
        #[derive(Clone)]
        struct Cand { id: u32, dist: f32 }
        impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
        impl Eq for Cand {}
        impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for Cand { fn cmp(&self, o: &Self) -> Ordering { o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) } }

        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut results: Vec<Cand> = Vec::new();
        heap.push(Cand { id: 0, dist: l2_distance(query, provider.get_vector(0)) });
        let mut io_ops = 0u32;

        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            io_ops += 1;
            for &nbr in provider.get_adjacency_list(cur.id) {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                io_ops += 1;
                heap.push(Cand { id: nbr, dist: l2_distance(query, provider.get_vector(nbr)) });
            }
            results.push(cur);
        }
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        (results.iter().map(|c| c.id).collect(), io_ops)
    }

    // ========================================================================
    // PQ structures (for Option 2 split search)
    // ========================================================================

    struct PqCodes { data: Vec<u8>, num_points: usize, num_chunks: usize }

    impl PqCodes {
        fn from_file(path: &Path) -> Self {
            use std::io::Read;
            let mut f = std::fs::File::open(path).expect("Cannot open PQ codes file");
            let mut hdr = [0u8; 8];
            f.read_exact(&mut hdr).unwrap();
            let num_points = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
            let num_chunks = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
            let mut data = vec![0u8; num_points * num_chunks];
            f.read_exact(&mut data).unwrap();
            println!("    PQ codes: {} points x {} chunks", num_points, num_chunks);
            Self { data, num_points, num_chunks }
        }
        #[inline]
        fn get(&self, point: usize) -> &[u8] {
            let s = point * self.num_chunks;
            &self.data[s..s + self.num_chunks]
        }
    }

    struct PqDistTable { table: Vec<f32>, num_chunks: usize }

    impl PqDistTable {
        fn build(query: &[f32], centroids: &PqCentroids) -> Self {
            let nc = centroids.num_chunks;
            let sd = centroids.sub_dim;
            let ncent = centroids.num_centroids;
            let mut table = vec![0.0f32; nc * ncent];
            for chunk in 0..nc {
                let q_sub = &query[chunk * sd..(chunk + 1) * sd];
                for c in 0..ncent {
                    let c_sub = centroids.get(chunk, c);
                    let mut d = 0.0f32;
                    for i in 0..sd { let diff = q_sub[i] - c_sub[i]; d += diff * diff; }
                    table[chunk * ncent + c] = d;
                }
            }
            Self { table, num_chunks: nc }
        }
        #[inline]
        fn distance(&self, codes: &[u8]) -> f32 {
            let mut d = 0.0f32;
            for chunk in 0..self.num_chunks { d += self.table[chunk * 256 + codes[chunk] as usize]; }
            d
        }
    }

    struct PqCentroids { data: Vec<f32>, num_chunks: usize, num_centroids: usize, sub_dim: usize }

    impl PqCentroids {
        fn derive(vectors_ptr: *const u8, dims: usize, pq_codes: &PqCodes, num_points: usize) -> Self {
            let nc = pq_codes.num_chunks;
            let sub_dim = dims / nc;
            let ncent = 256;
            let mut sums = vec![0.0f64; nc * ncent * sub_dim];
            let mut counts = vec![0u32; nc * ncent];
            for pt in 0..num_points {
                let codes = pq_codes.get(pt);
                let vec_ptr = unsafe { vectors_ptr.add(pt * dims * 4) as *const f32 };
                for chunk in 0..nc {
                    let c = codes[chunk] as usize;
                    counts[chunk * ncent + c] += 1;
                    let sum_base = (chunk * ncent + c) * sub_dim;
                    for d in 0..sub_dim {
                        sums[sum_base + d] += unsafe { *vec_ptr.add(chunk * sub_dim + d) } as f64;
                    }
                }
            }
            let mut data = vec![0.0f32; nc * ncent * sub_dim];
            for i in 0..nc * ncent {
                if counts[i] > 0 {
                    for d in 0..sub_dim {
                        data[i * sub_dim + d] = (sums[i * sub_dim + d] / counts[i] as f64) as f32;
                    }
                }
            }
            Self { data, num_chunks: nc, num_centroids: ncent, sub_dim }
        }
        #[inline]
        fn get(&self, chunk: usize, centroid: usize) -> &[f32] {
            let s = (chunk * self.num_centroids + centroid) * self.sub_dim;
            &self.data[s..s + self.sub_dim]
        }
    }

    fn pq_split_search(
        graph: &SearchableIndex, pq_codes_ptr: *const u8, num_chunks: usize,
        cxl_vectors_ptr: *const u8, dist_table: &PqDistTable,
        query: &[f32], dims: usize, k: usize, search_l: usize, num_points: usize,
    ) -> (Vec<u32>, u32, u32) {
        use std::collections::{BinaryHeap, HashSet};
        use std::cmp::Ordering;
        #[derive(Clone)]
        struct Cand { id: u32, dist: f32 }
        impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
        impl Eq for Cand {}
        impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for Cand { fn cmp(&self, o: &Self) -> Ordering { o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) } }

        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        let mut beam: Vec<Cand> = Vec::new();
        let mut pq_lookups = 0u32;
        let mut full_reads = 0u32;

        let codes_0 = unsafe { std::slice::from_raw_parts(pq_codes_ptr, num_chunks) };
        pq_lookups += 1;
        heap.push(Cand { id: 0, dist: dist_table.distance(codes_0) });

        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            beam.push(cur.clone());
            for &nbr in graph.get_neighbors(cur.id) {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                let codes = unsafe { std::slice::from_raw_parts(pq_codes_ptr.add(nbr as usize * num_chunks), num_chunks) };
                pq_lookups += 1;
                heap.push(Cand { id: nbr, dist: dist_table.distance(codes) });
            }
        }

        beam.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        let rerank_count = (k * 2).min(beam.len());
        let mut reranked: Vec<Cand> = beam[..rerank_count].iter().map(|c| {
            let vec = unsafe { std::slice::from_raw_parts(cxl_vectors_ptr.add(c.id as usize * dims * 4) as *const f32, dims) };
            full_reads += 1;
            Cand { id: c.id, dist: l2_distance(query, vec) }
        }).collect();
        reranked.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        reranked.truncate(k);
        (reranked.iter().map(|c| c.id).collect(), pq_lookups, full_reads)
    }

    // ========================================================================
    // Mode: numa (Option 2) — mmap extraction, multi-node CXL
    // ========================================================================

    fn run_numa_mode(args: &BenchArgs, header: &ParsedHeader, num_points: usize, dims: usize, max_degree: usize, elem_size: usize) {
        let local_node = args.local_node;
        let cxl_nodes = &args.cxl_nodes;
        let src_vec_bytes = dims * elem_size;
        let dst_vec_bytes = dims * 4; // always float32 in our buffers

        if cxl_nodes.len() == 1 {
            println!("\n[2] NUMA split: DRAM=node {}, CXL=node {}", local_node, cxl_nodes[0]);
        } else {
            println!("\n[2] NUMA split: DRAM=node {}, CXL=nodes {:?} (interleaved)", local_node, cxl_nodes);
        }

        // ── Load PQ codes ──
        let pq_path = if !args.pq_codes_path.is_empty() {
            args.pq_codes_path.clone()
        } else {
            let idx_dir = Path::new(&args.index_path).parent().unwrap_or(Path::new("."));
            let found = std::fs::read_dir(idx_dir).ok()
                .and_then(|entries| {
                    entries.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().contains("pq_compressed")))
                });
            match found {
                Some(p) => p.to_string_lossy().to_string(),
                None => {
                    eprintln!("Error: PQ codes file not found. Use --pq-codes PATH");
                    std::process::exit(1);
                }
            }
        };

        println!("\n[3] Loading PQ codes from: {}", pq_path);
        let pq_codes = PqCodes::from_file(Path::new(&pq_path));

        // ── mmap disk index (zero-copy, no 300GB Vec) ──
        println!("\n[4] Memory-mapping disk index (zero-copy)...");
        let t0 = Instant::now();
        let index_mmap = unsafe {
            let file = std::fs::File::open(&args.index_path).expect("Cannot open index");
            memmap2::Mmap::map(&file).expect("mmap failed")
        };
        let index_len = index_mmap.len();
        println!("  Mapped {:.1} GB in {:.1} ms", index_len as f64 / 1e9, t0.elapsed().as_secs_f64() * 1000.0);

        let block_size = header.block_size as usize;
        let data_start = block_size;
        let offsets = NodeOffsetCalculator::new(block_size, src_vec_bytes, max_degree, data_start);

        let vector_bytes = num_points * dst_vec_bytes; // float32 destination
        let graph_node_stride = 4 + max_degree * 4;
        let graph_bytes = num_points * graph_node_stride;
        let pq_bytes = pq_codes.num_points * pq_codes.num_chunks;

        println!("  Source vectors: {:.1} GB ({}), dest: {:.1} GB (float32), graph: {:.1} GB, PQ: {:.1} MB",
            (num_points * src_vec_bytes) as f64 / 1e9,
            if elem_size == 1 { "int8" } else { "float32" },
            vector_bytes as f64 / 1e9, graph_bytes as f64 / 1e9, pq_bytes as f64 / 1e6);

        // ── Allocate buffers ──
        println!("\n[5] Allocating buffers...");
        println!("    PQ codes ({:.0} MB) → DRAM (node {})", pq_bytes as f64 / 1e6, local_node);
        let dram_pq = NumaBuf::alloc_local(pq_bytes);
        unsafe { std::ptr::copy_nonoverlapping(pq_codes.data.as_ptr(), dram_pq.as_mut_ptr(), pq_bytes); }

        println!("    Vectors+graph ({:.1} GB) → DRAM baseline", (vector_bytes + graph_bytes) as f64 / 1e9);
        let dram_vectors = NumaBuf::alloc_local(vector_bytes);
        let dram_graph = NumaBuf::alloc_local(graph_bytes);

        println!("    Vectors+graph ({:.1} GB) → CXL node(s) {:?}", (vector_bytes + graph_bytes) as f64 / 1e9, cxl_nodes);
        let cxl_vectors = NumaBuf::alloc(vector_bytes, cxl_nodes);
        let cxl_graph = NumaBuf::alloc(graph_bytes, cxl_nodes);

        // ── Extract from mmap, converting to float32 if needed ──
        println!("    Extracting {} nodes...", num_points);
        let t0 = Instant::now();
        for i in 0..num_points {
            let src_offset = offsets.node_offset(i as u32);
            if src_offset + offsets.raw_node_len() > index_len { break; }

            let dst_vec_off = i * dst_vec_bytes;
            unsafe {
                let src = index_mmap.as_ptr().add(src_offset);

                if elem_size == 4 {
                    std::ptr::copy_nonoverlapping(src, dram_vectors.as_mut_ptr().add(dst_vec_off), dst_vec_bytes);
                    std::ptr::copy_nonoverlapping(src, cxl_vectors.as_mut_ptr().add(dst_vec_off), dst_vec_bytes);
                } else {
                    // int8 → float32
                    let src_bytes = std::slice::from_raw_parts(src, src_vec_bytes);
                    let dram_f = std::slice::from_raw_parts_mut(dram_vectors.as_mut_ptr().add(dst_vec_off) as *mut f32, dims);
                    let cxl_f = std::slice::from_raw_parts_mut(cxl_vectors.as_mut_ptr().add(dst_vec_off) as *mut f32, dims);
                    for d in 0..dims {
                        let val = src_bytes[d] as i8 as f32;
                        dram_f[d] = val;
                        cxl_f[d] = val;
                    }
                }

                let graph_src = src.add(src_vec_bytes);
                let graph_dst = i * graph_node_stride;
                let graph_len = (offsets.raw_node_len() - src_vec_bytes).min(graph_node_stride);
                std::ptr::copy_nonoverlapping(graph_src, dram_graph.as_mut_ptr().add(graph_dst), graph_len);
                std::ptr::copy_nonoverlapping(graph_src, cxl_graph.as_mut_ptr().add(graph_dst), graph_len);
            }

            if i > 0 && i % 10_000_000 == 0 {
                println!("      {:.0}M / {:.0}M nodes...", i as f64 / 1e6, num_points as f64 / 1e6);
            }
        }
        println!("    Done in {:.1}s", t0.elapsed().as_secs_f64());
        drop(index_mmap);

        // ── Derive PQ centroids ──
        println!("\n[6] Deriving PQ centroids...");
        let centroids = PqCentroids::derive(dram_vectors.as_ptr(), dims, &pq_codes, num_points);
        println!("    {} chunks x {} centroids x {} sub_dims", centroids.num_chunks, centroids.num_centroids, centroids.sub_dim);

        let queries = load_queries(args, dims);
        let k = 10.min(num_points);
        let search_l = 50.min(num_points);

        // ══════════════════════════════════════════════════════════════
        println!("\n[7] Search A: ALL on DRAM (full L2, best case)");
        let dram_idx = SearchableIndex { vectors_ptr: dram_vectors.as_ptr(), graph_ptr: dram_graph.as_ptr(), dims, max_degree, graph_node_stride };
        let (avg, p50, p99, avg_io) = bench_search_index(&dram_idx, &queries, k, search_l, num_points, args.reps);
        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us  avg_io={:.1}", avg, p50, p99, avg_io);

        // ══════════════════════════════════════════════════════════════
        println!("\n[8] Search B: ALL on CXL (full L2, worst case)");
        let cxl_idx = SearchableIndex { vectors_ptr: cxl_vectors.as_ptr(), graph_ptr: cxl_graph.as_ptr(), dims, max_degree, graph_node_stride };
        let (avg, p50, p99, avg_io) = bench_search_index(&cxl_idx, &queries, k, search_l, num_points, args.reps);
        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us  avg_io={:.1}", avg, p50, p99, avg_io);

        // ══════════════════════════════════════════════════════════════
        println!("\n[9] Search C: PQ SPLIT — PQ on DRAM, vectors+graph on CXL");
        println!("    Beam search → PQ codes from DRAM");
        println!("    Reranking → full vectors from CXL (top-2k only)\n");

        let mut lats = Vec::new();
        let mut total_pq = 0u64;
        let mut total_full = 0u64;
        for _ in 0..args.reps {
            for query in &queries {
                let dt = PqDistTable::build(query, &centroids);
                let t = Instant::now();
                let (_, pq_ops, full_ops) = pq_split_search(
                    &cxl_idx, dram_pq.as_ptr(), pq_codes.num_chunks,
                    cxl_vectors.as_ptr(), &dt, query, dims, k, search_l, num_points);
                lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                total_pq += pq_ops as u64;
                total_full += full_ops as u64;
            }
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us",
            lats.iter().sum::<f64>() / n as f64, lats[n/2], lats[((n as f64*0.99) as usize).min(n-1)]);
        println!("  avg PQ lookups (DRAM): {:.0}   avg full-vector reads (CXL): {:.0}",
            total_pq as f64 / n as f64, total_full as f64 / n as f64);

        // ── Raw latency comparison ──
        println!("\n[10] Raw read latency: DRAM vs CXL");
        for (label, buf, stride) in [("DRAM PQ", &dram_pq, pq_codes.num_chunks), ("CXL vectors", &cxl_vectors, dims * 4)] {
            let nr = 100000.min(num_points * 1000);
            let mut rl: Vec<f64> = (0..nr).map(|i| {
                let t = Instant::now();
                unsafe { std::ptr::read_volatile(buf.as_ptr().add((i % num_points) * stride)); }
                t.elapsed().as_nanos() as f64
            }).collect();
            rl.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rn = rl.len();
            println!("  {}: avg={:.0}ns  p50={:.0}ns  p99={:.0}ns", label,
                rl.iter().sum::<f64>() / rn as f64, rl[rn/2], rl[((rn as f64*0.99) as usize).min(rn-1)]);
        }
        println!("\n=== Done ===");
    }

    fn bench_search_index(
        index: &SearchableIndex, queries: &[Vec<f32>],
        k: usize, search_l: usize, num_points: usize, reps: usize,
    ) -> (f64, f64, f64, f64) {
        let mut lats = Vec::new();
        let mut total_io = 0u64;
        for _ in 0..reps { for q in queries {
            let t = Instant::now();
            let (_, io) = greedy_search_generic(index, q, k, search_l, num_points);
            lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
            total_io += io as u64;
        }}
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        (lats.iter().sum::<f64>() / n as f64, lats[n/2], lats[((n as f64*0.99) as usize).min(n-1)], total_io as f64 / n as f64)
    }

    fn load_queries(args: &BenchArgs, dims: usize) -> Vec<Vec<f32>> {
        if !args.queries_path.is_empty() {
            println!("    Loading queries: {}", args.queries_path);
            read_fbin_queries(Path::new(&args.queries_path))
        } else {
            println!("    No queries — using zero vectors");
            (0..10).map(|_| vec![0.0f32; dims]).collect()
        }
    }

    // ========================================================================
    // Entry point + arg parsing
    // ========================================================================

    pub fn run(args: BenchArgs) {
        println!("=== CXL DiskANN Benchmark ===\n");
        println!("[1] Index: {}", args.index_path);
        let header = parse_disk_index_header(&args.index_path);
        let num_points = if args.num_points > 0 { args.num_points } else { header.num_pts as usize };
        let dims = if args.dim > 0 { args.dim } else { header.dims as usize };
        let elem_size = header.detect_elem_size();
        let max_degree = if args.max_degree > 0 { args.max_degree } else { header.max_degree(elem_size) };
        let dtype = if elem_size == 1 { "int8" } else { "float32" };
        println!("  num_pts={}, dims={}, max_degree={}, node_len={}, block_size={}, medoid={}, dtype={}",
            num_points, dims, max_degree, header.node_len, header.block_size, header.medoid, dtype);
        match args.mode.as_str() {
            "disk" => run_disk_mode(&args, &header, num_points, dims, max_degree, elem_size),
            "numa" => run_numa_mode(&args, &header, num_points, dims, max_degree, elem_size),
            other => { eprintln!("Unknown mode '{}'. Use 'disk' or 'numa'.", other); std::process::exit(1); }
        }
    }

    pub struct BenchArgs {
        pub mode: String,
        pub device: String,
        pub index_path: String,
        pub queries_path: String,
        pub pq_codes_path: String,
        pub dim: usize,
        pub max_degree: usize,
        pub block_size: usize,
        pub data_start: usize,
        pub num_points: usize,
        pub reps: usize,
        pub skip_copy: bool,
        pub local_node: u32,
        pub cxl_nodes: Vec<u32>,
    }

    impl BenchArgs {
        pub fn from_env() -> Self {
            let args: Vec<String> = std::env::args().collect();
            let mut r = BenchArgs {
                mode: "disk".into(), device: String::new(),
                index_path: String::new(), queries_path: String::new(), pq_codes_path: String::new(),
                dim: 0, max_degree: 0, block_size: 0, data_start: 0, num_points: 0,
                reps: 5, skip_copy: false, local_node: 0, cxl_nodes: vec![2],
            };
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--mode"       => { i += 1; r.mode = args[i].clone(); }
                    "--device"     => { i += 1; r.device = args[i].clone(); }
                    "--index"      => { i += 1; r.index_path = args[i].clone(); }
                    "--queries"    => { i += 1; r.queries_path = args[i].clone(); }
                    "--pq-codes"   => { i += 1; r.pq_codes_path = args[i].clone(); }
                    "--dim"        => { i += 1; r.dim = args[i].parse().expect("bad --dim"); }
                    "--max-degree" => { i += 1; r.max_degree = args[i].parse().expect("bad --max-degree"); }
                    "--block-size" => { i += 1; r.block_size = args[i].parse().expect("bad --block-size"); }
                    "--data-start" => { i += 1; r.data_start = args[i].parse().expect("bad --data-start"); }
                    "--num-points" => { i += 1; r.num_points = args[i].parse().expect("bad --num-points"); }
                    "--reps"       => { i += 1; r.reps = args[i].parse().expect("bad --reps"); }
                    "--skip-copy"  => { r.skip_copy = true; }
                    "--local-node" => { i += 1; r.local_node = args[i].parse().expect("bad --local-node"); }
                    "--cxl-nodes"  => { i += 1; r.cxl_nodes = parse_node_spec(&args[i]); }
                    "--help" | "-h" => {
                        eprintln!("Usage: cxl_bench [OPTIONS]\n");
                        eprintln!("Modes:");
                        eprintln!("  --mode disk    Option 3: mmap disk index (default)");
                        eprintln!("  --mode numa    Option 2: PQ-split NUMA benchmark\n");
                        eprintln!("Common:");
                        eprintln!("  --index PATH       Disk index file (required)");
                        eprintln!("  --queries PATH     Query file (.fbin)");
                        eprintln!("  --reps N           Repetitions (default: 5)");
                        eprintln!("  --dim/--max-degree/--num-points/--block-size/--data-start  (0=auto)\n");
                        eprintln!("Disk mode:");
                        eprintln!("  --device PATH      CXL device (e.g., /dev/dax1.0)");
                        eprintln!("  --skip-copy        Don't dd to device\n");
                        eprintln!("NUMA mode:");
                        eprintln!("  --local-node N     DRAM NUMA node (default: 0)");
                        eprintln!("  --cxl-nodes SPEC   CXL NUMA nodes (default: 2)");
                        eprintln!("                     Examples: 3  |  3,4  |  3-5  |  3-4,6");
                        eprintln!("  --pq-codes PATH    PQ codes file (auto-detected if omitted)");
                        std::process::exit(0);
                    }
                    other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
                }
                i += 1;
            }
            if r.index_path.is_empty() { eprintln!("Error: --index required. Use --help."); std::process::exit(1); }
            r
        }
    }
}

#[cfg(feature = "cxl")]
fn main() { bench::run(bench::BenchArgs::from_env()); }

#[cfg(not(feature = "cxl"))]
fn main() {
    eprintln!("CXL not enabled. Build with: cargo build --features diskann-disk/cxl");
    std::process::exit(1);
}
