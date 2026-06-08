//! CXL Smoke Test & Benchmark
//!
//! Two modes:
//!   --mode disk   (default) Option 3: mmap disk index, test CxlMmapReader + CxlVertexProvider
//!   --mode numa              Option 2: extract from disk index, place vectors+graph on CXL NUMA,
//!                                      PQ on local DRAM, compare latency
//!
//! Usage:
//!   # Option 3 — DRAM baseline
//!   cargo run --release -p diskann-disk --features diskann-disk/cxl --bin cxl_bench -- \
//!     --index test_data/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index \
//!     --queries test_data/disk_index_search/disk_index_sample_query_10pts.fbin
//!
//!   # Option 3 — on CXL devdax
//!   sudo cargo run --release -p diskann-disk --features diskann-disk/cxl --bin cxl_bench -- \
//!     --device /dev/dax1.0 --skip-copy \
//!     --index <index> --queries <queries>
//!
//!   # Option 2 — NUMA split (PQ→DRAM, vectors+graph→CXL)
//!   sudo cargo run --release -p diskann-disk --features diskann-disk/cxl --bin cxl_bench -- \
//!     --mode numa --local-node 0 --cxl-node 2 \
//!     --index <index> --queries <queries>

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
    // NUMA helpers (same as cxl_data_provider.rs)
    // ========================================================================

    const MPOL_BIND: i32 = 2;

    struct NumaBuf {
        ptr: *mut u8,
        size: usize,
    }

    impl NumaBuf {
        /// Allocate on a specific NUMA node via mmap + mbind.
        fn alloc(size: usize, numa_node: u32) -> Self {
            let size = size.max(4096); // at least one page
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(ptr != libc::MAP_FAILED, "mmap of {} bytes failed", size);

            // Set NUMA policy BEFORE touching pages — they'll fault onto the target node
            let nodemask: u64 = 1 << numa_node;
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_mbind,
                    ptr,
                    size,
                    MPOL_BIND,
                    &nodemask as *const u64,
                    64 as libc::c_ulong,
                    0 as libc::c_ulong, // no flags — just set policy, don't move
                ) as libc::c_int
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                unsafe { libc::munmap(ptr, size); }
                panic!("mbind to NUMA node {} failed: {} (errno {})",
                    numa_node, err, err.raw_os_error().unwrap_or(-1));
            }

            // Touch pages to fault them onto the target node
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }

            Self { ptr: ptr as *mut u8, size }
        }

        /// Allocate without NUMA binding — uses default policy (local DRAM).
        fn alloc_local(size: usize) -> Self {
            let size = size.max(4096);
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert!(ptr != libc::MAP_FAILED, "mmap of {} bytes failed", size);
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }
            Self { ptr: ptr as *mut u8, size }
        }

        fn as_ptr(&self) -> *const u8 { self.ptr }
        fn as_mut_ptr(&self) -> *mut u8 { self.ptr }
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
        fn max_degree_f32(&self) -> usize {
            let vector_len = self.dims as usize * std::mem::size_of::<f32>();
            let remaining = (self.node_len as usize)
                .saturating_sub(vector_len)
                .saturating_sub(self.associated_data_length as usize);
            (remaining / std::mem::size_of::<u32>()).saturating_sub(1)
        }
    }

    fn parse_disk_index_header(path: &str) -> ParsedHeader {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).expect("Cannot open index file");
        f.seek(SeekFrom::Start(0)).unwrap();
        let mut buf = [0u8; 104];
        f.read_exact(&mut buf).expect("Cannot read index header");
        let r64 = |off: usize| u64::from_le_bytes(buf[off..off+8].try_into().unwrap());
        ParsedHeader {
            num_pts: r64(8),
            dims: r64(16),
            medoid: r64(24),
            node_len: r64(32),
            associated_data_length: r64(80),
            block_size: r64(88),
        }
    }

    fn read_fbin_queries(path: &Path) -> Vec<Vec<f32>> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).expect("Cannot open query file");
        let mut hdr = [0u8; 8];
        file.read_exact(&mut hdr).unwrap();
        let npts = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let ndims = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        println!("  Queries: {} points, {} dims", npts, ndims);
        let mut queries = Vec::with_capacity(npts);
        for _ in 0..npts {
            let mut buf = vec![0u8; ndims * 4];
            file.read_exact(&mut buf).unwrap();
            queries.push(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        }
        queries
    }

    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    // ========================================================================
    // Greedy search — works for both modes
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
            let offset = id as usize * self.dims * 4;
            unsafe {
                std::slice::from_raw_parts(self.vectors_ptr.add(offset) as *const f32, self.dims)
            }
        }

        fn get_neighbors(&self, id: u32) -> &[u32] {
            let offset = id as usize * self.graph_node_stride;
            unsafe {
                let base = self.graph_ptr.add(offset);
                let num_nbrs = (*(base as *const u32)) as usize;
                let num_nbrs = num_nbrs.min(self.max_degree);
                std::slice::from_raw_parts(base.add(4) as *const u32, num_nbrs)
            }
        }
    }

    fn greedy_search_generic(
        index: &SearchableIndex,
        query: &[f32],
        k: usize,
        search_l: usize,
        num_points: usize,
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

        let d = l2_distance(query, index.get_vector(0));
        heap.push(Cand { id: 0, dist: d });
        let mut io_ops = 0u32;

        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            io_ops += 1;

            for &nbr in index.get_neighbors(cur.id) {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                let d = l2_distance(query, index.get_vector(nbr));
                io_ops += 1;
                heap.push(Cand { id: nbr, dist: d });
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

    fn run_disk_mode(args: &BenchArgs, header: &ParsedHeader, num_points: usize, dims: usize, max_degree: usize) {
        let block_size = if args.block_size > 0 { args.block_size } else { header.block_size as usize };
        let data_start = if args.data_start > 0 { args.data_start } else { block_size };

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
        println!("  Mapped {} bytes ({:.1} MB) in {:.1} ms",
            mapped_len, mapped_len as f64 / 1e6, t0.elapsed().as_secs_f64() * 1000.0);

        let offsets = NodeOffsetCalculator::new(block_size, dims * 4, max_degree, data_start);
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
                NodeOffsetCalculator::new(block_size, dims * 4, max_degree, data_start),
                dims, max_degree,
                CxlVertexProviderConfig { prefetch_ahead: pf, cache_enabled: pf > 0, cache_size: if pf > 0 { 64 } else { 0 } },
            );

            let (avg, p50, p99, avg_io) = bench_search_provider(&provider, &queries, k, search_l, num_points, args.reps);
            println!("  {:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1}", pf, avg, p50, p99, avg_io);
        }

        // Latency microbenchmarks
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
        provider: &CxlVertexProvider<f32>,
        queries: &[Vec<f32>],
        k: usize, search_l: usize, num_points: usize, reps: usize,
    ) -> (f64, f64, f64, f64) {
        // Wrap the provider in our generic search interface
        // We'll use the provider's methods directly instead
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
        provider: &CxlVertexProvider<f32>,
        query: &[f32], k: usize, search_l: usize, num_points: usize,
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
        let d = l2_distance(query, provider.get_vector(0));
        heap.push(Cand { id: 0, dist: d });
        let mut io_ops = 0u32;

        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            io_ops += 1;
            for &nbr in provider.get_adjacency_list(cur.id) {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                let d = l2_distance(query, provider.get_vector(nbr));
                io_ops += 1;
                heap.push(Cand { id: nbr, dist: d });
            }
            results.push(cur);
        }
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        (results.iter().map(|c| c.id).collect(), io_ops)
    }

    // ========================================================================
    // Mode: numa (Option 2) — PQ split: PQ→DRAM, vectors+graph→CXL
    // ========================================================================

    /// PQ codes for all points: pq_codes[point * num_chunks + chunk] = centroid_id (u8)
    struct PqCodes {
        data: Vec<u8>,
        num_points: usize,
        num_chunks: usize,
    }

    impl PqCodes {
        /// Read from DiskANN pq_compressed.bin format: [npts: u32][nchunks: u32][codes: u8*npts*nchunks]
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
            let start = point * self.num_chunks;
            &self.data[start..start + self.num_chunks]
        }
    }

    /// PQ distance table: for each chunk, for each centroid (256), the partial distance
    /// Table is computed per-query from centroids derived from the full vectors.
    struct PqDistTable {
        /// table[chunk * 256 + centroid_id] = partial squared L2 distance
        table: Vec<f32>,
        num_chunks: usize,
    }

    impl PqDistTable {
        /// Build distance table for a query given PQ centroids.
        /// centroids layout: [chunk][centroid_id][sub_dims] as f32
        fn build(query: &[f32], centroids: &PqCentroids) -> Self {
            let nc = centroids.num_chunks;
            let sd = centroids.sub_dim;
            let num_centroids = centroids.num_centroids;
            let mut table = vec![0.0f32; nc * num_centroids];

            for chunk in 0..nc {
                let q_start = chunk * sd;
                let q_sub = &query[q_start..q_start + sd];
                for c in 0..num_centroids {
                    let c_sub = centroids.get(chunk, c);
                    let mut dist = 0.0f32;
                    for d in 0..sd {
                        let diff = q_sub[d] - c_sub[d];
                        dist += diff * diff;
                    }
                    table[chunk * num_centroids + c] = dist;
                }
            }

            Self { table, num_chunks: nc }
        }

        /// Compute approximate PQ distance for a point using its codes.
        #[inline]
        fn distance(&self, codes: &[u8]) -> f32 {
            let mut dist = 0.0f32;
            // Unrolled for common case; compiler will vectorize
            for chunk in 0..self.num_chunks {
                dist += self.table[chunk * 256 + codes[chunk] as usize];
            }
            dist
        }
    }

    /// PQ centroids derived from the full vectors and assigned codes.
    struct PqCentroids {
        /// centroids[chunk][centroid_id][sub_dim]
        data: Vec<f32>,
        num_chunks: usize,
        num_centroids: usize,
        sub_dim: usize,
    }

    impl PqCentroids {
        /// Derive centroids from full vectors and PQ codes by averaging.
        fn derive(
            vectors_ptr: *const u8, dims: usize,
            pq_codes: &PqCodes, num_points: usize,
        ) -> Self {
            let nc = pq_codes.num_chunks;
            let sub_dim = dims / nc;
            let num_centroids = 256; // u8 codes → 256 possible centroids

            // Accumulate: sum + count per centroid
            let mut sums = vec![0.0f64; nc * num_centroids * sub_dim];
            let mut counts = vec![0u32; nc * num_centroids];

            for pt in 0..num_points {
                let codes = pq_codes.get(pt);
                let vec_offset = pt * dims * 4;
                let vec_ptr = unsafe { vectors_ptr.add(vec_offset) as *const f32 };

                for chunk in 0..nc {
                    let c = codes[chunk] as usize;
                    let v_start = chunk * sub_dim;
                    let sum_base = (chunk * num_centroids + c) * sub_dim;
                    counts[chunk * num_centroids + c] += 1;
                    for d in 0..sub_dim {
                        let val = unsafe { *vec_ptr.add(v_start + d) } as f64;
                        sums[sum_base + d] += val;
                    }
                }
            }

            // Average
            let mut data = vec![0.0f32; nc * num_centroids * sub_dim];
            for chunk in 0..nc {
                for c in 0..num_centroids {
                    let cnt = counts[chunk * num_centroids + c];
                    if cnt > 0 {
                        let sum_base = (chunk * num_centroids + c) * sub_dim;
                        let dst_base = (chunk * num_centroids + c) * sub_dim;
                        for d in 0..sub_dim {
                            data[dst_base + d] = (sums[sum_base + d] / cnt as f64) as f32;
                        }
                    }
                }
            }

            Self { data, num_chunks: nc, num_centroids, sub_dim }
        }

        #[inline]
        fn get(&self, chunk: usize, centroid: usize) -> &[f32] {
            let start = (chunk * self.num_centroids + centroid) * self.sub_dim;
            &self.data[start..start + self.sub_dim]
        }
    }

    /// Two-phase search: PQ beam traversal (DRAM) → full-vector rerank (CXL)
    fn pq_split_search(
        graph: &SearchableIndex,       // graph on CXL (but adjacency list is tiny)
        pq_codes_ptr: *const u8,       // PQ codes on DRAM
        num_chunks: usize,
        cxl_vectors_ptr: *const u8,    // full vectors on CXL (only for rerank)
        dist_table: &PqDistTable,
        query: &[f32],
        dims: usize,
        k: usize,
        search_l: usize,
        num_points: usize,
    ) -> (Vec<u32>, u32, u32) {
        // Returns (result_ids, pq_lookups_from_dram, full_vector_reads_from_cxl)
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

        // Start: PQ distance to medoid (DRAM access)
        let codes_0 = unsafe { std::slice::from_raw_parts(pq_codes_ptr, num_chunks) };
        let d = dist_table.distance(codes_0);
        pq_lookups += 1;
        heap.push(Cand { id: 0, dist: d });

        // Phase 1: Beam search using PQ distances (all from DRAM)
        while let Some(cur) = heap.pop() {
            if visited.contains(&cur.id) { continue; }
            if visited.len() >= search_l { break; }
            visited.insert(cur.id);
            beam.push(cur.clone());

            // Read adjacency list (from CXL, but it's small: ~max_degree * 4 bytes)
            let neighbors = graph.get_neighbors(cur.id);

            for &nbr in neighbors {
                if nbr as usize >= num_points || visited.contains(&nbr) { continue; }
                // PQ distance — reads PQ codes from DRAM
                let codes = unsafe {
                    std::slice::from_raw_parts(
                        pq_codes_ptr.add(nbr as usize * num_chunks),
                        num_chunks,
                    )
                };
                let d = dist_table.distance(codes);
                pq_lookups += 1;
                heap.push(Cand { id: nbr, dist: d });
            }
        }

        // Phase 2: Rerank top candidates with full vectors (CXL access)
        beam.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        let rerank_count = (k * 2).min(beam.len()); // rerank 2x candidates

        let mut reranked: Vec<Cand> = beam[..rerank_count].iter().map(|c| {
            let vec_ptr = unsafe { cxl_vectors_ptr.add(c.id as usize * dims * 4) as *const f32 };
            let vec = unsafe { std::slice::from_raw_parts(vec_ptr, dims) };
            let exact_dist = l2_distance(query, vec);
            full_reads += 1;
            Cand { id: c.id, dist: exact_dist }
        }).collect();

        reranked.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        reranked.truncate(k);

        (reranked.iter().map(|c| c.id).collect(), pq_lookups, full_reads)
    }

    fn run_numa_mode(args: &BenchArgs, header: &ParsedHeader, num_points: usize, dims: usize, max_degree: usize) {
        let local_node = args.local_node;
        let cxl_node = args.cxl_node;

        println!("\n[2] NUMA split: local DRAM=node {}, CXL=node {}", local_node, cxl_node);

        // ── Load PQ codes ──
        let pq_path = if !args.pq_codes_path.is_empty() {
            args.pq_codes_path.clone()
        } else {
            // Auto-detect: look for pq_compressed.bin next to the index
            let idx_dir = Path::new(&args.index_path).parent().unwrap_or(Path::new("."));
            let candidates = [
                "disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_compressed.bin",
                "pq_compressed.bin",
            ];
            let found = candidates.iter()
                .map(|c| idx_dir.join(c))
                .find(|p| p.exists());
            match found {
                Some(p) => p.to_string_lossy().to_string(),
                None => {
                    eprintln!("Error: PQ codes file not found. Use --pq-codes PATH");
                    eprintln!("  Looked in: {}", idx_dir.display());
                    std::process::exit(1);
                }
            }
        };

        println!("\n[3] Loading PQ codes from: {}", pq_path);
        let pq_codes = PqCodes::from_file(Path::new(&pq_path));

        // ── Extract vectors + graph from disk index ──
        println!("\n[4] Extracting vectors + graph from disk index...");
        let block_size = header.block_size as usize;
        let data_start = block_size;
        let offsets = NodeOffsetCalculator::new(block_size, dims * 4, max_degree, data_start);

        let index_data = std::fs::read(&args.index_path).expect("Cannot read index file");
        let vector_bytes = num_points * dims * 4;
        let graph_node_stride = 4 + max_degree * 4;
        let graph_bytes = num_points * graph_node_stride;
        let pq_bytes = pq_codes.num_points * pq_codes.num_chunks;

        println!("  Vectors: {:.1} MB (→CXL), Graph: {:.1} MB (→CXL), PQ codes: {:.1} MB (→DRAM)",
            vector_bytes as f64 / 1e6, graph_bytes as f64 / 1e6, pq_bytes as f64 / 1e6);

        // ── Allocate: PQ codes → DRAM, vectors+graph → DRAM (baseline) + CXL ──
        println!("\n[5] Allocating buffers...");
        println!("    PQ codes → DRAM (node {})", local_node);
        let dram_pq = NumaBuf::alloc_local(pq_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(pq_codes.data.as_ptr(), dram_pq.as_mut_ptr(), pq_bytes);
        }

        println!("    Vectors+graph → DRAM baseline (node {})", local_node);
        let dram_vectors = NumaBuf::alloc_local(vector_bytes);
        let dram_graph = NumaBuf::alloc_local(graph_bytes);

        println!("    Vectors+graph → CXL (node {})", cxl_node);
        let cxl_vectors = NumaBuf::alloc(vector_bytes, cxl_node);
        let cxl_graph = NumaBuf::alloc(graph_bytes, cxl_node);

        // Extract and copy to both tiers
        println!("    Extracting {} nodes...", num_points);
        for i in 0..num_points {
            let src_offset = offsets.node_offset(i as u32);
            if src_offset + offsets.raw_node_len() > index_data.len() { break; }

            let vec_src = &index_data[src_offset..src_offset + dims * 4];
            let vec_dst_off = i * dims * 4;
            unsafe {
                std::ptr::copy_nonoverlapping(vec_src.as_ptr(), dram_vectors.as_mut_ptr().add(vec_dst_off), dims * 4);
                std::ptr::copy_nonoverlapping(vec_src.as_ptr(), cxl_vectors.as_mut_ptr().add(vec_dst_off), dims * 4);
            }

            let graph_src_start = src_offset + dims * 4;
            let graph_copy_len = (offsets.raw_node_len() - dims * 4).min(graph_node_stride);
            let graph_dst_off = i * graph_node_stride;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    index_data[graph_src_start..].as_ptr(),
                    dram_graph.as_mut_ptr().add(graph_dst_off), graph_copy_len);
                std::ptr::copy_nonoverlapping(
                    index_data[graph_src_start..].as_ptr(),
                    cxl_graph.as_mut_ptr().add(graph_dst_off), graph_copy_len);
            }
        }

        // ── Derive PQ centroids from vectors + codes ──
        println!("\n[6] Deriving PQ centroids from vectors...");
        let centroids = PqCentroids::derive(dram_vectors.as_ptr(), dims, &pq_codes, num_points);
        println!("    {} chunks x {} centroids x {} sub_dims",
            centroids.num_chunks, centroids.num_centroids, centroids.sub_dim);

        let queries = load_queries(args, dims);
        let k = 10.min(num_points);
        let search_l = 50.min(num_points);

        // ══════════════════════════════════════════════════════════════════
        // Benchmark A: ALL on DRAM — full L2 search (baseline best-case)
        // ══════════════════════════════════════════════════════════════════
        println!("\n[7] Search A: ALL on DRAM (full L2, best case)");
        let dram_index = SearchableIndex {
            vectors_ptr: dram_vectors.as_ptr(), graph_ptr: dram_graph.as_ptr(),
            dims, max_degree, graph_node_stride,
        };
        let (avg, p50, p99, avg_io) = bench_search_index(&dram_index, &queries, k, search_l, num_points, args.reps);
        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us  avg_io={:.1}", avg, p50, p99, avg_io);

        // ══════════════════════════════════════════════════════════════════
        // Benchmark B: ALL on CXL — full L2 search (worst case)
        // ══════════════════════════════════════════════════════════════════
        println!("\n[8] Search B: ALL on CXL (full L2, worst case)");
        let cxl_index = SearchableIndex {
            vectors_ptr: cxl_vectors.as_ptr(), graph_ptr: cxl_graph.as_ptr(),
            dims, max_degree, graph_node_stride,
        };
        let (avg, p50, p99, avg_io) = bench_search_index(&cxl_index, &queries, k, search_l, num_points, args.reps);
        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us  avg_io={:.1}", avg, p50, p99, avg_io);

        // ══════════════════════════════════════════════════════════════════
        // Benchmark C: SPLIT — PQ beam (DRAM) + rerank with full vectors (CXL)
        // This is what CxlDataProvider does in production.
        // ══════════════════════════════════════════════════════════════════
        println!("\n[9] Search C: PQ SPLIT — PQ codes on DRAM, vectors on CXL");
        println!("    Beam search reads PQ codes from DRAM (~100ns)");
        println!("    Reranking reads full vectors from CXL (~300ns)");
        println!("    Only top-2k candidates touch CXL memory\n");

        let cxl_graph_index = SearchableIndex {
            vectors_ptr: cxl_vectors.as_ptr(), graph_ptr: cxl_graph.as_ptr(),
            dims, max_degree, graph_node_stride,
        };

        let mut lats = Vec::new();
        let mut total_pq_lookups = 0u64;
        let mut total_full_reads = 0u64;

        for _ in 0..args.reps {
            for query in &queries {
                let dist_table = PqDistTable::build(query, &centroids);
                let t = Instant::now();
                let (_ids, pq_ops, full_ops) = pq_split_search(
                    &cxl_graph_index,
                    dram_pq.as_ptr(),
                    pq_codes.num_chunks,
                    cxl_vectors.as_ptr(),
                    &dist_table,
                    query,
                    dims, k, search_l, num_points,
                );
                lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                total_pq_lookups += pq_ops as u64;
                total_full_reads += full_ops as u64;
            }
        }

        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        let avg = lats.iter().sum::<f64>() / n as f64;
        let p50 = lats[n / 2];
        let p99 = lats[((n as f64 * 0.99) as usize).min(n - 1)];
        let avg_pq = total_pq_lookups as f64 / n as f64;
        let avg_full = total_full_reads as f64 / n as f64;

        println!("  avg={:.1}us  p50={:.1}us  p99={:.1}us", avg, p50, p99);
        println!("  avg PQ lookups (DRAM): {:.0}   avg full-vector reads (CXL): {:.0}", avg_pq, avg_full);

        // ── Raw memory latency comparison ──
        println!("\n[10] Raw read latency: DRAM (PQ codes) vs CXL (vectors)");
        for (label, buf, stride) in [
            ("DRAM PQ codes", &dram_pq, pq_codes.num_chunks),
            ("CXL vectors", &cxl_vectors, dims * 4),
        ] {
            let nr = 100000.min(num_points * 1000);
            let mut read_lats: Vec<f64> = (0..nr).map(|i| {
                let off = (i % num_points) * stride;
                let t = Instant::now();
                unsafe { std::ptr::read_volatile(buf.as_ptr().add(off)); }
                t.elapsed().as_nanos() as f64
            }).collect();
            read_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rn = read_lats.len();
            println!("  {}: avg={:.0}ns  p50={:.0}ns  p99={:.0}ns",
                label,
                read_lats.iter().sum::<f64>() / rn as f64,
                read_lats[rn / 2],
                read_lats[((rn as f64 * 0.99) as usize).min(rn - 1)]);
        }

        println!("\n=== Done ===");
    }

    fn bench_search_index(
        index: &SearchableIndex, queries: &[Vec<f32>],
        k: usize, search_l: usize, num_points: usize, reps: usize,
    ) -> (f64, f64, f64, f64) {
        let mut lats = Vec::new();
        let mut total_io = 0u64;
        for _ in 0..reps {
            for q in queries {
                let t = Instant::now();
                let (_, io) = greedy_search_generic(index, q, k, search_l, num_points);
                lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                total_io += io as u64;
            }
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        (lats.iter().sum::<f64>() / n as f64, lats[n/2],
         lats[((n as f64*0.99) as usize).min(n-1)], total_io as f64 / n as f64)
    }

    // ========================================================================
    // Shared helpers
    // ========================================================================

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
    // Entry point
    // ========================================================================

    pub fn run(args: BenchArgs) {
        println!("=== CXL DiskANN Benchmark ===\n");

        println!("[1] Index: {}", args.index_path);
        let header = parse_disk_index_header(&args.index_path);

        let num_points = if args.num_points > 0 { args.num_points } else { header.num_pts as usize };
        let dims = if args.dim > 0 { args.dim } else { header.dims as usize };
        let max_degree = if args.max_degree > 0 { args.max_degree } else { header.max_degree_f32() };

        println!("  num_pts={}, dims={}, max_degree={}, node_len={}, block_size={}, medoid={}",
            num_points, dims, max_degree, header.node_len, header.block_size, header.medoid);

        match args.mode.as_str() {
            "disk" => run_disk_mode(&args, &header, num_points, dims, max_degree),
            "numa" => run_numa_mode(&args, &header, num_points, dims, max_degree),
            other => {
                eprintln!("Unknown mode '{}'. Use 'disk' or 'numa'.", other);
                std::process::exit(1);
            }
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
        pub cxl_node: u32,
    }

    impl BenchArgs {
        pub fn from_env() -> Self {
            let args: Vec<String> = std::env::args().collect();
            let mut r = BenchArgs {
                mode: "disk".into(),
                device: String::new(),
                index_path: String::new(),
                queries_path: String::new(),
                pq_codes_path: String::new(),
                dim: 0, max_degree: 0, block_size: 0, data_start: 0, num_points: 0,
                reps: 5, skip_copy: false,
                local_node: 0, cxl_node: 2,
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
                    "--cxl-node"   => { i += 1; r.cxl_node = args[i].parse().expect("bad --cxl-node"); }
                    "--help" | "-h" => {
                        eprintln!("Usage: cxl_bench [OPTIONS]");
                        eprintln!();
                        eprintln!("Modes:");
                        eprintln!("  --mode disk    Option 3: mmap disk index (default)");
                        eprintln!("  --mode numa    Option 2: PQ-split NUMA benchmark");
                        eprintln!();
                        eprintln!("Common:");
                        eprintln!("  --index PATH       Disk index file (required)");
                        eprintln!("  --queries PATH     Query file (.fbin)");
                        eprintln!("  --reps N           Repetitions (default: 5)");
                        eprintln!("  --dim/--max-degree/--num-points/--block-size/--data-start  Overrides (0=auto)");
                        eprintln!();
                        eprintln!("Disk mode:");
                        eprintln!("  --device PATH      CXL device (e.g., /dev/dax1.0)");
                        eprintln!("  --skip-copy        Don't dd to device");
                        eprintln!();
                        eprintln!("NUMA mode:");
                        eprintln!("  --local-node N     Local DRAM NUMA node (default: 0)");
                        eprintln!("  --cxl-node N       CXL NUMA node (default: 2)");
                        eprintln!("  --pq-codes PATH    PQ compressed codes file (auto-detected if omitted)");
                        std::process::exit(0);
                    }
                    other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
                }
                i += 1;
            }
            if r.index_path.is_empty() {
                eprintln!("Error: --index required. Use --help.");
                std::process::exit(1);
            }
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
