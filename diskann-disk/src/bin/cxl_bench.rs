//! CXL DiskANN Benchmark
//!
//! Modes:
//!   --mode disk   Option 3: mmap disk index via CxlMmapReader + CxlVertexProvider
//!   --mode numa   Option 2: PQ-split NUMA (PQ→DRAM, vectors+graph→CXL)
//!
//! Outputs: console table + JSON results file
//!
//! Examples:
//!   ./cxl_bench --mode numa --local-node 0 --cxl-nodes 3,4 \
//!     --index spacev_R32_disk.index --queries query.30K.i8bin \
//!     --pq-codes spacev_R32_pq_compressed.bin --groundtruth groundtruth.30K.i32bin \
//!     --output results.json

#[cfg(feature = "cxl")]
mod bench {
    use std::path::Path;
    use std::time::Instant;

    use diskann_disk::utils::aligned_file_reader::cxl_mmap_reader::{CxlMmapConfig, CxlMmapReader};
    use diskann_disk::search::provider::cxl_sector_graph::NodeOffsetCalculator;
    use diskann_disk::search::provider::cxl_vertex_provider::{
        CxlVertexProvider, CxlVertexProviderConfig,
    };

    // ====================================================================
    // NUMA helpers
    // ====================================================================
    const MPOL_BIND: i32 = 2;
    const MPOL_INTERLEAVE: i32 = 3;

    fn parse_node_spec(spec: &str) -> Vec<u32> {
        let mut nodes = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.contains('-') {
                let b: Vec<&str> = part.split('-').collect();
                if b.len() == 2 {
                    let lo: u32 = b[0].trim().parse().expect("bad range start");
                    let hi: u32 = b[1].trim().parse().expect("bad range end");
                    for n in lo..=hi { nodes.push(n); }
                }
            } else { nodes.push(part.parse().expect("bad node")); }
        }
        nodes.sort(); nodes.dedup();
        assert!(!nodes.is_empty(), "empty node spec");
        nodes
    }

    fn nodes_to_mask(nodes: &[u32]) -> u64 {
        let mut m: u64 = 0;
        for &n in nodes { assert!(n < 64); m |= 1 << n; }
        m
    }

    struct NumaBuf { ptr: *mut u8, size: usize }
    impl NumaBuf {
        fn alloc(size: usize, nodes: &[u32]) -> Self {
            let size = size.max(4096);
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ|libc::PROT_WRITE,
                libc::MAP_PRIVATE|libc::MAP_ANONYMOUS, -1, 0) };
            assert!(ptr != libc::MAP_FAILED, "mmap {} bytes failed", size);
            let mask = nodes_to_mask(nodes);
            let policy = if nodes.len() == 1 { MPOL_BIND } else { MPOL_INTERLEAVE };
            let ret = unsafe { libc::syscall(libc::SYS_mbind, ptr, size, policy,
                &mask as *const u64, 64 as libc::c_ulong, 0 as libc::c_ulong) as i32 };
            if ret != 0 { let e = std::io::Error::last_os_error(); unsafe { libc::munmap(ptr, size); }
                panic!("mbind to {:?} failed: {}", nodes, e); }
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }
            Self { ptr: ptr as *mut u8, size }
        }
        fn alloc_local(size: usize) -> Self {
            let size = size.max(4096);
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ|libc::PROT_WRITE,
                libc::MAP_PRIVATE|libc::MAP_ANONYMOUS, -1, 0) };
            assert!(ptr != libc::MAP_FAILED);
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size); }
            Self { ptr: ptr as *mut u8, size }
        }
        fn as_ptr(&self) -> *const u8 { self.ptr }
        fn as_mut_ptr(&self) -> *mut u8 { self.ptr }
        #[allow(dead_code)] fn len(&self) -> usize { self.size }
    }
    impl Drop for NumaBuf { fn drop(&mut self) { unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size); } } }

    // ====================================================================
    // Header parsing
    // ====================================================================
    struct ParsedHeader { num_pts: u64, dims: u64, medoid: u64, node_len: u64, block_size: u64, associated_data_length: u64 }
    impl ParsedHeader {
        fn max_degree(&self, es: usize) -> usize {
            let vl = self.dims as usize * es;
            let r = (self.node_len as usize).saturating_sub(vl).saturating_sub(self.associated_data_length as usize);
            (r / 4).saturating_sub(1)
        }
        fn detect_elem_size(&self) -> usize {
            let d4 = self.max_degree(4); let d1 = self.max_degree(1);
            if d4 > 0 && d4 < 1024 { 4 } else if d1 > 0 && d1 < 1024 { 1 } else { 4 }
        }
    }
    fn parse_disk_index_header(path: &str) -> ParsedHeader {
        use std::io::Read;
        let mut f = std::fs::File::open(path).expect("Cannot open index");
        let mut buf = [0u8; 104];
        f.read_exact(&mut buf).expect("Cannot read header");
        let r = |o: usize| u64::from_le_bytes(buf[o..o+8].try_into().unwrap());
        ParsedHeader { num_pts: r(8), dims: r(16), medoid: r(24), node_len: r(32), associated_data_length: r(80), block_size: r(88) }
    }

    // ====================================================================
    // File I/O
    // ====================================================================
    fn read_fbin_queries(path: &Path) -> Vec<Vec<f32>> {
        use std::io::Read;
        let fsz = std::fs::metadata(path).expect("stat query").len() as usize;
        let mut f = std::fs::File::open(path).expect("open query");
        let mut h = [0u8; 8]; f.read_exact(&mut h).unwrap();
        let n = u32::from_le_bytes(h[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
        let bpv = (fsz - 8) / n;
        if bpv == d * 4 {
            println!("    Queries: {} x {} (float32)", n, d);
            (0..n).map(|_| { let mut b = vec![0u8; d*4]; f.read_exact(&mut b).unwrap();
                b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }).collect()
        } else if bpv == d {
            println!("    Queries: {} x {} (int8→f32)", n, d);
            (0..n).map(|_| { let mut b = vec![0u8; d]; f.read_exact(&mut b).unwrap();
                b.iter().map(|&v| v as i8 as f32).collect() }).collect()
        } else { panic!("Unknown query format: {} bytes for {}x{}", fsz-8, n, d); }
    }

    /// Groundtruth: [npts: u32][k: u32][ids: i32 * npts * k]
    struct GroundTruth { ids: Vec<Vec<u32>>, k: usize }
    impl GroundTruth {
        fn load(path: &Path) -> Self {
            use std::io::Read;
            let mut f = std::fs::File::open(path).expect("Cannot open groundtruth");
            let mut h = [0u8; 8]; f.read_exact(&mut h).unwrap();
            let n = u32::from_le_bytes(h[0..4].try_into().unwrap()) as usize;
            let k = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
            println!("    Groundtruth: {} queries x {} neighbors", n, k);
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                let mut row = vec![0u8; k * 4]; f.read_exact(&mut row).unwrap();
                ids.push(row.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect());
            }
            Self { ids, k }
        }
        fn recall_at(&self, query_idx: usize, results: &[u32], at_k: usize) -> f64 {
            if query_idx >= self.ids.len() { return 0.0; }
            let gt: std::collections::HashSet<u32> = self.ids[query_idx].iter().take(at_k).copied().collect();
            let found = results.iter().take(at_k).filter(|id| gt.contains(id)).count();
            found as f64 / at_k as f64
        }
    }

    // ====================================================================
    // Metrics
    // ====================================================================
    #[derive(Clone)]
    struct SearchMetrics {
        mode: String, search_l: usize, recall_at_10: f64,
        qps: f64, avg_us: f64, p50_us: f64, p95_us: f64, p99_us: f64,
        avg_dist_comps: f64, avg_vec_reads: f64,
    }

    fn print_metrics_header() {
        println!("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
            "L", "recall@10", "QPS", "avg_us", "p50_us", "p95_us", "p99_us", "dist_comps", "vec_reads");
        println!("  {}", "-".repeat(100));
    }

    fn print_metrics_row(m: &SearchMetrics) {
        println!("  {:>6} {:>10.4} {:>10.0} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>10.1}",
            m.search_l, m.recall_at_10, m.qps, m.avg_us, m.p50_us, m.p95_us, m.p99_us,
            m.avg_dist_comps, m.avg_vec_reads);
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    fn write_json(path: &str, all_metrics: &[(String, Vec<SearchMetrics>)], header: &ParsedHeader,
                  num_points: usize, dims: usize, max_degree: usize, elem_size: usize) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).expect("Cannot create output JSON");
        writeln!(f, "{{").unwrap();
        writeln!(f, "  \"index\": {{").unwrap();
        writeln!(f, "    \"num_points\": {},", num_points).unwrap();
        writeln!(f, "    \"dims\": {},", dims).unwrap();
        writeln!(f, "    \"max_degree\": {},", max_degree).unwrap();
        writeln!(f, "    \"node_len\": {},", header.node_len).unwrap();
        writeln!(f, "    \"block_size\": {},", header.block_size).unwrap();
        writeln!(f, "    \"elem_size\": {},", elem_size).unwrap();
        writeln!(f, "    \"data_type\": \"{}\"", if elem_size == 1 { "int8" } else { "float32" }).unwrap();
        writeln!(f, "  }},").unwrap();
        writeln!(f, "  \"benchmarks\": [").unwrap();
        for (bi, (name, metrics)) in all_metrics.iter().enumerate() {
            writeln!(f, "    {{").unwrap();
            writeln!(f, "      \"name\": \"{}\",", name).unwrap();
            writeln!(f, "      \"results\": [").unwrap();
            for (mi, m) in metrics.iter().enumerate() {
                writeln!(f, "        {{").unwrap();
                writeln!(f, "          \"search_L\": {},", m.search_l).unwrap();
                writeln!(f, "          \"recall_at_10\": {:.6},", m.recall_at_10).unwrap();
                writeln!(f, "          \"qps\": {:.1},", m.qps).unwrap();
                writeln!(f, "          \"avg_us\": {:.2},", m.avg_us).unwrap();
                writeln!(f, "          \"p50_us\": {:.2},", m.p50_us).unwrap();
                writeln!(f, "          \"p95_us\": {:.2},", m.p95_us).unwrap();
                writeln!(f, "          \"p99_us\": {:.2},", m.p99_us).unwrap();
                writeln!(f, "          \"avg_dist_comps\": {:.1},", m.avg_dist_comps).unwrap();
                writeln!(f, "          \"avg_vec_reads\": {:.1}", m.avg_vec_reads).unwrap();
                write!(f, "        }}").unwrap();
                if mi + 1 < metrics.len() { writeln!(f, ",").unwrap(); } else { writeln!(f).unwrap(); }
            }
            writeln!(f, "      ]").unwrap();
            write!(f, "    }}").unwrap();
            if bi + 1 < all_metrics.len() { writeln!(f, ",").unwrap(); } else { writeln!(f).unwrap(); }
        }
        writeln!(f, "  ]").unwrap();
        writeln!(f, "}}").unwrap();
        println!("\n  Results written to: {}", path);
    }

    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    // ====================================================================
    // Search implementations — all return (result_ids, dist_comps, vec_reads)
    // ====================================================================
    struct SearchableIndex { vectors_ptr: *const u8, graph_ptr: *const u8, dims: usize, max_degree: usize, graph_stride: usize }
    impl SearchableIndex {
        fn get_vector(&self, id: u32) -> &[f32] {
            unsafe { std::slice::from_raw_parts(self.vectors_ptr.add(id as usize * self.dims * 4) as *const f32, self.dims) }
        }
        fn get_neighbors(&self, id: u32) -> &[u32] {
            unsafe { let b = self.graph_ptr.add(id as usize * self.graph_stride);
                let n = (*(b as *const u32)) as usize;
                std::slice::from_raw_parts(b.add(4) as *const u32, n.min(self.max_degree)) }
        }
    }

    /// Full L2 search. dist_comps == vec_reads (every distance reads a full vector).
    fn full_l2_search(idx: &SearchableIndex, query: &[f32], k: usize, search_l: usize, num_points: usize)
        -> (Vec<u32>, u32, u32)
    {
        use std::collections::{BinaryHeap, HashSet}; use std::cmp::Ordering;
        #[derive(Clone)] struct C { id: u32, d: f32 }
        impl PartialEq for C { fn eq(&self, o: &Self) -> bool { self.d == o.d } } impl Eq for C {}
        impl PartialOrd for C { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for C { fn cmp(&self, o: &Self) -> Ordering { o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal) } }

        let mut vis = HashSet::new(); let mut heap = BinaryHeap::new(); let mut res: Vec<C> = Vec::new();
        let mut ops = 0u32;
        heap.push(C { id: 0, d: l2_distance(query, idx.get_vector(0)) }); ops += 1;
        while let Some(c) = heap.pop() {
            if vis.contains(&c.id) { continue; }
            if vis.len() >= search_l { break; }
            vis.insert(c.id);
            for &nb in idx.get_neighbors(c.id) {
                if nb as usize >= num_points || vis.contains(&nb) { continue; }
                ops += 1; heap.push(C { id: nb, d: l2_distance(query, idx.get_vector(nb)) });
            }
            res.push(c);
        }
        res.sort_by(|a,b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal)); res.truncate(k);
        let ids: Vec<u32> = res.iter().map(|c| c.id).collect();
        (ids, ops, ops) // full L2: every dist comp reads a full vector
    }

    /// PQ split search: PQ beam (DRAM) + full-vector rerank (CXL).
    /// dist_comps = pq_lookups + full_reads. vec_reads = full_reads only.
    fn pq_split_search(
        graph: &SearchableIndex, pq_ptr: *const u8, nc: usize,
        vec_ptr: *const u8, dt: &PqDistTable, query: &[f32],
        dims: usize, k: usize, search_l: usize, np: usize,
    ) -> (Vec<u32>, u32, u32) {
        use std::collections::{BinaryHeap, HashSet}; use std::cmp::Ordering;
        #[derive(Clone)] struct C { id: u32, d: f32 }
        impl PartialEq for C { fn eq(&self, o: &Self) -> bool { self.d == o.d } } impl Eq for C {}
        impl PartialOrd for C { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for C { fn cmp(&self, o: &Self) -> Ordering { o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal) } }

        let mut vis = HashSet::new(); let mut heap = BinaryHeap::new(); let mut beam: Vec<C> = Vec::new();
        let mut pq_ops = 0u32; let mut full_ops = 0u32;
        let c0 = unsafe { std::slice::from_raw_parts(pq_ptr, nc) };
        pq_ops += 1; heap.push(C { id: 0, d: dt.distance(c0) });

        while let Some(c) = heap.pop() {
            if vis.contains(&c.id) { continue; }
            if vis.len() >= search_l { break; }
            vis.insert(c.id); beam.push(c.clone());
            for &nb in graph.get_neighbors(c.id) {
                if nb as usize >= np || vis.contains(&nb) { continue; }
                let codes = unsafe { std::slice::from_raw_parts(pq_ptr.add(nb as usize * nc), nc) };
                pq_ops += 1; heap.push(C { id: nb, d: dt.distance(codes) });
            }
        }
        beam.sort_by(|a,b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal));
        let rerank = (k * 2).min(beam.len());
        let mut rr: Vec<C> = beam[..rerank].iter().map(|c| {
            let v = unsafe { std::slice::from_raw_parts(vec_ptr.add(c.id as usize * dims * 4) as *const f32, dims) };
            full_ops += 1; C { id: c.id, d: l2_distance(query, v) }
        }).collect();
        rr.sort_by(|a,b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal)); rr.truncate(k);
        (rr.iter().map(|c| c.id).collect(), pq_ops + full_ops, full_ops)
    }

    // ====================================================================
    // PQ structures
    // ====================================================================
    struct PqCodes { data: Vec<u8>, num_points: usize, num_chunks: usize }
    impl PqCodes {
        fn from_file(path: &Path) -> Self {
            use std::io::Read;
            let mut f = std::fs::File::open(path).expect("open PQ codes");
            let mut h = [0u8; 8]; f.read_exact(&mut h).unwrap();
            let n = u32::from_le_bytes(h[0..4].try_into().unwrap()) as usize;
            let c = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
            let mut data = vec![0u8; n*c]; f.read_exact(&mut data).unwrap();
            println!("    PQ codes: {} x {}", n, c); Self { data, num_points: n, num_chunks: c }
        }
        #[inline] fn get(&self, p: usize) -> &[u8] { let s=p*self.num_chunks; &self.data[s..s+self.num_chunks] }
    }
    struct PqDistTable { table: Vec<f32>, num_chunks: usize }
    impl PqDistTable {
        fn build(query: &[f32], cent: &PqCentroids) -> Self {
            let (nc, sd, ncent) = (cent.num_chunks, cent.sub_dim, cent.num_centroids);
            let mut t = vec![0.0f32; nc*ncent];
            for ch in 0..nc { let q = &query[ch*sd..(ch+1)*sd];
                for c in 0..ncent { let cv = cent.get(ch, c);
                    t[ch*ncent+c] = (0..sd).map(|i| { let d=q[i]-cv[i]; d*d }).sum(); } }
            Self { table: t, num_chunks: nc }
        }
        #[inline] fn distance(&self, codes: &[u8]) -> f32 {
            (0..self.num_chunks).map(|i| self.table[i*256 + codes[i] as usize]).sum()
        }
    }
    struct PqCentroids { data: Vec<f32>, num_chunks: usize, num_centroids: usize, sub_dim: usize }
    impl PqCentroids {
        fn derive(vp: *const u8, dims: usize, pq: &PqCodes, np: usize) -> Self {
            let nc = pq.num_chunks; let sd = dims/nc; let ncent = 256;
            let mut sums = vec![0.0f64; nc*ncent*sd]; let mut counts = vec![0u32; nc*ncent];
            for pt in 0..np { let codes = pq.get(pt);
                let vptr = unsafe { vp.add(pt*dims*4) as *const f32 };
                for ch in 0..nc { let c = codes[ch] as usize; counts[ch*ncent+c] += 1;
                    let sb = (ch*ncent+c)*sd;
                    for d in 0..sd { sums[sb+d] += unsafe { *vptr.add(ch*sd+d) } as f64; } } }
            let mut data = vec![0.0f32; nc*ncent*sd];
            for i in 0..nc*ncent { if counts[i]>0 { for d in 0..sd { data[i*sd+d] = (sums[i*sd+d]/counts[i] as f64) as f32; } } }
            Self { data, num_chunks: nc, num_centroids: ncent, sub_dim: sd }
        }
        #[inline] fn get(&self, ch: usize, c: usize) -> &[f32] {
            let s=(ch*self.num_centroids+c)*self.sub_dim; &self.data[s..s+self.sub_dim]
        }
    }

    // ====================================================================
    // Sweep runner — runs search across L values, computes metrics
    // ====================================================================
    const SEARCH_L_VALUES: &[usize] = &[10, 20, 50, 100, 200];

    fn run_sweep<F>(
        name: &str, queries: &[Vec<f32>], gt: &Option<GroundTruth>,
        num_points: usize, reps: usize, search_fn: F,
    ) -> Vec<SearchMetrics>
    where F: Fn(&[f32], usize) -> (Vec<u32>, u32, u32) // query, search_l -> (ids, dist_comps, vec_reads)
    {
        println!("\n  === {} ===", name);
        print_metrics_header();
        let mut results = Vec::new();

        for &sl in SEARCH_L_VALUES {
            let sl = sl.min(num_points);
            let k = 10.min(num_points);
            let mut lats = Vec::new();
            let mut total_dc = 0u64; let mut total_vr = 0u64;
            let mut total_recall = 0.0f64; let mut recall_count = 0usize;

            for _ in 0..reps {
                for (qi, query) in queries.iter().enumerate() {
                    let t = Instant::now();
                    let (ids, dc, vr) = search_fn(query, sl);
                    lats.push(t.elapsed().as_nanos() as f64 / 1000.0);
                    total_dc += dc as u64; total_vr += vr as u64;
                    if let Some(ref g) = gt {
                        total_recall += g.recall_at(qi, &ids, k);
                        recall_count += 1;
                    }
                }
            }

            lats.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let n = lats.len();
            let avg = lats.iter().sum::<f64>() / n as f64;
            let m = SearchMetrics {
                mode: name.to_string(), search_l: sl,
                recall_at_10: if recall_count > 0 { total_recall / recall_count as f64 } else { -1.0 },
                qps: 1_000_000.0 / avg,
                avg_us: avg, p50_us: percentile(&lats, 0.50),
                p95_us: percentile(&lats, 0.95), p99_us: percentile(&lats, 0.99),
                avg_dist_comps: total_dc as f64 / n as f64,
                avg_vec_reads: total_vr as f64 / n as f64,
            };
            print_metrics_row(&m);
            results.push(m);
        }
        results
    }

    // ====================================================================
    // Multi-device mmap — round-robins node reads across devices
    // ====================================================================
    struct MultiMmap {
        ptrs: Vec<*const u8>,
        size: usize, // mapped size (same for all devices)
    }

    impl MultiMmap {
        fn open(paths: &[String], index_size: usize) -> Self {
            let align = 2 * 1024 * 1024; // 2MB devdax alignment
            let map_size = ((index_size + align - 1) / align) * align;

            let ptrs: Vec<*const u8> = paths.iter().map(|p| {
                let is_devdax = p.contains("/dev/dax");
                if is_devdax {
                    // devdax: use libc::mmap with MAP_SHARED
                    let fd = unsafe { libc::open(
                        std::ffi::CString::new(p.as_bytes()).unwrap().as_ptr(),
                        libc::O_RDONLY) };
                    assert!(fd >= 0, "Cannot open {}: {}", p, std::io::Error::last_os_error());
                    let ptr = unsafe { libc::mmap(
                        std::ptr::null_mut(), map_size,
                        libc::PROT_READ, libc::MAP_SHARED,
                        fd, 0) };
                    unsafe { libc::close(fd); }
                    assert!(ptr != libc::MAP_FAILED,
                        "mmap {} failed (size={:.1}GB): {}", p, map_size as f64/1e9, std::io::Error::last_os_error());
                    ptr as *const u8
                } else {
                    // Regular file: use memmap2
                    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("Cannot open {}: {}", p, e));
                    let m = unsafe { memmap2::Mmap::map(&f).unwrap_or_else(|e| panic!("mmap {}: {}", p, e)) };
                    let ptr = m.as_ptr();
                    std::mem::forget(m); // leak — lives for program duration
                    ptr
                }
            }).collect();

            Self { ptrs, size: map_size }
        }

        fn len(&self) -> usize { self.size }
        fn num_devices(&self) -> usize { self.ptrs.len() }

        #[inline(always)]
        unsafe fn ptr_at(&self, offset: usize, node_id: u32) -> *const u8 {
            let dev = node_id as usize % self.ptrs.len();
            self.ptrs[dev].add(offset)
        }
    }

    /// SearchableIndex backed by multi-device mmap — distributes reads across devices
    struct MultiDeviceIndex {
        multi: MultiMmap,
        offsets: NodeOffsetCalculator,
        dims: usize,
        max_degree: usize,
        src_vec_bytes: usize,
        elem_size: usize,
    }

    impl MultiDeviceIndex {
        /// Compute L2 distance directly from mmap, handling int8→f32 conversion inline.
        #[inline]
        fn distance_to(&self, id: u32, query: &[f32]) -> f32 {
            let off = self.offsets.node_offset(id);
            unsafe {
                let ptr = self.multi.ptr_at(off, id);
                if self.elem_size == 1 {
                    let mut sum = 0.0f32;
                    for i in 0..self.dims {
                        let v = *ptr.add(i) as i8 as f32;
                        let diff = query[i] - v;
                        sum += diff * diff;
                    }
                    sum
                } else {
                    let vec = std::slice::from_raw_parts(ptr as *const f32, self.dims);
                    l2_distance(query, vec)
                }
            }
        }
        fn get_neighbors(&self, id: u32) -> &[u32] {
            unsafe {
                let base = self.multi.ptr_at(self.offsets.node_offset(id) + self.src_vec_bytes, id);
                let n = (*(base as *const u32)) as usize;
                std::slice::from_raw_parts(base.add(4) as *const u32, n.min(self.max_degree))
            }
        }
    }

    fn multi_device_search(
        idx: &MultiDeviceIndex, query: &[f32], k: usize, sl: usize, np: usize,
    ) -> (Vec<u32>, u32, u32) {
        use std::collections::{BinaryHeap, HashSet}; use std::cmp::Ordering;
        #[derive(Clone)] struct C{id:u32,d:f32}
        impl PartialEq for C{fn eq(&self,o:&Self)->bool{self.d==o.d}} impl Eq for C{}
        impl PartialOrd for C{fn partial_cmp(&self,o:&Self)->Option<Ordering>{Some(self.cmp(o))}}
        impl Ord for C{fn cmp(&self,o:&Self)->Ordering{o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)}}

        let mut vis=HashSet::new(); let mut heap=BinaryHeap::new(); let mut res:Vec<C>=Vec::new();
        let mut ops=0u32;
        heap.push(C{id:0,d:idx.distance_to(0, query)}); ops+=1;
        while let Some(c)=heap.pop() {
            if vis.contains(&c.id){continue;} if vis.len()>=sl{break;} vis.insert(c.id);
            for &nb in idx.get_neighbors(c.id) {
                if nb as usize>=np||vis.contains(&nb){continue;}
                ops+=1; heap.push(C{id:nb,d:idx.distance_to(nb, query)});
            } res.push(c);
        }
        res.sort_by(|a,b|a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal)); res.truncate(k);
        (res.iter().map(|c|c.id).collect(), ops, ops)
    }

    // ====================================================================
    // Mode: disk (Option 3)
    // ====================================================================
    fn run_disk_mode(args: &BenchArgs, header: &ParsedHeader, np: usize, dims: usize, md: usize, es: usize) {
        let bs = if args.block_size > 0 { args.block_size } else { header.block_size as usize };
        let ds = if args.data_start > 0 { args.data_start } else { bs };
        let svb = dims * es;

        // Parse device list: comma-separated paths
        let devices: Vec<String> = if args.device.is_empty() {
            vec![args.index_path.clone()]
        } else {
            args.device.split(',').map(|s| s.trim().to_string()).collect()
        };
        let is_multi = devices.len() > 1;

        // Copy index to devices if needed
        if !args.device.is_empty() && !args.skip_copy {
            let sz = std::fs::metadata(&args.index_path).unwrap().len();
            for dev in &devices {
                println!("[2] Copying index to {}...", dev);
                std::process::Command::new("sudo").args(["dd", &format!("if={}",args.index_path),
                    &format!("of={}",dev), "bs=4M",
                    &format!("count={}",(sz+4*1024*1024-1)/(4*1024*1024))]).status().ok();
            }
        }

        if is_multi {
            println!("\n[2] Multi-device: {} devices", devices.len());
            for d in &devices { println!("    {}", d); }
        } else if args.device.is_empty() {
            println!("\n[2] DRAM baseline (no --device)");
        } else {
            println!("\n[2] Single device: {}", devices[0]);
        }

        let queries = load_queries(args, dims);
        let gt = load_groundtruth(args);
        let mut all: Vec<(String, Vec<SearchMetrics>)> = Vec::new();

        let is_devdax = devices.iter().any(|d| d.contains("/dev/dax"));

        if is_multi || is_devdax || es != 4 {
            // Use MultiMmap+MultiDeviceIndex path for:
            // - multiple devices, devdax devices, or int8 data (needs inline conversion)
            let index_size = std::fs::metadata(&args.index_path).expect("Cannot stat index").len() as usize;
            let path_desc = if is_devdax { "devdax MAP_SHARED" } else { "regular file" };
            println!("\n[3] Opening {} mmap(s) ({:.1} GB, {})...",
                devices.len(), index_size as f64 / 1e9, path_desc);
            let multi = MultiMmap::open(&devices, index_size);
            println!("  Mapped {} x {:.1} GB", multi.num_devices(), multi.len() as f64 / 1e9);

            let offsets = NodeOffsetCalculator::new(bs, svb, md, ds);
            let idx = MultiDeviceIndex { multi, offsets, dims, max_degree: md, src_vec_bytes: svb, elem_size: es };

            let label = format!("disk_{}dev", devices.len());
            let metrics = run_sweep(&label, &queries, &gt, np, args.reps,
                |q, sl| multi_device_search(&idx, q, 10.min(np), sl, np));
            all.push((label, metrics));
        } else {
            // Single device: use CxlVertexProvider with prefetch
            let cxl_path = &devices[0];
            println!("\n[3] Opening mmap...");
            let reader = CxlMmapReader::with_config(cxl_path, CxlMmapConfig {
                prefault: true, advise_random: true, use_huge_pages: true }).expect("mmap failed");
            println!("  Mapped {:.1} GB", reader.mapped_len() as f64 / 1e9);

            // Create provider once, use for all queries
            let offsets = NodeOffsetCalculator::new(bs, svb, md, ds);
            let provider = CxlVertexProvider::<f32>::new(
                CxlMmapReader::with_config(cxl_path, CxlMmapConfig {
                    prefault: true, advise_random: true, use_huge_pages: true }).unwrap(),
                offsets, dims, md,
                CxlVertexProviderConfig { prefetch_ahead: 3, cache_enabled: true, cache_size: 64 },
            );

            let metrics = run_sweep("disk_1dev", &queries, &gt, np, args.reps, |query, sl| {
                use std::collections::{BinaryHeap, HashSet}; use std::cmp::Ordering;
                #[derive(Clone)] struct C{id:u32,d:f32}
                impl PartialEq for C{fn eq(&self,o:&Self)->bool{self.d==o.d}} impl Eq for C{}
                impl PartialOrd for C{fn partial_cmp(&self,o:&Self)->Option<Ordering>{Some(self.cmp(o))}}
                impl Ord for C{fn cmp(&self,o:&Self)->Ordering{o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)}}
                let mut vis=HashSet::new(); let mut heap=BinaryHeap::new(); let mut res:Vec<C>=Vec::new();
                let mut ops=0u32; let k=10.min(np);
                heap.push(C{id:0,d:l2_distance(query,provider.get_vector(0))}); ops+=1;
                while let Some(c)=heap.pop() {
                    if vis.contains(&c.id){continue;} if vis.len()>=sl{break;} vis.insert(c.id);
                    for &nb in provider.get_adjacency_list(c.id) {
                        if nb as usize>=np||vis.contains(&nb){continue;}
                        ops+=1; heap.push(C{id:nb,d:l2_distance(query,provider.get_vector(nb))});
                    } res.push(c);
                }
                res.sort_by(|a,b|a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal)); res.truncate(k);
                (res.iter().map(|c|c.id).collect(), ops, ops)
            });
            all.push(("disk_1dev".into(), metrics));

            // Raw latency
            println!("\n  === Raw read latency ===");
            let offsets2 = NodeOffsetCalculator::new(bs, svb, md, ds);
            let mlen = reader.mapped_len();
            let nr = 100000.min(np*100);
            let mut rl: Vec<f64> = (0..nr).filter_map(|i| {
                let off = offsets2.node_offset((i%np) as u32);
                if off >= mlen { return None; }
                let t = Instant::now();
                unsafe { std::ptr::read_volatile(reader.ptr_at(off)); }
                Some(t.elapsed().as_nanos() as f64)
            }).collect();
            rl.sort_by(|a,b| a.partial_cmp(b).unwrap()); let n = rl.len();
            if n > 0 {
                println!("  {} reads: avg={:.0}ns p50={:.0}ns p95={:.0}ns p99={:.0}ns",
                    n, rl.iter().sum::<f64>()/n as f64, percentile(&rl,0.50), percentile(&rl,0.95), percentile(&rl,0.99));
            }
        }

        if !args.output_path.is_empty() { write_json(&args.output_path, &all, header, np, dims, md, es); }
    }

    // ====================================================================
    // Mode: numa (Option 2) — PQ split with sweep
    // ====================================================================
    fn run_numa_mode(args: &BenchArgs, header: &ParsedHeader, np: usize, dims: usize, md: usize, es: usize) {
        let cxl_nodes = &args.cxl_nodes;
        println!("\n[2] DRAM=node {}, CXL=nodes {:?}", args.local_node, cxl_nodes);

        // Load PQ codes
        let pq_path = if !args.pq_codes_path.is_empty() { args.pq_codes_path.clone() }
        else {
            let dir = Path::new(&args.index_path).parent().unwrap_or(Path::new("."));
            std::fs::read_dir(dir).ok().and_then(|e| e.filter_map(|e|e.ok()).map(|e|e.path())
                .find(|p| p.file_name().map_or(false,|n| n.to_string_lossy().contains("pq_compressed"))))
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| { eprintln!("PQ codes not found. Use --pq-codes"); std::process::exit(1); })
        };
        println!("\n[3] Loading PQ codes: {}", pq_path);
        let pq_codes = PqCodes::from_file(Path::new(&pq_path));

        // mmap disk index
        println!("\n[4] mmap disk index...");
        let mmap = unsafe { let f=std::fs::File::open(&args.index_path).expect("open"); memmap2::Mmap::map(&f).expect("mmap") };
        let mlen = mmap.len();
        println!("  Mapped {:.1} GB", mlen as f64 / 1e9);

        let bs = header.block_size as usize; let ds = bs;
        let svb = dims * es; let dvb = dims * 4;
        let offsets = NodeOffsetCalculator::new(bs, svb, md, ds);
        let vbytes = np * dvb; let gs = 4 + md * 4; let gbytes = np * gs;
        let pqbytes = pq_codes.num_points * pq_codes.num_chunks;

        println!("  Src vectors: {:.1}GB ({}), dst: {:.1}GB (f32), graph: {:.1}GB, PQ: {:.0}MB",
            (np*svb) as f64/1e9, if es==1{"i8"}else{"f32"}, vbytes as f64/1e9, gbytes as f64/1e9, pqbytes as f64/1e6);

        // Allocate
        println!("\n[5] Allocating...");
        let dram_pq = NumaBuf::alloc_local(pqbytes);
        unsafe { std::ptr::copy_nonoverlapping(pq_codes.data.as_ptr(), dram_pq.as_mut_ptr(), pqbytes); }
        let dram_vec = NumaBuf::alloc_local(vbytes);
        let dram_graph = NumaBuf::alloc_local(gbytes);
        let cxl_vec = NumaBuf::alloc(vbytes, cxl_nodes);
        let cxl_graph = NumaBuf::alloc(gbytes, cxl_nodes);

        // Extract with int8→f32 conversion
        println!("    Extracting {} nodes...", np);
        let t0 = Instant::now();
        for i in 0..np {
            let so = offsets.node_offset(i as u32);
            if so + offsets.raw_node_len() > mlen { break; }
            let dvo = i * dvb;
            unsafe {
                let src = mmap.as_ptr().add(so);
                if es == 4 {
                    std::ptr::copy_nonoverlapping(src, dram_vec.as_mut_ptr().add(dvo), dvb);
                    std::ptr::copy_nonoverlapping(src, cxl_vec.as_mut_ptr().add(dvo), dvb);
                } else {
                    let sb = std::slice::from_raw_parts(src, svb);
                    let df = std::slice::from_raw_parts_mut(dram_vec.as_mut_ptr().add(dvo) as *mut f32, dims);
                    let cf = std::slice::from_raw_parts_mut(cxl_vec.as_mut_ptr().add(dvo) as *mut f32, dims);
                    for d in 0..dims { let v = sb[d] as i8 as f32; df[d]=v; cf[d]=v; }
                }
                let gsrc = src.add(svb); let gd = i*gs;
                let gl = (offsets.raw_node_len()-svb).min(gs);
                std::ptr::copy_nonoverlapping(gsrc, dram_graph.as_mut_ptr().add(gd), gl);
                std::ptr::copy_nonoverlapping(gsrc, cxl_graph.as_mut_ptr().add(gd), gl);
            }
            if i>0 && i%10_000_000==0 { println!("      {}M / {}M", i/1_000_000, np/1_000_000); }
        }
        println!("    Done in {:.1}s", t0.elapsed().as_secs_f64());
        drop(mmap);

        // Derive PQ centroids
        println!("\n[6] Deriving PQ centroids...");
        let cent = PqCentroids::derive(dram_vec.as_ptr(), dims, &pq_codes, np);
        println!("    {}x{}x{}", cent.num_chunks, cent.num_centroids, cent.sub_dim);

        let queries = load_queries(args, dims);
        let gt = load_groundtruth(args);
        let mut all: Vec<(String, Vec<SearchMetrics>)> = Vec::new();

        // ─── A: ALL DRAM ───
        let dram_idx = SearchableIndex { vectors_ptr: dram_vec.as_ptr(), graph_ptr: dram_graph.as_ptr(), dims, max_degree: md, graph_stride: gs };
        let m = run_sweep("ALL_DRAM", &queries, &gt, np, args.reps,
            |q, sl| full_l2_search(&dram_idx, q, 10.min(np), sl, np));
        all.push(("ALL_DRAM".into(), m));

        // ─── B: ALL CXL ───
        let cxl_idx = SearchableIndex { vectors_ptr: cxl_vec.as_ptr(), graph_ptr: cxl_graph.as_ptr(), dims, max_degree: md, graph_stride: gs };
        let m = run_sweep("ALL_CXL", &queries, &gt, np, args.reps,
            |q, sl| full_l2_search(&cxl_idx, q, 10.min(np), sl, np));
        all.push(("ALL_CXL".into(), m));

        // ─── C: PQ SPLIT ───
        let m = run_sweep("PQ_SPLIT_DRAM_CXL", &queries, &gt, np, args.reps, |q, sl| {
            let dt = PqDistTable::build(q, &cent);
            pq_split_search(&cxl_idx, dram_pq.as_ptr(), pq_codes.num_chunks,
                cxl_vec.as_ptr(), &dt, q, dims, 10.min(np), sl, np)
        });
        all.push(("PQ_SPLIT_DRAM_CXL".into(), m));

        // ─── Raw latency ───
        println!("\n  === Raw read latency ===");
        for (label, buf, stride) in [("DRAM_PQ", &dram_pq, pq_codes.num_chunks), ("CXL_vectors", &cxl_vec, dvb)] {
            let nr = 100000.min(np*1000);
            let mut rl: Vec<f64> = (0..nr).map(|i| {
                let t=Instant::now(); unsafe{std::ptr::read_volatile(buf.as_ptr().add((i%np)*stride));}
                t.elapsed().as_nanos() as f64 }).collect();
            rl.sort_by(|a,b|a.partial_cmp(b).unwrap()); let n=rl.len();
            println!("  {}: avg={:.0}ns p50={:.0}ns p95={:.0}ns p99={:.0}ns", label,
                rl.iter().sum::<f64>()/n as f64, percentile(&rl,0.50), percentile(&rl,0.95), percentile(&rl,0.99));
        }

        if !args.output_path.is_empty() { write_json(&args.output_path, &all, header, np, dims, md, es); }
    }

    // ====================================================================
    // Helpers
    // ====================================================================
    fn load_queries(args: &BenchArgs, dims: usize) -> Vec<Vec<f32>> {
        if !args.queries_path.is_empty() { read_fbin_queries(Path::new(&args.queries_path)) }
        else { println!("    No queries — zero vectors"); (0..10).map(|_|vec![0.0f32;dims]).collect() }
    }
    fn load_groundtruth(args: &BenchArgs) -> Option<GroundTruth> {
        if !args.gt_path.is_empty() { Some(GroundTruth::load(Path::new(&args.gt_path))) } else {
            println!("    No groundtruth — recall will show -1"); None
        }
    }

    // ====================================================================
    // Entry + args
    // ====================================================================
    pub fn run(args: BenchArgs) {
        println!("=== CXL DiskANN Benchmark ===\n");
        println!("[1] Index: {}", args.index_path);
        let hdr = parse_disk_index_header(&args.index_path);
        let np = if args.num_points>0{args.num_points}else{hdr.num_pts as usize};
        let dims = if args.dim>0{args.dim}else{hdr.dims as usize};
        let es = hdr.detect_elem_size();
        let md = if args.max_degree>0{args.max_degree}else{hdr.max_degree(es)};
        println!("  n={}, d={}, R={}, node_len={}, block_size={}, medoid={}, dtype={}",
            np, dims, md, hdr.node_len, hdr.block_size, hdr.medoid, if es==1{"int8"}else{"f32"});
        match args.mode.as_str() {
            "disk" => run_disk_mode(&args, &hdr, np, dims, md, es),
            "numa" => run_numa_mode(&args, &hdr, np, dims, md, es),
            _ => { eprintln!("Unknown mode. Use disk or numa."); std::process::exit(1); }
        }
        println!("\n=== Done ===");
    }

    pub struct BenchArgs {
        pub mode: String, pub device: String, pub index_path: String,
        pub queries_path: String, pub pq_codes_path: String, pub gt_path: String,
        pub output_path: String,
        pub dim: usize, pub max_degree: usize, pub block_size: usize,
        pub data_start: usize, pub num_points: usize, pub reps: usize,
        pub skip_copy: bool, pub local_node: u32, pub cxl_nodes: Vec<u32>,
    }

    impl BenchArgs {
        pub fn from_env() -> Self {
            let a: Vec<String> = std::env::args().collect();
            let mut r = BenchArgs {
                mode:"disk".into(), device:String::new(), index_path:String::new(),
                queries_path:String::new(), pq_codes_path:String::new(), gt_path:String::new(),
                output_path:String::new(),
                dim:0, max_degree:0, block_size:0, data_start:0, num_points:0,
                reps:3, skip_copy:false, local_node:0, cxl_nodes:vec![2],
            };
            let mut i = 1;
            while i < a.len() {
                match a[i].as_str() {
                    "--mode"       => { i+=1; r.mode=a[i].clone(); }
                    "--device"     => { i+=1; r.device=a[i].clone(); }
                    "--index"      => { i+=1; r.index_path=a[i].clone(); }
                    "--queries"    => { i+=1; r.queries_path=a[i].clone(); }
                    "--pq-codes"   => { i+=1; r.pq_codes_path=a[i].clone(); }
                    "--groundtruth"=> { i+=1; r.gt_path=a[i].clone(); }
                    "--output"     => { i+=1; r.output_path=a[i].clone(); }
                    "--dim"        => { i+=1; r.dim=a[i].parse().unwrap(); }
                    "--max-degree" => { i+=1; r.max_degree=a[i].parse().unwrap(); }
                    "--block-size" => { i+=1; r.block_size=a[i].parse().unwrap(); }
                    "--data-start" => { i+=1; r.data_start=a[i].parse().unwrap(); }
                    "--num-points" => { i+=1; r.num_points=a[i].parse().unwrap(); }
                    "--reps"       => { i+=1; r.reps=a[i].parse().unwrap(); }
                    "--skip-copy"  => { r.skip_copy=true; }
                    "--local-node" => { i+=1; r.local_node=a[i].parse().unwrap(); }
                    "--cxl-nodes"  => { i+=1; r.cxl_nodes=parse_node_spec(&a[i]); }
                    "--help"|"-h"  => {
                        eprintln!("Usage: cxl_bench [OPTIONS]\n");
                        eprintln!("Modes:  --mode disk | numa\n");
                        eprintln!("Common:");
                        eprintln!("  --index PATH       Disk index (required)");
                        eprintln!("  --queries PATH     Query file (.fbin/.i8bin)");
                        eprintln!("  --groundtruth PATH Groundtruth (.i32bin) for recall");
                        eprintln!("  --output PATH      JSON results file");
                        eprintln!("  --reps N           Repetitions (default: 3)\n");
                        eprintln!("Disk:   --device PATH[,PATH2]  --skip-copy");
                        eprintln!("NUMA:   --local-node N  --cxl-nodes SPEC (e.g. 3,4 or 3-5)");
                        eprintln!("        --pq-codes PATH\n");
                        eprintln!("Overrides: --dim --max-degree --num-points --block-size --data-start");
                        std::process::exit(0);
                    }
                    o => { eprintln!("Unknown: {}", o); std::process::exit(1); }
                }
                i+=1;
            }
            if r.index_path.is_empty() { eprintln!("--index required"); std::process::exit(1); }
            r
        }
    }
}

#[cfg(feature = "cxl")]
fn main() { bench::run(bench::BenchArgs::from_env()); }
#[cfg(not(feature = "cxl"))]
fn main() { eprintln!("Build with: cargo build --features diskann-disk/cxl"); std::process::exit(1); }
