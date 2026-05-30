//! CXL NUMA-Aware DataProvider
//!
//! Option 2 implementation: a new DataProvider that explicitly places data
//! across DRAM and CXL NUMA nodes using the in-memory GraphIndex path.
//!
//! Architecture:
//!   Local DRAM (NUMA 0/1): PQ codebooks, distance tables, search state, PQ codes
//!   CXL Memory (NUMA 2+):  Full-precision vectors, graph adjacency lists

use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use diskann::{ANNError, ANNResult};

// NUMA policy constants — not always exported by the libc crate
const MPOL_BIND: i32 = 2;
const MPOL_MF_MOVE: i32 = 1 << 1;
const MPOL_MF_STRICT: i32 = 1 << 0;

/// CXL NUMA node ID. Detected via `numactl --hardware` or /sys/devices/system/node/.
pub type NumaNode = u32;

/// NUMA memory binding policy.
#[derive(Debug, Clone)]
pub enum NumaPolicy {
    Bind(NumaNode),
    Preferred(NumaNode),
    Interleave(Vec<NumaNode>),
    Default,
}

/// Helper to create an ANNError from a string message.
fn cxl_io_error(msg: impl Into<String>) -> ANNError {
    ANNError::log_io_error(std::io::Error::new(std::io::ErrorKind::Other, msg.into()))
}

// ============================================================================
// NUMA-aware allocation
// ============================================================================

/// Allocate memory on a specific NUMA node using mbind syscall.
///
/// # Safety
/// Returns raw pointer. Caller must manage lifetime and deallocation.
unsafe fn numa_alloc(size: usize, node: NumaNode) -> ANNResult<*mut u8> {
    let layout = Layout::from_size_align(size, 64)
        .map_err(|e| cxl_io_error(format!("Layout error: {}", e)))?;
    let ptr = alloc(layout);
    if ptr.is_null() {
        return Err(cxl_io_error(format!("Failed to allocate {} bytes", size)));
    }

    // Touch pages to ensure allocation
    std::ptr::write_bytes(ptr, 0, size);

    // Bind to NUMA node via raw syscall (avoids libnuma dependency)
    let nodemask: u64 = 1 << node;

    let ret = libc::syscall(
        libc::SYS_mbind,
        ptr as *mut libc::c_void,
        size as libc::c_ulong,
        MPOL_BIND as libc::c_long,
        &nodemask as *const u64 as libc::c_long,
        64 as libc::c_long,
        (MPOL_MF_MOVE) as libc::c_long,
    ) as libc::c_int;

    if ret != 0 {
        let err = std::io::Error::last_os_error();
        dealloc(ptr, layout);
        return Err(cxl_io_error(format!(
            "mbind to NUMA node {} failed: {}. \
             Check: numactl --hardware shows CXL node. \
             Try: numactl --membind={} to verify.",
            node, err, node
        )));
    }

    Ok(ptr)
}

/// NUMA-bound buffer. Deallocates on drop.
struct NumaBuffer {
    ptr: *mut u8,
    layout: Layout,
}

impl NumaBuffer {
    fn new(size: usize, node: NumaNode) -> ANNResult<Self> {
        let layout = Layout::from_size_align(size, 64)
            .map_err(|e| cxl_io_error(format!("Layout error: {}", e)))?;
        let ptr = unsafe { numa_alloc(size, node)? };
        Ok(Self { ptr, layout })
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.layout.size()
    }

    unsafe fn as_slice<T>(&self, count: usize) -> &[T] {
        std::slice::from_raw_parts(self.ptr as *const T, count)
    }

    #[allow(dead_code)]
    unsafe fn as_mut_slice<T>(&self, count: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr as *mut T, count)
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

unsafe impl Send for NumaBuffer {}
unsafe impl Sync for NumaBuffer {}

// ============================================================================
// CXL Data Provider
// ============================================================================

/// CXL-aware DataProvider with explicit NUMA placement.
///
/// Separates hot (PQ) data in local DRAM from cold (full vectors + graph)
/// data in CXL memory.
pub struct CxlDataProvider<T: Copy> {
    // ── Local DRAM tier ──
    pq_codes: NumaBuffer,
    pq_codebook: NumaBuffer,
    #[allow(dead_code)]
    local_numa_node: NumaNode,

    // ── CXL tier ──
    vectors_mmap: Option<Mmap>,
    graph_mmap: Option<Mmap>,
    vectors_numa: Option<NumaBuffer>,
    graph_numa: Option<NumaBuffer>,
    #[allow(dead_code)]
    cxl_numa_node: NumaNode,

    // ── Metadata ──
    num_vectors: usize,
    dimensions: usize,
    max_degree: usize,
    pq_subspaces: usize,
    pq_centroids: usize,
    pq_sub_dim: usize,

    // ── Access mode ──
    mode: CxlAccessMode,

    _phantom: PhantomData<T>,
}

/// How CXL data is accessed.
#[derive(Debug, Clone, Copy)]
pub enum CxlAccessMode {
    /// mmap from DAX filesystem (file-backed)
    MmapDax,
    /// NUMA-allocated buffers (in-memory, loaded from files at startup)
    NumaBuffers,
}

/// Configuration for CXL data provider.
#[derive(Debug, Clone)]
pub struct CxlProviderConfig {
    pub local_numa_node: NumaNode,
    pub cxl_numa_node: NumaNode,
    pub mode: CxlAccessMode,
    pub pq_subspaces: usize,
    pub pq_centroids: usize,
    pub max_degree: usize,
    pub dimensions: usize,
}

impl<T: Copy + 'static> CxlDataProvider<T> {
    /// Create provider with mmap'd CXL files (DAX mode).
    pub fn from_dax_files(
        vectors_path: &Path,
        graph_path: &Path,
        pq_codes_path: &Path,
        pq_codebook_path: &Path,
        num_vectors: usize,
        config: &CxlProviderConfig,
    ) -> ANNResult<Self> {
        let pq_sub_dim = config.dimensions / config.pq_subspaces;

        // ── Load PQ data into local DRAM ──
        let pq_codes_size = num_vectors * config.pq_subspaces;
        let pq_codes = NumaBuffer::new(pq_codes_size, config.local_numa_node)?;
        load_file_into_buffer(pq_codes_path, pq_codes.as_mut_ptr(), pq_codes_size)?;

        let codebook_size =
            config.pq_centroids * config.pq_subspaces * pq_sub_dim * std::mem::size_of::<f32>();
        let pq_codebook = NumaBuffer::new(codebook_size, config.local_numa_node)?;
        load_file_into_buffer(pq_codebook_path, pq_codebook.as_mut_ptr(), codebook_size)?;

        // ── mmap vectors + graph from CXL DAX ──
        let vectors_file = std::fs::File::open(vectors_path).map_err(ANNError::log_io_error)?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file).map_err(ANNError::log_io_error)? };
        unsafe {
            libc::madvise(
                vectors_mmap.as_ptr() as *mut libc::c_void,
                vectors_mmap.len(),
                libc::MADV_RANDOM,
            );
        }

        let graph_file = std::fs::File::open(graph_path).map_err(ANNError::log_io_error)?;
        let graph_mmap = unsafe { Mmap::map(&graph_file).map_err(ANNError::log_io_error)? };
        unsafe {
            libc::madvise(
                graph_mmap.as_ptr() as *mut libc::c_void,
                graph_mmap.len(),
                libc::MADV_RANDOM,
            );
        }

        Ok(Self {
            pq_codes,
            pq_codebook,
            local_numa_node: config.local_numa_node,
            vectors_mmap: Some(vectors_mmap),
            graph_mmap: Some(graph_mmap),
            vectors_numa: None,
            graph_numa: None,
            cxl_numa_node: config.cxl_numa_node,
            num_vectors,
            dimensions: config.dimensions,
            max_degree: config.max_degree,
            pq_subspaces: config.pq_subspaces,
            pq_centroids: config.pq_centroids,
            pq_sub_dim,
            mode: CxlAccessMode::MmapDax,
            _phantom: PhantomData,
        })
    }

    /// Create provider with NUMA-bound memory buffers.
    pub fn from_numa_buffers(
        vectors_path: &Path,
        graph_path: &Path,
        pq_codes_path: &Path,
        pq_codebook_path: &Path,
        num_vectors: usize,
        config: &CxlProviderConfig,
    ) -> ANNResult<Self> {
        let elem_size = std::mem::size_of::<T>();
        let pq_sub_dim = config.dimensions / config.pq_subspaces;

        // ── PQ data → local DRAM ──
        let pq_codes_size = num_vectors * config.pq_subspaces;
        let pq_codes = NumaBuffer::new(pq_codes_size, config.local_numa_node)?;
        load_file_into_buffer(pq_codes_path, pq_codes.as_mut_ptr(), pq_codes_size)?;

        let codebook_size =
            config.pq_centroids * config.pq_subspaces * pq_sub_dim * std::mem::size_of::<f32>();
        let pq_codebook = NumaBuffer::new(codebook_size, config.local_numa_node)?;
        load_file_into_buffer(pq_codebook_path, pq_codebook.as_mut_ptr(), codebook_size)?;

        // ── Vectors + graph → CXL NUMA node ──
        let vectors_size = num_vectors * config.dimensions * elem_size;
        let vectors_buf = NumaBuffer::new(vectors_size, config.cxl_numa_node)?;
        load_file_into_buffer(vectors_path, vectors_buf.as_mut_ptr(), vectors_size)?;

        let graph_node_size = 4 + config.max_degree * 4;
        let graph_size = num_vectors * graph_node_size;
        let graph_buf = NumaBuffer::new(graph_size, config.cxl_numa_node)?;
        load_file_into_buffer(graph_path, graph_buf.as_mut_ptr(), graph_size)?;

        Ok(Self {
            pq_codes,
            pq_codebook,
            local_numa_node: config.local_numa_node,
            vectors_mmap: None,
            graph_mmap: None,
            vectors_numa: Some(vectors_buf),
            graph_numa: Some(graph_buf),
            cxl_numa_node: config.cxl_numa_node,
            num_vectors,
            dimensions: config.dimensions,
            max_degree: config.max_degree,
            pq_subspaces: config.pq_subspaces,
            pq_centroids: config.pq_centroids,
            pq_sub_dim,
            mode: CxlAccessMode::NumaBuffers,
            _phantom: PhantomData,
        })
    }

    /// Get full-precision vector. Returns reference into CXL memory.
    #[inline(always)]
    pub fn get_full_vector(&self, id: u32) -> &[T] {
        let idx = id as usize;
        debug_assert!(idx < self.num_vectors);
        let elem_size = std::mem::size_of::<T>();
        let offset = idx * self.dimensions * elem_size;

        match self.mode {
            CxlAccessMode::MmapDax => {
                let mmap = self.vectors_mmap.as_ref().unwrap();
                unsafe {
                    let ptr = mmap.as_ptr().add(offset) as *const T;
                    std::slice::from_raw_parts(ptr, self.dimensions)
                }
            }
            CxlAccessMode::NumaBuffers => {
                let buf = self.vectors_numa.as_ref().unwrap();
                unsafe {
                    let ptr = buf.as_ptr().add(offset) as *const T;
                    std::slice::from_raw_parts(ptr, self.dimensions)
                }
            }
        }
    }

    /// Get PQ codes for a vector. Returns reference into local DRAM.
    #[inline(always)]
    pub fn get_pq_codes(&self, id: u32) -> &[u8] {
        let idx = id as usize;
        debug_assert!(idx < self.num_vectors);
        let offset = idx * self.pq_subspaces;
        unsafe {
            std::slice::from_raw_parts(self.pq_codes.as_ptr().add(offset), self.pq_subspaces)
        }
    }

    /// Get PQ codebook. Returns reference into local DRAM.
    #[inline(always)]
    pub fn get_codebook(&self) -> &[f32] {
        let count = self.pq_centroids * self.pq_subspaces * self.pq_sub_dim;
        unsafe { self.pq_codebook.as_slice::<f32>(count) }
    }

    /// Get neighbor IDs for a node. Returns reference into CXL memory.
    #[inline(always)]
    pub fn get_neighbors(&self, id: u32) -> &[u32] {
        let idx = id as usize;
        debug_assert!(idx < self.num_vectors);

        let node_size = 4 + self.max_degree * 4;
        let offset = idx * node_size;

        match self.mode {
            CxlAccessMode::MmapDax => {
                let mmap = self.graph_mmap.as_ref().unwrap();
                unsafe {
                    let base = mmap.as_ptr().add(offset);
                    let num_nbrs = (*(base as *const u32)) as usize;
                    let num_nbrs = num_nbrs.min(self.max_degree);
                    let nbrs = base.add(4) as *const u32;
                    std::slice::from_raw_parts(nbrs, num_nbrs)
                }
            }
            CxlAccessMode::NumaBuffers => {
                let buf = self.graph_numa.as_ref().unwrap();
                unsafe {
                    let base = buf.as_ptr().add(offset);
                    let num_nbrs = (*(base as *const u32)) as usize;
                    let num_nbrs = num_nbrs.min(self.max_degree);
                    let nbrs = base.add(4) as *const u32;
                    std::slice::from_raw_parts(nbrs, num_nbrs)
                }
            }
        }
    }

    /// Prefetch a vector from CXL into CPU cache.
    #[inline(always)]
    pub fn prefetch_vector(&self, id: u32) {
        let idx = id as usize;
        if idx >= self.num_vectors {
            return;
        }
        let elem_size = std::mem::size_of::<T>();
        let offset = idx * self.dimensions * elem_size;
        let total_bytes = self.dimensions * elem_size;
        let num_cls = (total_bytes + 63) / 64;

        let base_ptr = match self.mode {
            CxlAccessMode::MmapDax => {
                let mmap = self.vectors_mmap.as_ref().unwrap();
                unsafe { mmap.as_ptr().add(offset) }
            }
            CxlAccessMode::NumaBuffers => {
                let buf = self.vectors_numa.as_ref().unwrap();
                unsafe { buf.as_ptr().add(offset) }
            }
        };

        unsafe {
            for cl in 0..num_cls {
                let addr = base_ptr.add(cl * 64);
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::x86_64::_mm_prefetch(
                        addr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::aarch64::_prefetch(addr as *const i8, 0, 3, 1);
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    let _ = addr;
                }
            }
        }
    }

    /// Prefetch graph adjacency list from CXL.
    #[inline(always)]
    pub fn prefetch_neighbors(&self, id: u32) {
        let idx = id as usize;
        if idx >= self.num_vectors {
            return;
        }
        let node_size = 4 + self.max_degree * 4;
        let offset = idx * node_size;
        let num_cls = (node_size + 63) / 64;

        let base_ptr = match self.mode {
            CxlAccessMode::MmapDax => {
                let mmap = self.graph_mmap.as_ref().unwrap();
                unsafe { mmap.as_ptr().add(offset) }
            }
            CxlAccessMode::NumaBuffers => {
                let buf = self.graph_numa.as_ref().unwrap();
                unsafe { buf.as_ptr().add(offset) }
            }
        };

        unsafe {
            for cl in 0..num_cls {
                let addr = base_ptr.add(cl * 64);
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::x86_64::_mm_prefetch(
                        addr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::aarch64::_prefetch(addr as *const i8, 0, 3, 1);
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    let _ = addr;
                }
            }
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Memory usage summary.
    pub fn memory_stats(&self) -> MemoryStats {
        let elem_size = std::mem::size_of::<T>();
        let vector_bytes = self.num_vectors * self.dimensions * elem_size;
        let graph_bytes = self.num_vectors * (4 + self.max_degree * 4);
        let pq_codes_bytes = self.pq_codes.len();
        let pq_codebook_bytes = self.pq_codebook.len();

        MemoryStats {
            local_dram_bytes: pq_codes_bytes + pq_codebook_bytes,
            cxl_bytes: vector_bytes + graph_bytes,
            pq_codes_bytes,
            pq_codebook_bytes,
            vector_bytes,
            graph_bytes,
        }
    }
}

/// Memory usage breakdown.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub local_dram_bytes: usize,
    pub cxl_bytes: usize,
    pub pq_codes_bytes: usize,
    pub pq_codebook_bytes: usize,
    pub vector_bytes: usize,
    pub graph_bytes: usize,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Local DRAM: {:.1} MB (PQ codes: {:.1} MB, codebook: {:.1} MB)\n\
             CXL Memory: {:.1} MB (vectors: {:.1} MB, graph: {:.1} MB)",
            self.local_dram_bytes as f64 / 1e6,
            self.pq_codes_bytes as f64 / 1e6,
            self.pq_codebook_bytes as f64 / 1e6,
            self.cxl_bytes as f64 / 1e6,
            self.vector_bytes as f64 / 1e6,
            self.graph_bytes as f64 / 1e6,
        )
    }
}

// ============================================================================
// NUMA verification utilities
// ============================================================================

/// Verify CXL NUMA node is available and has expected capacity.
pub fn verify_cxl_numa(node: NumaNode) -> ANNResult<NumaNodeInfo> {
    let sysfs_path = format!("/sys/devices/system/node/node{}", node);
    if !Path::new(&sysfs_path).exists() {
        return Err(cxl_io_error(format!(
            "NUMA node {} not found. Check:\n\
             1. CXL device is recognized: lspci | grep CXL\n\
             2. Device configured: daxctl list\n\
             3. Mode is system-ram: daxctl reconfigure-device --mode=system-ram dax0.0\n\
             4. Verify: numactl --hardware",
            node
        )));
    }

    let meminfo_path = format!("{}/meminfo", sysfs_path);
    let content = std::fs::read_to_string(&meminfo_path)
        .map_err(ANNError::log_io_error)?;

    let mut total_kb = 0u64;
    let mut free_kb = 0u64;
    for line in content.lines() {
        if line.contains("MemTotal") {
            total_kb = parse_meminfo_value(line);
        }
        if line.contains("MemFree") {
            free_kb = parse_meminfo_value(line);
        }
    }

    Ok(NumaNodeInfo {
        node,
        total_bytes: total_kb * 1024,
        free_bytes: free_kb * 1024,
    })
}

fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace()
        .filter_map(|w| w.parse::<u64>().ok())
        .next()
        .unwrap_or(0)
}

#[derive(Debug)]
pub struct NumaNodeInfo {
    pub node: NumaNode,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

impl std::fmt::Display for NumaNodeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NUMA node {}: {:.1} GB total, {:.1} GB free",
            self.node,
            self.total_bytes as f64 / 1e9,
            self.free_bytes as f64 / 1e9,
        )
    }
}

// ============================================================================
// File loading utility
// ============================================================================

fn load_file_into_buffer(path: &Path, dst: *mut u8, expected_size: usize) -> ANNResult<()> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(ANNError::log_io_error)?;

    let file_size = file.metadata().map_err(ANNError::log_io_error)?.len() as usize;
    if file_size < expected_size {
        return Err(cxl_io_error(format!(
            "File {} is {} bytes, expected at least {}",
            path.display(),
            file_size,
            expected_size
        )));
    }

    let buf = unsafe { std::slice::from_raw_parts_mut(dst, expected_size) };
    file.read_exact(buf).map_err(ANNError::log_io_error)?;

    Ok(())
}

/// Arc-wrapped provider for multi-threaded search.
pub type SharedCxlProvider<T> = Arc<CxlDataProvider<T>>;

/// Benchmark JSON config for CXL provider.
#[derive(Debug, Clone)]
pub struct CxlBenchmarkConfig {
    pub local_numa_node: NumaNode,
    pub cxl_numa_node: NumaNode,
    pub mode: CxlAccessMode,
    pub vectors_path: String,
    pub graph_path: String,
    pub pq_codes_path: String,
    pub pq_codebook_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_display() {
        let stats = MemoryStats {
            local_dram_bytes: 32_000_000 + 1_000_000,
            cxl_bytes: 512_000_000 + 256_000_000,
            pq_codes_bytes: 32_000_000,
            pq_codebook_bytes: 1_000_000,
            vector_bytes: 512_000_000,
            graph_bytes: 256_000_000,
        };
        let display = format!("{}", stats);
        assert!(display.contains("Local DRAM"));
        assert!(display.contains("CXL Memory"));
    }

    #[test]
    fn test_align_calculation() {
        let n = 1_000_000usize;
        let d = 128usize;
        let m = 32usize;
        let r = 64usize;

        let vector_bytes = n * d * 4;
        let graph_bytes = n * (4 + r * 4);
        let pq_bytes = n * m;

        assert!(vector_bytes + graph_bytes > 700_000_000);
        assert!(pq_bytes < 40_000_000);
    }

    #[test]
    fn test_numa_node_info_display() {
        let info = NumaNodeInfo {
            node: 2,
            total_bytes: 64_000_000_000,
            free_bytes: 60_000_000_000,
        };
        let display = format!("{}", info);
        assert!(display.contains("NUMA node 2"));
        assert!(display.contains("64.0 GB"));
    }

    #[test]
    fn test_numa_policy_variants() {
        let _bind = NumaPolicy::Bind(2);
        let _pref = NumaPolicy::Preferred(0);
        let _inter = NumaPolicy::Interleave(vec![0, 1, 2]);
        let _def = NumaPolicy::Default;
    }
}
