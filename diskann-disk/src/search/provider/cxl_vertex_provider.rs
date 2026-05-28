//! CXL Prefetching Vertex Provider
//!
//! Option 3, Changes 3+4+6: Wraps existing VertexProvider with:
//! - Software prefetching to hide CXL latency (~300ns per cache line)
//! - Reduced/optional node caching (cache value drops ~500x vs SSD)
//! - Zero-copy access path via mmap (bypass AlignedRead overhead)

use std::marker::PhantomData;

use crate::utils::aligned_file_reader::cxl_mmap_reader::CxlMmapReader;
use crate::search::provider::cxl_sector_graph::NodeOffsetCalculator;
// use crate::stubs::ANNResult;
use diskann::ANNResult;
//use diskann::error::ann_error::ANNResult;

/// Number of nodes to prefetch ahead in the beam search loop.
const DEFAULT_PREFETCH_AHEAD: usize = 3;

/// Maximum nodes to cache (medoid + immediate neighbors).
const DEFAULT_CXL_CACHE_SIZE: usize = 64;

/// CXL-optimized vertex provider.
///
/// Instead of using AlignedRead + io_uring, directly accesses mmap'd CXL memory.
/// Provides zero-copy node access and integrated software prefetching.
pub struct CxlVertexProvider<T> {
    reader: CxlMmapReader,
    offsets: NodeOffsetCalculator,
    prefetch_ahead: usize,
    cache: Option<HotNodeCache>,
    io_count: u32,
    dimensions: usize,
    max_degree: usize,
    _phantom: PhantomData<T>,
}

/// Minimal cache for frequently accessed nodes (medoid + level-1 neighbors).
struct HotNodeCache {
    entries: std::collections::HashMap<u32, CachedNode>,
    capacity: usize,
}

struct CachedNode {
    vector: Vec<u8>,
    neighbors: Vec<u32>,
}

impl HotNodeCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&self, id: u32) -> Option<&CachedNode> {
        self.entries.get(&id)
    }

    fn insert(&mut self, id: u32, node: CachedNode) {
        if self.entries.len() < self.capacity {
            self.entries.insert(id, node);
        }
    }

    fn contains(&self, id: &u32) -> bool {
        self.entries.contains_key(id)
    }
}

/// Configuration for CXL vertex provider.
#[derive(Debug, Clone)]
pub struct CxlVertexProviderConfig {
    pub prefetch_ahead: usize,
    pub cache_enabled: bool,
    pub cache_size: usize,
}

impl Default for CxlVertexProviderConfig {
    fn default() -> Self {
        Self {
            prefetch_ahead: DEFAULT_PREFETCH_AHEAD,
            cache_enabled: true,
            cache_size: DEFAULT_CXL_CACHE_SIZE,
        }
    }
}

impl CxlVertexProviderConfig {
    /// No-cache config for benchmarking cache impact.
    pub fn no_cache() -> Self {
        Self {
            cache_enabled: false,
            cache_size: 0,
            ..Default::default()
        }
    }
}

impl<T: Copy + 'static> CxlVertexProvider<T> {
    /// Create provider from mmap reader and layout parameters.
    pub fn new(
        reader: CxlMmapReader,
        offsets: NodeOffsetCalculator,
        dimensions: usize,
        max_degree: usize,
        config: CxlVertexProviderConfig,
    ) -> Self {
        let cache = if config.cache_enabled {
            Some(HotNodeCache::new(config.cache_size))
        } else {
            None
        };

        Self {
            reader,
            offsets,
            prefetch_ahead: config.prefetch_ahead,
            cache,
            io_count: 0,
            dimensions,
            max_degree,
            _phantom: PhantomData,
        }
    }

    /// Warm the cache with medoid and its immediate neighbors.
    pub fn warm_cache(&mut self, medoid_id: u32, medoid_neighbors: &[u32]) -> ANNResult<()> {
        // Read nodes first (borrows self immutably via reader/offsets),
        // then insert into cache separately to avoid borrow conflict.
        let cache_capacity = match &self.cache {
            Some(c) => c.capacity,
            None => return Ok(()),
        };

        let medoid_node = self.read_and_parse_node(medoid_id)?;

        let mut neighbor_nodes = Vec::new();
        for &nbr_id in medoid_neighbors {
            if 1 + neighbor_nodes.len() >= cache_capacity {
                break;
            }
            neighbor_nodes.push((nbr_id, self.read_and_parse_node(nbr_id)?));
        }

        if let Some(ref mut cache) = self.cache {
            cache.insert(medoid_id, medoid_node);
            for (id, node) in neighbor_nodes {
                cache.insert(id, node);
            }
        }
        Ok(())
    }

    /// Read and parse a single node from CXL memory.
    fn read_and_parse_node(&self, node_id: u32) -> ANNResult<CachedNode> {
        let offset = self.offsets.node_offset(node_id);
        let elem_size = std::mem::size_of::<T>();
        let vector_bytes = self.dimensions * elem_size;

        unsafe {
            let node_ptr = self.reader.ptr_at(offset);

            let vector = std::slice::from_raw_parts(node_ptr, vector_bytes).to_vec();

            let num_nbrs_ptr = node_ptr.add(vector_bytes) as *const u32;
            let num_nbrs = (*num_nbrs_ptr) as usize;
            let num_nbrs = num_nbrs.min(self.max_degree);

            let nbrs_ptr = node_ptr.add(vector_bytes + 4) as *const u32;
            let neighbors = std::slice::from_raw_parts(nbrs_ptr, num_nbrs).to_vec();

            Ok(CachedNode { vector, neighbors })
        }
    }

    /// Get full-precision vector for a node — zero-copy from CXL mmap.
    #[inline(always)]
    pub fn get_vector(&self, node_id: u32) -> &[T] {
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(node_id) {
                let ptr = cached.vector.as_ptr() as *const T;
                return unsafe { std::slice::from_raw_parts(ptr, self.dimensions) };
            }
        }

        let offset = self.offsets.node_offset(node_id);
        unsafe {
            let ptr = self.reader.ptr_at(offset) as *const T;
            std::slice::from_raw_parts(ptr, self.dimensions)
        }
    }

    /// Get adjacency list (neighbor IDs) for a node — zero-copy from CXL mmap.
    #[inline(always)]
    pub fn get_adjacency_list(&self, node_id: u32) -> &[u32] {
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(node_id) {
                return &cached.neighbors;
            }
        }

        let offset = self.offsets.node_offset(node_id);
        let vector_bytes = self.dimensions * std::mem::size_of::<T>();

        unsafe {
            let node_ptr = self.reader.ptr_at(offset);
            let num_nbrs_ptr = node_ptr.add(vector_bytes) as *const u32;
            let num_nbrs = (*num_nbrs_ptr) as usize;
            let num_nbrs = num_nbrs.min(self.max_degree);

            let nbrs_ptr = node_ptr.add(vector_bytes + 4) as *const u32;
            std::slice::from_raw_parts(nbrs_ptr, num_nbrs)
        }
    }

    /// Batch load vertices with integrated prefetching.
    pub fn load_vertices_prefetched(&mut self, node_ids: &[u32]) {
        let node_read_len = self.offsets.read_len(0);

        for (i, &_node_id) in node_ids.iter().enumerate() {
            if i + self.prefetch_ahead < node_ids.len() {
                let prefetch_id = node_ids[i + self.prefetch_ahead];

                let should_prefetch = match &self.cache {
                    Some(cache) => !cache.contains(&prefetch_id),
                    None => true,
                };

                if should_prefetch {
                    let prefetch_offset = self.offsets.node_offset(prefetch_id);
                    self.reader.prefetch_node(prefetch_offset, node_read_len);
                }
            }

            self.io_count += 1;
        }
    }

    /// Prefetch a specific set of candidate nodes.
    pub fn prefetch_candidates(&self, candidates: &[u32]) {
        let node_read_len = self.offsets.read_len(0);
        let count = candidates.len().min(self.prefetch_ahead * 2);

        for &id in candidates.iter().take(count) {
            let is_cached = self.cache.as_ref().map_or(false, |c| c.contains(&id));
            if !is_cached {
                let offset = self.offsets.node_offset(id);
                self.reader.prefetch_node(offset, node_read_len);
            }
        }
    }

    /// Number of I/O operations (node reads) for current query.
    pub fn io_operations(&self) -> u32 {
        self.io_count
    }

    /// Reset state between queries.
    pub fn clear(&mut self) {
        self.io_count = 0;
        self.reader.reset_io_count();
    }
}

/// Factory that creates CxlVertexProvider instances per search thread.
pub struct CxlVertexProviderFactory {
    index_path: String,
    block_size: usize,
    vector_size_bytes: usize,
    max_degree: usize,
    dimensions: usize,
    data_start_offset: usize,
    config: CxlVertexProviderConfig,
}

impl CxlVertexProviderFactory {
    pub fn new(
        index_path: String,
        block_size: usize,
        vector_size_bytes: usize,
        max_degree: usize,
        dimensions: usize,
        data_start_offset: usize,
        config: CxlVertexProviderConfig,
    ) -> Self {
        Self {
            index_path,
            block_size,
            vector_size_bytes,
            max_degree,
            dimensions,
            data_start_offset,
            config,
        }
    }

    /// Create a new vertex provider for a search thread.
    pub fn create<T: Copy + 'static>(&self) -> ANNResult<CxlVertexProvider<T>> {
        let reader = CxlMmapReader::new(&self.index_path)?;
        let offsets = NodeOffsetCalculator::new(
            self.block_size,
            self.vector_size_bytes,
            self.max_degree,
            self.data_start_offset,
        );

        Ok(CxlVertexProvider::new(
            reader,
            offsets,
            self.dimensions,
            self.max_degree,
            self.config.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = CxlVertexProviderConfig::default();
        assert_eq!(config.prefetch_ahead, 3);
        assert!(config.cache_enabled);
        assert_eq!(config.cache_size, 64);
    }

    #[test]
    fn test_config_no_cache() {
        let config = CxlVertexProviderConfig::no_cache();
        assert!(!config.cache_enabled);
        assert_eq!(config.cache_size, 0);
    }

    #[test]
    fn test_hot_node_cache() {
        let mut cache = HotNodeCache::new(2);
        assert!(!cache.contains(&0));

        cache.insert(
            0,
            CachedNode {
                vector: vec![1, 2, 3, 4],
                neighbors: vec![1, 2],
            },
        );
        assert!(cache.contains(&0));
        assert!(!cache.contains(&1));

        let node = cache.get(0).unwrap();
        assert_eq!(node.neighbors, vec![1, 2]);

        cache.insert(
            1,
            CachedNode {
                vector: vec![5, 6, 7, 8],
                neighbors: vec![0, 3],
            },
        );

        // Over capacity — should not insert
        cache.insert(
            2,
            CachedNode {
                vector: vec![9, 10],
                neighbors: vec![0],
            },
        );
        assert!(!cache.contains(&2));
    }

    #[test]
    fn test_vertex_provider_with_tempfile() {
        use std::io::Write;

        // Create a fake index: 3 nodes, f32, dim=4, max_degree=2
        // Node layout: [vector: 16 bytes][num_nbrs: 4][neighbors: 8] = 28 bytes
        // Aligned to 64: 64 bytes per node
        let dim = 4usize;
        let max_degree = 2usize;
        let _raw_node_len = dim * 4 + 4 + max_degree * 4; // 28
        let aligned_node_len = 64; // ceil(28/64)*64
        let num_nodes = 3u32;

        let mut data = vec![0u8; aligned_node_len * num_nodes as usize];

        for node_id in 0..num_nodes {
            let offset = node_id as usize * aligned_node_len;
            // Write vector: [node_id as f32, 0.0, 0.0, 0.0]
            let vec_val = node_id as f32;
            data[offset..offset + 4].copy_from_slice(&vec_val.to_ne_bytes());
            // Write num_nbrs
            let num_nbrs = if node_id == 0 { 2u32 } else { 1u32 };
            data[offset + 16..offset + 20].copy_from_slice(&num_nbrs.to_ne_bytes());
            // Write neighbors
            let nbr1 = (node_id + 1) % num_nodes;
            data[offset + 20..offset + 24].copy_from_slice(&nbr1.to_ne_bytes());
            if num_nbrs > 1 {
                let nbr2 = (node_id + 2) % num_nodes;
                data[offset + 24..offset + 28].copy_from_slice(&nbr2.to_ne_bytes());
            }
        }

        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&data).unwrap();
        f.flush().unwrap();

        let reader = CxlMmapReader::with_config(
            f.path(),
            crate::utils::aligned_file_reader::cxl_mmap_reader::CxlMmapConfig {
                prefault: false,
                advise_random: false,
                use_huge_pages: false,
            },
        )
        .unwrap();

        let offsets = NodeOffsetCalculator::new(64, dim * 4, max_degree, 0);
        let provider = CxlVertexProvider::<f32>::new(
            reader,
            offsets,
            dim,
            max_degree,
            CxlVertexProviderConfig::no_cache(),
        );

        // Verify zero-copy vector read
        let vec0 = provider.get_vector(0);
        assert_eq!(vec0[0], 0.0f32);

        let vec1 = provider.get_vector(1);
        assert_eq!(vec1[0], 1.0f32);

        let vec2 = provider.get_vector(2);
        assert_eq!(vec2[0], 2.0f32);

        // Verify adjacency list
        let adj0 = provider.get_adjacency_list(0);
        assert_eq!(adj0.len(), 2);
        assert_eq!(adj0[0], 1);
        assert_eq!(adj0[1], 2);

        let adj1 = provider.get_adjacency_list(1);
        assert_eq!(adj1.len(), 1);
        assert_eq!(adj1[0], 2);
    }
}
