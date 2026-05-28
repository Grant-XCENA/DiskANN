//! CXL-Optimized Disk Sector Graph Layout
//!
//! Option 3, Change 2: Eliminates 4KB sector padding for CXL.
//! Nodes packed at cache-line (64-byte) boundaries instead of 4KB blocks.
//! Reduces bandwidth waste from ~76% (960-byte nodes in 4KB sectors) to ~6%.

/// Cache-line size for CXL alignment (64 bytes on x86_64 and aarch64).
pub const CXL_ALIGN: usize = 64;

/// Minimum block size that indicates CXL layout in the graph header.
/// Any block_size < 512 signals CXL format (SSD requires ≥512 for O_DIRECT).
pub const CXL_BLOCK_SIZE_THRESHOLD: u16 = 512;

/// Node layout mode — detected from graph header block_size field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLayoutMode {
    /// Traditional SSD layout: nodes packed into 4KB sectors with padding.
    SsdSectorAligned {
        block_size: usize,
        nodes_per_sector: usize,
    },
    /// CXL layout: nodes at cache-line boundaries, no sector grouping.
    CxlCacheLineAligned {
        /// Node size rounded up to cache-line boundary
        aligned_node_len: usize,
    },
}

/// Computes node offsets for both SSD and CXL layouts.
#[derive(Debug, Clone)]
pub struct NodeOffsetCalculator {
    mode: NodeLayoutMode,
    /// Raw node length: sizeof(vector) + sizeof(u32) + max_degree * sizeof(u32)
    raw_node_len: usize,
    /// Header/metadata offset — data starts after this
    data_start_offset: usize,
}

impl NodeOffsetCalculator {
    /// Create calculator from graph header parameters.
    ///
    /// # Arguments
    /// * `block_size` - From GraphHeader. 4096 = SSD, 64 = CXL.
    /// * `vector_size_bytes` - dimensions * sizeof(T) (e.g., 128 * 4 = 512 for SIFT128 f32)
    /// * `max_degree` - Maximum neighbor list length (R parameter)
    /// * `data_start_offset` - Byte offset where node data begins (after header)
    pub fn new(
        block_size: usize,
        vector_size_bytes: usize,
        max_degree: usize,
        data_start_offset: usize,
    ) -> Self {
        // Node layout: [vector: vector_size_bytes][num_nbrs: 4][neighbors: max_degree * 4]
        let raw_node_len = vector_size_bytes + 4 + max_degree * 4;

        let mode = if block_size < CXL_BLOCK_SIZE_THRESHOLD as usize {
            let aligned = align_up(raw_node_len, CXL_ALIGN);
            NodeLayoutMode::CxlCacheLineAligned {
                aligned_node_len: aligned,
            }
        } else {
            let nodes_per_sector = block_size / raw_node_len;
            NodeLayoutMode::SsdSectorAligned {
                block_size,
                nodes_per_sector: nodes_per_sector.max(1),
            }
        };

        Self {
            mode,
            raw_node_len,
            data_start_offset,
        }
    }

    /// Detect layout mode from block_size in graph header.
    pub fn from_block_size(
        block_size: usize,
        vector_size_bytes: usize,
        max_degree: usize,
        data_start_offset: usize,
    ) -> Self {
        Self::new(block_size, vector_size_bytes, max_degree, data_start_offset)
    }

    /// Compute absolute byte offset of a node in the index file.
    #[inline(always)]
    pub fn node_offset(&self, node_id: u32) -> usize {
        let id = node_id as usize;

        match self.mode {
            NodeLayoutMode::CxlCacheLineAligned { aligned_node_len } => {
                self.data_start_offset + id * aligned_node_len
            }
            NodeLayoutMode::SsdSectorAligned {
                block_size,
                nodes_per_sector,
            } => {
                if self.raw_node_len <= block_size {
                    let sector_id = id / nodes_per_sector;
                    let offset_in_sector = (id % nodes_per_sector) * self.raw_node_len;
                    self.data_start_offset + sector_id * block_size + offset_in_sector
                } else {
                    let sectors_per_node = self.raw_node_len.div_ceil(block_size);
                    self.data_start_offset + id * sectors_per_node * block_size
                }
            }
        }
    }

    /// Compute byte length to read for a single node.
    #[inline(always)]
    pub fn read_len(&self, _node_id: u32) -> usize {
        match self.mode {
            NodeLayoutMode::CxlCacheLineAligned { aligned_node_len } => aligned_node_len,
            NodeLayoutMode::SsdSectorAligned {
                block_size,
                nodes_per_sector: _,
            } => {
                if self.raw_node_len <= block_size {
                    block_size
                } else {
                    let sectors_per_node = self.raw_node_len.div_ceil(block_size);
                    sectors_per_node * block_size
                }
            }
        }
    }

    /// Number of bytes actually used per node (before alignment padding).
    pub fn raw_node_len(&self) -> usize {
        self.raw_node_len
    }

    /// Effective bytes per node including alignment padding.
    pub fn effective_node_len(&self) -> usize {
        match self.mode {
            NodeLayoutMode::CxlCacheLineAligned { aligned_node_len } => aligned_node_len,
            NodeLayoutMode::SsdSectorAligned {
                block_size,
                nodes_per_sector,
            } => {
                if self.raw_node_len <= block_size {
                    block_size / nodes_per_sector
                } else {
                    self.raw_node_len.div_ceil(block_size) * block_size
                }
            }
        }
    }

    /// Bandwidth efficiency: ratio of useful bytes to total bytes read.
    pub fn efficiency(&self) -> f64 {
        self.raw_node_len as f64 / self.effective_node_len() as f64
    }

    /// Layout mode in use.
    pub fn mode(&self) -> NodeLayoutMode {
        self.mode
    }
}

/// Align value up to next multiple of alignment.
#[inline(always)]
const fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Extension to GraphHeader to support CXL layout.
#[derive(Debug, Clone)]
pub struct CxlLayoutInfo {
    pub is_cxl_layout: bool,
    pub alignment: usize,
    pub raw_node_size: usize,
    pub aligned_node_size: usize,
}

impl CxlLayoutInfo {
    /// Detect CXL layout from existing header block_size field.
    pub fn from_block_size(block_size: usize, raw_node_size: usize) -> Self {
        if block_size < CXL_BLOCK_SIZE_THRESHOLD as usize {
            Self {
                is_cxl_layout: true,
                alignment: block_size.max(CXL_ALIGN),
                raw_node_size,
                aligned_node_size: align_up(raw_node_size, block_size.max(CXL_ALIGN)),
            }
        } else {
            Self {
                is_cxl_layout: false,
                alignment: block_size,
                raw_node_size,
                aligned_node_size: block_size,
            }
        }
    }
}

/// CXL build configuration.
#[derive(Debug, Clone)]
pub struct CxlBuildConfig {
    /// Block size to write in graph header. Set to 64 for CXL.
    pub block_size: usize,
    /// Whether to write nodes with cache-line alignment
    pub cache_line_align: bool,
}

impl Default for CxlBuildConfig {
    fn default() -> Self {
        Self {
            block_size: CXL_ALIGN,
            cache_line_align: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIFT128_VECTOR_SIZE: usize = 512;
    const DEGREE_64: usize = 64;
    const HEADER_OFFSET: usize = 4096;

    #[test]
    fn test_cxl_layout_offset() {
        let calc = NodeOffsetCalculator::new(64, SIFT128_VECTOR_SIZE, DEGREE_64, HEADER_OFFSET);

        assert_eq!(calc.raw_node_len(), 772);
        let aligned = align_up(772, 64);
        assert_eq!(aligned, 832);

        assert_eq!(calc.node_offset(0), HEADER_OFFSET);
        assert_eq!(calc.node_offset(1), HEADER_OFFSET + 832);
        assert_eq!(calc.node_offset(100), HEADER_OFFSET + 100 * 832);
    }

    #[test]
    fn test_ssd_layout_offset() {
        let calc = NodeOffsetCalculator::new(4096, SIFT128_VECTOR_SIZE, DEGREE_64, HEADER_OFFSET);

        assert_eq!(calc.node_offset(0), HEADER_OFFSET);
        assert_eq!(calc.node_offset(4), HEADER_OFFSET + 4 * 772);
        assert_eq!(calc.node_offset(5), HEADER_OFFSET + 4096);
    }

    #[test]
    fn test_cxl_efficiency_vs_ssd() {
        let cxl = NodeOffsetCalculator::new(64, SIFT128_VECTOR_SIZE, DEGREE_64, 0);
        let ssd = NodeOffsetCalculator::new(4096, SIFT128_VECTOR_SIZE, DEGREE_64, 0);

        let cxl_eff = cxl.efficiency();
        let _ssd_eff = ssd.efficiency();

        assert!(cxl_eff > 0.90, "CXL efficiency={}", cxl_eff);
        println!("CXL efficiency: {:.1}%", cxl_eff * 100.0);
        println!("SSD efficiency: {:.1}%", _ssd_eff * 100.0);
    }

    #[test]
    fn test_layout_detection() {
        let info_cxl = CxlLayoutInfo::from_block_size(64, 772);
        assert!(info_cxl.is_cxl_layout);
        assert_eq!(info_cxl.alignment, 64);

        let info_ssd = CxlLayoutInfo::from_block_size(4096, 772);
        assert!(!info_ssd.is_cxl_layout);
        assert_eq!(info_ssd.alignment, 4096);
    }

    #[test]
    fn test_large_node_multi_sector() {
        let calc = NodeOffsetCalculator::new(4096, 4096, 128, 0);
        assert_eq!(calc.raw_node_len(), 4612);

        assert_eq!(calc.node_offset(0), 0);
        assert_eq!(calc.node_offset(1), 8192);

        let cxl = NodeOffsetCalculator::new(64, 4096, 128, 0);
        assert_eq!(cxl.node_offset(0), 0);
        assert_eq!(cxl.node_offset(1), 4672);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(772, 64), 832);
    }
}
