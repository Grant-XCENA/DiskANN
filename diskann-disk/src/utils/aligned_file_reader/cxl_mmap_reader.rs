//! CXL Memory-Mapped Aligned File Reader
//!
//! Option 3 implementation: replaces io_uring with direct mmap access for CXL
//! Type-3 memory expanders. CXL provides byte-addressable load/store at ~200-400ns,
//! eliminating need for O_DIRECT, io_uring, and 512-byte alignment.
//!
//! Mount CXL device as DAX filesystem before use:
//!   daxctl reconfigure-device --mode=devdax dax0.0
//!   mkfs.ext4 /dev/dax0.0
//!   mount -o dax=always /dev/dax0.0 /mnt/cxl

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use memmap2::Mmap;

//use crate::stubs::{A64, AlignedFileReader, AlignedRead, ANNResult};
use diskann::{ANNError, ANNResult};
//use diskann::error::ann_error::ANNResult;
use crate::utils::aligned_file_reader::traits::AlignedFileReader;
use crate::utils::aligned_file_reader::aligned_read::{A64, AlignedRead};

/// Cache-line alignment constant (64 bytes).
pub const CXL_CACHE_LINE: usize = 64;

/// CXL mmap-based file reader.
///
/// Maps entire index file into virtual address space via DAX-mounted CXL device.
/// Reads become direct memory loads — no syscalls, no io_uring, no kernel copies.
///
/// # Performance characteristics
/// - Read latency: ~200-400ns per cache line (vs ~70-100μs for NVMe)
/// - No alignment constraints beyond cache line (64 bytes)
/// - No async I/O overhead — synchronous memcpy
/// - Supports software prefetching to hide CXL latency
pub struct CxlMmapReader {
    /// Read-only memory map of the index file
    mmap: Mmap,
    /// I/O operation counter (per-node reads, not cache-line accesses)
    io_count: AtomicU32,
}

/// Configuration for CXL mmap reader behavior
#[derive(Debug, Clone)]
pub struct CxlMmapConfig {
    /// Use MADV_WILLNEED to prefault all pages at mmap time.
    pub prefault: bool,
    /// madvise MADV_RANDOM hint (correct for graph traversal).
    pub advise_random: bool,
    /// Enable MADV_HUGEPAGE for 2MB pages (reduces TLB misses).
    pub use_huge_pages: bool,
}

impl Default for CxlMmapConfig {
    fn default() -> Self {
        Self {
            prefault: true,
            advise_random: true,
            use_huge_pages: true,
        }
    }
}

impl CxlMmapReader {
    /// Create a new CXL mmap reader with default configuration.
    pub fn new<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        Self::with_config(path, CxlMmapConfig::default())
    }

    /// Create a new CXL mmap reader with custom configuration.
    pub fn with_config<P: AsRef<Path>>(path: P, config: CxlMmapConfig) -> ANNResult<Self> {
        /*let file = File::open(path.as_ref()).map_err(|e| {
            anyhow::anyhow!(
                "Failed to open index file '{}': {}. \
                 Ensure file exists on DAX-mounted CXL device \
                 (mount -o dax=always /dev/daxN.M /mnt/cxl)",
                path.as_ref().display(),
                e
            )
        })?;
*/
        let file = File::open(path.as_ref()).map_err(ANNError::log_io_error)?;

        let _file_len = file.metadata()?.len() as usize;

        // Read-only mmap — safe, no UB from concurrent writes since
        // CXL index files are immutable after build.
        /*let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to mmap index file (size={}): {}. \
                     Check CXL device capacity and DAX mount status.",
                    file_len,
                    e
                )
            })?
        };*/

        let mmap = unsafe {
            Mmap::map(&file).map_err(ANNError::log_io_error)?
        };

        // Apply madvise hints for access pattern optimization
        unsafe {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();

            if config.advise_random {
                libc::madvise(ptr, len, libc::MADV_RANDOM);
            }

            if config.use_huge_pages {
                libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
            }

            if config.prefault {
                libc::madvise(ptr, len, libc::MADV_WILLNEED);
            }
        }

        Ok(Self {
            mmap,
            io_count: AtomicU32::new(0),
        })
    }

    /// Get raw pointer to mapped memory at given offset.
    ///
    /// # Safety
    /// Caller must ensure offset does not exceed mapped region size.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, offset: usize) -> *const u8 {
        debug_assert!(offset < self.mmap.len(), "CXL mmap read out of bounds");
        self.mmap.as_ptr().add(offset)
    }

    /// Get a slice view into the mapped memory — zero-copy access.
    ///
    /// # Safety
    /// Caller must ensure offset..offset+len is within bounds and
    /// the returned slice is not used after the reader is dropped.
    #[inline(always)]
    pub unsafe fn slice_at(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(
            offset + len <= self.mmap.len(),
            "CXL mmap slice out of bounds: offset={} len={} total={}",
            offset,
            len,
            self.mmap.len()
        );
        std::slice::from_raw_parts(self.mmap.as_ptr().add(offset), len)
    }

    /// Software prefetch a node from CXL memory into CPU cache.
    #[inline(always)]
    pub fn prefetch_node(&self, offset: usize, node_size_bytes: usize) {
        if offset + node_size_bytes > self.mmap.len() {
            return; // Silently skip out-of-bounds prefetch
        }

        let num_cache_lines = (node_size_bytes + CXL_CACHE_LINE - 1) / CXL_CACHE_LINE;

        unsafe {
            let base = self.mmap.as_ptr().add(offset);
            for cl in 0..num_cache_lines {
                let addr = base.add(cl * CXL_CACHE_LINE);
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

    /// Total I/O operations (node-level reads) since creation.
    pub fn io_operations(&self) -> u32 {
        self.io_count.load(Ordering::Relaxed)
    }

    /// Reset I/O counter.
    pub fn reset_io_count(&self) {
        self.io_count.store(0, Ordering::Relaxed);
    }

    /// Size of the mapped region in bytes.
    pub fn mapped_len(&self) -> usize {
        self.mmap.len()
    }
}

impl AlignedFileReader for CxlMmapReader {
    type Alignment = A64;

    fn read(&mut self, read_requests: &mut [AlignedRead<u8, A64>]) -> ANNResult<()> {
        for req in read_requests.iter_mut() {
            let offset = req.offset() as usize;
            let buf = req.aligned_buf_mut();
            let len = buf.len();

            if offset + len > self.mmap.len() {
                return Err(ANNError::log_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("CXL mmap read out of bounds: offset={} len={} mapped_len={}", offset, len, self.mmap.len()),
                )));
                /*return Err(anyhow::anyhow!(
                    "CXL mmap read out of bounds: offset={} len={} mapped_len={}",
                    offset,
                    len,
                    self.mmap.len()
                ));*/
            }

            unsafe {
                let src = self.mmap.as_ptr().add(offset);
                std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), len);
            }

            self.io_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Reader type discriminant for factory selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CxlReaderType {
    MmapRandom,
    MmapSequential,
}

#[cfg(test)]
mod tests {
    use super::*;
//    use crate::stubs::AlignedFileReader;
    use crate::utils::aligned_file_reader::traits::AlignedFileReader;

    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_basic_mmap_read() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let file = create_test_file(&data);

        let reader = CxlMmapReader::with_config(
            file.path(),
            CxlMmapConfig {
                prefault: false,
                advise_random: true,
                use_huge_pages: false,
            },
        )
        .unwrap();

        assert_eq!(reader.mapped_len(), 4096);
        assert_eq!(reader.io_operations(), 0);

        unsafe {
            let slice = reader.slice_at(0, 256);
            for i in 0..256 {
                assert_eq!(slice[i], i as u8);
            }
        }
    }

    #[test]
    fn test_aligned_file_reader_trait() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let file = create_test_file(&data);

        let mut reader = CxlMmapReader::with_config(
            file.path(),
            CxlMmapConfig {
                prefault: false,
                advise_random: true,
                use_huge_pages: false,
            },
        )
        .unwrap();
        
        let layout = std::alloc::Layout::from_size_align(128, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let mut buf = unsafe { std::slice::from_raw_parts_mut(ptr, 128) };
        let mut requests = vec![AlignedRead::<u8, A64>::new(0, &mut buf).unwrap()];
        reader.read(&mut requests).unwrap();
        let buf = requests[0].aligned_buf();
       
        for i in 0..128 {
            assert_eq!(buf[i], i as u8);
        }
    }

    #[test]
    fn test_prefetch_no_panic() {
        let data = vec![0u8; 4096];
        let file = create_test_file(&data);

        let reader = CxlMmapReader::with_config(
            file.path(),
            CxlMmapConfig {
                prefault: false,
                advise_random: true,
                use_huge_pages: false,
            },
        )
        .unwrap();

        reader.prefetch_node(0, 960);
        reader.prefetch_node(3000, 960);
        reader.prefetch_node(5000, 960); // Beyond file — silently skipped
    }

    #[test]
    fn test_io_counter() {
        let data = vec![0u8; 4096];
        let file = create_test_file(&data);

        let reader = CxlMmapReader::with_config(
            file.path(),
            CxlMmapConfig {
                prefault: false,
                advise_random: true,
                use_huge_pages: false,
            },
        )
        .unwrap();

        assert_eq!(reader.io_operations(), 0);
        reader.reset_io_count();
        assert_eq!(reader.io_operations(), 0);
    }

    #[test]
    fn test_multiple_reads_at_offsets() {
        let node_size = 960;
        let total = node_size * 3;
        let mut data = vec![0u8; total];
        for i in 0..3 {
            data[i * node_size] = (i + 1) as u8;
        }

        let file = create_test_file(&data);
        let reader = CxlMmapReader::with_config(
            file.path(),
            CxlMmapConfig {
                prefault: false,
                advise_random: true,
                use_huge_pages: false,
            },
        )
        .unwrap();

        unsafe {
            assert_eq!(reader.slice_at(0, 1)[0], 1);
            assert_eq!(reader.slice_at(960, 1)[0], 2);
            assert_eq!(reader.slice_at(1920, 1)[0], 3);
        }
    }
}
