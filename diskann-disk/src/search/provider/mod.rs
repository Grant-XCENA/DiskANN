/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk vertex providers and related functionality.
//!
//! This module contains providers for reading vertex data from disk,
//! including cached providers, sector graphs, and factory patterns.

pub mod cached_disk_vertex_provider;
pub mod disk_provider;
pub mod disk_sector_graph;
pub mod disk_vertex_provider;
pub mod disk_vertex_provider_factory;

cfg_if::cfg_if! {
    if #[cfg(feature = "cxl")] {
        pub mod cxl_sector_graph;
        pub mod cxl_vertex_provider;
    }
}
