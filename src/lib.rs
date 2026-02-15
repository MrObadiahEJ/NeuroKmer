// src/lib.rs
// Robust public library – re-export everything cleanly
// Borrowed modular style from neural-rs + WheatNNLeek

pub mod models;
pub mod utils;
pub mod spiking_hash;
pub mod associative;

pub use models::*;
pub use utils::*;
pub use spiking_hash::*;
pub use associative::*;

use log::info;

/// Top-level result type for the crate – makes error handling consistent
pub type NeuroResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Initialize logging (call once at startup)
pub fn init_logging() -> NeuroResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("NeuroKmer logging initialized");
    Ok(())
}