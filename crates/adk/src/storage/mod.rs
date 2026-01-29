//! Storage layer for persisting threads and messages.
//!
//! This module provides traits for storage operations and implementations:
//!
//! - [`ThreadStorage`] - Trait for thread CRUD operations
//! - [`MessageStorage`] - Trait for message CRUD operations
//! - [`FileMetadataStorage`] - Trait for file metadata operations
//! - [`Storage`] - Combined trait with migration support
//! - [`PostgresStorage`] - PostgreSQL implementation (requires `postgres` feature)

mod types;

#[cfg(feature = "postgres")]
mod postgres;

pub use types::{FileMetadataStorage, MessageStorage, Storage, ThreadStorage};

#[cfg(feature = "postgres")]
pub use postgres::{PostgresConfig, PostgresStorage};
