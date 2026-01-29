//! Streaming support for agent responses.
//!
//! This module provides:
//!
//! - [`SseEvent`] - Server-Sent Event types for streaming
//! - [`StreamConfig`] - Configuration for streaming behavior
//! - [`StreamExecutor`] - Executor for streaming responses with tool loops

mod config;
mod events;
mod executor;

pub use config::StreamConfig;
pub use events::SseEvent;
pub use executor::StreamExecutor;
