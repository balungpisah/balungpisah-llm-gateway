//! Streaming support for agent responses.
//!
//! This module provides:
//!
//! - [`SseEvent`] - Server-Sent Event types for streaming
//! - [`SseStream`] - Type alias for raw SSE string stream
//! - [`StreamConfig`] - Configuration for streaming behavior
//! - [`StreamExecutor`] - Executor for streaming responses with tool loops

use futures::Stream;
use std::pin::Pin;

mod config;
mod events;
mod executor;

pub use config::StreamConfig;
pub use events::{generate_block_id, generate_message_id, SseError, SseEvent};
pub use executor::StreamExecutor;

/// Error type for stream operations
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("Stream error: {0}")]
    Stream(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Type alias for SSE stream - returns raw SSE-formatted strings
///
/// Each item is a complete SSE event string like:
/// ```text
/// event: block.delta
/// data: {"message_id":"msg_...","block_id":"block_...","delta":{"text":"Hello"}}
///
/// ```
pub type SseStream = Pin<Box<dyn Stream<Item = Result<String, StreamError>> + Send>>;
