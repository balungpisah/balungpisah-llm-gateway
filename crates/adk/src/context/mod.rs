//! Context management for agent conversations.
//!
//! This module provides utilities for managing conversation context:
//!
//! - [`ContextConfig`] - Context configuration with message limits and tool handling
//! - [`ToolsConfig`] - Configuration for tool message handling (retain_last, limit_per_message, deduplicate)
//! - Message conversion utilities for TensorZero format

mod config;
mod convert;

pub use config::{ContextConfig, ToolsConfig};
pub use convert::{
    convert_message_to_input, convert_messages_to_input, convert_response_to_message_content,
};
