//! Context management for agent conversations.
//!
//! This module provides utilities for managing conversation context:
//!
//! - [`ContextConfig`] - High-level context configuration with tool handling
//! - [`ToolsContextConfig`] - Configuration for tool message handling
//! - [`ContextFilter`] - Low-level message filtering
//! - Message conversion utilities for TensorZero format

mod config;
mod convert;
mod filter;

pub use config::{ContextConfig, ToolsContextConfig};
pub use convert::{
    convert_message_to_input, convert_messages_to_input, convert_response_to_message_content,
};
pub use filter::ContextFilter;
