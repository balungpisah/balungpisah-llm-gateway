//! Context management for agent conversations.
//!
//! This module provides utilities for managing conversation context:
//!
//! - [`ContextFilter`] - Filter messages to fit context windows
//! - Message conversion utilities for TensorZero format

mod convert;
mod filter;

pub use convert::{
    convert_message_to_input, convert_messages_to_input, convert_response_to_message_content,
};
pub use filter::ContextFilter;
