//! Data models for the Agent Development Kit.
//!
//! This module contains the core data structures used throughout the SDK:
//!
//! - [`Thread`] - A conversation thread that groups related messages
//! - [`Message`] - A single message in a thread
//! - [`MessageContent`] and [`ContentBlock`] - Message content types

mod content;
mod message;
mod thread;

pub use content::{ContentBlock, ImageSource, MessageContent};
pub use message::{Message, Role, ToolResultMessageBuilder};
pub use thread::{Thread, ThreadOptions};
