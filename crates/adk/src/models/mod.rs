//! Data models for the Agent Development Kit.
//!
//! This module contains the core data structures used throughout the SDK:
//!
//! - [`Thread`] - A conversation thread that groups related messages
//! - [`Message`] - A single message in a thread
//! - [`MessageContent`] and [`ContentBlock`] - Message content types
//! - [`FileMetadata`] - Metadata for files stored in object storage

mod content;
mod file_metadata;
mod message;
mod thread;

pub use content::{ContentBlock, ImageSource, MessageContent};
pub use file_metadata::FileMetadata;
pub use message::{Message, Role, ToolResultMessageBuilder};
pub use thread::{Thread, ThreadOptions};
