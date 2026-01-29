//! Agent module containing the core Agent struct and builder.
//!
//! - [`Agent`] - The main agent struct for conversations
//! - [`AgentBuilder`] - Builder with compile-time validation
//! - [`ModelSpec`] - Model specification (function or model name)

mod builder;
mod core;

pub use builder::{AgentBuilder, ModelSpec};
pub use core::{Agent, ChatRequest, ChatResponse, ChatStreamResponse, Usage};
