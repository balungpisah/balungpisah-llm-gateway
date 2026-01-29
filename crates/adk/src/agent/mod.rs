//! Agent module containing the core Agent struct and builder.
//!
//! - [`Agent`] - The main agent struct for conversations
//! - [`AgentBuilder`] - Builder with compile-time validation

mod builder;
mod core;

pub use builder::AgentBuilder;
pub use core::{Agent, ChatRequest, ChatResponse, Usage};
