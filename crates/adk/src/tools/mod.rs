//! Tool system for extending agent capabilities.
//!
//! This module provides the infrastructure for defining and executing tools:
//!
//! - [`ToolDefinition`] - Describes a tool's name, description, and parameters
//! - [`ToolExecutor`] - Trait for implementing tool execution logic
//! - [`ToolContext`] - Context provided during tool execution
//! - [`ToolResult`] - Result of tool execution
//! - [`ToolRegistry`] - Collection of available tools
//!
//! # Example
//!
//! ```
//! use balungpisah_adk::tools::{ToolDefinition, FnToolExecutor, ToolResult, ToolContext};
//! use serde_json::{json, Value};
//!
//! // Define a tool
//! let definition = ToolDefinition::builder("get_weather")
//!     .description("Get the current weather for a location")
//!     .string_param("location", "The city and state, e.g. San Francisco, CA")
//!     .build();
//!
//! // Create an executor
//! let executor = FnToolExecutor::new(definition, |args: Value, ctx: ToolContext| async move {
//!     let location = args.get("location")
//!         .and_then(|v| v.as_str())
//!         .unwrap_or("unknown");
//!
//!     ToolResult::success(&ctx.tool_call_id, &ctx.tool_name, format!("Weather in {}: Sunny, 72°F", location))
//! });
//! ```

mod context;
mod definition;
mod executor;
mod result;

pub use context::{ToolContext, ToolContextBuilder};
pub use definition::{ToolDefinition, ToolDefinitionBuilder};
pub use executor::{FnToolExecutor, ToolExecutor, ToolFn, ToolRegistry};
pub use result::{IntoToolResult, ToolResult};
