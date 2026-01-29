//! Tool execution context.

use crate::models::Thread;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Context provided to tool executors during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// ID of the tool call being executed.
    pub tool_call_id: String,
    /// Name of the tool being executed.
    pub tool_name: String,
    /// Thread the tool is being executed in.
    pub thread: Arc<Thread>,
    /// Custom data that can be passed to tools.
    pub data: HashMap<String, Value>,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        thread: Arc<Thread>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            thread,
            data: HashMap::new(),
        }
    }

    /// Add custom data to the context.
    pub fn with_data(mut self, key: impl Into<String>, value: Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    /// Get custom data from the context.
    pub fn get_data(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Get typed custom data from the context.
    pub fn get_data_as<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.data
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get the thread ID.
    pub fn thread_id(&self) -> Uuid {
        self.thread.id
    }

    /// Get the external ID from the thread.
    pub fn external_id(&self) -> &str {
        &self.thread.external_id
    }
}

/// Builder for tool context.
#[derive(Debug)]
pub struct ToolContextBuilder {
    tool_call_id: String,
    tool_name: String,
    thread: Option<Arc<Thread>>,
    data: HashMap<String, Value>,
}

impl ToolContextBuilder {
    /// Create a new builder.
    pub fn new(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            thread: None,
            data: HashMap::new(),
        }
    }

    /// Set the thread.
    pub fn thread(mut self, thread: Arc<Thread>) -> Self {
        self.thread = Some(thread);
        self
    }

    /// Add custom data.
    pub fn data(mut self, key: impl Into<String>, value: Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    /// Build the context.
    pub fn build(self) -> Option<ToolContext> {
        Some(ToolContext {
            tool_call_id: self.tool_call_id,
            tool_name: self.tool_name,
            thread: self.thread?,
            data: self.data,
        })
    }
}
