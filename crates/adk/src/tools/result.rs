//! Tool execution results.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID of the tool call this result is for.
    pub tool_call_id: String,
    /// Name of the tool that was called.
    pub tool_name: String,
    /// The result content.
    pub content: String,
    /// Whether the execution was an error.
    pub is_error: bool,
    /// Optional structured data in the result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolResult {
    /// Create a successful result with string content.
    pub fn success(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: content.into(),
            is_error: false,
            data: None,
        }
    }

    /// Create a successful result with JSON data.
    pub fn success_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        data: Value,
    ) -> Self {
        let content = serde_json::to_string_pretty(&data).unwrap_or_else(|_| data.to_string());
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            is_error: false,
            data: Some(data),
        }
    }

    /// Create an error result.
    pub fn error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: error.into(),
            is_error: true,
            data: None,
        }
    }

    /// Create an error result from an Error trait object.
    pub fn from_error<E: std::error::Error>(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: E,
    ) -> Self {
        Self::error(tool_call_id, tool_name, error.to_string())
    }

    /// Add structured data to the result.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Check if this is a successful result.
    pub fn is_success(&self) -> bool {
        !self.is_error
    }

    /// Get the data as a specific type.
    pub fn data_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        self.data
            .as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Extension trait for converting Result types to ToolResult.
pub trait IntoToolResult {
    /// Convert to a ToolResult.
    fn into_tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> ToolResult;
}

impl<T, E> IntoToolResult for Result<T, E>
where
    T: std::fmt::Display,
    E: std::error::Error,
{
    fn into_tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> ToolResult {
        match self {
            Ok(value) => ToolResult::success(tool_call_id, tool_name, value.to_string()),
            Err(error) => ToolResult::from_error(tool_call_id, tool_name, error),
        }
    }
}

impl IntoToolResult for Value {
    fn into_tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> ToolResult {
        ToolResult::success_json(tool_call_id, tool_name, self)
    }
}

impl IntoToolResult for String {
    fn into_tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> ToolResult {
        ToolResult::success(tool_call_id, tool_name, self)
    }
}

impl IntoToolResult for &str {
    fn into_tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> ToolResult {
        ToolResult::success(tool_call_id, tool_name, self)
    }
}
