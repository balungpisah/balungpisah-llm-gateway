//! SSE Event types for streaming agent responses
//!
//! This module provides strongly-typed SSE events that follow the
//! Server-Sent Events specification for real-time agent communication.
//!
//! ## Improved SSE Response Format v2
//!
//! This implements a frontend-first streaming design with:
//! - Block-based architecture with unique IDs
//! - Self-contained events with full context
//! - Clear lifecycle tracking (streaming → executing → done)
//! - Tool call transparency for real-time progress
//! - Comprehensive error handling

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// SSE Event wrapper for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    /// Event type identifier (e.g., "message.started", "block.delta")
    pub event: String,
    /// Agent ID that generated this event
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Event payload data
    pub data: serde_json::Value,
}

impl SseEvent {
    /// Create a new SSE event
    pub fn new(event: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            event: event.into(),
            agent_id: None,
            data,
        }
    }

    /// Set agent_id for this event
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Convert to SSE-formatted string
    pub fn to_sse_string(&self) -> Result<String, SseError> {
        let data_str = serde_json::to_string(&self.data)
            .map_err(|e| SseError::SerializationError(e.to_string()))?;

        let mut output = format!("event: {}\n", self.event);
        if let Some(ref agent_id) = self.agent_id {
            output.push_str(&format!("id: {}\n", agent_id));
        }
        output.push_str(&format!("data: {}\n\n", data_str));

        Ok(output)
    }

    /// Check if this is a terminal event.
    pub fn is_terminal(&self) -> bool {
        self.event == "message.completed" || self.event == "error"
    }

    // ==================== Message Lifecycle Events ====================

    /// Message started event (v2 format)
    pub fn message_started(
        message_id: String,
        thread_id: String,
        role: String,
        model: String,
    ) -> Self {
        Self::new(
            "message.started",
            serde_json::json!({
                "message_id": message_id,
                "thread_id": thread_id,
                "role": role,
                "model": model,
                "timestamp": Utc::now().to_rfc3339(),
            }),
        )
    }

    /// Message usage event (token counting)
    pub fn message_usage(message_id: String, input_tokens: u32, output_tokens: u32) -> Self {
        Self::new(
            "message.usage",
            serde_json::json!({
                "message_id": message_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }),
        )
    }

    /// Message completed event
    pub fn message_completed(
        message_id: String,
        thread_id: String,
        total_blocks: usize,
        finish_reason: String,
    ) -> Self {
        Self::new(
            "message.completed",
            serde_json::json!({
                "message_id": message_id,
                "thread_id": thread_id,
                "total_blocks": total_blocks,
                "finish_reason": finish_reason,
                "timestamp": Utc::now().to_rfc3339(),
            }),
        )
    }

    // ==================== Block Lifecycle Events ====================

    /// Block created event - text block
    pub fn block_created_text(message_id: String, block_id: String, index: usize) -> Self {
        Self::new(
            "block.created",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "text",
                "index": index,
            }),
        )
    }

    /// Block created event - thought block
    pub fn block_created_thought(message_id: String, block_id: String, index: usize) -> Self {
        Self::new(
            "block.created",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "thought",
                "index": index,
            }),
        )
    }

    /// Block created event - tool call block
    pub fn block_created_tool_call(
        message_id: String,
        block_id: String,
        index: usize,
        tool_name: String,
        tool_call_id: String,
    ) -> Self {
        Self::new(
            "block.created",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "tool_call",
                "index": index,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
            }),
        )
    }

    /// Block created event - tool result block
    pub fn block_created_tool_result(
        message_id: String,
        block_id: String,
        index: usize,
        tool_call_id: String,
        tool_name: String,
    ) -> Self {
        Self::new(
            "block.created",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "tool_result",
                "index": index,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            }),
        )
    }

    /// Block delta event - text content
    pub fn block_delta_text(message_id: String, block_id: String, text: String) -> Self {
        Self::new(
            "block.delta",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "text",
                "delta": { "text": text }
            }),
        )
    }

    /// Block delta event - thought content (with optional signature)
    pub fn block_delta_thought(
        message_id: String,
        block_id: String,
        text: Option<String>,
        signature: Option<String>,
    ) -> Self {
        let mut delta = serde_json::json!({});
        if let Some(text_value) = text {
            delta["text"] = serde_json::json!(text_value);
        }
        if let Some(sig_value) = signature {
            delta["signature"] = serde_json::json!(sig_value);
        }

        Self::new(
            "block.delta",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "thought",
                "delta": delta
            }),
        )
    }

    /// Block delta event - tool call arguments (with partial accumulation)
    pub fn block_delta_tool_call(
        message_id: String,
        block_id: String,
        tool_name: String,
        tool_call_id: String,
        arguments_delta: String,
        partial_arguments: String,
    ) -> Self {
        Self::new(
            "block.delta",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "tool_call",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "delta": { "arguments": arguments_delta },
                "partial_arguments": partial_arguments,
            }),
        )
    }

    /// Block delta event - tool result (execution result or error)
    pub fn block_delta_tool_result(
        message_id: String,
        block_id: String,
        tool_call_id: String,
        tool_name: String,
        result: Option<serde_json::Value>,
        error: Option<String>,
    ) -> Self {
        let mut delta = serde_json::json!({});

        if let Some(result_value) = result {
            delta["result"] = result_value;
            delta["success"] = serde_json::json!(true);
        } else if let Some(error_msg) = error {
            delta["error"] = serde_json::json!(error_msg);
            delta["success"] = serde_json::json!(false);
        }

        Self::new(
            "block.delta",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "tool_result",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "delta": delta,
            }),
        )
    }

    /// Block completed event - text block
    pub fn block_completed_text(
        message_id: String,
        block_id: String,
        final_content: String,
    ) -> Self {
        Self::new(
            "block.completed",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "text",
                "final_content": final_content,
            }),
        )
    }

    /// Block completed event - thought block (with optional signature)
    pub fn block_completed_thought(
        message_id: String,
        block_id: String,
        final_content: String,
        signature: Option<String>,
    ) -> Self {
        let mut data = serde_json::json!({
            "message_id": message_id,
            "block_id": block_id,
            "block_type": "thought",
            "final_content": final_content,
        });

        if let Some(sig_value) = signature {
            data["signature"] = serde_json::json!(sig_value);
        }

        Self::new("block.completed", data)
    }

    /// Block completed event - tool call block (arguments only, no result)
    pub fn block_completed_tool_call(
        message_id: String,
        block_id: String,
        tool_name: String,
        tool_call_id: String,
        final_arguments: String,
        parsed_arguments: serde_json::Value,
    ) -> Self {
        Self::new(
            "block.completed",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": "tool_call",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "final_arguments": final_arguments,
                "parsed_arguments": parsed_arguments,
            }),
        )
    }

    /// Block completed event - tool result block (with execution result)
    pub fn block_completed_tool_result(
        message_id: String,
        block_id: String,
        tool_call_id: String,
        tool_name: String,
        result: Option<serde_json::Value>,
        error: Option<String>,
        execution_time_ms: u64,
    ) -> Self {
        let mut data = serde_json::json!({
            "message_id": message_id,
            "block_id": block_id,
            "block_type": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "execution_time_ms": execution_time_ms,
        });

        // Add result or error
        if let Some(result_value) = result {
            data["success"] = serde_json::json!(true);
            data["result"] = result_value;
        } else if let Some(error_msg) = error {
            data["success"] = serde_json::json!(false);
            data["error"] = serde_json::json!(error_msg);
        }

        Self::new("block.completed", data)
    }

    /// Block error event
    pub fn block_error(
        message_id: String,
        block_id: String,
        block_type: String,
        error_code: String,
        error_message: String,
        error_details: Option<String>,
    ) -> Self {
        let mut error_obj = serde_json::json!({
            "code": error_code,
            "message": error_message,
        });

        if let Some(details) = error_details {
            error_obj["details"] = serde_json::json!(details);
        }

        Self::new(
            "block.error",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "block_type": block_type,
                "error": error_obj,
            }),
        )
    }

    // ==================== Tool Execution Events ====================

    /// Tool execution started event
    pub fn tool_execution_started(
        message_id: String,
        block_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    ) -> Self {
        Self::new(
            "tool.execution_started",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "started_at": Utc::now().to_rfc3339(),
            }),
        )
    }

    /// Tool execution completed event (success)
    pub fn tool_execution_completed(
        message_id: String,
        block_id: String,
        tool_call_id: String,
        tool_name: String,
        result: serde_json::Value,
        execution_time_ms: u64,
        _started_at: DateTime<Utc>,
    ) -> Self {
        Self::new(
            "tool.execution_completed",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "success": true,
                "result": result,
                "execution_time_ms": execution_time_ms,
                "completed_at": Utc::now().to_rfc3339(),
            }),
        )
    }

    /// Tool execution failed event (error)
    #[allow(clippy::too_many_arguments)]
    pub fn tool_execution_failed(
        message_id: String,
        block_id: String,
        tool_call_id: String,
        tool_name: String,
        error_code: String,
        error_message: String,
        error_details: Option<String>,
        execution_time_ms: u64,
    ) -> Self {
        let mut error_obj = serde_json::json!({
            "code": error_code,
            "message": error_message,
        });

        if let Some(details) = error_details {
            error_obj["details"] = serde_json::json!(details);
        }

        Self::new(
            "tool.execution_failed",
            serde_json::json!({
                "message_id": message_id,
                "block_id": block_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "success": false,
                "error": error_obj,
                "execution_time_ms": execution_time_ms,
                "failed_at": Utc::now().to_rfc3339(),
            }),
        )
    }

    // ==================== Utility Events ====================

    /// Generic error event
    pub fn error(error_type: String, message: String) -> Self {
        Self::new(
            "error",
            serde_json::json!({
                "type": error_type,
                "message": message,
            }),
        )
    }

    /// Agent delegation start event (for multi-agent systems)
    pub fn agent_delegation_start(
        from_agent: impl Into<String>,
        to_agent: impl Into<String>,
        request: impl Into<String>,
    ) -> Self {
        let from = from_agent.into();
        Self::new(
            "agent.delegation_started",
            serde_json::json!({
                "from": from.clone(),
                "to": to_agent.into(),
                "request": request.into(),
                "timestamp": Utc::now().to_rfc3339(),
            }),
        )
        .with_agent_id(from)
    }

    /// Agent delegation complete event
    pub fn agent_delegation_complete(
        agent_id: impl Into<String>,
        result_summary: impl Into<String>,
        message_count: usize,
    ) -> Self {
        let id = agent_id.into();
        Self::new(
            "agent.delegation_completed",
            serde_json::json!({
                "agent_id": id.clone(),
                "result_summary": result_summary.into(),
                "message_count": message_count,
                "timestamp": Utc::now().to_rfc3339(),
            }),
        )
        .with_agent_id(id)
    }
}

/// Helper to generate unique block IDs
pub fn generate_block_id() -> String {
    format!("block_{}", Uuid::new_v4())
}

/// Helper to generate unique message IDs
pub fn generate_message_id() -> String {
    format!("msg_{}", Uuid::new_v4())
}

/// SSE-specific errors
#[derive(Debug, thiserror::Error)]
pub enum SseError {
    #[error("Failed to serialize SSE data: {0}")]
    SerializationError(String),
}

impl fmt::Display for SseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = format!("event: {}\n", self.event);
        if let Some(ref agent_id) = self.agent_id {
            output.push_str(&format!("id: {}\n", agent_id));
        }
        output.push_str(&format!("data: {}\n\n", self.data));
        write!(f, "{}", output)
    }
}
