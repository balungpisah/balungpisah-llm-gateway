//! SSE event types for streaming responses.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Server-Sent Event for agent responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseEvent {
    /// Stream started with IDs.
    Start {
        /// Inference ID from TensorZero.
        inference_id: Uuid,
        /// Episode ID for the conversation.
        episode_id: Uuid,
        /// Thread ID.
        thread_id: Uuid,
    },

    /// Text content delta.
    TextDelta {
        /// The text fragment.
        text: String,
    },

    /// Tool use started.
    ToolUseStart {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
    },

    /// Tool arguments delta.
    ToolUseDelta {
        /// Tool call ID.
        id: String,
        /// Arguments fragment.
        arguments: String,
    },

    /// Tool use complete.
    ToolUseComplete {
        /// Tool call ID.
        id: String,
        /// Complete arguments JSON.
        arguments: String,
    },

    /// Tool execution started.
    ToolExecutionStart {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
    },

    /// Tool execution completed.
    ToolExecutionComplete {
        /// Tool call ID.
        id: String,
        /// Tool result.
        result: String,
        /// Whether it was an error.
        is_error: bool,
    },

    /// Iteration started (for multi-turn tool loops).
    IterationStart {
        /// Current iteration number (1-indexed).
        iteration: usize,
        /// Maximum iterations.
        max_iterations: usize,
    },

    /// Iteration completed.
    IterationComplete {
        /// Current iteration number.
        iteration: usize,
    },

    /// Error occurred.
    Error {
        /// Error message.
        message: String,
        /// Error code if available.
        code: Option<String>,
    },

    /// Stream finished successfully.
    Done {
        /// Final text content (if any).
        text: Option<String>,
        /// Number of iterations completed.
        iterations: usize,
    },
}

impl SseEvent {
    /// Create a start event.
    pub fn start(inference_id: Uuid, episode_id: Uuid, thread_id: Uuid) -> Self {
        Self::Start {
            inference_id,
            episode_id,
            thread_id,
        }
    }

    /// Create a text delta event.
    pub fn text_delta(text: impl Into<String>) -> Self {
        Self::TextDelta { text: text.into() }
    }

    /// Create a tool use start event.
    pub fn tool_use_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolUseStart {
            id: id.into(),
            name: name.into(),
        }
    }

    /// Create a tool use delta event.
    pub fn tool_use_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolUseDelta {
            id: id.into(),
            arguments: arguments.into(),
        }
    }

    /// Create a tool use complete event.
    pub fn tool_use_complete(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolUseComplete {
            id: id.into(),
            arguments: arguments.into(),
        }
    }

    /// Create a tool execution start event.
    pub fn tool_execution_start(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolExecutionStart {
            id: id.into(),
            name: name.into(),
        }
    }

    /// Create a tool execution complete event.
    pub fn tool_execution_complete(
        id: impl Into<String>,
        result: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolExecutionComplete {
            id: id.into(),
            result: result.into(),
            is_error,
        }
    }

    /// Create an error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            code: None,
        }
    }

    /// Create an error event with a code.
    pub fn error_with_code(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            code: Some(code.into()),
        }
    }

    /// Create a done event.
    pub fn done(text: Option<String>, iterations: usize) -> Self {
        Self::Done { text, iterations }
    }

    /// Format as SSE data line.
    pub fn to_sse_data(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        format!("data: {}\n\n", json)
    }

    /// Check if this is a terminal event.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }
}

impl std::fmt::Display for SseEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_sse_data())
    }
}
