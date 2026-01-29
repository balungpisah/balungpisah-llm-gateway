//! Response types for TensorZero inference.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Response from a chat inference request.
#[derive(Debug, Clone, Deserialize)]
pub struct InferenceResponse {
    /// Unique inference ID.
    pub inference_id: Uuid,
    /// Episode ID for conversation tracking.
    pub episode_id: Uuid,
    /// Response content blocks.
    pub content: Vec<ContentBlock>,
    /// Usage statistics.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl InferenceResponse {
    /// Get the text content from the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool calls from the response.
    pub fn tool_calls(&self) -> Vec<&ToolCallBlock> {
        self.content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolCall(tc) = block {
                    Some(tc)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if the response contains any tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    }
}

/// Content block in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text { text: String },
    /// Tool call from the model.
    ToolCall(ToolCallBlock),
}

/// A tool call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallBlock {
    /// Unique ID for this tool call.
    pub id: String,
    /// Validated name of the tool (null if invalid).
    #[serde(default)]
    pub name: Option<String>,
    /// Validated arguments as parsed JSON (null if invalid).
    #[serde(default)]
    pub arguments: Option<Value>,
    /// Raw name from the model (before normalization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_name: Option<String>,
    /// Raw arguments from the model as a string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

impl ToolCallBlock {
    /// Get the tool name, preferring validated name over raw_name.
    pub fn tool_name(&self) -> Option<&str> {
        self.name.as_deref().or(self.raw_name.as_deref())
    }

    /// Get the arguments as a Value.
    ///
    /// Returns the validated arguments if available, otherwise tries to parse raw_arguments.
    pub fn parse_arguments(&self) -> Result<Value, serde_json::Error> {
        if let Some(ref args) = self.arguments {
            Ok(args.clone())
        } else if let Some(ref raw) = self.raw_arguments {
            serde_json::from_str(raw)
        } else {
            Ok(Value::Null)
        }
    }

    /// Get the arguments string (for use in input messages).
    ///
    /// Returns raw_arguments if available, otherwise serializes the validated arguments.
    pub fn arguments_string(&self) -> String {
        if let Some(ref raw) = self.raw_arguments {
            raw.clone()
        } else if let Some(ref args) = self.arguments {
            serde_json::to_string(args).unwrap_or_default()
        } else {
            "{}".to_string()
        }
    }

    /// Parse the arguments into a specific type.
    pub fn parse_arguments_as<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, serde_json::Error> {
        let args = self.parse_arguments()?;
        serde_json::from_value(args)
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Usage {
    /// Number of input tokens.
    #[serde(default)]
    pub input_tokens: u32,
    /// Number of output tokens.
    #[serde(default)]
    pub output_tokens: u32,
}

impl Usage {
    /// Get total tokens used.
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// Response from a JSON inference request.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonInferenceResponse {
    /// Unique inference ID.
    pub inference_id: Uuid,
    /// Episode ID for conversation tracking.
    pub episode_id: Uuid,
    /// The parsed JSON output.
    pub output: Output,
    /// Usage statistics.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Output from JSON inference.
#[derive(Debug, Clone, Deserialize)]
pub struct Output {
    /// Raw JSON string from the model.
    pub raw: String,
    /// Parsed JSON value.
    pub parsed: Option<Value>,
}

impl JsonInferenceResponse {
    /// Get the parsed output value.
    pub fn parsed(&self) -> Option<&Value> {
        self.output.parsed.as_ref()
    }

    /// Parse the output into a specific type.
    pub fn parse_as<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        match &self.output.parsed {
            Some(value) => serde_json::from_value(value.clone()),
            None => serde_json::from_str(&self.output.raw),
        }
    }
}

/// Streaming chunk from TensorZero.
///
/// TensorZero sends each SSE event as a complete JSON chunk containing
/// inference_id, episode_id, variant_name, content array, and optionally usage.
/// This is simpler than Anthropic's content_block_start/delta/stop pattern.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamEvent {
    /// Inference ID for this request.
    pub inference_id: Uuid,
    /// Episode ID for conversation tracking.
    pub episode_id: Uuid,
    /// Variant name used for inference.
    pub variant_name: String,
    /// Content blocks (deltas) in this chunk.
    pub content: Vec<StreamContentBlock>,
    /// Usage statistics (usually present in final chunks).
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Content block in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlock {
    /// Text content delta.
    Text {
        /// Block ID.
        id: String,
        /// Text delta.
        text: String,
    },
    /// Tool call delta.
    ToolCall {
        /// Tool call ID.
        id: String,
        /// Raw name of the tool (may be partial).
        raw_name: String,
        /// Raw arguments string delta.
        raw_arguments: String,
    },
    /// Thought content delta (from reasoning models).
    Thought {
        /// Block ID.
        id: String,
        /// Thought text delta.
        #[serde(default)]
        text: Option<String>,
        /// Signature (usually comes at the end).
        #[serde(default)]
        signature: Option<String>,
    },
}

impl StreamContentBlock {
    /// Get the block ID.
    pub fn id(&self) -> &str {
        match self {
            StreamContentBlock::Text { id, .. } => id,
            StreamContentBlock::ToolCall { id, .. } => id,
            StreamContentBlock::Thought { id, .. } => id,
        }
    }

    /// Get the block type as a string.
    pub fn block_type(&self) -> &'static str {
        match self {
            StreamContentBlock::Text { .. } => "text",
            StreamContentBlock::ToolCall { .. } => "tool_call",
            StreamContentBlock::Thought { .. } => "thought",
        }
    }
}

// Legacy type aliases for backward compatibility during migration
// TODO: Remove these after updating executor.rs

/// Legacy: Initial chunk event (now just StreamEvent).
pub type ChunkEvent = StreamEvent;

/// Legacy: Content block start event.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStartEvent {
    /// Index of this content block.
    pub index: usize,
    /// Type of content block.
    pub content_block: ContentBlockType,
}

/// Legacy: Type of content block being streamed.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockType {
    /// Text content block.
    Text { text: String },
    /// Tool call content block.
    ToolCall(StreamToolCall),
}

/// Legacy: Tool call in streaming response.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamToolCall {
    /// Unique ID for this tool call.
    pub id: String,
    /// Raw name of the tool.
    #[serde(default)]
    pub raw_name: Option<String>,
    /// Raw arguments as a string.
    #[serde(default)]
    pub raw_arguments: Option<String>,
    /// Name of the tool (for compatibility).
    #[serde(default)]
    pub name: Option<String>,
    /// Arguments (for compatibility).
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Legacy: Delta update for a content block.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockDeltaEvent {
    /// Index of the content block being updated.
    pub index: usize,
    /// The delta content.
    pub delta: Delta,
}

/// Legacy: Delta content types.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    /// Text delta.
    TextDelta { text: String },
    /// Tool call arguments delta.
    ToolCallDelta { arguments: String },
}

/// Legacy: End of a content block.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStopEvent {
    /// Index of the finished content block.
    pub index: usize,
}

/// Legacy: Stream finished event.
#[derive(Debug, Clone, Deserialize)]
pub struct DoneEvent {
    /// Usage statistics for the complete response.
    #[serde(default)]
    pub usage: Option<Usage>,
}
