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
    /// Name of the tool to call.
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
    /// Raw name from the model (before normalization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_name: Option<String>,
    /// Raw arguments from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

impl ToolCallBlock {
    /// Parse the arguments as JSON.
    pub fn parse_arguments(&self) -> Result<Value, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }

    /// Parse the arguments into a specific type.
    pub fn parse_arguments_as<T: serde::de::DeserializeOwned>(
        &self,
    ) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
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

/// Streaming event from TensorZero.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Initial chunk with IDs.
    Chunk(ChunkEvent),
    /// Content block start.
    ContentBlockStart(ContentBlockStartEvent),
    /// Content block delta.
    ContentBlockDelta(ContentBlockDeltaEvent),
    /// Content block stop.
    ContentBlockStop(ContentBlockStopEvent),
    /// Stream finished.
    Done(DoneEvent),
}

/// Initial chunk event with inference and episode IDs.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkEvent {
    /// Inference ID for this request.
    pub inference_id: Uuid,
    /// Episode ID for conversation tracking.
    pub episode_id: Uuid,
}

/// Start of a content block.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStartEvent {
    /// Index of this content block.
    pub index: usize,
    /// Type of content block.
    pub content_block: ContentBlockType,
}

/// Type of content block being streamed.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockType {
    /// Text content block.
    Text { text: String },
    /// Tool call content block.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

/// Delta update for a content block.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockDeltaEvent {
    /// Index of the content block being updated.
    pub index: usize,
    /// The delta content.
    pub delta: Delta,
}

/// Delta content types.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    /// Text delta.
    TextDelta { text: String },
    /// Tool call arguments delta.
    ToolCallDelta { arguments: String },
}

/// End of a content block.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStopEvent {
    /// Index of the finished content block.
    pub index: usize,
}

/// Stream finished event.
#[derive(Debug, Clone, Deserialize)]
pub struct DoneEvent {
    /// Usage statistics for the complete response.
    #[serde(default)]
    pub usage: Option<Usage>,
}
