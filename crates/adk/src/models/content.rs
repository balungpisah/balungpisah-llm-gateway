//! Content block types for messages.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text { text: String },

    /// Tool use request from the assistant.
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Name of the tool.
        name: String,
        /// Arguments as JSON.
        input: Value,
    },

    /// Tool result from execution.
    ToolResult {
        /// ID of the tool use this is responding to.
        tool_use_id: String,
        /// Name of the tool.
        name: String,
        /// Result content.
        content: String,
        /// Whether the tool execution failed.
        #[serde(default)]
        is_error: bool,
    },

    /// Image content.
    Image {
        /// Base64-encoded image data or URL.
        source: ImageSource,
    },
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool use content block.
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool result content block.
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            name: name.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error tool result content block.
    pub fn tool_error(
        tool_use_id: impl Into<String>,
        name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            name: name.into(),
            content: error.into(),
            is_error: true,
        }
    }

    /// Check if this is a text block.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Check if this is a tool use block.
    pub fn is_tool_use(&self) -> bool {
        matches!(self, Self::ToolUse { .. })
    }

    /// Check if this is a tool result block.
    pub fn is_tool_result(&self) -> bool {
        matches!(self, Self::ToolResult { .. })
    }

    /// Get the text content if this is a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Get the tool use details if this is a tool use block.
    pub fn as_tool_use(&self) -> Option<(&str, &str, &Value)> {
        match self {
            Self::ToolUse { id, name, input } => Some((id, name, input)),
            _ => None,
        }
    }
}

/// Image source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// MIME type (e.g., "image/png").
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// URL to an image.
    Url {
        /// Image URL.
        url: String,
    },
}

/// Content of a message, which can be a simple string or structured blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Create from a simple string.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create from content blocks.
    pub fn blocks(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }

    /// Get as a string (concatenates text blocks if structured).
    pub fn as_string(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.as_text())
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Get all content blocks (wraps text in a block if simple).
    pub fn as_blocks(&self) -> Vec<ContentBlock> {
        match self {
            Self::Text(text) => vec![ContentBlock::text(text)],
            Self::Blocks(blocks) => blocks.clone(),
        }
    }

    /// Check if this content contains any tool uses.
    pub fn has_tool_uses(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Blocks(blocks) => blocks.iter().any(|b| b.is_tool_use()),
        }
    }

    /// Get all tool use blocks from this content.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        match self {
            Self::Text(_) => vec![],
            Self::Blocks(blocks) => blocks.iter().filter(|b| b.is_tool_use()).collect(),
        }
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<Vec<ContentBlock>> for MessageContent {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }
}
