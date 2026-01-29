//! Request types for TensorZero inference.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Tool choice configuration for inference requests.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to use tools.
    #[default]
    Auto,
    /// Force the model to use a specific tool.
    Tool(String),
    /// Prevent the model from using any tools.
    None,
    /// Require the model to use at least one tool.
    Required,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    /// The role of the message sender.
    pub role: MessageRole,
    /// The content of the message.
    pub content: Vec<InputContentBlock>,
}

impl InputMessage {
    /// Create a new user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![InputContentBlock::Text {
                r#type: "text".to_string(),
                text: text.into(),
            }],
        }
    }

    /// Create a new assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![InputContentBlock::Text {
                r#type: "text".to_string(),
                text: text.into(),
            }],
        }
    }

    /// Create a new assistant message with a tool call.
    pub fn assistant_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![InputContentBlock::ToolCall {
                r#type: "tool_call".to_string(),
                id: id.into(),
                name: name.into(),
                arguments: arguments.to_string(),
            }],
        }
    }

    /// Create a new tool result message.
    pub fn tool_result(id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![InputContentBlock::ToolResult {
                r#type: "tool_result".to_string(),
                id: id.into(),
                result: result.into(),
            }],
        }
    }

    /// Add a content block to this message.
    pub fn with_content(mut self, block: InputContentBlock) -> Self {
        self.content.push(block);
        self
    }
}

/// Role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// User message.
    User,
    /// Assistant (model) message.
    Assistant,
}

/// Content block in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContentBlock {
    /// Text content.
    Text { r#type: String, text: String },
    /// Tool call from the assistant.
    ToolCall {
        r#type: String,
        id: String,
        name: String,
        arguments: String,
    },
    /// Tool result from execution.
    ToolResult {
        r#type: String,
        id: String,
        result: String,
    },
}

/// Tool definition for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON schema for the tool parameters.
    pub parameters: Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Request payload for chat inference.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceRequest {
    /// Function name configured in TensorZero.
    pub function_name: String,
    /// Optional episode ID for tracking conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<Uuid>,
    /// Input messages for the conversation.
    pub input: InferenceInput,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Tool choice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Additional tools available for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_tools: Option<Vec<ToolDefinition>>,
    /// Template parameters for dynamic prompts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// Input for inference request.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceInput {
    /// System prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Conversation messages.
    pub messages: Vec<InputMessage>,
}

/// Builder for constructing inference requests.
#[derive(Debug, Default)]
pub struct InferenceRequestBuilder {
    function_name: Option<String>,
    episode_id: Option<Uuid>,
    messages: Vec<InputMessage>,
    system: Option<String>,
    stream: bool,
    tool_choice: Option<ToolChoice>,
    additional_tools: Option<Vec<ToolDefinition>>,
    params: Option<Value>,
}

impl InferenceRequestBuilder {
    /// Create a new request builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the function name.
    pub fn function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    /// Set the episode ID.
    pub fn episode_id(mut self, id: Uuid) -> Self {
        self.episode_id = Some(id);
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    /// Add a message to the request.
    pub fn message(mut self, message: InputMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Add multiple messages to the request.
    pub fn messages(mut self, messages: impl IntoIterator<Item = InputMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Enable streaming for this request.
    pub fn stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Set the tool choice.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Add additional tools.
    pub fn additional_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.additional_tools = Some(tools);
        self
    }

    /// Set template parameters.
    pub fn params(mut self, params: Value) -> Self {
        self.params = Some(params);
        self
    }

    /// Build the inference request.
    pub fn build(self) -> crate::error::Result<InferenceRequest> {
        let function_name =
            self.function_name
                .ok_or_else(|| crate::error::TensorZeroError::InvalidRequest {
                    message: "function_name is required".to_string(),
                })?;

        Ok(InferenceRequest {
            function_name,
            episode_id: self.episode_id,
            input: InferenceInput {
                system: self.system,
                messages: self.messages,
            },
            stream: self.stream,
            tool_choice: self.tool_choice,
            additional_tools: self.additional_tools,
            params: self.params,
        })
    }
}

/// Request payload for JSON inference.
#[derive(Debug, Clone, Serialize)]
pub struct JsonInferenceRequest {
    /// Function name configured in TensorZero.
    pub function_name: String,
    /// Optional episode ID for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode_id: Option<Uuid>,
    /// Input for the JSON inference.
    pub input: JsonInferenceInput,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Output schema for the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

/// Input for JSON inference request.
#[derive(Debug, Clone, Serialize)]
pub struct JsonInferenceInput {
    /// System prompt override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Conversation messages.
    pub messages: Vec<InputMessage>,
}

/// Builder for JSON inference requests.
#[derive(Debug, Default)]
pub struct JsonInferenceRequestBuilder {
    function_name: Option<String>,
    episode_id: Option<Uuid>,
    messages: Vec<InputMessage>,
    system: Option<String>,
    stream: bool,
    output_schema: Option<Value>,
}

impl JsonInferenceRequestBuilder {
    /// Create a new JSON request builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the function name.
    pub fn function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    /// Set the episode ID.
    pub fn episode_id(mut self, id: Uuid) -> Self {
        self.episode_id = Some(id);
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    /// Add a message.
    pub fn message(mut self, message: InputMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Add multiple messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = InputMessage>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Enable streaming.
    pub fn stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Set the output schema.
    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Build the JSON inference request.
    pub fn build(self) -> crate::error::Result<JsonInferenceRequest> {
        let function_name =
            self.function_name
                .ok_or_else(|| crate::error::TensorZeroError::InvalidRequest {
                    message: "function_name is required".to_string(),
                })?;

        Ok(JsonInferenceRequest {
            function_name,
            episode_id: self.episode_id,
            input: JsonInferenceInput {
                system: self.system,
                messages: self.messages,
            },
            stream: self.stream,
            output_schema: self.output_schema,
        })
    }
}
