//! Request types for TensorZero inference.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Tool choice configuration for inference requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to use tools.
    #[default]
    Auto,
    /// Force the model to use a specific tool.
    #[serde(rename = "specific")]
    Specific { name: String },
    /// Prevent the model from using any tools.
    None,
    /// Require the model to use at least one tool.
    Required,
}

impl ToolChoice {
    /// Create a tool choice that forces a specific tool.
    pub fn specific(name: impl Into<String>) -> Self {
        Self::Specific { name: name.into() }
    }
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
                arguments,
            }],
        }
    }

    /// Create a new tool result message.
    pub fn tool_result(
        id: impl Into<String>,
        name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![InputContentBlock::ToolResult {
                r#type: "tool_result".to_string(),
                id: id.into(),
                name: name.into(),
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
        /// Arguments as a JSON object (can also be a string for backwards compatibility).
        arguments: Value,
    },
    /// Tool result from execution.
    ToolResult {
        r#type: String,
        id: String,
        name: String,
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

/// Credentials for dynamic API keys.
///
/// TensorZero supports dynamic API keys via the `credentials` field.
/// Keys like `system_api_key` can be provided at request time.
pub type Credentials = Value;

/// Request payload for chat inference.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceRequest {
    /// Function name configured in TensorZero.
    /// Either `function_name` or `model` must be provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,

    /// Model name for ad-hoc inference (alternative to function_name).
    /// When using model directly, TensorZero uses the built-in `tensorzero::default` function.
    ///
    /// Formats:
    /// - `"my_model"` - references `[models.my_model]` in config
    /// - `"openai::gpt-4o"` - calls provider API directly (shorthand)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

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

    /// Whether to allow parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Template parameters for dynamic prompts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,

    /// Dynamic credentials (e.g., API keys).
    /// Use this to pass `system_api_key` and other dynamic credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<Credentials>,

    /// Tags for tracking and observability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Value>,
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
    model_name: Option<String>,
    episode_id: Option<Uuid>,
    messages: Vec<InputMessage>,
    system: Option<String>,
    stream: bool,
    tool_choice: Option<ToolChoice>,
    additional_tools: Option<Vec<ToolDefinition>>,
    parallel_tool_calls: Option<bool>,
    params: Option<Value>,
    credentials: Option<Credentials>,
    tags: Option<Value>,
}

impl InferenceRequestBuilder {
    /// Create a new request builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the function name.
    ///
    /// Either `function_name` or `model` must be set before building.
    pub fn function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    /// Set the model name for ad-hoc inference.
    ///
    /// Use this as an alternative to `function_name` when you want to use
    /// a model directly without a pre-configured function.
    ///
    /// Formats:
    /// - `"my_model"` - references `[models.my_model]` in config
    /// - `"openai::gpt-4o"` - calls provider API directly (shorthand)
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
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

    /// Set whether to allow parallel tool calls.
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set template parameters.
    pub fn params(mut self, params: Value) -> Self {
        self.params = Some(params);
        self
    }

    /// Set dynamic credentials (e.g., API keys).
    ///
    /// # Example
    ///
    /// ```
    /// use balungpisah_tensorzero::InferenceRequestBuilder;
    /// use serde_json::json;
    ///
    /// let request = InferenceRequestBuilder::new()
    ///     .model("gpt-4o")
    ///     .credentials(json!({
    ///         "system_api_key": "sk-..."
    ///     }))
    ///     .build();
    /// ```
    pub fn credentials(mut self, credentials: Value) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set tags for tracking and observability.
    pub fn tags(mut self, tags: Value) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Build the inference request.
    ///
    /// Either `function_name` or `model_name` must be set.
    pub fn build(self) -> crate::error::Result<InferenceRequest> {
        // Validate that either function_name or model_name is set
        if self.function_name.is_none() && self.model_name.is_none() {
            return Err(crate::error::TensorZeroError::InvalidRequest {
                message: "either function_name or model_name must be set".to_string(),
            });
        }

        Ok(InferenceRequest {
            function_name: self.function_name,
            model_name: self.model_name,
            episode_id: self.episode_id,
            input: InferenceInput {
                system: self.system,
                messages: self.messages,
            },
            stream: self.stream,
            tool_choice: self.tool_choice,
            additional_tools: self.additional_tools,
            parallel_tool_calls: self.parallel_tool_calls,
            params: self.params,
            credentials: self.credentials,
            tags: self.tags,
        })
    }
}

/// Request payload for JSON inference.
#[derive(Debug, Clone, Serialize)]
pub struct JsonInferenceRequest {
    /// Function name configured in TensorZero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_name: Option<String>,

    /// Model name for ad-hoc inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

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

    /// Dynamic credentials.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<Credentials>,

    /// Tags for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Value>,
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
    model_name: Option<String>,
    episode_id: Option<Uuid>,
    messages: Vec<InputMessage>,
    system: Option<String>,
    stream: bool,
    output_schema: Option<Value>,
    credentials: Option<Credentials>,
    tags: Option<Value>,
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

    /// Set the model name.
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
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

    /// Set dynamic credentials.
    pub fn credentials(mut self, credentials: Value) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set tags for tracking.
    pub fn tags(mut self, tags: Value) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Build the JSON inference request.
    pub fn build(self) -> crate::error::Result<JsonInferenceRequest> {
        if self.function_name.is_none() && self.model_name.is_none() {
            return Err(crate::error::TensorZeroError::InvalidRequest {
                message: "either function_name or model_name must be set".to_string(),
            });
        }

        Ok(JsonInferenceRequest {
            function_name: self.function_name,
            model_name: self.model_name,
            episode_id: self.episode_id,
            input: JsonInferenceInput {
                system: self.system,
                messages: self.messages,
            },
            stream: self.stream,
            output_schema: self.output_schema,
            credentials: self.credentials,
            tags: self.tags,
        })
    }
}
