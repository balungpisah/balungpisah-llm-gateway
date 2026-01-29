//! Core Agent implementation.

use crate::context::{
    convert_messages_to_input, convert_response_to_message_content, ContextConfig,
};
use crate::error::{AgentError, Result};
use crate::models::{Message, MessageContent, Role, Thread, ThreadOptions};
use crate::storage::Storage;
use crate::stream::{SseEvent, StreamConfig, StreamExecutor};
use crate::tools::{ToolContext, ToolRegistry, ToolResult};
use balungpisah_tensorzero::{
    InferenceRequestBuilder, InferenceResponse, TensorZeroClient, ToolChoice,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, instrument};
use uuid::Uuid;

use super::builder::ModelSpec;

/// An AI agent that can have conversations and use tools.
pub struct Agent<S>
where
    S: Storage + 'static,
{
    pub(crate) client: TensorZeroClient,
    pub(crate) storage: Arc<S>,
    pub(crate) model_spec: ModelSpec,
    pub(crate) tools: ToolRegistry,
    pub(crate) max_iterations: usize,
    pub(crate) stream_config: StreamConfig,
    pub(crate) context_config: Option<ContextConfig>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) credentials: Option<Value>,
    pub(crate) tags: Option<Value>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) parallel_tool_calls: Option<bool>,
}

impl<S> std::fmt::Debug for Agent<S>
where
    S: Storage + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model_spec", &self.model_spec)
            .field("max_iterations", &self.max_iterations)
            .field("tools", &self.tools.names())
            .finish()
    }
}

impl<S> Agent<S>
where
    S: Storage + 'static,
{
    /// Get or create a thread for the given external ID.
    #[instrument(skip(self, options))]
    pub async fn get_or_create_thread(
        &self,
        external_id: &str,
        options: Option<ThreadOptions>,
    ) -> Result<Thread> {
        let options = options.unwrap_or_default();

        if !options.force_new {
            if let Some(thread) = self.storage.get_thread_by_external_id(external_id).await? {
                return Ok(thread);
            }
        }

        let mut thread = Thread::new(external_id);
        if let Some(metadata) = options.metadata {
            thread = thread.with_metadata(metadata);
        }

        self.storage.create_thread(&thread).await?;
        debug!("Created new thread: {}", thread.id);

        Ok(thread)
    }

    /// Get a thread by ID.
    pub async fn get_thread(&self, thread_id: Uuid) -> Result<Option<Thread>> {
        Ok(self.storage.get_thread(thread_id).await?)
    }

    /// Get all messages for a thread.
    pub async fn get_thread_messages(&self, thread_id: Uuid) -> Result<Vec<Message>> {
        Ok(self.storage.get_thread_messages(thread_id).await?)
    }

    /// Get a message by ID.
    pub async fn get_message(&self, message_id: Uuid) -> Result<Option<Message>> {
        Ok(self.storage.get_message(message_id).await?)
    }

    /// Update a message.
    pub async fn update_message(&self, message: &Message) -> Result<()> {
        Ok(self.storage.update_message(message).await?)
    }

    /// Delete a message.
    pub async fn delete_message(&self, message_id: Uuid) -> Result<()> {
        Ok(self.storage.delete_message(message_id).await?)
    }

    /// Chat with the agent (non-streaming).
    ///
    /// This method sends a message and waits for the complete response,
    /// including any tool execution loops.
    #[instrument(skip(self, message), fields(thread_id = %thread_id))]
    pub async fn chat(&self, thread_id: Uuid, message: impl Into<String>) -> Result<ChatResponse> {
        let thread = self
            .storage
            .get_thread(thread_id)
            .await?
            .ok_or(AgentError::ThreadNotFound { thread_id })?;

        let user_message = message.into();

        // Save user message
        let user_msg = Message::user(thread_id, user_message.clone());
        self.storage.create_message(&user_msg).await?;

        // Get context messages
        let stored_messages = self.storage.get_thread_messages(thread_id).await?;
        let messages = self.filter_context_messages(stored_messages, false);
        let input_messages = convert_messages_to_input(&messages);

        // Run the chat loop
        self.chat_loop(Arc::new(thread), input_messages).await
    }

    /// Filter messages based on context configuration.
    fn filter_context_messages(&self, messages: Vec<Message>, is_loop: bool) -> Vec<Message> {
        let config = match &self.context_config {
            Some(cfg) => {
                if is_loop {
                    cfg.for_loop()
                } else {
                    cfg
                }
            }
            None => return messages,
        };

        config.filter_messages(messages)
    }

    /// Internal chat loop that handles tool execution.
    async fn chat_loop(
        &self,
        thread: Arc<Thread>,
        mut messages: Vec<balungpisah_tensorzero::InputMessage>,
    ) -> Result<ChatResponse> {
        let mut iteration = 0;
        let mut total_usage = Usage::default();

        loop {
            iteration += 1;
            let is_loop = iteration > 1;

            if iteration > self.max_iterations {
                return Err(AgentError::MaxIterationsExceeded {
                    max: self.max_iterations,
                });
            }

            // Re-filter messages for loop iterations if needed
            if is_loop {
                // For now, we keep the messages as-is in the loop
                // The context_config.for_loop() can be used for more aggressive filtering
            }

            // Build request
            let request = self.build_inference_request(&thread, &messages)?;

            // Send request
            let response = self.client.inference(request).await?;

            // Track usage
            if let Some(usage) = &response.usage {
                total_usage.input_tokens += usage.input_tokens;
                total_usage.output_tokens += usage.output_tokens;
            }

            // Check for tool calls
            let tool_calls = response.tool_calls();

            if tool_calls.is_empty() {
                // No tool calls - we're done
                let text = response.text();

                // Save assistant message
                let content = convert_response_to_message_content(&response.content);
                let assistant_msg = Message::assistant(thread.id, content)
                    .with_inference_id(response.inference_id)
                    .with_episode_id(response.episode_id);
                self.storage.create_message(&assistant_msg).await?;

                // Update thread episode ID if needed
                if thread.episode_id.is_none() {
                    let mut updated_thread = (*thread).clone();
                    updated_thread.episode_id = Some(response.episode_id);
                    self.storage.update_thread(&updated_thread).await?;
                }

                return Ok(ChatResponse {
                    text,
                    inference_id: response.inference_id,
                    episode_id: response.episode_id,
                    iterations: iteration,
                    usage: total_usage,
                });
            }

            // Save assistant message with tool calls
            let content = convert_response_to_message_content(&response.content);
            let assistant_msg = Message::assistant(thread.id, content)
                .with_inference_id(response.inference_id)
                .with_episode_id(response.episode_id);
            self.storage.create_message(&assistant_msg).await?;

            // Add assistant response to context
            messages.push(response_to_input_message(&response));

            // Execute tools
            let tool_results = self.execute_tools(&thread, &tool_calls).await;

            // Save tool results
            let tool_result_msg = tool_results_to_message(thread.id, &tool_results);
            self.storage.create_message(&tool_result_msg).await?;

            // Add tool results to context
            for result in &tool_results {
                messages.push(balungpisah_tensorzero::InputMessage::tool_result(
                    &result.tool_call_id,
                    &result.tool_name,
                    &result.content,
                ));
            }
        }
    }

    /// Build an inference request with all configured options.
    fn build_inference_request(
        &self,
        thread: &Thread,
        messages: &[balungpisah_tensorzero::InputMessage],
    ) -> Result<balungpisah_tensorzero::InferenceRequest> {
        let mut builder = InferenceRequestBuilder::new().messages(messages.iter().cloned());

        // Set model or function
        match &self.model_spec {
            ModelSpec::Function(name) => {
                builder = builder.function_name(name);
            }
            ModelSpec::Model(name) => {
                builder = builder.model(name);
            }
        }

        // Set episode ID
        if let Some(episode_id) = thread.episode_id {
            builder = builder.episode_id(episode_id);
        }

        // Set system prompt
        if let Some(ref prompt) = self.system_prompt {
            builder = builder.system(prompt);
        }

        // Set credentials
        if let Some(ref creds) = self.credentials {
            builder = builder.credentials(creds.clone());
        }

        // Set tags
        if let Some(ref tags) = self.tags {
            builder = builder.tags(tags.clone());
        }

        // Set tools
        if !self.tools.is_empty() {
            builder = builder.additional_tools(self.tools.tensorzero_definitions());

            // Set tool choice (default to Auto if tools are present)
            let choice = self.tool_choice.clone().unwrap_or(ToolChoice::Auto);
            builder = builder.tool_choice(choice);
        }

        // Set parallel tool calls
        if let Some(parallel) = self.parallel_tool_calls {
            builder = builder.parallel_tool_calls(parallel);
        }

        builder.build().map_err(|e| AgentError::Inference {
            message: e.to_string(),
            status_code: None,
        })
    }

    /// Execute tool calls.
    async fn execute_tools(
        &self,
        thread: &Arc<Thread>,
        tool_calls: &[&balungpisah_tensorzero::ToolCallBlock],
    ) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for tc in tool_calls {
            let tool_name = tc.tool_name().unwrap_or("unknown");
            let result = if let Some(executor) = self.tools.get(tool_name) {
                let args: Value = tc.parse_arguments().unwrap_or(Value::Null);
                let context = ToolContext::new(&tc.id, tool_name, thread.clone());

                debug!("Executing tool '{}' with args: {}", tool_name, args);
                executor.execute(args, context).await
            } else {
                ToolResult::error(&tc.id, tool_name, format!("Tool '{}' not found", tool_name))
            };

            results.push(result);
        }

        results
    }

    /// Chat with streaming response.
    ///
    /// Returns a channel receiver that will emit SSE events as they occur.
    #[instrument(skip(self, message), fields(thread_id = %thread_id))]
    pub async fn chat_stream(
        &self,
        thread_id: Uuid,
        message: impl Into<String>,
    ) -> Result<mpsc::Receiver<SseEvent>> {
        let thread = self
            .storage
            .get_thread(thread_id)
            .await?
            .ok_or(AgentError::ThreadNotFound { thread_id })?;

        let thread = Arc::new(thread);
        let user_message = message.into();

        let (tx, rx) = mpsc::channel(self.stream_config.buffer_size);

        // Get model/function name for the executor
        let model_or_function = match &self.model_spec {
            ModelSpec::Function(name) => name.clone(),
            ModelSpec::Model(name) => name.clone(),
        };

        // Create stream executor
        let mut executor = StreamExecutor::new(
            self.client.clone(),
            self.storage.clone(),
            self.tools.clone(),
            &model_or_function,
        )
        .max_iterations(self.max_iterations)
        .config(self.stream_config.clone());

        if let Some(ref config) = self.context_config {
            executor = executor.context_config(config.clone());
        }

        // Spawn execution task
        tokio::spawn(async move {
            if let Err(e) = executor.execute(thread, &user_message, tx.clone()).await {
                let _ = tx.send(SseEvent::error(e.to_string())).await;
            }
        });

        Ok(rx)
    }

    /// Get reference to the tool registry.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// Get mutable reference to the tool registry.
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// Get the model specification.
    pub fn model_spec(&self) -> &ModelSpec {
        &self.model_spec
    }

    /// Get the function name (if using a function).
    pub fn function_name(&self) -> Option<&str> {
        self.model_spec.as_function()
    }

    /// Get the model name (if using a model directly).
    pub fn model_name(&self) -> Option<&str> {
        self.model_spec.as_model()
    }

    /// Get the max iterations setting.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
}

/// Response from a chat request.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The final text response.
    pub text: String,
    /// Inference ID from TensorZero.
    pub inference_id: Uuid,
    /// Episode ID for the conversation.
    pub episode_id: Uuid,
    /// Number of iterations (tool execution loops) completed.
    pub iterations: usize,
    /// Token usage statistics.
    pub usage: Usage,
}

impl ChatResponse {
    /// Get the text content.
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens generated.
    pub output_tokens: u32,
}

impl Usage {
    /// Get total tokens used.
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// A simple chat request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// The message text.
    pub message: String,
    /// Optional metadata for the message.
    pub metadata: Option<Value>,
}

impl ChatRequest {
    /// Create a new chat request.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            metadata: None,
        }
    }

    /// Add metadata to the request.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl From<String> for ChatRequest {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ChatRequest {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

// Helper functions

fn response_to_input_message(response: &InferenceResponse) -> balungpisah_tensorzero::InputMessage {
    use balungpisah_tensorzero::{ContentBlock, InputContentBlock, InputMessage, MessageRole};

    let content: Vec<InputContentBlock> = response
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => InputContentBlock::Text {
                r#type: "text".to_string(),
                text: text.clone(),
            },
            ContentBlock::ToolCall(tc) => {
                // Get the arguments as a Value for input
                let args = tc.parse_arguments().unwrap_or(Value::Null);
                let name = tc.tool_name().unwrap_or("unknown").to_string();
                InputContentBlock::ToolCall {
                    r#type: "tool_call".to_string(),
                    id: tc.id.clone(),
                    name,
                    arguments: args,
                }
            }
        })
        .collect();

    InputMessage {
        role: MessageRole::Assistant,
        content,
    }
}

fn tool_results_to_message(thread_id: Uuid, results: &[ToolResult]) -> Message {
    use crate::models::ContentBlock;

    let blocks: Vec<ContentBlock> = results
        .iter()
        .map(|r| {
            if r.is_error {
                ContentBlock::tool_error(&r.tool_call_id, &r.tool_name, &r.content)
            } else {
                ContentBlock::tool_result(&r.tool_call_id, &r.tool_name, &r.content)
            }
        })
        .collect();

    Message {
        id: Uuid::now_v7(),
        thread_id,
        role: Role::User,
        content: MessageContent::Blocks(blocks),
        created_at: chrono::Utc::now(),
        inference_id: None,
        episode_id: None,
    }
}
