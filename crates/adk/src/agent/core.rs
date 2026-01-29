//! Core Agent implementation.

use crate::context::{
    convert_messages_to_input, convert_response_to_message_content, ContextFilter,
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

/// An AI agent that can have conversations and use tools.
pub struct Agent<S>
where
    S: Storage + 'static,
{
    pub(crate) client: TensorZeroClient,
    pub(crate) storage: Arc<S>,
    pub(crate) function_name: String,
    pub(crate) tools: ToolRegistry,
    pub(crate) max_iterations: usize,
    pub(crate) stream_config: StreamConfig,
    pub(crate) context_filter: Option<ContextFilter>,
}

impl<S> std::fmt::Debug for Agent<S>
where
    S: Storage + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("function_name", &self.function_name)
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
        let messages = if let Some(ref filter) = self.context_filter {
            filter.filter_messages(stored_messages)
        } else {
            stored_messages
        };
        let input_messages = convert_messages_to_input(&messages);

        // Run the chat loop
        self.chat_loop(Arc::new(thread), input_messages).await
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

            if iteration > self.max_iterations {
                return Err(AgentError::MaxIterationsExceeded {
                    max: self.max_iterations,
                });
            }

            // Build request
            let additional_tools = if !self.tools.is_empty() {
                Some(self.tools.tensorzero_definitions())
            } else {
                None
            };

            let mut request_builder = InferenceRequestBuilder::new()
                .function_name(&self.function_name)
                .messages(messages.clone());

            if let Some(episode_id) = thread.episode_id {
                request_builder = request_builder.episode_id(episode_id);
            }

            if let Some(tools) = additional_tools {
                request_builder = request_builder
                    .additional_tools(tools)
                    .tool_choice(ToolChoice::Auto);
            }

            let request = request_builder.build()?;

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
                    &result.content,
                ));
            }
        }
    }

    /// Execute tool calls.
    async fn execute_tools(
        &self,
        thread: &Arc<Thread>,
        tool_calls: &[&balungpisah_tensorzero::ToolCallBlock],
    ) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for tc in tool_calls {
            let result = if let Some(executor) = self.tools.get(&tc.name) {
                let args: Value = tc.parse_arguments().unwrap_or(Value::Null);
                let context = ToolContext::new(&tc.id, &tc.name, thread.clone());

                debug!("Executing tool '{}' with args: {}", tc.name, args);
                executor.execute(args, context).await
            } else {
                ToolResult::error(&tc.id, format!("Tool '{}' not found", tc.name))
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

        // Create stream executor
        let executor = StreamExecutor::new(
            self.client.clone(),
            self.storage.clone(),
            self.tools.clone(),
            &self.function_name,
        )
        .max_iterations(self.max_iterations)
        .config(self.stream_config.clone());

        let executor = if let Some(ref filter) = self.context_filter {
            executor.context_filter(filter.clone())
        } else {
            executor
        };

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

    /// Get the function name.
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Get the max iterations setting.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
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
            ContentBlock::ToolCall(tc) => InputContentBlock::ToolCall {
                r#type: "tool_call".to_string(),
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            },
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
                ContentBlock::tool_error(&r.tool_call_id, &r.content)
            } else {
                ContentBlock::tool_result(&r.tool_call_id, &r.content)
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
