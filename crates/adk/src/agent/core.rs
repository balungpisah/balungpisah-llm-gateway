//! Core Agent implementation.

use crate::context::{convert_messages_to_input, ContextConfig};
use crate::error::{AgentError, Result};
use crate::models::{Message, MessageContent, Role, Thread, ThreadOptions};
use crate::storage::Storage;
use crate::stream::{SseEvent, StreamConfig, StreamExecutor};
use crate::tools::{ToolContext, ToolRegistry, ToolResult};
use balungpisah_tensorzero::{
    InferenceRequestBuilder, InferenceResponse, TensorZeroClient, ToolChoice,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use super::builder::ModelSpec;

/// Maximum length for auto-generated thread titles.
const MAX_TITLE_LENGTH: usize = 50;

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

    /// Update a thread.
    pub async fn update_thread(&self, thread: &Thread) -> Result<()> {
        Ok(self.storage.update_thread(thread).await?)
    }

    /// Delete a message.
    pub async fn delete_message(&self, message_id: Uuid) -> Result<()> {
        Ok(self.storage.delete_message(message_id).await?)
    }

    /// Delete all messages in a thread after a specific message.
    /// This is used for the "edit and resubmit" workflow.
    /// Returns the number of messages deleted.
    pub async fn delete_messages_after(&self, thread_id: Uuid, after_id: Uuid) -> Result<u64> {
        Ok(self
            .storage
            .delete_messages_after(thread_id, after_id)
            .await?)
    }

    /// Resolve or create a thread based on the request.
    ///
    /// Thread lifecycle:
    /// - `thread_id = None`: Create new thread with auto-generated ID
    /// - `thread_id = Some(id)` not found: Create thread with the provided ID (optimistic UI)
    /// - `thread_id = Some(id)` found: Verify ownership and return existing thread
    async fn resolve_thread(
        &self,
        external_id: &str,
        thread_id: Option<Uuid>,
        agent_slug: Option<&str>,
        title: Option<&str>,
        metadata: Option<Value>,
    ) -> Result<Thread> {
        match thread_id {
            None => {
                // Create new thread with auto-generated ID
                let mut thread = Thread::new(external_id);
                if let Some(slug) = agent_slug {
                    thread = thread.with_agent_slug(slug);
                }
                if let Some(t) = title {
                    thread = thread.with_title(t);
                }
                if let Some(m) = metadata {
                    thread = thread.with_metadata(m);
                }

                self.storage.create_thread(&thread).await?;
                info!(thread_id = %thread.id, "Created new thread");
                Ok(thread)
            }
            Some(id) => {
                // Check if thread exists
                if let Some(existing) = self.storage.get_thread(id).await? {
                    // Verify ownership
                    if existing.external_id != external_id {
                        return Err(AgentError::ThreadAccessDenied {
                            thread_id: id,
                            reason: "Thread belongs to another user".to_string(),
                        });
                    }
                    debug!(thread_id = %id, "Using existing thread");
                    Ok(existing)
                } else {
                    // Create thread with FE-provided ID (optimistic UI)
                    let mut thread = Thread::with_id(id, external_id);
                    if let Some(slug) = agent_slug {
                        thread = thread.with_agent_slug(slug);
                    }
                    if let Some(t) = title {
                        thread = thread.with_title(t);
                    }
                    if let Some(m) = metadata {
                        thread = thread.with_metadata(m);
                    }

                    self.storage.create_thread(&thread).await?;
                    info!(thread_id = %id, "Created thread with FE-provided ID (optimistic UI)");
                    Ok(thread)
                }
            }
        }
    }

    /// Resolve or create a user message based on the request.
    ///
    /// Message lifecycle:
    /// - `user_message_id = None`: Create new message with auto-generated ID
    /// - `user_message_id = Some(id)` not found: Create message with provided ID (optimistic UI)
    /// - `user_message_id = Some(id)` found: Edit mode - update and delete messages after
    async fn resolve_user_message(
        &self,
        thread_id: Uuid,
        user_message_id: Option<Uuid>,
        content: MessageContent,
    ) -> Result<Message> {
        match user_message_id {
            None => {
                // Create new message with auto-generated ID
                let msg = Message::user(thread_id, content);
                self.storage.create_message(&msg).await?;
                debug!(message_id = %msg.id, "Created new user message");
                Ok(msg)
            }
            Some(id) => {
                // Check if message exists
                if let Some(existing) = self.storage.get_message(id).await? {
                    // Verify it belongs to this thread
                    if existing.thread_id != thread_id {
                        return Err(AgentError::MessageAccessDenied {
                            message_id: id,
                            reason: "Message belongs to another thread".to_string(),
                        });
                    }

                    // Verify it's a user message
                    if existing.role != Role::User {
                        return Err(AgentError::MessageAccessDenied {
                            message_id: id,
                            reason: "Only user messages can be edited".to_string(),
                        });
                    }

                    // Edit mode: delete all messages after this one
                    let deleted = self.storage.delete_messages_after(thread_id, id).await?;
                    info!(
                        message_id = %id,
                        deleted_count = deleted,
                        "Edit mode: deleted messages after target"
                    );

                    // Update the message content
                    let mut updated = existing;
                    updated.content = content;
                    updated.touch();
                    self.storage.update_message(&updated).await?;

                    Ok(updated)
                } else {
                    // Create message with FE-provided ID (optimistic UI)
                    let msg = Message::user_with_id(id, thread_id, content);
                    self.storage.create_message(&msg).await?;
                    info!(message_id = %id, "Created message with FE-provided ID (optimistic UI)");
                    Ok(msg)
                }
            }
        }
    }

    /// Generate a title from message content.
    fn generate_title(content: &MessageContent) -> Option<String> {
        let text = match content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Blocks(blocks) => {
                // Find first text block
                blocks
                    .iter()
                    .find_map(|b| b.as_text().map(|s| s.to_string()))?
            }
        };

        if text.is_empty() {
            return None;
        }

        // Truncate to MAX_TITLE_LENGTH chars, preserving word boundaries
        let title = if text.len() <= MAX_TITLE_LENGTH {
            text
        } else {
            let truncated = &text[..MAX_TITLE_LENGTH];
            // Try to find a word boundary
            if let Some(pos) = truncated.rfind(char::is_whitespace) {
                format!("{}...", &truncated[..pos])
            } else {
                format!("{}...", truncated)
            }
        };

        Some(title)
    }

    /// Chat with the agent using a structured request.
    ///
    /// This is the main entry point that handles the full thread/message lifecycle:
    /// - Thread creation/lookup with optimistic UI support
    /// - Message creation/edit with optimistic UI support
    /// - Auto-generated titles
    ///
    /// Returns a streaming response.
    #[instrument(skip(self, request), fields(external_id = %external_id))]
    pub async fn chat_with_request(
        &self,
        external_id: &str,
        request: ChatRequest,
    ) -> Result<ChatStreamResponse> {
        // Check if this is the first message (for title generation)
        let is_first_message = match request.thread_id {
            None => true,
            Some(id) => {
                // Check if thread has any messages
                if self.storage.get_thread(id).await?.is_some() {
                    let messages = self.storage.get_thread_messages(id).await?;
                    messages.is_empty()
                } else {
                    true // Thread doesn't exist yet
                }
            }
        };

        // Generate title from first message if needed
        let title = if is_first_message {
            Self::generate_title(&request.content)
        } else {
            None
        };

        // Resolve thread
        let mut thread = self
            .resolve_thread(
                external_id,
                request.thread_id,
                request.agent_slug.as_deref(),
                title.as_deref(),
                request.metadata.clone(),
            )
            .await?;

        // Update title if this is the first message and thread didn't have one
        if is_first_message && thread.title.is_none() && title.is_some() {
            thread.title = title;
            self.storage.update_thread(&thread).await?;
        }

        // Resolve user message
        let _user_msg = self
            .resolve_user_message(thread.id, request.user_message_id, request.content.clone())
            .await?;

        // Start streaming
        let (tx, rx) = mpsc::channel(self.stream_config.buffer_size);

        let thread = Arc::new(thread);
        let thread_id = thread.id;

        // Create stream executor
        let mut executor = StreamExecutor::new(
            self.client.clone(),
            self.storage.clone(),
            self.tools.clone(),
            self.model_spec.clone(),
        )
        .max_iterations(self.max_iterations)
        .config(self.stream_config.clone());

        // Pass all configurations
        if let Some(ref config) = self.context_config {
            executor = executor.context_config(config.clone());
        }
        if let Some(ref prompt) = self.system_prompt {
            executor = executor.system_prompt(prompt.clone());
        }
        if let Some(ref creds) = self.credentials {
            executor = executor.credentials(creds.clone());
        }
        if let Some(ref tags) = self.tags {
            executor = executor.tags(tags.clone());
        }
        if let Some(ref choice) = self.tool_choice {
            executor = executor.tool_choice(choice.clone());
        }
        if let Some(parallel) = self.parallel_tool_calls {
            executor = executor.parallel_tool_calls(parallel);
        }

        // Spawn execution task
        // Note: We pass the content for reference, but execute_without_saving will
        // load messages from storage (which includes the message we just saved)
        let content = request.content;
        tokio::spawn(async move {
            if let Err(e) = executor
                .execute_without_saving(thread, content, tx.clone())
                .await
            {
                let error_event = SseEvent::error("agent_error".to_string(), e.to_string());
                if let Ok(sse_str) = error_event.to_sse_string() {
                    let _ = tx.send(sse_str).await;
                }
            }
        });

        Ok(ChatStreamResponse {
            thread_id,
            user_message_id: request.user_message_id,
            stream: rx,
        })
    }

    /// Chat with the agent (non-streaming).
    ///
    /// This method sends a message and waits for the complete response,
    /// including any tool execution loops.
    ///
    /// Accepts either a simple string or structured `MessageContent` for multimodal input.
    #[instrument(skip(self, content), fields(thread_id = %thread_id))]
    pub async fn chat(
        &self,
        thread_id: Uuid,
        content: impl Into<MessageContent>,
    ) -> Result<ChatResponse> {
        let thread = self
            .storage
            .get_thread(thread_id)
            .await?
            .ok_or(AgentError::ThreadNotFound { thread_id })?;

        let message_content = content.into();

        // Save user message
        let user_msg = Message::user(thread_id, message_content);
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
    ///
    /// This method implements unified message storage: all content from the tool execution
    /// loop (tool_use, tool_result, text) is accumulated and saved as a single assistant
    /// message with one episode_id at the end of the loop.
    async fn chat_loop(
        &self,
        thread: Arc<Thread>,
        mut messages: Vec<balungpisah_tensorzero::InputMessage>,
    ) -> Result<ChatResponse> {
        let mut iteration = 0;
        let mut total_usage = Usage::default();

        // Unified storage: capture episode_id from first inference and accumulate all blocks
        let mut loop_episode_id: Option<Uuid> = None;
        let mut accumulated_blocks: Vec<crate::models::ContentBlock> = Vec::new();

        loop {
            iteration += 1;
            let is_loop = iteration > 1;

            if iteration > self.max_iterations {
                // Don't save partial content on max iterations exceeded
                return Err(AgentError::MaxIterationsExceeded {
                    max: self.max_iterations,
                });
            }

            // Re-filter messages for loop iterations if needed
            if is_loop {
                // For now, we keep the messages as-is in the loop
                // The context_config.for_loop() can be used for more aggressive filtering
            }

            // Build request (pass episode_id from first iteration to subsequent requests)
            let request = self.build_inference_request(&thread, &messages, loop_episode_id)?;

            // Send request
            let response = self.client.inference(request).await?;

            // Track usage
            if let Some(usage) = &response.usage {
                total_usage.input_tokens += usage.input_tokens;
                total_usage.output_tokens += usage.output_tokens;
            }

            // Capture episode_id from first iteration only
            if loop_episode_id.is_none() {
                loop_episode_id = Some(response.episode_id);
            }

            // Accumulate response content blocks
            for block in &response.content {
                accumulated_blocks.push(convert_tz_block_to_content_block(block));
            }

            // Check for tool calls
            let tool_calls = response.tool_calls();

            if tool_calls.is_empty() {
                // No tool calls - loop complete
                let text = response.text();

                // Save unified assistant message with all accumulated content
                let unified_msg =
                    Message::assistant(thread.id, MessageContent::Blocks(accumulated_blocks))
                        .with_episode_id(loop_episode_id.unwrap_or(response.episode_id));
                self.storage.create_message(&unified_msg).await?;

                return Ok(ChatResponse {
                    text,
                    inference_id: response.inference_id,
                    episode_id: loop_episode_id.unwrap_or(response.episode_id),
                    iterations: iteration,
                    usage: total_usage,
                });
            }

            // Add assistant response to context for next iteration (not saving to storage yet)
            messages.push(response_to_input_message(&response));

            // Execute tools
            let tool_results = self.execute_tools(&thread, &tool_calls).await;

            // Accumulate tool results (not saving as separate message)
            for result in &tool_results {
                accumulated_blocks.push(if result.is_error {
                    crate::models::ContentBlock::tool_error(
                        &result.tool_call_id,
                        &result.tool_name,
                        &result.content,
                    )
                } else {
                    crate::models::ContentBlock::tool_result(
                        &result.tool_call_id,
                        &result.tool_name,
                        &result.content,
                    )
                });
            }

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
        _thread: &Thread,
        messages: &[balungpisah_tensorzero::InputMessage],
        episode_id: Option<Uuid>,
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

        // Pass episode_id to ensure all inferences in the loop share the same episode
        if let Some(eid) = episode_id {
            builder = builder.episode_id(eid);
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

    /// Chat with streaming response (simple version).
    ///
    /// Returns a channel receiver that will emit raw SSE-formatted strings.
    ///
    /// For full lifecycle support (thread/message creation), use `chat_with_request` instead.
    #[instrument(skip(self, content), fields(thread_id = %thread_id))]
    pub async fn chat_stream(
        &self,
        thread_id: Uuid,
        content: impl Into<MessageContent>,
    ) -> Result<mpsc::Receiver<String>> {
        let thread = self
            .storage
            .get_thread(thread_id)
            .await?
            .ok_or(AgentError::ThreadNotFound { thread_id })?;

        let thread = Arc::new(thread);
        let message_content = content.into();

        let (tx, rx) = mpsc::channel(self.stream_config.buffer_size);

        // Create stream executor with full model spec and configuration
        let mut executor = StreamExecutor::new(
            self.client.clone(),
            self.storage.clone(),
            self.tools.clone(),
            self.model_spec.clone(),
        )
        .max_iterations(self.max_iterations)
        .config(self.stream_config.clone());

        // Pass context config
        if let Some(ref config) = self.context_config {
            executor = executor.context_config(config.clone());
        }

        // Pass system prompt
        if let Some(ref prompt) = self.system_prompt {
            executor = executor.system_prompt(prompt.clone());
        }

        // Pass credentials
        if let Some(ref creds) = self.credentials {
            executor = executor.credentials(creds.clone());
        }

        // Pass tags
        if let Some(ref tags) = self.tags {
            executor = executor.tags(tags.clone());
        }

        // Pass tool choice
        if let Some(ref choice) = self.tool_choice {
            executor = executor.tool_choice(choice.clone());
        }

        // Pass parallel tool calls setting
        if let Some(parallel) = self.parallel_tool_calls {
            executor = executor.parallel_tool_calls(parallel);
        }

        // Spawn execution task
        tokio::spawn(async move {
            if let Err(e) = executor.execute(thread, message_content, tx.clone()).await {
                // Send error as SSE-formatted string
                let error_event = SseEvent::error("agent_error".to_string(), e.to_string());
                if let Ok(sse_str) = error_event.to_sse_string() {
                    let _ = tx.send(sse_str).await;
                }
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

/// Response from chat_with_request containing the stream and metadata.
pub struct ChatStreamResponse {
    /// The thread ID (may be newly created).
    pub thread_id: Uuid,
    /// The user message ID (if provided).
    pub user_message_id: Option<Uuid>,
    /// The SSE stream receiver.
    pub stream: mpsc::Receiver<String>,
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

/// A chat request with full lifecycle support.
///
/// ## Thread Lifecycle
/// - `thread_id = None`: Create new thread with auto-generated ID
/// - `thread_id = Some(id)` not found: Create thread with provided ID (optimistic UI)
/// - `thread_id = Some(id)` found: Use existing thread (verifies ownership)
///
/// ## Message Lifecycle
/// - `user_message_id = None`: Create new message with auto-generated ID
/// - `user_message_id = Some(id)` not found: Create message with provided ID (optimistic UI)
/// - `user_message_id = Some(id)` found: Edit mode - update and delete subsequent messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Message content (text or multimodal blocks).
    pub content: MessageContent,

    /// Optional thread ID.
    /// - `None`: Create new thread
    /// - `Some(id)`: Use existing or create with this ID (optimistic UI)
    #[serde(default)]
    pub thread_id: Option<Uuid>,

    /// Optional user message ID for optimistic UI or edit mode.
    /// - `None`: Auto-generate message ID
    /// - `Some(id)` not found: Create with this ID (optimistic UI)
    /// - `Some(id)` found: Edit mode - update message and delete all after it
    #[serde(default)]
    pub user_message_id: Option<Uuid>,

    /// Optional agent slug for debugging/filtering.
    #[serde(default)]
    pub agent_slug: Option<String>,

    /// Optional metadata for the thread/message.
    #[serde(default)]
    pub metadata: Option<Value>,
}

impl ChatRequest {
    /// Create a new chat request with text content.
    pub fn new(content: impl Into<MessageContent>) -> Self {
        Self {
            content: content.into(),
            thread_id: None,
            user_message_id: None,
            agent_slug: None,
            metadata: None,
        }
    }

    /// Set the thread ID.
    pub fn thread_id(mut self, id: Uuid) -> Self {
        self.thread_id = Some(id);
        self
    }

    /// Set the user message ID.
    pub fn user_message_id(mut self, id: Uuid) -> Self {
        self.user_message_id = Some(id);
        self
    }

    /// Set the agent slug.
    pub fn agent_slug(mut self, slug: impl Into<String>) -> Self {
        self.agent_slug = Some(slug.into());
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: Value) -> Self {
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
        Self::new(message.to_string())
    }
}

// Helper functions

/// Convert TensorZero content block to our ContentBlock type.
fn convert_tz_block_to_content_block(
    block: &balungpisah_tensorzero::ContentBlock,
) -> crate::models::ContentBlock {
    match block {
        balungpisah_tensorzero::ContentBlock::Text { text } => {
            crate::models::ContentBlock::text(text)
        }
        balungpisah_tensorzero::ContentBlock::ToolCall(tc) => {
            let name = tc.tool_name().unwrap_or("unknown");
            let input = tc.parse_arguments().unwrap_or(Value::Null);
            crate::models::ContentBlock::tool_use(&tc.id, name, input)
        }
    }
}

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
