//! Stream executor for handling streaming responses with tool execution.

use super::config::StreamConfig;
use super::events::{generate_block_id, generate_message_id, SseEvent};
use crate::agent::ModelSpec;
use crate::context::ContextConfig;
use crate::error::{AgentError, Result};
use crate::models::{ContentBlock, Message, MessageContent, Role, Thread};
use crate::storage::{MessageStorage, ThreadStorage};
use crate::tools::{ToolContext, ToolRegistry, ToolResult};
use balungpisah_tensorzero::{
    ContentBlock as TzContentBlock, InferenceRequestBuilder, InputMessage, StreamContentBlock,
    TensorZeroClient, ToolChoice,
};
use chrono::Utc;
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

/// Executor for streaming agent responses with tool execution loop.
pub struct StreamExecutor<S>
where
    S: ThreadStorage + MessageStorage + Send + Sync,
{
    /// TensorZero client.
    client: TensorZeroClient,
    /// Storage backend.
    storage: Arc<S>,
    /// Tool registry.
    tools: ToolRegistry,
    /// Model specification (function or model name).
    model_spec: ModelSpec,
    /// Maximum iterations for tool execution loop.
    max_iterations: usize,
    /// Stream configuration.
    config: StreamConfig,
    /// Context configuration.
    context_config: Option<ContextConfig>,
    /// System prompt override.
    system_prompt: Option<String>,
    /// Dynamic credentials (e.g., API keys).
    credentials: Option<Value>,
    /// Tags for tracking.
    tags: Option<Value>,
    /// Tool choice strategy.
    tool_choice: Option<ToolChoice>,
    /// Whether to allow parallel tool calls.
    parallel_tool_calls: Option<bool>,
}

impl<S> std::fmt::Debug for StreamExecutor<S>
where
    S: ThreadStorage + MessageStorage + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamExecutor")
            .field("model_spec", &self.model_spec)
            .field("max_iterations", &self.max_iterations)
            .field("tools", &self.tools.names())
            .finish()
    }
}

/// Helper to send SSE event as raw string
async fn send_sse(sender: &mpsc::Sender<String>, event: SseEvent) {
    if let Ok(sse_string) = event.to_sse_string() {
        let _ = sender.send(sse_string).await;
    }
}

impl<S> StreamExecutor<S>
where
    S: ThreadStorage + MessageStorage + Send + Sync + 'static,
{
    /// Create a new stream executor.
    pub fn new(
        client: TensorZeroClient,
        storage: Arc<S>,
        tools: ToolRegistry,
        model_spec: ModelSpec,
    ) -> Self {
        Self {
            client,
            storage,
            tools,
            model_spec,
            max_iterations: 10,
            config: StreamConfig::default(),
            context_config: None,
            system_prompt: None,
            credentials: None,
            tags: None,
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set stream configuration.
    pub fn config(mut self, config: StreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Set context configuration.
    pub fn context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set dynamic credentials (e.g., API keys).
    pub fn credentials(mut self, credentials: Value) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set tags for tracking.
    pub fn tags(mut self, tags: Value) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set the tool choice strategy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set whether to allow parallel tool calls.
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Execute a chat request with streaming.
    ///
    /// Emits raw SSE-formatted strings to the sender channel.
    /// Accepts either a simple string or structured `MessageContent` for multimodal input.
    #[instrument(skip(self, sender, content), fields(thread_id = %thread.id))]
    pub async fn execute(
        &self,
        thread: Arc<Thread>,
        content: impl Into<MessageContent>,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        // Save user message
        let user_msg = Message::user(thread.id, content.into());
        self.storage.create_message(&user_msg).await?;

        // Build context messages
        let messages = self.build_context_messages(&thread).await?;

        // Run the execution loop
        self.execute_loop(thread, messages, sender).await
    }

    /// Execute a chat request with streaming, without saving the user message.
    ///
    /// Use this when the user message has already been saved (e.g., by `chat_with_request`
    /// for optimistic UI or edit mode support).
    ///
    /// Emits raw SSE-formatted strings to the sender channel.
    #[instrument(skip(self, sender, _content), fields(thread_id = %thread.id))]
    pub async fn execute_without_saving(
        &self,
        thread: Arc<Thread>,
        _content: MessageContent,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        // Build context messages (includes the already-saved user message)
        let messages = self.build_context_messages(&thread).await?;

        // Run the execution loop
        self.execute_loop(thread, messages, sender).await
    }

    /// Execute with existing messages (for tool result continuations).
    async fn execute_loop(
        &self,
        thread: Arc<Thread>,
        mut messages: Vec<InputMessage>,
        sender: mpsc::Sender<String>,
    ) -> Result<()> {
        let mut iteration = 0;

        // Generate a single message_id for entire conversation
        let message_id = generate_message_id();
        let thread_id_str = thread.id.to_string();
        let mut message_started_sent = false;
        let mut total_block_index: usize = 0;
        let mut total_input_tokens: u32 = 0;
        let mut total_output_tokens: u32 = 0;

        loop {
            iteration += 1;

            if iteration > self.max_iterations {
                send_sse(
                    &sender,
                    SseEvent::error(
                        "max_iterations".to_string(),
                        "Max iterations exceeded".to_string(),
                    ),
                )
                .await;
                return Err(AgentError::MaxIterationsExceeded {
                    max: self.max_iterations,
                });
            }

            // Build request
            let mut request_builder = InferenceRequestBuilder::new()
                .messages(messages.clone())
                .stream();

            // Set model or function based on spec
            match &self.model_spec {
                ModelSpec::Function(name) => {
                    request_builder = request_builder.function_name(name);
                }
                ModelSpec::Model(name) => {
                    request_builder = request_builder.model(name);
                }
            }

            // Set system prompt if provided
            if let Some(ref prompt) = self.system_prompt {
                request_builder = request_builder.system(prompt);
            }

            // Set credentials if provided
            if let Some(ref creds) = self.credentials {
                request_builder = request_builder.credentials(creds.clone());
            }

            // Set tags if provided
            if let Some(ref tags) = self.tags {
                request_builder = request_builder.tags(tags.clone());
            }

            // Set tools if available
            if !self.tools.is_empty() {
                request_builder =
                    request_builder.additional_tools(self.tools.tensorzero_definitions());

                // Set tool choice (default to Auto if tools are present)
                let choice = self.tool_choice.clone().unwrap_or(ToolChoice::Auto);
                request_builder = request_builder.tool_choice(choice);
            }

            // Set parallel tool calls if specified
            if let Some(parallel) = self.parallel_tool_calls {
                request_builder = request_builder.parallel_tool_calls(parallel);
            }

            let request = request_builder.build()?;

            // Stream the response
            let (response_content, _inference_id, episode_id, chunk_usage) = self
                .stream_response(
                    request,
                    &sender,
                    &message_id,
                    &thread_id_str,
                    &mut message_started_sent,
                    &mut total_block_index,
                )
                .await?;

            // Accumulate usage
            total_input_tokens += chunk_usage.0;
            total_output_tokens += chunk_usage.1;

            // Check if we need to execute tools
            let tool_calls = extract_tool_calls(&response_content);

            if tool_calls.is_empty() {
                // No tool calls - we're done
                // Save assistant message
                let assistant_msg = Message::assistant(
                    thread.id,
                    response_content_to_message_content(&response_content),
                )
                .with_episode_id(episode_id);
                self.storage.create_message(&assistant_msg).await?;

                // Emit message.usage
                send_sse(
                    &sender,
                    SseEvent::message_usage(
                        message_id.clone(),
                        total_input_tokens,
                        total_output_tokens,
                    ),
                )
                .await;

                // Emit message.completed
                send_sse(
                    &sender,
                    SseEvent::message_completed(
                        message_id,
                        thread_id_str,
                        total_block_index,
                        "stop".to_string(),
                    ),
                )
                .await;

                return Ok(());
            }

            // Save assistant message with tool calls
            let assistant_msg = Message::assistant(
                thread.id,
                response_content_to_message_content(&response_content),
            )
            .with_episode_id(episode_id);
            self.storage.create_message(&assistant_msg).await?;

            // Add assistant message to context
            messages.push(response_to_input_message(&response_content));

            // Execute tools
            let tool_results = self
                .execute_tools(
                    &thread,
                    tool_calls,
                    &sender,
                    &message_id,
                    &mut total_block_index,
                )
                .await;

            // Save tool results as user message
            let tool_result_msg = tool_results_to_message(thread.id, &tool_results);
            self.storage.create_message(&tool_result_msg).await?;

            // Add tool results to context for next iteration
            for result in &tool_results {
                messages.push(InputMessage::tool_result(
                    &result.tool_call_id,
                    &result.tool_name,
                    &result.content,
                ));
            }
        }
    }

    /// Stream a single inference response.
    ///
    /// Returns (content_blocks, inference_id, episode_id, (input_tokens, output_tokens))
    async fn stream_response(
        &self,
        request: balungpisah_tensorzero::InferenceRequest,
        sender: &mpsc::Sender<String>,
        message_id: &str,
        thread_id: &str,
        message_started_sent: &mut bool,
        total_block_index: &mut usize,
    ) -> Result<(Vec<TzContentBlock>, Uuid, Uuid, (u32, u32))> {
        let mut stream = self.client.inference_stream(request).await?;

        let mut inference_id = Uuid::nil();
        let mut episode_id = Uuid::nil();
        // Track content blocks by their string ID
        let mut content_blocks: HashMap<String, ContentBlockBuilder> = HashMap::new();
        let mut input_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;

        while let Some(event) = stream.next().await {
            match event {
                Ok(chunk) => {
                    // Update IDs from chunk
                    inference_id = chunk.inference_id;
                    episode_id = chunk.episode_id;

                    // Update usage if present
                    if let Some(usage) = &chunk.usage {
                        input_tokens = usage.input_tokens;
                        output_tokens = usage.output_tokens;
                    }

                    // Send message.started event on first chunk
                    if !*message_started_sent {
                        send_sse(
                            sender,
                            SseEvent::message_started(
                                message_id.to_string(),
                                thread_id.to_string(),
                                "assistant".to_string(),
                                chunk.variant_name.clone(),
                            ),
                        )
                        .await;
                        *message_started_sent = true;
                    }

                    // Process each content block in the chunk
                    for block in &chunk.content {
                        self.handle_stream_content_block(
                            block,
                            &mut content_blocks,
                            sender,
                            message_id,
                            total_block_index,
                        )
                        .await;
                    }
                }
                Err(e) => {
                    warn!("Stream error: {}", e);
                    // Send error event to client
                    send_sse(
                        sender,
                        SseEvent::error("stream_error".to_string(), e.to_string()),
                    )
                    .await;
                }
            }
        }

        // Emit block.completed for all completed blocks
        for (_, builder) in content_blocks.iter_mut() {
            if !builder.complete {
                builder.complete = true;

                // Emit appropriate block.completed event
                if let Some(text) = &builder.text {
                    if !text.is_empty() {
                        send_sse(
                            sender,
                            SseEvent::block_completed_text(
                                message_id.to_string(),
                                builder.block_id.clone(),
                                text.clone(),
                            ),
                        )
                        .await;
                    }
                } else if let (Some(tool_call_id), Some(tool_name), Some(args)) = (
                    &builder.tool_call_id,
                    &builder.tool_call_name,
                    &builder.tool_call_arguments,
                ) {
                    let parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
                    send_sse(
                        sender,
                        SseEvent::block_completed_tool_call(
                            message_id.to_string(),
                            builder.block_id.clone(),
                            tool_name.clone(),
                            tool_call_id.clone(),
                            args.clone(),
                            parsed,
                        ),
                    )
                    .await;
                }
            }
        }

        // Build final content blocks
        let content: Vec<TzContentBlock> = content_blocks
            .into_iter()
            .filter_map(|(_, b)| b.build())
            .collect();

        Ok((
            content,
            inference_id,
            episode_id,
            (input_tokens, output_tokens),
        ))
    }

    /// Handle a single content block from a streaming chunk.
    async fn handle_stream_content_block(
        &self,
        block: &StreamContentBlock,
        blocks: &mut HashMap<String, ContentBlockBuilder>,
        sender: &mpsc::Sender<String>,
        message_id: &str,
        total_block_index: &mut usize,
    ) {
        let tensor_zero_id = block.id().to_string();

        match block {
            StreamContentBlock::Text { id: _, text } => {
                if let Some(builder) = blocks.get_mut(&tensor_zero_id) {
                    // Existing block - append text delta
                    builder.append_text(text);
                    if !text.is_empty() {
                        send_sse(
                            sender,
                            SseEvent::block_delta_text(
                                message_id.to_string(),
                                builder.block_id.clone(),
                                text.clone(),
                            ),
                        )
                        .await;
                    }
                } else {
                    // New text block
                    let block_id = generate_block_id();
                    let index = *total_block_index;
                    *total_block_index += 1;

                    // Emit block.created
                    send_sse(
                        sender,
                        SseEvent::block_created_text(
                            message_id.to_string(),
                            block_id.clone(),
                            index,
                        ),
                    )
                    .await;

                    blocks.insert(
                        tensor_zero_id,
                        ContentBlockBuilder::text(block_id.clone(), text.clone()),
                    );

                    // Emit first delta if non-empty
                    if !text.is_empty() {
                        send_sse(
                            sender,
                            SseEvent::block_delta_text(
                                message_id.to_string(),
                                block_id,
                                text.clone(),
                            ),
                        )
                        .await;
                    }
                }
            }
            StreamContentBlock::ToolCall {
                id,
                raw_name,
                raw_arguments,
            } => {
                if let Some(builder) = blocks.get_mut(&tensor_zero_id) {
                    // Existing tool call - append arguments delta
                    builder.append_arguments(raw_arguments);
                    if !raw_arguments.is_empty() {
                        send_sse(
                            sender,
                            SseEvent::block_delta_tool_call(
                                message_id.to_string(),
                                builder.block_id.clone(),
                                builder.tool_call_name.clone().unwrap_or_default(),
                                id.clone(),
                                raw_arguments.clone(),
                                builder.tool_call_arguments.clone().unwrap_or_default(),
                            ),
                        )
                        .await;
                    }
                } else {
                    // New tool call block
                    let block_id = generate_block_id();
                    let index = *total_block_index;
                    *total_block_index += 1;

                    // Emit block.created
                    send_sse(
                        sender,
                        SseEvent::block_created_tool_call(
                            message_id.to_string(),
                            block_id.clone(),
                            index,
                            raw_name.clone(),
                            id.clone(),
                        ),
                    )
                    .await;

                    blocks.insert(
                        tensor_zero_id,
                        ContentBlockBuilder::tool_call(
                            block_id.clone(),
                            id.clone(),
                            raw_name.clone(),
                            raw_arguments.clone(),
                        ),
                    );

                    // Emit first delta if non-empty
                    if !raw_arguments.is_empty() {
                        send_sse(
                            sender,
                            SseEvent::block_delta_tool_call(
                                message_id.to_string(),
                                block_id,
                                raw_name.clone(),
                                id.clone(),
                                raw_arguments.clone(),
                                raw_arguments.clone(),
                            ),
                        )
                        .await;
                    }
                }
            }
            StreamContentBlock::Thought {
                id: _,
                text,
                signature,
            } => {
                // Handle thought blocks similar to text (for reasoning models)
                if let Some(builder) = blocks.get_mut(&tensor_zero_id) {
                    if let Some(thought_text) = text {
                        builder.append_text(thought_text);
                        send_sse(
                            sender,
                            SseEvent::block_delta_thought(
                                message_id.to_string(),
                                builder.block_id.clone(),
                                Some(thought_text.clone()),
                                signature.clone(),
                            ),
                        )
                        .await;
                    }
                } else if text.is_some() || signature.is_some() {
                    let block_id = generate_block_id();
                    let index = *total_block_index;
                    *total_block_index += 1;

                    // Emit block.created
                    send_sse(
                        sender,
                        SseEvent::block_created_thought(
                            message_id.to_string(),
                            block_id.clone(),
                            index,
                        ),
                    )
                    .await;

                    blocks.insert(
                        tensor_zero_id,
                        ContentBlockBuilder::text(
                            block_id.clone(),
                            text.clone().unwrap_or_default(),
                        ),
                    );

                    // Emit first delta
                    send_sse(
                        sender,
                        SseEvent::block_delta_thought(
                            message_id.to_string(),
                            block_id,
                            text.clone(),
                            signature.clone(),
                        ),
                    )
                    .await;
                }
            }
        }
    }

    /// Execute tool calls.
    async fn execute_tools(
        &self,
        thread: &Arc<Thread>,
        tool_calls: Vec<(String, String, String, String)>, // (block_id, tool_call_id, name, arguments)
        sender: &mpsc::Sender<String>,
        message_id: &str,
        total_block_index: &mut usize,
    ) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for (block_id, tool_call_id, name, arguments) in tool_calls {
            // Parse arguments
            let args: Value = serde_json::from_str(&arguments).unwrap_or(Value::Null);

            // Emit tool.execution_started
            let started_at = Utc::now();
            send_sse(
                sender,
                SseEvent::tool_execution_started(
                    message_id.to_string(),
                    block_id.clone(),
                    tool_call_id.clone(),
                    name.clone(),
                    args.clone(),
                ),
            )
            .await;

            let result = if let Some(executor) = self.tools.get(&name) {
                // Create context
                let context = ToolContext::new(&tool_call_id, &name, thread.clone());

                // Execute
                debug!("Executing tool '{}' with args: {}", name, args);
                executor.execute(args.clone(), context).await
            } else {
                ToolResult::error(&tool_call_id, &name, format!("Tool '{}' not found", name))
            };

            let execution_time_ms = Utc::now()
                .signed_duration_since(started_at)
                .num_milliseconds() as u64;

            // Create tool_result block
            let result_block_id = generate_block_id();
            let result_index = *total_block_index;
            *total_block_index += 1;

            // Emit block.created for tool_result
            send_sse(
                sender,
                SseEvent::block_created_tool_result(
                    message_id.to_string(),
                    result_block_id.clone(),
                    result_index,
                    tool_call_id.clone(),
                    name.clone(),
                ),
            )
            .await;

            // Emit block.delta and block.completed for tool_result
            if result.is_error {
                send_sse(
                    sender,
                    SseEvent::block_delta_tool_result(
                        message_id.to_string(),
                        result_block_id.clone(),
                        tool_call_id.clone(),
                        name.clone(),
                        None,
                        Some(result.content.clone()),
                    ),
                )
                .await;

                send_sse(
                    sender,
                    SseEvent::block_completed_tool_result(
                        message_id.to_string(),
                        result_block_id,
                        tool_call_id.clone(),
                        name.clone(),
                        None,
                        Some(result.content.clone()),
                        execution_time_ms,
                    ),
                )
                .await;

                send_sse(
                    sender,
                    SseEvent::tool_execution_failed(
                        message_id.to_string(),
                        block_id,
                        tool_call_id,
                        name,
                        "tool_error".to_string(),
                        result.content.clone(),
                        None,
                        execution_time_ms,
                    ),
                )
                .await;
            } else {
                let result_json: Value = serde_json::from_str(&result.content)
                    .unwrap_or(Value::String(result.content.clone()));

                send_sse(
                    sender,
                    SseEvent::block_delta_tool_result(
                        message_id.to_string(),
                        result_block_id.clone(),
                        tool_call_id.clone(),
                        name.clone(),
                        Some(result_json.clone()),
                        None,
                    ),
                )
                .await;

                send_sse(
                    sender,
                    SseEvent::block_completed_tool_result(
                        message_id.to_string(),
                        result_block_id,
                        tool_call_id.clone(),
                        name.clone(),
                        Some(result_json.clone()),
                        None,
                        execution_time_ms,
                    ),
                )
                .await;

                send_sse(
                    sender,
                    SseEvent::tool_execution_completed(
                        message_id.to_string(),
                        block_id,
                        tool_call_id,
                        name,
                        result_json,
                        execution_time_ms,
                        started_at,
                    ),
                )
                .await;
            }

            results.push(result);
        }

        results
    }

    /// Build context messages from thread history.
    async fn build_context_messages(&self, thread: &Thread) -> Result<Vec<InputMessage>> {
        let stored_messages = self.storage.get_thread_messages(thread.id).await?;

        // Apply context config if configured
        let messages = if let Some(ref config) = self.context_config {
            config.filter_messages(stored_messages)
        } else {
            stored_messages
        };

        Ok(crate::context::convert_messages_to_input(&messages))
    }
}

/// Helper to build content blocks during streaming.
#[derive(Debug)]
struct ContentBlockBuilder {
    block_id: String,
    text: Option<String>,
    tool_call_id: Option<String>,
    tool_call_name: Option<String>,
    tool_call_arguments: Option<String>,
    complete: bool,
}

impl ContentBlockBuilder {
    fn text(block_id: String, initial: String) -> Self {
        Self {
            block_id,
            text: Some(initial),
            tool_call_id: None,
            tool_call_name: None,
            tool_call_arguments: None,
            complete: false,
        }
    }

    fn tool_call(block_id: String, tool_call_id: String, name: String, arguments: String) -> Self {
        Self {
            block_id,
            text: None,
            tool_call_id: Some(tool_call_id),
            tool_call_name: Some(name),
            tool_call_arguments: Some(arguments),
            complete: false,
        }
    }

    fn append_text(&mut self, text: &str) {
        if let Some(ref mut t) = self.text {
            t.push_str(text);
        }
    }

    fn append_arguments(&mut self, args: &str) {
        if let Some(ref mut a) = self.tool_call_arguments {
            a.push_str(args);
        }
    }

    fn build(self) -> Option<TzContentBlock> {
        if let Some(text) = self.text {
            Some(TzContentBlock::Text { text })
        } else if let (Some(id), Some(name), Some(arguments)) = (
            self.tool_call_id,
            self.tool_call_name,
            self.tool_call_arguments,
        ) {
            // Parse arguments string to Value for the new struct format
            let args_value: Option<Value> = serde_json::from_str(&arguments).ok();
            Some(TzContentBlock::ToolCall(
                balungpisah_tensorzero::ToolCallBlock {
                    id,
                    name: Some(name.clone()),
                    arguments: args_value,
                    raw_name: Some(name),
                    raw_arguments: Some(arguments),
                },
            ))
        } else {
            None
        }
    }
}

/// Extract tool calls from response content.
/// Returns tuples of (block_id, tool_call_id, name, arguments_string).
fn extract_tool_calls(content: &[TzContentBlock]) -> Vec<(String, String, String, String)> {
    content
        .iter()
        .filter_map(|block| {
            if let TzContentBlock::ToolCall(tc) = block {
                let name = tc.tool_name().unwrap_or("unknown").to_string();
                let args = tc.arguments_string();
                // Generate a block_id for tool execution tracking
                Some((generate_block_id(), tc.id.clone(), name, args))
            } else {
                None
            }
        })
        .collect()
}

/// Convert TensorZero content blocks to message content.
fn response_content_to_message_content(content: &[TzContentBlock]) -> MessageContent {
    let blocks: Vec<ContentBlock> = content
        .iter()
        .map(|block| match block {
            TzContentBlock::Text { text } => ContentBlock::text(text),
            TzContentBlock::ToolCall(tc) => {
                let input: Value = tc.parse_arguments().unwrap_or(Value::Null);
                let name = tc.tool_name().unwrap_or("unknown");
                ContentBlock::tool_use(&tc.id, name, input)
            }
        })
        .collect();

    MessageContent::Blocks(blocks)
}

/// Convert response content to input message for next iteration.
fn response_to_input_message(content: &[TzContentBlock]) -> InputMessage {
    let mut input_blocks = Vec::new();

    for block in content {
        match block {
            TzContentBlock::Text { text } => {
                input_blocks.push(balungpisah_tensorzero::InputContentBlock::Text {
                    r#type: "text".to_string(),
                    text: text.clone(),
                });
            }
            TzContentBlock::ToolCall(tc) => {
                let name = tc.tool_name().unwrap_or("unknown").to_string();
                let args = tc.parse_arguments().unwrap_or(Value::Null);
                input_blocks.push(balungpisah_tensorzero::InputContentBlock::ToolCall {
                    r#type: "tool_call".to_string(),
                    id: tc.id.clone(),
                    name,
                    arguments: args,
                });
            }
        }
    }

    InputMessage {
        role: balungpisah_tensorzero::MessageRole::Assistant,
        content: input_blocks,
    }
}

/// Convert tool results to a user message.
fn tool_results_to_message(thread_id: Uuid, results: &[ToolResult]) -> Message {
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

    let now = chrono::Utc::now();
    Message {
        id: Uuid::now_v7(),
        thread_id,
        role: Role::User,
        content: MessageContent::Blocks(blocks),
        episode_id: None,
        created_at: now,
        updated_at: now,
    }
}
