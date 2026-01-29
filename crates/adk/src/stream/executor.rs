//! Stream executor for handling streaming responses with tool execution.

use super::config::StreamConfig;
use super::events::SseEvent;
use crate::context::ContextFilter;
use crate::error::{AgentError, Result};
use crate::models::{ContentBlock, Message, MessageContent, Role, Thread};
use crate::storage::{MessageStorage, ThreadStorage};
use crate::tools::{ToolContext, ToolRegistry, ToolResult};
use balungpisah_tensorzero::{
    ContentBlock as TzContentBlock, ContentBlockDeltaEvent, ContentBlockStartEvent,
    ContentBlockType, Delta, InferenceRequestBuilder, InputMessage, StreamEvent, TensorZeroClient,
    ToolChoice,
};
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
    /// TensorZero function name.
    function_name: String,
    /// Maximum iterations for tool execution loop.
    max_iterations: usize,
    /// Stream configuration.
    config: StreamConfig,
    /// Context filter.
    context_filter: Option<ContextFilter>,
}

impl<S> std::fmt::Debug for StreamExecutor<S>
where
    S: ThreadStorage + MessageStorage + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamExecutor")
            .field("function_name", &self.function_name)
            .field("max_iterations", &self.max_iterations)
            .field("tools", &self.tools.names())
            .finish()
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
        function_name: impl Into<String>,
    ) -> Self {
        Self {
            client,
            storage,
            tools,
            function_name: function_name.into(),
            max_iterations: 10,
            config: StreamConfig::default(),
            context_filter: None,
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

    /// Set context filter.
    pub fn context_filter(mut self, filter: ContextFilter) -> Self {
        self.context_filter = Some(filter);
        self
    }

    /// Execute a chat request with streaming.
    #[instrument(skip(self, sender), fields(thread_id = %thread.id))]
    pub async fn execute(
        &self,
        thread: Arc<Thread>,
        user_message: &str,
        sender: mpsc::Sender<SseEvent>,
    ) -> Result<()> {
        // Save user message
        let user_msg = Message::user(thread.id, user_message);
        self.storage.create_message(&user_msg).await?;

        // Build context messages
        let messages = self.build_context_messages(&thread).await?;

        // Run the execution loop
        self.execute_loop(thread, messages, sender).await
    }

    /// Execute with existing messages (for tool result continuations).
    async fn execute_loop(
        &self,
        thread: Arc<Thread>,
        mut messages: Vec<InputMessage>,
        sender: mpsc::Sender<SseEvent>,
    ) -> Result<()> {
        let mut iteration = 0;

        loop {
            iteration += 1;

            if iteration > self.max_iterations {
                let _ = sender
                    .send(SseEvent::error("Max iterations exceeded"))
                    .await;
                return Err(AgentError::MaxIterationsExceeded {
                    max: self.max_iterations,
                });
            }

            // Send iteration start
            let _ = sender
                .send(SseEvent::IterationStart {
                    iteration,
                    max_iterations: self.max_iterations,
                })
                .await;

            // Build request
            let additional_tools = if !self.tools.is_empty() {
                Some(self.tools.tensorzero_definitions())
            } else {
                None
            };

            let mut request_builder = InferenceRequestBuilder::new()
                .function_name(&self.function_name)
                .messages(messages.clone())
                .stream();

            if let Some(episode_id) = thread.episode_id {
                request_builder = request_builder.episode_id(episode_id);
            }

            if let Some(tools) = additional_tools {
                request_builder = request_builder
                    .additional_tools(tools)
                    .tool_choice(ToolChoice::Auto);
            }

            let request = request_builder.build()?;

            // Stream the response
            let (response_content, inference_id, episode_id) =
                self.stream_response(request, &sender).await?;

            // Send iteration complete
            let _ = sender.send(SseEvent::IterationComplete { iteration }).await;

            // Check if we need to execute tools
            let tool_calls = extract_tool_calls(&response_content);

            if tool_calls.is_empty() {
                // No tool calls - we're done
                let final_text = extract_text(&response_content);

                // Save assistant message
                let assistant_msg = Message::assistant(
                    thread.id,
                    response_content_to_message_content(&response_content),
                )
                .with_inference_id(inference_id)
                .with_episode_id(episode_id);
                self.storage.create_message(&assistant_msg).await?;

                // Update thread episode ID if needed
                if thread.episode_id.is_none() {
                    let mut updated_thread = (*thread).clone();
                    updated_thread.episode_id = Some(episode_id);
                    self.storage.update_thread(&updated_thread).await?;
                }

                let _ = sender.send(SseEvent::done(final_text, iteration)).await;
                return Ok(());
            }

            // Save assistant message with tool calls
            let assistant_msg = Message::assistant(
                thread.id,
                response_content_to_message_content(&response_content),
            )
            .with_inference_id(inference_id)
            .with_episode_id(episode_id);
            self.storage.create_message(&assistant_msg).await?;

            // Add assistant message to context
            messages.push(response_to_input_message(&response_content));

            // Execute tools
            let tool_results = self.execute_tools(&thread, tool_calls, &sender).await;

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
    async fn stream_response(
        &self,
        request: balungpisah_tensorzero::InferenceRequest,
        sender: &mpsc::Sender<SseEvent>,
    ) -> Result<(Vec<TzContentBlock>, Uuid, Uuid)> {
        let mut stream = self.client.inference_stream(request).await?;

        let mut inference_id = Uuid::nil();
        let mut episode_id = Uuid::nil();
        let mut content_blocks: HashMap<usize, ContentBlockBuilder> = HashMap::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::Chunk(chunk)) => {
                    inference_id = chunk.inference_id;
                    episode_id = chunk.episode_id;
                }
                Ok(StreamEvent::ContentBlockStart(start)) => {
                    self.handle_content_block_start(start, &mut content_blocks, sender)
                        .await;
                }
                Ok(StreamEvent::ContentBlockDelta(delta)) => {
                    self.handle_content_block_delta(delta, &mut content_blocks, sender)
                        .await;
                }
                Ok(StreamEvent::ContentBlockStop(stop)) => {
                    if let Some(builder) = content_blocks.get_mut(&stop.index) {
                        builder.complete = true;

                        // Send tool use complete if applicable
                        if let Some((id, args)) = builder.as_tool_call() {
                            let _ = sender.send(SseEvent::tool_use_complete(id, args)).await;
                        }
                    }
                }
                Ok(StreamEvent::Done(_)) => {
                    break;
                }
                Err(e) => {
                    warn!("Stream error: {}", e);
                }
            }
        }

        // Build final content blocks
        let content: Vec<TzContentBlock> = content_blocks
            .into_iter()
            .filter_map(|(_, b)| b.build())
            .collect();

        Ok((content, inference_id, episode_id))
    }

    async fn handle_content_block_start(
        &self,
        event: ContentBlockStartEvent,
        blocks: &mut HashMap<usize, ContentBlockBuilder>,
        sender: &mpsc::Sender<SseEvent>,
    ) {
        match &event.content_block {
            ContentBlockType::Text { text } => {
                blocks.insert(event.index, ContentBlockBuilder::text(text.clone()));
                if !text.is_empty() {
                    let _ = sender.send(SseEvent::text_delta(text)).await;
                }
            }
            ContentBlockType::ToolCall(tc) => {
                // Get name from raw_name or name field
                let name = tc.raw_name.clone()
                    .or_else(|| tc.name.clone())
                    .unwrap_or_default();
                // Get arguments from raw_arguments or arguments field
                let args = tc.raw_arguments.clone()
                    .or_else(|| tc.arguments.clone())
                    .unwrap_or_default();
                blocks.insert(
                    event.index,
                    ContentBlockBuilder::tool_call(tc.id.clone(), name.clone(), args),
                );
                let _ = sender.send(SseEvent::tool_use_start(&tc.id, &name)).await;
            }
        }
    }

    async fn handle_content_block_delta(
        &self,
        event: ContentBlockDeltaEvent,
        blocks: &mut HashMap<usize, ContentBlockBuilder>,
        sender: &mpsc::Sender<SseEvent>,
    ) {
        if let Some(builder) = blocks.get_mut(&event.index) {
            match &event.delta {
                Delta::TextDelta { text } => {
                    builder.append_text(text);
                    let _ = sender.send(SseEvent::text_delta(text)).await;
                }
                Delta::ToolCallDelta { arguments } => {
                    let tool_id = builder.tool_call_id().map(|s| s.to_string());
                    builder.append_arguments(arguments);
                    if let Some(id) = tool_id {
                        let _ = sender.send(SseEvent::tool_use_delta(id, arguments)).await;
                    }
                }
            }
        }
    }

    /// Execute tool calls.
    async fn execute_tools(
        &self,
        thread: &Arc<Thread>,
        tool_calls: Vec<(String, String, String)>, // (id, name, arguments)
        sender: &mpsc::Sender<SseEvent>,
    ) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for (id, name, arguments) in tool_calls {
            // Send execution start
            let _ = sender
                .send(SseEvent::tool_execution_start(&id, &name))
                .await;

            let result = if let Some(executor) = self.tools.get(&name) {
                // Parse arguments
                let args: Value = serde_json::from_str(&arguments).unwrap_or(Value::Null);

                // Create context
                let context = ToolContext::new(&id, &name, thread.clone());

                // Execute
                debug!("Executing tool '{}' with args: {}", name, args);
                executor.execute(args, context).await
            } else {
                ToolResult::error(&id, &name, format!("Tool '{}' not found", name))
            };

            // Send execution complete
            let _ = sender
                .send(SseEvent::tool_execution_complete(
                    &id,
                    &result.content,
                    result.is_error,
                ))
                .await;

            results.push(result);
        }

        results
    }

    /// Build context messages from thread history.
    async fn build_context_messages(&self, thread: &Thread) -> Result<Vec<InputMessage>> {
        let stored_messages = self.storage.get_thread_messages(thread.id).await?;

        // Apply context filter if configured
        let messages = if let Some(ref filter) = self.context_filter {
            filter.filter_messages(stored_messages)
        } else {
            stored_messages
        };

        Ok(crate::context::convert_messages_to_input(&messages))
    }
}

/// Helper to build content blocks during streaming.
#[derive(Debug)]
struct ContentBlockBuilder {
    text: Option<String>,
    tool_call_id: Option<String>,
    tool_call_name: Option<String>,
    tool_call_arguments: Option<String>,
    complete: bool,
}

impl ContentBlockBuilder {
    fn text(initial: String) -> Self {
        Self {
            text: Some(initial),
            tool_call_id: None,
            tool_call_name: None,
            tool_call_arguments: None,
            complete: false,
        }
    }

    fn tool_call(id: String, name: String, arguments: String) -> Self {
        Self {
            text: None,
            tool_call_id: Some(id),
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

    fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    fn as_tool_call(&self) -> Option<(&str, &str)> {
        match (&self.tool_call_id, &self.tool_call_arguments) {
            (Some(id), Some(args)) => Some((id, args)),
            _ => None,
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
/// Returns tuples of (id, name, arguments_string).
fn extract_tool_calls(content: &[TzContentBlock]) -> Vec<(String, String, String)> {
    content
        .iter()
        .filter_map(|block| {
            if let TzContentBlock::ToolCall(tc) = block {
                let name = tc.tool_name().unwrap_or("unknown").to_string();
                let args = tc.arguments_string();
                Some((tc.id.clone(), name, args))
            } else {
                None
            }
        })
        .collect()
}

/// Extract text from response content.
fn extract_text(content: &[TzContentBlock]) -> Option<String> {
    let text: String = content
        .iter()
        .filter_map(|block| {
            if let TzContentBlock::Text { text } = block {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();

    if text.is_empty() {
        None
    } else {
        Some(text)
    }
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
