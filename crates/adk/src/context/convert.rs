//! Message format conversion utilities.

use crate::models::{ContentBlock, Message, MessageContent, Role};
use balungpisah_tensorzero::{InputContentBlock, InputMessage, MessageRole};

/// Convert stored messages to TensorZero input messages.
///
/// Handles both legacy format (separate messages for tool_use and tool_result)
/// and unified format (single assistant message with all content blocks).
pub fn convert_messages_to_input(messages: &[Message]) -> Vec<InputMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role {
            Role::User => {
                result.push(convert_message_to_input(message));
            }
            Role::Assistant => {
                if is_unified_format(&message.content) {
                    result.extend(split_unified_message(message));
                } else {
                    result.push(convert_message_to_input(message));
                }
            }
        }
    }

    result
}

/// Check if message contains both tool_use and tool_result (unified format).
///
/// The unified format stores all content from a tool execution loop in a single
/// assistant message: [tool_use, tool_result, text, tool_use, tool_result, ...]
fn is_unified_format(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(_) => false,
        MessageContent::Blocks(blocks) => {
            blocks.iter().any(|b| b.is_tool_use()) && blocks.iter().any(|b| b.is_tool_result())
        }
    }
}

/// Split unified message into LLM-expected format.
///
/// Takes a single assistant message with mixed content and splits it into
/// alternating assistant (tool_use, text) and user (tool_result) messages.
///
/// Example:
/// Input:  [tool_use_1, tool_result_1, text, tool_use_2, tool_result_2, final_text]
/// Output: [assistant(tool_use_1), user(tool_result_1), assistant(text, tool_use_2), user(tool_result_2), assistant(final_text)]
fn split_unified_message(message: &Message) -> Vec<InputMessage> {
    let blocks = match &message.content {
        MessageContent::Text(text) => {
            return vec![InputMessage {
                role: MessageRole::Assistant,
                content: vec![InputContentBlock::Text {
                    r#type: "text".to_string(),
                    text: text.clone(),
                }],
            }];
        }
        MessageContent::Blocks(blocks) => blocks,
    };

    let mut result = Vec::new();
    let mut assistant_buffer: Vec<InputContentBlock> = Vec::new();
    let mut user_buffer: Vec<InputContentBlock> = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::Text { .. } | ContentBlock::ToolUse { .. } => {
                // Flush user buffer first if not empty (role transition)
                if !user_buffer.is_empty() {
                    result.push(InputMessage {
                        role: MessageRole::User,
                        content: std::mem::take(&mut user_buffer),
                    });
                }

                // Add to assistant buffer
                if let Some(input_block) = convert_content_block(block) {
                    assistant_buffer.push(input_block);
                }
            }
            ContentBlock::ToolResult { .. } => {
                // Flush assistant buffer first if not empty (role transition)
                if !assistant_buffer.is_empty() {
                    result.push(InputMessage {
                        role: MessageRole::Assistant,
                        content: std::mem::take(&mut assistant_buffer),
                    });
                }

                // Add to user buffer
                if let Some(input_block) = convert_content_block(block) {
                    user_buffer.push(input_block);
                }
            }
            ContentBlock::Image { .. } | ContentBlock::File { .. } => {
                // Images and files go to assistant buffer
                if !user_buffer.is_empty() {
                    result.push(InputMessage {
                        role: MessageRole::User,
                        content: std::mem::take(&mut user_buffer),
                    });
                }

                if let Some(input_block) = convert_content_block(block) {
                    assistant_buffer.push(input_block);
                }
            }
        }
    }

    // Flush remaining buffers
    if !assistant_buffer.is_empty() {
        result.push(InputMessage {
            role: MessageRole::Assistant,
            content: assistant_buffer,
        });
    }
    if !user_buffer.is_empty() {
        result.push(InputMessage {
            role: MessageRole::User,
            content: user_buffer,
        });
    }

    result
}

/// Convert a single stored message to TensorZero input message.
pub fn convert_message_to_input(message: &Message) -> InputMessage {
    let role = match message.role {
        Role::User => MessageRole::User,
        Role::Assistant => MessageRole::Assistant,
    };

    let content = match &message.content {
        MessageContent::Text(text) => {
            vec![InputContentBlock::Text {
                r#type: "text".to_string(),
                text: text.clone(),
            }]
        }
        MessageContent::Blocks(blocks) => blocks.iter().filter_map(convert_content_block).collect(),
    };

    InputMessage { role, content }
}

/// Convert a content block to TensorZero input content block.
fn convert_content_block(block: &ContentBlock) -> Option<InputContentBlock> {
    match block {
        ContentBlock::Text { text } => Some(InputContentBlock::Text {
            r#type: "text".to_string(),
            text: text.clone(),
        }),
        ContentBlock::ToolUse { id, name, input } => Some(InputContentBlock::ToolCall {
            r#type: "tool_call".to_string(),
            id: id.clone(),
            name: name.clone(),
            arguments: input.clone(),
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            name,
            content,
            is_error: _,
        } => Some(InputContentBlock::ToolResult {
            r#type: "tool_result".to_string(),
            id: tool_use_id.clone(),
            name: name.clone(),
            result: content.clone(),
        }),
        ContentBlock::Image { .. } => {
            // Images not directly supported in TensorZero input format
            // Could be converted to base64 data URLs if needed
            None
        }
        ContentBlock::File {
            file_id,
            url,
            mime_type,
            data,
        } => {
            // Convert file blocks to appropriate format
            // Priority: URL > base64 data > file_id reference
            if let Some(url) = url {
                // If we have a URL, check if it's an image
                let is_image = mime_type
                    .as_ref()
                    .map(|m| m.starts_with("image/"))
                    .unwrap_or(false);

                if is_image {
                    // For images with URL, include as text with URL reference
                    // TensorZero may support image URLs directly in the future
                    Some(InputContentBlock::Text {
                        r#type: "text".to_string(),
                        text: format!("[Image: {}]", url),
                    })
                } else {
                    // For non-image files, include as text with URL reference
                    Some(InputContentBlock::Text {
                        r#type: "text".to_string(),
                        text: format!("[File: {}]", url),
                    })
                }
            } else if let (Some(_data), Some(mime)) = (data, mime_type) {
                // If we have base64 data and it's an image, we could potentially
                // convert to a data URL, but for now we'll note it as attached
                if mime.starts_with("image/") {
                    Some(InputContentBlock::Text {
                        r#type: "text".to_string(),
                        text: format!("[Attached image: {}]", mime),
                    })
                } else {
                    Some(InputContentBlock::Text {
                        r#type: "text".to_string(),
                        text: format!("[Attached file: {}]", mime),
                    })
                }
            } else {
                // If we only have a file_id, reference it
                // The URL should be resolved before sending to the model
                file_id.as_ref().map(|id| InputContentBlock::Text {
                    r#type: "text".to_string(),
                    text: format!("[File reference: {}]", id),
                })
            }
        }
    }
}

/// Convert TensorZero response content to stored message content.
pub fn convert_response_to_message_content(
    content: &[balungpisah_tensorzero::ContentBlock],
) -> MessageContent {
    let blocks: Vec<ContentBlock> = content
        .iter()
        .map(|block| match block {
            balungpisah_tensorzero::ContentBlock::Text { text } => ContentBlock::text(text),
            balungpisah_tensorzero::ContentBlock::ToolCall(tc) => {
                let input = tc.parse_arguments().unwrap_or(serde_json::Value::Null);
                let name = tc.tool_name().unwrap_or("unknown");
                ContentBlock::tool_use(&tc.id, name, input)
            }
        })
        .collect();

    if blocks.len() == 1 {
        if let ContentBlock::Text { text } = &blocks[0] {
            return MessageContent::Text(text.clone());
        }
    }

    MessageContent::Blocks(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn test_convert_text_message() {
        let message = Message::new(Uuid::new_v4(), Role::User, "Hello, world!");

        let input = convert_message_to_input(&message);

        assert!(matches!(input.role, MessageRole::User));
        assert_eq!(input.content.len(), 1);
    }

    #[test]
    fn test_convert_tool_use_message() {
        let blocks = vec![ContentBlock::tool_use(
            "call_123",
            "get_weather",
            json!({"location": "San Francisco"}),
        )];
        let message = Message::new(
            Uuid::new_v4(),
            Role::Assistant,
            MessageContent::Blocks(blocks),
        );

        let input = convert_message_to_input(&message);

        assert!(matches!(input.role, MessageRole::Assistant));
        assert_eq!(input.content.len(), 1);
    }

    #[test]
    fn test_convert_tool_result_message() {
        let blocks = vec![ContentBlock::tool_result(
            "call_123",
            "get_weather",
            "Sunny, 72°F",
        )];
        let message = Message::new(Uuid::new_v4(), Role::User, MessageContent::Blocks(blocks));

        let input = convert_message_to_input(&message);

        assert!(matches!(input.role, MessageRole::User));
        assert_eq!(input.content.len(), 1);
    }

    #[test]
    fn test_is_unified_format_detects_mixed_blocks() {
        // Unified format: has both tool_use and tool_result
        let blocks = vec![
            ContentBlock::tool_use("call_1", "get_weather", json!({"location": "NYC"})),
            ContentBlock::tool_result("call_1", "get_weather", "Sunny"),
            ContentBlock::text("Based on the weather..."),
        ];
        let content = MessageContent::Blocks(blocks);
        assert!(is_unified_format(&content));

        // Legacy format: only tool_use (no tool_result)
        let blocks = vec![ContentBlock::tool_use(
            "call_1",
            "get_weather",
            json!({"location": "NYC"}),
        )];
        let content = MessageContent::Blocks(blocks);
        assert!(!is_unified_format(&content));

        // Text only
        let content = MessageContent::Text("Hello".to_string());
        assert!(!is_unified_format(&content));
    }

    #[test]
    fn test_split_unified_message_basic() {
        let thread_id = Uuid::new_v4();
        let blocks = vec![
            ContentBlock::tool_use("call_1", "get_weather", json!({"location": "NYC"})),
            ContentBlock::tool_result("call_1", "get_weather", "Sunny, 72°F"),
            ContentBlock::text("The weather is nice!"),
        ];
        let message = Message::new(thread_id, Role::Assistant, MessageContent::Blocks(blocks));

        let result = split_unified_message(&message);

        // Should produce: assistant(tool_use), user(tool_result), assistant(text)
        assert_eq!(result.len(), 3);
        assert!(matches!(result[0].role, MessageRole::Assistant));
        assert!(matches!(result[1].role, MessageRole::User));
        assert!(matches!(result[2].role, MessageRole::Assistant));
    }

    #[test]
    fn test_split_unified_message_multiple_iterations() {
        let thread_id = Uuid::new_v4();
        let blocks = vec![
            // First iteration
            ContentBlock::tool_use("call_1", "search", json!({"q": "rust"})),
            ContentBlock::tool_result("call_1", "search", "Found results..."),
            // Second iteration
            ContentBlock::tool_use("call_2", "read_file", json!({"path": "foo.rs"})),
            ContentBlock::tool_result("call_2", "read_file", "fn main() {}"),
            // Final text
            ContentBlock::text("Based on my analysis..."),
        ];
        let message = Message::new(thread_id, Role::Assistant, MessageContent::Blocks(blocks));

        let result = split_unified_message(&message);

        // Should produce alternating assistant/user messages
        assert_eq!(result.len(), 5);
        assert!(matches!(result[0].role, MessageRole::Assistant)); // tool_use 1
        assert!(matches!(result[1].role, MessageRole::User)); // tool_result 1
        assert!(matches!(result[2].role, MessageRole::Assistant)); // tool_use 2
        assert!(matches!(result[3].role, MessageRole::User)); // tool_result 2
        assert!(matches!(result[4].role, MessageRole::Assistant)); // final text
    }

    #[test]
    fn test_backwards_compat_legacy_format() {
        let thread_id = Uuid::new_v4();

        // Legacy assistant message with only tool_use
        let assistant_blocks = vec![ContentBlock::tool_use(
            "call_1",
            "get_weather",
            json!({"location": "NYC"}),
        )];
        let assistant_msg = Message::new(
            thread_id,
            Role::Assistant,
            MessageContent::Blocks(assistant_blocks),
        );

        // Legacy user message with only tool_result
        let user_blocks = vec![ContentBlock::tool_result("call_1", "get_weather", "Sunny")];
        let user_msg = Message::new(thread_id, Role::User, MessageContent::Blocks(user_blocks));

        let messages = vec![assistant_msg, user_msg];
        let result = convert_messages_to_input(&messages);

        // Should convert directly without splitting (backwards compatible)
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].role, MessageRole::Assistant));
        assert!(matches!(result[1].role, MessageRole::User));
    }

    #[test]
    fn test_convert_messages_handles_mixed_formats() {
        let thread_id = Uuid::new_v4();

        // User message (simple text)
        let user_msg = Message::new(thread_id, Role::User, "What's the weather?");

        // Unified assistant message
        let unified_blocks = vec![
            ContentBlock::tool_use("call_1", "get_weather", json!({"location": "NYC"})),
            ContentBlock::tool_result("call_1", "get_weather", "Sunny"),
            ContentBlock::text("It's sunny in NYC!"),
        ];
        let assistant_msg = Message::new(
            thread_id,
            Role::Assistant,
            MessageContent::Blocks(unified_blocks),
        );

        let messages = vec![user_msg, assistant_msg];
        let result = convert_messages_to_input(&messages);

        // User message + split unified message (3 parts)
        assert_eq!(result.len(), 4);
        assert!(matches!(result[0].role, MessageRole::User)); // Original user
        assert!(matches!(result[1].role, MessageRole::Assistant)); // tool_use
        assert!(matches!(result[2].role, MessageRole::User)); // tool_result
        assert!(matches!(result[3].role, MessageRole::Assistant)); // text
    }
}
