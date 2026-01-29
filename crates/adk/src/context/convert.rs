//! Message format conversion utilities.

use crate::models::{ContentBlock, Message, MessageContent, Role};
use balungpisah_tensorzero::{InputContentBlock, InputMessage, MessageRole};

/// Convert stored messages to TensorZero input messages.
pub fn convert_messages_to_input(messages: &[Message]) -> Vec<InputMessage> {
    messages.iter().map(convert_message_to_input).collect()
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
        let blocks = vec![ContentBlock::tool_result("call_123", "get_weather", "Sunny, 72°F")];
        let message = Message::new(Uuid::new_v4(), Role::User, MessageContent::Blocks(blocks));

        let input = convert_message_to_input(&message);

        assert!(matches!(input.role, MessageRole::User));
        assert_eq!(input.content.len(), 1);
    }
}
