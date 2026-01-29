//! Context filtering for message history.

use crate::models::{Message, Role};

/// Filter configuration for context messages.
#[derive(Debug, Clone)]
pub struct ContextFilter {
    /// Maximum number of messages to include.
    pub max_messages: Option<usize>,
    /// Maximum total characters in context.
    pub max_characters: Option<usize>,
    /// Whether to always include the system message.
    pub include_system: bool,
    /// Whether to always include tool result pairs.
    pub preserve_tool_pairs: bool,
}

impl Default for ContextFilter {
    fn default() -> Self {
        Self {
            max_messages: None,
            max_characters: None,
            include_system: true,
            preserve_tool_pairs: true,
        }
    }
}

impl ContextFilter {
    /// Create a new filter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of messages.
    pub fn max_messages(mut self, max: usize) -> Self {
        self.max_messages = Some(max);
        self
    }

    /// Set maximum total characters.
    pub fn max_characters(mut self, max: usize) -> Self {
        self.max_characters = Some(max);
        self
    }

    /// Set whether to include system messages.
    pub fn include_system(mut self, include: bool) -> Self {
        self.include_system = include;
        self
    }

    /// Set whether to preserve tool call/result pairs.
    pub fn preserve_tool_pairs(mut self, preserve: bool) -> Self {
        self.preserve_tool_pairs = preserve;
        self
    }

    /// Filter messages according to the configuration.
    pub fn filter_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        if messages.is_empty() {
            return messages;
        }

        let mut filtered = messages;

        // Apply message count limit
        if let Some(max) = self.max_messages {
            if filtered.len() > max {
                // Keep the most recent messages, but ensure we don't break tool pairs
                let start_idx = filtered.len().saturating_sub(max);
                filtered = filtered.into_iter().skip(start_idx).collect();

                // Ensure we start with a user message if possible
                while !filtered.is_empty() && filtered[0].role == Role::Assistant {
                    filtered.remove(0);
                }
            }
        }

        // Apply character limit
        if let Some(max_chars) = self.max_characters {
            let mut total_chars = 0;
            let mut keep_from = 0;

            // Start from the end and work backwards
            for (i, msg) in filtered.iter().enumerate().rev() {
                let msg_chars = msg.text().len();
                if total_chars + msg_chars > max_chars && i > 0 {
                    keep_from = i + 1;
                    break;
                }
                total_chars += msg_chars;
            }

            if keep_from > 0 {
                filtered = filtered.into_iter().skip(keep_from).collect();

                // Ensure we start with a user message
                while !filtered.is_empty() && filtered[0].role == Role::Assistant {
                    filtered.remove(0);
                }
            }
        }

        // Ensure tool call/result pairs are preserved
        if self.preserve_tool_pairs {
            filtered = Self::ensure_tool_pairs(filtered);
        }

        filtered
    }

    /// Ensure tool call and result pairs are not separated.
    fn ensure_tool_pairs(messages: Vec<Message>) -> Vec<Message> {
        if messages.is_empty() {
            return messages;
        }

        let mut result: Vec<Message> = Vec::with_capacity(messages.len());
        let mut i = 0;

        while i < messages.len() {
            let msg = &messages[i];

            // If this is a tool result without its preceding tool call, skip it
            if msg.role == Role::User {
                let content_blocks = msg.content.as_blocks();
                let is_tool_result = content_blocks.iter().any(|b| b.is_tool_result());

                if is_tool_result && result.is_empty() {
                    // Skip orphaned tool results at the start
                    i += 1;
                    continue;
                }

                if is_tool_result {
                    // Check if the previous message has the corresponding tool call
                    if let Some(prev) = result.last() {
                        if !prev.has_tool_uses() {
                            // Skip orphaned tool result
                            i += 1;
                            continue;
                        }
                    }
                }
            }

            result.push(messages[i].clone());
            i += 1;
        }

        // Ensure we don't end with an assistant message with tool calls that have no results
        while !result.is_empty() {
            let last = result.last().unwrap();
            if last.role == Role::Assistant && last.has_tool_uses() {
                result.pop();
            } else {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MessageContent;
    use uuid::Uuid;

    fn make_message(role: Role, text: &str) -> Message {
        Message::new(Uuid::new_v4(), role, MessageContent::text(text))
    }

    #[test]
    fn test_max_messages_filter() {
        let filter = ContextFilter::new().max_messages(2);

        let messages = vec![
            make_message(Role::User, "First"),
            make_message(Role::Assistant, "Response 1"),
            make_message(Role::User, "Second"),
            make_message(Role::Assistant, "Response 2"),
        ];

        let filtered = filter.filter_messages(messages);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].text(), "Second");
    }

    #[test]
    fn test_empty_messages() {
        let filter = ContextFilter::new();
        let filtered = filter.filter_messages(vec![]);
        assert!(filtered.is_empty());
    }
}
