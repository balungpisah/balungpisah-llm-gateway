//! Context configuration for agent conversations.
//!
//! This module provides configuration for how conversation context is managed,
//! including message limits, tool message handling, and loop-specific overrides.

use serde::{Deserialize, Serialize};

/// Configuration for how tool-related messages are handled in context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Number of recent tool call/result pairs to retain.
    /// Older tool interactions will be summarized or removed.
    pub retain_last: usize,

    /// Maximum number of tool results to include per message.
    /// If a message has more tool results, older ones are truncated.
    pub limit_per_message: Option<usize>,

    /// Whether to deduplicate repeated tool calls with same arguments.
    pub deduplicate: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            retain_last: 5,
            limit_per_message: None,
            deduplicate: false,
        }
    }
}

impl ToolsConfig {
    /// Create a new tools context config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of recent tool pairs to retain.
    pub fn retain_last(mut self, count: usize) -> Self {
        self.retain_last = count;
        self
    }

    /// Set the maximum tool results per message.
    pub fn limit_per_message(mut self, limit: usize) -> Self {
        self.limit_per_message = Some(limit);
        self
    }

    /// Enable or disable tool call deduplication.
    pub fn deduplicate(mut self, enabled: bool) -> Self {
        self.deduplicate = enabled;
        self
    }
}

/// Configuration for conversation context management.
///
/// Controls how messages are filtered and managed when building
/// context for inference requests.
///
/// # Example
///
/// ```
/// use balungpisah_adk::context::{ContextConfig, ToolsConfig};
///
/// let config = ContextConfig::new()
///     .max_messages(10)
///     .tools(ToolsConfig::new()
///         .retain_last(3)
///         .limit_per_message(3))
///     .loop_override(ContextConfig::new()
///         .max_messages(3)
///         .tools(ToolsConfig::new().retain_last(1)));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum number of messages to include in context.
    pub max_messages: usize,

    /// Configuration for tool message handling.
    pub tools: ToolsConfig,

    /// Override configuration for tool execution loops.
    /// During multi-turn tool execution, this config is used instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_override: Option<Box<ContextConfig>>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_messages: 20,
            tools: ToolsConfig::default(),
            loop_override: None,
        }
    }
}

impl ContextConfig {
    /// Create a new context config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of messages.
    pub fn max_messages(mut self, max: usize) -> Self {
        self.max_messages = max;
        self
    }

    /// Set the tools configuration.
    pub fn tools(mut self, config: ToolsConfig) -> Self {
        self.tools = config;
        self
    }

    /// Set an override config for tool execution loops.
    pub fn loop_override(mut self, config: ContextConfig) -> Self {
        self.loop_override = Some(Box::new(config));
        self
    }

    /// Get the config to use for a tool execution loop.
    pub fn for_loop(&self) -> &ContextConfig {
        self.loop_override.as_deref().unwrap_or(self)
    }

    /// Filter messages according to this configuration.
    ///
    /// Applies the following filters in order:
    /// 1. Limit total messages to `max_messages`
    /// 2. Apply `tools.deduplicate` to remove repeated tool calls
    /// 3. Apply `tools.retain_last` to keep only recent tool pairs
    /// 4. Apply `tools.limit_per_message` to truncate tool results per message
    ///
    /// The filtering ensures conversation coherence by:
    /// - Starting with a user message
    /// - Not breaking tool call/result pairs
    pub fn filter_messages(
        &self,
        messages: Vec<crate::models::Message>,
    ) -> Vec<crate::models::Message> {
        use crate::models::Role;

        if messages.is_empty() {
            return messages;
        }

        let mut filtered = messages;

        // Step 1: Apply max_messages limit
        if filtered.len() > self.max_messages {
            let start_idx = filtered.len().saturating_sub(self.max_messages);
            filtered = filtered.into_iter().skip(start_idx).collect();
        }

        // Step 2: Apply tools.deduplicate - remove repeated tool calls with same name+args
        if self.tools.deduplicate {
            filtered = self.apply_deduplicate(filtered);
        }

        // Step 3: Apply tools.retain_last - keep only N recent tool call/result pairs
        filtered = self.apply_retain_last(filtered);

        // Step 4: Apply tools.limit_per_message - limit tool blocks per message
        if let Some(limit) = self.tools.limit_per_message {
            filtered = self.apply_limit_per_message(filtered, limit);
        }

        // Step 5: Ensure we start with a user message (not an assistant or tool result)
        while !filtered.is_empty() && filtered[0].role == Role::Assistant {
            filtered.remove(0);
        }

        // Step 6: Ensure we don't have orphaned tool results at the start
        filtered = self.remove_orphaned_tool_results(filtered);

        filtered
    }

    /// Apply retain_last to keep only N recent tool call/result pairs.
    fn apply_retain_last(
        &self,
        messages: Vec<crate::models::Message>,
    ) -> Vec<crate::models::Message> {
        use crate::models::Role;

        let retain_last = self.tools.retain_last;
        if retain_last == 0 {
            // Remove all tool-related messages
            return messages
                .into_iter()
                .filter(|m| !m.has_tool_uses() && !self.is_tool_result_message(m))
                .collect();
        }

        // Count tool call/result pairs from the end
        let mut tool_pair_count = 0;
        let mut pair_boundaries: Vec<usize> = Vec::new();

        // Find tool pairs by scanning backwards
        let mut i = messages.len();
        while i > 0 {
            i -= 1;
            let msg = &messages[i];

            // If this is a user message with tool results, it's the end of a pair
            if msg.role == Role::User && self.is_tool_result_message(msg) {
                // Find the preceding assistant message with tool calls
                if i > 0 && messages[i - 1].role == Role::Assistant && messages[i - 1].has_tool_uses()
                {
                    tool_pair_count += 1;
                    if tool_pair_count > retain_last {
                        pair_boundaries.push(i - 1); // Mark the assistant message for removal
                        pair_boundaries.push(i); // Mark the tool result message for removal
                    }
                    i -= 1; // Skip the assistant message we just processed
                }
            }
        }

        if pair_boundaries.is_empty() {
            return messages;
        }

        // Remove marked messages
        messages
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !pair_boundaries.contains(idx))
            .map(|(_, m)| m)
            .collect()
    }

    /// Check if a message contains tool results.
    fn is_tool_result_message(&self, message: &crate::models::Message) -> bool {
        message
            .content
            .as_blocks()
            .iter()
            .any(|b| b.is_tool_result())
    }

    /// Apply deduplication to remove repeated tool calls with same name and arguments.
    fn apply_deduplicate(
        &self,
        messages: Vec<crate::models::Message>,
    ) -> Vec<crate::models::Message> {
        use crate::models::{ContentBlock, MessageContent, Role};
        use std::collections::HashSet;

        let mut seen_tool_calls: HashSet<String> = HashSet::new();
        let mut result = Vec::with_capacity(messages.len());

        for msg in messages {
            if msg.role == Role::Assistant && msg.has_tool_uses() {
                // Filter tool use blocks to remove duplicates
                let blocks = msg.content.as_blocks();
                let mut unique_blocks: Vec<ContentBlock> = Vec::new();
                let mut has_duplicates = false;

                for block in blocks {
                    if let Some((_, name, input)) = block.as_tool_use() {
                        // Create a fingerprint from name + arguments
                        let fingerprint = format!("{}:{}", name, input);
                        if seen_tool_calls.contains(&fingerprint) {
                            has_duplicates = true;
                            continue; // Skip duplicate
                        }
                        seen_tool_calls.insert(fingerprint);
                    }
                    unique_blocks.push(block.clone());
                }

                if has_duplicates {
                    // Reconstruct message with unique blocks only
                    if unique_blocks.is_empty() {
                        continue; // Skip entirely empty messages
                    }
                    let mut new_msg = msg.clone();
                    new_msg.content = MessageContent::Blocks(unique_blocks);
                    result.push(new_msg);
                } else {
                    result.push(msg);
                }
            } else {
                result.push(msg);
            }
        }

        result
    }

    /// Apply limit_per_message to truncate tool blocks in each message.
    fn apply_limit_per_message(
        &self,
        messages: Vec<crate::models::Message>,
        limit: usize,
    ) -> Vec<crate::models::Message> {
        use crate::models::{ContentBlock, MessageContent, Role};

        messages
            .into_iter()
            .map(|msg| {
                // Only limit tool-related messages
                let is_tool_msg = (msg.role == Role::Assistant && msg.has_tool_uses())
                    || (msg.role == Role::User && self.is_tool_result_message(&msg));

                if !is_tool_msg {
                    return msg;
                }

                let blocks = msg.content.as_blocks();

                // Count tool-related blocks
                let tool_block_count = blocks
                    .iter()
                    .filter(|b| b.is_tool_use() || b.is_tool_result())
                    .count();

                if tool_block_count <= limit {
                    return msg;
                }

                // Keep only the last `limit` tool blocks, preserve text blocks
                let mut tool_count = 0;
                let filtered_blocks: Vec<ContentBlock> = blocks
                    .into_iter()
                    .rev()
                    .filter(|b| {
                        if b.is_tool_use() || b.is_tool_result() {
                            tool_count += 1;
                            tool_count <= limit
                        } else {
                            true // Keep text blocks
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();

                let mut new_msg = msg;
                new_msg.content = MessageContent::Blocks(filtered_blocks);
                new_msg
            })
            .collect()
    }

    /// Remove orphaned tool results at the start of the conversation.
    fn remove_orphaned_tool_results(
        &self,
        messages: Vec<crate::models::Message>,
    ) -> Vec<crate::models::Message> {
        use crate::models::Role;

        if messages.is_empty() {
            return messages;
        }

        let mut result = Vec::with_capacity(messages.len());
        let mut skip_until_non_tool = true;

        for msg in messages {
            if skip_until_non_tool {
                // Skip tool result messages at the start
                if msg.role == Role::User && self.is_tool_result_message(&msg) {
                    continue;
                }
                skip_until_non_tool = false;
            }
            result.push(msg);
        }

        result
    }

    /// Create a minimal config for tight context (useful for loops).
    pub fn minimal() -> Self {
        Self {
            max_messages: 5,
            tools: ToolsConfig {
                retain_last: 1,
                limit_per_message: Some(3),
                deduplicate: false,
            },
            loop_override: None,
        }
    }

    /// Create a standard config for typical conversations.
    pub fn standard() -> Self {
        Self {
            max_messages: 10,
            tools: ToolsConfig {
                retain_last: 3,
                limit_per_message: Some(5),
                deduplicate: false,
            },
            loop_override: Some(Box::new(Self::minimal())),
        }
    }

    /// Create a large context config for complex conversations.
    pub fn large() -> Self {
        Self {
            max_messages: 30,
            tools: ToolsConfig {
                retain_last: 10,
                limit_per_message: None,
                deduplicate: true,
            },
            loop_override: Some(Box::new(Self {
                max_messages: 10,
                tools: ToolsConfig {
                    retain_last: 3,
                    limit_per_message: Some(5),
                    deduplicate: false,
                },
                loop_override: None,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ContextConfig::default();
        assert_eq!(config.max_messages, 20);
        assert_eq!(config.tools.retain_last, 5);
        assert!(config.loop_override.is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let config = ContextConfig::new()
            .max_messages(15)
            .tools(ToolsConfig::new().retain_last(3).deduplicate(true));

        assert_eq!(config.max_messages, 15);
        assert_eq!(config.tools.retain_last, 3);
        assert!(config.tools.deduplicate);
    }

    #[test]
    fn test_loop_override() {
        let config = ContextConfig::new()
            .max_messages(10)
            .loop_override(ContextConfig::new().max_messages(3));

        assert_eq!(config.max_messages, 10);
        assert_eq!(config.for_loop().max_messages, 3);
    }

    #[test]
    fn test_for_loop_without_override() {
        let config = ContextConfig::new().max_messages(10);

        // Without override, for_loop returns self
        assert_eq!(config.for_loop().max_messages, 10);
    }

    #[test]
    fn test_preset_configs() {
        let minimal = ContextConfig::minimal();
        assert_eq!(minimal.max_messages, 5);

        let standard = ContextConfig::standard();
        assert_eq!(standard.max_messages, 10);
        assert!(standard.loop_override.is_some());

        let large = ContextConfig::large();
        assert_eq!(large.max_messages, 30);
        assert!(large.tools.deduplicate);
    }

    #[test]
    fn test_filter_messages_max() {
        use crate::models::{Message, MessageContent, Role};
        use uuid::Uuid;

        let config = ContextConfig::new().max_messages(2);

        let thread_id = Uuid::new_v4();
        let messages = vec![
            Message::new(thread_id, Role::User, MessageContent::text("First")),
            Message::new(thread_id, Role::Assistant, MessageContent::text("Response 1")),
            Message::new(thread_id, Role::User, MessageContent::text("Second")),
            Message::new(thread_id, Role::Assistant, MessageContent::text("Response 2")),
        ];

        let filtered = config.filter_messages(messages);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].text(), "Second");
        assert_eq!(filtered[1].text(), "Response 2");
    }

    #[test]
    fn test_filter_messages_empty() {
        let config = ContextConfig::new();
        let filtered = config.filter_messages(vec![]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_messages_starts_with_user() {
        use crate::models::{Message, MessageContent, Role};
        use uuid::Uuid;

        let config = ContextConfig::new().max_messages(2);

        let thread_id = Uuid::new_v4();
        let messages = vec![
            Message::new(thread_id, Role::User, MessageContent::text("First")),
            Message::new(thread_id, Role::Assistant, MessageContent::text("Response 1")),
            Message::new(thread_id, Role::Assistant, MessageContent::text("Response 2")),
            Message::new(thread_id, Role::User, MessageContent::text("Second")),
        ];

        let filtered = config.filter_messages(messages);
        // Should skip the assistant message at index 2 and keep user + assistant
        assert!(!filtered.is_empty());
        assert_eq!(filtered[0].role, Role::User);
    }
}
