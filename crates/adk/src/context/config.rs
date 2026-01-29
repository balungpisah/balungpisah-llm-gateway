//! Context configuration for agent conversations.
//!
//! This module provides configuration for how conversation context is managed,
//! including message limits, tool message handling, and loop-specific overrides.

use serde::{Deserialize, Serialize};

/// Configuration for how tool-related messages are handled in context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsContextConfig {
    /// Number of recent tool call/result pairs to retain.
    /// Older tool interactions will be summarized or removed.
    pub retain_last: usize,

    /// Maximum number of tool results to include per message.
    /// If a message has more tool results, older ones are truncated.
    pub limit_per_message: Option<usize>,

    /// Whether to deduplicate repeated tool calls with same arguments.
    pub deduplicate: bool,
}

impl Default for ToolsContextConfig {
    fn default() -> Self {
        Self {
            retain_last: 5,
            limit_per_message: None,
            deduplicate: false,
        }
    }
}

impl ToolsContextConfig {
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
/// use balungpisah_adk::context::{ContextConfig, ToolsContextConfig};
///
/// let config = ContextConfig::new()
///     .max_messages(10)
///     .tools(ToolsContextConfig::new()
///         .retain_last(3)
///         .limit_per_message(3))
///     .loop_override(ContextConfig::new()
///         .max_messages(3)
///         .tools(ToolsContextConfig::new().retain_last(1)));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum number of messages to include in context.
    pub max_messages: usize,

    /// Configuration for tool message handling.
    pub tools: ToolsContextConfig,

    /// Override configuration for tool execution loops.
    /// During multi-turn tool execution, this config is used instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_override: Option<Box<ContextConfig>>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_messages: 20,
            tools: ToolsContextConfig::default(),
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
    pub fn tools(mut self, config: ToolsContextConfig) -> Self {
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

    /// Create a minimal config for tight context (useful for loops).
    pub fn minimal() -> Self {
        Self {
            max_messages: 5,
            tools: ToolsContextConfig {
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
            tools: ToolsContextConfig {
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
            tools: ToolsContextConfig {
                retain_last: 10,
                limit_per_message: None,
                deduplicate: true,
            },
            loop_override: Some(Box::new(Self {
                max_messages: 10,
                tools: ToolsContextConfig {
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
            .tools(ToolsContextConfig::new().retain_last(3).deduplicate(true));

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
}
