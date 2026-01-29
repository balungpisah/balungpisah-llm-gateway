//! Message model for conversation history.

use super::content::{ContentBlock, MessageContent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "postgres", derive(sqlx::Type))]
#[cfg_attr(
    feature = "postgres",
    sqlx(type_name = "message_role", rename_all = "snake_case")
)]
pub enum Role {
    /// Message from the user.
    User,
    /// Message from the assistant.
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// A message in a conversation thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message identifier.
    pub id: Uuid,
    /// Thread this message belongs to.
    pub thread_id: Uuid,
    /// Role of the sender.
    pub role: Role,
    /// Message content.
    pub content: MessageContent,
    /// TensorZero episode ID.
    pub episode_id: Option<Uuid>,
    /// When the message was created.
    pub created_at: DateTime<Utc>,
    /// When the message was last updated.
    pub updated_at: DateTime<Utc>,
}

impl Message {
    /// Create a new message.
    pub fn new(thread_id: Uuid, role: Role, content: impl Into<MessageContent>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            thread_id,
            role,
            content: content.into(),
            episode_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new user message.
    pub fn user(thread_id: Uuid, content: impl Into<MessageContent>) -> Self {
        Self::new(thread_id, Role::User, content)
    }

    /// Create a new assistant message.
    pub fn assistant(thread_id: Uuid, content: impl Into<MessageContent>) -> Self {
        Self::new(thread_id, Role::Assistant, content)
    }

    /// Set the episode ID.
    pub fn with_episode_id(mut self, episode_id: Uuid) -> Self {
        self.episode_id = Some(episode_id);
        self
    }

    /// Update the `updated_at` timestamp.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Get the text content of this message.
    pub fn text(&self) -> String {
        self.content.as_string()
    }

    /// Check if this message contains tool uses.
    pub fn has_tool_uses(&self) -> bool {
        self.content.has_tool_uses()
    }

    /// Get all tool use blocks from this message.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content.tool_uses()
    }
}

/// Builder for creating messages with tool results.
#[derive(Debug, Default)]
pub struct ToolResultMessageBuilder {
    thread_id: Option<Uuid>,
    results: Vec<ContentBlock>,
}

impl ToolResultMessageBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thread ID.
    pub fn thread_id(mut self, thread_id: Uuid) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    /// Add a tool result.
    pub fn result(
        mut self,
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        self.results
            .push(ContentBlock::tool_result(tool_use_id, tool_name, content));
        self
    }

    /// Add an error result.
    pub fn error(
        mut self,
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        self.results
            .push(ContentBlock::tool_error(tool_use_id, tool_name, error));
        self
    }

    /// Build the message.
    pub fn build(self) -> Option<Message> {
        let thread_id = self.thread_id?;
        if self.results.is_empty() {
            return None;
        }

        let now = Utc::now();
        Some(Message {
            id: Uuid::now_v7(),
            thread_id,
            role: Role::User,
            content: MessageContent::Blocks(self.results),
            episode_id: None,
            created_at: now,
            updated_at: now,
        })
    }
}
