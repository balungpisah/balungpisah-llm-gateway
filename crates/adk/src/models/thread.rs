//! Thread model for conversation management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// A conversation thread that groups related messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    /// Unique thread identifier.
    pub id: Uuid,
    /// External identifier (e.g., user ID, session ID).
    pub external_id: String,
    /// TensorZero episode ID for this thread.
    pub episode_id: Option<Uuid>,
    /// Optional metadata for the thread.
    pub metadata: Option<Value>,
    /// When the thread was created.
    pub created_at: DateTime<Utc>,
    /// When the thread was last updated.
    pub updated_at: DateTime<Utc>,
}

impl Thread {
    /// Create a new thread.
    pub fn new(external_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            external_id: external_id.into(),
            episode_id: None,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new thread with a specific ID.
    pub fn with_id(id: Uuid, external_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id,
            external_id: external_id.into(),
            episode_id: None,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the episode ID.
    pub fn with_episode_id(mut self, episode_id: Uuid) -> Self {
        self.episode_id = Some(episode_id);
        self
    }

    /// Set the metadata.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Update the `updated_at` timestamp.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// Options for creating or retrieving a thread.
#[derive(Debug, Clone, Default)]
pub struct ThreadOptions {
    /// Metadata to set on the thread.
    pub metadata: Option<Value>,
    /// Force creation of a new thread even if one exists.
    pub force_new: bool,
}

impl ThreadOptions {
    /// Create new options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Force creation of a new thread.
    pub fn force_new(mut self) -> Self {
        self.force_new = true;
        self
    }
}
