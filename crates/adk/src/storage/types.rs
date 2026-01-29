//! Storage trait definitions.

use crate::error::StorageResult;
use crate::models::{Message, Thread};
use async_trait::async_trait;
use uuid::Uuid;

/// Storage interface for thread operations.
#[async_trait]
pub trait ThreadStorage: Send + Sync {
    /// Create a new thread.
    async fn create_thread(&self, thread: &Thread) -> StorageResult<()>;

    /// Get a thread by ID.
    async fn get_thread(&self, id: Uuid) -> StorageResult<Option<Thread>>;

    /// Get a thread by external ID.
    async fn get_thread_by_external_id(&self, external_id: &str) -> StorageResult<Option<Thread>>;

    /// Update a thread.
    async fn update_thread(&self, thread: &Thread) -> StorageResult<()>;

    /// Delete a thread and all its messages.
    async fn delete_thread(&self, id: Uuid) -> StorageResult<()>;

    /// List threads, optionally filtered by external ID prefix.
    async fn list_threads(
        &self,
        external_id_prefix: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Thread>>;
}

/// Storage interface for message operations.
#[async_trait]
pub trait MessageStorage: Send + Sync {
    /// Create a new message.
    async fn create_message(&self, message: &Message) -> StorageResult<()>;

    /// Get a message by ID.
    async fn get_message(&self, id: Uuid) -> StorageResult<Option<Message>>;

    /// Get all messages for a thread, ordered by creation time.
    async fn get_thread_messages(&self, thread_id: Uuid) -> StorageResult<Vec<Message>>;

    /// Get the most recent N messages for a thread.
    async fn get_recent_messages(
        &self,
        thread_id: Uuid,
        limit: usize,
    ) -> StorageResult<Vec<Message>>;

    /// Update a message.
    async fn update_message(&self, message: &Message) -> StorageResult<()>;

    /// Delete a message.
    async fn delete_message(&self, id: Uuid) -> StorageResult<()>;

    /// Delete all messages for a thread.
    async fn delete_thread_messages(&self, thread_id: Uuid) -> StorageResult<()>;

    /// Count messages in a thread.
    async fn count_thread_messages(&self, thread_id: Uuid) -> StorageResult<usize>;
}

/// Combined storage interface for convenience.
#[async_trait]
pub trait Storage: ThreadStorage + MessageStorage {
    /// Run database migrations.
    async fn migrate(&self) -> StorageResult<()>;

    /// Check if the storage is healthy.
    async fn health_check(&self) -> StorageResult<bool>;
}
