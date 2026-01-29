//! Storage trait definitions.

use crate::error::StorageResult;
use crate::models::{FileMetadata, Message, Thread};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
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

    /// Delete all messages in a thread created after the specified message.
    /// This is used for the "edit and resubmit" workflow.
    /// Returns the number of messages deleted.
    async fn delete_messages_after(&self, thread_id: Uuid, after_id: Uuid) -> StorageResult<u64>;
}

/// Storage interface for file metadata operations.
#[async_trait]
pub trait FileMetadataStorage: Send + Sync {
    /// Create a new file metadata entry.
    async fn create_file(&self, file: &FileMetadata) -> StorageResult<()>;

    /// Get file metadata by ID.
    async fn get_file(&self, id: Uuid) -> StorageResult<Option<FileMetadata>>;

    /// Update the presigned URL for a file.
    async fn update_file_url(
        &self,
        id: Uuid,
        url: String,
        expires_at: DateTime<Utc>,
    ) -> StorageResult<()>;

    /// Delete file metadata.
    async fn delete_file(&self, id: Uuid) -> StorageResult<()>;

    /// List all files for a thread.
    async fn list_thread_files(&self, thread_id: Uuid) -> StorageResult<Vec<FileMetadata>>;

    /// List files for a specific message.
    async fn list_message_files(&self, message_id: Uuid) -> StorageResult<Vec<FileMetadata>>;

    /// Find a file by checksum within a thread (for deduplication).
    async fn find_by_checksum(
        &self,
        thread_id: Uuid,
        checksum: &str,
    ) -> StorageResult<Option<FileMetadata>>;
}

/// Combined storage interface for convenience.
#[async_trait]
pub trait Storage: ThreadStorage + MessageStorage + FileMetadataStorage {
    /// Run database migrations.
    async fn migrate(&self) -> StorageResult<()>;

    /// Check if the storage is healthy.
    async fn health_check(&self) -> StorageResult<bool>;
}
