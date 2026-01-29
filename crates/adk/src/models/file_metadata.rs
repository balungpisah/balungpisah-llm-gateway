//! File metadata model for stored files.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Metadata for a file stored in object storage (MinIO/S3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Unique identifier for the file.
    pub id: Uuid,
    /// Thread this file belongs to.
    pub thread_id: Uuid,
    /// Message this file is attached to (if any).
    pub message_id: Option<Uuid>,
    /// Original filename.
    pub filename: String,
    /// MIME type (e.g., "image/jpeg", "application/pdf").
    pub mime_type: String,
    /// File size in bytes.
    pub size_bytes: i64,
    /// SHA256 checksum of the file content.
    pub checksum: String,
    /// Storage bucket name.
    pub storage_bucket: String,
    /// Storage object key/path.
    pub storage_key: String,
    /// When the file was uploaded.
    pub created_at: DateTime<Utc>,
    /// Cached presigned URL for access.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presigned_url: Option<String>,
    /// When the presigned URL expires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_expires_at: Option<DateTime<Utc>>,
}

impl FileMetadata {
    /// Create a new file metadata entry.
    pub fn new(
        thread_id: Uuid,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
        size_bytes: i64,
        checksum: impl Into<String>,
        storage_bucket: impl Into<String>,
        storage_key: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            thread_id,
            message_id: None,
            filename: filename.into(),
            mime_type: mime_type.into(),
            size_bytes,
            checksum: checksum.into(),
            storage_bucket: storage_bucket.into(),
            storage_key: storage_key.into(),
            created_at: Utc::now(),
            presigned_url: None,
            url_expires_at: None,
        }
    }

    /// Associate this file with a message.
    pub fn with_message_id(mut self, message_id: Uuid) -> Self {
        self.message_id = Some(message_id);
        self
    }

    /// Set the presigned URL and expiration.
    pub fn with_presigned_url(mut self, url: String, expires_at: DateTime<Utc>) -> Self {
        self.presigned_url = Some(url);
        self.url_expires_at = Some(expires_at);
        self
    }

    /// Check if the presigned URL is still valid.
    pub fn is_url_valid(&self) -> bool {
        match &self.url_expires_at {
            Some(expires) => Utc::now() < *expires,
            None => false,
        }
    }

    /// Check if the file is an image based on MIME type.
    pub fn is_image(&self) -> bool {
        self.mime_type.starts_with("image/")
    }

    /// Get the full storage path (bucket/key).
    pub fn storage_path(&self) -> String {
        format!("{}/{}", self.storage_bucket, self.storage_key)
    }
}
