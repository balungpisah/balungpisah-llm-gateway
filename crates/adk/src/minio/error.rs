//! MinIO/S3 error types.

use thiserror::Error;

/// Errors that can occur during MinIO/S3 operations.
#[derive(Debug, Error)]
pub enum MinioError {
    /// Failed to initialize the S3 client.
    #[error("Failed to initialize S3 client: {message}")]
    Initialization { message: String },

    /// Failed to upload a file.
    #[error("Failed to upload file: {message}")]
    Upload { message: String },

    /// Failed to download a file.
    #[error("Failed to download file: {message}")]
    Download { message: String },

    /// Failed to delete a file.
    #[error("Failed to delete file: {message}")]
    Delete { message: String },

    /// Failed to generate presigned URL.
    #[error("Failed to generate presigned URL: {message}")]
    PresignedUrl { message: String },

    /// Failed to create bucket.
    #[error("Failed to create bucket: {message}")]
    BucketCreation { message: String },

    /// Invalid configuration.
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
}

/// Result type for MinIO operations.
pub type MinioResult<T> = Result<T, MinioError>;
