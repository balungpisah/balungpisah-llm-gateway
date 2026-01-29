//! MinIO/S3 configuration.

use std::time::Duration;

/// Configuration for MinIO/S3 client.
#[derive(Debug, Clone)]
pub struct MinioConfig {
    /// Endpoint URL (e.g., "http://localhost:9000" or "https://s3.amazonaws.com").
    pub endpoint: String,
    /// Access key ID.
    pub access_key: String,
    /// Secret access key.
    pub secret_key: String,
    /// Default bucket name.
    pub bucket: String,
    /// AWS region (default: "us-east-1").
    pub region: String,
    /// Whether to use path-style addressing (required for MinIO).
    pub path_style: bool,
    /// Presigned URL expiration duration.
    pub presigned_url_expiry: Duration,
}

impl MinioConfig {
    /// Create a new MinIO configuration.
    pub fn new(
        endpoint: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            bucket: bucket.into(),
            region: "us-east-1".to_string(),
            path_style: true, // Default to path-style for MinIO
            presigned_url_expiry: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Set the AWS region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Set path-style addressing.
    pub fn path_style(mut self, path_style: bool) -> Self {
        self.path_style = path_style;
        self
    }

    /// Set presigned URL expiration duration.
    pub fn presigned_url_expiry(mut self, duration: Duration) -> Self {
        self.presigned_url_expiry = duration;
        self
    }

    /// Create a configuration from environment variables.
    ///
    /// Reads from:
    /// - `MINIO_ENDPOINT` or `S3_ENDPOINT`
    /// - `MINIO_ACCESS_KEY` or `AWS_ACCESS_KEY_ID`
    /// - `MINIO_SECRET_KEY` or `AWS_SECRET_ACCESS_KEY`
    /// - `MINIO_BUCKET` or `S3_BUCKET`
    /// - `MINIO_REGION` or `AWS_REGION` (optional, defaults to "us-east-1")
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("MINIO_ENDPOINT")
            .or_else(|_| std::env::var("S3_ENDPOINT"))
            .ok()?;
        let access_key = std::env::var("MINIO_ACCESS_KEY")
            .or_else(|_| std::env::var("AWS_ACCESS_KEY_ID"))
            .ok()?;
        let secret_key = std::env::var("MINIO_SECRET_KEY")
            .or_else(|_| std::env::var("AWS_SECRET_ACCESS_KEY"))
            .ok()?;
        let bucket = std::env::var("MINIO_BUCKET")
            .or_else(|_| std::env::var("S3_BUCKET"))
            .ok()?;
        let region = std::env::var("MINIO_REGION")
            .or_else(|_| std::env::var("AWS_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        Some(Self::new(endpoint, access_key, secret_key, bucket).region(region))
    }
}
