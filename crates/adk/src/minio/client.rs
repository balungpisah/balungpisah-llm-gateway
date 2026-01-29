//! MinIO/S3 client implementation.

use super::config::MinioConfig;
use super::error::{MinioError, MinioResult};
use s3::bucket::Bucket;
use s3::creds::Credentials;
use s3::Region;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tracing::{debug, instrument};

/// Client for MinIO/S3 object storage operations.
#[derive(Clone)]
pub struct MinioClient {
    bucket: Arc<Bucket>,
    config: MinioConfig,
}

impl std::fmt::Debug for MinioClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinioClient")
            .field("bucket", &self.config.bucket)
            .field("endpoint", &self.config.endpoint)
            .finish()
    }
}

/// Result of a file upload operation.
#[derive(Debug, Clone)]
pub struct UploadResult {
    /// The storage key where the file was uploaded.
    pub key: String,
    /// SHA256 checksum of the uploaded data.
    pub checksum: String,
    /// Size of the uploaded data in bytes.
    pub size_bytes: i64,
}

impl MinioClient {
    /// Create a new MinIO client.
    pub fn new(config: MinioConfig) -> MinioResult<Self> {
        let region = Region::Custom {
            region: config.region.clone(),
            endpoint: config.endpoint.clone(),
        };

        let credentials = Credentials::new(
            Some(&config.access_key),
            Some(&config.secret_key),
            None,
            None,
            None,
        )
        .map_err(|e| MinioError::Initialization {
            message: e.to_string(),
        })?;

        let mut bucket = Bucket::new(&config.bucket, region, credentials).map_err(|e| {
            MinioError::Initialization {
                message: e.to_string(),
            }
        })?;

        if config.path_style {
            bucket = bucket.with_path_style();
        }

        Ok(Self {
            bucket: Arc::new(bucket),
            config,
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> MinioResult<Option<Self>> {
        match MinioConfig::from_env() {
            Some(config) => Ok(Some(Self::new(config)?)),
            None => Ok(None),
        }
    }

    /// Upload a file to storage.
    ///
    /// Returns the storage key, checksum, and size.
    #[instrument(skip(self, data))]
    pub async fn upload_file(
        &self,
        key: &str,
        data: &[u8],
        mime_type: &str,
    ) -> MinioResult<UploadResult> {
        debug!("Uploading file to {}", key);

        // Calculate checksum
        let checksum = Self::calculate_checksum(data);
        let size_bytes = data.len() as i64;

        // Upload to S3
        self.bucket
            .put_object_with_content_type(key, data, mime_type)
            .await
            .map_err(|e| MinioError::Upload {
                message: e.to_string(),
            })?;

        debug!("Successfully uploaded {} bytes to {}", size_bytes, key);

        Ok(UploadResult {
            key: key.to_string(),
            checksum,
            size_bytes,
        })
    }

    /// Upload base64-encoded data.
    #[instrument(skip(self, base64_data))]
    pub async fn upload_base64(
        &self,
        key: &str,
        base64_data: &str,
        mime_type: &str,
    ) -> MinioResult<UploadResult> {
        use base64::{engine::general_purpose::STANDARD, Engine};

        let data = STANDARD
            .decode(base64_data)
            .map_err(|e| MinioError::Upload {
                message: format!("Invalid base64 data: {}", e),
            })?;

        self.upload_file(key, &data, mime_type).await
    }

    /// Generate a presigned URL for downloading a file.
    #[instrument(skip(self))]
    pub async fn generate_presigned_url(&self, key: &str) -> MinioResult<String> {
        debug!("Generating presigned URL for {}", key);

        let expiry_secs = self.config.presigned_url_expiry.as_secs() as u32;

        let url = self
            .bucket
            .presign_get(key, expiry_secs, None)
            .await
            .map_err(|e| MinioError::PresignedUrl {
                message: e.to_string(),
            })?;

        Ok(url)
    }

    /// Delete a file from storage.
    #[instrument(skip(self))]
    pub async fn delete_file(&self, key: &str) -> MinioResult<()> {
        debug!("Deleting file {}", key);

        self.bucket
            .delete_object(key)
            .await
            .map_err(|e| MinioError::Delete {
                message: e.to_string(),
            })?;

        debug!("Successfully deleted {}", key);
        Ok(())
    }

    /// Download a file from storage.
    #[instrument(skip(self))]
    pub async fn download_file(&self, key: &str) -> MinioResult<Vec<u8>> {
        debug!("Downloading file {}", key);

        let response = self
            .bucket
            .get_object(key)
            .await
            .map_err(|e| MinioError::Download {
                message: e.to_string(),
            })?;

        Ok(response.to_vec())
    }

    /// Check if a file exists.
    #[instrument(skip(self))]
    pub async fn file_exists(&self, key: &str) -> MinioResult<bool> {
        match self.bucket.head_object(key).await {
            Ok(_) => Ok(true),
            Err(s3::error::S3Error::HttpFailWithBody(404, _)) => Ok(false),
            Err(e) => Err(MinioError::Download {
                message: e.to_string(),
            }),
        }
    }

    /// Get the bucket name.
    pub fn bucket_name(&self) -> &str {
        &self.config.bucket
    }

    /// Get the presigned URL expiry duration in seconds.
    pub fn presigned_url_expiry_secs(&self) -> u64 {
        self.config.presigned_url_expiry.as_secs()
    }

    /// Calculate SHA256 checksum of data.
    fn calculate_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Generate a storage key for a file.
    ///
    /// Format: `{thread_id}/{file_id}/{filename}`
    pub fn generate_key(thread_id: &uuid::Uuid, file_id: &uuid::Uuid, filename: &str) -> String {
        format!("{}/{}/{}", thread_id, file_id, filename)
    }
}
