//! MinIO/S3 object storage integration.
//!
//! This module provides a client for storing and retrieving files from MinIO or S3-compatible
//! object storage. It is used for multimodal content (images, documents) in messages.
//!
//! # Features
//!
//! This module requires the `minio` feature flag:
//!
//! ```toml
//! [dependencies]
//! balungpisah-adk = { version = "...", features = ["minio"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use balungpisah_adk::minio::{MinioClient, MinioConfig};
//!
//! // Create a client
//! let config = MinioConfig::new(
//!     "http://localhost:9000",
//!     "minioadmin",
//!     "minioadmin",
//!     "my-bucket",
//! );
//! let client = MinioClient::new(config)?;
//!
//! // Upload a file
//! let result = client.upload_file("path/to/file.jpg", &data, "image/jpeg").await?;
//!
//! // Generate a presigned URL for access
//! let url = client.generate_presigned_url(&result.key).await?;
//! ```

mod client;
mod config;
mod error;

pub use client::{MinioClient, UploadResult};
pub use config::MinioConfig;
pub use error::{MinioError, MinioResult};
