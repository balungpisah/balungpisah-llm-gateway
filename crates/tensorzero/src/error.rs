//! Error types for TensorZero client operations.

use thiserror::Error;

/// Errors that can occur when using the TensorZero client.
#[derive(Debug, Error)]
pub enum TensorZeroError {
    /// Failed to build the HTTP client.
    #[error("failed to build HTTP client: {message}")]
    ClientBuild { message: String },

    /// Invalid URL provided.
    #[error("invalid URL: {url}")]
    InvalidUrl { url: String },

    /// HTTP request failed.
    #[error("HTTP request failed: {message}")]
    Request {
        message: String,
        #[source]
        source: Option<reqwest::Error>,
    },

    /// Server returned an error response.
    #[error("server error (status {status_code}): {message}")]
    ServerError { status_code: u16, message: String },

    /// Failed to parse response.
    #[error("failed to parse response: {message}")]
    ParseError {
        message: String,
        #[source]
        source: Option<serde_json::Error>,
    },

    /// Streaming error.
    #[error("streaming error: {message}")]
    StreamError { message: String },

    /// Invalid request configuration.
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
}

impl TensorZeroError {
    /// Create a request error from a reqwest error.
    pub fn from_reqwest(err: reqwest::Error) -> Self {
        Self::Request {
            message: err.to_string(),
            source: Some(err),
        }
    }

    /// Create a parse error from a serde_json error.
    pub fn from_json(err: serde_json::Error) -> Self {
        Self::ParseError {
            message: err.to_string(),
            source: Some(err),
        }
    }
}

/// Result type alias for TensorZero operations.
pub type Result<T> = std::result::Result<T, TensorZeroError>;
