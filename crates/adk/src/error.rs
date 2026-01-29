//! Error types for the Agent Development Kit.

use thiserror::Error;

/// Primary error type for agent operations.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Error from storage operations.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// Error from tool execution.
    #[error("tool error: {0}")]
    Tool(#[from] ToolError),

    /// Tool not found by name.
    #[error("tool '{tool_name}' not found")]
    ToolNotFound { tool_name: String },

    /// Maximum iterations exceeded during agent loop.
    #[error("max iterations ({max}) exceeded")]
    MaxIterationsExceeded { max: usize },

    /// Error from TensorZero inference.
    #[error("inference failed: {message}")]
    Inference {
        message: String,
        status_code: Option<u16>,
    },

    /// Invalid configuration.
    #[error("invalid configuration: {message}")]
    Configuration { message: String },

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Thread not found.
    #[error("thread not found: {thread_id}")]
    ThreadNotFound { thread_id: uuid::Uuid },

    /// Stream error.
    #[error("stream error: {message}")]
    Stream { message: String },
}

impl From<balungpisah_tensorzero::TensorZeroError> for AgentError {
    fn from(err: balungpisah_tensorzero::TensorZeroError) -> Self {
        match &err {
            balungpisah_tensorzero::TensorZeroError::ServerError {
                status_code,
                message,
            } => AgentError::Inference {
                message: message.clone(),
                status_code: Some(*status_code),
            },
            _ => AgentError::Inference {
                message: err.to_string(),
                status_code: None,
            },
        }
    }
}

/// Error type for storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database connection error.
    #[error("connection error: {message}")]
    Connection { message: String },

    /// Query execution error.
    #[error("query error: {message}")]
    Query { message: String },

    /// Record not found.
    #[error("not found: {entity} with id {id}")]
    NotFound { entity: String, id: String },

    /// Serialization error for JSON fields.
    #[error("serialization error: {message}")]
    Serialization { message: String },

    /// Migration error.
    #[error("migration error: {message}")]
    Migration { message: String },

    /// Transaction error.
    #[error("transaction error: {message}")]
    Transaction { message: String },
}

#[cfg(feature = "postgres")]
impl From<sqlx::Error> for StorageError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => StorageError::NotFound {
                entity: "record".to_string(),
                id: "unknown".to_string(),
            },
            sqlx::Error::Configuration(msg) => StorageError::Connection {
                message: msg.to_string(),
            },
            _ => StorageError::Query {
                message: err.to_string(),
            },
        }
    }
}

#[cfg(feature = "postgres")]
impl From<sqlx::migrate::MigrateError> for StorageError {
    fn from(err: sqlx::migrate::MigrateError) -> Self {
        StorageError::Migration {
            message: err.to_string(),
        }
    }
}

/// Error type for tool operations.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool execution failed.
    #[error("execution failed: {message}")]
    ExecutionFailed { message: String },

    /// Invalid arguments provided to tool.
    #[error("invalid arguments: {message}")]
    InvalidArguments { message: String },

    /// Tool timeout.
    #[error("tool execution timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },

    /// Permission denied.
    #[error("permission denied: {message}")]
    PermissionDenied { message: String },

    /// Internal tool error.
    #[error("internal error: {message}")]
    Internal { message: String },
}

/// Result type alias for agent operations.
pub type Result<T> = std::result::Result<T, AgentError>;

/// Result type alias for storage operations.
pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Result type alias for tool operations.
pub type ToolResult<T> = std::result::Result<T, ToolError>;
