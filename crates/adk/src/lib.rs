//! Balungpisah Agent Development Kit (ADK)
//!
//! A Rust SDK for building AI agents with TensorZero inference gateway integration.
//!
//! # Features
//!
//! - **Agent Management**: Create and manage AI agents with conversation history
//! - **Tool System**: Define and execute custom tools with type-safe interfaces
//! - **Storage**: PostgreSQL storage for threads and messages
//! - **Streaming**: Server-Sent Events support for real-time responses
//! - **Context Filtering**: Smart context management for conversation windows
//!
//! # Quick Start
//!
//! ```ignore
//! use balungpisah_adk::prelude::*;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), AgentError> {
//!     // Create TensorZero client
//!     let client = TensorZeroClient::new("http://localhost:3000")?;
//!
//!     // Create storage
//!     let storage = Arc::new(
//!         PostgresStorage::connect_url("postgres://localhost/agents").await?
//!     );
//!
//!     // Build agent
//!     let agent = AgentBuilder::new()
//!         .tensorzero_client(client)
//!         .storage(storage)
//!         .function_name("my_assistant")
//!         .build()?;
//!
//!     // Create thread and chat
//!     let thread = agent.get_or_create_thread("user-123", None).await?;
//!     let response = agent.chat(thread.id, "Hello!").await?;
//!
//!     println!("Agent: {}", response.text());
//!     Ok(())
//! }
//! ```
//!
//! # Crate Features
//!
//! - `postgres` (default): Enable PostgreSQL storage support
//! - `minio`: Enable MinIO/S3 object storage support for multimodal content

pub mod agent;
pub mod context;
pub mod error;
pub mod models;
pub mod storage;
pub mod stream;
pub mod tools;

#[cfg(feature = "minio")]
pub mod minio;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder, ChatRequest, ChatResponse, ChatStreamResponse};
    pub use crate::context::{ContextConfig, ToolsConfig};
    pub use crate::error::{AgentError, Result, StorageError, ToolError};
    pub use crate::models::{
        ContentBlock, FileMetadata, Message, MessageContent, Role, Thread, ThreadOptions,
    };
    pub use crate::storage::FileMetadataStorage;
    pub use crate::tools::{ToolContext, ToolDefinition, ToolExecutor, ToolRegistry, ToolResult};

    #[cfg(feature = "postgres")]
    pub use crate::storage::{PostgresConfig, PostgresStorage, Storage};

    #[cfg(feature = "minio")]
    pub use crate::minio::{MinioClient, MinioConfig, MinioError, MinioResult};

    pub use balungpisah_tensorzero::{TensorZeroClient, ToolChoice};
}

// Re-export main types at crate root
pub use agent::{
    Agent, AgentBuilder, ChatRequest, ChatResponse, ChatStreamResponse, ModelSpec, Usage,
};
pub use context::{ContextConfig, ToolsConfig};
pub use error::{AgentError, Result, StorageError, ToolError};
pub use models::{
    ContentBlock, FileMetadata, Message, MessageContent, Role, Thread, ThreadOptions,
};
pub use storage::{FileMetadataStorage, MessageStorage, Storage, ThreadStorage};
pub use stream::{SseEvent, StreamConfig, StreamExecutor};
pub use tools::{
    FnToolExecutor, ToolContext, ToolDefinition, ToolDefinitionBuilder, ToolExecutor, ToolRegistry,
    ToolResult,
};

#[cfg(feature = "postgres")]
pub use storage::{PostgresConfig, PostgresStorage};

// Re-export tensorzero types that users commonly need
pub use balungpisah_tensorzero::{TensorZeroClient, ToolChoice};
