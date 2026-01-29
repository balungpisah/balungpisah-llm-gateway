//! TensorZero client library for Rust.
//!
//! This crate provides a client for interacting with the TensorZero inference gateway.
//!
//! # Example
//!
//! ```no_run
//! use balungpisah_tensorzero::{TensorZeroClient, InferenceRequestBuilder, InputMessage};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a client
//! let client = TensorZeroClient::new("http://localhost:3000")?;
//!
//! // Build an inference request
//! let request = InferenceRequestBuilder::new()
//!     .function_name("my_assistant")
//!     .message(InputMessage::user("Hello, how are you?"))
//!     .build()?;
//!
//! // Send the request
//! let response = client.inference(request).await?;
//!
//! println!("Response: {}", response.text());
//! # Ok(())
//! # }
//! ```

mod client;
pub mod error;
pub mod request;
pub mod response;

// Re-export main types at crate root
pub use client::TensorZeroClient;
pub use error::{Result, TensorZeroError};
pub use request::{
    Credentials, InferenceInput, InferenceRequest, InferenceRequestBuilder, InputContentBlock,
    InputMessage, JsonInferenceInput, JsonInferenceRequest, JsonInferenceRequestBuilder,
    MessageRole, ToolChoice, ToolDefinition,
};
pub use response::{
    ChunkEvent, ContentBlock, ContentBlockDeltaEvent, ContentBlockStartEvent,
    ContentBlockStopEvent, ContentBlockType, Delta, DoneEvent, InferenceResponse,
    JsonInferenceResponse, Output, StreamContentBlock, StreamEvent, ToolCallBlock, Usage,
};
