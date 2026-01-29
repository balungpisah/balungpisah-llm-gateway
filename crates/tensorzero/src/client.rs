//! TensorZero HTTP client for inference.

use crate::error::{Result, TensorZeroError};
use crate::request::{InferenceRequest, JsonInferenceRequest};
use crate::response::{InferenceResponse, JsonInferenceResponse, StreamEvent};
use eventsource_stream::Eventsource;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use std::pin::Pin;
use tracing::{debug, error, instrument};

/// Client for interacting with TensorZero inference gateway.
#[derive(Clone)]
pub struct TensorZeroClient {
    client: Client,
    base_url: String,
}

impl std::fmt::Debug for TensorZeroClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorZeroClient")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl TensorZeroClient {
    /// Create a new TensorZero client.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the TensorZero gateway (e.g., "http://localhost:3000")
    ///
    /// # Example
    ///
    /// ```no_run
    /// use balungpisah_tensorzero::TensorZeroClient;
    ///
    /// let client = TensorZeroClient::new("http://localhost:3000").unwrap();
    /// ```
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();

        // Validate URL format
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(TensorZeroError::InvalidUrl { url: base_url });
        }

        let client = Client::builder()
            .build()
            .map_err(|e| TensorZeroError::ClientBuild {
                message: e.to_string(),
            })?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Create a new client with a custom reqwest client.
    pub fn with_client(base_url: impl Into<String>, client: Client) -> Result<Self> {
        let base_url = base_url.into();

        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(TensorZeroError::InvalidUrl { url: base_url });
        }

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Get the base URL of this client.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Perform a chat inference request.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use balungpisah_tensorzero::{TensorZeroClient, InferenceRequestBuilder, InputMessage};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = TensorZeroClient::new("http://localhost:3000")?;
    ///
    /// let request = InferenceRequestBuilder::new()
    ///     .function_name("my_function")
    ///     .message(InputMessage::user("Hello!"))
    ///     .build()?;
    ///
    /// let response = client.inference(request).await?;
    /// println!("Response: {}", response.text());
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, request), fields(function = %request.function_name))]
    pub async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let url = format!("{}/inference", self.base_url);

        debug!("Sending inference request to {}", url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            error!("Inference failed with status {}: {}", status, message);
            return Err(TensorZeroError::ServerError {
                status_code: status.as_u16(),
                message,
            });
        }

        let body = response
            .text()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        serde_json::from_str(&body).map_err(TensorZeroError::from_json)
    }

    /// Perform a streaming chat inference request.
    ///
    /// Returns a stream of `StreamEvent` that can be processed as they arrive.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use balungpisah_tensorzero::{TensorZeroClient, InferenceRequestBuilder, InputMessage};
    /// use futures::StreamExt;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = TensorZeroClient::new("http://localhost:3000")?;
    ///
    /// let request = InferenceRequestBuilder::new()
    ///     .function_name("my_function")
    ///     .message(InputMessage::user("Hello!"))
    ///     .stream()
    ///     .build()?;
    ///
    /// let mut stream = client.inference_stream(request).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         Ok(event) => println!("Event: {:?}", event),
    ///         Err(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, request), fields(function = %request.function_name))]
    pub async fn inference_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let url = format!("{}/inference", self.base_url);

        debug!("Sending streaming inference request to {}", url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            error!(
                "Streaming inference failed with status {}: {}",
                status, message
            );
            return Err(TensorZeroError::ServerError {
                status_code: status.as_u16(),
                message,
            });
        }

        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|result| async move {
                match result {
                    Ok(event) => {
                        // Skip empty data or comments
                        if event.data.is_empty() || event.data == "[DONE]" {
                            return None;
                        }

                        match serde_json::from_str::<StreamEvent>(&event.data) {
                            Ok(stream_event) => Some(Ok(stream_event)),
                            Err(e) => {
                                // Log parse errors but continue processing
                                debug!(
                                    "Failed to parse stream event: {} - data: {}",
                                    e, event.data
                                );
                                None
                            }
                        }
                    }
                    Err(e) => Some(Err(TensorZeroError::StreamError {
                        message: e.to_string(),
                    })),
                }
            });

        Ok(Box::pin(stream))
    }

    /// Perform a JSON inference request.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use balungpisah_tensorzero::{TensorZeroClient, JsonInferenceRequestBuilder, InputMessage};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = TensorZeroClient::new("http://localhost:3000")?;
    ///
    /// let request = JsonInferenceRequestBuilder::new()
    ///     .function_name("extract_data")
    ///     .message(InputMessage::user("Extract the name and age from: John is 30."))
    ///     .output_schema(json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "name": { "type": "string" },
    ///             "age": { "type": "integer" }
    ///         }
    ///     }))
    ///     .build()?;
    ///
    /// let response = client.json_inference(request).await?;
    /// println!("Output: {:?}", response.parsed());
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, request), fields(function = %request.function_name))]
    pub async fn json_inference(
        &self,
        request: JsonInferenceRequest,
    ) -> Result<JsonInferenceResponse> {
        let url = format!("{}/inference", self.base_url);

        debug!("Sending JSON inference request to {}", url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            error!("JSON inference failed with status {}: {}", status, message);
            return Err(TensorZeroError::ServerError {
                status_code: status.as_u16(),
                message,
            });
        }

        let body = response
            .text()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        serde_json::from_str(&body).map_err(TensorZeroError::from_json)
    }

    /// Check if the TensorZero gateway is healthy.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(TensorZeroError::from_reqwest)?;

        Ok(response.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = TensorZeroClient::new("http://localhost:3000");
        assert!(client.is_ok());
    }

    #[test]
    fn test_client_invalid_url() {
        let client = TensorZeroClient::new("invalid-url");
        assert!(client.is_err());
    }

    #[test]
    fn test_client_base_url_normalization() {
        let client = TensorZeroClient::new("http://localhost:3000/").unwrap();
        assert_eq!(client.base_url(), "http://localhost:3000");
    }
}
