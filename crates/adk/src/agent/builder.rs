//! Agent builder with typestate pattern for compile-time validation.

use crate::context::ContextConfig;
use crate::error::{AgentError, Result};
use crate::storage::Storage;
use crate::stream::StreamConfig;
use crate::tools::ToolRegistry;
use balungpisah_tensorzero::{TensorZeroClient, ToolChoice};
use serde_json::Value;
use std::marker::PhantomData;
use std::sync::Arc;

use super::core::Agent;

/// Marker type for unset builder state.
pub struct Unset;

/// Marker type for set builder state.
pub struct Set<T>(pub T);

/// Model specification - either a function name or model name.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// Use a pre-configured TensorZero function.
    Function(String),
    /// Use a model directly (ad-hoc function).
    Model(String),
}

impl ModelSpec {
    /// Check if this is a function spec.
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_))
    }

    /// Check if this is a model spec.
    pub fn is_model(&self) -> bool {
        matches!(self, Self::Model(_))
    }

    /// Get the function name if this is a function spec.
    pub fn as_function(&self) -> Option<&str> {
        match self {
            Self::Function(name) => Some(name),
            _ => None,
        }
    }

    /// Get the model name if this is a model spec.
    pub fn as_model(&self) -> Option<&str> {
        match self {
            Self::Model(name) => Some(name),
            _ => None,
        }
    }
}

/// Builder for creating Agent instances with compile-time validation.
///
/// Uses the typestate pattern to ensure required fields (tensorzero_client, storage,
/// and model/function) are provided before building.
///
/// # Example
///
/// ```ignore
/// use balungpisah_adk::prelude::*;
/// use serde_json::json;
///
/// // Using a pre-configured function
/// let agent = AgentBuilder::new()
///     .tensorzero_client(client)
///     .storage(storage)
///     .function_name("my_assistant")
///     .system_prompt("You are a helpful assistant.")
///     .max_iterations(10)
///     .build()?;
///
/// // Using a model directly with dynamic API key
/// let agent = AgentBuilder::new()
///     .tensorzero_client(client)
///     .storage(storage)
///     .model_name("gpt-4o")
///     .credentials(json!({ "system_api_key": "sk-..." }))
///     .build()?;
/// ```
pub struct AgentBuilder<C = Unset, S = Unset, M = Unset> {
    client: C,
    storage: S,
    model_spec: M,
    tools: ToolRegistry,
    max_iterations: usize,
    stream_config: StreamConfig,
    context_config: Option<ContextConfig>,
    system_prompt: Option<String>,
    credentials: Option<Value>,
    tags: Option<Value>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    _phantom: PhantomData<(C, S, M)>,
}

impl Default for AgentBuilder<Unset, Unset, Unset> {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder<Unset, Unset, Unset> {
    /// Create a new agent builder.
    pub fn new() -> Self {
        Self {
            client: Unset,
            storage: Unset,
            model_spec: Unset,
            tools: ToolRegistry::new(),
            max_iterations: 10,
            stream_config: StreamConfig::default(),
            context_config: None,
            system_prompt: None,
            credentials: None,
            tags: None,
            tool_choice: None,
            parallel_tool_calls: None,
            _phantom: PhantomData,
        }
    }
}

impl<S, M> AgentBuilder<Unset, S, M> {
    /// Set the TensorZero client.
    pub fn tensorzero_client(
        self,
        client: TensorZeroClient,
    ) -> AgentBuilder<Set<TensorZeroClient>, S, M> {
        AgentBuilder {
            client: Set(client),
            storage: self.storage,
            model_spec: self.model_spec,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_config: self.context_config,
            system_prompt: self.system_prompt,
            credentials: self.credentials,
            tags: self.tags,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            _phantom: PhantomData,
        }
    }
}

impl<C, M> AgentBuilder<C, Unset, M> {
    /// Set the storage backend.
    pub fn storage<T>(self, storage: Arc<T>) -> AgentBuilder<C, Set<Arc<T>>, M>
    where
        T: Storage + 'static,
    {
        AgentBuilder {
            client: self.client,
            storage: Set(storage),
            model_spec: self.model_spec,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_config: self.context_config,
            system_prompt: self.system_prompt,
            credentials: self.credentials,
            tags: self.tags,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            _phantom: PhantomData,
        }
    }
}

impl<C, S> AgentBuilder<C, S, Unset> {
    /// Set the TensorZero function name.
    ///
    /// Use this when you have a pre-configured function in TensorZero.
    pub fn function_name(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<ModelSpec>> {
        AgentBuilder {
            client: self.client,
            storage: self.storage,
            model_spec: Set(ModelSpec::Function(name.into())),
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_config: self.context_config,
            system_prompt: self.system_prompt,
            credentials: self.credentials,
            tags: self.tags,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            _phantom: PhantomData,
        }
    }

    /// Alias for function_name for convenience.
    pub fn function(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<ModelSpec>> {
        self.function_name(name)
    }

    /// Set the model name for ad-hoc inference.
    ///
    /// Use this when you want to use a model directly without a pre-configured function.
    /// When using this, you typically also need to provide credentials with `system_api_key`.
    pub fn model_name(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<ModelSpec>> {
        AgentBuilder {
            client: self.client,
            storage: self.storage,
            model_spec: Set(ModelSpec::Model(name.into())),
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_config: self.context_config,
            system_prompt: self.system_prompt,
            credentials: self.credentials,
            tags: self.tags,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            _phantom: PhantomData,
        }
    }

    /// Alias for model_name for convenience.
    pub fn model(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<ModelSpec>> {
        self.model_name(name)
    }
}

impl<C, S, M> AgentBuilder<C, S, M> {
    /// Set the tool registry.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Set the maximum number of tool execution iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the stream configuration.
    pub fn stream_config(mut self, config: StreamConfig) -> Self {
        self.stream_config = config;
        self
    }

    /// Set the context configuration.
    ///
    /// This controls how conversation history is filtered and managed.
    pub fn context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Set the system prompt.
    ///
    /// This overrides any system prompt configured in the TensorZero function.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set dynamic credentials (e.g., API keys).
    ///
    /// Use this to pass `system_api_key` and other dynamic credentials to TensorZero.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use serde_json::json;
    ///
    /// builder.credentials(json!({
    ///     "system_api_key": "sk-your-api-key"
    /// }))
    /// ```
    pub fn credentials(mut self, credentials: Value) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set tags for tracking and observability.
    ///
    /// These are passed to TensorZero for logging and analytics.
    pub fn tags(mut self, tags: Value) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set the tool choice strategy.
    ///
    /// Controls how the model decides to use tools.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set whether to allow parallel tool calls.
    ///
    /// When enabled, the model may request multiple tool calls in a single response.
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }
}

impl<T> AgentBuilder<Set<TensorZeroClient>, Set<Arc<T>>, Set<ModelSpec>>
where
    T: Storage + 'static,
{
    /// Build the agent.
    ///
    /// This method is only available when all required fields have been set.
    pub fn build(self) -> Result<Agent<T>> {
        let client = self.client.0;
        let storage = self.storage.0;
        let model_spec = self.model_spec.0;

        // Validate model spec
        match &model_spec {
            ModelSpec::Function(name) if name.is_empty() => {
                return Err(AgentError::Configuration {
                    message: "function_name cannot be empty".to_string(),
                });
            }
            ModelSpec::Model(name) if name.is_empty() => {
                return Err(AgentError::Configuration {
                    message: "model_name cannot be empty".to_string(),
                });
            }
            _ => {}
        }

        Ok(Agent {
            client,
            storage,
            model_spec,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_config: self.context_config,
            system_prompt: self.system_prompt,
            credentials: self.credentials,
            tags: self.tags,
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let _builder = AgentBuilder::new();
    }

    #[test]
    fn test_model_spec() {
        let func = ModelSpec::Function("chat".to_string());
        assert!(func.is_function());
        assert!(!func.is_model());
        assert_eq!(func.as_function(), Some("chat"));
        assert_eq!(func.as_model(), None);

        let model = ModelSpec::Model("gpt-4o".to_string());
        assert!(!model.is_function());
        assert!(model.is_model());
        assert_eq!(model.as_function(), None);
        assert_eq!(model.as_model(), Some("gpt-4o"));
    }
}
