//! Agent builder with typestate pattern for compile-time validation.

use crate::context::ContextFilter;
use crate::error::{AgentError, Result};
use crate::storage::Storage;
use crate::stream::StreamConfig;
use crate::tools::ToolRegistry;
use balungpisah_tensorzero::TensorZeroClient;
use std::marker::PhantomData;
use std::sync::Arc;

use super::core::Agent;

/// Marker type for unset builder state.
pub struct Unset;

/// Marker type for set builder state.
pub struct Set<T>(T);

/// Builder for creating Agent instances with compile-time validation.
///
/// Uses the typestate pattern to ensure required fields (tensorzero_client, storage,
/// and function_name) are provided before building.
///
/// # Example
///
/// ```ignore
/// let agent = AgentBuilder::new()
///     .tensorzero_client(client)
///     .storage(storage)
///     .function_name("my_assistant")
///     .max_iterations(10)
///     .build()?;
/// ```
pub struct AgentBuilder<C = Unset, S = Unset, F = Unset> {
    client: C,
    storage: S,
    function_name: F,
    tools: ToolRegistry,
    max_iterations: usize,
    stream_config: StreamConfig,
    context_filter: Option<ContextFilter>,
    _phantom: PhantomData<(C, S, F)>,
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
            function_name: Unset,
            tools: ToolRegistry::new(),
            max_iterations: 10,
            stream_config: StreamConfig::default(),
            context_filter: None,
            _phantom: PhantomData,
        }
    }
}

impl<S, F> AgentBuilder<Unset, S, F> {
    /// Set the TensorZero client.
    pub fn tensorzero_client(
        self,
        client: TensorZeroClient,
    ) -> AgentBuilder<Set<TensorZeroClient>, S, F> {
        AgentBuilder {
            client: Set(client),
            storage: self.storage,
            function_name: self.function_name,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_filter: self.context_filter,
            _phantom: PhantomData,
        }
    }
}

impl<C, F> AgentBuilder<C, Unset, F> {
    /// Set the storage backend.
    pub fn storage<T>(self, storage: Arc<T>) -> AgentBuilder<C, Set<Arc<T>>, F>
    where
        T: Storage + 'static,
    {
        AgentBuilder {
            client: self.client,
            storage: Set(storage),
            function_name: self.function_name,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_filter: self.context_filter,
            _phantom: PhantomData,
        }
    }
}

impl<C, S> AgentBuilder<C, S, Unset> {
    /// Set the TensorZero function name.
    pub fn function_name(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<String>> {
        AgentBuilder {
            client: self.client,
            storage: self.storage,
            function_name: Set(name.into()),
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_filter: self.context_filter,
            _phantom: PhantomData,
        }
    }

    /// Alias for function_name for convenience.
    pub fn function(self, name: impl Into<String>) -> AgentBuilder<C, S, Set<String>> {
        self.function_name(name)
    }
}

impl<C, S, F> AgentBuilder<C, S, F> {
    /// Set the tool registry.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Add tools from another registry.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        for def in tools.definitions() {
            if tools.get(&def.name).is_some() {
                // Tools are Arc-wrapped, so they're already shareable
            }
        }
        // For now, just replace
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

    /// Set the context filter.
    pub fn context_filter(mut self, filter: ContextFilter) -> Self {
        self.context_filter = Some(filter);
        self
    }
}

impl<T> AgentBuilder<Set<TensorZeroClient>, Set<Arc<T>>, Set<String>>
where
    T: Storage + 'static,
{
    /// Build the agent.
    ///
    /// This method is only available when all required fields have been set.
    pub fn build(self) -> Result<Agent<T>> {
        let client = self.client.0;
        let storage = self.storage.0;
        let function_name = self.function_name.0;

        if function_name.is_empty() {
            return Err(AgentError::Configuration {
                message: "function_name cannot be empty".to_string(),
            });
        }

        Ok(Agent {
            client,
            storage,
            function_name,
            tools: self.tools,
            max_iterations: self.max_iterations,
            stream_config: self.stream_config,
            context_filter: self.context_filter,
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

    // Note: Full integration tests require a mock storage implementation
}
