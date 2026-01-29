//! Tool executor traits and implementations.

use super::context::ToolContext;
use super::definition::ToolDefinition;
use super::result::ToolResult;
use async_trait::async_trait;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Trait for types that can execute tools.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Get the tool definition.
    fn definition(&self) -> &ToolDefinition;

    /// Execute the tool with the given arguments.
    async fn execute(&self, args: Value, context: ToolContext) -> ToolResult;
}

/// Type alias for async tool execution functions.
pub type ToolFn = Box<
    dyn Fn(Value, ToolContext) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> + Send + Sync,
>;

/// A tool executor backed by a function.
pub struct FnToolExecutor {
    definition: ToolDefinition,
    executor: ToolFn,
}

impl std::fmt::Debug for FnToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnToolExecutor")
            .field("definition", &self.definition)
            .finish()
    }
}

impl FnToolExecutor {
    /// Create a new function-based tool executor.
    pub fn new<F, Fut>(definition: ToolDefinition, f: F) -> Self
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        Self {
            definition,
            executor: Box::new(move |args, ctx| Box::pin(f(args, ctx))),
        }
    }

    /// Create a new function-based tool executor with a simple function.
    ///
    /// This variant takes a function that doesn't need the context.
    pub fn simple<F, Fut>(definition: ToolDefinition, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        Self {
            definition,
            executor: Box::new(move |args, ctx| {
                let result = f(args);
                Box::pin(async move {
                    let mut result = result.await;
                    if result.tool_call_id.is_empty() {
                        result.tool_call_id = ctx.tool_call_id;
                    }
                    if result.tool_name.is_empty() {
                        result.tool_name = ctx.tool_name;
                    }
                    result
                })
            }),
        }
    }
}

#[async_trait]
impl ToolExecutor for FnToolExecutor {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, args: Value, context: ToolContext) -> ToolResult {
        (self.executor)(args, context).await
    }
}

/// Registry of tool executors.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn ToolExecutor>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool executor.
    pub fn register<T: ToolExecutor + 'static>(&mut self, executor: T) {
        self.tools.push(Arc::new(executor));
    }

    /// Register a function as a tool.
    pub fn register_fn<F, Fut>(&mut self, definition: ToolDefinition, f: F)
    where
        F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        self.register(FnToolExecutor::new(definition, f));
    }

    /// Register a simple function as a tool (no context needed).
    pub fn register_simple_fn<F, Fut>(&mut self, definition: ToolDefinition, f: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        self.register(FnToolExecutor::simple(definition, f));
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn ToolExecutor>> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .cloned()
    }

    /// Get all tool definitions.
    pub fn definitions(&self) -> Vec<&ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    /// Get all tool definitions in TensorZero format.
    pub fn tensorzero_definitions(&self) -> Vec<balungpisah_tensorzero::ToolDefinition> {
        self.definitions()
            .into_iter()
            .map(|d| d.to_tensorzero())
            .collect()
    }

    /// Check if a tool exists.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.definition().name == name)
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get all tool names.
    pub fn names(&self) -> Vec<&str> {
        self.tools
            .iter()
            .map(|t| t.definition().name.as_str())
            .collect()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}
