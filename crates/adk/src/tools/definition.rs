//! Tool definition types and builder.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Definition of a tool that can be used by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name of the tool.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON schema for the tool's parameters.
    pub parameters: Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Create a builder for a tool definition.
    pub fn builder(name: impl Into<String>) -> ToolDefinitionBuilder {
        ToolDefinitionBuilder::new(name)
    }

    /// Convert to TensorZero tool definition format.
    pub fn to_tensorzero(&self) -> balungpisah_tensorzero::ToolDefinition {
        balungpisah_tensorzero::ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

/// Builder for creating tool definitions with a fluent API.
#[derive(Debug)]
pub struct ToolDefinitionBuilder {
    name: String,
    description: Option<String>,
    parameters: ParametersBuilder,
}

impl ToolDefinitionBuilder {
    /// Create a new builder with the given tool name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: ParametersBuilder::new(),
        }
    }

    /// Set the tool description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a required string parameter.
    pub fn string_param(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.parameters = self.parameters.string(name, description, true);
        self
    }

    /// Add an optional string parameter.
    pub fn optional_string_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.string(name, description, false);
        self
    }

    /// Add a required number parameter.
    pub fn number_param(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.parameters = self.parameters.number(name, description, true);
        self
    }

    /// Add an optional number parameter.
    pub fn optional_number_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.number(name, description, false);
        self
    }

    /// Add a required integer parameter.
    pub fn integer_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.integer(name, description, true);
        self
    }

    /// Add an optional integer parameter.
    pub fn optional_integer_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.integer(name, description, false);
        self
    }

    /// Add a required boolean parameter.
    pub fn boolean_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.boolean(name, description, true);
        self
    }

    /// Add an optional boolean parameter.
    pub fn optional_boolean_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters = self.parameters.boolean(name, description, false);
        self
    }

    /// Add a required enum parameter.
    pub fn enum_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: &[&str],
    ) -> Self {
        self.parameters = self.parameters.enum_type(name, description, values, true);
        self
    }

    /// Add an optional enum parameter.
    pub fn optional_enum_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: &[&str],
    ) -> Self {
        self.parameters = self.parameters.enum_type(name, description, values, false);
        self
    }

    /// Add a required array parameter.
    pub fn array_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        item_type: &str,
    ) -> Self {
        self.parameters = self.parameters.array(name, description, item_type, true);
        self
    }

    /// Add an optional array parameter.
    pub fn optional_array_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        item_type: &str,
    ) -> Self {
        self.parameters = self.parameters.array(name, description, item_type, false);
        self
    }

    /// Add a custom parameter with a raw JSON schema.
    pub fn custom_param(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        self.parameters = self.parameters.custom(name, schema, required);
        self
    }

    /// Build the tool definition.
    pub fn build(self) -> ToolDefinition {
        ToolDefinition {
            name: self.name,
            description: self.description.unwrap_or_default(),
            parameters: self.parameters.build(),
        }
    }
}

/// Builder for JSON schema parameters.
#[derive(Debug, Default)]
struct ParametersBuilder {
    properties: Vec<(String, Value)>,
    required: Vec<String>,
}

impl ParametersBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn string(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "string",
                "description": description.into()
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn number(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "number",
                "description": description.into()
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn integer(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "integer",
                "description": description.into()
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn boolean(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "boolean",
                "description": description.into()
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn enum_type(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: &[&str],
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "string",
                "description": description.into(),
                "enum": values
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn array(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        item_type: &str,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.push((
            name.clone(),
            json!({
                "type": "array",
                "description": description.into(),
                "items": { "type": item_type }
            }),
        ));
        if required {
            self.required.push(name);
        }
        self
    }

    fn custom(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        let name = name.into();
        self.properties.push((name.clone(), schema));
        if required {
            self.required.push(name);
        }
        self
    }

    fn build(self) -> Value {
        let properties: serde_json::Map<String, Value> = self.properties.into_iter().collect();

        json!({
            "type": "object",
            "properties": properties,
            "required": self.required
        })
    }
}
