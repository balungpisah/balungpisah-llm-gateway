//! Simple Agent Example
//!
//! This example demonstrates how to create a basic AI agent with:
//! - TensorZero integration
//! - PostgreSQL storage
//! - Custom tools
//!
//! # Prerequisites
//!
//! 1. TensorZero gateway running at http://localhost:3000
//! 2. PostgreSQL database with the required schema
//!
//! # Running
//!
//! ```bash
//! # Set up the database (optional - agent will create tables)
//! export DATABASE_URL=postgres://localhost/agents
//!
//! # Run the example
//! cargo run -p simple_agent
//! ```

use balungpisah_adk::{
    AgentBuilder, PostgresConfig, PostgresStorage, Storage, TensorZeroClient, ToolContext,
    ToolDefinition, ToolRegistry, ToolResult,
};
use serde_json::{json, Value};
use std::sync::Arc;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("simple_agent=debug".parse()?)
                .add_directive("balungpisah_adk=debug".parse()?),
        )
        .init();

    tracing::info!("Starting simple agent example");

    // Get configuration from environment
    let tensorzero_url =
        std::env::var("TENSORZERO_URL").unwrap_or_else(|_| "http://localhost:3000".to_string());
    let database_url =
        std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://localhost/agents".to_string());

    // Create TensorZero client
    let client = TensorZeroClient::new(&tensorzero_url)?;
    tracing::info!("Connected to TensorZero at {}", tensorzero_url);

    // Create storage
    let storage = Arc::new(PostgresStorage::connect(PostgresConfig::new(&database_url)).await?);
    tracing::info!("Connected to PostgreSQL");

    // Run migrations
    storage.migrate().await?;
    tracing::info!("Database migrations complete");

    // Create tool registry
    let mut tools = ToolRegistry::new();

    // Register a simple weather tool
    let weather_tool = ToolDefinition::builder("get_weather")
        .description("Get the current weather for a location")
        .string_param("location", "The city and state, e.g. San Francisco, CA")
        .optional_enum_param("unit", "Temperature unit", &["celsius", "fahrenheit"])
        .build();

    tools.register_fn(weather_tool, |args: Value, ctx: ToolContext| async move {
        let location = args
            .get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let unit = args
            .get("unit")
            .and_then(|v| v.as_str())
            .unwrap_or("fahrenheit");

        tracing::info!(
            "Getting weather for {} (tool_call_id: {})",
            location,
            ctx.tool_call_id
        );

        // Simulate weather API call
        let (temp, symbol) = if unit == "celsius" {
            (22, "°C")
        } else {
            (72, "°F")
        };

        ToolResult::success_json(
            &ctx.tool_call_id,
            json!({
                "location": location,
                "temperature": temp,
                "unit": symbol,
                "conditions": "Sunny",
                "humidity": "45%"
            }),
        )
    });

    // Register a calculator tool
    let calc_tool = ToolDefinition::builder("calculate")
        .description("Perform basic arithmetic calculations")
        .string_param(
            "expression",
            "The mathematical expression to evaluate (e.g., '2 + 2')",
        )
        .build();

    tools.register_fn(calc_tool, |args: Value, ctx: ToolContext| async move {
        let expression = args
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("0");

        tracing::info!("Calculating: {}", expression);

        // Simple expression evaluation (in production, use a proper parser)
        match evaluate_simple_expression(expression) {
            Ok(value) => ToolResult::success(&ctx.tool_call_id, format!("{}", value)),
            Err(e) => ToolResult::error(&ctx.tool_call_id, e),
        }
    });

    tracing::info!("Registered {} tools: {:?}", tools.len(), tools.names());

    // Build the agent
    let agent = AgentBuilder::new()
        .tensorzero_client(client)
        .storage(storage)
        .function_name("simple_assistant") // Must match TensorZero config
        .tools(tools)
        .max_iterations(5)
        .build()?;

    tracing::info!("Agent built successfully");

    // Get or create a thread
    let thread = agent.get_or_create_thread("example-user-001", None).await?;
    tracing::info!(
        "Using thread: {} (external_id: {})",
        thread.id,
        thread.external_id
    );

    // Example conversation
    let messages = vec![
        "Hello! What can you help me with?",
        "What's the weather like in San Francisco?",
        "Can you calculate 15 * 7 + 3 for me?",
    ];

    for user_message in messages {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("User: {}", user_message);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let response = agent.chat(thread.id, user_message).await?;

        println!("\nAssistant: {}", response.text());
        println!(
            "\n[iterations: {}, tokens: {} in / {} out]",
            response.iterations, response.usage.input_tokens, response.usage.output_tokens
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example complete!");

    Ok(())
}

/// Simple expression evaluator (for demonstration purposes).
fn evaluate_simple_expression(expr: &str) -> std::result::Result<f64, String> {
    // Very basic expression parsing - in production use a proper parser
    let expr = expr.trim();

    // Try to parse as a simple number first
    if let Ok(num) = expr.parse::<f64>() {
        return Ok(num);
    }

    // Try basic operations
    for (op, op_char) in [("*", '*'), ("+", '+'), ("-", '-'), ("/", '/')] {
        if let Some(pos) = expr.rfind(op_char) {
            if pos > 0 && pos < expr.len() - 1 {
                let left = evaluate_simple_expression(&expr[..pos])?;
                let right = evaluate_simple_expression(&expr[pos + 1..])?;

                return match op {
                    "*" => Ok(left * right),
                    "+" => Ok(left + right),
                    "-" => Ok(left - right),
                    "/" => {
                        if right == 0.0 {
                            Err("Division by zero".to_string())
                        } else {
                            Ok(left / right)
                        }
                    }
                    _ => Err(format!("Unknown operator: {}", op)),
                };
            }
        }
    }

    Err(format!("Could not parse expression: {}", expr))
}
