//! Simple Agent Example
//!
//! This example demonstrates how to create a basic AI agent with:
//! - TensorZero integration with dynamic API keys
//! - Model or function-based configuration
//! - PostgreSQL storage with message editing
//! - Context configuration for conversation management
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
//! # Option 1: Use a pre-configured function
//! cargo run -p simple_agent
//!
//! # Option 2: Use model directly with dynamic API key
//! export OPENAI_API_KEY=sk-...
//! cargo run -p simple_agent -- --use-model
//! ```

use balungpisah_adk::{
    AgentBuilder, ContextConfig, PostgresConfig, PostgresStorage, Storage, TensorZeroClient,
    ToolChoice, ToolContext, ToolDefinition, ToolRegistry, ToolResult, ToolsConfig,
};
use serde_json::{json, Value};
use std::sync::Arc;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("simple_agent=debug".parse()?)
                .add_directive("balungpisah_adk=debug".parse()?),
        )
        .init();

    tracing::info!("Starting simple agent example");

    // Check for --use-model flag to demonstrate model-based configuration
    let use_model = std::env::args().any(|arg| arg == "--use-model");

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
            &ctx.tool_name,
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
            Ok(value) => {
                ToolResult::success(&ctx.tool_call_id, &ctx.tool_name, format!("{}", value))
            }
            Err(e) => ToolResult::error(&ctx.tool_call_id, &ctx.tool_name, e),
        }
    });

    tracing::info!("Registered {} tools: {:?}", tools.len(), tools.names());

    // Create context configuration for conversation management
    let context_config = ContextConfig::new()
        .max_messages(20)
        .tools(ToolsConfig::new().retain_last(5).limit_per_message(3))
        .loop_override(
            // Use more aggressive context limits during tool loops
            ContextConfig::new()
                .max_messages(5)
                .tools(ToolsConfig::new().retain_last(2)),
        );

    // Build the agent with either function_name or model_name
    let agent = if use_model {
        // Option 2: Use model directly with dynamic API key
        // This is useful for multi-tenant applications where each user has their own API key
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY must be set when using --use-model flag");

        tracing::info!("Building agent with model-based configuration (gpt-5-nano)");

        AgentBuilder::new()
            .tensorzero_client(client)
            .storage(storage.clone())
            .model_name("gpt-5-nano") // Use model directly
            .credentials(json!({
                "system_api_key": api_key
            }))
            .system_prompt(
                "You are a helpful assistant with access to weather and calculator tools. \
                 Be concise but friendly in your responses.",
            )
            .tools(tools)
            .tool_choice(ToolChoice::Auto)
            .parallel_tool_calls(true)
            .context_config(context_config)
            .max_iterations(5)
            .tags(json!({
                "example": "simple_agent",
                "mode": "model"
            }))
            .build()?
    } else {
        // Option 1: Use a pre-configured TensorZero function
        tracing::info!("Building agent with function-based configuration (simple_assistant)");

        AgentBuilder::new()
            .tensorzero_client(client)
            .storage(storage.clone())
            .function_name("simple_assistant") // Must match TensorZero config
            .system_prompt(
                "You are a helpful assistant with access to weather and calculator tools. \
                 Be concise but friendly in your responses.",
            )
            .tools(tools)
            .context_config(context_config)
            .max_iterations(5)
            .tags(json!({
                "example": "simple_agent",
                "mode": "function"
            }))
            .build()?
    };

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

    // Demonstrate message management
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Message Management Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let thread_messages = agent.get_thread_messages(thread.id).await?;
    println!(
        "\nThread has {} messages. First few message roles:",
        thread_messages.len()
    );

    for (i, msg) in thread_messages.iter().take(5).enumerate() {
        println!("  {}. {:?} - {}", i + 1, msg.role, msg.id);
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
