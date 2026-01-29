# Balungpisah LLM Gateway

An open-source Rust SDK for building AI agents with TensorZero inference gateway integration.

## Crates

| Crate | Description |
|-------|-------------|
| [`balungpisah-tensorzero`](crates/tensorzero) | TensorZero client for inference |
| [`balungpisah-adk`](crates/adk) | Agent Development Kit |

## Quick Start

Add the ADK to your `Cargo.toml`:

```toml
[dependencies]
balungpisah-adk = "0.1"
```

### Basic Agent Example

```rust
use balungpisah_adk::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    // Create TensorZero client
    let tz_client = TensorZeroClient::new("http://localhost:3000")?;

    // Create storage
    let storage = Arc::new(
        PostgresStorage::connect("postgres://localhost/agents").await?
    );

    // Build agent
    let agent = AgentBuilder::new()
        .tensorzero_client(tz_client)
        .storage(storage)
        .function("my_assistant")
        .max_iterations(10)
        .build()?;

    // Create or load thread
    let thread = agent.get_or_create_thread("user-123", None).await?;

    // Chat with the agent
    let request = ChatRequest::new("Hello, how can you help me?");
    let response = agent.chat(&thread.id, request).await?;

    println!("Agent: {}", response.text());

    Ok(())
}
```

## Features

- **TensorZero Integration**: First-class support for TensorZero inference gateway
- **Tool System**: Define and execute tools with type-safe interfaces
- **Persistent Storage**: PostgreSQL storage for threads and messages
- **Streaming**: Server-Sent Events (SSE) support for real-time responses
- **Context Management**: Smart context filtering and message conversion
- **Builder Pattern**: Compile-time validated agent configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                        │
├─────────────────────────────────────────────────────────────┤
│                    balungpisah-adk                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  Agent  │  │  Tools  │  │ Stream  │  │ Context Filter  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       │            │            │                 │          │
│  ┌────┴────────────┴────────────┴─────────────────┴────┐    │
│  │                     Storage                          │    │
│  │               (PostgreSQL)                           │    │
│  └──────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                 balungpisah-tensorzero                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              TensorZeroClient                         │   │
│  │         (HTTP + SSE Streaming)                        │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    TensorZero Gateway                        │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Rust 1.75+
- PostgreSQL 14+ (for storage)
- TensorZero gateway instance

## Development

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --workspace
```

## License

MIT License - see [LICENSE](LICENSE) for details.
