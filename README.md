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
- Docker and Docker Compose (for development environment)

## Development Setup

### 1. Start the development services

```bash
# Copy environment variables
cp .env.example .env

# Start all services
docker compose up -d
```

This starts:
- **PostgreSQL** (TimescaleDB) on port 5432
- **TensorZero Gateway** on port 3000
- **TensorZero UI** on port 4000
- **ClickHouse** on port 8123 (for TensorZero observability)

### 2. Verify services are running

```bash
# Check service health
docker compose ps

# View logs
docker compose logs -f
```

### 3. Access the services

| Service | URL |
|---------|-----|
| PostgreSQL | `postgres://postgres:postgres@localhost:5432/agents` |
| TensorZero Gateway | http://localhost:3000 |
| TensorZero UI | http://localhost:4000 |

### 4. Build and run

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

### Stop services

```bash
docker compose down

# To also remove volumes (reset data)
docker compose down -v
```

## Configuration

The TensorZero configuration is in `config/tensorzero/tensorzero.toml`.

**Dynamic API Keys**: All models use `api_key_location = "dynamic::system_api_key"`, meaning API keys are provided by ADK consumers at request time rather than configured in the gateway. This allows each consumer to use their own API keys.

Customize the config to:
- Add more functions/variants
- Configure different models
- Add metrics for observability

## License

MIT License - see [LICENSE](LICENSE) for details.
