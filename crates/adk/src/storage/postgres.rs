//! PostgreSQL storage implementation.

use crate::error::{StorageError, StorageResult};
use crate::models::{Message, Role, Thread};
use crate::storage::types::{MessageStorage, Storage, ThreadStorage};
use async_trait::async_trait;
use serde_json::Value;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use tracing::{debug, instrument};
use uuid::Uuid;

/// PostgreSQL storage configuration.
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    /// Database connection URL.
    pub url: String,
    /// Maximum number of connections in the pool.
    pub max_connections: u32,
    /// Minimum number of connections to maintain.
    pub min_connections: u32,
}

impl PostgresConfig {
    /// Create a new configuration with the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_connections: 10,
            min_connections: 1,
        }
    }

    /// Set the maximum number of connections.
    pub fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Set the minimum number of connections.
    pub fn min_connections(mut self, min: u32) -> Self {
        self.min_connections = min;
        self
    }
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            url: "postgres://localhost/agents".to_string(),
            max_connections: 10,
            min_connections: 1,
        }
    }
}

/// PostgreSQL storage implementation.
#[derive(Clone)]
pub struct PostgresStorage {
    pool: PgPool,
}

impl std::fmt::Debug for PostgresStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStorage")
            .field("pool", &"PgPool { ... }")
            .finish()
    }
}

impl PostgresStorage {
    /// Connect to PostgreSQL using the provided configuration.
    pub async fn connect(config: PostgresConfig) -> StorageResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .connect(&config.url)
            .await
            .map_err(|e| StorageError::Connection {
                message: e.to_string(),
            })?;

        Ok(Self { pool })
    }

    /// Connect using a connection string.
    pub async fn connect_url(url: &str) -> StorageResult<Self> {
        Self::connect(PostgresConfig::new(url)).await
    }

    /// Create from an existing pool.
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get a reference to the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[async_trait]
impl ThreadStorage for PostgresStorage {
    #[instrument(skip(self, thread), fields(thread_id = %thread.id))]
    async fn create_thread(&self, thread: &Thread) -> StorageResult<()> {
        debug!("Creating thread");

        sqlx::query(
            r#"
            INSERT INTO threads (id, external_id, agent_slug, title, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(thread.id)
        .bind(&thread.external_id)
        .bind(&thread.agent_slug)
        .bind(&thread.title)
        .bind(&thread.metadata)
        .bind(thread.created_at)
        .bind(thread.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_thread(&self, id: Uuid) -> StorageResult<Option<Thread>> {
        debug!("Getting thread");

        let row = sqlx::query(
            r#"
            SELECT id, external_id, agent_slug, title, metadata, created_at, updated_at
            FROM threads
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Thread {
            id: r.get("id"),
            external_id: r.get("external_id"),
            agent_slug: r.get("agent_slug"),
            title: r.get("title"),
            metadata: r.get("metadata"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    #[instrument(skip(self))]
    async fn get_thread_by_external_id(&self, external_id: &str) -> StorageResult<Option<Thread>> {
        debug!("Getting thread by external ID");

        let row = sqlx::query(
            r#"
            SELECT id, external_id, agent_slug, title, metadata, created_at, updated_at
            FROM threads
            WHERE external_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(external_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Thread {
            id: r.get("id"),
            external_id: r.get("external_id"),
            agent_slug: r.get("agent_slug"),
            title: r.get("title"),
            metadata: r.get("metadata"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    #[instrument(skip(self, thread), fields(thread_id = %thread.id))]
    async fn update_thread(&self, thread: &Thread) -> StorageResult<()> {
        debug!("Updating thread");

        sqlx::query(
            r#"
            UPDATE threads
            SET external_id = $2, agent_slug = $3, title = $4, metadata = $5, updated_at = $6
            WHERE id = $1
            "#,
        )
        .bind(thread.id)
        .bind(&thread.external_id)
        .bind(&thread.agent_slug)
        .bind(&thread.title)
        .bind(&thread.metadata)
        .bind(thread.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_thread(&self, id: Uuid) -> StorageResult<()> {
        debug!("Deleting thread");

        // Messages are deleted via CASCADE
        sqlx::query("DELETE FROM threads WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn list_threads(
        &self,
        external_id_prefix: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Thread>> {
        debug!("Listing threads");

        let rows = match external_id_prefix {
            Some(prefix) => {
                let pattern = format!("{}%", prefix);
                sqlx::query(
                    r#"
                    SELECT id, external_id, agent_slug, title, metadata, created_at, updated_at
                    FROM threads
                    WHERE external_id LIKE $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(pattern)
                .bind(limit as i64)
                .bind(offset as i64)
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query(
                    r#"
                    SELECT id, external_id, agent_slug, title, metadata, created_at, updated_at
                    FROM threads
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    "#,
                )
                .bind(limit as i64)
                .bind(offset as i64)
                .fetch_all(&self.pool)
                .await?
            }
        };

        Ok(rows
            .into_iter()
            .map(|r| Thread {
                id: r.get("id"),
                external_id: r.get("external_id"),
                agent_slug: r.get("agent_slug"),
                title: r.get("title"),
                metadata: r.get("metadata"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }
}

#[async_trait]
impl MessageStorage for PostgresStorage {
    #[instrument(skip(self, message), fields(message_id = %message.id, thread_id = %message.thread_id))]
    async fn create_message(&self, message: &Message) -> StorageResult<()> {
        debug!("Creating message");

        let content_json =
            serde_json::to_value(&message.content).map_err(|e| StorageError::Serialization {
                message: e.to_string(),
            })?;

        sqlx::query(
            r#"
            INSERT INTO messages (id, thread_id, role, content, episode_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(message.id)
        .bind(message.thread_id)
        .bind(message.role.to_string())
        .bind(content_json)
        .bind(message.episode_id)
        .bind(message.created_at)
        .bind(message.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_message(&self, id: Uuid) -> StorageResult<Option<Message>> {
        debug!("Getting message");

        let row = sqlx::query(
            r#"
            SELECT id, thread_id, role, content, episode_id, created_at, updated_at
            FROM messages
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|r| {
            let role_str: String = r.get("role");
            let content_json: Value = r.get("content");

            Ok(Message {
                id: r.get("id"),
                thread_id: r.get("thread_id"),
                role: parse_role(&role_str),
                content: serde_json::from_value(content_json).map_err(|e| {
                    StorageError::Serialization {
                        message: e.to_string(),
                    }
                })?,
                episode_id: r.get("episode_id"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
        })
        .transpose()
    }

    #[instrument(skip(self))]
    async fn get_thread_messages(&self, thread_id: Uuid) -> StorageResult<Vec<Message>> {
        debug!("Getting thread messages");

        let rows = sqlx::query(
            r#"
            SELECT id, thread_id, role, content, episode_id, created_at, updated_at
            FROM messages
            WHERE thread_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|r| {
                let role_str: String = r.get("role");
                let content_json: Value = r.get("content");

                Ok(Message {
                    id: r.get("id"),
                    thread_id: r.get("thread_id"),
                    role: parse_role(&role_str),
                    content: serde_json::from_value(content_json).map_err(|e| {
                        StorageError::Serialization {
                            message: e.to_string(),
                        }
                    })?,
                    episode_id: r.get("episode_id"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                })
            })
            .collect()
    }

    #[instrument(skip(self))]
    async fn get_recent_messages(
        &self,
        thread_id: Uuid,
        limit: usize,
    ) -> StorageResult<Vec<Message>> {
        debug!("Getting recent messages");

        let rows = sqlx::query(
            r#"
            SELECT id, thread_id, role, content, episode_id, created_at, updated_at
            FROM messages
            WHERE thread_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(thread_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        // Reverse to get chronological order
        let mut messages: Vec<Message> = rows
            .into_iter()
            .map(|r| {
                let role_str: String = r.get("role");
                let content_json: Value = r.get("content");

                Ok(Message {
                    id: r.get("id"),
                    thread_id: r.get("thread_id"),
                    role: parse_role(&role_str),
                    content: serde_json::from_value(content_json).map_err(|e| {
                        StorageError::Serialization {
                            message: e.to_string(),
                        }
                    })?,
                    episode_id: r.get("episode_id"),
                    created_at: r.get("created_at"),
                    updated_at: r.get("updated_at"),
                })
            })
            .collect::<StorageResult<Vec<_>>>()?;

        messages.reverse();
        Ok(messages)
    }

    #[instrument(skip(self, message), fields(message_id = %message.id))]
    async fn update_message(&self, message: &Message) -> StorageResult<()> {
        debug!("Updating message");

        let content_json =
            serde_json::to_value(&message.content).map_err(|e| StorageError::Serialization {
                message: e.to_string(),
            })?;

        sqlx::query(
            r#"
            UPDATE messages
            SET role = $2, content = $3, episode_id = $4, updated_at = $5
            WHERE id = $1
            "#,
        )
        .bind(message.id)
        .bind(message.role.to_string())
        .bind(content_json)
        .bind(message.episode_id)
        .bind(message.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_message(&self, id: Uuid) -> StorageResult<()> {
        debug!("Deleting message");

        sqlx::query("DELETE FROM messages WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_thread_messages(&self, thread_id: Uuid) -> StorageResult<()> {
        debug!("Deleting thread messages");

        sqlx::query("DELETE FROM messages WHERE thread_id = $1")
            .bind(thread_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn count_thread_messages(&self, thread_id: Uuid) -> StorageResult<usize> {
        debug!("Counting thread messages");

        let row = sqlx::query("SELECT COUNT(*) as count FROM messages WHERE thread_id = $1")
            .bind(thread_id)
            .fetch_one(&self.pool)
            .await?;

        let count: i64 = row.get("count");
        Ok(count as usize)
    }

    #[instrument(skip(self))]
    async fn delete_messages_after(&self, thread_id: Uuid, after_id: Uuid) -> StorageResult<u64> {
        debug!("Deleting messages after message");

        let result = sqlx::query(
            r#"
            DELETE FROM messages
            WHERE thread_id = $1
              AND created_at > (SELECT created_at FROM messages WHERE id = $2)
            "#,
        )
        .bind(thread_id)
        .bind(after_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }
}

#[async_trait]
impl Storage for PostgresStorage {
    async fn migrate(&self) -> StorageResult<()> {
        // Run migrations using sqlx's built-in migration system
        // This will:
        // - Create _sqlx_migrations table if it doesn't exist
        // - Track which migrations have been applied
        // - Only run new migrations
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::Migration {
                message: e.to_string(),
            })?;

        debug!("Database migrations completed successfully");
        Ok(())
    }

    async fn health_check(&self) -> StorageResult<bool> {
        let result = sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map(|_| true)
            .unwrap_or(false);

        Ok(result)
    }
}

fn parse_role(role_str: &str) -> Role {
    match role_str {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => Role::User, // Default to user for unknown roles
    }
}
