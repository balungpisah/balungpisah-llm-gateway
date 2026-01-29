-- Initial schema for balungpisah-adk
-- This migration creates the threads and messages tables

-- Threads table
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY,
    external_id TEXT NOT NULL,
    agent_slug TEXT,
    title TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for external_id lookups
CREATE INDEX IF NOT EXISTS idx_threads_external_id ON threads(external_id);

-- Index for listing threads by creation time
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at DESC);

-- Index for agent_slug lookups
CREATE INDEX IF NOT EXISTS idx_threads_agent_slug ON threads(agent_slug) WHERE agent_slug IS NOT NULL;

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content JSONB NOT NULL,
    episode_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fetching messages by thread
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);

-- Index for ordering messages within a thread
CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON messages(thread_id, created_at);

-- Index for updated_at (edit tracking)
CREATE INDEX IF NOT EXISTS idx_messages_updated_at ON messages(updated_at);
