-- Initial schema for balungpisah-adk
-- This migration creates the threads and messages tables

-- Threads table
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY,
    external_id TEXT NOT NULL,
    episode_id UUID,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for external_id lookups
CREATE INDEX IF NOT EXISTS idx_threads_external_id ON threads(external_id);

-- Index for listing threads by creation time
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at DESC);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    inference_id UUID,
    episode_id UUID
);

-- Index for fetching messages by thread
CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);

-- Index for ordering messages within a thread
CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON messages(thread_id, created_at);

-- Index for inference lookups
CREATE INDEX IF NOT EXISTS idx_messages_inference_id ON messages(inference_id) WHERE inference_id IS NOT NULL;
