-- Add files table for storing file metadata
-- This migration adds support for multimodal content (images, documents, etc.)

CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    filename VARCHAR(255) NOT NULL,
    mime_type VARCHAR(127) NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    storage_bucket VARCHAR(63) NOT NULL,
    storage_key VARCHAR(1024) NOT NULL,
    presigned_url TEXT,
    url_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fetching files by thread
CREATE INDEX IF NOT EXISTS idx_files_thread_id ON files(thread_id);

-- Index for fetching files by message
CREATE INDEX IF NOT EXISTS idx_files_message_id ON files(message_id);

-- Index for finding files by checksum (deduplication)
CREATE INDEX IF NOT EXISTS idx_files_checksum ON files(checksum);
