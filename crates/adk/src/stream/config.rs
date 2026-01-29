//! Stream configuration options.

/// Configuration for streaming behavior.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Whether to enable streaming.
    pub enabled: bool,
    /// Buffer size for stream events.
    pub buffer_size: usize,
    /// Timeout for individual stream events in milliseconds.
    pub event_timeout_ms: Option<u64>,
    /// Whether to flush events immediately.
    pub flush_immediate: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_size: 32,
            event_timeout_ms: Some(30000), // 30 seconds
            flush_immediate: true,
        }
    }
}

impl StreamConfig {
    /// Create a new stream configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable streaming.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the event timeout.
    pub fn event_timeout_ms(mut self, timeout: u64) -> Self {
        self.event_timeout_ms = Some(timeout);
        self
    }

    /// Disable event timeout.
    pub fn no_timeout(mut self) -> Self {
        self.event_timeout_ms = None;
        self
    }

    /// Set whether to flush events immediately.
    pub fn flush_immediate(mut self, flush: bool) -> Self {
        self.flush_immediate = flush;
        self
    }
}
