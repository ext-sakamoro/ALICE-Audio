//! Fixed-size ring buffer for audio sample storage.

/// Fixed-size ring buffer for audio sample storage.
pub struct RingBuffer {
    buf: Vec<f64>,
    write_pos: usize,
}

impl RingBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            buf: vec![0.0; size],
            write_pos: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.buf.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Push a sample, overwriting the oldest.
    pub fn push(&mut self, sample: f64) {
        if self.buf.is_empty() {
            return;
        }
        self.buf[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buf.len();
    }

    /// Read a sample `delay` steps behind the write head.
    pub fn read(&self, delay: usize) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let len = self.buf.len();
        let idx = (self.write_pos + len - delay % len) % len;
        self.buf[idx]
    }

    /// Reset all samples to zero.
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
    }
}
