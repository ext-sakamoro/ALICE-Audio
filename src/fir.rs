//! `FirFilter` — Finite Impulse Response filter.

use core::f64::consts::PI;

use crate::ring_buffer::RingBuffer;

/// Finite Impulse Response filter.
pub struct FirFilter {
    pub(crate) coeffs: Vec<f64>,
    buffer: RingBuffer,
}

impl FirFilter {
    /// Create a new FIR filter with the given coefficients.
    pub fn new(coeffs: Vec<f64>) -> Self {
        let len = coeffs.len();
        Self {
            coeffs,
            buffer: RingBuffer::new(len),
        }
    }

    /// Design a low-pass FIR filter using windowed sinc (Hamming window).
    pub fn low_pass(cutoff: f64, sample_rate: f64, order: usize) -> Self {
        let fc = cutoff / sample_rate;
        let mid = order / 2;
        let coeffs: Vec<f64> = (0..=order)
            .map(|i| {
                let n = i as f64 - mid as f64;
                let sinc = if n.abs() < 1e-12 {
                    2.0 * PI * fc
                } else {
                    (2.0 * PI * fc * n).sin() / n
                };
                let window = 0.46f64.mul_add(-(2.0 * PI * i as f64 / order as f64).cos(), 0.54);
                sinc * window
            })
            .collect();
        Self::new(coeffs)
    }

    /// Process a single sample.
    pub fn process(&mut self, sample: f64) -> f64 {
        self.buffer.push(sample);
        let mut out = 0.0;
        for (i, &c) in self.coeffs.iter().enumerate() {
            out += c * self.buffer.read(i);
        }
        out
    }

    /// Process a buffer of samples.
    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}
