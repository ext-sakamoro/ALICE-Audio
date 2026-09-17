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
    ///
    /// `h[n] = 2fc·sinc(2fc·n)·w[n] = sin(2πfc·n)/(πn)·w[n]`, so the DC gain
    /// is ≈ 1 (exactly 1 after the normalisation below).  Until 2026-09-17
    /// the `1/π` was missing and every design had a DC gain of π (oracle
    /// `tests/analytic_oracle.rs`).
    pub fn low_pass(cutoff: f64, sample_rate: f64, order: usize) -> Self {
        let fc = cutoff / sample_rate;
        let mid = order as f64 / 2.0;
        let mut coeffs: Vec<f64> = (0..=order)
            .map(|i| {
                let n = i as f64 - mid;
                let sinc = if n.abs() < 1e-12 {
                    2.0 * fc
                } else {
                    (2.0 * PI * fc * n).sin() / (PI * n)
                };
                let window = 0.46f64.mul_add(-(2.0 * PI * i as f64 / order as f64).cos(), 0.54);
                sinc * window
            })
            .collect();
        // normalise the DC gain to exactly 1
        let sum: f64 = coeffs.iter().sum();
        if sum.abs() > 1e-15 {
            for c in &mut coeffs {
                *c /= sum;
            }
        }
        Self::new(coeffs)
    }

    /// Process a single sample: `y[n] = Σ c[k]·x[n−k]`.
    ///
    /// `RingBuffer::read(k)` is `k` steps behind the write head, so after
    /// the push the current sample is `read(1)`.  Until 2026-09-17 this used
    /// `read(k)`, which applied `c[0]` to the oldest sample and every other
    /// tap one sample late — a circular rotation of the impulse response
    /// (oracle `tests/analytic_oracle.rs`).
    pub fn process(&mut self, sample: f64) -> f64 {
        self.buffer.push(sample);
        let mut out = 0.0;
        for (i, &c) in self.coeffs.iter().enumerate() {
            out = c.mul_add(self.buffer.read(i + 1), out);
        }
        out
    }

    /// Process a buffer of samples.
    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}
