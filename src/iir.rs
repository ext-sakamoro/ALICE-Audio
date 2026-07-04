//! `IirFilter` — second-order IIR (biquad) filter.

use core::f64::consts::PI;

/// Second-order IIR (biquad) filter.
///
/// Transfer function: `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
#[derive(Debug, Clone)]
pub struct IirFilter {
    pub(crate) b0: f64,
    pub(crate) b1: f64,
    pub(crate) b2: f64,
    pub(crate) a1: f64,
    pub(crate) a2: f64,
    pub(crate) x1: f64,
    pub(crate) x2: f64,
    pub(crate) y1: f64,
    pub(crate) y2: f64,
}

impl IirFilter {
    pub const fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Design a low-pass biquad filter.
    pub fn low_pass(cutoff: f64, sample_rate: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Design a high-pass biquad filter.
    pub fn high_pass(cutoff: f64, sample_rate: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = f64::midpoint(1.0, cos_w0);
        let b1 = -(1.0 + cos_w0);
        let b2 = f64::midpoint(1.0, cos_w0);
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Design a band-pass biquad filter.
    pub fn band_pass(center: f64, sample_rate: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * center / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Process a single sample.
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.a2.mul_add(
            -self.y2,
            self.a1.mul_add(
                -self.y1,
                self.b2
                    .mul_add(self.x2, self.b0.mul_add(x, self.b1 * self.x1)),
            ),
        );
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Process a buffer of samples.
    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }

    /// Reset the filter state.
    pub const fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}
