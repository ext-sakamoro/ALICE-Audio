//! Effects: `Delay`, `Reverb`, `Chorus`, `EqBand`, `Equalizer`.

use core::f64::consts::PI;

use crate::iir::IirFilter;
use crate::ring_buffer::RingBuffer;

/// Simple delay effect.
pub struct Delay {
    buffer: RingBuffer,
    delay_samples: usize,
    feedback: f64,
    mix: f64,
}

impl Delay {
    pub fn new(delay_samples: usize, feedback: f64, mix: f64) -> Self {
        Self {
            buffer: RingBuffer::new(delay_samples + 1),
            delay_samples,
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        let delayed = self.buffer.read(self.delay_samples);
        let input = delayed.mul_add(self.feedback, sample);
        self.buffer.push(input);
        sample.mul_add(1.0 - self.mix, delayed * self.mix)
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}

/// Simple reverb using multiple comb filters and all-pass filters.
pub struct Reverb {
    pub(crate) combs: Vec<CombFilter>,
    pub(crate) allpasses: Vec<AllPassFilter>,
    mix: f64,
}

impl Reverb {
    /// Create a reverb with default room parameters.
    pub fn new(sample_rate: f64, mix: f64) -> Self {
        let comb_delays_ms = [29.7, 37.1, 41.1, 43.7];
        let comb_feedback = 0.84;
        let combs: Vec<CombFilter> = comb_delays_ms
            .iter()
            .map(|&ms| {
                let samples = (ms * sample_rate / 1000.0) as usize;
                CombFilter::new(samples, comb_feedback)
            })
            .collect();

        let ap_delays_ms = [5.0, 1.7];
        let ap_gain = 0.7;
        let allpasses: Vec<AllPassFilter> = ap_delays_ms
            .iter()
            .map(|&ms| {
                let samples = (ms * sample_rate / 1000.0) as usize;
                AllPassFilter::new(samples, ap_gain)
            })
            .collect();

        Self {
            combs,
            allpasses,
            mix: mix.clamp(0.0, 1.0),
        }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        let mut comb_sum = 0.0;
        for comb in &mut self.combs {
            comb_sum += comb.process(sample);
        }
        comb_sum /= self.combs.len() as f64;

        let mut out = comb_sum;
        for ap in &mut self.allpasses {
            out = ap.process(out);
        }

        sample.mul_add(1.0 - self.mix, out * self.mix)
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}

/// Feedback comb filter `H(z) = z⁻ᴰ / (1 − g z⁻ᴰ)` (reverb building block).
pub struct CombFilter {
    buffer: RingBuffer,
    delay: usize,
    feedback: f64,
}

impl CombFilter {
    #[must_use]
    pub fn new(delay: usize, feedback: f64) -> Self {
        Self {
            buffer: RingBuffer::new(delay + 1),
            delay,
            feedback,
        }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        let delayed = self.buffer.read(self.delay);
        let out = delayed.mul_add(self.feedback, sample);
        self.buffer.push(out);
        delayed
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}

/// Schroeder all-pass `H(z) = (−g + z⁻ᴰ) / (1 − g z⁻ᴰ)`, `|H(ω)| = 1`
/// (reverb building block).
pub struct AllPassFilter {
    buffer: RingBuffer,
    delay: usize,
    gain: f64,
}

impl AllPassFilter {
    #[must_use]
    pub fn new(delay: usize, gain: f64) -> Self {
        Self {
            buffer: RingBuffer::new(delay + 1),
            delay,
            gain,
        }
    }

    /// `v[n] = x[n] + g·v[n−D]`, `y[n] = −g·v[n] + v[n−D]`.
    ///
    /// Until 2026-09-17 the output used `−g·x[n]` (the raw input instead of
    /// the delay-line input `v`), which is not all-pass: |H| reached 1.7 at
    /// low frequencies (oracle `tests/analytic_oracle.rs`).
    pub fn process(&mut self, sample: f64) -> f64 {
        let delayed = self.buffer.read(self.delay);
        let input = delayed.mul_add(self.gain, sample);
        self.buffer.push(input);
        input.mul_add(-self.gain, delayed)
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}

/// Simple chorus effect using modulated delay.
pub struct Chorus {
    buffer: RingBuffer,
    rate: f64,
    depth: f64,
    pub(crate) mix: f64,
    phase: f64,
    sample_rate: f64,
}

impl Chorus {
    /// Create a chorus effect.
    ///
    /// - `rate`: LFO rate in Hz
    /// - `depth`: modulation depth in samples
    /// - `mix`: wet/dry mix (0.0 to 1.0)
    pub fn new(sample_rate: f64, rate: f64, depth: f64, mix: f64) -> Self {
        let max_delay = (depth * 2.0) as usize + 128;
        Self {
            buffer: RingBuffer::new(max_delay),
            rate,
            depth,
            mix: mix.clamp(0.0, 1.0),
            phase: 0.0,
            sample_rate,
        }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        self.buffer.push(sample);

        let lfo = (2.0 * PI * self.phase).sin();
        self.phase += self.rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        let delay = self.depth.mul_add(lfo, self.depth) as usize;
        let delayed = self.buffer.read(delay.max(1));

        sample.mul_add(1.0 - self.mix, delayed * self.mix)
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}

/// Parametric EQ band (wrapper around biquad).
pub struct EqBand {
    pub(crate) filter: IirFilter,
}

impl EqBand {
    /// Create a peaking EQ band.
    pub fn peaking(center: f64, sample_rate: f64, gain_db: f64, q: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * center / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

        Self {
            filter: IirFilter::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0),
        }
    }

    /// Process a single sample.
    pub fn process(&mut self, sample: f64) -> f64 {
        self.filter.process(sample)
    }

    /// Process a buffer.
    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        self.filter.process_buffer(samples)
    }
}

/// Multi-band parametric EQ.
pub struct Equalizer {
    bands: Vec<EqBand>,
}

impl Equalizer {
    pub const fn new(bands: Vec<EqBand>) -> Self {
        Self { bands }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        let mut out = sample;
        for band in &mut self.bands {
            out = band.process(out);
        }
        out
    }

    pub fn process_buffer(&mut self, samples: &[f64]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s)).collect()
    }
}
