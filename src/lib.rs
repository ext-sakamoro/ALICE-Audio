#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::struct_field_names
)]

//! ALICE-Audio: General-purpose audio processing library.
//!
//! Provides FFT (Cooley-Tukey), FIR/IIR filters, multi-channel mixer,
//! effects (reverb, delay, chorus, EQ), waveform generation,
//! sample rate conversion, ADSR envelope, and ring buffer.

use core::f64::consts::PI;

// ============================================================
// Complex number
// ============================================================

/// Minimal complex number for FFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl core::ops::Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

// ============================================================
// FFT (Cooley-Tukey radix-2 DIT)
// ============================================================

/// Compute the FFT of `data` in-place. Length must be a power of two.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn fft(data: &mut [Complex]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be power of two");
    if n <= 1 {
        return;
    }
    bit_reverse_permutation(data);
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = -2.0 * PI / size as f64;
        for k in 0..half {
            let w = Complex::new((angle_step * k as f64).cos(), (angle_step * k as f64).sin());
            let mut j = k;
            while j < n {
                let t = w * data[j + half];
                let u = data[j];
                data[j] = u + t;
                data[j + half] = u - t;
                j += size;
            }
        }
        size *= 2;
    }
}

/// Compute the inverse FFT in-place.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
pub fn ifft(data: &mut [Complex]) {
    let n = data.len();
    // Conjugate
    for c in data.iter_mut() {
        c.im = -c.im;
    }
    fft(data);
    let inv = 1.0 / n as f64;
    for c in data.iter_mut() {
        c.re *= inv;
        c.im = -c.im * inv;
    }
}

fn bit_reverse_permutation(data: &mut [Complex]) {
    let n = data.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);
        if i < rev {
            data.swap(i, rev);
        }
    }
}

// ============================================================
// Ring Buffer
// ============================================================

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

// ============================================================
// Waveform Generation
// ============================================================

/// Generate a sine wave.
pub fn gen_sine(frequency: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
    (0..num_samples)
        .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin())
        .collect()
}

/// Generate a square wave.
pub fn gen_square(frequency: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
    (0..num_samples)
        .map(|i| {
            let phase = (frequency * i as f64 / sample_rate).fract();
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Generate a sawtooth wave.
pub fn gen_sawtooth(frequency: f64, sample_rate: f64, num_samples: usize) -> Vec<f64> {
    (0..num_samples)
        .map(|i| {
            let phase = (frequency * i as f64 / sample_rate).fract();
            2.0f64.mul_add(phase, -1.0)
        })
        .collect()
}

/// Generate white noise using a simple LCG PRNG.
pub fn gen_noise(num_samples: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(1);
    (0..num_samples)
        .map(|_| {
            // LCG parameters from Numerical Recipes
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Map to [-1, 1]
            (state >> 33) as f64 / (f64::from(u32::MAX) / 2.0) - 1.0
        })
        .collect()
}

// ============================================================
// ADSR Envelope
// ============================================================

/// ADSR envelope generator. Times are in samples.
#[derive(Debug, Clone, Copy)]
pub struct Adsr {
    pub attack: usize,
    pub decay: usize,
    pub sustain_level: f64,
    pub release: usize,
}

impl Adsr {
    pub const fn new(attack: usize, decay: usize, sustain_level: f64, release: usize) -> Self {
        Self {
            attack,
            decay,
            sustain_level,
            release,
        }
    }

    /// Generate the full envelope for a note of `hold_samples` duration
    /// (attack + decay + sustain hold + release).
    pub fn generate(&self, hold_samples: usize) -> Vec<f64> {
        let total = hold_samples + self.release;
        let mut env = Vec::with_capacity(total);
        for i in 0..total {
            env.push(self.sample(i, hold_samples));
        }
        env
    }

    /// Get envelope value at sample index `i` with note-off at `hold_samples`.
    pub fn sample(&self, i: usize, hold_samples: usize) -> f64 {
        if i < self.attack {
            // Attack
            i as f64 / self.attack.max(1) as f64
        } else if i < self.attack + self.decay {
            // Decay
            let t = (i - self.attack) as f64 / self.decay.max(1) as f64;
            1.0 - t * (1.0 - self.sustain_level)
        } else if i < hold_samples {
            // Sustain
            self.sustain_level
        } else if i < hold_samples + self.release {
            // Release
            let t = (i - hold_samples) as f64 / self.release.max(1) as f64;
            self.sustain_level * (1.0 - t)
        } else {
            0.0
        }
    }
}

// ============================================================
// FIR Filter
// ============================================================

/// Finite Impulse Response filter.
pub struct FirFilter {
    coeffs: Vec<f64>,
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
                // Hamming window
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

// ============================================================
// IIR Filter (Biquad)
// ============================================================

/// Second-order IIR (biquad) filter.
///
/// Transfer function: `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
#[derive(Debug, Clone)]
pub struct IirFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
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

// ============================================================
// Mixer (multi-channel)
// ============================================================

/// Mix multiple mono channels into a single output with per-channel gain.
pub fn mix_channels(channels: &[&[f64]], gains: &[f64]) -> Vec<f64> {
    if channels.is_empty() {
        return Vec::new();
    }
    let max_len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut output = vec![0.0; max_len];
    for (ch_idx, &channel) in channels.iter().enumerate() {
        let gain = gains.get(ch_idx).copied().unwrap_or(1.0);
        for (i, &sample) in channel.iter().enumerate() {
            output[i] += sample * gain;
        }
    }
    output
}

/// Pan a mono signal to stereo. `pan` ranges from -1.0 (left) to 1.0 (right).
pub fn pan_stereo(mono: &[f64], pan: f64) -> (Vec<f64>, Vec<f64>) {
    let pan_clamped = pan.clamp(-1.0, 1.0);
    let left_gain = ((1.0 - pan_clamped) / 2.0).sqrt();
    let right_gain = f64::midpoint(1.0, pan_clamped).sqrt();
    let left: Vec<f64> = mono.iter().map(|&s| s * left_gain).collect();
    let right: Vec<f64> = mono.iter().map(|&s| s * right_gain).collect();
    (left, right)
}

// ============================================================
// Effects
// ============================================================

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
    combs: Vec<CombFilter>,
    allpasses: Vec<AllPassFilter>,
    mix: f64,
}

impl Reverb {
    /// Create a reverb with default room parameters.
    pub fn new(sample_rate: f64, mix: f64) -> Self {
        // Comb filter delay times in ms (Schroeder reverb)
        let comb_delays_ms = [29.7, 37.1, 41.1, 43.7];
        let comb_feedback = 0.84;
        let combs: Vec<CombFilter> = comb_delays_ms
            .iter()
            .map(|&ms| {
                let samples = (ms * sample_rate / 1000.0) as usize;
                CombFilter::new(samples, comb_feedback)
            })
            .collect();

        // All-pass filter delay times in ms
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
        // Sum comb filter outputs
        let mut comb_sum = 0.0;
        for comb in &mut self.combs {
            comb_sum += comb.process(sample);
        }
        comb_sum /= self.combs.len() as f64;

        // Series all-pass filters
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

/// Comb filter used internally by reverb.
struct CombFilter {
    buffer: RingBuffer,
    delay: usize,
    feedback: f64,
}

impl CombFilter {
    fn new(delay: usize, feedback: f64) -> Self {
        Self {
            buffer: RingBuffer::new(delay + 1),
            delay,
            feedback,
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        let delayed = self.buffer.read(self.delay);
        let out = delayed.mul_add(self.feedback, sample);
        self.buffer.push(out);
        delayed
    }
}

/// All-pass filter used internally by reverb.
struct AllPassFilter {
    buffer: RingBuffer,
    delay: usize,
    gain: f64,
}

impl AllPassFilter {
    fn new(delay: usize, gain: f64) -> Self {
        Self {
            buffer: RingBuffer::new(delay + 1),
            delay,
            gain,
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        let delayed = self.buffer.read(self.delay);
        let input = delayed.mul_add(self.gain, sample);
        self.buffer.push(input);
        sample.mul_add(-self.gain, delayed)
    }
}

/// Simple chorus effect using modulated delay.
pub struct Chorus {
    buffer: RingBuffer,
    rate: f64,
    depth: f64,
    mix: f64,
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
    filter: IirFilter,
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

// ============================================================
// Sample Rate Conversion (linear interpolation)
// ============================================================

/// Resample audio using linear interpolation.
pub fn resample(input: &[f64], from_rate: f64, to_rate: f64) -> Vec<f64> {
    if input.is_empty() || from_rate <= 0.0 || to_rate <= 0.0 {
        return Vec::new();
    }
    let ratio = from_rate / to_rate;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let s0 = input[idx.min(input.len() - 1)];
        let s1 = input[(idx + 1).min(input.len() - 1)];
        output.push((s1 - s0).mul_add(frac, s0));
    }
    output
}

// ============================================================
// Utility
// ============================================================

/// Compute RMS of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&s| s * s).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

/// Compute peak amplitude.
pub fn peak(signal: &[f64]) -> f64 {
    signal.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)
}

/// Normalize a signal to peak amplitude of 1.0.
pub fn normalize(signal: &mut [f64]) {
    let p = peak(signal);
    if p > 1e-12 {
        let inv = 1.0 / p;
        for s in signal.iter_mut() {
            *s *= inv;
        }
    }
}

/// Apply gain in decibels.
pub fn apply_gain_db(signal: &mut [f64], db: f64) {
    let gain = 10.0_f64.powf(db / 20.0);
    for s in signal.iter_mut() {
        *s *= gain;
    }
}

/// Hard clip a signal to [-threshold, threshold].
pub fn hard_clip(signal: &mut [f64], threshold: f64) {
    for s in signal.iter_mut() {
        *s = s.clamp(-threshold, threshold);
    }
}

/// Soft clip using tanh.
pub fn soft_clip(signal: &mut [f64], drive: f64) {
    for s in signal.iter_mut() {
        *s = (*s * drive).tanh();
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // -- Complex tests --

    #[test]
    fn test_complex_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a + b;
        assert!(approx_eq(c.re, 4.0, EPSILON));
        assert!(approx_eq(c.im, 6.0, EPSILON));
    }

    #[test]
    fn test_complex_sub() {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(2.0, 1.0);
        let c = a - b;
        assert!(approx_eq(c.re, 3.0, EPSILON));
        assert!(approx_eq(c.im, 2.0, EPSILON));
    }

    #[test]
    fn test_complex_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        assert!(approx_eq(c.re, -5.0, EPSILON));
        assert!(approx_eq(c.im, 10.0, EPSILON));
    }

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!(approx_eq(c.magnitude(), 5.0, EPSILON));
    }

    #[test]
    fn test_complex_phase() {
        let c = Complex::new(1.0, 1.0);
        assert!(approx_eq(c.phase(), PI / 4.0, EPSILON));
    }

    #[test]
    fn test_complex_zero() {
        let c = Complex::new(0.0, 0.0);
        assert!(approx_eq(c.magnitude(), 0.0, EPSILON));
    }

    // -- FFT tests --

    #[test]
    fn test_fft_single() {
        let mut data = vec![Complex::new(1.0, 0.0)];
        fft(&mut data);
        assert!(approx_eq(data[0].re, 1.0, EPSILON));
        assert!(approx_eq(data[0].im, 0.0, EPSILON));
    }

    #[test]
    fn test_fft_two() {
        let mut data = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
        fft(&mut data);
        assert!(approx_eq(data[0].re, 0.0, EPSILON));
        assert!(approx_eq(data[1].re, 2.0, EPSILON));
    }

    #[test]
    fn test_fft_four_dc() {
        let mut data = vec![Complex::new(1.0, 0.0); 4];
        fft(&mut data);
        assert!(approx_eq(data[0].re, 4.0, EPSILON));
        for i in 1..4 {
            assert!(approx_eq(data[i].magnitude(), 0.0, 1e-6));
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let original: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut data = original.clone();
        fft(&mut data);
        ifft(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!(approx_eq(a.re, b.re, 1e-6));
            assert!(approx_eq(a.im, b.im, 1e-6));
        }
    }

    #[test]
    fn test_fft_parseval() {
        let mut data: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64 * 0.3).sin(), 0.0))
            .collect();
        let time_energy: f64 = data.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        fft(&mut data);
        let freq_energy: f64 = data.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        assert!(approx_eq(time_energy * 16.0, freq_energy, 1e-6));
    }

    #[test]
    fn test_fft_linearity() {
        let a: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let b: Vec<Complex> = (0..8)
            .map(|i| Complex::new((i as f64).sin(), 0.0))
            .collect();
        let sum: Vec<Complex> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let mut fa = a;
        let mut fb = b;
        let mut fs = sum;
        fft(&mut fa);
        fft(&mut fb);
        fft(&mut fs);

        for i in 0..8 {
            let expected = fa[i] + fb[i];
            assert!(approx_eq(fs[i].re, expected.re, 1e-6));
            assert!(approx_eq(fs[i].im, expected.im, 1e-6));
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn test_fft_non_power_of_two() {
        let mut data = vec![Complex::new(0.0, 0.0); 3];
        fft(&mut data);
    }

    #[test]
    fn test_fft_large() {
        let n = 256;
        let mut data: Vec<Complex> = (0..n)
            .map(|i| Complex::new((2.0 * PI * 4.0 * i as f64 / n as f64).sin(), 0.0))
            .collect();
        fft(&mut data);
        // Bin 4 should have the peak
        let peak_bin = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.magnitude().partial_cmp(&b.magnitude()).unwrap())
            .unwrap()
            .0;
        assert!(peak_bin == 4 || peak_bin == n - 4);
    }

    // -- Ring Buffer tests --

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RingBuffer::new(4);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        assert!(approx_eq(rb.read(1), 3.0, EPSILON));
        assert!(approx_eq(rb.read(2), 2.0, EPSILON));
        assert!(approx_eq(rb.read(3), 1.0, EPSILON));
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut rb = RingBuffer::new(3);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        rb.push(4.0); // overwrites 1.0
        assert!(approx_eq(rb.read(1), 4.0, EPSILON));
        assert!(approx_eq(rb.read(3), 2.0, EPSILON));
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut rb = RingBuffer::new(4);
        rb.push(5.0);
        rb.push(6.0);
        rb.clear();
        assert!(approx_eq(rb.read(1), 0.0, EPSILON));
    }

    #[test]
    fn test_ring_buffer_len() {
        let rb = RingBuffer::new(10);
        assert_eq!(rb.len(), 10);
        assert!(!rb.is_empty());
    }

    #[test]
    fn test_ring_buffer_empty() {
        let mut rb = RingBuffer::new(0);
        assert!(rb.is_empty());
        rb.push(1.0);
        assert!(approx_eq(rb.read(0), 0.0, EPSILON));
    }

    // -- Waveform tests --

    #[test]
    fn test_gen_sine_length() {
        let wave = gen_sine(440.0, 44100.0, 1000);
        assert_eq!(wave.len(), 1000);
    }

    #[test]
    fn test_gen_sine_range() {
        let wave = gen_sine(440.0, 44100.0, 44100);
        for &s in &wave {
            assert!(s >= -1.0 - EPSILON && s <= 1.0 + EPSILON);
        }
    }

    #[test]
    fn test_gen_sine_zero_crossing() {
        let wave = gen_sine(1.0, 100.0, 100);
        // At sample 0, sin(0) = 0
        assert!(approx_eq(wave[0], 0.0, 1e-6));
    }

    #[test]
    fn test_gen_sine_peak() {
        // Sine at quarter period should be ~1.0
        let wave = gen_sine(1.0, 100.0, 100);
        assert!(approx_eq(wave[25], 1.0, 0.1));
    }

    #[test]
    fn test_gen_square_values() {
        let wave = gen_square(1.0, 100.0, 100);
        for &s in &wave {
            assert!(approx_eq(s, 1.0, EPSILON) || approx_eq(s, -1.0, EPSILON));
        }
    }

    #[test]
    fn test_gen_square_length() {
        let wave = gen_square(440.0, 44100.0, 500);
        assert_eq!(wave.len(), 500);
    }

    #[test]
    fn test_gen_sawtooth_range() {
        let wave = gen_sawtooth(440.0, 44100.0, 44100);
        for &s in &wave {
            assert!(s >= -1.0 - EPSILON && s <= 1.0 + EPSILON);
        }
    }

    #[test]
    fn test_gen_sawtooth_start() {
        let wave = gen_sawtooth(1.0, 100.0, 100);
        assert!(approx_eq(wave[0], -1.0, EPSILON));
    }

    #[test]
    fn test_gen_noise_length() {
        let noise = gen_noise(1000, 42);
        assert_eq!(noise.len(), 1000);
    }

    #[test]
    fn test_gen_noise_range() {
        let noise = gen_noise(10000, 42);
        for &s in &noise {
            assert!(s >= -1.5 && s <= 1.5); // Approximately bounded
        }
    }

    #[test]
    fn test_gen_noise_different_seeds() {
        let n1 = gen_noise(100, 1);
        let n2 = gen_noise(100, 2);
        // Should not be identical
        let diff: f64 = n1.iter().zip(n2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0);
    }

    // -- ADSR tests --

    #[test]
    fn test_adsr_attack() {
        let env = Adsr::new(10, 10, 0.5, 10);
        assert!(approx_eq(env.sample(0, 100), 0.0, EPSILON));
        assert!(approx_eq(env.sample(5, 100), 0.5, EPSILON));
        assert!(approx_eq(env.sample(10, 100), 1.0, EPSILON));
    }

    #[test]
    fn test_adsr_decay() {
        let env = Adsr::new(10, 10, 0.5, 10);
        assert!(approx_eq(env.sample(10, 100), 1.0, EPSILON));
        assert!(approx_eq(env.sample(20, 100), 0.5, EPSILON));
    }

    #[test]
    fn test_adsr_sustain() {
        let env = Adsr::new(10, 10, 0.7, 10);
        assert!(approx_eq(env.sample(50, 100), 0.7, EPSILON));
    }

    #[test]
    fn test_adsr_release() {
        let env = Adsr::new(10, 10, 0.5, 20);
        assert!(approx_eq(env.sample(100, 100), 0.5, EPSILON));
        assert!(approx_eq(env.sample(110, 100), 0.25, EPSILON));
        assert!(approx_eq(env.sample(120, 100), 0.0, EPSILON));
    }

    #[test]
    fn test_adsr_generate_length() {
        let env = Adsr::new(10, 10, 0.5, 20);
        let out = env.generate(100);
        assert_eq!(out.len(), 120); // hold + release
    }

    #[test]
    fn test_adsr_zero_attack() {
        let env = Adsr::new(0, 10, 0.5, 10);
        // With 0 attack, first sample should be at decay start (1.0)
        assert!(approx_eq(env.sample(0, 100), 1.0, 0.01));
    }

    // -- FIR Filter tests --

    #[test]
    fn test_fir_identity() {
        // Identity filter: [1.0]
        let mut f = FirFilter::new(vec![1.0]);
        let out = f.process(0.5);
        assert!(approx_eq(out, 0.5, EPSILON));
    }

    #[test]
    fn test_fir_moving_average() {
        let mut f = FirFilter::new(vec![0.5, 0.5]);
        let _ = f.process(1.0);
        let out = f.process(1.0);
        assert!(approx_eq(out, 1.0, EPSILON));
    }

    #[test]
    fn test_fir_low_pass_construction() {
        let f = FirFilter::low_pass(1000.0, 44100.0, 32);
        assert_eq!(f.coeffs.len(), 33);
    }

    #[test]
    fn test_fir_process_buffer() {
        let mut f = FirFilter::new(vec![1.0]);
        let input = vec![1.0, 2.0, 3.0];
        let output = f.process_buffer(&input);
        assert_eq!(output.len(), 3);
        assert!(approx_eq(output[0], 1.0, EPSILON));
    }

    #[test]
    fn test_fir_low_pass_attenuates_high_freq() {
        let mut f = FirFilter::low_pass(1000.0, 44100.0, 64);
        // High frequency signal
        let high = gen_sine(10000.0, 44100.0, 4410);
        let filtered = f.process_buffer(&high);
        // After settling, output should be significantly attenuated
        let tail = &filtered[200..];
        let input_rms = rms(&high[200..]);
        let output_rms = rms(tail);
        assert!(output_rms < input_rms * 0.5);
    }

    // -- IIR Filter tests --

    #[test]
    fn test_iir_pass_through() {
        let mut f = IirFilter::new(1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(approx_eq(f.process(1.0), 1.0, EPSILON));
    }

    #[test]
    fn test_iir_low_pass_creation() {
        let f = IirFilter::low_pass(1000.0, 44100.0, 0.707);
        // Verify coefficients are finite
        assert!(f.b0.is_finite());
        assert!(f.a1.is_finite());
    }

    #[test]
    fn test_iir_high_pass_creation() {
        let f = IirFilter::high_pass(1000.0, 44100.0, 0.707);
        assert!(f.b0.is_finite());
    }

    #[test]
    fn test_iir_band_pass_creation() {
        let f = IirFilter::band_pass(1000.0, 44100.0, 1.0);
        assert!(f.b0.is_finite());
    }

    #[test]
    fn test_iir_reset() {
        let mut f = IirFilter::low_pass(1000.0, 44100.0, 0.707);
        f.process(1.0);
        f.process(0.5);
        f.reset();
        assert!(approx_eq(f.x1, 0.0, EPSILON));
        assert!(approx_eq(f.y1, 0.0, EPSILON));
    }

    #[test]
    fn test_iir_process_buffer() {
        let mut f = IirFilter::low_pass(5000.0, 44100.0, 0.707);
        let input = vec![1.0; 100];
        let output = f.process_buffer(&input);
        assert_eq!(output.len(), 100);
        // DC signal through low-pass should converge to ~1.0
        assert!(approx_eq(output[99], 1.0, 0.01));
    }

    #[test]
    fn test_iir_low_pass_attenuates_high() {
        let mut f = IirFilter::low_pass(1000.0, 44100.0, 0.707);
        let high = gen_sine(15000.0, 44100.0, 4410);
        let filtered = f.process_buffer(&high);
        let tail = &filtered[500..];
        assert!(rms(tail) < rms(&high[500..]) * 0.3);
    }

    // -- Mixer tests --

    #[test]
    fn test_mix_empty() {
        let out = mix_channels(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_mix_single_channel() {
        let ch = [1.0, 2.0, 3.0];
        let out = mix_channels(&[&ch], &[0.5]);
        assert!(approx_eq(out[0], 0.5, EPSILON));
        assert!(approx_eq(out[1], 1.0, EPSILON));
    }

    #[test]
    fn test_mix_two_channels() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let out = mix_channels(&[&a, &b], &[1.0, 1.0]);
        assert!(approx_eq(out[0], 1.0, EPSILON));
        assert!(approx_eq(out[1], 1.0, EPSILON));
    }

    #[test]
    fn test_mix_different_lengths() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0];
        let out = mix_channels(&[&a, &b], &[1.0, 1.0]);
        assert_eq!(out.len(), 3);
        assert!(approx_eq(out[0], 2.0, EPSILON));
        assert!(approx_eq(out[2], 3.0, EPSILON));
    }

    #[test]
    fn test_pan_center() {
        let mono = [1.0; 10];
        let (left, right) = pan_stereo(&mono, 0.0);
        assert!(approx_eq(left[0], right[0], 1e-6));
    }

    #[test]
    fn test_pan_hard_left() {
        let mono = [1.0; 10];
        let (left, right) = pan_stereo(&mono, -1.0);
        assert!(left[0] > right[0]);
        assert!(approx_eq(right[0], 0.0, 1e-6));
    }

    #[test]
    fn test_pan_hard_right() {
        let mono = [1.0; 10];
        let (left, right) = pan_stereo(&mono, 1.0);
        assert!(right[0] > left[0]);
        assert!(approx_eq(left[0], 0.0, 1e-6));
    }

    // -- Delay tests --

    #[test]
    fn test_delay_basic() {
        let mut delay = Delay::new(3, 0.0, 1.0);
        assert!(approx_eq(delay.process(1.0), 0.0, EPSILON)); // No delayed signal yet
        assert!(approx_eq(delay.process(0.0), 0.0, EPSILON));
        assert!(approx_eq(delay.process(0.0), 0.0, EPSILON));
        assert!(approx_eq(delay.process(0.0), 1.0, EPSILON)); // Delayed by 3
    }

    #[test]
    fn test_delay_dry_wet() {
        let mut delay = Delay::new(10, 0.0, 0.0);
        // Full dry
        assert!(approx_eq(delay.process(1.0), 1.0, EPSILON));
    }

    #[test]
    fn test_delay_buffer() {
        let mut delay = Delay::new(2, 0.0, 0.5);
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let output = delay.process_buffer(&input);
        assert_eq!(output.len(), 5);
    }

    // -- Reverb tests --

    #[test]
    fn test_reverb_creation() {
        let reverb = Reverb::new(44100.0, 0.3);
        assert_eq!(reverb.combs.len(), 4);
        assert_eq!(reverb.allpasses.len(), 2);
    }

    #[test]
    fn test_reverb_process() {
        let mut reverb = Reverb::new(44100.0, 0.5);
        let out = reverb.process(1.0);
        assert!(out.is_finite());
    }

    #[test]
    fn test_reverb_buffer() {
        let mut reverb = Reverb::new(44100.0, 0.3);
        let input = gen_sine(440.0, 44100.0, 1000);
        let output = reverb.process_buffer(&input);
        assert_eq!(output.len(), 1000);
    }

    #[test]
    fn test_reverb_adds_tail() {
        let mut reverb = Reverb::new(44100.0, 1.0);
        // Impulse followed by silence
        let mut input = vec![0.0; 2000];
        input[0] = 1.0;
        let output = reverb.process_buffer(&input);
        // There should be some energy in the tail
        let tail_energy: f64 = output[100..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.0);
    }

    // -- Chorus tests --

    #[test]
    fn test_chorus_creation() {
        let chorus = Chorus::new(44100.0, 1.5, 20.0, 0.5);
        assert!(approx_eq(chorus.mix, 0.5, EPSILON));
    }

    #[test]
    fn test_chorus_process() {
        let mut chorus = Chorus::new(44100.0, 1.5, 20.0, 0.5);
        let out = chorus.process(1.0);
        assert!(out.is_finite());
    }

    #[test]
    fn test_chorus_buffer() {
        let mut chorus = Chorus::new(44100.0, 1.5, 20.0, 0.5);
        let input = gen_sine(440.0, 44100.0, 1000);
        let output = chorus.process_buffer(&input);
        assert_eq!(output.len(), 1000);
    }

    // -- EQ tests --

    #[test]
    fn test_eq_band_creation() {
        let band = EqBand::peaking(1000.0, 44100.0, 6.0, 1.0);
        let out = band.filter.b0;
        assert!(out.is_finite());
    }

    #[test]
    fn test_eq_band_process() {
        let mut band = EqBand::peaking(1000.0, 44100.0, 0.0, 1.0);
        // 0 dB gain should roughly pass through
        let input = gen_sine(1000.0, 44100.0, 4410);
        let output = band.process_buffer(&input);
        let tail_in = rms(&input[500..]);
        let tail_out = rms(&output[500..]);
        assert!(approx_eq(tail_in, tail_out, 0.1));
    }

    #[test]
    fn test_equalizer_multi_band() {
        let bands = vec![
            EqBand::peaking(200.0, 44100.0, 3.0, 1.0),
            EqBand::peaking(1000.0, 44100.0, -3.0, 1.0),
            EqBand::peaking(5000.0, 44100.0, 6.0, 1.0),
        ];
        let mut eq = Equalizer::new(bands);
        let input = gen_sine(440.0, 44100.0, 1000);
        let output = eq.process_buffer(&input);
        assert_eq!(output.len(), 1000);
    }

    // -- Sample Rate Conversion tests --

    #[test]
    fn test_resample_identity() {
        let input = gen_sine(440.0, 44100.0, 441);
        let output = resample(&input, 44100.0, 44100.0);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_resample_downsample() {
        let input = gen_sine(440.0, 44100.0, 44100);
        let output = resample(&input, 44100.0, 22050.0);
        // Should be roughly half the length
        assert!(output.len() >= 22000 && output.len() <= 22200);
    }

    #[test]
    fn test_resample_upsample() {
        let input = gen_sine(440.0, 22050.0, 22050);
        let output = resample(&input, 22050.0, 44100.0);
        assert!(output.len() >= 44000 && output.len() <= 44200);
    }

    #[test]
    fn test_resample_empty() {
        let output = resample(&[], 44100.0, 22050.0);
        assert!(output.is_empty());
    }

    // -- Utility tests --

    #[test]
    fn test_rms_sine() {
        let wave = gen_sine(440.0, 44100.0, 44100);
        let r = rms(&wave);
        // RMS of sine wave is 1/sqrt(2) ≈ 0.707
        assert!(approx_eq(r, 1.0 / 2.0_f64.sqrt(), 0.01));
    }

    #[test]
    fn test_rms_empty() {
        assert!(approx_eq(rms(&[]), 0.0, EPSILON));
    }

    #[test]
    fn test_rms_dc() {
        let dc = vec![0.5; 100];
        assert!(approx_eq(rms(&dc), 0.5, EPSILON));
    }

    #[test]
    fn test_peak_basic() {
        let signal = [0.5, -0.8, 0.3];
        assert!(approx_eq(peak(&signal), 0.8, EPSILON));
    }

    #[test]
    fn test_peak_empty() {
        assert!(approx_eq(peak(&[]), 0.0, EPSILON));
    }

    #[test]
    fn test_normalize() {
        let mut signal = vec![0.5, -0.8, 0.3];
        normalize(&mut signal);
        assert!(approx_eq(peak(&signal), 1.0, EPSILON));
    }

    #[test]
    fn test_normalize_silent() {
        let mut signal = vec![0.0; 10];
        normalize(&mut signal);
        assert!(approx_eq(peak(&signal), 0.0, EPSILON));
    }

    #[test]
    fn test_apply_gain_db() {
        let mut signal = vec![1.0];
        apply_gain_db(&mut signal, 6.0);
        // +6dB ≈ 2x
        assert!(approx_eq(signal[0], 10.0_f64.powf(6.0 / 20.0), 0.01));
    }

    #[test]
    fn test_apply_gain_db_negative() {
        let mut signal = vec![1.0];
        apply_gain_db(&mut signal, -20.0);
        // -20dB = 0.1
        assert!(approx_eq(signal[0], 0.1, 0.001));
    }

    #[test]
    fn test_hard_clip() {
        let mut signal = vec![2.0, -3.0, 0.5];
        hard_clip(&mut signal, 1.0);
        assert!(approx_eq(signal[0], 1.0, EPSILON));
        assert!(approx_eq(signal[1], -1.0, EPSILON));
        assert!(approx_eq(signal[2], 0.5, EPSILON));
    }

    #[test]
    fn test_soft_clip() {
        let mut signal = vec![0.0, 100.0, -100.0];
        soft_clip(&mut signal, 1.0);
        assert!(approx_eq(signal[0], 0.0, EPSILON));
        assert!(signal[1] > 0.99 && signal[1] <= 1.0);
        assert!(signal[2] < -0.99 && signal[2] >= -1.0);
    }

    #[test]
    fn test_soft_clip_drive() {
        let mut s1 = vec![0.5];
        let mut s2 = vec![0.5];
        soft_clip(&mut s1, 1.0);
        soft_clip(&mut s2, 5.0);
        // Higher drive should produce more clipping (closer to 1.0)
        assert!(s2[0] > s1[0]);
    }

    // -- Integration tests --

    #[test]
    fn test_fft_sine_detection() {
        let n = 1024;
        let sr = 44100.0;
        let freq = 1000.0;
        let sine = gen_sine(freq, sr, n);
        let mut data: Vec<Complex> = sine.iter().map(|&s| Complex::new(s, 0.0)).collect();
        fft(&mut data);
        // Find peak bin
        let peak_bin = data[..n / 2]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.magnitude().partial_cmp(&b.magnitude()).unwrap())
            .unwrap()
            .0;
        let detected_freq = peak_bin as f64 * sr / n as f64;
        assert!((detected_freq - freq).abs() < sr / n as f64 * 2.0);
    }

    #[test]
    fn test_filter_chain() {
        let mut lp = IirFilter::low_pass(5000.0, 44100.0, 0.707);
        let mut hp = IirFilter::high_pass(200.0, 44100.0, 0.707);
        let input = gen_sine(1000.0, 44100.0, 4410);
        let mid = lp.process_buffer(&input);
        let output = hp.process_buffer(&mid);
        // 1000Hz is in the passband of both filters
        let out_rms = rms(&output[1000..]);
        assert!(out_rms > 0.3);
    }

    #[test]
    fn test_waveform_with_adsr() {
        let adsr = Adsr::new(100, 50, 0.7, 200);
        let sine = gen_sine(440.0, 44100.0, 500);
        let env = adsr.generate(300);
        let shaped: Vec<f64> = sine.iter().zip(env.iter()).map(|(&s, &e)| s * e).collect();
        assert_eq!(shaped.len(), 500);
        // First sample near zero (attack start)
        assert!(shaped[0].abs() < 0.1);
    }

    #[test]
    fn test_process_chain_delay_reverb() {
        let input = gen_sine(440.0, 44100.0, 4410);
        let mut delay = Delay::new(4410, 0.3, 0.5);
        let delayed = delay.process_buffer(&input);
        let mut reverb = Reverb::new(44100.0, 0.3);
        let output = reverb.process_buffer(&delayed);
        assert_eq!(output.len(), input.len());
        // Output should have energy
        assert!(rms(&output) > 0.0);
    }

    #[test]
    fn test_mix_and_pan() {
        let sine = gen_sine(440.0, 44100.0, 1000);
        let noise = gen_noise(1000, 42);
        let mixed = mix_channels(&[&sine, &noise], &[0.8, 0.2]);
        let (left, right) = pan_stereo(&mixed, -0.5);
        assert_eq!(left.len(), 1000);
        assert_eq!(right.len(), 1000);
        // Left should be louder
        assert!(rms(&left) > rms(&right));
    }

    #[test]
    fn test_resample_and_filter() {
        let input = gen_sine(440.0, 44100.0, 44100);
        let resampled = resample(&input, 44100.0, 22050.0);
        let mut f = FirFilter::low_pass(5000.0, 22050.0, 32);
        let filtered = f.process_buffer(&resampled);
        assert!(rms(&filtered[500..]) > 0.3);
    }

    #[test]
    fn test_equalizer_boost_cut() {
        let bands = vec![EqBand::peaking(500.0, 44100.0, 12.0, 0.5)];
        let mut eq = Equalizer::new(bands);
        let input = gen_sine(500.0, 44100.0, 4410);
        let output = eq.process_buffer(&input);
        // Boosted signal should be louder
        let in_rms = rms(&input[1000..]);
        let out_rms = rms(&output[1000..]);
        assert!(out_rms > in_rms);
    }

    #[test]
    fn test_complex_equality() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(1.0, 2.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_complex_debug() {
        let c = Complex::new(1.0, 2.0);
        let s = format!("{c:?}");
        assert!(s.contains("1.0"));
    }

    #[test]
    fn test_adsr_clone() {
        let a = Adsr::new(10, 20, 0.5, 30);
        let b = a;
        assert_eq!(a.attack, b.attack);
    }

    #[test]
    fn test_iir_clone() {
        let a = IirFilter::low_pass(1000.0, 44100.0, 0.707);
        let b = a.clone();
        assert!(approx_eq(a.b0, b.b0, EPSILON));
    }

    #[test]
    fn test_gen_sine_half_period() {
        let wave = gen_sine(1.0, 100.0, 100);
        // At half period (sample 50), sin(pi) ≈ 0
        assert!(wave[50].abs() < 0.1);
    }

    #[test]
    fn test_gen_square_first_half() {
        let wave = gen_square(1.0, 100.0, 100);
        assert!(approx_eq(wave[0], 1.0, EPSILON));
        assert!(approx_eq(wave[49], 1.0, EPSILON));
    }

    #[test]
    fn test_gen_sawtooth_midpoint() {
        let wave = gen_sawtooth(1.0, 100.0, 100);
        // Midpoint should be near 0
        assert!(wave[50].abs() < 0.1);
    }

    #[test]
    fn test_ring_buffer_sequential() {
        let mut rb = RingBuffer::new(5);
        for i in 0..5 {
            rb.push(i as f64);
        }
        assert!(approx_eq(rb.read(1), 4.0, EPSILON));
        assert!(approx_eq(rb.read(5), 0.0, EPSILON));
    }

    #[test]
    fn test_delay_feedback() {
        let mut delay = Delay::new(3, 0.5, 1.0);
        // Impulse + silence; with feedback the echo should repeat
        let mut output = Vec::new();
        output.push(delay.process(1.0));
        for _ in 0..12 {
            output.push(delay.process(0.0));
        }
        // There should be multiple echoes (energy beyond first impulse)
        let tail_energy: f64 = output[4..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.01);
    }

    #[test]
    fn test_mix_with_zero_gain() {
        let ch = [1.0, 2.0, 3.0];
        let out = mix_channels(&[&ch], &[0.0]);
        assert!(approx_eq(out[0], 0.0, EPSILON));
    }

    #[test]
    fn test_resample_short() {
        let input = vec![1.0, 0.0, 1.0, 0.0];
        let output = resample(&input, 4.0, 2.0);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut signal = vec![1.0, -0.5, 0.3];
        normalize(&mut signal);
        assert!(approx_eq(signal[0], 1.0, EPSILON));
    }

    #[test]
    fn test_hard_clip_within_range() {
        let mut signal = vec![0.3, -0.2, 0.5];
        hard_clip(&mut signal, 1.0);
        assert!(approx_eq(signal[0], 0.3, EPSILON));
    }
}
