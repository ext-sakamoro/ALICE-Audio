//! Waveform generation: sine / square / sawtooth / noise.

use core::f64::consts::PI;

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

/// Generate white noise using a simple LCG PRNG, uniform on `[−1, 1)`.
///
/// Until 2026-09-17 the top 31 bits were scaled as if they were 32, so the
/// output was uniform on `[−1, 0)` — a −0.5 DC offset (oracle
/// `tests/analytic_oracle.rs`).
pub fn gen_noise(num_samples: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_add(1);
    (0..num_samples)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // top 32 bits → [0, 1) → [−1, 1)
            (state >> 32) as f64 / 2_147_483_648.0 - 1.0
        })
        .collect()
}
