//! Utility: RMS, peak, normalize, gain, clip.

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
