//! Multi-channel mixer + stereo pan.

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
