//! Sample rate conversion (linear interpolation).

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
