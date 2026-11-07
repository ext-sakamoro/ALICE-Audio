//! WebAssembly: an impact's samples for a page to play. Only on `wasm32` with the `wasm`
//! feature. A page puts the array in an `AudioBuffer`; starting audio stays with the page.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::impact::{render_impact, ImpactMaterial, ImpactVoice, ImpactWave};

const fn material_of(kind: u32) -> Option<ImpactMaterial> {
    match kind {
        0 => Some(ImpactMaterial::HardMetal),
        1 => Some(ImpactMaterial::Soft),
        _ => None,
    }
}

/// The samples of an impact of `material` (0 hard metal, 1 soft) at a speed change of
/// `speed` m/s. Empty for an unknown material or a sample rate that is not positive.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "renderImpactAtSpeed"))]
#[must_use]
pub fn render_impact_at_speed(material: u32, speed: f64, sample_rate: f64) -> Vec<f32> {
    material_of(material).map_or_else(Vec::new, |m| {
        render_impact(&ImpactVoice::from_speed(m, speed), sample_rate)
    })
}

/// The samples of an impact from its numbers (see `ImpactVoice`); `wave` is 0 sine, 1
/// triangle. Empty when a number is refused.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "renderImpact"))]
#[must_use]
#[allow(clippy::too_many_arguments, reason = "a page passes plain numbers")]
pub fn render_impact_from_numbers(
    freq: f64,
    slide: f64,
    decay_s: f64,
    gain: f64,
    wave: u32,
    noise_gain: f64,
    noise_s: f64,
    noise_cutoff: f64,
    sample_rate: f64,
) -> Vec<f32> {
    let wave = match wave {
        0 => ImpactWave::Sine,
        1 => ImpactWave::Triangle,
        _ => return Vec::new(),
    };
    render_impact(
        &ImpactVoice {
            freq,
            slide,
            decay_s,
            gain,
            wave,
            noise_gain,
            noise_s,
            noise_cutoff,
        },
        sample_rate,
    )
}
