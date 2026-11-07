//! The sound of one impact, computed from a few numbers.
//!
//! A decaying tone and a short burst of filtered noise. No audio data is involved, and
//! the samples are a pure function of the arguments on every target.
//!
//! The tone's level rises for [`ATTACK_S`], then falls exponentially to [`END_LEVEL`] at
//! `decay_s`; its pitch moves exponentially from `freq` to `freq * slide` over the same
//! time. The noise runs at a constant level for `noise_s` through a low-pass.

use alice_det_math::{clamp64, exp64, ln64, max64, min64, sin64};
use core::f64::consts::TAU;

use crate::iir::IirFilter;
use crate::waveform::gen_noise;

/// Rise of the tone, in seconds.
pub const ATTACK_S: f64 = 0.004;
/// Level the tone's decay reaches at `decay_s`.
pub const END_LEVEL: f64 = 0.0001;
/// Silence kept after the tone, in seconds.
pub const TAIL_S: f64 = 0.02;
/// Speed at which an impact is at full strength, in m/s.
pub const FULL_SPEED: f64 = 6.0;
/// The most samples one impact can have: about 87 s at 48 kHz.
pub const MAX_IMPACT_SAMPLES: usize = 1 << 22;
/// Q of the noise's low-pass.
const NOISE_Q: f64 = 1.122_018_454_301_963_3;
const NOISE_SEED: u64 = 0x9e37_79b9;

/// The shape of the tone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImpactWave {
    /// A sine.
    Sine,
    /// A triangle, with its partials falling as the square of their number.
    Triangle,
}

/// What strikes: sets the character of [`ImpactVoice::from_speed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImpactMaterial {
    /// A bright, short ring.
    HardMetal,
    /// A low thud falling in pitch.
    Soft,
}

/// The numbers one impact is made from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactVoice {
    /// Pitch at the start, in Hz.
    pub freq: f64,
    /// Pitch at `decay_s`, as a multiple of `freq`.
    pub slide: f64,
    /// Length of the tone, in seconds.
    pub decay_s: f64,
    /// Peak level of the tone.
    pub gain: f64,
    /// Shape of the tone.
    pub wave: ImpactWave,
    /// Level of the noise.
    pub noise_gain: f64,
    /// Length of the noise, in seconds.
    pub noise_s: f64,
    /// Cutoff of the noise's low-pass, in Hz.
    pub noise_cutoff: f64,
}

impl ImpactVoice {
    /// The impact of `material` at a speed change of `speed` m/s: stronger and brighter
    /// up to [`FULL_SPEED`], the same beyond it. A speed that is not a positive number
    /// counts as 0.
    #[must_use]
    #[allow(
        clippy::suboptimal_flops,
        reason = "the sums, as plain arithmetic on every target"
    )]
    pub fn from_speed(material: ImpactMaterial, speed: f64) -> Self {
        let k = if speed > 0.0 {
            clamp64(speed / FULL_SPEED, 0.0, 1.0)
        } else {
            0.0
        };
        match material {
            ImpactMaterial::HardMetal => Self {
                freq: 1900.0 + 600.0 * k,
                slide: 1.0,
                decay_s: 0.12,
                gain: 0.12 + 0.25 * k,
                wave: ImpactWave::Triangle,
                noise_gain: 0.3 * k,
                noise_s: 0.04,
                noise_cutoff: 3000.0,
            },
            ImpactMaterial::Soft => Self {
                freq: 170.0 + 60.0 * k,
                slide: 0.6,
                decay_s: 0.16,
                gain: 0.25 + 0.35 * k,
                wave: ImpactWave::Sine,
                noise_gain: 0.15 * k,
                noise_s: 0.03,
                noise_cutoff: 900.0,
            },
        }
    }

    fn is_valid(&self, sample_rate: f64) -> bool {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let not_negative = |x: f64| x.is_finite() && x >= 0.0;
        let highest_pitch = self.freq * max64(self.slide, 1.0);
        positive(sample_rate)
            && positive(self.freq)
            && positive(self.slide)
            && highest_pitch <= sample_rate / 2.0
            && self.decay_s.is_finite()
            && self.decay_s > ATTACK_S
            && (self.decay_s + TAIL_S) * sample_rate < (MAX_IMPACT_SAMPLES + 1) as f64
            && not_negative(self.gain)
            && not_negative(self.noise_gain)
            && not_negative(self.noise_s)
            && positive(self.noise_cutoff)
    }
}

#[allow(clippy::suboptimal_flops, reason = "plain arithmetic on every target")]
fn wave_at(wave: ImpactWave, phase: f64) -> f64 {
    match wave {
        ImpactWave::Sine => sin64(phase),
        ImpactWave::Triangle => {
            let t = (phase / TAU).rem_euclid(1.0);
            if t < 0.25 {
                4.0 * t
            } else if t < 0.75 {
                2.0 - 4.0 * t
            } else {
                4.0 * t - 4.0
            }
        }
    }
}

/// The samples of one impact.
///
/// Empty when a number is not finite or out of range: `decay_s` must exceed [`ATTACK_S`], `sample_rate` must be positive, the pitch at both
/// ends must be at most half of `sample_rate`, the length at most [`MAX_IMPACT_SAMPLES`],
/// and every sample must fit an `f32`.
#[must_use]
#[allow(clippy::suboptimal_flops, reason = "plain arithmetic on every target")]
pub fn render_impact(voice: &ImpactVoice, sample_rate: f64) -> Vec<f32> {
    if !voice.is_valid(sample_rate) {
        return Vec::new();
    }
    let tone_n = (voice.decay_s * sample_rate) as usize;
    let total = ((voice.decay_s + TAIL_S) * sample_rate) as usize;
    let mut out = vec![0.0_f64; total];
    let decay_rate = -ln64(END_LEVEL) / (voice.decay_s - ATTACK_S);
    let slide_rate = ln64(voice.slide) / voice.decay_s;
    let mut phase = 0.0;
    for (i, sample) in out.iter_mut().enumerate().take(tone_n) {
        let t = i as f64 / sample_rate;
        let level = if t < ATTACK_S {
            t / ATTACK_S
        } else {
            exp64(-decay_rate * (t - ATTACK_S))
        };
        *sample = voice.gain * level * wave_at(voice.wave, phase);
        phase = (phase + TAU * voice.freq * exp64(slide_rate * t) / sample_rate).rem_euclid(TAU);
    }
    if voice.noise_gain > 0.0 && voice.noise_s > 0.0 {
        let noise_n = ((voice.noise_s * sample_rate) as usize).min(total);
        let cutoff = min64(voice.noise_cutoff, 0.45 * sample_rate);
        let mut filter = IirFilter::low_pass(cutoff, sample_rate, NOISE_Q);
        for (sample, n) in out.iter_mut().zip(gen_noise(noise_n, NOISE_SEED)) {
            *sample += voice.noise_gain * filter.process(n);
        }
    }
    let samples: Vec<f32> = out.iter().map(|&x| x as f32).collect();
    if samples.iter().all(|x| x.is_finite()) {
        samples
    } else {
        Vec::new()
    }
}
