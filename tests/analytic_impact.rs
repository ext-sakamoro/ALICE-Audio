//! An impact against what its numbers say: pitch by zero crossings, level by the decay
//! law, length, and the speed mapping. Bits are pinned in `tests/cross_target_bits.rs`.

// the expected values come from the platform's arithmetic, independent of the code under test
#![allow(clippy::disallowed_methods)]

use alice_audio::{
    render_impact, ImpactMaterial, ImpactVoice, ImpactWave, ATTACK_S, END_LEVEL,
    MAX_IMPACT_SAMPLES, TAIL_S,
};

const SR: f64 = 48_000.0;

fn tone(wave: ImpactWave, slide: f64) -> ImpactVoice {
    ImpactVoice {
        freq: 1000.0,
        slide,
        decay_s: 0.5,
        gain: 0.5,
        wave,
        noise_gain: 0.0,
        noise_s: 0.0,
        noise_cutoff: 1000.0,
    }
}

fn rising_crossings(x: &[f32]) -> Vec<f64> {
    x.windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] < 0.0 && w[1] >= 0.0)
        .map(|(i, w)| i as f64 + f64::from(-w[0] / (w[1] - w[0])))
        .collect()
}

fn pitch_between(x: &[f32], from_s: f64, to_s: f64) -> f64 {
    let a = (from_s * SR) as usize;
    let b = (to_s * SR) as usize;
    let c = rising_crossings(&x[a..b]);
    assert!(c.len() > 4, "{} crossings", c.len());
    (c.len() - 1) as f64 * SR / (c[c.len() - 1] - c[0])
}

fn peak_near(x: &[f32], t: f64) -> f64 {
    let a = (t * SR) as usize;
    x[a..a + 96]
        .iter()
        .fold(0.0_f64, |m, v| m.max(f64::from(v.abs())))
}

#[test]
fn the_length_is_the_tone_and_its_tail() {
    let x = render_impact(&tone(ImpactWave::Sine, 1.0), SR);
    assert_eq!(x.len(), ((0.5 + TAIL_S) * SR) as usize);
    assert!(x[(0.5 * SR) as usize..].iter().all(|&v| v == 0.0));
}

#[test]
fn the_pitch_is_the_frequency_for_each_wave() {
    for wave in [ImpactWave::Sine, ImpactWave::Triangle] {
        let x = render_impact(&tone(wave, 1.0), SR);
        let f = pitch_between(&x, 0.01, 0.2);
        assert!((f - 1000.0).abs() < 0.5, "{wave:?}: {f}");
    }
}

#[test]
fn the_pitch_slides_exponentially() {
    let x = render_impact(&tone(ImpactWave::Sine, 0.5), SR);
    for t in [0.05, 0.125, 0.25] {
        let f = pitch_between(&x, t - 0.01, t + 0.01);
        let want = 1000.0 * 0.5_f64.powf(t / 0.5);
        assert!((f - want).abs() < 0.01 * want, "t {t}: {f} against {want}");
    }
}

#[test]
fn the_level_follows_the_decay_law() {
    let x = render_impact(&tone(ImpactWave::Sine, 1.0), SR);
    for t in [0.02, 0.05, 0.1, 0.2] {
        let got = peak_near(&x, t);
        let want = 0.5 * END_LEVEL.powf((t - ATTACK_S) / (0.5 - ATTACK_S));
        assert!(
            (got - want).abs() < 0.03 * want,
            "t {t}: {got} against {want}"
        );
    }
    assert!(f64::from(x[0].abs()) < 1e-9, "starts from silence");
    assert!(peak_near(&x, ATTACK_S - 0.0005) > 0.3);
}

#[test]
fn the_rise_is_a_straight_line_from_silence() {
    let v = ImpactVoice {
        freq: 12_000.0,
        ..tone(ImpactWave::Triangle, 1.0)
    };
    let x = render_impact(&v, SR);
    // a quarter period every sample at 12 kHz, so every fourth sample starting at 1 is a crest
    for i in (1..(ATTACK_S * SR) as usize).step_by(4) {
        let want = 0.5 * i as f64 / SR / ATTACK_S;
        assert!(
            (f64::from(x[i]) - want).abs() < 1e-5,
            "sample {i}: {} against {want}",
            x[i]
        );
    }
}

/// The samples of a steady triangle, phase by phase, under the decay law.
#[test]
fn a_triangle_is_a_triangle() {
    let x = render_impact(&tone(ImpactWave::Triangle, 1.0), SR);
    for (i, &sample) in x.iter().enumerate().take(960).skip(480) {
        let t = i as f64 / SR;
        let level = 0.5 * END_LEVEL.powf((t - ATTACK_S) / (0.5 - ATTACK_S));
        let p = (t * 1000.0).fract();
        let shape = if p < 0.25 {
            4.0 * p
        } else if p < 0.75 {
            2.0 - 4.0 * p
        } else {
            4.0 * p - 4.0
        };
        let want = level * shape;
        assert!(
            (f64::from(sample) - want).abs() < 2e-3,
            "sample {i}: {} against {want}",
            sample
        );
    }
}

#[test]
fn a_triangle_has_the_stated_peak_and_no_more() {
    let v = ImpactVoice {
        decay_s: 0.5,
        ..tone(ImpactWave::Triangle, 1.0)
    };
    let x = render_impact(&v, SR);
    let peak = x.iter().fold(0.0_f64, |m, s| m.max(f64::from(s.abs())));
    assert!(peak <= 0.5 + 1e-6 && peak > 0.45, "{peak}");
}

#[test]
fn the_noise_lasts_its_length_and_stays_under_its_cutoff() {
    let v = ImpactVoice {
        gain: 0.0,
        noise_gain: 0.5,
        noise_s: 0.1,
        noise_cutoff: 1000.0,
        ..tone(ImpactWave::Sine, 1.0)
    };
    let x = render_impact(&v, SR);
    let end = (0.1 * SR) as usize;
    assert!(x[end + 1..].iter().all(|&s| s == 0.0));
    assert!(x[..end].iter().any(|&s| s != 0.0));
    // the mean squared step of a signal low-passed at 1 kHz is far below that of white noise
    let power: f64 = x[..end].iter().map(|&s| f64::from(s * s)).sum();
    let steps: f64 = x[..end]
        .windows(2)
        .map(|w| f64::from((w[1] - w[0]).powi(2)))
        .sum();
    assert!(steps / power < 0.1, "{}", steps / power);
}

#[test]
fn the_speed_sets_strength_up_to_full_speed() {
    let at = |m, s| ImpactVoice::from_speed(m, s);
    let cases = [
        (
            ImpactMaterial::HardMetal,
            1900.0,
            2500.0,
            0.12,
            0.37,
            0.0,
            0.3,
        ),
        (ImpactMaterial::Soft, 170.0, 230.0, 0.25, 0.6, 0.0, 0.15),
    ];
    for (m, f0, f1, g0, g1, n0, n1) in cases {
        let (quiet, full) = (at(m, 0.0), at(m, 6.0));
        assert_eq!((quiet.freq, quiet.gain, quiet.noise_gain), (f0, g0, n0));
        assert_eq!(full.freq, f1);
        assert!((full.gain - g1).abs() < 1e-12);
        assert_eq!(full.noise_gain, n1);
        assert_eq!(at(m, 60.0), full);
        let half = at(m, 3.0);
        assert!((half.freq - 0.5 * (f0 + f1)).abs() < 1e-9);
        for bad in [f64::NAN, -1.0, f64::NEG_INFINITY] {
            assert_eq!(at(m, bad), quiet, "{bad}");
        }
        assert!(!render_impact(&full, SR).is_empty());
    }
}

#[test]
fn a_number_out_of_range_gives_no_samples() {
    let good = tone(ImpactWave::Sine, 1.0);
    let cases = [
        ImpactVoice { freq: 0.0, ..good },
        ImpactVoice {
            freq: f64::NAN,
            ..good
        },
        ImpactVoice { slide: 0.0, ..good },
        ImpactVoice {
            decay_s: ATTACK_S,
            ..good
        },
        ImpactVoice {
            decay_s: f64::INFINITY,
            ..good
        },
        ImpactVoice { gain: -1.0, ..good },
        ImpactVoice {
            noise_gain: f64::NAN,
            ..good
        },
        ImpactVoice {
            noise_s: -1.0,
            ..good
        },
        ImpactVoice {
            noise_cutoff: 0.0,
            ..good
        },
    ];
    for v in cases {
        assert!(render_impact(&v, SR).is_empty(), "{v:?}");
    }
    assert!(render_impact(&good, 0.0).is_empty());
    assert!(render_impact(&good, f64::NAN).is_empty());
    assert!(!render_impact(&good, SR).is_empty());
}

#[test]
fn two_renders_are_the_same() {
    for m in [ImpactMaterial::HardMetal, ImpactMaterial::Soft] {
        let v = ImpactVoice::from_speed(m, 4.0);
        let (a, b) = (render_impact(&v, SR), render_impact(&v, SR));
        assert_eq!(
            a.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
        );
    }
}

fn silent(freq: f64, decay_s: f64) -> ImpactVoice {
    ImpactVoice {
        freq,
        slide: 1.0,
        decay_s,
        gain: 0.5,
        wave: ImpactWave::Sine,
        noise_gain: 0.0,
        noise_s: 0.0,
        noise_cutoff: 1000.0,
    }
}

/// The largest length is given, and one sample more is refused before anything is allocated.
#[test]
fn the_length_has_a_stated_limit() {
    let rate = 1000.0;
    let limit = MAX_IMPACT_SAMPLES as f64;
    let at_limit = silent(100.0, (limit + 0.5) / rate - TAIL_S);
    assert_eq!(render_impact(&at_limit, rate).len(), MAX_IMPACT_SAMPLES);
    let beyond = silent(100.0, (limit + 1.5) / rate - TAIL_S);
    assert!(render_impact(&beyond, rate).is_empty());
}

/// Lengths that cannot be allocated give no samples: neither an abort nor a panic.
#[test]
fn a_length_too_large_to_allocate_gives_no_samples() {
    assert!(render_impact(&silent(100.0, 1e12), SR).is_empty());
    assert!(render_impact(&silent(100.0, 1e30), SR).is_empty());
    assert!(render_impact(&silent(100.0, 0.1), 1e300).is_empty());
    let mut noise = silent(100.0, 0.1);
    noise.noise_gain = 0.1;
    noise.noise_s = 1e300;
    assert_eq!(
        render_impact(&noise, SR).len(),
        render_impact(&silent(100.0, 0.1), SR).len()
    );
}

/// The pitch at either end stays at or under half the sample rate.
#[test]
fn a_pitch_over_half_the_sample_rate_gives_no_samples() {
    let half = SR / 2.0;
    assert!(!render_impact(&silent(half, 0.1), SR).is_empty());
    assert!(render_impact(&silent(half * 1.001, 0.1), SR).is_empty());
    assert!(render_impact(&silent(1e308, 0.1), SR).is_empty());
    let mut up = silent(half / 2.0, 0.1);
    up.slide = 2.0;
    assert!(!render_impact(&up, SR).is_empty());
    up.slide = 2.01;
    assert!(render_impact(&up, SR).is_empty());
    up.slide = 1e308;
    assert!(render_impact(&up, SR).is_empty());
}

/// A level that leaves the range of a sample gives no samples, not some infinite ones.
#[test]
fn a_level_out_of_the_sample_range_gives_no_samples() {
    for gain in [1e39, 1e308] {
        let mut v = silent(100.0, 0.1);
        v.gain = gain;
        assert!(render_impact(&v, SR).is_empty(), "gain {gain}");
    }
    let mut v = silent(100.0, 0.1);
    v.noise_gain = 1e308;
    v.noise_s = 0.02;
    assert!(render_impact(&v, SR).is_empty());
    // large but in range stays
    let mut v = silent(100.0, 0.1);
    v.gain = 1e30;
    let x = render_impact(&v, SR);
    assert!(!x.is_empty() && x.iter().all(|s| s.is_finite()));
}
