//! Analytic oracles — closed-form checks for the audio laws in ALICE-Audio
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or an independent f64 DFT written
//! in this file, never from the crate function under test.
//!
//! Oracle sources:
//! - waveforms: Fourier series (square 4/(πn) odd, saw 2/(πn)), sine line
//!   amplitude 1, uniform LCG noise mean 0 / variance 1/3 / range [−1, 1)
//! - FIR: the impulse response IS the coefficient vector; windowed sinc
//!   low-pass has DC gain 1, −6 dB at the cutoff, Hamming stopband ≤ −50 dB
//! - IIR / EQ (RBJ cookbook): Butterworth Q = 1/√2 ⇒ |H(fc)| = 1/√2, exact
//!   prewarped roll-off 1/√(1 + Ω⁴), peaking EQ |H(fc)| = 10^(g/20) with
//!   DC and Nyquist gain 1
//! - delay: impulse returns at D, 2D, … scaled mix·fb^(k−1); comb
//!   |H| ∈ [1/(1+g), 1/(1−g)] with peaks at ω = 2πk/D; Schroeder all-pass
//!   |H(ω)| = 1 for every ω; reverb with mix 0 is the identity and its
//!   impulse response decays (fb < 1)
//! - ADSR: linear ramps with the documented sample counts
//! - mixer: superposition Σ gain·channel, constant-power pan L² + R² = 1
//! - resample: linear ×2 upsampling has line gain (1 + cos(πk/N))/2 and an
//!   image (1 − cos(πk/N))/2 at N − k; ÷2 is exact decimation; a ramp is exact
//! - utility: rms of a sine A/√2, dB gain 10^(dB/20), tanh slope = drive

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::f64::consts::{PI, SQRT_2, TAU};

use alice_audio::adsr::Adsr;
use alice_audio::effects::{AllPassFilter, Chorus, CombFilter, Delay, EqBand, Equalizer, Reverb};
use alice_audio::fir::FirFilter;
use alice_audio::iir::IirFilter;
use alice_audio::mixer::{mix_channels, pan_stereo};
use alice_audio::resample::resample;
use alice_audio::utility::{apply_gain_db, hard_clip, normalize, peak, rms, soft_clip};
use alice_audio::waveform::{gen_noise, gen_sawtooth, gen_sine, gen_square};

/// Amplitude of the DFT line at `bin` (a tone of amplitude A on that bin
/// reads A).
fn line_amplitude(x: &[f64], bin: usize) -> f64 {
    let n = x.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let ang = TAU * bin as f64 * i as f64 / n;
        re += v * ang.cos();
        im -= v * ang.sin();
    }
    2.0 * re.hypot(im) / n
}

/// Steady-state gain at `f` Hz of a sample-by-sample processor at
/// `sample_rate`, measured on a sine through it (independent of any H(z)).
fn measured_gain(mut process: impl FnMut(f64) -> f64, f: f64, sample_rate: f64) -> f64 {
    let n = 16_384usize;
    // even bin so the second half still holds whole periods
    let bin = 2 * (f * n as f64 / sample_rate / 2.0).round().max(1.0) as usize;
    let y: Vec<f64> = (0..n)
        .map(|i| process((TAU * bin as f64 * i as f64 / n as f64).sin()))
        .collect();
    line_amplitude(&y[n / 2..], bin / 2)
}

/// The frequency `measured_gain` actually drives for a requested `f`.
fn snapped(f: f64, sample_rate: f64) -> f64 {
    let n = 16_384usize;
    let bin = 2 * (f * n as f64 / sample_rate / 2.0).round().max(1.0) as usize;
    bin as f64 * sample_rate / n as f64
}

#[test]
fn measured_gain_helper_reads_unity_on_a_pass_through() {
    for f in [100.0, 1000.0, 10_000.0] {
        assert!((measured_gain(|x| x, f, 48_000.0) - 1.0).abs() < 1e-12);
    }
}

// ───────────────────────── waveforms ──────────────────────────────────────

#[test]
fn generators_follow_their_fourier_series() {
    // 48 kHz, 375 Hz ⇒ 128 samples per period, 32 periods
    let (sr, f, n) = (48_000.0, 375.0, 4096usize);
    let periods = 32;
    let sine = gen_sine(f, sr, n);
    assert!((line_amplitude(&sine, periods) - 1.0).abs() < 1e-12);
    assert!(line_amplitude(&sine, 2 * periods) < 1e-12);
    let square = gen_square(f, sr, n);
    for k in 1..=7 {
        let a = line_amplitude(&square, periods * k);
        let expected = if k % 2 == 1 {
            4.0 / (PI * k as f64)
        } else {
            0.0
        };
        assert!(
            (a - expected).abs() < 0.02 * expected + 1e-9,
            "square harmonic {k}: {a} vs {expected}"
        );
    }
    let saw = gen_sawtooth(f, sr, n);
    for k in 1..=7 {
        let a = line_amplitude(&saw, periods * k);
        let expected = 2.0 / (PI * k as f64);
        assert!(
            (a - expected).abs() < 0.02 * expected + 1e-9,
            "saw harmonic {k}: {a} vs {expected}"
        );
    }
    assert!(saw.iter().all(|v| (-1.0..1.0).contains(v)) && square.iter().all(|v| v.abs() == 1.0));
    // uniform white noise on [−1, 1): mean 0, variance 1/3, flat spectrum
    let noise = gen_noise(200_000, 42);
    assert!(noise.iter().all(|v| (-1.0..1.0).contains(v)), "noise range");
    let mean = noise.iter().sum::<f64>() / noise.len() as f64;
    let var = noise.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / noise.len() as f64;
    assert!(mean.abs() < 0.01, "noise mean {mean}");
    assert!((var - 1.0 / 3.0).abs() < 0.01, "noise variance {var}");
    assert_ne!(gen_noise(16, 1), gen_noise(16, 2), "seed matters");
    assert_eq!(gen_noise(16, 7), gen_noise(16, 7), "deterministic");
}

// ───────────────────────── FIR ────────────────────────────────────────────

#[test]
fn fir_impulse_response_is_the_coefficient_vector() {
    let coeffs = vec![0.1, -0.4, 0.7, 0.25, 0.05];
    let mut f = FirFilter::new(coeffs.clone());
    let mut imp = vec![0.0; 8];
    imp[0] = 1.0;
    let h = f.process_buffer(&imp);
    for i in 0..8 {
        let expected = coeffs.get(i).copied().unwrap_or(0.0);
        assert!(
            (h[i] - expected).abs() < 1e-15,
            "h[{i}] = {} vs {expected}",
            h[i]
        );
    }
    // linear convolution: y[n] = Σ c[k] x[n−k]
    let x = [1.0, 2.0, -1.0, 0.5, 3.0, -2.0];
    let mut f = FirFilter::new(coeffs.clone());
    let y = f.process_buffer(&x);
    for n in 0..x.len() {
        let expected: f64 = (0..=n)
            .map(|k| coeffs.get(k).copied().unwrap_or(0.0) * x[n - k])
            .sum();
        assert!(
            (y[n] - expected).abs() < 1e-12,
            "y[{n}] = {} vs {expected}",
            y[n]
        );
    }
}

#[test]
fn windowed_sinc_low_pass_has_unit_dc_and_hamming_stopband() {
    let (sr, fc, order) = (48_000.0, 4_800.0, 64usize);
    let f = FirFilter::low_pass(fc, sr, order);
    let taps = {
        let mut g = FirFilter::low_pass(fc, sr, order);
        let mut imp = vec![0.0; order + 1];
        imp[0] = 1.0;
        g.process_buffer(&imp)
    };
    assert_eq!(taps.len(), order + 1);
    let dc: f64 = taps.iter().sum();
    assert!((dc - 1.0).abs() < 1e-9, "DC gain {dc}");
    for i in 0..=order {
        assert!(
            (taps[i] - taps[order - i]).abs() < 1e-12,
            "linear phase symmetry {i}"
        );
    }
    drop(f);
    let g = |hz: f64| {
        let mut f = FirFilter::low_pass(fc, sr, order);
        measured_gain(move |x| f.process(x), hz, sr)
    };
    assert!((g(500.0) - 1.0).abs() < 0.01, "passband {}", g(500.0));
    let at_fc = 20.0 * g(fc).log10();
    assert!(
        (at_fc + 6.0).abs() < 0.5,
        "−6 dB at cutoff, got {at_fc:.2} dB"
    );
    // Hamming transition ≈ 3.3·fs/(order+1) ≈ 2.4 kHz; beyond it ≤ −50 dB
    for hz in [9_000.0, 12_000.0, 18_000.0, 23_000.0] {
        let db = 20.0 * g(hz).log10();
        assert!(db < -50.0, "stopband {hz} Hz: {db:.1} dB");
    }
}

// ───────────────────────── IIR / EQ ───────────────────────────────────────

#[test]
fn rbj_biquads_and_peaking_eq_match_the_cookbook_closed_forms() {
    let (sr, fc) = (48_000.0, 4_800.0);
    let q = 1.0 / SQRT_2;
    let gain = |mk: &dyn Fn() -> IirFilter, hz: f64| {
        let mut f = mk();
        measured_gain(move |x| f.process(x), hz, sr)
    };
    let lp = || IirFilter::low_pass(fc, sr, q);
    assert!((gain(&lp, 100.0) - 1.0).abs() < 1e-3, "LP DC");
    assert!(
        (gain(&lp, fc) - 1.0 / SQRT_2).abs() < 2e-3,
        "LP −3 dB at fc: {}",
        gain(&lp, fc)
    );
    // exact digital Butterworth: Ω = tan(πf/fs)/tan(πfc/fs), |H| = 1/√(1 + Ω⁴)
    let omega = |hz: f64| (PI * hz / sr).tan() / (PI * fc / sr).tan();
    for hz in [2_400.0, 8_000.0, 15_000.0] {
        let expected = 1.0 / (1.0 + omega(snapped(hz, sr)).powi(4)).sqrt();
        assert!(
            (gain(&lp, hz) - expected).abs() < 2e-3,
            "LP {hz}: {} vs {expected}",
            gain(&lp, hz)
        );
    }
    let hp = || IirFilter::high_pass(fc, sr, q);
    assert!(gain(&hp, 100.0) < 0.01, "HP DC");
    assert!(
        (gain(&hp, fc) - 1.0 / SQRT_2).abs() < 2e-3,
        "HP −3 dB at fc"
    );
    for hz in [2_400.0, 8_000.0, 15_000.0] {
        let expected = 1.0 / (1.0 + omega(snapped(hz, sr)).powi(-4)).sqrt();
        assert!(
            (gain(&hp, hz) - expected).abs() < 2e-3,
            "HP {hz}: {} vs {expected}",
            gain(&hp, hz)
        );
    }
    let bp = || IirFilter::band_pass(fc, sr, 2.0);
    assert!((gain(&bp, fc) - 1.0).abs() < 2e-3, "BP unit peak");
    // band edges: |Ω − 1/Ω| = 1/Q
    for sign in [-1.0f64, 1.0] {
        let big = (sign / 2.0 + 4.25f64.sqrt()) / 2.0;
        let edge = (big * (PI * fc / sr).tan()).atan() * sr / PI;
        assert!(
            (gain(&bp, edge) - 1.0 / SQRT_2).abs() < 3e-3,
            "BP edge {edge}"
        );
    }
    // peaking EQ: |H(fc)| = 10^(g/20), unity at DC and near Nyquist, cut is the inverse
    for g_db in [6.0, -12.0] {
        let mk = || EqBand::peaking(fc, sr, g_db, 1.0);
        let at = |hz: f64| {
            let mut b = mk();
            measured_gain(move |x| b.process(x), hz, sr)
        };
        let expected = 10f64.powf(g_db / 20.0);
        assert!(
            (at(fc) - expected).abs() < 2e-3 * expected,
            "EQ {g_db} dB at fc: {}",
            at(fc)
        );
        assert!(
            (at(100.0) - 1.0).abs() < 1e-3 && (at(23_000.0) - 1.0).abs() < 1e-2,
            "EQ shoulders"
        );
    }
    // boost then cut of the same amount is the identity
    let mut eq = Equalizer::new(vec![
        EqBand::peaking(fc, sr, 9.0, 1.5),
        EqBand::peaking(fc, sr, -9.0, 1.5),
    ]);
    for hz in [500.0, fc, 12_000.0] {
        let g = measured_gain(|x| eq.process(x), hz, sr);
        assert!((g - 1.0).abs() < 1e-6, "boost+cut at {hz}: {g}");
    }
    // the difference equation is exact
    let mut f = IirFilter::new(0.5, 0.25, 0.125, -0.5, 0.25);
    let y = f.process_buffer(&[1.0, 0.0, 0.0]);
    assert_eq!(
        y,
        vec![0.5, 0.25 + 0.5 * 0.5, 0.125 + 0.5 * 0.5 - 0.25 * 0.5]
    );
    f.reset();
    assert_eq!(f.process(1.0), 0.5);
}

// ───────────────────────── delay / comb / all-pass / reverb ───────────────

#[test]
fn delay_comb_and_allpass_impulse_and_frequency_responses() {
    let d = 37usize;
    let (fb, mix) = (0.5, 0.75);
    let mut delay = Delay::new(d, fb, mix);
    let mut imp = vec![0.0; 5 * d];
    imp[0] = 1.0;
    let y = delay.process_buffer(&imp);
    for (i, &v) in y.iter().enumerate() {
        let expected = if i == 0 {
            1.0 - mix
        } else if i % d == 0 {
            mix * fb.powi((i / d - 1) as i32)
        } else {
            0.0
        };
        assert!(
            (v - expected).abs() < 1e-12,
            "delay y[{i}] = {v} vs {expected}"
        );
    }
    // comb H = z^−D / (1 − g z^−D): |H| = 1/(1−g) at ω = 2πk/D, 1/(1+g) between
    let g = 0.6;
    let sr = 48_000.0;
    let d = 48usize; // peaks every 1 kHz
    let comb_gain = |hz: f64| {
        let mut c = CombFilter::new(d, g);
        measured_gain(move |x| c.process(x), hz, sr)
    };
    assert!(
        (comb_gain(3_000.0) - 1.0 / (1.0 - g)).abs() < 1e-3,
        "comb peak {}",
        comb_gain(3_000.0)
    );
    assert!(
        (comb_gain(2_500.0) - 1.0 / (1.0 + g)).abs() < 1e-3,
        "comb notch {}",
        comb_gain(2_500.0)
    );
    // general closed form |H(ω)|² = 1/(1 − 2g cos(ωD) + g²)
    for hz in [700.0, 4_250.0, 9_100.0] {
        let w = TAU * snapped(hz, sr) / sr;
        let expected = 1.0 / (1.0 - 2.0 * g * (w * d as f64).cos() + g * g).sqrt();
        assert!(
            (comb_gain(hz) - expected).abs() < 2e-3,
            "comb {hz}: {} vs {expected}",
            comb_gain(hz)
        );
    }
    // Schroeder all-pass: |H(ω)| = 1 for every ω
    for hz in [100.0, 700.0, 3_000.0, 4_250.0, 9_100.0, 20_000.0] {
        let mut ap = AllPassFilter::new(d, 0.7);
        let gain = measured_gain(move |x| ap.process(x), hz, sr);
        assert!((gain - 1.0).abs() < 2e-3, "all-pass {hz} Hz: {gain}");
    }
    // all-pass impulse response: −g, then (1 − g²) g^(k−1) at kD
    let mut ap = AllPassFilter::new(5, 0.7);
    let mut imp = vec![0.0; 21];
    imp[0] = 1.0;
    let h = ap.process_buffer(&imp);
    assert!((h[0] + 0.7).abs() < 1e-12);
    for k in 1..=4 {
        let expected = (1.0 - 0.49) * 0.7f64.powi(k as i32 - 1);
        assert!(
            (h[5 * k] - expected).abs() < 1e-12,
            "all-pass h[{}] = {} vs {expected}",
            5 * k,
            h[5 * k]
        );
    }
    // reverb: mix 0 is the identity, impulse response decays (all feedback < 1)
    let mut dry = Reverb::new(sr, 0.0);
    let x = gen_sine(440.0, sr, 512);
    assert_eq!(dry.process_buffer(&x), x);
    let mut wet = Reverb::new(sr, 1.0);
    let mut imp = vec![0.0; 48_000 * 4];
    imp[0] = 1.0;
    let h = wet.process_buffer(&imp);
    let early: f64 = h[..4_800].iter().map(|v| v * v).sum();
    let late: f64 = h[h.len() - 4_800..].iter().map(|v| v * v).sum();
    assert!(
        early > 0.0 && late < early * 1e-4,
        "reverb decays: early {early} late {late}"
    );
    assert!(h.iter().all(|v| v.is_finite() && v.abs() < 50.0));
    // chorus with zero depth and full mix is the identity
    let mut ch = Chorus::new(sr, 2.0, 0.0, 1.0);
    assert_eq!(ch.process_buffer(&x), x);
}

// ───────────────────────── ADSR ───────────────────────────────────────────

#[test]
fn adsr_is_piecewise_linear_with_the_configured_sample_counts() {
    let env = Adsr::new(10, 20, 0.25, 40);
    let hold = 100;
    let e = env.generate(hold);
    assert_eq!(e.len(), hold + 40);
    for i in 0..e.len() {
        let expected = if i < 10 {
            i as f64 / 10.0
        } else if i < 30 {
            1.0 - (i - 10) as f64 / 20.0 * 0.75
        } else if i < hold {
            0.25
        } else {
            0.25 * (1.0 - (i - hold) as f64 / 40.0)
        };
        assert!(
            (e[i] - expected).abs() < 1e-12,
            "env[{i}] = {} vs {expected}",
            e[i]
        );
    }
    assert_eq!(env.sample(hold + 40, hold), 0.0);
    assert!(e.iter().all(|v| (0.0..=1.0).contains(v)));
    // zero-length attack starts at full level; zero decay drops straight to sustain
    let z = Adsr::new(0, 0, 0.5, 4);
    assert_eq!(z.sample(0, 10), 0.5);
    assert_eq!(Adsr::new(0, 5, 0.5, 4).sample(0, 10), 1.0);
}

// ───────────────────────── mixer ──────────────────────────────────────────

#[test]
fn mixer_is_superposition_and_pan_is_constant_power() {
    let a = [1.0, 2.0, 3.0];
    let b = [0.5, -1.0];
    let out = mix_channels(&[&a, &b], &[2.0, 4.0]);
    assert_eq!(out, vec![2.0 + 2.0, 4.0 - 4.0, 6.0]);
    assert_eq!(mix_channels(&[&a], &[]), a.to_vec(), "missing gain = 1");
    assert!(mix_channels(&[], &[]).is_empty());
    let mono = [1.0, -0.5, 0.25];
    for pan in [-1.0, -0.5, 0.0, 0.3, 1.0] {
        let (l, r) = pan_stereo(&mono, pan);
        for i in 0..mono.len() {
            let p = l[i] * l[i] + r[i] * r[i];
            assert!(
                (p - mono[i] * mono[i]).abs() < 1e-12,
                "constant power at pan {pan}"
            );
        }
    }
    let (l, r) = pan_stereo(&mono, 0.0);
    assert!(
        (l[0] - 1.0 / SQRT_2).abs() < 1e-12 && (r[0] - 1.0 / SQRT_2).abs() < 1e-12,
        "centre −3 dB"
    );
    let (l, r) = pan_stereo(&mono, -1.0);
    assert_eq!((l[0], r[0]), (1.0, 0.0), "hard left");
    let (l, r) = pan_stereo(&mono, 5.0);
    assert_eq!((l[0], r[0]), (0.0, 1.0), "clamped hard right");
}

// ───────────────────────── resample ───────────────────────────────────────

#[test]
fn linear_resampling_has_the_closed_form_line_gain_and_image() {
    let n = 1024usize;
    let k = 40usize;
    let x: Vec<f64> = (0..n)
        .map(|i| (TAU * k as f64 * i as f64 / n as f64).cos())
        .collect();
    // ×2: y[2i] = x[i], y[2i+1] = (x[i] + x[i+1])/2 ⇒ zero-stuff ⊛ [½ 1 ½]
    let up = resample(&x, 48_000.0, 96_000.0);
    assert_eq!(up.len(), 2 * n);
    for i in 0..n - 1 {
        assert_eq!(up[2 * i], x[i]);
        assert!((up[2 * i + 1] - f64::midpoint(x[i], x[i + 1])).abs() < 1e-12);
    }
    let w = PI * k as f64 / n as f64;
    assert!(
        (line_amplitude(&up, k) - (1.0 + w.cos()) / 2.0).abs() < 1e-3,
        "line gain"
    );
    assert!(
        (line_amplitude(&up, n - k) - (1.0 - w.cos()) / 2.0).abs() < 1e-3,
        "image at N − k"
    );
    // ÷2 is exact decimation
    let down = resample(&x, 96_000.0, 48_000.0);
    assert_eq!(down.len(), n / 2);
    for i in 0..n / 2 {
        assert_eq!(down[i], x[2 * i]);
    }
    // a linear ramp is reproduced exactly at any ratio
    let ramp: Vec<f64> = (0..100).map(|i| 0.5 * i as f64 - 3.0).collect();
    let r = resample(&ramp, 44_100.0, 48_000.0);
    assert_eq!(r.len(), (100.0 * 48_000.0 / 44_100.0f64).ceil() as usize);
    for (j, &v) in r.iter().enumerate() {
        let t = j as f64 * 44_100.0 / 48_000.0;
        if t <= 99.0 {
            assert!((v - (0.5 * t - 3.0)).abs() < 1e-9, "ramp {j}");
        }
    }
    assert_eq!(resample(&x, 48_000.0, 48_000.0), x, "identity");
    assert!(resample(&[], 1.0, 2.0).is_empty() && resample(&x, 0.0, 2.0).is_empty());
}

// ───────────────────────── utility ────────────────────────────────────────

#[test]
fn utility_closed_forms() {
    let a = 0.8;
    let mut s: Vec<f64> = gen_sine(1_000.0, 48_000.0, 4_800)
        .iter()
        .map(|v| v * a)
        .collect();
    assert!((rms(&s) - a / SQRT_2).abs() < 1e-9, "rms A/√2");
    assert!((peak(&s) - a).abs() < 1e-9);
    apply_gain_db(&mut s, 20.0);
    assert!((peak(&s) - 10.0 * a).abs() < 1e-9, "+20 dB = ×10");
    apply_gain_db(&mut s, -20.0 * 2f64.log10());
    assert!((peak(&s) - 5.0 * a).abs() < 1e-9, "−6.02 dB = ×½");
    normalize(&mut s);
    assert!((peak(&s) - 1.0).abs() < 1e-12);
    let mut silent = vec![0.0; 8];
    normalize(&mut silent);
    assert!(silent.iter().all(|v| *v == 0.0), "normalize(0) stays 0");
    let mut c = vec![-2.0, -0.3, 0.0, 0.3, 2.0];
    hard_clip(&mut c, 0.5);
    assert_eq!(c, vec![-0.5, -0.3, 0.0, 0.3, 0.5]);
    let mut t = vec![1e-6, 0.5, 100.0, -100.0];
    soft_clip(&mut t, 3.0);
    assert!((t[0] - 3e-6).abs() < 1e-15, "small-signal slope = drive");
    assert!((t[1] - 1.5f64.tanh()).abs() < 1e-15);
    assert!(
        (t[2] - 1.0).abs() < 1e-12 && (t[3] + 1.0).abs() < 1e-12,
        "bounded"
    );
    assert_eq!(rms(&[]), 0.0);
    assert_eq!(peak(&[]), 0.0);
}
