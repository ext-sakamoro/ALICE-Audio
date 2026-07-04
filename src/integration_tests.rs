//! Cross-module integration tests.

#![allow(
    clippy::doc_markdown,
    clippy::assertions_on_constants,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::manual_range_contains
)]

use core::f64::consts::PI;

use crate::adsr::*;
use crate::complex::*;
use crate::effects::*;
use crate::fir::*;
use crate::iir::*;
use crate::mixer::*;
use crate::resample::*;
use crate::ring_buffer::*;
use crate::utility::*;
use crate::waveform::*;

const EPSILON: f64 = 1e-10;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

// -- Complex --

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

// -- FFT --

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
    let peak_bin = data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.magnitude().partial_cmp(&b.magnitude()).unwrap())
        .unwrap()
        .0;
    assert!(peak_bin == 4 || peak_bin == n - 4);
}

// -- Ring Buffer --

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
    rb.push(4.0);
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

// -- Waveform --

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
    assert!(approx_eq(wave[0], 0.0, 1e-6));
}

#[test]
fn test_gen_sine_peak() {
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
        assert!(s >= -1.5 && s <= 1.5);
    }
}

#[test]
fn test_gen_noise_different_seeds() {
    let n1 = gen_noise(100, 1);
    let n2 = gen_noise(100, 2);
    let diff: f64 = n1.iter().zip(n2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1.0);
}

// -- ADSR --

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
    assert_eq!(out.len(), 120);
}

#[test]
fn test_adsr_zero_attack() {
    let env = Adsr::new(0, 10, 0.5, 10);
    assert!(approx_eq(env.sample(0, 100), 1.0, 0.01));
}

// -- FIR --

#[test]
fn test_fir_identity() {
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
    let high = gen_sine(10000.0, 44100.0, 4410);
    let filtered = f.process_buffer(&high);
    let tail = &filtered[200..];
    let input_rms = rms(&high[200..]);
    let output_rms = rms(tail);
    assert!(output_rms < input_rms * 0.5);
}

// -- IIR --

#[test]
fn test_iir_pass_through() {
    let mut f = IirFilter::new(1.0, 0.0, 0.0, 0.0, 0.0);
    assert!(approx_eq(f.process(1.0), 1.0, EPSILON));
}

#[test]
fn test_iir_low_pass_creation() {
    let f = IirFilter::low_pass(1000.0, 44100.0, 0.707);
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

// -- Mixer --

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

// -- Delay --

#[test]
fn test_delay_basic() {
    let mut delay = Delay::new(3, 0.0, 1.0);
    assert!(approx_eq(delay.process(1.0), 0.0, EPSILON));
    assert!(approx_eq(delay.process(0.0), 0.0, EPSILON));
    assert!(approx_eq(delay.process(0.0), 0.0, EPSILON));
    assert!(approx_eq(delay.process(0.0), 1.0, EPSILON));
}

#[test]
fn test_delay_dry_wet() {
    let mut delay = Delay::new(10, 0.0, 0.0);
    assert!(approx_eq(delay.process(1.0), 1.0, EPSILON));
}

#[test]
fn test_delay_buffer() {
    let mut delay = Delay::new(2, 0.0, 0.5);
    let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let output = delay.process_buffer(&input);
    assert_eq!(output.len(), 5);
}

// -- Reverb --

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
    let mut input = vec![0.0; 2000];
    input[0] = 1.0;
    let output = reverb.process_buffer(&input);
    let tail_energy: f64 = output[100..].iter().map(|s| s * s).sum();
    assert!(tail_energy > 0.0);
}

// -- Chorus --

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

// -- EQ --

#[test]
fn test_eq_band_creation() {
    let band = EqBand::peaking(1000.0, 44100.0, 6.0, 1.0);
    let out = band.filter.b0;
    assert!(out.is_finite());
}

#[test]
fn test_eq_band_process() {
    let mut band = EqBand::peaking(1000.0, 44100.0, 0.0, 1.0);
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

// -- Sample Rate Conversion --

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

// -- Utility --

#[test]
fn test_rms_sine() {
    let wave = gen_sine(440.0, 44100.0, 44100);
    let r = rms(&wave);
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
    assert!(approx_eq(signal[0], 10.0_f64.powf(6.0 / 20.0), 0.01));
}

#[test]
fn test_apply_gain_db_negative() {
    let mut signal = vec![1.0];
    apply_gain_db(&mut signal, -20.0);
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
    assert!(s2[0] > s1[0]);
}

// -- Integration --

#[test]
fn test_fft_sine_detection() {
    let n = 1024;
    let sr = 44100.0;
    let freq = 1000.0;
    let sine = gen_sine(freq, sr, n);
    let mut data: Vec<Complex> = sine.iter().map(|&s| Complex::new(s, 0.0)).collect();
    fft(&mut data);
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
    let mut output = Vec::new();
    output.push(delay.process(1.0));
    for _ in 0..12 {
        output.push(delay.process(0.0));
    }
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
