//! The output bits of every public function on fixed inputs, pinned: the same pins must hold on every
//! target CI builds (Linux, macOS, Windows; x86_64 and aarch64). The arithmetic goes through
//! alice-det-math and clippy.toml bans the platform transcendentals, so a target that computes
//! different bits is a regression, not noise.
//!
//! The inputs are built without any transcendental: the crate's own LCG noise (integer arithmetic)
//! and impulses. Each pin is FNV-1a 64 over the little-endian bytes of the output's `f64` bits.
//! `ALICE_AUDIO_PRINT_PINS=1 cargo test --test cross_target_bits -- --nocapture` prints every pin.
//!
//! Two inputs are chosen so that a platform `sin` / `cos` put back in place of det-math moves a pin
//! on a target whose libm gives other bits there (found by comparing `f64::sin` / `f64::cos` with
//! `sin64` / `cos64` on the arguments these functions form, measured on aarch64 macOS only; Linux and
//! Windows were not measured, so these two inputs are known to catch the change on the macOS lane):
//! - `fir_low_pass_order_55`: the window `cos(2π i / 55)` differs in the last bit at 6 of its 56 terms
//!   (the other FIR pins use orders 31 and 63, where no term, or one that rounds away, differs);
//! - `chorus_lfo_edge`: at rate 1.5 Hz the lfo `sin(2π·phase)` differs in the last bit at 135 of the
//!   first 4096 samples, but the delay is `(depth·lfo + depth) as usize`, which a last-bit change
//!   rarely moves; with `depth = f64::from_bits(0x3ffb_4be0_9a58_bf2f)` (≈ 1.706) the delay at sample
//!   882 is 1 with det-math and 2 with that libm.

use alice_audio::{
    fft, gen_noise, gen_sawtooth, gen_sine, gen_square, ifft, mix_channels, pan_stereo, resample,
    utility, Adsr, AllPassFilter, Chorus, CombFilter, Complex, Delay, EqBand, Equalizer, FirFilter,
    IirFilter, Reverb,
};

const N: usize = 4096;
const SR: f64 = 48_000.0;

/// FNV-1a 64 over the little-endian bytes of each value's bits
fn fold(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn noise() -> Vec<f64> {
    gen_noise(N, 7)
}

fn impulse() -> Vec<f64> {
    let mut x = vec![0.0; N];
    x[0] = 1.0;
    x
}

/// (name, measured output bits) for every case
fn cases() -> Vec<(&'static str, u64)> {
    let x = noise();
    let mut out = vec![
        ("gen_sine", fold(gen_sine(441.0, SR, N))),
        ("gen_square", fold(gen_square(441.0, SR, N))),
        ("gen_sawtooth", fold(gen_sawtooth(441.0, SR, N))),
        ("gen_noise", fold(x.clone())),
        (
            "iir_low_pass",
            fold(IirFilter::low_pass(1_000.0, SR, 0.707).process_buffer(&x)),
        ),
        (
            "iir_high_pass",
            fold(IirFilter::high_pass(1_000.0, SR, 0.707).process_buffer(&x)),
        ),
        (
            "iir_band_pass",
            fold(IirFilter::band_pass(1_000.0, SR, 2.0).process_buffer(&x)),
        ),
        (
            "iir_low_pass_impulse",
            fold(IirFilter::low_pass(3_000.0, SR, 0.707).process_buffer(&impulse())),
        ),
        (
            "fir_low_pass",
            fold(FirFilter::low_pass(1_000.0, SR, 63).process_buffer(&x)),
        ),
        (
            "eq_peaking",
            fold(EqBand::peaking(1_000.0, SR, 6.0, 0.7).process_buffer(&x)),
        ),
        (
            "equalizer",
            fold(
                Equalizer::new(vec![
                    EqBand::peaking(200.0, SR, -3.0, 0.7),
                    EqBand::peaking(4_000.0, SR, 4.5, 1.2),
                ])
                .process_buffer(&x),
            ),
        ),
        ("delay", fold(Delay::new(480, 0.5, 0.5).process_buffer(&x))),
        (
            "delay_clamped",
            fold(Delay::new(480, 1.7, -0.25).process_buffer(&x)),
        ),
        ("reverb", fold(Reverb::new(SR, 0.4).process_buffer(&x))),
        (
            "comb",
            fold(CombFilter::new(1_103, 0.84).process_buffer(&x)),
        ),
        (
            "all_pass",
            fold(AllPassFilter::new(556, 0.5).process_buffer(&x)),
        ),
        (
            "chorus",
            fold(Chorus::new(SR, 1.5, 24.0, 0.5).process_buffer(&x)),
        ),
        (
            "chorus_lfo_edge",
            fold(
                Chorus::new(SR, 1.5, f64::from_bits(0x3ffb_4be0_9a58_bf2f), 0.5).process_buffer(&x),
            ),
        ),
        (
            "fir_low_pass_order_55",
            fold(FirFilter::low_pass(1_000.0, SR, 55).process_buffer(&x)),
        ),
        ("resample", fold(resample(&x, SR, 44_100.0))),
        (
            "adsr",
            fold(
                Adsr {
                    attack: 480,
                    decay: 960,
                    sustain_level: 0.6,
                    release: 2_400,
                }
                .generate(4_800),
            ),
        ),
    ];
    // the FFT, its inverse and the polar form of each bin
    let mut spec: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft(&mut spec);
    out.push(("fft", fold(spec.iter().flat_map(|c| [c.re, c.im]))));
    out.push(("fft_magnitude", fold(spec.iter().map(|c| c.magnitude()))));
    out.push(("fft_phase", fold(spec.iter().map(|c| c.phase()))));
    ifft(&mut spec);
    out.push(("ifft", fold(spec.iter().flat_map(|c| [c.re, c.im]))));
    // utility
    out.push(("rms", fold([utility::rms(&x)])));
    out.push(("peak", fold([utility::peak(&x)])));
    let mut y = x.clone();
    utility::normalize(&mut y);
    out.push(("normalize", fold(y)));
    let mut y = x.clone();
    utility::apply_gain_db(&mut y, -6.0);
    out.push(("apply_gain_db", fold(y)));
    let mut y = x.clone();
    utility::soft_clip(&mut y, 2.5);
    out.push(("soft_clip", fold(y)));
    let mut y = x.clone();
    utility::hard_clip(&mut y, 0.5);
    out.push(("hard_clip", fold(y)));
    // mixer
    let (l, r) = pan_stereo(&x, 0.3);
    out.push(("pan_stereo", fold(l.into_iter().chain(r))));
    let (l, r) = pan_stereo(&x, 1.8);
    out.push(("pan_stereo_clamped", fold(l.into_iter().chain(r))));
    let s = gen_sine(441.0, SR, N);
    out.push(("mix_channels", fold(mix_channels(&[&x, &s], &[0.7, 0.3]))));
    // design sweeps: many cutoffs, gains and rates, so a last-bit difference in one evaluation of a
    // transcendental reaches a pin (one cutoff alone rarely shows it)
    let imp = impulse();
    let cutoffs: Vec<f64> = (0..512).map(|i| 20.0 + 37.5 * i as f64).collect();
    let qs = [0.5, 0.707, 1.0, 4.0];
    for (name, design) in [
        (
            "iir_low_pass_sweep",
            IirFilter::low_pass as fn(f64, f64, f64) -> IirFilter,
        ),
        ("iir_high_pass_sweep", IirFilter::high_pass),
        ("iir_band_pass_sweep", IirFilter::band_pass),
    ] {
        let bits = cutoffs.iter().enumerate().flat_map(|(i, &fc)| {
            let mut f = design(fc, SR, qs[i % qs.len()]);
            f.process_buffer(&imp[..16])
        });
        out.push((name, fold(bits.collect::<Vec<_>>())));
    }
    let eq = cutoffs.iter().enumerate().flat_map(|(i, &fc)| {
        EqBand::peaking(fc, SR, -12.0 + 0.05 * i as f64, qs[i % qs.len()])
            .process_buffer(&imp[..16])
    });
    out.push(("eq_peaking_sweep", fold(eq.collect::<Vec<_>>())));
    let fir = (0..64).flat_map(|i| {
        FirFilter::low_pass(100.0 + 350.0 * f64::from(i), SR, 31).process_buffer(&imp[..32])
    });
    out.push(("fir_low_pass_sweep", fold(fir.collect::<Vec<_>>())));
    let gains = (0..512).flat_map(|i| {
        let mut y = vec![0.75, -0.3];
        utility::apply_gain_db(&mut y, -60.0 + 0.2 * f64::from(i));
        y
    });
    out.push(("apply_gain_db_sweep", fold(gains.collect::<Vec<_>>())));
    let chorus = [0.25, 1.5, 7.0]
        .into_iter()
        .flat_map(|rate| Chorus::new(SR, rate, 24.0, 0.5).process_buffer(&x));
    out.push(("chorus_rates", fold(chorus.collect::<Vec<_>>())));
    out
}

/// The recorded pins (from `ALICE_AUDIO_PRINT_PINS=1`; debug and release print the same)
const PINS: &[(&str, u64)] = &[
    ("gen_sine", 0x75d48b3e614196c4),
    ("gen_square", 0x470db53e16fa5ba5),
    ("gen_sawtooth", 0x5d31cb714420ebe6),
    ("gen_noise", 0x9df7f6d797992d56),
    ("iir_low_pass", 0x8fb30bec2b7636c3),
    ("iir_high_pass", 0xf8a25a00ef53a55a),
    ("iir_band_pass", 0xf41910b58fea9582),
    ("iir_low_pass_impulse", 0x2a936b3684e91be5),
    ("fir_low_pass", 0xe27f650bf45a8d09),
    ("eq_peaking", 0x3c262def38f19972),
    ("equalizer", 0x32833cea08f9acfe),
    ("delay", 0x2301ff8c90a344f9),
    ("delay_clamped", 0x9df7f6d797992d56),
    ("reverb", 0x59d3465a206aa624),
    ("comb", 0xdcf2b42fc28d179b),
    ("all_pass", 0xfe1c70b6b0711121),
    ("chorus", 0xdffafe03dcbd8caa),
    ("chorus_lfo_edge", 0xabc5733282570ad8),
    ("fir_low_pass_order_55", 0x5fe48fab89ebe759),
    ("resample", 0xd6953a2e57df7439),
    ("adsr", 0x6b7c29a184a6613c),
    ("fft", 0xfab9eac9d6365caf),
    ("fft_magnitude", 0x85ce22b550b14323),
    ("fft_phase", 0x9834202818048e9b),
    ("ifft", 0xa9b3d52a8a810c44),
    ("rms", 0xe2e47cb9c006ff99),
    ("peak", 0xb540ea9c10162178),
    ("normalize", 0x735aedc6dd17f07a),
    ("apply_gain_db", 0x7c91d4156dc69cc6),
    ("soft_clip", 0x03b3eeb86d6cc95c),
    ("hard_clip", 0xef63c3fe4a6c8423),
    ("pan_stereo", 0xbe3506d78ed06bac),
    ("pan_stereo_clamped", 0x49abea2a42aa8056),
    ("mix_channels", 0xe00e375cce176f13),
    ("iir_low_pass_sweep", 0xc8ee0610b24268be),
    ("iir_high_pass_sweep", 0xc41f2aad1726d90d),
    ("iir_band_pass_sweep", 0x59175b461cd9841f),
    ("eq_peaking_sweep", 0xf95fa6a457e1374b),
    ("fir_low_pass_sweep", 0x9a98aac5e9366827),
    ("apply_gain_db_sweep", 0x2f79c559d2260373),
    ("chorus_rates", 0x0606a7a2b77cadf3),
];

#[test]
fn every_output_matches_its_pin_on_this_target() {
    let got = cases();
    if std::env::var_os("ALICE_AUDIO_PRINT_PINS").is_some() {
        for (name, h) in &got {
            println!("    (\"{name}\", 0x{h:016x}),");
        }
        return;
    }
    assert!(!got.is_empty(), "no case computed");
    let mut compared = 0;
    for (name, h) in &got {
        let want = PINS
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("`{name}` has no pin"));
        assert_eq!(
            *h, want.1,
            "`{name}`: output bits 0x{h:016x}, pinned 0x{:016x}",
            want.1
        );
        compared += 1;
    }
    assert_eq!(compared, PINS.len(), "a pin names no case");
    // CI greps this line on every target (a lane that ran no comparison has no such line)
    println!("pins compared: {compared}");
}
