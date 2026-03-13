**English** | [日本語](README_JP.md)

# ALICE-Audio

General-purpose audio processing library for [Project A.L.I.C.E.](https://github.com/anthropics/alice)

## Overview

`alice-audio` provides a comprehensive audio processing toolkit in pure Rust — FFT, filters, mixing, effects, waveform generation, sample rate conversion, and more.

## Features

- **FFT** — Cooley-Tukey radix-2 DIT (in-place, power-of-two)
- **Inverse FFT** — frequency-domain to time-domain reconstruction
- **FIR/IIR Filters** — configurable digital filter implementations
- **Multi-channel Mixer** — channel mixing with gain control
- **Effects** — reverb, delay, chorus, parametric EQ
- **Waveform Generation** — sine, square, sawtooth, triangle, white noise
- **Sample Rate Conversion** — resampling with interpolation
- **ADSR Envelope** — attack/decay/sustain/release envelope generator
- **Ring Buffer** — lock-free circular buffer for real-time audio

## Quick Start

```rust
use alice_audio::{Complex, fft};

let mut data: Vec<Complex> = (0..1024)
    .map(|i| Complex::new((i as f64 * 0.1).sin(), 0.0))
    .collect();
fft(&mut data);
```

## Architecture

```
alice-audio
├── Complex        — minimal complex number type
├── fft / ifft     — Cooley-Tukey FFT & inverse
├── filter         — FIR & IIR digital filters
├── mixer          — multi-channel audio mixer
├── effects        — reverb, delay, chorus, EQ
├── waveform       — oscillator / waveform generators
├── resample       — sample rate conversion
├── envelope       — ADSR envelope generator
└── ring_buffer    — circular buffer for streaming
```

## License

MIT OR Apache-2.0
