[English](README.md) | **日本語**

# ALICE-Audio

[Project A.L.I.C.E.](https://github.com/anthropics/alice) の汎用オーディオ処理ライブラリ

## 概要

`alice-audio` は純Rustによる包括的なオーディオ処理ツールキットです。FFT、フィルタ、ミキシング、エフェクト、波形生成、サンプルレート変換などを提供します。

## 機能

- **FFT** — Cooley-Tukey基数2 DIT（インプレース、2の冪長）
- **逆FFT** — 周波数領域から時間領域への復元
- **FIR/IIRフィルタ** — 設定可能なデジタルフィルタ
- **マルチチャンネルミキサー** — ゲイン制御付きチャンネルミキシング
- **エフェクト** — リバーブ、ディレイ、コーラス、パラメトリックEQ
- **波形生成** — サイン波、矩形波、ノコギリ波、三角波、ホワイトノイズ
- **サンプルレート変換** — 補間によるリサンプリング
- **ADSRエンベロープ** — アタック/ディケイ/サスティン/リリース包絡線生成
- **リングバッファ** — リアルタイムオーディオ用循環バッファ

## クイックスタート

```rust
use alice_audio::{Complex, fft};

let mut data: Vec<Complex> = (0..1024)
    .map(|i| Complex::new((i as f64 * 0.1).sin(), 0.0))
    .collect();
fft(&mut data);
```

## アーキテクチャ

```
alice-audio
├── Complex        — 最小複素数型
├── fft / ifft     — Cooley-Tukey FFT & 逆変換
├── filter         — FIR & IIRデジタルフィルタ
├── mixer          — マルチチャンネルオーディオミキサー
├── effects        — リバーブ、ディレイ、コーラス、EQ
├── waveform       — オシレータ/波形ジェネレータ
├── resample       — サンプルレート変換
├── envelope       — ADSRエンベロープ生成
└── ring_buffer    — ストリーミング用循環バッファ
```

## ライセンス

MIT OR Apache-2.0
