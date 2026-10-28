# Changelog

All notable changes to this crate are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking (2.0.0):** 出力の bit が変わる 超越関数と `f64` の max / clamp を全て `alice-det-math` 0.6.1 に通した (IIR の low / high / band-pass と EQ の peaking の係数設計の sin / cos / powf、FIR の窓関数付き sinc の sin / cos、FFT の回転因子の sin / cos、`Complex::magnitude` の hypot と `Complex::phase` の atan2、chorus の LFO の sin、`apply_gain_db` の powf、`soft_clip` の tanh、`gen_sine` の sin、`peak` の max、delay / reverb / chorus の feedback・mix、`pan_stereo` の pan、`hard_clip` の clamp) これで出力は全ての target (Linux・macOS・Windows・wasm32) で入力の bit だけで決まる 以前は platform の libm で、target によって最後の bit が違った 値の差は各関数の誤差 (≤ 1〜2 ulp) の範囲で、既存の試験 (114 件) は全て許容の中で通る
- **Breaking (2.0.0):** `hard_clip` の `threshold` が負 (下限 > 上限) でも panic しない (G2 の clamp: 上限が勝つ) 以前の `f64::clamp` は panic した delay / reverb / chorus の `feedback` / `mix` と `pan_stereo` の `pan` も同じ規則で NaN を境界にしない

### Added

- `clippy.toml`: platform の超越関数 (`f64` / `f32` の sin、cos、tan、asin、acos、atan、atan2、sinh、cosh、tanh、exp、exp2、exp_m1、ln、ln_1p、log、log2、log10、powf、powi、cbrt、hypot) と max / min / clamp を禁止 (`cargo clippy --all-targets -- -D warnings` で検出、独立な参照値を platform の libm で計算する `tests/analytic_oracle.rs` だけは理由つきで許可)
- `tests/cross_target_bits.rs`: 全ての公開関数の出力の bit を固定入力 (crate 自身の LCG の noise と impulse、超越関数を使わない) で 39 件 pin (FNV-1a 64) 係数の設計は cutoff・Q・gain を掃引して、1 点では出ない最後の bit の差も pin に届くようにした CI の test job を Linux・macOS・Windows の matrix にし、debug と release で同じ pin を確かめる (preflight も同じ step)
- 依存: `alice-det-math` 0.6.1 (git rev、`std`)
