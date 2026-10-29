#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

need actionlint "brew install actionlint"

step "ci.yml / fmt: Check formatting"
( export CARGO_TERM_COLOR="always"; cargo fmt -- --check )

step "ci.yml / test: Clippy"
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / test: Test"
( export CARGO_TERM_COLOR="always"; cargo test )

step "ci.yml / test: Analytic oracle"
( export CARGO_TERM_COLOR="always"; cargo test --test analytic_oracle )
step "ci.yml / test: Cross-target bit pins"
( export CARGO_TERM_COLOR="always"
  for profile in "" "--release"; do
    out=$(cargo test $profile --test cross_target_bits -- --nocapture 2>&1) || { echo "$out"; exit 1; }
    echo "$out" | grep -E '^test |pins compared'
    echo "$out" | grep -qE 'pins compared: [1-9][0-9]*[[:space:]]*$' || { echo "no pins compared" >&2; exit 1; }
  done )

step "ci.yml / wasm: Cross-target bit pins via wasmtime"
need wasmtime "brew install wasmtime"
rustup target list --installed | grep -q '^wasm32-wasip1$' || rustup target add wasm32-wasip1
( export CARGO_TERM_COLOR="always" CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime"
  for flag in "" "--release"; do
    out=$(cargo test $flag --target wasm32-wasip1 --test cross_target_bits -- --nocapture 2>&1) || { echo "$out"; exit 1; }
    echo "$out" | grep -E 'Running|^test |pins compared'
    echo "$out" | grep -qE 'pins compared: [1-9][0-9]*[[:space:]]*$' || { echo "no pins compared ($flag)" >&2; exit 1; }
  done )

echo; echo "preflight OK"
