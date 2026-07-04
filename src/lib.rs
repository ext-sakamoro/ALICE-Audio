#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::struct_field_names
)]

//! ALICE-Audio: General-purpose audio processing library.
//!
//! Provides FFT (Cooley-Tukey), FIR/IIR filters, multi-channel mixer,
//! effects (reverb, delay, chorus, EQ), waveform generation,
//! sample rate conversion, ADSR envelope, and ring buffer.

pub mod adsr;
pub mod complex;
pub mod effects;
pub mod fir;
pub mod iir;
pub mod mixer;
pub mod prelude;
pub mod resample;
pub mod ring_buffer;
pub mod utility;
pub mod waveform;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::adsr::*;
pub use crate::complex::*;
pub use crate::effects::*;
pub use crate::fir::*;
pub use crate::iir::*;
pub use crate::mixer::*;
pub use crate::resample::*;
pub use crate::ring_buffer::*;
pub use crate::utility::*;
pub use crate::waveform::*;
