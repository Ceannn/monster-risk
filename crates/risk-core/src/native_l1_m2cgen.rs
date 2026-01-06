// Auto-generated L1 model (m2cgen) glue.
//
// This module is only compiled when the `native_l1_m2cgen` feature is enabled.
//
// The actual model code lives in `native_l1_m2cgen_gen.rs` which should be
// generated from `models/ieee_l1` using `scripts/ml/export_l1_m2cgen_rust.py`.
//
// We keep the generated code as a plain Rust source file so we don't need any
// build-time Python dependency in `cargo` (generation is an explicit step).

#![allow(clippy::all)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

mod gen {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/native_l1_m2cgen_gen.rs"));
}

pub use gen::{score, OUTPUT_DIM};

#[inline(always)]
pub fn proba_pos(out: &[f64; OUTPUT_DIM]) -> f64 {
    // For binary classification, m2cgen typically emits 2-class probabilities.
    // We treat index 1 as the positive class.
    if OUTPUT_DIM >= 2 { out[1] } else { out[0] }
}
