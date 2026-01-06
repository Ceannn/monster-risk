// This file must be generated from the XGBoost L1 model via m2cgen.
//
// Run:
//
//   python3 scripts/ml/export_l1_m2cgen_rust.py \
/*       --model-dir models/ieee_l1 \
       --out crates/risk-core/src/native_l1_m2cgen_gen.rs */

//
// After generation, this file will contain a fast, allocation-free Rust scorer
// with the following API:
//
//   pub const OUTPUT_DIM: usize = 2;
//   pub fn score(input: &[f64], out: &mut [f64; OUTPUT_DIM]);
//
compile_error!("native_l1_m2cgen is enabled, but native_l1_m2cgen_gen.rs has not been generated yet. See the header comment for the generation command.");
