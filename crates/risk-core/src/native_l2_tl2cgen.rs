//! Native L2 predictor compiled from XGBoost model via TL2cgen (Treelite successor).
//!
//! Place generated C sources under:
//!   crates/risk-core/native/tl2cgen/ieee_l2/
//! and build with `--features native_l2_tl2cgen`.
//!
//! ⚠️ We prefix the exported C symbols at compile time (see build.rs) to avoid
//! link-time collisions with L1 TL2cgen symbols.

use anyhow::{anyhow, ensure, Result};
use std::cell::RefCell;
use std::mem;
use std::sync::OnceLock;

#[repr(C)]
#[derive(Copy, Clone)]
pub union Entry {
    pub missing: i32,
    pub fvalue: f32,
}

extern "C" {
    #[link_name = "ieee_l2_get_num_feature"]
    fn get_num_feature() -> usize;

    #[link_name = "ieee_l2_get_num_target"]
    fn get_num_target() -> usize;

    // TL2cgen generated signature:
    //   void predict(Entry* data, int pred_margin, float* out_result);
    #[link_name = "ieee_l2_predict"]
    fn predict(data: *const Entry, pred_margin: i32, out_result: *mut f32);

    // TL2cgen generated signature:
    //   size_t postprocess(float* out_result, float* out_pred);
    #[link_name = "ieee_l2_postprocess"]
    fn postprocess(out_result: *mut f32, out_pred: *mut f32) -> usize;
}

static NUM_FEATURE: OnceLock<usize> = OnceLock::new();
static NUM_TARGET: OnceLock<usize> = OnceLock::new();

#[inline(always)]
pub fn num_feature() -> usize {
    *NUM_FEATURE.get_or_init(|| unsafe { get_num_feature() })
}

#[inline(always)]
pub fn num_target() -> usize {
    *NUM_TARGET.get_or_init(|| unsafe { get_num_target() })
}

thread_local! {
    static TLS_ENTRY_BUF: RefCell<Vec<Entry>> = RefCell::new(Vec::new());
    static TLS_OUT_RESULT: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static TLS_OUT_PRED: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static TLS_ALIGN_F32: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// Predict probability for a single dense row.
///
/// Semantics:
/// - `row[i] == NaN` => missing
#[inline]
pub fn predict_proba_dense_1row(row: &[f32]) -> Result<f32> {
    let n = num_feature();
    ensure!(
        row.len() == n,
        "tl2cgen(L2) num_feature mismatch: got row_len={} expect={}",
        row.len(),
        n
    );

    TLS_ENTRY_BUF.with(|entry_cell| {
        TLS_OUT_RESULT.with(|out_cell| {
            TLS_OUT_PRED.with(|pred_cell| {
                let mut entry = entry_cell.borrow_mut();
                if entry.len() != n {
                    // any non -1 value means "present"
                    entry.resize(n, Entry { missing: 0 });
                }

                // pass-1: contiguous stores (cache friendly)
                for (i, &x) in row.iter().enumerate() {
                    entry[i] = Entry { fvalue: x };
                }
                // pass-2: cold path: only flip NaNs to missing
                for (i, &x) in row.iter().enumerate() {
                    if x.is_nan() {
                        entry[i] = Entry { missing: -1 };
                    }
                }

                let nt = num_target().max(1);

                let mut out = out_cell.borrow_mut();
                if out.len() != nt {
                    out.resize(nt, 0.0);
                }

                // pred_margin=0 => probability (for logistic objectives) after postprocess
                unsafe {
                    predict(entry.as_ptr(), 0, out.as_mut_ptr());
                }

                let mut pred = pred_cell.borrow_mut();
                if pred.len() != nt {
                    pred.resize(nt, 0.0);
                }

                let res_len = unsafe { postprocess(out.as_mut_ptr(), pred.as_mut_ptr()) };
                ensure!(res_len >= 1, "tl2cgen(L2) postprocess returned len={res_len}");

                let p = pred[0];
                if p.is_finite() {
                    Ok(p)
                } else {
                    Err(anyhow!("tl2cgen(L2) returned non-finite prediction: {p}"))
                }
            })
        })
    })
}

/// Predict from dense f32 bytes (little-endian), with a zero-copy fast-path when aligned.
#[inline]
pub fn predict_proba_dense_bytes_le(bytes: &[u8], dim: usize) -> Result<f32> {
    ensure!(
        bytes.len() == dim * mem::size_of::<f32>(),
        "tl2cgen(L2) bytes len mismatch: got {} expect {}",
        bytes.len(),
        dim * mem::size_of::<f32>()
    );

    // Fast path: aligned &[f32] view (zero-copy). Only valid on little-endian.
    #[cfg(target_endian = "little")]
    {
        let (head, f32s, tail) = unsafe { bytes.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() {
            return predict_proba_dense_1row(f32s);
        }
    }

    // Slow path: copy/convert into aligned TLS buffer
    TLS_ALIGN_F32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() != dim {
            buf.resize(dim, 0.0);
        }

        for i in 0..dim {
            let off = i * 4;
            let b = <[u8; 4]>::try_from(&bytes[off..off + 4]).unwrap();
            buf[i] = f32::from_le_bytes(b);
        }

        predict_proba_dense_1row(&buf)
    })
}
