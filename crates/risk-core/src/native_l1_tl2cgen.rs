//! Native L1 predictor compiled from XGBoost model via TL2cgen (successor of Treelite).
//!
//! The generated C sources must be placed under:
//!   crates/risk-core/native/tl2cgen/ieee_l1/
//! and will be compiled & statically linked when the `native_l1_tl2cgen` feature is enabled.
//!
//! ⚠️ Important: TL2cgen may generate *different* C signatures depending on options
//! (e.g. quantize / multiclass / multi-target). Always match the generated `header.h`.
//! In our IEEE L1 export (quantize=1 by default), the generated API is:
//!
//!   - int32_t get_num_feature(void);
//!   - int32_t get_num_target(void);
//!   - void    get_num_class(int32_t* out);              // out len = N_TARGET
//!   - void    predict(union Entry* data, int pred_margin, float* result);
//!   - void    postprocess(float* result);
//!
//! Where `Entry` is a union with fields `missing` (int32), `fvalue` (float32), and `qvalue` (int32).
//! `missing == -1` indicates a missing value; otherwise the value is present.

use anyhow::{anyhow, ensure, Result};
use std::cell::RefCell;
use std::ptr;
use std::sync::OnceLock;

#[repr(C)]
#[derive(Copy, Clone)]
pub union Entry {
    pub missing: i32,
    pub fvalue: f32,
    /// Present in some TL2cgen outputs (e.g. quantized predictors). We keep it to match C layout.
    pub qvalue: i32,
}

extern "C" {
    fn get_num_feature() -> i32;
    fn get_num_target() -> i32;
    fn get_num_class(out: *mut i32);

    fn predict(data: *mut Entry, pred_margin: i32, result: *mut f32);
    fn postprocess(result: *mut f32);
}

static NUM_FEATURE: OnceLock<usize> = OnceLock::new();
static OUT_LEN: OnceLock<usize> = OnceLock::new();

#[inline(always)]
pub fn num_feature() -> usize {
    *NUM_FEATURE.get_or_init(|| {
        let n = unsafe { get_num_feature() };
        // TL2cgen uses int32_t for this.
        usize::try_from(n.max(0) as i64).unwrap_or(0)
    })
}

#[inline(always)]
fn out_len() -> usize {
    *OUT_LEN.get_or_init(|| unsafe {
        let nt = get_num_target().max(1) as usize;
        let mut cls = vec![1i32; nt];
        get_num_class(cls.as_mut_ptr());
        let max_cls = cls.iter().copied().max().unwrap_or(1).max(1) as usize;
        nt * max_cls
    })
}

thread_local! {
    static TLS_ENTRY_BUF: RefCell<Vec<Entry>> = RefCell::new(Vec::new());
    static TLS_OUT_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    // For the unaligned-bytes path: record missing indices, then patch them in a tight loop.
    static TLS_MISS_IDX: RefCell<Vec<u16>> = RefCell::new(Vec::new());
}

#[cold]
#[inline(never)]
fn set_missing(entry_ptr: *mut Entry, idx: usize) {
    // SAFETY: caller guarantees idx in-bounds.
    unsafe {
        *entry_ptr.add(idx) = Entry { missing: -1 };
    }
}

#[inline(always)]
fn ensure_len_entry(buf: &mut Vec<Entry>, n: usize) {
    if buf.len() != n {
        // Initialize to present values (any non -1 int is treated as present).
        *buf = vec![Entry { missing: 0 }; n];
    }
}

#[inline(always)]
fn ensure_len_out(out: &mut Vec<f32>) {
    let need = out_len();
    if out.len() != need {
        *out = vec![0.0; need];
    }
}

/// Predict probability (or score) for a single dense row.
///
/// Input semantics follow existing pipeline:
/// - `row[i] == NaN` represents missing value.
///
/// Micro-optimizations:
/// - Pass 1: bulk-copy all `f32` into the `Entry` buffer (contiguous write).
/// - Pass 2: only patch NaN positions to `missing = -1` (cold path).
#[inline]
pub fn predict_proba_dense_1row(row: &[f32]) -> Result<f32> {
    let n = num_feature();
    ensure!(
        row.len() == n,
        "tl2cgen num_feature mismatch: got row_len={} expect={}",
        row.len(),
        n
    );

    TLS_ENTRY_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        ensure_len_entry(&mut buf, n);

        // Pass 1: bulk write f32 into Entry storage.
        // SAFETY: Entry is a 4-byte C union; writing f32 bits into it is ABI-correct.
        unsafe {
            let dst = buf.as_mut_ptr() as *mut f32;
            ptr::copy_nonoverlapping(row.as_ptr(), dst, n);
        }

        // Pass 2: patch missing positions (NaN => missing = -1).
        let entry_ptr = buf.as_mut_ptr();
        for (i, &x) in row.iter().enumerate() {
            if x.is_nan() {
                set_missing(entry_ptr, i);
            }
        }

        TLS_OUT_BUF.with(|out_cell| {
            let mut out = out_cell.borrow_mut();
            ensure_len_out(&mut out);

            // pred_margin=0 => follow standard Treelite/TL2cgen semantics (final transform applied).
            unsafe {
                predict(entry_ptr, 0, out.as_mut_ptr());
                // If your generated code requires explicit postprocess, enable this:
                // postprocess(out.as_mut_ptr());
                let _ = postprocess as usize; // keep symbol linked even if not called.
            }

            let p = out[0];
            if p.is_finite() {
                Ok(p)
            } else {
                Err(anyhow!("tl2cgen returned non-finite prediction: {p}"))
            }
        })
    })
}

/// Predict probability for a single dense row, directly from f32 little-endian bytes.
///
/// This is the hot path used by `/score_dense_f32_bin`:
/// - Fast path: if `bytes` is properly aligned, reinterpret as `&[f32]` (zero-copy) and use the
///   same two-pass fill as `predict_proba_dense_1row`.
/// - Fallback: decode into the `Entry` buffer directly (no intermediate `Vec<f32>`), record missing
///   indices, then patch them.
#[inline]
pub fn predict_proba_dense_bytes_le(bytes: &[u8], ncols: usize) -> Result<f32> {
    let n = num_feature();
    ensure!(
        ncols == n,
        "tl2cgen num_feature mismatch: got ncols={} expect={}",
        ncols,
        n
    );

    let b = bytes;
    let need = n.checked_mul(4).ok_or_else(|| anyhow!("ncols too large"))?;
    ensure!(
        b.len() == need,
        "dense bytes size mismatch: got {}, need {}",
        b.len(),
        need
    );

    TLS_ENTRY_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        ensure_len_entry(&mut buf, n);
        let entry_ptr = buf.as_mut_ptr();

        // 4.1 alignment fast-path (little-endian only)
        if cfg!(target_endian = "little") {
            // SAFETY: we accept the aligned case only.
            let (head, body, tail) = unsafe { b.align_to::<f32>() };
            if head.is_empty() && tail.is_empty() && body.len() == n {
                // Pass 1: bulk-copy
                unsafe {
                    let dst = entry_ptr as *mut f32;
                    ptr::copy_nonoverlapping(body.as_ptr(), dst, n);
                }

                // Pass 2: patch NaNs
                for (i, &x) in body.iter().enumerate() {
                    if x.is_nan() {
                        set_missing(entry_ptr, i);
                    }
                }

                return TLS_OUT_BUF.with(|out_cell| {
                    let mut out = out_cell.borrow_mut();
                    ensure_len_out(&mut out);
                    unsafe {
                        predict(entry_ptr, 0, out.as_mut_ptr());
                        let _ = postprocess as usize;
                    }
                    let p = out[0];
                    if p.is_finite() {
                        Ok(p)
                    } else {
                        Err(anyhow!("tl2cgen returned non-finite prediction: {p}"))
                    }
                });
            }
        }

        // 4.1 fallback: unaligned bytes => decode + fill Entry directly.
        TLS_MISS_IDX.with(|miss_cell| {
            let mut miss = miss_cell.borrow_mut();
            miss.clear();

            // Pass 1: decode & write f32 values contiguously into Entry storage.
            // SAFETY: Entry is 4 bytes; we write f32 bits into it.
            let dst = entry_ptr as *mut f32;
            for (i, c) in b.chunks_exact(4).enumerate() {
                let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                unsafe { *dst.add(i) = x };
                if x.is_nan() {
                    // n is small (432); u16 is enough and keeps the vector tight.
                    miss.push(i as u16);
                }
            }

            // Pass 2: patch missing positions.
            for &i in miss.iter() {
                set_missing(entry_ptr, i as usize);
            }

            TLS_OUT_BUF.with(|out_cell| {
                let mut out = out_cell.borrow_mut();
                ensure_len_out(&mut out);
                unsafe {
                    predict(entry_ptr, 0, out.as_mut_ptr());
                    let _ = postprocess as usize;
                }
                let p = out[0];
                if p.is_finite() {
                    Ok(p)
                } else {
                    Err(anyhow!("tl2cgen returned non-finite prediction: {p}"))
                }
            })
        })
    })
}
