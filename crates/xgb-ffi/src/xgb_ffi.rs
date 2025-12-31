//! Minimal, fast XGBoost C-API bindings for inference.
//!
//! Goals:
//! - Safe-ish wrappers around XGBoost C-API
//! - A reusable `__array_interface__` buffer for 1-row dense inputs
//! - Helper methods used by `risk-core` (`predict_proba_dense_1row`, `predict_contribs_dense_1row`)

use anyhow::{anyhow, bail, Context};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint, c_void};
use std::path::Path;
use std::ptr;

pub type BstUlong = u64;
pub type DMatrixHandle = *mut c_void;
pub type BoosterHandle = *mut c_void;

extern "C" {
    fn XGBGetLastError() -> *const c_char;

    fn XGBoosterCreate(dmats: *const DMatrixHandle, len: BstUlong, out: *mut BoosterHandle) -> c_uint;
    fn XGBoosterFree(handle: BoosterHandle) -> c_uint;
    fn XGBoosterLoadModel(handle: BoosterHandle, fname: *const c_char) -> c_uint;

    fn XGBoosterPredictFromDense(
        handle: BoosterHandle,
        array_interface: *const c_char,
        config: *const c_char,
        out_shape: *mut *const BstUlong,
        out_dim: *mut BstUlong,
        out_result: *mut *const f32,
    ) -> c_uint;
}

fn last_error() -> anyhow::Error {
    let p = unsafe { XGBGetLastError() };
    if p.is_null() {
        anyhow!("xgboost: unknown error (XGBGetLastError returned null)")
    } else {
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        anyhow!("xgboost: {s}")
    }
}

fn check(rc: c_uint) -> anyhow::Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}

/// Output of a predict call.
#[derive(Debug, Clone)]
pub struct PredictOutput {
    pub shape: Vec<BstUlong>,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct Booster {
    handle: BoosterHandle,
}

// Not safe for concurrent calls across threads; OK to move to a worker thread.
unsafe impl Send for Booster {}

impl Drop for Booster {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { XGBoosterFree(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl Booster {
    /// Create an empty booster and load model from file.
    pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
        let mut handle: BoosterHandle = ptr::null_mut();
        unsafe {
            check(XGBoosterCreate(ptr::null(), 0, &mut handle)).context("XGBoosterCreate")?;
        }
        if handle.is_null() {
            bail!("xgboost: XGBoosterCreate returned null handle");
        }

        let cpath = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow!("model path is not valid UTF-8: {path:?}"))?,
        )?;

        unsafe { check(XGBoosterLoadModel(handle, cpath.as_ptr())).context("XGBoosterLoadModel")? };

        Ok(Self { handle })
    }

    /// Back-compat alias used by older `risk-core` code.
    pub fn load_model(path: &Path) -> Result<Self, String> {
        Self::load_from_file(path).map_err(|e| e.to_string())
    }

    /// Generic dense prediction entry point.
    pub fn predict_from_dense(&self, array_interface: &CStr, config: &CStr) -> anyhow::Result<PredictOutput> {
        let mut out_shape_ptr: *const BstUlong = ptr::null();
        let mut out_dim: BstUlong = 0;
        let mut out_result_ptr: *const f32 = ptr::null();

        unsafe {
            check(XGBoosterPredictFromDense(
                self.handle,
                array_interface.as_ptr(),
                config.as_ptr(),
                &mut out_shape_ptr,
                &mut out_dim,
                &mut out_result_ptr,
            ))
            .context("XGBoosterPredictFromDense")?;
        }

        let shape = if out_dim == 0 || out_shape_ptr.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(out_shape_ptr, out_dim as usize) }.to_vec()
        };

        let n = shape
            .iter()
            .copied()
            .fold(1u64, |acc, v| acc.saturating_mul(v))
            as usize;

        let values = if n == 0 || out_result_ptr.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(out_result_ptr, n) }.to_vec()
        };

        Ok(PredictOutput { shape, values })
    }

    /// Fast helper: predict probability for a single dense row.
    ///
    /// Uses per-thread cached `DenseArrayInterface` + config.
    pub fn predict_proba_dense_1row(&self, row: &[f32]) -> Result<f32, String> {
        TLS_DENSE.with(|cell| {
            let mut tls = cell.borrow_mut();
            if tls.as_ref().map(|x| x.ncols) != Some(row.len()) {
                *tls = Some(TlsDense::new(row.len())?);
            }
            let tls = tls.as_mut().expect("initialized");
            tls.ai.data_mut().copy_from_slice(row);
            let out = self.predict_from_dense(tls.ai.as_cstr(), tls.cfg_pred.as_c_str())?;
            Ok(out.values.get(0).copied().unwrap_or(0.0))
        }).map_err(|e| e.to_string())
    }

    /// Fast helper: predict contribution vector for a single dense row.
    ///
    /// `values` length is typically `ncols + 1` (last element = bias).
    pub fn predict_contribs_dense_1row(&self, row: &[f32]) -> Result<PredictOutput, String> {
        TLS_DENSE.with(|cell| {
            let mut tls = cell.borrow_mut();
            if tls.as_ref().map(|x| x.ncols) != Some(row.len()) {
                *tls = Some(TlsDense::new(row.len())?);
            }
            let tls = tls.as_mut().expect("initialized");
            tls.ai.data_mut().copy_from_slice(row);
            self.predict_from_dense(tls.ai.as_cstr(), tls.cfg_contrib.as_c_str())
        }).map_err(|e| e.to_string())
    }
}

/// A reusable dense `__array_interface__` for **one row**.
#[derive(Debug)]
pub struct DenseArrayInterface {
    data: Vec<f32>,
    cstr: CString,
}

impl DenseArrayInterface {
    pub fn new_1row(ncols: usize) -> anyhow::Result<Self> {
        let mut data = vec![0.0f32; ncols];
        // Pointer is stable as long as `data` is not reallocated (length fixed).
        let ptr = data.as_mut_ptr() as usize;

        // Numpy-style array interface (version 3). Little-endian f32.
        // data: [ptr, read_only]
        // shape: [rows, cols]
        let s = format!(
            r#"{{"data":[{},false],"shape":[1,{}],"typestr":"<f4","version":3}}"#,
            ptr, ncols
        );
        let cstr = CString::new(s)?;

        Ok(Self { data, cstr })
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    pub fn as_cstr(&self) -> &CStr {
        &self.cstr
    }
}

struct TlsDense {
    ncols: usize,
    ai: DenseArrayInterface,
    cfg_pred: CString,
    cfg_contrib: CString,
}

impl TlsDense {
    fn new(ncols: usize) -> anyhow::Result<Self> {
        let ai = DenseArrayInterface::new_1row(ncols)?;
        let cfg_pred = CString::new(
            r#"{"training":false,"type":0,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#,
        )?;
        let cfg_contrib = CString::new(
            r#"{"training":false,"type":0,"iteration_begin":0,"iteration_end":0,"strict_shape":false,"pred_contribs":true}"#,
        )?;
        Ok(Self {
            ncols,
            ai,
            cfg_pred,
            cfg_contrib,
        })
    }
}

thread_local! {
    static TLS_DENSE: RefCell<Option<TlsDense>> = const { RefCell::new(None) };
}
