use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_void},
    ptr,
};

pub type BstUlong = u64;
pub type DMatrixHandle = *mut c_void;
pub type BoosterHandle = *mut c_void;

extern "C" {
    fn XGBGetLastError() -> *const c_char;

    fn XGDMatrixCreateFromMat(
        data: *const f32,
        nrow: BstUlong,
        ncol: BstUlong,
        missing: f32,
        out: *mut DMatrixHandle,
    ) -> c_int;

    fn XGDMatrixFree(handle: DMatrixHandle) -> c_int;

    fn XGBoosterCreate(
        dmats: *const DMatrixHandle,
        len: BstUlong,
        out: *mut BoosterHandle,
    ) -> c_int;

    fn XGBoosterFree(handle: BoosterHandle) -> c_int;

    fn XGBoosterLoadModel(handle: BoosterHandle, fname: *const c_char) -> c_int;

    fn XGBoosterPredictFromDMatrix(
        handle: BoosterHandle,
        dmat: DMatrixHandle,
        config: *const c_char,
        out_shape: *mut *const BstUlong,
        out_dim: *mut BstUlong,
        out_result: *mut *const f32,
    ) -> c_int;
}

fn last_error() -> String {
    unsafe {
        let p = XGBGetLastError();
        if p.is_null() {
            "XGBoost error: <null>".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn xgb_check(rc: c_int, what: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("{what}\n\nCaused by:\n    {}", last_error()))
    }
}

pub struct DMatrix {
    handle: DMatrixHandle,
}

impl DMatrix {
    pub fn from_dense(
        data: &[f32],
        nrow: usize,
        ncol: usize,
        missing: f32,
    ) -> Result<Self, String> {
        if data.len() != nrow * ncol {
            return Err(format!(
                "DMatrix::from_dense: data.len()={} != nrow*ncol={}*{}={}",
                data.len(),
                nrow,
                ncol,
                nrow * ncol
            ));
        }
        let mut h: DMatrixHandle = ptr::null_mut();
        unsafe {
            xgb_check(
                XGDMatrixCreateFromMat(
                    data.as_ptr(),
                    nrow as BstUlong,
                    ncol as BstUlong,
                    missing,
                    &mut h,
                ),
                "XGDMatrixCreateFromMat",
            )?;
        }
        Ok(Self { handle: h })
    }

    #[inline]
    pub fn handle(&self) -> DMatrixHandle {
        self.handle
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = XGDMatrixFree(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

pub struct Booster {
    handle: BoosterHandle,
}

#[derive(Debug, Clone)]
pub struct PredictOutput {
    pub values: Vec<f32>,
    pub shape: Vec<usize>, // strict_shape=true 时是多维；false 时通常就是 [out_len]
}

impl Booster {
    pub fn load_model(model_path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        // 为了保险：给一个“非空地址”的 dmats 指针，但 len=0
        let dummy: [DMatrixHandle; 1] = [ptr::null_mut()];
        let mut h: BoosterHandle = ptr::null_mut();

        unsafe {
            xgb_check(
                XGBoosterCreate(dummy.as_ptr(), 0, &mut h),
                "XGBoosterCreate",
            )?;
        }

        let cpath = CString::new(model_path.as_ref().to_string_lossy().as_bytes().to_vec())
            .map_err(|e| format!("CString model_path: {e}"))?;

        unsafe {
            xgb_check(XGBoosterLoadModel(h, cpath.as_ptr()), "XGBoosterLoadModel")?;
        }

        Ok(Self { handle: h })
    }

    #[inline]
    pub fn handle(&self) -> BoosterHandle {
        self.handle
    }

    fn predict_from_dmatrix(
        &self,
        dmat: &DMatrix,
        config_json: &str,
    ) -> Result<PredictOutput, String> {
        let cconfig = CString::new(config_json).map_err(|e| format!("CString config: {e}"))?;

        let mut out_shape_ptr: *const BstUlong = ptr::null();
        let mut out_dim: BstUlong = 0;
        let mut out_result_ptr: *const f32 = ptr::null();

        unsafe {
            xgb_check(
                XGBoosterPredictFromDMatrix(
                    self.handle,
                    dmat.handle(),
                    cconfig.as_ptr(),
                    &mut out_shape_ptr,
                    &mut out_dim,
                    &mut out_result_ptr,
                ),
                "XGBoosterPredictFromDMatrix",
            )?;

            if out_dim == 0 || out_shape_ptr.is_null() || out_result_ptr.is_null() {
                return Err(format!(
                    "predict: got null output (out_dim={out_dim}, out_shape_ptr={:?}, out_result_ptr={:?})",
                    out_shape_ptr, out_result_ptr
                ));
            }

            // ⚠️ 关键：shape/result 都是 XGBoost 内部指针，只能“立刻拷贝”，不能接管所有权
            let shape_u64 = std::slice::from_raw_parts(out_shape_ptr, out_dim as usize);
            let shape: Vec<usize> = shape_u64.iter().map(|&x| x as usize).collect();

            let out_len: usize = shape.iter().copied().product();
            let vals = std::slice::from_raw_parts(out_result_ptr, out_len).to_vec();

            Ok(PredictOutput {
                values: vals,
                shape,
            })
        }
    }

    /// type=0: normal prediction
    pub fn predict_proba_dense_1row(&self, row: &[f32]) -> Result<f32, String> {
        let dmat = DMatrix::from_dense(row, 1, row.len(), f32::NAN)?;
        // XGBoost 3.x 要求 iteration_begin / iteration_end 必须给
        let cfg = r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":true}"#;
        let out = self.predict_from_dmatrix(&dmat, cfg)?;
        Ok(*out.values.get(0).unwrap_or(&0.0))
    }

    /// type=2: SHAP contributions（最后一列是 bias）
    pub fn predict_contribs_dense_1row(&self, row: &[f32]) -> Result<PredictOutput, String> {
        let dmat = DMatrix::from_dense(row, 1, row.len(), f32::NAN)?;
        let cfg = r#"{"type":2,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":true}"#;
        self.predict_from_dmatrix(&dmat, cfg)
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                //let _ = XGBoosterFree(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}
