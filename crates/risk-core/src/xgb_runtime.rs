use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "xgb_ffi")]
use xgb_ffi::{Booster, DMatrix};

#[cfg(feature = "native_l1_tl2cgen")]
use crate::native_l1_tl2cgen;
#[cfg(feature = "native_l2_tl2cgen")]
use crate::native_l2_tl2cgen;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Policy {
    pub review_threshold: f32,
    pub deny_threshold: f32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            review_threshold: 0.5,
            deny_threshold: 0.95,
        }
    }
}

fn looks_like_l2_path(p: &Path) -> bool {
    let s = p.to_string_lossy().to_ascii_lowercase();
    // 兼容: .../ieee_l2... 或 .../l2_judge...
    s.contains("l2")
}

fn select_model_file(dir: &Path) -> Option<PathBuf> {
    let p0 = dir.join("xgb_model.json");
    if p0.exists() {
        return Some(p0);
    }

    // 兼容: xgb_model_iter1800.json / xgb_model_iter1801.json ...
    let mut cands: Vec<(u32, PathBuf)> = vec![];
    if let Ok(rd) = fs::read_dir(dir) {
        for ent in rd.flatten() {
            let path = ent.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if let Some(rest) = name.strip_prefix("xgb_model_iter") {
                if let Some(rest) = rest.strip_suffix(".json") {
                    if let Ok(it) = rest.parse::<u32>() {
                        cands.push((it, path));
                    }
                }
            }
        }
    }
    cands.sort_by_key(|(it, _)| *it);
    cands.first().map(|(_, p)| p.clone())
}

fn load_feature_names(dir: &Path) -> Result<Vec<String>> {
    let json_path = dir.join("feature_names.json");
    if json_path.exists() {
        let s = fs::read_to_string(&json_path)
            .with_context(|| format!("read feature_names.json: {}", json_path.display()))?;
        let names: Vec<String> = serde_json::from_str(&s)
            .with_context(|| format!("parse feature_names.json: {}", json_path.display()))?;
        return Ok(names);
    }

    let txt_path = dir.join("features.txt");
    if txt_path.exists() {
        let s = fs::read_to_string(&txt_path)
            .with_context(|| format!("read features.txt: {}", txt_path.display()))?;
        let mut out: Vec<String> = vec![];
        for line in s.lines() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            out.push(t.to_string());
        }
        return Ok(out);
    }

    Err(anyhow!(
        "missing feature schema in model_dir={}, expected feature_names.json or features.txt",
        dir.display()
    ))
}

fn load_cat_maps(dir: &Path) -> Result<HashMap<String, HashMap<String, u32>>> {
    // 兼容: cat_maps.json.gz (我们训练脚本会写)
    let gz = dir.join("cat_maps.json.gz");
    if gz.exists() {
        let f = fs::File::open(&gz)
            .with_context(|| format!("open cat_maps.json.gz: {}", gz.display()))?;
        let dec = flate2::read::GzDecoder::new(f);
        let maps: HashMap<String, HashMap<String, u32>> = serde_json::from_reader(dec)
            .with_context(|| format!("parse cat_maps.json.gz: {}", gz.display()))?;
        return Ok(maps);
    }

    // 允许没有 cat_maps（纯数值 / 已经提前做 one-hot / hash 等）
    Ok(HashMap::new())
}

fn load_policy(dir: &Path) -> Result<Policy> {
    let p = dir.join("policy.json");
    if !p.exists() {
        return Ok(Policy::default());
    }
    let s = fs::read_to_string(&p).with_context(|| format!("read policy.json: {}", p.display()))?;
    let v: Policy =
        serde_json::from_str(&s).with_context(|| format!("parse policy.json: {}", p.display()))?;
    Ok(v)
}

#[derive(Debug)]
pub struct XgbRuntime {
    pub model_dir: PathBuf,
    pub model_path: PathBuf,
    pub feature_names: Vec<String>,
    pub cat_maps: HashMap<String, HashMap<String, u32>>,
    pub policy: Policy,
    is_l2: bool,

    #[cfg(feature = "xgb_ffi")]
    booster: Booster,
}

impl XgbRuntime {
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let model_dir = dir.to_path_buf();
        let is_l2 = looks_like_l2_path(dir);

        let feature_names = load_feature_names(dir)?;
        let cat_maps = load_cat_maps(dir)?;
        let policy = load_policy(dir)?;

        // xgb_model.json 在 TL2CGEN 静态推理模式下不一定“必须”，但保留路径用于日志/兼容。
        let model_path = select_model_file(dir).unwrap_or_else(|| dir.join("xgb_model.json"));

        #[cfg(feature = "xgb_ffi")]
        {
            if !model_path.exists() {
                return Err(anyhow!(
                    "xgb_ffi enabled but model file missing: {}",
                    model_path.display()
                ));
            }
            let booster = Booster::from_file(&model_path)
                .with_context(|| format!("load xgboost model: {}", model_path.display()))?;
            Ok(Self {
                model_dir,
                model_path,
                feature_names,
                cat_maps,
                policy,
                is_l2,
                booster,
            })
        }

        #[cfg(not(feature = "xgb_ffi"))]
        {
            Ok(Self {
                model_dir,
                model_path,
                feature_names,
                cat_maps,
                policy,
                is_l2,
            })
        }
    }

    #[inline]
    pub fn is_l2(&self) -> bool {
        self.is_l2
    }

    #[inline]
    pub fn decide(&self, score: f32) -> &'static str {
        if score >= self.policy.deny_threshold {
            "deny"
        } else if score >= self.policy.review_threshold {
            "review"
        } else {
            "allow"
        }
    }

    /// 从 JSON object（宽表字段）构造模型输入 row（dense f32）
    ///
    /// - 数值：直接转 f32
    /// - 字符串：查 cat_maps 映射到整数 id，再 cast 为 f32（与训练一致）
    /// - 缺失：NaN（TL2CGEN / XGBoost 都用 NaN 表示 missing）
    pub fn build_row(&self, obj: &Map<String, Value>) -> Vec<f32> {
        self.feature_names
            .iter()
            .map(|k| match obj.get(k) {
                None => f32::NAN,
                Some(Value::Null) => f32::NAN,
                Some(Value::Bool(b)) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN) as f32,
                Some(Value::String(s)) => {
                    // 注意：未知类别 -> 0（训练时也会把未知映射到 0 或者 special bucket）
                    let map = self.cat_maps.get(k);
                    let id = map.and_then(|m| m.get(s).copied()).unwrap_or(0);
                    id as f32
                }
                Some(v) => {
                    // 兜底：数组/对象等不支持
                    let _ = v;
                    f32::NAN
                }
            })
            .collect()
    }

    /// 预测概率（单行 dense）
    pub fn predict_proba(&self, row: &[f32]) -> Result<f32> {
        // TL2CGEN 静态推理优先（如果编译了）
        #[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
        {
            if self.is_l2 {
                #[cfg(feature = "native_l2_tl2cgen")]
                {
                    return native_l2_tl2cgen::predict_proba_dense_1row(row);
                }
                #[cfg(not(feature = "native_l2_tl2cgen"))]
                {
                    return Err(anyhow!(
                        "L2 model_dir={} but feature native_l2_tl2cgen is not enabled",
                        self.model_dir.display()
                    ));
                }
            } else {
                #[cfg(feature = "native_l1_tl2cgen")]
                {
                    return native_l1_tl2cgen::predict_proba_dense_1row(row);
                }
                #[cfg(not(feature = "native_l1_tl2cgen"))]
                {
                    // no L1 native
                }
            }
        }

        #[cfg(feature = "xgb_ffi")]
        {
            let dm = DMatrix::from_dense(row, 1, row.len(), f32::NAN)?;
            let pred = self.booster.predict(&dm, false)?;
            pred.get(0)
                .copied()
                .ok_or_else(|| anyhow!("xgb_ffi predict returned empty"))
        }

        #[cfg(not(feature = "xgb_ffi"))]
        {
            Err(anyhow!(
                "no predictor backend enabled: build with native_*_tl2cgen and/or xgb_ffi"
            ))
        }
    }

    /// 预测概率：输入为小端 f32 的 bytes（常见于 /score_dense_f32_bin 这种二进制接口）
    pub fn predict_proba_dense_bytes_le(&self, bytes: &[u8], ncols: usize) -> Result<f32> {
        #[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
        {
            if self.is_l2 {
                #[cfg(feature = "native_l2_tl2cgen")]
                {
                    return native_l2_tl2cgen::predict_proba_dense_bytes_le(bytes, ncols);
                }
                #[cfg(not(feature = "native_l2_tl2cgen"))]
                {
                    return Err(anyhow!(
                        "L2 model_dir={} but feature native_l2_tl2cgen is not enabled",
                        self.model_dir.display()
                    ));
                }
            } else {
                #[cfg(feature = "native_l1_tl2cgen")]
                {
                    return native_l1_tl2cgen::predict_proba_dense_bytes_le(bytes, ncols);
                }
                #[cfg(not(feature = "native_l1_tl2cgen"))]
                {
                    // no L1 native
                }
            }
        }

        // fallback：解码 bytes -> row -> predict
        if bytes.len() != ncols * 4 {
            return Err(anyhow!(
                "dense bytes len mismatch: got={} expect={}",
                bytes.len(),
                ncols * 4
            ));
        }
        let mut row = vec![0f32; ncols];
        for i in 0..ncols {
            let b = &bytes[i * 4..i * 4 + 4];
            row[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        self.predict_proba(&row)
    }

    /// top-k feature contribution（仅 xgb_ffi 支持；TL2CGEN 纯推理不提供解释接口）
    pub fn topk_contrib(&self, row: &[f32], topk: usize) -> Result<Vec<(String, f32)>> {
        #[cfg(feature = "xgb_ffi")]
        {
            let mut row2 = Vec::with_capacity(row.len() + 1);
            row2.extend_from_slice(row);
            row2.push(1.0); // bias

            let dm = DMatrix::from_dense(&row2, 1, row2.len(), f32::NAN)?;
            let contrib = self.booster.predict_contrib(&dm, false)?;

            // contrib 形状: [n_features + 1]，最后是 bias
            let mut pairs: Vec<(usize, f32)> = contrib
                .into_iter()
                .enumerate()
                .map(|(i, c)| (i, c))
                .collect();
            pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

            let names = self.contrib_names();
            let mut out = Vec::with_capacity(topk.min(pairs.len()));
            for (i, c) in pairs.into_iter().take(topk) {
                let name = names.get(i).cloned().unwrap_or_else(|| format!("f{i}"));
                out.push((name, c));
            }
            Ok(out)
        }

        #[cfg(not(feature = "xgb_ffi"))]
        {
            let _ = (row, topk);
            Ok(vec![])
        }
    }

    pub fn contrib_names(&self) -> Vec<String> {
        let mut names = self.feature_names.clone();
        names.push("bias".to_string());
        names
    }
}
