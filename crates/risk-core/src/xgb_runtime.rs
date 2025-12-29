use anyhow::Context;
use flate2::read::GzDecoder;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use xgb_ffi::Booster;

#[derive(Debug, Clone, Deserialize)]
pub struct Policy {
    pub review_threshold: f32,
    pub deny_threshold: f32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            review_threshold: 0.5,
            deny_threshold: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct XgbRuntime {
    pub model_path: PathBuf,
    pub feature_names: Vec<String>,
    pub cat_maps: HashMap<String, HashMap<String, f32>>,
    pub policy: Policy,
}

impl XgbRuntime {
    pub fn load_from_dir(dir: &Path) -> anyhow::Result<Self> {
        // 约定：
        // - feature_names.json: ["f1","f2",...]
        // - cat_maps.json.gz:  {"col":{"A":1.0,"B":2.0}, ...}
        // - policy.json (可选): {"review_threshold":0.5,"deny_threshold":0.9}
        // - model: ieee_xgb.ubj / xgb_model.json / ieee_xgb.bin
        let feature_names_path = dir.join("feature_names.json");
        let cat_maps_gz_path = dir.join("cat_maps.json.gz");
        let policy_path = dir.join("policy.json");

        let model_path_candidates = [
            // ✅ 最高优先：UBJSON（跨版本更稳）
            dir.join("ieee_xgb.ubj"),
            // 次优先：JSON
            dir.join("xgb_model.json"),
            // 兜底：老二进制
            dir.join("ieee_xgb.bin"),
        ];

        let feature_names = {
            let s = fs::read_to_string(&feature_names_path)
                .with_context(|| format!("read {}", feature_names_path.display()))?;
            serde_json::from_str::<Vec<String>>(&s)
                .with_context(|| format!("parse {}", feature_names_path.display()))?
        };

        let cat_maps = if cat_maps_gz_path.exists() {
            read_gz_json_map(&cat_maps_gz_path)
                .with_context(|| format!("read {}", cat_maps_gz_path.display()))?
        } else {
            HashMap::new()
        };

        let policy = if policy_path.exists() {
            let s = fs::read_to_string(&policy_path)
                .with_context(|| format!("read {}", policy_path.display()))?;
            serde_json::from_str::<Policy>(&s)
                .with_context(|| format!("parse {}", policy_path.display()))?
        } else {
            Policy::default()
        };

        let model_path = model_path_candidates
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot find model file: expected ieee_xgb.ubj / xgb_model.json / ieee_xgb.bin under {}",
                    dir.display()
                )
            })?;

        Ok(Self {
            model_path,
            feature_names,
            cat_maps,
            policy,
        })
    }

    /// 把输入 JSON object（列名->值）按 feature_names 顺序拼成 dense row（缺失=NaN，字符串=cat_maps 编码）
    pub fn build_row(&self, obj: &Map<String, Value>) -> Vec<f32> {
        let mut row = Vec::with_capacity(self.feature_names.len());

        for name in &self.feature_names {
            let v = obj.get(name);
            let x = match v {
                None | Some(Value::Null) => f32::NAN,
                Some(Value::Bool(b)) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN) as f32,
                Some(Value::String(s)) => {
                    // 未知类别 → 0（也可以改成 NaN；0 更稳，线上不容易炸）
                    self.cat_maps
                        .get(name)
                        .and_then(|m| m.get(s))
                        .copied()
                        .unwrap_or(0.0)
                }
                _ => f32::NAN,
            };
            row.push(x);
        }

        row
    }

    /// Thread-local Booster：避免 Booster !Send/!Sync + 避免锁
    fn with_booster<F, R>(&self, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&Booster) -> anyhow::Result<R>,
    {
        thread_local! {
            static TL: std::cell::RefCell<Option<(PathBuf, Booster)>> = std::cell::RefCell::new(None);
        }

        TL.with(|cell| -> anyhow::Result<R> {
            let mut guard = cell.borrow_mut();

            let need_reload = match guard.as_ref() {
                None => true,
                Some((path, _)) => path != &self.model_path,
            };

            if need_reload {
                let booster = Booster::load_model(&self.model_path)
                    .map_err(|e: String| anyhow::anyhow!(e))
                    .with_context(|| format!("load_model {}", self.model_path.display()))?;
                *guard = Some((self.model_path.clone(), booster));
            }

            let (_, booster) = guard.as_ref().unwrap();
            f(booster)
        })
    }

    /// 预测概率（1 行）
    pub fn predict_proba(&self, row: &[f32]) -> anyhow::Result<f32> {
        self.with_booster(|bst| {
            let p = bst
                .predict_proba_dense_1row(row)
                .map_err(|e: String| anyhow::anyhow!(e))
                .context("predict_proba_dense_1row")?;
            Ok(p)
        })
    }

    /// contrib（返回：features contrib + bias）
    /// - 一般 contrib 长度 = n_features + 1（最后一项 bias）
    pub fn contrib_row_with_bias(&self, row: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.with_booster(|bst| {
            let out = bst
                .predict_contribs_dense_1row(row)
                .map_err(|e: String| anyhow::anyhow!(e))
                .context("predict_contribs_dense_1row")?;

            // strict_shape=true 时通常是 3D: [1, groups, m]
            // 有时也可能是 2D: [1, m]
            let contrib_row: Vec<f32> = match out.shape.as_slice() {
                [1, _groups, m] => out.values[..*m].to_vec(),
                [1, m] => out.values[..*m].to_vec(),
                _ => anyhow::bail!("unexpected contrib shape: {:?}", out.shape),
            };

            Ok(contrib_row)
        })
    }

    /// 解释：取 TopK 贡献（绝对值排序），返回 (feature_name, contribution)
    /// 注意：默认忽略最后的 bias（如果你想要 bias，我也可以给你单独返回）
    pub fn topk_contrib(&self, row: &[f32], k: usize) -> anyhow::Result<Vec<(String, f32)>> {
        let n = self.feature_names.len();
        if row.len() != n {
            anyhow::bail!("feature size mismatch: got {}, expected {}", row.len(), n);
        }

        let contrib = self.contrib_row_with_bias(row)?;

        if contrib.len() < n {
            anyhow::bail!(
                "contrib too short: got {}, expected >= {}",
                contrib.len(),
                n
            );
        }

        // 只取前 n 个（忽略 bias）
        let mut pairs: Vec<(usize, f32)> = (0..n).map(|i| (i, contrib[i])).collect();
        pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        Ok(pairs
            .into_iter()
            .take(k.min(n))
            .map(|(i, c)| (self.feature_names[i].clone(), c))
            .collect::<Vec<(String, f32)>>())
    }

    pub fn decide(&self, score: f32) -> &'static str {
        if score >= self.policy.deny_threshold {
            "deny"
        } else if score >= self.policy.review_threshold {
            "review"
        } else {
            "allow"
        }
    }
}

fn read_gz_json_map(path: &Path) -> anyhow::Result<HashMap<String, HashMap<String, f32>>> {
    let mut f = fs::File::open(path)?;
    let mut gz = GzDecoder::new(&mut f);
    let mut buf = Vec::new();
    gz.read_to_end(&mut buf)?;
    let m = serde_json::from_slice::<HashMap<String, HashMap<String, f32>>>(&buf)?;
    Ok(m)
}
