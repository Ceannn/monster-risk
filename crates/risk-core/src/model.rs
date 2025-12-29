use crate::util::{clamp01, sigmoid};

/// 最小闭环版模型：
/// - L1：非常快的线性模型（模拟 LR）
/// - L2：稍重一点（仍然线性，但特征更多）
///
/// 后续升级路线：
/// - L2 换成 GBDT（并可“编译到 Rust 数组 + early-exit”）
/// - L3 换成 Transformer（触发式、受预算约束）
#[derive(Debug, Clone)]
pub struct Models {
    pub l1: LinearModel,
    pub l2: LinearModel,
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub bias: f64,
    pub weights: Vec<(String, f64)>, // (feature_name, weight)
}

impl LinearModel {
    pub fn score(&self, feats: &[(String, f64)]) -> f64 {
        // 为了简单：O(n^2) 查找；后续可用固定 index 或 phf 直接 O(n)
        let mut z = self.bias;
        for (fname, w) in &self.weights {
            if let Some((_, v)) = feats.iter().find(|(n, _)| n == fname) {
                z += w * (*v);
            }
        }
        clamp01(sigmoid(z))
    }
}

impl Default for Models {
    fn default() -> Self {
        // 这些权重只是占位：目的是跑通“分层触发+决策+reason”链路
        // 你后续会用离线训练出来的权重/树模型替换它们。
        let l1 = LinearModel {
            bias: -2.0,
            weights: vec![
                ("amount_log".into(), 0.8),
                ("is_foreign".into(), 1.2),
                ("velocity_60s".into(), 0.15),
                ("device_risky".into(), 0.6),
                ("ip_risky".into(), 0.4),
            ],
        };
        let l2 = LinearModel {
            bias: -1.5,
            weights: vec![
                ("amount_log".into(), 0.7),
                ("is_foreign".into(), 1.0),
                ("velocity_300s".into(), 0.08),
                ("amount_sum_300s".into(), 0.002),
                ("inter_arrival_ms".into(), -0.0003),
                ("mcc_risk".into(), 0.9),
                ("is_3ds".into(), -0.8),
            ],
        };
        Self { l1, l2 }
    }
}

impl Models {
    pub fn new() -> Self {
        Self::default()
    }
}
