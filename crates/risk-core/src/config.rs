use serde::{Deserialize, Serialize};

/// 运行时配置：后续可以换成文件读取 / 动态热更。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// 端到端预算（用于 Router 的 deadline / 超时降级）
    pub slo_p99_ms: u64,

    /// L1 不确定区间：落在 (low, high) 的样本才会触发 L2/L3
    pub l1_uncertain_low: f64,
    pub l1_uncertain_high: f64,

    /// 风险阈值（用于 decision）
    pub deny_threshold: f64,
    pub review_threshold: f64,

    /// FeatureStore：滑窗参数（秒）
    pub win_60s: u64,
    pub win_300s: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            slo_p99_ms: 10,
            l1_uncertain_low: 0.35,
            l1_uncertain_high: 0.65,
            deny_threshold: 0.85,
            review_threshold: 0.65,
            win_60s: 60,
            win_300s: 300,
        }
    }
}
