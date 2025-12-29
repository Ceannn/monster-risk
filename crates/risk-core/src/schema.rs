// crates/risk-core/src/schema.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Pos,
    Ecom,
    Atm,
    Transfer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRequest {
    /// 可选：客户端传入；否则服务端生成
    pub trace_id: Option<Uuid>,

    /// 事件时间（ms since epoch），用于点时一致/回放（不使用 arrival time）
    pub event_time_ms: i64,

    pub user_id: String,
    pub card_id: String,
    pub merchant_id: String,
    pub mcc: i32,

    pub amount: f64,
    pub currency: String,
    pub country: String,

    pub channel: Channel,
    pub device_id: String,
    pub ip_prefix: String,

    pub is_3ds: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allow,
    ManualReview,
    Deny,
    /// 超时/预算不足时的“保守降级”：为了不误杀，直接放行
    DegradeAllow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasonItem {
    pub signal: String,
    pub value: f64,
    pub baseline_p95: f64,
    pub direction: String, // "risk_up" / "risk_down" / "info"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub trace_id: Uuid,
    pub score: f64,
    pub decision: Decision,
    pub reason: Vec<ReasonItem>,
    /// 分段耗时（微秒）
    pub timings_us: TimingsUs,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingsUs {
    pub parse: u64,
    pub feature: u64,
    pub router: u64,
    pub l1: u64,
    pub l2: u64,
    pub serialize: u64,
}
