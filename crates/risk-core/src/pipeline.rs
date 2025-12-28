use crate::{
    config::Config,
    feature_store::FeatureStore,
    model::Models,
    schema::{Decision, ReasonItem, ScoreRequest, ScoreResponse, TimingsUs},
    util::now_us,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppCore {
    pub cfg: Config,
    pub store: Arc<FeatureStore>,
    pub models: Models,
}

impl AppCore {
    pub fn new(cfg: Config) -> Self {
        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        Self {
            cfg,
            store,
            models: Models::default(),
        }
    }

    pub fn score(&self, req: ScoreRequest) -> ScoreResponse {
        let t0 = Instant::now();
        let trace_id = req.trace_id.unwrap_or_else(Uuid::new_v4);

        // deadline：预算式编程的核心
        let budget = Duration::from_millis(self.cfg.slo_p99_ms);
        let deadline = t0 + budget;

        let mut timings = TimingsUs::default();

        // ---- feature stage
        let tf0 = Instant::now();
        let feats = self.extract_features(&req);
        timings.feature = now_us(tf0);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        // ---- router + L1
        let tr0 = Instant::now();
        let tl10 = Instant::now();
        let s1 = self.models.l1.score(&feats);
        timings.l1 = now_us(tl10);
        metrics::histogram!("stage_l1_us").record(timings.l1 as f64);

        // 不确定区间触发 L2
        let mut score = s1;
        let mut used_l2 = false;

        if s1 > self.cfg.l1_uncertain_low && s1 < self.cfg.l1_uncertain_high {
            // 预算检查：只在剩余预算足够时才触发
            let now = Instant::now();
            if now < deadline {
                let remaining = deadline.duration_since(now);
                // 经验阈值：留 1ms 给序列化等尾部
                if remaining > Duration::from_millis(1) {
                    let tl20 = Instant::now();
                    let s2 = self.models.l2.score(&feats);
                    timings.l2 = now_us(tl20);
                    metrics::histogram!("stage_l2_us").record(timings.l2 as f64);
                    score = s2;
                    used_l2 = true;
                    metrics::counter!("router_l2_trigger_total").increment(1);
                } else {
                    metrics::counter!("router_l2_skipped_budget_total").increment(1);
                }
            } else {
                metrics::counter!("router_timeout_before_l2_total").increment(1);
            }
        }

        timings.router = now_us(tr0);
        metrics::histogram!("stage_router_us").record(timings.router as f64);

        // ---- decision
        let decision = if Instant::now() > deadline {
            metrics::counter!("router_deadline_miss_total").increment(1);
            Decision::DegradeAllow
        } else if score >= self.cfg.deny_threshold {
            Decision::Deny
        } else if score >= self.cfg.review_threshold {
            Decision::ManualReview
        } else {
            Decision::Allow
        };

        // ---- reason：给出“证据条目”（不是黑箱分数）
        let mut reason = self.build_reason(&feats);
        if used_l2 {
            reason.push(ReasonItem {
                signal: "layer2_used".into(),
                value: 1.0,
                baseline_p95: 1.0,
                direction: "info".into(),
            });
        }

        // ---- serialize stage（这里只是统计，真正的序列化在 server 层）
        let ts0 = Instant::now();
        timings.serialize = now_us(ts0);
        metrics::histogram!("stage_serialize_us").record(timings.serialize as f64);

        // end-to-end
        metrics::histogram!("e2e_us").record(now_us(t0) as f64);

        ScoreResponse {
            trace_id,
            score,
            decision,
            reason,
            timings_us: timings,
        }
    }

    fn extract_features(&self, req: &ScoreRequest) -> Vec<(String, f64)> {
        // store features (query_then_update, 先读后写)
        let sf = self
            .store
            .query_then_update(&req.user_id, req.event_time_ms, req.amount);

        let amount_log = (req.amount.max(0.01)).ln();
        let is_foreign = if req.country.as_str() != "JP" {
            1.0
        } else {
            0.0
        };

        let mcc_risk = match req.mcc {
            7995 => 1.0, // gambling
            6011 => 0.8, // atm
            4829 => 0.6, // wire transfer
            _ => 0.2,
        };

        let device_risky = if req.device_id.ends_with("0") {
            1.0
        } else {
            0.0
        };
        let ip_risky = if req.ip_prefix.starts_with("198.") {
            1.0
        } else {
            0.0
        };
        let is_3ds = if req.is_3ds { 1.0 } else { 0.0 };

        vec![
            ("amount_log".into(), amount_log),
            ("is_foreign".into(), is_foreign),
            ("mcc_risk".into(), mcc_risk),
            ("device_risky".into(), device_risky),
            ("ip_risky".into(), ip_risky),
            ("is_3ds".into(), is_3ds),
            ("velocity_60s".into(), sf.velocity_60s),
            ("velocity_300s".into(), sf.velocity_300s),
            ("amount_sum_60s".into(), sf.amount_sum_60s),
            ("amount_sum_300s".into(), sf.amount_sum_300s),
            ("inter_arrival_ms".into(), sf.inter_arrival_ms),
            ("oo_order_flag".into(), sf.oo_order_flag),
        ]
    }

    fn build_reason(&self, feats: &[(String, f64)]) -> Vec<ReasonItem> {
        // 简化：挑几个典型信号返回
        let mut out = Vec::with_capacity(4);

        let get = |k: &str| {
            feats
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| *v)
                .unwrap_or(0.0)
        };

        let amount_log = get("amount_log");
        out.push(ReasonItem {
            signal: "amount_log".into(),
            value: amount_log,
            baseline_p95: 6.5,
            direction: if amount_log > 6.5 {
                "risk_up"
            } else {
                "risk_down"
            }
            .into(),
        });

        let v60 = get("velocity_60s");
        out.push(ReasonItem {
            signal: "velocity_60s".into(),
            value: v60,
            baseline_p95: 4.0,
            direction: if v60 > 4.0 { "risk_up" } else { "risk_down" }.into(),
        });

        let foreign = get("is_foreign");
        out.push(ReasonItem {
            signal: "is_foreign".into(),
            value: foreign,
            baseline_p95: 1.0,
            direction: if foreign > 0.5 {
                "risk_up"
            } else {
                "risk_down"
            }
            .into(),
        });

        let mcc = get("mcc_risk");
        out.push(ReasonItem {
            signal: "mcc_risk".into(),
            value: mcc,
            baseline_p95: 0.8,
            direction: if mcc > 0.8 { "risk_up" } else { "risk_down" }.into(),
        });

        out
    }
}
