use crate::xgb_pool::{XgbPool, XgbPoolError};
use crate::xgb_runtime::XgbRuntime;
use crate::{
    config::Config,
    feature_store::FeatureStore,
    model::Models,
    schema::{Decision, ReasonItem, ScoreRequest, ScoreResponse, TimingsUs},
    util::now_us,
};

use anyhow::Context;
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppCore {
    pub cfg: Config,
    pub store: Arc<FeatureStore>,
    pub models: Models,
    pub xgb: Option<Arc<XgbRuntime>>,

    /// ✅ 专用推理池（N OS 线程、每线程一份 Booster、有界队列）
    pub xgb_pool: Option<Arc<XgbPool>>,
}

impl AppCore {
    pub fn new(cfg: Config) -> Self {
        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        Self {
            cfg,
            store,
            models: Models::default(),
            xgb: None,
            xgb_pool: None,
        }
    }

    pub fn new_with_xgb(cfg: Config, model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let xgb = Arc::new(XgbRuntime::load_from_dir(model_dir)?);

        // ====== 专用推理池参数（不动 Config，靠 env 控制，实验很舒服）======
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let default_threads = avail.min(8).max(1);

        let pool_threads: usize = std::env::var("XGB_POOL_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);

        let pool_cap: usize = std::env::var("XGB_POOL_QUEUE_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(pool_threads.saturating_mul(64).max(64));

        let warmup_iters: usize = std::env::var("XGB_POOL_WARMUP_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);

        let xgb_pool = {
            let feature_names = Arc::new(xgb.feature_names.clone());
            let model_path = xgb.model_path.clone();
            let pool = XgbPool::new(
                model_path,
                feature_names,
                pool_threads,
                pool_cap,
                warmup_iters,
            )?;
            Some(Arc::new(pool))
        };

        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        Ok(Self {
            cfg,
            store,
            models: Models::default(),
            xgb: Some(xgb),
            xgb_pool,
        })
    }

    /// 启动预热：消掉第一次请求的冷启动尖刺（TLS booster 路径的 warmup）
    pub fn warmup_xgb(&self, iters: usize) -> anyhow::Result<()> {
        let xgb = self.xgb.as_ref().context("xgb not enabled")?;
        let n = xgb.feature_names.len();
        let row = vec![f32::NAN; n];

        let iters = iters.max(1);
        for _ in 0..iters {
            let _ = xgb.predict_proba(&row)?;
        }

        // contrib 通常是更“冷”的路径，best-effort 打一下
        let _ = xgb.topk_contrib(&row, 1);
        Ok(())
    }

    /// XGB 在线推理（TLS Booster 对照版本）：输入 JSON object（宽表字段）
    pub fn score_xgb(
        &self,
        parse_us: u64,
        obj: &Map<String, Value>,
    ) -> anyhow::Result<ScoreResponse> {
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_millis(self.cfg.slo_p99_ms as u64);

        let mut timings = TimingsUs::default();
        timings.parse = parse_us;

        let xgb = self
            .xgb
            .as_ref()
            .context("xgb not enabled: start server with --model-dir")?;

        // feature build（编码+缺失处理）
        let t_feat = Instant::now();
        let row: Vec<f32> = xgb.build_row(obj);
        timings.feature = now_us(t_feat);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        // L1 = XGB predict proba
        let t_l1 = Instant::now();
        let score_f32: f32 = xgb.predict_proba(&row)?;
        timings.xgb = now_us(t_l1);
        metrics::histogram!("stage_xgb_us").record(timings.xgb as f64);

        // decision（把 xgb.decide() 的字符串映射到 Decision enum）
        let decision: Decision = decision_from_str(xgb.decide(score_f32));
        let want_explain = matches!(decision, Decision::ManualReview);

        // router（把 explain 也算进 router 时间）
        let t_router = Instant::now();
        let explain_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 3).max(1));
        let mut reasons: Vec<ReasonItem> = vec![];

        if want_explain {
            let now = Instant::now();
            if now + explain_budget > deadline {
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                metrics::counter!("router_l2_trigger_total").increment(1);
                let t_l2 = Instant::now();

                // 解释链路：失败也不把请求打成 500，优雅降级
                match xgb.topk_contrib(&row, 5) {
                    Ok(top) => {
                        timings.l2 = now_us(t_l2);
                        metrics::histogram!("stage_l2_us").record(timings.l2 as f64);

                        for (name, c) in top {
                            let c = c as f64;
                            reasons.push(ReasonItem {
                                signal: name,
                                baseline_p95: 0.0,
                                direction: if c >= 0.0 {
                                    "risk_up".into()
                                } else {
                                    "risk_down".into()
                                },
                            });
                        }
                    }
                    Err(_e) => {
                        timings.l2 = now_us(t_l2);
                        metrics::counter!("router_l2_failed_total").increment(1);
                    }
                }
            }
        }

        timings.router = now_us(t_router);
        metrics::histogram!("stage_router_us").record(timings.router as f64);

        // response
        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_f32 as f64,
            decision,
            reason: reasons,
            timings_us: timings,
        };

        // serialize（统计用；真正响应序列化在 server 层）
        let t_ser = Instant::now();
        let _ = serde_json::to_vec(&resp);
        resp.timings_us.serialize = now_us(t_ser);
        metrics::histogram!("stage_serialize_us").record(resp.timings_us.serialize as f64);

        // e2e
        metrics::histogram!("e2e_us").record(now_us(t0) as f64);

        Ok(resp)
    }

    /// ✅ 专用推理池版本（async）
    /// - 队列满会返回 XgbPoolError::QueueFull（server 侧映射成 429）
    pub async fn score_xgb_pool_async(
        &self,
        parse_us: u64,
        obj: &Map<String, Value>,
    ) -> anyhow::Result<ScoreResponse> {
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_millis(self.cfg.slo_p99_ms as u64);

        let mut timings = TimingsUs::default();
        timings.parse = parse_us;

        let xgb = self
            .xgb
            .as_ref()
            .context("xgb not enabled: start server with --model-dir")?;
        let pool = self.xgb_pool.as_ref().context("xgb_pool not enabled")?;

        // feature build
        let t_feat = Instant::now();
        let row: Vec<f32> = xgb.build_row(obj);
        timings.feature = now_us(t_feat);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        // L1：score（pool）
        let t_l1_total = Instant::now();
        let rx = pool
            .try_submit(row.clone(), 0)
            .map_err(|e| anyhow::Error::new(e))?;

        let out = rx
            .await
            .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

        timings.xgb = now_us(t_l1_total);
        metrics::histogram!("stage_xgb_us").record(timings.xgb as f64);

        // 额外拆分指标（论文图很好看）
        metrics::histogram!("xgb_pool_queue_wait_us").record(out.queue_wait_us as f64);
        metrics::histogram!("xgb_pool_xgb_compute_us").record(out.xgb_us as f64);

        let score_f32 = out.score;
        let decision: Decision = decision_from_str(xgb.decide(score_f32));
        let want_explain = matches!(decision, Decision::ManualReview);

        // router + explain
        let t_router = Instant::now();
        let explain_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 3).max(1));
        let mut reasons: Vec<ReasonItem> = vec![];

        if want_explain {
            let now = Instant::now();
            if now + explain_budget > deadline {
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                metrics::counter!("router_l2_trigger_total").increment(1);

                let t_l2_total = Instant::now();
                let rx2 = pool.try_submit(row, 5).map_err(|e| anyhow::Error::new(e))?;

                let out2 = rx2
                    .await
                    .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

                timings.l2 = now_us(t_l2_total);
                metrics::histogram!("stage_l2_us").record(timings.l2 as f64);

                if let Some(top) = out2.contrib_topk {
                    for (name, c) in top {
                        let c = c as f64;
                        reasons.push(ReasonItem {
                            signal: name,
                            baseline_p95: 0.0,
                            direction: if c >= 0.0 {
                                "risk_up".into()
                            } else {
                                "risk_down".into()
                            },
                        });
                    }
                }
            }
        }

        timings.router = now_us(t_router);
        metrics::histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_f32 as f64,
            decision,
            reason: reasons,
            timings_us: timings,
        };

        let t_ser = Instant::now();
        let _ = serde_json::to_vec(&resp);
        resp.timings_us.serialize = now_us(t_ser);
        metrics::histogram!("stage_serialize_us").record(resp.timings_us.serialize as f64);

        metrics::histogram!("e2e_us").record(now_us(t0) as f64);

        Ok(resp)
    }

    /// baseline：结构化 ScoreRequest → features → L1/L2 → decision（保持你原始逻辑）
    pub fn score(&self, req: ScoreRequest) -> ScoreResponse {
        let t0 = Instant::now();
        let trace_id = req.trace_id.unwrap_or_else(Uuid::new_v4);

        let budget = Duration::from_millis(self.cfg.slo_p99_ms);
        let deadline = t0 + budget;

        let mut timings = TimingsUs::default();

        // feature
        let tf0 = Instant::now();
        let feats = self.extract_features(&req);
        timings.feature = now_us(tf0);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        // router + L1
        let tr0 = Instant::now();
        let tl10 = Instant::now();
        let s1: f64 = self.models.l1.score(&feats) as f64;
        timings.xgb = now_us(tl10);
        metrics::histogram!("stage_xgb_us").record(timings.xgb as f64);

        // optional L2（预算式）
        let mut score: f64 = s1;
        let mut used_l2 = false;

        if s1 > self.cfg.l1_uncertain_low as f64 && s1 < self.cfg.l1_uncertain_high as f64 {
            let now = Instant::now();
            if now < deadline {
                let remaining = deadline.duration_since(now);
                if remaining > Duration::from_millis(1) {
                    let tl20 = Instant::now();
                    let s2: f64 = self.models.l2.score(&feats) as f64;
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

        // decision
        let decision = if Instant::now() > deadline {
            metrics::counter!("router_deadline_miss_total").increment(1);
            Decision::DegradeAllow
        } else if score >= self.cfg.deny_threshold as f64 {
            Decision::Deny
        } else if score >= self.cfg.review_threshold as f64 {
            Decision::ManualReview
        } else {
            Decision::Allow
        };

        // reason
        let mut reason = self.build_reason(&feats);
        if used_l2 {
            reason.push(ReasonItem {
                signal: "layer2_used".into(),
                baseline_p95: 1.0,
                direction: "info".into(),
            });
        }

        // serialize（统计）
        let ts0 = Instant::now();
        timings.serialize = now_us(ts0);
        metrics::histogram!("stage_serialize_us").record(timings.serialize as f64);

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
            7995 => 1.0,
            6011 => 0.8,
            4829 => 0.6,
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
            baseline_p95: 6.5,
            direction: (if amount_log > 6.5 {
                "risk_up"
            } else {
                "risk_down"
            })
            .into(),
        });

        let v60 = get("velocity_60s");
        out.push(ReasonItem {
            signal: "velocity_60s".into(),
            baseline_p95: 4.0,
            direction: (if v60 > 4.0 { "risk_up" } else { "risk_down" }).into(),
        });

        let foreign = get("is_foreign");
        out.push(ReasonItem {
            signal: "is_foreign".into(),
            baseline_p95: 1.0,
            direction: (if foreign > 0.5 {
                "risk_up"
            } else {
                "risk_down"
            })
            .into(),
        });

        let mcc = get("mcc_risk");
        out.push(ReasonItem {
            signal: "mcc_risk".into(),
            baseline_p95: 0.8,
            direction: (if mcc > 0.8 { "risk_up" } else { "risk_down" }).into(),
        });

        out
    }
}

fn decision_from_str<S: AsRef<str>>(s: S) -> Decision {
    match s.as_ref() {
        "allow" => Decision::Allow,
        "deny" => Decision::Deny,
        "review" | "manual_review" => Decision::ManualReview,
        "degrade_allow" | "degraded_allow" => Decision::DegradeAllow,
        _ => Decision::ManualReview,
    }
}
