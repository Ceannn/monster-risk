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

    /// ✅ L2 更重模型（可选）
    pub xgb_l2: Option<Arc<XgbRuntime>>,

    /// ✅ 专用推理池（N OS 线程、每线程一份 Booster、有界队列）
    pub xgb_pool: Option<Arc<XgbPool>>,

    /// ✅ L2 专用推理池（隔离更重模型，可选）
    pub xgb_pool_l2: Option<Arc<XgbPool>>,
}

impl AppCore {
    pub fn new(cfg: Config) -> Self {
        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        Self {
            cfg,
            store,
            models: Models::default(),
            xgb: None,
            xgb_l2: None,
            xgb_pool: None,
            xgb_pool_l2: None,
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

        // Optional CPU pinning for XGB workers.
        // Example: XGB_POOL_PIN_CPUS="2,4,6,8" (linux only)
        let pin_cpus: Option<Vec<usize>> = std::env::var("XGB_POOL_PIN_CPUS").ok().and_then(|s| {
            let v: Vec<usize> = s
                .split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(|x| x.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        });

        let xgb_pool = {
            let feature_names = Arc::new(xgb.feature_names.clone());
            let model_path = xgb.model_path.clone();
            let pool = XgbPool::new_with_pinning(
                model_path,
                feature_names,
                pool_threads,
                pool_cap,
                warmup_iters,
                pin_cpus,
            )?;
            Some(Arc::new(pool))
        };

        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        Ok(Self {
            cfg,
            store,
            models: Models::default(),
            xgb: Some(xgb),
            xgb_l2: None,
            xgb_pool,
            xgb_pool_l2: None,
        })
    }

    /// ✅ L1/L2 两套 XGB（可选 L2）：
    /// - L1：小模型 + 大吞吐，负责绝大多数请求
    /// - L2：大模型 + 独立线程池，只在 L1 进入灰区时触发
    ///
    /// 你可以用不同的 env 分别控制两套 pool：
    /// - L1：XGB_L1_POOL_THREADS / XGB_L1_POOL_QUEUE_CAP / XGB_L1_POOL_WARMUP_ITERS / XGB_L1_POOL_PIN_CPUS
    /// - L2：XGB_L2_POOL_THREADS / XGB_L2_POOL_QUEUE_CAP / XGB_L2_POOL_WARMUP_ITERS / XGB_L2_POOL_PIN_CPUS
    ///
    /// 兼容旧 env（只配一套时）：XGB_POOL_*
    pub fn new_with_xgb_l1_l2<P1: AsRef<Path>, P2: AsRef<Path>>(
        cfg: Config,
        l1_model_dir: P1,
        l2_model_dir: Option<P2>,
    ) -> anyhow::Result<Self> {
        fn parse_usize_env(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }

        fn parse_pin_env(key: &str) -> Option<Vec<usize>> {
            std::env::var(key).ok().and_then(|s| {
                let v: Vec<usize> = s
                    .split(',')
                    .map(|x| x.trim())
                    .filter(|x| !x.is_empty())
                    .map(|x| x.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                if v.is_empty() {
                    None
                } else {
                    Some(v)
                }
            })
        }

        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));

        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        // ---------- L1 ----------
        let l1_dir = l1_model_dir.as_ref();
        let xgb1 = Arc::new(XgbRuntime::load_from_dir(l1_dir)?);

        let default_l1_threads = avail.min(8).max(1);
        let l1_threads = std::env::var("XGB_L1_POOL_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                std::env::var("XGB_POOL_THREADS")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(default_l1_threads);

        let l1_cap = std::env::var("XGB_L1_POOL_QUEUE_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                std::env::var("XGB_POOL_QUEUE_CAP")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(l1_threads.saturating_mul(64).max(64));

        let l1_warm = std::env::var("XGB_L1_POOL_WARMUP_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                std::env::var("XGB_POOL_WARMUP_ITERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(50);

        let l1_pin =
            parse_pin_env("XGB_L1_POOL_PIN_CPUS").or_else(|| parse_pin_env("XGB_POOL_PIN_CPUS"));

        let xgb_pool1 = {
            let feature_names = Arc::new(xgb1.feature_names.clone());
            let model_path = xgb1.model_path.clone();
            let pool = XgbPool::new_with_pinning(
                model_path,
                feature_names,
                l1_threads,
                l1_cap,
                l1_warm,
                l1_pin,
            )?;
            Some(Arc::new(pool))
        };

        // ---------- L2 (optional) ----------
        let mut xgb2: Option<Arc<XgbRuntime>> = None;
        let mut xgb_pool2: Option<Arc<XgbPool>> = None;

        if let Some(l2_dir) = l2_model_dir {
            let l2_dir = l2_dir.as_ref();
            let x = Arc::new(XgbRuntime::load_from_dir(l2_dir)?);

            // 默认给 L2 少一点线程（因为只跑灰区），但你可以用 env 覆盖
            let default_l2_threads = (avail / 2).max(1).min(8);
            let l2_threads = parse_usize_env("XGB_L2_POOL_THREADS", default_l2_threads);

            let l2_cap = parse_usize_env(
                "XGB_L2_POOL_QUEUE_CAP",
                l2_threads.saturating_mul(32).max(32),
            );

            let l2_warm = parse_usize_env("XGB_L2_POOL_WARMUP_ITERS", 50);
            let l2_pin = parse_pin_env("XGB_L2_POOL_PIN_CPUS");

            let feature_names = Arc::new(x.feature_names.clone());
            let model_path = x.model_path.clone();
            let pool = XgbPool::new_with_pinning(
                model_path,
                feature_names,
                l2_threads,
                l2_cap,
                l2_warm,
                l2_pin,
            )?;
            xgb2 = Some(x);
            xgb_pool2 = Some(Arc::new(pool));
        }

        Ok(Self {
            cfg,
            store,
            models: Models::default(),
            xgb: Some(xgb1),
            xgb_l2: xgb2,
            xgb_pool: xgb_pool1,
            xgb_pool_l2: xgb_pool2,
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

        // 如果启用了 L2，也预热一下（避免第一次触发 L2 时尖刺）
        if let Some(xgb2) = self.xgb_l2.as_ref() {
            let n2 = xgb2.feature_names.len();
            let row2 = vec![f32::NAN; n2];
            for _ in 0..iters {
                let _ = xgb2.predict_proba(&row2)?;
            }
            let _ = xgb2.topk_contrib(&row2, 1);
        }
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

        let xgb1 = self
            .xgb
            .as_ref()
            .context("xgb not enabled: start server with --model-dir")?;

        let t_feat = Instant::now();
        let row1 = xgb1.build_row(obj);
        timings.feature = now_us(t_feat);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        let t_l1 = Instant::now();
        let score1 = xgb1.predict_proba(&row1)?;
        timings.xgb = now_us(t_l1);
        metrics::histogram!("stage_xgb_us").record(timings.xgb as f64);

        let mut score_final = score1;
        let mut decision: Decision = decision_from_str(xgb1.decide(score1));
        let mut reasons: Vec<ReasonItem> = vec![];

        let t_router = Instant::now();

        let l2_enabled = self.xgb_l2.is_some();

        if matches!(decision, Decision::ManualReview) && l2_enabled {
            let xgb2 = self.xgb_l2.as_ref().unwrap();

            let l2_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 2).max(1));
            let now = Instant::now();
            if now + l2_budget > deadline {
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                metrics::counter!("router_l2_trigger_total").increment(1);

                let row2 = xgb2.build_row(obj);

                let t_l2 = Instant::now();
                score_final = xgb2.predict_proba(&row2)?;
                // 解释（topk）放在 L2 路径里，成本只在灰区才付
                for (name, c) in xgb2.topk_contrib(&row2, 5)? {
                    let c = c as f64;
                    reasons.push(ReasonItem {
                        signal: name,
                        value: c,
                        baseline_p95: 0.0,
                        direction: if c >= 0.0 {
                            "risk_up".into()
                        } else {
                            "risk_down".into()
                        },
                    });
                }
                timings.l2 = now_us(t_l2);
                metrics::histogram!("stage_l2_us").record(timings.l2 as f64);

                decision = decision_from_str(xgb2.decide(score_final));

                reasons.push(ReasonItem {
                    signal: "l1_score".into(),
                    value: score1 as f64,
                    baseline_p95: 0.0,
                    direction: "info".into(),
                });
            }
        }

        timings.router = now_us(t_router);
        metrics::histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
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

        let xgb1 = self
            .xgb
            .as_ref()
            .context("xgb not enabled: start server with --model-dir")?;
        let pool1 = self.xgb_pool.as_ref().context("xgb_pool not enabled")?;

        // feature build（只做一次：L1 必定要跑）
        let t_feat = Instant::now();
        let row1: Vec<f32> = xgb1.build_row(obj);
        timings.feature = now_us(t_feat);
        metrics::histogram!("stage_feature_us").record(timings.feature as f64);

        // L1：score（pool）
        let t_l1_total = Instant::now();
        let rx = pool1
            .try_submit(row1.clone(), 0)
            .map_err(|e| anyhow::Error::new(e))?;

        let out1 = rx
            .await
            .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

        timings.xgb = now_us(t_l1_total);
        metrics::histogram!("stage_xgb_us").record(timings.xgb as f64);

        let score1 = out1.score;
        let mut score_final = score1;
        let mut decision: Decision = decision_from_str(xgb1.decide(score1));

        let mut reasons: Vec<ReasonItem> = vec![];

        // router + (optional) L2
        let t_router = Instant::now();

        let l2_enabled = self.xgb_l2.is_some() && self.xgb_pool_l2.is_some();

        if matches!(decision, Decision::ManualReview) && l2_enabled {
            // ✅ L1 进入灰区：尝试触发 L2（大模型，独立 pool）
            let xgb2 = self.xgb_l2.as_ref().unwrap();
            let pool2 = self.xgb_pool_l2.as_ref().unwrap();

            // 给 L2 留一点预算，避免拖炸整条链路（默认：SLO 的一半，你也可以后面再做成 env/flag）
            let l2_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 2).max(1));

            let now = Instant::now();
            if now + l2_budget > deadline {
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                metrics::counter!("router_l2_trigger_total").increment(1);

                // 只有真的要跑 L2 时才 build row2（两套模型特征顺序可能不同）
                let row2: Vec<f32> = xgb2.build_row(obj);

                let t_l2_total = Instant::now();
                let rx2 = pool2
                    .try_submit(row2, 5)
                    .map_err(|e| anyhow::Error::new(e))?;

                match rx2.await {
                    Ok(inner) => {
                        let out2 = inner?;
                        timings.l2 = now_us(t_l2_total);
                        metrics::histogram!("stage_l2_us").record(timings.l2 as f64);

                        score_final = out2.score;
                        decision = decision_from_str(xgb2.decide(out2.score));

                        for (name, c) in out2.contrib_topk {
                            let c = c as f64;
                            reasons.push(ReasonItem {
                                signal: name,
                                value: c,
                                baseline_p95: 0.0,
                                direction: if c >= 0.0 {
                                    "risk_up".into()
                                } else {
                                    "risk_down".into()
                                },
                            });
                        }

                        // 额外给一个信息项：让你线上 debug 更爽（不会影响排序/决策）
                        reasons.push(ReasonItem {
                            signal: "l1_score".into(),
                            value: score1 as f64,
                            baseline_p95: 0.0,
                            direction: "info".into(),
                        });
                    }
                    Err(_) => {
                        // oneshot canceled（极少见）：回退到 L1 结果
                        metrics::counter!("router_deadline_miss_total").increment(1);
                    }
                }
            }
        } else if matches!(decision, Decision::ManualReview) {
            // 兼容：没有 L2 时，仍然用 L1 做一版解释（论文图也好看）
            let explain_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 3).max(1));
            let now = Instant::now();
            if now + explain_budget > deadline {
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                metrics::counter!("router_l2_trigger_total").increment(1);

                let t_l2_total = Instant::now();
                let rx2 = pool1
                    .try_submit(row1, 5)
                    .map_err(|e| anyhow::Error::new(e))?;

                let out2 = rx2
                    .await
                    .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

                timings.l2 = now_us(t_l2_total);
                metrics::histogram!("stage_l2_us").record(timings.l2 as f64);

                for (name, c) in out2.contrib_topk {
                    let c = c as f64;
                    reasons.push(ReasonItem {
                        signal: name,
                        value: c,
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

        timings.router = now_us(t_router);
        metrics::histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
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
        metrics::counter!("router_l2_trigger_total").increment(0);
        metrics::counter!("router_l2_skipped_budget_total").increment(0);
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
                value: 1.0,
                baseline_p95: 1.0,
                direction: "info".into(),
            });
        }

        let mut resp = ScoreResponse {
            trace_id,
            score,
            decision,
            reason,
            timings_us: timings,
        };

        // serialize（统计用；真正响应序列化在 server 层）
        let ts0 = Instant::now();
        let _ = serde_json::to_vec(&resp);
        resp.timings_us.serialize = now_us(ts0);
        metrics::histogram!("stage_serialize_us").record(resp.timings_us.serialize as f64);

        metrics::histogram!("e2e_us").record(now_us(t0) as f64);

        resp
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
            value: amount_log,
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
            value: v60,
            baseline_p95: 4.0,
            direction: (if v60 > 4.0 { "risk_up" } else { "risk_down" }).into(),
        });

        let foreign = get("is_foreign");
        out.push(ReasonItem {
            signal: "is_foreign".into(),
            value: foreign,
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
            value: mcc,
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
