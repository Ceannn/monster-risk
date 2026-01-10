use crate::stateful_l2::{DenseStatefulCtx, StatefulL2Augmenter};
use crate::xgb_pool::{XgbPool, XgbPoolError, EXTRA_MAX};
use crate::xgb_runtime::XgbRuntime;
use crate::{
    config::Config,
    feature_store::FeatureStore,
    model::Models,
    schema::{Decision, ReasonItem, ScoreRequest, ScoreResponse, TimingsUs},
    util::{mix_u64, now_us},
};

use anyhow::Context;
use bytes::Bytes;
use serde_json::{Map, Value};
use std::cell::Cell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// A very small per-second budget (rate limiter) for L2 triggers.
///
/// This is intentionally approximate and lock-free: good enough for overload control.
#[derive(Debug)]
struct RateBudget {
    limit_per_sec: u64,
    start: Instant,
    sec: AtomicU64,
    count: AtomicU64,
}

impl RateBudget {
    fn new(limit_per_sec: u64) -> Self {
        Self {
            limit_per_sec,
            start: Instant::now(),
            sec: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn now_sec(&self) -> u64 {
        self.start.elapsed().as_secs()
    }

    /// Returns true if a token is granted.
    fn try_acquire(&self) -> bool {
        if self.limit_per_sec == 0 {
            return true;
        }

        let now = self.now_sec();
        let cur = self.sec.load(Ordering::Relaxed);
        if cur != now {
            // Try to advance the window. If we win the CAS, reset count.
            if self
                .sec
                .compare_exchange(cur, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.count.store(0, Ordering::Relaxed);
            }
        }

        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        n <= self.limit_per_sec
    }
}

thread_local! {
    static TLS_L2_SAMPLE_RNG: Cell<u64> = const { Cell::new(0) };
}

fn tls_rand_u64() -> u64 {
    TLS_L2_SAMPLE_RNG.with(|c| {
        let mut state = c.get();
        if state == 0 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let salt = (std::process::id() as u64).wrapping_mul(0x9e3779b97f4a7c15);
            let addr = (&*c as *const Cell<u64> as usize) as u64;
            state = now ^ salt ^ addr;
        }
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let out = mix_u64(state);
        c.set(state);
        out
    })
}

#[derive(Debug)]
struct L2Control {
    /// Hard cap: max L2 triggers per second (0 = unlimited).
    #[allow(dead_code)]
    max_triggers_per_sec: u64,
    rate_budget: Option<RateBudget>,

    /// If L2 pool queue waterline exceeds this threshold, skip L2 (1.0 = disable).
    max_queue_waterline: f64,

    /// Minimal remaining time budget (microseconds) required to attempt L2.
    min_remaining_us: u64,

    /// Optional queue-wait budget passed to XgbPool (microseconds). 0 disables.
    queue_wait_budget_us: u64,

    /// Base sampling ratio for L2 triggers (ppm: 0..=1_000_000).
    sample_base_ppm: u32,
    /// Minimum sampling ratio after feedback (ppm).
    sample_min_ppm: u32,
    /// Dynamic sampling ratio (ppm).
    sample_dyn_ppm: AtomicU64,
    /// Enable dynamic feedback controller.
    sample_dyn_enable: bool,
    /// Last feedback update time (ms since start).
    sample_last_update_ms: AtomicU64,
    /// Counter to throttle feedback updates.
    sample_update_ctr: AtomicU64,
    /// Waterline feedback target and bounds.
    sample_waterline_target: f64,
    sample_waterline_hi: f64,
    sample_waterline_lo: f64,
    /// Cached L2 queue waterline (bits).
    waterline_cached_bits: AtomicU64,
    /// Last waterline refresh time (ms since start).
    waterline_last_update_ms: AtomicU64,
    /// Counter to throttle waterline refreshes.
    waterline_update_ctr: AtomicU64,
    /// Local timebase for feedback throttling.
    start: Instant,
}

impl L2Control {
    fn from_env(cfg: &Config) -> Self {
        fn env_u64(key: &str) -> Option<u64> {
            std::env::var(key).ok().and_then(|s| s.parse::<u64>().ok())
        }
        fn env_f64(key: &str) -> Option<f64> {
            std::env::var(key).ok().and_then(|s| s.parse::<f64>().ok())
        }

        let max_triggers_per_sec = env_u64("ROUTER_L2_MAX_TRIGGERS_PER_SEC").unwrap_or(0);
        let rate_budget = if max_triggers_per_sec > 0 {
            Some(RateBudget::new(max_triggers_per_sec))
        } else {
            None
        };

        // 1.0 means disable waterline gating by default (keep old behavior unless configured).
        let max_queue_waterline = env_f64("ROUTER_L2_MAX_QUEUE_WATERLINE").unwrap_or(1.0);

        // Default: half of end-to-end SLO (in us), but at least 1000us.
        let default_min_remaining_us = ((cfg.slo_p99_ms * 1000) / 2).max(1_000);
        let min_remaining_us =
            env_u64("ROUTER_L2_MIN_REMAINING_US").unwrap_or(default_min_remaining_us);

        let queue_wait_budget_us = env_u64("ROUTER_L2_QUEUE_WAIT_BUDGET_US").unwrap_or(0);

        let base_ppm = env_u64("ROUTER_L2_SAMPLE_PPM")
            .or_else(|| {
                env_f64("ROUTER_L2_SAMPLE_RATIO").map(|v| {
                    (v.max(0.0).min(1.0) * 1_000_000.0).round() as u64
                })
            })
            .unwrap_or(300_000)
            .min(1_000_000) as u32;

        let mut min_ppm = env_f64("ROUTER_L2_SAMPLE_MIN_RATIO")
            .map(|v| (v.max(0.0).min(1.0) * 1_000_000.0).round() as u32)
            .unwrap_or_else(|| ((base_ppm as f64) * 0.1).round() as u32);
        if min_ppm > base_ppm {
            min_ppm = base_ppm;
        }

        let sample_dyn_enable = env_u64("ROUTER_L2_DYN_ENABLE").unwrap_or(1) != 0;

        let clamp01 = |v: f64| if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
        let sample_waterline_target =
            clamp01(env_f64("ROUTER_L2_WATERLINE_TARGET").unwrap_or(0.60));
        let mut sample_waterline_hi = clamp01(env_f64("ROUTER_L2_WATERLINE_HI").unwrap_or(0.85));
        let mut sample_waterline_lo = clamp01(env_f64("ROUTER_L2_WATERLINE_LO").unwrap_or(0.40));
        if sample_waterline_lo > sample_waterline_hi {
            std::mem::swap(&mut sample_waterline_lo, &mut sample_waterline_hi);
        }

        metrics::gauge!("router_l2_sample_ratio")
            .set(base_ppm as f64 / 1_000_000.0);

        Self {
            max_triggers_per_sec,
            rate_budget,
            max_queue_waterline,
            min_remaining_us,
            queue_wait_budget_us,
            sample_base_ppm: base_ppm,
            sample_min_ppm: min_ppm,
            sample_dyn_ppm: AtomicU64::new(base_ppm as u64),
            sample_dyn_enable,
            sample_last_update_ms: AtomicU64::new(0),
            sample_update_ctr: AtomicU64::new(0),
            sample_waterline_target,
            sample_waterline_hi,
            sample_waterline_lo,
            waterline_cached_bits: AtomicU64::new(0.0f64.to_bits()),
            waterline_last_update_ms: AtomicU64::new(0),
            waterline_update_ctr: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    #[inline]
    fn allow_by_rate(&self) -> bool {
        match &self.rate_budget {
            Some(b) => b.try_acquire(),
            None => true,
        }
    }

    #[inline]
    fn sample_ppm(&self) -> u32 {
        if self.sample_dyn_enable {
            self.sample_dyn_ppm.load(Ordering::Relaxed).min(1_000_000) as u32
        } else {
            self.sample_base_ppm
        }
    }

    #[inline]
    fn allow_by_sample(&self) -> bool {
        let ppm = self.sample_ppm();
        if ppm >= 1_000_000 {
            return true;
        }
        if ppm == 0 {
            return false;
        }
        (tls_rand_u64() % 1_000_000) < (ppm as u64)
    }

    #[inline]
    fn sample_ratio(&self) -> f64 {
        self.sample_ppm() as f64 / 1_000_000.0
    }

    #[inline]
    fn sample_base_ratio(&self) -> f64 {
        self.sample_base_ppm as f64 / 1_000_000.0
    }

    #[inline]
    fn sample_dyn_ratio(&self) -> f64 {
        self.sample_dyn_ppm.load(Ordering::Relaxed).min(1_000_000) as f64 / 1_000_000.0
    }

    #[inline]
    fn waterline_cached_or_update<F>(&self, fetch: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        const UPDATE_EVERY: u64 = 1024;
        const UPDATE_INTERVAL_MS: u64 = 200;

        let cnt = self.waterline_update_ctr.fetch_add(1, Ordering::Relaxed) + 1;
        let now_ms = self.start.elapsed().as_millis() as u64;
        let last = self.waterline_last_update_ms.load(Ordering::Relaxed);
        if (cnt % UPDATE_EVERY != 0) && now_ms.saturating_sub(last) < UPDATE_INTERVAL_MS {
            return f64::from_bits(self.waterline_cached_bits.load(Ordering::Relaxed));
        }
        if self
            .waterline_last_update_ms
            .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return f64::from_bits(self.waterline_cached_bits.load(Ordering::Relaxed));
        }
        self.waterline_update_ctr.store(0, Ordering::Relaxed);

        let waterline = fetch();
        self.waterline_cached_bits
            .store(waterline.to_bits(), Ordering::Relaxed);
        self.feedback_waterline(waterline);
        waterline
    }

    fn feedback_overload(&self) {
        if !self.sample_dyn_enable {
            return;
        }
        let base = self.sample_base_ppm as f64;
        let min = self.sample_min_ppm as f64;
        let cur = self.sample_dyn_ppm.load(Ordering::Relaxed) as f64;
        let mut next = cur * 0.7;
        if next < min {
            next = min;
        }
        if next > base {
            next = base;
        }
        let next_ppm = next.round() as u64;
        let cur_ppm = cur.round() as u64;
        if next_ppm != cur_ppm {
            self.sample_dyn_ppm.store(next_ppm, Ordering::Relaxed);
            metrics::gauge!("router_l2_sample_ratio").set(next / 1_000_000.0);
        }
        crate::batched_counter!("router_l2_feedback_overload_total").increment(1);
    }

    fn feedback_waterline(&self, waterline: f64) {
        if !self.sample_dyn_enable {
            return;
        }

        const UPDATE_EVERY: u64 = 1024;
        const UPDATE_INTERVAL_MS: u64 = 200;

        let cnt = self.sample_update_ctr.fetch_add(1, Ordering::Relaxed) + 1;
        let now_ms = self.start.elapsed().as_millis() as u64;
        let last = self.sample_last_update_ms.load(Ordering::Relaxed);
        if (cnt % UPDATE_EVERY != 0) && now_ms.saturating_sub(last) < UPDATE_INTERVAL_MS {
            return;
        }
        if self
            .sample_last_update_ms
            .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        self.sample_update_ctr.store(0, Ordering::Relaxed);

        let base = self.sample_base_ppm as f64;
        let min = self.sample_min_ppm as f64;
        let cur = self.sample_dyn_ppm.load(Ordering::Relaxed) as f64;
        let mut next = cur;

        if waterline > self.sample_waterline_hi {
            next = cur * 0.85;
        } else if waterline < self.sample_waterline_lo {
            next = (cur * 1.03).min(base);
        } else {
            next = cur + (base - cur) * 0.01;
        }

        if next < min {
            next = min;
        }
        if next > base {
            next = base;
        }

        let next_ppm = next.round() as u64;
        let cur_ppm = cur.round() as u64;
        if next_ppm != cur_ppm {
            if next > cur {
                crate::batched_counter!("router_l2_feedback_relax_total").increment(1);
            }
            self.sample_dyn_ppm.store(next_ppm, Ordering::Relaxed);
            metrics::gauge!("router_l2_sample_ratio").set(next / 1_000_000.0);
        }
    }
}

#[derive(Clone)]
pub struct L2ControlView {
    inner: Arc<L2Control>,
}

impl L2ControlView {
    #[inline]
    pub fn allow_by_rate(&self) -> bool {
        self.inner.allow_by_rate()
    }

    #[inline]
    pub fn allow_by_sample(&self) -> bool {
        self.inner.allow_by_sample()
    }

    #[inline]
    pub fn sample_ratio(&self) -> f64 {
        self.inner.sample_ratio()
    }

    #[inline]
    pub fn sample_base_ratio(&self) -> f64 {
        self.inner.sample_base_ratio()
    }

    #[inline]
    pub fn sample_dyn_ratio(&self) -> f64 {
        self.inner.sample_dyn_ratio()
    }

    #[inline]
    pub fn waterline_cached_or_update<F>(&self, fetch: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        self.inner.waterline_cached_or_update(fetch)
    }

    #[inline]
    pub fn sample_waterline_target(&self) -> f64 {
        self.inner.sample_waterline_target
    }

    #[inline]
    pub fn sample_waterline_hi(&self) -> f64 {
        self.inner.sample_waterline_hi
    }

    #[inline]
    pub fn sample_waterline_lo(&self) -> f64 {
        self.inner.sample_waterline_lo
    }

    #[inline]
    pub fn feedback_overload(&self) {
        self.inner.feedback_overload();
    }

    #[inline]
    pub fn feedback_waterline(&self, waterline: f64) {
        self.inner.feedback_waterline(waterline);
    }

    #[inline]
    pub fn max_queue_waterline(&self) -> f64 {
        self.inner.max_queue_waterline
    }

    #[inline]
    pub fn min_remaining_us(&self) -> u64 {
        self.inner.min_remaining_us
    }

    #[inline]
    pub fn queue_wait_budget_us(&self) -> u64 {
        self.inner.queue_wait_budget_us
    }
}

#[inline]
fn record_serialize_metrics(resp: &mut ScoreResponse) {
    let ser_hist = crate::sampled_histogram!("stage_serialize_us");
    if ser_hist.enabled() {
        let t_ser = Instant::now();
        let _ = serde_json::to_vec(resp);
        resp.timings_us.serialize = now_us(t_ser);
        ser_hist.record(resp.timings_us.serialize as f64);
    }
}

#[inline]
fn record_e2e_metrics(t0: Instant) {
    crate::sampled_histogram!("e2e_us").record(now_us(t0) as f64);
}

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

    /// ✅ L2 stateful 增强器：当 L2 特征维度 > L1（L2 是 L1 前缀 + 额外特征）时启用。
    ///
    /// 典型用途：在 L2 阶段加入本地“有状态”特征（velocity / aggregation / graph），让 L2 和 L1 成为两种物种。
    pub stateful_l2: Option<Arc<StatefulL2Augmenter>>,

    // Router-side overload control for L2 triggering (budget / waterline / etc).
    l2_ctrl: Arc<L2Control>,
}

impl AppCore {
    pub fn new(cfg: Config) -> Self {
        let store = Arc::new(FeatureStore::new(cfg.win_60s, cfg.win_300s));
        let l2_ctrl = Arc::new(L2Control::from_env(&cfg));
        Self {
            cfg,
            store,
            models: Models::default(),
            xgb: None,
            xgb_l2: None,
            xgb_pool: None,
            xgb_pool_l2: None,
            stateful_l2: None,
            l2_ctrl,
        }
    }

    pub fn new_with_xgb(cfg: Config, model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model_dir = model_dir.as_ref();
        let xgb = Arc::new(XgbRuntime::load_from_dir_l1(model_dir)?);
        let l2_ctrl = Arc::new(L2Control::from_env(&cfg));

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
            stateful_l2: None,
            l2_ctrl,
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

        fn parse_u64_env(key: &str, default: u64) -> u64 {
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
        let l2_ctrl = Arc::new(L2Control::from_env(&cfg));

        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        // ---------- L1 ----------
        let l1_dir = l1_model_dir.as_ref();
        let xgb1 = Arc::new(XgbRuntime::load_from_dir_l1(l1_dir)?);

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
            let mut pool = XgbPool::new_with_pinning(
                model_path,
                feature_names,
                l1_threads,
                l1_cap,
                l1_warm,
                l1_pin,
            )?;

            // Admission-control backpressure (fail fast with 429) to keep client-visible tail stable.
            // Default: half of SLO budget (microseconds). Override with env:
            //   - XGB_L1_POOL_EARLY_REJECT_US
            //   - (fallback) XGB_POOL_EARLY_REJECT_US
            let default_early_us = (cfg.slo_p99_ms as u64)
                .saturating_mul(1000)
                .saturating_div(2)
                .max(0);
            let l1_early_us = std::env::var("XGB_L1_POOL_EARLY_REJECT_US")
                .ok()
                .and_then(|s| s.parse().ok())
                .or_else(|| {
                    std::env::var("XGB_POOL_EARLY_REJECT_US")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(default_early_us);
            pool.set_early_reject_pred_wait_us(l1_early_us);
            Some(Arc::new(pool))
        };

        // ---------- L2 (optional) ----------
        let mut xgb2: Option<Arc<XgbRuntime>> = None;
        let mut xgb_pool2: Option<Arc<XgbPool>> = None;

        if let Some(l2_dir) = l2_model_dir {
            let l2_dir = l2_dir.as_ref();
            let x = Arc::new(XgbRuntime::load_from_dir_l2(l2_dir)?);

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
            let mut pool = XgbPool::new_with_pinning_role(
                model_path,
                feature_names,
                l2_threads,
                l2_cap,
                l2_warm,
                l2_pin,
                true,
            )?;

            // L2 is slower; default to a smaller budget (quarter of SLO) unless overridden.
            let default_l2_early_us = (cfg.slo_p99_ms as u64)
                .saturating_mul(1000)
                .saturating_div(4)
                .max(0);
            let l2_early_us = parse_u64_env("XGB_L2_POOL_EARLY_REJECT_US", default_l2_early_us);
            pool.set_early_reject_pred_wait_us(l2_early_us);
            xgb2 = Some(x);
            xgb_pool2 = Some(Arc::new(pool));
        }

        let stateful_l2 = if let Some(ref x2) = xgb2 {
            StatefulL2Augmenter::try_new(&cfg, &xgb1.feature_names, &x2.feature_names)?
                .map(Arc::new)
        } else {
            None
        };

        Ok(Self {
            cfg,
            store,
            models: Models::default(),
            xgb: Some(xgb1),
            xgb_l2: xgb2,
            xgb_pool: xgb_pool1,
            xgb_pool_l2: xgb_pool2,
            stateful_l2,
            l2_ctrl,
        })
    }

    /// 启动预热：消掉第一次请求的冷启动尖刺（TLS booster 路径的 warmup）

    pub fn l2_ctrl(&self) -> L2ControlView {
        L2ControlView {
            inner: std::sync::Arc::clone(&self.l2_ctrl),
        }
    }

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
        crate::sampled_histogram!("stage_feature_us").record(timings.feature as f64);

        let t_l1 = Instant::now();
        let score1 = xgb1.predict_proba(&row1)?;
        timings.xgb = now_us(t_l1);
        crate::sampled_histogram!("stage_xgb_us").record(timings.xgb as f64);

        let mut score_final = score1;
        let mut decision: Decision = decision_from_str(xgb1.decide(score1));
        let mut reasons: Vec<ReasonItem> = vec![];

        let t_router = Instant::now();

        let l2_enabled = self.xgb_l2.is_some();

        if matches!(decision, Decision::ManualReview) && l2_enabled {
            let xgb2 = self.xgb_l2.as_ref().unwrap();

            let now = Instant::now();
            let remaining_us = deadline
                .checked_duration_since(now)
                .unwrap_or_else(|| Duration::from_micros(0))
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;

            if remaining_us < self.l2_ctrl.min_remaining_us {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                if now >= deadline {
                    crate::batched_counter!("router_timeout_before_l2_total").increment(1);
                } else {
                    crate::batched_counter!("router_l2_skipped_deadline_budget_total").increment(1);
                }
            } else if !self.l2_ctrl.allow_by_rate() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_rate_total").increment(1);
            } else if !self.l2_ctrl.allow_by_sample() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_sample_total").increment(1);
            } else {
                crate::batched_counter!("router_l2_trigger_total").increment(1);

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
                crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

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
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
            decision,
            reason: reasons,
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);

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
        crate::sampled_histogram!("stage_feature_us").record(timings.feature as f64);

        // L1：score（pool）
        let t_l1_total = Instant::now();
        let rx = pool1
            .try_submit(row1.clone(), 0)
            .map_err(|e| anyhow::Error::new(e))?;

        let out1 = rx
            .await
            .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

        timings.xgb = now_us(t_l1_total);
        crate::sampled_histogram!("stage_xgb_us").record(timings.xgb as f64);

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

            // --- L2 admission control ---
            let now = Instant::now();
            let remaining_us = deadline
                .checked_duration_since(now)
                .unwrap_or_else(|| Duration::from_micros(0))
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;

            // 1) deadline / remaining budget
            if remaining_us < self.l2_ctrl.min_remaining_us {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                if now >= deadline {
                    crate::batched_counter!("router_timeout_before_l2_total").increment(1);
                } else {
                    crate::batched_counter!("router_l2_skipped_deadline_budget_total").increment(1);
                }
            }
            // 2) per-second budget (rate limiter)
            else if !self.l2_ctrl.allow_by_rate() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_rate_total").increment(1);
            }
            // 3) sample gate
            else if !self.l2_ctrl.allow_by_sample() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_sample_total").increment(1);
            } else {
                // 4) queue waterline gate (avoid knee-cliff when L2 backlog grows)
                let waterline = self
                    .l2_ctrl
                    .waterline_cached_or_update(|| pool2.stats().queue_waterline());
                if self.l2_ctrl.max_queue_waterline < 1.0
                    && waterline >= self.l2_ctrl.max_queue_waterline
                {
                    crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                    crate::batched_counter!("router_l2_skipped_waterline_total").increment(1);
                } else {
                    // Only build row2 when we really plan to run L2.
                    let row2: Vec<f32> = xgb2.build_row(obj);

                    let t_l2_total = Instant::now();

                    // Optional queue-wait budget for L2 pool.
                    let mut q_budget_us = self.l2_ctrl.queue_wait_budget_us;
                    if q_budget_us > 0 {
                        q_budget_us = q_budget_us.min(remaining_us);
                    }

                    let rx2 = (if q_budget_us > 0 {
                        pool2.try_submit_with_budget(row2, 5, q_budget_us)
                    } else {
                        pool2.try_submit(row2, 5)
                    })
                    .map_err(|e| {
                        self.l2_ctrl.feedback_overload();
                        anyhow::Error::new(e)
                    })?;

                    // Count as triggered only after we successfully enqueued.
                    crate::batched_counter!("router_l2_trigger_total").increment(1);

                    match rx2.await {
                        Ok(inner) => match inner {
                            Ok(out2) => {
                                timings.l2 = now_us(t_l2_total);
                                crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

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
                            Err(e) => {
                                // If L2 missed its queue-wait budget, degrade to L1 result (200) rather than fail the whole request.
                                if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                                    match pe {
                                        XgbPoolError::DeadlineExceeded => {
                                            self.l2_ctrl.feedback_overload();
                                            timings.l2 = now_us(t_l2_total);
                                            crate::sampled_histogram!("stage_l2_us")
                                                .record(timings.l2 as f64);
                                            crate::batched_counter!("router_deadline_miss_total")
                                                .increment(1);
                                        }
                                        _ => return Err(e),
                                    }
                                } else {
                                    return Err(e);
                                }
                            }
                        },
                        Err(_) => {
                            // oneshot canceled（极少见）：回退到 L1 结果
                            crate::batched_counter!("router_deadline_miss_total").increment(1);
                        }
                    }
                }
            }
        } else if matches!(decision, Decision::ManualReview) {
            // 兼容：没有 L2 时，仍然用 L1 做一版解释（论文图也好看）
            let explain_budget = Duration::from_millis(((self.cfg.slo_p99_ms as u64) / 3).max(1));
            let now = Instant::now();
            if now + explain_budget > deadline {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
            } else {
                crate::batched_counter!("router_l2_trigger_total").increment(1);

                let t_l2_total = Instant::now();
                let rx2 = pool1
                    .try_submit(row1, 5)
                    .map_err(|e| anyhow::Error::new(e))?;

                let out2 = rx2
                    .await
                    .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

                timings.l2 = now_us(t_l2_total);
                crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

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
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
            decision,
            reason: reasons,
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);

        Ok(resp)
    }

    /// ✅ Dense 直传版本（async）
    /// - 调用方直接给 dense row（f32，顺序必须与 feature_names.json 一致）
    /// - 这条路径**不做 feature build**，用来把 JSON parse/Map lookup 的 CPU 和分配成本全部抹掉
    /// - 当前实现默认只跑 L1（如果你打开了 L2，也会在 decision=review 时跳过并计数）
    pub async fn score_xgb_pool_dense_async(
        &self,
        parse_us: u64,
        row1: Vec<f32>,
    ) -> anyhow::Result<ScoreResponse> {
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_millis(self.cfg.slo_p99_ms as u64);

        let mut timings = TimingsUs::default();
        timings.parse = parse_us;
        timings.feature = 0;
        crate::sampled_histogram!("stage_feature_us").record(0.0);

        let xgb1 = self
            .xgb
            .as_ref()
            .context("xgb not enabled: start server with --model-dir")?;
        let pool1 = self.xgb_pool.as_ref().context("xgb_pool not enabled")?;

        // 维度校验：避免客户端/模型不匹配时 silent corruption
        if row1.len() != xgb1.feature_names.len() {
            anyhow::bail!(
                "dense feature size mismatch: got {}, expected {}",
                row1.len(),
                xgb1.feature_names.len()
            );
        }

        // L1：score（pool）
        let t_l1_total = Instant::now();
        let rx = pool1
            .try_submit(row1, 0)
            .map_err(|e| anyhow::Error::new(e))?;

        let out1 = rx
            .await
            .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

        timings.xgb = now_us(t_l1_total);
        crate::sampled_histogram!("stage_xgb_us").record(timings.xgb as f64);

        let score1 = out1.score;
        let score_final = score1;
        let decision: Decision = decision_from_str(xgb1.decide(score1));

        let reasons: Vec<ReasonItem> = vec![];

        // router：当前 dense 版本默认不跑 L2（因为缺少 obj，无法按 L2 的 feature_names 重建 row2）
        let t_router = Instant::now();
        if matches!(decision, Decision::ManualReview) && self.xgb_l2.is_some() {
            crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
            crate::batched_counter!("router_l2_skipped_dense_unsupported_total").increment(1);

            // 如果已经接近 deadline，就把原因单独记一笔（论文里很好解释）
            let now = Instant::now();
            let remaining_us = deadline
                .checked_duration_since(now)
                .unwrap_or_else(|| Duration::from_micros(0))
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;
            if remaining_us < self.l2_ctrl.min_remaining_us {
                crate::batched_counter!("router_l2_skipped_deadline_budget_total").increment(1);
            }
        }
        timings.router = now_us(t_router);
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
            decision,
            reason: reasons,
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);

        Ok(resp)
    }

    /// Like `score_xgb_pool_dense_async`, but accepts a **little-endian raw f32 byte buffer**.
    ///
    /// This is designed for extreme throughput:
    /// - avoids JSON parse
    /// - avoids allocating / building `Vec<f32>` on the Tokio side
    /// - still keeps the XgbPool isolation + bounded-queue semantics
    pub async fn score_xgb_pool_dense_bytes_async(
        &self,
        parse_us: u64,
        row_bytes_le: Bytes,
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

        // 维度校验：避免客户端/模型不匹配时 silent corruption
        let l1_dim = xgb1.feature_names.len();
        let expected_len = l1_dim.saturating_mul(4);
        anyhow::ensure!(
            row_bytes_le.len() == expected_len,
            "dense bytes length mismatch: got={}, expected={} (dim={})",
            row_bytes_le.len(),
            expected_len,
            l1_dim
        );

        // (可选) stateful pre-L1：更新本地 history store，为 L2 构造额外特征做准备
        let mut stateful_ctx: Option<DenseStatefulCtx> = None;
        let mut feature_us: u64 = 0;
        if let Some(aug) = self.stateful_l2.as_ref() {
            let (ctx, t) = aug.pre_l1(&row_bytes_le);
            stateful_ctx = Some(ctx);
            feature_us = feature_us.saturating_add(t.pre_l1_us);
        }

        // L2 可能需要 payload 的 clone（Bytes clone 是 O(1)）
        let payload_for_l2 = row_bytes_le.clone();

        // L1：score（pool）
        let t_l1_total = Instant::now();
        let rx = pool1
            .try_submit_dense_bytes_le(row_bytes_le, l1_dim, 0)
            .map_err(|e| anyhow::Error::new(e))?;

        let out1 = match tokio::time::timeout(Duration::from_millis(self.cfg.slo_p99_ms as u64), rx)
            .await
        {
            Ok(Ok(v)) => v?,
            Ok(Err(_)) => anyhow::bail!(XgbPoolError::Canceled),
            Err(_) => anyhow::bail!(XgbPoolError::DeadlineExceeded),
        };

        timings.xgb = now_us(t_l1_total);
        crate::sampled_histogram!("stage_xgb_us").record(timings.xgb as f64);

        let score1 = out1.score;
        let mut score_final = score1;
        let mut decision: Decision = decision_from_str(xgb1.decide(score1));

        // router + (optional) L2
        let t_router = Instant::now();

        let l2_enabled = self.xgb_l2.is_some() && self.xgb_pool_l2.is_some();

        if matches!(decision, Decision::ManualReview) && l2_enabled {
            let xgb2 = self.xgb_l2.as_ref().unwrap();
            let pool2 = self.xgb_pool_l2.as_ref().unwrap();

            let now = Instant::now();
            let remaining_us = deadline
                .checked_duration_since(now)
                .unwrap_or_else(|| Duration::from_micros(0))
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;

            // 1) deadline / remaining budget
            if remaining_us < self.l2_ctrl.min_remaining_us {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                if now >= deadline {
                    crate::batched_counter!("router_timeout_before_l2_total").increment(1);
                } else {
                    crate::batched_counter!("router_l2_skipped_deadline_budget_total").increment(1);
                }
            }
            // 2) per-second budget (rate limiter)
            else if !self.l2_ctrl.allow_by_rate() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_rate_total").increment(1);
            }
            // 3) sample gate
            else if !self.l2_ctrl.allow_by_sample() {
                crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                crate::batched_counter!("router_l2_skipped_sample_total").increment(1);
            } else {
                // 4) queue waterline gate
                let waterline = self
                    .l2_ctrl
                    .waterline_cached_or_update(|| pool2.stats().queue_waterline());
                if self.l2_ctrl.max_queue_waterline < 1.0
                    && waterline >= self.l2_ctrl.max_queue_waterline
                {
                    crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                    crate::batched_counter!("router_l2_skipped_waterline_total").increment(1);
                } else {
                    // Build L2 payload only when we really plan to run L2.
                    let use_stateful = self.stateful_l2.is_some();

                    let t_l2_total = Instant::now();

                    // Optional queue-wait budget for L2 pool.
                    let mut q_budget_us = self.l2_ctrl.queue_wait_budget_us;
                    if q_budget_us > 0 {
                        q_budget_us = q_budget_us.min(remaining_us);
                    }

                    let rx2 = if use_stateful {
                        let aug = self.stateful_l2.as_ref().unwrap();
                        let ctx = stateful_ctx.as_ref().expect("stateful ctx missing");
                        let mut extra = [0.0f32; EXTRA_MAX];
                        extra[..aug.extra_dim()].copy_from_slice(&ctx.extra);
                        if q_budget_us > 0 {
                            pool2.try_submit_dense_bytes_le_extra_with_budget(
                                payload_for_l2,
                                l1_dim,
                                extra,
                                aug.extra_dim(),
                                0,
                                q_budget_us,
                            )
                        } else {
                            pool2.try_submit_dense_bytes_le_extra(
                                payload_for_l2,
                                l1_dim,
                                extra,
                                aug.extra_dim(),
                                0,
                            )
                        }
                    } else if q_budget_us > 0 {
                        pool2.try_submit_dense_bytes_le_with_budget(
                            payload_for_l2,
                            xgb2.feature_names.len(),
                            0,
                            q_budget_us,
                        )
                    } else {
                        pool2.try_submit_dense_bytes_le(payload_for_l2, xgb2.feature_names.len(), 0)
                    }
                    .map_err(|e| anyhow::Error::new(e));

                    match rx2 {
                        Ok(rx2) => {
                            // Count as triggered only after we successfully enqueued.
                            crate::batched_counter!("router_l2_trigger_total").increment(1);

                            let rem = deadline.saturating_duration_since(Instant::now());
                            match tokio::time::timeout(rem, rx2).await {
                                Ok(Ok(inner)) => match inner {
                                    Ok(out2) => {
                                        timings.l2 = now_us(t_l2_total);
                                        crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

                                        score_final = out2.score;
                                        decision = decision_from_str(xgb2.decide(out2.score));
                                    }
                                    Err(e) => {
                                        // If L2 missed its queue-wait budget, degrade to L1 result rather than fail the request.
                                        if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                                            if matches!(pe, XgbPoolError::DeadlineExceeded) {
                                                self.l2_ctrl.feedback_overload();
                                                timings.l2 = now_us(t_l2_total);
                                                crate::sampled_histogram!("stage_l2_us")
                                                    .record(timings.l2 as f64);
                                                crate::batched_counter!("router_deadline_miss_total")
                                                    .increment(1);
                                            } else {
                                                return Err(e);
                                            }
                                        } else {
                                            return Err(e);
                                        }
                                    }
                                },
                                Ok(Err(_)) => {
                                    // oneshot canceled: degrade to L1
                                    crate::batched_counter!("router_deadline_miss_total").increment(1);
                                }
                                Err(_) => {
                                    // timeout: degrade to L1
                                    self.l2_ctrl.feedback_overload();
                                    timings.l2 = now_us(t_l2_total);
                                    crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);
                                    crate::batched_counter!("router_deadline_miss_total").increment(1);
                                }
                            }
                        }
                        Err(_submit_err) => {
                            // Could not enqueue L2; degrade to L1 decision.
                            self.l2_ctrl.feedback_overload();
                            crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                        }
                    }
                }
            }
        }

        timings.feature = feature_us;
        crate::sampled_histogram!("stage_feature_us").record(feature_us as f64);

        timings.router = now_us(t_router);
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: score_final as f64,
            decision,
            reason: vec![],
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);

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
        crate::sampled_histogram!("stage_feature_us").record(timings.feature as f64);

        // router + L1
        let tr0 = Instant::now();
        let tl10 = Instant::now();
        let s1: f64 = self.models.l1.score(&feats) as f64;
        timings.xgb = now_us(tl10);
        crate::sampled_histogram!("stage_xgb_us").record(timings.xgb as f64);

        // optional L2（预算式）
        let mut score: f64 = s1;
        let mut used_l2 = false;
        crate::batched_counter!("router_l2_trigger_total").increment(0);
        crate::batched_counter!("router_l2_skipped_budget_total").increment(0);
        if s1 > self.cfg.l1_uncertain_low as f64 && s1 < self.cfg.l1_uncertain_high as f64 {
            let now = Instant::now();
            if now < deadline {
                let remaining = deadline.duration_since(now);
                if remaining > Duration::from_millis(1) {
                    let tl20 = Instant::now();
                    let s2: f64 = self.models.l2.score(&feats) as f64;
                    timings.l2 = now_us(tl20);
                    crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

                    score = s2;
                    used_l2 = true;
                    crate::batched_counter!("router_l2_trigger_total").increment(1);
                } else {
                    crate::batched_counter!("router_l2_skipped_budget_total").increment(1);
                }
            } else {
                crate::batched_counter!("router_timeout_before_l2_total").increment(1);
            }
        }

        timings.router = now_us(tr0);
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        // decision
        let decision = if Instant::now() > deadline {
            crate::batched_counter!("router_deadline_miss_total").increment(1);
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
        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);

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
