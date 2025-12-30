use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use crossbeam_channel as cb;
use futures_channel::oneshot;

use xgb_ffi::Booster;

/// 队列满 / 断开 / worker 返回错误等（用于 server 侧 downcast -> 429）
#[derive(Debug, Clone)]
pub enum XgbPoolError {
    QueueFull,
    Disconnected,
    Canceled,
    Worker(String),
}

impl fmt::Display for XgbPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XgbPoolError::QueueFull => write!(f, "xgb pool queue full"),
            XgbPoolError::Disconnected => write!(f, "xgb pool disconnected"),
            XgbPoolError::Canceled => write!(f, "xgb pool response canceled"),
            XgbPoolError::Worker(s) => write!(f, "xgb worker error: {s}"),
        }
    }
}
impl std::error::Error for XgbPoolError {}

#[derive(Debug, Clone)]
pub struct XgbOut {
    pub score: f32,
    pub queue_wait_us: u64,
    pub xgb_us: u64,
    pub contrib_topk: Option<Vec<(String, f32)>>,
}

struct XgbJob {
    row: Vec<f32>,
    topk: usize,
    enq: Instant,
    tx: oneshot::Sender<Result<XgbOut, XgbPoolError>>,
}

pub struct XgbPool {
    tx: cb::Sender<XgbJob>,
    q_depth: Arc<AtomicUsize>,
    cap: usize,
    feature_names: Arc<Vec<String>>,
    _joins: Arc<Vec<thread::JoinHandle<()>>>,
}

#[cfg(target_os = "linux")]
fn parse_cpuset_env(name: &str) -> Option<Vec<usize>> {
    let s = std::env::var(name).ok()?;
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Ok(v) = p.parse::<usize>() {
            out.push(v);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(target_os = "linux")]
fn default_even_cpus() -> Vec<usize> {
    // 7945HX/WSL2 常见 CPU 列表：0,1 是同一核 SMT；2,3 同一核；...
    // 默认优先用偶数 CPU，减少 SMT 互抢
    let mut v = Vec::new();
    if let Some(ids) = core_affinity::get_core_ids() {
        for c in ids {
            let id = c.id;
            if id % 2 == 0 {
                v.push(id);
            }
        }
    }
    if v.is_empty() {
        v.push(0);
    }
    v
}

#[cfg(target_os = "linux")]
fn pin_current_thread(wid: usize) {
    let cpus = parse_cpuset_env("XGB_POOL_CPUS").unwrap_or_else(default_even_cpus);
    let pick = cpus[wid % cpus.len()];

    if let Some(ids) = core_affinity::get_core_ids() {
        if let Some(core) = ids.into_iter().find(|c| c.id == pick) {
            core_affinity::set_for_current(core);
            if std::env::var("XGB_POOL_AFFINITY_DEBUG").ok().as_deref() == Some("1") {
                eprintln!("[xgb-worker-{wid}] pinned to cpu={pick}");
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_wid: usize) {}

impl XgbPool {
    /// - workers: OS 线程数（每线程一份 Booster）
    /// - cap: 有界队列容量（满了直接 429）
    /// - warmup_iters: 每个 worker 启动后做多少次 dummy predict（消掉冷启动尖刺）
    pub fn new(
        model_path: PathBuf,
        feature_names: Arc<Vec<String>>,
        workers: usize,
        cap: usize,
        warmup_iters: usize,
    ) -> anyhow::Result<Self> {
        if workers == 0 {
            anyhow::bail!("xgb pool workers must be > 0");
        }
        if cap == 0 {
            anyhow::bail!("xgb pool cap must be > 0");
        }
        ensure_exists(&model_path)?;

        let (tx, rx) = cb::bounded::<XgbJob>(cap);
        let q_depth = Arc::new(AtomicUsize::new(0));

        // worker 启动回执：每个线程加载+warmup 完成后发一个 Ok(())
        // 如果加载失败发 Err(msg)，new() 直接返回错误，避免“启动成功但池不可用”
        let (ready_tx, ready_rx) = cb::bounded::<Result<(), String>>(workers);

        let mut joins = Vec::with_capacity(workers);
        for wid in 0..workers {
            let rx = rx.clone();
            let mp = model_path.clone();
            let fnames = feature_names.clone();
            let qd = q_depth.clone();
            let rtx = ready_tx.clone();

            let j = thread::Builder::new()
                .name(format!("xgb-worker-{wid}"))
                .spawn(move || worker_loop(wid, rx, mp, fnames, qd, cap, warmup_iters, rtx))
                .context("spawn xgb worker")?;

            joins.push(j);
        }

        // 等待所有 worker ready（避免压测初期冷启动把队列顶满造成 dropped）
        for _ in 0..workers {
            match ready_rx.recv_timeout(Duration::from_secs(30)) {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => anyhow::bail!("xgb pool worker init failed: {msg}"),
                Err(cb::RecvTimeoutError::Timeout) => {
                    anyhow::bail!("xgb pool init timeout waiting workers ready")
                }
                Err(cb::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("xgb pool init failed: ready channel disconnected")
                }
            }
        }

        Ok(Self {
            tx,
            q_depth,
            cap,
            feature_names,
            _joins: Arc::new(joins),
        })
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn q_depth(&self) -> usize {
        self.q_depth.load(Ordering::Relaxed)
    }

    /// try_submit：满了直接 QueueFull（用于 429）
    pub fn try_submit(
        &self,
        row: Vec<f32>,
        topk: usize,
    ) -> Result<oneshot::Receiver<Result<XgbOut, XgbPoolError>>, XgbPoolError> {
        // 用原子 q_depth 做快速拒绝（严格性由 bounded channel 保证）
        let mut cur = self.q_depth.load(Ordering::Relaxed);
        loop {
            if cur >= self.cap {
                metrics::counter!("xgb_pool_reject_full_total").increment(1);
                return Err(XgbPoolError::QueueFull);
            }
            match self.q_depth.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }

        let (tx, rx) = oneshot::channel();
        let job = XgbJob {
            row,
            topk,
            enq: Instant::now(),
            tx,
        };

        match self.tx.try_send(job) {
            Ok(()) => {
                metrics::counter!("xgb_pool_submit_total").increment(1);
                metrics::gauge!("xgb_pool_q_depth").set(self.q_depth() as f64);
                Ok(rx)
            }
            Err(cb::TrySendError::Full(_)) => {
                // rollback
                self.q_depth.fetch_sub(1, Ordering::Relaxed);
                metrics::counter!("xgb_pool_reject_full_total").increment(1);
                metrics::gauge!("xgb_pool_q_depth").set(self.q_depth() as f64);
                Err(XgbPoolError::QueueFull)
            }
            Err(cb::TrySendError::Disconnected(_)) => {
                self.q_depth.fetch_sub(1, Ordering::Relaxed);
                metrics::gauge!("xgb_pool_q_depth").set(self.q_depth() as f64);
                Err(XgbPoolError::Disconnected)
            }
        }
    }

    /// contrib topk 的默认实现（用于 debug 或测试）
    pub fn topk_from_contrib_row_with_bias(
        &self,
        contrib_row_with_bias: &[f32],
        k: usize,
    ) -> Vec<(String, f32)> {
        topk_contrib(&self.feature_names, contrib_row_with_bias, k)
    }
}

fn ensure_exists(p: &Path) -> anyhow::Result<()> {
    std::fs::metadata(p).with_context(|| format!("model_path not found: {}", p.display()))?;
    Ok(())
}

fn worker_loop(
    wid: usize,
    rx: cb::Receiver<XgbJob>,
    model_path: PathBuf,
    feature_names: Arc<Vec<String>>,
    q_depth: Arc<AtomicUsize>,
    cap: usize,
    warmup_iters: usize,
    ready_tx: cb::Sender<Result<(), String>>,
) {
    // ✅ 0) 先绑核（减少迁移导致的推理抖动）
    pin_current_thread(wid);

    // 1) load model（每线程一份 Booster）
    let booster = match Booster::load_model(&model_path) {
        Ok(b) => b,
        Err(e) => {
            let _ = ready_tx.send(Err(format!(
                "[worker-{wid}] failed to load model {}: {e}",
                model_path.display()
            )));
            return;
        }
    };

    // ✅ 1.5) 强制单线程推理（1-row 推理非常关键）
    let _ = booster.set_param("nthread", "1");
    let _ = booster.set_param("predictor", "cpu_predictor");

    // 2) warmup（每线程）
    //   - clamp：避免 warmup_iters 设置过大导致启动极慢
    //   - row 用 0.0：更温和，也能触发内部 lazy init
    let iters = warmup_iters.min(50);
    if iters > 0 {
        let n = feature_names.len().max(1);
        let row = vec![0.0f32; n];
        let t = Instant::now();
        for _ in 0..iters {
            let _ = booster.predict_proba_dense_1row(&row);
        }
        let _ = booster.predict_contribs_dense_1row(&row);
        eprintln!(
            "[xgb-worker-{wid}] warmup done iters={iters} cost={}ms",
            t.elapsed().as_millis()
        );
    } else {
        eprintln!("[xgb-worker-{wid}] warmup skipped");
    }

    // 3) 通知主线程：我 ready 了
    let _ = ready_tx.send(Ok(()));

    // 4) 正常处理 job
    while let Ok(job) = rx.recv() {
        // 取走一个请求 -> q_depth-1
        let prev = q_depth.fetch_sub(1, Ordering::Relaxed);
        if prev == 0 {
            q_depth.store(0, Ordering::Relaxed);
        }
        metrics::gauge!("xgb_pool_q_depth").set(q_depth.load(Ordering::Relaxed) as f64);

        let queue_wait_us = job.enq.elapsed().as_micros() as u64;
        metrics::histogram!("xgb_pool_queue_wait_us").record(queue_wait_us as f64);

        let t_xgb = Instant::now();
        let score = match booster.predict_proba_dense_1row(&job.row) {
            Ok(p) => p,
            Err(e) => {
                let _ = job.tx.send(Err(XgbPoolError::Worker(e)));
                metrics::counter!("xgb_pool_done_err_total").increment(1);
                continue;
            }
        };

        let mut contrib_topk_out: Option<Vec<(String, f32)>> = None;
        if job.topk > 0 {
            match booster.predict_contribs_dense_1row(&job.row) {
                Ok(out) => {
                    let contrib_row_with_bias: Vec<f32> = match out.shape.as_slice() {
                        [1, _groups, m] => out.values[..*m].to_vec(),
                        [1, m] => out.values[..*m].to_vec(),
                        _ => {
                            let _ = job.tx.send(Err(XgbPoolError::Worker(format!(
                                "unexpected contrib shape: {:?}",
                                out.shape
                            ))));
                            metrics::counter!("xgb_pool_done_err_total").increment(1);
                            continue;
                        }
                    };
                    contrib_topk_out = Some(topk_contrib(
                        &feature_names,
                        &contrib_row_with_bias,
                        job.topk,
                    ));
                }
                Err(e) => {
                    let _ = job.tx.send(Err(XgbPoolError::Worker(e)));
                    metrics::counter!("xgb_pool_done_err_total").increment(1);
                    continue;
                }
            }
        }

        let xgb_us = t_xgb.elapsed().as_micros() as u64;
        metrics::histogram!("xgb_pool_xgb_compute_us").record(xgb_us as f64);

        let out = XgbOut {
            score,
            queue_wait_us,
            xgb_us,
            contrib_topk: contrib_topk_out,
        };

        let _ = job.tx.send(Ok(out));
        metrics::counter!("xgb_pool_done_ok_total").increment(1);

        // 健康性：cap 的边界提醒（debug 用）
        let q = q_depth.load(Ordering::Relaxed);
        if q > cap * 2 {
            metrics::counter!("xgb_pool_q_depth_weird_total").increment(1);
        }
    }

    // channel 断开
    let _ = ready_tx.send(Err(format!("[worker-{wid}] rx disconnected")));
}

fn topk_contrib(
    feature_names: &Arc<Vec<String>>,
    contrib_row_with_bias: &[f32],
    k: usize,
) -> Vec<(String, f32)> {
    if k == 0 || contrib_row_with_bias.is_empty() {
        return vec![];
    }

    let m = contrib_row_with_bias.len();
    let feat_maybe_bias = m.saturating_sub(1);

    let mut top: Vec<(usize, f32, f32)> = Vec::with_capacity(feat_maybe_bias);
    for idx in 0..feat_maybe_bias {
        let v = contrib_row_with_bias[idx];
        top.push((idx, v, v.abs()));
    }

    // 按 abs 降序
    top.sort_by(|a, b| b.2.total_cmp(&a.2));
    top.truncate(k);

    top.into_iter()
        .map(|(idx, v, _)| {
            let name = feature_names
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("f{idx}"));
            (name, v)
        })
        .collect()
}
