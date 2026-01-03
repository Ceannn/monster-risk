//! A dedicated inference pool for XGBoost.
//!
//! Why this exists:
//! - XGBoost C-API is blocking and can be CPU heavy; running it on Tokio workers
//!   causes head-of-line blocking and latency spikes.
//! - We instead create N dedicated OS threads, each owning its own Booster.
//! - Submission is non-blocking (`try_submit`); when all queues are full we fail fast.
//!
//! This file intentionally keeps the public API tiny and stable.

use anyhow::Context;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use xgb_ffi::Booster;

#[derive(Debug)]
pub enum XgbPoolError {
    QueueFull,
    WorkerDown,
    Canceled,
    DeadlineExceeded,
}

impl std::fmt::Display for XgbPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XgbPoolError::QueueFull => write!(f, "xgb pool queue full"),
            XgbPoolError::WorkerDown => write!(f, "xgb pool worker down"),
            XgbPoolError::Canceled => write!(f, "xgb pool job canceled"),
            XgbPoolError::DeadlineExceeded => write!(f, "xgb pool deadline exceeded"),
        }
    }
}
impl std::error::Error for XgbPoolError {}

#[derive(Debug, Clone)]
pub struct XgbOut {
    pub score: f32,
    pub queue_wait_us: u64,
    pub xgb_us: u64,
    pub contrib_topk: Vec<(String, f32)>,
}

struct XgbJob {
    enq_at: Instant,
    deadline_at: Option<Instant>,
    row: Vec<f32>,
    contrib_topk: usize,
    resp_tx: oneshot::Sender<anyhow::Result<XgbOut>>,
}

struct WorkerHandle {
    tx: mpsc::Sender<XgbJob>,
}

#[derive(Debug, Clone, Copy)]
pub struct XgbPoolStats {
    pub n_workers: usize,
    /// Total queue capacity across all workers (rounded up by per-worker split).
    pub cap_total: usize,
    /// Jobs currently waiting in worker queues (does not include running jobs).
    pub queued: usize,
    /// Jobs currently executing inside workers.
    pub running: usize,
}

impl XgbPoolStats {
    #[inline]
    pub fn inflight(&self) -> usize {
        self.queued + self.running
    }

    #[inline]
    pub fn queue_waterline(&self) -> f64 {
        if self.cap_total == 0 {
            0.0
        } else {
            (self.queued as f64) / (self.cap_total as f64)
        }
    }
}

/// Decrements `running` when dropped.
struct RunningGuard(Arc<AtomicUsize>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct XgbPool {
    workers: Vec<WorkerHandle>,
    rr: AtomicUsize,

    cap_total: usize,
    queued: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,
}

impl XgbPool {
    /// Build an inference pool.
    ///
    /// - `model_path`: path to the model file (e.g. ieee_xgb.ubj / xgb_model.json / ieee_xgb.bin)
    /// - `feature_names`: feature order used by the model (for contrib top-k naming)
    /// - `n_workers`: number of dedicated XGB threads
    /// - `queue_cap_total`: total queue capacity across all workers (will be evenly split)
    /// - `warmup_iters`: per-worker warmup iterations to eliminate cold-start spikes
    pub fn new(
        model_path: impl AsRef<Path>,
        feature_names: Arc<Vec<String>>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_pinning(
            model_path,
            feature_names,
            n_workers,
            queue_cap_total,
            warmup_iters,
            None,
        )
    }

    /// Same as [`XgbPool::new`], but pins each worker thread to a specific CPU id (linux only).
    ///
    /// `pin_cpus`: optional CPU ids to pin each worker to (extra ids ignored).
    pub fn new_with_pinning(
        model_path: impl AsRef<Path>,
        feature_names: Arc<Vec<String>>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
        pin_cpus: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(n_workers > 0, "xgb pool n_workers must be > 0");
        anyhow::ensure!(queue_cap_total > 0, "xgb pool queue_cap_total must be > 0");

        // Split total cap across workers to keep admission-control semantics intuitive.
        // Example: n_workers=8, queue_cap_total=512 => per_worker_cap=64.
        let per_worker_cap = (queue_cap_total + n_workers - 1) / n_workers;
        let cap_total = per_worker_cap * n_workers;

        let model_path = Arc::new(model_path.as_ref().to_path_buf());

        // Track queue/running counts for admission control & debugging (best-effort).
        let queued = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::with_capacity(n_workers);
        let pin_cpus = pin_cpus.unwrap_or_default();

        for wid in 0..n_workers {
            let (tx, mut rx) = mpsc::channel::<XgbJob>(per_worker_cap);
            let model_path = Arc::clone(&model_path);
            let feature_names = Arc::clone(&feature_names);
            let cpu = pin_cpus.get(wid).copied();
            let warmup_iters = warmup_iters;
            let queued = Arc::clone(&queued);
            let running = Arc::clone(&running);

            thread::Builder::new()
                .name(format!("xgb-worker-{wid}"))
                .spawn(move || {
                    if let Some(cpu) = cpu {
                        #[cfg(target_os = "linux")]
                        {
                            if let Err(e) = pin_current_thread(cpu) {
                                eprintln!("xgb-worker-{wid}: failed to pin to cpu {cpu}: {e}");
                            } else {
                                eprintln!("xgb-worker-{wid}: pinned to cpu={cpu}");
                            }
                        }
                    }

                    let bst = match Booster::load_model(model_path.as_ref()) {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("xgb-worker-{wid}: failed to load model: {e}");
                            return;
                        }
                    };

                    // Warmup: initialize thread-local buffers & JIT-ish internals.
                    if warmup_iters > 0 {
                        let ncols = feature_names.len();
                        let row = vec![f32::NAN; ncols];
                        for _ in 0..warmup_iters {
                            let _ = bst.predict_proba_dense_1row(&row);
                        }
                        // Best-effort warmup for contrib path (usually colder).
                        let _ = bst.predict_contribs_dense_1row(&row);
                        eprintln!("xgb-worker-{wid}: warmup done iters={warmup_iters}");
                    }

                    // Main loop
                    while let Some(job) = rx.blocking_recv() {
                        // Best-effort counters (admission control / debug).
                        queued.fetch_sub(1, Ordering::Relaxed);
                        running.fetch_add(1, Ordering::Relaxed);
                        let _running_guard = RunningGuard(Arc::clone(&running));

                        let now = Instant::now();
                        let queue_wait_us =
                            now.duration_since(job.enq_at)
                                .as_micros()
                                .min(u128::from(u64::MAX)) as u64;

                        metrics::histogram!("xgb_pool_queue_wait_us").record(queue_wait_us as f64);

                        if let Some(deadline_at) = job.deadline_at {
                            if now >= deadline_at {
                                metrics::counter!("xgb_pool_deadline_miss_total").increment(1);
                                let _ = job
                                    .resp_tx
                                    .send(Err(anyhow::Error::new(XgbPoolError::DeadlineExceeded)));
                                continue;
                            }
                        }

                        let t0 = Instant::now();
                        let score = match bst.predict_proba_dense_1row(&job.row) {
                            Ok(s) => s,
                            Err(e) => {
                                let _ = job.resp_tx.send(Err(anyhow::anyhow!(e)));
                                continue;
                            }
                        };

                        let mut contrib_topk = Vec::new();
                        if job.contrib_topk > 0 {
                            match bst.predict_contribs_dense_1row(&job.row) {
                                Ok(out) => {
                                    // Usually len = ncols + 1 (bias). We ignore bias for topk.
                                    let n = feature_names.len().min(out.values.len());
                                    contrib_topk = topk_abs_named(
                                        &out.values[..n],
                                        &feature_names,
                                        job.contrib_topk,
                                    );
                                }
                                Err(e) => {
                                    let _ = job.resp_tx.send(Err(anyhow::anyhow!(e)));
                                    continue;
                                }
                            }
                        }

                        let xgb_us = t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                        metrics::histogram!("xgb_pool_xgb_compute_us").record(xgb_us as f64);

                        let _ = job.resp_tx.send(Ok(XgbOut {
                            score,
                            queue_wait_us,
                            xgb_us,
                            contrib_topk,
                        }));
                    }
                })
                .context("spawn xgb worker")?;

            workers.push(WorkerHandle { tx });
        }

        Ok(Self {
            workers,
            rr: AtomicUsize::new(0),
            cap_total,
            queued,
            running,
        })
    }

    /// Convenience: build a pool directly from a model directory (same format as `XgbRuntime::load_from_dir`).
    pub fn new_from_dir(
        model_dir: impl AsRef<Path>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
        pin_cpus: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        let rt = crate::xgb_runtime::XgbRuntime::load_from_dir(model_dir.as_ref())
            .with_context(|| format!("load model dir: {}", model_dir.as_ref().display()))?;
        Self::new_with_pinning(
            rt.model_path,
            Arc::new(rt.feature_names),
            n_workers,
            queue_cap_total,
            warmup_iters,
            pin_cpus,
        )
    }

    /// Snapshot stats for router-side admission control / debugging.
    #[inline]
    pub fn stats(&self) -> XgbPoolStats {
        XgbPoolStats {
            n_workers: self.workers.len(),
            cap_total: self.cap_total,
            queued: self.queued.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }

    /// Non-blocking submission. Returns a oneshot receiver to await the result.
    pub fn try_submit(
        &self,
        row: Vec<f32>,
        contrib_topk: usize,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        let n = self.workers.len();
        if n == 0 {
            return Err(XgbPoolError::WorkerDown);
        }

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let mut job = XgbJob {
            enq_at: Instant::now(),
            deadline_at: None,
            row,
            contrib_topk,
            resp_tx,
        };

        let start = self.rr.fetch_add(1, Ordering::Relaxed);
        let mut down = 0usize;

        for i in 0..n {
            let idx = (start + i) % n;
            // NOTE: increment before enqueue to avoid underflow race with worker-side fetch_sub.
            self.queued.fetch_add(1, Ordering::Relaxed);
            match self.workers[idx].tx.try_send(job) {
                Ok(()) => return Ok(resp_rx),
                Err(mpsc::error::TrySendError::Full(j)) => {
                    self.queued.fetch_sub(1, Ordering::Relaxed);
                    job = j;
                    continue;
                }
                Err(mpsc::error::TrySendError::Closed(j)) => {
                    self.queued.fetch_sub(1, Ordering::Relaxed);
                    job = j;
                    down += 1;
                    continue;
                }
            }
        }

        if down == n {
            Err(XgbPoolError::WorkerDown)
        } else {
            Err(XgbPoolError::QueueFull)
        }
    }

    /// Non-blocking submission with a queue-wait budget (microseconds).
    ///
    /// - `budget_us = 0` disables the budget.
    /// - If the job waits in the queue longer than `budget_us`, the worker will
    ///   reply with an `XgbPoolError::DeadlineExceeded` without running inference.
    pub fn try_submit_with_budget(
        &self,
        row: Vec<f32>,
        contrib_topk: usize,
        budget_us: u64,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        if budget_us == 0 {
            return self.try_submit(row, contrib_topk);
        }

        let n = self.workers.len();
        if n == 0 {
            return Err(XgbPoolError::WorkerDown);
        }

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();

        let enq_at = Instant::now();
        let deadline_at = enq_at + Duration::from_micros(budget_us);

        let mut job = XgbJob {
            enq_at,
            deadline_at: Some(deadline_at),
            row,
            contrib_topk,
            resp_tx,
        };

        let start = self.rr.fetch_add(1, Ordering::Relaxed);
        let mut down = 0usize;

        for i in 0..n {
            let idx = (start + i) % n;
            // NOTE: increment before enqueue to avoid underflow race with worker-side fetch_sub.
            self.queued.fetch_add(1, Ordering::Relaxed);
            match self.workers[idx].tx.try_send(job) {
                Ok(()) => return Ok(resp_rx),
                Err(mpsc::error::TrySendError::Full(j)) => {
                    self.queued.fetch_sub(1, Ordering::Relaxed);
                    job = j;
                    continue;
                }
                Err(mpsc::error::TrySendError::Closed(j)) => {
                    self.queued.fetch_sub(1, Ordering::Relaxed);
                    job = j;
                    down += 1;
                    continue;
                }
            }
        }

        if down == n {
            Err(XgbPoolError::WorkerDown)
        } else {
            Err(XgbPoolError::QueueFull)
        }
    }

    /// Convenience async wrapper with budget.
    pub async fn submit_with_budget(
        &self,
        row: Vec<f32>,
        contrib_topk: usize,
        budget_us: u64,
    ) -> Result<anyhow::Result<XgbOut>, XgbPoolError> {
        let rx = self.try_submit_with_budget(row, contrib_topk, budget_us)?;
        match rx.await {
            Ok(v) => Ok(v),
            Err(_) => Err(XgbPoolError::Canceled),
        }
    }

    /// Convenience async wrapper.
    pub async fn submit(
        &self,
        row: Vec<f32>,
        contrib_topk: usize,
    ) -> Result<anyhow::Result<XgbOut>, XgbPoolError> {
        let rx = self.try_submit(row, contrib_topk)?;
        match rx.await {
            Ok(v) => Ok(v),
            Err(_) => Err(XgbPoolError::Canceled),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct TopkItem {
    abs: f32,
    idx: usize,
    val: f32,
}

impl PartialEq for TopkItem {
    fn eq(&self, other: &Self) -> bool {
        self.abs.to_bits() == other.abs.to_bits() && self.idx == other.idx
    }
}
impl Eq for TopkItem {}

impl PartialOrd for TopkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopkItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.abs
            .total_cmp(&other.abs)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Select top-k contributions by absolute value, returning named pairs.
///
/// Complexity: O(n log k). This avoids allocating and sorting a full `Vec` of length `n`.
fn topk_abs_named(values: &[f32], feature_names: &[String], k: usize) -> Vec<(String, f32)> {
    if k == 0 || values.is_empty() {
        return Vec::new();
    }

    let n = values.len().min(feature_names.len());
    let k = k.min(n);

    let mut heap: BinaryHeap<Reverse<TopkItem>> = BinaryHeap::with_capacity(k + 1);
    for i in 0..n {
        let v = values[i];
        let item = TopkItem {
            abs: v.abs(),
            idx: i,
            val: v,
        };

        if heap.len() < k {
            heap.push(Reverse(item));
            continue;
        }

        if let Some(Reverse(min_item)) = heap.peek() {
            if item.abs > min_item.abs {
                let _ = heap.pop();
                heap.push(Reverse(item));
            }
        }
    }

    let mut out: Vec<TopkItem> = heap.into_iter().map(|Reverse(it)| it).collect();
    out.sort_by(|a, b| b.abs.total_cmp(&a.abs));

    let mut named = Vec::with_capacity(out.len());
    for it in out {
        named.push((feature_names[it.idx].clone(), it.val));
    }
    named
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> std::io::Result<()> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let tid = libc::pthread_self();
        let rc = libc::pthread_setaffinity_np(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) -> std::io::Result<()> {
    Ok(())
}
