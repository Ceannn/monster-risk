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
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use xgb_ffi::Booster;

#[derive(Debug)]
pub enum XgbPoolError {
    QueueFull,
    WorkerDown,
    Canceled,
}

impl std::fmt::Display for XgbPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XgbPoolError::QueueFull => write!(f, "xgb pool queue full"),
            XgbPoolError::WorkerDown => write!(f, "xgb pool worker down"),
            XgbPoolError::Canceled => write!(f, "xgb pool job canceled"),
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
    row: Vec<f32>,
    contrib_topk: usize,
    resp_tx: oneshot::Sender<anyhow::Result<XgbOut>>,
}

struct WorkerHandle {
    tx: mpsc::Sender<XgbJob>,
}

pub struct XgbPool {
    workers: Vec<WorkerHandle>,
    rr: AtomicUsize,
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

        let model_path = Arc::new(model_path.as_ref().to_path_buf());

        let mut workers = Vec::with_capacity(n_workers);
        let pin_cpus = pin_cpus.unwrap_or_default();

        for wid in 0..n_workers {
            let (tx, mut rx) = mpsc::channel::<XgbJob>(per_worker_cap);
            let model_path = Arc::clone(&model_path);
            let feature_names = Arc::clone(&feature_names);
            let cpu = pin_cpus.get(wid).copied();
            let warmup_iters = warmup_iters;

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
                        let now = Instant::now();
                        let queue_wait_us =
                            now.duration_since(job.enq_at)
                                .as_micros()
                                .min(u128::from(u64::MAX)) as u64;

                        metrics::histogram!("xgb_pool_queue_wait_us").record(queue_wait_us as f64);

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
                                    let mut tmp: Vec<(usize, f32)> =
                                        (0..n).map(|i| (i, out.values[i])).collect();
                                    tmp.sort_by(|a, b| {
                                        b.1.abs()
                                            .partial_cmp(&a.1.abs())
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                    contrib_topk = tmp
                                        .into_iter()
                                        .take(job.contrib_topk.min(n))
                                        .map(|(i, v)| (feature_names[i].clone(), v))
                                        .collect();
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
            row,
            contrib_topk,
            resp_tx,
        };

        let start = self.rr.fetch_add(1, Ordering::Relaxed);
        let mut down = 0usize;

        for i in 0..n {
            let idx = (start + i) % n;
            match self.workers[idx].tx.try_send(job) {
                Ok(()) => return Ok(resp_rx),
                Err(mpsc::error::TrySendError::Full(j)) => {
                    job = j;
                    continue;
                }
                Err(mpsc::error::TrySendError::Closed(j)) => {
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
