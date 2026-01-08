//! A dedicated inference pool for XGBoost.
//!
//! Why this exists:
//! - XGBoost C-API is blocking and can be CPU heavy; running it on Tokio workers
//!   causes head-of-line blocking and latency spikes.
//! - We instead create N dedicated OS threads, each owning its own Booster (or native predictor).
//! - Submission is non-blocking (`try_submit*`); when the pool is saturated we fail fast.
//!
//! This file intentionally keeps the public API tiny and stable.
//!
//! Monster optimization (Work-stealing + per-core sharding):
//! - Instead of N bounded MPSC queues (Tokio mpsc), we shard submissions by *current CPU*
//!   into multiple lock-free injectors (crossbeam_deque::Injector).
//! - Each worker has a local deque (Worker) and pulls work by stealing batches from injectors,
//!   which reduces cross-core contention in the hot path.

use anyhow::Context;
use bytes::Bytes;
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Steal, Worker as DequeWorker};
use crossbeam_utils::Backoff;

#[cfg(all(
    not(feature = "xgb_ffi"),
    not(feature = "native_l1_tl2cgen"),
    not(feature = "native_l2_tl2cgen")
))]
compile_error!("risk-core: enable feature `xgb_ffi` or `native_*_tl2cgen` to build xgb_pool");

// NOTE: runtime-agnostic oneshot.
// - Tokio server can still await it directly.
// - Glommio server can await it without pulling in Tokio sync internals.
use futures_channel::oneshot;

#[cfg(feature = "xgb_ffi")]
use xgb_ffi::Booster;
#[cfg(feature = "native_l1_tl2cgen")]
use crate::native_l1_tl2cgen;
#[cfg(feature = "native_l2_tl2cgen")]
use crate::native_l2_tl2cgen;

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

pub(crate) const EXTRA_MAX: usize = 32;

/// Input row representation.
///
/// - `OwnedDense`: legacy path (already materialized Vec<f32>)
/// - `DenseBytesLe`: raw f32 little-endian bytes (len = ncols * 4)
///   This avoids building a `Vec<f32>` on the server thread.
enum XgbRow {
    OwnedDense(Vec<f32>),
    DenseBytesLe { bytes: Bytes, ncols: usize },
    DenseBytesLeExtra {
        bytes: Bytes,
        ncols: usize,
        extra: [f32; EXTRA_MAX],
        extra_dim: usize,
    },
}

struct XgbJob {
    enq_at: Instant,
    deadline_at: Option<Instant>,
    row: XgbRow,
    contrib_topk: usize,
    resp_tx: oneshot::Sender<anyhow::Result<XgbOut>>,
}

#[derive(Debug, Clone, Copy)]
pub struct XgbPoolStats {
    pub n_workers: usize,
    /// Total queue capacity across all workers (rounded up by per-worker split).
    pub cap_total: usize,
    /// Jobs currently waiting in pool queues (best-effort; does not include running jobs).
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

thread_local! {
    // Per-thread RR to spread bursts across shards without a global atomic.
    static TLS_RR: Cell<usize> = const { Cell::new(0) };
    // Cache current CPU id for threads that are pinned (Glommio executors, pinned workers).
    // Avoids a `sched_getcpu()` syscall on every submission.
    static TLS_CPU_ID: Cell<i32> = const { Cell::new(-1) };
}

/// Work-stealing XGB pool.
pub struct XgbPool {
    n_workers: usize,
    model_dim: usize,
    is_l2: bool,

    // Sharded injectors keyed by current CPU (best-effort).
    shards: Arc<Vec<Arc<Injector<XgbJob>>>>,
    // cpu_id -> shard_id mapping for fast lookup (i16: -1 means unknown)
    io_cpu_to_shard: Arc<Vec<i16>>,

    cap_total: usize,
    queued: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,

    // Best-effort liveness tracking: if all workers died, fail fast.
    live_workers: Arc<AtomicUsize>,

    /// If non-zero: reject at admission when predicted queue wait exceeds this budget.
    ///
    /// This is the "real" backpressure knob that keeps client-visible P99 stable.
    early_reject_pred_wait_us: u64,
    /// EWMA of compute time (best-effort, microseconds).
    compute_ema_us: Arc<AtomicU64>,
}

impl XgbPool {
    /// Build an inference pool.
    ///
    /// - `model_path`: path to the model file (e.g. ieee_xgb.ubj / xgb_model.json / ieee_xgb.bin)
    /// - `feature_names`: feature order used by the model (for contrib top-k naming)
    /// - `n_workers`: number of dedicated XGB threads
    /// - `queue_cap_total`: total queue capacity across all workers (will be evenly split for legacy semantics)
    /// - `warmup_iters`: per-worker warmup iterations to eliminate cold-start spikes
    #[cfg(feature = "xgb_ffi")]
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
    #[cfg(any(
        feature = "xgb_ffi",
        feature = "native_l1_tl2cgen",
        feature = "native_l2_tl2cgen"
    ))]
    pub fn new_with_pinning(
        model_path: impl AsRef<Path>,
        feature_names: Arc<Vec<String>>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
        pin_cpus: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        Self::new_with_pinning_role(
            model_path,
            feature_names,
            n_workers,
            queue_cap_total,
            warmup_iters,
            pin_cpus,
            false,
        )
    }

    /// Same as [`XgbPool::new_with_pinning`], but allows specifying the pool role (L1/L2).
    #[cfg(any(
        feature = "xgb_ffi",
        feature = "native_l1_tl2cgen",
        feature = "native_l2_tl2cgen"
    ))]
    pub fn new_with_pinning_role(
        model_path: impl AsRef<Path>,
        feature_names: Arc<Vec<String>>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
        pin_cpus: Option<Vec<usize>>,
        is_l2: bool,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(n_workers > 0, "xgb pool n_workers must be > 0");
        anyhow::ensure!(queue_cap_total > 0, "xgb pool queue_cap_total must be > 0");

        // Keep legacy semantics: queue_cap_total is "total capacity", but the old implementation
        // rounded it up by per-worker split. Preserve that so tests/knobs stay intuitive.
        let per_worker_cap = (queue_cap_total + n_workers - 1) / n_workers;
        let cap_total = per_worker_cap * n_workers;

        let model_path = Arc::new(model_path.as_ref().to_path_buf());
        let model_dim = feature_names.len();

        #[cfg(all(
            not(feature = "xgb_ffi"),
            not(feature = "native_l2_tl2cgen")
        ))]
        if is_l2 {
            anyhow::bail!("xgb pool L2 requested but no L2 backend enabled");
        }

        // Counters for admission control (best-effort).
        let queued = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicUsize::new(0));

        // Initialize EMA with a conservative default (microseconds).
        let compute_ema_us = Arc::new(AtomicU64::new(500));

        let pin_cpus = pin_cpus.unwrap_or_default();

        eprintln!(
            "xgb-pool: role={} backend={} workers={} cap_total={}",
            if is_l2 { "L2" } else { "L1" },
            pool_backend_name(is_l2),
            n_workers,
            cap_total
        );

        // Determine I/O CPUs (where requests are produced) vs worker CPUs.
        // If we can infer an affinity mask, shard by the CPUs that are NOT used by XGB workers.
        // If we can't, fall back to 1 shard.
        let io_cpus = infer_io_cpus(&pin_cpus);
        let (shards, io_cpu_to_shard) = build_shards(&io_cpus);

        let shards = Arc::new(shards);
        let io_cpu_to_shard = Arc::new(io_cpu_to_shard);

        let live_workers = Arc::new(AtomicUsize::new(n_workers));

        for wid in 0..n_workers {
            #[cfg(feature = "xgb_ffi")]
            let model_path = Arc::clone(&model_path);
            #[cfg(not(feature = "xgb_ffi"))]
            let _model_path = Arc::clone(&model_path);
            let feature_names = Arc::clone(&feature_names);
            let cpu = pin_cpus.get(wid).copied();
            let warmup_iters = warmup_iters;
            let queued = Arc::clone(&queued);
            let running = Arc::clone(&running);
            let compute_ema_us = Arc::clone(&compute_ema_us);
            let shards = Arc::clone(&shards);
            let io_cpu_to_shard = Arc::clone(&io_cpu_to_shard);
            let live_workers2 = Arc::clone(&live_workers);
            #[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
            let is_l2 = is_l2;
            #[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
            let model_dim = model_dim;

            thread::Builder::new()
                .name(format!("xgb-worker-{wid}"))
                .spawn(move || {
                    // Ensure live counter is decremented even on panic / early return.
                    struct LiveGuard(Arc<AtomicUsize>);
                    impl Drop for LiveGuard {
                        fn drop(&mut self) {
                            self.0.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                    let _lg = LiveGuard(live_workers2);

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

                    let n_shards = shards.len().max(1);
                    let home_shard = {
                        // Best-effort: if we know the current CPU, map to a nearby IO shard
                        // so the worker tends to pull from the shard "closest" to its CPU.
                        // Otherwise just distribute by wid.
                        let cur = current_cpu_id();
                        if let Some(cur) = cur {
                            // Pick nearest IO CPU, then map to shard.
                            // If mapping fails, fall back to wid-based.
                            nearest_shard(cur, &io_cpu_to_shard, n_shards).unwrap_or(wid % n_shards)
                        } else {
                            wid % n_shards
                        }
                    };

                    // Local deque as a cache to reduce contention (batch steal from injectors).
                    let local = DequeWorker::new_fifo();
                    let mut rr = wid;

                    #[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
                    {
                        let ncols = native_num_feature(is_l2);
                        if model_dim != ncols {
                            eprintln!(
                                "xgb-worker-{wid}: tl2cgen num_feature mismatch: feature_names.len()={} num_feature={}",
                                model_dim,
                                ncols
                            );
                        }

                        if warmup_iters > 0 {
                            let row = vec![f32::NAN; ncols];
                            for _ in 0..warmup_iters {
                                let _ = native_predict_proba_dense_1row(is_l2, &row);
                            }
                            eprintln!("xgb-worker-{wid}: warmup done iters={warmup_iters} (tl2cgen)");
                        }

                        let mut scratch_row: Vec<f32> = Vec::with_capacity(model_dim);
                        let mut last_extra_dim: Option<usize> = None;
                        let backoff = Backoff::new();

                        loop {
                            let job = match pop_job(&local, &shards, home_shard, &mut rr) {
                                Some(j) => {
                                    backoff.reset();
                                    j
                                }
                                None => {
                                    backoff.snooze();
                                    continue;
                                }
                            };

                            running.fetch_add(1, Ordering::Relaxed);
                            let _running_guard = RunningGuard(Arc::clone(&running));

                            let now = Instant::now();
                            let queue_wait_us =
                                now.duration_since(job.enq_at)
                                    .as_micros()
                                    .min(u128::from(u64::MAX)) as u64;
                            metrics::histogram!("xgb_pool_queue_wait_us").record(queue_wait_us as f64);

                            // We've started processing this job => decrement queued.
                            queued.fetch_sub(1, Ordering::Relaxed);

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

                            let (score, contrib_topk) = match job.row {
                                XgbRow::OwnedDense(row) => {
                                    let s = match native_predict_proba_dense_1row(is_l2, &row) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            let _ = job.resp_tx.send(Err(e));
                                            continue;
                                        }
                                    };
                                    (s, Vec::new())
                                }
                                XgbRow::DenseBytesLe { bytes, ncols } => {
                                    let s = match native_predict_proba_dense_bytes_le(is_l2, &bytes, ncols) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            let _ = job.resp_tx.send(Err(e));
                                            continue;
                                        }
                                    };
                                    (s, Vec::new())
                                }
                                XgbRow::DenseBytesLeExtra {
                                    bytes,
                                    ncols,
                                    extra,
                                    extra_dim,
                                } => {
                                    if ncols + extra_dim != model_dim || extra_dim > EXTRA_MAX {
                                        let _ = job
                                            .resp_tx
                                            .send(Err(anyhow::anyhow!(
                                                "dense bytes+extra dim mismatch: base={} extra={} model_dim={}",
                                                ncols,
                                                extra_dim,
                                                model_dim
                                            )));
                                        continue;
                                    }

                                    let t_build = Instant::now();
                                    let fallback = match decode_f32le_into(&bytes, ncols, &mut scratch_row) {
                                        Ok(v) => v,
                                        Err(msg) => {
                                            let _ = job.resp_tx.send(Err(anyhow::anyhow!(msg)));
                                            continue;
                                        }
                                    };
                                    if extra_dim > 0 {
                                        scratch_row.reserve(extra_dim);
                                        scratch_row.extend_from_slice(&extra[..extra_dim]);
                                    }
                                    let build_us =
                                        t_build.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                                    metrics::histogram!("l2_payload_extra_build_us")
                                        .record(build_us as f64);
                                    if last_extra_dim != Some(extra_dim) {
                                        metrics::gauge!("l2_payload_extra_dim")
                                            .set(extra_dim as f64);
                                        last_extra_dim = Some(extra_dim);
                                    }
                                    if fallback {
                                        metrics::counter!("l2_payload_decode_fallback_total").increment(1);
                                    }

                                    let s = match native_predict_proba_dense_1row(is_l2, &scratch_row) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            let _ = job.resp_tx.send(Err(e));
                                            continue;
                                        }
                                    };
                                    (s, Vec::new())
                                }
                            };

                            let xgb_us = t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                            metrics::histogram!("xgb_pool_xgb_compute_us").record(xgb_us as f64);

                            // Best-effort compute EWMA update.
                            let old = compute_ema_us.load(Ordering::Relaxed);
                            let new = if old == 0 {
                                xgb_us.max(1)
                            } else {
                                (old.saturating_mul(7) + xgb_us) / 8
                            };
                            compute_ema_us.store(new.max(1), Ordering::Relaxed);

                            let _ = job.resp_tx.send(Ok(XgbOut {
                                score,
                                queue_wait_us,
                                xgb_us,
                                contrib_topk,
                            }));
                        }
                    }

                    #[cfg(all(
                        feature = "xgb_ffi",
                        not(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))
                    ))]
                    {
                        let bst = match Booster::load_model(model_path.as_ref()) {
                            Ok(b) => b,
                            Err(e) => {
                                eprintln!("xgb-worker-{wid}: failed to load model: {e}");
                                return;
                            }
                        };

                        // Warmup: initialize thread-local buffers & internals.
                        if warmup_iters > 0 {
                            let ncols = feature_names.len();
                            let row = vec![f32::NAN; ncols];
                            for _ in 0..warmup_iters {
                                let _ = bst.predict_proba_dense_1row(&row);
                            }
                            let _ = bst.predict_contribs_dense_1row(&row);
                            eprintln!("xgb-worker-{wid}: warmup done iters={warmup_iters}");
                        }

                        // Main loop
                        let mut scratch_f32: Vec<f32> = Vec::new();
                        let backoff = Backoff::new();

                        loop {
                            let job = match pop_job(&local, &shards, home_shard, &mut rr) {
                                Some(j) => {
                                    backoff.reset();
                                    j
                                }
                                None => {
                                    backoff.snooze();
                                    continue;
                                }
                            };

                            running.fetch_add(1, Ordering::Relaxed);
                            let _running_guard = RunningGuard(Arc::clone(&running));

                            let now = Instant::now();
                            let queue_wait_us =
                                now.duration_since(job.enq_at)
                                    .as_micros()
                                    .min(u128::from(u64::MAX)) as u64;

                            metrics::histogram!("xgb_pool_queue_wait_us").record(queue_wait_us as f64);

                            // Started => dequeue.
                            queued.fetch_sub(1, Ordering::Relaxed);

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

                            // Run inference with either owned row or raw bytes.
                            let (score, contrib_topk) = match job.row {
                                XgbRow::OwnedDense(row) => {
                                    let s = match bst.predict_proba_dense_1row(&row) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            let _ = job.resp_tx.send(Err(anyhow::anyhow!(e)));
                                            continue;
                                        }
                                    };

                                    let mut topk = Vec::new();
                                    if job.contrib_topk > 0 {
                                        match bst.predict_contribs_dense_1row(&row) {
                                            Ok(out) => {
                                                let n = feature_names.len().min(out.values.len());
                                                topk = topk_abs_named(
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
                                    (s, topk)
                                }
                                XgbRow::DenseBytesLe { bytes, ncols } => {
                                    let row = match bytes_as_f32le(&bytes, ncols, &mut scratch_f32) {
                                        Ok(s) => s,
                                        Err(msg) => {
                                            let _ = job.resp_tx.send(Err(anyhow::anyhow!(msg)));
                                            continue;
                                        }
                                    };

                                    let s = match bst.predict_proba_dense_1row(row) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            let _ = job.resp_tx.send(Err(anyhow::anyhow!(e)));
                                            continue;
                                        }
                                    };

                                    let mut topk = Vec::new();
                                    if job.contrib_topk > 0 {
                                        match bst.predict_contribs_dense_1row(row) {
                                            Ok(out) => {
                                                let n = feature_names.len().min(out.values.len());
                                                topk = topk_abs_named(
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
                                    (s, topk)
                                }
                            };

                            let xgb_us = t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                            metrics::histogram!("xgb_pool_xgb_compute_us").record(xgb_us as f64);

                            // EWMA update: new = old*7/8 + xgb*1/8
                            let old = compute_ema_us.load(Ordering::Relaxed);
                            let new = if old == 0 {
                                xgb_us.max(1)
                            } else {
                                (old.saturating_mul(7) + xgb_us) / 8
                            };
                            compute_ema_us.store(new.max(1), Ordering::Relaxed);

                            let _ = job.resp_tx.send(Ok(XgbOut {
                                score,
                                queue_wait_us,
                                xgb_us,
                                contrib_topk,
                            }));
                        }
                    }
                })
                .context("spawn xgb worker")?;
        }

        Ok(Self {
            n_workers,
            model_dim,
            is_l2,
            shards,
            io_cpu_to_shard,
            cap_total,
            queued,
            running,
            live_workers,
            early_reject_pred_wait_us: 0,
            compute_ema_us,
        })
    }

    /// Enable admission-control style backpressure.
    ///
    /// When enabled, `try_submit*` will **fail fast** with `QueueFull` if the
    /// predicted queue wait exceeds `pred_wait_budget_us`.
    pub fn set_early_reject_pred_wait_us(&mut self, pred_wait_budget_us: u64) {
        self.early_reject_pred_wait_us = pred_wait_budget_us;
    }

    /// Convenience: build a pool directly from a model directory (same format as `XgbRuntime::load_from_dir`).
    #[cfg(any(
        feature = "xgb_ffi",
        feature = "native_l1_tl2cgen",
        feature = "native_l2_tl2cgen"
    ))]
    pub fn new_from_dir(
        model_dir: impl AsRef<Path>,
        n_workers: usize,
        queue_cap_total: usize,
        warmup_iters: usize,
        pin_cpus: Option<Vec<usize>>,
    ) -> anyhow::Result<Self> {
        let rt = crate::xgb_runtime::XgbRuntime::load_from_dir(model_dir.as_ref())
            .with_context(|| format!("load model dir: {}", model_dir.as_ref().display()))?;
        let is_l2 = rt.is_l2();
        let feature_names = Arc::new(rt.feature_names);
        let model_path = rt.model_path;
        Self::new_with_pinning_role(
            model_path,
            feature_names,
            n_workers,
            queue_cap_total,
            warmup_iters,
            pin_cpus,
            is_l2,
        )
    }

    /// Snapshot stats for router-side admission control / debugging.
    #[inline]
    pub fn stats(&self) -> XgbPoolStats {
        XgbPoolStats {
            n_workers: self.n_workers,
            cap_total: self.cap_total,
            queued: self.queued.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }

    #[inline]
    fn ensure_alive(&self) -> Result<(), XgbPoolError> {
        if self.live_workers.load(Ordering::Relaxed) == 0 {
            Err(XgbPoolError::WorkerDown)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn admission_control(&self) -> Result<(), XgbPoolError> {
        self.ensure_alive()?;

        if self.early_reject_pred_wait_us > 0 {
            let inflight = (self.queued.load(Ordering::Relaxed) + self.running.load(Ordering::Relaxed)) as u64;
            let ema = self.compute_ema_us.load(Ordering::Relaxed).max(1);
            let pred_wait_us = inflight.saturating_mul(ema) / (self.n_workers as u64).max(1);
            metrics::histogram!("xgb_pool_pred_wait_us").record(pred_wait_us as f64);
            if pred_wait_us >= self.early_reject_pred_wait_us {
                metrics::counter!("xgb_pool_admission_reject_total").increment(1);
                return Err(XgbPoolError::QueueFull);
            }
        }

        // Hard cap: global queued count.
        let prev = self.queued.fetch_add(1, Ordering::Relaxed);
        if prev >= self.cap_total {
            self.queued.fetch_sub(1, Ordering::Relaxed);
            return Err(XgbPoolError::QueueFull);
        }

        Ok(())
    }

    /// Non-blocking submission. Returns a oneshot receiver to await the result.
    pub fn try_submit(
        &self,
        row: Vec<f32>,
        contrib_topk: usize,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let enq_at = Instant::now();
        let deadline_at = if self.early_reject_pred_wait_us > 0 {
            Some(enq_at + Duration::from_micros(self.early_reject_pred_wait_us))
        } else {
            None
        };
        let job = XgbJob {
            enq_at,
            deadline_at,
            row: XgbRow::OwnedDense(row),
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
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

        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();

        let enq_at = Instant::now();
        let deadline_at = enq_at + Duration::from_micros(budget_us);

        let job = XgbJob {
            enq_at,
            deadline_at: Some(deadline_at),
            row: XgbRow::OwnedDense(row),
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
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

    /// Non-blocking submission for dense bytes payload (f32 little-endian).
    pub fn try_submit_dense_bytes_le(
        &self,
        bytes: Bytes,
        ncols: usize,
        contrib_topk: usize,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        // Validate shape early to avoid silent corruption.
        let need = ncols.checked_mul(4).ok_or(XgbPoolError::QueueFull)?;
        if bytes.len() != need {
            // Treat as bad request upstream; here we surface it as QueueFull to keep API stable.
            return Err(XgbPoolError::QueueFull);
        }

        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let enq_at = Instant::now();
        let deadline_at = if self.early_reject_pred_wait_us > 0 {
            Some(enq_at + Duration::from_micros(self.early_reject_pred_wait_us))
        } else {
            None
        };
        let job = XgbJob {
            enq_at,
            deadline_at,
            row: XgbRow::DenseBytesLe { bytes, ncols },
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
    }

    /// Non-blocking submission for dense bytes payload (f32 little-endian) with a queue-wait budget (microseconds).
    ///
    /// - `budget_us = 0` disables the budget (falls back to `try_submit_dense_bytes_le`).
    /// - The effective queue-wait deadline is the minimum of:
    ///   - pool-level `early_reject_pred_wait_us` (if > 0)
    ///   - per-call `budget_us` (if > 0)
    pub fn try_submit_dense_bytes_le_with_budget(
        &self,
        bytes: Bytes,
        ncols: usize,
        contrib_topk: usize,
        budget_us: u64,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        if budget_us == 0 {
            return self.try_submit_dense_bytes_le(bytes, ncols, contrib_topk);
        }

        // Validate shape early to avoid silent corruption.
        let need = ncols.checked_mul(4).ok_or(XgbPoolError::QueueFull)?;
        if bytes.len() != need {
            // Treat as bad request upstream; here we surface it as QueueFull to keep API stable.
            return Err(XgbPoolError::QueueFull);
        }

        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let enq_at = Instant::now();

        // Per-call budget + pool-level early-reject => pick the tighter deadline.
        let mut eff_us = budget_us;
        if self.early_reject_pred_wait_us > 0 {
            eff_us = eff_us.min(self.early_reject_pred_wait_us);
        }
        let deadline_at = Some(enq_at + Duration::from_micros(eff_us));

        let job = XgbJob {
            enq_at,
            deadline_at,
            row: XgbRow::DenseBytesLe { bytes, ncols },
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
    }

    /// Non-blocking submission for dense bytes payload (f32 little-endian) with extra f32 tail.
    ///
    /// This is intended for L2 only (extra features appended inside worker scratch).
    pub fn try_submit_dense_bytes_le_extra(
        &self,
        bytes: Bytes,
        ncols: usize,
        extra: [f32; EXTRA_MAX],
        extra_dim: usize,
        contrib_topk: usize,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        if extra_dim == 0 {
            return self.try_submit_dense_bytes_le(bytes, ncols, contrib_topk);
        }
        if extra_dim > EXTRA_MAX || !self.is_l2 {
            return Err(XgbPoolError::QueueFull);
        }

        // Validate shape early to avoid silent corruption.
        let need = ncols.checked_mul(4).ok_or(XgbPoolError::QueueFull)?;
        if bytes.len() != need {
            return Err(XgbPoolError::QueueFull);
        }
        if ncols.saturating_add(extra_dim) != self.model_dim {
            return Err(XgbPoolError::QueueFull);
        }

        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let enq_at = Instant::now();
        let deadline_at = if self.early_reject_pred_wait_us > 0 {
            Some(enq_at + Duration::from_micros(self.early_reject_pred_wait_us))
        } else {
            None
        };
        let job = XgbJob {
            enq_at,
            deadline_at,
            row: XgbRow::DenseBytesLeExtra {
                bytes,
                ncols,
                extra,
                extra_dim,
            },
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
    }

    /// Non-blocking submission for dense bytes payload (f32 little-endian) with extra tail,
    /// with a queue-wait budget (microseconds).
    pub fn try_submit_dense_bytes_le_extra_with_budget(
        &self,
        bytes: Bytes,
        ncols: usize,
        extra: [f32; EXTRA_MAX],
        extra_dim: usize,
        contrib_topk: usize,
        budget_us: u64,
    ) -> Result<oneshot::Receiver<anyhow::Result<XgbOut>>, XgbPoolError> {
        if budget_us == 0 {
            return self.try_submit_dense_bytes_le_extra(bytes, ncols, extra, extra_dim, contrib_topk);
        }
        if extra_dim == 0 {
            return self.try_submit_dense_bytes_le_with_budget(bytes, ncols, contrib_topk, budget_us);
        }
        if extra_dim > EXTRA_MAX || !self.is_l2 {
            return Err(XgbPoolError::QueueFull);
        }

        // Validate shape early to avoid silent corruption.
        let need = ncols.checked_mul(4).ok_or(XgbPoolError::QueueFull)?;
        if bytes.len() != need {
            return Err(XgbPoolError::QueueFull);
        }
        if ncols.saturating_add(extra_dim) != self.model_dim {
            return Err(XgbPoolError::QueueFull);
        }

        self.admission_control()?;

        let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<XgbOut>>();
        let enq_at = Instant::now();

        // Per-call budget + pool-level early-reject => pick the tighter deadline.
        let mut eff_us = budget_us;
        if self.early_reject_pred_wait_us > 0 {
            eff_us = eff_us.min(self.early_reject_pred_wait_us);
        }
        let deadline_at = Some(enq_at + Duration::from_micros(eff_us));

        let job = XgbJob {
            enq_at,
            deadline_at,
            row: XgbRow::DenseBytesLeExtra {
                bytes,
                ncols,
                extra,
                extra_dim,
            },
            contrib_topk,
            resp_tx,
        };

        let sid = pick_shard(&self.io_cpu_to_shard, self.shards.len());
        self.shards[sid].push(job);

        Ok(resp_rx)
    }

    /// Convenience async wrapper.
    pub async fn submit_dense_bytes_le(
        &self,
        bytes: Bytes,
        ncols: usize,
        contrib_topk: usize,
    ) -> Result<anyhow::Result<XgbOut>, XgbPoolError> {
        let rx = self.try_submit_dense_bytes_le(bytes, ncols, contrib_topk)?;
        match rx.await {
            Ok(v) => Ok(v),
            Err(_) => Err(XgbPoolError::Canceled),
        }
    }
}

fn pool_backend_name(is_l2: bool) -> &'static str {
    if is_l2 {
        #[cfg(feature = "native_l2_tl2cgen")]
        {
            return "native_tl2cgen";
        }
        #[cfg(all(not(feature = "native_l2_tl2cgen"), feature = "xgb_ffi"))]
        {
            return "xgb_ffi";
        }
        #[cfg(all(not(feature = "native_l2_tl2cgen"), not(feature = "xgb_ffi")))]
        {
            return "disabled";
        }
    } else {
        #[cfg(feature = "native_l1_tl2cgen")]
        {
            return "native_tl2cgen";
        }
        #[cfg(all(not(feature = "native_l1_tl2cgen"), feature = "xgb_ffi"))]
        {
            return "xgb_ffi";
        }
        #[cfg(all(not(feature = "native_l1_tl2cgen"), not(feature = "xgb_ffi")))]
        {
            return "disabled";
        }
    }
    "disabled"
}

#[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
fn native_num_feature(is_l2: bool) -> usize {
    if is_l2 {
        #[cfg(feature = "native_l2_tl2cgen")]
        {
            return native_l2_tl2cgen::num_feature();
        }
        #[cfg(not(feature = "native_l2_tl2cgen"))]
        {
            return 0;
        }
    }

    #[cfg(feature = "native_l1_tl2cgen")]
    {
        return native_l1_tl2cgen::num_feature();
    }
    #[cfg(not(feature = "native_l1_tl2cgen"))]
    {
        0
    }
}

#[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
fn native_predict_proba_dense_1row(is_l2: bool, row: &[f32]) -> anyhow::Result<f32> {
    if is_l2 {
        #[cfg(feature = "native_l2_tl2cgen")]
        {
            return native_l2_tl2cgen::predict_proba_dense_1row(row);
        }
        #[cfg(not(feature = "native_l2_tl2cgen"))]
        {
            return Err(anyhow::anyhow!(
                "native_l2_tl2cgen disabled but L2 pool requested"
            ));
        }
    }

    #[cfg(feature = "native_l1_tl2cgen")]
    {
        return native_l1_tl2cgen::predict_proba_dense_1row(row);
    }
    #[cfg(not(feature = "native_l1_tl2cgen"))]
    {
        Err(anyhow::anyhow!(
            "native_l1_tl2cgen disabled but L1 pool requested"
        ))
    }
}

#[cfg(any(feature = "native_l1_tl2cgen", feature = "native_l2_tl2cgen"))]
fn native_predict_proba_dense_bytes_le(
    is_l2: bool,
    bytes: &Bytes,
    ncols: usize,
) -> anyhow::Result<f32> {
    if is_l2 {
        #[cfg(feature = "native_l2_tl2cgen")]
        {
            return native_l2_tl2cgen::predict_proba_dense_bytes_le(bytes, ncols);
        }
        #[cfg(not(feature = "native_l2_tl2cgen"))]
        {
            return Err(anyhow::anyhow!(
                "native_l2_tl2cgen disabled but L2 pool requested"
            ));
        }
    }

    #[cfg(feature = "native_l1_tl2cgen")]
    {
        return native_l1_tl2cgen::predict_proba_dense_bytes_le(bytes, ncols);
    }
    #[cfg(not(feature = "native_l1_tl2cgen"))]
    {
        Err(anyhow::anyhow!(
            "native_l1_tl2cgen disabled but L1 pool requested"
        ))
    }
}

#[inline]
fn pop_job(
    local: &DequeWorker<XgbJob>,
    shards: &Vec<Arc<Injector<XgbJob>>>,
    home_shard: usize,
    rr: &mut usize,
) -> Option<XgbJob> {
    if let Some(j) = local.pop() {
        return Some(j);
    }

    let n = shards.len().max(1);

    // 1) Try home shard first (hot path locality).
    match steal_one(&shards[home_shard % n], local) {
        Some(j) => return Some(j),
        None => {}
    }

    // 2) Try a small number of other shards (work stealing).
    // Rotate which shards we try first to avoid hammering shard 0.
    let tries = n.min(4);
    let start = *rr % n;
    *rr = rr.wrapping_add(1);

    for k in 0..tries {
        let sid = (start + k) % n;
        if sid == home_shard {
            continue;
        }
        if let Some(j) = steal_one(&shards[sid], local) {
            return Some(j);
        }
    }

    None
}

#[inline]
fn steal_one(inj: &Injector<XgbJob>, local: &DequeWorker<XgbJob>) -> Option<XgbJob> {
    loop {
        match inj.steal_batch_and_pop(local) {
            Steal::Success(j) => return Some(j),
            Steal::Empty => return None,
            Steal::Retry => continue,
        }
    }
}

#[inline]
fn pick_shard(io_cpu_to_shard: &[i16], n_shards: usize) -> usize {
    let n = n_shards.max(1);
    let cpu = current_cpu_id_fast();
    let base = io_cpu_to_shard.get(cpu).copied().unwrap_or(-1);

    // If we can map CPU -> shard, use it; else hash by cpu.
    let mut sid = if base >= 0 { base as usize } else { cpu % n };

    // Add per-thread RR to reduce collisions when multiple tasks run on the same CPU.
    TLS_RR.with(|c| {
        let cur = c.get();
        c.set(cur.wrapping_add(1));
        sid = sid.wrapping_add(cur) % n;
    });

    sid
}

#[inline]
fn nearest_shard(worker_cpu: usize, io_cpu_to_shard: &[i16], n_shards: usize) -> Option<usize> {
    if n_shards == 0 {
        return None;
    }
    // Prefer exact mapping if exists.
    if let Some(s) = io_cpu_to_shard.get(worker_cpu).copied() {
        if s >= 0 {
            return Some(s as usize);
        }
    }
    // Otherwise search outward a bit for the nearest known IO cpu (cheap, bounded).
    // (This matters only when workers are pinned but IO CPUs are disjoint.)
    for d in 1..=8usize {
        if worker_cpu >= d {
            if let Some(s) = io_cpu_to_shard.get(worker_cpu - d).copied() {
                if s >= 0 {
                    return Some(s as usize);
                }
            }
        }
        if let Some(s) = io_cpu_to_shard.get(worker_cpu + d).copied() {
            if s >= 0 {
                return Some(s as usize);
            }
        }
    }
    None
}

fn build_shards(io_cpus: &[usize]) -> (Vec<Arc<Injector<XgbJob>>>, Vec<i16>) {
    // If we know IO CPUs, one shard per IO CPU. Otherwise 1 shard.
    let n_shards = io_cpus.len().max(1).min(64);
    let mut shards = Vec::with_capacity(n_shards);
    for _ in 0..n_shards {
        shards.push(Arc::new(Injector::new()));
    }

    // Build cpu_id -> shard_id table.
    let max_cpu = max_cpu_id().unwrap_or(0);
    let mut map = vec![-1i16; max_cpu + 1];
    if !io_cpus.is_empty() {
        for (sid, &cpu) in io_cpus.iter().take(n_shards).enumerate() {
            if cpu < map.len() {
                map[cpu] = sid as i16;
            }
        }
    }

    (shards, map)
}

/// Infer the CPU list used by "I/O producers" (HTTP runtime threads),
/// by taking the current process affinity mask and removing worker-pinned CPUs.
///
/// If we fail to read affinity mask, returns empty => caller falls back to 1 shard.
fn infer_io_cpus(pin_cpus: &[usize]) -> Vec<usize> {
    let allowed = get_self_affinity_cpus().unwrap_or_default();
    if allowed.is_empty() {
        return Vec::new();
    }
    if pin_cpus.is_empty() {
        return allowed;
    }

    let mut mark = vec![false; max_cpu_id().unwrap_or(0) + 1];
    for &c in pin_cpus {
        if c < mark.len() {
            mark[c] = true;
        }
    }
    allowed.into_iter().filter(|&c| c < mark.len() && !mark[c]).collect()
}

/// Convert raw f32 little-endian bytes into a `&[f32]` view when possible.
/// Falls back to decoding into `scratch` when the byte slice is unaligned.
fn decode_f32le_into(bytes: &Bytes, ncols: usize, scratch: &mut Vec<f32>) -> Result<bool, String> {
    let b = bytes.as_ref();
    let need = ncols
        .checked_mul(4)
        .ok_or_else(|| "ncols too large".to_string())?;
    if b.len() != need {
        return Err(format!(
            "dense bytes size mismatch: got {}, need {}",
            b.len(),
            need
        ));
    }

    scratch.clear();
    scratch.reserve(ncols);

    if cfg!(target_endian = "little") {
        // SAFETY: We only accept the aligned case (head/tail empty). All bit patterns are valid f32.
        let (head, body, tail) = unsafe { b.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() && body.len() == ncols {
            scratch.extend_from_slice(body);
            return Ok(false);
        }
    }

    for c in b.chunks_exact(4) {
        scratch.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(true)
}

/// Convert raw f32 little-endian bytes into a `&[f32]` view when possible.
/// Falls back to decoding into `scratch` when the byte slice is unaligned.
fn bytes_as_f32le<'a>(
    bytes: &'a Bytes,
    ncols: usize,
    scratch: &'a mut Vec<f32>,
) -> Result<&'a [f32], String> {
    let b = bytes.as_ref();
    let need = ncols
        .checked_mul(4)
        .ok_or_else(|| "ncols too large".to_string())?;
    if b.len() != need {
        return Err(format!(
            "dense bytes size mismatch: got {}, need {}",
            b.len(),
            need
        ));
    }

    // Fast path: reinterpret when aligned (little-endian only).
    if cfg!(target_endian = "little") {
        // SAFETY: We only accept the aligned case (head/tail empty). All bit patterns are valid f32.
        let (head, body, tail) = unsafe { b.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() && body.len() == ncols {
            return Ok(body);
        }
    }

    // Fallback: decode safely into scratch.
    scratch.clear();
    scratch.reserve(ncols);
    for c in b.chunks_exact(4) {
        scratch.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(scratch.as_slice())
}

fn bytes_as_f64le(bytes: &[u8], ncols: usize, scratch: &mut Vec<f64>) -> anyhow::Result<()> {
    let need = ncols * 4;
    anyhow::ensure!(
        bytes.len() == need,
        "DenseBytesLe: expected {} bytes (ncols={} * 4), got {}",
        need,
        ncols,
        bytes.len()
    );

    if scratch.len() != ncols {
        scratch.resize(ncols, 0.0);
    }

    for i in 0..ncols {
        let off = i * 4;
        let c = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        scratch[i] = f32::from_le_bytes(c) as f64;
    }

    Ok(())
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

#[cfg(target_os = "linux")]
fn current_cpu_id() -> Option<usize> {
    // SAFETY: sched_getcpu has no side effects.
    let cpu = unsafe { libc::sched_getcpu() };
    if cpu >= 0 {
        Some(cpu as usize)
    } else {
        None
    }
}


#[inline]
fn current_cpu_id_fast() -> usize {
    TLS_CPU_ID.with(|c| {
        let v = c.get();
        if v >= 0 {
            v as usize
        } else {
            let cpu = current_cpu_id().unwrap_or(0) as i32;
            c.set(cpu);
            cpu as usize
        }
    })
}
#[cfg(not(target_os = "linux"))]
fn current_cpu_id() -> Option<usize> {
    None
}

#[cfg(target_os = "linux")]
fn max_cpu_id() -> Option<usize> {
    // This is the smallest safe bound we can get without external deps.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
    if n > 0 {
        Some((n as usize).saturating_sub(1))
    } else {
        None
    }
}

#[cfg(not(target_os = "linux"))]
fn max_cpu_id() -> Option<usize> {
    None
}

#[cfg(target_os = "linux")]
fn get_self_affinity_cpus() -> std::io::Result<Vec<usize>> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        let tid = libc::pthread_self();
        let rc = libc::pthread_getaffinity_np(tid, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
        let mut out = Vec::new();
        // cpu_set_t size is platform-specific; CPU_ISSET is safe to query up to CONF cpus.
        let max = max_cpu_id().unwrap_or(0);
        for cpu in 0..=max {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        Ok(out)
    }
}

#[cfg(not(target_os = "linux"))]
fn get_self_affinity_cpus() -> std::io::Result<Vec<usize>> {
    Ok(Vec::new())
}
