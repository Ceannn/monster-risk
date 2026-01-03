use anyhow::Context;
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PacerMode {
    /// Fixed interval per shard (+ optional jitter).
    Periodic,
    /// Poisson arrivals (exponential inter-arrival time) per shard.
    Poisson,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum XgbBodyFormat {
    /// Decide based on path: dir => dir, *.jsonl/*.ndjson => jsonl, otherwise json.
    Auto,
    /// Treat --xgb-body-file as a single JSON blob (sent verbatim each request).
    Json,
    /// Treat --xgb-body-file as JSONL/NDJSON (one JSON object per line; sampled per request).
    Jsonl,
    /// Treat --xgb-body-file as a directory of JSON files (each file is one body; sampled per request).
    Dir,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080/score")]
    url: String,

    /// Prometheus text endpoint (optional; for counters columns)
    #[arg(long, default_value = "http://127.0.0.1:8080/metrics")]
    metrics_url: String,

    /// JSON body file for /score_xgb* endpoints (default {})
    #[arg(long)]
    xgb_body_file: Option<String>,

    /// How to interpret --xgb-body-file (default: auto)
    #[arg(long, value_enum, default_value_t = XgbBodyFormat::Auto)]
    xgb_body_format: XgbBodyFormat,

    /// Max number of request bodies to load from corpus (jsonl/dir). Helps cap RAM.
    #[arg(long, default_value_t = 20000)]
    xgb_body_max: usize,

    /// Dense f32 corpus file (concatenated records). When set, bench sends application/octet-stream.
    /// Record format: raw little-endian f32 array, length = --xgb-dense-dim.
    #[arg(long)]
    xgb_dense_file: Option<String>,

    /// Dense vector dimension (required when using --xgb-dense-file)
    #[arg(long, default_value_t = 0)]
    xgb_dense_dim: usize,

    #[arg(long, default_value_t = 1000)]
    rps: u64,

    #[arg(long, default_value_t = 20)]
    duration: u64,

    /// Max in-flight. In open-loop mode, if full we drop immediately (no queueing).
    #[arg(long, default_value_t = 200)]
    concurrency: usize,

    /// CSV output path
    #[arg(long, default_value = "results/bench_summary.csv")]
    out: String,

    /// Optional label column for CSV (e.g. "l2_pct=10" or "l2_budget=1400")
    #[arg(long, default_value = "")]
    label: String,

    /// Scrape /metrics and write counters
    #[arg(long, default_value_t = true)]
    scrape_metrics: bool,

    /// Comma-separated RPS list, e.g. 1000,2000,4000,8000
    #[arg(long)]
    sweep_rps: Option<String>,

    /// Pause between sweep points (ms)
    #[arg(long, default_value_t = 500)]
    sweep_pause_ms: u64,

    /// Multiple targets:
    /// --target tokio,http://127.0.0.1:8080/score_xgb_pool,http://127.0.0.1:8080/metrics
    #[arg(long, value_name = "NAME,SCORE_URL,METRICS_URL")]
    target: Vec<String>,

    /// Find knee automatically
    #[arg(long, default_value_t = false)]
    find_knee: bool,

    #[arg(long, default_value_t = 1000)]
    knee_start_rps: u64,

    #[arg(long, default_value_t = 1000)]
    knee_step_rps: u64,

    #[arg(long, default_value_t = 64000)]
    knee_max_rps: u64,

    /// Relative jump threshold: p99 > last_p99 * (1 + pct)
    #[arg(long, default_value_t = 0.30)]
    knee_rise_pct: f64,

    /// Absolute jump threshold: p99 > last_p99 + abs_us
    #[arg(long, default_value_t = 200)]
    knee_abs_us: u64,

    // -------------------------
    // Pacer controls
    // -------------------------
    /// Arrival process: poisson (default) or periodic
    #[arg(long, value_enum, default_value_t = PacerMode::Poisson)]
    pacer: PacerMode,

    /// Number of pacer shards (0 = auto)
    #[arg(long, default_value_t = 0)]
    pacer_shards: usize,

    /// Periodic-only: add 0..jitter_us microseconds to each interval
    #[arg(long, default_value_t = 0)]
    pacer_jitter_us: u64,

    /// RNG seed for reproducibility (0 = time-based)
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Request timeout (ms)
    #[arg(long, default_value_t = 2000)]
    req_timeout_ms: u64,
}

#[derive(Debug, Clone)]
struct Target {
    runtime: String,
    url: String,
    metrics_url: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScoreRequest {
    trace_id: Option<String>,
    event_time_ms: i64,
    user_id: String,
    card_id: String,
    merchant_id: String,
    mcc: i32,
    amount: f64,
    currency: String,
    country: String,
    channel: String,
    device_id: String,
    ip_prefix: String,
    is_3ds: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimingsUs {
    #[serde(default)]
    parse: u64,
    #[serde(default)]
    feature: u64,
    #[serde(default)]
    router: u64,
    #[serde(default, alias = "l1")]
    xgb: u64,
    #[serde(default)]
    l2: u64,
    #[serde(default)]
    serialize: u64,
}

/// Parse RSK1 binary response (48 bytes) and extract timings.
/// Layout: magic 'RSK1', ver(u16)=1, flags(u16), trace_id(u64), score(f32), decision(u8), pad[3], timings[6](u32).
fn try_parse_rsk1_timings(buf: &[u8]) -> Option<TimingsUs> {
    if buf.len() < 48 {
        return None;
    }
    if &buf[0..4] != b"RSK1" {
        return None;
    }
    let ver = u16::from_le_bytes([buf[4], buf[5]]);
    if ver != 1 {
        return None;
    }

    #[inline]
    fn read_u32_le(b: &[u8], off: usize) -> u64 {
        u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as u64
    }

    // timings start at offset 24
    Some(TimingsUs {
        parse: read_u32_le(buf, 24),
        feature: read_u32_le(buf, 28),
        router: read_u32_le(buf, 32),
        xgb: read_u32_le(buf, 36),
        l2: read_u32_le(buf, 40),
        serialize: read_u32_le(buf, 44),
    })
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScoreResponse {
    trace_id: String,
    score: f64,
    decision: String,
    reason: Vec<serde_json::Value>,
    timings_us: TimingsUs,
}

#[derive(Debug, Clone, Default)]
struct Sample {
    e2e_us: u64,

    // server stage breakdown
    parse_us: u64,
    feature_us: u64,
    router_us: u64,
    xgb_us: u64,
    l2_us: u64,
    serialize_us: u64,

    // bench side schedule lag / oversleep (microseconds)
    bench_lag_us: u64,

    ok: bool,
}

#[derive(Debug, Clone, Default)]
struct Summary {
    samples_ok: usize,
    samples_err: usize,
    dropped: u64,

    // HTTP status breakdown for attempted requests (early-drop not counted here)
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    http_timeout: u64,

    p50: u64,
    p95: u64,
    p99: u64,

    // bench-side lag stats
    bench_lag_p99_us: u64,
    bench_lag_max_us: u64,
    missed_ticks_total: u64,

    // stage p99
    e2e_p99: u64,
    parse_p99: u64,
    feature_p99: u64,
    router_p99: u64,
    xgb_p99: u64,
    l2_p99: u64,
    serialize_p99: u64,

    // counters from /metrics（可能为空）
    router_l2_trigger_total: Option<f64>,
    router_deadline_miss_total: Option<f64>,
    router_l2_skipped_budget_total: Option<f64>,
    router_timeout_before_l2_total: Option<f64>,
    // more router counters (may be absent)
    router_l2_skipped_rate_total: Option<f64>,
    router_l2_skipped_waterline_total: Option<f64>,
    router_l2_skipped_deadline_budget_total: Option<f64>,

    // per-run deltas (end - start; useful when not restarting server between runs)
    router_l2_trigger_delta: Option<f64>,
    router_deadline_miss_delta: Option<f64>,
    router_l2_skipped_budget_delta: Option<f64>,
    router_l2_skipped_rate_delta: Option<f64>,
    router_l2_skipped_waterline_delta: Option<f64>,
    router_l2_skipped_deadline_budget_delta: Option<f64>,
    router_timeout_before_l2_delta: Option<f64>,

    // xgb_pool metrics (scraped from /metrics if available)
    xgb_pool_deadline_miss_total: Option<u64>,
    xgb_pool_deadline_miss_delta: Option<u64>,
    xgb_pool_xgb_compute_p99_est_us: Option<u64>,
    xgb_pool_queue_wait_p99_us: Option<u64>,
    xgb_pool_xgb_compute_p99_us: Option<u64>,
}

static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ensure output dir exists
    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let targets = parse_targets(&args)?;
    let (xgb_bodies, xgb_content_type) = load_xgb_bodies(&args)?;

    // RPS sequence
    let rps_list: Vec<u64> = if args.find_knee {
        let mut v = Vec::new();
        let mut r = args.knee_start_rps.max(1);
        while r <= args.knee_max_rps {
            v.push(r);
            r = r.saturating_add(args.knee_step_rps.max(1));
            if r == 0 {
                break;
            }
        }
        v
    } else if let Some(s) = &args.sweep_rps {
        parse_rps_list(s)?
    } else {
        vec![args.rps]
    };

    // previous p99 by runtime
    let mut prev_p99: HashMap<String, u64> = HashMap::new();

    for (i, rps) in rps_list.iter().copied().enumerate() {
        let mut knee_reasons: Vec<String> = Vec::new();

        for t in &targets {
            let summary = run_once(
                &args,
                rps,
                &t.url,
                &t.metrics_url,
                &xgb_bodies,
                xgb_content_type,
            )
            .await?;

            if summary.samples_ok == 0 {
                println!(
                    "[runtime={} rps={}] no successful samples collected (errors={}, dropped={})",
                    t.runtime, rps, summary.samples_err, summary.dropped
                );
            } else {
                let dur_s = args.duration.max(1) as f64;

                let ok_u64 = summary.samples_ok as u64;
                let err_u64 = summary.samples_err as u64;
                let dropped_u64 = summary.dropped;

                let attempted = ok_u64 + err_u64 + dropped_u64;

                let attempted_rps = attempted as f64 / dur_s;
                let ok_rps = ok_u64 as f64 / dur_s;
                let drop_rps = dropped_u64 as f64 / dur_s;

                let drop_pct = if attempted > 0 {
                    (dropped_u64 as f64) * 100.0 / (attempted as f64)
                } else {
                    0.0
                };

                println!(
                    "[runtime={} rps={}] ok_samples={} err_samples={} dropped={}  p50={}us  p95={}us  p99={}us  attempted_rps={:.1} ok_rps={:.1} drop_rps={:.1} drop_pct={:.3}%",
                    t.runtime,
                    rps,
                    summary.samples_ok,
                    summary.samples_err,
                    summary.dropped,
                    summary.p50,
                    summary.p95,
                    summary.p99,
                    attempted_rps,
                    ok_rps,
                    drop_rps,
                    drop_pct,
                );

                println!(
                    "[runtime={} rps={}] http: 2xx={} 429={} 5xx={} timeout={}",
                    t.runtime,
                    rps,
                    summary.http_2xx,
                    summary.http_429,
                    summary.http_5xx,
                    summary.http_timeout
                );

                println!(
                    "[runtime={} rps={}] stage_p99(us): parse={} feature={} router={} xgb={} l2={} serialize={}",
                    t.runtime,
                    rps,
                    summary.parse_p99,
                    summary.feature_p99,
                    summary.router_p99,
                    summary.xgb_p99,
                    summary.l2_p99,
                    summary.serialize_p99
                );

                if summary.xgb_pool_deadline_miss_total.is_some()
                    || summary.xgb_pool_xgb_compute_p99_est_us.is_some()
                    || summary.xgb_pool_queue_wait_p99_us.is_some()
                    || summary.xgb_pool_xgb_compute_p99_us.is_some()
                {
                    let miss = summary.xgb_pool_deadline_miss_total.unwrap_or(0);
                    let est = summary.xgb_pool_xgb_compute_p99_est_us.unwrap_or(0);
                    let q = summary.xgb_pool_queue_wait_p99_us.unwrap_or(0);
                    let c = summary.xgb_pool_xgb_compute_p99_us.unwrap_or(0);
                    println!(
                        "[runtime={} rps={}] xgb_pool: deadline_miss_total={} compute_p99_est_us={} queue_wait_p99_us={} compute_p99_us={}",
                        t.runtime, rps, miss, est, q, c
                    );
                }

                println!(
                    "[runtime={} rps={}] bench_lag_p99={}us bench_lag_max={}us missed_ticks_total={}",
                    t.runtime,
                    rps,
                    summary.bench_lag_p99_us,
                    summary.bench_lag_max_us,
                    summary.missed_ticks_total
                );
            }

            write_csv_row(
                &out_path,
                &args,
                &t.runtime,
                &t.url,
                &t.metrics_url,
                rps,
                &summary,
            )?;

            if args.find_knee {
                if summary.dropped > 0 {
                    knee_reasons.push(format!(
                        "{}: dropped>0 ({} drops)",
                        t.runtime, summary.dropped
                    ));
                } else if summary.samples_ok > 0 {
                    let last = *prev_p99.get(&t.runtime).unwrap_or(&0);
                    if last > 0 {
                        let threshold_rel =
                            (last as f64 * (1.0 + args.knee_rise_pct)).round() as u64;
                        let threshold_abs = last.saturating_add(args.knee_abs_us);

                        if summary.p99 > threshold_rel && summary.p99 > threshold_abs {
                            knee_reasons.push(format!(
                                "{}: p99 jump {}us -> {}us (> {}us and > {}us, +{:.0}%)",
                                t.runtime,
                                last,
                                summary.p99,
                                threshold_rel,
                                threshold_abs,
                                args.knee_rise_pct * 100.0
                            ));
                        }
                    }
                    prev_p99.insert(t.runtime.clone(), summary.p99);
                }
            }
        }

        if args.find_knee && !knee_reasons.is_empty() {
            println!(
                "
=== KNEE HIT at rps={} ===",
                rps
            );
            for r in knee_reasons {
                println!(" - {}", r);
            }
            println!();
            break;
        }

        if (args.find_knee || args.sweep_rps.is_some())
            && i + 1 < rps_list.len()
            && args.sweep_pause_ms > 0
        {
            tokio::time::sleep(Duration::from_millis(args.sweep_pause_ms)).await;
        }
    }

    Ok(())
}

fn percentile(sorted_us: &[u64], p: f64) -> u64 {
    if sorted_us.is_empty() {
        return 0;
    }
    let n = sorted_us.len();
    let rank = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted_us[rank.min(n - 1)]
}

#[allow(dead_code)]
fn gen_req(rng: &mut StdRng) -> ScoreRequest {
    let user = rng.gen_range(1..=10_000);
    let card = rng.gen_range(1..=50_000);
    let merch = rng.gen_range(1..=2_000);
    let mccs = [5411, 5732, 7995, 6011, 4812, 4829, 5311];
    let mcc = mccs[rng.gen_range(0..mccs.len())];

    let amount = (rng.gen::<f64>().powf(3.0) * 20000.0).max(1.0); // heavy tail
    let country = if rng.gen::<f64>() < 0.92 { "JP" } else { "US" };
    let channel = if rng.gen::<f64>() < 0.7 {
        "ecom"
    } else {
        "pos"
    };
    let is_3ds = rng.gen::<f64>() < 0.6;

    let now_ms = current_time_ms();

    ScoreRequest {
        trace_id: None,
        event_time_ms: now_ms,
        user_id: format!("u_{user}"),
        card_id: format!("c_{card}"),
        merchant_id: format!("m_{merch}"),
        mcc,
        amount,
        currency: "JPY".into(),
        country: country.into(),
        channel: channel.into(),
        device_id: format!("d_{}", rng.gen_range(1..=2000)),
        ip_prefix: if rng.gen::<f64>() < 0.02 {
            "198.51.100".into()
        } else {
            "203.0.113".into()
        },
        is_3ds,
    }
}

fn parse_rps_list(s: &str) -> anyhow::Result<Vec<u64>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let v: u64 = t
            .parse()
            .with_context(|| format!("invalid rps in --sweep-rps: {t}"))?;
        if v == 0 {
            continue;
        }
        out.push(v);
    }
    anyhow::ensure!(!out.is_empty(), "--sweep-rps parsed to empty list");
    Ok(out)
}

fn build_client(args: &Args) -> anyhow::Result<reqwest::Client> {
    let timeout = Duration::from_millis(args.req_timeout_ms.max(1));

    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_millis(500))
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(args.concurrency.max(1))
        .tcp_nodelay(true)
        .build()
        .context("build reqwest client")
}

#[allow(dead_code)]
fn prepare_score_bodies(seed: u64, n: usize) -> anyhow::Result<Vec<Bytes>> {
    // Pre-generate request bodies to avoid serde/alloc noise during benchmark
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n.max(1));
    for _ in 0..n.max(1) {
        let req = gen_req(&mut rng);
        let v = serde_json::to_vec(&req).context("serialize ScoreRequest")?;
        out.push(Bytes::from(v));
    }
    Ok(out)
}

/// inter-arrival ~ Exp(rate)
fn sample_exp_interval(rng: &mut StdRng, rate_per_sec: f64) -> Duration {
    let u: f64 = 1.0 - rng.gen::<f64>(); // (0,1]
    let dt_s = -u.ln() / rate_per_sec.max(1e-12);
    let ns = (dt_s * 1_000_000_000.0).ceil() as u64;
    Duration::from_nanos(ns.max(1))
}

async fn run_once(
    args: &Args,
    rps: u64,
    url: &str,
    metrics_url: &str,
    xgb_bodies: &Arc<Vec<Bytes>>,
    xgb_content_type: &str,
) -> anyhow::Result<Summary> {
    // seed=0 => time-based (less deterministic but closer to reality)
    let seed = if args.seed == 0 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        t.as_nanos() as u64
    } else {
        args.seed
    };

    let client = Arc::new(build_client(args)?);

    // /score_xgb /score_xgb_async /score_xgb_pool etc
    let is_xgb =
        url.contains("/score_xgb") || url.contains("/score_dense") || args.xgb_dense_file.is_some();

    // Pre-generated bodies only for non-xgb endpoints
    let score_bodies = if is_xgb {
        None
    } else {
        let n = args.concurrency.max(256).min(8192);
        Some(Arc::new(
            prepare_score_bodies(42u64, n).context("prepare score bodies")?,
        ))
    };

    // Unbounded to avoid backpressure deadlocks inside the bench itself.
    let (sample_tx, mut sample_rx) = mpsc::unbounded_channel::<Sample>();
    let (lag_tx, mut lag_rx) = mpsc::unbounded_channel::<u64>();

    // Collector task
    let collector = tokio::spawn(async move {
        let mut e2e = Vec::<u64>::with_capacity(200_000);
        let mut parse = Vec::<u64>::with_capacity(200_000);
        let mut feature = Vec::<u64>::with_capacity(200_000);
        let mut router = Vec::<u64>::with_capacity(200_000);
        let mut xgb = Vec::<u64>::with_capacity(200_000);
        let mut l2 = Vec::<u64>::with_capacity(200_000);
        let mut serialize = Vec::<u64>::with_capacity(200_000);

        let mut ok = 0usize;
        let mut err = 0usize;

        while let Some(s) = sample_rx.recv().await {
            if s.ok {
                ok += 1;
                e2e.push(s.e2e_us);
                parse.push(s.parse_us);
                feature.push(s.feature_us);
                router.push(s.router_us);
                xgb.push(s.xgb_us);
                l2.push(s.l2_us);
                serialize.push(s.serialize_us);
            } else {
                err += 1;
            }
        }

        (e2e, parse, feature, router, xgb, l2, serialize, ok, err)
    });

    // In-flight limit
    let workers = args.concurrency.max(1);
    let inflight = Arc::new(Semaphore::new(workers));

    // Only parse timings
    #[derive(Debug, Deserialize)]
    struct RespTimings {
        timings_us: TimingsUs,
    }

    let url_s = Arc::new(url.to_string());
    let xgb_ct = Arc::new(xgb_content_type.to_string());
    let xgb_bodies = xgb_bodies.clone();

    let snap_before = if args.scrape_metrics {
        fetch_metrics_snapshot(args, metrics_url).await
    } else {
        None
    };

    let start = tokio::time::Instant::now();
    let end = start + Duration::from_secs(args.duration);

    // pacer shards
    let mut shards = if args.pacer_shards == 0 {
        args.concurrency.max(1).clamp(32, 256)
    } else {
        args.pacer_shards.max(1).min(4096)
    };
    shards = shards.min(rps.max(1) as usize).max(1);

    let dropped = Arc::new(AtomicU64::new(0));
    // HTTP status counters (attempted requests only; early-drop not counted)
    let http_2xx = Arc::new(AtomicU64::new(0));
    let http_429 = Arc::new(AtomicU64::new(0));
    let http_5xx = Arc::new(AtomicU64::new(0));
    let http_timeout = Arc::new(AtomicU64::new(0));

    // ✅ missed threshold: >= 5ms (only count real pauses)
    let missed = Arc::new(AtomicU64::new(0));
    let miss_threshold_us: u64 = 5_000;

    let mut pacers = Vec::with_capacity(shards);

    for shard_idx in 0..shards {
        // Split RPS across shards as evenly as possible
        let base = rps / shards as u64;
        let extra = (shard_idx as u64) < (rps % shards as u64);
        let my_rps = base + if extra { 1 } else { 0 };
        if my_rps == 0 {
            continue;
        }

        let rate = my_rps as f64;

        let client = client.clone();
        let url = url_s.clone();
        // `tokio::spawn(async move { .. })` below would otherwise move `xgb_ct`.
        // Clone per-shard so the outer loop can keep spawning shards.
        let xgb_ct = xgb_ct.clone();
        let xgb_bodies = xgb_bodies.clone();
        let score_bodies = score_bodies.clone();
        let inflight = inflight.clone();
        let sample_tx = sample_tx.clone();
        let lag_tx = lag_tx.clone();
        let dropped = dropped.clone();
        let missed = missed.clone();
        let http_2xx = http_2xx.clone();
        let http_429 = http_429.clone();
        let http_5xx = http_5xx.clone();
        let http_timeout = http_timeout.clone();

        let mut rng = StdRng::seed_from_u64(
            seed.wrapping_add((shard_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        );

        let pacer_mode = args.pacer;
        let jitter_us = args.pacer_jitter_us;

        let h = tokio::spawn(async move {
            while tokio::time::Instant::now() < end {
                let now = tokio::time::Instant::now();

                // Next inter-arrival
                let mut dt = match pacer_mode {
                    PacerMode::Poisson => sample_exp_interval(&mut rng, rate),
                    PacerMode::Periodic => {
                        let ns = (1_000_000_000u64 / my_rps.max(1)).max(1);
                        Duration::from_nanos(ns)
                    }
                };

                // Optional jitter for periodic
                if matches!(pacer_mode, PacerMode::Periodic) && jitter_us > 0 {
                    let j = rng.gen_range(0..=jitter_us) as u64;
                    dt += Duration::from_micros(j);
                }

                if now + dt >= end {
                    break;
                }

                // Delay semantics: do not catch up (avoids self-inflicted burst)
                let sleep_start = tokio::time::Instant::now();
                tokio::time::sleep(dt).await;
                let actual = sleep_start.elapsed();
                let lag = actual.saturating_sub(dt);
                let lag_us = lag.as_micros() as u64;

                let _ = lag_tx.send(lag_us);
                if lag_us >= miss_threshold_us {
                    missed.fetch_add(1, Ordering::Relaxed);
                }

                // early drop if in-flight is full
                let permit = match inflight.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        dropped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                };

                let req_id = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
                let body: Bytes = if is_xgb {
                    {
                        let bodies = xgb_bodies.as_ref();
                        let idx = rng.gen_range(0..bodies.len());
                        bodies[idx].clone()
                    }
                } else {
                    let bodies = score_bodies.as_ref().expect("score_bodies");
                    let idx = (req_id as usize) % bodies.len();
                    bodies[idx].clone()
                };

                let client2 = client.clone();
                let url2 = url.clone();
                let tx2 = sample_tx.clone();
                let xgb_ct2 = xgb_ct.clone();
                let http_2xx2 = http_2xx.clone();
                let http_4292 = http_429.clone();
                let http_5xx2 = http_5xx.clone();
                let http_timeout2 = http_timeout.clone();

                tokio::spawn(async move {
                    let _permit = permit; // released on drop

                    let t0 = Instant::now();
                    let mut sample = Sample::default();
                    sample.bench_lag_us = lag_us;

                    let send_res = client2
                        .post(url2.as_str())
                        .header(
                            "content-type",
                            if is_xgb {
                                xgb_ct2.as_str()
                            } else {
                                "application/json"
                            },
                        )
                        .body(body)
                        .send()
                        .await;

                    match send_res {
                        Ok(resp) => {
                            let status = resp.status();
                            let code = status.as_u16();
                            if (200..=299).contains(&code) {
                                http_2xx2.fetch_add(1, Ordering::Relaxed);
                            } else if code == 429 {
                                http_4292.fetch_add(1, Ordering::Relaxed);
                            } else if (500..=599).contains(&code) {
                                http_5xx2.fetch_add(1, Ordering::Relaxed);
                            }

                            let status_ok = status.is_success();
                            match resp.bytes().await {
                                Ok(bytes) => {
                                    sample.e2e_us = t0.elapsed().as_micros() as u64;
                                    if status_ok {
                                        if let Some(t) = try_parse_rsk1_timings(&bytes) {
                                            sample.parse_us = t.parse;
                                            sample.feature_us = t.feature;
                                            sample.router_us = t.router;
                                            sample.xgb_us = t.xgb;
                                            sample.l2_us = t.l2;
                                            sample.serialize_us = t.serialize;
                                            sample.ok = true;
                                        } else if let Ok(r) =
                                            serde_json::from_slice::<RespTimings>(&bytes)
                                        {
                                            sample.parse_us = r.timings_us.parse;
                                            sample.feature_us = r.timings_us.feature;
                                            sample.router_us = r.timings_us.router;
                                            sample.xgb_us = r.timings_us.xgb;
                                            sample.l2_us = r.timings_us.l2;
                                            sample.serialize_us = r.timings_us.serialize;
                                            sample.ok = true;
                                        }
                                    }
                                }
                                Err(_) => {
                                    sample.e2e_us = t0.elapsed().as_micros() as u64;
                                }
                            }
                        }
                        Err(e) => {
                            if e.is_timeout() {
                                http_timeout2.fetch_add(1, Ordering::Relaxed);
                            }
                            sample.e2e_us = t0.elapsed().as_micros() as u64;
                        }
                    }

                    let _ = tx2.send(sample);
                });
            }
        });

        pacers.push(h);
    }

    // wait for pacers
    for h in pacers {
        let _ = h.await;
    }

    // wait for all in-flight requests to finish
    let mut permits = Vec::with_capacity(workers);
    for _ in 0..workers {
        if let Ok(p) = inflight.clone().acquire_owned().await {
            permits.push(p);
        }
    }
    drop(permits);

    // close channels
    drop(sample_tx);
    drop(lag_tx);

    // collect lag samples
    let mut bench_lag_us_vec: Vec<u64> = Vec::new();
    while let Some(v) = lag_rx.recv().await {
        bench_lag_us_vec.push(v);
    }

    let (mut e2e, mut parse, mut feature, mut router, mut xgb, mut l2, mut serialize, ok, err) =
        collector.await?;

    let mut summary = Summary::default();
    summary.samples_ok = ok;
    summary.samples_err = err;
    summary.dropped = dropped.load(Ordering::Relaxed);
    summary.http_2xx = http_2xx.load(Ordering::Relaxed);
    summary.http_429 = http_429.load(Ordering::Relaxed);
    summary.http_5xx = http_5xx.load(Ordering::Relaxed);
    summary.http_timeout = http_timeout.load(Ordering::Relaxed);

    bench_lag_us_vec.sort_unstable();
    summary.bench_lag_p99_us = percentile(&bench_lag_us_vec, 99.0);
    summary.bench_lag_max_us = *bench_lag_us_vec.last().unwrap_or(&0);
    summary.missed_ticks_total = missed.load(Ordering::Relaxed);

    if ok > 0 {
        e2e.sort_unstable();
        parse.sort_unstable();
        feature.sort_unstable();
        router.sort_unstable();
        xgb.sort_unstable();
        l2.sort_unstable();
        serialize.sort_unstable();

        summary.p50 = percentile(&e2e, 50.0);
        summary.p95 = percentile(&e2e, 95.0);
        summary.p99 = percentile(&e2e, 99.0);

        summary.e2e_p99 = percentile(&e2e, 99.0);
        summary.parse_p99 = percentile(&parse, 99.0);
        summary.feature_p99 = percentile(&feature, 99.0);
        summary.router_p99 = percentile(&router, 99.0);
        summary.xgb_p99 = percentile(&xgb, 99.0);
        summary.l2_p99 = percentile(&l2, 99.0);
        summary.serialize_p99 = percentile(&serialize, 99.0);
    }

    if args.scrape_metrics {
        let snap_after = fetch_metrics_snapshot(args, metrics_url).await;

        // Totals
        if let Some(snap) = &snap_after {
            summary.router_l2_trigger_total = snap.router_l2_trigger_total;
            summary.router_deadline_miss_total = snap.router_deadline_miss_total;
            summary.router_l2_skipped_budget_total = snap.router_l2_skipped_budget_total;
            summary.router_l2_skipped_rate_total = snap.router_l2_skipped_rate_total;
            summary.router_l2_skipped_waterline_total = snap.router_l2_skipped_waterline_total;
            summary.router_l2_skipped_deadline_budget_total =
                snap.router_l2_skipped_deadline_budget_total;
            summary.router_timeout_before_l2_total = snap.router_timeout_before_l2_total;

            summary.xgb_pool_deadline_miss_total = snap.xgb_pool_deadline_miss_total;
            summary.xgb_pool_xgb_compute_p99_est_us = snap.xgb_pool_xgb_compute_p99_est_us;
            summary.xgb_pool_queue_wait_p99_us = snap.xgb_pool_queue_wait_p99_us;
            summary.xgb_pool_xgb_compute_p99_us = snap.xgb_pool_xgb_compute_p99_us;
        }

        // Deltas (end - start): useful if you keep the same server process across a sweep.
        if let Some(snap) = snap_after {
            summary.router_l2_trigger_delta = diff_f64(
                snap.router_l2_trigger_total,
                snap_before.as_ref().and_then(|s| s.router_l2_trigger_total),
            );
            summary.router_deadline_miss_delta = diff_f64(
                snap.router_deadline_miss_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_deadline_miss_total),
            );
            summary.router_l2_skipped_budget_delta = diff_f64(
                snap.router_l2_skipped_budget_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_l2_skipped_budget_total),
            );
            summary.router_l2_skipped_rate_delta = diff_f64(
                snap.router_l2_skipped_rate_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_l2_skipped_rate_total),
            );
            summary.router_l2_skipped_waterline_delta = diff_f64(
                snap.router_l2_skipped_waterline_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_l2_skipped_waterline_total),
            );
            summary.router_l2_skipped_deadline_budget_delta = diff_f64(
                snap.router_l2_skipped_deadline_budget_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_l2_skipped_deadline_budget_total),
            );
            summary.router_timeout_before_l2_delta = diff_f64(
                snap.router_timeout_before_l2_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.router_timeout_before_l2_total),
            );

            summary.xgb_pool_deadline_miss_delta = diff_u64(
                snap.xgb_pool_deadline_miss_total,
                snap_before
                    .as_ref()
                    .and_then(|s| s.xgb_pool_deadline_miss_total),
            );
        }
    }

    Ok(summary)
}

fn diff_f64(end: Option<f64>, start: Option<f64>) -> Option<f64> {
    match (end, start) {
        (Some(e), Some(s)) if e.is_finite() && s.is_finite() => Some((e - s).max(0.0)),
        (Some(e), None) if e.is_finite() => Some(e.max(0.0)),
        _ => None,
    }
}

fn diff_u64(end: Option<u64>, start: Option<u64>) -> Option<u64> {
    match (end, start) {
        (Some(e), Some(s)) => Some(e.saturating_sub(s)),
        (Some(e), None) => Some(e),
        _ => None,
    }
}

async fn fetch_metrics_snapshot(args: &Args, metrics_url: &str) -> Option<MetricsSnapshot> {
    // use a separate client to avoid disturbing the main connection pool
    let client = build_client(args).ok()?;
    let resp = client.get(metrics_url).send().await.ok()?;
    let text = resp.text().await.ok()?;
    Some(scrape_metrics_snapshot(&text))
}

fn current_time_ms() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as i64
}

fn write_csv_row(
    out_path: &PathBuf,
    args: &Args,
    runtime: &str,
    url: &str,
    metrics_url: &str,
    rps_used: u64,
    s: &Summary,
) -> anyhow::Result<()> {
    let need_header = !out_path.exists();

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .with_context(|| format!("open csv: {}", out_path.display()))?;

    if need_header {
        writeln!(
            f,
            "ts_ms,label,runtime,url,metrics_url,target_rps,duration_s,concurrency,\
attempted,attempted_rps,ok,ok_rps,err,err_rps,dropped,drop_rps,drop_pct,\
http_2xx,http_429,http_5xx,http_timeout,\
p50_us,p95_us,p99_us,\
parse_p99_us,feature_p99_us,router_p99_us,xgb_p99_us,l2_p99_us,serialize_p99_us,\
router_l2_trigger_total,router_l2_trigger_delta,router_deadline_miss_total,router_deadline_miss_delta,router_l2_skipped_budget_total,router_l2_skipped_budget_delta,router_l2_skipped_rate_total,router_l2_skipped_rate_delta,router_l2_skipped_waterline_total,router_l2_skipped_waterline_delta,router_l2_skipped_deadline_budget_total,router_l2_skipped_deadline_budget_delta,router_timeout_before_l2_total,router_timeout_before_l2_delta,\
xgb_pool_deadline_miss_total,xgb_pool_deadline_miss_delta,xgb_pool_xgb_compute_p99_est_us,xgb_pool_queue_wait_p99_us,xgb_pool_xgb_compute_p99_us,\
bench_lag_p99_us,bench_lag_max_us,missed_ticks_total"
        )?;
    }

    let dur_s_u64 = args.duration.max(1);
    let dur_s = dur_s_u64 as f64;

    let ok_u64 = s.samples_ok as u64;
    let err_u64 = s.samples_err as u64;
    let dropped_u64 = s.dropped;
    let attempted = ok_u64 + err_u64 + dropped_u64;

    let attempted_rps = attempted as f64 / dur_s;
    let ok_rps = ok_u64 as f64 / dur_s;
    let err_rps = err_u64 as f64 / dur_s;
    let drop_rps = dropped_u64 as f64 / dur_s;

    let drop_pct = if attempted > 0 {
        (dropped_u64 as f64) * 100.0 / (attempted as f64)
    } else {
        0.0
    };

    let ts_ms = current_time_ms();

    let mut row: Vec<String> = Vec::with_capacity(64);
    row.push(ts_ms.to_string());
    row.push(escape_csv(&args.label));
    row.push(escape_csv(runtime));
    row.push(escape_csv(url));
    row.push(escape_csv(metrics_url));
    row.push(rps_used.to_string());
    row.push(dur_s_u64.to_string());
    row.push(args.concurrency.to_string());

    row.push(attempted.to_string());
    row.push(format!("{:.3}", attempted_rps));
    row.push(ok_u64.to_string());
    row.push(format!("{:.3}", ok_rps));
    row.push(err_u64.to_string());
    row.push(format!("{:.3}", err_rps));
    row.push(dropped_u64.to_string());
    row.push(format!("{:.3}", drop_rps));
    row.push(format!("{:.6}", drop_pct));

    row.push(s.http_2xx.to_string());
    row.push(s.http_429.to_string());
    row.push(s.http_5xx.to_string());
    row.push(s.http_timeout.to_string());

    row.push(s.p50.to_string());
    row.push(s.p95.to_string());
    row.push(s.p99.to_string());

    row.push(s.parse_p99.to_string());
    row.push(s.feature_p99.to_string());
    row.push(s.router_p99.to_string());
    row.push(s.xgb_p99.to_string());
    row.push(s.l2_p99.to_string());
    row.push(s.serialize_p99.to_string());

    row.push(opt_f64(s.router_l2_trigger_total));
    row.push(opt_f64(s.router_l2_trigger_delta));
    row.push(opt_f64(s.router_deadline_miss_total));
    row.push(opt_f64(s.router_deadline_miss_delta));
    row.push(opt_f64(s.router_l2_skipped_budget_total));
    row.push(opt_f64(s.router_l2_skipped_budget_delta));
    row.push(opt_f64(s.router_l2_skipped_rate_total));
    row.push(opt_f64(s.router_l2_skipped_rate_delta));
    row.push(opt_f64(s.router_l2_skipped_waterline_total));
    row.push(opt_f64(s.router_l2_skipped_waterline_delta));
    row.push(opt_f64(s.router_l2_skipped_deadline_budget_total));
    row.push(opt_f64(s.router_l2_skipped_deadline_budget_delta));
    row.push(opt_f64(s.router_timeout_before_l2_total));
    row.push(opt_f64(s.router_timeout_before_l2_delta));

    row.push(opt_u64(s.xgb_pool_deadline_miss_total));
    row.push(opt_u64(s.xgb_pool_deadline_miss_delta));
    row.push(opt_u64(s.xgb_pool_xgb_compute_p99_est_us));
    row.push(opt_u64(s.xgb_pool_queue_wait_p99_us));
    row.push(opt_u64(s.xgb_pool_xgb_compute_p99_us));

    row.push(s.bench_lag_p99_us.to_string());
    row.push(s.bench_lag_max_us.to_string());
    row.push(s.missed_ticks_total.to_string());

    writeln!(f, "{}", row.join(","))?;
    Ok(())
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|x| format!("{:.0}", x)).unwrap_or_default()
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

#[derive(Debug, Default, Clone)]
struct MetricsSnapshot {
    // counters from router layer (may be absent)
    router_l2_trigger_total: Option<f64>,
    router_deadline_miss_total: Option<f64>,
    router_l2_skipped_budget_total: Option<f64>,
    router_timeout_before_l2_total: Option<f64>,
    router_l2_skipped_rate_total: Option<f64>,
    router_l2_skipped_waterline_total: Option<f64>,
    router_l2_skipped_deadline_budget_total: Option<f64>,

    // xgb_pool counters/gauges (may be absent)
    xgb_pool_deadline_miss_total: Option<u64>,
    xgb_pool_deadline_miss_delta: Option<u64>,
    xgb_pool_xgb_compute_p99_est_us: Option<u64>,

    // xgb_pool quantiles (prefer quantile label form; fall back to *_p99 gauges)
    xgb_pool_queue_wait_p99_us: Option<u64>,
    xgb_pool_xgb_compute_p99_us: Option<u64>,
}

fn scrape_metrics_snapshot(text: &str) -> MetricsSnapshot {
    fn set_u64(dst: &mut Option<u64>, v: f64) {
        if v.is_finite() {
            let v = if v < 0.0 { 0.0 } else { v };
            *dst = Some(v.round() as u64);
        }
    }

    let mut out = MetricsSnapshot::default();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Format: <name{labels}> <value> [timestamp]
        let mut it = line.split_whitespace();
        let name_labels = match it.next() {
            Some(v) => v,
            None => continue,
        };
        let val_str = match it.next() {
            Some(v) => v,
            None => continue,
        };
        let val = match val_str.parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Split base name and labels (if any)
        let (base, labels_opt) = if let Some(brace) = name_labels.find('{') {
            let base = &name_labels[..brace];
            let labels = name_labels
                .get(brace + 1..name_labels.rfind('}').unwrap_or(name_labels.len()))
                .unwrap_or("");
            (base, Some(labels))
        } else {
            (name_labels, None)
        };

        // Router counters (no labels)
        match base {
            "router_l2_trigger_total" => out.router_l2_trigger_total = Some(val),
            "router_deadline_miss_total" => out.router_deadline_miss_total = Some(val),
            "router_l2_skipped_budget_total" => out.router_l2_skipped_budget_total = Some(val),
            "router_l2_skipped_rate_total" => out.router_l2_skipped_rate_total = Some(val),
            "router_l2_skipped_waterline_total" => {
                out.router_l2_skipped_waterline_total = Some(val)
            }
            "router_l2_skipped_deadline_budget_total" => {
                out.router_l2_skipped_deadline_budget_total = Some(val)
            }
            "router_timeout_before_l2_total" => out.router_timeout_before_l2_total = Some(val),
            _ => {}
        }

        // xgb_pool counters/gauges (no labels)
        match base {
            "xgb_pool_deadline_miss_total" => set_u64(&mut out.xgb_pool_deadline_miss_total, val),
            "xgb_pool_xgb_compute_p99_est_us" => {
                set_u64(&mut out.xgb_pool_xgb_compute_p99_est_us, val)
            }
            _ => {}
        }

        // xgb_pool quantiles - label form
        if let Some(labels) = labels_opt {
            let is_p99 =
                labels.contains("quantile=\"0.99\"") || labels.contains("quantile=\"0.990\"");
            if is_p99 {
                if base == "xgb_pool_queue_wait_us" {
                    set_u64(&mut out.xgb_pool_queue_wait_p99_us, val);
                    continue;
                }
                if base == "xgb_pool_xgb_compute_us" {
                    set_u64(&mut out.xgb_pool_xgb_compute_p99_us, val);
                    continue;
                }
            }
        }

        // xgb_pool quantiles - gauge fallback (common pattern)
        match name_labels {
            "xgb_pool_queue_wait_us_p99" => set_u64(&mut out.xgb_pool_queue_wait_p99_us, val),
            "xgb_pool_xgb_compute_us_p99" => set_u64(&mut out.xgb_pool_xgb_compute_p99_us, val),
            // some exporters use p99 suffix without underscore
            "xgb_pool_queue_wait_us99" => set_u64(&mut out.xgb_pool_queue_wait_p99_us, val),
            "xgb_pool_xgb_compute_us99" => set_u64(&mut out.xgb_pool_xgb_compute_p99_us, val),
            _ => {}
        }
    }

    out
}

fn slice_trim_ascii(buf: &Bytes) -> Bytes {
    let mut s = 0usize;
    let mut e = buf.len();
    while s < e && buf[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && buf[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    buf.slice(s..e)
}

fn detect_xgb_body_format(path: &std::path::Path) -> XgbBodyFormat {
    if path.is_dir() {
        return XgbBodyFormat::Dir;
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "jsonl" || ext == "ndjson" {
        XgbBodyFormat::Jsonl
    } else {
        XgbBodyFormat::Json
    }
}

/// Load request bodies for xgb endpoints.
/// - Json: single body (verbatim)
/// - Jsonl: one body per non-empty line
/// - Dir: one body per file
fn load_xgb_bodies(args: &Args) -> anyhow::Result<(Arc<Vec<Bytes>>, &'static str)> {
    let max = args.xgb_body_max.max(1);

    // --- Dense f32 corpus ---
    if let Some(path_s) = args.xgb_dense_file.as_deref() {
        let dim = args.xgb_dense_dim;
        if dim == 0 {
            anyhow::bail!("--xgb-dense-dim is required when using --xgb-dense-file");
        }

        let rec_len = dim.checked_mul(4).context("xgb_dense_dim too large")?;

        let path = std::path::PathBuf::from(path_s);
        let buf = Bytes::from(std::fs::read(&path)?);

        let n_records = buf.len() / rec_len;
        if n_records == 0 {
            anyhow::bail!(
                "dense corpus too small: file_len={} rec_len={}",
                buf.len(),
                rec_len
            );
        }

        let take = n_records.min(max);
        let mut out: Vec<Bytes> = Vec::with_capacity(take);
        for i in 0..take {
            let s = i * rec_len;
            let e = s + rec_len;
            out.push(buf.slice(s..e));
        }

        return Ok((Arc::new(out), "application/octet-stream"));
    }

    // --- JSON corpus (default / backwards compatible) ---
    let Some(path_s) = args.xgb_body_file.as_deref() else {
        return Ok((
            Arc::new(vec![Bytes::from_static(b"{}")]),
            "application/json",
        ));
    };

    let path = std::path::PathBuf::from(path_s);
    let fmt = match args.xgb_body_format {
        XgbBodyFormat::Auto => detect_xgb_body_format(&path),
        other => other,
    };

    let mut out: Vec<Bytes> = Vec::new();

    match fmt {
        XgbBodyFormat::Json => {
            let buf = Bytes::from(std::fs::read(&path)?);
            let body = slice_trim_ascii(&buf);
            if body.is_empty() {
                out.push(Bytes::from_static(b"{}"));
            } else {
                out.push(body);
            }
        }
        XgbBodyFormat::Jsonl => {
            let buf = Bytes::from(std::fs::read(&path)?);
            let mut line_start = 0usize;
            for i in 0..=buf.len() {
                let at_end = i == buf.len();
                if at_end || buf[i] == b'\n' {
                    let mut s = line_start;
                    let mut e = i;
                    while s < e && buf[s].is_ascii_whitespace() {
                        s += 1;
                    }
                    while e > s && buf[e - 1].is_ascii_whitespace() {
                        e -= 1;
                    }
                    if s < e {
                        out.push(buf.slice(s..e));
                        if out.len() >= max {
                            break;
                        }
                    }
                    line_start = i.saturating_add(1);
                }
            }
            if out.is_empty() {
                // Fallback to treating as a single JSON blob (e.g. pretty-printed json)
                let body = slice_trim_ascii(&buf);
                if body.is_empty() {
                    out.push(Bytes::from_static(b"{}"));
                } else {
                    out.push(body);
                }
            }
        }
        XgbBodyFormat::Dir => {
            let mut entries: Vec<_> = std::fs::read_dir(&path)?
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|e| e.path())
                .collect();
            entries.sort();
            for p in entries {
                let buf = Bytes::from(std::fs::read(&p)?);
                let body = slice_trim_ascii(&buf);
                if !body.is_empty() {
                    out.push(body);
                    if out.len() >= max {
                        break;
                    }
                }
            }
            if out.is_empty() {
                out.push(Bytes::from_static(b"{}"));
            }
        }
        XgbBodyFormat::Auto => unreachable!("auto resolved above"),
    }

    Ok((Arc::new(out), "application/json"))
}

fn parse_targets(args: &Args) -> anyhow::Result<Vec<Target>> {
    if args.target.is_empty() {
        return Ok(vec![Target {
            runtime: "default".into(),
            url: args.url.clone(),
            metrics_url: args.metrics_url.clone(),
        }]);
    }

    let mut out = Vec::new();
    for t in &args.target {
        let parts: Vec<&str> = t
            .split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .collect();
        anyhow::ensure!(
            parts.len() == 3,
            "invalid --target '{}', expect NAME,SCORE_URL,METRICS_URL",
            t
        );
        out.push(Target {
            runtime: parts[0].to_string(),
            url: parts[1].to_string(),
            metrics_url: parts[2].to_string(),
        });
    }
    Ok(out)
}
