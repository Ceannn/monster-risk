use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
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
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080/score")]
    url: String,

    /// Prometheus 文本暴露端点（可选；用于写入路由/降级计数器）
    #[arg(long, default_value = "http://127.0.0.1:8080/metrics")]
    metrics_url: String,

    /// 当目标 URL 包含 /score_xgb 时使用的 JSON 请求体文件（默认 {}）
    #[arg(long)]
    xgb_body_file: Option<String>,

    #[arg(long, default_value_t = 1000)]
    rps: u64,

    #[arg(long, default_value_t = 20)]
    duration: u64,

    /// 这里我们将 concurrency 解释为“bench 侧常驻 sender worker 数”（也等价于最大 in-flight）
    #[arg(long, default_value_t = 200)]
    concurrency: usize,

    /// 输出 CSV（默认写到 results/bench_summary.csv）
    #[arg(long, default_value = "results/bench_summary.csv")]
    out: String,

    /// 是否抓取 /metrics 写入计数器列
    #[arg(long, default_value_t = true)]
    scrape_metrics: bool,

    /// 逗号分隔的 RPS 列表，如：1000,2000,4000,8000
    #[arg(long)]
    sweep_rps: Option<String>,

    /// sweep 时每个档位之间休息 N ms（让系统冷却一下）
    #[arg(long, default_value_t = 500)]
    sweep_pause_ms: u64,

    /// 目标列表：可重复传参
    /// 用法：--target tokio,http://127.0.0.1:8080/score_xgb,http://127.0.0.1:8080/metrics
    ///      --target glommio,http://127.0.0.1:8081/score_xgb,http://127.0.0.1:8081/metrics
    #[arg(long, value_name = "NAME,SCORE_URL,METRICS_URL")]
    target: Vec<String>,

    /// 自动找拐点：从 start 开始，每次 +step，直到 dropped>0 或 p99 明显上扬
    #[arg(long, default_value_t = false)]
    find_knee: bool,

    #[arg(long, default_value_t = 1000)]
    knee_start_rps: u64,

    #[arg(long, default_value_t = 1000)]
    knee_step_rps: u64,

    #[arg(long, default_value_t = 64000)]
    knee_max_rps: u64,

    /// “明显上扬”的阈值：本档 p99 > 上一档 p99 * (1 + knee_rise_pct)
    #[arg(long, default_value_t = 0.30)]
    knee_rise_pct: f64,

    /// “明显上扬”的绝对门槛：本档 p99 必须至少比上一档高 knee_abs_us 才算拐点
    #[arg(long, default_value_t = 200)]
    knee_abs_us: u64,
}

#[derive(Debug, Clone)]
struct Target {
    runtime: String,
    url: String,
    metrics_url: String,
}

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

    // bench side schedule lag (actual - planned)
    bench_lag_us: u64,

    ok: bool,
}

#[derive(Debug, Clone, Default)]
struct Summary {
    samples_ok: usize,
    samples_err: usize,
    dropped: u64,

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
}

#[derive(Debug, Clone)]
struct Job {
    req_id: u64,
    bench_lag_us: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ensure output dir exists
    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let targets = parse_targets(&args)?;
    let xgb_body = load_xgb_body(&args)?;

    // 生成 rps 序列
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

    // 记录每个 runtime 的上一档 p99（用于判断“明显上扬”）
    let mut prev_p99: HashMap<String, u64> = HashMap::new();

    for (i, rps) in rps_list.iter().copied().enumerate() {
        let mut knee_reasons: Vec<String> = Vec::new();

        for t in &targets {
            let summary = run_once(&args, rps, &t.url, &t.metrics_url, &xgb_body).await?;

            if summary.samples_ok == 0 {
                println!(
                    "[runtime={} rps={}] no successful samples collected (errors={}, dropped={})",
                    t.runtime, rps, summary.samples_err, summary.dropped
                );
            } else {
                println!(
                    "[runtime={} rps={}] ok_samples={} err_samples={} dropped={}  p50={}us  p95={}us  p99={}us",
                    t.runtime,
                    rps,
                    summary.samples_ok,
                    summary.samples_err,
                    summary.dropped,
                    summary.p50,
                    summary.p95,
                    summary.p99
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
            println!("\n=== KNEE HIT at rps={} ===", rps);
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
    reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(args.concurrency.max(1))
        .tcp_nodelay(true)
        .build()
        .context("build reqwest client")
}

fn prepare_score_bodies(seed: u64, n: usize) -> anyhow::Result<Vec<Bytes>> {
    // 预生成 N 个请求体，避免压测过程中反复 serde_json 序列化造成噪声和抖动
    // 不改变“字段集合”，只是把“每次生成”改成“提前生成 + 循环使用”
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n.max(1));
    for _ in 0..n.max(1) {
        let req = gen_req(&mut rng);
        let v = serde_json::to_vec(&req).context("serialize ScoreRequest")?;
        out.push(Bytes::from(v));
    }
    Ok(out)
}

async fn run_once(
    args: &Args,
    rps: u64,
    url: &str,
    metrics_url: &str,
    xgb_body: &Bytes,
) -> anyhow::Result<Summary> {
    use serde::Deserialize;

    let client = build_client(args)?;

    // 命中 /score_xgb /score_xgb_async /score_xgb_pool 等
    let is_xgb = url.contains("/score_xgb");

    // 预生成 score 的请求体池（仅在非 xgb 路径用）
    let score_bodies = if is_xgb {
        None
    } else {
        // 用一个固定池大小：至少 1024，最多 8192（够随机、内存也很小）
        let n = args.concurrency.max(256).min(8192);
        Some(Arc::new(
            prepare_score_bodies(42u64, n).context("prepare score bodies")?,
        ))
    };

    // samples channel
    let (sample_tx, mut sample_rx) = mpsc::channel::<Sample>(256_000);

    // collector: gather into vectors (依旧沿用你实验字段，不新增)
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

    // job queue：这是关键！有界队列 + try_send，满了就 dropped（避免制造突发洪峰）
    let job_q_cap = (args.concurrency.max(1) * 4).max(1024).min(200_000);
    let (job_tx, job_rx) = mpsc::channel::<Job>(job_q_cap);

    // 固定 sender workers（替代 per-tick tokio::spawn）
    let workers = args.concurrency.max(1);

    // 仅解析 timings（bench 端不需要完整 ScoreResponse）
    #[derive(Debug, Deserialize)]
    struct RespXgb {
        timings_us: TimingsXgb,
    }
    #[derive(Debug, Deserialize)]
    struct TimingsXgb {
        #[serde(default)]
        parse: u64,
        #[serde(default)]
        feature: u64,
        #[serde(default)]
        router: u64,
        #[serde(default)]
        xgb: u64,
        #[serde(default)]
        l2: u64,
        #[serde(default)]
        serialize: u64,
    }

    // 兼容旧字段 l1
    #[derive(Debug, Deserialize)]
    struct RespL1 {
        timings_us: TimingsL1,
    }
    #[derive(Debug, Deserialize)]
    struct TimingsL1 {
        #[serde(default)]
        parse: u64,
        #[serde(default)]
        feature: u64,
        #[serde(default)]
        router: u64,
        #[serde(default)]
        l1: u64,
        #[serde(default)]
        l2: u64,
        #[serde(default)]
        serialize: u64,
    }

    // worker 共享 rx：用 Mutex 包装（tokio mpsc Receiver 不能 clone）
    // 注意：这不是性能瓶颈，因为每个 job 只锁一次取任务；真正成本在网络/推理上
    let shared_rx = Arc::new(tokio::sync::Mutex::new(job_rx));

    static REQ_SEQ: AtomicU64 = AtomicU64::new(1);
    let url_s = Arc::new(url.to_string());
    let xgb_body = Arc::new(xgb_body.clone());
    let client = Arc::new(client);

    let mut worker_handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let rx = shared_rx.clone();
        let tx = sample_tx.clone();
        let client = client.clone();
        let url = url_s.clone();
        let xgb_body = xgb_body.clone();
        let score_bodies = score_bodies.clone();

        let h = tokio::spawn(async move {
            loop {
                let job = {
                    let mut g = rx.lock().await;
                    g.recv().await
                };
                let Some(job) = job else {
                    break;
                };

                let t0 = Instant::now();
                let mut sample = Sample::default();
                sample.bench_lag_us = job.bench_lag_us;

                let send_res = if is_xgb {
                    client
                        .post(url.as_str())
                        .header("content-type", "application/json")
                        .body((*xgb_body).clone())
                        .send()
                        .await
                } else {
                    // 从预生成池取一个 body（避免 serde+alloc 噪声）
                    let bodies = score_bodies.as_ref().expect("score_bodies");
                    let idx = (job.req_id as usize) % bodies.len();
                    client
                        .post(url.as_str())
                        .header("content-type", "application/json")
                        .body(bodies[idx].clone())
                        .send()
                        .await
                };

                match send_res {
                    Ok(resp) => match resp.bytes().await {
                        Ok(bytes) => {
                            sample.e2e_us = t0.elapsed().as_micros() as u64;

                            if let Ok(r) = serde_json::from_slice::<RespXgb>(&bytes) {
                                sample.parse_us = r.timings_us.parse;
                                sample.feature_us = r.timings_us.feature;
                                sample.router_us = r.timings_us.router;
                                sample.xgb_us = r.timings_us.xgb;
                                sample.l2_us = r.timings_us.l2;
                                sample.serialize_us = r.timings_us.serialize;
                                sample.ok = true;
                            } else if let Ok(r) = serde_json::from_slice::<RespL1>(&bytes) {
                                sample.parse_us = r.timings_us.parse;
                                sample.feature_us = r.timings_us.feature;
                                sample.router_us = r.timings_us.router;
                                sample.xgb_us = r.timings_us.l1;
                                sample.l2_us = r.timings_us.l2;
                                sample.serialize_us = r.timings_us.serialize;
                                sample.ok = true;
                            } else {
                                sample.ok = false;
                            }
                        }
                        Err(_) => sample.ok = false,
                    },
                    Err(_) => sample.ok = false,
                }

                let _ = tx.send(sample).await;
            }
        });
        worker_handles.push(h);
    }
    drop(sample_tx); // collector 只要等所有 worker 退出即可结束

    let start = Instant::now();
    let end = start + Duration::from_secs(args.duration);

    // 固定速率 open-loop（Delay：不追赶补发）
    let ns = (1_000_000_000u64 / rps.max(1)).max(1);
    let interval = Duration::from_nanos(ns);

    let first_tick = tokio::time::Instant::now() + interval;
    let mut tick = tokio::time::interval_at(first_tick, interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // bench 侧滞后统计：actual - planned
    let interval_ns: u128 = interval.as_nanos().max(1);
    let mut seq: u64 = 0;

    let mut bench_lag_us_vec: Vec<u64> =
        Vec::with_capacity(((args.duration as usize).saturating_mul(rps as usize)).min(2_000_000));
    let mut missed_ticks_total: u64 = 0;

    let mut dropped: u64 = 0;

    while Instant::now() < end {
        tick.tick().await;

        // planned tick 时间点 + lag
        let off_ns: u128 = interval_ns.saturating_mul(seq as u128);
        let off_ns_u64: u64 = off_ns.min(u64::MAX as u128) as u64;
        let planned = first_tick + Duration::from_nanos(off_ns_u64);

        let now = tokio::time::Instant::now();
        let lag = now.saturating_duration_since(planned);
        let lag_us: u64 = lag.as_micros() as u64;
        bench_lag_us_vec.push(lag_us);

        if lag.as_nanos() > interval_ns {
            missed_ticks_total += (lag.as_nanos() / interval_ns) as u64;
        }
        seq = seq.wrapping_add(1);

        let req_id = REQ_SEQ.fetch_add(1, Ordering::Relaxed);
        let job = Job {
            req_id,
            bench_lag_us: lag_us,
        };

        // 关键：try_send，队列满就 dropped（不阻塞，不补发，不制造 burst）
        if job_tx.try_send(job).is_err() {
            dropped += 1;
        }
    }

    drop(job_tx);

    for h in worker_handles {
        let _ = h.await;
    }

    let (mut e2e, mut parse, mut feature, mut router, mut xgb, mut l2, mut serialize, ok, err) =
        collector.await?;

    let mut summary = Summary::default();
    summary.samples_ok = ok;
    summary.samples_err = err;
    summary.dropped = dropped;

    bench_lag_us_vec.sort_unstable();
    summary.bench_lag_p99_us = percentile(&bench_lag_us_vec, 99.0);
    summary.bench_lag_max_us = *bench_lag_us_vec.last().unwrap_or(&0);
    summary.missed_ticks_total = missed_ticks_total;

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
        // 注意：这里用单独 client 抓 metrics 不会污染主压测连接池（这一点你之前就做对了）
        if let Ok(resp) = build_client(args)?.get(metrics_url).send().await {
            if let Ok(text) = resp.text().await {
                let (l2_trig, miss, skip, before_l2) = scrape_counters(&text);
                summary.router_l2_trigger_total = l2_trig;
                summary.router_deadline_miss_total = miss;
                summary.router_l2_skipped_budget_total = skip;
                summary.router_timeout_before_l2_total = before_l2;
            }
        }
    }

    Ok(summary)
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
            "ts_ms,runtime,url,metrics_url,rps,duration_s,concurrency,ok,err,dropped,p50_us,p95_us,p99_us,parse_p99_us,feature_p99_us,router_p99_us,xgb_p99_us,l2_p99_us,serialize_p99_us,router_l2_trigger_total,router_deadline_miss_total,router_l2_skipped_budget_total,router_timeout_before_l2_total,bench_lag_p99_us,bench_lag_max_us,missed_ticks_total"
        )?;
    }

    let ts_ms = current_time_ms();
    writeln!(
        f,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        ts_ms,
        escape_csv(runtime),
        escape_csv(url),
        escape_csv(metrics_url),
        rps_used,
        args.duration,
        args.concurrency,
        s.samples_ok,
        s.samples_err,
        s.dropped,
        s.p50,
        s.p95,
        s.p99,
        s.parse_p99,
        s.feature_p99,
        s.router_p99,
        s.xgb_p99,
        s.l2_p99,
        s.serialize_p99,
        opt_f64(s.router_l2_trigger_total),
        opt_f64(s.router_deadline_miss_total),
        opt_f64(s.router_l2_skipped_budget_total),
        opt_f64(s.router_timeout_before_l2_total),
        s.bench_lag_p99_us,
        s.bench_lag_max_us,
        s.missed_ticks_total,
    )?;

    println!("csv appended: {}", out_path.display());
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

fn scrape_counters(text: &str) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let mut l2 = None;
    let mut miss = None;
    let mut skip = None;
    let mut before = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut it = line.split_whitespace();
        let name = match it.next() {
            Some(x) => x,
            None => continue,
        };
        let val = match it.next() {
            Some(v) => v.parse::<f64>().ok(),
            None => None,
        };
        if val.is_none() {
            continue;
        }
        let val = val.unwrap();

        match name {
            "router_l2_trigger_total" => l2 = Some(val),
            "router_deadline_miss_total" => miss = Some(val),
            "router_l2_skipped_budget_total" => skip = Some(val),
            "router_timeout_before_l2_total" => before = Some(val),
            _ => {}
        }
    }

    (l2, miss, skip, before)
}

fn load_xgb_body(args: &Args) -> anyhow::Result<Bytes> {
    if let Some(p) = &args.xgb_body_file {
        let b = std::fs::read(p).with_context(|| format!("read --xgb-body-file: {}", p))?;
        Ok(Bytes::from(b))
    } else {
        Ok(Bytes::from_static(b"{}"))
    }
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
