use anyhow::Context;
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
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

    #[arg(long, default_value_t = 1000)]
    rps: u64,

    #[arg(long, default_value_t = 20)]
    duration: u64,

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TimingsUs {
    parse: u64,
    feature: u64,
    router: u64,
    l1: u64,
    l2: u64,
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
    feature_us: u64,
    router_us: u64,
    l1_us: u64,
    l2_us: u64,
    serialize_us: u64,

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

    // stage p99（你可以在论文里直接对比 Tokio vs Glommio 的 stage 尾巴）
    e2e_p99: u64,
    feature_p99: u64,
    router_p99: u64,
    l1_p99: u64,
    l2_p99: u64,
    serialize_p99: u64,

    // counters from /metrics（可能为空）
    router_l2_trigger_total: Option<f64>,
    router_deadline_miss_total: Option<f64>,
    router_l2_skipped_budget_total: Option<f64>,
    router_timeout_before_l2_total: Option<f64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ensure output dir exists
    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // 如果传了 --sweep-rps 就跑 sweep，否则只跑单次 args.rps
    let rps_list = if let Some(s) = &args.sweep_rps {
        parse_rps_list(s)?
    } else {
        vec![args.rps]
    };

    for (i, rps) in rps_list.iter().copied().enumerate() {
        let summary = run_once(&args, rps).await?;

        // 打印（带上当前 rps）
        if summary.samples_ok == 0 {
            println!(
                "[rps={}] no successful samples collected (errors={}, dropped={})",
                rps, summary.samples_err, summary.dropped
            );
        } else {
            println!(
                "[rps={}] ok_samples={} err_samples={} dropped={}  p50={}us  p95={}us  p99={}us  (target p99<=10000us)",
                rps, summary.samples_ok, summary.samples_err, summary.dropped, summary.p50, summary.p95, summary.p99
            );
            println!(
                "[rps={}] stage_p99(us): feature={} router={} l1={} l2={} serialize={}",
                rps,
                summary.feature_p99,
                summary.router_p99,
                summary.l1_p99,
                summary.l2_p99,
                summary.serialize_p99
            );
        }

        // 写 CSV：注意这里用 rps（不是 args.rps）
        write_csv_row(&out_path, &args, rps, &summary)?;

        // sweep 档位间暂停（最后一个不暂停）
        if args.sweep_rps.is_some() && i + 1 < rps_list.len() && args.sweep_pause_ms > 0 {
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

async fn run_once(args: &Args, rps: u64) -> anyhow::Result<Summary> {
    // 每次 run 独立 client，避免跨档位复用连接造成“状态污染”
    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(args.concurrency)
        .tcp_nodelay(true)
        .build()?;

    // channel for samples
    let (tx, mut rx) = mpsc::channel::<Sample>(1_000_000);

    // collector task: gather into vectors
    let collector = tokio::spawn(async move {
        let mut e2e = Vec::<u64>::with_capacity(200_000);
        let mut feature = Vec::<u64>::with_capacity(200_000);
        let mut router = Vec::<u64>::with_capacity(200_000);
        let mut l1 = Vec::<u64>::with_capacity(200_000);
        let mut l2 = Vec::<u64>::with_capacity(200_000);
        let mut serialize = Vec::<u64>::with_capacity(200_000);

        let mut ok = 0usize;
        let mut err = 0usize;

        while let Some(s) = rx.recv().await {
            if s.ok {
                ok += 1;
                e2e.push(s.e2e_us);
                feature.push(s.feature_us);
                router.push(s.router_us);
                l1.push(s.l1_us);
                l2.push(s.l2_us);
                serialize.push(s.serialize_us);
            } else {
                err += 1;
            }
        }

        (e2e, feature, router, l1, l2, serialize, ok, err)
    });

    let start = Instant::now();
    let end = start + Duration::from_secs(args.duration);

    // fixed-rate launcher（用传入的 rps）
    let ns = (1_000_000_000u64 / rps.max(1)).max(1);
    let interval = Duration::from_nanos(ns);
    let mut tick = tokio::time::interval(interval);

    let sem = Arc::new(tokio::sync::Semaphore::new(args.concurrency));

    // tracking
    let mut dropped: u64 = 0;
    let seed = 42u64;
    static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

    while Instant::now() < end {
        tick.tick().await;

        let permit = match sem.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                dropped += 1;
                continue;
            }
        };

        let tx2 = tx.clone();
        let client2 = client.clone();
        let url = args.url.clone();
        let req_id = REQ_SEQ.fetch_add(1, Ordering::Relaxed);

        tokio::spawn(async move {
            let _permit = permit;

            let mut rng = StdRng::seed_from_u64(seed ^ req_id);
            let req = gen_req(&mut rng);

            let t0 = Instant::now();
            let mut sample = Sample::default();

            match client2.post(url).json(&req).send().await {
                Ok(resp) => match resp.bytes().await {
                    Ok(bytes) => {
                        sample.e2e_us = t0.elapsed().as_micros() as u64;
                        if let Ok(sr) = serde_json::from_slice::<ScoreResponse>(&bytes) {
                            sample.feature_us = sr.timings_us.feature;
                            sample.router_us = sr.timings_us.router;
                            sample.l1_us = sr.timings_us.l1;
                            sample.l2_us = sr.timings_us.l2;
                            sample.serialize_us = sr.timings_us.serialize;
                            sample.ok = true;
                        } else {
                            sample.ok = false;
                        }
                    }
                    Err(_) => sample.ok = false,
                },
                Err(_) => sample.ok = false,
            }

            let _ = tx2.send(sample).await;
        });
    }

    drop(tx);

    let (mut e2e, mut feature, mut router, mut l1, mut l2, mut serialize, ok, err) =
        collector.await?;

    let mut summary = Summary::default();
    summary.samples_ok = ok;
    summary.samples_err = err;
    summary.dropped = dropped;

    if ok > 0 {
        // sort for percentiles
        e2e.sort_unstable();
        feature.sort_unstable();
        router.sort_unstable();
        l1.sort_unstable();
        l2.sort_unstable();
        serialize.sort_unstable();

        summary.p50 = percentile(&e2e, 50.0);
        summary.p95 = percentile(&e2e, 95.0);
        summary.p99 = percentile(&e2e, 99.0);

        summary.e2e_p99 = percentile(&e2e, 99.0);
        summary.feature_p99 = percentile(&feature, 99.0);
        summary.router_p99 = percentile(&router, 99.0);
        summary.l1_p99 = percentile(&l1, 99.0);
        summary.l2_p99 = percentile(&l2, 99.0);
        summary.serialize_p99 = percentile(&serialize, 99.0);
    }

    if args.scrape_metrics {
        if let Ok(resp) = client.get(&args.metrics_url).send().await {
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
            "ts_ms,url,rps,duration_s,concurrency,ok,err,dropped,p50_us,p95_us,p99_us,feature_p99_us,router_p99_us,l1_p99_us,l2_p99_us,serialize_p99_us,router_l2_trigger_total,router_deadline_miss_total,router_l2_skipped_budget_total,router_timeout_before_l2_total"
        )?;
    }

    let ts_ms = current_time_ms();
    writeln!(
        f,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        ts_ms,
        escape_csv(&args.url),
        rps_used,
        args.duration,
        args.concurrency,
        s.samples_ok,
        s.samples_err,
        s.dropped,
        s.p50,
        s.p95,
        s.p99,
        s.feature_p99,
        s.router_p99,
        s.l1_p99,
        s.l2_p99,
        s.serialize_p99,
        opt_f64(s.router_l2_trigger_total),
        opt_f64(s.router_deadline_miss_total),
        opt_f64(s.router_l2_skipped_budget_total),
        opt_f64(s.router_timeout_before_l2_total),
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

/// 从 Prometheus 文本格式中抓取几个关键 counter
/// 从 Prometheus 文本格式中抓取几个关键 counter（无 regex 依赖）
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

        // 形如：metric_name 123
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
