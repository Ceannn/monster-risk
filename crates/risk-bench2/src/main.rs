use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use clap::Parser;
use hdrhistogram::Histogram;
use http::{header, HeaderValue, Method, Request, Uri};
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper::StatusCode;
use hyper_util::rt::TokioIo;
use rand::{Rng, SeedableRng};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

/// risk-bench2: 每核一线程 + hyper http1 keepalive conn 的 open-loop 压测器。
///
/// 设计目标：
/// - bench 自己不成为瓶颈（每线程独立 runtime、独立连接、独立统计，无共享锁）
/// - open-loop（Poisson 或 fixed interval）能稳定打出 knee 曲线
/// - inflight 有界（bench 侧背压），避免“bench 把自己压爆”污染结论
#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// Target URL (http://host:port/path)
    #[arg(long)]
    url: String,

    /// Target RPS (attempted, open-loop)
    #[arg(long, default_value_t = 8000)]
    rps: u64,

    /// Duration seconds
    #[arg(long, default_value_t = 20)]
    duration: u64,

    /// Warmup seconds (send requests but do not record stats).
    #[arg(long, default_value_t = 0)]
    warmup: u64,

    /// Rolling window in milliseconds for per-thread timeseries CSV (0 disables).
    #[arg(long, default_value_t = 200)]
    window_ms: u64,

    /// Output path/prefix for rolling window CSV.
    /// - If contains "{t}", it will be replaced by the thread index.
    /// - Otherwise, if threads>1, suffix _t{t}.csv is added.
    #[arg(long)]
    window_csv: Option<String>,

    /// Number of shard threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Total concurrency cap (in-flight). Will be divided across threads.
    /// NOTE: For HTTP/1.1, real on-wire in-flight is bounded by total connections.
    #[arg(long, default_value_t = 1024)]
    concurrency: usize,

    /// Connections per thread (keepalive). Each conn has its own send loop.
    #[arg(long, default_value_t = 4)]
    conns_per_thread: usize,

    /// Per-connection queue cap (1 = real in-flight semantics).
    #[arg(long, default_value_t = 1)]
    conn_queue: usize,

    /// Request timeout ms (0 = no timeout)
    #[arg(long, default_value_t = 2000)]
    timeout_ms: u64,

    /// Pacer: poisson | fixed | poisson_spin | fixed_spin
    #[arg(long, default_value = "poisson")]
    pacer: String,

    /// Max jitter added to each schedule tick (microseconds, uniform [0, jitter])
    #[arg(long, default_value_t = 0)]
    pacer_jitter_us: u64,

    /// For *_spin pacers: busy-spin in the last N microseconds before the scheduled tick.
    /// 0 disables spinning (default).
    #[arg(long, default_value_t = 0)]
    pacer_spin_threshold_us: u64,

    /// Max catch-up for fixed pacer (in ticks) when the pacer wakes up late.
    /// If we are behind by more than this, we skip the extra ticks to avoid bursty catch-up.
    #[arg(long, default_value_t = 2)]
    pacer_max_catchup_ticks: u64,

    /// Max catch-up for poisson pacer (in microseconds). If lag exceeds this, we reset next tick base to "now".
    /// 0 means "always reset on any lag".
    #[arg(long, default_value_t = 10_000)]
    pacer_max_catchup_us: u64,

    /// Dense f32le file (binary), layout: rows * (dense_dim * 4 bytes)
    #[arg(long)]
    xgb_dense_file: String,

    /// Dense vector dimension (number of f32)
    #[arg(long)]
    xgb_dense_dim: usize,

    /// Content-Type header (default application/octet-stream)
    #[arg(long, default_value = "application/octet-stream")]
    content_type: String,

    /// Print per-second progress
    #[arg(long, default_value_t = false)]
    progress: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct StageTimingsUs {
    parse: u32,
    feature: u32,
    router: u32,
    xgb: u32,
    l2: u32,
    serialize: u32,
}

#[derive(Clone, Copy, Debug)]
struct Rsk1 {
    status: StatusCode,
    decision: u8,
    score_bits: u32,
    timings: StageTimingsUs,
}

fn decode_rsk1(status: StatusCode, buf: &[u8]) -> Option<Rsk1> {
    if buf.len() != 48 {
        return None;
    }
    if &buf[0..4] != b"RSK1" {
        return None;
    }
    // version, flags
    let _ver = u16::from_le_bytes([buf[4], buf[5]]);
    let _flags = u16::from_le_bytes([buf[6], buf[7]]);
    // trace_id ignored
    // score
    let score_bits = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let decision = buf[20];

    // timings_us[6] u32, starting at offset 24
    let mut read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };

    let t = StageTimingsUs {
        parse: read_u32(24),
        feature: read_u32(28),
        router: read_u32(32),
        xgb: read_u32(36),
        l2: read_u32(40),
        serialize: read_u32(44),
    };

    Some(Rsk1 {
        status,
        decision,
        score_bits,
        timings: t,
    })
}

struct ShardStats {
    ok: u64,
    err: u64,
    dropped: u64,

    /// Dropped because in-flight cap was hit (bench bottleneck)
    drop_inflight_cap: u64,
    /// Dropped because connection queue was full (bench bottleneck)
    drop_conn_queue_full: u64,
    timeout: u64,

    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,

    missed_ticks: u64,

    /// Number of pacer lag samples recorded
    lag_samples: u64,
    /// Number of responses that carried rsk1 stage timings
    rsk1_samples: u64,

    lat_us: Histogram<u64>,
    client_qwait_us: Histogram<u64>,
    lag_us: Histogram<u64>,

    stage_parse: Histogram<u64>,
    stage_feature: Histogram<u64>,
    stage_router: Histogram<u64>,
    stage_xgb: Histogram<u64>,
    stage_l2: Histogram<u64>,
    stage_serialize: Histogram<u64>,
}

impl ShardStats {
    fn new() -> Result<Self> {
        Ok(Self {
            ok: 0,
            err: 0,
            dropped: 0,
            drop_inflight_cap: 0,
            drop_conn_queue_full: 0,
            timeout: 0,

            http_2xx: 0,
            http_429: 0,
            http_5xx: 0,

            missed_ticks: 0,

            lag_samples: 0,
            rsk1_samples: 0,

            lat_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            client_qwait_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            lag_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,

            // HdrHistogram lower bound must be >= 1; we clamp 0 -> 1 when recording.
            stage_parse: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_feature: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_router: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_xgb: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_l2: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_serialize: Histogram::new_with_bounds(1, 60_000_000, 3)?,
        })
    }

    fn merge_from(&mut self, other: &ShardStats) -> Result<()> {
        self.ok += other.ok;
        self.err += other.err;
        self.dropped += other.dropped;
        self.drop_inflight_cap += other.drop_inflight_cap;
        self.drop_conn_queue_full += other.drop_conn_queue_full;
        self.timeout += other.timeout;
        self.http_2xx += other.http_2xx;
        self.http_429 += other.http_429;
        self.http_5xx += other.http_5xx;
        self.missed_ticks += other.missed_ticks;

        self.lag_samples += other.lag_samples;
        self.rsk1_samples += other.rsk1_samples;

        self.lat_us.add(&other.lat_us)?;
        self.client_qwait_us.add(&other.client_qwait_us)?;
        self.lag_us.add(&other.lag_us)?;
        self.stage_parse.add(&other.stage_parse)?;
        self.stage_feature.add(&other.stage_feature)?;
        self.stage_router.add(&other.stage_router)?;
        self.stage_xgb.add(&other.stage_xgb)?;
        self.stage_l2.add(&other.stage_l2)?;
        self.stage_serialize.add(&other.stage_serialize)?;
        Ok(())
    }
}

// ---------------- Rolling window stats (per shard) ----------------

struct WindowAgg {
    attempted: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    dropped: u64,
    drop_inflight_cap: u64,
    drop_conn_queue_full: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    missed_ticks: u64,

    lat_us: Histogram<u64>,
    client_qwait_us: Histogram<u64>,
    lag_us: Histogram<u64>,

    stage_parse: Histogram<u64>,
    stage_feature: Histogram<u64>,
    stage_router: Histogram<u64>,
    stage_xgb: Histogram<u64>,
    stage_l2: Histogram<u64>,
    stage_serialize: Histogram<u64>,
}

impl WindowAgg {
    fn new() -> Result<Self> {
        Ok(Self {
            attempted: 0,
            ok: 0,
            err: 0,
            timeout: 0,
            dropped: 0,
            drop_inflight_cap: 0,
            drop_conn_queue_full: 0,
            http_2xx: 0,
            http_429: 0,
            http_5xx: 0,
            missed_ticks: 0,

            lat_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            client_qwait_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            lag_us: Histogram::new_with_bounds(1, 60_000_000, 3)?,

            stage_parse: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_feature: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_router: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_xgb: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_l2: Histogram::new_with_bounds(1, 60_000_000, 3)?,
            stage_serialize: Histogram::new_with_bounds(1, 60_000_000, 3)?,
        })
    }

    fn reset(&mut self) {
        self.attempted = 0;
        self.ok = 0;
        self.err = 0;
        self.timeout = 0;
        self.dropped = 0;
        self.drop_inflight_cap = 0;
        self.drop_conn_queue_full = 0;
        self.http_2xx = 0;
        self.http_429 = 0;
        self.http_5xx = 0;
        self.missed_ticks = 0;

        self.lat_us.reset();
        self.client_qwait_us.reset();
        self.lag_us.reset();

        self.stage_parse.reset();
        self.stage_feature.reset();
        self.stage_router.reset();
        self.stage_xgb.reset();
        self.stage_l2.reset();
        self.stage_serialize.reset();
    }
}

fn hist_q(hist: &Histogram<u64>, q: f64) -> u64 {
    if hist.len() == 0 {
        0
    } else {
        hist.value_at_quantile(q)
    }
}

fn make_window_path(template: &str, shard_id: usize, threads: usize) -> String {
    if template.contains("{t}") {
        template.replace("{t}", &shard_id.to_string())
    } else if threads > 1 {
        format!("{template}_t{shard_id}.csv")
    } else {
        template.to_string()
    }
}

struct Target {
    uri: Uri,
    host_header: HeaderValue,
    path_and_query: http::uri::PathAndQuery,
    addr: String,
}

fn parse_target(url: &str) -> Result<Target> {
    let uri: Uri = url.parse().context("invalid --url")?;
    let scheme = uri
        .scheme_str()
        .ok_or_else(|| anyhow!("url missing scheme (http://...)"))?;
    if scheme != "http" {
        return Err(anyhow!("only http:// is supported in bench2 for now"));
    }

    let authority = uri
        .authority()
        .ok_or_else(|| anyhow!("url missing authority (host:port)"))?;
    let host = authority.host();
    let port = authority.port_u16().unwrap_or(80);

    let path_and_query = uri
        .path_and_query()
        .cloned()
        .unwrap_or_else(|| http::uri::PathAndQuery::from_static("/"));

    let host_header = HeaderValue::from_str(authority.as_str())?;
    let addr = format!("{}:{}", host, port);

    // quick resolve sanity
    let _ = addr
        .to_socket_addrs()
        .context("failed to resolve host")?
        .next()
        .ok_or_else(|| anyhow!("no resolved address"))?;

    Ok(Target {
        uri,
        host_header,
        path_and_query,
        addr,
    })
}

#[derive(Clone, Copy, Debug)]
enum PacerKind {
    Poisson,
    Fixed,
    PoissonSpin,
    FixedSpin,
}

impl fmt::Display for PacerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacerKind::Poisson => write!(f, "poisson"),
            PacerKind::Fixed => write!(f, "fixed"),
            PacerKind::PoissonSpin => write!(f, "poisson_spin"),
            PacerKind::FixedSpin => write!(f, "fixed_spin"),
        }
    }
}

fn parse_pacer(s: &str) -> Result<PacerKind> {
    match s {
        "poisson" => Ok(PacerKind::Poisson),
        "fixed" => Ok(PacerKind::Fixed),
        // spin variants: same distribution, but busy-spin in the final `--pacer-spin-threshold-us`
        "poisson_spin" | "poisson-spin" => Ok(PacerKind::PoissonSpin),
        "fixed_spin" | "fixed-spin" => Ok(PacerKind::FixedSpin),
        _ => Err(anyhow!(
            "invalid --pacer: {} (poisson|fixed|poisson_spin|fixed_spin)",
            s
        )),
    }
}

async fn tick_wait(pacer: PacerKind, next_at: Instant, spin_threshold_us: u64) {
    // Normal mode: use tokio timer.
    let spin_enabled =
        spin_threshold_us > 0 && matches!(pacer, PacerKind::PoissonSpin | PacerKind::FixedSpin);
    if !spin_enabled {
        tokio::time::sleep_until(tokio::time::Instant::from_std(next_at)).await;
        return;
    }

    let spin_dur = Duration::from_micros(spin_threshold_us);
    if let Some(sleep_until) = next_at.checked_sub(spin_dur) {
        tokio::time::sleep_until(tokio::time::Instant::from_std(sleep_until)).await;
    } else {
        // Too close to the deadline: give other tasks a chance.
        tokio::task::yield_now().await;
    }

    while Instant::now() < next_at {
        std::hint::spin_loop();
    }
}

fn exp_interval_us(rng: &mut rand::rngs::SmallRng, lambda: f64) -> u64 {
    // inverse CDF of exponential: -ln(U)/lambda
    let mut u: f64 = rng.gen();
    if u <= 0.0 {
        u = 1e-12;
    }
    let dt = -u.ln() / lambda; // seconds
    let us = (dt * 1_000_000.0) as u64;
    us.max(1)
}

#[derive(Clone)]
struct PayloadRing {
    base: Bytes,
    row_bytes: usize,
    rows: usize,
}

const PAYLOAD_CACHE_MAX_ROWS: usize = 1_000_000;

impl PayloadRing {
    fn load(path: &str, dense_dim: usize) -> Result<Self> {
        let row_bytes = dense_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("dense_dim too large"))?;
        let data = std::fs::read(path).with_context(|| format!("read dense file: {}", path))?;
        if data.len() < row_bytes {
            return Err(anyhow!(
                "dense file too small: len={} row_bytes={}",
                data.len(),
                row_bytes
            ));
        }
        if data.len() % row_bytes != 0 {
            return Err(anyhow!(
                "dense file size not multiple of row_bytes: len={} row_bytes={}",
                data.len(),
                row_bytes
            ));
        }
        let rows = data.len() / row_bytes;
        Ok(Self {
            base: Bytes::from(data),
            row_bytes,
            rows,
        })
    }

    fn cache_rows(&self) -> Vec<Bytes> {
        (0..self.rows).map(|i| self.row_slice(i)).collect()
    }

    #[inline]
    fn row_slice(&self, idx: usize) -> Bytes {
        let i = idx % self.rows;
        let off = i * self.row_bytes;
        self.base.slice(off..off + self.row_bytes)
    }
}

struct ReqMsg {
    req_id: u64,
    body: Bytes,
    t_sched: Instant,
}

struct RespInfo {
    status: StatusCode,
    latency_us: u64,
    client_qwait_us: u64,
    rsk1: Option<Rsk1>,
}

struct RespEvent {
    req_id: u64,
    info: Option<RespInfo>,
}

struct InflightInfo {
    measuring: bool,
    t_sched: Instant,
}

async fn conn_worker(
    target: Target,
    content_type: HeaderValue,
    mut rx: mpsc::Receiver<ReqMsg>,
    resp_tx: mpsc::Sender<RespEvent>,
) -> Result<()> {
    let stream = TcpStream::connect(&target.addr)
        .await
        .with_context(|| format!("connect {}", target.addr))?;
    stream.set_nodelay(true).ok();

    let io = TokioIo::new(stream);
    let (mut sender, conn) = http1::handshake(io).await.context("http1 handshake")?;

    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("[bench2] conn terminated: {e}");
        }
    });

    // warm ready
    sender.ready().await.context("sender ready")?;

    let base_req = Request::builder()
        .method(Method::POST)
        .uri(target.path_and_query.as_str())
        .header(header::HOST, target.host_header.clone())
        .header(header::CONTENT_TYPE, content_type.clone())
        .body(())
        .map_err(|e| anyhow!("build request: {e}"))?;
    let (base_parts, _) = base_req.into_parts();

    while let Some(msg) = rx.recv().await {
        let req = Request::from_parts(base_parts.clone(), Full::new(msg.body));
        let t_sched = msg.t_sched;
        let req_id = msg.req_id;
        let t0 = Instant::now();
        let client_qwait_us = t0.duration_since(t_sched).as_micros().min(u64::MAX as u128) as u64;
        let res: Result<(StatusCode, Bytes)> = async {
            let resp = sender.send_request(req).await.context("send_request")?;
            let status = resp.status();
            let body_bytes = collect_body(resp.into_body()).await?;
            Ok((status, body_bytes))
        }
        .await;

        let latency_us = t0.elapsed().as_micros() as u64;

        let info = res
            .map(|(status, body_bytes)| RespInfo {
                status,
                latency_us,
                client_qwait_us,
                rsk1: decode_rsk1(status, &body_bytes),
            })
            .ok();

        // receiver might have been dropped; ignore send error
        let _ = resp_tx.send(RespEvent { req_id, info }).await;
    }

    Ok(())
}

async fn collect_body(body: Incoming) -> Result<Bytes> {
    // For RSK1 it's fixed 48 bytes; keep it simple & robust.
    use http_body_util::BodyExt;
    let collected = body
        .collect()
        .await
        .map_err(|e| anyhow!("collect body: {e}"))?;
    Ok(collected.to_bytes())
}

async fn run_shard(
    shard_id: usize,
    threads_total: usize,
    args: Args,
    target: Target,
    payload: PayloadRing,
    inflight_cap: usize,
    rps_shard: f64,
) -> Result<ShardStats> {
    let pacer = parse_pacer(&args.pacer)?;
    let spin_threshold_us = args.pacer_spin_threshold_us;
    let jitter = args.pacer_jitter_us;
    let rsk1_sample_log2 = std::env::var("RISK_BENCH2_RSK1_SAMPLE_LOG2")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
        .min(20);

    let content_type =
        HeaderValue::from_str(&args.content_type).context("invalid content-type header value")?;

    let mut stats = ShardStats::new()?;

    // ---------------- connections ----------------
    let conns = args.conns_per_thread.max(1);
    let conn_queue = args.conn_queue.max(1);
    let mut txs = Vec::with_capacity(conns);
    let resp_chan_cap = (conns * (conn_queue + 1)).max(1024);
    let (resp_tx, mut resp_rx) = mpsc::channel::<RespEvent>(resp_chan_cap);

    for _ in 0..conns {
        let (tx, rx) = mpsc::channel::<ReqMsg>(conn_queue);
        txs.push(tx);
        let target2 = Target {
            uri: target.uri.clone(),
            host_header: target.host_header.clone(),
            path_and_query: target.path_and_query.clone(),
            addr: target.addr.clone(),
        };
        let ct2 = content_type.clone();
        let resp_tx2 = resp_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = conn_worker(target2, ct2, rx, resp_tx2).await {
                eprintln!("[bench2] conn_worker error: {e:?}");
            }
        });
    }

    // ---------------- warmup & windows ----------------
    let start = Instant::now();
    let warmup_dur = Duration::from_secs(args.warmup);
    let measure_start = start + warmup_dur;
    let measure_end = measure_start + Duration::from_secs(args.duration);

    let window_enabled = args.window_ms > 0;
    let window_dur = Duration::from_millis(args.window_ms.max(1) as u64);

    let mut win = if window_enabled {
        Some(WindowAgg::new()?)
    } else {
        None
    };
    let mut win_idx: u64 = 0;
    let mut win_next = measure_start + window_dur;

    let mut win_writer: Option<std::io::BufWriter<std::fs::File>> = if window_enabled {
        let tmpl = args
            .window_csv
            .clone()
            .unwrap_or_else(|| format!("bench2_window_rps{}_t{{t}}.csv", args.rps));
        let p = make_window_path(&tmpl, shard_id, threads_total);
        let f = std::fs::File::create(&p).with_context(|| format!("create window csv: {}", p))?;
        let mut w = std::io::BufWriter::new(f);
        // header
        use std::io::Write;
        writeln!(
            w,
            "t_ms,win_idx,inflight,attempted,ok,err,timeout,dropped,drop_inflight_cap,drop_conn_queue_full,http_2xx,http_429,http_5xx,lat_p50_us,lat_p95_us,lat_p99_us,client_qwait_p99_us,lag_p99_us,lag_max_us,stage_parse_p99_us,stage_feature_p99_us,stage_router_p99_us,stage_xgb_p99_us,stage_l2_p99_us,stage_serialize_p99_us,missed_ticks"
        )?;
        Some(w)
    } else {
        None
    };

    // ---------------- pacer ----------------
    // each shard has its own RNG so no sharing
    let mut rng = rand::rngs::SmallRng::from_entropy();
    let lambda = rps_shard.max(1.0); // per second

    let mut next_at = Instant::now();
    let fixed_period_us = if rps_shard > 0.0 {
        (1_000_000.0 / rps_shard) as u64
    } else {
        1_000_000
    };
    if matches!(pacer, PacerKind::Fixed | PacerKind::FixedSpin) {
        let phase_us = if threads_total > 0 {
            fixed_period_us.saturating_mul(shard_id as u64) / threads_total as u64
        } else {
            0
        };
        next_at += Duration::from_micros(phase_us);
    }

    let timeout_dur = if args.timeout_ms > 0 {
        Some(Duration::from_millis(args.timeout_ms))
    } else {
        None
    };
    let mut inflight: HashMap<u64, InflightInfo> = HashMap::with_capacity(inflight_cap * 2);
    let mut deadlines: BinaryHeap<Reverse<(Instant, u64)>> = BinaryHeap::new();

    let mut seq: u64 = 0;
    let mut last_progress = Instant::now();
    let payload_rows = if payload.rows <= PAYLOAD_CACHE_MAX_ROWS {
        Some(payload.cache_rows())
    } else {
        None
    };
    let mut resp_closed = false;

    // helper: flush one window
    let mut flush_window = |now: Instant,
                            inflight_len: usize,
                            win_idx: u64,
                            win: &mut WindowAgg,
                            w: &mut std::io::BufWriter<std::fs::File>|
     -> Result<()> {
        use std::io::Write;
        let t_ms = if now > measure_start {
            (now - measure_start).as_millis() as u64
        } else {
            0
        };

        let lat_p50 = hist_q(&win.lat_us, 0.50);
        let lat_p95 = hist_q(&win.lat_us, 0.95);
        let lat_p99 = hist_q(&win.lat_us, 0.99);

        let qwait_p99 = hist_q(&win.client_qwait_us, 0.99);

        let lag_p99 = hist_q(&win.lag_us, 0.99);
        let lag_max = if win.lag_us.len() == 0 {
            0
        } else {
            win.lag_us.max()
        };

        let sp = (
            hist_q(&win.stage_parse, 0.99),
            hist_q(&win.stage_feature, 0.99),
            hist_q(&win.stage_router, 0.99),
            hist_q(&win.stage_xgb, 0.99),
            hist_q(&win.stage_l2, 0.99),
            hist_q(&win.stage_serialize, 0.99),
        );

        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            t_ms,
            win_idx,
            inflight_len,
            win.attempted,
            win.ok,
            win.err,
            win.timeout,
            win.dropped,
            win.drop_inflight_cap,
            win.drop_conn_queue_full,
            win.http_2xx,
            win.http_429,
            win.http_5xx,
            lat_p50,
            lat_p95,
            lat_p99,
            qwait_p99,
            lag_p99,
            lag_max,
            sp.0,
            sp.1,
            sp.2,
            sp.3,
            sp.4,
            sp.5,
            win.missed_ticks
        )?;
        Ok(())
    };

    // ---------------- main loop ----------------
    while Instant::now() < measure_end || !inflight.is_empty() {
        let in_window = Instant::now() < measure_end;
        let timeout_enabled = timeout_dur.is_some();
        let next_deadline = if timeout_enabled {
            deadlines.peek().map(|entry| (entry.0).0)
        } else {
            None
        };
        let mut deadline_sleep = if timeout_enabled {
            if let Some(deadline) = next_deadline {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline))
            } else {
                tokio::time::sleep(Duration::from_secs(3600))
            }
        } else {
            tokio::time::sleep(Duration::from_secs(3600))
        };
        tokio::pin!(deadline_sleep);

        tokio::select! {
                    // tick: schedule request
                    _ = tick_wait(pacer, next_at, spin_threshold_us), if in_window => {
                        let now = Instant::now();
                        let measuring = now >= measure_start;

                        // pacer lag + bounded catch-up (RefineA)
        // Record lag histogram, but only count "missed_ticks" when we SKIP excessive catch-up.
        let mut base_at = next_at;
        if now > next_at {
            let lag_us = (now - next_at).as_micros() as u64;

            if measuring {
                let _ = stats.lag_us.record(lag_us.max(1));
                stats.lag_samples += 1;

                if let Some(w) = win.as_mut() {
                    let _ = w.lag_us.record(lag_us.max(1));
                }
            }

            match pacer {
                PacerKind::Fixed | PacerKind::FixedSpin => {
                    let period = fixed_period_us.max(1);
                    let missed = lag_us / period;
                    let max_catchup = args.pacer_max_catchup_ticks;
                    if missed > max_catchup {
                        let skip = missed - max_catchup;
                        // Advance base_at so next_at won't stay far in the past (prevents burst catch-up).
                        let adv_us = (skip as u128) * (period as u128);
                        base_at += Duration::from_micros(adv_us.min(u64::MAX as u128) as u64);

                        if measuring {
                            stats.missed_ticks += skip;
                            if let Some(w) = win.as_mut() {
                                w.missed_ticks += skip;
                            }
                        }
                    }
                }
                PacerKind::Poisson | PacerKind::PoissonSpin => {
                    let max_us = args.pacer_max_catchup_us;
                    if max_us == 0 || lag_us > max_us {
                        // Reset poisson base to "now" to avoid tight catch-up loops after a long stall.
                        base_at = now;

                        if measuring {
                            stats.missed_ticks += 1;
                            if let Some(w) = win.as_mut() {
                                w.missed_ticks += 1;
                            }
                        }
                    }
                }
            }
        }

        // compute next schedule
                        let dt_us = match pacer {
                            PacerKind::Poisson | PacerKind::PoissonSpin => exp_interval_us(&mut rng, lambda),
                            PacerKind::Fixed | PacerKind::FixedSpin => fixed_period_us.max(1),
                        };
                        let jitter_us = if jitter > 0 { rng.gen_range(0..=jitter) } else { 0 };
                        next_at = base_at + Duration::from_micros(dt_us + jitter_us);

                        // bench-side inflight backpressure
                        if inflight.len() >= inflight_cap {
                            if measuring {
                                stats.dropped += 1;
                                stats.drop_inflight_cap += 1;
                                if let Some(w) = win.as_mut() {
                                    w.dropped += 1;
                                    w.drop_inflight_cap += 1;
                                    w.attempted += 1;
                                }
                            }
                            continue;
                        }

                        let req_id = seq;
                        let row_idx = ((req_id as usize).wrapping_mul(1315423911) ^ shard_id) % payload.rows;
                        let body = match payload_rows.as_ref() {
                            Some(rows) => rows[row_idx].clone(),
                            None => payload.row_slice(row_idx),
                        };
                        seq = seq.wrapping_add(1);

                        let t_sched = now;
                        let mut msg = ReqMsg { req_id, body, t_sched };
                        let start_idx = (seq as usize) % txs.len();
                        let mut sent = false;
                        for i in 0..txs.len() {
                            let conn_idx = (start_idx + i) % txs.len();
                            match txs[conn_idx].try_send(msg) {
                                Ok(()) => {
                                    sent = true;
                                    break;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Full(m)) => {
                                    msg = m;
                                    continue;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(m)) => {
                                    msg = m;
                                    continue;
                                }
                            }
                        }
                        if !sent {
                            if measuring {
                                stats.dropped += 1;
                                stats.drop_conn_queue_full += 1;
                                if let Some(w) = win.as_mut() {
                                    w.dropped += 1;
                                    w.drop_conn_queue_full += 1;
                                    w.attempted += 1;
                                }
                            }
                            continue;
                        }

                        // count attempted at send time (measured only)
                        if measuring {
                            if let Some(w) = win.as_mut() {
                                w.attempted += 1;
                            }
                        }

                        inflight.insert(
                            req_id,
                            InflightInfo {
                                measuring,
                                t_sched,
                            },
                        );
                        if let Some(timeout) = timeout_dur {
                            deadlines.push(Reverse((t_sched + timeout, req_id)));
                        }

                        if args.progress && last_progress.elapsed() >= Duration::from_secs(1) {
                            eprintln!(
                                "[bench2] shard={} inflight={} t={:.1}s (measuring={})",
                                shard_id,
                                inflight.len(),
                                start.elapsed().as_secs_f64(),
                                measuring
                            );
                            last_progress = Instant::now();
                        }
                    }

                    // window tick
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(win_next)), if window_enabled && Instant::now() >= measure_start && (Instant::now() < measure_end || !inflight.is_empty()) => {
                        if let (Some(wagg), Some(writer)) = (win.as_mut(), win_writer.as_mut()) {
                            let now = Instant::now();
                            flush_window(now, inflight.len(), win_idx, wagg, writer)?;
                            wagg.reset();
                            win_idx += 1;
                            win_next += window_dur;
                        }
                    }

                    // deadline tick (timeout)
                    _ = &mut deadline_sleep, if timeout_enabled => {
                        let now = Instant::now();
                        while let Some(Reverse((deadline, req_id))) = deadlines.peek().copied() {
                            if deadline > now {
                                break;
                            }
                            let _ = deadlines.pop();
                            let info = inflight.remove(&req_id);
                            let Some(info) = info else {
                                continue;
                            };
                            if !info.measuring {
                                continue;
                            }
                            stats.timeout += 1;
                            let qwait_us = now
                                .duration_since(info.t_sched)
                                .as_micros()
                                .min(u64::MAX as u128) as u64;
                            let _ = stats.client_qwait_us.record(qwait_us.max(1));
                            if let Some(w) = win.as_mut() {
                                let _ = w.client_qwait_us.record(qwait_us.max(1));
                                w.timeout += 1;
                            }
                        }
                    }

                    // response completion
                    resp = resp_rx.recv(), if !resp_closed => {
                        let Some(event) = resp else {
                            resp_closed = true;
                            continue;
                        };
                        let Some(info) = inflight.remove(&event.req_id) else {
                            continue;
                        };
                        if !info.measuring {
                            continue;
                        }

                        if let Some(info) = event.info {
                            if info.status.is_success() {
                                stats.ok += 1;
                                stats.http_2xx += 1;
                                if let Some(w) = win.as_mut() {
                                    w.ok += 1;
                                    w.http_2xx += 1;
                                }
                            } else {
                                stats.err += 1;
                                if info.status == StatusCode::TOO_MANY_REQUESTS {
                                    stats.http_429 += 1;
                                    if let Some(w) = win.as_mut() { w.err += 1; w.http_429 += 1; }
                                } else if info.status.is_server_error() {
                                    stats.http_5xx += 1;
                                    if let Some(w) = win.as_mut() { w.err += 1; w.http_5xx += 1; }
                                } else {
                                    if let Some(w) = win.as_mut() { w.err += 1; }
                                }
                            }

                            let _ = stats.lat_us.record(info.latency_us.max(1));
                            if let Some(w) = win.as_mut() {
                                let _ = w.lat_us.record(info.latency_us.max(1));
                            }
                            let _ = stats
                                .client_qwait_us
                                .record(info.client_qwait_us.max(1));
                            if let Some(w) = win.as_mut() {
                                let _ = w
                                    .client_qwait_us
                                    .record(info.client_qwait_us.max(1));
                            }

                            if let Some(r) = info.rsk1 {
                                stats.rsk1_samples += 1;

                                // Sample stage hist to reduce overhead at high QPS.
                                let sample_ok = if rsk1_sample_log2 == 0 {
                                    true
                                } else {
                                    let mask = (1u64 << rsk1_sample_log2) - 1;
                                    (stats.rsk1_samples & mask) == 0
                                };
                                if sample_ok {
                                    let _ = stats.stage_parse.record((r.timings.parse as u64).max(1));
                                    let _ = stats.stage_feature.record((r.timings.feature as u64).max(1));
                                    let _ = stats.stage_router.record((r.timings.router as u64).max(1));
                                    let _ = stats.stage_xgb.record((r.timings.xgb as u64).max(1));
                                    let _ = stats.stage_l2.record((r.timings.l2 as u64).max(1));
                                    let _ = stats.stage_serialize.record((r.timings.serialize as u64).max(1));

                                    if let Some(w) = win.as_mut() {
                                        let _ = w.stage_parse.record((r.timings.parse as u64).max(1));
                                        let _ = w.stage_feature.record((r.timings.feature as u64).max(1));
                                        let _ = w.stage_router.record((r.timings.router as u64).max(1));
                                        let _ = w.stage_xgb.record((r.timings.xgb as u64).max(1));
                                        let _ = w.stage_l2.record((r.timings.l2 as u64).max(1));
                                        let _ = w.stage_serialize.record((r.timings.serialize as u64).max(1));
                                    }
                                }
                            }
                        } else {
                            // timeout or internal error
                            stats.timeout += 1;
                            let qwait_us = Instant::now()
                                .duration_since(info.t_sched)
                                .as_micros()
                                .min(u64::MAX as u128) as u64;
                            let _ = stats.client_qwait_us.record(qwait_us.max(1));
                            if let Some(w) = win.as_mut() {
                                let _ = w.client_qwait_us.record(qwait_us.max(1));
                                w.timeout += 1;
                            }
                        }
                    }
                }
    }

    // final flush
    if window_enabled {
        if let (Some(wagg), Some(writer)) = (win.as_mut(), win_writer.as_mut()) {
            flush_window(Instant::now(), inflight.len(), win_idx, wagg, writer)?;
            use std::io::Write;
            let _ = writer.flush();
        }
    }

    Ok(stats)
}

fn pick_threads(args: &Args) -> usize {
    if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let target = parse_target(&args.url)?;
    let payload = PayloadRing::load(&args.xgb_dense_file, args.xgb_dense_dim)?;

    let threads = pick_threads(&args).max(1);
    let conc_total = args.concurrency.max(threads);
    let inflight_per = (conc_total + threads - 1) / threads;
    let conn_queue = args.conn_queue.max(1);
    let conns_total = threads * args.conns_per_thread.max(1);
    let inflight_per_eff = inflight_per.min(args.conns_per_thread.max(1) * conn_queue);

    let rps_total = args.rps.max(1);
    let rps_per = rps_total as f64 / threads as f64;

    eprintln!(
        "[bench2] url={} rps={} warmup={}s duration={}s threads={} inflight_total={} inflight_per={} inflight_per_eff={} conns_per_thread={} conn_queue={} conns_total={} pacer={} jitter_us={} catchup_ticks={} catchup_us={} timeout_ms={} payload_rows={} dim={} content_type={} window_ms={} window_csv={}",
        args.url,
        rps_total,
        args.warmup,
        args.duration,
        threads,
        conc_total,
        inflight_per,
        inflight_per_eff,
        args.conns_per_thread,
        conn_queue,
        conns_total,
        args.pacer,
        args.pacer_jitter_us,
        args.pacer_max_catchup_ticks,
        args.pacer_max_catchup_us,
        args.timeout_ms,
        payload.rows,
        args.xgb_dense_dim,
        args.content_type,
        args.window_ms,
        args.window_csv.clone().unwrap_or_else(|| "(auto)".to_string()),
    );

    let args_arc = Arc::new(args);
    let payload_arc = Arc::new(payload);

    let start = Instant::now();

    let mut handles = Vec::with_capacity(threads);
    for shard_id in 0..threads {
        let args2 = (*args_arc).clone();
        let target2 = Target {
            uri: target.uri.clone(),
            host_header: target.host_header.clone(),
            path_and_query: target.path_and_query.clone(),
            addr: target.addr.clone(),
        };
        let payload2 = (*payload_arc).clone();

        let h = thread::spawn(move || -> Result<ShardStats> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .enable_time()
                .build()
                .context("build per-thread tokio runtime")?;

            rt.block_on(run_shard(
                shard_id,
                threads,
                args2,
                target2,
                payload2,
                inflight_per,
                rps_per,
            ))
        });
        handles.push(h);
    }

    let mut total = ShardStats::new()?;
    for h in handles {
        let st = h.join().map_err(|_| anyhow!("thread panicked"))??;
        total.merge_from(&st)?;
    }

    let wall_elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let measured_elapsed = (args_arc.duration as f64).max(1e-9);

    let attempted = (total.ok + total.err + total.timeout + total.dropped) as f64;
    let attempted_rps = attempted / measured_elapsed;
    let ok_rps = total.ok as f64 / measured_elapsed;
    let drop_rps = total.dropped as f64 / measured_elapsed;
    let drop_pct = if attempted > 0.0 {
        (total.dropped as f64) * 100.0 / attempted
    } else {
        0.0
    };
    let drop_inflight_pct = if attempted > 0.0 {
        (total.drop_inflight_cap as f64) * 100.0 / attempted
    } else {
        0.0
    };
    let drop_conn_pct = if attempted > 0.0 {
        (total.drop_conn_queue_full as f64) * 100.0 / attempted
    } else {
        0.0
    };

    let p50 = hist_q(&total.lat_us, 0.50);
    let p95 = hist_q(&total.lat_us, 0.95);
    let p99 = hist_q(&total.lat_us, 0.99);

    println!(
        "[bench2 rps={}] ok_samples={} err_samples={} timeout={} dropped={} p50={}us p95={}us p99={}us attempted_rps={:.1} ok_rps={:.1} drop_rps={:.1} drop_pct={:.3}% (wall_elapsed={:.2}s)",
        args_arc.rps,
        total.ok,
        total.err,
        total.timeout,
        total.dropped,
        p50,
        p95,
        p99,
        attempted_rps,
        ok_rps,
        drop_rps,
        drop_pct,
        wall_elapsed,
    );

    println!(
        "[bench2] http: 2xx={} 429={} 5xx={} timeout={} missed_ticks_total={} drop_inflight_cap={} drop_conn_queue_full={}",
        total.http_2xx,
        total.http_429,
        total.http_5xx,
        total.timeout,
        total.missed_ticks,
        total.drop_inflight_cap,
        total.drop_conn_queue_full
    );

    // stage p99 (from RSK1 if available)
    let sp = (
        hist_q(&total.stage_parse, 0.99),
        hist_q(&total.stage_feature, 0.99),
        hist_q(&total.stage_router, 0.99),
        hist_q(&total.stage_xgb, 0.99),
        hist_q(&total.stage_l2, 0.99),
        hist_q(&total.stage_serialize, 0.99),
    );

    println!(
        "[bench2] stage_p99(us): parse={} feature={} router={} xgb={} l2={} serialize={} (rsk1_samples={})",
        sp.0, sp.1, sp.2, sp.3, sp.4, sp.5, total.rsk1_samples
    );

    let lag_p99 = hist_q(&total.lag_us, 0.99);
    let lag_max = if total.lag_us.len() == 0 {
        0
    } else {
        total.lag_us.max()
    };

    println!(
        "[bench2] pacer_lag_p99={}us pacer_lag_max={}us (lag_samples={})",
        lag_p99, lag_max, total.lag_samples
    );

    let qwait_p99 = hist_q(&total.client_qwait_us, 0.99);
    let qwait_max = if total.client_qwait_us.len() == 0 {
        0
    } else {
        total.client_qwait_us.max()
    };
    println!(
        "[bench2] client_qwait_p99={}us client_qwait_max={}us (qwait_samples={})",
        qwait_p99,
        qwait_max,
        total.client_qwait_us.len()
    );
    println!(
        "[bench2] bench_limits: drop_inflight_cap={} ({:.3}%) drop_conn_queue_full={} ({:.3}%) client_qwait_p99={}us",
        total.drop_inflight_cap,
        drop_inflight_pct,
        total.drop_conn_queue_full,
        drop_conn_pct,
        qwait_p99
    );

    // ---------------- quality gate ----------------
    // FAIL should mean: *the bench* is preventing us from hitting the target (or keeping the in-flight cap),
    // not merely that the OS scheduler injected micro-jitter while we still achieved target RPS.
    let mut fail_reasons: Vec<String> = Vec::new();
    let mut warn_reasons: Vec<String> = Vec::new();

    // Hard signals that the load generator is the bottleneck / not stable at the requested load:
    if total.drop_inflight_cap > 0 {
        fail_reasons.push(format!(
            "drop_inflight_cap={} ({:.3}%)",
            total.drop_inflight_cap,
            drop_inflight_pct
        ));
    }
    if total.drop_conn_queue_full > 0 {
        fail_reasons.push(format!(
            "drop_conn_queue_full={} ({:.3}%)",
            total.drop_conn_queue_full,
            drop_conn_pct
        ));
    }
    if total.timeout > 0 {
        fail_reasons.push(format!("timeout={}", total.timeout));
    }
    // If we cannot even attempt close to target, we are bench-limited.
    if attempted_rps < (rps_total as f64) * 0.98 {
        fail_reasons.push(format!("attempted_rps={:.1} (<98% target)", attempted_rps));
    }

    // Soft signals: timing jitter. Useful for tuning (e.g. enable *_spin pacer), but not an automatic FAIL.
    if total.missed_ticks > 0 {
        warn_reasons.push(format!("missed_ticks_total={}", total.missed_ticks));
    }
    if lag_p99 >= 5_000 {
        warn_reasons.push(format!("pacer_lag_p99={}us", lag_p99));
    }
    if lag_max >= 50_000 {
        warn_reasons.push(format!("pacer_lag_max={}us", lag_max));
    }
    if total.client_qwait_us.len() > 0 {
        let target_period_us =
            ((1_000_000.0 * threads as f64) / (rps_total as f64)).max(1.0) as u64;
        let qwait_fail =
            qwait_p99 >= 5_000 || qwait_p99 >= target_period_us.saturating_mul(10);
        if qwait_fail {
            fail_reasons.push(format!("client_qwait_p99={}us", qwait_p99));
        } else if qwait_p99 >= 1_000 {
            warn_reasons.push(format!("client_qwait_p99={}us", qwait_p99));
        }
        if qwait_max >= 50_000 {
            warn_reasons.push(format!("client_qwait_max={}us", qwait_max));
        }
    }

    if fail_reasons.is_empty() && warn_reasons.is_empty() {
        println!("[bench2] quality_gate: PASS");
    } else if fail_reasons.is_empty() {
        println!(
            "[bench2] quality_gate: PASS (with warnings) warnings={}",
            warn_reasons.join(", ")
        );
    } else {
        // include warnings so the print is still actionable
        let mut all = fail_reasons;
        all.extend(warn_reasons);
        println!(
            "[bench2] quality_gate: FAIL (bench-limited) reasons={}",
            all.join(", ")
        );
    }
    Ok(())
}
