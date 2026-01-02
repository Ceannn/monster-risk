use anyhow::Context;
use axum::extract::DefaultBodyLimit;
use axum::{
    body::Bytes,
    error_handling::HandleErrorLayer,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{Map, Value};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;
use tower::limit::ConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::BoxError;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

mod pin;

use risk_core::{
    config::Config,
    pipeline::AppCore,
    schema::{Decision, ReasonItem, ScoreRequest, ScoreResponse, TimingsUs},
    xgb_pool::{XgbPool, XgbPoolError},
    xgb_runtime::XgbRuntime,
};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// L1 模型目录（兼容你之前的 --model-dir 语义）
    /// 目录里应包含：ieee_xgb.ubj/ieee_xgb.bin、policy.json、feature_names.json、cat_maps.json.gz
    #[arg(long)]
    model_dir: Option<String>,

    /// L2 模型目录（可选；启用后 /score_xgb_pool 会走 L1->L2 级联）
    #[arg(long)]
    model_l2_dir: Option<String>,

    /// 启动后预热次数（TLS 直调用路径用；pool 自己会 warmup）
    #[arg(long, default_value_t = 100)]
    warmup_iters: usize,
}

#[derive(Clone)]
struct AppState {
    core_l1: Arc<AppCore>,
    core_l2: Option<Arc<AppCore>>,
    prom: PrometheusHandle,
    xgb_blocking_sem: Arc<Semaphore>,
}

#[derive(Clone, Debug)]
struct PoolConf {
    threads: usize,
    cap_total: usize,
    warmup_iters: usize,
    pin_cpus: Option<Vec<usize>>,
}

fn env_usize(keys: &[&str], default: usize) -> usize {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            if let Ok(x) = v.parse::<usize>() {
                return x;
            }
        }
    }
    default
}

fn env_cpu_list(keys: &[&str]) -> Option<Vec<usize>> {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            let mut out = Vec::new();
            for part in v.split(',') {
                let s = part.trim();
                if s.is_empty() {
                    continue;
                }
                if let Ok(x) = s.parse::<usize>() {
                    out.push(x);
                } else {
                    eprintln!("[WARN] invalid cpu id in {k}={v}: '{s}'");
                }
            }
            if !out.is_empty() {
                return Some(out);
            }
        }
    }
    None
}

fn default_pool_threads() -> usize {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    avail.min(8).max(1)
}

fn read_pool_conf_l1() -> PoolConf {
    let dt = default_pool_threads();
    let threads = env_usize(&["XGB_L1_POOL_THREADS", "XGB_POOL_THREADS"], dt);
    let cap_total = env_usize(
        &["XGB_L1_POOL_QUEUE_CAP", "XGB_POOL_QUEUE_CAP"],
        threads.saturating_mul(64).max(64),
    );
    let warmup_iters = env_usize(&["XGB_L1_POOL_WARMUP_ITERS", "XGB_POOL_WARMUP_ITERS"], 50);
    let pin_cpus = env_cpu_list(&["XGB_L1_POOL_PIN_CPUS", "XGB_POOL_PIN_CPUS"]);
    PoolConf {
        threads,
        cap_total,
        warmup_iters,
        pin_cpus,
    }
}

fn read_pool_conf_l2() -> PoolConf {
    let dt = default_pool_threads();
    let threads = env_usize(&["XGB_L2_POOL_THREADS"], dt.max(1));
    let cap_total = env_usize(
        &["XGB_L2_POOL_QUEUE_CAP"],
        threads.saturating_mul(64).max(64),
    );
    let warmup_iters = env_usize(&["XGB_L2_POOL_WARMUP_ITERS"], 50);
    let pin_cpus = env_cpu_list(&["XGB_L2_POOL_PIN_CPUS"]);
    PoolConf {
        threads,
        cap_total,
        warmup_iters,
        pin_cpus,
    }
}

/// 用指定 pool 配置构造一个带 XGB runtime + pool 的 AppCore
fn build_core_with_pool(cfg: Config, model_dir: &Path, pc: PoolConf) -> anyhow::Result<AppCore> {
    let mut core = AppCore::new(cfg);

    let xgb = Arc::new(XgbRuntime::load_from_dir(model_dir)?);
    let feature_names = Arc::new(xgb.feature_names.clone());

    let pool = XgbPool::new_with_pinning(
        xgb.model_path.clone(),
        feature_names,
        pc.threads,
        pc.cap_total,
        pc.warmup_iters,
        pc.pin_cpus,
    )?;

    core.xgb = Some(xgb);
    core.xgb_pool = Some(Arc::new(pool));
    Ok(core)
}

#[inline]
fn decision_from_xgb_str(s: &str) -> Decision {
    match s {
        "deny" => Decision::Deny,
        "review" => Decision::ManualReview,
        "allow" => Decision::Allow,
        _ => Decision::ManualReview,
    }
}

/// 只做 “score-only” 的 pool 推理（不算 contrib）
/// - 带 queue-wait budget（动态 reserve = compute_p99_est + safety）
async fn score_only_with_budget(
    core: &AppCore,
    obj: &Map<String, Value>,
    deadline: Instant,
    safety_us: u64,
    reserve_clamp_hi: u64,
) -> anyhow::Result<(f32, u64 /*feat_us*/, u64 /*xgb_total_us*/)> {
    let xgb = core.xgb.as_ref().context("xgb not enabled")?;
    let pool = core.xgb_pool.as_ref().context("xgb_pool not enabled")?;

    // feature build
    let t_feat = Instant::now();
    let row: Vec<f32> = xgb.build_row(obj);
    let feat_us = t_feat.elapsed().as_micros() as u64;
    metrics::histogram!("stage_feature_us").record(feat_us as f64);

    // budget
    let remain_us: u64 = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO)
        .as_micros()
        .min(u128::from(u64::MAX)) as u64;

    let compute_p99_us = pool.xgb_compute_p99_est_us();
    let reserve_us: u64 = compute_p99_us
        .saturating_add(safety_us)
        .clamp(800, reserve_clamp_hi);

    let budget_us: u64 = remain_us.saturating_sub(reserve_us);
    if budget_us == 0 {
        metrics::counter!("router_deadline_miss_total").increment(1);
        return Err(anyhow::Error::new(XgbPoolError::DeadlineExceeded));
    }

    let t_xgb = Instant::now();
    let rx = pool
        .try_submit_with_budget(row, 0, budget_us)
        .map_err(|e| anyhow::Error::new(e))?;

    let out = rx
        .await
        .map_err(|_| anyhow::Error::new(XgbPoolError::Canceled))??;

    let xgb_us = t_xgb.elapsed().as_micros() as u64;
    metrics::histogram!("stage_xgb_us").record(xgb_us as f64);

    Ok((out.score, feat_us, xgb_us))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // metrics recorder
    let prom = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder");

    // upkeep: flush quantiles periodically
    let upkeep = prom.clone();
    tokio::spawn(async move {
        let mut itv = tokio::time::interval(Duration::from_secs(1));
        loop {
            itv.tick().await;
            upkeep.run_upkeep();
        }
    });

    // IO 核：只放 Tokio runtime + HTTP/metrics/upkeep
    pin::pin_tokio_runtime_workers(&[18, 20, 22, 24])?;

    let cfg = Config::default();

    // -------- build L1 core --------
    let core_l1: Arc<AppCore> = if let Some(dir) = args.model_dir.as_deref() {
        let dir = PathBuf::from(dir);
        tracing::info!("L1 enabled, model_dir={}", dir.display());

        let pc = read_pool_conf_l1();
        tracing::info!("L1 pool conf: {:?}", pc);

        let core = build_core_with_pool(cfg.clone(), &dir, pc)?;
        let core = Arc::new(core);

        if args.warmup_iters > 0 {
            let t = Instant::now();
            tracing::info!("warmup (TLS path) start iters={}", args.warmup_iters);
            core.warmup_xgb(args.warmup_iters)?;
            tracing::info!("warmup done cost={}ms", t.elapsed().as_millis());
        }

        core
    } else {
        tracing::info!("XGB disabled, baseline /score only");
        Arc::new(AppCore::new(cfg.clone()))
    };

    // -------- build L2 core (optional) --------
    let core_l2: Option<Arc<AppCore>> = if let Some(dir2) = args.model_l2_dir.as_deref() {
        let dir2 = PathBuf::from(dir2);
        tracing::info!("L2 enabled, model_dir={}", dir2.display());

        let pc = read_pool_conf_l2();
        tracing::info!("L2 pool conf: {:?}", pc);

        let core = build_core_with_pool(cfg.clone(), &dir2, pc)?;
        let core = Arc::new(core);

        if args.warmup_iters > 0 {
            let _ = core.warmup_xgb(args.warmup_iters);
        }

        Some(core)
    } else {
        None
    };

    let state = AppState {
        core_l1,
        core_l2,
        prom,
        xgb_blocking_sem: Arc::new(Semaphore::new(64)),
    };

    let xgb_routes = Router::new()
        .route("/score_xgb", post(score_xgb))
        .route("/score_xgb_async", post(score_xgb_async))
        .layer(DefaultBodyLimit::max(256 * 1024))
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|err: BoxError| async move {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("xgb overloaded: {err}"),
                    )
                }))
                .layer(LoadShedLayer::new())
                .layer(ConcurrencyLimitLayer::new(128)),
        );

    let app = Router::new()
        .route("/score", post(score))
        .route("/metrics", get(metrics))
        // 主力压测口：默认 score-only，并支持 L1->L2 级联
        .route("/score_xgb_pool", post(score_xgb_pool))
        .merge(xgb_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = args
        .listen
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --listen '{}': {e}", args.listen))?;

    tracing::info!("risk-server-tokio listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;

    Ok(())
}

/// baseline：结构化 ScoreRequest
async fn score(State(state): State<AppState>, Json(req): Json<ScoreRequest>) -> impl IntoResponse {
    let resp = state.core_l1.score(req);
    Json(resp)
}

/// XGB：TLS 直调用（对照用）
async fn score_xgb(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let t_parse = Instant::now();

    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid json body: {e}")).into_response()
        }
    };
    let parse_us = t_parse.elapsed().as_micros() as u64;

    let obj: Map<String, Value> = match v {
        Value::Object(mut m) => {
            if let Some(Value::Object(features)) = m.remove("features") {
                features
            } else {
                m
            }
        }
        _ => return (StatusCode::BAD_REQUEST, "expected json object").into_response(),
    };

    match state.core_l1.score_xgb(parse_us, &obj) {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("score_xgb failed: {e:#}"),
        )
            .into_response(),
    }
}

/// 危险对照组：把 parse+infer 丢 blocking（你保留它用于论文对照很有价值）
async fn score_xgb_async(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let permit = match state.xgb_blocking_sem.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return (StatusCode::TOO_MANY_REQUESTS, "xgb busy").into_response(),
    };

    let core = state.core_l1.clone();

    let join = tokio::task::spawn_blocking(move || -> anyhow::Result<ScoreResponse> {
        let _permit = permit;

        let t_parse = Instant::now();
        let v: serde_json::Value = serde_json::from_slice(&body).context("invalid json body")?;
        let parse_us = t_parse.elapsed().as_micros() as u64;

        let obj = match v {
            serde_json::Value::Object(mut m) => {
                if let Some(serde_json::Value::Object(features)) = m.remove("features") {
                    features
                } else {
                    m
                }
            }
            _ => anyhow::bail!("expected json object"),
        };

        core.score_xgb(parse_us, &obj)
    });

    match join.await {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, format!("{e:#}")).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("join failed: {e}"),
        )
            .into_response(),
    }
}

async fn score_xgb_pool(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let t_parse = Instant::now();

    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid json body: {e}")).into_response()
        }
    };
    let parse_us = t_parse.elapsed().as_micros() as u64;

    let obj: Map<String, Value> = match v {
        Value::Object(mut m) => {
            if let Some(Value::Object(features)) = m.remove("features") {
                features
            } else {
                m
            }
        }
        _ => return (StatusCode::BAD_REQUEST, "expected json object").into_response(),
    };

    // -------- L1 -> (optional) L2 cascade --------
    match score_xgb_pool_cascade(state.core_l1.clone(), state.core_l2.clone(), parse_us, &obj).await
    {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => {
            if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                if matches!(pe, XgbPoolError::QueueFull | XgbPoolError::DeadlineExceeded) {
                    return (StatusCode::TOO_MANY_REQUESTS, "xgb pool busy").into_response();
                }
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("score_xgb_pool failed: {e:#}"),
            )
                .into_response()
        }
    }
}

async fn score_xgb_pool_cascade(
    core_l1: Arc<AppCore>,
    core_l2: Option<Arc<AppCore>>,
    parse_us: u64,
    obj: &Map<String, Value>,
) -> anyhow::Result<ScoreResponse> {
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_millis(core_l1.cfg.slo_p99_ms);

    let mut timings = TimingsUs::default();
    timings.parse = parse_us;

    // L1 score-only
    let (s1, feat_us, xgb_us) = score_only_with_budget(&core_l1, obj, deadline, 400, 8_000).await?;
    timings.feature = feat_us;
    timings.xgb = xgb_us;

    let xgb1 = core_l1.xgb.as_ref().context("l1 xgb not enabled")?;
    let mut score: f64 = s1 as f64;
    let mut decision: Decision = decision_from_xgb_str(xgb1.decide(s1));

    let t_router = Instant::now();
    let mut reason: Vec<ReasonItem> = Vec::new();

    // L2 only if L1 says "review"
    if let (Decision::ManualReview, Some(core2)) = (&decision, core_l2) {
        metrics::counter!("router_l2_trigger_total").increment(1);

        // L2 的 budget 更保守一点（一般更重/更慢）
        match score_only_with_budget(&core2, obj, deadline, 600, 12_000).await {
            Ok((s2, feat2_us, l2_us)) => {
                // 把 L2 的 feature build + infer 全算进 l2 stage（更符合“第二层成本”）
                let _ = feat2_us;
                timings.l2 = l2_us;

                let xgb2 = core2.xgb.as_ref().context("l2 xgb not enabled")?;
                score = s2 as f64;
                decision = decision_from_xgb_str(xgb2.decide(s2));

                reason.push(ReasonItem {
                    signal: "layer2_used".into(),
                    value: 1.0,
                    baseline_p95: 1.0,
                    direction: "info".into(),
                });
            }
            Err(e) => {
                // ✅ L2 失败不影响请求：回退 L1 结果（这能显著减少 429）
                if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                    if matches!(pe, XgbPoolError::QueueFull | XgbPoolError::DeadlineExceeded) {
                        metrics::counter!("router_l2_skipped_budget_total").increment(1);
                    } else {
                        metrics::counter!("router_timeout_before_l2_total").increment(1);
                    }
                } else {
                    metrics::counter!("router_timeout_before_l2_total").increment(1);
                }
                reason.push(ReasonItem {
                    signal: "layer2_fallback".into(),
                    value: 1.0,
                    baseline_p95: 1.0,
                    direction: "info".into(),
                });
            }
        }
    }

    timings.router = t_router.elapsed().as_micros() as u64;
    metrics::histogram!("stage_router_us").record(timings.router as f64);

    let mut resp = ScoreResponse {
        trace_id: uuid::Uuid::new_v4(),
        score,
        decision,
        reason,
        timings_us: timings,
    };

    let t_ser = Instant::now();
    let _ = serde_json::to_vec(&resp);
    resp.timings_us.serialize = t_ser.elapsed().as_micros() as u64;
    metrics::histogram!("stage_serialize_us").record(resp.timings_us.serialize as f64);

    metrics::histogram!("e2e_us").record(t0.elapsed().as_micros() as f64);

    Ok(resp)
}

/// Prometheus metrics
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.prom.render()
}
