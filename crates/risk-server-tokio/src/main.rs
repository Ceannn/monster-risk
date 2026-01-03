use std::{net::SocketAddr, str::FromStr, sync::Arc, time::Instant};

use anyhow::Context;
use axum::{
    body::Bytes,
    error_handling::HandleErrorLayer,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{
    config::Config,
    pipeline::AppCore,
    schema::{ScoreRequest, ScoreResponse},
    xgb_pool::XgbPoolError,
};
use tower::{BoxError, ServiceBuilder};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "risk-server-tokio", version, about)]
struct Args {
    /// Listen address, e.g. 127.0.0.1:8080
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// L1 model directory (required)
    #[arg(long, value_name = "DIR")]
    model_dir: String,

    /// L2 model directory (optional). If provided, enables L1/L2 cascade inside risk-core.
    #[arg(long = "model-l2-dir", value_name = "DIR")]
    model_l2_dir: Option<String>,

    /// Max in-flight HTTP requests (tower concurrency_limit)
    #[arg(long, default_value_t = 4096)]
    max_in_flight: usize,
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
}

fn env_usize(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_value)
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn install_prometheus() -> PrometheusHandle {
    PrometheusBuilder::new()
        .add_global_label("service", "risk-server-tokio")
        .install_recorder()
        .expect("install prometheus recorder")
}

/// Parse JSON body into a JSON object map.
/// Accepts either:
///  1) {"features": {...}, ...}  -> use the "features" object
///  2) {...}                    -> use the root object
fn json_obj_from_body(body: &Bytes) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let v: serde_json::Value = serde_json::from_slice(body).context("invalid json")?;

    if let Some(features) = v.get("features").and_then(|x| x.as_object()) {
        return Ok(features.clone());
    }
    if let Some(obj) = v.as_object() {
        return Ok(obj.clone());
    }

    anyhow::bail!("json must be an object (or contain an object field `features`)")
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(st): State<AppState>) -> String {
    st.prom.render()
}

/// Full pipeline endpoint (if you still want it).
/// NOTE: risk-core 的 score() 在你项目里是同步的，所以这里不 await。
async fn score(State(st): State<AppState>, Json(req): Json<ScoreRequest>) -> Json<ScoreResponse> {
    Json(st.core.score(req))
}

/// Baseline: single-model, but still CPU-bound; we push it into spawn_blocking.
async fn score_xgb(State(st): State<AppState>, body: Bytes) -> Response {
    let t0 = Instant::now();
    let obj = match json_obj_from_body(&body) {
        Ok(obj) => obj,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let parse_us = t0.elapsed().as_micros() as u64;

    let core = st.core.clone();
    let res = tokio::task::spawn_blocking(move || core.score_xgb(parse_us, &obj))
        .await
        .map_err(|e| anyhow::anyhow!(e))
        .and_then(|x| x);

    match res {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => {
            error!(error = %e, "score_xgb failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

/// Main endpoint: XGB pool (and cascade inside risk-core if enabled).
/// We parse JSON here to get obj reference, then call score_xgb_pool_async(parse_us, obj).
async fn score_xgb_pool(State(st): State<AppState>, body: Bytes) -> Response {
    let t0 = Instant::now();
    let v: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("invalid json: {e}")).into_response(),
    };
    let parse_us = t0.elapsed().as_micros() as u64;

    // Borrow the object map across the await (v lives for the whole function).
    let obj = if let Some(features) = v.get("features").and_then(|x| x.as_object()) {
        features
    } else if let Some(obj) = v.as_object() {
        obj
    } else {
        return (StatusCode::BAD_REQUEST, "json must be an object").into_response();
    };

    let res = st.core.score_xgb_pool_async(parse_us, obj).await;

    match res {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => {
            // ✅ 把“可预期的背压”映射为 429
            if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                match pe {
                    XgbPoolError::QueueFull => {
                        return (StatusCode::TOO_MANY_REQUESTS, "QueueFull").into_response()
                    }
                    XgbPoolError::DeadlineExceeded => {
                        return (StatusCode::TOO_MANY_REQUESTS, "DeadlineExceeded").into_response()
                    }
                    XgbPoolError::WorkerDown => {
                        // worker 掉了属于服务不可用（和背压不一样）
                        return (StatusCode::SERVICE_UNAVAILABLE, "WorkerDown").into_response();
                    }
                    // 兼容未来 enum 扩展（不炸 match）
                    _ => {}
                }
            }

            error!(error = %e, "score_xgb_pool failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

/// tower 的 load_shed / concurrency_limit 早拒绝会走到这里。
/// 我们统一变成 429 overloaded（不再出现你日志里的 503 latency=0ms）。
async fn handle_tower_overload(err: BoxError) -> Response {
    warn!(error = %err, "request rejected by middleware");
    (StatusCode::TOO_MANY_REQUESTS, "overloaded").into_response()
}

async fn async_main(
    args: Args,
    worker_threads: usize,
    max_blocking_threads: usize,
) -> anyhow::Result<()> {
    init_tracing();
    info!(
        "tokio runtime: worker_threads={} max_blocking_threads={}",
        worker_threads, max_blocking_threads
    );

    let prom = install_prometheus();

    let cfg = Config::default();

    let core = AppCore::new_with_xgb_l1_l2(cfg, &args.model_dir, args.model_l2_dir.as_deref())
        .context("init AppCore")?;

    if let Some(dir2) = args.model_l2_dir.as_deref() {
        info!(
            "XGB cascade enabled: l1_dir={} l2_dir={}",
            args.model_dir, dir2
        );
    } else {
        info!("XGB single-model: l1_dir={}", args.model_dir);
    }

    let st = AppState {
        core: Arc::new(core),
        prom,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/score", post(score))
        .route("/score_xgb", post(score_xgb))
        .route("/score_xgb_pool", post(score_xgb_pool))
        .with_state(st)
        .layer(
            ServiceBuilder::new()
                // ✅ 关键：HandleErrorLayer 必须包在最外层，才能把 overload 变成 429
                .layer(HandleErrorLayer::new(handle_tower_overload))
                .layer(tower::load_shed::LoadShedLayer::new())
                .layer(tower::limit::ConcurrencyLimitLayer::new(args.max_in_flight))
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(DefaultMakeSpan::new().include_headers(false))
                        .on_failure(DefaultOnFailure::new()),
                ),
        );

    let addr = SocketAddr::from_str(&args.listen).context("invalid --listen")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("listening on http://{}", addr);

    axum::serve(listener, app.into_make_service())
        .await
        .context("server failed")?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    // 继续支持你现在的环境变量启动方式
    let worker_threads = env_usize("TOKIO_WORKER_THREADS", 4);
    let max_blocking_threads = env_usize("TOKIO_MAX_BLOCKING_THREADS", 4);

    let args = Args::parse();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(worker_threads)
        .max_blocking_threads(max_blocking_threads)
        .build()
        .context("build tokio runtime")?;

    rt.block_on(async_main(args, worker_threads, max_blocking_threads))
}
