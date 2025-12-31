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
use risk_core::schema::ScoreResponse;
use serde_json::{Map, Value};
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tokio::sync::Semaphore;
use tower::limit::ConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::BoxError;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
mod pin;
use risk_core::{config::Config, pipeline::AppCore, schema::ScoreRequest};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// 可选：启用 XGB 在线推理（指向模型目录）
    /// 目录里应包含：ieee_xgb.bin / policy.json / feature_names.json / cat_maps.json.gz（以及其它导出文件）
    #[arg(long)]
    model_dir: Option<String>,

    /// 启动后预热次数（仅当启用 --model-dir 时生效；0=不预热）
    #[arg(long, default_value_t = 100)]
    warmup_iters: usize,
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
    xgb_blocking_sem: Arc<Semaphore>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    // logging（如果你项目里已有 tracing 初始化，可以删掉这段）
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // metrics recorder
    let prom = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder");
    // ✅ 让 summary / histogram 这类聚合分位数能“刷新出来”

    let upkeep = prom.clone();
    tokio::spawn(async move {
        let mut itv = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            itv.tick().await;
            upkeep.run_upkeep();
        }
    });
    // config + core
    let cfg = Config::default();

    let core = if let Some(dir) = args.model_dir.as_deref() {
        tracing::info!("XGB enabled, loading model_dir={}", dir);
        let core = Arc::new(AppCore::new_with_xgb(cfg, dir)?);

        if args.warmup_iters > 0 {
            let t = Instant::now();
            tracing::info!("warmup_xgb start iters={}", args.warmup_iters);
            core.warmup_xgb(args.warmup_iters)?;
            tracing::info!("warmup_xgb done cost={}ms", t.elapsed().as_millis());
        }

        core
    } else {
        tracing::info!("XGB disabled, baseline /score only");
        Arc::new(AppCore::new(cfg))
    };
    // IO 核：只放 Tokio runtime + HTTP/metrics/upkeep
    pin::pin_tokio_runtime_workers(&[18, 20, 22, 24])?;
    let state = AppState {
        core,
        prom,
        xgb_blocking_sem: Arc::new(Semaphore::new(64)),
    }; // ✅ 建议先 64；你想更狠就 32
    let xgb_routes = Router::new()
        .route("/score_xgb", post(score_xgb))
        .route("/score_xgb_async", post(score_xgb_async))
        // 防止请求体太大（按你实际 xgb_body 大小调；先给 256KB 很安全）
        .layer(DefaultBodyLimit::max(256 * 1024))
        // 关键：先 shed 再限流（过载直接拒绝，别排队）
        .layer(
            ServiceBuilder::new()
                // ✅ 把 tower error 变成 HTTP 响应 -> Router error 变为 Infallible
                .layer(HandleErrorLayer::new(|err: BoxError| async move {
                    // 过载/限流时统一返回 503（你也可以改成 429）
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("xgb overloaded: {err}"),
                    )
                }))
                // ✅ 过载直接 shed（不排队、不读 body），防 OOM 的关键闸门
                .layer(LoadShedLayer::new())
                // ✅ 并发闸门：建议从 64/128 起
                .layer(ConcurrencyLimitLayer::new(128)),
        );
    // router
    let app = Router::new()
        .route("/score", post(score))
        .route("/metrics", get(metrics))
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

/// baseline：结构化 ScoreRequest（你现在 bench 默认打的那条）
async fn score(State(state): State<AppState>, Json(req): Json<ScoreRequest>) -> impl IntoResponse {
    let resp = state.core.score(req);
    Json(resp)
}

/// XGB：接受宽表 JSON object（IEEE-CIS 那类字段）
/// - 如果 body 是 {"features": {...}} 则取 features
/// - 如果 body 本身就是 {...} 则直接用
async fn score_xgb(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let t_parse = Instant::now();

    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid json body: {e}")).into_response()
        }
    };

    let parse_us = t_parse.elapsed().as_micros() as u64;

    let obj = match v {
        Value::Object(m) => {
            if let Some(Value::Object(features)) = m.get("features") {
                features.clone()
            } else {
                m
            }
        }
        _ => return (StatusCode::BAD_REQUEST, "expected json object").into_response(),
    };

    match state.core.score_xgb(parse_us, &obj) {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("score_xgb failed: {e:#}"),
        )
            .into_response(),
    }
}

async fn score_xgb_async(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    // 1) 先背压：忙就拒绝（不要解析 JSON，不要分配 Map）
    let permit = match state.xgb_blocking_sem.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return (StatusCode::TOO_MANY_REQUESTS, "xgb busy").into_response(),
    };

    let core = state.core.clone();

    // 2) 解析 + 推理全部丢到 blocking 线程池里
    let join = tokio::task::spawn_blocking(move || -> anyhow::Result<ScoreResponse> {
        let _permit = permit; // ✅ closure 结束才释放

        let t_parse = Instant::now();
        let v: serde_json::Value = serde_json::from_slice(&body).context("invalid json body")?;
        let parse_us = t_parse.elapsed().as_micros() as u64;

        let obj = match v {
            serde_json::Value::Object(mut m) => {
                // ✅ 不 clone：move 出 features
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

// 注意：下面这个路径按你工程真实 crate 名调整
use risk_core::xgb_pool::XgbPoolError;

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
        Value::Object(m) => {
            if let Some(Value::Object(features)) = m.get("features") {
                features.clone()
            } else {
                m
            }
        }
        _ => return (StatusCode::BAD_REQUEST, "expected json object").into_response(),
    };

    match state.core.score_xgb_pool_async(parse_us, &obj).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => {
            // ✅ 队列满：429（背压）
            if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                if matches!(pe, XgbPoolError::QueueFull) {
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

/// Prometheus metrics
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.prom.render()
}
