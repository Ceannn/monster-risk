use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::Value;
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tower_http::trace::TraceLayer;

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
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
}

#[tokio::main]
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

    // config + core
    let cfg = Config::default();

    let core = if let Some(dir) = args.model_dir.as_deref() {
        tracing::info!("XGB enabled, loading model_dir={}", dir);
        Arc::new(AppCore::new_with_xgb(cfg, dir)?)
    } else {
        tracing::info!("XGB disabled, baseline /score only");
        Arc::new(AppCore::new(cfg))
    };

    let state = AppState { core, prom };

    // router
    let app = Router::new()
        .route("/score", post(score))
        .route("/score_xgb", post(score_xgb))
        .route("/metrics", get(metrics))
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

/// Prometheus metrics
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.prom.render()
}
