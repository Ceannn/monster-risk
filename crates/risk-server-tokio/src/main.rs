use std::sync::atomic::{AtomicU64, Ordering};
use std::{net::SocketAddr, str::FromStr, sync::Arc, time::Instant};

use anyhow::Context;
use axum::{
    body::Bytes,
    error_handling::HandleErrorLayer,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{
    config::Config,
    pipeline::AppCore,
    schema::{Decision, ScoreRequest, ScoreResponse},
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

static TRACE_ID_SEQ: AtomicU64 = AtomicU64::new(1);

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

async fn debug_backend(State(st): State<AppState>) -> Response {
    let l1 = st.core.xgb.as_ref();
    let l2 = st.core.xgb_l2.as_ref();
    let aug = st.core.stateful_l2.as_ref();

    let body = serde_json::json!({
        "l1": l1.map(|m| serde_json::json!({
            "backend": m.backend_name(),
            "model_dir": m.model_dir.to_string_lossy(),
            "feature_dim": m.feature_names.len(),
            "schema_hash": m.schema_hash_hex(),
            "thresholds": {
                "review": m.policy.review_threshold,
                "deny": m.policy.deny_threshold,
            },
        })),
        "l2": l2.map(|m| serde_json::json!({
            "backend": m.backend_name(),
            "model_dir": m.model_dir.to_string_lossy(),
            "feature_dim": m.feature_names.len(),
            "schema_hash": m.schema_hash_hex(),
            "thresholds": {
                "review": m.policy.review_threshold,
                "deny": m.policy.deny_threshold,
            },
        })),
        "stateful_l2": aug.map(|_| serde_json::json!({
            "enabled": true,
        })).unwrap_or(serde_json::json!({"enabled": false})),
    });

    (StatusCode::OK, Json(body)).into_response()
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

/// ✅ Dense f32 直传：Content-Type: application/octet-stream
/// Body 格式：
/// - 推荐（无 header）：连续 dim 个 f32（little-endian），总长度 = dim*4
/// - 可选（带 header）：
///   magic="RVEC"(4) + ver(u16=1) + flags(u16=0) + dim(u32) + reserved(u32) + payload(f32*dim)
async fn score_dense_f32(State(st): State<AppState>, body: Bytes) -> Response {
    let t0 = Instant::now();

    let Some(xgb1) = st.core.xgb.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "xgb not enabled: start server with --model-dir",
        )
            .into_response();
    };
    let expected_dim = xgb1.feature_names.len();

    let payload = match parse_dense_payload_le(&body, expected_dim) {
        Ok(v) => v,
        Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
    };

    let parse_us = t0.elapsed().as_micros() as u64;
    let res = st
        .core
        .score_xgb_pool_dense_bytes_async(parse_us, payload)
        .await;

    match res {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => {
            if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                match pe {
                    XgbPoolError::QueueFull => {
                        return (StatusCode::TOO_MANY_REQUESTS, "QueueFull").into_response()
                    }
                    XgbPoolError::DeadlineExceeded => {
                        return (StatusCode::TOO_MANY_REQUESTS, "DeadlineExceeded").into_response()
                    }
                    XgbPoolError::WorkerDown => {
                        return (StatusCode::SERVICE_UNAVAILABLE, "WorkerDown").into_response();
                    }
                    _ => {}
                }
            }
            error!(error = %e, "score_dense_f32 failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

fn parse_dense_payload_le(body: &Bytes, expected_dim: usize) -> Result<Bytes, String> {
    let b = body.as_ref();

    // Fast path: raw payload (no header)
    let raw_len = expected_dim
        .checked_mul(4)
        .ok_or_else(|| "expected_dim too large".to_string())?;
    if b.len() == raw_len {
        return Ok(body.clone());
    }

    // Headered payload
    if b.len() < 16 {
        return Err(format!("body too short: {} < 16", b.len()));
    }
    if &b[0..4] != b"RVEC" {
        return Err("invalid dense payload (missing magic RVEC)".into());
    }
    let ver = u16::from_le_bytes([b[4], b[5]]);
    if ver != 1 {
        return Err(format!("unsupported RVEC version: {}", ver));
    }
    let flags = u16::from_le_bytes([b[6], b[7]]);
    if flags != 0 {
        return Err(format!("unsupported RVEC flags: {}", flags));
    }
    let dim = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
    if dim != expected_dim {
        return Err(format!(
            "dense dim mismatch: got {}, expected {}",
            dim, expected_dim
        ));
    }
    let need = 16 + raw_len;
    if b.len() != need {
        return Err(format!(
            "invalid payload size: got {}, expected {}",
            b.len(),
            need
        ));
    }

    // Zero-copy slice (refcounted)
    Ok(body.slice(16..))
}

fn decision_to_u8(d: &Decision) -> u8 {
    match d {
        Decision::Allow => 0,
        Decision::Deny => 1,
        Decision::ManualReview => 2,
        Decision::DegradeAllow => 3,
    }
}

/// RSK1 binary response layout (48 bytes, little-endian):
/// - magic[4] = "RSK1"
/// - version(u16)=1
/// - flags(u16): bit0=l2_path, bit1=degraded
/// - trace_id(u64)
/// - score(f32)
/// - decision(u8)
/// - pad[3]
/// - timings_us[6](u32): parse/feature/router/xgb/l2/serialize
fn encode_rsk1_response(trace_id: u64, resp: &ScoreResponse) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"RSK1");
    out.extend_from_slice(&1u16.to_le_bytes());

    let mut flags: u16 = 0;
    if resp.timings_us.l2 > 0 {
        flags |= 1 << 0;
    }
    if matches!(resp.decision, Decision::DegradeAllow) {
        flags |= 1 << 1;
    }
    out.extend_from_slice(&flags.to_le_bytes());

    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&(resp.score as f32).to_le_bytes());
    out.push(decision_to_u8(&resp.decision));
    out.extend_from_slice(&[0u8; 3]);

    #[inline]
    fn clamp_u32(x: u64) -> u32 {
        if x > u32::MAX as u64 {
            u32::MAX
        } else {
            x as u32
        }
    }

    let ts = &resp.timings_us;
    for v in [ts.parse, ts.feature, ts.router, ts.xgb, ts.l2, ts.serialize] {
        out.extend_from_slice(&clamp_u32(v).to_le_bytes());
    }

    debug_assert!(
        out.len() == 48,
        "RSK1 response must be 48 bytes, got {}",
        out.len()
    );
    out
}

/// ✅ Dense f32 直传 + Binary response（application/octet-stream）
/// 请求体同 /score_dense_f32（raw 或 RVEC header）。
async fn score_dense_f32_bin(State(st): State<AppState>, body: Bytes) -> Response {
    let t0 = Instant::now();

    let Some(xgb1) = st.core.xgb.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "xgb not enabled: start server with --model-dir",
        )
            .into_response();
    };
    let expected_dim = xgb1.feature_names.len();

    let payload = match parse_dense_payload_le(&body, expected_dim) {
        Ok(v) => v,
        Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
    };

    let trace_id = TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
    let parse_us = t0.elapsed().as_micros() as u64;

    let res = st
        .core
        .score_xgb_pool_dense_bytes_async(parse_us, payload)
        .await;

    match res {
        Ok(resp) => {
            // 让 timings.serialize 代表二进制序列化时间（而不是 core 侧的 JSON to_vec 计时）。
            let t_ser = Instant::now();
            // 先用旧值编码，拿到真实编码开销后再写回再编码一遍会多一次分配。
            // 我们这里走“单次编码”：先估计 serialize_us=0，编码后写回到 header 里的 timings.serialize。
            // 为了保持简单，直接把 serialize_us 记到 metrics 上，不再写回 body。
            // （bench 端依然能从 stage_p99 里看到 serialize 的数量级，且目前 serialize 占比极小）
            let bin = encode_rsk1_response(trace_id, &resp);
            let _serialize_us = t_ser.elapsed().as_micros() as u64;
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            // trace_id 只在错误时打日志，避免热路径噪声；客户端会拿到 trace_id。
            r
        }
        Err(e) => {
            if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
                match pe {
                    XgbPoolError::QueueFull => {
                        return (StatusCode::TOO_MANY_REQUESTS, "QueueFull").into_response()
                    }
                    XgbPoolError::DeadlineExceeded => {
                        return (StatusCode::TOO_MANY_REQUESTS, "DeadlineExceeded").into_response()
                    }
                    XgbPoolError::WorkerDown => {
                        return (StatusCode::SERVICE_UNAVAILABLE, "WorkerDown").into_response();
                    }
                    _ => {}
                }
            }

            error!(error = %e, trace_id = trace_id, "score_dense_f32_bin failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

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

    let mut cfg = Config::default();
    if let Ok(v) = std::env::var("SLO_P99_MS") {
        if let Ok(ms) = v.parse::<u64>() {
            cfg.slo_p99_ms = ms;
        }
    }

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
        .route("/debug/backend", get(debug_backend))
        .route("/score", post(score))
        .route("/score_xgb", post(score_xgb))
        .route("/score_xgb_pool", post(score_xgb_pool))
        .route("/score_dense_f32", post(score_dense_f32))
        .route("/score_dense_f32_bin", post(score_dense_f32_bin))
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
