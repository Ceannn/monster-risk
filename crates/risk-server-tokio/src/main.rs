use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse},
    routing::{get, post},
    Json, Router,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{config::Config, pipeline::AppCore, schema::{ScoreRequest, ScoreResponse}};
use std::{net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
}

#[tokio::main]
async fn main() {
    // tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // metrics
    let prom = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder");

    let cfg = Config::default();
    let core = Arc::new(AppCore::new(cfg));

    let state = AppState { core, prom };

    let app = Router::new()
        .route("/score", post(score))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!("risk-server-tokio listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
        .await
        .unwrap();
}

async fn score(State(st): State<AppState>, Json(req): Json<ScoreRequest>) -> Result<Json<ScoreResponse>, (StatusCode, String)> {
    // 这里 parse 耗时几乎忽略；为了示例保留 timings_us.parse 字段（后续可在 extractor 中统计）
    let resp = st.core.score(req);
    Ok(Json(resp))
}

async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, st.prom.render())
}
