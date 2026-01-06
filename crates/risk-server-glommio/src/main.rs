use clap::Parser;
use bytes::{Bytes, BytesMut};
use futures_lite::{AsyncReadExt, AsyncWriteExt};
use glommio::net::TcpListener;
use glommio::sync::{Permit, Semaphore};
use glommio::{LocalExecutorBuilder, Placement};

use std::io::Write as _;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{
    config::Config,
    pipeline::AppCore,
    schema::ScoreRequest,
    util::now_us,
    xgb_pool::XgbPoolError,
};

use serde_json::{Map, Value};

use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

static TRACE_ID_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: usize,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// 可选：启用 XGB 在线推理（指向模型目录）
    #[arg(long)]
    model_dir: Option<String>,

    /// 可选：启用 L2 模型（指向 L2 模型目录）
    #[arg(long)]
    model_l2_dir: Option<String>,

    /// HTTP 层最大并发（超出直接 429），用于保护服务端排队不会失控
    #[arg(long, default_value_t = 4096)]
    max_in_flight: usize,

    /// 启动后预热次数（仅当启用 --model-dir 时生效；0=不预热）
    #[arg(long, default_value_t = 100)]
    warmup_iters: usize,
}
fn main() -> anyhow::Result<()> {
    // tracing

    let args = Args::parse();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // metrics recorder（进程内全局一次）
    let prom = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder");
    let mut cfg = Config::default();

    // 与 tokio server 保持一致：允许用环境变量覆盖端到端预算（用于超时/降级/背压）。
    if let Ok(v) = std::env::var("SLO_P99_MS") {
        if let Ok(ms) = v.parse::<u64>() {
            cfg.slo_p99_ms = ms;
            tracing::info!("SLO_P99_MS override: {} ms", ms);
        }
    }

    let core = if let Some(dir) = args.model_dir.as_deref() {
        tracing::info!("XGB enabled, loading L1 model_dir={}", dir);
        Arc::new(AppCore::new_with_xgb_l1_l2(cfg, dir, args.model_l2_dir.as_deref())?)
    } else {
        tracing::info!("XGB disabled, baseline /score only");
        Arc::new(AppCore::new(cfg))
    };

    let state = AppState {
        core,
        prom,
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_in_flight: args.max_in_flight,
    };

    // ✅ 预热只做一次（避免每个 executor 重复跑）
    if args.model_dir.is_some() && args.warmup_iters > 0 {
        let t = std::time::Instant::now();
        tracing::info!("warmup_xgb start iters={}", args.warmup_iters);
        state.core.warmup_xgb(args.warmup_iters)?;
        tracing::info!("warmup_xgb done cost={}ms", t.elapsed().as_millis());
    }

    // ✅ thread-per-core：优先在“当前进程允许的 CPU 集合”里，扣掉 XGB pool pin 过的 CPU，剩下的给 Glommio IO executors。
    //    如果扣完为空，则退化为使用全部允许 CPU（runtime 与 XGB 可能共享核，但至少不会跑到 taskset 之外）。
    let io_cpus = select_glommio_io_cpus_from_env()?;
    let shards = io_cpus.len().max(1);
    let per_core_in_flight = std::cmp::max(1, (args.max_in_flight + shards - 1) / shards);

    tracing::info!(
        "glommio thread-per-core: io_shards={} io_cpus={:?} global_max_in_flight={} per_core_max_in_flight={}",
        shards,
        io_cpus,
        args.max_in_flight,
        per_core_in_flight
    );

    let addr = args.listen.clone();
    let mut handles = Vec::new();

    if io_cpus.is_empty() {
        // 极端兜底：无法解析 affinity / cpu 列表
        let st = state.clone();
        let addr2 = addr.clone();
        let per_core_in_flight2 = per_core_in_flight as u64;
        let h = LocalExecutorBuilder::new(Placement::Unbound)
            .name("risk-glommio-io")
            .spawn(move || async move {
                run_accept_loop(addr2, st, Rc::new(Semaphore::new(per_core_in_flight2)), true).await
            })
            .unwrap();
        handles.push(h);
    } else {
        for (idx, cpu) in io_cpus.into_iter().enumerate() {
            let st = state.clone();
            let addr2 = addr.clone();
            let per_core_in_flight2 = per_core_in_flight as u64;
            let name = format!("risk-glommio-io-cpu{}", cpu);
            let h = LocalExecutorBuilder::new(Placement::Fixed(cpu))
                .name(&name)
                .spawn(move || async move {
                    // glommio::net::TcpListener::bind() 会设置 SO_REUSEPORT，允许多 executor 同地址并行 accept。
                    run_accept_loop(addr2, st, Rc::new(Semaphore::new(per_core_in_flight2)), idx == 0).await
                })
                .unwrap();
            handles.push(h);
        }
    }

    // 阻塞主线程（任意一个 executor panic/退出都会让 join 返回，便于暴露问题）
    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}

async fn run_accept_loop(addr: String, state: AppState, core_sem: Rc<Semaphore>, log_listen: bool) {
    let listener = TcpListener::bind(addr.as_str()).expect("bind failed");
    if log_listen {
        tracing::info!("risk-server-glommio listening on http://{}", addr);
    }

    loop {
        match listener.accept().await {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);

                let st = state.clone();
                let sem = Rc::clone(&core_sem);
                glommio::spawn_local(async move {
                    if let Err(e) = handle_conn(stream, st, sem).await {
                        tracing::debug!("conn error: {}", e);
                    }
                })
                .detach();
            }
            Err(e) => {
                tracing::warn!("accept error: {}", e);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ReqMeta {
    method: String,
    path: String,
    content_len: Option<usize>,
    chunked: bool,
    expect_100: bool,
    want_close: bool,
}

async fn handle_conn(
    mut stream: glommio::net::TcpStream,
    st: AppState,
    core_sem: Rc<Semaphore>,
) -> anyhow::Result<()> {
    let mut stash: BytesMut = BytesMut::with_capacity(64 * 1024);
    let mut buf = vec![0u8; 64 * 1024];
    // 连接级复用：避免每个请求都重新分配响应 buffer
    let mut out: Vec<u8> = Vec::with_capacity(256);

    loop {
        // 1) 读到 header 结束
        while find_double_crlf(stash.as_ref()).is_none() {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                return Ok(());
            }
            stash.extend_from_slice(&buf[..n]);
            if stash.len() > 1024 * 1024 {
                anyhow::bail!("header too large");
            }
        }

        let header_end = find_double_crlf(stash.as_ref()).unwrap();

        // 2) 解析 header（用短作用域避免借用冲突）
        let meta = {
            let head = std::str::from_utf8(&stash[..header_end])?;
            parse_request_head(head)
        };

        // 3) 如果有 Expect: 100-continue，先回 100
        if meta.expect_100 {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            stream.flush().await?;
        }

        let body_start = header_end + 4;

        // 4) 读 body（content-length 或 chunked）
        let body: Bytes = if meta.chunked {
            // stash 里可能还没有完整 chunked body，继续读直到能解析
            loop {
                if let Some((end, body_vec)) = try_parse_chunked(stash.as_ref(), body_start) {
                    // drop consumed bytes (header + chunks)
                    let _ = stash.split_to(end);
                    break Bytes::from(body_vec);
                }
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    anyhow::bail!("eof while reading chunked body");
                }
                stash.extend_from_slice(&buf[..n]);
            }
        } else {
            let need = meta.content_len.unwrap_or(0);
            while stash.len() < body_start + need {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    anyhow::bail!("eof while reading body");
                }
                stash.extend_from_slice(&buf[..n]);
            }

            if need == 0 {
                // drop header only
                let _ = stash.split_to(body_start);
                Bytes::new()
            } else {
                // Split without copying: [header + body] is removed from stash, leaving keep-alive tail.
                let total = body_start + need;
                let mut head_body = stash.split_to(total);
                head_body.split_off(body_start).freeze()
            }
        };

        // 6) 路由
        let keep_alive = !meta.want_close;

        if meta.method == "GET" && meta.path == "/health" {
            write_http(&mut stream, &mut out, 200, "text/plain", b"ok", keep_alive).await?;

        } else if meta.method == "POST" && meta.path == "/score" {
            let req: ScoreRequest = serde_json::from_slice(body.as_ref())?;
            let resp = st.core.score(req);
            let body = serde_json::to_vec(&resp)?;
            write_http(&mut stream, &mut out, 200, "application/json", &body, keep_alive).await?;

        } else if meta.method == "POST" && (meta.path == "/score_xgb" || meta.path == "/score_xgb_pool") {
            // ✅ 统一走 pool（避免把 CPU-bound 推理塞进 glommio reactor）
            let t_parse = std::time::Instant::now();

            let v: Value = match serde_json::from_slice(body.as_ref()) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("invalid json body: {}", e);
                    write_http(&mut stream, &mut out, 400, "text/plain", msg.as_bytes(), keep_alive).await?;
                    continue;
                }
            };

            let parse_us: u64 = t_parse.elapsed().as_micros() as u64;

            let obj = match v {
                Value::Object(m) => {
                    if let Some(Value::Object(features)) = m.get("features") {
                        features.clone()
                    } else {
                        m
                    }
                }
                _ => {
                    write_http(&mut stream, &mut out, 400, "text/plain", b"expected json object", keep_alive).await?;
                    continue;
                }
            };

            // HTTP 层背压：超出并发上限就直接 429（避免服务端排队雪球）
            let (_g, _p) = match try_acquire_inflight(&st, core_sem.as_ref()) {
                Some(g) => g,
                None => {
                    write_http(&mut stream, &mut out, 429, "text/plain", b"overloaded", keep_alive).await?;
                    continue;
                }
            };

            match st.core.score_xgb_pool_async(parse_us, &obj).await {
                Ok(resp) => {
                    let body = serde_json::to_vec(&resp)?;
                    write_http(&mut stream, &mut out, 200, "application/json", &body, keep_alive).await?;
                }
                Err(e) => {
                    let (code, msg) = map_xgb_error(&e);
                    write_http(&mut stream, &mut out, code, "text/plain", msg.as_bytes(), keep_alive).await?;
                }
            }

        } else if meta.method == "POST" && meta.path == "/score_dense_f32_bin" {
            // ✅ bench2/bench3 主压测口：dense f32le -> RSK1 (48B)
            let t0 = std::time::Instant::now();

            // HTTP 层背压：超出并发上限就直接 429
            let (_g, _p) = match try_acquire_inflight(&st, core_sem.as_ref()) {
                Some(g) => g,
                None => {
                    write_http(&mut stream, &mut out, 429, "text/plain", b"overloaded", keep_alive).await?;
                    continue;
                }
            };

            // 必须启用 XGB + pool
            let xgb = match st.core.xgb.as_ref() {
                Some(x) => x,
                None => {
                    write_http(&mut stream, &mut out, 500, "text/plain", b"xgb not enabled", keep_alive).await?;
                    continue;
                }
            };
            let pool = match st.core.xgb_pool.as_ref() {
                Some(p) => p,
                None => {
                    write_http(&mut stream, &mut out, 500, "text/plain", b"xgb_pool not enabled", keep_alive).await?;
                    continue;
                }
            };

            let dim = xgb.feature_names.len();

            // parse payload (raw bytes or RVEC header)
            let t_parse = std::time::Instant::now();
            let payload = match parse_dense_payload_le(&body, dim) {
                Ok(p) => p,
                Err(msg) => {
                    write_http(&mut stream, &mut out, 400, "text/plain", msg.as_bytes(), keep_alive).await?;
                    continue;
                }
            };
            let parse_us = now_us(t_parse);

            // submit to XgbPool (bounded queue => fast fail)
            let t_xgb = std::time::Instant::now();
            let rx = match pool.try_submit_dense_bytes_le(payload, dim, 0) {
                Ok(rx) => rx,
                Err(e) => {
                    let (code, msg) = map_xgb_pool_err(&e);
                    write_http(&mut stream, &mut out, code, "text/plain", msg.as_bytes(), keep_alive).await?;
                    continue;
                }
            };

            // 与 tokio 版本的“端到端预算”语义对齐：超过 SLO 预算的请求直接 429（快速、可预测），避免 client timeout。
            let budget = std::time::Duration::from_millis(st.core.cfg.slo_p99_ms.max(1));
            let rx_res = match glommio::timer::timeout(budget, async { Ok(rx.await) }).await {
                Ok(v) => v,
                Err(_) => {
                    metrics::counter!("xgb_deadline_exceeded_total").increment(1);
                    write_http(&mut stream, &mut out, 429, "text/plain", b"xgb deadline exceeded", keep_alive).await?;
                    continue;
                }
            };

            let out1 = match rx_res {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    let msg = format!("xgb inference failed: {:#}", e);
                    write_http(&mut stream, &mut out, 500, "text/plain", msg.as_bytes(), keep_alive).await?;
                    continue;
                }
                Err(_) => {
                    write_http(&mut stream, &mut out, 503, "text/plain", b"xgb worker canceled", keep_alive).await?;
                    continue;
                }
            };

            let xgb_us = now_us(t_xgb);
            metrics::histogram!("stage_feature_us").record(0.0);
            metrics::histogram!("stage_xgb_us").record(xgb_us as f64);

            // router (decision mapping)
            let t_router = std::time::Instant::now();
            let decision_u8 = decision_str_to_u8(xgb.decide(out1.score));
            if decision_u8 == 2 && st.core.xgb_l2.is_some() {
                // 与 tokio 版本的 dense-bytes 路径保持一致：此端点不触发 L2，统计为“跳过”。
                metrics::counter!("router_l2_skipped_budget_total").increment(1);
            }
            let router_us = now_us(t_router);
            metrics::histogram!("stage_router_us").record(router_us as f64);

            // serialize (RSK1)
            let t_ser = std::time::Instant::now();
            let trace_id = TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
            // NOTE: 序列化耗时本身不能“精确地”写回同一个包（会有轻微递归依赖）。
            // 这里用两步：先用 serialize_us=0 编码，然后把测得的 serialize_us patch 回固定 offset(44..48)。
            let mut rsk1 = encode_rsk1(trace_id, out1.score, decision_u8, parse_us, 0, router_us, xgb_us, 0, 0);
            let ser_us = now_us(t_ser);
            patch_rsk1_serialize_us(&mut rsk1, ser_us);
            metrics::histogram!("stage_serialize_us").record(ser_us as f64);
            metrics::histogram!("e2e_us").record(now_us(t0) as f64);

            write_http(&mut stream, &mut out, 200, "application/octet-stream", &rsk1, keep_alive).await?;

        } else if meta.method == "GET" && meta.path == "/metrics" {
            let text = st.prom.render();
            write_http(
                &mut stream,
                &mut out,
                200,
                "text/plain; version=0.0.4",
                text.as_bytes(),
                keep_alive,
            )
            .await?;
        } else {
            write_http(&mut stream, &mut out, 404, "text/plain", b"not found", keep_alive).await?;
        }

        if meta.want_close {
            return Ok(());
        }
    }
}

/// RAII guard for global in-flight accounting.
struct InFlightGuard {
    ctr: Arc<AtomicUsize>,
}
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.ctr.fetch_sub(1, Ordering::Relaxed);
    }
}

fn try_acquire_inflight<'a>(st: &AppState, core_sem: &'a Semaphore) -> Option<(InFlightGuard, Permit<'a>)> {
    // 先拿全局 token，再拿本核 token；本核拿不到就回滚全局。
    let cur = st.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
    if st.max_in_flight > 0 && cur > st.max_in_flight {
        st.in_flight.fetch_sub(1, Ordering::Relaxed);
        return None;
    }

    let g = InFlightGuard {
        ctr: Arc::clone(&st.in_flight),
    };

    match core_sem.try_acquire_permit(1) {
        Ok(p) => Some((g, p)),
        Err(_) => {
            // 回滚全局
            drop(g);
            None
        }
    }
}

/// 从环境变量 / 进程 affinity 里推导 Glommio IO executors 应该绑定的 CPU 列表。
///
/// 规则：
/// 1) 先读当前进程允许的 CPU（比如通过 taskset/cpuset 限制）。
/// 2) 再读 XGB pool 的 pin 列表（XGB_L1_POOL_PIN_CPUS / XGB_L2_POOL_PIN_CPUS / XGB_POOL_PIN_CPUS）。
/// 3) IO_Cpus = Allowed - XGB_Pinned；如果为空，就退化为 Allowed（避免跑到 taskset 之外）。
fn select_glommio_io_cpus_from_env() -> anyhow::Result<Vec<usize>> {
    let allowed = get_self_affinity_cpus()?;
    if allowed.is_empty() {
        return Ok(Vec::new());
    }

    let mut pinned: HashSet<usize> = HashSet::new();
    pinned.extend(parse_cpu_env("XGB_L1_POOL_PIN_CPUS"));
    pinned.extend(parse_cpu_env("XGB_L2_POOL_PIN_CPUS"));
    pinned.extend(parse_cpu_env("XGB_POOL_PIN_CPUS"));

    let io: Vec<usize> = allowed
        .iter()
        .copied()
        .filter(|cpu| !pinned.contains(cpu))
        .collect();

    if io.is_empty() {
        Ok(allowed)
    } else {
        Ok(io)
    }
}

fn parse_cpu_env(var: &str) -> HashSet<usize> {
    match std::env::var(var) {
        Ok(s) => parse_cpu_csv(&s),
        Err(_) => HashSet::new(),
    }
}

fn parse_cpu_csv(s: &str) -> HashSet<usize> {
    let mut out = HashSet::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Ok(v) = p.parse::<usize>() {
            out.insert(v);
        }
    }
    out
}

fn get_self_affinity_cpus() -> anyhow::Result<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let rc = libc::sched_getaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &mut set,
            );
            if rc != 0 {
                return Err(anyhow::anyhow!(std::io::Error::last_os_error()));
            }

            let mut cpus = Vec::new();
            for cpu in 0..(libc::CPU_SETSIZE as usize) {
                if libc::CPU_ISSET(cpu, &set) {
                    cpus.push(cpu);
                }
            }
            cpus.sort_unstable();
            Ok(cpus)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Ok((0..n).collect())
    }
}

fn map_xgb_error(e: &anyhow::Error) -> (u16, String) {
    if let Some(pe) = e.downcast_ref::<XgbPoolError>() {
        let (code, msg) = map_xgb_pool_err(pe);
        return (code, msg);
    }
    (500, format!("internal error: {:#}", e))
}

fn map_xgb_pool_err(e: &XgbPoolError) -> (u16, String) {
    match e {
        XgbPoolError::QueueFull => (429, "xgb pool queue full".to_string()),
        XgbPoolError::DeadlineExceeded => (429, "xgb pool deadline exceeded".to_string()),
        XgbPoolError::WorkerDown => (503, "xgb pool worker down".to_string()),
        XgbPoolError::Canceled => (503, "xgb pool canceled".to_string()),
    }
}

/// Parse dense payload.
/// - raw: exactly dim*4 bytes
/// - RVEC: magic="RVEC"(4) + ver(u16=1) + flags(u16=0) + dim(u32) + reserved(u32) + payload(f32*dim)
fn parse_dense_payload_le(body: &Bytes, expected_dim: usize) -> Result<Bytes, String> {
    let expected_len = expected_dim
        .checked_mul(4)
        .ok_or_else(|| "expected_dim too large".to_string())?;

    // raw fast path
    if body.len() == expected_len {
        return Ok(body.clone());
    }

    // RVEC header
    if body.len() != 16 + expected_len {
        return Err(format!(
            "invalid dense payload size: got {}, expected {} (raw) or {} (RVEC)",
            body.len(),
            expected_len,
            16 + expected_len
        ));
    }
    let b = body.as_ref();
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
        return Err(format!("RVEC dim mismatch: got {}, expected {}", dim, expected_dim));
    }
    Ok(body.slice(16..))
}

#[inline]
fn decision_str_to_u8(s: &str) -> u8 {
    match s {
        "allow" => 0,
        "deny" => 1,
        "review" | "manual_review" => 2,
        "degrade_allow" | "degraded_allow" => 3,
        _ => 2,
    }
}

/// Encode 48-byte RSK1 response.
/// Layout (compatible with risk-bench2 decoder):
/// - magic 'RSK1'
/// - ver(u16)=1
/// - flags(u16)
/// - trace_id(u64)
/// - score(f32)
/// - decision(u8) + pad[3]
/// - timings[6] u32: parse, feature, router, xgb, l2, serialize
fn encode_rsk1(
    trace_id: u64,
    score: f32,
    decision: u8,
    parse_us: u64,
    feature_us: u64,
    router_us: u64,
    xgb_us: u64,
    l2_us: u64,
    serialize_us: u64,
) -> Vec<u8> {
    let mut flags: u16 = 0;
    if l2_us > 0 {
        flags |= 1;
    }
    if decision == 3 {
        flags |= 2;
    }

    #[inline]
    fn clamp_u32(x: u64) -> u32 {
        (x.min(u32::MAX as u64)) as u32
    }

    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"RSK1");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&score.to_le_bytes());
    out.push(decision);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&clamp_u32(parse_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(feature_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(router_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(xgb_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(l2_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(serialize_us).to_le_bytes());

    debug_assert_eq!(out.len(), 48);
    out
}

#[inline]
fn patch_rsk1_serialize_us(buf: &mut [u8], serialize_us: u64) {
    if buf.len() < 48 {
        return;
    }
    let v = (serialize_us.min(u32::MAX as u64) as u32).to_le_bytes();
    // timings[5] offset: 24 + 5*4 = 44
    buf[44..48].copy_from_slice(&v);
}

fn parse_request_head(head: &str) -> ReqMeta {
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut it = first.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("").to_string();

    let mut content_len: Option<usize> = None;
    let mut chunked = false;
    let mut expect_100 = false;
    let mut want_close = false;

    for line in lines {
        let line = line.trim();

        if let Some(v) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_len = v.trim().parse::<usize>().ok();
        }

        if let Some(v) = line
            .strip_prefix("Transfer-Encoding:")
            .or_else(|| line.strip_prefix("transfer-encoding:"))
        {
            if v.split(',').any(|t| t.trim().eq_ignore_ascii_case("chunked")) {
                chunked = true;
            }
        }

        if line.eq_ignore_ascii_case("Expect: 100-continue") {
            expect_100 = true;
        }

        if line.eq_ignore_ascii_case("Connection: close") {
            want_close = true;
        }
    }

    ReqMeta {
        method,
        path,
        content_len,
        chunked,
        expect_100,
        want_close,
    }
}

fn find_double_crlf(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n")
}

async fn write_http(
    stream: &mut glommio::net::TcpStream,
    scratch: &mut Vec<u8>,
    code: u16,
    ctype: &str,
    body: &[u8],
    keep_alive: bool,
) -> anyhow::Result<()> {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        500 => "Internal Server Error",
        _ => "OK",
    };

    let conn = if keep_alive { "keep-alive" } else { "close" };

    // 关键：复用 scratch，合并 header+body，一次性写出，避免 Nagle/delayed-ack 造成 ~40ms 抖动 + 避免每请求分配
    scratch.clear();
    // Vec<u8> 实现了 std::io::Write，所以 write! 不会额外分配 String。
    write!(
        scratch,
        "HTTP/1.1 {} {}
Content-Type: {}
Content-Length: {}
Connection: {}

",
        code,
        status,
        ctype,
        body.len(),
        conn
    )?;
    scratch.extend_from_slice(body);

    stream.write_all(scratch).await?;
    Ok(())
}

fn find_crlf(b: &[u8]) -> Option<usize> {
    b.windows(2).position(|w| w == b"\r\n")
}

/// 尝试从 stash[start..] 解析 chunked body。
/// 成功则返回 (consumed_end_index, body_bytes)。
fn try_parse_chunked(stash: &[u8], start: usize) -> Option<(usize, Vec<u8>)> {
    let mut i = start;
    let mut out = Vec::new();

    loop {
        // chunk size line
        let rel = find_crlf(&stash.get(i..)?)?;
        let line_end = i + rel;
        let line = std::str::from_utf8(&stash[i..line_end]).ok()?.trim();
        let size = usize::from_str_radix(line, 16).ok()?;
        let mut j = line_end + 2; // skip \r\n

        if size == 0 {
            // 终止块：期待 "\r\n"
            if stash.len() < j + 2 {
                return None;
            }
            if &stash[j..j + 2] != b"\r\n" {
                return None;
            }
            j += 2;
            return Some((j, out));
        }

        // need chunk data + trailing \r\n
        if stash.len() < j + size + 2 {
            return None;
        }
        out.extend_from_slice(&stash[j..j + size]);
        j += size;

        if &stash[j..j + 2] != b"\r\n" {
            return None;
        }
        i = j + 2;
    }
}