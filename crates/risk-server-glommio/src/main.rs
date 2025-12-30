use clap::Parser;
use futures_lite::{AsyncReadExt, AsyncWriteExt};
use glommio::net::TcpListener;
use glommio::{LocalExecutorBuilder, Placement};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{config::Config, pipeline::AppCore, schema::ScoreRequest};

use serde_json::{Map, Value};

use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8081")]
    listen: String,

    /// 可选：启用 XGB 在线推理（指向模型目录）
    #[arg(long)]
    model_dir: Option<String>,

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
    let cfg = Config::default();

    let core = if let Some(dir) = args.model_dir.as_deref() {
        tracing::info!("XGB enabled, loading model_dir={}", dir);
        Arc::new(AppCore::new_with_xgb(cfg, dir)?)
    } else {
        tracing::info!("XGB disabled, baseline /score only");
        Arc::new(AppCore::new(cfg))
    };

    let state = AppState { core, prom };

    LocalExecutorBuilder::new(Placement::Unbound)
        .name("risk-glommio")
        .spawn(move || async move {
            let addr = args.listen.as_str();
            if args.model_dir.is_some() && args.warmup_iters > 0 {
                let t = std::time::Instant::now();
                tracing::info!("warmup_xgb start iters={}", args.warmup_iters);
                match state.core.warmup_xgb(args.warmup_iters) {
                    Ok(()) => {
                        tracing::info!("warmup_xgb done cost={}ms", t.elapsed().as_millis());
                    }
                    Err(e) => {
                        tracing::error!("warmup_xgb failed: {:#}", e);
                        return;
                    }
                }
            }

            let listener = TcpListener::bind(addr).expect("bind failed");
            tracing::info!("risk-server-glommio listening on http://{}", addr);

            loop {
                match listener.accept().await {
                    Ok(stream) => {
                        // 这里 stream 才是 TcpStream
                        let _ = stream.set_nodelay(true);

                        let st = state.clone();
                        glommio::spawn_local(async move {
                            if let Err(e) = handle_conn(stream, st).await {
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
        })
        .unwrap()
        .join()
        .unwrap();

    Ok(())
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

async fn handle_conn(mut stream: glommio::net::TcpStream, st: AppState) -> anyhow::Result<()> {
    let mut stash: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut buf = vec![0u8; 64 * 1024];

    loop {
        // 1) 读到 header 结束
        while find_double_crlf(&stash).is_none() {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                return Ok(());
            }
            stash.extend_from_slice(&buf[..n]);
            if stash.len() > 1024 * 1024 {
                anyhow::bail!("header too large");
            }
        }

        let header_end = find_double_crlf(&stash).unwrap();

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
        let (consumed_end, body) = if meta.chunked {
            // stash 里可能还没有完整 chunked body，继续读直到能解析
            loop {
                if let Some((end, body)) = try_parse_chunked(&stash, body_start) {
                    break (end, body);
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
            let body = if need > 0 {
                stash[body_start..body_start + need].to_vec()
            } else {
                Vec::new()
            };
            (body_start + need, body)
        };

        // 5) 丢掉已消费字节，剩余留给下一轮 keep-alive
        stash.drain(..consumed_end);

        // 6) 路由
        if meta.method == "POST" && meta.path == "/score" {
            let req: ScoreRequest = serde_json::from_slice(&body)?;
            let resp = st.core.score(req);
            let out = serde_json::to_vec(&resp)?;
            write_http(&mut stream, 200, "application/json", &out).await?;
        } else if meta.method == "POST" && meta.path == "/score_xgb" {
            let t_parse = std::time::Instant::now();

            let v: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("invalid json body: {}", e);
                    write_http(&mut stream, 400, "text/plain", msg.as_bytes()).await?;
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
                    write_http(&mut stream, 400, "text/plain", b"expected json object").await?;
                    continue;
                }
            };

            match st.core.score_xgb(parse_us, &obj) {
                Ok(resp) => {
                    let out = serde_json::to_vec(&resp)?;
                    write_http(&mut stream, 200, "application/json", &out).await?;
                }
                Err(e) => {
                    let msg = format!("score_xgb failed: {:#}", e);
                    write_http(&mut stream, 500, "text/plain", msg.as_bytes()).await?;
                }
            }
        } else if meta.method == "GET" && meta.path == "/metrics" {
            let text = st.prom.render();
            write_http(
                &mut stream,
                200,
                "text/plain; version=0.0.4",
                text.as_bytes(),
            )
            .await?;
        } else {
            write_http(&mut stream, 404, "text/plain", b"not found").await?;
        }

        if meta.want_close {
            return Ok(());
        }
    }
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

        if line.to_ascii_lowercase().starts_with("transfer-encoding:")
            && line.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
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
    code: u16,
    ctype: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        code,
        status,
        ctype,
        body.len()
    );
    // 关键：合并成一个 buffer，一次性写出，避免 Nagle/delayed-ack 造成 ~40ms 抖动
    let mut out = Vec::with_capacity(header.len() + body.len());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(body);

    stream.write_all(&out).await?;
    // 一般不需要 flush；如果你想保险留着也行
    stream.flush().await?;
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
