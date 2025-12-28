use futures_lite::{AsyncReadExt, AsyncWriteExt};
use glommio::net::TcpListener;
use glommio::{LocalExecutorBuilder, Placement};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{config::Config, pipeline::AppCore, schema::ScoreRequest};

use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
}

fn main() {
    // tracing
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
    let core = Arc::new(AppCore::new(cfg));
    let state = AppState { core, prom };

    LocalExecutorBuilder::new(Placement::Unbound)
        .name("risk-glommio")
        .spawn(move || async move {
            let addr = "127.0.0.1:8081";
            let listener = TcpListener::bind(addr).expect("bind failed");
            tracing::info!("risk-server-glommio listening on http://{}", addr);

            loop {
                let stream = listener.accept().await;
                match stream {
                    Ok(stream) => {
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
}

async fn handle_conn(mut stream: glommio::net::TcpStream, st: AppState) -> anyhow::Result<()> {
    let mut stash: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut buf = vec![0u8; 64 * 1024];

    loop {
        // 1) 读到 header 结束（\r\n\r\n）
        while find_double_crlf(&stash).is_none() {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                return Ok(()); // client closed
            }
            stash.extend_from_slice(&buf[..n]);
            if stash.len() > 1024 * 1024 {
                anyhow::bail!("request header too large");
            }
        }

        let header_end = find_double_crlf(&stash).unwrap();
        let (method, path, content_len, want_close) = {
            let head = std::str::from_utf8(&stash[..header_end])?;
            parse_request_head(head)
        };

        // 2) 读取 body（若有）
        let body_start = header_end + 4;
        let need_body = content_len.unwrap_or(0);

        while stash.len() < body_start + need_body {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                anyhow::bail!("unexpected eof while reading body");
            }
            stash.extend_from_slice(&buf[..n]);
        }

        let body = if need_body > 0 {
            stash[body_start..body_start + need_body].to_vec()
        } else {
            Vec::new()
        };

        // 3) 丢掉已消费的 bytes，剩余留给下一轮（这是 keep-alive 必需）
        stash.drain(..body_start + need_body);

        // 4) 路由处理
        if method == "POST" && path == "/score" {
            let req: ScoreRequest = serde_json::from_slice(&body)?;
            let resp = st.core.score(req);
            let out = serde_json::to_vec(&resp)?;
            write_http(&mut stream, 200, "application/json", &out).await?;
        } else if method == "GET" && path == "/metrics" {
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

        // 5) 如果客户端要求 close，就退出
        if want_close {
            return Ok(());
        }
    }
}

fn parse_request_head(head: &str) -> (String, String, Option<usize>, bool) {
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut it = first.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("").to_string();

    let mut content_len: Option<usize> = None;
    let mut want_close = false;

    for line in lines {
        let line = line.trim();
        if let Some(v) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_len = v.trim().parse::<usize>().ok();
        }
        if line.eq_ignore_ascii_case("Connection: close") {
            want_close = true;
        }
    }

    (method, path, content_len, want_close)
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
        404 => "Not Found",
        _ => "OK",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: keep-alive
        Keep-Alive: timeout=5
\r\n\r\n",
        code,
        status,
        ctype,
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}
