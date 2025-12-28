# Monster Risk (Tokio starter)

这是一个“先跑通最小闭环、再逐步变 monster”的风控实时评分系统原型骨架（Tokio 版）。
目标：**端到端 SLO p99 <= 10ms**（先跑通，再逐步优化；之后会加 Glommio 版做 p99 对比）。

## Quickstart

```bash
# 1) 启动服务（默认 127.0.0.1:8080）
cargo run -p risk-server-tokio

# 2) 压测（固定速率，默认 1000 rps，20s）
cargo run -p risk-bench -- --rps 1000 --duration 20
```

- 评分接口：POST `http://127.0.0.1:8080/score`
- 指标接口：GET  `http://127.0.0.1:8080/metrics`

## Example request

```json
{
  "trace_id": null,
  "event_time_ms": 1735344000123,
  "user_id": "u_42",
  "card_id": "c_42",
  "merchant_id": "m_7",
  "mcc": 5411,
  "amount": 128.50,
  "currency": "JPY",
  "country": "JP",
  "channel": "ECOM",
  "device_id": "d_1",
  "ip_prefix": "203.0.113",
  "is_3ds": true
}
```

