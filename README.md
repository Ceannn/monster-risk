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


## Paper experiments (L2 budget sweep)

### Bench: add a label column

`risk-bench` supports `--label` which will be written into the CSV (useful for sweeps).

### Manual sweep helper

If you prefer手动起服务 + 手动改 env，这个脚本会按顺序跑 bench，并把每次 run 以 label 写进同一个 CSV：

```bash
./scripts/bench/tokio_sweep_l2_budget_manual.sh 14000 "0 1 5 10 20"
# 输出：results/l2_budget_sweep_*/sweep_l2_budget.csv
# 可选：python3 scripts/plot/plot_l2_budget_sweep.py results/l2_budget_sweep_*/sweep_l2_budget.csv
```

The sweep is designed around the idea: keep RPS fixed, change L2 trigger budget, and observe the phase transition (ok_rps/p99/429 + queue_wait).


This patch implements XgbPool option (B): per-core sharded queues + lock-free work-stealing.

Files:
- crates/risk-core/src/xgb_pool.rs  (replaces the old tokio::mpsc-based implementation)
- crates/risk-core/Cargo.toml      (adds crossbeam-deque + crossbeam-utils deps)

How to apply:
1) Copy the two files into your repo at the same paths (overwrite).
2) `cargo build -p risk-server-glommio --release ...` as usual.

Notes:
- Public API of XgbPool stays the same.
- Admission control still uses:
  - XGB_L1_POOL_QUEUE_CAP  -> total queue capacity (rounded up by per-worker split)
  - XGB_L1_POOL_EARLY_REJECT_US -> optional predicted-queue-wait budget
