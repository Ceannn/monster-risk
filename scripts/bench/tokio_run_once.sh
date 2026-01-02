#!/usr/bin/env bash
set -euo pipefail

RPS="${1:-12000}"
DURATION="${DURATION:-20}"
CONCURRENCY="${CONCURRENCY:-256}"
PACER_SHARDS="${PACER_SHARDS:-512}"
PACER_JITTER_US="${PACER_JITTER_US:-0}"

URL="${URL:-http://127.0.0.1:8080/score_xgb_pool}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8080/metrics}"
CPU_BENCH_TASKSET="${CPU_BENCH_TASKSET:-26-31}"

taskset -c "${CPU_BENCH_TASKSET}" \
  target/release/risk-bench \
    --url "${URL}" \
    --metrics-url "${METRICS_URL}" \
    --rps "${RPS}" \
    --duration "${DURATION}" \
    --concurrency "${CONCURRENCY}" \
    --pacer-shards "${PACER_SHARDS}" \
    --pacer-jitter-us "${PACER_JITTER_US}"
