#!/usr/bin/env bash
set -euo pipefail

# ========= 可改参数（也支持用环境变量覆盖） =========
LISTEN="${LISTEN:-127.0.0.1:8080}"
URL="${URL:-http://127.0.0.1:8080/score_xgb_pool}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8080/metrics}"

MODEL_L1_DIR="${MODEL_L1_DIR:-models/ieee_l1}"
MODEL_L2_DIR="${MODEL_L2_DIR:-models/ieee_l2}"

# Poisson / open-loop 参数（你的 bench 已经支持）
DURATION="${DURATION:-20}"
CONCURRENCY="${CONCURRENCY:-256}"
PACER_SHARDS="${PACER_SHARDS:-512}"
PACER_JITTER_US="${PACER_JITTER_US:-0}"

# sweep 区间
RPS_START="${RPS_START:-8000}"
RPS_END="${RPS_END:-14000}"
RPS_STEP="${RPS_STEP:-500}"

# ======== 绑核（按你机器调整）========
# 典型三分区：L1快核 / L2稳核 / Tokio IO核，bench 再独立一组
CPU_SERVER_TASKSET="${CPU_SERVER_TASKSET:-2-25}"
CPU_BENCH_TASKSET="${CPU_BENCH_TASKSET:-26-31}"

# L1/L2 pool 各自绑核（逗号分隔）
XGB_L1_POOL_PIN_CPUS="${XGB_L1_POOL_PIN_CPUS:-2,4,6,8,10,12,14,16}"
XGB_L2_POOL_PIN_CPUS="${XGB_L2_POOL_PIN_CPUS:-19,21,23,25}"

# pool 配置（建议先这样起步）
XGB_L1_POOL_THREADS="${XGB_L1_POOL_THREADS:-8}"
XGB_L1_POOL_QUEUE_CAP="${XGB_L1_POOL_QUEUE_CAP:-1024}"

XGB_L2_POOL_THREADS="${XGB_L2_POOL_THREADS:-4}"
XGB_L2_POOL_QUEUE_CAP="${XGB_L2_POOL_QUEUE_CAP:-256}"

# Tokio 配置（你的 main 里 worker_threads=4，这里环境变量主要压住 blocking 池 & allocator）
TOKIO_WORKER_THREADS="${TOKIO_WORKER_THREADS:-4}"
TOKIO_MAX_BLOCKING_THREADS="${TOKIO_MAX_BLOCKING_THREADS:-4}"

# WSL2/多线程 allocator 止血阀
MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 输出目录
OUT_DIR="${OUT_DIR:-results/poisson_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

SERVER_LOG="${OUT_DIR}/server.log"
BENCH_LOG="${OUT_DIR}/bench.log"
CSV_OUT="${OUT_DIR}/sweep.csv"

echo "[OUT] ${OUT_DIR}"
echo "[CFG] URL=${URL}"
echo "[CFG] METRICS_URL=${METRICS_URL}"
echo "[CFG] RPS ${RPS_START}..${RPS_END} step ${RPS_STEP}"
echo "[CFG] duration=${DURATION}s concurrency=${CONCURRENCY} shards=${PACER_SHARDS} jitter_us=${PACER_JITTER_US}"
echo

cleanup() {
  echo "[CLEANUP] stopping server..."
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    sleep 1
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ========= 1) 编译（只编一次） =========
echo "[BUILD] cargo build --release (server+bench)"
cargo build -p risk-server-tokio -p risk-bench --release

# ========= 2) 启动 server（后台） =========
echo "[SERVER] starting..."
(
  export RUST_LOG="${RUST_LOG:-info}"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX}"
  export TOKIO_WORKER_THREADS="${TOKIO_WORKER_THREADS}"
  export TOKIO_MAX_BLOCKING_THREADS="${TOKIO_MAX_BLOCKING_THREADS}"

  export XGB_L1_POOL_THREADS="${XGB_L1_POOL_THREADS}"
  export XGB_L1_POOL_QUEUE_CAP="${XGB_L1_POOL_QUEUE_CAP}"
  export XGB_L1_POOL_PIN_CPUS="${XGB_L1_POOL_PIN_CPUS}"

  export XGB_L2_POOL_THREADS="${XGB_L2_POOL_THREADS}"
  export XGB_L2_POOL_QUEUE_CAP="${XGB_L2_POOL_QUEUE_CAP}"
  export XGB_L2_POOL_PIN_CPUS="${XGB_L2_POOL_PIN_CPUS}"

  # 注意：这里使用 target/release 直接跑，比 cargo run 更干净更稳定
  taskset -c "${CPU_SERVER_TASKSET}" \
    target/release/risk-server-tokio \
      --listen "${LISTEN}" \
      --model-dir "${MODEL_L1_DIR}" \
      --model-l2-dir "${MODEL_L2_DIR}" \
    2>&1 | tee "${SERVER_LOG}"
) &
SERVER_PID=$!
echo "[SERVER] pid=${SERVER_PID}"

# ========= 3) 等待 server ready =========
echo "[WAIT] waiting for /metrics..."
for i in {1..60}; do
  if curl -sf "${METRICS_URL}" >/dev/null; then
    echo "[WAIT] server ready"
    break
  fi
  sleep 0.5
done

# ========= 4) sweep =========
echo "[SWEEP] writing CSV -> ${CSV_OUT}"
echo "ts_ms,runtime,url,metrics_url,rps,duration_s,concurrency,pacer_shards,pacer_jitter_us,ok,err,dropped,p50_us,p95_us,p99_us,parse_p99_us,feature_p99_us,router_p99_us,xgb_p99_us,l2_p99_us,serialize_p99_us,http_2xx,http_429,http_5xx,timeout,attempted_rps,ok_rps,drop_rps,drop_pct,bench_lag_p99_us,bench_lag_max_us,missed_ticks_total,xgb_deadline_miss_total,xgb_compute_p99_est_us,xgb_queue_wait_p99_us,xgb_compute_p99_us" > "${CSV_OUT}"

for ((rps=RPS_START; rps<=RPS_END; rps+=RPS_STEP)); do
  echo
  echo "==================== RPS=${rps} ===================="

  # bench 单次运行，追加到总日志 & 输出 CSV 一行（你的 bench 已经支持 CSV 输出）
  taskset -c "${CPU_BENCH_TASKSET}" \
    target/release/risk-bench \
      --url "${URL}" \
      --metrics-url "${METRICS_URL}" \
      --rps "${rps}" \
      --duration "${DURATION}" \
      --concurrency "${CONCURRENCY}" \
      --pacer-shards "${PACER_SHARDS}" \
      --pacer-jitter-us "${PACER_JITTER_US}" \
      --csv-append "${CSV_OUT}" \
    2>&1 | tee -a "${BENCH_LOG}"

  # 小休息：让系统“回到稳态”
  sleep 1
done

echo
echo "[DONE] sweep finished."
echo "[OUT] CSV: ${CSV_OUT}"
echo "[OUT] server log: ${SERVER_LOG}"
echo "[OUT] bench  log: ${BENCH_LOG}"
