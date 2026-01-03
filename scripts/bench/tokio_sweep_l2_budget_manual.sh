#!/usr/bin/env bash
set -euo pipefail

# 手动工作流：你自己起 server（每个点可选择重启），脚本负责按顺序跑 bench 并把结果落到一个 CSV。
#
# 用法：
#   ./scripts/bench/tokio_sweep_l2_budget_manual.sh 14000 "0 1 5 10 20"
#
# 你可以通过环境变量覆盖：
#   URL, METRICS_URL, DURATION, CONCURRENCY, PACER_SHARDS, PACER_JITTER_US, CPU_BENCH_TASKSET
#   XGB_BODY_FILE (推荐 jsonl), XGB_BODY_FORMAT (auto|json|jsonl|dir)
#   ROUTER_L2_MAX_QUEUE_WATERLINE, ROUTER_L2_QUEUE_WAIT_BUDGET_US, ROUTER_L2_MIN_REMAINING_US
#
# 说明：
# - server 的 L2 预算（ROUTER_L2_MAX_TRIGGERS_PER_SEC）必须在 server 进程环境里设置；
#   如果你不重启 server，则预算不会变。但 bench CSV 会记录每次跑的 label（便于对齐）。

RPS="${1:-14000}"
PCTS_STR="${2:-0 1 5 10 20}"

URL="${URL:-http://127.0.0.1:8080/score_xgb_pool}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8080/metrics}"

DURATION="${DURATION:-20}"
CONCURRENCY="${CONCURRENCY:-256}"
PACER_SHARDS="${PACER_SHARDS:-512}"
PACER_JITTER_US="${PACER_JITTER_US:-0}"
CPU_BENCH_TASKSET="${CPU_BENCH_TASKSET:-26-31}"

XGB_BODY_FILE="${XGB_BODY_FILE:-}"
XGB_BODY_FORMAT="${XGB_BODY_FORMAT:-auto}"

# 给 L2 的"安全阀"默认值：你也可以在启动 server 时自己设。
ROUTER_L2_MAX_QUEUE_WATERLINE="${ROUTER_L2_MAX_QUEUE_WATERLINE:-0.85}"
ROUTER_L2_QUEUE_WAIT_BUDGET_US="${ROUTER_L2_QUEUE_WAIT_BUDGET_US:-2000}"
ROUTER_L2_MIN_REMAINING_US="${ROUTER_L2_MIN_REMAINING_US:-1000}"

OUT_DIR="${OUT_DIR:-results/l2_budget_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"
CSV_OUT="${OUT_DIR}/sweep_l2_budget.csv"

# build once (bench)
echo "[BUILD] cargo build -p risk-bench --release"
cargo build -p risk-bench --release

# optional body args
BODY_ARGS=()
if [[ -n "${XGB_BODY_FILE}" ]]; then
  BODY_ARGS+=(--xgb-body-file "${XGB_BODY_FILE}")
  BODY_ARGS+=(--xgb-body-format "${XGB_BODY_FORMAT}")
fi

IFS=' ' read -r -a PCTS <<< "${PCTS_STR}"

echo "[OUT] ${CSV_OUT}"
echo "[CFG] fixed_rps=${RPS} duration=${DURATION}s concurrency=${CONCURRENCY} shards=${PACER_SHARDS} jitter_us=${PACER_JITTER_US}"
echo "[CFG] url=${URL} metrics=${METRICS_URL}"
if [[ -n "${XGB_BODY_FILE}" ]]; then
  echo "[CFG] xgb_body_file=${XGB_BODY_FILE} (${XGB_BODY_FORMAT})"
fi

echo
for pct in "${PCTS[@]}"; do
  # integer budget per sec
  budget=$(( RPS * pct / 100 ))

  echo "============================================================"
  echo "[POINT] L2 pct=${pct}% => budget_per_sec=${budget} (at rps=${RPS})"
  echo
  echo "你需要让 server 以如下 ENV 运行（改完记得重启 server 才生效）："
  echo "  export ROUTER_L2_MAX_TRIGGERS_PER_SEC=${budget}"
  echo "  export ROUTER_L2_MAX_QUEUE_WATERLINE=${ROUTER_L2_MAX_QUEUE_WATERLINE}"
  echo "  export ROUTER_L2_QUEUE_WAIT_BUDGET_US=${ROUTER_L2_QUEUE_WAIT_BUDGET_US}"
  echo "  export ROUTER_L2_MIN_REMAINING_US=${ROUTER_L2_MIN_REMAINING_US}"
  echo
  echo "准备好后按 Enter 跑本点 bench（会 append 到同一个 CSV）..."
  read -r _

  label="l2_pct=${pct};l2_budget_per_sec=${budget}"

  taskset -c "${CPU_BENCH_TASKSET}" \
    target/release/risk-bench \
      --label "${label}" \
      --url "${URL}" \
      --metrics-url "${METRICS_URL}" \
      --rps "${RPS}" \
      --duration "${DURATION}" \
      --concurrency "${CONCURRENCY}" \
      --pacer-shards "${PACER_SHARDS}" \
      --pacer-jitter-us "${PACER_JITTER_US}" \
      --out "${CSV_OUT}" \
      "${BODY_ARGS[@]}"

done

echo
echo "[DONE] results saved to: ${CSV_OUT}"
