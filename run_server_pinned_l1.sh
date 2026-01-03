#!/usr/bin/env bash
set -euo pipefail

# ===== 基础环境（WSL2 推荐）=====
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

# tokio 线程数 = 你要 pin 的 tokio 核数
export TOKIO_WORKER_THREADS=4
export TOKIO_MAX_BLOCKING_THREADS=4

# ===== L1 XGB pool：只用偶数 CPU，避免 SMT 同核争用 =====
export XGB_L1_POOL_THREADS=8
export XGB_L1_POOL_QUEUE_CAP=512
export XGB_L1_POOL_PIN_CPUS="2,4,6,8,10,12,14,16"

# 先做 L1-only baseline：别开 L2（不要传 --model-l2-dir）
unset XGB_L2_POOL_THREADS || true
unset XGB_L2_POOL_QUEUE_CAP || true
unset XGB_L2_POOL_PIN_CPUS || true

# ===== 启动 server：进程允许的 CPU 集合（包含 XGB+Tokio+垃圾核）=====
taskset -c 2-25 \
target/release/risk-server-tokio \
  --model-dir models/ieee_l1 &

pid=$!
echo "server pid=$pid"

# 等 tokio 线程起来（避免“pin 早了，后面又新起线程没 pin”的坑）
for _ in $(seq 1 50); do
  n=$(ps -T -p "$pid" -o tid,comm | awk '/tokio/ && /runtime/ {c++} END{print c+0}')
  if [ "$n" -ge 4 ]; then break; fi
  sleep 0.05
done

# ===== pin tokio worker 到 18,20,22,24 =====
TOKIO_CPUS=(18 20 22 24)
i=0
for tid in $(ps -T -p "$pid" -o tid,comm | awk '/tokio/ && /runtime/ {print $1}'); do
  cpu=${TOKIO_CPUS[$i]}
  taskset -cp "$cpu" "$tid" >/dev/null
  i=$(( (i+1) % ${#TOKIO_CPUS[@]} ))
done

# 主线程也 pin 到 tokio 核（避免抢 XGB）
taskset -cp 18 "$pid" >/dev/null

# ===== 把 cuda/杂线程踢到垃圾核 25 =====
for tid in $(ps -T -p "$pid" -o tid,comm | awk '/cuda/ {print $1}'); do
  taskset -cp 25 "$tid" >/dev/null
done

# 再补 pin 一次，防止 tokio 后续又起线程（WSL2 下很常见）
sleep 0.2
i=0
for tid in $(ps -T -p "$pid" -o tid,comm | awk '/tokio/ && /runtime/ {print $1}'); do
  cpu=${TOKIO_CPUS[$i]}
  taskset -cp "$cpu" "$tid" >/dev/null
  i=$(( (i+1) % ${#TOKIO_CPUS[@]} ))
done

echo "verify (psr=current cpu):"
ps -T -p "$pid" -o tid,psr,comm | egrep 'risk-server|tokio|xgb-worker|cuda' | sort -k2n

echo ""
echo "verify (affinity mask):"
for tid in $(ps -T -p "$pid" -o tid,comm | egrep 'risk-server|tokio|xgb-worker|cuda' | awk '{print $1}'); do
  taskset -cp "$tid"
done

wait "$pid"
