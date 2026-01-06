#!/usr/bin/env bash
set -euo pipefail

LD="/home/ceann/miniforge3/envs/ml/lib:/home/ceann/miniforge3/envs/ml/lib/python3.12/site-packages/xgboost/lib:/home/ceann/miniforge3/envs/ml/lib/python3.12/site-packages/xgboost/xgboost.libs"

sudo env \
  LD_LIBRARY_PATH="$LD" \
  OMP_NUM_THREADS=1 \
  MALLOC_ARENA_MAX=2 \
  prlimit --memlock=unlimited -- \
  taskset -c 2,4,6,8,10,12,14,16,17,19,21,23 \
  target/release/risk-server-glommio \
    --listen 127.0.0.1:8080 \
    --model-dir models/ieee_l1 \
    --max-in-flight 4096
