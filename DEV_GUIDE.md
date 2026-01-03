1. 项目目标（论文核心目标）

本项目实现一个面向高频信用卡交易欺诈检测的在线推理服务，强调 吞吐极限 + 可控 P99：

吞吐目标：单机尽可能推高 RPS（最终目标可达 10k~20k+，视硬件）

尾延迟目标：P99 在 SLO 范围内稳定（核心研究对象：排队/调度造成的尾爆）

方法论：

CPU-bound 推理必须隔离（线程/模型）

必须显式背压（bounded queue / 429/降级）

尾延迟必须可观测分解（queue_wait vs compute）

压测必须用 open-loop/Poisson 才能测 knee（closed-loop 会掩盖拥塞）

引入级联（L1/L2）后，研究触发率预算导致的“相变曲线”

2. Repo 结构与职责（crates）
2.1 crates/risk-server-tokio

HTTP 服务（axum/hyper）

暴露 endpoints：

POST /score_xgb_pool：主压测口（走 XgbPool / 或 L1+L2 级联）

GET /metrics：Prometheus 指标

负责：

JSON parse（支持 body 或 body.features）

调用 risk-core 的 AppCore pipeline

将错误映射为 HTTP（200/429/503）

tower trace / metrics 暴露

2.2 crates/risk-core

核心 pipeline + 模型运行时封装：

schema.rs：协议唯一真理

ScoreRequest/ScoreResponse/Decision/ReasonItem/TimingsUs

pipeline.rs：AppCore

将输入 object（Map）跑完：parse → feature → router → xgb(L1) → (可选 L2) → serialize

注意：很多函数是 sync（不要 .await），以免误以为 async

xgb_pool.rs：推理隔离池

N OS threads，每线程持有一份 Booster

bounded queue（cap）

warmup

metrics：queue_wait_us、xgb_compute_us、deadline_miss_total、QueueFull 等

xgb_runtime.rs：单 Booster 封装

predict_proba_* 等；（性能优化点：dense buffer 复用、scalar 快路径、topk partial select）

2.3 crates/xgb-ffi

XGBoost C API 的薄封装

只做 handle 管理、predict、错误处理

不掺业务字段/不掺 pipeline

2.4 crates/risk-bench

压测工具（极关键）：

open-loop/Poisson pacer

JSONL body 池（避免重复同 body 导致分布失真）

并发控制（concurrency 为上限，避免无界 in-flight）

输出 stdout + CSV：

ok/err/dropped、p50/p95/p99、attempted_rps/ok_rps/drop_pct

HTTP 状态分类：http_2xx/http_429/http_5xx/timeout

stage_p99：parse/feature/router/xgb/l2/serialize

xgb_pool：deadline_miss_total、queue_wait_p99、compute_p99

bench 生成器：bench_lag_p99 / missed_ticks_total

3. 协议与接口（HTTP/JSON）
请求体

必须是 JSON object

若外层存在 "features": {...} 则取内部 object

否则整个 object 作为 features

响应体（ScoreResponse）

典型字段：

{
  "trace_id": "...",
  "score": 0.0,
  "decision": "allow|deny|manual_review|degrade_allow",
  "reason": [],
  "timings_us": {
    "parse":0, "feature":0, "router":0, "xgb":0, "l2":0, "serialize":0
  }
}

4. 运行方式（服务端）
4.1 基本准则（必须遵守）

CPU-bound 推理不允许无界 async 化

必须：bounded queue + 429（或降级）

XGB worker 核 / Tokio runtime 核 / bench 核 必须分区不重叠

XGBoost 内部线程数固定：OMP_NUM_THREADS=1

WSL2 下 RSS 控制：MALLOC_ARENA_MAX=2

4.2 推荐启动命令（Tokio glue + L1/L2 pool）
OMP_NUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
TOKIO_WORKER_THREADS=4 \
TOKIO_MAX_BLOCKING_THREADS=4 \
XGB_L1_POOL_THREADS=8 \
XGB_L1_POOL_QUEUE_CAP=512 \
XGB_L1_POOL_PIN_CPUS="2,4,6,8,10,12,14,16" \
XGB_L2_POOL_THREADS=4 \
XGB_L2_POOL_QUEUE_CAP=256 \
XGB_L2_POOL_PIN_CPUS="19,21,23,25" \
taskset -c 2-25 \
cargo run -p risk-server-tokio --release -- \
  --model-dir models/ieee_l1 \
  --model-l2-dir models/ieee_l2


注意：CLI 选项以 risk-server-tokio --help 为准（历史上出现过 --listen 不被识别的变更）。

5. 压测方式（bench）
5.1 基本压测（bench 绑核 26-31）
taskset -c 26-31 \
target/release/risk-bench \
  --url http://127.0.0.1:8080/score_xgb_pool \
  --metrics-url http://127.0.0.1:8080/metrics \
  --rps 14000 --duration 20 \
  --concurrency 256 \
  --pacer-shards 512 --pacer-jitter-us 0 \
  --xgb-body-file data/bench/ieee_20k.jsonl

5.2 knee sweep（手动扫 RPS）

例如从 14k 到 20k，每档 20s，记录 CSV

关注：

ok_rps 与 attempted_rps 的差距（drop_pct）

http_429/5xx 是否出现

stage_p99 是否由 xgb_pool.queue_wait 主导

bench_lag 与 missed_ticks 是否失真（bench 自己跟不上会污染结论）

5.3 并发选择规则（避免 bench 误伤）

concurrency 要足够大，粗略估计：
required_concurrency ≈ rps * p95_latency

如果 p95=60ms、rps=14k，则并发约需要 840；否则会出现大量 dropped（bench 的锅，不是 server）。

6. Prometheus 指标（读指标定位瓶颈）
server / pool 侧关键指标

xgb_pool_queue_wait_us（summary）

0.99/0.999 上升 ⇒ 排队尾爆

xgb_pool_xgb_compute_us（summary）

compute 稳定但整体 p99 爆 ⇒ 不是模型慢，是排队/调度/生成器抖动

xgb_pool_deadline_miss_total（counter）

deadline 策略触发次数

级联相关：

router_l2_trigger_total

router_l2_skipped_budget_total

stage_l2_us（summary）

7. IEEE-CIS 训练（L1/L2）
数据位置

data/raw/ieee-cis/ 下包含 train_transaction/train_identity 等

训练脚本（Polars Streaming）

使用 Polars Lazy/Streaming 避免一次性爆内存

输出：

models/ieee_l1/（含 model + policy.json）

models/ieee_l2/（含 model + 可选 policy）

训练结果示例（全量）

L1：valid AUC≈0.9089，AUPRC≈0.554；生成 review/deny 阈值

L2：valid AUC≈0.9227，AUPRC≈0.596；提升不大但更强

8. 已踩过的坑（防止新 Chat 重蹈覆辙）

CPU-bound 被 async 化 → OOM/尾爆
解决：XgbPool + bounded queue + 拒绝策略

只 pin XGB 不够
Tokio runtime 线程不 pin 会和 XGB 抢核导致随机尾爆
解决：核分区（XGB/Tokio/bench）

bench 用同一个 body 重复压测 → 级联测不出
解决：JSONL body 池，随机轮转

级联 L2 触发率过高会把系统拖死
需要 L2 budget/deadline 控制，否则 stage_l2_us 会出现 10~50ms p99，吞吐断崖

CLI 参数变更（--listen 不存在）
任何运行命令先 --help 校验

9. 当前项目状态（截止本次 commit/tag）

单 L1（或 L1-only xgb_pool）已达到高吞吐低 p99 的稳定区间（knee 右移明显）

bench 已支持 Poisson/open-loop、JSONL、HTTP 状态分类输出

L1/L2 模型已训练完成并产出 policy 阈值

L1/L2 级联已初步接入，但仍需：

L2 触发预算

bench 端可控触发率实验

明确 5xx/429/降级策略（论文要写清楚“早拒绝 vs 晚成功”）

10. 下一步计划（论文产出最大）
必做（论文核心图）

固定 RPS（如 14k），扫 L2 触发率（0%、1%、5%、10%、20%）

输出曲线：ok_rps、p99、http_429/5xx、stage_l2_p99、queue_wait_p99

写结论：级联触发率预算导致吞吐/尾延迟“相变”（断崖）

可选（runtime 对比）

等级联预算稳定后，再做 Tokio vs Glommio 对比（控制变量更干净）

目标：证明 runtime 是否成为结构性瓶颈，以及 glommio shard 是否右移 knee / 降抖动1. 项目目标（论文核心目标）

本项目实现一个面向高频信用卡交易欺诈检测的在线推理服务，强调 吞吐极限 + 可控 P99：

吞吐目标：单机尽可能推高 RPS（最终目标可达 10k~20k+，视硬件）

尾延迟目标：P99 在 SLO 范围内稳定（核心研究对象：排队/调度造成的尾爆）

方法论：

CPU-bound 推理必须隔离（线程/模型）

必须显式背压（bounded queue / 429/降级）

尾延迟必须可观测分解（queue_wait vs compute）

压测必须用 open-loop/Poisson 才能测 knee（closed-loop 会掩盖拥塞）

引入级联（L1/L2）后，研究触发率预算导致的“相变曲线”

2. Repo 结构与职责（crates）
2.1 crates/risk-server-tokio

HTTP 服务（axum/hyper）

暴露 endpoints：

POST /score_xgb_pool：主压测口（走 XgbPool / 或 L1+L2 级联）

GET /metrics：Prometheus 指标

负责：

JSON parse（支持 body 或 body.features）

调用 risk-core 的 AppCore pipeline

将错误映射为 HTTP（200/429/503）

tower trace / metrics 暴露

2.2 crates/risk-core

核心 pipeline + 模型运行时封装：

schema.rs：协议唯一真理

ScoreRequest/ScoreResponse/Decision/ReasonItem/TimingsUs

pipeline.rs：AppCore

将输入 object（Map）跑完：parse → feature → router → xgb(L1) → (可选 L2) → serialize

注意：很多函数是 sync（不要 .await），以免误以为 async

xgb_pool.rs：推理隔离池

N OS threads，每线程持有一份 Booster

bounded queue（cap）

warmup

metrics：queue_wait_us、xgb_compute_us、deadline_miss_total、QueueFull 等

xgb_runtime.rs：单 Booster 封装

predict_proba_* 等；（性能优化点：dense buffer 复用、scalar 快路径、topk partial select）

2.3 crates/xgb-ffi

XGBoost C API 的薄封装

只做 handle 管理、predict、错误处理

不掺业务字段/不掺 pipeline

2.4 crates/risk-bench

压测工具（极关键）：

open-loop/Poisson pacer

JSONL body 池（避免重复同 body 导致分布失真）

并发控制（concurrency 为上限，避免无界 in-flight）

输出 stdout + CSV：

ok/err/dropped、p50/p95/p99、attempted_rps/ok_rps/drop_pct

HTTP 状态分类：http_2xx/http_429/http_5xx/timeout

stage_p99：parse/feature/router/xgb/l2/serialize

xgb_pool：deadline_miss_total、queue_wait_p99、compute_p99

bench 生成器：bench_lag_p99 / missed_ticks_total

3. 协议与接口（HTTP/JSON）
请求体

必须是 JSON object

若外层存在 "features": {...} 则取内部 object

否则整个 object 作为 features

响应体（ScoreResponse）

典型字段：

{
  "trace_id": "...",
  "score": 0.0,
  "decision": "allow|deny|manual_review|degrade_allow",
  "reason": [],
  "timings_us": {
    "parse":0, "feature":0, "router":0, "xgb":0, "l2":0, "serialize":0
  }
}

4. 运行方式（服务端）
4.1 基本准则（必须遵守）

CPU-bound 推理不允许无界 async 化

必须：bounded queue + 429（或降级）

XGB worker 核 / Tokio runtime 核 / bench 核 必须分区不重叠

XGBoost 内部线程数固定：OMP_NUM_THREADS=1

WSL2 下 RSS 控制：MALLOC_ARENA_MAX=2

4.2 推荐启动命令（Tokio glue + L1/L2 pool）
OMP_NUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
TOKIO_WORKER_THREADS=4 \
TOKIO_MAX_BLOCKING_THREADS=4 \
XGB_L1_POOL_THREADS=8 \
XGB_L1_POOL_QUEUE_CAP=512 \
XGB_L1_POOL_PIN_CPUS="2,4,6,8,10,12,14,16" \
XGB_L2_POOL_THREADS=4 \
XGB_L2_POOL_QUEUE_CAP=256 \
XGB_L2_POOL_PIN_CPUS="19,21,23,25" \
taskset -c 2-25 \
cargo run -p risk-server-tokio --release -- \
  --model-dir models/ieee_l1 \
  --model-l2-dir models/ieee_l2


注意：CLI 选项以 risk-server-tokio --help 为准（历史上出现过 --listen 不被识别的变更）。

5. 压测方式（bench）
5.1 基本压测（bench 绑核 26-31）
taskset -c 26-31 \
target/release/risk-bench \
  --url http://127.0.0.1:8080/score_xgb_pool \
  --metrics-url http://127.0.0.1:8080/metrics \
  --rps 14000 --duration 20 \
  --concurrency 256 \
  --pacer-shards 512 --pacer-jitter-us 0 \
  --xgb-body-file data/bench/ieee_20k.jsonl

5.2 knee sweep（手动扫 RPS）

例如从 14k 到 20k，每档 20s，记录 CSV

关注：

ok_rps 与 attempted_rps 的差距（drop_pct）

http_429/5xx 是否出现

stage_p99 是否由 xgb_pool.queue_wait 主导

bench_lag 与 missed_ticks 是否失真（bench 自己跟不上会污染结论）

5.3 并发选择规则（避免 bench 误伤）

concurrency 要足够大，粗略估计：
required_concurrency ≈ rps * p95_latency

如果 p95=60ms、rps=14k，则并发约需要 840；否则会出现大量 dropped（bench 的锅，不是 server）。

6. Prometheus 指标（读指标定位瓶颈）
server / pool 侧关键指标

xgb_pool_queue_wait_us（summary）

0.99/0.999 上升 ⇒ 排队尾爆

xgb_pool_xgb_compute_us（summary）

compute 稳定但整体 p99 爆 ⇒ 不是模型慢，是排队/调度/生成器抖动

xgb_pool_deadline_miss_total（counter）

deadline 策略触发次数

级联相关：

router_l2_trigger_total

router_l2_skipped_budget_total

stage_l2_us（summary）

7. IEEE-CIS 训练（L1/L2）
数据位置

data/raw/ieee-cis/ 下包含 train_transaction/train_identity 等

训练脚本（Polars Streaming）

使用 Polars Lazy/Streaming 避免一次性爆内存

输出：

models/ieee_l1/（含 model + policy.json）

models/ieee_l2/（含 model + 可选 policy）

训练结果示例（全量）

L1：valid AUC≈0.9089，AUPRC≈0.554；生成 review/deny 阈值

L2：valid AUC≈0.9227，AUPRC≈0.596；提升不大但更强

8. 已踩过的坑（防止新 Chat 重蹈覆辙）

CPU-bound 被 async 化 → OOM/尾爆
解决：XgbPool + bounded queue + 拒绝策略

只 pin XGB 不够
Tokio runtime 线程不 pin 会和 XGB 抢核导致随机尾爆
解决：核分区（XGB/Tokio/bench）

bench 用同一个 body 重复压测 → 级联测不出
解决：JSONL body 池，随机轮转

级联 L2 触发率过高会把系统拖死
需要 L2 budget/deadline 控制，否则 stage_l2_us 会出现 10~50ms p99，吞吐断崖

CLI 参数变更（--listen 不存在）
任何运行命令先 --help 校验

9. 当前项目状态（截止本次 commit/tag）

单 L1（或 L1-only xgb_pool）已达到高吞吐低 p99 的稳定区间（knee 右移明显）

bench 已支持 Poisson/open-loop、JSONL、HTTP 状态分类输出

L1/L2 模型已训练完成并产出 policy 阈值

L1/L2 级联已初步接入，但仍需：

L2 触发预算

bench 端可控触发率实验

明确 5xx/429/降级策略（论文要写清楚“早拒绝 vs 晚成功”）

10. 下一步计划（论文产出最大）
必做（论文核心图）

固定 RPS（如 14k），扫 L2 触发率（0%、1%、5%、10%、20%）

输出曲线：ok_rps、p99、http_429/5xx、stage_l2_p99、queue_wait_p99

写结论：级联触发率预算导致吞吐/尾延迟“相变”（断崖）

可选（runtime 对比）

等级联预算稳定后，再做 Tokio vs Glommio 对比（控制变量更干净）

目标：证明 runtime 是否成为结构性瓶颈，以及 glommio shard 是否右移 knee / 降抖动
