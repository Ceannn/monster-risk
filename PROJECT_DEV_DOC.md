# Risk Online Inference Project — 开发文档（给下一轮 Chat / 新同学上手用）

> 适用范围：本仓库实现“高频在线欺诈检测推理服务”，核心目标是 **吞吐极限 + 可控 P99**，并且把**排队/调度导致的尾延迟**拆到可观测。fileciteturn10file0L3-L19
> 这份文档基于当前 repo 的 `DEV_GUIDE` 以及本次对 bench2/bench3 的演进经验整理；**如果 CLI/接口有漂移，一律以各 crate 的 `--help` 为准**。fileciteturn10file0L152-L154

---

## 0. 你需要先记住的 3 个“物理事实”

1. **CPU-bound 推理必须隔离**：不能把模型推理塞在无界 async 里，否则尾延迟会被 runtime 调度放大，甚至 OOM。fileciteturn10file0L11-L16
2. **必须显式背压**：bounded queue / 429 / 降级（否则 knee 左移 + P99 爆）。fileciteturn10file0L13-L16
3. **测 knee 必须 open-loop**（Poisson 最经典）：closed-loop 会“自适应降速”，把拥塞藏起来。fileciteturn10file0L17-L19

把它想成“高速公路”：

- **吞吐**像每秒通过车辆数；
- **P99**像最堵那 1% 车辆的到达时间；
- **背压**像匝道红绿灯，避免主路彻底瘫痪；
- **open-loop**像你坚持按固定车流往里送车，才能看到道路在什么车流下崩盘（knee）。

---

## 1. 仓库结构总览（Workspace / crates）

### 1.1 crate 关系图（宏观）

```
            ┌──────────────────────────────┐
HTTP Client │  risk-bench / risk-bench2    │  open-loop 生成负载 + 统计
            └──────────────┬───────────────┘
                           │ HTTP
                           ▼
            ┌──────────────────────────────┐
            │     risk-server-tokio         │  axum/hyper: 解析、路由、错误映射、metrics
            └──────────────┬───────────────┘
                           │ 调用
                           ▼
            ┌──────────────────────────────┐
            │          risk-core            │  AppCore pipeline：parse→feature→router→xgb→(l2)→serialize
            │    ├─ schema.rs (协议真理)    │
            │    ├─ pipeline.rs (AppCore)   │
            │    ├─ xgb_pool.rs (隔离池)    │
            │    └─ xgb_runtime.rs          │
            └──────────────┬───────────────┘
                           │ FFI
                           ▼
            ┌──────────────────────────────┐
            │           xgb-ffi             │  XGBoost C API 薄封装
            └──────────────────────────────┘
```

> `risk-server-tokio` 暴露 `/score_xgb_pool`（主压测口）和 `/metrics`。fileciteturn10file0L22-L31
> `risk-core` 的 pipeline 顺序与 stage 统计口径也在 `DEV_GUIDE` 中定义。fileciteturn10file0L46-L55

### 1.2 crates/risk-server-tokio（HTTP 服务层）

职责（服务层做“薄胶水”）：

- HTTP server（axum/hyper）fileciteturn10file0L22-L25
- endpoints：
  - `POST /score_xgb_pool`：主压测口（走 XgbPool 或 L1+L2 级联）fileciteturn10file0L26-L29
  - `GET /metrics`：Prometheus 指标 fileciteturn10file0L30-L31
- JSON parse（支持 `body` 或 `body.features`）fileciteturn10file0L34-L37
- 错误映射到 HTTP：典型是 200/429/503（取决于队列满、deadline miss、内部错误等）fileciteturn10file0L38-L40

> 本次 bench2/bench3 还在压测 `POST /score_dense_f32_bin`（binary dense f32le 快路径）。这个接口属于“当前进度”而非 `DEV_GUIDE` 旧文明确写死的口径：如果你是新 Chat，第一件事是 `risk-server-tokio --help` + 搜索路由表确认它仍存在。fileciteturn10file0L152-L154

### 1.3 crates/risk-core（核心 pipeline + 运行时封装）

核心文件与职责：

- `schema.rs`：协议唯一真理（ScoreRequest/ScoreResponse/Decision/ReasonItem/TimingsUs）fileciteturn10file0L46-L50
- `pipeline.rs`：`AppCore`，按 stage 串联：`parse → feature → router → xgb(L1) → (可选 L2) → serialize`fileciteturn10file0L50-L53
  - **重要提示**：很多函数是 sync，不要乱 `.await`，否则会误导你对执行/调度的理解。fileciteturn10file0L54-L55
- `xgb_pool.rs`：推理隔离池（每 OS 线程持有一份 Booster），核心是 bounded queue + 拒绝策略 + warmup + 指标。fileciteturn10file0L56-L65
- `xgb_runtime.rs`：单 Booster 的封装（predict*proba*\* 等；dense buffer 复用是常见优化点）。fileciteturn10file0L66-L69

### 1.4 crates/xgb-ffi（XGBoost C API 薄封装）

- 只做 handle 管理、predict、错误处理；不掺业务与 pipeline。fileciteturn10file0L70-L77

### 1.5 crates/risk-bench（旧 bench，JSONL 负载）

- open-loop / Poisson pacerfileciteturn10file0L82-L85
- JSONL body 池（避免重复同 body 造成分布失真）fileciteturn10file0L84-L87
- 输出：ok/err/dropped、p50/p95/p99、attempted_rps/ok_rps/drop_pct、HTTP 状态分类、stage_p99、bench 生成器 lag/missed_ticks。fileciteturn10file0L88-L99

### 1.6 crates/risk-bench2（新 bench / bench3 演进，binary dense）

这部分是本次 chat 的“主战场”：

- 目标：把 bench 变成“稳定、可控、超高频 HTTP 轰炸机”，并且能自证 **bench 不是瓶颈**（quality gate / tick lag / qwait 等）。
- 当前 CLI（你贴的 `--help`）已经收敛为：`--rps/--warmup/--duration/--threads/--concurrency/--conns-per-thread/--timeout-ms/--pacer/...`
- 负载：`--xgb-dense-file`（f32le 二进制矩阵） + `--xgb-dense-dim`（维度） + `Content-Type: application/octet-stream`
- 输出：你看到的 `[bench2] stage_p99(us): ...`、`pacer_lag_p99`、`client_qwait_*`、`quality_gate` 等，都是为了让 bench 自己“可验真”。

---

## 2. 对外协议与接口（HTTP）

### 2.1 JSON 口（/score_xgb_pool）

- 请求体：必须是 JSON object；若外层存在 `"features": {...}` 则取内部 object，否则整个 object 作为 features。fileciteturn10file0L100-L108
- 响应体：`ScoreResponse`，典型字段包含 `trace_id/score/decision/reason/timings_us`。fileciteturn10file0L109-L121
- `timings_us` 的 stage 划分是后续分析的“时间轴标准”。fileciteturn10file0L118-L120

### 2.2 Binary 口（/score_dense_f32_bin，本次压测在用）

> 这是本次 bench2/bench3 的主压测口（基于你提供的压测日志）。它的目的是去掉 JSON parse 的不确定性，让压力更“纯粹”，更像真实线上 RPC 的固定体积 payload。

**请求**（bench2 发出的约定）：

- HTTP：`POST /score_dense_f32_bin`
- Header：`Content-Type: application/octet-stream`
- Body：连续的 `f32` little-endian，布局：`rows * (dense_dim * 4 bytes)`（bench2 默认会从文件按 row 取 slice）。

**响应**：仍然是 JSON（便于 stage 拆分、timings 输出），bench2 会解析 timings 做 stage_p99。

---

## 3. 推理流水线（AppCore）怎么跑

### 3.1 Stage 语义（你在日志里看到的 parse/feature/router/xgb/l2/serialize）

在 `risk-core` 中，pipeline 顺序固定：fileciteturn10file0L50-L53

1. **parse**：把输入（JSON 或 binary）转成内部表示
2. **feature**：特征工程 / 派生特征（可能含规则或缺失值填充）
3. **router**：路由决策（是否触发 L2、是否预算允许、是否 deadline 允许）
4. **xgb(L1)**：主模型推理
5. **(可选) L2**：级联模型推理（更准、更慢、触发率决定系统相变）fileciteturn10file0L19-L20
6. **serialize**：组装 ScoreResponse（timings/reason/decision）

> 这套 stage 拆分的意义：当你看到 p99 爆掉，要能回答“是模型慢？还是排队慢？还是 bench 自己抖？”——这就是论文要写清楚的部分。fileciteturn10file0L15-L19

### 3.2 XgbPool：为什么它是“尾延迟的保险丝”

XgbPool 的核心设计点（都在 `DEV_GUIDE` 写了）：

- N 个 OS threads，每线程持有 Booster，隔离 CPU-bound 推理 fileciteturn10file0L56-L60
- bounded queue（cap）：队列满就拒绝（避免无限排队把 P99 拖穿）fileciteturn10file0L60-L64
- 指标：`queue_wait_us`、`xgb_compute_us`、`deadline_miss_total`、QueueFull 等 fileciteturn10file0L64-L65

把它比喻成“厨房出餐口”：

- 顾客（HTTP 请求）可以不断下单；
- 厨师（XGB worker threads）数量固定；
- **出餐口队列**有容量，满了就让顾客别排（429/降级），否则店里会挤爆，最后大家都等到崩溃（P99 爆炸）。

---

## 4. 指标与定位：如何判断瓶颈到底在谁

### 4.1 Prometheus（服务侧）你最该看什么

服务侧关键指标建议按这个顺序看：fileciteturn10file0L187-L209

1. `xgb_pool_queue_wait_us`（summary）：p99/0.999 上升 ⇒ 排队尾爆
2. `xgb_pool_xgb_compute_us`（summary）：compute 稳但整体 p99 爆 ⇒ 不是模型慢，是排队/调度/bench 抖
3. `xgb_pool_deadline_miss_total`：deadline 策略触发次数
4. 级联：`router_l2_trigger_total`、`router_l2_skipped_budget_total`、`stage_l2_us`

### 4.2 bench2/bench3（客户端侧）你现在已经“能自证”什么

你最新一批日志非常关键：

- `missed_ticks_total=0`、`client_qwait_* = 0`、20k~22k RPS PASS，这基本说明：**bench 生成器已经不是主要瓶颈**（至少 tick 这条线干净了）。
- 失败点从 “bench-limited missed_ticks” 转成 “timeout / 429 / conn_queue_full”，这反而是好事：说明你正在更真实地撞服务端的 knee（或服务端 timeout/背压策略）。

---

## 5. 运行方式（强烈推荐的“经典命令”）

### 5.1 服务端（Tokio server）推荐启动命令（带绑核 + pool 配置）

下面这套是 `DEV_GUIDE` 的“经典模板”，适合作为默认 baseline：fileciteturn10file0L136-L151

```bash
# 建议先确认 CLI：cargo run -p risk-server-tokio --release -- --help
# （历史上出现过 --listen 不存在的变更）fileciteturn10file0L152-L154

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
```

> 这套命令背后的原则：XGB worker 核 / Tokio runtime 核 / bench 核 **必须分区不重叠**。fileciteturn10file0L130-L131

### 5.2 bench（risk-bench）经典跑法（JSONL /score_xgb_pool）

`DEV_GUIDE` 原版命令如下：fileciteturn10file0L155-L165

```bash
taskset -c 26-31 \
target/release/risk-bench \
  --url http://127.0.0.1:8080/score_xgb_pool \
  --metrics-url http://127.0.0.1:8080/metrics \
  --rps 14000 --duration 20 \
  --concurrency 256 \
  --pacer-shards 512 --pacer-jitter-us 0 \
  --xgb-body-file data/bench/ieee_20k.jsonl
```

### 5.3 bench2/bench3（risk-bench2）经典跑法（binary /score_dense_f32_bin）

> 下面命令按你当前 CLI（`--warmup/--duration` 是 **秒整数**）写。
> 你机器上已有：`data/bench/ieee_20k.f32`，dense dim=432（来自 feature_names.json 长度）。
> 如果你想“复刻你刚刚 PASS 的那档”，直接从这里开始。

```bash
# 建议 release 构建
cargo build -p risk-bench2 --release

taskset -c 26-31 \
target/release/risk-bench2 \
  --url http://127.0.0.1:8080/score_dense_f32_bin \
  --rps 20000 \
  --warmup 10 \
  --duration 30 \
  --threads 6 \
  --concurrency 2048 \
  --conns-per-thread 8 \
  --timeout-ms 50 \
  --pacer fixed \
  --pacer-jitter-us 0 \
  --window-ms 200 \
  --window-csv bench3_ts_fixed_rps20000_t{t}.csv \
  --xgb-dense-file data/bench/ieee_20k.f32 \
  --xgb-dense-dim 432 \
  --content-type application/octet-stream \
  --progress
```

#### knee sweep（手动扫 RPS，找 knee）

`DEV_GUIDE` 的思路是“从低到高扫，每档 20s+，记录 ok_rps/p99/429/5xx/queue_wait”。fileciteturn10file0L166-L179
对 bench2 来说，你可以直接这么做：

```bash
for rps in 20000 21000 22000 23000 24000 25000; do
  taskset -c 26-31 target/release/risk-bench2 \
    --url http://127.0.0.1:8080/score_dense_f32_bin \
    --rps $rps --warmup 10 --duration 30 \
    --threads 6 --concurrency 2048 --conns-per-thread 8 \
    --timeout-ms 50 --pacer fixed \
    --xgb-dense-file data/bench/ieee_20k.f32 --xgb-dense-dim 432 \
    --window-ms 200 --window-csv bench3_ts_fixed_rps${rps}_t{t}.csv \
    --progress
done
```

---

## 6. 实验方法论（怎么做“对论文有用的图”）

### 6.1 你现在的“正确下一步”是什么？

你前面那句“4 核给 bench，在螺丝壳里做道场吗？”其实非常到位。
当 bench 已经能在 20k~22k 稳定 PASS（并且 missed_ticks=0）时，继续在 bench 上抠 50us 意义不大——下一步应该把精力回到 **论文核心变量**：

- **固定 RPS**（例如 14k / 18k / 20k），扫 **L2 触发率**：0%、1%、5%、10%、20%
- 画出：ok_rps、p99、http_429/5xx、stage_l2_p99、queue_wait_p99
- 写出结论：触发率预算导致吞吐/尾延迟“相变曲线”。fileciteturn10file0L265-L273

> 这部分已经在 `DEV_GUIDE` 的“下一步计划”里点名了，是最值钱的产出。fileciteturn10file0L542-L549

### 6.2 并发怎么选，避免 bench 误伤

`DEV_GUIDE` 给了一个非常实用的粗估：
`required_concurrency ≈ rps * p95_latency`。fileciteturn10file0L180-L185
换成直觉：你要保证“路上有足够多的车”，不然 open-loop 的发车频率根本跑不满目标 RPS。

---

## 7. 本次 chat 我们到底做了什么（给下一轮 Chat 的“交接清单”）

> 你可以把这一段当成“变更摘要”，下一轮 Chat 只要从这里接着干。

1. **重构并修到能用的 risk-bench2（bench3 演进）**
   - 目标：变成高精度 open-loop HTTP loadgen，能跑到 20k+ RPS，且自带质量门（quality gate）。
   - 过程中修了大量 Rust 编译错误（字段名漂移、宏/derive 冲突、move 语义、格式化参数、Result 返回类型等），把 CLI 也改到“简单好用”。
2. **把“bench-limited”从主要噪声源降下去**
   - 你最新日志显示 `missed_ticks_total=0`、`quality_gate: PASS`，说明 tick 生成器已经跟得上目标节奏。
3. **把实验焦点拉回“knee 右移”与“超载行为”**
   - 你已经看到 23k+ 开始出现 timeout/429/drop_conn_queue_full，这就是典型 knee 之后的行为，接下来应该对照服务侧 metrics（queue_wait/compute/deadline miss）解释原因。fileciteturn10file0L187-L199
4. **零拷贝/减少拷贝（bench 与 server 的 I/O）**
   - bench2 使用二进制 payload（dense f32le）是“减少 parse 不确定性”的重要一步；是否进一步做到 server 端真正零拷贝，需要看 server handler 对 body 的读取方式（建议下一轮 Chat 直接从 server 的 handler 开始做 profile）。

---

## 8. 仍然值得 refine 的点（但建议按优先级来）

### P0（直接影响论文 / 真正的系统结论）

- **L2 触发率 sweep**（相变曲线）fileciteturn10file0L265-L273
- 明确并记录 429/503/降级策略（“早拒绝 vs 晚成功”）fileciteturn10file0L540-L541

### P1（把结论做得更干净：减少测量噪声）

- 服务器侧：Tokio runtime 线程也 pin（避免与 XGB 抢核导致随机尾爆）fileciteturn10file0L513-L515
- bench2：把 `/metrics` 采样也接入（像 risk-bench 一样），让每档压测自动保存服务侧关键 summary（queue_wait/compute/deadline miss）。

### P2（锦上添花：bench 本身再“monster”一点）

- 真正的 `*_spin` pacer（最后 N 微秒 busy-spin），用于把 tick jitter 压到极限（但要小心它会吃掉 CPU，必须独立绑核）。
- 更强的“质量门”：除 missed_ticks 外，加上 “pacer_lag_p99 超阈值就 WARN/FAIL”。
- 更丰富的 CSV：把 stage 分布 + HTTP 分类 + 服务侧 metrics 合并到同一时间轴（做论文图特别爽）。

---

## 9. 最后：新同学/新 Chat 的启动 checklist

1. `cargo run -p risk-server-tokio --release -- --help`（确认 CLI 没漂移）fileciteturn10file0L152-L154
2. 按“经典命令”启动 server（先 L1-only baseline，再开 L2）。fileciteturn10file0L136-L150
3. 先跑 `risk-bench2` 固定 20k、22k，看是否 PASS（missed_ticks=0）。
4. 开始 knee sweep：20k→26k，记录拐点（timeout/429/drop）。fileciteturn10file0L166-L179
5. 把拐点与 Prometheus `queue_wait_us / compute_us / deadline_miss_total` 对齐解释。fileciteturn10file0L187-L199
6. 进入论文核心实验：固定 RPS 扫 L2 触发率，画相变曲线。fileciteturn10file0L265-L273

---

> 如果你把这份文档交给下一轮 Chat：
>
> - 直接给它你的最新压测日志 + `/metrics` 的关键曲线（queue_wait/compute/deadline miss），它就能快速进入“分析 knee 与相变”的状态，而不是再陪编译器打拳。
