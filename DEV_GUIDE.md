# DEV_GUIDE

> 这份文档是“能一键跑起来 + 能稳定压测 + 能解释结果”的最小闭环指南。  
> 目标是：**用同一套命令**把 `risk-server-tokio` 跑稳、把 `risk-bench2` 打爆，并且能快速判断：到底是 **server-limited** 还是 **bench-limited**。

---

## 1. 项目目标（我们到底在追什么）

1) **把线上“高频打分服务”拆成可控的实验台：**
- Tokio HTTP server：`crates/risk-server-tokio`
- 推理核心：`crates/risk-core`（L1/L2、XGBoost pool、背压/降级）
- 压测器：`crates/risk-bench2`（benchV3 方向：每核一线程 + 多连接 keepalive + 精准 pacer）

2) **强约束：不允许“测不准”。**  
bench 不能成为瓶颈，否则你看到的 P99 就像“拿温度计测电压”——数字很漂亮，但结论不可信。

3) **接口方向：从 JSON → 二进制（更接近真实线上）。**
- 请求：dense f32 little-endian bytes（`application/octet-stream`）
- 响应：`RSK1` 二进制封包（包含 score + stage timings）

---

## 2. 本地开发环境（建议配置）

- Rust stable（建议 `rustup default stable`）
- Linux（本地 loopback 压测 OK；想测网络栈/网卡请用两机）
- 建议编译参数：
  ```bash
  export RUSTFLAGS="-C target-cpu=native"
  ```

> 小比喻：`-C target-cpu=native` 就像“让编译器知道你这台 CPU 真的有 AVX2/Zen4/…”，否则它会按保守档跑。

---

## 3. 数据与模型准备

### 3.1 常用数据文件（你现在已经有了）
你当前目录里看到的这些就够了：

- `data/bench/ieee_20k.f32`：**dense f32le 二进制**（推荐用于压测）
- `data/bench/ieee_20k.jsonl` / `ieee_20k_posmix.jsonl`：历史 JSONL（更多用于功能对齐/调试）

维度：
- `models/ieee_l1/feature_names.json` 长度应为 **432**（你测的就是 432）

### 3.2 dense f32le 文件格式（重要！）
`ieee_20k.f32` 的布局：

- 每行 = `dense_dim * 4` 字节（f32 little-endian）
- 文件总大小 = `rows * dense_dim * 4`

比如 `rows=20000`，`dense_dim=432`：
- 单行字节数 = 432 * 4 = **1728 bytes**
- 文件大小约 = 20000 * 1728 ≈ **33MB**（与你 ls 看到的吻合）

---

## 4. 启动 risk-server-tokio（Tokio server）

### 4.1 Server CLI（以当前接口为准）
`risk-server-tokio` 目前核心参数是：

- `--listen <IP:PORT>`：监听地址（**没有 `--host`**，所以你之前会报错）
- `--model-dir <DIR>`：L1 模型目录
- `--model-l2-dir <DIR>`：L2 模型目录
- `--max-in-flight <N>`：HTTP 层最大并发（超出会 429，保护服务）

此外，Tokio runtime 线程数从环境变量读：
- `TOKIO_WORKER_THREADS`（默认 4）
- `TOKIO_MAX_BLOCKING_THREADS`（默认 4）

### 4.2 推荐启动命令（经典“绑核 + 限制线程 + 背压”）
下面给你一套“最经典的那种”，你直接复制就能跑：

```bash
# 1) 编译（可选，但推荐先 build）
export RUSTFLAGS="-C target-cpu=native"
cargo build -p risk-server-tokio --release

# 2) 绑核启动（示例：给 server 8 个逻辑核；按你机器改）
OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 TOKIO_WORKER_THREADS=4 TOKIO_MAX_BLOCKING_THREADS=4 XGB_L1_POOL_THREADS=8 XGB_L1_POOL_QUEUE_CAP=512 XGB_L1_POOL_WARMUP_ITERS=1000 XGB_L1_POOL_PIN_CPUS=0,1,2,3,4,5,6,7 XGB_L1_POOL_EARLY_REJECT_US=50 XGB_L2_POOL_THREADS=4 XGB_L2_POOL_QUEUE_CAP=256 XGB_L2_POOL_WARMUP_ITERS=200 XGB_L2_POOL_PIN_CPUS=8,9,10,11 XGB_L2_POOL_EARLY_REJECT_US=50 taskset -c 0-11 cargo run -p risk-server-tokio --release --   --listen 127.0.0.1:8080   --model-dir models/ieee_l1   --model-l2-dir models/ieee_l2   --max-in-flight 4096
```

说明（你会用得上）：
- `OMP_NUM_THREADS=1`：强制 XGBoost 单线程（否则会在 pool 里“线程套线程”爆炸）
- `*_POOL_THREADS / *_POOL_QUEUE_CAP`：XGB worker 数与队列上限（核心背压开关）
- `*_POOL_PIN_CPUS`：把 worker 固定到核上（避免跑来跑去抖动 P99）
- `*_POOL_EARLY_REJECT_US`：当预测排队等待超过阈值，提前拒绝（可选，但对尾延迟有帮助）

---

## 5. 接口（对齐当前进度）

### 5.1 推荐压测接口：`POST /score_dense_f32_bin`
- Request:
  - `Content-Type: application/octet-stream`
  - Body：**exactly `dense_dim` 个 f32le**（即 `dense_dim * 4` bytes）
- Response:
  - `Content-Type: application/octet-stream`
  - Body：48 bytes 的 `RSK1` 二进制封包（小、稳定、便于解析）

#### 5.1.1 `RSK1` 响应格式（48 bytes）
| offset | size | type | 含义 |
|---:|---:|---|---|
| 0 | 4 | bytes | magic = `RSK1` |
| 4 | 2 | u16le | version（当前 1） |
| 6 | 2 | u16le | flags（当前 0） |
| 8 | 8 | u64le | trace_id（服务端自增） |
| 16 | 4 | f32le | score |
| 20 | 24 | 6×u32le | stage_us：parse, feature, router, xgb, l2, serialize |
| 44 | 4 | u32le | reserved（0） |

> 类比：这就像给每个请求都发了一个“迷你 flight recorder”，你不用解析 JSON，也能知道每段耗时。

### 5.2 兼容接口（调试用，非压测首选）
- `/score_xgb_pool_async` / `/score_dense_f32` 等（JSON 体/JSON 回包）
- 这些保留用于对齐逻辑、debug 与回归；压测建议统一走 bin 版本，噪声更小。

---

## 6. 启动 risk-bench2（benchV3 方向）

### 6.1 bench2 CLI（你现在的 usage）
你目前看到的参数已经是“修到能用”的版本（重点变化）：

- `--duration` / `--warmup`：**单位是秒（整数）**，不能写 `10s`
- 并发用 `--concurrency`（替代你之前记忆里的 `--inflight-total`）
- payload 文件用 `--xgb-dense-file`（替代 `--payload-f32-file`）
- pacer：
  - `--pacer poisson|fixed`
  - `--pacer-spin-threshold-us N`：最后 N 微秒 busy-spin（用来榨干 tick 抖动）

### 6.2 经典压测命令（固定 pacer，逐步拉 RPS）
> 下面这套就是你一直在跑的“黄金模板”，我把坑都填平了。

```bash
export RUSTFLAGS="-C target-cpu=native"
cargo build -p risk-bench2 --release

# 给 bench 4~6 个核（示例：12-15）；确保不和 server 绑到同一批核
taskset -c 12-15 cargo run -p risk-bench2 --release --   --url http://127.0.0.1:8080/score_dense_f32_bin   --xgb-dense-file data/bench/ieee_20k.f32   --xgb-dense-dim 432   --content-type application/octet-stream   --rps 20000   --warmup 10   --duration 30   --threads 6   --concurrency 2048   --conns-per-thread 8   --timeout-ms 50   --pacer fixed   --pacer-jitter-us 0   --pacer-spin-threshold-us 0   --window-ms 200   --window-csv bench_ts_fixed_rps20000_t{t}.csv   --progress
```

#### 6.2.1 推荐的 ramp 计划（找拐点）
固定其他参数不动，只改 `--rps`：
- 20k → 21k → 22k → 23k → 24k → 25k …

你现在的数据很清楚：
- 20k、21k、22k：基本 **PASS**
- 23k 开始：出现 `timeout`，说明已触碰服务端尾延迟/容量极限
- 25k+：`drop_conn_queue_full` + 大量 timeout，说明 client 连接队列也开始爆

---

## 7. 输出怎么读（判断是不是“螺丝壳里做道场”）

bench2 输出里你最该盯的 5 个东西：

1) **ok_rps vs attempted_rps**  
- ok_rps 跟不上 attempted_rps：要么 server 顶不住，要么 client 自己卡住

2) **429 / timeout / dropped 的结构**
- `429`：服务端背压（通常是队列满/并发上限）
- `timeout`：尾部严重拖长（也可能是 client 排队过久）
- `drop_conn_queue_full`：client 的连接发送队列顶满（bench 自身开始爆）

3) **stage_p99(us)**
- 你现在经常看到 xgb p99 在 800~2300us 的区间波动  
  ——这基本就是“模型推理 + 池化调度”的真实成本

4) **pacer_lag_p99**
- 你之前看到 ~1900us 的 lag，一般是“tick 调度与 runtime 抖动”
- 开 `--pacer-spin-threshold-us` 的意义：把最后几十微秒变成自旋，减少 oversleep

5) **quality_gate**
- `PASS`：bench 自己没有明显掉链子（结果更可信）
- `FAIL (bench-limited)`：先别优化 server，先把 bench 拉直

> 你现在已经把 `missed_ticks_total` 压到 0，说明 bench 的“发车节奏”稳定很多；  
> 这就是从 benchV2 → benchV3 的质变点。

---

## 8. 常见坑（你踩过的我都写死在这里）

- `error: unexpected argument '--host' found`  
  ✅ server 用 `--listen 127.0.0.1:8080`

- `error: invalid value '10s' for '--warmup'`  
  ✅ `--warmup 10`（单位秒，整数）

- `error: unexpected argument '--inflight-total' found`  
  ✅ 用 `--concurrency 2048`

- `error: unexpected argument '--payload-f32-file' found`  
  ✅ 用 `--xgb-dense-file data/bench/ieee_20k.f32`

---

## 9. 这次我们到底做了什么（进度对齐用）

这一轮 chat 的核心成果可以总结成 5 条：

1) **risk-server-tokio：支持 dense f32 二进制请求**（减少 JSON 噪声）
2) **risk-server-tokio：增加 `RSK1` 二进制响应**（小回包 + 带 stage timing）
3) **推理链路：做了零拷贝/少拷贝路径**（`align_to::<f32>()` + fallback decode）
4) **背压体系更完整**：HTTP 层 `--max-in-flight` + pool queue cap + early reject → 429
5) **risk-bench2（benchV3 方向）**：CLI 变好用、pacer 更稳、quality gate 更可信、窗口 CSV 可观测

---

## 10. 下一步建议（别只在 4 核 bench 里“内卷”）

你说得非常对：如果 bench 只有 4 核，很多优化会进入“螺丝壳里做道场”。

下一步更值钱的方向（按优先级）：

1) **两机压测**（bench 一台、server 一台）  
   - 这样你测到的才是“网络栈 + 中断 + NIC + 调度”的真实形态  
   - loopback 更像“在同一个屋里扔球”，两机才是“隔着马路扔球”

2) **明确容量目标：以 timeout=0 的 ok_rps 找拐点**  
   - 把 22k~24k 区间的曲线拉细：22.5k/23k/23.5k/…  
   - 你现在已经看到了 23k 的边界迹象，这很宝贵

3) **做一套“固定配置的回归基线”**  
   - 固定：模型、dim、concurrency、conns、timeout、pacer  
   - 每次改动只比：p50/p99、xgb_p99、429、timeout

---

## 11. 代码结构速览（便于快速定位）

- `crates/risk-server-tokio`：HTTP server（路由、并发限制、请求解析、回包）
- `crates/risk-core`：pipeline（L1/L2 路由、XgbPool、背压/early reject、metrics）
- `crates/risk-bench2`：压测器（pacer、连接池、并发、统计、CSV、quality gate）

---

如果你要“最简单一口气跑起来”的版本：  
先跑第 4 章 server 命令，再跑第 6 章 bench 命令，然后从 `--rps 20000` 开始往上加。
