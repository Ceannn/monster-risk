# Monster Risk Backend — Project Dev Doc

> 最后更新：2026-01-06  
> 这份文档记录我们从 “Tokio 勉强顶 22k” 进化到 “Glommio + TL2CGEN 稳 40k” 的关键架构与优化路径，避免上下文被聊天淹没。

---

## 0. 项目全景（模块 / 接口 / 数据流）— *给新同学/新 AI 的“防幻觉说明书”*

> 这一节的目标：**用最少的“脑补空间”把系统讲清楚**。  
> 阅读完你应该能回答：请求从哪来？进了哪个线程？什么时候入队/出队？模型到底在哪里跑？怎么保证特征顺序不翻车？

把这个项目想成一条“高速收费站”流水线：

- **Glommio** = 多个收费站口（每个 CPU core 一个执行器），负责 *接车、验票、分流、回包*。
- **risk-core** = 收费规则 + 车辆信息标准（特征 schema）+ 两道闸（L1/L2）策略。
- **TL2CGEN（Treelite successor）静态模型** = “识别器芯片”，直接焊在二进制里（静态链接），不靠 `libxgboost.so`。
- **XgbPool** = 芯片旁边的“工位排班系统”（把推理任务在多个 worker 线程间分发，控制排队长度、超时、背压）。

---

### 0.1 仓库模块地图（按 crate / 文件定位）

> 以下路径以仓库根为准（示例：`crates/risk-core/...`）。你可以把它当成导航索引。

**服务端入口（HTTP Server）**
- `crates/risk-server-glommio/`  
  Glommio 版服务端（目标：极致吞吐/低延迟）。
- `crates/risk-server-tokio/`  
  Tokio 版服务端（基线/对照组/方便调试）。

**核心业务与推理运行时（核心都在 risk-core）**
- `crates/risk-core/src/pipeline.rs`  
  业务决策流水线：L1 →（可选）L2 → 决策合并 → 输出（Decision / reason / used_l2 等）。  
  *你看到的 `used_l2`、`review_th/deny_th` 等策略字段就在这里跑。*
- `crates/risk-core/src/xgb_runtime.rs`  
  推理后端的**统一门面**：  
  - 负责选择 “native tl2cgen” 还是 “xgb_ffi（动态库）”；
  - 提供 `predict_proba_*` 这一族 API；
  - 负责把 L1/L2 的阈值策略（decide）封装起来（供 pipeline 调用）。
- `crates/risk-core/src/xgb_pool.rs`  
  推理线程池/调度层（你现在的 monster 版就是从这里起飞的）：  
  - **per-core 本地队列 + work-stealing**（降低跨核竞争）  
  - 排队容量、早拒绝（early reject）、deadline  
  - CPU pin（绑定 worker 到指定 CPU）
- `crates/risk-core/src/native_l1_tl2cgen.rs` / `native_l2_tl2cgen.rs`  
  TL2CGEN 导出的 C 模型在 Rust 里的“薄胶水层”：  
  - 负责 `dense bytes (f32le) → Entry[]` 的转换  
  - 负责调用 `extern "C"` 的 `*_predict(...)` 并返回概率

**静态模型产物（由 TL2CGEN 生成，编译进二进制）**
- `crates/risk-core/native/tl2cgen/ieee_l1/`  
- `crates/risk-core/native/tl2cgen/ieee_l2/`  
  里面包含 `main.c / header.h / quantize.c / recipe.json / prefix.h` 等文件。  
  这些 **会被 `cc` crate 编译成 `libieee_l1.a / libieee_l2.a`，并静态链接进最终 binary**。

**性能验证与压测工具**
- `crates/risk-bench2/`（以及可能的 bench3 变体）  
  用固定 RPS/固定并发向服务端打压，输出 p50/p95/p99 + stage 分解。

**训练/导出脚本（Python/Polars/XGBoost → TL2CGEN）**
- `scripts/train_ieee_l1_l2_polars_hardneg.py`  
  时间切分 + L1 训练 + Hard Negative Mining + L2 训练 + 导出 feature schema / cat maps 等。

---

### 0.2 对外接口（HTTP 协议 & Payload 约束）

当前主力接口是：

- `POST /score_dense_f32_bin`
- `Content-Type: application/octet-stream`
- **Body**：一行 dense 特征，`f32` little-endian 连续排列  
  - 字节长度必须满足：`len == dim * 4`  
  - `dim` 必须与模型 schema 一致（例如 IEEE demo 常见是 432）
  - 缺失值语义：用 **NaN** 表示 missing（Rust 侧会把 NaN 映射到 tl2cgen 的 missing 标记）

> 直觉类比：这相当于你把“432 个 float”当成一根打包好的钢筋，直接塞进收费站窗口；服务端不会解析 JSON，也不会做字段映射，减少所有多余开销。

响应是 `application/json`（具体字段见本文后面的接口章节；不要在没看 server 的 Resp struct 前凭空加字段）。

---

### 0.3 端到端链路：从 socket 到 L1/L2 决策（线程视角）

按一次请求的生命周期拆开（**关键在“在哪个线程做什么”**）：

1) **Glommio accept & read**（Glommio executor 线程 / per-core）
- 监听 socket（通常 `SO_REUSEPORT` 多监听器 + 多 core accept）
- 读 HTTP 头 + body（body 就是 f32le bytes）
- 进行轻量 parse、路由（命中 `/score_dense_f32_bin`）

2) **背压点（Backpressure）**
- 入口会检查 inflight / 队列容量
- 过载时：要么 early reject（返回 429/自定义码），要么排队等待（取决于配置）
- 目标：**把“排队导致的尾延迟炸裂”挡在门口**，而不是让系统内部堆积成雪崩

3) **特征 bytes → 推理输入（极致路径）**
- 最热路径是：`Bytes` → `&[f32]`（能对齐就 0-copy）→ tl2cgen Entry buffer  
- 我们做过的 micro 优化（对齐 fast-path + 两段式写 Entry + 冷路径 NaN 修补）就是在这一段砍出来的

4) **推理执行（两种模式）**
- **native tl2cgen（推荐/默认）**：  
  - 推理函数来自静态链接的 `libieee_l1.a / libieee_l2.a`
  - 形式上是 `extern "C"`，但本质是普通函数调用（没有动态库加载/符号解析开销）
- **xgb_ffi（历史兼容）**：  
  - 通过 FFI 调用 `libxgboost.so`
  - 需要处理 `LD_LIBRARY_PATH / OMP_NUM_THREADS` 等运行时环境
  - 目前主要用于对照、debug 或在还没导出 tl2cgen 时兜底

5) **XgbPool 调度（是否“另开线程池”）**
- 项目里推理通常**不在 Glommio executor 线程里直接算**，而是投递到 `XgbPool` 的 worker 线程：
  - 好处：Glommio 线程保持“快进快出”，避免被 CPU-heavy 的推理卡住 accept/read/write
  - 你现在的 monster 版 `XgbPool`：**per-core 本地队列 + work-stealing**，尽量做到“请求在哪个核进来，就近完成推理”，减少 cache 抖动
- 特例：如果你将来要做“纯 inline 推理”（Glommio core 自己算），可以实现一个 `InlineBackend`，但必须非常小心：会把网络 IO 与推理互相拖慢，尾延迟可能变差。

6) **L1 → L2 策略（pipeline）**
- `pipeline.rs` 决定是否走 L2（典型：L1 低分直接 pass，高分/可疑进入 L2 精判）
- 最后合成最终 `Decision`（PASS/REVIEW/DENY）并回写 HTTP 响应

---

### 0.4 模型目录结构与“schema 对齐”铁律

**任何模型目录（L1/L2）都必须包含 feature schema**，否则服务端会拒绝启动：

- `feature_names.json` 或 `features.txt`（必须有其一）
- 类别特征映射（如果启用了 cat 特征）：`cat_maps.json.gz`  
  - 注意：Rust 侧解析期待特定类型（例如 u32），训练脚本必须输出对齐类型，否则会出现  
    `invalid type: floating point '1.0', expected u32`

> 这条铁律的理由很简单：XGBoost/TL2CGEN 对特征顺序非常敏感，错一位就是“silent failure”。  
> **宁可启动失败，也不要默默跑错。**

---

### 0.5 “我们是不是做了 int8 量化？”——关于 TL2CGEN 的 quantize

你在 `crates/risk-core/native/tl2cgen/ieee_l*/quantize.c` 里看到的 `quantize(...)`，本质是：

- **把输入 float 按 feature 的阈值/分桶规则量化成离散 bin**（例如 `qvalue: int`），便于模型用更紧凑的跳转逻辑走树
- 这并不等价于“权重量化成 int8”那种 NN/PTQ 的概念  
  - 输入量化的结果通常是 int32/uint32 级别的 bin id（具体取决于生成代码）
  - 目标是更好的 cache/分支预测/指令局部性

一句话：**这是 tree-inference 的“结构化加速”，不是把模型变成 int8 权重。**

---

### 0.6 编译开关（feature flags）与运行时开关（env/CLI）

**编译期开关（Cargo features）**
- `native_l1_tl2cgen`：启用 L1 静态模型
- `native_l2_tl2cgen`：启用 L2 静态模型
- `xgb_ffi`：启用动态库后端（历史兼容）

> 注意：feature 开关在 `risk-core` 里定义，server crate 需要把 feature **forward** 到 `risk-core`，否则会出现 “server 没有这个 feature” 的编译错误。

**运行时开关（典型）**
- CLI：`--model-dir <L1>`、`--model-l2-dir <L2>`、`--max-in-flight ...` 等
- env（线程池/调度）：`XGB_L1_POOL_THREADS` / `XGB_L2_POOL_THREADS`、`*_PIN_CPUS`、`*_QUEUE_CAP`、`*_EARLY_REJECT_US`
- env（内存/分配器）：`MALLOC_ARENA_MAX=2`
- `OMP_NUM_THREADS`：**仅对 xgb_ffi / OpenMP 场景有意义**；当你使用纯 TL2CGEN 静态推理时，通常可以不设（设了也不会影响推理线程数）。

---

### 0.7 你想改代码但不想踩雷：三个“必须先确认”的事实

1) **请求 payload 是 dense f32le 单行**（不是 JSON、不是多行 batch）。  
2) **L1/L2 的特征顺序必须严格对齐 schema 文件**（不要“看起来差不多就行”）。  
3) **Glommio 线程 ≠ 推理线程**（默认推理走 `XgbPool` worker；要改调度必须明确这条边界）。

这三条如果弄错，后面所有优化/调参都会在错误地基上“越练越偏”。

---

## 1. 项目目标

### 1.1 业务目标
- 构建一个 **低延迟、高吞吐** 的风控推理服务（HTTP），支持 **L1 Gatekeeper + L2 Judge** 两级模型
- 约束：WSL2 环境下训练内存约 **6GB**（超过会被系统干掉）

### 1.2 性能目标（本地单机）
- L1-only：目标 ≥ 30k RPS，p99 < 2ms（timeout=50ms 下无明显 drop）
- L1+L2：L2 只在必要时触发，整体吞吐尽量接近 L1-only
- 基准工具：`risk-bench2` 固定 rps 扫描，找 knee（从 30k 到 40k 以上）

---

## 2. 现状（截至 2026-01-06）

### 2.1 最关键的架构变更
1. **推理后端从 xgb_ffi（动态 libxgboost）切换到 TL2CGEN（静态编译进二进制）**
   - 彻底消灭 `libxgboost.so` / `libgomp` 动态加载与 OpenMP 线程干扰
2. **XgbPool 调度升级为 per-core 本地队列 + work-stealing（B 方案）**
   - 大幅减少跨核 contention（尤其在 38k~40k 区间很敏感）
3. **dense bytes → Entry 转换的微观优化**
   - 对齐 fast-path（`align_to::<f32>()` 尽可能零拷贝）
   - 两段式填充（先连续写 fvalue，再补 NaN→missing）
   - 缺失值冷路径化（更友好的分支预测）

### 2.2 结果（你实际跑出来的数字）
（以你贴出的 knee 扫描为准）

- 30k：稳定 PASS（p50≈0.48ms, p99≈1.0ms）
- 38k：稳定 PASS（p99≈1.5~1.8ms 量级，偶发少量 timeout）
- 40k：已能“扛住”并接近满吞吐（偶发 timeout / missed_ticks，属于 bench-limited）

> 解释：在 40k 区间，很多 FAIL 不是服务崩，而是 **bench 的 pacer/timeout、连接队列、tick miss** 开始成为瓶颈（你日志里已经出现）。

---

## 3. 端到端链路（Glommio → 推理 → 返回）

### 3.1 数据面（/score_dense_f32_bin）
- 输入：HTTP body = dense f32 little-endian，维度 `dim=432`
- 解析：校验 `len == dim*4`，尽量走对齐零拷贝
- 推理：把任务放入 XgbPool（L1 或 L2）
- 输出：返回 score + 决策（PASS/REVIEW/DENY）

### 3.2 控制面（模型加载）
- L1：`--model-dir <dir>`
- L2：`--model-l2-dir <dir>`（未指定则 L2 不初始化/不触发）
- 每个模型目录必须带 schema（`feature_names.json` 或 `features.txt`）与 cat maps（`cat_maps.json.gz`）

---

## 4. TL2CGEN 静态推理：为什么这么快？

### 4.1 从“调用动态库”变成“普通函数调用”
- `xgb_ffi`：每次推理要经过 FFI + 动态库符号 + 可能的 OpenMP runtime
- TL2CGEN：编译期把 C 代码塞进 `.text`，运行时就是一次 `call`（非常接近手写 C）

### 4.2 分支预测与 cache 友好
你之前提到的“无分支预测失败”是核心点：
- if-else 深链：CPU 很难预测（尤其树多时）
- TL2CGEN：把树展平、量化、表驱动，尽量减少不可预测分支

### 4.3 “int8 量化”到底是什么
TL2CGEN 生成的 `quantize.c` 会把 float 映射到离散 bin（整数），从而：
- 比较更简单
- 内存访问更规律
- 在树多、维度大时非常赚

---

## 5. XgbPool：为什么要独立推理 worker？

### 5.1 主要原因
- Glommio 的 executor 是 I/O 友好模型：你希望它 **永远快进快出**
- 推理是纯 CPU：如果放在 Glommio executor 上跑，容易把 accept/read/write 卡住

### 5.2 B 方案（当前）
- 每个 core 一个本地队列（push/pop 都是本核最便宜路径）
- 当本地队列空时，再去 steal 其他队列
- 配合 deadline/early-reject：在压力下更稳定，不容易让尾延迟爆炸

> 类比：每个收银台都有自己的队伍（本地队列），没人排队时才去别的收银台“抢客”（steal），而不是所有人挤在一个超级长队上（全局队列锁）。

---

## 6. L1/L2 模型策略（从“2000 棵树拉胯”到“两级体系”）

### 6.1 训练策略（IEEECIS）
- 时间切分（严禁随机切分）
- L1：更少树、更浅（你现在的“小钢炮版本”，目标高召回+快）
- HNM：用 L1 把“简单好人”踢掉，训练 L2 的 hard set
- L2：更深更准，但只在 REVIEW 场景触发

### 6.2 为何选择 L2 iter=1800
你已经观察到：
- 训练 AUC 接近完美后，valid AUC 反而下降（过拟合开始）
- AUCPR 微升并不代表整体泛化更好

结论：选 1800~2000 轮附近是更稳的折中（你已导出 iter1800）。

---

## 7. 运行与验证清单（写给未来的你）

### 7.1 确认二进制是“真静态推理”
```bash
ldd target/release/risk-server-glommio | rg -i "xgboost|gomp"
# TL2CGEN 模式应为空
```

### 7.2 确认模型目录齐全
- `feature_names.json` / `features.txt`
- `cat_maps.json.gz`（u32 修复版）
- `xgb_model*.json`

### 7.3 确认 L2 “真的被初始化”
- 启动参数带 `--model-l2-dir`
- 启动日志应打印 L2 加载（若代码里有 INFO）
- 构造触发 REVIEW 的请求，观察是否走到 L2 路径

---

## 8. 已知问题与处理

### 8.1 Glommio io_uring buffer 注册 OOM
- 多见于 WSL2 + memlock 限制
- 解决：`prlimit --memlock=unlimited`（或降低 executor/并发）

### 8.2 TL2CGEN 多模型符号冲突
- L1/L2 都生成 `main.c/quantize/predict_unit*` 等符号
- 解决：对 L2 做 **符号前缀化**（例如 `ieee_l2_*`），避免 duplicate symbol

### 8.3 `expf` / `-lm` 链接问题
- TL2CGEN 代码可能调用 `expf`
- 解决：在构建脚本里显式链接 math（或提供 shim 并确保最终链接到 `-lm`）

---

## 9. 下一步（AWS 真机阶段）

1. 在 AWS 选 2~3 种 CPU（例如：高频 x86、更多核心、不同 cache 配置）
2. 固定同一套 bench workload，测：
   - knee 点
   - p99/p999
   - CPU 利用率、cache miss、上下文切换
3. 结合拓扑做绑核策略（Glommio executor vs L1 pool vs L2 pool 的核心划分）
4. 决策：是否需要进一步把 “请求在哪个核进来就在哪个核完成推理” 做成更极致的 affinity（但要小心倒退）

---

## 10. 参考命令（记录可复现）

见 `DEV_GUIDE.md` 的 TL;DR 区：编译、运行、knee 压测命令都已整理。


---

## Router / L2 触发控制（新增：比例 Gate）

除了 rate limiter / waterline / deadline 预算外，新增 **比例 gate**（确定性采样）：

- 环境变量：`ROUTER_L2_MAX_TRIGGER_RATIO`（`0.0..=1.0`，默认 `1.0`=不限制）
- 指标：`router_l2_skipped_ratio_total`（被比例 gate 跳过的次数）

用于论文/实验时做 “L2 只吃 hard cases” 的可控触发分布（非常适合做对照实验：ratio=1.0 vs 0.25 vs 0.125）。

另外：高 RPS 下 histogram 写入有额外开销，可用 `RISK_METRICS_SAMPLE_LOG2` 采样写入 histogram（RSK1 响应 timings 不变）。

---

## 11. Policy 校准产物化：用分位点把 L2/L3 预算“锁死”

> 这一节是“论文装逼专用 + 线上工程极有用”。
> 我们把“希望 L2 触发多少”从口头目标变成 **policy.json 产物**，做到：
> - 可复现（同一份 score 分布 → 同一份阈值）
> - 可审计（policy_calibration.json 记录目标/实际/输入）
> - 可解释（分位点控制触发率 = 预算控制）

### 9.1 policy.json 在系统里的位置

- 每个 `model_dir` 都可以放一个 `policy.json`：
  - L1 的 `review_threshold`：进入 L2 的门槛（“二次检查线”）
  - L1 的 `deny_threshold`：本层直接 deny 的门槛（“当场拦截线”）
  - L2 同理（将来你加 L3：L2 的 review 就是“进入 L3 的线”）

服务端启动时会打印（你已经看到类似）：
- `L1 loaded ... thresholds: review=... deny=...`
- `L2 loaded ... thresholds: review=... deny=...`

### 9.2 为什么用分位点选阈值？

因为“触发率”本质上是概率分布上的面积。

设 score 越大越可疑：
- 想要本层 deny 大约 `deny_frac` 的样本，就取：`deny_th = Q(1 - deny_frac)`
- 想要路由到下一层 `route_frac` 的样本，就取：`review_th = Q(1 - (route_frac + deny_frac))`

这样就把 **预算控制** 变成一个数值稳定的计算：
- `P(score >= deny_th) ≈ deny_frac`
- `P(review_th <= score < deny_th) ≈ route_frac`

> 类比：你给收费站设置两道线，直接控制“拦多少 / 二查多少”。

### 9.3 工程工作流（离线脚本）

我们提供两步脚本（都在 `scripts/ml/`）：

1) **先离线打分**：dense f32le → scores f32le

```bash
python3 scripts/ml/score_xgb_dense_f32le.py \
  --model-dir models/ieee_l1l2_full/ieee_l1 \
  --dense-file data/bench/ieee_20k.f32 \
  --dim 432 \
  --out scores/l1_scores_20k.f32
```

2) **再校准阈值并写入 policy.json**（产物化）

```bash
python3 scripts/ml/calibrate_policy_quantiles.py \
  --model-dir models/ieee_l1l2_full/ieee_l1 \
  --scores-f32le scores/l1_scores_20k.f32 \
  --route-frac 0.10 \
  --deny-frac 0.001 \
  --layer L1
```

输出：
- `model_dir/policy.json`
- `model_dir/policy_calibration.json`（记录目标/实际/输入/时间戳）

### 9.4 policy.json vs runtime gate（谁负责什么）

- **policy.json**：语义层的“常态分流”，决定 L2/L3 平均要吃多少样本
- **runtime gate（ROUTER_L2_MAX_TRIGGER_RATIO / PER_SEC / waterline）**：运行时保险丝，防止分布漂移/突发把系统打爆

推荐顺序：
1) 先用 policy 校准到你希望的 L2/L3 预算
2) 再用 runtime gate 做兜底（保守一点也没坏处）
