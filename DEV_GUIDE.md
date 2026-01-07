# Dev Guide

> 最后更新：2026-01-06  
> 目标：让你在 **WSL2 / Linux** 上把 `risk-server-glommio` 跑起来、确认 **L1/L2 TL2CGEN 静态推理已启用**、然后用 `risk-bench2` 压到 knee。

---

## 0. TL;DR（最短路径）

### 0.1 编译（L1+L2 都走 TL2CGEN 静态推理）
```bash
TL2CGEN_MARCH_NATIVE=1 \
cargo build -p risk-server-glommio --release \
  --no-default-features \
  --features native_l1_tl2cgen,native_l2_tl2cgen
```

### 0.2 运行（建议绑核 + memlock）
> **TL2CGEN 模式不需要** `LD_LIBRARY_PATH` / `OMP_NUM_THREADS`（那是给 `xgb_ffi` 动态库 + OpenMP 用的）。

```bash
sudo env \
  MALLOC_ARENA_MAX=2 \
  XGB_L1_POOL_THREADS=6 \
  XGB_L1_POOL_QUEUE_CAP=512 \
  XGB_L1_POOL_EARLY_REJECT_US=2000 \
  XGB_L1_POOL_PIN_CPUS="2,4,6,8,10,12" \
  XGB_L2_POOL_THREADS=2 \
  XGB_L2_POOL_QUEUE_CAP=256 \
  XGB_L2_POOL_EARLY_REJECT_US=4000 \
  XGB_L2_POOL_PIN_CPUS="14,16" \
  prlimit --memlock=unlimited -- \
  taskset -c 2,4,6,8,10,12,14,16,18,20,22,24 \
  target/release/risk-server-glommio \
    --listen 127.0.0.1:8080 \
    --model-dir models/ieee_l1l2_full/ieee_l1 \
    --model-l2-dir models/ieee_l2_iter1800 \
    --max-in-flight 4096 \
    --warmup-iters 0
```

### 0.3 压测（knee：38k/40k）
> 你现在的 `risk-bench2` CLI **已经收敛成精简版**：它不再接受 `--inflight-total`（已由 `--concurrency` 取代；你已经踩过坑了）。  
> 以 `--help` 输出为准。

```bash
# dense 特征二进制样本文件（bench 需要它）
# --xgb-dense-file：f32 little-endian，按 row-major 平铺
taskset -c 1,3,5,7,9,11 \
target/release/risk-bench2 \
  --url http://127.0.0.1:8080/score_dense_f32_bin \
  --rps 40000 \
  --warmup 10 \
  --duration 30 \
  --threads 6 \
  --concurrency 2048 \
  --conns-per-thread 8 \
  --timeout-ms 50 \
  --pacer fixed \
  --pacer-jitter-us 0 \
  --pacer-spin-threshold-us 0 \
  --xgb-dense-file data/bench/xgb_dense_f32le.bin \
  --xgb-dense-dim 432 \
  --content-type application/octet-stream \
  --window-ms 200 \
  --window-csv results/bench_l1l2_t{t}.csv \
  --progress
```

---

## 1. 架构总览（你现在的“怪兽形态”）

### 1.1 请求链路（从 socket 到分数）
1. **Glommio** 线程（thread-per-core executor）
   - `accept()` 新连接（多 executor + `SO_REUSEPORT` 分流）
   - 读 HTTP 请求头 + body
   - body 是 **dense f32 little-endian**（维度 432）
2. 解析/校验
   - 校验 body 长度是否为 `dim * 4`
   - （可选）fast-path：`align_to::<f32>()` 零拷贝；否则拷贝到对齐 scratch
3. 推理调度
   - 把推理任务扔给 **XgbPool**（L1 pool / L2 pool）
   - pool 线程做 CPU 密集型推理，避免把 Glommio I/O 核拖死
4. 推理后处理
   - L1：给出 `score1`，按 policy 得出初步决策（PASS / REVIEW / DENY）
   - L2：只在需要时触发（通常是 L1 的 REVIEW 区间）
5. 写回 HTTP 响应
   - JSON / 二进制输出（取决于 endpoint）

> 直觉类比：Glommio 是“高速收费站闸机”，负责把车流分流并快速验票；XgbPool 是“背后的一排检票员”，专做重 CPU 的查验。你要的是 **闸机永远不堵**。

### 1.2 推理后端（TL2CGEN vs xgb_ffi）
| 后端 | 依赖 | 线程模型 | 适用 |
|---|---|---|---|
| `native_*_tl2cgen` | **静态链接**（C 代码编进二进制） | 建议走 XgbPool（隔离 CPU） | 线上/压测默认 |
| `xgb_ffi` | `libxgboost.so` 动态库 + 可能的 OpenMP | 必须谨慎限制线程数 | 兼容/对照实验 |

---

## 2. 模型目录规范（必须对齐，否则 silent failure）

### 2.1 必备文件
每个 `model_dir` 必须包含：
- `xgb_model*.json`（或你约定的模型文件名）
- `feature_names.json`（或者 `features.txt`）：**特征顺序**是生命线  
- `cat_maps.json.gz`：类别映射（注意：我们修过 `u32` 类型坑）
- （推荐）`policy.json`：阈值（L1 的 review/deny；L2 的 deny 等）

你遇到过的典型报错：
- `missing feature schema ... expected feature_names.json or features.txt`
- `parse cat_maps.json.gz ... invalid type: floating point 1.0, expected u32`

### 2.2 L1/L2 schema 对齐检查
你已经验证过：
```bash
wc -l models/ieee_l1l2_full/ieee_l1/feature_names.json
wc -l models/ieee_l2_iter1800/feature_names.json
# 两边都应一致（例如 433 行 ≈ 432 维 + JSON 结构）
```

---

## 3. TL2CGEN：静态编译到底发生了什么？

### 3.1 你现在是不是“int8 量化”？
严格说：**不是 XGBoost 的 int8 权重量化**，但 TL2CGEN 生成的 C 代码里确实有 `quantize.c` / `qvalue` 路径：

- 输入还是 `float`
- 会把 `float` 映射到某个整数 bin（quantized feature value）
- 树的分裂比较更多走整数/表驱动路径（对分支预测更友好）

你 grep 到的这些就是证据：
- `crates/risk-core/native/tl2cgen/ieee_l*/quantize.c`
- `main.c` 里 “Quantize data … data[i].qvalue = quantize(…)”

### 3.2 为什么 TL2CGEN 下不需要 `OMP_NUM_THREADS`？
因为 **没有 OpenMP 的并行 region**：TL2CGEN 的推理就是纯函数 + CPU 分支/数组访问。  
`OMP_NUM_THREADS=1` 只对 `xgb_ffi`（动态 `libxgboost.so`）那条链路有意义。

---

## 4. 资源与绑核（WSL2 尤其关键）

### 4.1 `memlock` 与 Glommio io_uring “Cannot allocate memory”
你看到过类似警告：
- `registering buffers … Skipping … code: 12 OutOfMemory`

这通常意味着：Glommio 想预注册固定 buffer，但 **RLIMIT_MEMLOCK 不够**（在 WSL2 非常常见）。  
你现在的最佳实践就是：
- `prlimit --memlock=unlimited -- <cmd>`
- 或者减少 executor 数 / 降低并发（当你只想先跑通）

> 注意：这类 warning 很多时候不会让服务挂掉，只是少了“固定 buffer 的加速档”。

### 4.2 绑核建议
- **server** 和 **bench** 尽量分开物理核（不要共用同一对 SMT 兄弟线程）
- 如果你不确定拓扑：用 `lscpu -e` 看 `CORE,CPU,SOCKET`

---

## 5. `risk-bench2`：当前 CLI 变化

你遇到过：
- `error: unexpected argument '--inflight-total' found`

说明当前 `risk-bench2` 已改成精简版（只保留必要参数）。  
文档里不要再“死记”参数：直接用：

```bash
target/release/risk-bench2 --help
```

并以 help 输出为准更新脚本。

---

## 6. 常见坑位速查（你已经踩过的都在这）

### 6.1 sudo 下找不到 cargo / rustup default
- 不要用 `sudo cargo …`（PATH 和 rustup toolchain 会炸）
- 推荐：
  - 先 `cargo build --release`
  - 然后 `sudo prlimit … taskset … target/release/<bin>`

### 6.2 `libxgboost.so: cannot open shared object file`
只在 `xgb_ffi` 动态库模式会出现。  
TL2CGEN 模式下：
```bash
ldd target/release/risk-server-glommio | rg -i "xgboost|gomp"
# 应该是空（你已经验证过）
```

---

## 7. 如何确认 L2 “真的在跑”？

仅仅看 bench 输出里的 `l2=1us` **不一定说明 L2 跑了**——更可能是“L2 stage 没触发，计时几乎为 0”。

你可以用三种方式确认（按成本从低到高）：
1. **启动日志**：启动时应该能看到 L2 模型加载（如果代码里有 INFO/WARN）
2. **构造高风险样本**：让 L1 score 落入 REVIEW 区间，触发 L2
3. **临时把 L1 policy 调成“全进 L2”**（如果你把阈值文件做成可配）

---

## 8. 附：训练脚本（Polars + Hard Negative Mining，6G 内存友好）

示例（注意参数名已经是 `--hnm-neg-th`，不是 `--l2-keep-neg-th`）：
```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
python3 scripts/train_ieee_l1_l2_polars_hardneg.py \
  --data-dir data/raw/ieee-cis \
  --out-root models \
  --valid-frac 0.20 \
  --sample 0.10 \
  --nthread 4 \
  --topk 256 \
  --max-cat-unique 50000 \
  --l1-rounds 200 --l1-max-depth 3 --l1-scale-pos-weight 50 \
  --hnm-neg-th 0.01 \
  --l2-rounds 3000 --l2-max-depth 7 --l2-scale-pos-weight 5 \
  --max-bin 256
```


### 8.3 重新炼丹：Distillation / Hard-Pos / 单调约束（建议直接用在 L1+L2）

这三个开关是为“级联（Cascade）”量身定制的：目标不是“线下 AUC 再涨 0.001”，而是 **让 L1 在固定预算下抓更多坏人**、同时减少 L2 过载。

**A) Distillation（L1 学 L2）**

让 L1 在 **不增加在线特征开销** 的前提下，学习 L2 在 hard cases 上的判别边界（尤其是那些 “L1 犹豫 / 打不准” 的样本）。

```bash
python3 scripts/train_ieee_l1_l2_polars_hardneg.py \
  ... \
  --l1-distill-alpha 0.30 \
  --l1-distill-rounds 200 \
  --l1-distill-early-stop 50
```

- `alpha` 越大，越像“模仿老师”；越小，越像“尊重真标签”。
- 这版实现是 **在 L2 mined set（pos + hard neg）上对 L1 做 fine-tune**，对内存更友好（不会再把全量 train 留在内存里）。

**B) Hard Positive Mining（抓住那些 L1 觉得像好人的 fraud）**

把 “L1 低分的正样本” 视作 hard positive，在 L2 训练时给更大权重，逼 L2 学会它们。

```bash
python3 scripts/train_ieee_l1_l2_polars_hardneg.py \
  ... \
  --hnm-hard-pos-th 0.20 \
  --hnm-hard-pos-mul 3.0
```

**C) Monotonic Constraints（安全锁，不是提分神器）**

给你 *确定单调* 的强特征加约束（例如金额/频率），避免模型学出反常识逻辑。

```bash
python3 scripts/train_ieee_l1_l2_polars_hardneg.py \
  ... \
  --l1-mono 'TransactionAmt:+1' \
  --l2-mono 'TransactionAmt:+1'
```

> 注意：不要给 hash/embedding/高基数离散特征乱加单调约束。

---

### 8.4 对抗验证（Adversarial Validation）：先排雷再炼丹

把 “train vs test（或 early vs late）” 当作标签训练一个分类器：
- AUC 接近 0.5：分布一致，比较安全。
- AUC 明显 > 0.7：强漂移/泄露风险，优先排查高重要性特征（比如 ID 类、时间戳类）。

```bash
python3 scripts/ml/adversarial_validate_ieee_polars.py \
  --data-dir data/raw/ieee-cis \
  --mode train_vs_test \
  --sample 0.20 \
  --max-cat-unique 50000 \
  --nthread 4
```

---

### 8.5 不确定性路由（Tree StdDev）离线分析

这是把 “score 中间” 和 “模型真的不确定” 区分开的一个可操作方案：  
计算每条样本 **各棵树 margin 贡献的标准差**（std 越大，树之间越打架）。

```bash
python3 scripts/ml/score_xgb_tree_std_dense_f32le.py \
  --model-dir models/ieee_l1l2_full/ieee_l1 \
  --dense-file data/bench/ieee_20k.f32 \
  --dim 432 \
  --out-std scores/l1_tree_std_20k.f32
```

然后用一个统一脚本扫 “route budget vs recall”：

```bash
python3 scripts/ml/sweep_scalar_frontier.py \
  --scalar-f32le scores/l1_tree_std_20k.f32 \
  --labels-u8 scores/l1_labels_20k.u8 \
  --desc \
  --rf 0.02 --rf 0.05 --rf 0.10 --rf 0.20
```



---

## 8. Monster 可调参数速查（L2 路由 + Metrics 采样）

### 8.1 Router / L2 控制（环境变量）
这些参数只影响 **L1 决策为 review 时** 是否尝试 L2（不会改变 L1 的 score 计算）。

- `ROUTER_L2_MAX_TRIGGERS_PER_SEC`：每秒最多触发多少次 L2（`0`=不限制）。  
- `ROUTER_L2_MAX_TRIGGER_RATIO`：按比例触发 L2（`0.0..=1.0`，默认 `1.0`=不限制）。  
  - 实现是 **确定性采样**（不会引入 RNG），并新增指标：`router_l2_skipped_ratio_total`  
- `ROUTER_L2_MAX_QUEUE_WATERLINE`：L2 队列水位阈值（`1.0`=不限制）。  
- `ROUTER_L2_MIN_REMAINING_US`：离截止时间不足该值则跳过 L2。  
- `ROUTER_L2_QUEUE_WAIT_BUDGET_US`：传给 L2 pool 的 queue-wait budget（`0`=不启用）。

### 8.2 Metrics 采样（环境变量）
高 RPS 下 `metrics::histogram!` 记录本身有原子开销；我们保持 **RSK1 响应里 timings 不变**，但允许采样写入 histogram。

- `RISK_METRICS_SAMPLE_LOG2`：  
  - `0`（默认）= 每个请求都 record  
  - `N>0` = 只 record `1/2^N` 的请求（确定性采样，低开销）


---

## 9. policy.json 离线校准（把“L2 触发率”变成可复现产物）

你可以把 **policy.json** 理解成“收费站的分流规则”：
- `review_threshold`：进入下一层（L2 / L3）的下限
- `deny_threshold`：本层直接拒绝（DENY）的下限

我们用 **分位点（quantile）** 来选阈值，把“我希望 L2 只吃 10% 的 hard cases”变成一个可复现的文件。

### 9.1 两步走：先离线打分，再选分位点

**Step A：用模型把样本打分成一个 score 分布文件（f32le）**

```bash
python3 scripts/ml/score_xgb_dense_f32le.py \
  --model-dir models/ieee_l1l2_full/ieee_l1 \
  --dense-file data/bench/ieee_20k.f32 \
  --dim 432 \
  --out scores/l1_scores_20k.f32
```

**Step B：用 score 分布选阈值，写入 `model_dir/policy.json`**

例子：目标是 **L1 → L2 触发率约 10%**，并且 **L1 本层硬拒绝 0.1%**：

```bash
python3 scripts/ml/calibrate_policy_quantiles.py \
  --model-dir models/ieee_l1l2_full/ieee_l1 \
  --scores-f32le scores/l1_scores_20k.f32 \
  --route-frac 0.10 \
  --deny-frac 0.001 \
  --layer L1
```

脚本会输出：
- `models/.../policy.json`（服务端启动会读它）
- `models/.../policy_calibration.json`（记录：目标比例、实际比例、输入文件、时间戳）

### 9.2 背后的“分位点公式”（写论文时最香的部分）

令 `score` 是模型输出概率（越大越可疑）：
- 选择 `deny_th = Q(1 - deny_frac)`
- 选择 `review_th = Q(1 - (route_frac + deny_frac))`

那么近似有：
- `P(score >= deny_th) ≈ deny_frac`
- `P(review_th <= score < deny_th) ≈ route_frac`

> 类比：你在门口贴了两条线：
> - 过了第一条线（review_th）就“叫保安二次检查”（进入下一层）
> - 过了第二条线（deny_th）就“当场拦下”（直接 deny）
> 
> 用分位点选线的位置，就等于你直接控制“要拦多少人 / 要二查多少人”。

### 9.3 与 Router 的 ratio gate 如何配合？

- `policy.json` 控制的是 **语义层的分流比例**（按 score 分布）
- `ROUTER_L2_MAX_TRIGGER_RATIO / ...PER_SEC / waterline...` 控制的是 **运行时的算力预算**（安全阀）

推荐策略：
1) 先用离线校准把 L2 触发率做“常态正确”（例如 5%~15%）
2) 再用 runtime gate 当保险丝（防止线上分布漂移把 L2 打爆）
