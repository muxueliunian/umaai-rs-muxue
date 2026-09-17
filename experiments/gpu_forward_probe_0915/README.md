# GPU 前向执行调度独立探针（2026-09-15）

只回答一个问题：**前向的执行调度是不是主要可优化部分**。
对照两臂：A = 当前三成员串行 eager 前向；B = 保持同一前向与集成表达、对整段做
CUDA Graph 捕获与 replay。

本目录**不修改**生产侧车、模型定义或 Rust 决策路径，也不做 baddbmm/vmap 重写、
不 `torch.compile`、不开 TF32、不改精度、不动成员数、不改管道。

## 复现命令（pwsh，工作目录 = workspace 根）

三步必须分进程跑：profiler 的插桩开销不能混进稳态计时。

```powershell
$CK = @("--checkpoint","target/arm_G2mix_seed1/step_030000.pt",
        "--checkpoint","target/arm_G2mix_seed2/step_030000.pt",
        "--checkpoint","target/arm_G2mix_seed3/step_030000.pt")
$DEC = @("--dec-csv","logs/perf_opt1_0915/chk_R1_base_dec.csv",
         "--dec-csv","logs/perf_opt1_0915/chk_R2_base_dec.csv")
$P = "experiments/gpu_forward_probe_0915/probe.py"

python $P @CK @DEC profile  --batch 512 --steps 5 --out-dir logs/gpu_probe_0915
python $P unit                                             # 口径合成回归，不需要 GPU
python $P @CK @DEC selftest --batch 512 --per-stage 60      # 再验收「验收器」
python $P @CK @DEC verify   --batch 512 --batch 1024 --per-stage 300
python $P @CK @DEC time     --batch 512 --batch 1024 --iters 50 --reps 5 --warmup 20
```

`selftest` 逐个注入错误（翻位、NaN、Inf、只脏补零尾段、翻转动作、只改吃面格位、
重放漂移），要求验收器**每一种都拦得住**。日志全绿只说明它没在报错，不说明它报得出来。

`time` 至少独立跑 3 次（本轮 rep1/rep2/rep3），本机整局墙钟自身离散已知偏大。

❗**日志不要写同名文件**：本轮的作废运行就是被同名重跑覆盖掉的。写之前先
`if (Test-Path $L) { throw }`，重跑一律用新编号。

## 输入来自哪里

754 维特征直接取自既有决策 CSV 的 `features` 列。该列是 9 位有效数字的十进制，
足以精确还原 f32，所以解析回来的就是当时实际编码的位模式。全程**只比较实际字段
与字节，不计算任何哈希或指纹**。

候选与阶段取自同一行的 `n_actions` / `actions` / `chosen`，覆盖全部五个决策阶段。

## 选分口径：分两类判据

`action_slots` / `argmax_logit` 逐条对应 `policy_schema.rs` 的格位布局与
`ramen_nn_trainer.rs` 的 `score_one` / `argmax_logit`（含并列时取较小候选下标）。
不用全局 234 维 argmax 代替。

❗分值一律在 **`np.float32`** 上产生：`RegionSelect` 的三格求和按 `ids` 的书写顺序
**f32 逐项累加**，与 `score_one` 的 `let mut sum = 0.0f32; sum += ...` 同序同精度。
用 Python float（f64）累加会换掉舍入点，极窄间隔或并列时可能给出与生产不同的 argmax。

❗`argmax_logit` 按 **`f32::total_cmp`** 的全序比较（`total_order_key`），不是 Python 的 `>`：
`-0.0 < +0.0`、`-NaN` 小于一切、`+NaN` 大于一切；并列仍取较小候选下标。
这三种值本轮样本里没出现过，但判据得先与生产一致。

这两条口径由 `unit` 子命令的合成用例守着（不需要 GPU，也不读 checkpoint）。

但**不是所有行都能严格复现生产口径**：

| 行类型 | 能否严格复现 | 判据 |
|---|---|---|
| Train / SuperRamenSelect / SpecialSelect / RegionSelect / 不吃面的 RamenSelect | 能 | 直接比生产 argmax |
| **选了某碗面的 RamenSelect** | **不能** | 相关格位**超集**逐位相同 |

❗后一类的生产口径是对 `rules::list_special_targets_for` 给出的合法用法集合 T 取 max，
而 T 依赖 `feeling_stock` / `special_feeling`，决策 CSV 没有记录、本探针无法复原。
`score_actions` 对这类行返回 `strict=False`，其 argmax 是**上界口径**，只作参考。
判定只走「相关格位超集逐位相同」——那样无论 T 是哪个子集，生产选出的动作都相同。
一旦出现位差，这一行按**无法判定**处理并计入未通过，**不会**因为「动作恰好一样」而放行。

## 日志

`logs/gpu_probe_0915/`：

| 文件 | 内容 |
|---|---|
| `01_profile_eager_B512.log` | profiler 能力、CPU 算子构成、图节点数、提交/完成分离 |
| `02_verify_graph.log` | 两个档位的 Graph 正确性与生产口径动作对比 |
| `03_timing_rep{1,2,3}.log` | 三次独立稳态交错计时 |
| `04_unit_tests_after_import_fix.log` | 导入清理后的针对性 Rust 测试 |
| `05_selftest_injection.log` | 验收器的错误注入自检 |
| `06_unit_tests_graph_switch.log` | 接入 `--sidecar-graph` 后的针对性 Rust 测试 |
| `07_smalln_graph_arms.log` | 小 n 两臂运行（缓存开 + eager / 缓存开 + Graph）|
| `08_smalln_compare.log` | 两臂 CSV 的逐字段比较 |
| `09_bench_n256_graph.log` | 三根 n=256 交错计时（空闲重跑）❗第一次被并发游戏污染的运行**被同名覆盖**，已不可复核 |
| `10_selftest_injection_f32.log` | f32 打分口径修正后重跑的注入自检，7/7 |
| `11_verify_graph_f32.log` | f32 打分口径修正后重跑的 Graph 正确性，49 项 0 未通过 |
| `12_game_pair_graph.log` | 一对整局（缓存+eager / 缓存+Graph）及逐行比较 |
| `game_{eager,graph}_steps.csv` | 两臂各 183 步逐步决策，含 `rng_probe` |
| `13_unit_scoring.log` | 合成回归 6 项 0 未通过（无 GPU）|
| `14_verify_graph_totalcmp.log` | `total_cmp` 修正后重跑的 Graph 正确性，49 项 0 未通过 |
| `15_selftest_totalcmp.log` | `total_cmp` 修正后重跑的注入自检，仍 7/7 |
| `trace_eager_B512.json` | chrome trace 原文 |
| `graph_nodes_B512.dot` | CUDA Graph 节点图原文 |

## 环境限制（如实记录，未绕过）

`torch._C._autograd._supported_activities()` 在本机只返回 `{CPU}`：
**CUPTI 不可用**，`ProfilerActivity.CUDA` 被静默忽略，设备侧 kernel 时间线采不到。
因此 `key_averages()` 的 CUDA 列是由 CPU 事件推导的，本轮**不采用它做任何归因**。
按约束未安装任何包、未改运行环境，改用两条不依赖 CUPTI 的证据：

1. CUDA Graph 节点图 → 整段前向的实际 kernel 节点数（设备侧事实）；
2. 提交 / 完成分离计时 → 判断 CPU 派发与 GPU 执行谁是节拍器。

## 侧车接线（本轮新增，默认关闭）

`scripts/ramen_nn/bench_sidecar.py` 加了 `--cuda-graph`；
`ramen_root_bench` 加了 `--sidecar-graph`。三道防线保证开关不会被静默忽略或误用：

1. `check_switch_support` 在建局、模型加载与侧车启动**之前**拒绝不消费该开关的模式，
   并拒绝 `--sidecar-graph` 与 `--adaptive-batch` 同时给出（图按固定行数捕获，换档要重捕）。
2. 侧车就绪行新增 `graph=on|off`；Rust 侧要求自报为 `on`，否则报错——
   只看「我们传了参数」不算数。
3. 侧车在开图时**拒绝** `OP_SET_BATCH` 控制消息，而不是沿用旧图算错行数。

隔离口径：固定根对照两臂**都开 `--policy-cache`、都固定 `--batch 512`、都不开自适应批**，
唯一差别是 `--sidecar-graph`。
