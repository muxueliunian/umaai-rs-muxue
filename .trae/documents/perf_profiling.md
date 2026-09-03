# 性能分析指南（flamegraph / pprof-rs 选择 + 复现）

> 记录"对手写逻辑 / MCTS 性能分析"系列试验的工具选择准则、最终结论与复现命令，便于以后重测与演进。

## 1. 背景

项目当前三类 CPU 性能 / 延迟分析工具（覆盖全栈 vs 按段 vs 全局三层视角）：

1. **手写策略批量火焰图**——`RecommendedRamenTrainer`，单线程跑批便于横向对比；走 `cargo flamegraph`（见附录 C）
2. **MCTS / 手写策略单局性能剖面**——`RamenMctsTrainer`，多线程 + 闭包 + inline 优化导致传统 perf 栈展开质量差；走 `pprof-rs` 用户态采样（`mcts_profiler` / `sim_profiler`，见附录 B）
3. **calc 链路按段拆解的 microbench**——不依赖栈展开，按数据流依赖链分桶测 ns/op（`calc_training_value_microbench`，见 §7）

工具选择与命令各不相同。本文给出明确决策准则，避免下次重复踩坑。

## 2. 工具选择：先选对，再开跑

**经验法则**：

| 场景 | 工具 | 输出 |
|---|---|---|
| 单线程 / 闭包少 / inline 温和（手写策略批量 / 业务逻辑全栈） | `cargo flamegraph` | SVG 火焰图 |
| 多线程 rayon / 闭包深度嵌套 / opt-level 任意档（MCTS / 搜索调度 / 高频 hot path 全栈） | `pprof-rs` 用户态采样（`mcts_profiler` / `sim_profiler`） | .pb protobuf → `go tool pprof` |
| 单函数 / 按段 ns/op + 占比 + 逐段优化回归（**calc / policy 链路定点打击**） | `calc_training_value_microbench` | stdout 表格（见 §7）|

为什么这样分？`cargo flamegraph` 走 perf + inferno，栈展开依赖 dwarf unwind tables。多线程场景下 perf.data 容易膨胀到几 GB + 大量子子 samples lost；MCTS 闭包密集，inline 后看到的都是 `closure_env#0` / `impl#N` / `[unknown]` 这样的代号，看不到真正的 hot path 函数名。

`pprof-rs` 用户态 backtrace + symbolization 不依赖 perf 工具链，闭包/`impl#N`/`{closure#0}` 全部能解析到原始函数名。多线程友好，产物 KB-MB 量级不会膨胀，采样率可调（默认 1kHz），输出 Google pprof protobuf 给 `go tool pprof` 或 `inferno-flamegraph` 解读。

`cargo flamegraph --dev` 看似退而求其次的选择，但在 Rust 项目下基本不可用——Rust dev profile 默认没有 unwind tables，栈帧全部归 `[unknown]`，试验实测有 80% 样本是 `[unknown]`。

**所以**：

- 看到手写策略 / 业务逻辑全栈 → `cargo flamegraph`
- 看到 MCTS / 搜索 / 任何"inline 重灾区"全栈 → 直接 `pprof-rs`，不要在 `cargo flamegraph` 上浪费时间
- 优化单函数 / 验证 calc 链路某段是否变小 → `calc_training_value_microbench`（pprof self time 给"占比"但给不出"绝对 ns/op"和"占比依赖关系"；microbench 补这块缺口）

## 3. 性能分析结论（当前方法论）

本节基于 microbench 多轮均值（§7 / 附录 B）替换 2026-09-01 的 pprof-flamagraph 基线——后者单局 RUNS=1 抖动盖住信号，已整体弃用（旧数据删于附录 B/C）。

### 3.1 关键发现：手写策略本体是 MCTS 的真正热点

当前测量手段（详见 §7 与附录 B）：

| 测量对象 | 工具 | N | 统计稳定性 |
|---|---|---|---|
| **单函数 NS/call** | d10872a microbench test | 100k × 3 round | std/mean ≤ 2.5% |
| **7 段整 Round NS/iter** | `calc_training_value_microbench` bin | 1000 × 3 round | std/mean ≤ 13%（B 段 warm-up 漂移） |

附录 B 显示手写策略 hot path 5 个核心函数全部 **NS/call 下降 50-90%**（vs d10872a 附录 C 同口径基线）：

| 函数 | d10872a 基线 | cleanup 后多轮均 | Δ |
|---|---:|---:|---:|
| `SupportCard::calc_training_effect` | ~102 | 13.77 | **-87%** |
| `default_calc_training_buff` | ~73 | 12.47 | **-83%** |
| `calc_training_value` | ~104 | 33.30 | **-68%** |
| `LocalRamenTrainer::select_action` | ~59 | 24.57 | **-59%** |
| `reserve_penalty` | ~8 | 3.93 | **-51%** |

加上 §7 段 F 实测（整回合打分 4177 ns/iter，冷 run + 稳态），与"手写策略本体是 MCTS 性能最大杠杆点"的历史结论一致——只是基线数字已是 cleanup 后新基线。

cleanup 三连改动的具体内容（变更轨迹）：

1. **`SupportCard::calc_training_effect` 简化签名 + 起点改基础面板**（最大单点改动）
2. **deyilv 路径去掉 `eff.clone()`**（owned 链一致性）
3. **`Game::deyilv` trait `Result<f32>` → `f32`**（减少分支预测 + Result 链一致性）

### 3.2 已淘汰结论

§3.x 的旧"次要发现"与 §3.3 的旧"优化优先级"基于 `cargo flamegraph` + 单局 pprof，单 RUNS=1 抖动无法给出对照结论，已整体弃用——其在 noise level 之上**反复跳动**，本轮三改动后 pprof 单局抽样未给出一致下降方向（SupportCard::calc_training_effect、calc_training_value 微降 5%；dynamic_status_adjustment 降 45% / score_train_action 涨 92% 同时出现，明显非代码因素）。

替代方法见 §7 / 附录 B 的 multi-run microbench 实测统计显著（std ≤ 2.5%）。未来 hot path 优化以 §3.1 / 附录 B 数字为锚定标准。

## 4. 试验历史：为什么 cargo flamegraph 对 MCTS 不可用

多次试验得到的明确结论：MCTS 这种多线程 + 闭包 + inline 重灾区，cargo flamegraph 在 opt-level 任意档、profile 任意档下都难以给出有用信息。要走 microbench × N（统计显著）或当前基线的 microbench 多轮均值。

详见文末附录 A。

## 5. 工具固化：三个 bin 互相补充

### 5.1 路径与依赖

`crates/umasim/tools/data_collection/` 下三个 bin 工具，统一约定 `cargo run --release --bin <name>`（无 required-features，由各自 cfg gate 决定可选编译）：

| Bin | 文件路径 | 必要性 | 何时用 |
|---|---|---|---|
| `sim_profiler` | `sim_profiler.rs` | `--features profiler`（Windows 不编译 pprof-rs） | 手写策略全栈 pprof（附录 C 同源，但 pprof 更准）|
| `mcts_profiler` | `mcts_profiler.rs` | `--features profiler` | MCTS 单局全栈 pprof（附录 B 同源）|
| `calc_training_value_microbench` | `calc_training_value_microbench.rs` | 无 | **calc / policy 链路按段 ns/op 拆解（§7，按段优化回归用）** |

`sim_profiler` 与 `mcts_profiler` 在源文件顶部带 `#![cfg(feature = "profiler")]`——cfg gate 在编译时跳过，与 `required-features` 等价；`calc_training_value_microbench` 不依赖 pprof-rs，所有平台默认可跑。

### 5.2 三个 bin 对比

| 维度 | sim_profiler | mcts_profiler | calc_training_value_microbench |
|---|---|---|---|
| 训练员 | `RecommendedRamenTrainer` | `RamenMctsTrainer` | （不调训练员，直接调 game calc / policy 层）|
| 测量视角 | 整局全栈 self time | 整局全栈 self time | **按段 ns/op**（7 段分桶）|
| 输出 | .pb protobuf → `go tool pprof` | .pb protobuf → `go tool pprof` | **stdout 表格** |
| rayon 线程数 | 全局默认（由 `game_config.toml` 的 `num_threads` 控制） | `MCTS_PROFILER_NUM_THREADS` 强制（默认 1） | 单线程（无 rayon） |
| SearchConfig | 无 | `search_n` / `stages` / `selection` / `ucb` / `radical_factor_max` 全套 | 无 |
| env vars | `SIM_PROFILER_RUNS` / `LABEL` / `FREQ` | `MCTS_PROFILER_RUNS` / `LABEL` / `FREQ` / `SEARCH_N` / `STAGES` / `NUM_THREADS` | `CT_MICROBENCH_RUNS` / `CT_MICROBENCH_WARMUP` |
| 输出文件 | `logs/profile/<label>.pb` | `logs/profile/<label>.pb` | （stdout） |
| 复现章节 | §6.1 + 附录 C | §6.2 + 附录 B | **§7 + 附录 E** |

### 5.3 pprof-rs 产物解读

```bash
# Top 函数（按 self time 排序）
go tool pprof -top logs/profile/mcts.pb

# 调用树
go tool pprof -tree logs/profile/mcts.pb

# 火焰图 SVG（需要 inferno-flamegraph）
inferno-flamegraph logs/profile/mcts.pb > logs/profile/mcts_flame.svg
```

pprof-rs 0.15 栈方向：`frames[0]=leaf`，`frames.last()=root`；在每个 `Frame` 内 `symbols[0]=leaf`（最近函数），最后 = caller。噪音帧（`backtrace::*` / `pprof::*` / `signal_handler`）需在 self time 聚合时跳过——`mcts_profiler.rs` 已实现 `is_noise` 函数处理。

## 6. 复现 checklist

### 6.1 重测手写策略 cargo flamegraph

```bash
# 0. 一次性 perf 权限（Ubuntu 默认 paranoid=4 阻断用户态 CPU event 采集）
sudo sysctl -w kernel.perf_event_paranoid=2

# 1. 备份 bench_config.toml
cp umaai-rs/bench_config.toml umaai-rs/logs/bench_config.toml.bak.flamegraph.$(date +%Y%m%d_%H%M%S)

# 2. 改 bench_config.toml：runs=100, trainer="handwritten", 仅留 [player_builds.speed]
# 3. 跑
cd umaai-rs
cargo flamegraph --release --no-default-features --bin bench_base -- --trainer handwritten

# 4. 恢复 bench_config.toml
cp logs/bench_config.toml.bak.flamegraph.<时间戳> bench_config.toml

# 产物：umaai-rs/flamegraph.svg、umaai-rs/logs/bench_base_results.csv
```

### 6.2 重测 MCTS pprof-rs

```bash
# 1. 编译
cd umaai-rs
cargo build --release --features profiler --bin mcts_profiler

# 2. 跑单局
MCTS_PROFILER_RUNS=1 MCTS_PROFILER_SEARCH_N=64 \
MCTS_PROFILER_STAGES="train,ramen,special" \
MCTS_PROFILER_NUM_THREADS=1 \
MCTS_PROFILER_LABEL=mcts \
./target/release/mcts_profiler

# 3. 解读
go tool pprof -top logs/profile/mcts.pb
inferno-flamegraph logs/profile/mcts.pb > logs/profile/mcts_flame.svg

# 产物：logs/profile/mcts.pb、logs/profile/mcts_<label>_stdout.log
```

### 6.3 清理约定

- `perf.data`（cargo flamegraph 中间产物）：**数 GB**，跑完即删
- `logs/profile/<label>.pb`：保留作为基线对比
- `logs/<...>.bak.flamegraph.<时间戳>`：保留作为回滚证据

### 6.4 重测 calc_training_value_microbench

```bash
# 默认 1000 iter / 段，warmup 1000
cd umaai-rs
cargo run --release --bin calc_training_value_microbench

# 小用例（验证可行性，~1.5 ms 墙钟）
CT_MICROBENCH_RUNS=100 CT_MICROBENCH_WARMUP=100 \
  cargo run --release --bin calc_training_value_microbench
```

结果以 stdout 表格输出（§7.2 格式），无中间产物文件。env vars 见附录 E。

---

## 7. 按段拆解的最坏路径基线（calc_training_value_microbench）

不替代附录 B/C 的全栈基线，而是按数据流依赖链分桶测 calc / policy 链路各段的 ns/op——本工具能给出 pprof self time 给不出的"绝对 ns"和"占比依赖关系"。

### 7.1 7 段设计（vs d10872a microbench）

| 维度 | d10872a microbench | 当前 microbench |
|---|---|---|
| 卡组 | speed build + 推 turn=30 | **speed build + friendship 全 100 + turn=30 + 拉面 buff 全开** |
| 单 / 5 train | 单 train=0 一次 | **5 train 一回合循环**（对齐 `LocalRamenTrainer::score_train_action`）|
| 拉面 buff | 自然 turn=30 state（可能不吃面） | **current_ramen=Some(5) + selected_regions=[5,7,9]，中山-全 region 命中全部 5 train** |
| 段数 | 6（reserve_penalty/calc_buff/calc_value/effect/clone/select_action） | **7**（A.distribute_all / B.calc_buff ×5 / C.calc_value ×5 / D.端到端 / E.score_train_action / F.decide_train 整回合 / G.calc_ramen_training_effect）|
| 私有方法可见性 | 同 crate 内联测试 | **`RamenPolicy::score_train_action` / `status_gain`，`LocalRamenTrainer::decide_train` / `dynamic_status_adjustment` / `reserve_penalty` 提到 pub**（产品路径不变）|

7 段各自测的对象与 pprof 对应：

| 段 | 函数 | 包含子操作 | pprof 对应 |
|---|---|---|---|
| **A** | `distribute_all` | reset + iterate persons + 多次 `distribute_person`（absent 判定 + 零分配分桶采样 + retry）| 内联未单独报 |
| **B** | 5 train × `default_calc_training_buff` | 遍历 dist[t] + `SupportCard::calc_training_effect` + `CardTrainingEffect::add` | 3.28% + 1.83% + 1.10% |
| **C** | 5 train × `calc_training_value` | `default_calc_training_value` 下层 + `calc_ramen_training_effect` + 上下层 clamp | 2.58% + 3.16% |
| **D** | 端到端一回合 = A + B + C | 完整 calc 链路（无 policy 层）| — |
| **E** | `RamenPolicy::score_train_action` ×1 | B + C + `calc_training_failure_rate` + 5 × `status_gain` + score 拆解 | 1.61% |
| **F** | `LocalRamenTrainer::decide_train`（7 candidates）| score_train_actions + 修复路径 + choose + phase + reserve_penalty/dynamic_status_adjustment 调整 | 0.54% + 2.54% + 1.68% |
| **G** | `calc_ramen_training_effect` ×1 | 拉面 buff 累乘（calc_normal_effect / calc_finals_effect） | 内联于 C 2.58% |

### 7.2 新基线（2026-09-02 采样替换后，可比口径 Run 2/3 均值，mean ns/iter）

```
段                              mean ns/iter  per-train  占 D
A.distribute_all                     633.9        —       52.4%
B.calc_training_buff ×5               257.2       51.4     21.3%
C.calc_training_value ×5              265.1       53.0     21.9%
D.端到端一回合(5train)                1209.6        —      100%
E.score_train_action x1              446.3        —       —
F.decide_train(7 候选)              4237.8        —       —
G.calc_ramen_training_effect x1       9.6        —       —
```

D − (A+B+C) ≈ D − 1156.2 = 53.4 ns（占 D 4.4%）——分桶测 cache 热、组合测 cache miss 边界，正常。

> 注：口径 = 第三次重测 Run 2/3 均值（cold Run 1 剔除，原始值见附录 B.3）。对照段（B/C/F/G，本次未动代码）回落上午 cleanup 基线 ±2%，判定测量环境可比；可比口径下段 A 800.97 → 633.9 ≈ **-21%**（采样替换端到端收益，与 §7.3 进程内对照交叉验证吻合）。同日另有两次漂移样本（798.7 / 907.0），见 B.3 注③。

### 7.3 关键观察（清理后多轮均值）

| 观察 | 数字 | 说明 |
|---|---|---|
| A 占比 52.4% | A/D = 633.9/1209.6 | distribute_all 仍是最大单一杠杆（与 §3.1 一致） |
| **A 段采样替换已落地** | WeightedIndex → 零分配分桶采样（`sample_bucket`）| 与 rand 整数 WeightedIndex 逐位等价（`traits.rs` 守门测试 7 权重组合 × 5 万次全一致 + 全量测试数值不变）；进程内对照每次采样 **-31~-42%（7-11 ns/call）**，可比口径整段 **-21%** |
| F ≈ 4.24 μs / 7 candidates | 4238 ns/iter | 整回合策略决策成本；与 1 局手写策略 mean 2.36 ms 的 ~77 个决策回合同数量级 |
| G = 9.6 ns | 拉面 buff 累乘路径已轻 | 内联到上层不进 Top |
| D − (A+B+C) ≈ 53.4 ns | 边界 4.4% | cache miss / cache 边界，正常 |

> 注：cleanup 后 cross-validate 见附录 B.2——3 个 hot path 函数 d10872a microbench 单测全部下降 50-90%（统计显著）。

优化优先级合入 §3.1，不在重复。

---

## 附录 A：试验历史（MCTS cargo flamegraph 失败记录）

| 版本 | 配置 | 结果 |
|---|---|---|
| v1 | release profile (opt-level='z'), num_threads=1, search_n=32, stages=train,ramen,special | 火焰图 Top 全是 init 阶段（gamedata JSON 加载 + BTreeMap insert 23.90%），MCTS 主循环函数全部被压成 `closure_env#0` / `impl#N`；看不到 `search_uniform` / `simulate_many` / `simulate_to_terminal` 实际开销 |
| v2 | release profile + 临时改 `Cargo.toml` opt-level=3, num_threads=1, search_n=256 | opt-level=3 让函数边界稍微清晰，`search_uniform` 7.41% 出现，但手写策略 hot path 函数（`default_calc_training_value` / `calc_training_value` 等）仍被 inline 吃掉；`RamenAction` 16.11% / `ActionResult` 11.10% 这些"类型相关帧"占据了真正 hot path 的位置 |
| v3 | **debug profile**（`cargo flamegraph --dev`）, num_threads=1, search_n=128 | **失败**——Rust dev profile 默认没有 unwind tables，79.99% 样本 `[unknown]`；perf.data 38.75 GB（debug 模式 binary 体积大）+ 9.13% samples lost |
| v4（成功）| pprof-rs（`mcts_profiler`）, num_threads=1, search_n=64 | 完美——手写策略 hot path 全部浮出水面：`default_calc_training_value` 3.16% / `default_calc_training_buff` 3.28% / `dynamic_status_adjustment` 2.54% / `reserve_penalty` 1.68% 等清晰可见 |
| v5（当前）| microbench × 多轮 mean (d10872a + calc_training_value_microbench) | **替代 v4 的 pprof 单局数据**：单 RUNS=1 pprof 抖动盖住真实信号，已弃用。本节值只保留 pprof 历史样例（不能作 cleanup 后对照基线）。§3.1 / 附录 B 为当前对照基准 |

## 附录 B：d10872a microbench 多轮均值基线（2026-09-02 cleanup 后）

测量方法：

```bash
cargo test --release -p umasim --lib \
  trainer::local_ramen_trainer::tests::microbench_top_fns \
  -- --ignored --nocapture
```

- 单函数 100,000 iter × 3 round = 300,000 sample
- round-min + round-mean 抓取
- 取 3 次完整运行的总 mean 进一步平均

### B.1 实测基线（3 次外部运行）

| 函数 | Run 1 | Run 2 | Run 3 | **总 mean** |
|---|---:|---:|---:|---:|
| `reserve_penalty` | 3.9 | 3.9 | 4.0 | **3.93** |
| `default_calc_training_buff` | 12.2 | 12.8 | 12.4 | **12.47** |
| `calc_training_value` | 32.9 | 33.5 | 33.5 | **33.30** |
| `SupportCard::calc_training_effect` | 13.6 | 13.8 | 13.9 | **13.77** |
| `CardTrainingEffect::clone` | 2.3 | 2.3 | 2.3 | **2.30** |
| `LocalRamenTrainer::select_action` | 24.5 | 24.5 | 24.7 | **24.57** |

std/mean ≤ 2.5%（最高 `default_calc_training_buff` 的 2.5%，其余 ≤ 1.5%）。统计显著，**可直接作 cleanup 后对照基线**。

### B.2 vs d10872a 原 commit（commit d10872a 测得，未 cleanup）

| 函数 | d10872a 基线 | B.1 总 mean | Δ | 备注 |
|---|---:|---:|---:|---|
| `reserve_penalty` | 7.7 | 3.93 | **-49%** | early-return 路径 + cache-friendly |
| `default_calc_training_buff` | 72.5 | 12.47 | **-83%** | cleanup 最大单点收益 |
| `calc_training_value` | 104.3 | 33.30 | **-68%** | |
| `SupportCard::calc_training_effect` | 101.7 | 13.77 | **-86%** | 起点改基础面板（不变叠加） |
| `CardTrainingEffect::clone` | 29.8 | 2.30 | **-92%** | |
| `LocalRamenTrainer::select_action` | 59.4 | 24.57 | **-59%** | 下游 calc_buff 受益传递 |

注意：d10872a commit 上的数字是单次跑（与 B.1 三次平均不可严格比对），但数量级差距足够大（-49%~-92%）说明 cleanup 是真实的优化，不是测量噪声。

### B.3 calc_training_value_microbench × 3（7 段 ns/iter，2026-09-02 第三次重测、可比口径）

测量方法：

```bash
cargo run --release --bin calc_training_value_microbench
```

| 段 | Run 1¹ | Run 2 | Run 3 | **总均²** | spread |
|---|---:|---:|---:|---:|---:|
| A. distribute_all | 797.5 | 631.7 | 636.0 | **633.9** | 0.7% |
| B. calc_training_buff ×5 | 530.6 | 255.6 | 258.7 | **257.2** | 1.2% |
| C. calc_training_value ×5 | 310.7 | 265.8 | 264.3 | **265.1** | 0.6% |
| D. 端到端一回合(5train) | 1814.5 | 1210.4 | 1208.8 | **1209.6** | 0.1% |
| E. score_train_action ×1 | 703.0 | 443.7 | 448.8 | **446.3** | 1.1% |
| F. decide_train(整回合) | 4169.1 | 4216.9 | 4258.6 | **4237.8** | 1.0% |
| G. calc_ramen_training_effect ×1 | 9.4 | 9.5 | 9.7 | **9.6** | 2.1% |

> 注① Run 1 为进程冷启动（全段偏高，B 段最甚 530.6），仅参考，**总均不含**。
> 注② 总均 = Run 2/3 均值。对照段（本次未触碰代码）B/C/F/G 回落上午 cleanup 基线 ±2%（257.2/265.1/4237.8/9.6 vs 260.20/270.39/4175.40/9.63）→ 测量环境与上午可比；可比口径下段 A 800.97 → 633.9 ≈ **-21%**（采样替换端到端收益，与 §7.3 进程内对照 -31~-42%/次 交叉验证吻合）。
> 注③ 机器性能漂移现象（2026-09-02 同 commit 连续 4 次重测）：段 A 漂移 634~907（±18%），未动代码的对照段同向漂移（B 255~382、F 4175~5662），每轮 Run 1 恒为冷启动峰值；G 相对稳定（9.4~13.7）。应对：**跨次数字不可直接对比**，先以「对照段是否回落基线」判可比性，定量一律以进程内对照为准。

## 附录 C：mcts_profiler env vars

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MCTS_PROFILER_RUNS` | 1 | 跑批局数 |
| `MCTS_PROFILER_FREQ` | 1000 | pprof 采样率（Hz） |
| `MCTS_PROFILER_LABEL` | "mcts" | 输出文件标签 |
| `MCTS_PROFILER_SEARCH_N` | 64 | 每候选 rollout 数 |
| `MCTS_PROFILER_STAGES` | "train,ramen,special" | 搜索阶段（逗号分隔） |
| `MCTS_PROFILER_NUM_THREADS` | 1 | rayon 线程数（1 = 单线程，栈最干净） |

## 附录 D：calc_training_value_microbench env vars

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CT_MICROBENCH_RUNS` | 1000 | 每段 iter 次数（每段内部 = 1 iter，对应 5 train 一回合 / 1 candidate 打分 / 1 distribute_all 等含义见 §7.2 表） |
| `CT_MICROBENCH_WARMUP` | 1000 | 每段 warmup 次数（让分配器 / cache 稳定，与 d10872a microbench 模板一致） |

**注**：本工具**没有** `LABEL` / `OUTPUT_PATH` env vars——与 `mcts_profiler` 不同，仅以表格形式 stdout 输出，墙钟总耗时也打屏，不需要落盘文件（每次跑直接对比数字即可）。

复现命令见 §6.4。