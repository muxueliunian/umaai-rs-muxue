# 性能分析指南（flamegraph / pprof-rs 选择 + 复现）

> 记录"对手写逻辑 / MCTS 性能分析"系列试验的工具选择准则、最终结论与复现命令，便于以后重测与演进。

## 1. 背景

项目当前两类 CPU 性能分析：

1. **手写策略批量火焰图**——`RecommendedRamenTrainer`，单线程跑批便于横向对比，结论与改动 d的 pprof + microbench 基线一致
2. **MCTS 单局性能剖面**——`RamenMctsTrainer`，多线程 + 闭包 + inline 优化导致传统 perf 栈展开质量差，需要换工具

两类分析的工具选择与命令各不相同。本文给出明确决策准则，避免下次重复踩坑。

## 2. 工具选择：先选对，再开跑

**经验法则**：

- 单线程 / 闭包少 / inline 温和（手写策略 / 规则层 / 业务逻辑）→ `cargo flamegraph`
- 多线程 rayon / 闭包深度嵌套 / opt-level 任意档（MCTS / 搜索调度 / 高频 hot path）→ `pprof-rs` 用户态采样（`mcts_profiler` / `sim_profiler`）

为什么这样分？`cargo flamegraph` 走 perf + inferno，栈展开依赖 dwarf unwind tables。多线程场景下 perf.data 容易膨胀到几 GB + 大量子子 samples lost；MCTS 闭包密集，inline 后看到的都是 `closure_env#0` / `impl#N` / `[unknown]` 这样的代号，看不到真正的 hot path 函数名。

`pprof-rs` 用户态 backtrace + symbolization 不依赖 perf 工具链，闭包/`impl#N`/`{closure#0}` 全部能解析到原始函数名。多线程友好，产物 KB-MB 量级不会膨胀，采样率可调（默认 1kHz），输出 Google pprof protobuf 给 `go tool pprof` 或 `inferno-flamegraph` 解读。

`cargo flamegraph --dev` 看似退而求其次的选择，但在 Rust 项目下基本不可用——Rust dev profile 默认没有 unwind tables，栈帧全部归 `[unknown]`，试验实测有 80% 样本是 `[unknown]`。

**所以**：

- 看到手写策略 / 业务逻辑 →`cargo flamegraph`
- 看到 MCTS / 搜索 / 任何"inline 重灾区"→ 直接 pprof-rs，不要在 cargo flamegraph 上浪费时间

## 3. MCTS 性能分析结论

### 3.1 关键发现：手写策略本体是 MCTS 的真正热点**

MCTS 单局（release + num_threads=1 + search_n=64 + train,ramen,special）耗时约 61.6s，其中 pprof 1kHz 采样开销约 27%。手写策略本体在 rollout 中被反复调用，**总占比约 22%**——是 MCTS 性能的最大单一杠杆点：

- `default_calc_training_value` / `default_calc_training_buff` / `calc_training_value`：训练数值计算三件套，每次 rollout 起点调用一次
- `SupportCard::calc_training_effect`：每张支援卡的 buff 计算，每次 rollout 调多次
- `RamenPolicy::score_train_action` + `RamenPolicy::status_gain`：候选打分与状态增益
- `LocalRamenTrainer::dynamic_status_adjustment` + `LocalRamenTrainer::reserve_penalty`：手写策略里的动态平衡与预留门限
- `RamenTrainingEffect::default` + `CardTrainingEffect::add`：训练效果结构体的默认初始化与累加

这与改动 d的 microbench baseline（`calc_training_value` 12ms/1000局 > `select_action` 8.5ms > `SupportCard::calc_training_effect` 5ms > `default_calc_training_buff` 4.45ms）一致——但 MCTS rollout 把这些热点的总采样放大了 N×M 倍（N 局 × 每局多次 rollout），优化手写策略任一 hot path 都会在 MCTS 路径上有乘法收益。

### 3.2 次要发现

**`f32::powi` 大整数运算占 ~6%**。`Big32x40::mul_pow2` + `Big32x40::div_rem_small` 共约 6%。这是 `f32::powi` 内部用 Big32x40 算法算 `2^n` 的开销，来源是手写策略里训练等级 / PT 增益的指数计算。**优化方向**：能换成整数乘法（如 `1.0 + level * 0.05` 代替 `1.05_f32.powi(level)`）的话可省约 6%。

**Vec 分配与 push 写入占 ~7%**。`score_train_action` 内部构造 `(String, f32)` 元组列表（`core::ptr::write<(String, f32)>` 2.77%）+ RamenAction 构造（0.76%）+ RamenPolicyOutput 构造（0.69%）+ RawVec deallocate/try_allocate_in/grow_amortized/finish_grow 合计约 3.9%。**优化方向**：`Vec::with_capacity` 预分配 + ActionResult 池化，能省约 3-4%。

**`f32 → String` 转换占 ~5%**。`flt2dec::format_exact_opt` + `cached_power` + `decoder::decode<f32>` + `String::write_str` + `core::fmt::write` 合计约 4.7%。主要来自 `println!` 输出与 pprof protobuf 序列化。

**`?` 运算符 unwrap 占 ~3.3%**。`Result<ActionValue, Error>::branch` 1.72% + `Result<(CardTrainingEffect, bool), Error>::branch` 1.10% + `Result<RamenPolicyOutput, Error>::branch` 0.53%。**优化方向**：hot path 里用 `let Ok(x) = ... else { unreachable!() }` 避免 branch 预测开销。

**`f32::clamp` / `f32::max` 占 ~3%**。多在 `calc_training_value` 内部做状态约束。

### 3.3 优化优先级

按杠杆降序：

1. **【最高】继续优化手写策略 hot path**——`default_calc_training_value` / `default_calc_training_buff` / `calc_training_value` / `score_train_action` / `SupportCard::calc_training_effect`。MCTS × N 局 × rollout 的乘法效应，远超单局手写策略的收益。
2. **【高】`f32::powi` → 整数乘法**。手写策略里等级加成 / PT 加成用整数乘法代替 `1.05_f32.powi(level)`，省约 6%。
3. **【中】`Vec::with_capacity` 预分配 + ActionResult 池化**。省约 3-4%。
4. **【中】消除 `?` branch**。hot path 里用 `let Ok(x) = ... else { unreachable!() }`，省约 3%。
5. **【低-中】`f32::clamp` → 手写比较 + 赋值**。省约 1.5%。
6. **【低】减少 `println!` 频率**。省约 5%（含 pprof 序列化）。

## 4. 试验历史：为什么 cargo flamegraph 对 MCTS 不可用

多次试验得到的明确结论：MCTS 这种多线程 + 闭包 + inline 重灾区，cargo flamegraph 在 opt-level 任意档、profile 任意档下都难以给出有用信息。要走 pprof-rs。

详见文末附录 A。

## 5. 工具固化：mcts_profiler

### 5.1 路径与依赖

- `crates/umasim/src/bin/mcts_profiler.rs`：MCTS pprof-rs profiler bin（d10872a `sim_profiler` 模板）
- 注册：放 `src/bin/` 自动注册为 bin target `mcts_profiler`
- 特性：`#![cfg(feature = "profiler")]`——Windows 不编译 pprof-rs，与 `sim_profiler` 同构
- 不需要 `[[bin`]` 显式 required-features：cfg gate 在源文件顶部，编译时直接跳过

### 5.2 与 sim_profiler 区别

| 维度 | sim_profiler | mcts_profiler |
|---|---|---|
| Trainer | `RecommendedRamenTrainer` | `RamenMctsTrainer` |
| rayon 线程数 | 全局默认（由 `game_config.toml` 的 `num_threads` 控制） | `MCTS_PROFILER_NUM_THREADS` 强制（默认 1） |
| SearchConfig | 无 | `search_n` / `stages` / `selection` / `ucb` / `radical_factor_max` 全套 |
| 跑批种子 | 与 `SIM_PROFILER_RUNS` 等 env vars 联动 | 与 `MCTS_PROFILER_*` env vars 联动 |
| 输出标签 | `SIM_PROFILER_LABEL`（默认 "baseline"） | `MCTS_PROFILER_LABEL`（默认 "mcts"） |

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

---

## 附录 A：试验历史（MCTS cargo flamegraph 失败记录）

| 版本 | 配置 | 结果 |
|---|---|---|
| v1 | release profile (opt-level='z'), num_threads=1, search_n=32, stages=train,ramen,special | 火焰图 Top 全是 init 阶段（gamedata JSON 加载 + BTreeMap insert 23.90%），MCTS 主循环函数全部被压成 `closure_env#0` / `impl#N`；看不到 `search_uniform` / `simulate_many` / `simulate_to_terminal` 实际开销 |
| v2 | release profile + 临时改 `Cargo.toml` opt-level=3, num_threads=1, search_n=256 | opt-level=3 让函数边界稍微清晰，`search_uniform` 7.41% 出现，但手写策略 hot path 函数（`default_calc_training_value` / `calc_training_value` 等）仍被 inline 吃掉；`RamenAction` 16.11% / `ActionResult` 11.10% 这些"类型相关帧"占据了真正 hot path 的位置 |
| v3 | **debug profile**（`cargo flamegraph --dev`）, num_threads=1, search_n=128 | **失败**——Rust dev profile 默认没有 unwind tables，79.99% 样本 `[unknown]`；perf.data 38.75 GB（debug 模式 binary 体积大）+ 9.13% samples lost |
| v4（成功）| pprof-rs（`mcts_profiler`）, num_threads=1, search_n=64 | 完美——手写策略 hot path 全部浮出水面：`default_calc_training_value` 3.16% / `default_calc_training_buff` 3.28% / `dynamic_status_adjustment` 2.54% / `reserve_penalty` 1.68% 等清晰可见 |

## 附录 B：MCTS pprof-rs 单局实测数据（2026-09-01）

### B.1 配置

- profile：release（opt-level='z'，项目预设）
- rayon 线程数：1
- search_n：64
- search_stages：train,ramen,special
- Trainer：`RamenMctsTrainer`（fallback = `RecommendedRamenTrainer`）
- Uma：美浦波旁（102601）+ speed build（速3 耐1 智1 + 友人 303054）
- base_seed：61444

### B.2 跑分

- 单局耗时：**61.6s**（含 pprof 1kHz backtrace 开销约 27%）
- pprof 采样：44599 ticks, 11252 stacks
- CPU 时间 ≈ 44.6s（与 wall-clock 61.6s 的差距是 pprof-rs 自身采样信号 handler 占用）
- 分数：67522（UA9）
- RMJ：3/3，自选比赛达标

### B.3 Top 函数（self time 聚合）

| ticks | % | 函数 |
|---:|---:|---|
| 1749 | 3.92 | `core::num::imp::bignum::Big32x40::mul_pow2` |
| 1464 | 3.28 | `RamenGame::default_calc_training_buff` |
| 1412 | 3.16 | `RamenGame::default_calc_training_value` |
| 1234 | 2.77 | `core::ptr::write<(String, f32)>` |
| 1152 | 2.58 | `RamenGame::calc_training_value` |
| 1132 | 2.54 | `LocalRamenTrainer::dynamic_status_adjustment` |
| 927 | 2.08 | `Big32x40::div_rem_small` |
| 814 | 1.83 | `SupportCard::calc_training_effect<RamenGame>` |
| 786 | 1.76 | `f32::clamp` |
| 765 | 1.72 | `Result<ActionValue, Error>::branch` |
| 749 | 1.68 | `LocalRamenTrainer::reserve_penalty` |
| 719 | 1.61 | `RamenPolicy::score_train_action` |
| 648 | 1.45 | `RamenPolicy::status_gain` |
| 578 | 1.30 | `RawVec::deallocate` |
| 554 | 1.24 | `RawVec::try_allocate_in` |
| 541 | 1.21 | `f32::max` |
| 513 | 1.15 | `RamenTrainingEffect::default` |
| 512 | 1.15 | `flt2dec::strategy::grisu::format_exact_opt` |
| 493 | 1.10 | `CardTrainingEffect::add` |
| 489 | 1.10 | `Result<(CardTrainingEffect, bool), Error>::branch` |
| 467 | 1.05 | `Option<Ordering>::is_some_and` |
| 366 | 0.82 | `Zip<IterMut<u32>, Iter<u32>>::next` |
| 363 | 0.81 | `flt2dec::cached_power` |
| 353 | 0.79 | `atomic_add<usize, usize>` |
| 339 | 0.76 | `core::ptr::write<RamenAction>` |
| 335 | 0.75 | `RawVec::grow_amortized` |
| 334 | 0.75 | `calc_normal_effect` |
| 330 | 0.74 | `flt2dec::decoder::decode<f32>` |
| 317 | 0.71 | `core::fmt::write` |
| 310 | 0.69 | `core::ptr::write<RamenPolicyOutput>` |
| 289 | 0.65 | `core::ptr::read<u8>` |
| 287 | 0.64 | `String::write_str` |
| 285 | 0.64 | `GameConstants::status_final_score` |
| 282 | 0.63 | `flt2dec::to_exact_fixed_str<f32>` |
| 277 | 0.62 | `usize::max` |
| 270 | 0.61 | `RawVec::finish_grow` |
| 251 | 0.56 | `RamenGame::is_shining_at` |
| 246 | 0.55 | `*mut u32::add` |
| 242 | 0.54 | `LocalRamenTrainer::decide_train` |
| 236 | 0.53 | `Result<RamenPolicyOutput, Error>::branch` |

## 附录 C：手写策略 100 局 cargo flamegraph 基线（2026-09-01）

### C.1 命令

```bash
cd umaai-rs
cargo flamegraph --release --no-default-features --bin bench_base -- --trainer handwritten
```

### C.2 实测基线

- 单局 mean **2.359ms**（p50 2.297 / p90 2.480 / p99 3.715）
- 吞吐 **424 局/s**
- 分数 mean 56687 std 2286，RMJ 2.38/3，自选比赛达标 100%

### C.3 关键前提

- `perf_event_paranoid` ≤ 2（本机默认 = 4，需 `sudo sysctl -w kernel.perf_event_paranoid=2`，临时）
- 项目 `profile.release opt-level='z'` 是项目预设，cargo flamegraph 默认沿用

### C.4 与 MCTS pprof-rs 对比

| 函数 | 手写策略 100 局 cargo flamegraph | MCTS 1 局 pprof-rs |
|---|---|---|
| `default_calc_training_value` | 4.85% | 3.16% |
| `calc_training_value` | 2.42% | 2.58% |
| `score_train_action` | 2.44% | 1.61% |
| `distribute_person` | 2.41% | < 0.5% |
| `apply_event` | 2.35% | < 0.5% |

pprof-rs 让手写策略 hot path 函数名**真正浮出水面**——之前 cargo flamegraph 把这些帧全部算成 inline closure 消失了。

## 附录 D：mcts_profiler env vars

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MCTS_PROFILER_RUNS` | 1 | 跑批局数 |
| `MCTS_PROFILER_FREQ` | 1000 | pprof 采样率（Hz） |
| `MCTS_PROFILER_LABEL` | "mcts" | 输出文件标签 |
| `MCTS_PROFILER_SEARCH_N` | 64 | 每候选 rollout 数 |
| `MCTS_PROFILER_STAGES` | "train,ramen,special" | 搜索阶段（逗号分隔） |
| `MCTS_PROFILER_NUM_THREADS` | 1 | rayon 线程数（1 = 单线程，栈最干净） |