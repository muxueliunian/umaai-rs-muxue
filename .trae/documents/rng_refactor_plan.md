# 随机数受控重构方案（RNG Refactor Plan）

> **摘要**：现在所有随机事件（人头分布、事件、训练成败）共用一个随机数序列，不同策略消耗次数不同，会导致后续随机全部错位——比较策略时结果混入了运气差。改造后：每个随机点由 `(种子, 回合, 类型, 本回合第几次)` 直接算出，互不牵连；人头分布/事件这类"每回合固定随机"与策略完全无关——同一种子、同一回合，任何策略看到的局面逐位相同，分数差就是策略本身的差距。MCTS 将来也能让所有候选动作在同一套随机局面下对比，结论更可信。

## 1. 背景与动机

bench 需要在**相同随机局面**下比较不同策略（如 Random / 手写策略 / 未来 MCTS），使分数差异反映"纯策略差异"而非随机噪声。当前实现无法做到：

- 策略的决策随机消耗会直接改变规则层随机序列（决策与规则共流）。
- 策略对吃面/训练/比赛的选择次数不同 → 规则随机消耗量不同 → 后续所有回合的分布与事件漂移。

目标：让"每回合固定发生的随机"（人头分布、hint、事件生成）与策略完全解耦——无论策略如何，同一 `(seed, 回合)` 下回合分布逐位一致；策略只影响"策略交互随机"（训练成功率、分身、比赛结果）。

## 2. 设计目标

1. **跨策略局面一致**：每回合分布/角标/hint/事件生成由 `(rule_master, turn)` 唯一决定，与策略无关。
2. **决策与规则彻底分离**：规则层不再使用 Trainer 传入的 rng（现状 bug，见 §3.2）。
3. **局号进种子**：每局初始规则种子不同，由 bench 局号哈希派生。
4. **随机点无状态**：第 N 次随机 = 纯函数 `f(master, ...)`，不依赖此前消耗次数——随机点不累计错位。
5. **MCTS 预留**：搜索随机完全自由；rollout 克隆局面独立流；第 k 轮 rollout 所有候选共享同一随机骨架（公共随机数 CRN）。

## 3. 现状分析

### 3.1 RNG 使用链

```
run_full_game(trainer, &mut decision_rng)
  ├─ Trainer 决策（RandomTrainer 每次 shuffle；手写策略几乎不消耗）
  └─ 规则层全部共用同一个 decision_rng：
       run_distribute（人头/角标/hint）· generate_events · do_train（成功率/分身）· 比赛
  └─ internal_rng（rule_rng，独立流）仅用于 2 处：
       吃面效果落地（ground_ramen_effects_with_internal_rng）+ RMJ 结算
```

- `seeded_rngs(seed)`：`(decision_rng, rule_rng)`，rule_rng 经 `set_internal_rng` 注入。
- 多局：`seed = base + run_idx`（简单加法，bench_base / bench_compositions）。
- `internal_rng` 是**持续流**（take / 放回），无回合重置。

### 3.2 漂移源（两个）

1. **决策与规则共流**：策略的决策随机消耗（RandomTrainer shuffle vs 手写策略几乎不消耗）直接改变后续分布/事件/训练随机。bench 注释声称的"规则层可复现"实际仅对吃面落地/RMJ 两处成立。
2. **规则流内部漂移**：策略吃面/训练次数不同 → internal_rng 消耗量不同 → 后续漂移。

## 4. 核心设计

### 4.1 无状态流 `SplitmixRng`

```
struct SplitmixRng { master: u64, counter: u64 }
impl rand::Rng:
    next_u64 = splitmix64(master ^ counter)   // 固定常数迭代的 SplitMix64 哈希
    counter += 1
```

- **无状态**：第 N 次随机 = `splitmix64(master ^ N)`，与"之前消耗多少次"无关。
- **值语义**：Clone 即获得独立流实例（counter 各自推进，互不影响）——MCTS 克隆局面的隔离基础。
- 暴露 `master()` / `counter()` 供测试观测。

### 4.2 种子公式（局号进种子）

```
第 i 局（bench 局号）：
  规则种子   rule_master_i = splitmix64(base_seed ^ i)       ← 每局独立（替代现状 base+i 加法）
  决策种子   decision_i    = splitmix64(base_seed ^ i ^ DECISION_TAG)

每回合（turn，回合开始重置）：
  回合固定流 master = splitmix64(rule_master_i ^ turn)
  策略流     master = splitmix64(rule_master_i ^ turn ^ STRATEGY_TAG)
  counter 本回合内从 0 计数（已确认：回合切换时归零）
```

派生常数（TAG、SplitMix64 gamma）固定不变，代码演进不得改动（同 `seeded_rngs` 的 `0x9E37_79B9_7F4A_7C15` 约定）。

### 4.3 流分类与归属

| 流 | 随机点 | 受控性 |
|---|---|---|
| **回合固定流** | 人头分布、训练角标、hint 分配、回合事件生成 | 每回合从 `(rule_master_i, turn)` 派生，n 从 0 计数——**与策略完全无关** |
| **策略流** | 训练成功率/大失败、分身分配、比赛结果、吃面效果落地（诀窍消耗随机等） | 从 `(rule_master_i, turn, STRATEGY_TAG)` 派生；仅 apply 真实动作时消耗 |
| **决策流** | Trainer 选择（含 MCTS 搜索） | 独立 `decision_i`，规则层不再触碰 |

归属：`RamenGame` 持有 `turn_fixed_rng` + `strategy_rng`（或统一"注入 master + 内部按需派生"），替换/并存现有 `internal_rng` 字段。

### 4.4 模块结构（新 `crates/umasim/src/rng.rs`）

```
rng.rs（原语级，lib.rs 注册 pub mod rng）
├── splitmix64(x) -> u64        唯一权威哈希实现
├── derive_seed(base, parts...) 种子派生（迭代 splitmix64 混入各分量）
├── SplitmixRng                  无状态流包装器（impl rand::Rng，见 §4.1）
└── StreamTag                    流标记枚举（TurnFixed / Strategy / Probe）
```

**明确不做**：不做 RngContext / 管理器 / 注册表——谁持有哪条流由 `RamenGame` 自行组织（已有 internal_rng 先例）。`seeded_rngs` 升级后保留在 bench.rs（局号派生属 bench 语义），内部调 `rng::derive_seed`。

## 5. MCTS 集成

MCTS 是**决策层**：不感知底层规则随机受控，其搜索随机完全自由。三层约束：

1. **搜索选择（UCB 等）**：用决策流 rng，自由消耗。
2. **rollout 必须分叉**：每次 rollout 前从决策流抽种子 `roll_seed`，克隆局面的流 master = `roll_seed`（否则所有 rollout 从相同起点 → 探测雷同，搜索退化）。
3. **公共随机数（CRN，旧搜索未实现）**：第 k 轮 rollout，所有候选共享同一随机骨架：
   ```
   第 k 轮：master_k = derive_seed(base, k)
     对每个候选 Ai：克隆局面 → 流 master = master_k → apply Ai → 模拟未来
   ```
   同一轮内候选面对逐位一致的随机未来，分数差 = 动作真实价值（无随机噪声）；跨轮 master 变化提供统计多样性。配对采样下噪声相关抵消，比较灵敏度远高于传统 rollout——适合拉面杯"低随机、选择差异微妙"的局面。

规则层改造成"无状态流 + 可注入 master"后，rollout 只需克隆局面并设置 master_k，自动获得 CRN 语义——MCTS 代码无需感知受控细节。

### 5.1 与以往随机的改进（传统 rollout vs CRN）

**传统方式（连续流 rollout，旧搜索实现）**：

```
候选 A 先模拟：随机序列 S1, S2, S3, ...
候选 B 后模拟：随机序列从 S4 开始（A 已消耗 3 次）→ B 的"随机未来"与 A 完全不同
分数差 = 动作真实价值 + 随机噪声差（两候选的随机环境互不相关）
```

- 问题：A/B 的比较混入两套独立随机环境的差异；当动作收益差小于随机波动时（拉面杯常见：吃面 vs 不吃面收益差微妙），需要海量样本才能统计显著。

**本方案（CRN 公共随机数 rollout）**：

```
第 k 轮：master_k = derive_seed(base, k)     ← 本轮所有候选共享同一随机骨架
  候选 A：流 master = master_k → apply A → 模拟
  候选 B：流 master = master_k → apply B → 模拟（同一随机未来，逐位一致）
分数差 = 动作真实价值（随机环境相同，噪声相关抵消）
```

- 改进：同一轮内候选面对**逐位一致的随机未来**（分布、事件、hint 全部相同），分数差仅反映动作本身；跨轮 master 变化提供统计多样性，多轮均值消除单轮偶然性。配对采样（paired sampling）下噪声相关抵消，同等样本量下动作分辨力显著高于传统 rollout。
- 顺带收益：rollout 随机点由 `(master_k, turn, n)` 唯一决定，与探测顺序/数量无关——rollout 可复现、可并行（各 rollout 从自己的克隆流取随机，无共享状态）。

## 6. 测试策略（三层）

**层 1：rng 单元测试**（rng.rs 内）
- 同一 (master, n) 两次计算同值（确定性）
- 不同 master 序列不同（无相关性）
- 消费 k 次后从第 k+1 次继续（无状态）
- 不同 StreamTag 派生序列不重叠（流间隔离）

**层 2：跨策略逐回合一致**（集成，验证"固定"核心效果）
- seed 固定，RandomTrainer vs RamenHandwrittenTrainer 各跑 N 回合
- 每回合 Distribute 后记录 distribution 快照 + 事件 ID 序列
- 输出对比：两策略逐回合分布序列**完全一致**

**层 3：隔离性**
- 回合重置：策略 A（狂训练）vs B（狂休息）跑 20 回合，回合 15 分布一致（前 14 回合消耗不影响第 15 回合）
- probe 克隆隔离：克隆局面、消费克隆流，原局面流不动（MCTS rollout 隔离原子验证）
- 流间不污染：策略流消耗后，回合固定流下一值不变（同回合内）

测试均按项目规范用 println 输出对比结果，不 assert。`SplitmixRng::master()/counter()` 提供可观测性。

## 7. 实施步骤（建议顺序）

1. **rng.rs 模块**：splitmix64 / derive_seed / SplitmixRng / StreamTag + 层 1 单元测试。
2. **RamenGame 流字段**：`turn_fixed_rng` + `strategy_rng`（替换/并存 internal_rng）；回合切换（run_begin / next）重置固定流。
3. **规则层去参数 rng**：`run_distribute` / `generate_events` / `do_train` / `ground_ramen_effects` 等约 8-10 处函数签名去掉 rng 参数，改从 self 流取随机。
4. **bench 局号派生**：`seeded_rngs` 升级入参加局号 i；bench_base / bench_compositions 传 `run_idx`。
5. **层 2 / 层 3 集成测试**。
6. **全量回归**：固定种子结果会变（测试多为 println，风险低），跑通确认。

## 8. 风险与待确认项

- **固定种子结果变化**：流结构改变所有固定种子模拟的随机序列，既有测试/基准结果全部变化，需重新校准基准基线。
- **规则层去参改动面**：8-10 处函数签名 + 调用链，需谨慎避免漏改导致随机来源混用。
- **派生常数冻结**：splitmix 迭代与 TAG 常量一旦发布不可变（可复现性契约），演进需走版本化。

（「第 N 次随机」计数范围已确认：本回合内从 0 计数，见 §4.2。）
