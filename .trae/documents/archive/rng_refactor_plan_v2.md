# 随机数受控重构方案 v2（RNG Refactor Plan v2）

> **v3 修订（2026-08-22 实施期）**：流分类从双流升级为**三流**——新增 `EventRng` 事件轴（§4.3）。
> 实施中发现：事件（马娘事件/支援卡连续事件）的**是否触发**依赖事件历史（max_time / 卡事件 8001-8003，策略状态），但其随机本身与策略/局面无关；独立成轴后，事件历史的差异只影响事件流自身，不污染局面流。
> 层 2 验收口径同步修正：**角标/人头分布/固定流消费量逐位一致**为硬指标；hint 位依赖剧本 PT 档位（策略状态，游戏机制），不参与一致性判定。
> CRN 退役范围修正：`reseed_for_stage`/`stage_seed`/`crn_stage_reseed` **仅拉面路径退役**（规则层接管）；onsen 规则层未改造，保留外挂 CRN。

> **摘要**：在 v1（`.trae/documents/rng_refactor_plan.md`，已归档）基础上，融合 PR #13（search 模块重构）**已落地**的 MCTS/CRN 实现与上游评审意见（`.trae/documents/rng_reply.md`，已归档），重新整理实施蓝图。
>
> **v2 相对 v1 的三大变化**：
> 1. §5 MCTS/CRN 从「计划」变为「现状 + 接驳」——PR #13 已实现搜索层 CRN，本方案的规则层改造需把 CRN 改接到新流上，而不是重新实现；
> 2. §4.1 流公式改用**加法派生**（采纳上游意见，消除 XOR 撞车）；§4.3 流归属升级为**编译期类型隔离**（`TurnFixedRng` / `StrategyRng`，接错直接编译不过）；
> 3. `search/seeds.rs` 的 `splitmix64` 提升为顶层 `rng.rs`（全仓库唯一权威哈希实现）。

## 1. 背景与动机

bench 需要在**相同随机局面**下比较不同策略（Random / 手写策略 / MCTS），使分数差异反映"纯策略差异"而非随机噪声。当前实现无法做到：

- 策略的决策随机消耗会直接改变规则层随机序列（决策与规则共流）。
- 策略对吃面/训练/比赛的选择次数不同 → 规则随机消耗量不同 → 后续所有回合的分布与事件漂移。

目标：让"每回合固定发生的随机"（人头分布、hint、事件生成）与策略完全解耦——无论策略如何，同一 `(seed, 回合)` 下回合分布逐位一致；策略只影响"策略交互随机"（训练成功率、分身、比赛结果）。

## 2. 设计目标

1. **跨策略局面一致**：每回合分布/角标/hint/事件生成由 `(rule_master, turn)` 唯一决定，与策略无关。
2. **决策与规则彻底分离**：规则层不再使用 Trainer 传入的 rng（现状 bug，见 §3.2）。
3. **局号进种子**：每局初始规则种子不同，由 bench 局号哈希派生。
4. **随机点无状态**：第 N 次随机 = 纯函数 `f(master, N)`，不依赖此前消耗次数——随机点不累计错位。
5. **MCTS 预留**：搜索随机完全自由；rollout 克隆局面独立流；第 k 轮 rollout 所有候选共享同一随机骨架（公共随机数 CRN）。

> v2 补充：目标 5 在 PR #13 中已实现（搜索层阶段重播种），本方案的目标是让规则层无状态化后**接管** CRN 对齐职责（§5.2）。

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

### 3.3 PR #13 已落地内容（search 模块重构）

| v1 方案条目 | PR #13 现状 |
|---|---|
| §5.2 rollout 必须分叉 | `FlatSearchGame::fork_for_rollout`——「克隆局面」与「重置内部随机流」绑成不可分割操作；通用搜索一律经它建分支，不得直接 `clone()`（`search/searchable.rs`） |
| §5.3 CRN 公共随机数 | 已实现：按 `(rollout 种子, 回合, 阶段)` 重播种（`RolloutSeeds::stage_seed` + `FlatSearch::reseed_for_stage`），`SearchConfig::crn_stage_reseed` 控制（默认开，`gamedata/config.rs` MctsConfig + `default_config.toml`） |
| §5.3 rollout 可复现、可并行 | 已实现：1/2/8/24 线程结果逐位相同（`test_crn_pairing_gain*` 等回归） |
| §7.1 splitmix64 / derive_seed | 在 `crates/umasim/src/search/seeds.rs`（`pub(crate) fn splitmix64`，sampler 复用），**尚未提升为顶层 `rng.rs`** |

**关键约束（PR 已落地并有回归测试守护）**：**候选索引绝不能进种子派生**。CRN 的方差削减来自 `Var(X_a − X_b) = Var(X_a) + Var(X_b) − 2Cov(X_a, X_b)`，按 `hash(root, 候选, j)` 派生会让协方差归零，退化成「可复现的独立抽样」。`test_search_invariant_to_action_order` 守护此约束，v2 全部改造不得破坏。

**实测收益（PR #13 报告）**：配对相关系数 / 等效样本倍率——onsen 0.69 / 3.65x，拉面 0.69 / 3.52x（「仅共享起始种子」仅 1.3~1.5x）。

### 3.4 协调点（本方案必须解决的接驳问题）

搜索层 CRN 是**外挂式对齐**（按阶段边界重播种传入的 `&mut StdRng`）；规则层改造（§7 步骤 3「规则层去参数 rng」）落地后，规则层不再读传入 rng，`reseed_for_stage` 将**静默失效**——CRN 变空操作但测试多为 println 不会报警，搜索质量悄悄退回 1.3 倍。因此规则层改造必须同时完成 CRN 接驳（§5.2），并用 PR 已有回归测试验收。

## 4. 核心设计

### 4.1 无状态流 `SplitmixRng`

```
struct SplitmixRng { master: u64, counter: u64 }
impl rand::RngCore:
    next_u64 = splitmix64(master + counter * GAMMA)   // 加法派生（v2 采纳上游意见）
    counter += 1
```

- **无状态**：第 N 次随机 = `splitmix64(master + N·GAMMA)`，与"之前消耗多少次"无关。
- **值语义**：Clone 即获得独立流实例（counter 各自推进，互不影响）——MCTS 克隆局面的隔离基础。
- 暴露 `master()` / `counter()` 供测试观测；`reset(master)` 供回合切换 / rollout 注入。
- **为什么是加法而不是 v1 的 XOR**：`splitmix64(master ^ counter)` 有 `stream(A, n) == stream(A ^ k, n ^ k)` 的撞车性质（master 相近的两条流互相错位重叠）；`splitmix64(master + counter * GAMMA)` 无此性质，且与 PR 已落地的 `RolloutSeeds::seed_at` 公式同构，两边统一。`GAMMA = 0x9E37_79B9_7F4A_7C15`（黄金比例，与 `seeded_rngs` 历史约定同源，冻结不可改）。

### 4.2 种子公式（局号进种子，v1 不变）

```
第 i 局（bench 局号）：
  规则种子   rule_master_i = splitmix64(base_seed ^ i)       ← 每局独立（替代现状 base+i 加法）
  决策种子   decision_i    = splitmix64(base_seed ^ i ^ DECISION_TAG)

每回合（turn，回合开始重置）：
  回合固定流 master = splitmix64(rule_master_i ^ turn)
  策略流     master = splitmix64(rule_master_i ^ turn ^ STRATEGY_TAG)
  counter 本回合内从 0 计数（已确认：回合切换时归零）
```

派生常数（TAG、GAMMA、SplitMix64 混淆常数）固定不变，代码演进不得改动（同 `seeded_rngs` 的 `0x9E37_79B9_7F4A_7C15` 约定）。**注**：加法派生只用于**流内取值**（§4.1）；种子**派生公式**仍用 XOR 混合各分量（master 本身是 splitmix64 输出、散得开，无撞车风险，且与 PR 现有 `stage_seed` / `InternalSeed` 派生语义一致）。

### 4.3 流分类与归属（v3：三流，编译期类型隔离）

| 流 | 随机点 | 受控性 |
|---|---|---|
| **局面流** `TurnFixedRng` | 人头分布、训练角标、hint **触发位**（`distribute_hint` 的 random_bool） | 每回合从 `(rule_master_i, turn)` 派生，n 从 0 计数，`run_distribute` 独占——**与策略完全无关**（层 2 硬指标） |
| **事件流** `EventRng` | 回合开始事件链：休息心得结束判定、友人解锁判定、事件生成（weights/choose）、事件应用 | 从 `(rule_master_i, turn, EVENT_TAG)` 派生。事件的**是否触发**依赖事件历史（策略状态），但随机本身独立成轴——历史差异只影响事件流自身，不污染局面/策略流 |
| **策略流** `StrategyRng` | 训练成功率/大失败、分身分配（地区+超级拉面）、比赛结果、吃面效果落地、休息/外出结果、策略触发事件（失败/hint 事件判定/extra train）的应用结果 | 从 `(rule_master_i, turn, STRATEGY_TAG)` 派生；仅 apply 真实动作时消耗 |
| **决策流** | Trainer 选择（含 MCTS 搜索） | 独立 `decision_i`，规则层不再触碰 |

**类型隔离强制措施**（采纳上游意见 2，v3 扩展到三流）：`TurnFixedRng` / `EventRng` / `StrategyRng` 为不同类型（各自包装一个 `SplitmixRng`），任何规则层随机点接错流直接编译不过，把「8–10 处签名靠 review 盯」风险升级为编译期约束。

**hint 的两层随机归属**：①触发概率（`distribute_hint`，每回合必发生）→ 局面流；②hint 事件判定（`hint_persons.choose` / `attr_prob`，训练后策略触发）→ 策略流。hint 事件的发生次数不影响局面流（局面流在 `run_distribute` 内已消费完毕）。注意：**hint 概率值**依赖剧本 PT 档位（`hint_bonus_pct`，策略状态）——同序列不同概率时 `random_bool` 结果可不同，属游戏机制，非随机错位（层 2 测试锁定羁绊后，hint 位仍可能因 PT 档位不同而不一致，不参与一致性判定）。

归属：`RamenGame` 持有 `rule_master` + `turn_fixed` + `event` + `strategy`（未注入 rule_master 时全部为 None，规则随机回退旧行为），替换现有 `internal_rng` 字段。

### 4.4 模块结构（顶层 `crates/umasim/src/rng.rs`，lib.rs 注册 `pub mod rng`）

```
rng.rs（原语级）
├── splitmix64(x)                   唯一权威哈希实现（合并 search/seeds.rs 的 pub(crate) 版，
│                                   sampler / search / bench 全部改引用此处）
├── derive_seed(base, parts...)     种子派生（XOR 混入各分量后单次 splitmix64）
├── SplitmixRng                     无状态流核心（加法派生，impl rand::RngCore）
├── TurnFixedRng / EventRng / StrategyRng   类型隔离的三条流（§4.3）
├── StreamTag                       流标记枚举（TurnFixed / Strategy / Event / Probe）
└── GAMMA / DECISION_TAG / STRATEGY_TAG / EVENT_TAG / PROBE_TAG   冻结常数
```

- `search/seeds.rs` 保留 `RolloutSeeds`（搜索专用 CRN 载体），`splitmix64` 改引用 `crate::rng::splitmix64`（行为不变，seeds.rs 测试守护迁移正确性）。`InternalSeed` 已随拉面 CRN 接驳退役（规则层直接注入 rollout 种子，无需分频道派生）。
- **明确不做**：不做 RngContext / 管理器 / 注册表——谁持有哪条流由 `RamenGame` 自行组织（已有 internal_rng 先例）。`seeded_rngs` 升级后保留在 bench.rs（局号派生属 bench 语义），内部调 `rng::derive_seed`。

## 5. MCTS/CRN 集成（v2：现状 + 接驳）

### 5.1 现状（PR #13，无需重新实现）

- **搜索选择（UCB 等）**：用决策流 rng，自由消耗。
- **rollout 分叉**：`FlatSearchGame::fork_for_rollout(rollout_seed)`——克隆局面 + 重置规则层内部流（拉面注入 `rule_master = rollout_seed`，规则层按三流派生）。
- **CRN**：`RolloutSeeds::seed_at(j)`（所有候选共享第 j 次 rollout 种子）+ `reseed_for_stage` 按 `(rollout 种子, 回合, 阶段)` 重播种，`crn_stage_reseed` 默认开。
- **约束**：候选索引不进种子派生（`test_search_invariant_to_action_order` 守护）。

### 5.2 规则层落地后的接驳（本方案实施内容）

规则层改成"无状态流 + 可注入 master"后，CRN 职责从搜索层外挂重播种**移交**给规则层自身：

```
第 k 轮 rollout：master_k = seeds.seed_at(k)（RolloutSeeds，加法派生，所有候选共享）
  对每个候选 Ai：fork_for_rollout(master_k) = 克隆局面 + 注入 rule_master = master_k
                  → apply Ai → 模拟未来（每回合固定流/策略流由 (master_k, turn) 派生）
```

- 同一轮内候选面对**逐位一致的随机未来**（分布、事件、hint 全部相同——固定流按 `(master_k, turn)` 派生，天然对齐，无需再按阶段重播种），分数差 = 动作真实价值（CRN 配对，噪声相关抵消）。
- **退役内容（仅拉面路径）**：拉面的 `simulate_common` 不再调用 `reseed_for_stage`（规则层接管对齐）；`RamenGame::fork_for_rollout` 改为注入 `rule_master`。
- **保留内容（onsen）**：`FlatSearch::reseed_for_stage` / `RolloutSeeds::stage_seed` / `SearchConfig::crn_stage_reseed`（含配置字段与 default_config.toml 行）**全部保留**——onsen 规则层未改造，其 rollout 仍走传入 rng，外挂重播种是唯一的 CRN 对齐手段（拉面 v2 原文的「退役三件套」仅对拉面成立）。`FlatSearchGame::crn_stage_key` 保留（onsen 使用；拉面实现保留仅为满足 trait）。
- **保留内容**：`fork_for_rollout` 入口（签名不变，内部从 `set_internal_rng(StdRng::seed_from_u64(...))` 改为 `set_rule_master(master_k)`）；`RolloutSeeds::seed_at`（仍是 rollout 骨架种子来源）。
- **验收**：PR 已有回归测试直接复用——`test_crn_pairing_gain` / `test_crn_pairing_gain_ramen`（阶段重播种前后各候选均值与配对相关系数）、多线程逐位一致测试。改造后配对收益**不得回退**（目标保持 corr ≥ 0.6 / 等效样本 ≥ 3x 量级；拉面规则层接管后理论上应不低于现状）。

### 5.3 与以往随机的改进（传统 rollout vs CRN，v1 保留）

**传统方式（连续流 rollout，未实现 CRN 之前）**：候选 A 先模拟消耗随机序列，候选 B 从错位的位置开始 → 分数差 = 动作真实价值 + 随机噪声差，拉面杯"低随机、选择差异微妙"的局面需要海量样本。

**本方案（CRN）**：同一轮候选共享 `master_k`，逐位一致的随机未来 → 分数差仅反映动作本身；跨轮 master 变化提供统计多样性。rollout 随机点由 `(master_k, turn, n)` 唯一决定，与探测顺序/数量无关——rollout 可复现、可并行（各 rollout 从自己的克隆流取随机，无共享状态）。

## 6. 测试策略（三层 + PR 回归复用）

**层 1：rng 单元测试**（rng.rs 内，沿用 seeds.rs 迁移来的测试）
- 同一 (master, n) 两次计算同值（确定性）
- 不同 master 序列不同（无相关性）
- 消费 k 次后从第 k+1 次继续（无状态）
- Clone 值语义：克隆后各自推进互不影响（MCTS 隔离原子验证）
- 不同 StreamTag 派生序列不重叠（流间隔离）
- **加法派生防撞车**：`stream(master, n) != stream(master^k, n^k)` 抽样验证（v2 新增，对应上游意见 1）

**层 2：跨策略逐回合一致**（集成，验证"固定"核心效果，`rng_consistency.rs`）
- seed 固定，RandomTrainer vs RamenHandwrittenTrainer 各跑 N 回合（测试锁支援卡羁绊=100，消除得意率状态差异）
- 每回合 Distribute 后记录 distribution 快照 + 角标 + hint + 固定流消费量 + 事件增量
- 输出对比：**角标 / 分布表 / 固定流消费量逐位一致**（硬指标，实测 19/19）；hint 位因依赖剧本 PT 档位不参与判定；事件增量逐位一致（事件流独立）

**层 3：隔离性**（`rng_consistency.rs`）
- 回合重置：策略 A（狂训练）vs B（狂休息）跑 20 回合，分布/角标/固定流消费逐位一致（实测 0 不一致，前 14 回合消耗不影响第 15 回合）
- probe 克隆隔离：克隆局面、消费克隆流，原局面流不动（MCTS rollout 隔离原子验证）
- 流间不污染：策略流消耗后，回合固定流下一值不变（同回合内）

**PR 回归复用（搜索层，禁止破坏）**
- `test_search_invariant_to_action_order`：候选顺序不变性（候选索引不进种子派生的守护）
- `test_crn_pairing_gain` / `test_crn_pairing_gain_ramen`：CRN 配对收益验收（接驳后复用）
- seeds.rs 迁移测试：`splitmix64` 提升为顶层实现后行为逐位不变

测试均按项目规范用 println 输出对比结果，不 assert（seeds.rs / flat_search.rs 中已有 assert 的测试保持原样，属 PR 已落地内容）。

## 7. 实施步骤（建议顺序，v2 更新）

1. **rng.rs 模块**（已完成）：`splitmix64` 提升为顶层唯一实现（sampler / seeds.rs / bench 改引用）；`derive_seed` / `SplitmixRng`（加法派生防 XOR 撞车）/ `TurnFixedRng` / `EventRng` / `StrategyRng`（类型隔离三流）/ `StreamTag`（含 Event 变体）+ 层 1 单元测试。
2. **RamenGame 流字段**（已完成）：`rule_master` + `turn_fixed` + `event` + `strategy`（新类型）替换 `internal_rng`；`set_rule_master(u64)`；回合切换（run_begin 开头）重置三条流。
3. **规则层去参数 rng + CRN 接驳**（已完成）：
   - 规则层签名泛型化 `&mut impl Rng`（Trainer trait 保持 `&mut StdRng` 决策流）；规则随机按三流归属改从 self 流取（run_begin 事件链→事件流、run_distribute→局面流、apply 动作→策略流），未注入 rule_master 时回退旧行为（传入 rng / internal_rng 兜底）；
   - `RamenGame::fork_for_rollout` 注入 `rule_master = rollout 种子`（拉面 CRN 由规则层接管）；拉面 `simulate_common` 退役 `reseed_for_stage` 调用；onsen 保留 `reseed_for_stage` / `stage_seed` / `crn_stage_reseed`；`InternalSeed` 退役；
   - `test_crn_pairing_gain*`（onsen）保持有效。
4. **bench 局号派生**：`seeded_rngs` 升级入参加局号 i；bench_base / bench_compositions 传 `run_idx`。
5. **层 2 / 层 3 集成测试**。
6. **全量回归**：固定种子结果会变（测试多为 println，风险低），跑通确认；`test_*_reproducible` 类（同 seed 两次一致）必须保持通过。

## 8. 风险与待确认项

- **固定种子结果变化**：流结构改变所有固定种子模拟的随机序列，既有测试/基准结果全部变化，需重新校准基准基线（`test_*_reproducible` 同 seed 两次一致的测试不受影响，仍须通过）。
- **规则层去参改动面**：8-10 处函数签名 + 调用链，需谨慎避免漏改导致随机来源混用——v2 已用类型隔离（§4.3）把此风险从 review 约束升级为编译期约束。
- **CRN 接驳正确性**：退役搜索层重播种后配对收益不得回退——`test_crn_pairing_gain*` 是硬验收；若规则层接管后收益反而下降，需在实施时对比诊断（固定流消费顺序是否与 PR 阶段对齐口径一致）。
- **派生常数冻结**：splitmix 迭代与 TAG 常量一旦发布不可变（可复现性契约），演进需走版本化。v2 冻结清单：`GAMMA`（0x9E37_79B9_7F4A_7C15）、`MIX_A` / `MIX_B`（SplitMix64 混淆常数）、`DECISION_TAG` / `STRATEGY_TAG` / `PROBE_TAG`、`INTERNAL_STREAM_TAG`（seeds.rs 现有）。
- **splitmix64 提升迁移**：从 search/seeds.rs `pub(crate)` 提升到顶层 `pub` 时不得改变函数行为（纯迁移，seeds.rs 测试守护）。
- **rollout 决策器与决策流**：搜索 rollout 的 Trainer 选择仍用决策流（rollout 内部自由消耗），与规则层无状态流互不干扰——实施时确认 rollout 训练器路径不误读规则流。

（「第 N 次随机」计数范围已确认：本回合内从 0 计数，见 §4.2。事件应用随机归属、未注入 rule_master 的回退、bench 接口形态均已在实施前与用户确认：按触发来源分流 / 回退旧行为用传入 rng / `seeded_rngs(base_seed, run_idx) -> (StdRng, rule_master)`。）
