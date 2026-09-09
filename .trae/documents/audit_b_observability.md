# 观测/配置层缺陷审计结论（2026-08-24）

审计基线 commit：`a223a6e`
方法：brief 交给 gpt-5.6-sol（effort=high）与 grok-4.6（effort=xhigh）**独立只读审计**，
Claude 先自行推一遍结论作对照基准，再逐条核验二者主张。
本文件**不进 git**（属进度记录，非代码文档）。

---

## Bug 1 —— 局末 PT 和吃面次数恒为 0

**现象**　`results.csv` 的 `scenario_pt`、`eat_count` 两列每局都是 0。

**机理**　RMJ 结算回合（`state.rs:353` = turn 23/47/71）把两个计数器清零
（`game.rs:153` / `game.rs:186`）；游戏在 turn 77 结束，而 turn 72–77 跳过 `RamenSelect`
直接进 Train（`game.rs:112`），不再吃面。故最后一次清零后两值恒为 0。

**取证**　`logs/bench_base_results.csv` 2100 局，两列非零个数均为 **0**。

**两 agent**　均判「是缺陷 / 无争议」。

**影响**　剧本 PT 是拉面杯核心指标，bench 输出完全看不到，手写策略调参无依据。
`bin/minimal_strategy_ab.rs:25-46` 已在外部重算绕过——且它重算的是**三年合计**，
与 `GameOutcome` 文档写的「当年」不是同一指标，同名字段在不同 binary 中语义已分叉。

**方案**　`RamenState` 增逐年归档数组，在 RMJ 归零**之前**写入；`GameOutcome` 暴露数组；
CSV 改 `scenario_pt_y1..y3` / `eat_count_y1..y3` / `region_y1..y3`。删除 `minimal_strategy_ab` 旁路重算。
**用户裁定：只留逐年三列，不留三年合计。**

**实现陷阱**（Codex 提出）　归档地区必须用**显式 `year_idx`**，不能用 `current_year()` 推断——
turn 23 时 `current_year()` 仍判第一年，但当时选的是第二年地区。

---

## Bug 2 —— CRN 收益测量工具在拉面上是空跑

**现象**　`flat_search.rs:1181` `test_crn_pairing_gain_ramen` 用 `crn_stage_reseed` 做 A/B，
但该开关只在 `impl FlatSearch<OnsenGame>` 的 `reseed_for_stage` 路径生效，
拉面走 `simulate_common`，完全不读 `self.config`。两轮输出必然逐位相同。
成因：从温泉 `test_crn_pairing_gain` 拷贝时把 `simulate` 换成了 `simulate_common`，开关没跟着换。

**两 agent**　均判「是缺陷 / 无争议」。

**Grok 的纠正（已采信）**　「打印的等效倍率在测量一个不存在的差异」**过杀**。
不存在的只是**两臂之差**；每一臂自身的倍率测的是「共享 `seed_at(j)` → `rule_master`」的真实配对，
该机制是活的：`searchable.rs:158` `fork_for_rollout` 把 rollout 种子注入为 `rule_master`，
且 `RolloutSeeds::seed_at(j)` 刻意不含候选索引。

**影响**　这是决定能否下调 `search_n` 的唯一依据，现状给不出任何有效结论。

**方案**　A/B 轴换成 `shared_rule_master ∈ {true,false}`：
独立臂给 `rule_master` 带上候选身份（如 `derive_seed(seed_at(j), [i])`），共享臂维持现状。
断言必须含**独立臂增益 ≈ 1**（尺子没坏的锚点）与**共享臂 > 事先写死的下限**。

**实测结果（2026-08-24，修复后首次有效测量）**　拉面根局面（回合 0 / Train / 7 候选，
200 rollouts，21 个候选对）：

| 臂 | 平均 corr | 平均等效倍率 | 区间 |
|---|---|---|---|
| 独立 `rule_master` | 0.0019 | **1.01x** | [0.90, 1.18] |
| 共享 `rule_master`（生产） | 0.7736 | **4.86x** | [3.28, 11.61] |

独立臂 corr 近零、倍率贴 1，证明测量尺子本身是准的；共享臂 4.86x 即拉面规则层 CRN
的实际配对方差削减效果。这是「能否下调 search_n」所缺的那个数，也是 Bug 6 定标
`expected_search_stdev` 的前置数据。


**红线（两 agent 一致）**　生产 CRN 是对的，错的是测试。
**不许**为做对照去改 `seed_at` 或 `fork_for_rollout`。

---

## Bug 3 —— 同一测量工具的配对错位

**现象**　`flat_search.rs:1200` 各候选先各自 `flatten()` 掉失败样本，再按压缩后下标配对。
候选 A 的第 5 个成功样本会被配到候选 B 的第 6 个。仅打印警告，静默继续计算。

**两 agent**　Codex 发现；Grok 未提。

**方案**　保留原始 `j`，只对「双方在同一 `j` 都成功」的交集计算相关与差值方差。与 Bug 2 同批修。

---

## Bug 4 —— UCB 首组无视预算，小预算下自适应不发生

**现象**　`search_ucb`（`flat_search.rs:355-387`）第一阶段无条件给每候选跑满 `group_size`；
`group_size > search_n` 时超预算，且第二阶段 `max_planned >= search_n` 立刻成立，自适应零次。

**两 agent 分歧**
- Codex：**不是缺陷**。`search_n` 语义在均匀路径是「每候选定额」、UCB 是「最优候选停搜阈值」，
  与上游 `searchSingleMax` 一致，属参数语义过载而非实现错。
- Grok：**拆成两半**。语义确属上游契约（同意 Codex）；但「首组超预算」是独立的无争议实现 bug。

**裁定：采信 Grok。** 铁证：`crates/unused/bin/generate_mean_filtered_data.rs:232`
**已经写了这条 clamp**——作者知道该问题，只是补在外围而非内核。

**Claude 自我更正**　先前把此项描述为「`search_n` 被误当成每候选上限」不准确，已撤回。

**影响**　生产默认（`search_n=12288 > group=2048`）踩不到。踩到的是小预算：bench、
测试里 `with_search_n(16)` 忘改 group。`bench_base.rs:102-113` 因此把 `search_ucb` 默认设为 false。

**方案**　首组改 `group_size.min(search_n)`，把 clamp 从 unused 收进搜索内核。生产行为不变。

**红线（两 agent + Claude 一致否决）**　不要把 `search_n` 改成「决策点总预算」。
均匀路径现为 5×12288=61440 次 rollout，改后每候选仅剩约 2458 次，质量断崖且全部基线作废。

---

## Bug 5 —— 用户配置的搜索参数被静默丢弃

**现象**　`config.rs:862-863` 的 `OverrideGameConfig::merge` 只有两行：

```rust
ret.mcts.search_n = self.mcts.search_n;
ret.mcts.radical_factor_max = self.mcts.radical_factor_max;
```

`OverrideGameConfig.mcts` 是完整 `MctsConfig`（非 Option 结构；`OverrideMctsConfig` 类型不存在）。后果：

1. 用户在 `game_config.toml` 的 `[mcts]` 写 `search_group_size` / `expected_search_stdev` /
   `use_ucb` / `search_cpuct` / `max_depth` / `policy_delta` / `ramen_search_stages` —— **全部静默失效**。
2. 赋值**无条件**：写个空 `[mcts]` 就会让 `search_n` 被 serde 缺省 10240 覆盖掉
   `default_config.toml` 的 12288，**静默改小搜索预算**。

**两 agent**　独立发现，结论一致。

**定位**　这是 Bug 6 数值打架的**真正根因**：参数没有完整的覆盖链。

**方案**　改 Option 式逐字段覆盖。
**副作用需先查**：改后「用户 toml 省略 `search_n`」会从「被 10240 覆盖」变成「保留 12288」，属行为变化。

---

## Bug 6 —— `expected_search_stdev` 等参数四处取值矛盾

| 来源 | `search_group_size` | `expected_search_stdev` | 谁在用 |
|---|---|---|---|
| `SearchConfig::default()` / `::ucb()` | 256 | 2200 | 单测、**bench_base** |
| `MctsConfig` serde 默认 | 512 | 2200 | 基本是死默认 |
| `default_config.toml [mcts]` | 2048 | **15000** | **正式运行的拉面 MCTS** |
| `default_config.toml [collector]` | 256 | 2200 | 数据采集 |

**两 agent**　均判「部分成立 / 有争议」。不一致是客观事实，「哪个数对」是调参非 bug。

**Claude 自我更正**　先前称「15000 会让 UCB 退化成均匀分配」——**错误，已撤回**。
两 agent 各自算得：`group=2048`、5～7 候选时探索项约 741～877 分，
而候选间均分差常达数千分，UCB 仍会集中火力。15000 只是探索偏猛（约 5～7 倍实测 σ），
且「期望标准差」名不副实（其物理含义应为单条 rollout 终局分的标准差，与 score 同量纲）。

**实测锚点**　`logs/bench_base_results.csv` 2100 局整局分 σ ≈ **2919**（Grok 取前 49 局得 2250）。
注意：这是跨种子整局分 σ，非同一根局面下条件 rollout 的 σ，只能作量级参考。
**不可引用 `nn_pipeline_plan.md` §12 的 σ 数据——该文档基线已第四次作废。**

**影响**　① bench（256/2200）与主程序（2048/15000）跑的不是同一套 UCB 参数，
拿基线论证 toml 参数会混变量；② 探索强度是否合理无人知晓。

**方案**　拆两步。
- 卫生（方向唯一）：补 rustdoc 说清物理含义；bench 与主程序走同一配置源。
- 定标（需用户拍板）：15000 改回 ~2200，或保留 15000 但改名为 `ucb_exploration_scale`。
  **建议等 Bug 2 修好、能打出真实配对增益与根节点 `ActionResult::stdev()` 之后再定。**

---

## 分批（用户已确认）

根因数为 4（Bug 2+3 同根因，Bug 5+6 同根因）。用户要求分两次，故将有争议项整体剔出：

| 批次 | 内容 | 争议 | 动基线 |
|---|---|---|---|
| **第一次** | Bug 1 | 无 | 否 |
| **第二次** | Bug 2 + Bug 3 + Bug 4 | 无 | 否 |
| **暂不做** | Bug 5 + Bug 6 | 有 | Bug 6 定标会动 |

顺序依赖：Bug 6 待定的数值需靠 Bug 2 修好后产出的数据支撑，故非「简单先做」。

## 执行方式（用户指定）

第一次由 grok 实施、Claude review；review 通过后再做第二次。
