# UmaAI-RS 问题记录

本文件用于记载较复杂问题（需要用户协助解决的）的解决过程。

## 问题记录模板

```
## [问题标题]
- **日期**：YYYY-MM-DD
- **状态**：待解决 / 解决中 / 已解决
- **问题描述**：简要描述问题现象
- **排查过程**：记录排查步骤和发现
- **解决方案**：记录最终解决方法
- **备注**：其他相关信息
```

---

## 人头（person_index）问题解决现状一览（2026-08-25 复核）

| # | 问题 | 状态 | 简要说明 |
|---|---|---|---|
| 1 | 人头下标当卡组下标（`< 6` 守卫全线失效） | ✅ 已解决（08-23） | 拉面规则层 `default_calc_training_buff` / `add_friendship` / `deyilv` / hint 路径全部改 `card_id` 反查，回归测试在；base/onsen 两处 `< 6` 回写布局下正确、未动 |
| 2 | 训练人数加成硬编码「理事长=6、记者=7」 | ✅ 已解决（08-24） | `count_training_persons` 按 `PersonType` 判定，拉面四项全错的旧过滤器废弃，三个回归用例在 |
| 3 | 超级拉面分身漏友人卡 | ✅ 已解决（08-24） | 候选收集由 `0..6` 改全扫全体人头 + 每训练一个友人约束 |
| 4 | 分身分配假失败 / 顺序饿死 | ✅ 已解决（08-24） | 概率重试改合法集直选、友人卡优先分配、真无解跳过不中断回合；随机流改用按 `(rule_master, turn, TAG)` 派生局部流 |
| 5 | 诀窍槽训练加成按旧布局索引 | ✅ 已解决 | `fill_feeling_gauge` 改按 `PersonType` 统计，与显示层 `collect_train_lines` 同口径 |
| 6 | hint 路径无守卫 `deck[person_index]`（base/onsen） | ⚠️ 待解决（潜伏） | base `basic.rs` / onsen hint 路径原样保留；生产只对 `PersonType::Card` 打 hint 踩不到越界 |
| 7 | `card_id` 重复校验缺失 | ⚠️ 待解决（防御） | `validate_game_config` 只查 `cards.len()==6`，手写卡组可能触发 `deck_index_of` 静默命中第一张 |
| 8 | `/* */` 调试块含旧假设 | ⚠️ 待解决（死代码） | `ramen/action.rs` 调试块仍含 `pidx < 6` + `deck[pidx]`，解开注释才会出问题 |
| 9 | rng_consistency 的 `0..6` 遍历 | ⚠️ 待解决（测试语义） | `run_turns` 同时锁 `deck[i]`/`persons[i]`，友人卡人头 6 锁不到；不影响该测试结论 |
| 10 | onsen 挖土过滤器 `*p != 6 && *p != 7` | ⚠️ 待统一化 | onsen 布局下正确，按上游意愿未动，属防回归而非 bug |
| 11 | 手写策略评估器人头计数 `head_count` | ⚠️ 待解决（仅 onsen） | `HandwrittenEvaluator` 只 `impl Evaluator<OnsenGame>`，供温泉 MCTS rollout/leaf 估值；拉面走 `RamenPolicy`、base 走 `HandwrittenTrainer`，均不经此文件 |

> 结论：拉面剧本的「人头」问题已全部清零；剩余未解决项集中在已搁置的温泉剧本与 base 潜伏项，以及拉面侧的死代码 / 测试语义 / 跨剧本防御缺口。详见下文各条目。


## 自动 vs 玩家手动：吃面-训练覆盖差距（玩家 87% vs 自动 52%）

- **日期**：2026-08-26
- **状态**：已解决（2026-08-26，C 方案简化落地 + 弱位 boost 未满修复）
- **问题描述**：同种子（seed=61444, stamina build）对比玩家手动局与推荐策略自动局的吃面-训练行为。决策日志逐回合解析"`RamenSelect` 吃哪碗面 → `Train` 练哪个位"，检查训练位是否落在吃面 `at_trains` 覆盖内：
  - **玩家**：训练 60 回合，吃面 31 碗且全部跟训练，其中 **27/31（87%）训练位在 at_trains 覆盖内**；仅 4 次错配（t4 吃速练力 / t40/t53/t64 吃耐练速）
  - **自动**：训练仅 52 回合（少 8），吃面 28 碗少 3 碗；吃面跟训练 **13/25（52%）覆盖**，12 次错配（t2 吃速练智 / t4 吃智练力 / t7 吃速练耐 / t10 吃智练速 / t13 吃速练耐 / t16 吃智练耐 / t26、t38 吃耐力练智 / t40 吃耐根练速 / t61 吃耐练根 / t64、t66 吃耐练速）
  - speed build 同差异：玩家 90% vs 自动 46%
- **分数**：玩家 stamina 64196 vs 自动 54639（差 −9557）；speed 63851 vs 57110（−6741）。五维差距集中在智位（玩家 1783 vs 自动 973）与训练回合数（60 vs 52）
- **排查过程**：
  - 自动的 `ramen_train_coupling_weight=2.0`（at_trains 覆盖位加分）与弱位 boost 都在，但**加分不足以压过其他训练位的自然高分**（t2 吃速面后智训练 245 分 > 速训练 83 分——智的早期羁绊/动态平衡/弱位偏好加成叠加后仍高于速+耦合）
  - 自动选面（`decide_ramen`）阶段**不前瞻"吃完练哪个位"**——选面独立于训练选择，导致"吃了速面却练智"的错配
  - 玩家行为隐含"吃面→练覆盖位"的两步联动（面选好即锁定最有效的训练窗口）
- **解决方案（C 方案简化，用户拍板）**：
  1. 新增 `LocalRamenConfig.eat_requires_covered_train`（推荐 preset 开启）——`decide_ramen` 对每个吃面候选预演"该面落地后的 `decide_train` 最优动作"（clone + `current_ramen=Some(rid)` + `clear_pending()`，不落地随机分身），最优动作不是该面 `at_trains` 覆盖训练位时把该吃面候选降为 `NEG_INFINITY`。覆盖位判定用完整 `decide_train`（coupling/弱位 boost/体力门禁全部参与），非简单 at_trains 静态匹配
  2. 弱位 boost（`ramen_weak_train_boost` 的 decide_train 分支）与 `ramen_window_alignment` 的 `weak_mult` 补 **"未满"条件**（`five_status[tr] < five_status_limit[tr]`）——已满位只剩 PT 收益，弱位加成本意为培养副属性，对已满位放大只会虚高其训练分（把"无属性价值"的位错抬成最优，测试构造中暴露）
- **验证**：stamina seed=61444 × 10 seed：吃面训练 at_trains 覆盖 **80%→99%**，score_mean **58249→58972（+723）**、skill_pt_mean **7825→7970（+145）**。新增单测 `eat_covered_train_gate_blocks_mismatched_ramen`（局面 A 智满→智面拒；局面 B 其他满智低→智面过/速面拒）；全量 306 passed / 4 failed（pre-existing 基线），零新增失败。**改变拉面模拟数值，基线作废**
- **备选（未采纳）**：A 强化 coupling（软加分不彻底）；B 弱位 boost 限定覆盖位（已随"未满"修复部分吸收）；C 完整 `ramen_lookahead`（clone+ground 分身随机，成本高且旧实验关闭）
- **备注**：本条目是"玩家吃面是否更覆盖有效训练"的直接回答——**是，玩家 87% vs 自动 52%**；修复后自动达 99%。吃面-训练联动是继训练等级马太效应后的第二大分差来源，但单点启发式可修（无需 MCTS）。

---


## 地区选择修正公式落地（你qing 在 at_trains 内独立生效）+ low_count_youqing 弃用

- **日期**：2026-08-25
- **状态**：已解决
- **问题描述**：`score_region` 原公式 `bias_sum × (xunlian × 40 + youqing × 1) + pt × 30 + hint × 15` 让 Y2 id 5（中山-全，youqing=10 覆盖 5 位）天然胜出——`bias_sum` 给高分（5 位总和=6）但实际你qing 仅在该训练位触发一次（per_train=2）；Y3 地区因 `pt_bonus` 几乎全 50、`youqing` 40~60 梯度窄，跨 build 区分度低。先前为缓解"少卡位不被加权"问题引入的 `low_count_youqing` 在全 101 种验证下反转——智向 build 严重受损（-3447）：把 build 次要位（耐/力）对应地区评分压过主训位（智）对应地区（id 9→id 7）。
- **排查过程**（**两阶段修正**）：
  1. 首轮 480 局验证 `low_count_youqing_weight=0.5, k=3` 在 4 个 [player_builds] build 上 3/4 一致正向（加权均值 +337），但仅看 4 build 不够全面
  2. 全 101 种 × 20 局配对验证同参数：52/101（51.5%）build 正向、加权均值 +242，但智向 build 严重受伤（最差 `1sta+2pow+2wis` -3447）；speed/power_wisdom/spd2_gut0 上永远 delta=0（b 选了与 a 完全相同的地区组合，delta=0 来自 RNG 流偏移）
  3. 诊断智向 build 受伤机制：Y2 `id 9`（小仓智，at_trains=[智]，youqing=50）→ `id 7`（京都耐根，at_trains=[耐,根]，youqing=50）；lowc 加权让 `bias=1` 的位（耐、智）对应地区评分压过 `bias=2` 的主训位（智）对应地区——本意是"补 build 不在乎的位"，实际是"反 build 主训位加权"
  4. 用户洞察"少但不是没有"（仅 bias==1 加权，bias=0 排除）——修正后 3 速 build 上 +541 但智向 build 仍受伤，问题不在 bias 数量定义而在加权"次要位 vs 主训位"的根本错向
  5. 重新审视基线：`plain`（无 lowc 加权）的 a_score 在智向 build 上比 lowc50 高 3447 分——说明 `bias_sum` 已经自然把 build 主训位对应地区评分最高，`low_count_youqing` 反而打破这种自然偏向
  6. **第一阶段修正**：提出 `youqing / |at_trains|` 标准化为单格友情加成，让 Y2 id 5 (per_train=2) 自动输给单点 (per_train=50)。全 101 种验证：新公式 1104/2020（54.7%）build 优于旧 plain，加权均值 +160.5；但**`deck_can_split=true`（玩家真实 build）平均 -337 略损**——之前结论被残缺 build 拉高
  7. **用户再次洞察游戏机制**：`region.youqing` 在 `at_trains` 内**每个训练位独立生效**——三点组合 youqing=40 at_trains=[速力智] 在速/力/智**每个**位训练时都获得 +40，**不是分散到 13.3**。`/n_trains` 标准化**方向错误**——应该是"覆盖训练位越多越好"
  8. **第二阶段修正（当前）**：去掉 `max(0.5)` 兜底（无卡位贡献 0），加"无卡位惩罚"（每个 `build 在该位无卡` 的训练位 -10）。公式：`bias_sum_in_at_trains × youqing + pt_bonus×30 + hint×15 - waste×10`。`max(0.5)` 让"覆盖广且含无卡位"地区（Y2 id 5）评分偏高，无卡位惩罚让 id 5 (含 2 无卡位 = 50-20=30) 显著低于 id 9 智单点 (无 waste = 50)
- **解决方案**：
  1. `score_region` 改为：循环 `at_trains`，累加 `bias[t] > 0` 的位贡献到 `bias_sum`，累加 `bias[t] == 0` 的位到 `n_waste`
  2. 返回 `bias_sum × (xunlian×40 + youqing×1) + pt_bonus×30 + hint×15 - n_waste × 10`
  3. 删除 `RamenPolicy::low_count_youqing` 函数（约 26 行）
  4. `decide_region` 移除 `if low_count_youqing_weight != 0.0 { ... }` 分支
  5. `RamenPolicyConfig` 删除 `region_low_count_youqing_weight` / `region_low_count_k` 字段及 default
  6. `RecommendedRamenTrainer::with_region_overrides` 标 `#[deprecated]` 等价 `new()`（保留接口防外部实验覆盖）
  7. region_matrix 加 `SAMPLE_BUILDS` 按 build 名抽样 + `ALL_COMPOSITIONS=1` 跑全 101 种 + `OUT_PREFIX` 分文件保留（避免覆盖）
- **验证**：全 101 种 × 20 局同种子配对（`region-matrix-baseline-plain.csv` 旧 vs `region-matrix-corrected-plain.csv` 新）：
  - **`deck_can_split=true`（种类 ≥ 4，玩家真实 build，420 局）平均 +99.9 正向**——核心指标，说明对真实玩家 build 修正公式确实更优
  - `deck_can_split=false`（种类 < 4，残缺 build，1600 局）平均 -7.3 中性略负——这些是 hint_special/地区分身/finals extra 都失效的残缺 build，不是实际玩家 build
  - 总加权 +15（被残缺 build 拉低）；最差 build `1speed+1stamina+1guts+2wisdom` -2074
  - top 1 build `1speed+2stamina+1guts+1wisdom` +1611（覆盖耐/根/智多点，你qing 40 × bias_sum=3 + waste=0）
  - 各 build 实际选区典型：top 1 Y2 选 id 7（耐根 waste=0）→ Y3 选 id 11/17/19（覆盖 build 实际卡位的三点组合）
  - `test_region_build_sensitivity`（速向 vs 智向选不同 Y3 组合）继续通过
- **备注**：原 `workbench_improve_1.md` §1 规划的 `feeling_yield` / `recipe_balance` / `low_count_youqing_bonus` 三指标中，`low_count_youqing` 思路被"修正公式"取代——`bias_sum × youqing` 隐式表达 build 训练倾向 + `waste` 惩罚排除"覆盖广但无卡位"的反例地区。`youqing / |at_trains|` 标准化方向错误（用户洞察：三点组合 youqing 不分散到各位）。**地区基线作废，需重抓 bench/搜索/MCTS 三处快照**


## 地区选择诀窍类指标弃用（净获得 / 配方失衡 / 吃出碗数）

- **日期**：2026-08-25
- **状态**：已解决（结论 = 弃用）
- **问题描述**：为让地区选择"看出更多东西"（workbench_improve_1 §1），实现了三个组合级指标——全年诀窍净获得（`gain - overflow`）、配方消耗失衡（`recipe_balance`）、卡少训练位 youqing 加权（`low_count_youqing`）。后两者（前者经用户建议改消费端）实测均无区分度。
- **排查过程**：
  1. 最小实验一（同种子配对扫权重）：`feelyield`/`recipebal` 权重扫到百级仍 0/20 改变选择，`lowc` 20/20 改变但平均分 -72
  2. 诊断：第 3 年 120 组合 baseline 总分 top 12 只差 40 分，但净获得全范围仅 40-46（top 内 41-43 恒定）——**总量守恒**（`base_dist` 恒 10/回合 → 总清零 ≈ 总填充/7）
  3. 固定地区 A/B 整局对比（3 速 build 固定卡组，20 局同种子）：
     - 旧数据组：A=60069 vs B=59222（A +847），B 诀窍获得更多、溢出更少却更低分 → 生产端与评分**逆行**
     - 新数据组（第二年 5→7/8→7）：A=58785 vs B=59006（B +221），net/imb 与评分方向不稳定，baseline 静态分也不能预测整局（A 静态全面占优却输）
  4. 消费端（吃出碗数 `meals_eaten`，加库存检查）：A 19 碗 vs B 19 碗**完全守恒**——产能是瓶颈时消费端 = 生产端，依旧守恒
- **解决方案**：三个诀窍类指标全部弃用（代码删除，仅保留观测埋点 `yearly_friend_turns/gauge_gain/gauge_overflow` 与 `region_gauge_diagnostic` 采集工具）。地区选择保留「卡组 bias × 词条」静态打分为主 + `low_count_youqing` 为唯一组合级加权（有真实区分度，参数待调）。验证手段固化：`region_matrix` 的 `FIXED_AB` 固定地区整局同种子对比是唯一可信判据。
- **备注**：数学本质——`base_dist` 总量守恒使生产端/消费端任何口径都≈恒定，±5 是全部信号；地区选择的真实价值由整局动态（训练位实际利用率、吃面窗口）决定，静态公式难以建模，未来可考虑接入整局搜索验证。


## 拉面分身分配「假失败」与「顺序饿死」（概率重试 + 友人卡垫底）

- **日期**：2026-08-24（提交 a223a6e，随 PR #19 于 2026-08-25 合入 main）
- **状态**：已解决
- **问题描述**：两条分身路径（超级拉面 per-卡、地区拉面 per-训练位）原先都是「随机抽一个落点、放不下再抽」的概率重试，失败仅打 diag 日志、不计数不告警。三处问题：
  1. **假失败**：超级侧对 4 个候选位有放回抽样、上限 8 次——只剩一格合法位时 `(3/4)^8 ≈ 10%` 概率明明放得下却放弃；友人卡受「每训练一个友人」约束、合法位常只剩一格，且人头下标最大恒排最后，伤害几乎全落在它身上
  2. **顺序饿死**：六张卡在线贪心放置，前面的普通卡会把后面友人卡的唯一合法位先填满（5 个非 NPC 名额占满）
  3. **地区侧白耗随机数**：原实现「先抽再查满员」，抽到满员位也算一次随机消耗，且满员是规格内跳过、不是分配失败
- **排查过程**：
  - `card_indices` 升序收集，拉面友人卡人头下标必然最大、永远最后分配，只能捡剩位
  - 两条路径各写一份满员 / 挤 NPC 规则，存在规则漂移；入参负数与越界下标有 panic 风险
  - 真无解时若返回 `Err`，会经 `run_distribute` 的 `?` 中断整个回合、作废整次搜索 rollout，且只丢弃无解局面这一类盘面——给搜索样本引入系统性缺失偏差
  - 分身随机直接吃策略流：消耗随重试次数浮动（6~48 次），分配算法的任何改动都会平移同回合后续的训练成败 / 休息 / 外出；地区分身在吃面落地时执行，上游任一处位移都会改掉选卡；MCTS 同回合各候选走过不同路径后策略流长度不同，原方案让各候选的分身随机性互相去相关
- **解决方案**：
  1. 先过滤合法落点再均匀抽取：落点条件分布不变（两者都在合法集上均匀），假失败归零，RNG 消耗从 1~8 次降为 1 次
  2. 友人卡优先分配（稳定排序、普通卡相对顺序不变）；友人先放后普通卡不需回溯（候选位分身前最多 8 个非 NPC，失败要求另外三个候选位各满 5 人共 15 人、矛盾）
  3. 真无解（三友人占三格 + 第四格 5 张本体）跳过该卡、不返回 `Err`——不让无解盘面作废 rollout
  4. 地区侧改为先过滤合法卡再抽；候选收集改按 `PersonType` 扫全体人头
  5. 抽出共用 `can_place_clone`（纯判定）+ `place_clone`（写入），消除两条路径的规则漂移；负数与越界下标由可能 panic 收成拒绝
  6. 分身随机改用按 `(rule_master, turn, TAG)` 派生的局部流（新增两个冻结 tag）：策略流消耗从浮动 6~48 次降为 0，同回合各候选拿到同一份分身随机性以加强配对方差削减；未注入 `rule_master` 的旧路径回退为从父流取一个字派生，保持既有可复现性契约
- **验证**：新增 4 个用例（41 项观测），逐条变异测试——关友人优先 → 可解反例 2 项 NG；绕过局部流 → 父流消耗 NG；退回有放回重试 → 场景 C 在 256 种子中漏放 31 次（12.1%，理论 10.0%）；退回地区「先抽再跳过」→ 场景 3 NG。原场景 C 断言只写「不该有什么」，友人分身没放出来时同样成立，已补「必须放出来」。挤 NPC 分支此前零覆盖，改打原语直接测。Release 全量 243 passed / 0 failed，clippy 警告数未增
- **备注**：**改变拉面模拟数值，既有基线作废；温泉与 base 逐位不变**。关联本文件「超级拉面分身失败是静默的」一条——其问题已随本修复解决（假失败归零）。「拉面规则层四处数值修复」中的训练人数加成、友人卡分身两条属 main 上更早的 08-24 提交，不在本 PR 范围

---

## 拉面杯观测出口局末恒零（scenario_pt / eat_count 未按年归档）

- **日期**：2026-08-24（提交 4fa1ee6）
- **状态**：已解决
- **问题描述**：`scenario_pt` 与 `eat_count` 在每年 RMJ 结算回合（23/47/71）清零，而育成结束在回合 77、72-77 不再吃面，局末读到的两个值**恒为 0**——实测 results.csv 这两列 2100 局全零
- **排查过程**：
  - `minimal_strategy_ab` 已在外部重算绕过，且口径与 `GameOutcome` 分叉（一个是三年合计、一个写着当年）
  - 地区选择同样无逐年归档
  - 年份索引不能用 `current_year()`：回合 23 时它仍判第一年，但那一刻选的是第二年地区
- **解决方案**：归零前按年归档 PT 与吃面次数，并新增逐年地区选择归档；CSV 换成 `scenario_pt_y1..y3` / `eat_count_y1..y3` / `region_y1..y3`，外部重算一并删除。地区归档的年份索引按回合硬编码（2/23/47 → 0/1/2），与阶段分发共用一处映射，顺带移除其中的 `unreachable!`
- **备注**：**纯观测出口，模拟数值逐位不变**——同种子分数与五维在改动前后完全相同，基线不受影响

---

## 搜索层 CRN 测量与 UCB 分配的三处错误

- **日期**：2026-08-24（提交 ae89a18）
- **状态**：已解决
- **问题描述**：
  1. **CRN 收益测量对照轴失效**：原用 `crn_stage_reseed` 分臂，该开关只在温泉路径生效、拉面 rollout 根本不读，两臂输入相同等于没有对照
  2. **CRN 失败样本配对错位**：各候选先各自压缩掉失败样本、再按新下标配对，一侧失败会让此后全部错位一格
  3. **UCB 首组越预算**：首组无条件跑满 `search_group_size`，大于 `search_n` 时超预算且第二阶段立即退出、自适应分配零次；该 clamp 原本只补在弃用工具的外围
- **解决方案**：
  1. 改按「候选间是否共享 `rule_master`」分臂，抽出双种子 rollout 入口（`simulate_common_with_seeds`）拆开决策流与规则主种子；生产路径 `simulate_common` 传相同两值、CRN 语义不变
  2. 按原始序号取双方都成功的交集计算相关与差值方差
  3. 首组步长收进 `search_n`；生产配置 12288 > 2048 场景行为不变
- **验证**：实测独立臂 1.01x（corr 0.002）、共享臂 4.86x（corr 0.77）——CRN 收益真实存在且大幅；生产共享语义与搜索分数逐位不变
- **备注**：无

---

## `[mcts]` 用户配置覆盖静默失效 + 主二进制 onsen 分支搜索参数手抄漏项

- **日期**：2026-08-24（提交 5915057）
- **状态**：已解决
- **问题描述**：
  1. 用户 toml 的 `[mcts]` 原是完整 `MctsConfig`，merge 只拷 `search_n` 与 `radical_factor_max` 两项，**其余 10 项静默失效**；且 serde 会把未写字段填成代码缺省（512/2200/32）而非 `default_config.toml` 的 2048/15000/64——那个残缺 merge 反倒在护着生产参数，改成整段赋值会直接打坏搜索
  2. 主二进制 onsen 分支手抄 8 个搜索字段且**漏了 `crn_stage_reseed`**
  3. `expected_search_stdev` 语义被误读：它是 UCB 探索项的缩放标尺、不是实测统计量
- **解决方案**：
  1. 改为 `OverrideMctsConfig` 全 Option + 逐字段覆盖 + `deny_unknown_fields`，整段 `[mcts]` 可省略；缺配置文件时不走 serde 的手写兜底同步改为全 None
  2. onsen 分支改调既有的 `SearchConfig::new_game_config`（拉面分支与 umaai 早已在用），不再手抄
  3. 补注释说明两处默认值服务不同场景、无需对齐
  4. 顺带修假绿测试：`test_override_config_denies_unknown_fields` 原先因「缺 `[mcts]` 段」报错而非因未知字段通过，补对照用例锁住
- **备注**：日常路径（配置文件存在且内容如现状）逐字段验证数值不变，生产分数不变

---

## 拉面合并动作路径的两处缺陷（搜索层静默清零 / 两次搜索互不可见）

- **日期**：2026-08-25（提交 2c5aea5 / 4432745）
- **状态**：已解决
- **问题描述**：
  1. **合并动作进搜索被静默破坏**：合并动作传进 `apply_root_action` 会被通用 `apply_action` 在 `RamenSelect` 只写 `pending_ramen`、清零 `special_targets`（隐藏风味）且照常返回成功——连和为 3 的非法组合都能悄悄落地；原文档为此明令禁止合并动作进入搜索
  2. **`RamenSelect` 拆成两次独立搜索**：吃哪碗面与用哪些隐藏风味分开搜，前一次搜索看不到后一次的收益；且 `apply_combined_ramen_decision` 此前只有单测跑过、从未在完整育成中执行过
- **解决方案**：
  1. `FlatSearchGame::apply_root_action` 新增合并分支，判别式「`RamenSelect` 阶段 + `StageOnly` + 携带 `special_targets`」三者合起来唯一（三阶段动作不带 targets、special_select 带但只在 SpecialSelect 阶段落地、比赛回合一体化动作 operation 非 StageOnly），转交 `apply_combined_ramen_decision`；补整局冒烟测试
  2. `RamenMctsTrainer` 在 `RamenSelect` 内部自建 `(ramen, targets)` 合并候选一次搜完，缓存 targets 供紧随的 `SpecialSelect` 直接取用。改在训练员内部而非游戏层——`select_action` 契约是返回传入候选的下标，在 `run_ramen_select` 加合并分支会让所有训练员（含手写基线）都收到合并候选；取缓存须早于阶段门控早退（否则 special 门控关闭时搜索选出的 targets 被静默丢弃、分数上看不出来），加缓存命中计数使该结构约定可被测试钉住。开关默认开启，关闭即退回三阶段分别搜
- **备注**：搜索对外层 rng 的消耗由两次降为一次，随机序列整体位移，**拉面基线作废**；三阶段动作行为逐位不变

---

## 拉面搜索阶段缺省缺 `ramen`——每局 61 个吃面决策点生产零搜索

- **日期**：2026-08-25（提交 426694a）
- **状态**：已解决
- **问题描述**：`ramen_search_stages` 缺省只搜 `train`，且 `default_config.toml` / `game_config.toml` 都没写这一项——**每局 61 个 `RamenSelect` 决策点在生产里一次 rollout 都不跑**（实测该阶段平均决策耗时 2.4us，对比 Train 的 120ms）
- **排查过程**：三臂对照，每臂 42 局（7 build × 6 局），同 build 同种子配对：

  | 臂 | 配置 | 耗时 | 均分 |
  |---|---|---|---|
  | A | train n=64 | 8.27s/局 | 55806.6 |
  | B | train,ramen n=64 | 12.29s/局 | **58112.6** |
  | C | train n=96 | 10.95s/局 | 55846.0 |

  - B−A = +2306.0 ± 290.4（t=7.94，39/42 胜，corr 0.8168）；B−C = +2266.6 ± 342.3（等墙钟对照结论不变）；C−A = +39.4 ± 273.9——**train 一侧已饱和**，同一笔算力加到 `train` 的 `search_n` 只值 +39 分
  - 七个 build 全部为正（+491 speed ~ +4522 wisdom），无一反向
- **解决方案**：缺省改为 `"train,ramen"`；`default_config.toml` 同步显式写出该项（附实测注释），bench_base 缺省对齐
- **备注**：**改变 MCTS 生产行为与拉面基线**。测量在 radical_factor=0 下进行，生产的 1.4 一档尚未测。此前缺省改回 `train` 全量测试照常绿的问题，由下条守门测试解决

---

## MCTS 测试地基修补（P0 安全网 + 审查后四处修补）

- **日期**：2026-08-25（提交 3167f2f / c284bb9）
- **状态**：已解决
- **问题描述**：审查 PR 范围内新增测试后发现多处「测试形同虚设」：
  1. `ramen_search_stages` 缺省从 `"train"` 改成 `"train,ramen"`（值 +2306 分）后，把它改回去**全量测试照常绿**（实测 277 passed）——生产配置零防线
  2. 合并候选峰值测试只有 `peak <= 28` 上限，去掉「不吃面」候选后峰值 28→27 仍然绿
  3. `test_combined_vs_threestage_pairing` 测量壳只断言「局数对得上、搜索次数 > 0」，合并搜索完全坏掉（两边都走三阶段）断言仍绿；其 search_n=4 对比口径也已被三臂实验取代
  4. `score_parts().total() == calc_score()` 在当前实现下是转发恒真，容易被误读成硬 oracle
  5. 阶段 one-hot 宽度「正好等于当前阶段数」——将来新增阶段变体就会改掉输入维度、令已落盘教师数据作废
  6. 文档卫生：`MctsTrainer` 死字段 `last_game`、`rollout_batch_size` 整条配置链未接线空转
- **解决方案**：
  1. 新增 `test_production_default_searches_ramen_stage`：同时钉 serde 缺省函数与 `default_config.toml` 两个真值源，按 `RamenSearchStages::parse` 语义判定（"ramen,train" 等价重排不误报）；变异三组——改坏 serde 侧红、改坏 toml 侧红、等价重排绿
  2. 峰值测试改为断言结构恒等式「候选数 == 1 + 三地区各自 targets 数之和」（与 gamedata 数值无关）并补下限防动作空间静默收缩；`assert!` 换 `Checks`
  3. 删除无效测量壳及其四个统计辅助函数
  4. `score_parts` 与 `calc_score` 那条转发断言补注「不是公式 oracle」（真 oracle 是 `expected_score_parts()`）
  5. 阶段 one-hot 预留两个恒零空槽
  6. 删 `last_game`；补动作空间不变量钉死（`special_targets` 之和恒 ≤ 2——budget 公式推出，`validate_special_targets` 为第二道防线——与合并候选峰值上限）、新增 `Uma::score_parts()` 且 `calc_score` 改为对其求和（分量粒度只到 3 项 / 7 项）、温泉 CRN 阶段重播种双向契约测试（避开本就不重播种的 Dig/Upgrade 以免假红）
- **备注**：**输入维度变化（教师数据需重生成），模拟数值逐位不变**；顺带把 `flat_search.rs` 的 anyhow 导入移进 `mod tests`（五处用法全在测试内）

---

## 人头下标与卡组槽位的对应关系在拉面剧本被打破（`person_index < 6` 守卫全线失效）

- **日期**：2026-08-23
- **状态**：已解决（2026-08-23，上游作者已授权我们修）
- **问题描述**：base / onsen 的代码用 `person_index < 6` 作为「这个人头是卡组里的支援卡」的判据，隐含假设 `persons[0..6)` 与 `deck[0..6)` 一一对应。拉面剧本的 `init_persons` 打破了这个假设，但所有 `< 6` 守卫都原样保留，导致理事长被当成卡组第 6 张卡、友人卡被当成「不是卡」。
- **排查过程**：
  - 实际人头布局（`game/ramen/game.rs:48-61`、`game/ramen/state.rs:269-282`）：
    | 下标 | 是谁 | 何时加入 |
    |---|---|---|
    | 0-4 | `card_type < 5` 的 5 张训练卡 | 开局 `init_persons` |
    | **5** | **理事长** | 开局 `init_persons` 末尾 |
    | **6** | **友人卡**（`card_type >= 5`） | 回合 2 `add_friend_and_npcs` |
    | 7-11 | 5 个 NPC | 回合 2 |
    | 12 | 记者 | 回合 12 |
  - 而卡组 `deck[5]` 正是友人卡（`sampler.rs` 与 `bench_config.toml` 的 build 均把友人放末位）
  - 原设计意图的证据：`BasePerson::yayoi()`（`game/base/person.rs:55-65`）把 `person_index` 硬编码成 **6**。这个值随后被 `add_person`（`state.rs:304`）覆盖成实际下标，但它表明原布局是「0-5 六张卡 + 6 理事长」——正是 `< 6` 守卫成立的前提。拉面把友人卡延迟到回合 2 才加，5 张训练卡只占 0-4，理事长顺位落到 5。
  - 受影响的 `< 6` 守卫：
    - `game/ramen/state.rs:314-316` `add_friendship`：羁绊只在 `person_index < 6` 时回写 `deck`
    - `game/ramen/game.rs:469-480` `deyilv`：`person_index < 6` 时读 `deck[person_index]`，否则返回 0.0
    - `game/ramen/action.rs:560-562`：`pidx < 6` 时用 `deck[pidx]` 重算卡效果
  - **实测确认（2026-08-23，`game/ramen/game.rs` 新增两个只读诊断测试）**：
    - `test_person_deck_index_mismatch_full_game`：跑 3 局完整拉面杯，局末 `deck[5].friendship` 恒等于 `persons[5]`（理事长）的羁绊（64 / 100 / 96），与 `persons[6]`（友人卡本人，恒为 100）分叉。开局 `deck[5].friendship = 30`（`initialJiBan`）会被理事长从 0 起算的计数覆盖掉。前 5 张训练卡的两份羁绊始终一致，作为对照。
    - `test_training_buff_index_mismatch`：只把理事长（人头 5）放进训练，`calc_training_buff` 返回的是 **`deck[5]`（友人卡）的完整卡效果**（xunlian=20、saihou=5、fail_rate_drop=15、vital_cost_drop=30、event_effect_up=30、event_recovery_amount_up=60）；只把友人卡（人头 6）放进训练，返回**全零**。
- **后果一（修正：不是「永不解锁」，而是「按理事长的羁绊解锁」）**：`SupportCard::calc_training_effect`（`game/support_card.rs:270-282`）判定固有用的是 `self.friendship`，即卡组那份拷贝，而这份拷贝被理事长的羁绊回写。实测把 `deck[5].friendship` 设为 60 时，理事长所在训练位的 buff 多出 `bonus[1] += 1`、`bonus[5] += 1`（即 `30305 [友]骏川手纲` 的 `uniqueEffectParam = [101, 60, 4, 1, 30, 1]`）；设为 59 时不触发。也就是说友人卡的固有**确实会生效，但阈值读的是理事长的羁绊，且加成落在理事长的人头上**。
- **后果二（核心，此前未列出）**：`traits.rs:335-360` 的 `default_calc_training_buff` 用 `*index >= 0 && *index < 6` 把**人头下标当卡组下标**，拉面未重写该方法（全仓库只有 `traits.rs:335` 一处 `fn calc_training_buff`）。于是：
  - 理事长出现在训练里时，会顶着**友人卡的全部训练加成**参与计算（含 youqing / 失败率下降 / 体力消耗下降）
  - 友人卡本人（人头 6）参与训练时，**训练加成完全为零**
  - 这条直接改变模拟数值，影响面远大于羁绊串位本身
- **后果三（原「后果二」的一半，已排除）**：理事长**不会**走进 `deyilv()`。`traits.rs:206` 是 `let train_type = person.train_type() as usize;`，`train_type()` 返回 `i32`（`base/person.rs:87`），理事长的 `-1` 转 `usize` 后是 `usize::MAX`，`traits.rs:219` 的 `train_type <= 4` 不成立。故「`deyilv(5)` 拿友人卡得意率当理事长的」不成立。
- **影响范围确认**：base（`base/basic.rs:235-247`）与 onsen（`onsen/game.rs:1329-1351`）的 `init_persons` 都把 `deck` 的 6 张卡**按序全部**推入 `persons`，再追加理事长，`< 6` 判据在这两个剧本成立。**本问题是拉面独有的回归，不是三剧本通病。**
- **既有测试把错误行为固化了**：`game/ramen/game.rs:2427` 断言 `deyilv(6) == 0.0` 并注释「person_index >= 6 返回 0」，但下标 6 正是友人卡。需上游确认这是有意还是当初未意识到 6 号是谁。
- **次要影响**：`action.rs:720/734/753` 的 hint 路径同样按 `< 6` 取卡——友人卡（人头 6）拿不到 `hint_count_bonus`，且在 `push_hint_event` 里被当作非支援卡处理（hint_level 固定 1、不出技能提示）。理事长 `is_hint = false`，不进 hint 路径，故这一侧无串位。
- **解决方案**：按 `card_id` 反查（经 GPT-5.6 codex 与 Grok 4.6 两方独立评审，方案一致）。
  1. `Person` trait 新增 `card_id()`，`Game` trait 新增 provided 方法 `deck_index_of(person_index) -> Option<usize>`：由人头的 `card_id` 在 `deck` 里 `position()` 反查槽位，无卡人头（理事长 / 记者 / NPC）返回 `None`
  2. 全部改为反查的调用点：`traits.rs` 的 `default_calc_training_buff`（核心数值路径）与 `distribute_hint`、`ramen/state.rs` 的 `add_friendship`、`ramen/game.rs` 的 `deyilv` 与 `distribute_hint` override、`ramen/action.rs` 的 `handle_hint_event` / `push_hint_event`
  3. **两个下标严格分开**：`is_shining_at` / `distribution` / `EventData` 一律用人头下标，只有 `deck[..]` 用反查出的卡组下标
  4. `persons_mut()` 循环内不能调 `deck_index_of`（借用冲突），改为循环外预抽 `(card_id, hint_prob_increase)`、循环内用 `Person::card_id()` 匹配
  5. `deyilv` 的负数人头下标（`-1`）此前会算出 `deck[usize::MAX]` 而 panic，改用 `usize::try_from` + `Option` 后安全返回 0
- **未采纳的方案**：
  - 给 `BasePerson` 加 `deck_index` 字段：第二份真理，会与 `card_id` 漂移，且要动公开结构体的 `Serialize` / `PartialEq` 与全部构造点
  - 给友人卡预留下标 5：只在「友人恒为 `deck[5]`」时成立，而 `RamenGame::newgame` 只检查卡组**含有**友人卡、不检查位置
  - 拉面 `init_persons` 改成 6 张全入 + `person_is_available` 门控：能让 `< 6` 重新凑效，但地雷仍在，以后再延迟插入任何卡就会复发
- **验证**：两个诊断测试改造为回归测试（`test_person_deck_index_mapping_full_game` / `test_training_buff_person_deck_mapping`）。修复后 3 局完整育成中 `deck[5].friendship` 恒等于 `persons[6]`（友人本人）而非 `persons[5]`（理事长）；只放理事长的训练 buff 全零、只放友人的 buff 等于 `deck[5]` 的卡效果。全量 `cargo test --release` 由 227 passed 增至 228 passed（新增编码器交叉反查回归），0 failed。
- **备注**：
  - `game/base/basic.rs:218` 与 `game/onsen/game.rs:161` 有同样的 `< 6` 回写模式，但这两个剧本的 `persons[0..6)` 与 `deck` 确实同序，不受影响（见上「影响范围确认」）。**本次未改动这两处**——它们当前行为正确，改动属于防回归而非修 bug，为控制上游 diff 面暂缓。
  - 本条由 NN 特征编码器（`game/ramen/features.rs`）的评审发现——编码器原本也照抄了「人头下标与卡组同序」的假设。**订正：编码器侧此前并未真正修正**（`ef99478` 的 changelog 与本文件都误记为已修），本次连同规则层一并改为 `card_id` 反查。
  - **复核（2026-08-25，PR #19 合入后）**：拉面侧修复仍全部有效——`default_calc_training_buff`（`traits.rs` 经 `deck_index_of`）、`add_friendship`（`state.rs:384`）、`deyilv`（`game.rs:484`）、默认 `distribute_hint`（`traits.rs` 按 `card_id` 反查）、`handle_hint_event`/`push_hint_event`（`action.rs:776/787/804`）均未回退，回归测试（`test_person_deck_index_mapping_full_game` / `test_training_buff_person_deck_mapping`）仍在。base/onsen 两处 `< 6` 回写仍未动，两剧本布局下当前正确。
  - `deck_index_of` 的前置条件是 `deck` 内 `card_id` 唯一。`SupportCard::new` 取 `idrank / 10` 作 `card_id`，同一张卡的不同突破（如 `302751` / `302754`）`card_id` 相同；现有构造路径均不去重，重复时会静默命中第一张。默认配置 / sampler / bench 不触发，已在 `deck_index_of` 的文档注释里写明，见下条独立记录。
  - **数值影响**：本修复改变拉面模拟结果，不是静默重构。效果从「理事长出现的位置」搬到「友人出现的位置」，友人固有的解锁时序也随之改变（阈值由理事长羁绊改为友人本人羁绊），训练选择 / 失败 / 体力 / 事件轨迹整体分叉。bench 基线、sampler 根局面、按旧编码器落盘的教师数据全部作废，须重跑。

---

## 拉面人头布局派生的其余上游问题（本次未修，待与上游确认）

- **日期**：2026-08-23
- **状态**：一、二已解决（2026-08-24），三 / 四 / 五待解决（**上游遗留，未与上游确认**）
- **问题描述**：修「人头下标与卡组槽位错位」时，两方评审（GPT-5.6 codex / Grok 4.6）另找出五处同源或相邻的问题。与上游确认的授权范围只覆盖「人头下标当卡组下标」与「`current_effect` 死字段」两条，故这五处**本次一行未改**，在此完整记录，待上游确认后另开。全部已本地 grep 复核属实。
- **一、训练人数加成硬编码「理事长 = 6、记者 = 7」（独立的数值 bug，影响最大）** —— **已修（2026-08-24）**：抽出 `Game::count_training_persons` 改按 `PersonType` 判定，负数与越界下标一并不计；温泉 / base 逐位不变（`test_count_training_persons_onsen_unchanged` 守门），`onsen/game.rs` 的挖土处按上游意愿未动。
  - `game/traits.rs` 的 `default_calc_training_value`：`.filter(|p| **p != 6 && **p != 7).count()`，注释写「包括 NPC 和分身，排除掉理事长和记者」。
  - base / onsen 布局下正确，拉面布局下四项全错：

    | 人头下标 | 是谁 | 应该 | 当前 |
    |---|---|---|---|
    | 5 | 理事长 | 排除 | **计入** |
    | 6 | 友人卡 | 计入 | **排除** |
    | 7 | NPC | 计入 | **排除** |
    | 12 | 记者 | 排除 | **计入** |

  - 拉面的 `calc_training_value` 直接调这个默认实现，人数直接进 `1 + 0.05 × 人数`，误差在数个百分点量级。
  - 修法：改成按 `PersonType` 排除 `Yayoi | Reporter`。`game/onsen/game.rs` 的挖土处有同样的 `*p != 6 && *p != 7`，onsen 布局下碰巧正确，建议一并改成类型判断。（`ramen/action.rs` 的诀窍槽已修过同样的问题，训练人数加成漏了。）
  - **注意**：上游作者原话「理事长记者位置需要反查」很可能指的正是这一条——`记者` 在「人头下标当卡组下标」里根本不参与（记者是人头 12、无卡，反查恒为 `None`），唯一同时涉及理事长与记者的就是这个过滤器。下次沟通需要澄清。
- **二、超级拉面分身漏掉友人卡** —— **已修（2026-08-24）**：候选遍历改 `0..persons.len()`；同时给分身补上「每训练一个友人」约束（见下文独立条目）。地区分身按其「不含友人卡」语义**未动**。
  - `game/ramen/action.rs` 的 `card_indices: Vec<i32> = (0..6i32)`：注释写「获取所有支援卡索引（含友人卡，index 0-5）」，但 `0..6` 里没有下标 6，**友人卡永远不参与超级拉面分身**。
  - `game/ramen/game.rs` 的地区分身同样是 `(0..6i32)`，但它只取 `PersonType::Card`：友人在卡组末位时 0-4 恰为训练卡、5 是理事长被类型过滤掉，**碰巧正确**；友人不在末位时会漏卡。
  - 修法：两处都改成遍历 `0..persons.len()` 再按 `PersonType` 过滤。
- **三、Hint 路径的无守卫 `deck[..]` 访问（潜伏越界）**
  - `game/base/basic.rs` 与 `game/onsen/game.rs` 的 hint 路径仍有 `deck[person_index]` 形式的无守卫访问，`deck` 长度恒 6，人头下标 ≥ 6 会直接越界 panic。
  - 拉面侧本次已随「人头下标」修复一并收紧（改反查 + 无卡人头取人头自己的名字）；base / onsen 两处未动。
  - 生产路径上 `distribute_hint` 只给 `PersonType::Card` 打 hint，目前踩不到，属潜伏项。另 `handle_hint_event` 的随机分支没有像 hint_special 分支那样过滤 `PersonType::Card`。
- **四、`BaseGame::new` 不校验卡组内 `card_id` 重复**
  - `SupportCard::new` 取 `idrank / 10` 作 `card_id`，同一张卡的不同突破（如 `302751` / `302754`）`card_id` 相同。
  - 现有构造路径一处都不去重：`BaseGame::new` 无校验、`validate_game_config` 只查 `len() == 6`、`RamenGame::newgame` 只查有无友人卡。
  - 本次新增的 `Game::deck_index_of` 依赖 `card_id` 唯一，重复时会静默命中第一张。默认配置 / sampler / bench 实际不会重复，任何手写卡组都能触发。
  - 本次只在 `deck_index_of` 的文档注释里写明前置条件，**未给上游的 `BaseGame::new` 加 `bail!`**（那是给上游加校验，超出授权范围）。
- **五、`ramen/action.rs` 注释掉的调试块仍含旧假设**
  - `/* */` 块里仍有 `if pidx < 6 { game.deck[pidx] }`。目前是死代码，但解开注释会把友人卡打成「非支援卡」。
- **其他待办（非 bug，属清理）**：
  - `game/ramen/rng_consistency.rs` 的 `run_turns` 用 `for i in 0..6` 同时锁 `deck[i]` 与 `persons[i]` 的羁绊。拉面下 `persons[5]` 是理事长、友人卡（人头 6）没被锁到。当前不影响该测试要验证的结论（分布权重走 `deyilv` → 读 `deck`，6 张卡都锁到了；友人卡靠 group buff 闪彩、不看羁绊），但语义上应改成按 `PersonType::Card | ScenarioCard` 遍历。
  - ~~`features.rs` 的 `person_train_slots` 对分身仍是 last-write one-hot~~ —— **已修（2026-08-23）**。本文件「NN 特征编码器首版的五处编码缺陷」原记为已改 multi-hot，实际仍是 `slots[idx] = Some(t)` 后写覆盖（Codex 评审独立复现）。已改为 `Vec<[bool; TRAIN_NUM]>` 掩码，cards 段与 persons 段同步改为逐位写标志，维度不变。该文件是我们自己的代码，不涉及上游授权范围。
- **复核（2026-08-25，PR #19 合入后）**：一（`count_training_persons` 按 `PersonType` 判定）、二（分身候选全扫 + 每训练一个友人约束）修复仍有效，回归测试（`test_count_training_persons_*` 三个用例）仍在；三（base `basic.rs` hint 路径的 `*p < 6` + `deck[*p]`、onsen hint 路径的 `deck[person_index]`）、四（`validate_game_config` 仍只查 `cards.len()==6`，无 `card_id` 重复校验）、五（`ramen/action.rs` 的 `/* */` 调试块仍含 `pidx < 6` + `deck[pidx]`）均原样未动，与本记录一致；onsen 挖土处（`game.rs:342` 的 `*p != 6 && *p != 7` 过滤器）按上游意愿仍未动，onsen 布局下正确；rng_consistency 待办也未改。**注：「手写策略评估器的人头计数」一项经核实只作用于温泉路径**——`HandwrittenEvaluator` 仅 `impl Evaluator<OnsenGame>`（`handwritten_evaluator.rs:593`），拉面手写策略走 `RamenPolicy`（`RamenHandwrittenTrainer`），与该文件无关，详见下文独立条目。

---

## 超级拉面分身失败是静默的（无计数、无告警）

- **日期**：2026-08-24
- **状态**：已解决（2026-08-24，提交 a223a6e，随 PR #19 合入 main）
- **问题描述**：`ramen/action.rs` 的 `distribute_super_ramen_clones` 对每张支援卡最多重试 `option_trains.len() * 2` 次，全部失败时只走 `diag!(">> 超级拉面分身失败: ...")`，不返回 `Err`、不计数、不进任何统计。生产跑批时分身丢失完全无声。
- **排查过程**：
  - `choose(rng)` 是有放回采样，选项二只有 4 个训练位 → 8 次重试。若 4 个位里有 3 个被堵死，全落空概率约 `0.75^8 ≈ 10%`。
  - `card_indices` 升序收集，拉面友人卡人头下标必然大于全部训练卡（训练卡开局占 0-4，友人卡回合 2 才加入），**友人卡永远最后一个分配**，只能捡剩位，失败概率系统性最高。
  - 2026-08-24 给分身补了「每训练一个友人」约束后，友人卡又多了一条拒绝理由。生产中理事长 / 记者 `absent_rate = 200`，多数回合不在场，估计失败率在 1% 以下，但非零且不可观测。
- **解决方案**：按「拉面分身分配『假失败』与『顺序饿死』」条目的修复一并落地：
  1. 假失败归零——改为先过滤合法落点再均匀抽取，只剩一格合法位时不再以 `0.75^8 ≈ 10%` 概率放弃，RNG 消耗降为 1 次
  2. 友人卡优先分配，消除「永远最后分配、只能捡剩位」的系统性劣势
  3. 真无解（三友人占三格 + 第四格 5 张本体）时跳过该卡、不返回 `Err`——返回错误会中断整个回合并作废整次搜索 rollout；跳过仍无计数，但已不再静默丢失合法分身
  4. 原「把失败次数计入 bench 统计」的方案未采纳：假失败归零后真正无解只剩构造性盘面，跳过即可，不额外引入统计接线
- **备注**：这个函数在 2026-08-24 之前零测试覆盖，「友人卡分身从未生成」正是靠静默藏了三个月；本次随修复补了 4 个用例（41 项观测）与变异测试，挤 NPC 分支零覆盖问题一并解决。

## 手写策略评估器的人头计数未随人数加成一并修正

- **日期**：2026-08-24
- **状态**：待解决
- **问题描述**：`neural/handwritten_evaluator.rs` 的 `let head_count = game.distribution()[train].len();` 与 `default_calc_training_value` 是同一个「人数」概念，但既不排除理事长 / 记者，也不过滤负数，还是裸下标索引（`distribution` 未初始化会直接 panic，而 `count_training_persons` 返回 0）。
- **排查过程**：该值用在三处阈值判断（`head_count >= 3`、`head_count >= ABSENT_HEAD_THRESHOLD + 1`），理事长在场会让阈值提前触发。同一文件的 `other_max_head` 用 `distribution()[t].len()` 亦然。
- **解决方案**：改用 `Game::count_training_persons`。影响的是**手写策略打分依据**而非模拟数值，故与 2026-08-24 的两处数值修复分开，不混进同一次改动。
- **复核（2026-08-25，PR #19 合入后）**：仍未解决——`handwritten_evaluator.rs:503` 的 `head_count = game.distribution()[train].len()` 与 `:555` 的 `other_max_head` 原样保留，三处阈值判断（`:560` 前期智力攒羁绊、`:565` 多人头部分解锁、`:571` 被迫选择）仍在用裸分布长度（含理事长 / 记者 / 未初始化 panic 风险）。**影响范围订正：该评估器只服务温泉路径**——`HandwrittenEvaluator` 仅 `impl Evaluator<OnsenGame>`（`handwritten_evaluator.rs:593`），供温泉 MCTS 的 rollout 与 leaf 估值使用；拉面手写策略是 `RamenHandwrittenTrainer`（`RamenPolicy`），base 是 `trainer/handwritten_trainer.rs` 的另一套，均不经由此文件。修复不涉及本 PR 改动的文件，维持待解决（影响面限已搁置的温泉剧本）。
- **备注**：属「人头下标当卡组下标」的同族残留——2026-08-23 修了取值型，2026-08-24 修了计数型与遍历型的模拟侧，策略侧的计数型漏了。

---

## `RamenGame::current_effect` 是从未被写入的死字段

- **日期**：2026-08-23
- **状态**：编码器侧已解决（2026-08-23）；**上游字段本身保留，待上游确认**
- **问题描述**：`RamenGame.current_effect: RamenEffect`（`game/ramen/state.rs:159-160`）的文档写「当前生效的拉面效果（每回合重新计算）」，但全仓库对它的写入只有 `newgame` 里的一次 `RamenEffect::default()`（`state.rs:237`）。它在整局中恒为全零。
- **排查过程**：
  - `grep -rn "current_effect" crates/umasim/src/` 的结果只有两行：`state.rs:160`（字段声明）与 `state.rs:237`（初始化为 `default()`）。没有任何赋值点。
  - 真正的拉面效果由 `game/ramen/effects.rs` 的 `calc_ramen_training_effect` 按**训练位**现场计算，算完直接用，从不写回 `current_effect`。
  - 之所以从来没暴露：没有任何读取方依赖它，所以恒零不影响模拟结果。
  - 触发发现的场景：NN 特征编码器照着字段名和文档，为它保留了 14 维特征，结果这 14 维在所有样本里恒为 0。
  - 设计上也确实**装不下**：`youqing` 只在友情训练时非零，`xunlian` 随训练位不同，不存在一份能代表所有训练位的单一「当前效果」。
- **解决方案**：待与上游确认。两个方向：
  1. 删除该字段（推荐）——没有读取方，删掉零风险，同时消除误导
  2. 若确实想要缓存，需改成「每训练位一份」并在 `run_distribute` 后填充；但这等于把派生量塞进状态，与「状态只存原始量」相悖
- **本次处理（2026-08-23）**：只删编码器这一侧——移除 `features.rs` 的 `G_EFFECT` 常量、`effect` 特征块与 `encode_effect` 函数，`GLOBAL_DIM` 由 166 降至 152、`INPUT_DIM` 由 766 降至 752。**上游的 `RamenGame::current_effect` 字段与 `RamenEffect` 类型未动**：删除公开字段会扩大与上游的合并冲突面，而 14 维恒零的实际危害完全在编码器侧，本地删干净即可。字段的去留留给上游决定。
- **备注**：`features.rs` 的模块文档已加「已知未覆盖」一节记录此事——若上游后续真正填充该字段，需要重新加回特征块并给样本打新的 schema 版本号。

---

## NN 特征编码器首版的五处编码缺陷（自查 + 三方评审）

- **日期**：2026-08-23
- **状态**：已解决（2026-08-23）
- **问题描述**：`game/ramen/features.rs` 首版存在五处会污染教师数据的编码缺陷。教师数据一旦落盘，编码错误无法回溯修正，故在接入 Phase 3 之前全部修掉。
- **排查过程**：派 Grok / Codex / Gemini 三方并行评审首版实现（Gemini 三次均因 agy 打开不存在的路径而硬失败，未产出）。Grok 与 Codex 的发现互补，逐条经本地 grep 复核后确认成立：
  1. **友人卡与人头错配**（Grok 提出）：编码器假设「人头下标 0..5 与卡组槽位同序」，实际下标 5 是理事长、友人卡在 6。卡槽 5 会读到理事长的训练位，人头 6 的卡链接丢失。根因是上游的布局问题，见本文件「人头下标与卡组槽位的对应关系在拉面剧本被打破」一条。
  2. **分身被后写覆盖**（Grok 提出）：分身不新建人头，而是把同一 `person_idx` 再 `push` 进另一个训练位（`game/ramen/game.rs:947`、`:960`），人头会同时出现在 `distribution` 的多行。`person_train_slots` 用 `slots[idx] = Some(t)` 后写覆盖，只保留编号最大的训练位——彩圈分身在特征里直接消失。
  3. **`current_effect` 14 维恒为 0**（Grok 与 Codex 独立提出）：见本文件对应条目。
  4. **`selected_regions` 默认值被当真数据**（Grok 与 Codex 独立提出）：默认 `[0,0,0]`，第 1 年地区选择（回合 2 的 `run_begin` 内联）之前会把地区 0（札幌-速）的效果编三遍。**注意：这不是游戏 bug**——`game/ramen/game.rs:118` 用 `turn < 2` 跳过 `RamenSelect`、`:262` 用 `turn >= 2` 门控候选面，游戏逻辑在回合 2 之前从不读该字段，只有编码器会读。
  5. **Train 阶段 `pending_ramen` 与 `current_ramen` 重复编码**（Codex 提出）：`ground_ramen_effects`（`game.rs:775-778`）设置 `current_ramen` 后不清 `pending_ramen`，`clear_pending()` 要到下一回合 `run_begin`（`game.rs:1109`）才调用。对游戏无害，但编码器把同一个选择编了两遍，其中 pending 已是语义上的残留。
- **解决方案**：
  1. 卡与人头的对应改为按 `card_id` 双向 `position()` 反查，不再依赖下标相等
  2. 训练位编码由 one-hot 改 multi-hot（维度不变），正确表达分身 —— **订正：此条在 `ef99478` 同样未落地**，`person_train_slots` 直到 2026-08-23 才真正改为 multi-hot 掩码
  3. 移除 `current_effect` 的 14 维块
  4. 新增 `regions_ready` 掩码位，未就绪时地区块整体写 0，不查 id 0
  5. Train 阶段只编 `current_ramen`
  6. 另按评审意见调整：归一化尺度改用真实上限（`scenario_pt` 1000→5000、`five_status` 1200→2800 等）；`onehot` 拆成 `onehot_optional`（合法缺席填 0）与 `onehot_checked`（非法下标 `bail!`），与模块文档的约束一致；补上地区的诀窍配方 `region_feeling`
- **备注**：
  - **订正（2026-08-23）**：上表的解决方案第 1 条（卡与人头按 `card_id` 双向反查）与第 3 条（移除 `current_effect` 的 14 维块）在 `ef99478` 里**实际并未落地**，本文件与 changelog 都误记为已完成。两条在「人头下标与卡组槽位错位」的修复中才真正实施。第 2、4、5、6 条在 `ef99478` 已落地，不受影响。
  - Codex 另给出了 Phase 3 接线的具体障碍清单（`training_sample.rs:17-38` 的 1121/587/89 常量与 `assert_eq!` 校验、`sample_collector.rs` 的 `OnsenAction` 映射、`search/result.rs:278` 只为 `SearchOutput<OnsenAction>` 实现、`collector.rs` 的 `ShardWriter` 未泛型化）。结论：拉面必须单开 schema，不能原地改温泉常量。留作 Phase 3 输入。
  - Codex 建议把 `remaining_race_slots` 从 `game/ramen/policy.rs` 挪到 `Uma` / `BaseGame` 层——它只处理 `FreeRaceData::mask` 与通用自选比赛回合范围，与拉面策略权重无关，现在让编码器反向依赖了策略模块。认同，待办。
  - `sampler.rs:38-40` 的模块注释仍写 `default_config.toml` 是 `fixed`，提交 `875f61c` 已改回 `all`，注释与事实相反，待订正。

---

## distribute_person 中"不出现"判定受得意率影响

- **日期**：2026-08-19
- **状态**：已解决（2026-08-25）
- **问题描述**：当前 `Game::distribute_person`（`traits.rs`）将"不出现"判定和"训练位置分配"混在一起，不在率 = `absent_rate / (500 + absent_rate + deyilv)`，导致得意率会影响"不出现"概率。按剧本原始规则，"不出现"概率应不受得意率影响，得意率只影响训练位置的权重分配。
- **排查过程**：
  - 用户给出剧本原始算法：
    1. 用基础权重 [100,100,100,100,100,absent_rate] 判定"不出现"，概率 = `absent_rate / (500 + absent_rate)`（**不含得意率**）
    2. 判定为出现后，按 [100+deyilv, 100, 100, 100, 100]（不含"不出现"项）随机分配训练位置
  - 当前算法：
    - 不在率 = `absent_rate / (500 + absent_rate + deyilv)`（含得意率）
    - 训练位置按 [100+deyilv, 100, 100, 100, 100, absent_rate]（含"不出现"项）分配
  - 关键差异：得意率会拉高"不出现"判定概率（deyilv 越大，不出现概率越低）——这是错误的
- **解决方案**：2026-08-25 按用户确认的**实际规则**落地（用户补充了与文档记载不同的细节）：
  1. **两步判定**（`traits.rs` `distribute_person` 重写）：第一步先判「不在」，概率 = `absent_rate / (500 + absent_rate)`，**不含得意率**，判定不在即返回 -1；第二步判定出现后才按 `[100+deyilv, 100, 100, 100, 100]` 分配训练位。不在判定不再调用 `deyilv`（旧实现判定前就调了，有写卡效果副作用的浪费）
  2. **不在权重类型表**（新 `Game::absent_weight`）：支援卡 `50 - absent_rate_drop`、友人/团队卡 `100 - absent_rate_drop`、**理事长/记者固定 200**（不受 `absent_rate_drop` 影响）、NPC 0（必定出现）
  3. **NPC 必定出现双保险**：`absent_weight` 对 Npc 返回 0 + `distribute_all` 对 NPC 传 `allow_absent=false`（采纳用户建议：把「NPC 无不在率」作为拉面规则放在分布入口，而非只写死在类型表）
  4. **不在记录**（用户需求 3）：所有被判定不在的人头（**含理事长/记者**，简化后剧本侧再按类型筛）经新的 `Game::record_absent_person` 写入 `RamenState.absent_cards`，`run_distribute` 每回合 `distribute_all` 前清空，供后续剧本机制使用
  5. **地区分身缺席优先**（用户需求 4，`distribute_region_clones`）：本回合判定「不在」的支援卡（`PersonType::Card`）按缺席记录顺序优先补进 `at_trains` 分身位（先缺谁补谁，放不下顺延）；全部支援卡都在训练后，剩余分身位才随机复制在场支援卡。示例：region 15（中山-速力智，`at_trains=[0,2,4]`）且有支援卡 1 缺席 → 位置 0 补卡 1
- **验证**：新增 4 个单元测试（不在权重类型表 / `AlwaysTrueRng` 下两步行为验证 / 24 轮集成：NPC 必现 + 记录互斥 + 理事长入记录 / 地区分身缺席优先三场景）。`AlwaysTrueRng` 测试中踩到 rand 0.9 的坑：恒 0 RNG 会让均匀整数采样的 Canon rejection（`lo >= thresh`）无限重试，改返回 1 并注释。**改变拉面模拟数值**：bench seed=42 基线 51731→52739（不在率不含得意率 + 缺席卡优先补分身位），三处逐位基准快照已重抓；全量 279 passed / 0 failed。该修复落在 `distribute_person` 默认实现上，**对 base/onsen 同样生效，属剧本通用正确行为，无需为其重抓基线**（用户确认）
- **备注**：2026-08-19 曾确认暂不动 absent_rate 相关逻辑（涉及 `absent_rate_drop` 等其他领域知识），当时只修复了 `RamenGame::deyilv` 缺剧本加成的问题；本次按用户确认的实际规则一并落地。`absent_cards` 目前无消费方（用户需求 3 的「后续剧本计算」待继续描述）。

---

## 第三年地区选择组合过多

- **日期**：2026-08-17
- **状态**：已解决（2026-08-20）
- **问题描述**：第3年可选地区为10-19共10个，C(10,3)=120种组合，动作空间过大，影响搜索效率
- **排查过程**：
  - 第1/2年各5个地区，C(5,3)=10种组合，可接受
  - 第3年120种组合导致 Trainer 需要评估120个动作，计算开销显著增加
- **解决方案**：第3年地区选择默认 Fixed，走固定组合 `[[11,14,15]]`，跳过 120 组合枚举（2026-08-20 实现，见 changelog）；后续如需动态策略，可再按"先定主方向、再子集内选组合"的预筛选方案扩展
- **备注**：第3年地区还包含pt_bonus效果，选择策略需要同时考虑youqing/pt_bonus和配方匹配

---

## Ubuntu 下 umaai 二进制图标方案待定

- **日期**：2026-08-17
- **状态**：待解决（暂缓）
- **问题描述**：umaai 的 Windows 版通过 build.rs + winscribe 把 `.ico` 图标嵌入二进制；Ubuntu 下 ELF 二进制没有 Windows 资源节的图标嵌入机制，需要确定 Linux 版的图标落地方式
- **排查过程**：
  - Linux ELF 无原生图标资源节，无法直接嵌入 `.ico`
  - 候选方案一：`.desktop` 文件 + 外置 PNG/SVG 图标（桌面菜单展示，Linux 惯例）
  - 候选方案二：`include_bytes!` 把 PNG 数据嵌入二进制（自包含、单文件分发，需补充 PNG 资源）
  - 已顺带修复 umaai 的 Linux 构建：winscribe 改为仅在 `cfg(windows)` 目标下作为 build-dependency；build.rs 的 Windows 资源编译逻辑用 `#[cfg(windows)]` 包裹；`windows` crate 及其依赖链（windows-future 0.3.2 与 windows-core 0.62.2 不兼容，Linux 上编译失败）改为 `cfg(windows)` 限定依赖；补声明 Linux 专用的 `libc` 依赖；Linux 版 `get_stack_size` 补 `pub` 修饰
- **解决方案**：暂不处理，待用户明确图标使用场景（桌面菜单展示或二进制自包含）后再实施
- **备注**：umaai 为 CLI + TUI 程序（clap + ratatui），桌面图标应用场景有限

---

## constants.json 排名数据需人工更新

- **日期**：2026-08-19
- **状态**：已解决（2026-08-20，数据经用户确认）
- **问题描述**：constants.json 中的 `rank_scores`、`rank_names`、`five_status_final_score` 为游戏排名相关数据，当前数值可能已过期，需要稍后按最新游戏版本人工核对更新
- **排查过程**：配置系统整理（Phase 2）甄别 constants.json 各项归属时确认：这三项属固定游戏数据、不随剧本变化，保留在 constants.json，由人工更新
- **解决方案**：用户提供最新数据后更新三个数组（提交 `aa756d9`）：`rank_scores` / `rank_names` 补齐至 LS24（共 298 档，速度档位上调），`five_status_final_score` 同步核对（3399 档）
- **备注**：与配置整理方案一致，见 .trae/documents/config_refactor_plan.md

---

## 友人事件效果未应用「事件效果提高」「恢复量提高」词条

- **日期**：2026-08-19
- **状态**：已解决（2026-08-20）
- **问题描述**：友人卡的支援卡词条「事件效果提高」（`event_effect_up`）和「恢复量提高」（`event_recovery_amount_up`）当前在游戏逻辑中没有被应用到友人事件上。`FriendState` 正确读取并保存了这两个字段（`event_bonus`、`vital_bonus`），但所有 `apply_event` 路径都没有引用它们——base/onsei/ramen 三个剧本都是如此。这导致友人事件（登场/点击/解锁/出行 1-5）的实际效果与支援卡词条描述不符。
- **排查过程**：
  - `crates/umasim/src/game/mod.rs:138-158` `FriendState::new` 从 `card.card_value[rank].event_recovery_amount_up` / `event_effect_up` 读取并写入 `vital_bonus` / `event_bonus`。
  - 全仓 grep `event_bonus` / `vital_bonus` / `event_effect_up` / `event_recovery_amount_up`（排除 `features[...]` 神经网络特征值）：
    - `event_effect_up` / `event_recovery_amount_up` 只出现在 `FriendState::new` 的读取、`SupportCardValue::explain`（仅打印描述）、`onsen/game.rs:1328-1329`（神经网络特征归一化）
    - `friend.vital_bonus` / `friend.event_bonus` 在游戏逻辑（`apply_event` 路径）**完全无引用**
  - `apply_event` 调用链：
    - `RamenGame::apply_event`（`game.rs:378`）→ `self.base.apply_event(event, choice, rng)` → `BaseGame::apply_event`（`base/mod.rs:151`）→ `self.uma.add_value(&choice_result.value)` 直接结算效果，**未乘 `friend.event_bonus` / `friend.vital_bonus`**
    - `OnsenGame::apply_event` / `BasicGame::apply_event` 同理，未引用 friend bonus
  - 影响范围：所有剧本（包括拉面杯、温泉、基础）的友人事件；只有 `FriendCardState` 为 `SSR`/`R` 的卡组才会触发
- **解决方案**：用户已确认精确语义，最终在 `BaseGame` 统一修复：
  1. `BaseGame` 新增 `friend_event_ids: HashSet<u32>` 字段；`BaseGame::new` 从 `global_events().friend_events.values()` 派生 base/onsen 友人事件 ID；`RamenGame::newgame` 额外 extend `RAMENDATA.friend_events.values()` 合并 ramen 友人事件 ID
  2. `BaseGame::apply_event` 在结算前判定 `friend_event_ids.contains(&event.id)`，命中则调用新增的 `apply_friend_bonus` 私有方法
  3. `apply_friend_bonus` 按用户确认语义乘算：`status_pt[i] * (100 + event_bonus) / 100`（floor）仅作用于 `status_pt[0..6]`；`vital * (100 + vital_bonus) / 100`（仅 `vital > 0`）；不影响 `max_vital` / `motivation` / `hint_level` / `friendship`
  4. base / onsen / ramen 三剧本统一受益，trait override (`BasicGame/OnsenGame/RamenGame::apply_event`) 无需改动
  5. `event_bonus == 0 && vital_bonus == 0`（未携带友人卡）时分支跳过，行为与现状一致
  6. 新增 7 个单元测试（`test_apply_friend_bonus_*` × 5 + `test_apply_event_*_integration` × 2），全 124 lib 测试通过
- **备注**：数据结构（`EventCollection` / `RamenScenarioData`）未修改，所有友人事件 ID 集合从 `friend_events.values()` 在 `BaseGame::new` / `RamenGame::newgame` 时派生，O(1) HashSet 查询；同时删除与本次修改冲突的 `test_ramen_region_strategy_fixed_skips_enumeration` 测试。

---

## stable 工具链下 cargo fmt 破坏 Nightly 格式

- **日期**：2026-08-21
- **状态**：已解决（2026-08-22，手动执行方案；钩子自动化已撤销）
- **问题描述**：`rustfmt.toml` 使用 8 个 Nightly-only 选项（`imports_granularity` / `group_imports` / `trailing_comma` / `wrap_comments` 等），在 stable 工具链执行 `cargo fmt` 会静默忽略这些选项，把整个仓库重排成 stable 风格——与 git 历史 `ffddd1a`（应用仓库 rustfmt 格式）的 Nightly 格式不一致，产生大量无关 diff
- **排查过程**：
  - `rustfmt.toml` 含 `imports_granularity = "Crate"`、`group_imports = "StdExternalCrate"`、`trailing_comma = "Never"` 等 Nightly 特性，stable rustfmt 不支持且无报错（仅 warning）
  - 环境无 `rust-toolchain` 文件、无 Nightly 工具链（`rustup toolchain list` 仅 stable），`cargo fmt` 实际以 stable 行为执行
  - 实际触发：bench 重构时执行 `cargo fmt` 误将 55 个文件、约 1900 行重排为 stable 风格，已全部还原
  - 2026-08-22 进一步发现：Nightly 为滚动版本，`ffddd1a` 格式化时与当前 nightly（rustfmt 1.10.0-nightly 2026-08-20）规则存在漂移（trailing_comma 去尾逗号、imports 合并、多行表达式压缩等），`cargo +nightly fmt --all -- --check` 报 40 文件 387 处差异
- **解决方案**：
  1. AGENTS.md 固化规则：项目使用 **Nightly** 格式规则，stable 工具链下**禁止执行 cargo fmt**
  2. **cargo fmt 只能由用户手动执行**（2026-08-22 更新）：AGENTS.md 明确「禁用 cargo fmt」——格式化由用户手动执行（`cargo +nightly fmt --all`），Agent 不执行 fmt，避免强制重新读取代码；编译仍用 stable，互不影响
  3. **钩子自动化已撤销（2026-08-22 用户决策）**：cargo-husky 依赖、`.cargo-husky/hooks/pre-commit` 与生成的 `.git/hooks/pre-commit` 均已移除，不再自动检查格式——从源头防止 stable fmt 改为**流程约定**（提交前用户手动跑 nightly fmt）；全库已应用当前 nightly 格式（提交 `fd144af`，42 文件），该次格式化保留
- **备注**：Nightly 为滚动版本，rustfmt 输出偶有细微变化（本次漂移即一例）；如需完全固定可锁定指定日期（如 `nightly-YYYY-MM-DD`）或引入 `rust-toolchain.toml` 固定工具链

## game_config.toml 从未被加载（路径 bug）+ [config_override] 字段不合并

- **日期**：2026-08-22
- **状态**：已解决（2026-08-22）
- **问题描述**：用户在 `game_config.toml` 修改 `uma`/`cards`/`extra_count` 后实际不生效——`load_game_config` 合并结果仍是 default 值（`uma=102601`、`extra_count=[0;6]`），用户配置形同虚设
- **排查过程**：
  - 临时诊断测试（load_game_config 打印合并结果）发现 `extra_count=[0;6]` ——这正是「用户配置不存在」兜底分支的默认值，证明走了兜底而非正常解析
  - 根因一（路径）：`USER_CONFIG_REL_PATH = "../game_config.toml"`（相对 `gamedata/` 的语义，Phase 2 步骤 4 引入），但 `resolve_user_config_path` 用 `current_dir().join(..)` 拼接，解析为「工作目录上一级」——文件不存在 → `cfg_path.exists()` 为 false → 永远走兜底，game_config.toml 从未被解析
  - 根因二（字段）：`OverrideConfig` 只有 `extra_count` 等 7 个字段且均必填（无 `serde(default)`），`uma`/`cards`/`blue_count` 不在其中——即便路径修复，这些字段也会被 serde 静默忽略；且 merge 无条件覆盖（缺写时用兜底值覆盖 default，与「只写要改的项」注释语义冲突）
  - 兜底机制（用户配置不存在时静默回退）掩盖了此 bug：程序一直正常跑，只是配置从未生效
- **解决方案**：
  1. 路径修复：`USER_CONFIG_REL_PATH` 改为 `"game_config.toml"`（工作目录根，与注释语义一致）
  2. `OverrideConfig` 全字段 `Option` 化（`#[serde(default)]`，`None` = 不覆盖）：新增 `uma`/`cards`/`blue_count`，现有 `extra_count`/`mcts_selected_onsen`/`log_level`/`num_threads` 改可选；merge 全部 `if let Some` 覆盖——真正实现「只写你要改的项」
  3. 加固：`#[serde(deny_unknown_fields)]`——拼错/未支持的字段显式报错，杜绝静默忽略
  4. `game_config.toml`：顶层 `mcts_selected_onsen`（原在遗留段，同样静默失效）移入 `[config_override]` 段；注释更新可选覆盖语义
  5. 测试 +4：merge 全 None 不覆盖 / 部分覆盖生效 / deny_unknown_fields 报错 / 用户配置路径定位 + 真实文件合并集成验证（`uma=100901` 生效）
- **备注**：`ramen_region_strategy` / `ramen_region_fixed`（game_config.toml 注释中的「顶层覆盖」）同样不在 `OverrideGameConfig` 结构内，目前无法通过 game_config.toml 覆盖（未启用，暂缓处理）；`OverrideGameConfig` 顶层未知字段（如遗留的顶层 `mcts_selected_onsen` 写法）仍静默忽略，如需可后续加 deny

---

## 第2/3年地区选择无 build 自适应（score_region 对无 xunlian 的地区无区分度）

- **日期**：2026-08-22
- **状态**：已解决（2026-08-23）
- **问题描述**：bench 已支持不同卡组（build）跑批，但第 3 年地区选择仍是固定值 `[[11, 14, 15]]`（default_config.toml `ramen_region_fixed`），所有 build 共用同一组合；即使恢复 120 组合全枚举，当前 `score_region` 打分对第 3 年地区**没有 build 区分度**——速度向与智力向卡组都会选中同一组合
- **排查过程**：
  - 临时测试 `test_region_build_sensitivity`（policy.rs）实测：速度向卡组（3速）与智力向卡组（3智）在第 3 年 120 组合打分下均选中 `[10, 11, 12]`，score 相同（4500）
  - 根因一（fixed 绕过策略）：`ramen_region_strategy="fixed"` 时第 3 年直接 apply `ramen_region_fixed[0]`，不经过 `decide_region` 打分，build 差异无从体现
  - 根因二（打分失效）：`score_region` 的 build 自适应依赖 `region.xunlian × 卡组 bias`，但第 3 年地区（id 10-19）`xunlian` 全为 0 → bias 项恒零；而 `pt_bonus` 全部相同（50）、`hint` 全 0 → 所有组合分数相同，argmax 取第一个
  - 第 3 年地区的真实差异在 `youqing`（40-60）与 `at_trains` 覆盖（单属性 vs 多属性），当前打分完全未纳入
  - **实施时补充发现：第 2 年（id 5-9）同样失效**。这批地区的 `xunlian` 也全为 0，`hint_count` 恒为 2，故 10 个组合（C(5,3)）一直同分取第一个。第 2 年不受 `ramen_region_strategy` 影响（`fixed` 只作用于第 3 年），所以这条一直存在、与固定值配置无关。本 issue 范围因此扩大到第 2/3 年
- **解决方案**（已定，待实施）：
  1. 增强 `score_region`：新增 `youqing × at_trains × 卡组 bias` 项（如 `Σ_{t∈at_trains} youqing × bias[t]`），使第 3 年打分随 build 训练倾向变化；权重沿用/扩展 `RamenPolicyConfig`
  2. `default_config.toml` 恢复 `ramen_region_strategy = "all"`（120 组合枚举，O(360) 打分已标注便宜）
  3. 验收：不同 build 选出不同第 3 年地区组合（更新 `test_region_build_sensitivity` 断言方向）
- **实施结果**（2026-08-23）：`xunlian` 与 `youqing` 统一按 `bias_sum` 缩放（新增 `region_youqing_weight`）。因同一年内 `pt_bonus` / `hint_count` 恒定、且 `xunlian` 与 `youqing` 不会同时非零，该权重的绝对值不影响 argmax，只影响打印的分数量级。验收通过：速度向选 `[15,17,18]`、智力向选 `[15,17,19]`。手写策略基线（8 马娘 × 7 build）聚合 49586 → 51340（+1754），其中唯一含「根」的 build `sta0_wis2` 增幅最大（+3572）——固定值 `[11,14,15]` 的训练位并集是 `{速,耐,力,智}`，恰好漏掉根
- **备注**：
  - 备选方案 B：按卡组主属性预筛候选范围（120 → 20~30 个）再打分——打分增强后不稳定时再启用，避免裁剪丢最优解
  - `test_region_build_sensitivity` 已由临时验证转为断言测试（`assert_ne!` 两 build 选中组合不同）
  - 与既有 issue「第三年地区选择组合过多」（已解决：Fixed）相关联：Fixed 是性能临时方案，本 issue 是在 build 维度上的功能补齐。恢复 `all` 后实测整局耗时 2.9ms 不变，120 组合枚举无可测代价
  - **影响采样器复现基座**：`sampler.rs` 的 `run_region_select` 读 `GAMECONFIG.ramen_region_strategy`，本次由 `fixed` 改 `all` 后同一条 `SampleSpec` 的轨迹已不同（结构性指标不变：74/78 回合覆盖、卡组分层 min==max）。Phase 3 落盘的配置签名须用改后这套
