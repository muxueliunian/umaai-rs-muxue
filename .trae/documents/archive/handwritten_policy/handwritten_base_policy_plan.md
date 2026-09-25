# 手写基础策略（HandwrittenTrainer）实现思路

> 目的：给蒙特卡洛搜索提供稳定、快速、低开销的基础策略（base policy）。
> 本文档记录当前讨论的定位、已拍板决策、地基方案、输出分层与 AI 自主探索的准备。

## 1. 背景与定位

- 拉面杯当前只有固定策略（地区顺序、超级拉面选项二），`HandwrittenTrainer` / `MctsTrainer` / `FlatSearch`
  仍绑定旧温泉杯，尚未适配 RamenGame。
- 文档已定案的粒度分工：**手写逻辑走三阶段**（RamenSelect→SpecialSelect→Train），
  **MctsTrainer 走合并决策路径**（方案 E）。手写逻辑天然是 MCTS rollout / 叶估值的基策。
- **本轮范围**：只做 rollout 用的基础策略；`HandwrittenTrainer` 仅作简单测试壳（跑完整局验证策略效果），
  不是交付主体。

## 2. 已拍板决策

| # | 决策 | 设计含义 |
|---|------|---------|
| 1 | `DecisionInfo` 只是 stub，需按上层需求调整 | 策略核心输出独立于协议格式；`DecisionInfo` 是它的序列化视图之一，接入时再映射 |
| 2 | 只做 rollout 基策，`HandwrittenTrainer` 仅测试用 | 交付物是纯打分函数核心 + 两个薄包装：确定性 argmax（rollout 用）+ HandwrittenTrainer（整局测试用） |
| 3 | 策略形态要利于调参 | 仿 onsen `HandwrittenEvaluator`：结构体存权重/阈值/加成常量 + 预设构造器（default / speed_build 等）；后续可挂入已有 `PolicyConfig`（文档预留了手写策略参数位） |
| 4 | 确定性：相同局面给出相同决策 | 策略是 `f(state, candidates) -> index` 纯函数：不用 RNG、平局按候选固定顺序、不依赖 HashMap 迭代序。RNG 仅存在于规则层（人头分配/事件） |
| 5 | 尽量复用现有公式算法 | 训练数值直接调 `calc_training_value` / `effects.rs` 既有计算；按 1c 计划做**每回合上下文缓存**（基础训练值、人头分布、失败率、可用面、静态效果只算一次），否则 rollout 性能不达标 |
| 6 | 卡组适配：初期 3-5 个固定卡组夹具，最终任意合法卡组 | 常量分两层：**通用守门规则**（体力/心情/生病/诀窍/面/PT，与卡组无关）+ **卡组派生系数**（从 cardDB 统计卡种数量/稀有度/技能分布、友人卡存在性，推导训练与外出优先级；参考 onsen `ABSENT_WEIGHT` 先例）。固定卡组作为调参夹具与回归测试，代码路径须对任意含新友人卡的合法卡组可运行 |
| 7 | 暂无性能指标，做好后再测 | 基准设施先建，指标后补 |

## 3. 地基：RandomTrainer 与简单基础策略的分工

结论：**两个都要，分工不同**——随机负责鲁棒性/覆盖率，简单基础策略负责质量基线与调参载体。

| | RandomTrainer | 简单基础策略 |
|---|---|---|
| 成本 | 零开发，已泛型可用 | 很小（守门规则 + 期望值排序） |
| 状态覆盖 | 广，含极端/烂局面 | 窄，有盲区 |
| 状态代表性 | 差（不代表真实对局） | 好（rollout 生成的状态正是基策自己的分布） |
| 调参信号 | 方差大、不敏感 | 敏感，整局分数随规则改进可观测 |
| 额外价值 | 压力测试（合法性、不 panic） | 它就是策略 v0，迭代即打磨它 |

落地组合：

1. **先建地基设施**（与选谁无关）：固定种子批量跑 + 决策日志落盘 + 分数分布输出（CSV/JSON）。
2. **RandomTrainer 用途**：跑 N 局产出状态语料 + 合法性回归门槛——每个策略版本须在语料上返回合法索引、不 panic、耗时达标。
3. **简单基础策略用途**：体力<45 休息、心情<5 外出、生病治病 + 按 `calc_training_value` 期望值排序选训练 + 面/诀窍贪心，跑整局产出 Random vs Basic 基线对比，直接开始调参。
4. （可选，暂不做）基础策略 + ε 随机探索，得到"分布内又带探索"的语料，留待未来训练 NN / 离线评估。

如果只先做一个：**先简单基础策略**（地基的目的就是让手写策略可测量、可迭代，它同时就是测量对象），RandomTrainer 作为配套测试随后补。

## 4. 输出分层：GameView / GameEvent / DecisionInfo / 决策日志

三者现有状态：`GameView` 已实现（状态快照，协议用）；`GameEvent` 仅规划（过程事件流，可回放/汇总/转 UI，当前无代码）；`DecisionInfo` 已实现但明确是 stub（决策输出协议格式）。

| 类型 | 记录对象 | 消费方 | 粒度 |
|---|---|---|---|
| GameView | 世界现在长什么样（状态快照） | 运行时下游（umaai/Android/MCP） | 每回合一次，schema 稳定 |
| GameEvent | 世界发生了什么（过程事件流） | 回放/汇总/转 UI（未来） | 每回合若干条 |
| DecisionInfo | 决策结果 + 概要理由（协议格式） | 下游协议 | 每次决策，schema 稳定 |
| 决策日志 | 决策者怎么想的：候选、各维度评分分解、原因、耗时 | 开发调参 | rollout 场景成千上万条/局，默认关闭 |

结论：调参需要的是**开发格式**（高保真、随意演进），协议需要的是**稳定格式**，不应是同一个类型。分层：

   策略核心输出（内部结构，随意演进）
     ├─ RamenPolicyOutput：每个候选 { score, score_breakdown, reason } + 耗时
     ├─ 决策日志（开发用）：dump 成 CSV/JSON，带 turn/stage/候选摘要，全量高保真
     └─ DecisionInfo（协议用）：按需从核心输出映射，schema 稳定

本轮只需要**决策日志**：在基准 bin / 测试壳里以内部结构 + CSV/JSON 落盘实现，
不动 GameView、不设计 GameEvent；拉面特有上下文（诀窍库存、PT、吃面计数等）
直接作为日志 context 字段，不扩公共 GameView。

## 5. 玩家经验标签（好/坏概念）

- 来源：玩家经验提供"已知好/坏结果"的语义概念（如隐藏风味溢出、训练失败是坏的），
  这是人类向策略注入知识的接口——数值机制 AI 可从代码读到，但"溢出是坏事"这类极性判断只有玩家能给。
- 三个用途：
  1. **打分项**：进入 score_breakdown 的奖励/惩罚（只用于期望值未建模的语义，如溢出这种机会浪费）
  2. **过程健康指标**：跑批统计坏事件频率，作为调参信号（最终分数不知道改哪，坏事件清单直接指到规则）
  3. **未来数据标注先验**：将来训 NN / 离线评估的 reward shaping 先验（现在不实现，保持清单）
- 注意：
  - 标签是**先验不是硬约束**：如溢出可能当期无面可吃、不可避免，更适合警告/诊断而非直接罚
  - **避免双重计分**：训练失败已被失败率公式计入期望值，走健康指标而非打分项
- 草稿清单见同目录 `good_bad_labels_draft.md`（玩家确认后成为正式输入）。
- **决策频率 × 影响**：高频小影响（训练/吃面/隐藏风味，每回合）→ 简单贪心 + 轻日志 + 统计调参，追求"平均正确"，单步错可被后续回合稀释；低频大影响（地区选择，每年 1 次、影响一整年；超级拉面与事件同理）→ 允许更仔细评估 + 重日志 + 调参优先，追求"单次正确"。规则复杂度按影响分配、性能约束按频率分配（高频必须 O(候选数) 简单，正合 rollout 低开销要求）

## 6. AI 自主探索的准备（输入资料包）

- 领域规则：`ramen_memo_cn.md`（诀窍/隐藏风味/RMJ/地区/分身/友人/事件触发表）+ `scenario_ramen.json` 字段说明。
- 可复用规则 API：`rules.rs` 的 `list_special_targets_for` / `calc_ramen_pt_gain` / `check_rmj` /
  `get_region_combinations` / `calc_region_bonus` 等；`effects.rs` 训练数值公式（下半值 × 训练加成 × 友情 × PT）。
- 状态模型：`RamenState`（feeling_stock/slot/queue、special_feeling、scenario_pt、rmj_results、
  train_level_bonus、eat_count、super_ramen、selected_regions）+ `BaseGame`（五维/体力/心情/技能点/羁绊/训练等级/回合/卡组）。
- 决策点清单：①RamenSelect 选面/不吃面 ②SpecialSelect 隐藏风味用法 ③Train 五训练+比赛+休息+外出+治病
  ④年度 RegionSelect（第3年 120 组合，注意 `ramen_region_strategy` 配置）⑤`player_select=true` 事件选项
  （4009/4010/友人事件）⑥超级拉面回合与 race 回合短路（72-77 不决策）。
- 运行环境：release 构建、固定种子、默认卡组（含友人卡 303051-303054）与马娘；
  `test_ramen_silent_loop` / `ramen_manual` 作为正确性锚点；`RandomTrainer` 作为对照基线。
- 编码守则：AGENTS.md 全量（中文、Result 优先、测试用 println、文档注释、release 模式、依赖用 cargo add、提交前统一更新 changelog）。

## 7. 期望的输出与数据（验收物）

- 代码：拉面手写策略模块（参数化结构体）+ `Trainer<RamenGame>` 测试壳 + 每个启发式分支单元测试 + 完整 77 回合集成测试。
- 决策轨迹数据：每回合每阶段 turn/stage/候选数/选中动作/score_breakdown/耗时（CSV 落盘到 `logs/`）。
- A/B 数据：同一批固定种子 Random vs Handwritten（可加权重变体档）各 N 局，输出评分分布
  （mean/median/min/max/std）、三年 RMJ 成败、PT 与五维期末值——回答"手写策略是否稳定"。
- 性能数据：各阶段 select_action 平均/最大耗时、整局耗时、rollout 模拟速度——回答"快速/低开销"。
- 调参记录：常量表（权重/阈值/加成）+ 每次改动依据（哪局/哪回合翻车触发）。
- 接入说明：作为 MCTS rollout 基策的接入点约定（RamenGame 版 SimulationTrainer / FastPolicy 接口）。

## 8. 实施主线

1. **先立地基**：固定种子跑批 + 决策日志 + RandomTrainer 基线分布（没有基线无法量化改进）。
2. **收益排序 + 安全剪枝起步**（1c 计划一致，不含复杂搜索）：
   - Train：期望收益 ≈ 训练数值 ×(1−失败率) + 彩圈/羁绊/得意率加成 + PT 折算；守门规则（体力/心情/生病）；
     比赛按 `race_grades` 与时机；夏合宿/友人解锁/前期智力策略参考 onsen 手写先例。
   - RamenSelect：按可做性 + 库存 + 隐藏风味预算 + PT 增益 + 地区效果打分，贪心选面。
   - SpecialSelect：以 `min_needed` 补缺口、最小化隐藏风味浪费、保留库存（候选仅 1~10 个，轻量贪心）。
   - RegionSelect：按地区效果静态排序粗筛（第3年 120 组合不全搜）。
   - 事件：按效果类型价值打分（对齐 onsen `evaluate_choice` 做法）。
3. **决策日志驱动调参**：找"体力崩 / RMJ 失败 / PT 落后"的回合，针对性补规则；所有数字进常量表。
4. **性能裁剪与 MCTS 接入**：拆分"精版"（在线）与"简版"（rollout，跳过 explain/日志、避免分配）；
   验证"手写 vs 随机"作为基策的搜索增益（对应 `mcts_turn_bonus` 语义）。
5. **收尾**：测试、性能基准、策略说明、changelog。

## 9. 下一步

- 先产出「探索任务书」（第 5~6 节固化为自包含任务 + 验收标准）。
- 同时补最小基准 bin：固定种子批量跑 RandomTrainer，输出分数分布 CSV——一切后续工作的地基。
- 待用户确认地基组合（第 3 节）与策略形态（第 2 节 #3）后开工。
