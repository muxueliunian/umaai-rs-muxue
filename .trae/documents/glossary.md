# 项目术语表（Glossary）

## 核心游戏

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Game` | 局面状态类型 | Game Trait |
| `GameData` | 静态数据 | 马娘/支援卡/事件定义/剧本基础数据 |
| `BaseGame` | 基础游戏 | 通用状态+逻辑 |
| `BasicGame` | 基础实现 | 无剧本的基础训练逻辑 |
| `Turn` | 回合 | 从0开始 |
| `TurnStage` | 回合阶段 | 不同剧本有不同的回合阶段 |
| `GameStatus` | 协议状态 | 外部协议 |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `GameState` | 当前可变状态 | 与 GameData 分离 |
| `Scenario` | 剧本 | 一套固定规则 |
| `ScenarioState` | 剧本状态 | 仅属某剧本 |
| `ScenarioData` | 剧本数据 | 配方/效果定义 |
| `GameResult` | 最终结果 | 评分+汇总 |
| `SimContext` | 模拟上下文 | RNG/输出模式 |

## 阶段 Phase

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Begin` | 开始阶段 | 处理回合开始触发的事件 |
| `Distribute` | 分配阶段 | 把人头分配到训练 |
| `Train` | 训练阶段 | 交给Trainer选择 |
| `AfterTrain` | 训练后 | 处理训练后的追加事件 |
| `End` | 结束阶段 | 部分剧本事件要在这里处理 |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Distribution` | 人物分配 | |
| `MainAction` | 主行动 | 训练/比赛/休息等 |
| `AfterAction` | 行动后 | 事件+状态处理 |
| `GameFinish` | 游戏结束 | 终局结算 |

## 动作与决策

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Decision` | 决策 | 简称 |
| `list_actions` | 列动作 | 列出所有可能的动作 |
| `apply_action` | 应用动作 | |
| `next` | 下一阶段 | |
| `list_and_apply_action` | 选并执行 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Operation` | 基础操作 | 一次基础行动 |
| `Action` | 完整动作 | 原子状态迁移 |
| `DecisionRequest` | 决策请求 | 游戏向策略提问 |
| `DecisionResponse` | 决策响应 | 策略回答 |
| `DecisionCandidate` | 候选项 | |
| `ActionPreview` | 动作预览 | 不修改状态 |
| `ActionValidationError` | 动作错误 | 不合法原因 |
| `MainActionDecision` | 主行动决策 | |
| `YearlyRamenSelection` | 年度选面 | 拉面杯决策 |
| `EventOptionDecision` | 事件决策 | |
| `validate_action` | 校验动作 | |
| `preview_action` | 预览动作 | |
| `apply_operation` | 应用操作 | |
| `advance_stage` | 推进阶段 | |
| `select_and_apply_action` | 选并执行 | 含策略选择 |

## 基础操作

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `BaseAction` | 基础动作 | 枚举 |
| `Train(i32)` | 属性训练 | |
| `Race` | 比赛 | |
| `Sleep` | 休息 | |
| `FriendOuting` | 友人外出 | |
| `NormalOuting` | 普通外出 | |
| `Clinic` | 治病 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Rest` | 休息 | 替代 Sleep |
| `Outing` | 外出 | 统一友人/普通 |

## 策略与评估

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Trainer` | 决策器 | 现有 Trait，保留 |
| `RandomTrainer` | 随机决策器 | |
| `HandwrittenTrainer` | 手写决策器 | |
| `MctsTrainer` | 搜索决策器 | 实际用 FlatSearch |
| `NeuralNetTrainer` | 神经网络决策器 | |
| `ManualTrainer` | 手动决策器 | |
| `Evaluator` | 评估器 | 现有 Trait |
| `LeafEvaluator` | 叶节点评估器 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Policy` | 决策策略 | 替代 Trainer |
| `RandomPolicy` | 随机策略 | |
| `HandwrittenPolicy` | 手写策略 | |
| `SearchPolicy` | 搜索策略 | |
| `NeuralPolicy` | 神经网络策略 | |
| `InteractivePolicy` | 交互策略 | |
| `ActionEvaluator` | 动作评估器 | |
| `ValueEvaluator` | 局面评估器 | |
| `ChoiceEvaluator` | 选项评估器 | |
| `ActionEvaluation` | 动作评估 | |

## 搜索

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `FlatSearch` | 扁平搜索 | |
| `Rollout` | 模拟展开 | |
| `search_group_size` | 搜索分组数 | |
| `max_depth` | 最大深度 | |
| `cpuct` | PUCT 系数 | |
| `search_cpuct` | 搜索 PUCT 系数 | |
| `radical_factor` | 激进度 | |
| `policy_delta` | 策略差值 | 含义待确认 |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `SearchBudget` | 搜索预算 | 次数/时间上限 |
| `SearchBatchSize` | 搜索批大小 | |
| `MaxDepth` | 最大搜索深度 | |
| `Cpuct` | PUCT 系数 | 统一名称 |
| `RadicalFactor` | 激进度因子 | |
| `PolicyTemperature` | 策略温度 | 替代 policy_delta |

**算法名（外部引用）**

| 用语 | 中文 | 备注 |
|---|---|---|
| `MCTS` | 蒙特卡洛树搜索 | 真正树搜索时使用 |
| `UCB` | 上置信界 | |

## 数值与状态

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Status` | 属性 | 五维 |
| `SkillPt` | 技能点 | |
| `Vital` | 体力 | |
| `Motivation` | 干劲 | 同 ganjing|
| `Friendship` | 羁绊 | 同 jiban |
| `HintLevel` | Hint 等级 | |
| `FailureRate` | 失败率 | |
| `TrainingValue` | 训练值 | |
| `TrainingBonus` | 训练加成 | |
| `UpperValue` | 上半训练值 | 剧本加成后总值，减下半值反映剧本加成 |
| `UpperLimit` | 上半训练值上限 | 可变 |
| `LowerValue` | 下半训练值 | 原始支援卡训练值，不受剧本加成 |
| `LowerLimit` | 下半训练值上限 | 固定为100 |
| `UpperDisplay` | 上半显示值 | UpperValue - LowerValue |
| `LowerDisplay` | 下半显示值 | |
| `ActionValue` | 效果变化 | 现有结构，保留 |
| `Buff` | 增益 | |
| `Flag` | 状态标识 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `StatusType` | 属性类型 | |
| `StatusDelta` | 属性变化 | |
| `VitalLimit` | 体力上限 | 动态上限 |
| `TrainingLimit` | 训练上限 | |
| `Modifier` | 修正项 | |
| `Counter` | 计数 | |
| `Duration` | 持续回合 | |

## 人物与训练

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Person` | 训练人物 | |
| `NPC` | 非支援卡、记者、理事长的其他人物 |
| `Reporter` | 记者 | |
| `Yayoi` | 理事长 | |
| `Link` | 剧本链接人物 | |
| `Hint` | Hint | 游戏数据库/英文 |
| `SpecialtyRate` | 得意率 | 游戏数据库/英文，优先使用 deyilv |
| `youqing` | 友情 | 旧 C/拼音 |
| `xunlian` | 训练 | 旧 C/拼音 |
| `deyilv` | 得意率 | 旧 C/拼音 |
| `ganjing` | 干劲 | 旧 C/拼音 |
| `saihou` | 赛后 | 旧 C/拼音 |
| `RaceBonus` | 比赛加成 | 同 saihou |
| `Shining` | 友情训练 | 同 `彩圈` | 

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `PersonType` | 人物类型 | |
| `ShiningBonus` | 友情加成 | 同 youqing |
| `PersonDistribution` | 训练分布 | |
| `MotivationBonus` | 干劲加成 | |

**拉面杯术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `ClonedPerson` | 分身人物 | 拉面 Buff 添加 |

## 事件

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `EventData` | 事件数据 | |
| `EventChoice` | 事件结果分支 | 现有结构，保留 |
| `ChoiceResult` | 结果类型 | 正常/成功/大成功/失败/大失败 |
| `choices` | 选项/结果 | 现有字段 |
| `prob` | 概率 | 现有字段 |
| `StoryStatus` | 故事状态载荷 | 协议层 |
| `StoryChoice` | 故事选项载荷 | 协议层 |
| `StoryEffectValue` | 故事效果载荷 | 协议层 |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Event` | 事件 | 统一领域名 |
| `PendingEvent` | 待处理事件 | 代替 unresolved_event |
| `EventHistory` | 事件历史 | |

**协议 vs 领域**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Story*` | 外部故事数据 | 协议层名称 |
| `Event*` | 内部事件数据 | 模拟器层名称 |

## 评分与统计

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Score` | 游戏评分 | |
| `ActionResult` | 动作统计 | 多次 rollout 统计 |
| `num` | 样本数 | 字段名 |
| `count` | 样本数 | 方法名 |
| `distribution` | 分数分布 | 字段名 |
| `mean` | 均值 | |
| `stdev` | 标准差 | |
| `weighted_mean` | 加权均值 | |
| `SearchOutput` | 搜索输出 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `FinalScore` | 最终评分 | |
| `EvaluationScore` | 评估分 | |
| `SelectionValue` | 决策价值 | |
| `RiskAdjustedScore` | 风险评分 | |
| `SkillPtFavoredScore` | 技能点偏好分 | |
| `ActionSimulationStats` | 动作模拟统计 | 替代 ActionResult |
| `ScoreHistogram` | 分数直方图 | 替代 distribution |
| `RiskAdjustedMean` | 风险调整均值 | 替代 weighted_mean |
| `TurnSnapshot` | 回合快照 | |
| `SelectReason` | 选择原因 | |
| `RejectReason` | 淘汰原因 | |

**统计级别**

| 用语 | 中文 | 备注 |
|---|---|---|
| `None` | 不收集 | |
| `Summary` | 汇总 | |
| `Turn` | 回合级 | |
| `Detailed` | 详细 | |

## 日志与展示

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Collector` | 收集器 | |
| `ratatui` | 终端界面库 | |
| `worker` | 工作线程 | 并行模拟 |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `Diagnostics` | 诊断日志 | 面向开发者 |
| `GameView` | 展示模型 | 面向用户/UI |
| `Observer` | 观察器 | 接收 GameEvent |
| `SilentMode` | 静默模式 | 搜索/批量模拟 |
| `OutputMode` | 输出模式 | |

## 配置

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `GameConfig` | 游戏配置 | |
| `MctsConfig` | 搜索配置 | |
| `CollectorConfig` | 收集配置 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `RuntimeConfig` | 运行配置 | 合并后有效值 |
| `SimConfig` | 模拟配置 | 局数/种子 |
| `SearchConfig` | 搜索配置 | 预算/深度/UCB |
| `PolicyConfig` | 策略配置 | |
| `OutputConfig` | 输出配置 | |
| `DevConfig` | 开发配置 | 调试开关 |
| `DevDefaults` | 开发默认值 | 来源 default_config.toml |
| `UserOverride` | 用户覆盖值 | 来源 game_config.toml |
| `ActiveConfig` | 最终配置 | |

## 神经网络与样本

**现有术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `TrainingSample` | 训练样本 | |
| `PolicyTarget` | 策略目标 | |
| `ChoiceTarget` | 选项目标 | |
| `ValueTarget` | 价值目标 | |
| `SampleCollector` | 样本收集器 | |

**规划术语**

| 用语 | 中文 | 备注 |
|---|---|---|
| `ModelTraining` | 模型训练 | 与游戏 Train 区分 |
| `Batch` | 样本批次 | |
| `FeatureVec` | 特征向量 | |
| `Mask` | 动作掩码 | |
| `GameSample` | 单局样本 | |
| `Exploration` | 探索 | |
| `Exploitation` | 利用 | |
| `ExplorationRate` | 探索率 | |
| `RolloutsPerOption` | 每项模拟数 | |
| `CRN` | 共同随机数 | |

## 拉面杯

| 用语 | 中文 | 备注 |
|---|---|---|
| `RamenScenario` | 拉面杯剧本 | |
| `RamenGame` | 拉面杯游戏 | |
| `RamenState` | 拉面杯状态 | |
| `RamenData` | 拉面杯数据 | 配方/面种/效果 |
| `BasicEffect` | 拉面基础效果 | |
| `RamenPoint` | 剧本点数 | 盛り上がりPt |
| `Feeling` | 诀窍 | 原"食材" |
| `FeelingType` | 诀窍类型 | 三种普通诀窍 |
| `Feelings` | 诀窍库存 | |
| `SpecialFeeling` | 隐藏风味 | 原"万能食材" |
| `FeelingLimit` | 诀窍上限 | 固定为10 |
| `FeelingBonus` | 诀窍加成 | |
| `FeelingCost` | 诀窍消耗 | |
| `Region` | 地域 | |
| `RamenType` | 面种类 | 固定配方+Buff |
| `RamenRecipe` | 面配方 | 5 个诀窍 |
| `RamenInventory` | 面库存 | |
| `Eat` | 吃面 | 做了就立即吃，是一个步骤 |
| `RamenAction` | 拉面动作 | 完整组合动作 |
| `AvailableRamen` | 可用面 | |
| `SelectedRegions` | 选择地区集合 | |
| `TwinkleRamen` | 超级拉面 | 73-78 回合 |
| `TwinkleEffect` | 超级面效果 | |
| `RamenSelectionPolicy` | 选面策略 | |
| `RegionSelectionPolicy` | 地区选择策略 | |
| `RamenEffect` | 面效果 | |
| `RamenBuff` | 面 Buff | |
| `LimitModifier` | 训练上限修正 | |
| `CloneEffect` | 分身效果，包括分身位置和数量的列表 | |
| `ExtraHintEffect` | 额外 Hint | |
| `FailRateDrop` | 失败率修正 | |

**地域名称**

札幌、函馆、新潟、福岛、东京、中山、中京、京都、阪神、小仓

## 旧温泉杯

| 用语 | 中文 | 备注 |
|---|---|---|
| `OnsenGame` | 温泉杯游戏 | onsen_backup |
| `OnsenAction` | 温泉杯动作 | onsen_backup |
| `OnsenTurnStage` | 温泉回合阶段 | onsen_backup |
| `OnsenBuff` | 温泉 Buff | onsen_backup |
| `OnsenEffect` | 温泉效果 | onsen_backup |
| `OnsenInfo` | 温泉信息 | onsen_backup |
| `OnsenOrder` | 温泉指令 | onsen_backup |
| `BathingInfo` | 泡澡信息 | onsen_backup |
| `Bathing` | 泡澡 | onsen_backup |
| `UseTicket` | 用券 | onsen_backup |
| `Dig` | 挖掘 | onsen_backup |
| `Upgrade` | 升级 | onsen_backup |
| `dig_remain` | 剩余挖掘 | onsen_backup |
| `dig_progress` | 挖掘进度 | onsen_backup |
| `dig_count` | 挖掘次数 | onsen_backup |
| `dig_power` | 挖掘力 | onsen_backup |
| `dig_level` | 挖掘等级 | onsen_backup |
| `dig_vital_cost` | 挖掘体力消耗 | onsen_backup |
| `current_onsen` | 当前温泉 | onsen_backup |
| `onsen_state` | 温泉状态 | onsen_backup |
| `pending_selection` | 待选的剧本回合选项 | onsen_backup |
| `is_super` | 强化状态 | onsen_backup |
| `is_super_ready` | 强化就绪 | onsen_backup |

## crate 与外部协议

**现有**

| 用语 | 中文 | 备注 |
|---|---|---|
| `umasim` | 模拟器核心 | crate |
| `umaai` | AI 应用层 | crate |
| `analyzer` | 分析工具 | crate |
| `UraFileWatcher` | Ura 文件监听 | |

**规划**

| 用语 | 中文 | 备注 |
|---|---|---|
| `ProtocolPayload` | 协议载荷 | 外部 JSON |
| `ProtocolAdapter` | 协议适配器 | 转为内部领域 |
| `GameStatePayload` | 状态载荷 | |

## 缩写与来源

| 用语 | 全称 | 中文 |
|---|---|---|
| `AI` | Artificial Intelligence | 人工智能 |
| `NN` | Neural Network | 神经网络 |
| `MCTS` | Monte Carlo Tree Search | 蒙特卡洛树搜索 |
| `UCB` | Upper Confidence Bound | 上置信界 |
| `PUCT` | Predictor + UCT | 预测引导搜索 |
| `CRN` | Common Random Numbers | 共同随机数 |
| `RNG` | Random Number Generator | 随机数生成器 |
| `UI` | User Interface | 用户界面 |
| `PT` / `Pt` | Skill Point | 技能点 |
| 拼音 | 旧版 C 代码命名 | 历史来源 |
| 英文 | 程序/游戏数据库命名 | 历史来源 |
| 日文 | 游戏数据库命名 | 历史来源 |