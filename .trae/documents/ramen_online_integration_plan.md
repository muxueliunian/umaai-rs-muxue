# 拉面杯在线模式对接计划（SendGameStatusPlugin ↔ umaai）

> 状态：草案 v1（方案已与用户确认方向，待实机数据核实后定稿）
> 上游：URA `SendGameStatusPlugin` / `RamenScenarioAnalyzer`
> 下游：`umaai-rs`（crates/umaai 通道层 + crates/umasim 核心层）

## 1. Summary

为拉面杯建立在线模式 AI：上游 C# 插件把游戏回合数据写入 `thisTurn.json`，下游 umaai 通过文件监听获得状态、重建 `RamenGame`、计算并**在屏幕上直接输出决策**。

三项已确认的方向性决策：

1. **传输通道延续"写文件 + inotify"**，不引入 WS/管道；文件名维持 `thisTurn.json` 单文件混流，按 `baseGame.scenarioId` 分发解析。
2. **拉面杯与温泉杯共存于单 binary**，不做独立二进制；分发钩子复用 `parse_game` 既有检查。
3. **决策输出走屏幕展示**，暂不落决策 JSON 文件（契约草案 §4.1 的决策输出仅作为未来 Web/MCP 的参考形态保留）。

在线模式特有的决策模型是**两阶段决策**（见 §7）。

## 2. 现状核对

### 2.1 数据源（游戏侧）

- 拉面杯走独立端点族 `/umamusume/single_mode_ramen/*`（check_event / load / exec_command / tasting / select_region / check_point …）。
- 响应类型为 `SingleModeRamenLoadResponse` / `SingleModeRamenExecCommandResponse`，与普通剧本的 `SingleModeCheckEventResponse` **不是同一包装类**；但 `chara_info`（`SingleModeChara`）/ `home_info`（`SingleModeHomeInfo`）子结构同构。
- 拉面专用数据在 `data.ramen_data_set`（`SingleModeRamenDataSet`），关键字段：

| 字段 | 内容 |
|---|---|
| `feeling_info_array` | 诀窍库存（feeling_index 获得顺序 + feeling_id 类型） |
| `feeling_turn_info_array` | 诀窍槽（feeling_id, remain_turn） |
| `feeling_reduce_turn_info_array` / `command_feeling_info_array` | 各训练的心得收益 / 训练角标（command_id, feeling_id） |
| `special_feeling_num` | 隐藏风味数量 |
| `all_selected_region_id_array` | 当年已选地区 |
| `active_effect_array` | 当前生效效果（category/id/value） |
| `uraf_effect_info` | 超级拉面状态（type/state） |
| `training_exec_info_array` | 各训练已执行次数 |
| `command_info_array` / `evaluation_info_array` | 训练参数 / 羁绊 |

### 2.2 上游 C#

- `RamenScenarioAnalyzer` 已注册拉面端点 Analyzer，能拿到完整 `RamenScenarioResponseData`（CharaInfo + DataSet + HomeInfo + UncheckedEventArray + CommandResult），含 `TurnInfoRamen`（训练 id 映射 101/105/102/103/106 + 合宿 601-605）与 `RamenTrainingStatsCalculator`（五维/PT/体力/失败率预览），但仅用于 UI 展示，未输出文件。
- `SendGameStatusPlugin` 无拉面分支；`ScenarioType` 枚举无 Ramen 项；`GameStatusSend_Base` 构造函数绑死 `SingleModeCheckEventResponse`。

### 2.3 下游 Rust

- `umaai::protocol`：`GameStatus` trait（`scenario_id()` + `into_game()`）、`GameStatusBase`（camelCase，`parse_basegame()` 派生逻辑）、`GameStatusOnsen` 已完成；`UraFileWatcher`（notify）+ `parse_game`（检查 `baseGame.scenarioId`，分发钩子已预留）。
- `umasim`：`RamenGame`/`RamenState` 字段全 pub；`RamenStage` 三阶段（`RamenSelect`/`SpecialSelect`/`Train`）+ 合并决策接口 `list_combined_ramen_select_actions()` / `apply_combined_ramen_decision()`；`output/` 已有 `DecisionInfo` 等结构化输出。
- umaai 主循环目前硬编码 `GameStatusOnsen`/`OnsenGame`，无拉面路径。

## 3. 总体架构

```text
游戏 ── single_mode_ramen/* 响应
   → C# SendGameStatusPlugin：GameStatusSend_Ramen（新增）
       通用字段复用 Base 逻辑；ramen_data_set → ramen 段；跨回合累计见 §6
   → 写 PluginData/SendGameStatusPlugin/thisTurn.json（通道不变，§8 加固）
   → Rust UraFileWatcher（不动）→ parse_game 窥探 scenarioId 分发
   → protocol/ramen.rs：GameStatusRamen → RamenGame（turn import，§5.2）
   → 两阶段决策（§7）→ 屏幕输出（玩家可读，含预览收益与理由）
```

## 4. 通信协议：thisTurn.json（拉面版）

沿用 `GameStatusBase` 的 camelCase 命名，顶层为 `baseGame` + `ramen` 两段（温泉同构，`scenarioId` 区分）：

```jsonc
{
  "baseGame": { /* 与温泉版字段一致 */ },
  "ramen": {
    "feelingStock": [3, 2, 1],        // 诀窍库存 [A,B,C]
    "feelingQueue": [0, 1, 0, 2],     // 获得顺序队列（类型角标序列）
    "feelingSlot": [5, 7, 0],         // 诀窍槽 [A,B,C]（换算见 §10-Q4）
    "specialFeeling": 2,              // 隐藏风味
    "selectedRegions": [3, 7, 12],    // 当年已选地区（region id）
    "eatenRamen": -1,                 // 本回合已吃面（-1 未吃；>=0 为 region 下标，见 §7）
    "scenarioPt": 1240,               // 剧本 Pt（来源见 §6）
    "eatCount": 2,                    // 当年吃面次数（来源见 §6）
    "rmjResults": [true, false],      // 历次结算成败（来源见 §6）
    "trainFeelingTypes": [0, 2, 1, 0, 2], // 本回合各训练角标（A/B/C）
    "superRamen": -1                  // 超级拉面选项（-1 未选；71 结束后 0/1）
  }
}
```

设计要点：

- `eatenRamen` 由 C# **显式报告**（不在 Rust 端推断），是两阶段决策的阶段开关。
- `scenario_pt`/`eat_count`/`rmj_results` 上游游戏 JSON 无直接字段，来源待人工确认（§6）：优先从现有数据点推导，跨回合累计仅作兜底。
- 拉面 `scenarioId` 数值待实机确认（URA 枚举无 Ramen 项，见 §10-Q1）。

## 5. 下游改造（umaai）

### 5.1 主循环分发

- `parse_game` 前窥探 `baseGame.scenarioId`：温泉 → `GameStatusOnsen`，拉面 → `GameStatusRamen`，其余报剧本不支持。
- 新游戏检测按剧本分开（现 `is_next_of` 为温泉逻辑）；`scenarioId` 变化即视为新游戏。
- 两剧本共用 `MctsTrainer`（搜索层已泛型化）与 watcher；玩家同一时刻只进行一个剧本，无并发冲突。

### 5.2 `protocol/ramen.rs`（turn import）

`GameStatusRamen { base_game: GameStatusBase, ramen: RamenStatus }` 实现 `GameStatus`：

- 通用段复用 `GameStatusBase::parse_basegame()`（拉面剧本友人 chara_id 传拉面对应值）。
- `into_game()` 恢复顺序：
  1. `RamenState` 直接字段回填（stock/queue/slot/special/regions/pt/eat_count/rmj/super_ramen/train_feeling_type）；
  2. 人头重建（deck + 理事长/记者 + 拉面 NPC 五人）；`absent_cards` 由分布与卡组对比派生；
  3. `deck_can_split`（卡型 ≥5 种）、`current_effect`（`eatenRamen >= 0` 时按地区效果重算）；
  4. 阶段映射：未吃面 → `RamenSelect`；已吃面 → `Train`；事件等待 → `Distribute`；地区/超级拉面选择按 `playing_state`（待 §10-Q2 调查）。
- RNG 流（rule_master/turn_fixed/strategy/event）在线模式留 `None`（走旧行为）。

### 5.3 决策输出

- 屏幕输出沿用 umaai 现有对话流风格（候选内联预览 + 理由），数据源用 `DecisionInfo` + `GameView`；**不写决策 JSON 文件**。
- 事件选项决策沿用温泉模式（`select_event_choice`）。

## 6. 缺口数据来源（优先从现有数据点推导，跨回合累计仅作兜底）

`scenario_pt` / `eat_count` / `rmj_results` 在 `chara_info` 与 `ramen_data_set` 均无直接字段，但游戏 UI 上这些值均可见，客户端必有数据源——**很可能可从现有数据点推出或算出，需人工确认**（实机查看各端点响应 / 对照游戏 UI 数值）：

| 数据 | 候选推导路径（按希望排序） | 兜底（确认不可推导时） |
|---|---|---|
| `scenario_pt` | ① `check_point` 端点响应可能含当前累计值；② `active_effect_array` 中 PT 相关效果；③ 抓包全量搜索响应字段定位 UI 数据源 | 训练/吃面响应 `target_type=30` 增量累计，RMJ 结算回合后归零 |
| `eat_count` | ① `tasting` / `exec_command` 响应字段；② `training_exec_info_array` 或 `active_effect_array` 中吃面相关计数 | `tasting` 响应时 +1，RMJ 结算后归零 |
| `rmj_results` | ① `active_effect_array`：RMJ 成败改变下一年常驻效果，效果值差异可反推（最有希望）；② `check_point` 响应历史 | `check_point` 响应时记录 `result_state` |

确认结论回填本节后再冻结 §4 协议字段来源。**若全部可推导，C# 端可保持无状态（无跨回合累计），插件为纯函数式，可靠性与可测试性更好。**

胜场 `raceHistory` 沿用既有机制。

## 7. 两阶段决策流程（在线模式核心时序）

拉面杯"做面不消耗回合"：玩家在回合内先吃面、再选训练，游戏在这两步间各推送一次状态。AI 相应输出两次决策：

```text
回合开始推送（eatenRamen = -1）
   → RamenGame stage=RamenSelect
   → list_combined_ramen_select_actions → 搜索/手写决策
   → 屏幕输出：吃面 X + 隐藏风味 [a,b,c]（含预览收益）
玩家操作吃面（可能未照推荐）→ 游戏落地 → 同回合第二次推送（eatenRamen = Y）
   → RamenGame stage=Train，current_ramen=Some(Y)，效果按真实吃的面重算
   → 仅决策基础操作（训练/比赛/休息/外出…）
   → 屏幕输出
玩家执行训练 → 下一回合
```

设计要点：

- 第二阶段基于吃面后的**真实状态**重新决策，玩家没照第一推荐执行也不影响第二阶段质量。
- 同一 `turn` 的两次推送需在 C# 端去重语义上明确（各写一次 `thisTurn.json`，内容因 `eatenRamen` 不同而不同，Rust 端 `watch()` 的内容变更检测天然支持）。
- 年初地区选择 / 71 回合结束的超级拉面选择是独立决策点，由 `playing_state` 或端点（`select_region`）识别（待 §10-Q2），AI 各给一次推荐。

## 8. 通道可靠性加固（inotify 丢数据的已知问题）

现状：C# `File.WriteAllText` 非原子（清空→写入），watcher 可能在写入中途读到半截 JSON；此外存在事件时序丢失与目录定位（.portable / LOCALAPPDATA）不一致问题。两侧顺带加固：

1. **C# 端原子写**：写 `thisTurn.json.tmp` 后 `File.Replace`/`File.Move` 原子替换（`doSend` 一处改动，温泉同时受益）。
2. **Rust 端读取重试**：`do_poll` 读到空/非法 JSON 时短暂延时重读（有限次），失败才报错等待下一事件；`watch()` 初始化兜底已有（启动时直接读现存文件），保持。

## 9. 上游改造（SendGameStatusPlugin）

1. 新增 `GameStatusSend_Ramen.cs`：
   - `baseGame` 段复用 `GameStatusSend_Base` 提取逻辑（构造函数重载或抽公共接口以接受拉面响应类型；`chara_info`/`home_info` 同构，迁移成本集中在构造入口）；
   - `ramen` 段从 `SingleModeRamenDataSet` 映射（§4）+ 缺口字段实现（§6，推导优先、累计兜底，Phase B 回填）。
2. 新增拉面端点 Analyzer（注册 `SingleModeRamenLoadResponse` / `SingleModeRamenExecCommandResponse`，仿 RamenScenarioAnalyzer 的注册方式）；拉面响应不能走现有 `jo.ToObject<SingleModeCheckEventResponse>()` 旧入口。
3. `playing_state` 合法值过滤按拉面实测值域补分支。
4. `doSend` 原子写改造（§8）。

## 10. 待调查项（实机抓包/参考 RamenScenarioAnalyzer 一次解决）

| # | 问题 | 影响 |
|---|---|---|
| Q1 | 拉面杯 `scenario_id` 数值（URA `ScenarioType` 枚举无 Ramen 项） | `parse_game` 分发检查、C# 分支条件 |
| Q2 | 拉面杯 `playing_state` 值域：普通回合 / 已吃面 / 年初地区选择 / 超级拉面选择 / 事件等待 | 阶段映射（§5.2）、两阶段开关（§7）、C# islegal 过滤 |
| Q3 | `eatenRamen` 识别：`active_effect_array` 能否稳定反推本回合所吃面 | §6 回合内累计的兜底选择 |
| Q4 | `feeling_turn_info_array.remain_turn` 与诀窍槽值的换算（槽值 = 7 − remain_turn？feeling_id 与 A/B/C 的对应表） | `feelingSlot` 映射正确性 |
| Q5 | EventLogger `GameStats` 对拉面端点族的覆盖情况 | 训练等级计数是否需要补记录（§6） |
| Q6 | `training_exec_info_array.exec_count` 能否替代/校验 `eat_count` 累计 | 冗余校验，降低推导漂移风险 |
| Q7 | §6 三个缺口数据的推导路径人工确认（含 `current_ramen` 回合内识别，即 Q3） | 决定 C# 端是否需要跨回合状态；回填 §6 |

## 11. 验证策略

1. **协议对拍**：离线用保存的 `turn*.json` 样例驱动 `GameStatusRamen::into_game()`，打印重建状态与 `explain_ramen_info` 人工核对（诀窍/槽/地区/PT/效果）。
2. **一致性守门**：`into_game()` → 立即序列化回 JSON，关键字段 roundtrip 不变（仿温泉 `From<&OnsenGame>` 测试）。
3. **两阶段端到端**：模拟"未吃面 → 推荐 → 已吃面 → 基础操作推荐"的连续两份 JSON，确认阶段映射与 `current_ramen` 恢复。
4. **共存回归**：温泉 `thisTurn.json` 走原路径不回归（`scenarioId` 分发不破坏现有解析）。
5. **实机联调**：真实育成一局，覆盖地区选择 / 事件 / RMJ 结算 / 超级拉面 / 合宿回合。

## 12. 实施顺序

1. **Phase A（C# 端先行）**：`GameStatusSend_Ramen` 框架 + Base 通用段复用 + `ramen` 段映射（§4）；§6 缺口字段按候选路径留桩（TODO + 临时实现），不阻塞框架开发。
2. **Phase B**：实机人工确认 §10 全部问题（重点 Q7 → 回填 §6 结论）→ 冻结协议字段 → C# 端补齐缺口字段实现。
3. **Phase C（Rust 侧）**：`protocol/ramen.rs` + 主循环分发 + 离线样例测试（用实机抓包保存的 `turn*.json` 驱动）。
4. **Phase D**：两阶段决策屏幕输出联调 + 共存回归。

## 13. Assumptions & Decisions

1. 传输通道维持文件 + inotify；单文件混流按 `scenarioId` 分发。
2. 拉面/温泉共存单 binary；独立 binary 仅作为未来发布形态选项。
3. 决策输出走屏幕，不落决策 JSON；契约草案（上游需求文档 §4）保留为未来 Web/MCP 参考。
4. 在线决策采用两阶段模型；第二阶段基于真实状态重建。
5. 缺口数据（`scenario_pt` / `eat_count` / `rmj_results`）优先从现有数据点推导，待人工确认（§6）；确认不可推导时才由 C# 端跨回合累计兜底。Rust 端不自行推断剧本语义。
6. `RamenGame` 重建在 Rust 端完成（turn import 官方构造器），上游只负责忠实报告原始状态，不做剧本语义加工。
