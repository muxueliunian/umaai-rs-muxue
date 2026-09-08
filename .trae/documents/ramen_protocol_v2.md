# 拉面杯在线协议 V2（实测定稿）

> 状态：定稿 v1（2026-09 基于 chara 6204 全 78 回合 151 样本实测）
> 来源：`logs/GameStatusSend_Ramen/game6204_turn*.json` 151 份 + 用户原话校验
> 用途：`crates/umaai/src/protocol/ramen.rs` turn import 设计基线
> 取代：`ramen_online_integration_plan.md` §4 草案（已废弃）

## 0. 顶层结构

每份样本是 `thisTurn.json` 一次快照，两段结构：

```text
{
  "ramen": { ... },       // 拉面段：RamenStatus 镜像
  "baseGame": { ... }     // 基础段：与温泉剧本同构 + 拉面增量字段
}
```

`scenarioId = 14`（拉面）vs `12`（温泉），用于分发。

## 1. baseGame 段

字段命名沿用 onsen `GameStatusBase` 的 CamelCase 风格，便于共用解析层。

| 字段 | 类型 | 含义 | Rust 端映射 |
|---|---|---|---|
| `scenarioId` | int | 14=拉面 / 12=温泉 | 分发判据 |
| `umaId` | int | 马娘 ID | `Uma::uma_id` |
| `umaStar` | int | 星数 | `Uma::uma_star`（暂未用） |
| `turn` | int | 当前回合（0-77） | `BaseGame::turn` |
| `vital` / `maxVital` | int | 体力/上限 | `Uma::vital/max_vital` |
| `motivation` | int | 干劲 1-5 | `Uma::motivation` |
| `fiveStatus` / `fiveStatusLimit` | int[5] | 五维/上限 | `Uma::five_status/five_status_limit` |
| `skillPt` / `skillScore` | int | 技能点/已学技能分数 | `Uma::skill_pt/skill_score` |
| `totalHints` | int | 总 Hint 等级 | `Uma::total_hints` |
| `trainLevelCount` | int[5] | 训练设施等级 | `BaseGame::train_level_count` |
| `ptScoreRate` / `failureRateBias` | f32/int | PT 系数/失败率偏置 | `Uma::*` |
| `isQieZhe` / `isAiJiao` / `isPositiveThinking` / `isRefreshMind` / `isIll` / `isLucky` | bool | Uma flags | `Uma::flags` |
| `zhongMaBlueCount` | int[5] | 种马蓝因子 | `InheritInfo::blue_count` |
| `saihou` / `isRacing` | int/bool | 赛后加成/生涯比赛状态 | `Uma::race_bonus/is_race_turn` |
| `cardId` | int[] | 卡组（id*10+rank 编码） | `BaseGame::deck` |
| `persons` | Person[] | 人头羁绊 | `BaseGame::persons` |
| `personDistribution` | int[5][5] | 人头分布 | `BaseGame::distribution` |
| `lockedTrainingId` / `friendship_noncard_yayoi/reporter` / `friend_stage` / `friend_outgoingUsed` / `playing_state` / `raceHistory` / `story` | — | 与温泉同 | 同 |
| **`single_mode_chara_id`** | int | **新增**：账号总育成局数 | 切局判据（与 saved_game 对照） |
| **`source`** | string | **新增**：`event` / `command` / `load` / `special` | 决策循环路由 |

### 1.1 `source` 语义

| 值 | 含义 | Rust 端处理 |
|---|---|---|
| `command` | 普通回合推送，玩家需选训练 | 走 `RamenTrainer::select_action` |
| `event` | 事件已挂载，需玩家决策（带 `story`） | 走 `MctsTrainer::select_event_choice` |
| `load` | 重进游戏时一次性推送 | 完整重建 RamenGame，初始化 last_ramen |
| `special` | 剧本专用（**151 样本未出现**，预留） | 暂不处理，按 `command` 同处理 |

### 1.2 `playing_state` 完整值域（实测）

| 值 | 含义 | 样本数 | Rust 端 stage |
|---|---|---|---|
| **1** | 训练阶段（含吃面/比赛回合） | 139 | `RamenStage::Train`（含 3 阶段） |
| **5** | 事件等待，必带 `story` | 6 | `RamenStage::Event` |
| **45** | RMJ 结算出口（`feeling_stock` 清空） | 2 (turn23_3, 47_4) | `RamenStage::Settlement` |
| **46** | RMJ 结算入口 | 3 (turn23_2, 47_3, 71_3) | 同上（与 45 配对） |
| **48** | 超级拉面过渡（仅 turn71_4） | 1 | `RamenStage::SuperRamen` |

**2 <= ps <= 10**：比赛相关，Rust 端**排除**（沿用 §4 草案判断）。

**地区选择不在独立 playing_state**：turn 0/1 selected_regions=[0,0,0] 之后 turn 2 直接 =[1,4,5]，无中间 snapshot。可能在 load 时一次性快照。

### 1.3 拉面专用事件 ID 范围

| 事件 ID | Name | 触发回合 | 含义 |
|---|---|---|---|
| `400014016` | 貴方と私とラーメンと | 64 | 拉面伙伴选择（3 选项：ファインモーション / ナリタトップロード / カルストンライトオ） |
| `400014017` | 最後、究極の"仕込み" | 71 | 超级拉面选择（固定 1 选项，固定效果） |

通用事件 ID 范围：`5010067xx`（角色）/ `8090011xx`（支援卡）/ `5010061xx`（角色其他）。

## 2. ramen 段

```text
{
  "feeling_guage_gains": [[i32;3]; 5],   // 每训练×每类型回合增量（部分填）
  "feeling_guage":       [i32; 3],        // 诀窍槽当前值
  "feeling_stock":       [i32],           // 诀窍队列（按获得顺序）
  "special_feeling":     i32,             // 隐藏风味数量
  "train_feeling_type":  [i32; 5],        // 训练角标（A=1/B=2/C=3, 0=无）
  "active_effect_array": [{category, id, value}], // 当前生效效果
  "super_ramen":         i32,             // -1=未选 / 0/1/2=已选超级拉面档位
  "selected_regions":    [i32; 3],        // 当年已选地区（region_id）
  "feeling_guage_gain_base": [i32; 3],    // 基础增量（按 region 配方）
  "last_ramen":          i32,             // **直接 = region_id**（关键）
  "scenario_pt":         i32,             // 当前累计剧本 PT（RMJ 失败归零）
  "next_scenario_pt":    i32              // 下次吃面可获 PT
}
```

### 2.1 字段语义要点

- **`feeling_guage_gains[5][3]`**：训练索引按 command_id 严格顺序 `[101, 105, 102, 103, 106]`（速/耐/力/根/智），feel_id 按 `[1,2,3]`（A/B/C）。
- **`train_feeling_type[5]`**：0=本回合无角标 / 1/2/3=A/B/C。
- **`selected_regions[3]`**：当年度选定的 3 个 region_id；年初为 `[0,0,0]`，下回合起为实际选定值。
- **`last_ramen`**：**实测就是 region_id**，与 selected_regions 严格对齐。每次吃完面更新，可同回合多次变化（如 turn12 1→4），跨回合保持显示直到下回合。
- **`scenario_pt`**：Y1 max 2700（RMJ 通过 1500）/ Y2 5400（3000）/ Y3 7500（3500）；失败时归零重新累计。
- **`next_scenario_pt`**：下次吃面可获 PT。Y1≈450 / Y2 440-480 / Y3 RMJ 通过后 750。
- **`super_ramen`**：-1=未选；71 回合末事件 400014017 选完后变 0/1/2，对应三种超级拉面档位。
- **`active_effect_array`**：每项 `{category, id, value}`，混合基础效果+地区效果+RMJ 常驻。仅用于还原训练数值，**不需要反推** `last_ramen`（`last_ramen` 已独立给出）。

### 2.2 Rust 端映射

| 协议字段 | Rust 字段 | 备注 |
|---|---|---|
| `feeling_guage_gains` | `RamenState::feeling_guage_gains`（新增） | 直接覆写 |
| `feeling_guage` | `RamenState::feeling_slot`（or 新增 `feeling_guage`） | 直接覆写 |
| `feeling_stock` | `RamenState::feeling_stock` | 直接覆写 |
| `special_feeling` | `RamenState::special_feeling` | 直接覆写 |
| `train_feeling_type` | `RamenState::train_feeling_type` | 直接覆写 |
| `active_effect_array` | `RamenState::active_effect` 或拆为 `current_effect` + `scenario_buff` | 拆分逻辑见 §3 |
| `super_ramen` | `RamenState::super_ramen` | 直接覆写 |
| `selected_regions` | `RamenState::selected_regions` | 直接覆写 |
| `feeling_guage_gain_base` | `RamenState::feeling_guage_gain_base`（新增） | 直接覆写 |
| `last_ramen` | `RamenState::current_ramen` 或 `last_ramen` | 直接覆写 |
| `scenario_pt` | `RamenState::scenario_pt` | 直接覆写 |
| `next_scenario_pt` | `RamenState::next_scenario_pt`（新增） | 直接覆写 |

### 2.3 派生字段处理策略

**C# 端已归一化**：所有派生字段都已填好，Rust 端**直接覆写**即可，不需要：
- 从 `feeling_turn_info_array.remain_turn` 反推 `feeling_guage`（已给值）
- 从 `feeling_info_array` 过滤 `feeling_id==0`（已给队列）
- 从 `active_effect_array` 反推 `last_ramen`（已独立给出）

## 3. playing_state 阶段 dispatch 设计

```text
match playing_state {
    1 => RamenStage::Train,         // 走 3 阶段决策（吃面/特殊/训练）
    5 => RamenStage::Event,         // select_event_choice
    45 | 46 => RamenStage::Settlement, // RMJ 结算，UI 提示无需操作
    48 => RamenStage::SuperRamen,   // 超级拉面生效过渡，无需操作
    _ if (2..=10).contains(&ps) => skip, // 比赛回合，排除
    _ => warn + skip,               // 未知 ps
}
```

### 3.1 阶段间时序（实测）

**RMJ 结算回合（turn 23 / 47 / 71）**：

```text
turn N (RMJ 触发回合):
  snapshot N:   ps=1 source=event     story=触发事件
  snapshot N_2: ps=46 source=command   RMJ 入口
  snapshot N_3: ps=45 source=command   RMJ 出口（feeling_stock 清空）
```

**超级拉面选择（turn 71）**：

```text
snapshot 71:   ps=1   source=command  super_ramen=-1
snapshot 71_2: ps=5   source=event    story=400014017 终极仕込み
snapshot 71_3: ps=46  source=command  super_ramen=-1
snapshot 71_4: ps=48  source=command  super_ramen=-1  feeling_stock=[]
snapshot 72:   ps=1   source=command  super_ramen=2    ← 玩家选定
```

## 4. 测试基线

`crates/umasim/src/game/ramen/turn_import.rs`（新增）：

```text
test_ramen_turn_import_v2:
  - 驱动 151 份样本，逐份调用 RamenGame::from_external_state
  - 验证关键字段 roundtrip（last_ramen / selected_regions / feeling_guage / scenario_pt）
  - 验证 playing_state → stage dispatch 全覆盖
  - 验证 active_effect_array 拆分不改变训练数值（与 C# 端对齐）
```

## 5. 已废止假设（避免回踩）

1. ~~`last_ramen` 仅 load 时有值~~ → **错**：实测每回合吃完面都更新。
2. ~~需要用 active_effect_array 反推 last_ramen~~ → **错**：已独立给出。
3. ~~playing_state 44/45 是地区选择~~ → **错**：实测样本无 ps=44；地区选择不在独立 ps。
4. ~~scenario_pt_cap 反转语义~~ → **已修正**：原 `scenario_pt_cap` 改名为 `next_scenario_pt`。
5. ~~active_effect_array category 1/2/4 含义待查~~ → **搁置**：只需忠实映射，不需解读语义。

## 6. 后续动作

1. ~~补采 A1/A2/A3 关键阶段样本~~ → **已完成**（chara 6204 全 78 回合 151 样本）。
2. 契约定稿（本文件）。
3. 实现 `RamenGame::from_external_state`（核心层）。
4. 实现 `crates/umaai/src/protocol/ramen.rs`（通道层）。
5. 主循环 `parse_game` 按 `scenarioId` 分发。
6. 写 `test_ramen_turn_import_v2` 测试（驱动 151 样本）。
