# umaai-rs 重构需求与三层分离通信架构建议

> 状态：草案（供上游 umaai-rs 参考）
> 视角：下游消费方（hlpatch / uma-juece）与上游规划的对齐
> 原则：下游不要求上游"适配自己"，只建议上游在自身重构规划中顺手固化三个稳定点——核心库 API、数据契约、更新清单

---

## 1. 背景

### 1.1 上游现状与规划

- **现状**：umasim（模拟核心，含拉面杯闭环与合并决策接口）+ umaai（PC 通道，输入源为 PC 端小黑板 UmamusumeResponseAnalyzer 的 thisTurn.json，当前仅温泉杯）+ 独立更新工具（uma-autoupdate，version.toml 清单 + 云 bucket）。
- **规划**（按上游自身开发主线）：
  1. 移除暂不使用的重依赖（如 tract-onnx），此步可顺带做平台依赖分离；
  2. 快速迭代启发式选择算法，产出策略输出；
  3. 自身重构；
  4. 外部接口；
  5. 输出流程改造为 AI 对话流风格，或通过 Web / MCP 服务器提供策略产出服务（规划文档已明确：结构化数据 GameEvent / GameView / TurnSnapshot 的主要消费者是 AI 分析层和外部协议，核心逻辑不依赖终端 UI）。

### 1.2 下游诉求

- hlpatch / uma-juece 需要在 Android 手机独立运行（不依赖 PC 常驻、无 adb 桥接），秒级获得拉面杯决策与规则信息。
- 上游快速迭代期不自适应下游，下游自行适配；因此下游最需要的是：
  1. 一个可交叉编译的纯核心库（umasim 核心，无终端/平台依赖）；
  2. 一份稳定、版本化的数据契约（回合状态 JSON → 决策 JSON）；
  3. 可离线/可热更的数据与策略通道（更新清单格式与上游对齐）。

### 1.3 结论先行

上游规划与下游诉求在三点上天然重合，建议上游在重构时一并落地：
- 平台依赖分离（上游计划）⇔ 下游 Android 交叉编译需求；
- 决策结构化输出（上游对话流/Web/MCP 规划）⇔ 下游决策 JSON 契约；
- 从外部真实状态重建游戏（上游 MCTS 在线搜索刚需）⇔ 下游回合状态恢复需求。

---

## 2. 重构需求清单（按优先级）

### R1 核心 / 通道 / 更新 三层职责分离（确认并固化）

- uma-autoupdate（更新层）：独立仓库，version.toml 清单 + bucket，负责数据/二进制分发。
- umasim（核心层）：模拟 / 搜索 / 状态 / 策略，零 UI、零网络、零文件监视依赖。
- umaai（通道层）：CLI + 小黑板输入（现状）；AI 对话流 / Web / MCP（规划）。

依赖方向：umaai → umasim；核心层不反向依赖任何通道；更新层独立于两者。

核心层验收标准：不 use 任何终端 UI 库、不直接依赖网络/文件监视，可在无终端环境（Android target）编译。

现状差距：umasim lib 路径直接依赖 inquire（crossterm）、colored、comfy_table、tract-onnx、ndarray 等，需要按 R2 拆分。

### R2 平台依赖分离（配合重依赖移除，一步到位）

- 现状阻塞点：umasim 未做 feature 拆分，任何消费者都必须编译全部依赖；其中 inquire（crossterm）在 Android 上无终端语义且交叉编译风险高，tract-onnx 依赖链巨大且会显著膨胀产物。
- 建议 feature 设计：

| Feature | 依赖 | 说明 |
|---|---|---|
| core（无条件） | serde / serde_json / rand / rayon / anyhow / toml / chrono / bincode / hashbrown / thiserror / fs-err / log | 平台无关核心 |
| cli | inquire / colored / comfy-table / indicatif / env_logger / flexi_logger / clap | 终端交互与日志（现状 PC 行为） |
| onnx | tract-onnx / ndarray | 神经网络评估（暂不使用） |
| watcher（可选） | notify | 文件监视（现仅 umaai 使用） |

- 默认值：default = ["cli"]，保证 PC 现有行为不变；Android/嵌入式消费者使用 --no-default-features 或按需启用。
- 验收标准：
  - cargo build --release（PC）行为与现在一致；
  - cargo build --release --target aarch64-linux-android --no-default-features（或最小 feature 集）通过；
  - 下游可协助验证 Android 目标并反馈修复项。

### R3 公开库 API（对上游自身规划也是刚需，双赢）

#### R3.1 turn import：外部回合状态 → RamenGame 中途状态

- 语义：给定一份真实游戏回合状态（拉面杯，含 ramen_data_set 映射后的字段），构建处于"当前回合、当前阶段"的 RamenGame，供决策/搜索使用。
- 现状：BaseGame / RamenGame / RamenState 字段已全部 pub，下游可以 newgame() 后覆写字段实现；但派生字段（persons、distribution、current_effect、events 计数、unresolved_events、deck_can_split 等）的恢复语义没有官方定义。
- 建议：提供官方构造器（如 RamenGame::from_turn_json(&str) 或 from_external_state），内部统一派生字段恢复规则，并配套文档与测试。
- 为什么上游也需要：MCTS 在线搜索必须从真实状态出发（而非从第 0 回合随机模拟）；AI 对话流/Web/MCP 服务接收外部回合状态时同样需要此入口。

#### R3.2 decision out：RamenGame + Trainer → 决策 JSON

- 语义：给定重建后的 RamenGame，调用 Trainer（启发式或 MCTS）得到选定动作，输出结构化决策 JSON（含候选、理由、置信度）。
- 现状：决策逻辑存在于 Trainer trait 与 list_actions / list_combined_ramen_select_actions / apply_combined_ramen_decision 等接口中，但没有统一的序列化输出层。
- 建议：在核心层提供 decide(&RamenGame) → DecisionOut 风格的稳定库函数（不依赖任何通道），通道层（CLI/对话流/Web/MCP/下游 .so）都消费同一输出。
- 为什么上游也需要：AI 对话流需要"决策 + 可读说明"，Web/MCP 需要"决策 + 候选评分"，与下游浮窗显示是同一份输出。

### R4 数据契约冻结与版本化

- 契约内容：
  - 输入：回合状态 JSON（拉面杯，字段见第 4.2 节初稿）；
  - 输出：决策 JSON（字段见第 4.1 节初稿）。
- 规则：
  1. 契约以 schema_version 标识；非破坏性扩展（新增可选字段）不升版本；
  2. 破坏性变更必须递增版本并附迁移说明；
  3. 上游内部重构不得静默改变已发布契约的字段语义（尤其 ramen_data_set 相关字段）。
- 收益：下游适配层只依赖契约，不依赖上游内部结构；上游可放心快速迭代。

### R5 更新清单扩展（uma-autoupdate）

- 现状：version.toml 为「段名 → VersionInfo」的映射，字段含 name / date / index / filelist / sha1 / install_path（支持 %uraroot% 占位）；已有 [ai_data] 段发布 umaDB.json / cardDB.json / text_data_dict.json 到 gamedata/。
- 拉面杯缺口：[ai_data] 缺 constants.json、events.json、scenario_ramen.json、default_config.toml 四个文件。
- 建议：扩展 [ai_data] 的 filelist 补齐上述文件即可，清单 schema 保持不变；下游 Android 更新器实现同构的 JSON 版清单（见 hlpatch 开发计划阶段 4）。

### R6（可选）启发式权重/常量数据化

- 启发式权重、阈值、偏好常量从代码迁移到数据文件，随更新通道发布。
- 收益：下游可离线热更策略参数而不重编 .so；上游调参无需发版。

---

## 3. 三层分离通信架构建议

### 3.1 架构图

```text
                ┌────────────────────────────────────────────┐
                │  契约（唯一稳定边界，版本化）                 │
                │  输入：回合状态 JSON（含 ramen_data_set）      │
                │  输出：决策 JSON                             │
                └────────────────────────────────────────────┘
  上游 umaai-rs                        下游
  ┌────────────────────┐              ┌──────────────────────┐
  │ 通道层 umaai        │              │ hlpatch              │
  │  ├ PC CLI + 小黑板   │              │  └ 回合状态 JSON 输出   │
  │  ├ AI 对话流（规划） │              │ uma-juece             │
  │  ├ Web/MCP 服务(规划)│              │  ├ ramen-decide.so    │
  │  └ 下游 .so 消费     │              │  │  (umasim核心+适配层) │
  │ 核心层 umasim        │              │  ├ 更新器（JSON清单）   │
  │  ├ 模拟/搜索/策略     │              │  └ 浮窗渲染            │
  │  ├ turn import(建议) │              │                      │
  │  └ decision out(建议)│              │                      │
  └────────────────────┘              └──────────────────────┘
  更新层 uma-autoupdate（version.toml + bucket）→ 数据/策略分发
```

### 3.2 关键设计点

1. 契约是唯一稳定边界：上游所有通道（CLI/对话流/Web/MCP）与下游所有部署（本地 .so / 远程服务）共用同一份契约；任何一侧改内部实现，只要契约不变就互不干扰。这是下游应对上游快速迭代的隔离层。
2. 通道层可多形态并存：AI 对话流、Web/MCP 服务、下游 .so 都是"核心层 + 某通道"的组合；新增通道不修改核心层。
3. 版本三轴管理：
   - 契约版本（schema_version，控制兼容）；
   - 内核版本（umasim rev，控制能力）；
   - 数据版本（gamedata/策略文件，控制规则）。
   - 下游 .so 更新频率最低（随 APK），数据/策略热更频率最高（走更新通道）。
4. 协作模式：下游提供契约初稿（本文第 4 节）→ 上游在启发式落地时定稿 → 双方按 schema_version 维护兼容。

---

## 4. 决策 JSON 初稿（供参考，最终由上游定义）

### 4.1 输出契约（决策 JSON）示例

```json
{
  "schema_version": 1,
  "turn": 23,
  "stage": "RamenSelect",
  "decision": {
    "ramen": 3,
    "special_targets": [0, 1, 2],
    "operation": { "type": "Train", "train": 2 }
  },
  "explanation": {
    "reason": "体力充足，吃面后训练收益最高",
    "confidence": 0.82,
    "detail": "吃面(地区3)后训练：速+18 耐+6 技Pt+42"
  },
  "alternatives": [
    { "ramen": null, "operation": { "type": "Rest" }, "score": 61.4 },
    { "ramen": null, "operation": { "type": "Train", "train": 4 }, "score": 58.9 }
  ]
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|---|---|---|
| schema_version | int | 契约版本 |
| turn | int | 决策对应的回合（0-77） |
| stage | string | 当前阶段：Begin / Distribute / RamenSelect / SpecialSelect / Train / AfterTrain / NextTurn / RegionSelect / SuperRamenSelect / Settlement |
| decision.ramen | int 或 null | 选定的地区拉面下标；null = 不吃面 |
| decision.special_targets | [int;3] | 隐藏风味用量 [A, B, C] |
| decision.operation.type | string | Train / Race / Rest / NormalOuting / FriendOuting / Clinic / RegionSelect / StageOnly |
| decision.operation.train | int 或 null | type=Train 时的训练下标（0-4：速/耐/力/根/智） |
| decision.operation.regions | [int;3] 或 null | type=RegionSelect 时的地区组合 |
| explanation.reason | string | 人类可读理由（AI 对话流/浮窗直接展示） |
| explanation.confidence | float | 置信度（MCTS 就绪后可为搜索统计） |
| explanation.detail | string | 收益明细（可选） |
| alternatives[] | array | 候选动作与评分（可选，MCTS 就绪后启用） |

**与 RamenAction 的映射**：ramen + special_targets 对应合并决策路径（list_combined_ramen_select_actions + apply_combined_ramen_decision）；三阶段模式下 ramen/special_targets 分阶段输出，operation 在 Train 阶段输出。

### 4.2 输入契约初稿（回合状态 JSON）

**顶层**（沿用 umaai 现有 GameStatusBase 的 camelCase 命名，便于上游采纳）：
uma_id、uma_star、turn、vital、max_vital、motivation、five_status、five_status_limit、skill_pt、skill_score、total_hints、train_level_count、pt_score_rate、failure_rate_bias、is_ill、isQieZhe、isAiJiao、is_positive_thinking、is_refresh_mind、is_lucky、zhongMaBlueCount、is_racing、card_id、persons、person_distribution、locked_training_id、friendship_noncard_yayoi、friendship_noncard_reporter、friend_stage、friend_outgoingUsed、playing_state、race_history、story

**ramen 段**（拉面杯专用，映射自 ramen_data_set 提取结果）：
feeling_stock（诀窍库存 [A,B,C]）、feeling_slot（诀窍槽 [A,B,C]）、feeling_queue（获得顺序）、special_feeling（隐藏风味）、selected_regions（当年已选地区 [usize;3]）、current_ramen（当前回合吃面）、scenario_pt、rmj_results、train_level_bonus、super_ramen、eat_count、train_feeling_type、checkpoint_pt、stage（当前阶段）、pending_ramen / pending_special_targets / combined_decision（三阶段决策中间态，可选）

**来源与可信度标注**：hlpatch 现有 /summary 的 ramen 段 + /debug/rameninfo、/debug/ramenfields、/debug/ramengains、/debug/ramen_dataset_path 等端点可提取大部分字段；每字段应标注"已验证 / 启发式 / 待确认"（沿用 hlpatch 的证据分级原则），未验证字段不得冒充事实。

### 4.3 版本化与兼容建议

1. 契约文件单独存放（如 docs/contract/ramen_turn_v1.md、docs/contract/ramen_decision_v1.md）；
2. 上游定稿后由上游仓库维护；下游镜像并实现解析器；
3. 每次破坏性变更：升 schema_version + 变更说明 + 下游适配点清单。

---

## 5. 与上游规划时间线对齐

| 上游步骤 | 对应本文建议 | 下游受益 |
|---|---|---|
| 移除重依赖 + 平台依赖分离 | R2 | Android 交叉编译可行 |
| 启发式选择算法落地 | R3.2 / R4 | 决策 JSON 契约定稿，下游接入 |
| 自身重构 | R1（三层固化） | 核心 API 稳定 |
| MCTS 搜索 | R3.1（turn import） | 在线决策从真实状态出发 |
| 输出改造（AI 对话流 / Web / MCP） | R3.2 / R4 | 下游 .so 与远程服务同契约可切换 |
| 更新工具适配拉面杯 | R5 | 数据热更通道打通 |

> 备注：以上为建议，不要求上游为下游排期；下游在各项落地前均按"上游零改动"路径推进（详见 hlpatch 开发计划）。
