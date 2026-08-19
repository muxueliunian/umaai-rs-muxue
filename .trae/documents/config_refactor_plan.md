# 配置系统整理实施计划（Phase 2）

> 对应 ramen_refactor_development_plan.md Phase 2「配置系统整理」。
> 本文档为实施计划。**节奏已确认：渐进式（2026-08-19）**。

## 1. 背景与目标

- 保留"开发者默认配置 + 用户配置覆盖"两层模型，业务代码只接收**类型化运行时配置**。
- 加载集中处理相对路径、缺省值、覆盖规则和校验；业务模块不自行读取 TOML、不依赖当前工作目录。
- 拉面杯固定配方和规则不放进用户配置。
- 清理旧温泉字段（onsen_order 等）须等所有入口迁移完成（Phase 6 一并删除）。

## 2. 现状梳理

### 配置加载（utils.rs::load_game_config，全部入口共用）
```text
gamedata/default_config.toml → GameConfig（serde 全量反序列化）
game_config.toml              → OverrideGameConfig（仅 [onsen_order]/[config_override]/[mcts] 少数字段）
        → merge() → GameConfig
```

- 入口 8 处：umasim/main、umaai/main、umaai/protocol、analyzer/main、ramen_manual、handwritten_evaluator、unused 两个 bin；全部依赖工作目录相对路径。
- `GameConfig` 为平铺混合体（模拟参数/日志/MCTS/温泉/collector 混在一个 struct）。

### gamedata 三类数据

| 类 | 文件 | 归属 |
|---|---|---|
| 1. 开发期调整的剧本参数 | events.json（全局事件，onsen/ramen 共用）、scenario_ramen.json、scenario_onsen.json（待退役） | 剧本 |
| 2. 跟随游戏更新的数据 | umaDB.json、cardDB.json、text_data_dict.json | 游戏数据 |
| 3. 机制 + 偏好混合 | constants.json（19 项）、"constants - 不打泥地.json"（手工变体，不参与加载） | 见 §4 甄别 |

## 3. 目标架构

```text
gamedata/default_config.toml  ← 开发者默认（按段组织，详细）
game_config.toml              ← 用户覆盖（顶层常用区 + 低频段覆盖）
CLI 参数                       ← 按 bin 需要
        ↓ merge + 校验
RuntimeConfig（类型化）
```

### 五段职责

| 段 | 内容 |
|---|---|
| simulation | scenario、trainer、uma、cards、blue_count、extra_count、simulation_count |
| search | 现有 MctsConfig 全部 + mcts_selection + neuralnet_model_path + **mcts_turn_bonus** + **pt_favor_rate** |
| policy | 拉面杯地区选择策略、超级拉面选择策略（新增，见 §6） |
| output | log_level、统计级别、输出路径 |
| dev | collector、线程数、调试开关 |

### 用户覆盖层设计（回应"常用项横跨五类"问题）

**TOML 用户层与 Rust 类型结构解耦**：

- **game_config.toml 顶层常用区**：用户高频调整项直接顶层平铺（uma、cards、scenario、trainer、log_level、num_threads、ramen_region_strategy、ramen_region_fixed、race_grades 等），serde 映射到对应子配置——用户无需记忆段归属。
- **低频/详细项**才进入 [search]、[dev] 等段覆盖。
- 该设计与 Rust 侧是否保留聚合壳无关（渐进式下同样采用）。

## 4. constants.json 甄别结果（2026-08-19 用户确认）

| 处理 | 字段 | 去向 |
|---|---|---|
| 移入配置层（用户可调） | mcts_turn_bonus | → [search]（搜索参数） |
| | pt_favor_rate | → [search]（评分偏好，与 mcts_selection 同类） |
| | race_grades | → 顶层（用户常调比赛表；替换"不打泥地"变体文件方式） |
| 移入剧本数据（每剧本不同） | five_status_limit_base | → scenario_ramen.json（拉面杯覆盖全局默认） |
| | no_event_turns | → scenario_ramen.json（同上） |
| 保留 constants.json（固定） | 其余项（hint/event/rest/training_vital/names 等） | 不动 |
| 稍后人工更新 | rank_scores、rank_names、five_status_final_score | 已记入 issues.md，待人工更新 |

## 5. 拉面杯新增配置（顶层，用户常调）

- `ramen_region_strategy = "all" | "fixed"`：第三年地区选择策略；`"fixed"` 时跳过组合枚举，解决 120 组合爆炸。
- `ramen_region_fixed = [[y1], [y2], [y3]]`（可选）：`"fixed"` 时的三年固定地区组合；缺省回退当前固定顺序策略。

## 6. 温泉字段处理

- `onsen_order`、`mcts_selected_onsen` 保留并标注 deprecated（umaai 仍在使用），Phase 6 随温泉代码删除。
- `scenario_onsen.json` 暂留。

## 7. 实施步骤

> 渐进式路线：保留 GameConfig 聚合壳、内部按职责分组类型化；每步独立可编译可提交，调用点零改动。

1. **数据迁移（独立小步，两种节奏通用）**：
   - constants.json 移除三项（mcts_turn_bonus/pt_favor_rate/race_grades），默认值落入 default_config.toml 对应段；修改引用点（mcts_trainer.rs、uma.rs、比赛等级计算处）。
   - five_status_limit_base、no_event_turns 增加 scenario_ramen.json 剧本覆盖字段，加载时剧本数据优先。
2. **TOML 结构调整**：default_config.toml 按五段重组；game_config.toml 改为"顶层常用区 + 段覆盖"。
3. **config.rs 重构**：GameConfig 内部分组为子结构（simulation/search/policy/output/dev），serde 兼容新 TOML；OverrideGameConfig 改为同段 Option 化覆盖（未设置不覆盖）。
4. **加载集中化**：路径收敛为加载层常量（可选环境变量），集中校验（scenario/trainer 枚举、cards 长度等）。
5. **拉面杯策略配置接入**：policy 段 → RamenGame 地区选择读取 ramen_region_strategy/ramen_region_fixed，替代硬编码固定顺序。
6. **入口验证**：全部入口 load_game_config() 零改动运行；ramen_manual 从顶层读 uma/cards。
7. **文档收尾**：更新 project_context.md、changelog.md，开发计划 Phase 2 标记完成。

## 8. 验证

- 全 workspace release 编译 + 测试通过。
- 三层覆盖行为：default → user（顶层常用 + 段覆盖）→ CLI。
- ramen_manual 正常启动；拉面杯 fixed 地区策略生效（第三年不再枚举 120 组合）。
- onsen 剧本（umaai）行为不变（温泉字段 deprecated 但可用）。

## 9. 决策记录：渐进式（2026-08-19 用户确认）

- 配置拆得过细从架构上更直接，但可能更杂乱、难以记忆；渐进式保留 GameConfig 聚合壳、内部子配置类型化，收敛快、每步可提交。
- 用户常用项横跨五类的问题由"顶层常用区"设计解决（§3），与 Rust 拆分方式无关。
- 一步到位（彻底拆 5 个独立类型 + 全部调用点迁移）暂不采用；若未来需要无壳形态，可留到 Phase 6 温泉清理时评估。
