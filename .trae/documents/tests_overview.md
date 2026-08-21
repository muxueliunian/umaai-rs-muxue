# 测试一览

本文件按模块分类、用一句话描述每个测试的功能，便于快速定位和评估覆盖率。

总计：**155 个测试**（含 1 个 async），分布在 `crates/umasim/src/` 与 `crates/umaai/src/` 各文件中。

## 目录

- [拉面杯](#拉面杯) — 110 个
  - [`game.rs`](#gamers阶段流转与集成) — 36
  - [`rules.rs`](#rulesrs诀窍吃面与地区) — 27
  - [`effects.rs`](#effectsrs训练数值与得意率) — 15
  - [`action.rs`](#actionrs动作枚举与列表) — 14
  - [`events.rs`](#eventsrs友人事件与剧本事件) — 5
  - [`policy.rs`](#policyrs固定策略与手写策略核心) — 9
  - [`logging_trainer.rs`](#logging_trainerrs决策日志包装) — 2
  - [`ramen_handwritten_trainer.rs`](#ramen_handwritten_trainerrs手写策略测试壳) — 2
- [配置 / 数据加载](#配置--数据加载) — 10
- [基础游戏](#基础游戏) — 10
- [决策日志](#决策日志outputdecision_logrs) — 4
- [其他](#其他) — 4
- [基准测试](#基准测试)（bench_base / bench_compositions 运行说明）

---

## 拉面杯

### `game.rs`（阶段流转与集成）

**初始化 / 基础**
- `test_ramen_game_newgame` — 新建游戏合法性
- `test_ramen_newgame_requires_new_friend` — 缺新友人卡时拒绝开局

**训练参数 / 数值**
- `test_train_param_decomposition` — 训练参数拆解正确性
- `test_random_distribution_training_value` — 随机分配下训练数值计算
- `test_ramen_deyilv_includes_scenario_bonus` — 得意率叠加剧本加成
- `test_random_event_generation` — 随机事件生成

**端到端 / 集成**
- `test_ramen_game_full_loop` — 拉面杯基础闭环（带日志）
- `test_ramen_silent_loop` — 拉面杯基础闭环（关日志）
- `test_manual_trainer_full_game` — ManualTrainer 完整流程
- `test_manual_trainer_hint_special_path` — 第3年 hint_special 路径不崩溃

**决策路径**
- `test_three_stage_decision_flow` — 三阶段（RamenSelect→SpecialSelect→Train）衔接
- `test_combined_decision_path_skips_special_select` — 合并决策跳过 SpecialSelect
- `test_combined_decision_path_no_ramen` — 合并决策不吃面路径
- `test_combined_decision_invalid_targets_rejected` — 合并决策拒绝非法 targets
- `test_three_stage_path_unaffected_by_combined_flag` — combined flag 不污染三阶段路径

**RMJ / 剧本事件**
- `test_select_rmj_choice_by_result` — RMJ 事件按 result 选择分支
- `test_rmj_event_year` — RMJ 事件按回合映射年份
- `test_rmj_event_apply_success` — RMJ 成功 apply 正确
- `test_rmj_event_apply_fail` — RMJ 失败 apply 正确
- `test_rmj_event_immediate_apply_at_turn_23` — 第1年 RMJ 当回合立即触发
- `test_scenario_pt_reset_after_rmj` — RMJ 结算后 scenario_pt 归零
- `test_generate_events_uma_debut` — 马娘登场事件生成（turn=0）
- `test_generate_events_classic_newyear` — 经典新年事件（turn=24）
- `test_generate_events_ancient_newyear` — 古马新年事件（turn=48）
- `test_add_mandatory_events_ticket_at_48` — turn=48 抽签事件
- `test_add_mandatory_events_ending_at_77` — turn=77 结局事件

**超级拉面**
- `test_super_ramen_base_effect_vital_motivation` — 基础效果（体力+干劲）每回合生效
- `test_super_ramen_saihou_one_time_only` — 赛后加成仅 turn=72 一次性

**hint_special**
- `test_hint_special_inactive_without_ramen` — 未吃面时不激活
- `test_hint_special_inactive_year1_2` — 第1/2年不激活
- `test_hint_special_active_year3` — 第3年吃面后激活
- `test_hint_special_only_at_listed_trains` — 仅在配置的训练位置生效
- `test_hint_special_inactive_low_card_types` — 支援卡种类<4 时不激活

**第3年地区策略**
- `test_ramen_region_strategy_fixed_skips_enumeration` — 第3年 Fixed 策略跳过枚举
- `test_year1_2_always_all_regardless_of_strategy` — 第1/2年 Fixed 不生效

**回合菜单约束**
- `test_skip_ramen_select_for_turn_0_1_and_super_ramen` — 回合 0-1/超级拉面短路 Distribute→Train

### `rules.rs`（诀窍、吃面与地区）

**吃面消耗**
- `test_consume_for_ramen` — 吃面正常消耗
- `test_consume_for_ramen_errors` — 吃面错误（配方/隐藏风味/库存不足）
- `test_can_make_ramen` — 能否做面判定

**诀窍 / 诀窍槽**
- `test_feeling_overflow` — 诀窍总数超过 10 时淘汰最早
- `test_gauge_overflow` — 诀窍槽溢出也只加 1 诀窍并清零
- `test_gauge_base_distribution` — 按配方比例分配 base_sum
- `test_fill_gauge_after_train_normal` — 训练后填充（普通回合）
- `test_fill_gauge_after_train_xiahesu_max` — 训练后填充（夏合宿全 MAX）
- `test_fill_gauge_after_non_train_normal` — 非训练后填充
- `test_fill_gauge_after_non_train_xiahesu` — 非训练后填充（夏合宿全 MAX）
- `test_fill_gauge_after_non_train_xiahesu_partial` — 夏合宿非训练但部分 MAX
- `test_get_turn_special_feeling` — 固定回合隐藏风味加成表

**特殊目标枚举**
- `test_list_special_targets_full_stock_sapporo_9` — 库存富余候选（札幌 [2,2,1]）
- `test_list_special_targets_min_needed_a3b1c1` — 库存有缺口（A 缺3/B 缺1/C 缺1）
- `test_list_special_targets_impossible` — 库存完全不够
- `test_list_special_targets_no_special_feeling` — 无隐藏风味时只生成 0/0/0 候选
- `test_list_special_targets_recipe_with_zero_dim` — 配方含 0（如 [2,3,0]）
- `test_list_special_targets_sorted_ascending` — 候选按 sum(t) 升序

**训练加成**
- `test_train_feeling_bonus` — 训练角标加成

**PT 增量**
- `test_calc_ramen_pt_gain` — 吃面 PT 增量公式

**RMJ**
- `test_check_rmj` — RMJ 三种 result（Fail/Success/GreatSuccess）判定

**地区**
- `test_get_region_range` — 按年份取可选地区 ID 范围
- `test_get_region_clone_trains` — 地区分身位置
- `test_get_super_ramen_clone_train_options` — 超级拉面分身位置选项
- `test_calc_region_bonus` — 地区词条加成按 PT 档位
- `test_validate_region_selection` — 地区组合合法性校验

**NPC**
- `test_npc_chara_ids` — NPC chara_id 集合正确

### `effects.rs`（训练数值与得意率）

**训练数值计算**
- `test_apply_training_value_status` — 属性训练下层/上层数值
- `test_apply_training_value_pt` — PT 训练数值计算
- `test_apply_training_value_upper_limit` — 上层数值上限约束
- `test_apply_training_value_lower_cap` — 下层数值上限100

**训练效果来源**
- `test_calc_effect_pt_only` — PT 档位效果
- `test_calc_effect_with_eating` — 普通吃面 buff 叠加
- `test_calc_effect_rmj_success` — RMJ 成功加成
- `test_calc_effect_super_ramen` — 超级拉面效果
- `test_calc_effect_super_ramen_with_split` — 超级拉面 + 分身
- `test_calc_effect_non_shining` — 非友情训练无 youqing

**剧本得意率**
- `test_calc_scenario_deyilv_normal_pt_only` — 普通回合 PT 档位得意率
- `test_calc_scenario_deyilv_normal_with_rmj_success` — 普通回合 + RMJ 成功
- `test_calc_scenario_deyilv_normal_with_rmj_fail` — 普通回合 + RMJ 失败
- `test_calc_scenario_deyilv_super_ramen` — 超级拉面回合得意率
- `test_calc_scenario_deyilv_super_ramen_rmj_fail` — 超级拉面 + RMJ 失败

### `action.rs`（动作枚举与列表）

**枚举与显示**
- `test_ramen_action_display` — RamenAction 的 Display 输出格式
- `test_ramen_action_properties` — `is_eating_ramen` / `base_operation` getter
- `test_combined_select_keeps_targets_when_eating` — 合并决策吃面时保留 targets
- `test_combined_select_normalizes_targets_when_no_ramen` — 不吃面时 targets 归零

**列表生成**
- `test_list_ramen_choices` — 面选择枚举（含不吃）
- `test_list_operations` — Operation 列表（含夏合宿/友人/治病等条件）
- `test_list_all_actions` — 完整动作列表（吃面×操作笛卡尔积）
- `test_list_train_actions_no_ramen_field` — Train 阶段动作不携带 ramen/special_targets
- `test_list_ramen_select_actions_full` — 拉面选择阶段（3 面都可选）
- `test_list_ramen_select_actions_no_available` — 拉面选择阶段（无可选面）
- `test_list_special_select_actions_uses_special_targets` — 隐藏风味选择阶段
- `test_list_combined_ramen_select_actions_full` — 合并决策完整候选
- `test_list_combined_ramen_select_actions_no_available` — 合并决策无可选
- `test_get_available_ramens` — 当年可用面判定

### `events.rs`（友人事件与剧本事件）

- `test_event_ids` — 事件 ID 表正确
- `test_turn_special_feeling` — turn→特殊隐藏风味数量映射
- `test_friend_event_state_lifecycle` — 友人事件状态机（首次/点击/解锁/出行）
- `test_friend_visibility` — 友人可见性
- `test_assign_train_feeling_type` — 训练角标分配每种至少1次

### `policy.rs`（固定策略与手写策略核心）

**固定策略**
- `test_fixed_region_selection` — 各年份固定地区选择
- `test_fixed_super_ramen_selection` — 超级拉面固定选项二

**手写策略核心（RamenPolicy）**
- `test_gate_ill_clinic` — 守门：生病必治病
- `test_gate_vital_low_rest` — 守门：体力低必休息
- `test_gate_motivation_low_outing` — 守门：心情低必外出
- `test_train_selector_deterministic` — 健康局面确定性选训练（两次一致）
- `test_special_selector_min_hidden` — SpecialSelect 最省隐藏风味
- `test_event_selector_higher_value` — 事件选效果总值高者
- `test_region_selector_valid_and_deterministic` — 地区组合可打分且确定性

### `logging_trainer.rs`（决策日志包装）

- `test_logging_trainer_records_full_game` — 完整局决策记录覆盖（三阶段/事件/地区选择）
- `test_reproducible_same_seed` — 同 seed 两次整局决策序列与评分一致（可复现性）

### `ramen_handwritten_trainer.rs`（手写策略测试壳）

- `test_handwritten_full_game` — 完整 77 回合跑通（评分/RMJ/吃面数输出）
- `test_handwritten_reproducible` — 同 seed 两次整局评分一致

---

## 配置 / 数据加载

**`utils.rs`**
- `test_validate_game_config_scenario_enum` — scenario 枚举校验
- `test_validate_game_config_trainer_enum` — trainer 枚举校验
- `test_validate_game_config_ramen_region_fixed_length` — Fixed 策略下 ramen_region_fixed 长度校验
- `test_resolve_default_config_path` — 默认配置路径解析

**`gamedata/event.rs`**
- `test_load_and_explain_all_events` — 加载并 explain 全事件
- `test_load_and_explain_ramen_events` — 加载并 explain 拉面事件

**`gamedata/mod.rs`**
- `test_uma_data` — 马娘数据加载
- `test_support_data` — 支援卡数据加载
- `test_consts` — 常量加载
- `test_turn_mask` — 回合掩码

---

## 基础游戏

**`game/uma.rs`**
- `test_uma` — Uma 基础结构
- `test_win_races` — 比赛胜场计算

**`game/base/mod.rs`**
- `test_explain` — BaseGame explain
- `test_newgame` — 新建基础游戏
- `test_can_self_race_bounds` — 自选比赛边界（13-71 允许，URA 回合禁止）
- `test_can_friend_outing_bounds` — 友人出行边界（解锁/回合 <72/次数未用完）

**`game/base/basic.rs`**
- `test_newgame` — 新建基础游戏（BasicGame）

**`game/support_card.rs`**
- `test_support` — 支援卡基础结构

**`game/inherit.rs`**
- `test_inherit` — 继承值生成

**`game/mod.rs`**
- `test_friend` — FriendState 基础结构

---

## 决策日志（`output/decision_log.rs`）

- `test_csv_escape` — CSV 字段转义（逗号/引号/换行）
- `test_csv_row_and_header` — 单行序列化 + 表头格式
- `test_empty_log` — 空日志 CSV 输出
- `test_save_to_roundtrip` — 落盘与读取往返

---

## 其他

**`neural/evaluator.rs`**
- `test_random_evaluator_send_sync` — 随机评估器线程安全

**`crates/umasim/src/main.rs`**
- （无 `#[test]`，仅烟雾运行入口）

**`crates/umaai/src/main.rs`**
- `test_watch` — watch 模式（async）
- `test_urafile` — URA 存档文件解析

---

## 基准测试（`bin/bench_base.rs`）

固定种子批量跑批，产出 RandomTrainer 基线分布（分数/PT/RMJ/耗时）与决策轨迹，
用于量化手写策略的改进（对应手写策略计划 §8「先立地基」）。

```bash
# 默认读取 workspace 根 bench_config.toml（runs=20, seed=42）
cargo run --release --bin bench_base

# 自定义局数/种子/开启决策日志/输出目录（CLI 覆盖 config）
cargo run --release --bin bench_base -- --runs 100 --seed 7 --log --out logs
```

- 参数：`--runs N` 局数、`--seed S` 基础种子（第 i 局 = seed+i）、`--log` 落盘决策日志（默认关）、`--out DIR` 输出目录
- 产出（默认 `logs/`）：`bench_base_results.csv`（每局一行：seed/分数/rank/五维/PT/RMJ/吃面数/耗时）+ 汇总统计
  （分数 mean/median/min/max/std、按阶段分组的决策耗时、吞吐）；`--log` 时另产出 `bench_base_decision_<seed>.csv`
- 可复现性：同一参数下游戏结果完全一致（决策 RNG 与规则层 `internal_rng` 均由 seed 派生；`elapsed_ms` 属运行耗时，允许波动）
- 性能基线（2026-08-21 实测）：RandomTrainer 单局 ~1.2ms，吞吐 ~815 局/s——手写策略须保持 O(候选数) 简单才有 rollout 接入意义

---

### `bin/bench_compositions.rs`

固定种子遍历五种普通支援卡各 0..=3 张、合计 5 张再加固定友人的全部 101 种构成，输出评分、五维、训练技能 PT、RMJ 和友人出行聚合 CSV。运行设施复用 `umasim::bench`（`bin/bench_base.rs` 同）。

```bash
cargo run --release --bin bench_compositions -- --runs 100 --seed 42 --trainer handwritten --out logs/bench_compositions.csv
```

- 代表卡选择：各类型取最新 5 张满破 SSR 作候选池，跳过满破面板和值（友情+干劲+训练）<70 的弱卡，按 card_id 倒序取 3 张；`--min-panel N` / `--pool-size N` / `--pick N` 可调，`--cards-file cards.toml` 手动指定兜底（每类型满破 idrank 列表）
- `test_enumerate_all_101_compositions` — 严格验证合法构成总数为 101，且每种合计 5 张、单类型不超过 3 张
- `test_build_all_composition_decks` — 验证全部构成都生成 5 张普通卡 + 1 张固定友人
- `umasim::bench` 模块测试（4 个）：seed 双 RNG 可复现 / summarize 统计 / percentile 分位 / 真实 cardDB 默认参数选卡集成验证

## 未来缩减参考（规则固化后讨论）

**前提**：拉面杯目前仍在重构期，公式随时可能调整；规则固化后可重新评估公式测试的密度。

**公式测试的价值三层次**：
1. **回归保护** — 重构期防止破坏；规则固化后价值下降（停止改动后无破坏）
2. **文档化** — 把预期行为固化在代码里；规则固化后可由正式文档替代
3. **调试辅助** — 快速定位数值问题；规则固化后价值保留

**建议的缩减方向**（"保留 happy path + 关键边界，删除纯中间态"）：

| 系列 | 当前 | 可压缩到 | 缩减点 |
|------|------|---------|--------|
| `test_calc_scenario_deyilv_*` | 5 | 3 | normal_with_rmj_success + normal_with_rmj_fail 合并；super_ramen 单独保留 |
| `test_apply_training_value_*` | 4 | 2 | status + lower_cap 合并；pt + upper_limit 合并 |
| `test_calc_effect_*` | 6 | 3 | pt_only + eating 合并；rmj_success 单独；super_ramen + super_ramen_with_split 合并 |
| `test_list_special_targets_*` | 6 | 3-4 | 保留 full_stock + min_needed + impossible + sorted；no_special_feeling + recipe_with_zero_dim 可视为 full_stock 变体 |
| `test_fill_gauge_*` | 5 | 3 | train_normal + non_train_normal 合并；各自 xiahesu 路径合并；partial 单独保留 |

**预估可缩减 15-20 个**（121 → 约 100-105）。

**应保留**：
- 端到端（3 个）：整体回归必备
- 决策路径（5 个）：三阶段/合并决策是核心设计
- RMJ/事件（10+ 个）：关键业务流程
- `hint_special_*`（5 个）：每个 case 不同，全保留
- 每个公式函数至少 1 个 happy path

**风险**：游戏更新（数据库调整、剧情加强）时边界 case 测试可能重新需要；公式逻辑本身不会变，风险可控。

**讨论结论（2026-08-20）**：当前不执行——重构期公式随时可能改动，现在缩减可能需要后续重新加；待规则固化后重新评估。