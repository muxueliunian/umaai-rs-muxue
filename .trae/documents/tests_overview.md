# 测试一览

本文件按模块分类、用一行描述每个测试的功能，便于快速定位和评估覆盖率。
基于 2026-08-28 master 实测口径：`cargo test --release --lib` 共 **330 个**（326 passed / 4 ignored）；
另 `umaai` bin 有 2 个测试（test_watch / test_urafile），因依赖环境变量与工作目录当前失败，属历史遗留。

## 目录

- [拉面杯规则层](#拉面杯规则层) — 157
  - [`ramen/game.rs`](#ramengamers-48) — 48
  - [`ramen/rules.rs`](#ramenrulesrs-30) — 30
  - [`ramen/policy.rs`](#ramenpolicyrs-24) — 24
  - [`ramen/action.rs`](#ramenactionrs-21) — 21
  - [`ramen/effects.rs`](#rameneffectsrs-15) — 15
  - [`ramen/features.rs`](#ramenfeaturesrs-9) — 9
  - [`ramen/events.rs`](#rameneventsrs-5) — 5
  - [`ramen/rng_consistency.rs`](#ramenrng_consistencyrs-4) — 4
  - [`ramen/state.rs`](#ramenstaters-1) — 1
- [基础游戏](#基础游戏) — 20
- [搜索层](#搜索层search) — 34
- [训练员](#训练员trainer) — 32
- [输出层](#输出层output) — 30
- [配置 / 数据加载](#配置--数据加载) — 26
- [采样器](#采样器sampler) — 13
- [基准设施](#基准设施bench) — 9
- [RNG](#rngrng) — 8
- [神经评估](#神经评估neural) — 1
- [基准测试运行说明](#基准测试运行说明)

---

## 拉面杯规则层

### `ramen/game.rs` (48)

**初始化 / 端到端**
- `test_ramen_game_newgame` — 新建游戏合法性
- `test_ramen_newgame_requires_new_friend` — 缺新友人卡时拒绝开局
- `test_ramen_game_full_loop` — 拉面杯基础闭环（带日志）
- `test_ramen_silent_loop` — 拉面杯基础闭环（关日志）
- `test_manual_trainer_full_game` — ManualTrainer 完整流程
- `test_manual_trainer_hint_special_path` — 第3年 hint_special 路径不崩溃

**训练参数 / 数值 / 事件生成**
- `test_train_param_decomposition` — 训练参数拆解正确性
- `test_random_distribution_training_value` — 随机分配下训练数值计算
- `test_ramen_deyilv_includes_scenario_bonus` — 得意率叠加剧本加成
- `test_random_event_generation` — 随机事件生成

**决策路径（三阶段 / 合并）**
- `test_three_stage_decision_flow` — 三阶段（RamenSelect→SpecialSelect→Train）衔接
- `test_combined_decision_path_skips_special_select` — 合并决策跳过 SpecialSelect
- `test_combined_decision_path_no_ramen` — 合并决策不吃面路径
- `test_combined_decision_invalid_targets_rejected` — 合并决策拒绝非法 targets
- `test_three_stage_path_unaffected_by_combined_flag` — combined flag 不污染三阶段路径

**回合 / 阶段序列**
- `test_turn2_stage_sequence` — turn 2 阶段序列 Begin→RegionSelect→BeginAfterRegionSelect→Distribute
- `test_non_turn2_has_no_begin_after_region_select` — 非 turn 2 不产生 BeginAfterRegionSelect 阶段
- `test_skip_ramen_select_for_turn_0_1_and_super_ramen` — 回合 0-1/超级拉面短路 Distribute→Train

**地区选择（第1年枚举 / 第3年 Fixed / 分身 / 归档）**
- `test_year1_region_select_uses_full_enumeration` — 第 1 年地区选择走全枚举
- `test_year3_fixed_list_actions_single_candidate` — 第 3 年 Fixed 策略动作列表仅单个固定候选
- `test_region_clones_absent_priority` — 地区分身：缺席卡优先补位
- `test_region_clones_per_train_semantics` — 地区分身按训练位语义分配
- `test_rmj_archives_yearly_counters_before_reset` — RMJ 年度计数在归零前按年归档

**人头 / 缺席 / 人数计数**
- `test_absent_recorded_and_npc_always_present` — 分身分配缺席名单入状态且 NPC 恒不在缺席列
- `test_absent_weight_by_type` — 缺席权重按人头类型（NPC/卡）区分
- `test_distribute_person_two_stage_absent` — distribute_person 两步算法（合法集直选 + 缺席处理）
- `test_person_deck_index_mapping_full_game` — 人头与卡组槽位反查映射整局一致
- `test_training_buff_person_deck_mapping` — 训练 buff 计算按卡组槽位取人头
- `test_count_training_persons_by_type` — 训练人数按 `PersonType` 判定（替代硬编码下标）
- `test_count_training_persons_onsen_unchanged` — 温泉侧人数计数行为不变

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

### `ramen/rules.rs` (30)

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
- `test_train_feeling_bonus` — 训练角标加成

**特殊目标枚举**
- `test_list_special_targets_full_stock_sapporo_9` — 库存富余候选（札幌 [2,2,1]）
- `test_list_special_targets_min_needed_a3b1c1` — 库存有缺口（A 缺3/B 缺1/C 缺1）
- `test_list_special_targets_impossible` — 库存完全不够
- `test_list_special_targets_no_special_feeling` — 无隐藏风味时只生成 0/0/0 候选
- `test_list_special_targets_recipe_with_zero_dim` — 配方含 0（如 [2,3,0]）
- `test_list_special_targets_sorted_ascending` — 候选按 sum(t) 升序
- `test_special_targets_enumeration_is_within_ten` — 隐藏风味候选枚举不超过库存上限 10
- `test_special_targets_sum_invariant` — 候选配方之和不变量

**PT / RMJ / 地区 / NPC**
- `test_calc_ramen_pt_gain` — 吃面 PT 增量公式
- `test_check_rmj` — RMJ 三种 result（Fail/Success/GreatSuccess）判定
- `test_combined_ramen_actions_peak` — 合并拉面动作的收益峰值
- `test_get_region_range` — 按年份取可选地区 ID 范围
- `test_get_region_clone_trains` — 地区分身位置
- `test_get_super_ramen_clone_train_options` — 超级拉面分身位置选项
- `test_calc_region_bonus` — 地区词条加成按 PT 档位
- `test_validate_region_selection` — 地区组合合法性校验
- `test_npc_chara_ids` — NPC chara_id 集合正确

### `ramen/policy.rs` (24)

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
- `test_decide_super_ramen_finds_option_two` — decide_super_ramen 对样例局面选中选项二

**自选比赛守门 / 打分自洽性**
- `test_remaining_race_slots` — 区间内剩余可比赛回合数（按当前回合裁剪，排除回合 11-12 与 URA 段）
- `test_free_race_gate` — 硬守门四场景（宽裕不干预 / 紧张强制比赛 / 已达标 / 无要求马娘）
- `test_free_race_gate_giveup_recorded` — 自选比赛放弃被记录为显式决策
- `test_free_race_gate_quiet_after_done` — 比赛达标后守门静默不再强制
- `test_free_race_gate_skips_nonqualified_turn` — 非比赛窗口回合不触发守门
- `test_free_race_gate_oguri_two_intervals` — 小栗帽 100603 专项：两段区间从 DB 正确读出、限 G1 使第二段可比赛回合 12→7、两段守门均按缺口提前触发并返回「比赛」
- `test_free_race_gate_without_race_candidate` — 候选表不含「比赛」时返回 None 而非越界 panic
- `test_race_turn_qualified` — race_turn 回合达标判定
- `test_score_race_skips_nonqualified_turn` — 比赛打分跳过非窗口回合
- `test_score_race_panel_properties` — 比赛打分面板性质（面板不变量）

**地区打分 / 数值口径**
- `test_region_selection_per_build` — 按 build 输出三年地区选择（region_matrix 人工审查配套）
- `test_region_build_sensitivity` — 地区打分对 build 卡组构成有区分度
- `test_breakdown_sums_to_score` — 打分 breakdown 各项之和等于 score（决策日志自洽）
- `test_status_rate_is_linear` — `status_rate` 线性生效（防止重复相乘成平方）

### `ramen/action.rs` (21)

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

**分身 / 超级拉面 / 地区应用**
- `test_clone_placement_full_train_and_npc_eviction` — 分身 placement：全训练位 + NPC 人头被逐出
- `test_train_gauge_uses_actual_npc_count` — 训练诀窍槽按实际在场 NPC 数计
- `test_super_ramen_select_list_and_apply` — 超级拉面选择候选生成与应用
- `test_super_ramen_clones_include_friend_card` — 超级拉面分身候选含友人卡
- `test_super_ramen_clones_friend_priority_beats_greedy_starvation` — 分身友人优先不被贪心饿死
- `test_super_ramen_clones_decoupled_from_parent_stream` — 超级拉面分身分配与父策略流解耦
- `test_region_select_archives_explicit_year_idx` — RegionSelect 应用时显式归档年份下标

### `ramen/effects.rs` (15)

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

### `ramen/features.rs` (9)

局面特征编码器（NN 管线 / 搜索观测）：
- `test_encode_newgame` — 新局编码合法性
- `test_encode_deterministic` — 同局面两次编码逐位一致
- `test_encode_sampled_positions` — 采样根局面编码合法
- `test_card_person_cross_lookup` — 卡组槽位与人头互查（card_id 反查槽位）
- `test_split_person_multi_hot` — 分身人头 multi-hot 编码
- `test_stage_num_reserve_slots` — 阶段编号预留槽位编码
- `test_status_bonus_reaches_features` — 状态加成反映到特征向量
- `test_year1_region_root_features` — 第1年地区选择根局面特征
- `test_dim_constants_consistent` — 维度常量与实际编码长度一致

### `ramen/events.rs` (5)

- `test_event_ids` — 事件 ID 表正确
- `test_turn_special_feeling` — turn→特殊隐藏风味数量映射
- `test_friend_event_state_lifecycle` — 友人事件状态机（首次/点击/解锁/出行）
- `test_friend_visibility` — 友人可见性
- `test_assign_train_feeling_type` — 训练角标分配每种至少1次

### `ramen/rng_consistency.rs` (4)

RNG 三流受控重构的集成验证：
- `test_layer2_cross_strategy_consistency` — 层 2：跨策略 20 回合角标/分布/固定流消费量逐位一致
- `test_layer3_turn_reset_isolation` — 层 3a 回合重置隔离：前 14 回合策略流消耗不影响第 15 回合固定流
- `test_layer3_clone_isolation` — 层 3b 克隆隔离：克隆局面流消费不影响原局面（rollout 隔离原子）
- `test_layer3_stream_isolation` — 层 3c 流间不污染：同回合策略流消耗后回合固定流下一值不变

### `ramen/state.rs` (1)

- `test_region_archive_year_idx_not_current_year` — 地区归档下标陷阱：turn 23 归档到第 1 年（current_year()-1）

---

## 基础游戏

**`game/base/mod.rs`**
- `test_newgame` — 新建基础游戏
- `test_explain` — BaseGame explain
- `test_newgame_status_limit_is_scenario_base_plus_inherit` — 开局上限＝剧本基值＋继承增量（PR #25 契约）
- `test_can_self_race_bounds` — 自选比赛边界（13-71 允许，URA 回合禁止）
- `test_can_friend_outing_bounds` — 友人出行边界（解锁/回合 <72/次数未用完）
- `test_apply_friend_bonus_status_pt` — 友人词条 bonus 五维/PT 加成
- `test_apply_friend_bonus_vital` — 友人词条 bonus 体力加成（×1.6）
- `test_apply_friend_bonus_vital_negative_not_affected` — 负体力不受 bonus 放大
- `test_apply_friend_bonus_other_fields_unchanged` — bonus 只改目标字段其余不动
- `test_apply_friend_bonus_no_bonus` — 无 bonus 词条时不加成
- `test_apply_event_friend_bonus_integration` — 事件 apply 集成友人词条 bonus
- `test_apply_event_no_friend_bonus_backward_compatible` — 无友人时事件 apply 向后兼容

**`game/base/basic.rs`**
- `test_newgame` — 新建基础游戏（BasicGame）
- `test_view_default` — BasicGame 默认 GameView

**`game/uma.rs`**
- `test_uma` — Uma 基础结构
- `test_win_races` — 比赛胜场计算
- `test_score_parts_matches_calc_score` — 评分分项之和等于 calc_score

**其他文件**
- `test_inherit`（game/inherit.rs）— 继承值生成
- `test_support`（game/support_card.rs）— 支援卡基础结构
- `test_friend`（game/mod.rs）— FriendState 基础结构

---

## 搜索层（search）

**`flat_search.rs` (20)**
- `test_search_reproducible_same_seed` — 同种子两次搜索逐位一致
- `test_search_seed_actually_used` — 换种子结果确实变化
- `test_search_invariant_to_action_order` — 搜索结果对动作顺序的不变量
- `test_search_ucb_reproducible` — UCB 路径可复现
- `test_search_ucb_order_sensitivity` — UCB 对候选顺序敏感性的界定
- `test_ucb_first_group_clamps_to_search_n` — UCB 首组步长收进 search_n
- `test_onsen_crn_reseed_changes_result` — 温泉阶段重播种确实改变结果（真 CRN 生效）
- `test_simulate_common_matches_dual_seed_wrapper` — simulate_common 与双种子包装器一致
- `test_ramen_root_search_reproducible` — 拉面根搜索可复现（快照逐位）
- `test_ramen_root_search_seed_used` — 拉面根搜索种子生效
- `test_ramen_crn_seed_topology` — 拉面 CRN 种子拓扑（根/阶段/序号派生关系）
- `test_ramen_simulate_common_ignores_reseed` — simulate_common 不做阶段重播种（拉面 CRN 由规则层接管）
- `test_ramen_three_stage_action_unchanged` — 三阶段动作选择快照逐位不变
- `test_ramen_combined_action_full_game_smoke` — 拉面合并动作整局冒烟
- `test_ramen_combined_action_preserves_targets` — 合并动作保留 targets
- `test_ramen_combined_action_rejects_illegal_targets` — 合并动作拒绝非法 targets
- `test_crn_pair_alignment_keeps_original_j` — CRN 配对按原始序号 j 对齐（不因失败过滤错位）
- `test_crn_pairing_gain`（ignored）— CRN 配对收益测量（耗时，按需手动）
- `test_crn_pairing_gain_ramen`（ignored）— CRN 配对收益测量·拉面（耗时，按需手动）
- `test_crn_pairing_gain_ramen_small` — CRN 配对收益测量（拉面小样本，常规运行）

**`seeds.rs` (6)**
- `test_from_rng_follows_entry_seed` — RolloutSeeds::from_rng 跟随入口种子
- `test_seed_at_deterministic` — seed_at 确定性
- `test_seed_at_distinct_per_rollout` — 每个 rollout 序号种子互异
- `test_stage_seed_distinct_per_rollout` — 阶段种子按 rollout 互异
- `test_stage_seed_distinct_per_turn_and_stage` — 阶段种子随回合与阶段互异
- `test_distinct_root_distinct_sequence` — 不同根种子前 256 序号无碰撞

**`ramen_terminal.rs` (5)**
- `test_dim_keys_frozen` — 25 维键名与顺序冻结（FROZEN_DIM_KEYS 守门）
- `test_ramen_terminal_from_game` — 从真实局归约拉面终局 25 维
- `test_threshold_must_be_reduced_per_rollout` — 阈值类维度在 rollout 内归约（防均值丢信息）
- `test_gap_spread_separates_balance` — gap/spread 维度能区分均衡 build
- `test_visit_covers_all_dims` — 终局访问覆盖全部维度

**`terminal.rs` (2)**
- `test_moment_result` — MomentResult 统计（count/sum/mean/stdev）
- `test_no_terminal_is_zst` — 无终局记录时零尺寸

**`config.rs` (1)**
- `test_new_game_config_follows_crn_stage_reseed` — new_game_config 的 crn_stage_reseed 跟随 GameConfig

---

## 训练员（trainer)

**`ramen_mcts_trainer.rs` (15)**
- `test_mcts_reproducible` — MCTS 整局可复现（同 seed 逐位一致）
- `test_mcts_train_only_full_game` — train_only 阶段门控整局快照（逐位 62698/[3337,…]/searched=66）
- `test_combined_default_on` — 合并搜索缺省开启
- `test_combined_gate_off_full_game` — 合并门控关整局快照（gate-off 非 pure REC，special 仍搜）
- `test_combined_on_skips_special_search` — 合并开时 SpecialSelect 走缓存（29 调用 1 重搜逐位快照）
- `test_combined_cache_used_when_special_gate_off` — special 门控关时合并缓存命中不重搜
- `test_stages_none_matches_recommended` — stages=none 与纯推荐策略逐位一致（searched=0）
- `test_search_stages_parse` — ramen_search_stages 解析（train,ramen / none 等）
- `test_region_gate_three_years` — 地区阶段门控三年开关组合
- `test_year1_region_is_searched` — 第 1 年地区纳入搜索生效
- `test_year1_region_search_root_smoke` — 第 1 年地区搜索根冒烟
- `test_super_ramen_gate_searches_once` — 超级拉面门控只搜一次
- `test_super_ramen_search_root_smoke` — 超级拉面搜索根冒烟
- `test_root_action_uses_strategy_stream` — 根动作决策走策略流（与规则流解耦）
- `test_terminal_breakdown_demo`（ignored）— 终局差异日志整局演示（按需手动）

**`local_ramen_trainer.rs` (13)**
- `recommended_ramen_new_mechanisms_enabled` — 推荐策略新机制全启用（吃面联动/门限 40/友人节奏等）
- `recommended_ramen_uses_025_friend_pacing` — 推荐策略友人 0.25 节奏参数
- `recommended_region_select_year1_runs_policy` — 第 1 年地区选择走推荐策略打分
- `recommended_for_rollout_decisions_identical` — for_rollout 档与正式档决策逐位一致
- `local_single_candidate_breakdown_and_for_rollout` — 单候选 breakdown 采集与 for_rollout 模式
- `eat_covered_train_gate_blocks_mismatched_ramen` — 吃面-训练覆盖门控否决不覆盖 at_trains 的面
- `eat_guarantee_value_on_risky_train` — 吃面必成价值在风险训练位生效
- `train_coupling_bonus_on_eating` — 吃面-训练联动加成生效
- `ramen_weak_train_boost_effect` — 弱位训练偏好放大生效
- `cap_discount_ratio_behavior` — 残余收益折扣行为（快满位副属性打折）
- `friend_hidden_starve_and_overflow_guard` — 友人隐藏风味饥饿与溢出双向守卫
- `friend_future_hidden_supply` — 友人未来隐藏风味供给预估
- `microbench_top_fns`（ignored）— 热点函数微基准（进程级 CWD，按需手动）

**`ramen_handwritten_trainer.rs` (2)**
- `test_handwritten_full_game` — 完整 77 回合跑通（评分/RMJ/吃面数输出）
- `test_handwritten_reproducible` — 同 seed 两次整局评分一致

**`logging_trainer.rs` (2)**
- `test_logging_trainer_records_full_game` — 完整局决策记录覆盖（三阶段/事件/地区选择）
- `test_reproducible_same_seed` — 同 seed 两次整局决策序列与评分一致（可复现性）

---

## 输出层（output)

**`reason.rs` (7)** — 险胜决策理由
- `test_narrow_win_analysis` — 险胜分析：门限内选出 rivals 与分差
- `test_landslide_silent` — 悬殊局（超门限）静默不输出理由
- `test_max_display_truncate` — 未中选候选按 reason_max_display 截断
- `test_pt_metric` — pt_score 维度除回系数还原（-66/2=-33）
- `test_render_lines` — render_reason_lines 数据驱动渲染格式
- `test_gap_confidence_degenerate` — 零误差/零分差退化输入置信度不产生 NaN
- `test_noop_sink` — NoopSink 默认静默

**`decision.rs` (6)**
- `test_from_index_minimal` — DecisionInfo 最小构造
- `test_from_index_and_score` — DecisionInfo 带评分构造
- `test_default_is_zero_index` — 默认构造下标为 0
- `test_serde_roundtrip_minimal` — serde 往返（最小）
- `test_serde_roundtrip_full` — serde 往返（全字段）
- `test_serde_json_value_conversion` — serde_json::Value 转换

**`diagnostic.rs` (5)** — 诊断日志运行时开关
- `test_diag_expands_to_info` — feature 开时 diag! 展开为真实 info
- `test_diag_silent_when_disabled` — 开关关闭时 diag! 静默
- `test_guard_suppress_and_restore` — DiagGuard 抑制/恢复基本语义
- `test_guard_restores_prev_value` — guard 恢复进入前的值（非硬编码 true）
- `test_nested_guards` — 嵌套 guard 栈式恢复

**`view.rs` (4)**
- `test_default_view` — 默认 GameView
- `test_with_scenario_only_sets_scenario` — with_scenario 构造器只设置 scenario
- `test_view_as_serde_json_value` — GameView 转 serde_json::Value
- `test_serde_roundtrip` — GameView serde 往返

**`turn_flow.rs` (4)** — 回合流程渲染
- `test_turn_output_baseline` — 回合输出固定种子基线
- `test_distribution_colors` — 训练分布着色
- `test_vital_color` — 体力着色梯度
- `test_verbose_demo` — verbose 演示输出

**`decision_log.rs` (4)**
- `test_csv_escape` — CSV 字段转义（逗号/引号/换行）
- `test_csv_row_and_header` — 单行序列化 + 表头格式
- `test_empty_log` — 空日志 CSV 输出
- `test_save_to_roundtrip` — 落盘与读取往返

---

## 配置 / 数据加载

**`gamedata/config.rs` (12)**
- `test_scenario_status_limit_base_contract` — 三剧本上限基值字面量契约（守漂移 + 锁互异）
- `test_status_final_score_saturates_out_of_range` — 评分查表越界饱和到表末
- `test_production_default_searches_ramen_stage` — 生产缺省 ramen_search_stages 含 ramen
- `test_top_level_region_override_takes_effect` — 顶层 ramen_region_* 覆盖生效
- `test_mcts_override_daily_path_keeps_production` — 日常路径（无 override）保持生产搜索参数
- `test_mcts_override_denies_unknown_fields` — OverrideMctsConfig 拒绝未知字段
- `test_mcts_override_omitted_section_keeps_all_twelve` — 省略 [mcts] 段时全部字段保默认
- `test_mcts_override_partial_fields_apply` — 部分字段覆盖仅改写指定项
- `test_override_config_denies_unknown_fields` — OverrideConfig 拒绝未知字段
- `test_override_config_parses_without_mcts_or_bogus` — 无 mcts 段/杂散字段时解析行为
- `test_override_merge_all_none_keeps_default` — 全 None override 不改默认
- `test_override_merge_partial_overrides` — 部分 override 仅覆盖指定字段

**`utils.rs` (8)**
- `test_validate_game_config_scenario_enum` — scenario 枚举校验
- `test_validate_game_config_trainer_enum` — trainer 枚举校验
- `test_validate_game_config_ramen_region_fixed_length` — Fixed 策略下 ramen_region_fixed 长度校验
- `test_resolve_default_config_path` — 默认配置路径解析
- `test_resolve_user_config_path_points_to_workspace_root` — 用户配置路径指向 workspace 根
- `test_default_config_ramen_region_fixed` — default_config 的 Fixed 预设值加载
- `test_missing_user_config_keeps_production_mcts` — 缺用户配置时生产 mcts 参数不丢
- `test_override_config_trainer_overrides_default` — OverrideConfig.trainer 覆盖 default 的 trainer

**`gamedata/event.rs` (2)**
- `test_load_and_explain_all_events` — 加载并 explain 全事件
- `test_load_and_explain_ramen_events` — 加载并 explain 拉面事件

**`gamedata/mod.rs` (4)**
- `test_uma_data` — 马娘数据加载
- `test_support_data` — 支援卡数据加载
- `test_consts` — 常量加载
- `test_turn_mask` — 回合掩码

---

## 采样器（sampler)

教师数据根局面采样器：
- `test_spec_deterministic` — 采样规格确定性
- `test_spec_covers_turn_range_and_decks` — 规格覆盖回合区间与卡组
- `test_sample_reproducible` — 采样可复现
- `test_sample_seed_actually_used` — 采样种子生效（换种子采样变）
- `test_sample_covers_all_turns` — 采样覆盖全部回合
- `test_combinations_boundaries` — 分层组合边界（首末组合）
- `test_gen1_space_size` — gen1 采样空间大小
- `test_gen1_decks_wellformed` — gen1 卡组合法性
- `test_gen1_space_excludes_chara_conflict` — gen1 空间排除 chara 冲突
- `test_epsilon_perturbs_trajectory` — epsilon 扰动改变轨迹
- `test_epsilon_out_of_range_rejected` — 非法 epsilon 拒绝
- `test_sampled_position_is_advanceable` — 采样局面可推进（阶段入口契约）
- `test_sampled_position_feeds_search` — 采样局面可喂给搜索

---

## 基准设施（bench)

- `test_load_player_builds_from_config` — 从 bench_config.toml 加载 player_builds
- `test_validate_player_builds_rejects_bad` — 非法 build 配置拒绝
- `test_player_builds_preserve_order` — build 声明序保持
- `test_player_builds_make_deck_live_data` — build 生成真实卡组（live cardDB）
- `test_select_representatives_live_data` — 代表卡选择（真实 cardDB）
- `test_seeded_rngs_reproducible` — seeded_rngs 双 RNG（决策/规则主）可复现
- `test_summarize` — summarize 统计聚合
- `test_percentile` — 分位数统计
- `test_yearly_observability_full_game_and_csv` — 逐年观测整局 + CSV 产出

---

## RNG（rng)

顶层 `rng.rs` 三流体系单元测试：
- `test_deterministic` — 同种子确定性
- `test_masters_differ` — 不同 master 流互异
- `test_additive_no_xor_collision` — 加法派生流无 XOR 碰撞
- `test_stream_tags_isolated` — StreamTag 类型隔离
- `test_typed_streams_work` — 三类流（TurnFixed/Event/Strategy）各自可用
- `test_fork_local_stream` — 局部流 fork 正确
- `test_clone_independent` — 克隆流独立
- `test_counter_continues` — 克隆后计数器接续

---

## 神经评估（neural)

- `test_random_evaluator_send_sync` — 随机评估器线程安全

---

## 基准测试运行说明

### `bin/bench_base.rs`

固定种子批量跑批，产出基线分布（分数/PT/RMJ/耗时）与决策轨迹。

```bash
cargo run --release --bin bench_base
cargo run --release --bin bench_base -- --runs 100 --seed 7 --log --out logs
```

- 参数：`--runs N` 局数、`--seed S` 基础种子（第 i 局 = seed+i）、`--log` 落盘决策日志（默认关）、`--out DIR` 输出目录
- 产出（默认 `logs/`）：`bench_base_results.csv` + 汇总统计；`--log` 时另产出决策轨迹 CSV
- 可复现性：同一参数下游戏结果完全一致（`elapsed_ms` 属运行耗时，允许波动）

### `bin/bench_compositions.rs`

固定种子遍历五种普通支援卡各 0..=3 张、合计 5 张再加固定友人的全部 101 种构成。

```bash
cargo run --release --bin bench_compositions -- --runs 100 --seed 42 --trainer handwritten --out logs/bench_compositions.csv
```

- 代表卡选择：各类型取最新 5 张满破 SSR 作候选池，`--min-panel` / `--pool-size` / `--pick` 可调
- 构成总数 101 的验证由 `bench` 模块测试覆盖

## 附注

- **ignored 测试（4 个）**：`test_crn_pairing_gain` / `test_crn_pairing_gain_ramen`（CRN 收益测量，耗时）、`microbench_top_fns`（微基准）、`test_terminal_breakdown_demo`（整局诊断演示）——均按需手动运行
