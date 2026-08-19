# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-19

### 拉面杯三阶段决策重构：吃面效果立即落地

落实"Trainer 选完（是否吃面）+（是否使用隐藏诀窍）后，立即消耗诀窍 / 拉面效果生效 / 生成分身"的需求，让玩家在选训练动作前能看到完整 buff 和 distribution。

**核心改动**：
- **新增 `RamenGame::ground_ramen_effects(rng)` 公共方法**：整合所有"吃面后立即生效"的效果——消耗诀窍、PT 增量、`current_ramen`、地区拉面分身、羁绊效果、显示 `explain_ramen_info()` + `explain_distribution()`
- **阶段过渡时自动触发**：
  - 三阶段路径：`SpecialSelect → Train` 过渡时（`Game::next()`）
  - 合并决策路径：`RamenSelect → Train`（`combined_decision=true`）时
- **Train 阶段简化**：移除原 `RamenAction::apply_ramen` / `apply_ramen_friendship` / `distribute_clones`，统一搬到 `RamenGame` 上（`apply_ramen_friendship`、`distribute_region_clones`）；Train 阶段 `apply_action` 只执行 `operation`
- **`list_train_actions` 简化签名**：不再带 `pending_ramen` / `pending_targets`，只生成 operation-only `RamenAction::new(op)`
- **`RamenAction::new(op)` 新构造函数**（等价于 `no_ramen`），Train 动作 `ramen`/`special_targets` 字段统一为 None

**外部接口支持**：通信模块可直接调用 `RamenGame::ground_ramen_effects(rng)` 落地"已吃面但未训练"的中间状态，无需走 RamenSelect/SpecialSelect 流程。

**防御性修改**：`explain_distribution` / `default_calc_training_buff` / `default_calc_training_value` / `shining_count` 在 `distribution` 未初始化时安全返回，避免 `ground_ramen_effects` 在 distribute 之前触发时 panic。

**测试更新**：`test_three_stage_decision_flow` 验证 Train 动作 ramen/special_targets 已为 None（已 ground）；`test_list_train_actions_no_ramen_field` 替代旧 `test_list_train_actions_carries_pending`。

### 拉面杯 hint_special 全员触发 + hint_count_bonus 保留

落实 issues.md「basic_effect.hint_special 尚未处理」：

- **`distribute_hint`**：在 hint_special 生效时，强制将 `at_trains` 训练位置的所有 PersonType::Card 支援卡的 `is_hint` 设为 true
- **新增辅助函数**：`calc_hint_special_active`（吃面 + 第3年 + 支援卡种类≥4）、`calc_hint_special_at_trains`（地区拉面的 at_trains）、`is_hint_special_active_for_train`
- **`handle_hint_event`**：hint_special 路径下依次触发 hint_persons 中所有 PersonType::Card 的 hint 事件，每个支援卡按各自 `1 + hint_count_bonus` 次触发（保留温泉杯逻辑）
- **抽取 `push_hint_event` 辅助函数**：复用 hint事件生成逻辑（含 `hint_level / total_hints` 上限处理）
- 新增 5 个单元测试覆盖各种生效场景

### ManualTrainer 玩家测试 + 拉面杯完整流程测试

- **`ManualTrainer` 改造**：新增 `mock_inputs` 队列（`Rc<RefCell<VecDeque<String>>>`）和 `FallbackMode` 枚举（`Interactive` / `PickFirst`）
  - `ManualTrainer::new()`：真实玩家模式（inquire 终端交互）
  - `ManualTrainer::with_mock_inputs(inputs)`：测试模式，mock 队列优先消费，耗尽后 fallback 到 PickFirst（选第一个候选）
- **`utils.rs`**：新增 `init_logger_stdout` 函数，仅输出到 stdout 不写文件（玩家测试场景，与 inquire TUI 不冲突）
- **新增独立 bin `ramen_manual`**：`cargo run --release --bin ramen_manual` 启动一局拉面杯，玩家通过 inquire 真实选择动作
- 新增 2 个测试：`test_manual_trainer_full_game`（完整 77 回合流程跑通）+ `test_manual_trainer_hint_special_path`（第3年路径验证）

### 并发测试 init_logger 竞争问题修复

落实 issues.md「测试批量运行时的全局状态问题」：

- **问题根因**：`init_logger` 存在 TOCTOU 竞争——多个测试并行调用时，`LOGGER.get().is_some()` 检查通过后其他线程抢先 `set()`，后续线程调用 `flexi_logger.start()` 触发 "logger already initialized" 报错
- **`utils.rs` 修复**：
  - 新增 `LOGGER_INIT_DONE: AtomicBool` 记录 log crate 是否已成功 start
  - 新增 `INIT_LOCK: OnceLock<Mutex<()>>` 串行化整个初始化过程
  - 快速路径：已初始化直接 return；慢速路径：持锁 + 双重检查
  - `disable_log` / `enable_log` 增加 `if let Some(LOGGER.get())` 保护，未初始化时安全 return
- 所有 110 个测试在 release 模式下 3 次连续运行均稳定通过

### 文案修正：StageOnly 显示 "阶段阶段" → "下一步"，但注释为“中间步骤”

- `Operation::StageOnly` 的 Display 输出 `<阶段阶段>` 改为 `<下一步>`（修复玩家测试时inquire菜单显示空字符串的误解）
- 同步更新 action.rs / mod.rs / trainer/mod.rs 中 10 处 "阶段阶段" 注释为 "中间步骤"

## 2026-08-18

### scenario_pt 每年初归零

剧本PT（`scenario_pt`）在每年 RMJ 结算后归零，下一年重新累计。即：
- turn=23 RMJ 结算后 PT 归零，第2年（turn=24-47）从 0 开始累计
- turn=47 RMJ 结算后 PT 归零，第3年（turn=48-71）从 0 开始累计
- turn=71 RMJ 结算后 PT 归零（URA 阶段不再累计 PT）

逻辑位置：`Game::next()` 的 NextTurn 阶段，在 RMJ 结算 + 写入 `rmj_results` + apply RMJ 事件 之后立即归零。这样 `ramen_success_effect / ramen_fail_effect` 的常驻效果已可读取，新一年的 `ramen_pt_effect` 档位和 `region_bonus` 也会基于新的 PT 重新计算。

新增单元测试 `test_scenario_pt_reset_after_rmj` 验证 turn=23 末 RMJ 结算后 PT=2500 → 0。

### RMJ 事件触发时机修正 + 超级拉面基础效果自动应用

- **RMJ 事件触发时机修复**：RMJ 事件（401404/401405/401406）原在 turn=23/47/71 后**多延迟一整个回合**才触发（push 到 `unresolved_events` 后等 AfterTrain 阶段消费）。修复为在 NextTurn 阶段 RMJ 结算后**立即 apply**（不需要 Trainer，RMJ 事件无 player_select）。现在 turn=23/47/71 末（即 1-indexed 第 24/48/72 回合末）正确触发
- **超级拉面 base 效果应用**：
  - `vital+20`、`motivation+1`：每个 URA 回合（turn=72-77）Begin 阶段都生效
  - `saihou+100`（赛后加成）：**仅在 turn=72 一次性 +100**，之后回合保留已生效值，不重复累加（避免 race_bonus 无限增长到 160/260/360/...）。实测 race_bonus 从 60（支援卡）一次性提升到 160，后续 5 个 URA 比赛都是 160

新增 3 个单元测试：`test_rmj_event_immediate_apply_at_turn_23`（验证 turn=23 末立即触发 401404）、`test_super_ramen_base_effect_vital_motivation`（验证 turn=72 一次性+100）、`test_super_ramen_saihou_one_time_only`（验证 4 个连续 URA 回合 race_bonus 仅 +100）。

### 拉面杯 RMJ 结算与固定触发事件补全

落实 issues 中"RMJ 结算后触发对应事件"和"补充固定触发事件"两批修复：

- **RMJ 结算事件触发**：每年 RMJ 结算后（回合 23/47/71 结束阶段），将对应事件 push 到 `unresolved_events`：401404（第1年）/401405（第2年）/401406（第3年）；`apply_event` 增加分支选择逻辑，根据 `rmj_results[year_idx]` 选 result=2（成功）或 result=1（失败）的分支并直接 `add_value`
- **固定触发事件补全**：
  - 回合 0 开始：触发 400000400 马娘登场
  - 回合 24 开始：触发 4009 经典年新年
  - 回合 48 开始：触发 4010 古马年新年
  - 回合 48 结束：触发 4011 新年抽签（`system_events["ticket"]`，按 prob 加权选 result）
  - 回合 77 结束：触发 401407 + 5011（ending）+ 友人结束事件
- **`generate_events` 扩展**：除 ramen_data.scenario_events 外，再处理 `global_events().story_events` 的 Fixed 事件（400000400/4009/4010）
- **`run_event` 默认实现**：决策条件由 `event.choices.len() > 1` 改为 `event.player_select && event.choices.len() > 1`，保证 `player_select=true` 的事件无论选项数都交给 Trainer 决策
- **race_turn 短路修复**：race_turn 时 `run_ramen_select` 显式调用 `run_after_train` 处理 `unresolved_events` 后再 `stage = NextTurn`，避免 turn=77 的 ending/401407/友人结束事件因 stage 跳过 AfterTrain 而漏触发

新增 9 个单元测试覆盖：`select_rmj_choice_by_result`/`rmj_event_year`/`rmj_event_apply_success`/`rmj_event_apply_fail`/`rmj_event_push_to_unresolved`/`generate_events_uma_debut`/`generate_events_classic_newyear`/`generate_events_ancient_newyear`/`add_mandatory_events_ticket_at_48`/`add_mandatory_events_ending_at_77`。

### 拉面杯训练分布剧本得意率加成修复

`RamenGame::deyilv` 此前只返回卡的 deyilv，未加剧本加成。修正为：`effects.rs` 新增 `calc_scenario_deyilv` 汇总 `pt_effect + rmj_effect` 的 deyilv；`RamenGame::deyilv` 返回"卡 deyilv + 剧本 deyilv"。新增 6 个测试覆盖普通回合/超级拉面/RMJ 成功失败等场景。`Game::distribute_person` 中"不出现"判定仍受得意率影响的小问题留待后续，详见 issues.md。

### 夏令营期间拉面杯规则实现

落实夏合宿期间（回合36-39和60-63）的三条特殊规则：①三种诀窍全 MAX（带新友人）；②禁用普通外出/友人出行/治病；③休息自动清除 `ill`/`bad_trainer` flag。新增 `fill_gauge_after_non_train` 统一处理比赛/休息/外出/友人出行的诀窍槽基础值填充；`list_operations` / `list_train_actions` 增加 `is_xiahesu` 参数。`Operation::Clinic` 按设计不获得诀窍槽。详见 issues.md「夏合宿期间诀窍槽加成未实现」。

### 拉面杯合并决策接口（两阶段聚合）

为未来在线搜索/MctsTrainer 提供"选面+吃法"一次性决策的合并路径：方案 E 在 `RamenGame`（不动 Game trait）新增 `list_combined_ramen_select_actions` / `apply_combined_ramen_decision`；`RamenState` 加 `combined_decision` 标记位让 `Game::next()` 在 RamenSelect 阶段跳过 SpecialSelect 直接推 Train。粒度选择权交给 Trainer（HandwrittenTrainer 走三阶段，MctsTrainer 走合并）。详见 issues.md「拉面杯合并决策接口（两阶段聚合）」定案。

### 拉面杯三阶段决策重构（隐藏风味显式化）

将 `apply_ramen` 中硬编码的 `special_targets = [0,0,0]` 改为 Trainer 显式决策：`RamenStage` 拆为 `RamenSelect` → `SpecialSelect` → `Train` 三阶段；新增 `list_special_targets_for` 生成隐藏风味用法的候选集合；`RamenAction` 新增 `special_targets` 字段和 `StageOnly` 占位 operation。详见 issues.md「拉面杯动作阶段扩展：隐藏风味决策」定案记录。

- 新增 7 个单元测试覆盖特殊风味决策、回归测试全过
- 同步修复 `init_*` 系列函数 `set().expect()` → 幂等早退（解决 OnceLock 重复 set 级联失败的旧 issue）

## 2026-08-17

### umaai 跨平台构建支持与相关更新

**umaai Linux 构建修复（umaai 现可在 Ubuntu 下编译运行）：**
- `winscribe`（Windows 资源编译）改为仅在 `cfg(windows)` 目标下作为 build-dependency
- `windows` crate 改为 `cfg(windows)` 限定依赖（其依赖链 windows-future 0.3.2 与 windows-core 0.62.2 版本不兼容，在 Linux 上编译失败）
- build.rs 的 Windows 资源编译逻辑（.ico 图标 + `/STACK` 链接参数）用 `#[cfg(windows)]` 包裹
- 补声明 Linux 专用的 `libc` 依赖；`utils.rs` 中 Linux 版 `get_stack_size` 补 `pub`

**其他：**
- issues.md 新增"Ubuntu 下 umaai 二进制图标方案待定"记录（含两个候选方向的说明）
- umasim：ramen/game.rs 训练值计算拆分为属性与 PT 两次调用（PT 使用 status_pt[5]，补充计算公式注释）
- AGENTS.md 项目规则修订（语言要求措辞、文档载入策略、安全注意事项等）
- project_context.md 新增"测试"章节（ramen 完整流程测试 test_ramen_silent_loop 的说明）
- 备注：工作树其余文件的未提交改动均为行尾（LF/CRLF）层面的变化，无内容修改，按要求未处理，留待提交时自然消解或另行处理

### 拉面杯模块机制修正、显示改进与架构重构

**机制修正：**
- 友人事件(友人登场/点击/解锁/出行)改用拉面杯专用事件ID(830305101-830305115)，替代温泉杯事件
- 友人出行时每次获得2个隐藏风味
- 友人事件羁绊正确加到友人卡（修正person_index指向错误问题）
- RMJ结算成功时训练等级剧本加成+1
- 诀窍角标分配保证每种(A/B/C)至少出现1次
- 诀窍值初始化(回合2/24/48) 时隐藏风味不重复计算
- 地区选择时机修正：第1年在回合2 Begin阶段，第2/3年在回合23/47 NextTurn阶段（RMJ结算后）
- 休息心得（refresh_mind）：友人解锁事件后触发，每回合体力+5，概率结束

**显示改进：**
- 回合信息追加隐藏诀窍数量显示
- 训练明细显示诀窍槽每种类型明细(A+X B+Y C+Z)
- NPC显示为[NPC]，不显示角色名和羁绊
- 吃面后显示完整回合信息（含分布表和训练数值）
- 剧本机制未开启/URA回合时不显示诀窍角标和拉面效果
- 获得隐藏风味后重新打印回合信息
- 设施等级显示为加上剧本加成后的实际等级
- 拉面效果显示：普通回合显示当前拉面效果，超级拉面回合显示所有生效加成

**架构重构：**
- 地区选择改为通过Trainer统一接口(select_action)，新增Operation::RegionSelect变体
- 移除Trainer trait的select_regions方法
- 分身系统重构：分身使用本体ID，不再使用负数ID
  - 分身和本体共享羁绊
  - 分身只能在本体得意训练位置闪耀
  - 分身在非本体训练位置时不闪耀，友情加成不生效
  - 分身增加人头数、贡献buff、触发hint和友人点击
  - 地区拉面分身：每个at_trains位置随机选一个不重复的支援卡
  - 超级拉面分身：随机选择训练位置，失败则重试
- 超级拉面回合效果：ramen_pt_effect、ramen_basic_effect按最高档生效，RMJ结算效果生效，finals_effect生效
- RMJ结算效果生效时间：第1年结算→第2年生效，第2年结算→第3年生效，第3年结算→URA期间生效
- 猴子训练员：修复地区选择时总是选第一个的问题，现在在找不到匹配的BaseAction时随机选择
- Hint率计算应用剧本加成(ramen_pt_effect.hint + rmj_effect.hint)
- 拉面羁绊效果对卡组所有6张卡生效
- 年3地区配方复用年1配方（取模映射）
- ramen_story_flow文档更新

**已知问题：**
1. 训练数值不对，尤其是友情加成
2. 超级拉面得意率人物分配错误

### 拉面杯训练数值端到端观测测试

- `crates/umasim/src/game/ramen/game.rs` `tests` 模块新增 `test_random_distribution_training_value`：固定回合 30（第二年，Lv=4）、支援卡羁绊 100，分别打印不吃面和吃面场景下的训练分布与数值，便于排查"训练数值不对，尤其是友情加成"的问题

## 2026-08-16

### 拉面杯模块 1d 最小闭环实现

**核心实现：**
- 新增 `game.rs`：实现 Game trait for RamenGame
- 阶段流转设计：`RamenStage::next()` 负责回合内，`Game::next()` 负责跨阶段
- 实现 `run_stage` 各阶段逻辑：Begin/Distribute/Train/AfterTrain/RegionSelect/SuperRamenSelect
- 实现 `list_actions`：生成所有吃面/不吃面 × 操作的组合动作
- 实现 `generate_events`：复用 BasicGame 随机事件
- 实现 `apply_event`：处理继承、友人解锁等特殊事件

**动作执行（action.rs 重构）：**
- 吃面与训练严格分阶段：`apply_ramen()` → `do_train()`
- 吃面处理包括分身分配（distribute_clones）
- 拆分 `do_train` 为多个辅助函数：`calc_train_params`、`handle_train_success`、`handle_train_failure` 等
- 修复 `global!` 返回引用不需要再加 `&` 的问题

**人头管理：**
- `init_persons`：开局仅加入非友人卡 + 理事长
- 动态添加：第2回合（turn==2）添加友人卡和NPC，第12回合（turn==12）添加记者
- `add_friend_and_npcs()`、`add_reporter()` 方法

**其他修复：**
- 修复 `events.rs` 隐藏风味回合表与 `rules.rs` 一致
- 新增 `RamenStage::NextTurn` 阶段处理回合边界逻辑
- 测试卡组更新为最新卡：杏目、青春永驻、名将怒涛、洛林军歌、里见光钻、骏川手纲

**开发计划更新：**
- 标记 1d 步骤为已完成

### 拉面杯模块 1b 核心游戏机制 + 1c 动作预览和手写策略

**1b - 核心游戏机制：**
- 补充 RamenScenarioData 数据结构（FinalsEffect、RmjEffect、PtEffect 等）
- 实现效果计算模块（effects.rs）：calc_ramen_training_effect、apply_ramen_training_value
- 实现做面/吃面规则：支持手动指定隐藏风味替换目标（special_targets）
- 实现 RMJ 结算：新增 RmjResult 枚举（Fail/Success/GreatSuccess）
- 实现地区选择规则：get_region_range、validate_region_selection
- 实现分身规则：get_region_clone_trains、get_super_ramen_clone_train_options
- 实现隐藏风味分配：get_turn_special_feeling
- 实现友人事件状态管理（FriendEventState）
- 实现训练角标分配（assign_train_feeling_type）
- 修正隐藏风味回合表：2,24,36,48,60=>2; 37,38,39,61,62,63=>1

**1c - 动作预览和手写策略：**
- 实现地区选择策略（fixed_region_selection）
- 实现超级拉面选择策略（fixed_super_ramen_selection）
- 实现动作生成函数（list_ramen_choices、list_operations、list_all_actions）
- 动作采用分离决策模型：吃面选择 × 基础操作
- 清理 mod.rs 无用依赖

**开发计划更新：**
- 标记 1b 和 1c 步骤为已完成

### 拉面杯模块 1a 核心类型定义 + 1b-1 诀窍系统

- 建立 `game/ramen/` 模块：mod.rs、state.rs、action.rs、rules.rs、effects.rs、events.rs、policy.rs
- 定义核心类型：RamenGame、RamenState、RamenEffect、RamenAction、RamenStage、FeelingType、TrainingType、Operation
- RamenEffect 字段对应剧本加成词条（xunlian、youqing、pt_bonus、train_limit、pt_limit 等）
- RamenAction 采用组合动作模型（ramen + operation），Display 显示地区名称
- 使用 IntEnum derive 替代手写 index/from_index，去掉 OutingType 独立枚举
- 新增 RegionEffect 结构体和 region_feeling 字段到 RamenScenarioData
- 修正 RamenBasicEffect.jiban → friendship，与 JSON 数据对齐
- 实现诀窍系统规则函数：槽基础值分配、库存溢出管理、训练加成、友情加成
- 诀窍槽基础值分配算法：floor + 消耗≥1固定分配1 + 最小已分配优先补足
- 修正诀窍槽溢出逻辑：清零而非取余，超出部分不保留
- 改进测试：混合类型验证溢出丢弃顺序，去掉 assert 改用 println 输出

### 拉面重构计划调整与文档整理

- 将 `opt/` 目录重命名为 `archive/`，归档旧规划文档
- 将 `master/` 目录重命名为 `master_mdb_data/`，统一数据目录命名
- 新增 `ramen_phase_adjustment_analysis.md` 分析文档（已归档至 `archive/`）
- 重构开发计划 Phase 结构：合并原 Phase 1/4/5/6 为新的 Phase 1（1a-1d），缩减为6个 Phase
- 修正开发计划术语：食材→诀窍、万能食材→隐藏风味、3-78→72-77
- 补充训练诀窍角标分配细节（每回合每个训练随机分配 A/B/C 角标，需新增存储字段）
- 更新执行检查清单，标记前两项为已完成

## 2026-08-15

### 拉面剧本机制完善

- 补充友人解锁机制、诀窍槽算法、分身规则等核心机制文档
- 补充剧本机制初始化规则（第2回合开始时）
- 补充夏合宿规则（训练等级、事件触发）
- 补充超级拉面期间限制（不可吃其他面）
- 更新gamedata数据：调整事件概率、添加地域名称、完善超级拉面效果
- 更新AGENTS.md项目规则：完善提交规范和工作流程
- 添加ramen_story_flow.md拉面剧本流程文档
- 更新术语表：添加诀窍槽、友人解锁、复合宿等新术语
- 整理文档目录：将规划类文档移至opt子目录

## 2026-08-14

### 拉面剧本事件数据补充

- 在scenario_ramen.json中添加scenario_events和friend_events数据
- 更新RamenScenarioData结构体，添加对应的事件字段
- 添加单元测试验证事件数据加载

### EventData触发类型重构

- 新增TriggerType枚举：Random/Code/Fixed三种触发类型
- 移除EventData中的start_turn/end_turn/max_trigger_time字段
- 更新JSON数据文件和触发逻辑代码

## 2026-08-13

### 文档整理

- 创建了AGENTS.md项目规则总结文档
- 在.trae/documents/目录下整理相关文档

### 测试规范完善

- 在umasim::utils中新增get_workspace_root()函数，用于获取workspace根目录
- 修改了多个测试文件，在测试中使用get_workspace_root()切换到workspace根目录

### 拉面剧本数据完善

- 更新ramen_basic_effect：添加jiban/status_limit/hint_special字段，填充3年效果数据
- 添加finals_effect：定义超级拉面(含RMJ成功)的基础/额外/单独效果
- 添加ramen_region_effect：记录20条地域拉面效果数据
- 更新Rust结构体：添加RamenBasicEffect结构体
- 更新ramen_memo_cn.md文档：补充效果说明和字段定义
