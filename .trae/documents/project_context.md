# UmaAI-RS 项目特定上下文

## 项目结构

### 工作空间
- 使用Cargo工作空间管理多个crate
- 工作空间根目录为项目根目录（AGENTS.md所在目录）

### 主要crate
- `crates/analyzer`：分析工具
- `crates/umaai`：主要应用
- `crates/umasim`：游戏模拟器
- `crates/unused`：未使用的代码

## 配置文件

### 游戏数据
- `gamedata/`目录包含游戏配置和数据
- 主要文件：
  - `constants.json`：游戏常量配置
  - `events.json`：事件数据
  - `umaDB.json`：马娘数据
  - `cardDB.json`：支援卡数据
  - `scenario_onsen.json`：温泉剧本配置
  - `scenario_ramen.json`：拉面剧本配置
  - `text_data_dict.json`：文本数据字典
  - `default_config.toml`：默认游戏配置
- `game_config.toml`（工作空间根目录）：用户自定义游戏配置

### 游戏常量
- 回合数默认从0开始，特殊情况下会说明从1开始

### 配置格式
- 使用JSON和TOML格式

### 脚本工具
- `scripts/`目录包含Python脚本工具
- 主要用于数据导出和处理（如 `scripts/export_support_card/`）

## 开发环境

### 操作系统
- 以 Windows 为主
- umaai 已支持 Ubuntu/Linux 构建（`winscribe`/`windows` 依赖以 `cfg(windows)` 限定）

### Shell
- PowerShell

## 测试

### 模拟游戏流程测试
- 位置：`crates/umasim/src/game/ramen/game.rs`
- `test_ramen_silent_loop`：完整拉面剧本 77 回合静默流程（关闭日志，仅输出育成配置和最终结果），端到端流程验证
- `test_manual_trainer_full_game`：ManualTrainer（mock 输入 + PickFirst fallback）完整 77 回合流程
- `test_manual_trainer_hint_special_path`：第3年 hint_special 路径验证
- 运行命令：`cargo test -p umasim <测试名>`（release 模式）

### 玩家手动测试程序
- `crates/umasim/src/bin/ramen_manual.rs`：`cargo run --release --bin ramen_manual`
- 通过 inquire 终端交互逐动作/事件选择，验证游戏机制实际表现
- 读取 `game_config.toml`（强制 `scenario = "ramen"`、`trainer = "manual"`；卡组必须含新友人卡 303051-303054）

## 拉面杯模块结构

### 模块入口（mod.rs）
- `RamenStage`：回合阶段枚举（Begin/Distribute/RamenSelect/SpecialSelect/Train/AfterTrain/NextTurn/RegionSelect/SuperRamenSelect/Settlement）
- `FeelingType`：诀窍类型（A/B/C）
- `TrainingType`：训练类型（Speed/Stamina/Power/Guts/Wisdom）
- `Operation`：基础操作（Train/Race/Rest/NormalOuting/FriendOuting/Clinic/RegionSelect([usize; 3])/StageOnly）

### 核心类型（state.rs）
- `RamenGame`: 拉面杯游戏主状态，通过Deref暴露BaseGame
  - `newgame()`：校验卡组必须含新友人卡（idrank 303051-303054，rank=1-4）
- `RamenState`: 拉面杯专用状态
  - 诀窍系统：`feeling_stock`/`feeling_slot`/`feeling_queue`
  - 隐藏风味：`special_feeling`
  - 地区拉面：`selected_regions`/`current_ramen`
  - 剧本进度：`scenario_pt`/`rmj_results`/`train_level_bonus`/`super_ramen`
  - 剧本计数器：`eat_count`/`train_feeling_type`
  - 三阶段决策 pending：`pending_ramen`/`pending_special_targets`/`combined_decision`（由 `clear_pending()` 一并清空）
- `RamenEffect`: 效果合并（基础+地区+超级拉面+PT常驻）
- 辅助方法：`add_friend_and_npcs()`、`add_reporter()`、`add_person()`、`init_feeling_stocks()`、`add_friendship()`（NPC不增加羁绊）

### Game trait 实现（game.rs）
- 阶段流转：`RamenStage::next()` 负责回合内，`Game::next()` 负责跨阶段（三阶段 RamenSelect→SpecialSelect→Train；合并决策时跳过 SpecialSelect 直接落地吃面效果）
- `run_stage()`：分发到各阶段处理函数（run_begin/run_distribute/run_ramen_select/run_special_select/run_train/run_after_train 等）
- `ground_ramen_effects(rng)`：吃面效果立即落地（消耗诀窍、PT 增量、current_ramen、地区分身、羁绊、显示），阶段过渡时自动触发，也可由通信模块直接调用
- `list_combined_ramen_select_actions()` / `apply_combined_ramen_decision()`：合并决策接口（RamenSelect×SpecialSelect 聚合，为未来 MctsTrainer 预留）
- `deyilv()` override：卡得意率 + 剧本得意率加成（`calc_scenario_deyilv`）
- `distribute_hint()` override：应用剧本Hint率加成；hint_special 生效时强制全部支援卡出 Hint（`calc_hint_special_active`/`calc_hint_special_at_trains`/`is_hint_special_active_for_train`）
- `is_shining_at()` override：支援卡只能在得意训练位置闪耀
- `calc_training_value()`：两阶段训练数值计算（卡 buff 约束后叠加拉面 buff）
- `generate_events()`：剧本事件（Random/Fixed）→ 全局 Fixed 事件（400000400/4009/4010）→ 友人事件 → 基础随机事件；固定事件（ticket@48、ending@77）
- 动态人头管理：`manage_persons_on_turn_start()`、`update_refresh_mind()`

### 动作定义（action.rs）
- `RamenAction`: 三阶段承载结构（`ramen` + `special_targets` + `operation`），`apply` 按当前 stage 路由
  - 构造器：`new(operation)`（Train 阶段 operation-only）、`with_ramen`、`combined_select`
  - RamenSelect/SpecialSelect 中间步骤动作写 pending；Train 阶段动作只执行 operation（拉面效果已由 `ground_ramen_effects` 落地）
- `do_train()`：训练执行（拆分为多个辅助函数）
- `distribute_super_ramen_clones()`：超级拉面分身分配（随机选择训练位置，失败则重试）
- `try_add_clone()`：尝试添加分身（处理满员和挤NPC逻辑）
- `TrainParams`：训练参数缓存结构

### 规则函数（rules.rs）
- 诀窍系统：`add_gauge`、`add_feeling`、`calc_gauge_base_distribution`（floor + 消耗=1固定 + 最小已分配优先补足）
- 做面/吃面：`can_make_ramen`、`consume_for_ramen`、`list_special_targets_for`、`calc_ramen_pt_gain`
- RMJ结算：`check_rmj`（返回RmjResult枚举）
- 地区选择：`get_region_range`、`get_region_combinations`、`validate_region_selection`、`calc_region_bonus`
- 分身规则：`get_region_clone_trains`、`get_super_ramen_clone_train_options`
- 隐藏风味：`get_turn_special_feeling`
- 诀窍槽填充：`fill_gauge_after_train`、`fill_gauge_after_non_train`（夏合宿走 `fill_gauge_xiahesu_max` 全 MAX 路径）
- 训练加成：`calc_train_feeling_bonus`、`apply_friendship_gauge_bonus`

### 效果计算（effects.rs）
- `RamenTrainingEffect`: 合并所有来源的训练效果
- `calc_ramen_training_effect`: 普通/超级拉面回合的训练效果总入口
- `calc_finals_effect`: 超级拉面回合效果（ramen_pt_effect、ramen_basic_effect 按最高档，RMJ结算效果，finals_effect）
- `calc_normal_effect`: 普通回合效果（PT常驻 + RMJ常驻 + 吃面基础 + 地区效果）
- `calc_scenario_deyilv`: 剧本得意率总加成（pt_effect + rmj_effect）
- `apply_ramen_training_value`: 应用训练效果计算数值

### 事件处理（events.rs）
- `FriendEventState`: 友人事件状态管理
- `assign_train_feeling_type`: 训练角标分配（每种诀窍至少出现1次）
- `push_hint_event`: hint 事件生成（含 hint_level/total_hints 上限处理）

### 策略（policy.rs）
- `fixed_region_selection`: 地区选择策略（固定顺序）
- `fixed_super_ramen_selection`: 超级拉面选择策略（固定选项二）

## 训练员（Trainer）

### RandomTrainer（猴子训练员）
- **定义位置**：`crates/umasim/src/trainer/mod.rs`（泛型 `Trainer<G>`）
- **用途**：随机决策器，可用于测试和基线对比
- **决策逻辑**：
  - 体力 < 45 → 优先休息（Sleep）
  - 心情 < 5 → 优先外出（NormalOuting/FriendOuting）
  - 否则 → 优先训练（Train）
  - 都不满足 → 随机选择；三阶段决策中优先选有实质内容的候选（ramen 非 None 或 special_targets 非零），避免误选占位动作
- **使用位置**：
  - `main.rs`：模拟运行时使用
  - `game/base/basic.rs`、ramen 测试中使用
- **导入方式**：`use crate::trainer::RandomTrainer;`

### ManualTrainer（玩家训练员）
- **定义位置**：`crates/umasim/src/trainer/mod.rs`
- **用途**：玩家手动选择动作/事件（inquire 终端交互）
- **模式**：
  - `ManualTrainer::new()`：真实玩家模式（Interactive）
  - `ManualTrainer::with_mock_inputs(inputs)`：测试模式（mock 队列优先消费，耗尽后 fallback 到 PickFirst 选第一个候选）
- **使用位置**：`crates/umasim/src/bin/ramen_manual.rs`、ramen 完整流程测试

### HandwrittenTrainer / MctsTrainer
- **定义位置**：`crates/umasim/src/trainer/handwritten_trainer.rs`、`mcts_trainer.rs`
- 当前实现仍绑定旧温泉杯（`impl Trainer<OnsenGame>`），尚未适配 RamenGame；拉面杯决策粒度设计：HandwrittenTrainer 走三阶段（RamenSelect→SpecialSelect→Train），MctsTrainer 走合并决策路径（`list_combined_ramen_select_actions` + `apply_combined_ramen_decision`，方案 E）

## 输出与决策理由

### 决策理由模块（output/reason.rs）
- **定义位置**：`crates/umasim/src/output/reason.rs`
- **核心 API**：
  - `analyze_narrow_win(turn, metric, threshold, max_display, chosen, output) -> Option<DecisionReasonData>`：按评分降序取前 N，构造理由数据
  - `render_reason_lines(data) -> Vec<String>`：渲染可读文字（直接 `info!` 上屏）
  - `DecisionReasonSink` trait + `NoopSink` / `LogJsonSink` 实现：原始数据出口（`NoopSink` 为 umasim 默认）
- **触发逻辑**：每回合都输出；`reason_gap_threshold` / `SearchConfig::reason_gap_threshold` / `MctsConfig::reason_gap_threshold` 字段保留兼容值但不再用作触发器
- **行结构**：
  - 行 1：`[回合 N] 首选: <描述>`，固定亮绿色
  - 行 2..：`[回合 N] #K <描述>: ±分差 （优势/劣势子项）`，按评分降序编号，K = 2, 3, ...
- **颜色档位**：与首选差距 `<30` / `<100` / `<300` / 其余 → 亮绿 / 绿 / 黄 / 真彩色灰
- **调整阈值位置**：`reason.rs` 的 `reason_color(gap)` 函数（约 L107）—— 唯一一处，改完跑 `cargo test -p umasim --release --lib output::reason` 验证
- **口径**：颜色与文本内显示的 `±分差` 同源于 `r.gap`（候选 − 首选），避免撕裂（之前按"候选 − 最优项"曾出现"-95 黄、-96 绿"）
- **no-color feature**：`--features no-color` 或 `--no-default-features --features no-color` 编译通过，输出无 ANSI 序列；`test_color_thresholds` 在该 feature 下自动跳过

## 配置系统（Phase 2 已完成）

### 配置加载入口
- `crates/umasim/src/utils.rs::load_game_config()`：统一读取 `gamedata/default_config.toml` + `game_config.toml`，merge 后调用 `validate_game_config` 校验
- 路径解析：`resolve_default_config_path()`（环境变量 `UMAI_DATA_DIR` > 工作目录 + `gamedata/`）；`resolve_user_config_path()`
- 用户配置不存在时自动用兜底 `OverrideGameConfig`，不会阻塞启动

### 五个子配置结构（渐进式分组）
- `SimulationConfig`：剧本/训练员/马娘/卡组/模拟次数
- `SearchConfig`：MCTS + 用户可调搜索项（`mcts_turn_bonus` / `pt_favor_rate` / `race_grades`）
- `PolicyConfig`：拉面杯地区/超级拉面选择策略
- `OutputConfig`：日志级别（统计级别后续步骤接入）
- `DeveloperConfig`：collector + 线程数
- `GameConfig` 保留聚合壳 + `simulation()` / `search()` / `policy()` / `output()` / `dev()` 访问器；调用点可逐步迁移到子配置访问

### 用户可调项迁移（步骤 1）
- `mcts_turn_bonus` / `pt_favor_rate` / `race_grades` 从 `gamedata/constants.json` 迁出到 `gamedata/default_config.toml` 顶层；`game_config.toml` 可选覆盖
- `five_status_limit_base` 隔离到 `gamedata/scenario_ramen.json`（拉面杯覆盖）；basic/onsen 继续使用 `GAMECONSTANTS.five_status_limit_base`
- `no_event_turns` 保留为公共数据（constants.json）
- `rank_scores` / `rank_names` / `five_status_final_score` 待人工更新（见 issues.md）

### 拉面杯地区选择策略（步骤 5）
- `ramen_region_strategy = "all" | "fixed"`：默认 `"all"`（枚举所有组合）；`"fixed"` 跳过 120 组合枚举
- `ramen_region_fixed = [[id1, id2, id3]]`：长度 = 1，仅当 `fixed` 策略时生效
- **仅第3年（year_idx=2）生效**；第1/2年固定走 all 枚举
- 策略路由在 `RamenGame::run_region_select` 中按 `year_idx` 判定

### 温泉遗留段
- `onsen_order` / `mcts_selected_onsen` 等温泉专属配置移到 `default_config.toml` / `game_config.toml` 文件末尾，标注"Phase 6 预期删除"
- `game_config.toml` `[onsen_order]` / `[config_override]` 段保留（OverrideGameConfig 解析依赖）
