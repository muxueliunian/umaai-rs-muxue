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
  - `default_config.toml`：默认游戏配置
  - `game_config.toml`：用户自定义游戏配置

### 游戏常量
- 回合数默认从0开始，特殊情况下会说明从1开始

### 配置格式
- 使用JSON和TOML格式

### 脚本工具
- `scripts/`目录包含Python脚本工具
- 主要用于数据导出和处理

## 开发环境

### 操作系统
- Windows

### Shell
- PowerShell

## 测试

### 模拟游戏流程测试
- 位置：`crates/umasim/src/game/ramen/game.rs`
- 使用 `test_ramen_silent_loop` 测试用例验证完整的拉面剧本游戏流程
- 该测试关闭日志（`disable_log`）运行完整游戏，仅输出育成配置和最终结果，适合作为端到端的流程验证
- 运行命令：`cargo nextest run -p umasim test_ramen_silent_loop`（或 `cargo test -p umasim test_ramen_silent_loop`）

## 拉面杯模块结构

### 模块入口（mod.rs）
- `RamenStage`：回合阶段枚举（Begin/Distribute/Train/AfterTrain/NextTurn/RegionSelect/SuperRamenSelect/Settlement）
- `FeelingType`：诀窍类型（A/B/C）
- `TrainingType`：训练类型（Speed/Stamina/Power/Guts/Wisdom）
- `Operation`：基础操作（Train/Race/Rest/NormalOuting/FriendOuting/Clinic/RegionSelect([usize; 3])）

### 核心类型（state.rs）
- `RamenGame`: 拉面杯游戏主状态，通过Deref暴露BaseGame
- `RamenState`: 拉面杯专用状态（诀窍、隐藏风味、剧本PT等）
  - `train_level_bonus`: 训练等级剧本加成字段
  - `deck_can_split`: 支援卡种类>=4时为true，用于判断分身条件
- `RamenEffect`: 效果合并（基础+地区+超级拉面+PT常驻）
- 辅助方法：`add_friend_and_npcs()`、`add_reporter()`、`add_person()`
- `init_feeling_stocks()`: 诀窍初始化方法
- `add_friendship()`: NPC不增加羁绊

### Game trait 实现（game.rs）
- 阶段流转：`RamenStage::next()` 负责回合内，`Game::next()` 负责跨阶段
- `init_persons()`：开局仅加入非友人卡 + 理事长
- `run_stage()`：分发到各阶段处理函数（run_begin/run_distribute/run_train/run_after_train 等）
- `run_region_select()`：年度地区选择（第1年在回合2 Begin阶段，第2/3年在回合23/47 NextTurn阶段）
- `explain_ramen_info()`：格式化拉面杯剧本信息（包含拉面效果、诀窍、PT等）
- `init_feeling_stocks()`：诀窍值初始化/重置
- `distribute_hint()` override：应用剧本Hint率加成
- `calc_hint_bonus_pct()`：计算剧本Hint加成
- `list_actions()`：生成所有吃面/不吃面 × 操作的组合动作
- `generate_events()`：复用 BasicGame 随机事件
- 动态人头管理：`manage_persons_on_turn_start()`
- `update_refresh_mind()`：更新休息心得效果
- `is_shining_at()` override：闪耀判定（支援卡只能在得意训练位置闪耀）

### 动作定义（action.rs）
- `RamenAction`: 拉面杯动作（ramen + operation）
- `ActionEnum` 实现：吃面与训练严格分阶段执行
- `apply_ramen()`：吃面处理（消耗诀窍、获得PT、分身分配）
- `do_train()`：训练执行（拆分为多个辅助函数）
- `do_friend_outing`：友人出行（使用拉面杯事件 + 隐藏风味）
- RegionSelect动作处理
- `TrainParams`：训练参数缓存结构
- `distribute_clones()`：地区拉面分身分配（每个at_trains位置随机选一个不重复的支援卡）
- `distribute_super_ramen_clones()`：超级拉面分身分配（随机选择训练位置，失败则重试）
- `try_add_clone()`：尝试添加分身（处理满员和挤NPC逻辑）
- `apply_ramen_friendship()`：训练前应用拉面羁绊效果

### 规则函数（rules.rs）
- 诀窍系统：add_gauge、add_feeling、calc_gauge_base_distribution
- 做面/吃面：can_make_ramen、consume_for_ramen、calc_ramen_pt_gain
- RMJ结算：check_rmj（返回RmjResult枚举）
- 地区选择：get_region_range、validate_region_selection、get_region_combinations（生成所有3地区组合）
- 分身规则：get_region_clone_trains、get_super_ramen_clone_train_options
- 隐藏风味：get_turn_special_feeling

### 效果计算（effects.rs）
- `RamenTrainingEffect`: 合并所有来源的训练效果
- `calc_ramen_training_effect`: 计算拉面杯训练效果
- `calc_finals_effect`: 计算超级拉面回合效果（ramen_pt_effect、ramen_basic_effect按最高档，RMJ结算效果，finals_effect）
- `calc_normal_effect`: 计算普通回合效果（PT常驻+RMJ常驻+吃面基础+地区效果）
- `apply_ramen_training_value`: 应用训练效果计算数值

### 事件处理（events.rs）
- `FriendEventState`: 友人事件状态管理
- 训练角标分配：assign_train_feeling_type（每种诀窍至少出现1次）

### 策略（policy.rs）
- `fixed_region_selection`: 地区选择策略（固定顺序）
- `fixed_super_ramen_selection`: 超级拉面选择策略（固定选项二）

## 已知问题（Issues）

### 1. 训练数值不对，尤其是友情加成
- 问题描述：训练数值计算可能不正确，特别是友情加成（youqing）的生效条件
- 可能原因：
  - 闪耀判定逻辑：支援卡只能在本体的得意训练位置闪耀
  - 分身在非本体训练位置时不闪耀，友情加成不生效
  - 需要检查 `is_shining_at()` 函数的实现

### 2. 超级拉面得意率人物分配错误
- 问题描述：超级拉面分身分配时，得意率（deyilv）的计算可能不正确
- 可能原因：
  - 分身分配算法：当前使用随机选择训练位置，失败则重试
  - 需要检查是否应该按得意率权重分配
  - 需要检查 `distribute_super_ramen_clones()` 函数的实现

## 训练员（Trainer）

### RandomTrainer（猴子训练员）
- **定义位置**：`crates/umasim/src/trainer/mod.rs`
- **用途**：随机决策器，可用于测试和基线对比
- **决策逻辑**：
  - 体力 < 45 → 优先休息（Sleep）
  - 心情 < 5 → 优先外出（NormalOuting/FriendOuting）
  - 否则 → 优先训练（Train）
  - 都不满足 → 随机选择
- **使用位置**：
  - `main.rs`：模拟运行时使用
  - `game/base/basic.rs`：测试中使用
- **导入方式**：`use crate::trainer::RandomTrainer;`

