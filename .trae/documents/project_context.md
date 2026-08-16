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

## 拉面杯模块结构

### 核心类型（state.rs）
- `RamenGame`: 拉面杯游戏主状态，通过Deref暴露BaseGame
- `RamenState`: 拉面杯专用状态（诀窍、隐藏风味、剧本PT等）
- `RamenEffect`: 效果合并（基础+地区+超级拉面+PT常驻）

### 动作定义（action.rs）
- `RamenAction`: 拉面杯动作（ramen + operation）
- `list_ramen_choices()`: 阶段1 - 吃面选择
- `list_operations()`: 阶段2 - 基础操作
- `list_all_actions()`: 组合所有动作

### 规则函数（rules.rs）
- 诀窍系统：add_gauge、add_feeling、calc_gauge_base_distribution
- 做面/吃面：can_make_ramen、consume_for_ramen、calc_ramen_pt_gain
- RMJ结算：check_rmj（返回RmjResult枚举）
- 地区选择：get_region_range、validate_region_selection
- 分身规则：get_region_clone_trains、get_super_ramen_clone_train_options
- 隐藏风味：get_turn_special_feeling

### 效果计算（effects.rs）
- `RamenTrainingEffect`: 合并所有来源的训练效果
- `calc_ramen_training_effect`: 计算拉面杯训练效果
- `apply_ramen_training_value`: 应用训练效果计算数值

### 事件处理（events.rs）
- `FriendEventState`: 友人事件状态管理
- 训练角标分配：assign_train_feeling_type

### 策略（policy.rs）
- `fixed_region_selection`: 地区选择策略（固定顺序）
- `fixed_super_ramen_selection`: 超级拉面选择策略（固定选项二）

