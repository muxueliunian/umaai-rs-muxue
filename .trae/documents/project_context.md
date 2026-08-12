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

