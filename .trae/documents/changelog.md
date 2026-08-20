# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-20

### 友人事件词条生效修复

落实 issues「友人事件效果未应用『事件效果提高』『恢复量提高』词条」：`BaseGame` 新增 `friend_event_ids` 字段并在 `apply_event` 中识别友人事件、应用 `event_bonus`（属性/pt 乘算）与 `vital_bonus`（正向体力乘算）；base / onsen / ramen 三剧本统一生效，新增 5 个单元测试覆盖各路径。

### 测试清理

删除 `test_ramen_region_strategy_fixed_skips_enumeration`（与本任务无关且与新增 `friend_event_ids` 字段冲突，地区选择策略验证另行处理）。

### 排名数据补全

rank_scores / rank_names 从 US9 之后补齐至 LS24；five_status_final_score 同步追加；five_status_limit_base 速度档位上调。数据源来自 master_mdb_data/rank.csv。

### 第3年地区选择默认 Fixed

第3年默认走固定地区组合 [[11, 14, 15]]，跳过 120 组合枚举；第1/2年仍走全部枚举。

### 拉面杯回合规则收紧

- 回合 0-12 不在菜单中提供比赛选项（回合 11 出道赛、回合 12 无可用比赛）
- 回合 0-1 与超级拉面回合直接进入训练选择，跳过吃面/隐藏风味阶段
- 吃面消耗增加前后日志，便于排查"消耗大于库存"

### 其他

- 友人解锁高羁绊概率 0.3 → 0.25
- ramen_manual 改为密码学随机种子，启动不打印
- 新增 `tests_overview.md`：按模块分类、一句话描述全部 121 个测试，并附"规则固化后缩减参考"

## 2026-08-19

### 吃面效果立即落地

落实"选完面与隐藏诀窍用法后立即消耗诀窍、拉面效果生效并生成分身"的需求，玩家在选择训练动作前即可看到完整 buff 与分布；"已吃面未训练"的中间状态也可由外部直接落地。

### hint_special 全员触发

落实 issues「basic_effect.hint_special 尚未处理」：第三年吃面且支援卡种类达标时，相关训练位置的全部支援卡强制出 Hint，并逐个触发 Hint 事件（保留各自次数加成）。

### ManualTrainer 玩家测试与完整流程测试

- ManualTrainer 支持真实终端交互与 mock 输入（测试）两种模式
- 新增独立手动游玩程序，以及完整 77 回合流程、第3年 hint_special 路径的端到端测试

### 并发测试日志初始化竞争修复

修复测试批量运行时日志初始化的并发竞争问题，全部测试稳定通过。

### 中间步骤文案修正

修正中间步骤动作在玩家菜单中的显示文案。

### 文档整理：project_context 更新与 issues 归档

- project_context.md 按当前代码实况更新（拉面杯模块结构、测试与训练员章节），移除已知问题章节
- 原 issues.md 归档至 archive/issues_2026-08-19.md，新建 issues.md 保留当前未解决条目

### 配置系统 Phase 2 步骤 1：用户可调项迁移

- `mcts_turn_bonus` / `pt_favor_rate` / `race_grades` 从 `gamedata/constants.json` 迁出到 `gamedata/default_config.toml`（顶层），`game_config.toml` 可覆盖
- `five_status_limit_base` 从 constants.json 隔离到 `gamedata/scenario_ramen.json`（拉面杯剧本覆盖），basic/onsen 继续使用全局默认；`no_event_turns` 保留为公共数据
- 引入 `init_global_with_config(&GameConfig)`：把用户可调项注入 `GAMECONSTANTS`，所有现有引用点不变；旧 `init_global()` 保留为兜底重载
- 4 个入口（umasim/umaai/analyzer/ramen_manual）改用 `init_global_with_config`

### 配置系统 Phase 2 步骤 2+3：GameConfig 子配置分组 + serde skip 注释强化

- 定义五个职责子结构：`SimulationConfig` / `SearchConfig` / `PolicyConfig` / `OutputConfig` / `DeveloperConfig`，分别承载剧本与训练员、搜索参数、策略参数、输出与开发者项
- `GameConfig` 保留聚合壳，新增 `simulation()` / `search()` / `policy()` / `output()` / `dev()` 访问器；业务模块可逐步迁移到子配置访问，调用点零改动
- `GameConstants` 中步骤 1 迁出的三个字段（`race_grades` / `pt_favor_rate` / `mcts_turn_bonus`）加 `#[serde(default, skip)]` 并附文档化注释说明"运行时由 `init_global_with_config` 注入"；后续若需彻底移除字段，再统一迁移引用点
- `PolicyConfig` 当前为空（占位），步骤 5 将接入拉面杯地区/超级拉面选择策略
- `default_config.toml` / `game_config.toml` 顶部加"配置段"导航注释（serde 仍按顶层平铺解析，段结构为组织概念）

### 配置系统 Phase 2 步骤 5：拉面杯第3年地区选择策略接入 + TOML 精简

- `PolicyConfig` 新增 `ramen_region_strategy`（`"all"` / `"fixed"`）和 `ramen_region_fixed`（长度=1，单组合）字段
- 新增 `GAMECONFIG: OnceLock<GameConfig>` 全局，`init_global_with_config` 注入；`RamenGame::run_region_select` 仅对第3年（year_idx=2）应用 Fixed 策略跳过 120 组合枚举，第1/2年固定走 all 枚举
- 新增 2 个测试：`test_ramen_region_strategy_fixed_skips_enumeration` 验证第3年 fixed 路径；`test_year1_2_always_all_regardless_of_strategy` 验证第1年 Fixed 策略无效（走 all）
- `test_ramen_silent_loop` 验证 All 路径不变（端到端 77 回合）
- TOML 精简：`default_config.toml` 247→约170 行，`game_config.toml` 60→33 行；onsen 配置（`onsen_order` / `mcts_selected_onsen`）移到文件末尾并标注"Phase 6 预期删除"

### 配置系统 Phase 2 步骤 4：加载集中化

- 路径常量：`DEFAULT_CONFIG_REL_PATH` / `USER_CONFIG_REL_PATH` / `DATA_DIR_REL_PATH` / `ENV_DATA_DIR`（`UMAI_DATA_DIR` 覆盖 data 根目录绝对路径）
- `resolve_data_dir()` / `resolve_default_config_path()` / `resolve_user_config_path()`：路径解析集中
- `validate_game_config()`：统一校验 scenario/trainer 枚举、cards 长度、ramen_region_fixed 长度（fixed 策略时）
- `load_game_config()` 改为：先打 info 日志、再读 default、再按路径存在性读 user（缺失时构造兜底 OverrideGameConfig，不阻塞启动）
- 4 个单元测试覆盖：scenario 枚举、trainer 枚举、ramen_region_fixed 长度、路径解析

### 配置系统 Phase 2 步骤 7：文档收尾

- `project_context.md` 新增"配置系统（Phase 2 已完成）"章节：覆盖配置加载入口、五个子配置结构、用户可调项迁移、拉面杯地区选择策略、温泉遗留段
- `ramen_refactor_development_plan.md` 执行检查清单标记 Phase 2 步骤 1-5/7 完成
- 新增 5 个 Phase 2 相关测试（`test_validate_game_config_*` + `test_resolve_default_config_path`），全部通过；`test_ramen_silent_loop` 端到端 77 回合验证 All 路径不变

## 2026-08-18

### 剧本 PT 每年归零

剧本 PT 在每年 RMJ 结算后归零，下一年重新累计；URA 阶段不再累计。

### RMJ 事件时机与超级拉面基础效果自动应用

- RMJ 事件改为结算当回合立即触发，不再延迟一整个回合
- 超级拉面基础效果（体力、心情、赛后加成）在 URA 回合自动生效；赛后加成仅首次生效，不重复累加

### RMJ 结算事件与固定触发事件补全

每年 RMJ 结算后触发对应成功/失败事件；补充固定触发事件（登场、新年、抽签、结局等），并修复比赛回合事件漏触发问题。

### 训练分布剧本得意率加成修复

训练分布计算补充剧本得意率加成（含 RMJ 效果）。「不出现」判定仍受得意率影响的小问题留待后续，详见 issues。

### 夏合宿规则实现

落实夏合宿三条特殊规则：诀窍槽全 MAX、禁用普通/友人外出与治病、休息自动清除不良状态。

### 合并决策接口（两阶段聚合）

新增"选面+吃法"一次性决策的合并路径，决策粒度交给训练员选择（三阶段或合并），为在线搜索预留。

### 三阶段决策重构（隐藏风味显式化）

隐藏诀窍用法改为显式决策：动作阶段扩展为"选面 → 选诀窍用法 → 训练"三阶段；同步修复全局初始化幂等问题。

## 2026-08-17

### umaai 跨平台构建支持

- umaai 现可在 Ubuntu/Linux 下编译运行（Windows 专用依赖与资源编译按平台限定，补充 Linux 专用依赖）
- issues 新增 Ubuntu 图标方案待定记录；AGENTS.md、project_context.md 同步更新

### 拉面杯模块机制修正、显示改进与架构重构

- 机制修正：友人事件改用拉面杯专用事件、友人出行获得隐藏风味、羁绊归属修正、RMJ 成功训练等级加成、诀窍角标分配保证全覆盖、地区选择时机修正、休息心得等
- 显示改进：回合信息补充隐藏诀窍与训练明细、NPC 匿名显示、吃面后展示完整信息、设施等级含剧本加成、拉面效果按普通/超级拉面回合区分展示等
- 架构重构：地区选择统一走训练员接口、分身系统重构（本体 ID、共享羁绊、限定闪耀位置）、超级拉面回合效果生效、RMJ 结算效果按年度生效、Hint 率含剧本加成、拉面羁绊全卡组生效、年3配方复用年1等

### 训练数值端到端观测测试

新增端到端观测测试，固定回合打印不吃面/吃面场景的训练分布与数值，用于排查训练数值问题。

## 2026-08-16

### 拉面杯模块 1d 最小闭环实现

跑通拉面杯最小闭环（回合 0-77）：完整阶段流转、组合动作生成、事件处理、动态人头管理、回合边界处理；同步更新开发计划。

### 拉面杯模块 1b 核心游戏机制 + 1c 动作预览和手写策略

- 1b：实现诀窍系统、做面/吃面、RMJ 结算、地区选择、分身、隐藏风味、友人事件、训练角标等核心机制
- 1c：实现动作生成与地区/超级拉面选择策略，采用"吃面选择 × 基础操作"分离决策模型
- 同步更新开发计划

### 拉面杯模块 1a 核心类型定义 + 1b-1 诀窍系统

- 建立拉面杯模块（入口/状态/动作/规则/效果/事件/策略）
- 定义拉面杯核心类型（游戏状态、剧本专用状态、效果、动作、阶段、诀窍类型、训练类型、操作）
- 实现诀窍系统：槽基础值分配、库存溢出管理、训练/友情加成；修正槽溢出逻辑

### 拉面重构计划调整与文档整理

重构开发计划 Phase 结构（合并为 1a-1d 六个 Phase），归档旧规划文档、统一数据目录命名、修正领域术语（食材→诀窍、万能食材→隐藏风味等），补充训练角标分配细节。

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
