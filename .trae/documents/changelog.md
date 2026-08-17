# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

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
