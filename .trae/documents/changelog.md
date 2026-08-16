# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-16

### 拉面杯模块 1a 核心类型定义 + 1b-1 诀窍系统
- 建立 `game/ramen/` 模块：mod.rs、state.rs、action.rs、rules.rs、effects.rs、events.rs、policy.rs
- 定义核心类型：RamenGame、RamenState、RamenEffect、RamenAction、RamenStage、FeelingType、TrainingType、Operation
- RamenEffect 字段对应剧本加成词条（xunlian、youqing、pt_bonus、train_limit、pt_limit 等）
- RamenAction 采用组合动作模型（ramen + operation），Display 显示地区名称
- 使用 IntEnum derive 替代手写 index/from_index，去掉 OutingType 独立枚举
- 新增 RegionEffect 结构体和 ramen_region_effect 字段到 RamenScenarioData
- 修正 RamenBasicEffect.jiban → friendship，与 JSON 数据对齐
- 实现诀窍系统规则函数：槽基础值分配、库存溢出管理、训练加成、友情加成

### 拉面重构计划调整与文档整理
- 将 `opt/` 目录重命名为 `archive/`，归档旧规划文档
- 将 `master/` 目录重命名为 `master_mdb_data/`，统一数据目录命名
- 新增 `ramen_phase_adjustment_analysis.md` 分析文档（已归档至 `archive/`）
- 重构开发计划 Phase 结构：合并原 Phase 1/4/5/6 为新的 Phase 1（1a-1d），缩减为 6 个 Phase
- 修正开发计划术语：食材→诀窍、万能食材→隐藏风味、73-78→72-77
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
- 更新术语表：添加诀窍槽、友人解锁、夏合宿等新术语
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
- 更新ramen_basic_effect：添加jiban/status_limit/hint_special字段，填入3年效果数据
- 添加finals_effect：定义超级拉面(超RMJ極)的基础/额外/单独效果
- 添加ramen_region_effect：记录20条地域拉面效果数据
- 更新Rust结构体：添加RamenBasicEffect结构体
- 更新ramen_memo_cn.md文档：补充效果说明和字段文档
