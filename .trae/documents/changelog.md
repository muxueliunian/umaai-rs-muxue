# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-22

- **搜索层真 CRN（NN 管线 Phase 1.3）**：`RolloutSeeds::stage_seed` 支持按 `(回合, 阶段)` 重新派生随机流，`simulate` 改吃 rollout 种子并在每个阶段边界重播种，由 `SearchConfig::crn_stage_reseed` 开关控制（**默认开启**，可经 `[mcts] crn_stage_reseed` 从 toml 关闭）；新增配对相关实测（onsen 7 候选 × 200 rollout）：仅共享起始种子 corr 0.18 / 等效 1.31x，开启按阶段重播种 corr 0.69 / 等效 3.65x，证实朴素共享种子几乎无收益
- **搜索层可复现（NN 管线 Phase 1.1/1.2）**：新增 `search/seeds.rs` 的 `RolloutSeeds`（rollout 种子按序号派生，候选索引不参与，为后续 CRN 留位），移除 `flat_search` 全部 8 处 `from_os_rng`，改为按工作项播种；`simulate_many` 改吃种子表 + 偏移并返回失败计数，UCB 按「已计划次数」记账避免失败导致候选间种子错位；rollout 失败由静默丢弃改为计数告警（全失败才报错），补 `search_group_size > 0` 校验；`flat_search` 新增可复现性回归测试（同种子一致 / 换种子生效 / 候选顺序无关），实测 1~24 线程结果逐位相同
- **自选比赛守门测试补充**：小栗帽两段区间专项 + 无比赛候选时不 panic
- **cargo-husky 钩子引入并撤销（用户决策）+ cargo fmt 手动化**：pre-commit rustfmt 检查钩子（nightly 强制 / stable 跳过）引入后，因 fmt 改由用户手动执行而移除；AGENTS.md 新增「禁用 cargo fmt」规则；全库应用当前 nightly rustfmt 格式
- **bench 玩家 build 预置**：`DeckComposition` + `PLAYER_BUILDS`（7 种主流玩家 build，guts_wisdom 按玩家讨论删除）外置到 bench_config.toml `[player_builds]`（Map 形态一行一个，IndexMap 保声明序，无内置默认、未配置报错）
- **game_config.toml 加载修复**：修复用户配置路径错误（此前从未被加载、一直走兜底默认）；`[config_override]` 全字段可选覆盖（新增 uma/cards/blue_count）+ 未知字段显式报错
- **issues 更新**：constants 排名数据（补齐至 LS24）与 rustfmt 条目状态更新

## 2026-08-21

- **bench 设施与全卡型基准**：新增 `umasim::bench` 公共设施（双 RNG 分裂 / 单局运行 / 统计 / CSV / 代表性选卡）+ `bench_compositions`（101 种卡组构成跑批），bench_base / bench_compositions 复用瘦身
- **手写策略规划文档**：新增 handwritten_policy 目录：定位（MCTS rollout 基策）、策略形态（参数化利于调参）、输出分层（决策日志 / DecisionInfo / GameView）、玩家经验标签
- **手写策略三步交付**：① 地基：bench_base + 决策日志 + 规则层可复现性修复（Random 基线 mean=30432）② 核心：RamenPolicy 各阶段打分 + RamenHandwrittenTrainer（较 Random +39%）③ 自选比赛守门 + 打分自洽性修正（实测 +18.5%）
- **rustfmt 规则固化 + AGENTS.md 微调（用户）**：明确 Nightly 格式、stable 禁跑 cargo fmt；需求澄清与安全注意事项表述精简

## 2026-08-20

- **注释精简**：umasim/Cargo.toml 注释 38→14 行；Rust 长注释压缩 6 处（文件头、重复的 1121 维清单去重），保留 13 处高价值文档（公式 / 索引映射 / 机制契约）
- **colored 无条件加载**：colored 从 cli feature 移出改为无条件依赖（非 Windows 纯 std 实现，Android / 嵌入式交叉编译无风险），消除 9 个文件约 20 处彩色双版本 cfg gate 重复代码；no-color 编译期无色语义不变
- **Phase 4 步骤1：依赖边界整理 + feature 拆分**：删除 analyzer crate；umasim feature 三层设计（default = cli + diag，新增 no-color / onnx）；15+ 文件 cfg gate 治理；nn 模块整体 cfg gate 到 onnx；umaai 依赖瘦身（去掉 tract-onnx）；四种编译组合通过；暂不抽 umasim-core
- **日志模块重构（Phase 3）**：新增 output 模块（diag! 宏 / GameView）；142 处规则层日志迁至 diag!；GameView 扩至 8 字段并删除 disable_log / enable_log；LOGGER 锁合并为 OnceLock，release 编译零 warning
- **测试日志简化**：新增 init_test_logger（只输出 stderr 不写文件），100+ 处测试迁移
- **友人事件词条生效修复**：apply_event 应用"事件效果提高 / 恢复量提高"词条，三剧本统一生效
- **排名数据补全**：rank_scores / rank_names 补齐至 LS24，速度档位上调
- **第3年地区选择默认 Fixed**：走固定组合 [[11,14,15]]，跳过 120 组合枚举
- **拉面杯回合规则收紧**：回合 0-12 无自选比赛；回合 0-1 与超级拉面回合跳过吃面阶段
- **其他**：友人高羁绊概率 0.3→0.25；ramen_manual 改密码学随机种子；新增 tests_overview.md

## 2026-08-19

- **吃面效果立即落地**：选完面与隐藏诀窍用法后立即消耗诀窍、效果生效并生成分身，玩家选训练前可见完整 buff
- **hint_special 全员触发**：第三年吃面且支援卡种类达标时，相关训练位置全部支援卡强制出 Hint
- **ManualTrainer 玩家测试**：支持真实终端交互与 mock 两种模式；新增完整 77 回合与 hint_special 路径的端到端测试
- **修复并发测试日志初始化竞争**
- **配置系统 Phase 2**：用户可调项迁至 default_config.toml（步骤1）；GameConfig 五子配置分组（步骤2+3）；配置加载集中化 + 统一校验（步骤4）；拉面杯第3年地区选择策略接入 PolicyConfig + TOML 精简（步骤5）；文档收尾（步骤7）
- **文档整理**：project_context 按实况更新，旧 issues 归档

## 2026-08-18

- **剧本 PT 每年归零**：RMJ 结算后归零重新累计，URA 阶段不再累计
- **RMJ 事件时机修正**：结算当回合立即触发；超级拉面基础效果 URA 回合自动生效（赛后加成仅首次）
- **事件补全**：RMJ 结算成功 / 失败事件 + 固定触发事件（登场 / 新年 / 抽签 / 结局），修复比赛回合事件漏触发
- **训练分布剧本得意率加成修复**（含 RMJ 效果）
- **夏合宿规则实现**：诀窍槽全 MAX、禁用普通 / 友人外出与治病、休息自动清除不良状态
- **决策重构**：新增"选面 + 吃法"一次性合并决策接口；动作阶段扩展为"选面 → 选诀窍用法 → 训练"三阶段

## 2026-08-17

- **umaai 跨平台构建支持**：可在 Ubuntu / Linux 下编译运行（Windows 专用依赖按平台限定）
- **拉面杯模块机制修正、显示改进与架构重构**：友人事件 / 分身系统 / 地区选择 / RMJ 结算 / 超级拉面 / 诀窍角标等
- **训练数值端到端观测测试**：固定回合打印吃面 / 不吃面场景的训练分布与数值

## 2026-08-16

- **拉面杯模块 1d 最小闭环**：回合 0-77 完整阶段流转、组合动作生成、事件处理、动态人头管理、回合边界处理
- **1b 核心游戏机制 + 1c 动作预览和手写策略**：诀窍 / 做面吃面 / RMJ 结算 / 地区选择 / 分身 / 隐藏风味 / 友人事件；"吃面选择 × 基础操作"分离决策模型
- **1a 核心类型定义 + 1b-1 诀窍系统**：拉面杯模块结构与核心类型；诀窍槽基础值分配、库存溢出、训练 / 友情加成
- **拉面重构计划调整**：Phase 合并为 1a-1d，归档旧规划文档、统一领域术语（食材→诀窍等）

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
