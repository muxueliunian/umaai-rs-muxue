# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-22

- **Git 钩子自动化（cargo-husky）**：引入 cargo-husky 1.5.0（workspace dev-dependencies，`user-hooks` feature）；新增 `.cargo-husky/hooks/pre-commit`：有 nightly 工具链时 `cargo +nightly fmt --all -- --check` 强制检查，无 nightly 则跳过（对齐 AGENTS.md「stable 下禁止 cargo fmt」）；构建时自动分发到 `.git/hooks/`，协作者 clone 后跑一次 test 即自动生效。钩子文件权限要求 r-x r-x r-x（cargo-husky 以 `mode & 0o555` 判定可执行）
- **全库 rustfmt 格式化**：应用当前 nightly rustfmt（1.10.0-nightly 2026-08-20）格式，40 文件 387 处差异收敛；rustfmt.toml `struct_lit_width` 32→30（工具规范化写入）
- **seeded_rngs 注释补充**：`bench.rs` 说明 0x9E37_79B9_7F4A_7C15 为 SplitMix64 标准 gamma 增量常数（黄金比例，有据可依非任意指纹），派生规则保持固定以保证可复现性
- **issues 更新**：constants.json 排名数据 issue 标记已解决（rank_scores / rank_names 补齐至 LS24、five_status_final_score 核对，提交 aa756d9，数据经用户确认）；rustfmt 问题解决方案补充钩子自动化（cargo-husky pre-commit）与全库格式化（含 nightly 滚动版本漂移说明）
- **bench 通用卡组构成设施**：bench.rs 新增 `DeckComposition`（5 张普通卡数量分布 + 预设短名，`kind_count`/`name`/`name_zh`/`build_deck`/`make_deck` 一步自动选代表卡生成卡组）+ `PLAYER_BUILDS`（7 种主流玩家 build：speed 3,1,0,0,1 / stamina 2,2,0,0,1 / power_wisdom 0,0,2,0,3 / speed_wisdom 2,1,0,0,2 / wisdom 1,1,0,0,3 / average 1,0,1,1,2 / guts_wisdom 0,0,0,3,2，数据来自玩家经验）；bench_compositions 迁移复用（删本地 Composition/build_deck，101 枚举改用 DeckComposition）；测试 +2（build 形状 / 真实 cardDB 下 7 build 一步生成卡组）
- **guts_wisdom 删除（用户决策）**：按玩家讨论暂时删除 guts_wisdom build（原为 7 种中的对照项，仅 2 种普通卡、不满足拉面杯「支援卡种类≥4」门槛）；PLAYER_BUILDS 变 6 种，注释保留恢复说明
- **PLAYER_BUILDS 外置到 bench_config.toml（用户决策）**：`DeckComposition.name` 改为 `String`（支持 serde 解析）；`PLAYER_BUILDS` const 改为 `default_player_builds()`（内置兜底）+ `load_player_builds()`（读取 bench_config.toml 的 `[[player_builds]]` 段，文件缺失/未配置/为空时回退默认，校验名称非空唯一、合计 5 张、单类型 ≤3）；bench_config.toml 新增 6 个 build 段（用户可直接手动调整）；bench.rs 测试重构：默认形状 / 校验拒绝 / 真实配置读取 / 真实 cardDB 生成卡组（4 个）

## 2026-08-21

- **bench 重构**：新增 `umasim::bench` 公共设施（seed 双 RNG 分裂、`run_seeded` 单局运行、统计、CSV 落盘、代表性选卡），`bench_base` / `bench_compositions` 复用瘦身净减约 200 行；新增 `csv` / `lexopt` 依赖（无条件编译，保持 `--no-default-features` 可编译）；bench_compositions 选卡改「最新 5 张 + 面板和值≥70 + 倒序取 3」，阈值可配、`--cards-file` 兜底；屏幕输出中文（复用 `train_names`）、CSV 英文，代表卡列表移至跑批后并打印跳过卡；测试 +4（共 160）
- **rustfmt 规则固化（用户）+ issues 更新**：AGENTS.md 新增规则（项目使用 Nightly 格式，stable 下禁止 cargo fmt）；issues 新增 rustfmt 问题（已解决）并更新第三年地区选择状态为已解决
- **拉面杯全卡型构成基准**：新增 `bench_compositions`，枚举五种普通支援卡各 0..=3 张、合计 5 张再加固定友人的全部 101 种构成；支持固定种子、Random/Handwritten 训练员、CSV 聚合输出与异常局失败；新增构成数量和六卡结构测试
- **手写基础策略（HandwrittenTrainer）规划文档**：新增 `.trae/documents/handwritten_policy/` 目录（思路总纲 + 好/坏手法主线特征清单）；确定范围（只做 MCTS rollout 基策、HandwrittenTrainer 仅测试壳）、策略形态（参数化利于调参）、确定性要求、复用现有公式算法、卡组适配方案；明确输出分层（决策日志 / DecisionInfo / GameView 各司其职）与地基方案（RandomTrainer 补鲁棒性与覆盖率、简单基础策略做质量基线与调参载体）；收录玩家经验输入：好/坏手法主线特征（free_race 达标为硬守门、第三年隐藏风味需清空等）与决策频率×影响维度（高频小影响 vs 低频大影响）
- **手写策略第一步：地基设施**：新增 `bench_base` 基准 bin + `bench_config.toml`（固定种子跑批，结果/决策日志 CSV 落盘，汇总含分数分布、按阶段决策耗时、吞吐）；新增决策日志模块 `output/decision_log.rs` 与 `LoggingTrainer`（记录每次决策的耗时与动作）；修复规则层可复现性（`RamenGame.internal_rng` 替换 `Game::next()` 内 3 处 `from_os_rng`，固定种子整局可复现）；基线：Random 20 局 score mean=30432、整局 ~1.2ms；新增 decision_log 与 logging_trainer 单元测试
- **手写策略第二步：策略核心 + RamenHandwrittenTrainer**：
  - `game/ramen/policy.rs` 扩展为手写策略核心：`RamenPolicyConfig`（参数化权重/阈值 + default/speed_build 预设）+ `RamenPolicy`（Train 期望收益与守门 / RamenSelect PT+效果贪心 / SpecialSelect 最省隐藏风味 / RegionSelect 静态组合打分 / 事件效果折算，确定性 argmax）
  - 新增 `RamenHandwrittenTrainer`（`trainer/ramen_handwritten_trainer.rs`）：测试壳，直接在拉面杯规则层上重新实现，不依赖旧温泉实现
  - 规则补充（通用化）：URA 回合 72/74/76 不可自选比赛与友人出行 → 收敛为 `BaseGame::can_self_race` / `can_friend_outing`；`do_race` URA 回合自选比赛按 G1 防越界（手写策略主动选比赛暴露）
  - bench_base 支持 `--trainer random|handwritten` A/B；基线对比（seed 42-61，20 局）：Handwritten mean=42560 vs Random mean=30698（+39%），整局 1.15ms——v0 即可作 MCTS rollout 基策
  - 测试：policy 守门/打分/确定性 7 个 + handwritten 完整局/可复现 2 个 + base 通用规则 2 个（共 153 个）
- **手写策略第三步：自选比赛守门 + 打分自洽性修正**（仅改 `game/ramen/policy.rs`，不动规则层）：
  - 自选比赛（free_race）两层处理：硬守门（区间内剩余可比赛回合 ≤ 缺口+slack 时强制比赛，优先于生病/体力/心情）+ 软倾向打分（`urgency × 缺口 / 剩余回合`）；新增 `race_free_urgency_weight` / `race_gate_slack`，`race_grade_weight` 默认 900→0（实测主动比赛纯亏）
  - 打分自洽性三修：breakdown 各项之和现等于 score（原先失败率项与实际扣分口径不一致）、`status_rate` 去掉重复相乘（原为平方）、彩圈加成纳入 `×(1-失败率)`
  - 实测（bench_base，seed 42-81）：美浦波旁 102601 mean 43168→51168（+18.5%，UD/UC→UB/UA），一局比赛数 41→9；无声铃鹿 100201（回合 12-26 需 1 场自选比赛）无 free_race 感知时 mean 9849（育成失败）、仅软倾向 47590（20 局失败 1 局）、完整方案 49513 且零失败
  - 测试：新增 4 个（剩余可比赛回合掩码 / 守门四场景 / breakdown 求和 / status_rate 线性），umasim 全量 157 passed；tests_overview 155→159
- **AGENTS.md 规则微调（用户）**：精简需求澄清与重构期文档策略表述；安全注意事项补充"远程主机/设备"与"脚本间接操作需确认"

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
