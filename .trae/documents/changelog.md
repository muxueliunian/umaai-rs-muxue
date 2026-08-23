# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-24
- **训练人数加成按人头类型计数**：`1 + 0.05 × 人数` 乘区原按硬编码人头下标排除理事长与记者，那是温泉布局的常量；拉面的理事长、友人卡、NPC、记者位置全不同，四项判反。改为按人头类型判定并抽出 count_training_persons，负数与越界下标一并不计。**改变拉面模拟数值，既有基线作废；温泉与 base 逐位不变，有回归用例守门**
- **超级拉面分身补上友人卡**：候选收集写死卡组下标范围，取不到回合 2 才加入的友人卡，类型过滤里的友人分支成了死条件，友人卡分身从未生成过；同时给分身补上「每个训练只能出现一个友人」——本体由分配逻辑维护该约束，分身此前不受限，会出现与理事长 / 记者同格这种自然分配产生不出的局面。**改变拉面模拟数值**

## 2026-08-23
- **拉面杯 MCTS 训练员**：新增按阶段门控的搜索训练员，命中的决策点走扁平搜索、其余转发手写策略，门控全关时与纯手写逐位一致；bench_base 与主二进制接入搜索参数；搜索层新增由剧本指定的 rollout 根动作路径（拉面走策略流），rollout 基策关闭打分分解采集以降开销
- **拉面局面特征编码器**：新增 features 模块，把局面编码为定长向量（global / cards / persons 三段），较温泉版补齐成长率与属性上限并开启人头分支；分块记账、查表失败报错不填 0，只喂原始状态与纯计数派生量；同时移除恒为全零的 current_effect 特征块（上游字段暂留待确认）
- **人头下标与卡组槽位解耦**：新增按 card_id 反查卡组槽位，拉面下人头顺序与卡组顺序不一致，原先按 person_index 直接当卡组下标的调用点全部改为反查。**改变拉面模拟数值，既有基线与落盘教师数据作废**
- **手写策略地区打分覆盖第 1 年**：新增有效阶段判定，使回合开始阶段内联触发的第 1 年地区选择也进入打分；MCTS 门控仍用原始阶段，避免破坏 rollout 的阶段推进契约。**改变手写策略基线数值**
- **地区选择 build 自适应**：score_region 纳入 youqing 项，xunlian 与 youqing 统一按卡组 bias 缩放；配置恢复枚举全部组合
- **测试观测收集器**：新增 utils::Checks，测试全程 println 记 OK/NG、末尾汇总有失败才报错，兼顾诊断输出与回归防线；既有的裸断言与重复本地实现一并归拢

## 2026-08-22
- **基准新增自选比赛达标维度**：新增任意时点重比各区间完成场数的判定（原判定只在区间结束回合的下一回合执行，且不达标即终止育成），bench 结果与 CSV 加达标率并在每局 / 分组 / 总览打印；配套补两个守门测试（不改策略逻辑），逐回合扫描触发点以免随常量表调整失效
- **搜索层可复现 + 真 CRN + 泛型化（NN 管线 Phase 1，已完成）**：rollout 种子改为按序号确定性派生（候选索引不参与，否则协方差归零），移除全部随机播种，失败由静默丢弃改为计数告警；新增按阶段边界重播种的真 CRN（默认开启，可从 toml 关），实测朴素共享起始种子几乎无收益、按阶段重播种才显著；搜索结构泛型化并保留默认类型参数使活跃入口零改动，采用「公共内核 + rollout 闭包」规避泛型方法解析导致温泉特判静默失效；顺带修 NN leaf 微批路径漏重播种、UCB 终止判据用成功数会死循环两处缺陷，并把 rollout 基策的调试缓存改 Mutex 以满足跨线程共享
- **局面采样器（NN 管线 Phase 2 上半）**：为教师数据制造根局面——分层的采样空间、按工作项序号确定性导出采样任务（分片 / 续跑 / 改并行度均不变）、轨迹随机扰动、走真实决策路径截断捕获；根局面限定在阶段入口，回合开始阶段内联执行的决策点会破坏搜索的阶段推进契约
- **第三方库引用规范化**：搜索层与采样器中 anyhow 宏的全名引用改为 use 导入后直接调用
- **支援卡类型注释订正**：card_type 原注释与卡片数据实测相反（5 是友人、6 是团队）

- **RNG 受控重构（v3 三流，已实施）**：新增顶层 `rng.rs`（splitmix64 唯一实现 / 加法派生无状态流 SplitmixRng / 类型隔离三流 TurnFixedRng+EventRng+StrategyRng）；规则层随机改从 self 流取（run_distribute 独占局面流=角标/人头分布/hint 触发位，回合开始事件链走事件流，训练/分身/比赛走策略流），Trainer 决策流保持 StdRng；bench 局号进种子 `seeded_rngs(base,idx)→(StdRng,rule_master)`；拉面 CRN 由规则层接管（fork_for_rollout 注入 rule_master，simulate_common 退役阶段重播种），onsen 保留外挂 CRN；未注入 rule_master 时回退旧行为。验收：层 2/3 集成测试 `rng_consistency.rs`——跨策略 20 回合角标/分布/固定流消费量逐位一致（0 不一致），事件增量逐位一致；方案文档 `rng_refactor_plan.md` 更新为 v2/v3 并归档 v1，`rng_reply.md`（上游 CRN 评审意见）归档
- **umasim 主二进制接入拉面杯剧本**：main.rs 此前仅支持 onsen/basic（`scenario="ramen"` 时实际落 basic），新增 `run_ramen_once` 与 ramen 分发分支（random/handwritten/mcts 回退/默认 manual 均支持），handwritten 分支使用 RamenHandwrittenTrainer；`GameConfig::scenario` 注释补 ramen。实测主二进制跑通 77 回合拉面杯（UB2 49442 / PT 7941）
- **issues 更新**：第三年地区选择无 build 自适应（score_region 对第三年地区无区分度，实测各 build 同选一组合；方案已定待实施，含临时验证测试）
- **ramen_manual 屏幕输出整理（Agent 对话文本流风格）**：新增 turn_flow 渲染层与固定种子基线测试；候选内联预览（训练数值 / 吃面完整效果 / 诀窍配方）并分层着色；事件三段式、回合状态去重；ramen_manual 接入实时候选栏与选择确认；训练诊断输出暂屏蔽
- **第3年地区选择修复**：ramen_region 配置字段落错 TOML 段导致预设失效（恒枚举 120 组合），移回顶层后 fixed 预设生效
- **comfy-table custom_styling**：修复彩色表格 ANSI 宽度错乱
- **自选比赛守门 + 决策日志 breakdown**：等级过滤 / 摆烂判定 / 达标后停止，候选评分分解入决策日志
- **诀窍槽 NPC 按实际人数计算**、game_config.toml 加载修复、cargo-husky 撤销与 fmt 手动化、bench 玩家 build 外置与分组跑批
- **显示微调（用户）**：比赛加成信息亮品红；清理未使用 import
- **文档归档**：config_refactor_plan / log_refactor_plan 移入 archive

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
