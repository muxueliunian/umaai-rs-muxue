# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-23
- **拉面局面特征编码器（NN 管线 Phase 2 下半）**：新增 `game/ramen/features.rs`，把 `RamenGame` 编码为定长 `f32` 向量（global / cards / persons 三段，人头段按最大 13 个定长 + 已登场掩码）。相比温泉版补上了成长率与上限（多马娘教师数据的核心区分变量，温泉版一次都没编码）并开启人头分支；分块记账（每块声明宽度、写完即校验、末尾断言总维度）、归一化尺度全部注明取值依据、查表失败一律报错不填 0；只喂原始状态与纯计数派生量，不喂任何带权重的估值。经三方评审修掉五处编码缺陷：卡与人头改按 `card_id` 双向反查（规则层下标错位，见 issues）、训练位改 multi-hot 以正确表达分身、地区块加 `regions_ready` 掩码避免把默认值当真数据、Train 阶段只编 `current_ramen` 不重复编 `pending_ramen`。配套把 `policy::remaining_race_slots` 提升为 `pub(crate)`
- **两条上游遗留问题的实证与订正**：新增两个只读诊断测试（规则层未改一行）——`test_person_deck_index_mismatch_full_game` 跑 3 局完整育成，证实 `deck[5].friendship` 恒等于理事长的羁绊而非友人卡本人的；`test_training_buff_index_mismatch` 证实理事长会顶着友人卡的完整训练加成进训练、而友人卡本人的加成为零。据此订正 issues 记录：原「友人卡固有永不解锁」应为「按理事长的羁绊解锁」，原「`deyilv` 拿错得意率」不成立（理事长的 `train_type = -1` 转 `usize` 后进不了该分支），并补上此前漏记的核心后果（`traits.rs` 的 `default_calc_training_buff` 同样按 `< 6` 取卡，直接改变训练数值）；确认 base / onsen 的 `init_persons` 按序推入 6 张卡，该问题是拉面独有的回归。另订正：`current_effect` 的 14 维死特征块实际并未从编码器移除
- **地区选择 build 自适应**：`score_region` 新增 `youqing × at_trains × 卡组 bias` 项，并把 `xunlian` 与 `youqing` 统一按 `bias_sum` 缩放。原实现只算 `xunlian`，而第 2/3 年地区的 `xunlian` 恒为 0、`pt_bonus` 与 `hint_count` 同年恒定，导致同年所有候选同分、argmax 恒取第一个，卡组构成完全不参与决策（原 issue 只记录了第 3 年，实测第 2 年同样失效）；`default_config.toml` 的 `ramen_region_strategy` 由 `fixed` 恢复 `all`，`test_region_build_sensitivity` 由临时验证转为断言测试。实测手写策略基线聚合 +1754 分，其中含「根」的 build 增幅最大（固定值 `[11,14,15]` 的训练位并集恰好漏掉根）

## 2026-08-22
- **基准新增自选比赛达标维度**：`Uma::all_free_races_done` 在任意时点重新比对各区间完成场数（`BaseGame::check_free_race` 只在区间结束回合的下一回合判定，且不达标即终止育成，拿不到全局达标情况）；`bench::GameOutcome` 加 `free_race_ok`，`bench_base` CSV 加同名列并在每局 / 分组 / 总览三处打印达标率。不达标会大量早停拉垮分数分布，基线对比时应先看这一项
- **手写策略自选比赛守门补测试**：补两个用例（不改策略逻辑）——小栗帽 100603 两段区间且第二段限 G1（实测可比赛回合数 7 < 区间长度 12），逐回合扫描触发点而非硬编码回合号，避免随 `race_grades` 常量表调整失效；候选表不含 `Operation::Race` 时（生病 / 体力不足）须返回 `None` 而非越界取下标
- **局面采样器（NN 管线 Phase 2 上半）**：新增 `sampler.rs`，为教师数据制造根局面——第一代采样空间（7 马娘 × 11 张卡池 × 3 种构成，角色冲突由 `chara_id` 实测比对排除）、按工作项序号确定性导出采样任务（卡组分层，截断回合与种子 SplitMix64 分频道派生，分片 / 续跑 / 改并行度均不变）、ε 轨迹扰动、走真实 `run_stage → select_action` 路径截断捕获；`SampleSpec` 自包含扰动参数以保证跨机回放一致，`SampleOutcome::Exhausted` 携带停止回合以区分「截断落在 URA 之后」与「扰动致育成提前失败」。根局面限定在 `RamenSelect / SpecialSelect / Train / RegionSelect` 捕获——第 1 年地区选择由 `run_begin` 内联执行、`stage` 仍是 `Begin`，这类不在阶段入口的决策点会破坏搜索的 `apply_action → next()` 契约。模块文档写明两条使用约定：必须按 index 区间分片、复现基座含 `gamedata` 与 `GameConfig`
- **第三方库引用规范化**：`flat_search.rs` / `sampler.rs` 中 `anyhow::bail!` / `anyhow::ensure!` / `anyhow::anyhow!` 的全名引用改为 `use` 导入后直接调用
- **支援卡类型注释订正**：`SupportCardData::card_type` 原注释写作「5团队6友人」，与 `cardDB.json` 实测相反（30305[友]=5，团队卡=6）
- **搜索层泛型化（NN 管线 Phase 1.4，Phase 1 完成）**：新增 `search/searchable.rs` 的 `FlatSearchGame` trait（关联 rollout 训练员、CRN 阶段编号、`fork_for_rollout` 强制「克隆+重置内部 RNG」不可分割）；`FlatSearch<G>` 与 `SearchOutput<A>` 泛型化并保留默认类型参数，活跃入口零改动；采用「公共内核 + rollout 闭包」而非 trait 钩子，规避泛型 impl 方法解析导致温泉特判静默失效的陷阱；拉面根节点搜索跑通且 1/8/24 线程逐位可复现；拉面 CRN 重测 1.24x→3.73x（略优于温泉 3.65x）
- **搜索层两处缺陷修复**：① NN leaf 微批路径（`simulate_until_terminal_or_leaf`）此前未按阶段重播种，导致 `rollout_batch_size > 1` 时 CRN 开关实际不生效，改为与 `simulate` 一致接收 rollout 种子；② UCB 终止判据由成功样本数改为已计划次数——rollout 稳定失败时成功数永远达不到 `search_n`，会死循环且触及不到「零样本」检查；③ 补 UCB 路径回归（可复现性）与候选顺序敏感性诊断（实测该根局面顺序无关）
- **搜索层真 CRN（NN 管线 Phase 1.3）**：`RolloutSeeds::stage_seed` 支持按 `(回合, 阶段)` 重新派生随机流，`simulate` 改吃 rollout 种子并在每个阶段边界重播种，由 `SearchConfig::crn_stage_reseed` 开关控制（**默认开启**，可经 `[mcts] crn_stage_reseed` 从 toml 关闭）；新增配对相关实测（onsen 7 候选 × 200 rollout）：仅共享起始种子 corr 0.18 / 等效 1.31x，开启按阶段重播种 corr 0.69 / 等效 3.65x，证实朴素共享种子几乎无收益
- **搜索层可复现（NN 管线 Phase 1.1/1.2）**：新增 `search/seeds.rs` 的 `RolloutSeeds`（rollout 种子按序号派生，候选索引不参与，为后续 CRN 留位），移除 `flat_search` 全部 8 处 `from_os_rng`，改为按工作项播种；`simulate_many` 改吃种子表 + 偏移并返回失败计数，UCB 按「已计划次数」记账避免失败导致候选间种子错位；rollout 失败由静默丢弃改为计数告警（全失败才报错），补 `search_group_size > 0` 校验；`flat_search` 新增可复现性回归测试（同种子一致 / 换种子生效 / 候选顺序无关），实测 1~24 线程结果逐位相同
- **`RamenHandwrittenTrainer` 的 breakdown 缓存改 `Mutex`**：上游新增的 `RefCell<Option<String>>` 使其失去 `Sync`，而搜索层 rayon 跨线程共享同一个 rollout 决策器（`FlatSearch<RamenGame>` 因此整体不再 `Sync`，编译失败）；改 `Mutex` 恢复，锁中毒时静默跳过调试文本而非中断育成

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
