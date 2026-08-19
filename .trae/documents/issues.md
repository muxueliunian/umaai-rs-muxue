# UmaAI-RS 问题记录

本文件用于记载较复杂问题（需要用户协助解决的）的解决过程。

## 问题记录模板

```
## [问题标题]
- **日期**：YYYY-MM-DD
- **状态**：待解决 / 解决中 / 已解决
- **问题描述**：简要描述问题现象
- **排查过程**：记录排查步骤和发现
- **解决方案**：记录最终解决方法
- **备注**：其他相关信息
```

---

## basic_effect.hint_special 尚未处理

- **日期**：2026-08-19
- **状态**：已解决
- **问题描述**：`RamenBasicEffect.hint_special`（第三年 `true`）表示"支援卡类型>=4 时，除友人/团队卡以外的所有支援卡都出现 Hint，且训练后发动所有的 Hint 事件"。当前代码在 `effects.rs` 的 `calc_finals_effect` / `calc_normal_effect` 中已经把 `basic.hint_special` 合并到 `RamenTrainingEffect.hint_special` 字段，但是没有下游消费者：`distribute_hint` 仅按 `hint` 加成概率判定单个支援卡是否出现 Hint，没有按 `hint_special` 强制所有非友人/团队支援卡出现 Hint 的逻辑；hint 事件触发逻辑也没有"训练后发动所有 Hint 事件"的处理。
- **排查过程**：
  - `effects.rs:91, 167` 把 `basic.hint_special` 合并到 `effect.hint_special`，但只是布尔字段透传
  - `RamenGame::distribute_hint`（`game.rs:550`）只读取 `calc_hint_bonus_pct()` 的 hint 加成，不读取 `hint_special`
  - `handle_train_success` / hint 事件触发逻辑（`action.rs`）按单卡 hint 触发流程处理，未实现"训练后发动所有 Hint"
- **解决方案**：
  1. `RamenGame::distribute_hint` 在 hint_special 生效时，强制将 `at_trains` 训练位置的所有 PersonType::Card 支援卡的 `is_hint` 设为 true
  2. `RamenGame::calc_hint_special_active` 判断生效条件：吃面 + 第3年 + 支援卡种类>=4
  3. `RamenGame::calc_hint_special_at_trains` 获取当前回合生效的训练位置（地区拉面的 at_trains）
  4. `RamenGame::is_hint_special_active_for_train(train)` 供 `handle_hint_event` 调用
  5. `handle_hint_event` 在 hint_special 路径下：依次触发 hint_persons 中所有 PersonType::Card 的 hint 事件，每个支援卡按各自 `1 + hint_count_bonus` 次触发（保留温泉杯逻辑）
  6. 抽取 `push_hint_event` 辅助函数复用 hint事件生成逻辑（含 `hint_level / total_hints` 上限处理）
  7. 新增 5 个单元测试覆盖：不吃面/年1-2/年3/部分位置/支援卡种类<4
- **备注**：仅第三年生效（`ramen_basic_effect[2].hint_special = true`），生效范围与 hint_special 字段语义一致；每个支援卡的 `hint_count_bonus` 独立判断，按各自的 `1 + bonus` 次触发。

---

## 测试批量运行时的全局状态问题
- **日期**：2026-08-13
- **状态**：已解决
- **问题描述**：单独运行umasim的测试可以通过，但使用`cargo test -p umasim --lib`批量运行时部分测试失败
- **排查过程**：
  - 已确认工作目录问题已通过get_workspace_root()解决
  - `init_logger` 的 TOCTOU 竞争：多个测试同时调用时，`LOGGER.get().is_some()` 检查通过后其他线程抢先 `set()`，后续线程调用 `flexi_logger.start()` 触发 "logger already initialized" 报错
  - 之前的 "LOGGER OnceLock 已 set 失败但 LOGGER 仍为 None" 状态导致 log crate 全局状态被永久污染
- **解决方案**：
  - `utils.rs` 新增 `LOGGER_INIT_DONE: AtomicBool` 记录 log crate 是否已成功 start（不依赖 LOGGER 是否已 set）
  - 新增 `INIT_LOCK: OnceLock<Mutex<()>>` 串行化整个初始化过程（持锁 + 双重检查）
  - 快速路径：已初始化过直接 return Ok
  - 慢速路径：持锁后双重检查，避免 log crate 被多次初始化
  - `disable_log` / `enable_log` 增加 `if let Some(LOGGER.get())` 保护，LOGGER 未初始化时安全 return（不 panic）
- **备注**：所有 110 个测试在 release 模式下 3 次连续运行均稳定通过

---

## 诀窍槽基础值分配算法与文档数据不完全一致
- **日期**：2026-08-16
- **状态**：存疑
- **问题描述**：`calc_gauge_base_distribution` 实现了 floor + 消耗=1固定 + 最小已分配优先补足的算法，与 ramen_memo 中"使用新友人"的11个算例对比，降序值在大部分案例匹配，但部分案例因实际 region_feeling 配方与文档记录的消耗值不同（A/B位置对调），导致排序位置不一致。
- **排查过程**：
  - 初始使用 round 四舍五入，因浮点精度问题导致所有结果相同（fallback值）
  - 改用 floor + 最大余数优先，余数相同时结果不稳定
  - 改用 floor + 最小已分配优先补足，余数相同时按配方消耗量降序，大部分匹配
  - 引入消耗=1固定分配1的特殊规则后，所有11个算例的降序值均匹配
  - 但部分案例（如札幌中京小倉、札幌中京京都）的实际配方与文档记录不同，可能是文档数据不准确或 region_feeling 数据有误
- **解决方案**：算法逻辑已基本正确，待用户准备新的实测数据进行验证
- **备注**：ramen_memo 中的算例数据可能与实际 JSON 数据存在差异，需要以实际游戏数据为准

---

## 隐藏风味替换机制设计
- **日期**：2026-08-16
- **状态**：已解决
- **问题描述**：做面时隐藏风味的替换目标应该由玩家手动选择，而不是自动替换消耗最多的类型
- **排查过程**：
  - 最初实现为自动从消耗最高的类型开始替换
  - 用户指出应该支持手动选择替换目标
- **解决方案**：修改 `consume_for_ramen` 和 `can_make_ramen` 函数，接受 `special_targets: &[i32; 3]` 参数，允许指定每种类型替换几个
- **备注**：这使得做面决策更加灵活，但也增加了动作空间

---

## 动作空间压缩设计
- **日期**：2026-08-16
- **状态**：已解决
- **问题描述**：组合动作（吃面+操作）导致动作空间过大，需要压缩
- **排查过程**：
  - 最初实现为所有组合（吃面×操作），动作数量约40个
  - 尝试限制吃面只能与训练/比赛组合，减少到约20个
  - 尝试合并少见动作为"特殊吃面组合"，进一步减少到约19个
- **解决方案**：采用分离决策模型
  - 阶段1：选择吃面决策（不吃面/吃面X/Y/Z）
  - 阶段2：选择基础操作（所有操作都可以）
  - 搜索空间从4×10=40降到4+10=14
- **备注**：RamenAction结构保持不变，只是搜索策略分阶段进行

---

## 拉面杯 1d 最小闭环实现中的待确认问题
- **日期**：2026-08-16
- **状态**：大部分已解决，部分待后续处理
- **问题描述**：实现拉面杯完整回合流程（1d）过程中，发现以下需要后续处理的问题
- **排查过程**：
  1. **隐藏风味回合不一致**：`events.rs` 和 `rules.rs` 的 `get_turn_special_feeling` 返回值不同。已修复为与 rules.rs 一致（2/24/36/48/60 获得2个，37-39/61-63 获得1个）。✅ 已解决
  2. **train_feeling_type 类型转换**：`assign_train_feeling_type` 返回 `[i32; 5]`，但 `RamenState.train_feeling_type` 字段类型为 `Option<[FeelingType; 5]>`，需要显式转换。✅ 已解决
  3. **ActionValue 新增字段**：`ActionValue` 包含 `friendship`、`hint_level`、`max_vital` 字段，构造时需要 `..Default::default()`。✅ 已解决
  4. **超级拉面自动效果未完整实现**：回合 72-77 的超级拉面效果（vital/motivation/saihou 恢复）目前仅记录日志，未实际应用 `finals_effect.base` 的 vital 和 motivation 效果。⏳ 待后续处理
  5. **分身系统未实现**：地区拉面分身（id >= 5）和超级拉面分身（每个支援卡额外出现一次）的逻辑尚未实现，后续需要在 distribute_all 中集成。⏳ 待后续处理
  6. **特殊吃面决策**：当前 `do_train` 中吃面使用默认 `special_targets = [0,0,0]`（不使用隐藏风味）。✅ 已解决（定案见「拉面杯动作阶段扩展：隐藏风味决策」）
  7. **年初选面策略**：当前使用 `fixed_region_selection` 固定顺序，需要在未来接入搜索/RL 策略时替换。✅ 已解决（改为通过Trainer统一接口select_action决策）
  8. **随机事件复用**：当前 `generate_events` 复用 BasicGame 的随机事件（支援卡事件、马娘事件等），这些事件是否适用于拉面杯需要确认。⏳ 待确认
  9. **回合编号问题**：人头动态添加的回合编号需要从0开始计算（turn==2 为第2回合，turn==12 为第12回合）。✅ 已修复
- **解决方案**：
  - 问题 1-3、9 已修复
  - 问题 4-8 作为后续任务记录
- **备注**：完整流程已跑通（回合 0-77），但部分效果（分身、超级拉面完整效果）尚未实现

---

## RMJ成功/失败效果是否正确生效
- **日期**：2026-08-17
- **状态**：待确认
- **问题描述**：RMJ结算后，成功/失败效果（ramen_success_effect / ramen_fail_effect）是否正确应用到后续回合的训练计算中，需要确认
- **排查过程**：
  - 代码中 `calc_normal_effect` 在 `year_idx >= 1` 时读取 `rmj_results` 来决定使用 success 或 fail 效果
  - `check_rmj` 函数在 NextTurn 阶段执行，结果写入 `ramen.rmj_results`
  - `distribute_hint` 重写中也使用了 `calc_hint_bonus_pct` 来应用 RMJ hint 加成
  - 但未实际验证 RMJ 效果（youqing/deyilv/hint）在训练数值计算中是否正确体现
- **解决方案**：待实测验证，建议在测试中打印 RMJ 结算前后的效果对比
- **备注**：涉及 ramen_success_effect 的 youqing/deyilv/hint 三个字段

---

## 第三年地区选择组合过多
- **日期**：2026-08-17
- **状态**：待解决
- **问题描述**：第3年可选地区为10-19共10个，C(10,3)=120种组合，动作空间过大，影响搜索效率
- **排查过程**：
  - 第1/2年各5个地区，C(5,3)=10种组合，可接受
  - 第3年120种组合导致 Trainer 需要评估120个动作，计算开销显著增加
- **解决方案**：待讨论，可能的方向：
  1. 基于当前卡组和诀窍库存，预筛选出合理的候选组合（如基于配方消耗匹配度）
  2. 分两步决策：先选主方向（偏速/偏耐/偏力等），再在子集中选具体组合
  3. 使用评估函数对120个组合快速打分排序，只让Trainer从top-K中选择
- **备注**：第3年地区还包含pt_bonus效果，选择策略需要同时考虑youqing/pt_bonus和配方匹配

---

## 训练数值不对，尤其是友情加成
- **日期**：2026-08-17
- **状态**：已解决
- **问题描述**：训练数值计算可能不正确，特别是友情加成（youqing）的生效条件
- **排查过程**：
  - 闪耀判定逻辑：支援卡只能在本体的得意训练位置闪耀（train_type == train && friendship >= 80）
  - 分身在非本体训练位置时不闪耀，友情加成不生效
  - `is_shining_at()` 函数已重写，但可能仍有逻辑问题
  - `calc_training_buff()` 中，非闪耀时 effect.youqing = 0
  - `calc_ramen_training_effect()` 中，非闪耀时 effect.youqing = 0
  - 旧实现：`RamenGame::calc_training_value` 仅调用 `default_calc_training_value`（卡 buff），然后由 `apply_ramen_to_train_value` 用 `apply_ramen_training_value` 简版公式（`lower * xunlian * youqing`，缺干劲/人数/成长率）追加拉面 buff；同时 `explain_distribution` 也用简版公式，导致显示数值与实际生效数值不一致
- **解决方案**：
  - 重写 `RamenGame::calc_training_value` 为两阶段实现（模仿 OnsenGame）：
    - 阶段1：`default_calc_training_value` 应用卡 buff 后约束 status_pt[i] ≤ 100
    - 阶段2：拉面 buff 累乘到下层值上
      - `xunlian × youqing` 对 `status_pt[0..4]`（**5 个属性**，含副属性加成）都生效
      - `pt_bonus` 仅对 `status_pt[5]`（PT）生效
      - upper 上限 = 100 + status_limit（PT: 100 + status_limit + pt_limit）
  - `explain_distribution` 直接使用 `calc_training_value` 的结果，删除简版公式调用
  - `handle_train_success` 直接使用 `calc_training_value` 的结果
  - 删除 `apply_ramen_to_train_value` 函数（已无人调用）
- **备注**：
  - 测试 `test_train_param_decomposition` 已包含 3 张速卡 + 2 个 NPC 在速训练、不吃面/吃面 Some(5) 的场景验证
  - 完整77 回合 `test_ramen_silent_loop` 通过
  - 核心修正：副属性（来自 `buff.bonus`）也享受拉面 buff 加成，这之前被忽略

---

## 超级拉面得意率人物分配错误
- **日期**：2026-08-17
- **状态**：待解决
- **问题描述**：超级拉面分身分配时，得意率（deyilv）的计算可能不正确
- **排查过程**：
  - 当前使用随机选择训练位置，分配失败则重试（最多 option_trains.len() * 2 次）
  - 原始设计是按得意率权重分配，但实现时改为随机
  - `distribute_super_ramen_clones()` 函数需要检查是否应该按得意率权重分配
- **解决方案**：待确认正确的分配算法，可能需要恢复按得意率权重分配
- **备注**：得意率影响支援卡出现在彩圈（友情训练）位置的概率

---

## distribute_person 中"不出现"判定受得意率影响

- **日期**：2026-08-19
- **状态**：待解决
- **问题描述**：当前 `Game::distribute_person`（`traits.rs:192`）将"不出现"判定和"训练位置分配"混在一起，不在率 = `absent_rate / (500 + absent_rate + deyilv)`，导致得意率会影响"不出现"概率。按剧本原始规则，"不出现"概率应不受得意率影响，得意率只影响训练位置的权重分配。
- **排查过程**：
  - 用户给出剧本原始算法：
    1. 用基础权重 [100,100,100,100,100,absent_rate] 判定"不出现"，概率 = `absent_rate / (500 + absent_rate)`（**不含得意率**）
    2. 判定为出现后，按 [100+deyilv, 100, 100, 100, 100]（不含"不出现"项）随机分配训练位置
  - 当前算法：
    - 不在率 = `absent_rate / (500 + absent_rate + deyilv)`（含得意率）
    - 训练位置按 [100+deyilv, 100, 100, 100, 100, absent_rate]（含"不出现"项）分配
  - 关键差异：得意率会拉高"不出现"判定概率（deyilv 越大，不出现概率越低）——这是错误的
- **解决方案**：将 `distribute_person` 改为两步算法：
  1. 先用基础权重（不含 deyilv）判定"是否不出现"
  2. 判定为出现后，按训练位置权重（含 deyilv）随机分配训练位置
- **备注**：用户本次确认暂不动 absent_rate 相关逻辑（涉及 `absent_rate_drop` 等其他领域知识，留待后续）。本次只修复 RamenGame::deyilv 缺剧本加成的问题。`distribute_person` 修正留待后续 issue。

---

## 夏合宿期间诀窍槽加成未实现
- **日期**：2026-08-17
- **状态**：已解决
- **问题描述**：夏合宿期间（回合36-39和60-63），全部诀窍槽必定为+7，但当前未实现。同时，夏合宿期间不允许普通和友人外出。
- **排查过程**：
  - ramen_memo_cn.md 中记载：夏合宿期间全部训练等级为5，不会发生支援卡事件或掉心情事件
  - 但未记载诀窍槽+7的规则
  - 用户指出夏合宿期间全部诀窍槽必定为+7
  - 当前 `fill_feeling_gauge()` 函数未处理夏合宿的特殊规则
- **解决方案**：
  - `rules.rs` 新增私有 `fill_gauge_xiahesu_max`：三种槽各自补到 GAUGE_LIMIT，溢出自动 +1 诀窍（带新友人时）
  - `fill_gauge_after_train` 增加 `is_xiahesu` 参数，夏合宿时走全 MAX 路径
  - 新增 `fill_gauge_after_non_train`：比赛/休息/外出/友人出行的统一入口
  - `RamenAction::apply` 的非训练分支在 base action apply 后调用 fill_gauge_non_train
  - 休息 + 夏合宿自动清除 ill 和 bad_trainer flag（等同 Clinic 效果），因此夏合宿禁用 Clinic
  - `list_operations` 增加 `is_xiahesu` 参数：夏合宿时不返回 NormalOuting/FriendOuting/Clinic
  - `list_train_actions`/`list_all_actions`/`list_actions` 同步透传 is_xiahesu
- **备注**：
  - 日文原始规则见 `ramen_memo.md:133-135`「全習得ゲージMAX」（带新友人）
  - 治病的 apply 按用户确认不调用 fill_gauge_after_non_train
  - 已加 5 个单元测试覆盖全 MAX / 正常路径；端到端 77 回合流程跑通

---

## Ubuntu 下 umaai 二进制图标方案待定
- **日期**：2026-08-17
- **状态**：待解决（暂缓）
- **问题描述**：umaai 的 Windows 版通过 build.rs + winscribe 把 `.ico` 图标嵌入二进制；Ubuntu 下 ELF 二进制没有 Windows 资源节的图标嵌入机制，需要确定 Linux 版的图标落地方式
- **排查过程**：
  - Linux ELF 无原生图标资源节，无法直接嵌入 `.ico`
  - 候选方案一：`.desktop` 文件 + 外置 PNG/SVG 图标（桌面菜单展示，Linux 惯例）
  - 候选方案二：`include_bytes!` 把 PNG 数据嵌入二进制（自包含、单文件分发，需补充 PNG 资源）
  - 已顺带修复 umaai 的 Linux 构建：winscribe 改为仅在 `cfg(windows)` 目标下作为 build-dependency；build.rs 的 Windows 资源编译逻辑用 `#[cfg(windows)]` 包裹；`windows` crate 及其依赖链（windows-future 0.3.2 与 windows-core 0.62.2 不兼容，Linux 上编译失败）改为 `cfg(windows)` 限定依赖；补声明 Linux 专用的 `libc` 依赖；Linux 版 `get_stack_size` 补 `pub` 修饰
- **解决方案**：暂不处理，待用户明确图标使用场景（桌面菜单展示或二进制自包含）后再实施
- **备注**：umaai 为 CLI + TUI 程序（clap + ratatui），桌面图标应用场景有限

---

## 拉面杯动作阶段扩展：隐藏风味决策

- **日期**：2026-08-17
- **状态**：已定案（阶段机三阶段）
- **问题描述**：`apply_ramen`（`action.rs:175`）硬编码 `special_targets = [0,0,0]`，Trainer 无权决策隐藏诀窍用法；`list_all_actions`（`action.rs:824`）也没有该维度。`get_available_ramens`（`action.rs:847`）用 `can_make_ramen(state, recipe, &[0,0,0])` 过滤，把"用隐藏风味才做得出的面"也屏蔽了（如库存 A=5 B=0 C=0、recipe=[2,2,1]，用 1 个隐藏风味替代 B 本可做面）。函数层已就绪（`consume_for_ramen`/`can_make_ramen` 已接受 `special_targets` 参数），缺的是上层决策传递。
- **决策域规模（实测）**：targets 合法域为 `0 ≤ t[i] ≤ recipe[i]`、`sum(t) ≤ min(2, 隐藏风味库存)`；最小必要替换 `min_needed[i] = max(0, recipe[i] - stock[i])`，`sum(min_needed) > 2` 时该面不可做。库存紧张时每面可行方案仅 1~6 种，全富余时达 9~10 种/面（recipe=[2,2,1] 全富余正好 9 种）；3 面全富余（如库存 3+3+4、隐藏风味 4）时合并枚举峰值约 280~310 动作
- **方案对比**：
  - **笛卡尔积枚举**：`ramen × targets × operations` 三维组合，单回合 368~466 动作，涨一个数量级，否决
  - **默认 `[0,0,0]` + 扩展接口**：改动小但语义不完整，普通 Trainer 看不到非默认选项，否决
  - **改 `Trainer` trait 拆三接口**：搜索空间最自然，但要重写所有 Trainer 实现，且 `list_actions` 被 onsen/basic 共用不能破坏，否决
  - **聚合"用 0/1/2 个"再深入**：第二阶段仍需决策主体，在单次 `select_action` 接口下只能退化为规则默认（违背显式决策目标）或隐式变成更长阶段链，否决
  - **定案：阶段机三阶段** —— `RamenStage` 拆为 `RamenSelect`（选面）→ `SpecialSelect`（选诀窍用法）→ `Train`（选训练），不动 `Trainer` trait。理由：该维度规模波动大（1~6 ↔ 9~10 种）且"省哪类诀窍"有战略含义（影响后续做面），峰值只有树形分层能消化；与 `RegionSelect` 阶段同模式，MCTS/`run_stage` 循环天然适配
- **落地要点**：`RamenGame` 增加 pending（`pending_ramen`、`pending_special_targets`），`apply_ramen` 改从 pending 读 targets；新增 `list_special_targets_for(state, ramen_idx) -> Vec<[i32; 3]>`（从 `min_needed` 出发、剩余预算内分配），`get_available_ramens` 改为"候选非空即可选"；`SpecialSelect` 候选含 `[0,0,0]`（吃面但不省诀窍）与"改为不吃面"选项，保证完备；选"不吃面"时短路跳过该阶段
- **备注**：与第 84 行「特殊吃面决策」子条目呼应，两条目已同步标注已解决

---

## 拉面杯合并决策接口（两阶段聚合）

- **日期**：2026-08-18
- **状态**：已定案（方案 E，待落地）
- **问题描述**：未来在线搜索（umaai）的交互只需"选择拉面前 / 选择训练前"两个状态——玩家一次给出"选面+吃法"决策，再给训练决策。但当前三阶段阶段机（`RamenSelect`→`SpecialSelect`→`Train`）要求 Trainer 在 RamenSelect 阶段只决策 ramen，SpecialSelect 阶段才决策 targets。这与"玩家决策不可拆分"的真实交互不一致，MctsTrainer 无法在不打破决策分布的前提下做完整搜索。
- **Trainer 粒度偏好分析**：
  - RandomTrainer：随机，无所谓
  - HandwrittenTrainer / ManualTrainer：**倾向三阶段**——策略代码易表达"先选面→再选吃法→最后选训练"的固定流程；候选少、决策聚焦
  - MctsTrainer + umaai：**必须两阶段聚合**——"选面+选吃法"是玩家不可拆分的一次决策；搜索必须按真实决策分布（同时考虑 ramen 和 targets）展开
  - 结论：**粒度选择权交给 Trainer**，不是 Game 硬编码
- **方案对比**：
  - **A：Game 加模式开关（`combine_ramen_decision: bool`）**：Game list_actions 在开关打开时返回聚合候选。缺点是 Game 接口被开关污染，三/两阶段路径在 list_actions / apply / next 多处并存
  - **D：Trainer 内连续两次决策**（Game 零改动）：MctsTrainer 在 RamenSelect 阶段手动循环两次（apply ramen → next → apply targets → next）。缺点是每个 (ramen, targets) 组合评估需要 2 次 apply + 2 次 list_actions + 2 次 next，MCTS 深度搜索场景浪费明显
  - **E：Game 加"合并决策"接口（推荐）**：在 RamenGame 上新增两个方法 `list_combined_ramen_select_actions` 和 `apply_combined_ramen_decision`，Trainer 自主选择走三阶段路径或合并路径。next() 仍负责推进，RamenState 加 `combined_decision: bool` 标记让 next() 在 RamenSelect 阶段跳过 SpecialSelect
  - **F：阶段机加 CombinedRamenSelect 变体**：阶段枚举膨胀，三/两阶段代码路径完全分离
- **定案：方案 E**。理由：
  1. Game 接口最小扩展（仅 RamenGame 加 2 方法，不动 Game trait，避免污染通用接口）
  2. Trainer 粒度自由：RandomTrainer / HandwrittenTrainer 走三阶段原路径不变；MctsTrainer 走合并路径一次 apply 评估一个聚合组合
  3. 阶段机保持纯粹：仍是 RamenSelect / SpecialSelect / Train 三阶段，只是允许 Trainer 走"快捷路径"
  4. MCTS 搜索效率高：聚合组合只需 1 次 apply
- **落地要点**：
  1. `RamenAction` 新增 `combined_select(ramen_idx, targets)` 构造方法；约定 `ramen = None` 时强制 `targets = [0,0,0]`（即"不吃面"等价于"吃面 + targets=[0,0,0]"在合并决策视角下）
  2. `RamenGame`（仅 `RamenGame`，不动 `Game` trait）新增两个方法：
     - `pub fn list_combined_ramen_select_actions(&self) -> Vec<RamenAction>`：返回 RamenSelect × SpecialSelect 笛卡尔积的聚合候选；不吃面 + 每个面的所有合法 targets
     - `pub fn apply_combined_ramen_decision(&mut self, ramen: Option<usize>, targets: [i32; 3]) -> Result<()>`：写 `pending_ramen` + `pending_special_targets`，设置 `combined_decision = true`，**不直接设 stage**（保留 next() 推进语义，避免后续 next 混乱）
  3. `RamenState` 新增字段 `combined_decision: bool`，并在 `clear_pending()` 中一并清空
  4. `Game::next()` 在 RamenSelect 阶段检查 `combined_decision`：若 true 则直接推到 Train（不进入 SpecialSelect）；否则按现有 `pending_ramen` 逻辑
  5. Trainer trait 不变；MctsTrainer 在实现合并决策时调用上述两个新方法
  6. 现有 RamenSelect / SpecialSelect 路径（list_ramen_select_actions、list_special_select_actions、apply 中的 pending 写入）保持不变，三阶段行为完全向后兼容
- **MctsTrainer 搜索策略（仅记录，不实现）**：聚合候选生成后做 top-N 截断（`top_n: usize`，默认 3-5）+ adaptive UCB：
  - 第一阶段（探查）：所有候选均匀 rollout K 次获得初始均值
  - 第二阶段（深入）：只对 top-K' 候选用 PUCT 选择做深度搜索
  - 配置项：`exploration_rounds`、`exploitation_top_k`、`cpuct`
  - 实现细节留待"MctsTrainer 实现"阶段
- **备注**：方案 E 的两个新方法仅放在 RamenGame（具体类），不放在 Game trait，避免污染通用接口（与 `list_ramen_select_actions` / `list_special_select_actions` 同样仅 RamenGame 调用保持一致）
