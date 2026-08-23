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

## 人头下标与卡组槽位的对应关系在拉面剧本被打破（`person_index < 6` 守卫全线失效）

- **日期**：2026-08-23
- **状态**：已解决（2026-08-23，上游作者已授权我们修）
- **问题描述**：base / onsen 的代码用 `person_index < 6` 作为「这个人头是卡组里的支援卡」的判据，隐含假设 `persons[0..6)` 与 `deck[0..6)` 一一对应。拉面剧本的 `init_persons` 打破了这个假设，但所有 `< 6` 守卫都原样保留，导致理事长被当成卡组第 6 张卡、友人卡被当成「不是卡」。
- **排查过程**：
  - 实际人头布局（`game/ramen/game.rs:48-61`、`game/ramen/state.rs:269-282`）：
    | 下标 | 是谁 | 何时加入 |
    |---|---|---|
    | 0-4 | `card_type < 5` 的 5 张训练卡 | 开局 `init_persons` |
    | **5** | **理事长** | 开局 `init_persons` 末尾 |
    | **6** | **友人卡**（`card_type >= 5`） | 回合 2 `add_friend_and_npcs` |
    | 7-11 | 5 个 NPC | 回合 2 |
    | 12 | 记者 | 回合 12 |
  - 而卡组 `deck[5]` 正是友人卡（`sampler.rs` 与 `bench_config.toml` 的 build 均把友人放末位）
  - 原设计意图的证据：`BasePerson::yayoi()`（`game/base/person.rs:55-65`）把 `person_index` 硬编码成 **6**。这个值随后被 `add_person`（`state.rs:304`）覆盖成实际下标，但它表明原布局是「0-5 六张卡 + 6 理事长」——正是 `< 6` 守卫成立的前提。拉面把友人卡延迟到回合 2 才加，5 张训练卡只占 0-4，理事长顺位落到 5。
  - 受影响的 `< 6` 守卫：
    - `game/ramen/state.rs:314-316` `add_friendship`：羁绊只在 `person_index < 6` 时回写 `deck`
    - `game/ramen/game.rs:469-480` `deyilv`：`person_index < 6` 时读 `deck[person_index]`，否则返回 0.0
    - `game/ramen/action.rs:560-562`：`pidx < 6` 时用 `deck[pidx]` 重算卡效果
  - **实测确认（2026-08-23，`game/ramen/game.rs` 新增两个只读诊断测试）**：
    - `test_person_deck_index_mismatch_full_game`：跑 3 局完整拉面杯，局末 `deck[5].friendship` 恒等于 `persons[5]`（理事长）的羁绊（64 / 100 / 96），与 `persons[6]`（友人卡本人，恒为 100）分叉。开局 `deck[5].friendship = 30`（`initialJiBan`）会被理事长从 0 起算的计数覆盖掉。前 5 张训练卡的两份羁绊始终一致，作为对照。
    - `test_training_buff_index_mismatch`：只把理事长（人头 5）放进训练，`calc_training_buff` 返回的是 **`deck[5]`（友人卡）的完整卡效果**（xunlian=20、saihou=5、fail_rate_drop=15、vital_cost_drop=30、event_effect_up=30、event_recovery_amount_up=60）；只把友人卡（人头 6）放进训练，返回**全零**。
- **后果一（修正：不是「永不解锁」，而是「按理事长的羁绊解锁」）**：`SupportCard::calc_training_effect`（`game/support_card.rs:270-282`）判定固有用的是 `self.friendship`，即卡组那份拷贝，而这份拷贝被理事长的羁绊回写。实测把 `deck[5].friendship` 设为 60 时，理事长所在训练位的 buff 多出 `bonus[1] += 1`、`bonus[5] += 1`（即 `30305 [友]骏川手纲` 的 `uniqueEffectParam = [101, 60, 4, 1, 30, 1]`）；设为 59 时不触发。也就是说友人卡的固有**确实会生效，但阈值读的是理事长的羁绊，且加成落在理事长的人头上**。
- **后果二（核心，此前未列出）**：`traits.rs:335-360` 的 `default_calc_training_buff` 用 `*index >= 0 && *index < 6` 把**人头下标当卡组下标**，拉面未重写该方法（全仓库只有 `traits.rs:335` 一处 `fn calc_training_buff`）。于是：
  - 理事长出现在训练里时，会顶着**友人卡的全部训练加成**参与计算（含 youqing / 失败率下降 / 体力消耗下降）
  - 友人卡本人（人头 6）参与训练时，**训练加成完全为零**
  - 这条直接改变模拟数值，影响面远大于羁绊串位本身
- **后果三（原「后果二」的一半，已排除）**：理事长**不会**走进 `deyilv()`。`traits.rs:206` 是 `let train_type = person.train_type() as usize;`，`train_type()` 返回 `i32`（`base/person.rs:87`），理事长的 `-1` 转 `usize` 后是 `usize::MAX`，`traits.rs:219` 的 `train_type <= 4` 不成立。故「`deyilv(5)` 拿友人卡得意率当理事长的」不成立。
- **影响范围确认**：base（`base/basic.rs:235-247`）与 onsen（`onsen/game.rs:1329-1351`）的 `init_persons` 都把 `deck` 的 6 张卡**按序全部**推入 `persons`，再追加理事长，`< 6` 判据在这两个剧本成立。**本问题是拉面独有的回归，不是三剧本通病。**
- **既有测试把错误行为固化了**：`game/ramen/game.rs:2427` 断言 `deyilv(6) == 0.0` 并注释「person_index >= 6 返回 0」，但下标 6 正是友人卡。需上游确认这是有意还是当初未意识到 6 号是谁。
- **次要影响**：`action.rs:720/734/753` 的 hint 路径同样按 `< 6` 取卡——友人卡（人头 6）拿不到 `hint_count_bonus`，且在 `push_hint_event` 里被当作非支援卡处理（hint_level 固定 1、不出技能提示）。理事长 `is_hint = false`，不进 hint 路径，故这一侧无串位。
- **解决方案**：按 `card_id` 反查（经 GPT-5.6 codex 与 Grok 4.6 两方独立评审，方案一致）。
  1. `Person` trait 新增 `card_id()`，`Game` trait 新增 provided 方法 `deck_index_of(person_index) -> Option<usize>`：由人头的 `card_id` 在 `deck` 里 `position()` 反查槽位，无卡人头（理事长 / 记者 / NPC）返回 `None`
  2. 全部改为反查的调用点：`traits.rs` 的 `default_calc_training_buff`（核心数值路径）与 `distribute_hint`、`ramen/state.rs` 的 `add_friendship`、`ramen/game.rs` 的 `deyilv` 与 `distribute_hint` override、`ramen/action.rs` 的 `handle_hint_event` / `push_hint_event`
  3. **两个下标严格分开**：`is_shining_at` / `distribution` / `EventData` 一律用人头下标，只有 `deck[..]` 用反查出的卡组下标
  4. `persons_mut()` 循环内不能调 `deck_index_of`（借用冲突），改为循环外预抽 `(card_id, hint_prob_increase)`、循环内用 `Person::card_id()` 匹配
  5. `deyilv` 的负数人头下标（`-1`）此前会算出 `deck[usize::MAX]` 而 panic，改用 `usize::try_from` + `Option` 后安全返回 0
- **未采纳的方案**：
  - 给 `BasePerson` 加 `deck_index` 字段：第二份真理，会与 `card_id` 漂移，且要动公开结构体的 `Serialize` / `PartialEq` 与全部构造点
  - 给友人卡预留下标 5：只在「友人恒为 `deck[5]`」时成立，而 `RamenGame::newgame` 只检查卡组**含有**友人卡、不检查位置
  - 拉面 `init_persons` 改成 6 张全入 + `person_is_available` 门控：能让 `< 6` 重新凑效，但地雷仍在，以后再延迟插入任何卡就会复发
- **验证**：两个诊断测试改造为回归测试（`test_person_deck_index_mapping_full_game` / `test_training_buff_person_deck_mapping`）。修复后 3 局完整育成中 `deck[5].friendship` 恒等于 `persons[6]`（友人本人）而非 `persons[5]`（理事长）；只放理事长的训练 buff 全零、只放友人的 buff 等于 `deck[5]` 的卡效果。全量 `cargo test --release` 由 227 passed 增至 228 passed（新增编码器交叉反查回归），0 failed。
- **备注**：
  - `game/base/basic.rs:218` 与 `game/onsen/game.rs:161` 有同样的 `< 6` 回写模式，但这两个剧本的 `persons[0..6)` 与 `deck` 确实同序，不受影响（见上「影响范围确认」）。**本次未改动这两处**——它们当前行为正确，改动属于防回归而非修 bug，为控制上游 diff 面暂缓。
  - 本条由 NN 特征编码器（`game/ramen/features.rs`）的评审发现——编码器原本也照抄了「人头下标与卡组同序」的假设。**订正：编码器侧此前并未真正修正**（`ef99478` 的 changelog 与本文件都误记为已修），本次连同规则层一并改为 `card_id` 反查。
  - `deck_index_of` 的前置条件是 `deck` 内 `card_id` 唯一。`SupportCard::new` 取 `idrank / 10` 作 `card_id`，同一张卡的不同突破（如 `302751` / `302754`）`card_id` 相同；现有构造路径均不去重，重复时会静默命中第一张。默认配置 / sampler / bench 不触发，已在 `deck_index_of` 的文档注释里写明，见下条独立记录。
  - **数值影响**：本修复改变拉面模拟结果，不是静默重构。效果从「理事长出现的位置」搬到「友人出现的位置」，友人固有的解锁时序也随之改变（阈值由理事长羁绊改为友人本人羁绊），训练选择 / 失败 / 体力 / 事件轨迹整体分叉。bench 基线、sampler 根局面、按旧编码器落盘的教师数据全部作废，须重跑。

---

## 拉面人头布局派生的其余上游问题（本次未修，待与上游确认）

- **日期**：2026-08-23
- **状态**：一、二已解决（2026-08-24），三 / 四 / 五待解决（**上游遗留，未与上游确认**）
- **问题描述**：修「人头下标与卡组槽位错位」时，两方评审（GPT-5.6 codex / Grok 4.6）另找出五处同源或相邻的问题。与上游确认的授权范围只覆盖「人头下标当卡组下标」与「`current_effect` 死字段」两条，故这五处**本次一行未改**，在此完整记录，待上游确认后另开。全部已本地 grep 复核属实。
- **一、训练人数加成硬编码「理事长 = 6、记者 = 7」（独立的数值 bug，影响最大）** —— **已修（2026-08-24）**：抽出 `Game::count_training_persons` 改按 `PersonType` 判定，负数与越界下标一并不计；温泉 / base 逐位不变（`test_count_training_persons_onsen_unchanged` 守门），`onsen/game.rs` 的挖土处按上游意愿未动。
  - `game/traits.rs` 的 `default_calc_training_value`：`.filter(|p| **p != 6 && **p != 7).count()`，注释写「包括 NPC 和分身，排除掉理事长和记者」。
  - base / onsen 布局下正确，拉面布局下四项全错：

    | 人头下标 | 是谁 | 应该 | 当前 |
    |---|---|---|---|
    | 5 | 理事长 | 排除 | **计入** |
    | 6 | 友人卡 | 计入 | **排除** |
    | 7 | NPC | 计入 | **排除** |
    | 12 | 记者 | 排除 | **计入** |

  - 拉面的 `calc_training_value` 直接调这个默认实现，人数直接进 `1 + 0.05 × 人数`，误差在数个百分点量级。
  - 修法：改成按 `PersonType` 排除 `Yayoi | Reporter`。`game/onsen/game.rs` 的挖土处有同样的 `*p != 6 && *p != 7`，onsen 布局下碰巧正确，建议一并改成类型判断。（`ramen/action.rs` 的诀窍槽已修过同样的问题，训练人数加成漏了。）
  - **注意**：上游作者原话「理事长记者位置需要反查」很可能指的正是这一条——`记者` 在「人头下标当卡组下标」里根本不参与（记者是人头 12、无卡，反查恒为 `None`），唯一同时涉及理事长与记者的就是这个过滤器。下次沟通需要澄清。
- **二、超级拉面分身漏掉友人卡** —— **已修（2026-08-24）**：候选遍历改 `0..persons.len()`；同时给分身补上「每训练一个友人」约束（见下文独立条目）。地区分身按其「不含友人卡」语义**未动**。
  - `game/ramen/action.rs` 的 `card_indices: Vec<i32> = (0..6i32)`：注释写「获取所有支援卡索引（含友人卡，index 0-5）」，但 `0..6` 里没有下标 6，**友人卡永远不参与超级拉面分身**。
  - `game/ramen/game.rs` 的地区分身同样是 `(0..6i32)`，但它只取 `PersonType::Card`：友人在卡组末位时 0-4 恰为训练卡、5 是理事长被类型过滤掉，**碰巧正确**；友人不在末位时会漏卡。
  - 修法：两处都改成遍历 `0..persons.len()` 再按 `PersonType` 过滤。
- **三、Hint 路径的无守卫 `deck[..]` 访问（潜伏越界）**
  - `game/base/basic.rs` 与 `game/onsen/game.rs` 的 hint 路径仍有 `deck[person_index]` 形式的无守卫访问，`deck` 长度恒 6，人头下标 ≥ 6 会直接越界 panic。
  - 拉面侧本次已随「人头下标」修复一并收紧（改反查 + 无卡人头取人头自己的名字）；base / onsen 两处未动。
  - 生产路径上 `distribute_hint` 只给 `PersonType::Card` 打 hint，目前踩不到，属潜伏项。另 `handle_hint_event` 的随机分支没有像 hint_special 分支那样过滤 `PersonType::Card`。
- **四、`BaseGame::new` 不校验卡组内 `card_id` 重复**
  - `SupportCard::new` 取 `idrank / 10` 作 `card_id`，同一张卡的不同突破（如 `302751` / `302754`）`card_id` 相同。
  - 现有构造路径一处都不去重：`BaseGame::new` 无校验、`validate_game_config` 只查 `len() == 6`、`RamenGame::newgame` 只查有无友人卡。
  - 本次新增的 `Game::deck_index_of` 依赖 `card_id` 唯一，重复时会静默命中第一张。默认配置 / sampler / bench 实际不会重复，任何手写卡组都能触发。
  - 本次只在 `deck_index_of` 的文档注释里写明前置条件，**未给上游的 `BaseGame::new` 加 `bail!`**（那是给上游加校验，超出授权范围）。
- **五、`ramen/action.rs` 注释掉的调试块仍含旧假设**
  - `/* */` 块里仍有 `if pidx < 6 { game.deck[pidx] }`。目前是死代码，但解开注释会把友人卡打成「非支援卡」。
- **其他待办（非 bug，属清理）**：
  - `game/ramen/rng_consistency.rs` 的 `run_turns` 用 `for i in 0..6` 同时锁 `deck[i]` 与 `persons[i]` 的羁绊。拉面下 `persons[5]` 是理事长、友人卡（人头 6）没被锁到。当前不影响该测试要验证的结论（分布权重走 `deyilv` → 读 `deck`，6 张卡都锁到了；友人卡靠 group buff 闪彩、不看羁绊），但语义上应改成按 `PersonType::Card | ScenarioCard` 遍历。
  - ~~`features.rs` 的 `person_train_slots` 对分身仍是 last-write one-hot~~ —— **已修（2026-08-23）**。本文件「NN 特征编码器首版的五处编码缺陷」原记为已改 multi-hot，实际仍是 `slots[idx] = Some(t)` 后写覆盖（Codex 评审独立复现）。已改为 `Vec<[bool; TRAIN_NUM]>` 掩码，cards 段与 persons 段同步改为逐位写标志，维度不变。该文件是我们自己的代码，不涉及上游授权范围。

---

## 超级拉面分身失败是静默的（无计数、无告警）

- **日期**：2026-08-24
- **状态**：待解决
- **问题描述**：`ramen/action.rs` 的 `distribute_super_ramen_clones` 对每张支援卡最多重试 `option_trains.len() * 2` 次，全部失败时只走 `diag!(">> 超级拉面分身失败: ...")`，不返回 `Err`、不计数、不进任何统计。生产跑批时分身丢失完全无声。
- **排查过程**：
  - `choose(rng)` 是有放回采样，选项二只有 4 个训练位 → 8 次重试。若 4 个位里有 3 个被堵死，全落空概率约 `0.75^8 ≈ 10%`。
  - `card_indices` 升序收集，拉面友人卡人头下标必然大于全部训练卡（训练卡开局占 0-4，友人卡回合 2 才加入），**友人卡永远最后一个分配**，只能捡剩位，失败概率系统性最高。
  - 2026-08-24 给分身补了「每训练一个友人」约束后，友人卡又多了一条拒绝理由。生产中理事长 / 记者 `absent_rate = 200`，多数回合不在场，估计失败率在 1% 以下，但非零且不可观测。
- **解决方案**：待定。倾向把失败次数计入 bench 统计（与 rollout 失败计数同一处理方式），而非改成返回 `Err`——分身失败在规则上是允许的，不该中断育成。
- **备注**：这个函数在 2026-08-24 之前零测试覆盖，「友人卡分身从未生成」正是靠静默藏了三个月。

## 手写策略评估器的人头计数未随人数加成一并修正

- **日期**：2026-08-24
- **状态**：待解决
- **问题描述**：`neural/handwritten_evaluator.rs` 的 `let head_count = game.distribution()[train].len();` 与 `default_calc_training_value` 是同一个「人数」概念，但既不排除理事长 / 记者，也不过滤负数，还是裸下标索引（`distribution` 未初始化会直接 panic，而 `count_training_persons` 返回 0）。
- **排查过程**：该值用在三处阈值判断（`head_count >= 3`、`head_count >= ABSENT_HEAD_THRESHOLD + 1`），理事长在场会让阈值提前触发。同一文件的 `other_max_head` 用 `distribution()[t].len()` 亦然。
- **解决方案**：改用 `Game::count_training_persons`。影响的是**手写策略打分依据**而非模拟数值，故与 2026-08-24 的两处数值修复分开，不混进同一次改动。
- **备注**：属「人头下标当卡组下标」的同族残留——2026-08-23 修了取值型，2026-08-24 修了计数型与遍历型的模拟侧，策略侧的计数型漏了。

---

## `RamenGame::current_effect` 是从未被写入的死字段

- **日期**：2026-08-23
- **状态**：编码器侧已解决（2026-08-23）；**上游字段本身保留，待上游确认**
- **问题描述**：`RamenGame.current_effect: RamenEffect`（`game/ramen/state.rs:159-160`）的文档写「当前生效的拉面效果（每回合重新计算）」，但全仓库对它的写入只有 `newgame` 里的一次 `RamenEffect::default()`（`state.rs:237`）。它在整局中恒为全零。
- **排查过程**：
  - `grep -rn "current_effect" crates/umasim/src/` 的结果只有两行：`state.rs:160`（字段声明）与 `state.rs:237`（初始化为 `default()`）。没有任何赋值点。
  - 真正的拉面效果由 `game/ramen/effects.rs` 的 `calc_ramen_training_effect` 按**训练位**现场计算，算完直接用，从不写回 `current_effect`。
  - 之所以从来没暴露：没有任何读取方依赖它，所以恒零不影响模拟结果。
  - 触发发现的场景：NN 特征编码器照着字段名和文档，为它保留了 14 维特征，结果这 14 维在所有样本里恒为 0。
  - 设计上也确实**装不下**：`youqing` 只在友情训练时非零，`xunlian` 随训练位不同，不存在一份能代表所有训练位的单一「当前效果」。
- **解决方案**：待与上游确认。两个方向：
  1. 删除该字段（推荐）——没有读取方，删掉零风险，同时消除误导
  2. 若确实想要缓存，需改成「每训练位一份」并在 `run_distribute` 后填充；但这等于把派生量塞进状态，与「状态只存原始量」相悖
- **本次处理（2026-08-23）**：只删编码器这一侧——移除 `features.rs` 的 `G_EFFECT` 常量、`effect` 特征块与 `encode_effect` 函数，`GLOBAL_DIM` 由 166 降至 152、`INPUT_DIM` 由 766 降至 752。**上游的 `RamenGame::current_effect` 字段与 `RamenEffect` 类型未动**：删除公开字段会扩大与上游的合并冲突面，而 14 维恒零的实际危害完全在编码器侧，本地删干净即可。字段的去留留给上游决定。
- **备注**：`features.rs` 的模块文档已加「已知未覆盖」一节记录此事——若上游后续真正填充该字段，需要重新加回特征块并给样本打新的 schema 版本号。

---

## NN 特征编码器首版的五处编码缺陷（自查 + 三方评审）

- **日期**：2026-08-23
- **状态**：已解决（2026-08-23）
- **问题描述**：`game/ramen/features.rs` 首版存在五处会污染教师数据的编码缺陷。教师数据一旦落盘，编码错误无法回溯修正，故在接入 Phase 3 之前全部修掉。
- **排查过程**：派 Grok / Codex / Gemini 三方并行评审首版实现（Gemini 三次均因 agy 打开不存在的路径而硬失败，未产出）。Grok 与 Codex 的发现互补，逐条经本地 grep 复核后确认成立：
  1. **友人卡与人头错配**（Grok 提出）：编码器假设「人头下标 0..5 与卡组槽位同序」，实际下标 5 是理事长、友人卡在 6。卡槽 5 会读到理事长的训练位，人头 6 的卡链接丢失。根因是上游的布局问题，见本文件「人头下标与卡组槽位的对应关系在拉面剧本被打破」一条。
  2. **分身被后写覆盖**（Grok 提出）：分身不新建人头，而是把同一 `person_idx` 再 `push` 进另一个训练位（`game/ramen/game.rs:947`、`:960`），人头会同时出现在 `distribution` 的多行。`person_train_slots` 用 `slots[idx] = Some(t)` 后写覆盖，只保留编号最大的训练位——彩圈分身在特征里直接消失。
  3. **`current_effect` 14 维恒为 0**（Grok 与 Codex 独立提出）：见本文件对应条目。
  4. **`selected_regions` 默认值被当真数据**（Grok 与 Codex 独立提出）：默认 `[0,0,0]`，第 1 年地区选择（回合 2 的 `run_begin` 内联）之前会把地区 0（札幌-速）的效果编三遍。**注意：这不是游戏 bug**——`game/ramen/game.rs:118` 用 `turn < 2` 跳过 `RamenSelect`、`:262` 用 `turn >= 2` 门控候选面，游戏逻辑在回合 2 之前从不读该字段，只有编码器会读。
  5. **Train 阶段 `pending_ramen` 与 `current_ramen` 重复编码**（Codex 提出）：`ground_ramen_effects`（`game.rs:775-778`）设置 `current_ramen` 后不清 `pending_ramen`，`clear_pending()` 要到下一回合 `run_begin`（`game.rs:1109`）才调用。对游戏无害，但编码器把同一个选择编了两遍，其中 pending 已是语义上的残留。
- **解决方案**：
  1. 卡与人头的对应改为按 `card_id` 双向 `position()` 反查，不再依赖下标相等
  2. 训练位编码由 one-hot 改 multi-hot（维度不变），正确表达分身 —— **订正：此条在 `ef99478` 同样未落地**，`person_train_slots` 直到 2026-08-23 才真正改为 multi-hot 掩码
  3. 移除 `current_effect` 的 14 维块
  4. 新增 `regions_ready` 掩码位，未就绪时地区块整体写 0，不查 id 0
  5. Train 阶段只编 `current_ramen`
  6. 另按评审意见调整：归一化尺度改用真实上限（`scenario_pt` 1000→5000、`five_status` 1200→2800 等）；`onehot` 拆成 `onehot_optional`（合法缺席填 0）与 `onehot_checked`（非法下标 `bail!`），与模块文档的约束一致；补上地区的诀窍配方 `region_feeling`
- **备注**：
  - **订正（2026-08-23）**：上表的解决方案第 1 条（卡与人头按 `card_id` 双向反查）与第 3 条（移除 `current_effect` 的 14 维块）在 `ef99478` 里**实际并未落地**，本文件与 changelog 都误记为已完成。两条在「人头下标与卡组槽位错位」的修复中才真正实施。第 2、4、5、6 条在 `ef99478` 已落地，不受影响。
  - Codex 另给出了 Phase 3 接线的具体障碍清单（`training_sample.rs:17-38` 的 1121/587/89 常量与 `assert_eq!` 校验、`sample_collector.rs` 的 `OnsenAction` 映射、`search/result.rs:278` 只为 `SearchOutput<OnsenAction>` 实现、`collector.rs` 的 `ShardWriter` 未泛型化）。结论：拉面必须单开 schema，不能原地改温泉常量。留作 Phase 3 输入。
  - Codex 建议把 `remaining_race_slots` 从 `game/ramen/policy.rs` 挪到 `Uma` / `BaseGame` 层——它只处理 `FreeRaceData::mask` 与通用自选比赛回合范围，与拉面策略权重无关，现在让编码器反向依赖了策略模块。认同，待办。
  - `sampler.rs:38-40` 的模块注释仍写 `default_config.toml` 是 `fixed`，提交 `875f61c` 已改回 `all`，注释与事实相反，待订正。

---

## distribute_person 中"不出现"判定受得意率影响

- **日期**：2026-08-19
- **状态**：待解决
- **问题描述**：当前 `Game::distribute_person`（`traits.rs`）将"不出现"判定和"训练位置分配"混在一起，不在率 = `absent_rate / (500 + absent_rate + deyilv)`，导致得意率会影响"不出现"概率。按剧本原始规则，"不出现"概率应不受得意率影响，得意率只影响训练位置的权重分配。
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
- **备注**：2026-08-19 曾确认暂不动 absent_rate 相关逻辑（涉及 `absent_rate_drop` 等其他领域知识），当时只修复了 `RamenGame::deyilv` 缺剧本加成的问题；`distribute_person` 修正留待本 issue。RamenGame 当前未 override `distribute_person`，两步算法尚未实现。

---

## 第三年地区选择组合过多

- **日期**：2026-08-17
- **状态**：已解决（2026-08-20）
- **问题描述**：第3年可选地区为10-19共10个，C(10,3)=120种组合，动作空间过大，影响搜索效率
- **排查过程**：
  - 第1/2年各5个地区，C(5,3)=10种组合，可接受
  - 第3年120种组合导致 Trainer 需要评估120个动作，计算开销显著增加
- **解决方案**：第3年地区选择默认 Fixed，走固定组合 `[[11,14,15]]`，跳过 120 组合枚举（2026-08-20 实现，见 changelog）；后续如需动态策略，可再按"先定主方向、再子集内选组合"的预筛选方案扩展
- **备注**：第3年地区还包含pt_bonus效果，选择策略需要同时考虑youqing/pt_bonus和配方匹配

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

## constants.json 排名数据需人工更新

- **日期**：2026-08-19
- **状态**：已解决（2026-08-20，数据经用户确认）
- **问题描述**：constants.json 中的 `rank_scores`、`rank_names`、`five_status_final_score` 为游戏排名相关数据，当前数值可能已过期，需要稍后按最新游戏版本人工核对更新
- **排查过程**：配置系统整理（Phase 2）甄别 constants.json 各项归属时确认：这三项属固定游戏数据、不随剧本变化，保留在 constants.json，由人工更新
- **解决方案**：用户提供最新数据后更新三个数组（提交 `aa756d9`）：`rank_scores` / `rank_names` 补齐至 LS24（共 298 档，速度档位上调），`five_status_final_score` 同步核对（3399 档）
- **备注**：与配置整理方案一致，见 .trae/documents/config_refactor_plan.md

---

## 友人事件效果未应用「事件效果提高」「恢复量提高」词条

- **日期**：2026-08-19
- **状态**：已解决（2026-08-20）
- **问题描述**：友人卡的支援卡词条「事件效果提高」（`event_effect_up`）和「恢复量提高」（`event_recovery_amount_up`）当前在游戏逻辑中没有被应用到友人事件上。`FriendState` 正确读取并保存了这两个字段（`event_bonus`、`vital_bonus`），但所有 `apply_event` 路径都没有引用它们——base/onsei/ramen 三个剧本都是如此。这导致友人事件（登场/点击/解锁/出行 1-5）的实际效果与支援卡词条描述不符。
- **排查过程**：
  - `crates/umasim/src/game/mod.rs:138-158` `FriendState::new` 从 `card.card_value[rank].event_recovery_amount_up` / `event_effect_up` 读取并写入 `vital_bonus` / `event_bonus`。
  - 全仓 grep `event_bonus` / `vital_bonus` / `event_effect_up` / `event_recovery_amount_up`（排除 `features[...]` 神经网络特征值）：
    - `event_effect_up` / `event_recovery_amount_up` 只出现在 `FriendState::new` 的读取、`SupportCardValue::explain`（仅打印描述）、`onsen/game.rs:1328-1329`（神经网络特征归一化）
    - `friend.vital_bonus` / `friend.event_bonus` 在游戏逻辑（`apply_event` 路径）**完全无引用**
  - `apply_event` 调用链：
    - `RamenGame::apply_event`（`game.rs:378`）→ `self.base.apply_event(event, choice, rng)` → `BaseGame::apply_event`（`base/mod.rs:151`）→ `self.uma.add_value(&choice_result.value)` 直接结算效果，**未乘 `friend.event_bonus` / `friend.vital_bonus`**
    - `OnsenGame::apply_event` / `BasicGame::apply_event` 同理，未引用 friend bonus
  - 影响范围：所有剧本（包括拉面杯、温泉、基础）的友人事件；只有 `FriendCardState` 为 `SSR`/`R` 的卡组才会触发
- **解决方案**：用户已确认精确语义，最终在 `BaseGame` 统一修复：
  1. `BaseGame` 新增 `friend_event_ids: HashSet<u32>` 字段；`BaseGame::new` 从 `global_events().friend_events.values()` 派生 base/onsen 友人事件 ID；`RamenGame::newgame` 额外 extend `RAMENDATA.friend_events.values()` 合并 ramen 友人事件 ID
  2. `BaseGame::apply_event` 在结算前判定 `friend_event_ids.contains(&event.id)`，命中则调用新增的 `apply_friend_bonus` 私有方法
  3. `apply_friend_bonus` 按用户确认语义乘算：`status_pt[i] * (100 + event_bonus) / 100`（floor）仅作用于 `status_pt[0..6]`；`vital * (100 + vital_bonus) / 100`（仅 `vital > 0`）；不影响 `max_vital` / `motivation` / `hint_level` / `friendship`
  4. base / onsen / ramen 三剧本统一受益，trait override (`BasicGame/OnsenGame/RamenGame::apply_event`) 无需改动
  5. `event_bonus == 0 && vital_bonus == 0`（未携带友人卡）时分支跳过，行为与现状一致
  6. 新增 7 个单元测试（`test_apply_friend_bonus_*` × 5 + `test_apply_event_*_integration` × 2），全 124 lib 测试通过
- **备注**：数据结构（`EventCollection` / `RamenScenarioData`）未修改，所有友人事件 ID 集合从 `friend_events.values()` 在 `BaseGame::new` / `RamenGame::newgame` 时派生，O(1) HashSet 查询；同时删除与本次修改冲突的 `test_ramen_region_strategy_fixed_skips_enumeration` 测试。

---

## stable 工具链下 cargo fmt 破坏 Nightly 格式

- **日期**：2026-08-21
- **状态**：已解决（2026-08-22，手动执行方案；钩子自动化已撤销）
- **问题描述**：`rustfmt.toml` 使用 8 个 Nightly-only 选项（`imports_granularity` / `group_imports` / `trailing_comma` / `wrap_comments` 等），在 stable 工具链执行 `cargo fmt` 会静默忽略这些选项，把整个仓库重排成 stable 风格——与 git 历史 `ffddd1a`（应用仓库 rustfmt 格式）的 Nightly 格式不一致，产生大量无关 diff
- **排查过程**：
  - `rustfmt.toml` 含 `imports_granularity = "Crate"`、`group_imports = "StdExternalCrate"`、`trailing_comma = "Never"` 等 Nightly 特性，stable rustfmt 不支持且无报错（仅 warning）
  - 环境无 `rust-toolchain` 文件、无 Nightly 工具链（`rustup toolchain list` 仅 stable），`cargo fmt` 实际以 stable 行为执行
  - 实际触发：bench 重构时执行 `cargo fmt` 误将 55 个文件、约 1900 行重排为 stable 风格，已全部还原
  - 2026-08-22 进一步发现：Nightly 为滚动版本，`ffddd1a` 格式化时与当前 nightly（rustfmt 1.10.0-nightly 2026-08-20）规则存在漂移（trailing_comma 去尾逗号、imports 合并、多行表达式压缩等），`cargo +nightly fmt --all -- --check` 报 40 文件 387 处差异
- **解决方案**：
  1. AGENTS.md 固化规则：项目使用 **Nightly** 格式规则，stable 工具链下**禁止执行 cargo fmt**
  2. **cargo fmt 只能由用户手动执行**（2026-08-22 更新）：AGENTS.md 明确「禁用 cargo fmt」——格式化由用户手动执行（`cargo +nightly fmt --all`），Agent 不执行 fmt，避免强制重新读取代码；编译仍用 stable，互不影响
  3. **钩子自动化已撤销（2026-08-22 用户决策）**：cargo-husky 依赖、`.cargo-husky/hooks/pre-commit` 与生成的 `.git/hooks/pre-commit` 均已移除，不再自动检查格式——从源头防止 stable fmt 改为**流程约定**（提交前用户手动跑 nightly fmt）；全库已应用当前 nightly 格式（提交 `fd144af`，42 文件），该次格式化保留
- **备注**：Nightly 为滚动版本，rustfmt 输出偶有细微变化（本次漂移即一例）；如需完全固定可锁定指定日期（如 `nightly-YYYY-MM-DD`）或引入 `rust-toolchain.toml` 固定工具链

## game_config.toml 从未被加载（路径 bug）+ [config_override] 字段不合并

- **日期**：2026-08-22
- **状态**：已解决（2026-08-22）
- **问题描述**：用户在 `game_config.toml` 修改 `uma`/`cards`/`extra_count` 后实际不生效——`load_game_config` 合并结果仍是 default 值（`uma=102601`、`extra_count=[0;6]`），用户配置形同虚设
- **排查过程**：
  - 临时诊断测试（load_game_config 打印合并结果）发现 `extra_count=[0;6]` ——这正是「用户配置不存在」兜底分支的默认值，证明走了兜底而非正常解析
  - 根因一（路径）：`USER_CONFIG_REL_PATH = "../game_config.toml"`（相对 `gamedata/` 的语义，Phase 2 步骤 4 引入），但 `resolve_user_config_path` 用 `current_dir().join(..)` 拼接，解析为「工作目录上一级」——文件不存在 → `cfg_path.exists()` 为 false → 永远走兜底，game_config.toml 从未被解析
  - 根因二（字段）：`OverrideConfig` 只有 `extra_count` 等 7 个字段且均必填（无 `serde(default)`），`uma`/`cards`/`blue_count` 不在其中——即便路径修复，这些字段也会被 serde 静默忽略；且 merge 无条件覆盖（缺写时用兜底值覆盖 default，与「只写要改的项」注释语义冲突）
  - 兜底机制（用户配置不存在时静默回退）掩盖了此 bug：程序一直正常跑，只是配置从未生效
- **解决方案**：
  1. 路径修复：`USER_CONFIG_REL_PATH` 改为 `"game_config.toml"`（工作目录根，与注释语义一致）
  2. `OverrideConfig` 全字段 `Option` 化（`#[serde(default)]`，`None` = 不覆盖）：新增 `uma`/`cards`/`blue_count`，现有 `extra_count`/`mcts_selected_onsen`/`log_level`/`num_threads` 改可选；merge 全部 `if let Some` 覆盖——真正实现「只写你要改的项」
  3. 加固：`#[serde(deny_unknown_fields)]`——拼错/未支持的字段显式报错，杜绝静默忽略
  4. `game_config.toml`：顶层 `mcts_selected_onsen`（原在遗留段，同样静默失效）移入 `[config_override]` 段；注释更新可选覆盖语义
  5. 测试 +4：merge 全 None 不覆盖 / 部分覆盖生效 / deny_unknown_fields 报错 / 用户配置路径定位 + 真实文件合并集成验证（`uma=100901` 生效）
- **备注**：`ramen_region_strategy` / `ramen_region_fixed`（game_config.toml 注释中的「顶层覆盖」）同样不在 `OverrideGameConfig` 结构内，目前无法通过 game_config.toml 覆盖（未启用，暂缓处理）；`OverrideGameConfig` 顶层未知字段（如遗留的顶层 `mcts_selected_onsen` 写法）仍静默忽略，如需可后续加 deny

---

## 第2/3年地区选择无 build 自适应（score_region 对无 xunlian 的地区无区分度）

- **日期**：2026-08-22
- **状态**：已解决（2026-08-23）
- **问题描述**：bench 已支持不同卡组（build）跑批，但第 3 年地区选择仍是固定值 `[[11, 14, 15]]`（default_config.toml `ramen_region_fixed`），所有 build 共用同一组合；即使恢复 120 组合全枚举，当前 `score_region` 打分对第 3 年地区**没有 build 区分度**——速度向与智力向卡组都会选中同一组合
- **排查过程**：
  - 临时测试 `test_region_build_sensitivity`（policy.rs）实测：速度向卡组（3速）与智力向卡组（3智）在第 3 年 120 组合打分下均选中 `[10, 11, 12]`，score 相同（4500）
  - 根因一（fixed 绕过策略）：`ramen_region_strategy="fixed"` 时第 3 年直接 apply `ramen_region_fixed[0]`，不经过 `decide_region` 打分，build 差异无从体现
  - 根因二（打分失效）：`score_region` 的 build 自适应依赖 `region.xunlian × 卡组 bias`，但第 3 年地区（id 10-19）`xunlian` 全为 0 → bias 项恒零；而 `pt_bonus` 全部相同（50）、`hint` 全 0 → 所有组合分数相同，argmax 取第一个
  - 第 3 年地区的真实差异在 `youqing`（40-60）与 `at_trains` 覆盖（单属性 vs 多属性），当前打分完全未纳入
  - **实施时补充发现：第 2 年（id 5-9）同样失效**。这批地区的 `xunlian` 也全为 0，`hint_count` 恒为 2，故 10 个组合（C(5,3)）一直同分取第一个。第 2 年不受 `ramen_region_strategy` 影响（`fixed` 只作用于第 3 年），所以这条一直存在、与固定值配置无关。本 issue 范围因此扩大到第 2/3 年
- **解决方案**（已定，待实施）：
  1. 增强 `score_region`：新增 `youqing × at_trains × 卡组 bias` 项（如 `Σ_{t∈at_trains} youqing × bias[t]`），使第 3 年打分随 build 训练倾向变化；权重沿用/扩展 `RamenPolicyConfig`
  2. `default_config.toml` 恢复 `ramen_region_strategy = "all"`（120 组合枚举，O(360) 打分已标注便宜）
  3. 验收：不同 build 选出不同第 3 年地区组合（更新 `test_region_build_sensitivity` 断言方向）
- **实施结果**（2026-08-23）：`xunlian` 与 `youqing` 统一按 `bias_sum` 缩放（新增 `region_youqing_weight`）。因同一年内 `pt_bonus` / `hint_count` 恒定、且 `xunlian` 与 `youqing` 不会同时非零，该权重的绝对值不影响 argmax，只影响打印的分数量级。验收通过：速度向选 `[15,17,18]`、智力向选 `[15,17,19]`。手写策略基线（8 马娘 × 7 build）聚合 49586 → 51340（+1754），其中唯一含「根」的 build `sta0_wis2` 增幅最大（+3572）——固定值 `[11,14,15]` 的训练位并集是 `{速,耐,力,智}`，恰好漏掉根
- **备注**：
  - 备选方案 B：按卡组主属性预筛候选范围（120 → 20~30 个）再打分——打分增强后不稳定时再启用，避免裁剪丢最优解
  - `test_region_build_sensitivity` 已由临时验证转为断言测试（`assert_ne!` 两 build 选中组合不同）
  - 与既有 issue「第三年地区选择组合过多」（已解决：Fixed）相关联：Fixed 是性能临时方案，本 issue 是在 build 维度上的功能补齐。恢复 `all` 后实测整局耗时 2.9ms 不变，120 组合枚举无可测代价
  - **影响采样器复现基座**：`sampler.rs` 的 `run_region_select` 读 `GAMECONFIG.ramen_region_strategy`，本次由 `fixed` 改 `all` 后同一条 `SampleSpec` 的轨迹已不同（结构性指标不变：74/78 回合覆盖、卡组分层 min==max）。Phase 3 落盘的配置签名须用改后这套
