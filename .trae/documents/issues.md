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
