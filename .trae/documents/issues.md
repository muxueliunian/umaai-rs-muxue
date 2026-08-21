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
- **状态**：已解决（2026-08-22，钩子自动化 + 全库格式化）
- **问题描述**：`rustfmt.toml` 使用 8 个 Nightly-only 选项（`imports_granularity` / `group_imports` / `trailing_comma` / `wrap_comments` 等），在 stable 工具链执行 `cargo fmt` 会静默忽略这些选项，把整个仓库重排成 stable 风格——与 git 历史 `ffddd1a`（应用仓库 rustfmt 格式）的 Nightly 格式不一致，产生大量无关 diff
- **排查过程**：
  - `rustfmt.toml` 含 `imports_granularity = "Crate"`、`group_imports = "StdExternalCrate"`、`trailing_comma = "Never"` 等 Nightly 特性，stable rustfmt 不支持且无报错（仅 warning）
  - 环境无 `rust-toolchain` 文件、无 Nightly 工具链（`rustup toolchain list` 仅 stable），`cargo fmt` 实际以 stable 行为执行
  - 实际触发：bench 重构时执行 `cargo fmt` 误将 55 个文件、约 1900 行重排为 stable 风格，已全部还原
  - 2026-08-22 进一步发现：Nightly 为滚动版本，`ffddd1a` 格式化时与当前 nightly（rustfmt 1.10.0-nightly 2026-08-20）规则存在漂移（trailing_comma 去尾逗号、imports 合并、多行表达式压缩等），`cargo +nightly fmt --all -- --check` 报 40 文件 387 处差异
- **解决方案**：
  1. AGENTS.md 固化规则：项目使用 **Nightly** 格式规则，stable 工具链下**禁止执行 cargo fmt**
  2. 格式化走 Nightly：`rustup toolchain install nightly --component rustfmt` 后使用 `cargo +nightly fmt`；编译仍用 stable，互不影响
  3. **钩子自动化（2026-08-22）**：引入 cargo-husky pre-commit 钩子（`.cargo-husky/hooks/pre-commit`，构建时自动分发到 `.git/hooks/`）：有 nightly 工具链时 `cargo +nightly fmt --all -- --check` 强制检查，无 nightly 则跳过不阻塞——从源头防止 stable fmt / 未格式化代码入库；全库已应用当前 nightly 格式（提交 `fd144af`，42 文件）
- **备注**：Nightly 为滚动版本，rustfmt 输出偶有细微变化（本次漂移即一例）；如需完全固定可锁定指定日期（如 `nightly-YYYY-MM-DD`）或引入 `rust-toolchain.toml` 固定工具链
