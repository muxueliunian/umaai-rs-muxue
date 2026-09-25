# umaai `main.rs` 职责拆分重构

## Context（背景）

当前 `crates/umaai/src/main.rs`（\~900 行）把多类职责混在一个文件里：

* 主程序入口/CLI/初始化/watch 循环。

* 温泉逻辑（`calc_onsen_training` / `calc_onsen_event`）及 onsen 分支的整段编排（newgame 检测、`emit_info`、luck 挂载）。

* 拉面逻辑（`calc_ramen_training`）及 ramen 分支的整段编排。

* 决策后的处理（luck score 计算 + 决策输出）：`emit_with_luck` / `emit_with_luck_decision` / `LastReasonSink` / `fallback_decision` / `ramen_stage_kind`。

目的：按职责拆分清晰，`main.rs` 收敛为「薄调度」，各场景逻辑与决策输出各归其位。

**用户已确认的取舍**：

1. 拆到底 → main.rs 只保留 CLI / 初始化 / watch 循环 + 按 scenario 分发；onsen、ramen 两分支各自抽成一个进程函数（含 newgame 检测与 emit）。
2. 拉面专属辅助（`fallback_decision` / `ramen_stage_kind`）跟 `calc_ramen_training` 放一起进拉面模块；`LastReasonSink` 这类通用决策管道放 `decision/` 模块。

## 目标目录结构

```
src/
  main.rs              # 薄调度：Args/解析、初始化、watch 循环 + dispatch、run_evaluate、测试
  protocol/            # 不变
  utils.rs             # 不变
  decision/
    mod.rs             # 决策后处理：LastReasonSink、emit_with_luck、emit_with_luck_decision
    luck_score.rs      # 由 src/luck_score.rs 原样移入（LuckScoreTracker 归并到决策目录）
  scenario/
    mod.rs             # pub mod onsen; pub mod ramen;
    onsen.rs           # process_onsen + calc_onsen_training + calc_onsen_event
    ramen.rs           # process_ramen + calc_ramen_training + fallback_decision + ramen_stage_kind
```

> 文件移动：`src/luck_score.rs` → `src/decision/luck_score.rs`。所有原 `crate::luck_score::LuckScoreTracker` 引用改为走 `decision` 命名空间。`decision/mod.rs` 顶部 `pub mod luck_score;` + `pub use luck_score::LuckScoreTracker;` 方便 scenario 侧引用。`main.rs` 删除 `pub mod luck_score;`。

## 改动明细

### 1. 新增 `src/decision/mod.rs`（决策后处理）

从 main.rs 原样迁移（只改动可见性/依赖引用）：

* `LastReasonSink`：`pub struct`，`new()`/`take()` 改为 `pub(crate)`；`impl DecisionReasonSink` 不变。

* `emit_with_luck<G: Game>(trainer: &MctsTrainer, game, sink, tracker, chara_id, decision_kind)`：不变。

* `emit_with_luck_decision<G: Game>(last_decision, game, sink, tracker, chara_id, reason_data, decision_kind, ramen_action)`：原样迁移，`tracker: &mut LuckScoreTracker` 引 `super::luck_score::LuckScoreTracker`（同级 `pub mod luck_score;`），内部仍用 `umasim::global!`/`gamedata::GAMECONSTANTS`。

另含被移入的 `luck_score.rs`：`decision/mod.rs` 声明 `pub mod luck_score;` 并 `pub use luck_score::LuckScoreTracker;`。

`main.rs` 删除以上符号，并移除 `pub mod luck_score;`（已并入 `decision`）；新增 `pub mod decision;`、`pub mod scenario;`。

### 2. 新增 `src/scenario/onsen.rs`（温泉逻辑）

从 main.rs 迁移：

* `calc_onsen_training` / `calc_onsen_event`：原样，参数不变。

* `process_onsen`（新增，承接 main 的 onsen 分支）签名：

  ```rust
  pub fn process_onsen(
      mut game: OnsenGame,
      trainer: &MctsTrainer,
      sink: &Arc<dyn DecisionSink>,
      luck_tracker: &mut LuckScoreTracker,
      rng: &mut StdRng,
      json_mode: bool,
      emit_info: &dyn Fn(&str),
      game_config: &GameConfig,
  ) -> Result<()>
  ```

  * SAVED\_GAME 写库 / newgame 检测（`game.is_next_of(&saved)`）、`emit_info("new_game")`、`trainer.print_newgame_config(&game)`、打印 `game_config.onsen_order`、重置 luck\_tracker。

  * 事件/训练分发（`calc_onsen_event` / `calc_onsen_training`）。

  * `emit_with_luck(&trainer, &game, sink, luck_tracker, chara_id, onsen_kind)`。

  * `eprintln!("计算完成，等待新数据...")`。

### 3. 新增 `src/scenario/ramen.rs`（拉面逻辑）

从 main.rs 迁移：

* `calc_ramen_training`：`json_sink` 形参改为 `emit_info: &dyn Fn(&str)`，内部 `eprintln!("计算后续动作...")` + `emit_info("compute_next_step")`（替换原 `if let Some(js)=json_sink{...}`）。其余逻辑（decide 闭包、连续决策、reason 屏幕打印）原样。

* `fallback_decision` / `ramen_stage_kind`：原样。

* `process_ramen`（新增，承接 main 的 ramen 分支）签名：

  ```rust
  pub fn process_ramen(
      mut game: RamenGame,
      single_mode_chara_id: Option<u64>,
      trainer: &RamenMctsTrainer,
      reason_slot: &LastReasonSink,
      sink: &Arc<dyn DecisionSink>,
      luck_tracker: &mut LuckScoreTracker,
      rng: &mut StdRng,
      json_mode: bool,
      emit_info: &dyn Fn(&str),
  ) -> Result<()>
  ```

  * `game.stage == RamenStage::Begin` 早 continue（原逻辑：分支内 `continue` → 进程函数直接 return）。

  * 切局检测（chara\_id / `emit_info("new_game")` / reset luck\_tracker）。

  * human 屏幕打印（`explain` / `explain_ramen_info` / `explain_distribution`）。

  * 调 `calc_ramen_training`，遍历链式中间决策 `sink.emit`，末尾决策按 `candidate_scores.is_empty()` 分流：手写直接 `sink.emit` / 否则 `emit_with_luck_decision`。

  * `eprintln!("计算完成，等待新数据...")`。

### 4. 瘦身 `src/main.rs`

保留：`Args`/`parse_args`/`print_help_and_exit`、`run_evaluate`、全部初始化（sink/json\_sink/emit\_info/emit\_error、trainers、watcher、`connected` gate）、watch 循环 dispatch、`hit` 错误分支、`pause_on_exit`、`main`、现有 `#[cfg(test)]` 测试。

watch 循环 dispatch 改为：

```rust
match parse_game_by_scenario(&contents) {
    Ok(ParsedGame::Onsen(game)) =>
        scenario::onsen::process_onsen(game, &trainer, &sink, &mut luck_tracker, &mut rng, json_mode, &emit_info, &game_config)?,
    Ok(ParsedGame::Ramen { game, single_mode_chara_id }) =>
        scenario::ramen::process_ramen(game, single_mode_chara_id, &ramen_trainer, &reason_slot, &sink, &mut luck_tracker, &mut rng, json_mode, &emit_info)?,
    Err(e) => { /* emit_error + human 红字 */ }
}
```

* `emit_info("compute_start")` 保留在 `watch()` 之后、dispatch 之前（main 持有闭包）。

* 新增 `pub mod decision;`、`pub mod scenario;`。

* `json_mode` 仍用于 `connected` gate 与错误分支 Human 显示。

## 复用点

* `emit_info`/`emit_error` 闭包（main 持有，作为 `&dyn Fn(&str)` 传给进程函数）。

* `umasim::gamedata::GameConfig`（`gamedata::config` 已 `pub use config::*` 导出）。

* `process_ramen` 使用 `crate::decision::{emit_with_luck_decision, LastReasonSink, LuckScoreTracker}`；`process_onsen` 使用 `crate::decision::{emit_with_luck, LuckScoreTracker}`。`main.rs` 原 `pub mod luck_score;` 移除，`LuckScoreTracker` 统一经 `crate::decision` 访问。

## 验证

* `cargo check -p umaai`：无新增 warning/error（仅 retain 既有的 umasim lib 19 个无关 warning）。

* `cargo test -p umaai`：现有 `parse_args` 测试通过（Args 仍在 main.rs）。

* 行为等价：对照迁移前后 onsen/ramen 分支的 emit 顺序与 luck 挂载逻辑逐项核对（chain 中间决策直接 emit、末尾按候选胜负分支、`compute_next_step` 时机不变）。

