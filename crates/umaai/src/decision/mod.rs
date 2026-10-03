//! 决策后处理：从 Trainer 拿到决策结果后，挂 luck score、组装并 `emit` 到 sink。
//!
//! 职责边界：本模块只关心「一条决策如何变成输出」——T(n) baseline 计算 / luck 快照
//! 挂载 / reason 与 ramen_action 附注 / 最终 `sink.emit`。不包含场景逻辑
//! （温泉 / 拉面各自在 `crate::scenario` 下）与主程序调度（`crate::main`）。

pub mod luck_score;
pub use luck_score::LuckScoreTracker;

pub mod record;
pub use record::RecordingSink;

pub mod zip_export;

use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json, to_value};
use umasim::{
    game::{Game, Trainer},
    gamedata::GAMECONSTANTS,
    global,
    output::{
        DecisionInfo, DecisionReasonData, DecisionReasonSink, DecisionSink
    },
    trainer::MctsTrainer
};

/// 缓存最近一次决策理由的 sink（每回合覆写）
///
/// 接到 `RamenMctsTrainer::reason_sink`：把 `DecisionReasonData` 缓存到内部
/// `Mutex<Option<…>>`，由拉面模块在 human mode 下取出后调 `render_reason_lines`
/// 打印到屏幕。`emit_decision_reason` 内部的 `info!` 调用**已被 trainer
/// `.verbose(false)` 关闭**，避免双打印；同时也避开 umaai 默认关闭日志的现状。
pub struct LastReasonSink {
    inner: Mutex<Option<DecisionReasonData>>
}

impl LastReasonSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(None) })
    }

    pub(crate) fn take(&self) -> Option<DecisionReasonData> {
        // 取走副本，留 None 给下一次覆写
        self.inner.lock().expect("reason sink").take()
    }
}

impl DecisionReasonSink for LastReasonSink {
    fn emit(&self, reason: &DecisionReasonData) {
        *self.inner.lock().expect("reason sink") = Some(reason.clone());
    }
}

/// 把 trainer 的 last_decision 喂给 sink：先挂 luck score 字段，再 emit
///
/// **Step 5 改造**：从原 `emit_decision` 升级——每回合不再"select_action → 立即 emit"，
/// 而是把多次 select_action 的 last_decision 收集起来，由场景模块在 `calc_*`
/// 完成后**统一调一次本函数**：
///
/// 1. 取 `trainer.last_decision()`（最后一次 select_action 的数据）
/// 2. 算 T(n) baseline（按局数加权：Σ score × n / Σ n，与 onsen `update_score` 同口径）
/// 3. `tracker.on_new_turn(chara_id, t_n_baseline)` 更新 / 切局检测
/// 4. 挂 `tracker.snapshot()` + 每候选 `action_luck` 到 `info.scenario_extra`
/// 5. `sink.emit(&info, &game.view())`
///
/// `GameView::view()` 由 Game trait 默认实现填充；onsen scenario 字段留空。
///
/// **Step 7 改造**：拆出 [`emit_with_luck_decision`] 接收 `Option<DecisionInfo>`，
/// 让拉面分支（`RamenMctsTrainer` 等其他 trainer）也能复用 luck score 挂载逻辑，
/// 不必为每个 trainer 单独写一份。
pub fn emit_with_luck<G: Game>(
    trainer: &MctsTrainer, game: &G, sink: &Arc<dyn DecisionSink>, tracker: &mut LuckScoreTracker, chara_id: u64,
    decision_kind: &str
) {
    // onsen 路径没有 reason_sink，传 None——scenario_extra.reason 不挂
    emit_with_luck_decision(
        trainer.last_decision(), game, sink, tracker, chara_id, None, decision_kind, None, true
    );
}

/// 把已提取的 `DecisionInfo` 喂给 sink：挂 luck score 字段 + emit。
///
/// 与 [`emit_with_luck`] 区别在于**不依赖具体 trainer 类型**——只要 trainer 实现了
/// `Trainer<G>` 并返回 `DecisionInfo` 即可。拉面分支（`RamenMctsTrainer` 等）走这里。
///
/// **2026-09 扩展**：
/// - `reason_data`：拉面 MCTS 路径从 `LastReasonSink.take()` 取 `DecisionReasonData`，
///   挂到 `scenario_extra.reason` 让 AIRedirector 拿到完整 human mode reason 信息
///   （metric / chosen_desc / chosen_mean / chosen_n / rivals[]）。其他 trainer 传 `None`。
/// - `decision_kind`：由外部传入（拉面模块按 snapshot 时 stage 填）——标明这条决策
///   属于哪种（"ramen_select" / "special_select" / "train" / "region_select" /
///   "super_ramen_select" / "event"）。C# 端按此字段分发 partial decision。
/// - `ramen_action`：仅 ramen 路径传 `Some(&str)`——`RamenAction::to_string()` 的结果，
///   含吃面 + 隐藏诀窍 + 操作三阶段信息（按用户拍板"AIRed 端只显示不解析"）。
/// - `attach_luck`：是否把运气分（`tracker.snapshot()` 的三个键 + 每候选 `action_luck`）
///   挂到 `scenario_extra`。`region_select`（地区选择）传 `false`——那一回合的期望评分会因
///   **RMJ 结算事件触发时机的模拟差异**大幅波动，运气分不可信，故不计算也不下发
///   （`decisions.csv` 对应列随之为空），**同时不推进基线**：运气分停留在上一次真正
///   显示时的状态，地区选择那次的跳变被整段跳过，不算进任何回合的「本回合运气」。
pub fn emit_with_luck_decision<G: Game>(
    last_decision: Option<DecisionInfo>, game: &G, sink: &Arc<dyn DecisionSink>,
    tracker: &mut LuckScoreTracker, chara_id: u64,
    reason_data: Option<&DecisionReasonData>,
    decision_kind: &str,
    ramen_action: Option<&str>,
    attach_luck: bool
) {
    let Some(mut info) = last_decision else {
        return;
    };
    // T(n) baseline：按局数加权（手写 / 早期早退时 candidate_n 为空 → 退化为按候选数等权）
    let t_n_baseline: f64 = if info.candidate_n.is_empty() {
        if info.candidate_scores.is_empty() {
            0.0
        } else {
            info.candidate_scores.iter().map(|&s| s as f64).sum::<f64>()
                / info.candidate_scores.len() as f64
        }
    } else {
        let total_n: u32 = info.candidate_n.iter().sum();
        if total_n == 0 {
            info.candidate_scores.iter().map(|&s| s as f64).sum::<f64>()
                / info.candidate_scores.len().max(1) as f64
        } else {
            info.candidate_scores
                .iter()
                .zip(info.candidate_n.iter())
                .map(|(&s, &n)| (s as f64) * (n as f64))
                .sum::<f64>()
                / total_n as f64
        }
    };

    // Note: 增加 mcts_turn_bonus 的行为原本在MctsTrainer<OnsenGame> 实现，现在放在外部完成，MCTS只输出原始分数
    // 不挂运气分的决策（`attach_luck == false`，地区选择）**不推进基线**：
    // 让 `total_luck_score` / `last_turn_delta` 停留在「上一次真正显示运气分时的状态」，
    // 即下一个决策的回合运气 = 它相对上次显示点的变化（把地区选择那次的跳变整段跳过）。
    if attach_luck {
        tracker.on_new_turn(
            chara_id,
            t_n_baseline,
            game.turn(),
            game.max_turn(),
            global!(GAMECONSTANTS).mcts_turn_bonus,
        );
    }

    // 每候选 action_luck：T(n, action_i) - T(n)（AIRedirector 关心，玩家模式跳过）
    // `attach_luck == false`（region_select）时不算：避免为不可信的运气分做无用功
    let action_luck = attach_luck.then(|| {
        json!(
            info.candidate_scores
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, (s as f64) - t_n_baseline))
                .collect::<std::collections::HashMap<usize, f64>>()
        )
    });

    // 顶层 decision_kind（外部传入）
    info.decision_kind = decision_kind.to_string();

    // 挂载 scenario_extra：snapshot + action_luck（仅 `attach_luck`）+
    // reason（仅拉面 MCTS）+ ramen_action（仅 ramen 路径）
    //
    // 合并而不是覆盖：决策本身可能已带信息（网络模式的 `decision_source`、
    // `mcts_nn_hint` 的参考推荐），以它为底再盖上 luck 相关键，键名冲突时以 luck 为准。
    let mut merged = match info.scenario_extra.take() {
        Some(Value::Object(map)) => map,
        _ => Map::new()
    };
    // luck 三键（snapshot）+ action_luck：仅 `attach_luck` 时挂（region_select 不挂）
    if let Some(action_luck) = action_luck {
        if let Ok(Value::Object(snapshot)) = to_value(tracker.snapshot()) {
            for (k, v) in snapshot {
                merged.insert(k, v);
            }
        }
        merged.insert("action_luck".into(), action_luck);
    }
    // reason：拉面 MCTS 路径挂，其他 trainer 不挂
    if let Some(data) = reason_data {
        if let Ok(reason_v) = to_value(data) {
            merged.insert("reason".into(), reason_v);
        }
    }
    // ramen_action：仅 ramen 路径填（"吃面/X(替换Ax1+Bx2)" 等）
    if let Some(action_text) = ramen_action {
        merged.insert("ramen_action".into(), action_text.into());
    }
    info.scenario_extra = Some(Value::Object(merged));

    sink.emit(&info, &game.view());
}