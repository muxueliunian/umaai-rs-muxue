//! 决策后处理：从 Trainer 拿到决策结果后，挂 luck score、组装并 `emit` 到 sink。
//!
//! 职责边界：本模块只关心「一条决策如何变成输出」——T(n) baseline 计算 / luck 快照
//! 挂载 / reason 与 ramen_action 附注 / 最终 `sink.emit`。不包含场景逻辑
//! （温泉 / 拉面各自在 `crate::scenario` 下）与主程序调度（`crate::main`）。

pub mod luck_score;
pub use luck_score::LuckScoreTracker;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering}
};

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
    /// 新建一个空 sink
    ///
    /// 直接返回 `Arc`：它要同时交给训练员（`reason_sink`）与取用方，
    /// 两边共享同一个槽位。
    pub fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(None) })
    }

    /// 取走上一次缓存的决策理由，并把槽位清空
    ///
    /// 返回 `None` 表示自上次取用以来没有新的理由被 emit（例如合并
    /// `RamenSelect` 路径搜了但不暴露摘要）。清空是有意的：不清会让下一次
    /// 没有理由时读到上一回合的旧数据。
    pub fn take(&self) -> Option<DecisionReasonData> {
        // 取走副本，留 None 给下一次覆写
        self.inner.lock().expect("reason sink").take()
    }
}

impl DecisionReasonSink for LastReasonSink {
    fn emit(&self, reason: &DecisionReasonData) {
        *self.inner.lock().expect("reason sink") = Some(reason.clone());
    }
}

/// 可临时静音的理由出口（套在真正的 [`LastReasonSink`] 外面）
///
/// 地区对照模式会在**随机流副本**上多跑一次既有装配。那一次如果是真搜索，它照样会
/// 经 `RamenMctsTrainer::emit_decision_reason` 把理由写进共用的槽位；而本回合真正执行
/// 的是网络那条，屏幕上就会出现「参照搜索的首选」被当成网络推荐的理由。
///
/// 解决办法不是事后清空槽位——那会连带删掉链式决策里**其它步骤**已经写好的有效理由。
/// 这里在写入口上做门：参照那一跑之前静音，跑完立刻恢复，其它步骤的理由一个不少。
pub struct ReasonGate {
    /// 真正的理由槽位
    inner: Arc<dyn DecisionReasonSink>,
    /// 静音中（参照侧正在跑）
    muted: AtomicBool
}

impl ReasonGate {
    /// 把一个理由出口包成可静音的门
    pub fn new(inner: Arc<dyn DecisionReasonSink>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            muted: AtomicBool::new(false)
        })
    }

    /// 在守卫存活期间静音；守卫析构时自动恢复
    ///
    /// 用 RAII 而不是「手动置位 / 复位」：参照那一跑中间可能 `?` 早退，手动复位会被
    /// 跳过，此后整局的理由都被静音。
    pub fn mute(&self) -> ReasonMuteGuard<'_> {
        self.muted.store(true, Ordering::Relaxed);
        ReasonMuteGuard { gate: self }
    }

    /// 当前是否静音（测试与诊断用）
    pub fn is_muted(&self) -> bool {
        self.muted.load(Ordering::Relaxed)
    }
}

impl DecisionReasonSink for ReasonGate {
    fn emit(&self, reason: &DecisionReasonData) {
        if self.muted.load(Ordering::Relaxed) {
            return;
        }
        self.inner.emit(reason);
    }
}

/// [`ReasonGate::mute`] 的 RAII 守卫：析构即恢复
pub struct ReasonMuteGuard<'a> {
    /// 被静音的门
    gate: &'a ReasonGate
}

impl Drop for ReasonMuteGuard<'_> {
    fn drop(&mut self) {
        self.gate.muted.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod reason_gate_tests {
    use anyhow::{Result, bail};

    use super::*;
    use crate::utils::Checks;

    /// 造一条可辨认的理由数据（内容不参与判定，只看它到没到槽位）
    fn sample_reason(turn: i32) -> DecisionReasonData {
        DecisionReasonData {
            turn,
            metric: "score".to_string(),
            threshold: 0.0,
            max_display: 5,
            chosen_index: 0,
            chosen_desc: "候选A".to_string(),
            chosen_mean: 1.0,
            chosen_n: 1,
            rivals: Vec::new()
        }
    }

    /// 理由门：静音期间丢弃、恢复后照常写入，且**不清掉**已经写好的理由
    ///
    /// 第三条是关键：对照模式只想挡住「参照那一跑」，链式决策里前一步已经写进槽位的
    /// 理由必须原样留着——粗暴地事后清空槽位会把它一起删掉。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_reason_gate_mutes_only_the_guarded_run() -> Result<()> {
        let mut c = Checks::new();
        let slot = LastReasonSink::new();
        let gate = ReasonGate::new(slot.clone());

        // 1) 不静音：正常写入
        gate.emit(&sample_reason(1));
        let got = slot.take();
        println!("未静音 → {:?}", got.as_ref().map(|d| d.turn));
        c.check(got.map(|d| d.turn) == Some(1), "未静音时理由正常写入槽位");

        // 2) 链式前一步先写一条，再静音跑一次参照：前一条必须留着，参照那条被丢弃
        gate.emit(&sample_reason(2));
        {
            let _guard = gate.mute();
            c.check(gate.is_muted(), "守卫存活期间处于静音");
            gate.emit(&sample_reason(99));
        }
        c.check(!gate.is_muted(), "守卫析构后自动恢复");
        let got2 = slot.take();
        println!("静音一跑之后 → {:?}", got2.as_ref().map(|d| d.turn));
        c.check(got2.map(|d| d.turn) == Some(2), "❗前一步的理由还在，参照那条没进来");

        // 3) 恢复之后照常写入
        gate.emit(&sample_reason(3));
        c.check(slot.take().map(|d| d.turn) == Some(3), "恢复后理由照常写入");

        // 4) 守卫在 `?` 早退路径上同样会析构：用一个必定返回 Err 的闭包模拟
        let early = || -> Result<()> {
            let _guard = gate.mute();
            bail!("模拟参照侧中途报错");
        };
        println!("早退 → {:?}", early().err().map(|e| e.to_string()));
        c.check(!gate.is_muted(), "参照侧中途报错后静音也已恢复（没把整局静音掉）");
        c.finish()
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
    emit_with_luck_decision(trainer.last_decision(), game, sink, tracker, chara_id, None, decision_kind, None);
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
pub fn emit_with_luck_decision<G: Game>(
    last_decision: Option<DecisionInfo>, game: &G, sink: &Arc<dyn DecisionSink>,
    tracker: &mut LuckScoreTracker, chara_id: u64,
    reason_data: Option<&DecisionReasonData>,
    decision_kind: &str,
    ramen_action: Option<&str>
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
    let _turn_delta = tracker.on_new_turn(
        chara_id,
        t_n_baseline,
        game.turn(),
        game.max_turn(),
        global!(GAMECONSTANTS).mcts_turn_bonus,
    );

    // 每候选 action_luck：T(n, action_i) - T(n)（AIRedirector 关心，玩家模式跳过）
    let action_luck = json!(
        info.candidate_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, (s as f64) - t_n_baseline))
            .collect::<std::collections::HashMap<usize, f64>>()
    );

    // 顶层 decision_kind（外部传入）
    info.decision_kind = decision_kind.to_string();

    // 挂载 scenario_extra：snapshot + action_luck（必挂）+ reason（仅拉面 MCTS）+
    // ramen_action（仅 ramen 路径）
    //
    // ❗**合并而不是覆盖**：决策本身可能已经带了信息（`decision_source` 来源标签、
    // 地区对照模式的 `region_compare`）。旧实现直接 `info.scenario_extra = extra`，
    // 于是「执行侧是真地区搜索」那条决策一走 luck 挂载，对照结果与来源标签就在
    // JSON 里凭空消失。这里以决策自带的对象为底，再把 luck 相关键盖上去——
    // 键名冲突时以 luck 为准（与旧行为一致），不冲突的一律保留。
    let mut merged = match info.scenario_extra.take() {
        Some(Value::Object(map)) => map,
        _ => Map::new()
    };
    if let Ok(Value::Object(snapshot)) = to_value(tracker.snapshot()) {
        for (k, v) in snapshot {
            merged.insert(k, v);
        }
    }
    merged.insert("action_luck".into(), action_luck);
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