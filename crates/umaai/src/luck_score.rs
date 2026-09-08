//! Luck score 跟踪器：累计 T(n) baseline 与回合 / 全局运气分
//!
//! 口径与文档 §3.3 一致：
//! - T(n) baseline 由 `main.rs` 主循环从 MCTS `candidate_scores` + `candidate_n`
//!   **按局数加权**计算（`Σ (score × n) / Σ n`，与 onsen `update_score` 同口径）。
//! - 回合运气分 = T(n+1) − T(n)；全局运气分 = T(n+1) − T(1)。
//! - chara_id 切换时 T(1) 重置、total_luck 清零（基于一次育成内累积）。
//!
//! 与 `DecisionInfo` 配套：本模块的 [`LuckScoreSnapshot`] 会被挂到
//! `DecisionInfo::scenario_extra` 一起发给 AIRedirector（详见集成文档 §3.3.6）。

use serde::Serialize;

/// 一次 lucky score 快照（挂在 `DecisionInfo::scenario_extra` 下发给 AIRedirector）
///
/// 与 `DecisionInfo` 同模块的 `Serialize` 派生，便于直接 `to_value(snapshot)`
/// 嵌入 scenario_extra。
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LuckScoreSnapshot {
    /// T(1)：首次 MCTS 决策的 baseline terminal（一次性记录）
    ///
    /// chara_id 切换或 AI 首次启动时被覆写。
    pub initial_terminal_baseline: f64,

    /// T(n+1)：当前回合的 baseline terminal
    pub current_terminal_baseline: f64,

    /// T(n+1) − T(1)：全局运气分
    pub total_luck_score: f64,

    /// T(n+1) − T(n)：上一回合的回合运气分
    ///
    /// 首回合为 `None`（没有上一回合可比）。
    pub last_turn_delta: Option<f64>
}

/// Luck score 跟踪器（状态机）
///
/// 持有 chara_id 切局检测、初始 / 上回合 baseline 两份内部状态。
/// 每次 AI 推荐完成后由 `main.rs` 调 [`Self::on_new_turn`] 更新。
///
/// **切局键**：`single_mode_chara_id`（C# 端 `single_mode_chara_id`，单调递增）——
/// 比 `uma_id` 更准确（同一马娘 `uma_id` 重复训练时也能识别新局）。
pub struct LuckScoreTracker {
    initial_terminal_baseline: Option<f64>,
    prev_turn_terminal_baseline: Option<f64>,
    total_luck: f64,
    last_turn_delta: Option<f64>,
    last_single_mode_id: Option<u64>
}

impl Default for LuckScoreTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LuckScoreTracker {
    /// 新建空 tracker（所有 baseline 字段为 `None`，首回合 `on_new_turn` 返回 `None`）
    pub fn new() -> Self {
        Self {
            initial_terminal_baseline: None,
            prev_turn_terminal_baseline: None,
            total_luck: 0.0,
            last_turn_delta: None,
            last_single_mode_id: None
        }
    }

    /// 当前记录的 `single_mode_chara_id`（`None` 表示从未记录过）
    ///
    /// 拉面分支用此判断切局——不依赖 `SAVED_GAME`（那是 onsen 专用）。
    /// 用 `single_mode_chara_id` 而不是 `uma_id`：同一马娘（`uma_id` 相同）可重复训练，
    /// 但 `single_mode_chara_id` 单调递增，能识别"同一 chara 重开新一局"。
    pub fn last_single_mode_id(&self) -> Option<u64> {
        self.last_single_mode_id
    }

    /// AI 推荐后调用：传入当前回合 T(n) baseline（已按局数加权计算好的值）
    ///
    /// 返回：当前回合的回合运气分（`None` 表示首次 / 切局后的首回合）
    ///
    /// **切局检测**：`last_single_mode_id` 变化或初始 baseline 未记录 → 全部 reset：
    /// - `initial_terminal_baseline` / `prev_turn_terminal_baseline` = 本回合 baseline
    /// - `total_luck` = 0.0
    /// - `last_turn_delta` = None
    /// - `last_single_mode_id` = 新 chara_id
    pub fn on_new_turn(&mut self, single_mode_id: u64, t_n_baseline: f64) -> Option<f64> {
        // 切局检测：chara_id 变了或 AI 第一次启动
        if self.last_single_mode_id != Some(single_mode_id) || self.initial_terminal_baseline.is_none() {
            self.initial_terminal_baseline = Some(t_n_baseline);
            self.prev_turn_terminal_baseline = Some(t_n_baseline);
            self.total_luck = 0.0;
            self.last_turn_delta = None;
            self.last_single_mode_id = Some(single_mode_id);
            return None;
        }

        // T(n+1) - T(n) = t_n_baseline - prev_turn_terminal_baseline
        let delta = self.prev_turn_terminal_baseline.map(|p| t_n_baseline - p);
        if let Some(d) = delta {
            self.last_turn_delta = Some(d);
            // T(n+1) - T(1) = t_n_baseline - initial_terminal_baseline
            self.total_luck = t_n_baseline - self.initial_terminal_baseline.unwrap();
        }
        self.prev_turn_terminal_baseline = Some(t_n_baseline);
        delta
    }

    /// 拍快照（用于挂到 `DecisionInfo::scenario_extra`）
    ///
    /// 未初始化时（首回合 / tracker 全新）所有数值字段给默认值 0 / None —— 让
    /// JSON 序列化时字段齐全不丢 key，但语义上是"无数据"。
    pub fn snapshot(&self) -> LuckScoreSnapshot {
        LuckScoreSnapshot {
            initial_terminal_baseline: self.initial_terminal_baseline.unwrap_or(0.0),
            current_terminal_baseline: self.prev_turn_terminal_baseline.unwrap_or(0.0),
            total_luck_score: self.total_luck,
            last_turn_delta: self.last_turn_delta
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 首回合：`on_new_turn` 返回 `None`、baseline 记录为初始值
    #[test]
    fn test_first_turn_returns_none() {
        let mut t = LuckScoreTracker::new();
        assert_eq!(t.on_new_turn(42, 50000.0), None, "首回合应返回 None");
        let snap = t.snapshot();
        println!("首回合 snapshot: {snap:?}");
        assert_eq!(snap.initial_terminal_baseline, 50000.0);
        assert_eq!(snap.current_terminal_baseline, 50000.0);
        assert_eq!(snap.total_luck_score, 0.0);
        assert_eq!(snap.last_turn_delta, None);
    }

    /// 累加正确：第二回合返回 T(2)-T(1) = delta，total_luck 同步累加
    #[test]
    fn test_accumulation_two_turns() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0); // T(1)
        let delta = t.on_new_turn(42, 50150.0); // T(2) - T(1) = 150
        println!("第二回合 delta={delta:?}");
        assert_eq!(delta, Some(150.0));
        let snap = t.snapshot();
        assert_eq!(snap.total_luck_score, 150.0);
        assert_eq!(snap.last_turn_delta, Some(150.0));
    }

    /// 多回合累加：total_luck = T(n+1) - T(1)，与历次 delta 累加一致
    #[test]
    fn test_multi_turn_accumulation() {
        let mut t = LuckScoreTracker::new();
        let baselines = [50000.0, 50150.0, 50320.0, 50400.0, 50280.0];
        let mut prev = None;
        for (i, &b) in baselines.iter().enumerate() {
            let delta = t.on_new_turn(42, b);
            match (i, prev) {
                (0, _) => assert_eq!(delta, None, "首回合 None"),
                (_, Some(p)) => assert_eq!(delta, Some(b - p), "第 {} 回合 delta = T(n) - T(n-1)", i + 1),
                _ => unreachable!()
            }
            prev = Some(b);
        }
        let snap = t.snapshot();
        println!("5 回合后 snapshot: {snap:?}");
        assert_eq!(snap.initial_terminal_baseline, 50000.0);
        assert_eq!(snap.current_terminal_baseline, 50280.0);
        assert_eq!(snap.total_luck_score, 280.0); // 50280 - 50000
        assert_eq!(snap.last_turn_delta, Some(-120.0)); // 50280 - 50400
    }

    /// 切局检测：chara_id 变了 → total_luck 清零、baseline 重置
    #[test]
    fn test_chara_id_change_resets() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0);
        t.on_new_turn(42, 50150.0); // 累加 150
        let snap_before = t.snapshot();
        println!("切局前: {snap_before:?}");
        assert_eq!(snap_before.total_luck_score, 150.0);

        // 切局：新 chara_id
        let delta = t.on_new_turn(99, 48000.0);
        assert_eq!(delta, None, "切局后首回合 None");
        let snap_after = t.snapshot();
        println!("切局后: {snap_after:?}");
        assert_eq!(snap_after.initial_terminal_baseline, 48000.0);
        assert_eq!(snap_after.current_terminal_baseline, 48000.0);
        assert_eq!(snap_after.total_luck_score, 0.0, "切局 total_luck 清零");
        assert_eq!(snap_after.last_turn_delta, None, "切局 last_turn_delta 重置");
    }

    /// snapshot Serialize 包含全部 4 个字段（AIRedirector 解析不能缺 key）
    #[test]
    fn test_snapshot_serialize_all_fields() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0);
        t.on_new_turn(42, 50150.0);
        let snap = t.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        println!("snapshot json: {json}");
        assert!(json.contains("initial_terminal_baseline"));
        assert!(json.contains("current_terminal_baseline"));
        assert!(json.contains("total_luck_score"));
        assert!(json.contains("last_turn_delta"));
        // 关键：last_turn_delta 为 Some 也要序列化
        assert!(json.contains("150"));
    }
}