//! Luck score 跟踪器：累计 T(n) baseline 与回合 / 全局运气分
//!
//! 口径与文档 §3.3 一致：
//! - T(n) baseline 由 `umaai::main` 主循环从 MCTS `candidate_scores` + `candidate_n`
//!   **按局数加权**计算（`Σ (score × n) / Σ n`，与 onsen `update_score` 同口径）。
//! - **显示分换算**：baseline 存的是「原期望评分」，`on_new_turn` 入参把
//!   `mcts_turn_bonus` 叠加为显示分再存储/比较。公式
//!   `显示分 = 原期望评分 + (总回合数 − 回合) × mcts_turn_bonus`；
//!   `initial_terminal_baseline` 恒按 `turn = 0` 计算。
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
    /// T(1) 的显示分：首次 MCTS 决策的 baseline 终端值（一次性记录）
    ///
    /// 恒按 `turn = 0` 换算（`原分 + 总回合数 × mcts_turn_bonus`）。
    /// chara_id 切换或 AI 首次启动时被覆写。
    pub initial_terminal_baseline: f64,

    /// T(n+1) 的显示分：当前回合的 baseline 终端值（按当前回合换算）
    pub current_terminal_baseline: f64,

    /// T(n+1) − T(1)（显示分口径）：全局运气分
    pub total_luck_score: f64,

    /// T(n+1) − T(n)（显示分口径）：上一回合计到本回合的回合运气分
    ///
    /// 首回合为 `None`（没有上一回合可比）。
    pub last_turn_delta: Option<f64>
}

/// Luck score 跟踪器（状态机）
///
/// 持有 chara_id 切局检测、初始 / 上回合 baseline（**显示分**）两份内部状态。
/// 每次 AI 推荐完成后由 `umaai` 调 [`Self::on_new_turn`] 更新。
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

    /// 登记「当前正在处理的是哪一局」，返回**这一帧是不是新一局的开始**
    ///
    /// 与 [`Self::on_new_turn`] 的分工：本方法只管**育成身份**，
    /// `on_new_turn` 只管**运气数值**。两件事必须分开，因为不是每一帧都有搜索评分：
    /// 整局网络模式（`ramen_trainer_policy = "nn"`）一次搜索都不跑，
    /// `on_new_turn` 整局都不会被调用；身份登记若还挂在它上面，`last_single_mode_id`
    /// 就永远是 `None`，同一局的**每一帧**都会被判成新局——屏幕反复打印「育成开始」，
    /// 按协议处理 `new_game` 的客户端也会反复重置 UI。
    ///
    /// ❗**不注入任何假 baseline**：新局时把数值状态一并清空（与旧的
    /// `*tracker = LuckScoreTracker::new()` 等价），但 `initial_terminal_baseline`
    /// 保持 `None`。首次真有评分的那一回合，`on_new_turn` 里
    /// `initial_terminal_baseline.is_none()` 这一支照样会走重置路径并返回 `None`，
    /// 因此有搜索评分的那条路行为与本方法引入前逐字一致。
    ///
    /// 同一局重复调用是幂等的：返回 `false`，且不触碰任何已累积的数值。
    pub fn begin_game(&mut self, single_mode_id: u64) -> bool {
        if self.last_single_mode_id == Some(single_mode_id) {
            return false;
        }
        *self = Self::new();
        self.last_single_mode_id = Some(single_mode_id);
        true
    }

    /// 把「原期望评分」换算为显示分
    ///
    /// `display = raw + (max_turn − turn) × bonus`
    /// - `initial` 恒走 `turn = 0`
    /// - `current` 走实际回合 `turn`
    fn to_display(raw: f64, turn: i32, max_turn: i32, bonus: i32) -> f64 {
        raw + (max_turn - turn) as f64 * bonus as f64
    }

    /// AI 推荐后调用：传入当前回合 `mcts_turn_bonus` 前提下的 baseline（已带回合信息）
    ///
    /// 入参：
    /// - `single_mode_id`：切局键
    /// - `t_n_baseline`：当前回合的「原期望评分」（无 bonus）
    /// - `turn` / `max_turn`：当前回合 / 总回合数（`Game::turn()` / `Game::max_turn()`）
    /// - `bonus`：`mcts_turn_bonus`（`global!(GAMECONSTANTS).mcts_turn_bonus`）
    ///
    /// 返回：当前回合的回合运气分（**显示分口径**；`None` 表示首次 / 切局后的首回合）
    ///
    /// **切局检测**：`last_single_mode_id` 变化或初始 baseline 未记录 → 全部 reset：
    /// - `initial_terminal_baseline` = 本回合原分按 `turn=0` 换算的显示分
    /// - `prev_turn_terminal_baseline` = 本回合显示分
    /// - `total_luck` = 0.0
    /// - `last_turn_delta` = None
    /// - `last_single_mode_id` = 新 chara_id
    pub fn on_new_turn(
        &mut self,
        single_mode_id: u64,
        t_n_baseline: f64,
        turn: i32,
        max_turn: i32,
        bonus: i32
    ) -> Option<f64> {
        let current_display = Self::to_display(t_n_baseline, turn, max_turn, bonus);
        let initial_display = Self::to_display(t_n_baseline, 0, max_turn, bonus);

        // 切局检测：chara_id 变了或 AI 第一次启动
        if self.last_single_mode_id != Some(single_mode_id) || self.initial_terminal_baseline.is_none() {
            self.initial_terminal_baseline = Some(initial_display);
            self.prev_turn_terminal_baseline = Some(current_display);
            self.total_luck = 0.0;
            self.last_turn_delta = None;
            self.last_single_mode_id = Some(single_mode_id);
            return None;
        }

        // T(n+1) - T(n) = current_display - prev_display
        let delta = self.prev_turn_terminal_baseline.map(|p| current_display - p);
        if let Some(d) = delta {
            self.last_turn_delta = Some(d);
            // T(n+1) - T(1) = current_display - initial_display
            self.total_luck = current_display - self.initial_terminal_baseline.unwrap();
        }
        self.prev_turn_terminal_baseline = Some(current_display);
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
    use anyhow::Result;

    use super::*;
    use crate::utils::Checks;

    /// `begin_game`：同一局只报一次新局，换 ID 再报一次，且**不注入假 baseline**
    ///
    /// 这是整局网络模式（一次搜索都不跑、`on_new_turn` 整局不被调用）下
    /// 「每一帧都被判成新局」的回归。第 3 组观测守住「有评分那条路行为不变」：
    /// 登记身份之后 `initial_terminal_baseline` 仍是 `None`，首次真有评分的回合照样
    /// 走 `on_new_turn` 的重置支并返回 `None`。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_begin_game_registers_identity_without_baseline() -> Result<()> {
        let mut c = Checks::new();
        let mut t = LuckScoreTracker::new();

        // 1) 同一局连续多帧：只有第一帧算新局
        let first = t.begin_game(7);
        let second = t.begin_game(7);
        let third = t.begin_game(7);
        println!("同一局三帧 → {first} / {second} / {third}");
        c.check(first, "第 1 帧报新局");
        c.check(!second && !third, "同一局的后续帧不再报新局");
        c.check(t.last_single_mode_id() == Some(7), "身份已登记");

        // 2) 登记身份不造 baseline
        let snap = t.snapshot();
        println!("登记后 snapshot: {snap:?}");
        c.check(t.initial_terminal_baseline.is_none(), "❗没有注入假的 initial baseline");
        c.check(t.prev_turn_terminal_baseline.is_none(), "❗没有注入假的上回合 baseline");
        c.check(snap.total_luck_score == 0.0, "运气分仍是 0");
        c.check(snap.last_turn_delta.is_none(), "回合运气分仍是 None");

        // 3) 有评分的那条路不变：首次 on_new_turn 仍走重置支、仍返回 None
        let delta = t.on_new_turn(7, 50000.0, 5, 78, 1);
        println!("登记之后首次 on_new_turn → {delta:?}");
        c.check(delta.is_none(), "首次有评分的回合仍返回 None（与本方法引入前一致）");
        c.check(t.snapshot().initial_terminal_baseline == 50078.0, "initial 由真实评分建立");

        // 4) 换一局：再报一次新局，且已累积的数值被清掉
        t.on_new_turn(7, 50150.0, 6, 78, 1);
        c.check(t.snapshot().total_luck_score != 0.0, "换局前确有累积（否则下一条观测是空跑）");
        let changed = t.begin_game(8);
        println!("换局 → {changed}");
        c.check(changed, "换 ID 报新局");
        c.check(t.last_single_mode_id() == Some(8), "身份切到新一局");
        c.check(t.snapshot().total_luck_score == 0.0, "换局清空累积的运气分");
        c.check(t.initial_terminal_baseline.is_none(), "换局后同样不预置 baseline");
        c.finish()
    }

    /// 首回合：`on_new_turn` 返回 `None`、initial 按 turn=0 换算、current 按回合换算
    #[test]
    fn test_first_turn_returns_none() {
        let mut t = LuckScoreTracker::new();
        // bonus=1, max_turn=78, turn=5: current=50000+(78-5)=50073; initial=50000+78=50078
        assert_eq!(t.on_new_turn(42, 50000.0, 5, 78, 1), None, "首回合应返回 None");
        let snap = t.snapshot();
        println!("首回合 snapshot: {snap:?}");
        assert_eq!(snap.initial_terminal_baseline, 50078.0);
        assert_eq!(snap.current_terminal_baseline, 50073.0);
        assert_eq!(snap.total_luck_score, 0.0);
        assert_eq!(snap.last_turn_delta, None);
    }

    /// 累加正确：第二回合返回 T(2)-T(1) = delta，total_luck 同步累加（显示分口径）
    #[test]
    fn test_accumulation_two_turns() {
        let mut t = LuckScoreTracker::new();
        // 首回合 turn=0：initial=50000+78=50078, current=50000+78=50078
        t.on_new_turn(42, 50000.0, 0, 78, 1);
        // 第二回合 turn=1：current=50150+(78-1)=50227；delta=50227-50078=149；total=50227-50078=149
        let delta = t.on_new_turn(42, 50150.0, 1, 78, 1);
        println!("第二回合 delta={delta:?}");
        assert_eq!(delta, Some(149.0));
        let snap = t.snapshot();
        assert_eq!(snap.total_luck_score, 149.0);
        assert_eq!(snap.last_turn_delta, Some(149.0));
    }

    /// 多回合累加：total_luck = T(n+1) - T(1)（显示分口径），与历次 delta 累加一致
    #[test]
    fn test_multi_turn_accumulation() {
        let mut t = LuckScoreTracker::new();
        // 每回合 raw baseline，bonus=1, max_turn=78, turn=i
        let raws = [50000.0, 50150.0, 50320.0, 50400.0, 50280.0];
        let mut prev_display = None;
        for (i, &b) in raws.iter().enumerate() {
            let turn = i as i32;
            let display = b + (78 - turn) as f64;
            let delta = t.on_new_turn(42, b, turn, 78, 1);
            match (i, prev_display) {
                (0, _) => assert_eq!(delta, None, "首回合 None"),
                (_, Some(p)) => assert_eq!(delta, Some(display - p), "第 {} 回合 delta = T(n) - T(n-1)", i + 1),
                _ => unreachable!()
            }
            prev_display = Some(display);
        }
        let snap = t.snapshot();
        println!("5 回合后 snapshot: {snap:?}");
        // initial 恒按 turn=0：50000+78=50078
        assert_eq!(snap.initial_terminal_baseline, 50078.0);
        // current：50280+(78-4)=50354
        assert_eq!(snap.current_terminal_baseline, 50354.0);
        // total = 50354 - 50078 = 276
        assert_eq!(snap.total_luck_score, 276.0);
        // last_turn_delta = 50354 - [50400+(78-3)] = 50354 - 50475 = -121
        assert_eq!(snap.last_turn_delta, Some(-121.0));
    }

    /// 切局检测：chara_id 变了 → total_luck 清零、baseline 重置
    #[test]
    fn test_chara_id_change_resets() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0, 0, 78, 1);
        t.on_new_turn(42, 50150.0, 1, 78, 1);
        let snap_before = t.snapshot();
        println!("切局前: {snap_before:?}");
        assert_eq!(snap_before.total_luck_score, 149.0);

        // 切局：新 chara_id（turn 也回到 1）
        let delta = t.on_new_turn(99, 48000.0, 1, 78, 1);
        assert_eq!(delta, None, "切局后首回合 None");
        let snap_after = t.snapshot();
        println!("切局后: {snap_after:?}");
        assert_eq!(snap_after.initial_terminal_baseline, 48078.0);
        assert_eq!(snap_after.current_terminal_baseline, 48077.0);
        assert_eq!(snap_after.total_luck_score, 0.0, "切局 total_luck 清零");
        assert_eq!(snap_after.last_turn_delta, None, "切局 last_turn_delta 重置");
    }

    /// snapshot Serialize 包含全部 4 个字段（AIRedirector 解析不能缺 key）
    #[test]
    fn test_snapshot_serialize_all_fields() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0, 0, 78, 1);
        t.on_new_turn(42, 50150.0, 1, 78, 1);
        let snap = t.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        println!("snapshot json: {json}");
        assert!(json.contains("initial_terminal_baseline"));
        assert!(json.contains("current_terminal_baseline"));
        assert!(json.contains("total_luck_score"));
        assert!(json.contains("last_turn_delta"));
        // 关键：last_turn_delta 为 Some 也要序列化（149 = 50227-50078）
        assert!(json.contains("149"));
    }

    /// 显示分不依赖 test 常量：bonus=0 时显示分 = 原分（退化行为）
    #[test]
    fn test_zero_bonus_equals_raw() {
        let mut t = LuckScoreTracker::new();
        t.on_new_turn(42, 50000.0, 3, 78, 0);
        let snap = t.snapshot();
        assert_eq!(snap.initial_terminal_baseline, 50000.0);
        assert_eq!(snap.current_terminal_baseline, 50000.0);
    }
}