//! 检查项引擎（文档 §6.1 坏手法 / §6.6 超级拉面期 / §6.7 伪波动标记）
//!
//! 原则（§6.1）：**已验证判据**直接产 findings；**判据待实测**的项（训练与
//! 体力健康 / 吃面节奏 / 友人完成度 / free_race 次数 / 状态健康 / 属性溢出）
//! 不纳入自动 findings，留待 SKILL 层用 digest 数据人工判读。
//!
//! 代码常量（§10）：`YEAR_BOUNDARIES` / `INHERIT_TURNS` / `SUPER_RAMEN_START`。

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    decisions::{DecRow, FlaggedTurn},
    execution::{ExecutionResult, Finding},
    schedule::Schedule,
    timeline::TimelineRow
};

/// 剧本年份边界（代码常量，§10）
pub const YEAR_BOUNDARIES: [u32; 3] = [24, 48, 72];
/// 两次继承回合（代码常量，§10）
pub const INHERIT_TURNS: [u32; 2] = [30, 54];
/// 第 1 年地区选择回合（开局 2-3 回合；地区选择本身带来的期望跳变也是
/// 程序性波动——与年界前的 RegionSelect 正跳同源，用户拍板）
pub const Y1_REGION_SELECT_TURNS: [u32; 2] = [2, 3];
/// 超级拉面期起点（turn ≥ 72；实测分身自 72 起，用户拍板）
pub const SUPER_RAMEN_START: u32 = 72;

/// 友人出行次数上限（完成要求）
const FRIEND_OUTING_CAP: i32 = 5;
/// 友人次数用完距结束的「仍远」阈值（初判，待标定；game6234：turn 58 用完剩 19）
const FRIEND_EARLY_REMAINING: u32 = 6;
/// 体力低点阈值（与 §6.1「体力长期低」同档；实测 turn 69 跌至 27）
const LOW_VITAL: i32 = 35;
/// 心情掉落后未及时恢复的观察窗口 N（§6.1 建议 N=3）
const MOTIVATION_WINDOW: u32 = 3;

/// 伪波动标记（§6.7）：年界 / 继承 / RMJ 结算 / 开局第 1 年地区选择回合
///
/// - 年界窗口 = 边界前 2 回合至后 1 回合；**turn 72 双属性**：既标记为年界
///   波动回合，也仍算进超级拉面期统计（§6.6 例外：该回合份量实打实，
///   归因时注意其正跳不是纯程序性回吐）
/// - 继承回合（[30,54]，与年份边界错开 6 回合的独立事件）
/// - RMJ 结算（从 `playing_state` 46/48 自动检测，不硬编码位置）
/// - **开局 2-3 回合 = 第 1 年地区选择**：选择带来的期望跳变（小赚或小亏）
///   也是程序性波动，与年界前的 RegionSelect 正跳同源（用户拍板）
pub fn flagged_turns(tl: &[TimelineRow]) -> Vec<FlaggedTurn> {
    let mut flags: Vec<FlaggedTurn> = Vec::new();
    for &t in Y1_REGION_SELECT_TURNS.iter() {
        flags.push(FlaggedTurn { turn: t, reason: "region_select(y1)".to_string() });
    }
    for &b in YEAR_BOUNDARIES.iter() {
        for t in b.saturating_sub(2)..=b + 1 {
            flags.push(FlaggedTurn { turn: t, reason: format!("year_boundary({b})") });
        }
    }
    for &t in INHERIT_TURNS.iter() {
        flags.push(FlaggedTurn { turn: t, reason: "inherit".to_string() });
    }
    for r in tl {
        if let Some(reason) = &r.reason {
            if reason.starts_with("rmj_settle") || reason.starts_with("rmj_final") {
                flags.push(FlaggedTurn { turn: r.turn, reason: reason.clone() });
            }
        }
    }
    flags.sort_by_key(|f| f.turn);
    flags.dedup_by(|a, b| a.turn == b.turn && a.reason == b.reason);
    flags
}

/// 超级拉面期（turn ≥ 72）运气统计（§6.6）
///
/// B 类分身按机制保证 100% 落得意位、每回合只能选一个训练 →「没吃到彩圈」
/// 不算亏，**只能用该期运气分判盈亏**（§6.5.3）。
#[derive(Debug, Default, Clone, Serialize)]
pub struct SuperRamenStats {
    /// 逐回合合计 Δ（回合合计口径，§6.2）
    pub by_turn: BTreeMap<u32, f64>,
    /// 该期回合合计 Δ 总和（正 = 赚 / 负 = 亏）
    pub total: f64
}

/// 超级拉面期统计（窗口内无任何 Δ 数据时返回 `None`）
pub fn super_ramen_stats(dec: &[DecRow]) -> Option<SuperRamenStats> {
    let mut by_turn: BTreeMap<u32, f64> = BTreeMap::new();
    for r in dec {
        if r.turn >= SUPER_RAMEN_START {
            if let Some(d) = r.turn_delta {
                *by_turn.entry(r.turn).or_default() += d;
            }
        }
    }
    if by_turn.is_empty() {
        return None;
    }
    let total = by_turn.values().sum();
    Some(SuperRamenStats { by_turn, total })
}

/// 超级拉面期盈亏 finding（§6.6：实测 game6234 该期 −1482 判亏）
pub fn super_ramen_finding(stats: &SuperRamenStats) -> Finding {
    let detail = stats
        .by_turn
        .iter()
        .map(|(t, d)| format!("{t}:{d:.1}"))
        .collect::<Vec<_>>()
        .join(" ");
    let verdict = if stats.total < 0.0 { "判亏" } else { "判赚" };
    Finding {
        kind: "super_ramen_luck".to_string(),
        turn: SUPER_RAMEN_START,
        file: None,
        evidence: format!(
            "turn>=72 回合合计 Δ = {:.2}（{detail}）{verdict}；B 类分身机制保证落得意位，\
             没吃到彩圈不算亏，只能用该期运气分判盈亏",
            stats.total
        ),
        severity: if stats.total < 0.0 { "warn".to_string() } else { "info".to_string() }
    }
}

/// 坏手法检查项（§6.1 已验证判据 + 训练失败候选清单）
///
/// 覆盖：
/// 1. **目标赛未跑赢**（已验证）：必赛回合不在 `raceHistory`（raceHistory 只记
///    跑赢——实测 game6234 命中 turn 45）
/// 2. **关键资源过早耗尽**（已验证）：友人次数达上限距结束仍远 + 此后出现
///    体力低点（实测：turn 58 用完 → turn 69 体力 27）
/// 3. **心情掉落后未及时恢复**（已验证，§6.1 首个已验证项）：干劲下降后
///    N=3 回合内既无恢复动作（出行/友人出行/休息）也未回升
/// 4. **训练失败候选**（先出候选人工复核）：AI 建议训练 + 跨回合五维零增长
///    + 体力有训练量级消耗
pub fn bad_habits(
    tl: &[TimelineRow],
    exec: &ExecutionResult,
    sched: &Schedule,
    race_history: &[i32]
) -> Vec<Finding> {
    let mut out = Vec::new();
    let max_turn = tl.last().map(|r| r.turn).unwrap_or(77);

    // ① 目标赛未跑赢
    for &t in &sched.mandatory_turns {
        if !race_history.contains(&t) {
            out.push(Finding {
                kind: "mandatory_race_not_won".to_string(),
                turn: t as u32,
                file: None,
                evidence: "必赛回合未跑赢（raceHistory 只记跑赢的比赛）".to_string(),
                severity: "warn".to_string()
            });
        }
    }

    // ② 关键资源过早耗尽（友人次数）
    // 「用完」的判定：friend_outgoing_used 达 5 的首份快照在 t5 回合 —— 但第 5 次
    // 出行发生在 t5-1 回合（行动结果体现在下一回合快照，§5.3），对齐文档口径
    // （实测 game6234：turn 58 用完）
    if let Some(t5) = tl.iter().find(|r| r.friend_outgoing_used >= FRIEND_OUTING_CAP).map(|r| r.turn) {
        let t = t5.saturating_sub(1);
        let remaining = max_turn.saturating_sub(t);
        let min_vital = tl
            .iter()
            .filter(|r| r.turn > t)
            .map(|r| r.vital)
            .min();
        if remaining >= FRIEND_EARLY_REMAINING && min_vital.is_some_and(|v| v < LOW_VITAL) {
            out.push(Finding {
                kind: "friend_quota_exhausted_early".to_string(),
                turn: t,
                file: None,
                evidence: format!(
                    "turn {t} 友人出行次数用完（距结束还剩 {remaining} 回合），此后体力最低 \
                     跌至 {}（< {LOW_VITAL}）——一次性资源过早耗尽",
                    min_vital.unwrap_or(0)
                ),
                severity: "warn".to_string()
            });
        }
    }

    // ③ 心情掉落后未及时恢复
    let mot_by_turn: BTreeMap<u32, i32> = per_turn_last(tl, |r| r.motivation);
    let recovery_turns: Vec<u32> = exec
        .rows
        .iter()
        .filter(|r| matches!(r.actual_action.as_str(), "出行" | "友人出行" | "休息"))
        .map(|r| r.turn)
        .collect();
    let turns: Vec<u32> = mot_by_turn.keys().copied().collect();
    for (i, &t) in turns.iter().enumerate() {
        let Some(&prev) = i.checked_sub(1).and_then(|p| mot_by_turn.get(&turns[p])) else {
            continue;
        };
        let Some(&cur) = mot_by_turn.get(&t) else { continue };
        if cur >= prev {
            continue; // 未下降
        }
        // 窗口 t..=t+3：既无恢复动作、干劲也未回升（高于掉落后的值）
        let recovered = (1..=MOTIVATION_WINDOW).any(|k| {
            mot_by_turn
                .get(&(t + k))
                .is_some_and(|&m| m > cur)
        });
        let acted = (0..=MOTIVATION_WINDOW).any(|k| recovery_turns.contains(&(t + k)));
        if !recovered && !acted {
            out.push(Finding {
                kind: "motivation_drop_unrecovered".to_string(),
                turn: t,
                file: None,
                evidence: format!(
                    "干劲 {prev}→{cur} 后 {MOTIVATION_WINDOW} 回合内既无出行/休息恢复动作、\
                     干劲也未回升（§6.8：AI 认为后续支援卡事件会把心情补回来，不主动恢复）"
                ),
                severity: "warn".to_string()
            });
        }
    }

    // ④ 训练失败候选（人工复核：事件导致的属性变化可能误判）
    for row in &exec.rows {
        let five_total: i32 = row.evidence.five_status_delta.iter().sum();
        if row.ai_choice.contains("训练")
            && five_total == 0
            && row.evidence.vital_delta <= -10
        {
            out.push(Finding {
                kind: "train_failure_candidate".to_string(),
                turn: row.turn,
                file: None,
                evidence: format!(
                    "AI 建议 {}，体力消耗 {} 但五维零增长（训练失败候选——\
                     正常训练对应属性 +13~+134；需人工复核事件影响）",
                    row.ai_choice, row.evidence.vital_delta
                ),
                severity: "info".to_string()
            });
        }
    }

    out
}

/// 每回合最后一份快照的字段值（回合末状态口径）
fn per_turn_last<T: Copy>(tl: &[TimelineRow], f: impl Fn(&TimelineRow) -> T) -> BTreeMap<u32, T> {
    let mut m: BTreeMap<u32, T> = BTreeMap::new();
    for r in tl {
        m.insert(r.turn, f(r)); // timeline 升序 → 后写覆盖 = 回合末
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::TimelineRow;

    /// 构造测试 timeline 行（只填检查项相关字段）
    fn row(turn: u32, vital: i32, motivation: i32, friend: i32, reason: Option<&str>) -> TimelineRow {
        TimelineRow {
            turn,
            seq: 0,
            stage: "Train".to_string(),
            reason: reason.map(String::from),
            source: "command".to_string(),
            playing_state: 1,
            vital,
            max_vital: 100,
            motivation,
            five_status: [0; 5],
            five_status_limit: [3000; 5],
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: friend,
            selected_regions: vec![1, 2, 3],
            scenario_pt: 0,
            feeling_stock: vec![],
            super_ramen: -1,
            is_ill: false,
            race_count: 0,
            absent_persons: vec![]
        }
    }

    /// 年界 / 继承 / RMJ / turn72 例外
    #[test]
    fn test_flagged_turns() {
        let tl = vec![
            row(23, 80, 4, 0, None),
            row(46, 80, 4, 0, Some("rmj_settle(46)")),
            row(48, 80, 4, 0, Some("rmj_final(48)")),
            row(50, 80, 4, 0, None),
        ];
        let flags = flagged_turns(&tl);
        println!("flagged: {flags:?}");
        let has = |t: u32, reason: &str| {
            flags.iter().any(|f| f.turn == t && f.reason.contains(reason))
        };
        assert!(has(22, "year_boundary(24)") && has(25, "year_boundary(24)"));
        assert!(has(46, "year_boundary(48)") && has(49, "year_boundary(48)"));
        assert!(has(70, "year_boundary(72)") && has(71, "year_boundary(72)"));
        assert!(
            has(72, "year_boundary(72)"),
            "turn 72 双属性：既标记年界波动，也算进超级拉面统计"
        );
        assert!(has(73, "year_boundary(72)"));
        assert!(has(30, "inherit") && has(54, "inherit"));
        assert!(has(2, "region_select(y1)") && has(3, "region_select(y1)"), "开局第 1 年地区选择");
        assert!(has(46, "rmj_settle") && has(48, "rmj_final"), "RMJ 自动检测");
    }

    /// 超级拉面期统计（turn ≥ 72 回合合计）
    #[test]
    fn test_super_ramen_stats() {
        let dec = vec![
            dec_row(70, Some(500.0)),
            dec_row(72, Some(-800.0)),
            dec_row(72, Some(-200.0)),
            dec_row(74, Some(-482.13)),
        ];
        let stats = super_ramen_stats(&dec);
        println!("super_ramen: {:?}", stats);
        let s = stats.expect("turn>=72 有数据");
        assert_eq!(s.by_turn.len(), 2);
        assert!((s.total - (-1482.13)).abs() < 1e-6);
        let f = super_ramen_finding(&s);
        println!("finding: {f:?}");
        assert_eq!(f.severity, "warn", "合计为负 → 判亏");
        assert!(f.evidence.contains("判亏"));

        let none = super_ramen_stats(&[dec_row(70, Some(1.0))]);
        assert!(none.is_none(), "窗口内无数据 → None");
    }

    /// 坏手法四项（含 game6234 实测形态：友人 58 耗尽 + 体力 27 / 心情 45 掉落）
    #[test]
    fn test_bad_habits() {
        // 友人次数：turn 58 达 5，此后 turn 69 体力 27
        let mut tl = vec![];
        for t in 0..=77 {
            let friend = if t >= 58 { 5 } else { 0 };
            let vital = if t == 69 { 27 } else { 80 };
            // 心情：turn 44=5 → 45 掉到 4，46-49 停在 4，50 回 5（事件恢复）
            let motivation = match t {
                t if t < 45 => 5,
                t if (45..50).contains(&t) => 4,
                _ => 5
            };
            tl.push(row(t, vital, motivation, friend, None));
        }
        let exec = ExecutionResult {
            rows: vec![crate::execution::ExecRow {
                turn: 45,
                stage: "Train".to_string(),
                ai_choice: "速训练".to_string(),
                actual_action: "剧本".to_string(),
                matches: Some(false),
                evidence: crate::execution::Evidence {
                    five_status_delta: [0; 5],
                    vital_delta: -20,
                    motivation_delta: 0,
                    race_count_delta: 0,
                    friend_outgoing_delta: 0,
                    is_ill_cured: false,
                    scenario_pt_delta: 0,
                    feeling_stock_len_delta: 1,
                    super_ramen_delta: 0
                }
            }],
            findings: vec![],
            comparable: 1,
            matched: 0
        };
        let sched = Schedule {
            mandatory_turns: vec![45],
            free_races: vec![],
            notes: vec![]
        };
        let race_history: Vec<i32> = vec![]; // turn 45 未跑赢
        let findings = bad_habits(&tl, &exec, &sched, &race_history);
        for f in &findings {
            println!("{}: turn={} sev={} ev={}", f.kind, f.turn, f.severity, f.evidence);
        }
        let kind = |k: &str| findings.iter().filter(|f| f.kind == k).count();
        assert_eq!(kind("mandatory_race_not_won"), 1, "turn 45 必赛未跑赢");
        assert_eq!(kind("friend_quota_exhausted_early"), 1, "58 耗尽 + 69 体力 27");
        assert_eq!(kind("motivation_drop_unrecovered"), 1, "45 掉心情 3 回合未恢复");
        assert_eq!(kind("train_failure_candidate"), 1, "速训练五维零增长体力 -20");
    }

    /// 测试用决策行（只填 turn / turn_delta）
    fn dec_row(turn: u32, delta: Option<f64>) -> DecRow {
        DecRow {
            file: format!("f{turn}.json"),
            turn,
            seq: 0,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![],
            chosen: Default::default(),
            t_n_raw: None,
            t_n_display: None,
            total_luck: None,
            turn_delta: delta,
            chain_len: 1,
            outcome: "calc".to_string(),
            reason: String::new(),
            step: 0
        }
    }
}
