//! 实际执行动作推断（文档 §5.4、§6.9）与偏离清单（§11 步骤 4）
//!
//! 方法（§5.4）：取「回合末 train 决策点 → 下一回合首份快照」的**状态差**推断
//! 实际动作，回答「玩家有没有听 AI、听的是哪一选」。
//!
//! - **锚点**：每回合**最后一条 `decision_kind=train`** 的 calc 行。链序保证
//!   Train 阶段行在 RamenSelect 行之后（吃面路径），不吃面路径的
//!   RamenSelect 行即该回合最终项；地区选择是回合内子决策（kind≠train），
//!   自动排除（§5.3）
//! - **分类口径**（§12.5 修正方向 + game6234 实测校准）：
//!   1. 比赛 = `raceHistory` +1（跑赢）**或锚点回合 ∈ 必赛回合**——输掉的比赛
//!      `raceHistory` 不记（§6.1），用赛程表兜底（URA 决赛 73/75/77 剧本固定赛）
//!   2. 训练 = **主增量属性 ≥ 13**（成功训练 +13~+134，§6.1）——智训练**不耗
//!      体力甚至小回复**（实测 +5~+20），不能以体力负增量为必要条件；事件
//!      增益通常 <13，不与训练/休息混淆
//!   3. 休息 = 体力 +20 以上（事件给的小额属性不影响）
//! - **粒度限制**：只能识别「动作类别 + 目标属性」；事件给属性会混入训练
//!   判读（证据全量输出供 LLM 复核）
//! - **AI 自动执行局**（AIRedirector）本块仅作校验、不产生结论（§6.9）——
//!   bin 层无法判定是否自动执行，由 SKILL 层结合 context 解读

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use umasim::utils::Array5;

use crate::{decisions::DecRow, timeline::TimelineRow};

/// 五维属性名（训练动作的目标属性）
const ATTR_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];

/// 训练判定的主增量下限（成功训练 +13~+134，§6.1 实测；事件增益通常更小）
const TRAIN_MIN_DELTA: i32 = 13;

/// 休息判定的体力下限（休息 +25~+50；事件小额属性不影响）
const REST_MIN_VITAL: i32 = 20;

/// AI 建议 vs 实际执行对照行（digest execution 块）
#[derive(Debug, Clone, Serialize)]
pub struct ExecRow {
    pub turn: u32,
    /// 锚点决策的阶段（`Train`；不吃面路径为 `RamenSelect`）
    pub stage: String,
    /// AI 建议选中项描述（锚点 chosen_desc）
    pub ai_choice: String,
    /// 推断的实际动作（动作类别 + 目标属性）
    pub actual_action: String,
    /// 是否一致（`None` = ai_choice 无法映射到动作类别）
    pub matches: Option<bool>,
    /// 状态差证据（§5.4 签名表全量）
    pub evidence: Evidence,
}

/// 状态差证据（锚点快照 → 下一回合首份快照）
#[derive(Debug, Default, Clone, Serialize)]
pub struct Evidence {
    /// 五维增量
    pub five_status_delta: Array5,
    pub vital_delta: i32,
    pub motivation_delta: i32,
    /// `raceHistory` 长度增量（只记跑赢）
    pub race_count_delta: i64,
    /// 友人出行次数增量
    pub friend_outgoing_delta: i32,
    /// `isIll` true → false
    pub is_ill_cured: bool,
    /// 剧本 PT 增量
    pub scenario_pt_delta: i32,
    /// 诀窍队列长度增量
    pub feeling_stock_len_delta: i64,
    /// 超级拉面档位增量
    pub super_ramen_delta: i32,
}

/// 偏离/检查项命中（digest findings 块；execution 偏离 + M4 检查项共用）
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    /// 类型（`execution_mismatch` / `mandatory_race_not_won` / 检查项类型等）
    #[serde(rename = "type")]
    pub kind: String,
    pub turn: u32,
    /// 关联快照文件（检查项无关联文件时 `None`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// 证据摘要（人可读）
    pub evidence: String,
    pub severity: String,
}

/// 推断结果 + 一致率统计
#[derive(Debug, Default, Clone)]
pub struct ExecutionResult {
    pub rows: Vec<ExecRow>,
    pub findings: Vec<Finding>,
    /// 可比行数（ai_choice 可映射）
    pub comparable: usize,
    /// 一致行数
    pub matched: usize,
}

/// 推断主入口
///
/// - `tl`：timeline 行（须按 (turn, seq) 升序，与 `timeline::build` 产出同序）
/// - `dec`：calc 决策行（CSV 原序 = 回合链序）
/// - `race_turns`：必赛回合（+ URA 决赛固定赛）——输掉的比赛 `raceHistory`
///   不记，用赛程表兜底判「比赛」
pub fn build(tl: &[TimelineRow], dec: &[DecRow], race_turns: &[i32]) -> ExecutionResult {
    // (turn, seq) → timeline 下标
    let idx: HashMap<(u32, u32), usize> =
        tl.iter().enumerate().map(|(i, r)| ((r.turn, r.seq), i)).collect();

    // 每回合最后一条 train 决策（BTreeMap 同 key 后写覆盖 → 保留链序最后一条）
    let mut anchor_by_turn: BTreeMap<u32, &DecRow> = BTreeMap::new();
    for r in dec {
        if r.decision_kind == "train" {
            anchor_by_turn.insert(r.turn, r);
        }
    }

    let mut rows = Vec::new();
    let mut findings = Vec::new();
    let mut comparable = 0usize;
    let mut matched = 0usize;

    for (turn, anchor) in &anchor_by_turn {
        let Some(&ai) = idx.get(&(*turn, anchor.seq)) else { continue };
        // 下一回合首份快照（timeline 升序 → 从锚点起首个 turn 更大的行）
        let Some(next) = tl.iter().skip(ai).find(|r| r.turn > *turn) else { continue };
        let a = &tl[ai];

        let ev = evidence(a, next);
        let actual = classify(&ev, *turn, race_turns);
        let ai_mapped = map_choice(&anchor.chosen.desc);
        let matches = ai_mapped.map(|m| m == actual.as_str());
        if matches.is_some() {
            comparable += 1;
        }
        if matches == Some(true) {
            matched += 1;
        }
        if matches == Some(false) {
            findings.push(Finding {
                kind: "execution_mismatch".to_string(),
                turn: *turn,
                file: Some(anchor.file.clone()),
                evidence: format!(
                    "ai_choice={} actual={} five_delta={:?} vital={} friend={} race={}",
                    anchor.chosen.desc,
                    actual,
                    ev.five_status_delta,
                    ev.vital_delta,
                    ev.friend_outgoing_delta,
                    ev.race_count_delta
                ),
                severity: "info".to_string(),
            });
        }
        rows.push(ExecRow {
            turn: *turn,
            stage: anchor.stage.clone(),
            ai_choice: anchor.chosen.desc.clone(),
            actual_action: actual,
            matches,
            evidence: ev,
        });
    }

    ExecutionResult { rows, findings, comparable, matched }
}

/// 状态差证据（锚点快照 → 下一回合首份快照）
fn evidence(a: &TimelineRow, n: &TimelineRow) -> Evidence {
    let mut five = [0i32; 5];
    for i in 0..5 {
        five[i] = n.five_status[i] - a.five_status[i];
    }
    Evidence {
        five_status_delta: five,
        vital_delta: n.vital - a.vital,
        motivation_delta: n.motivation - a.motivation,
        race_count_delta: n.race_count as i64 - a.race_count as i64,
        friend_outgoing_delta: n.friend_outgoing_used - a.friend_outgoing_used,
        is_ill_cured: a.is_ill && !n.is_ill,
        scenario_pt_delta: n.scenario_pt - a.scenario_pt,
        feeling_stock_len_delta: n.feeling_stock.len() as i64 - a.feeling_stock.len() as i64,
        super_ramen_delta: n.super_ramen - a.super_ramen,
    }
}

/// 实际动作分类（优先级见模块头「分类口径」）
fn classify(ev: &Evidence, turn: u32, race_turns: &[i32]) -> String {
    let five_total: i32 = ev.five_status_delta.iter().sum();
    // ① 比赛：跑赢（raceHistory +1）或必赛回合（输掉的比赛赛程表兜底）
    if ev.race_count_delta > 0 || race_turns.contains(&(turn as i32)) {
        return "比赛".to_string();
    }
    if ev.friend_outgoing_delta > 0 {
        return "友人出行".to_string();
    }
    if ev.is_ill_cured {
        return "治病".to_string();
    }
    // ② 训练：主增量属性达训练量级（智训练不耗体力甚至小回复）
    let mut dom = 0usize;
    let mut dom_v = i32::MIN;
    for (i, &d) in ev.five_status_delta.iter().enumerate() {
        if d > dom_v {
            dom_v = d;
            dom = i;
        }
    }
    if dom_v >= TRAIN_MIN_DELTA {
        return format!("{}训练", ATTR_NAMES[dom]);
    }
    // ③ 休息：体力大幅回复（事件小额属性不影响）
    if ev.vital_delta >= REST_MIN_VITAL {
        return "休息".to_string();
    }
    if ev.motivation_delta > 0 && five_total == 0 {
        return "出行".to_string();
    }
    if ev.scenario_pt_delta != 0 || ev.feeling_stock_len_delta != 0 || ev.super_ramen_delta != 0
    {
        return "剧本".to_string();
    }
    "未知".to_string()
}

/// AI 建议描述 → 动作类别（关键词匹配，覆盖实测 desc 词表：
/// `速/耐/力/根/智训练` / `休息` / `普通出行` / `友人出行` / `比赛` / `治病` /
/// `吃面`；无法映射返回 `None` → matches = null）
fn map_choice(desc: &str) -> Option<&'static str> {
    if desc.contains("训练") {
        Some(match () {
            _ if desc.contains("速") => "速训练",
            _ if desc.contains("耐") => "耐训练",
            _ if desc.contains("力") => "力训练",
            _ if desc.contains("根") => "根训练",
            _ if desc.contains("智") => "智训练",
            _ => "训练"
        })
    } else if desc.contains("休息") {
        Some("休息")
    } else if desc.contains("友人") {
        Some("友人出行")
    } else if desc.contains("出行") || desc.contains("外出") {
        Some("出行")
    } else if desc.contains("治病") || desc.contains("治疗") {
        Some("治病")
    } else if desc.contains("比赛") || desc.contains("参赛") {
        Some("比赛")
    } else if desc.contains("吃面") {
        Some("吃面")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试 timeline 行（只填推断相关字段）
    fn tl_row(
        turn: u32,
        seq: u32,
        five: [i32; 5],
        vital: i32,
        motivation: i32,
        friend: i32,
        race: usize,
        ill: bool,
    ) -> TimelineRow {
        TimelineRow {
            turn,
            seq,
            stage: "Train".to_string(),
            reason: None,
            source: "command".to_string(),
            playing_state: 1,
            vital,
            max_vital: 100,
            motivation,
            five_status: five,
            five_status_limit: [3000; 5],
            skill_pt: 0,
            train_level_count: [1; 5],
            friend_outgoing_used: friend,
            selected_regions: vec![1, 2, 3],
            scenario_pt: 0,
            feeling_stock: vec![],
            super_ramen: -1,
            is_ill: ill,
            race_count: race,
            absent_persons: vec![],
        }
    }

    /// 构造测试决策行（train 决策）
    fn dec_row(turn: u32, seq: u32, chosen: &str) -> DecRow {
        DecRow {
            file: format!("f{turn}_{seq}.json"),
            turn,
            seq,
            stage: "Train".to_string(),
            decision_kind: "train".to_string(),
            candidates: vec![],
            chosen: crate::decisions::Chosen {
                idx: Some(0),
                desc: chosen.to_string(),
                action_luck: None,
            },
            t_n_raw: None,
            t_n_display: None,
            total_luck: None,
            turn_delta: None,
            chain_len: 1,
            outcome: "calc".to_string(),
            reason: String::new(),
            step: 0,
        }
    }

    /// 训练 / 休息 / 友人出行 / 比赛（跑赢 + 必赛兜底）/ 未映射 各分支 + findings
    #[test]
    fn test_execution_inference() {
        let tl = vec![
            // turn 5：行动前（速 100，体力 80）→ turn 6：速 +20、体力 -20 → 速训练
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 1, 2, false),
            tl_row(6, 0, [120, 100, 100, 100, 100], 60, 4, 1, 2, false),
            // turn 7 → 8：休息（体力 +30、五维不变）
            tl_row(7, 0, [120, 100, 100, 100, 100], 40, 4, 1, 2, false),
            tl_row(8, 0, [120, 100, 100, 100, 100], 70, 4, 1, 2, false),
            // turn 9 → 10：友人出行（friend +1、无五维）
            tl_row(9, 0, [120, 100, 100, 100, 100], 70, 3, 1, 2, false),
            tl_row(10, 0, [120, 100, 100, 100, 100], 70, 5, 2, 2, false),
            // turn 11 → 12：比赛跑赢（race +1）
            tl_row(11, 0, [120, 100, 100, 100, 100], 70, 4, 2, 2, false),
            tl_row(12, 0, [120, 100, 100, 100, 100], 70, 4, 2, 3, false),
            // turn 13 → 14：必赛回合输了（race 不变，赛程表兜底判比赛）
            tl_row(13, 0, [120, 100, 100, 100, 100], 70, 4, 2, 3, false),
            tl_row(14, 0, [127, 100, 100, 100, 107], 70, 4, 2, 3, false),
            // turn 15 → 16：智训练（智 +32、体力 +10 —— 智训练不耗体力）
            tl_row(15, 0, [127, 100, 100, 100, 107], 70, 4, 2, 3, false),
            tl_row(16, 0, [135, 100, 100, 100, 139], 80, 4, 2, 3, false),
        ];
        let dec = vec![
            dec_row(5, 0, "速训练"),   // 一致
            dec_row(7, 0, "休息"),     // 一致
            dec_row(9, 0, "普通出行"), // 实际友人出行 → 偏离
            dec_row(11, 0, "速训练"),  // 实际比赛 → 偏离
            dec_row(13, 0, "比赛"),    // 必赛兜底 → 一致
            dec_row(15, 0, "智训练"),  // 智训练不耗体力 → 一致
        ];
        let r = build(&tl, &dec, &[13]);
        for row in &r.rows {
            println!(
                "turn={} ai={} actual={} match={:?} ev={:?}",
                row.turn, row.ai_choice, row.actual_action, row.matches, row.evidence
            );
        }
        println!("findings: {:#?}", r.findings);
        assert_eq!(r.rows.len(), 6);
        assert_eq!(r.rows[0].actual_action, "速训练");
        assert_eq!(r.rows[0].matches, Some(true));
        assert_eq!(r.rows[1].actual_action, "休息");
        assert_eq!(r.rows[1].matches, Some(true));
        assert_eq!(r.rows[2].actual_action, "友人出行");
        assert_eq!(r.rows[2].matches, Some(false), "普通出行 vs 友人出行应偏离");
        assert_eq!(r.rows[3].actual_action, "比赛");
        assert_eq!(r.rows[3].matches, Some(false), "速训练 vs 比赛应偏离");
        assert_eq!(r.rows[4].actual_action, "比赛", "必赛回合兜底判比赛（输掉不记 raceHistory）");
        assert_eq!(r.rows[4].matches, Some(true));
        assert_eq!(r.rows[5].actual_action, "智训练", "智训练不耗体力（vital +10）仍判训练");
        assert_eq!(r.rows[5].matches, Some(true));
        assert_eq!(r.comparable, 6);
        assert_eq!(r.matched, 4);
        assert_eq!(r.findings.len(), 2);
        assert!(r.findings.iter().all(|f| f.kind == "execution_mismatch"));
    }

    /// 锚点规则：同回合多条 train 行取最后一条（RamenSelect 初判 → Train 最终）
    #[test]
    fn test_anchor_last_train_row() {
        let tl = vec![
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(5, 1, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(6, 0, [130, 100, 100, 100, 100], 55, 4, 0, 0, false),
        ];
        let mut first = dec_row(5, 0, "耐训练"); // RamenSelect 初判（seq 0）
        first.stage = "RamenSelect".to_string();
        let final_row = dec_row(5, 1, "速训练"); // Train 最终（seq 1）
        let r = build(&tl, &[first, final_row], &[]);
        println!("锚点: ai={} stage={}", r.rows[0].ai_choice, r.rows[0].stage);
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].ai_choice, "速训练", "取链序最后一条 train");
        assert_eq!(r.rows[0].stage, "Train");
        assert_eq!(r.rows[0].matches, Some(true));
    }

    /// ai_choice 无法映射 → matches = None（不计入一致率）
    #[test]
    fn test_unmappable_choice() {
        let tl = vec![
            tl_row(5, 0, [100, 100, 100, 100, 100], 80, 4, 0, 0, false),
            tl_row(6, 0, [120, 100, 100, 100, 100], 60, 4, 0, 0, false),
        ];
        let dec = vec![dec_row(5, 0, "超级迷惑操作")];
        let r = build(&tl, &dec, &[]);
        println!("未映射: match={:?}", r.rows[0].matches);
        assert_eq!(r.rows[0].matches, None);
        assert_eq!(r.comparable, 0);
        assert_eq!(r.matched, 0);
        assert!(r.findings.is_empty(), "未映射不产生 findings");
    }
}
