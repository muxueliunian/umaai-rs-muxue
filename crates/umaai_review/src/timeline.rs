//! 快照解析 → digest.timeline（文档 §3.5 timeline 块、§5 输入契约）
//!
//! 快照只走 `GameStatusRamen` 反序列化取字段（**不调 `into_game`**、无需构建
//! `RamenGame` / 初始化 gamedata —— §3.3 复用点 1）。
//!
//! 阶段判定 [`stage_of`] 复刻 `record::classify_begin_reason` 的判定顺序 +
//! `protocol/ramen.rs` 文件头的 stage dispatch 表（顺序一致，勿单独改动）。

use serde::Serialize;
use umasim::utils::Array5;

use crate::pack::SnapEntry;
use umaai::protocol::ramen::GameStatusRamen;

/// timeline 单行（快照状态序列；锚点语义见 §5.3）
#[derive(Debug, Clone, Serialize)]
pub struct TimelineRow {
    /// 回合
    pub turn: u32,
    /// 同回合写入序号（0 = 无后缀）
    pub seq: u32,
    /// 派发阶段（`Train` / `RamenSelect` / `RegionSelect`）；不派发的快照为 `Begin`
    pub stage: String,
    /// `Begin`（skip）原因（与 `record::classify_begin_reason` 同口径）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// `baseGame.source`（`command` / `event` / `special`）
    pub source: String,
    /// `baseGame.playing_state`
    pub playing_state: i32,
    pub vital: i32,
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    pub five_status: Array5,
    /// 五维显示值（小黑板口径；真实值 > 1200 的部分减半，见 `score::display_status_array`）
    pub five_status_display: Array5,
    pub five_status_limit: Array5,
    pub skill_pt: i32,
    pub train_level_count: Array5,
    /// 友人出行已用次数
    pub friend_outgoing_used: i32,
    /// 当年已选地区（region_id；开局空数组）
    pub selected_regions: Vec<i32>,
    /// 当前累计剧本 PT
    pub scenario_pt: i32,
    /// 诀窍队列（`feeling_stock`；吃面动作签名与吃面节奏检查用）
    pub feeling_stock: Vec<i32>,
    /// 超级拉面档位（-1 未选 / 0/1/2）
    pub super_ramen: i32,
    pub is_ill: bool,
    /// 跑赢的比赛数（`raceHistory` 只记跑赢的比赛）
    pub race_count: usize,
    /// 缺席人物索引 0-5（§5.5 反推；`personDistribution` 不可判定时为空）
    pub absent_persons: Vec<u32>,
}

/// timeline 解析结果
#[derive(Debug, Default)]
pub struct TimelineResult {
    /// 全部快照的 timeline 行（按 (turn, seq) 升序，与 pack.snaps 同序）
    pub rows: Vec<TimelineRow>,
    /// 反序列化失败的快照（file, 错误信息）→ coverage.parse_error
    pub parse_errors: Vec<(String, String)>,
    /// 首个解析成功的快照（取卡组等开局信息；无快照时 `None`）
    pub first_status: Option<GameStatusRamen>,
    /// 末个解析成功的快照（终局评分 / raceHistory；无快照时 `None`）
    pub last_status: Option<GameStatusRamen>,
}

/// 阶段判定（返回 (stage, skip 原因)；reason = `Some` 即不派发的 `Begin` 快照）
///
/// 判定顺序与 `record::classify_begin_reason` 完全一致（data_incomplete →
/// event → ps=5 → RMJ 46/48 → 超级拉面丢包），其后按协议 dispatch 表：
/// turn ≤ 1 → `Train`；ps=45 → `RegionSelect`；command+有效果（已吃面）→
/// `Train`；command+无效果 → `RamenSelect`；special → `RegionSelect`。
fn stage_of(s: &GameStatusRamen) -> (&'static str, Option<&'static str>) {
    let base = &s.base_game;
    let turn = base.turn;
    let ps = base.playing_state;
    let source = base.source.as_deref().unwrap_or("");
    let selected_all_zero = s.ramen.selected_regions.iter().all(|&r| r == 0);
    let effect_empty = s.ramen.active_effect_array.is_empty();
    if (2..=71).contains(&turn) && selected_all_zero {
        return ("Begin", Some("data_incomplete(selected_regions=0)"));
    }
    if source == "event" {
        return ("Begin", Some("event"));
    }
    if ps == 5 {
        return ("Begin", Some("playing_state=5(event)"));
    }
    if ps == 46 {
        return ("Begin", Some("rmj_settle(46)"));
    }
    if ps == 48 {
        return ("Begin", Some("rmj_final(48)"));
    }
    if turn >= 72 && effect_empty {
        return ("Begin", Some("super_ramen_drop(active_effect empty)"));
    }
    // stage dispatch（protocol/ramen.rs 文件头规则表）
    if turn <= 1 && ps == 1 {
        return ("Train", None);
    }
    if ps == 45 {
        return ("RegionSelect", None);
    }
    if source == "command" && !effect_empty {
        return ("Train", None);
    }
    if source == "command" && effect_empty {
        return ("RamenSelect", None);
    }
    if source == "special" {
        return ("RegionSelect", None);
    }
    ("Begin", Some("begin_unclassified"))
}

/// §5.5 人物缺席反推：persons 索引 0-5 不出现在任何训练位 → absent
///
/// - `personDistribution` 为空数组 / 全占位（-1）→ 无法判定，返回空（**不是**全员缺席）
/// - 索引 8 是 NPC 占位符（同位多个 8 = 多个 NPC），不影响 0-5 判定
pub fn absent_persons(dist: &[Vec<i32>]) -> Vec<u32> {
    let any_person = dist
        .iter()
        .any(|slot| slot.iter().any(|&p| p >= 0));
    if !any_person {
        return Vec::new();
    }
    (0..6u32)
        .filter(|i| {
            let idx = *i as i32;
            !dist.iter().any(|slot| slot.contains(&idx))
        })
        .collect()
}

/// 快照序列 → timeline（逐份反序列化；失败进 parse_errors 不阻断）
pub fn build(snaps: &[SnapEntry]) -> TimelineResult {
    let mut rows = Vec::with_capacity(snaps.len());
    let mut parse_errors = Vec::new();
    let mut first_status: Option<GameStatusRamen> = None;
    let mut last_status: Option<GameStatusRamen> = None;
    for s in snaps {
        let text = String::from_utf8_lossy(&s.bytes);
        match serde_json::from_str::<GameStatusRamen>(&text) {
            Ok(st) => {
                let (stage, reason) = stage_of(&st);
                let base = &st.base_game;
                rows.push(TimelineRow {
                    turn: s.turn,
                    seq: s.seq,
                    stage: stage.to_string(),
                    reason: reason.map(String::from),
                    source: base.source.clone().unwrap_or_default(),
                    playing_state: base.playing_state,
                    vital: base.vital,
                    max_vital: base.max_vital,
                    motivation: base.motivation,
                    five_status: base.five_status,
                    five_status_display: crate::score::display_status_array(base.five_status),
                    five_status_limit: base.five_status_limit,
                    skill_pt: base.skill_pt,
                    train_level_count: base.train_level_count,
                    friend_outgoing_used: base.friend_outgoing_used,
                    selected_regions: st.ramen.selected_regions.clone(),
                    scenario_pt: st.ramen.scenario_pt,
                    feeling_stock: st.ramen.feeling_stock.clone(),
                    super_ramen: st.ramen.super_ramen,
                    is_ill: base.is_ill,
                    race_count: base.race_history.len(),
                    absent_persons: absent_persons(&base.person_distribution),
                });
                if first_status.is_none() {
                    first_status = Some(st.clone());
                }
                last_status = Some(st);
            }
            Err(e) => parse_errors.push((s.file.clone(), e.to_string())),
        }
    }
    TimelineResult { rows, parse_errors, first_status, last_status }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造最小可解析快照 JSON（`GameStatusBase` 必填字段全覆盖）
    ///
    /// 参数：回合 / source / playing_state / active_effect 数量 / selected_regions
    fn fixture(turn: i32, source: &str, ps: i32, effects: usize, regions: &[i32]) -> String {
        let effect_arr: Vec<String> = (0..effects)
            .map(|i| format!(r#"{{"category":1,"id":{i},"value":1}}"#))
            .collect();
        format!(
            r#"{{
                "baseGame": {{
                    "scenarioId": 14, "umaId": 112402, "umaStar": 5, "turn": {turn},
                    "vital": 80, "maxVital": 100, "motivation": 4,
                    "fiveStatus": [100, 200, 300, 400, 500],
                    "fiveStatusLimit": [1200, 1200, 1200, 1200, 1200],
                    "skillPt": 10, "skillScore": 0, "totalHints": 5,
                    "trainLevelCount": [1, 2, 3, 4, 5],
                    "ptScoreRate": 2.0, "failureRateBias": 0,
                    "isIll": false, "isQieZhe": false, "isAiJiao": false,
                    "isPositiveThinking": false, "isRefreshMind": false, "isLucky": false,
                    "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                    "cardId": [302424, 302894, 303044, 302924, 303024, 303054],
                    "persons": [], "personDistribution": [[], [], [], [], []],
                    "lockedTrainingId": -1,
                    "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                    "friend_stage": 0, "friend_outgoingUsed": 0,
                    "playing_state": {ps}, "raceHistory": [11, 28], "story": null,
                    "source": "{source}"
                }},
                "ramen": {{
                    "active_effect_array": [{arr}],
                    "selected_regions": {regions:?}
                }}
            }}"#,
            regions = regions,
            arr = effect_arr.join(","),
            turn = turn,
            ps = ps,
            source = source
        )
    }

    /// 快照转 SnapEntry（走 build 全链路）
    fn snap(file: &str, turn: u32, seq: u32, json: &str) -> SnapEntry {
        SnapEntry {
            file: file.to_string(),
            game: 6234,
            turn,
            seq,
            bytes: json.as_bytes().to_vec(),
        }
    }

    /// stage dispatch 全分支（顺序与 classify_begin_reason / 协议表一致）
    #[test]
    fn test_stage_dispatch() {
        let cases: Vec<(&str, i32, i32, usize, &[i32], &str, Option<&str>)> = vec![
            // (source, turn, ps, effects, regions, 期望 stage, 期望 reason)
            ("command", 0, 1, 0, &[], "Train", None),                      // turn≤1 剧本未启动
            ("command", 5, 1, 0, &[1, 2, 3], "RamenSelect", None),         // 未吃面
            ("command", 5, 1, 2, &[1, 2, 3], "Train", None),               // 已吃面
            ("command", 23, 45, 0, &[1, 2, 3], "RegionSelect", None),      // 地区选择
            ("special", 23, 45, 0, &[1, 2, 3], "RegionSelect", None),
            ("event", 10, 1, 0, &[1, 2, 3], "Begin", Some("event")),
            ("command", 10, 5, 0, &[1, 2, 3], "Begin", Some("playing_state=5(event)")),
            ("command", 24, 46, 0, &[1, 2, 3], "Begin", Some("rmj_settle(46)")),
            ("command", 24, 48, 0, &[1, 2, 3], "Begin", Some("rmj_final(48)")),
            ("command", 5, 1, 0, &[], "Begin", Some("data_incomplete(selected_regions=0)")),
            ("command", 74, 1, 0, &[1, 2, 3], "Begin", Some("super_ramen_drop(active_effect empty)")),
        ];
        for (src, turn, ps, eff, regions, want_stage, want_reason) in cases {
            let json = fixture(turn, src, ps, eff, regions);
            let st: GameStatusRamen = serde_json::from_str(&json).unwrap();
            let (stage, reason) = stage_of(&st);
            println!("turn={turn} src={src} ps={ps} eff={eff} → {stage} / {reason:?}");
            assert_eq!(stage, want_stage, "turn={turn} src={src} ps={ps}");
            assert_eq!(reason, want_reason, "turn={turn} src={src} ps={ps}");
        }
    }

    /// build 全链路：行字段 + 解析失败隔离 + 首末快照
    #[test]
    fn test_build_rows() {
        let snaps = vec![
            snap("game6234_turn0.json", 0, 0, &fixture(0, "command", 1, 0, &[])),
            snap("game6234_turn5.json", 5, 0, &fixture(5, "command", 1, 0, &[1, 2, 3])),
            snap("game6234_turn6.json", 6, 0, "not json"),
        ];
        let r = build(&snaps);
        println!("rows={} parse_errors={:?} first={:?}",
            r.rows.len(), r.parse_errors, r.first_status.as_ref().map(|s| s.base_game.turn));
        assert_eq!(r.rows.len(), 2, "坏快照不应进 rows");
        assert_eq!(r.parse_errors.len(), 1);
        assert_eq!(r.rows[0].turn, 0);
        assert_eq!(r.rows[0].race_count, 2, "raceHistory=[11,28]");
        assert_eq!(r.rows[0].stage, "Train");
        assert_eq!(r.rows[0].five_status_display, [100, 200, 300, 400, 500], "阈值内显示值 = 真实值");
        assert_eq!(r.rows[1].selected_regions, vec![1, 2, 3]);
        assert_eq!(r.rows[1].scenario_pt, 0);
        assert!(r.first_status.is_some() && r.last_status.is_some());
        assert_eq!(r.last_status.as_ref().unwrap().base_game.turn, 5);
    }

    /// 人物缺席反推（含 NPC 占位 8 与全空数组容忍）
    #[test]
    fn test_absent_persons() {
        let dist = vec![
            vec![0, 8, 8],   // 速位：卡0 + 2 NPC
            vec![2],
            vec![8],
            vec![4],
            vec![],
        ];
        let absent = absent_persons(&dist);
        println!("absent = {absent:?}");
        assert_eq!(absent, vec![1, 3, 5], "0/2/4 在场，1/3/5 缺席；8 是 NPC 不算");
        println!("全空 = {:?}", absent_persons(&[vec![], vec![], vec![], vec![], vec![]]));
        assert!(absent_persons(&[vec![], vec![], vec![], vec![], vec![]]).is_empty(), "全空 → 无法判定");
        println!("全 -1 = {:?}", absent_persons(&[vec![-1, -1], vec![-1], vec![], vec![], vec![]]));
        assert!(
            absent_persons(&[vec![-1, -1], vec![-1], vec![], vec![], vec![]]).is_empty(),
            "全占位 → 无法判定"
        );
        println!("满员 = {:?}", absent_persons(&[vec![0], vec![1], vec![2], vec![3], vec![4, 5]]));
        assert!(absent_persons(&[vec![0], vec![1], vec![2], vec![3], vec![4, 5]]).is_empty());
    }
}
