//! 赛程与自由比赛（文档 §6.3）：umaDB `races` + `freeRaces` + 快照 `raceHistory` 交叉
//!
//! - `mandatory_turns`：必赛回合（umaDB `races`，`[11,28,41,52,59,66,67]` 形态）
//! - `free_races`：自由比赛区间 + 要求次数 + 区间内**自选**比赛回合
//!   （`raceHistory` 只记**跑赢**的比赛；`raceHistory − races` = 自选）
//! - `freeRaces` 的 null / 对象 / 数组三种形态由 `UmaData` 的 serde 统一吸收
//!   （复用 `GAMEDATA` 口径；上游结构变化时先改 umasim）
//! - 不可做：评价「该选哪一场」（比赛池不在 gamedata；`preferRaces` 等为空）

use serde::Serialize;
use umasim::gamedata::UmaData;

/// 赛程块（digest schedule）
#[derive(Debug, Default, Clone, Serialize)]
pub struct Schedule {
    /// 必赛回合（升序）
    pub mandatory_turns: Vec<i32>,
    /// 自由比赛区间
    pub free_races: Vec<FreeRaceInfo>,
    pub notes: Vec<String>,
}

/// 自由比赛区间 + 实跑统计
#[derive(Debug, Clone, Serialize)]
pub struct FreeRaceInfo {
    pub start_turn: u32,
    pub end_turn: u32,
    /// 要求次数
    pub required: u32,
    /// 比赛等级（可选）
    pub grade: Option<u32>,
    /// 区间内自选比赛回合（`raceHistory` − `races`）
    pub picked_turns: Vec<i32>,
}

/// 组装赛程块
///
/// - `uma_data`：`GAMEDATA.get_uma(uma_id)`（gamedata 缺失时 `None` → 降级 + 注记）
/// - `race_history`：末份快照的 `baseGame.raceHistory`（跑赢列表）
/// - `turn`：末回合（`raceHistory` 为空的告警需 turn > 12 口径）
pub fn build(uma_data: Option<&UmaData>, race_history: &[i32], turn: i32) -> Schedule {
    let mut sched = Schedule::default();
    let Some(uma) = uma_data else {
        sched.notes.push("gamedata 缺失：赛程与自由比赛区间不可用（纯快照口径）".to_string());
        return sched;
    };
    sched.mandatory_turns = uma.races.clone();
    sched.mandatory_turns.sort();

    // raceHistory 为空（turn > 12）→ 自选统计不可信（与 parse_basegame 告警同口径）
    if turn > 12 && race_history.is_empty() {
        sched
            .notes
            .push("raceHistory 为空（turn>12）：自选比赛统计不可用，需更新小黑板插件".to_string());
    }

    for f in &uma.free_races {
        let picked: Vec<i32> = race_history
            .iter()
            .copied()
            .filter(|t| {
                (*t as u32) >= f.start_turn
                    && (*t as u32) <= f.end_turn
                    && !uma.races.contains(t)
            })
            .collect();
        sched.free_races.push(FreeRaceInfo {
            start_turn: f.start_turn,
            end_turn: f.end_turn,
            required: f.count,
            grade: f.grade,
            picked_turns: picked,
        });
    }

    // 目标赛未跑赢（§6.1 已验证检查项的数据面；findings 判定在 M4）
    let not_won: Vec<i32> = sched
        .mandatory_turns
        .iter()
        .copied()
        .filter(|t| !race_history.contains(t))
        .collect();
    if !not_won.is_empty() {
        sched.notes.push(format!("未跑赢的必赛回合: {not_won:?}"));
    }
    // 区间外自选赛（不在任何 free 区间内的额外比赛）
    // ⚠ URA 决赛 72-77 是剧本固定赛（`Uma::is_race_turn` 对 73/75/77 恒真），
    // 不在 umaDB `races` 里、也不是自选 —— 从自选判定中排除
    let in_free = |t: i32| {
        uma.free_races
            .iter()
            .any(|f| (t as u32) >= f.start_turn && (t as u32) <= f.end_turn)
    };
    let extra: Vec<i32> = race_history
        .iter()
        .copied()
        .filter(|t| {
            !uma.races.contains(t) && !in_free(*t) && !(72..=77).contains(t)
        })
        .collect();
    if !extra.is_empty() {
        sched.notes.push(format!("自由比赛区间外的自选比赛: {extra:?}"));
    }
    sched
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试用 UmaData（races + 两个 free 区间）
    fn uma_fixture() -> UmaData {
        UmaData {
            game_id: 112402,
            star: 5,
            name: "[URA]テスト".to_string(),
            five_status_bonus: [0; 5],
            five_status_initial: [0; 5],
            races: vec![11, 28, 41],
            free_races: vec![
                umasim::gamedata::FreeRaceData {
                    start_turn: 24,
                    end_turn: 47,
                    count: 2,
                    grade: None,
                    mask: 0,
                },
                umasim::gamedata::FreeRaceData {
                    start_turn: 48,
                    end_turn: 71,
                    count: 1,
                    grade: Some(1),
                    mask: 0,
                },
            ],
        }
    }

    /// 区间内 picked / 目标赛未跑赢 / 区间外自选 / gamedata 缺失降级
    #[test]
    fn test_build_schedule() {
        let uma = uma_fixture();
        // raceHistory：11 跑赢；28/41 未跑赢；30（区间1内自选）；50（区间2内自选）；
        // 73（URA 决赛固定赛，应被排除）；20（区间外自选）
        let history = vec![11, 20, 30, 50, 73];
        let sched = build(Some(&uma), &history, 70);
        println!(
            "mandatory={:?}\nfree={:#?}\nnotes={:?}",
            sched.mandatory_turns, sched.free_races, sched.notes
        );
        assert_eq!(sched.mandatory_turns, vec![11, 28, 41]);
        assert_eq!(sched.free_races[0].picked_turns, vec![30]);
        assert_eq!(sched.free_races[1].picked_turns, vec![50]);
        assert!(sched.notes.iter().any(|n| n.contains("28")), "28 未跑赢应注记");
        assert!(sched.notes.iter().any(|n| n.contains("20")), "20 区间外自选应注记");
        assert!(
            !sched.notes.iter().any(|n| n.contains("73")),
            "73 是 URA 决赛固定赛，不应算自选"
        );

        // gamedata 缺失 → 降级注记
        let degraded = build(None, &history, 70);
        println!("降级 notes: {:?}", degraded.notes);
        assert!(degraded.mandatory_turns.is_empty());
        assert_eq!(degraded.notes.len(), 1);
    }
}
