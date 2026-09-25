//! 终局评分与等级换算（文档 §3.4）
//!
//! 快照里没有最终评分（`skillScore` 恒 0），需自行计算：**五维加权 + 技能分**。
//! 口径与 `umasim::game::Uma::calc_score` 同源（`status_final_score` 查表 +
//! `total_pt × pt_score_rate` + `skill_score`），等级换算直接复用
//! `GameConstants::get_rank_name`（与 `rank.csv` 同表）。
//!
//! ⚠ 已学技能分数无法从包内还原（快照 `skillScore` 恒 0）→ `final_score`
//! **仅供参考**：略低于实际小黑板分数，且未计入「努力家」等新状态；
//! 该前提写进 digest.context。

use umaai::protocol::GameStatusBase;
use umasim::{game::Uma, global, gamedata::GAMECONSTANTS, utils::Array5};

/// 终局评分（`Uma::calc_score` 同源口径；需已 `gdata::init`）
pub fn final_score(base: &GameStatusBase) -> i32 {
    let uma = Uma {
        five_status: base.five_status,
        five_status_limit: base.five_status_limit,
        skill_pt: base.skill_pt,
        skill_score: base.skill_score,
        total_hints: base.total_hints,
        ..Default::default()
    };
    uma.calc_score()
}

/// 评分 → 等级名（`GameConstants::get_rank_name`，与 rank.csv 同表；需已 init）
pub fn rank_name(score: i32) -> String {
    global!(GAMECONSTANTS).get_rank_name(score)
}

/// 显示值减半阈值（小黑板口径：真实值超过该值的部分减半显示）
const DISPLAY_STATUS_THRESHOLD: i32 = 1200;

/// 单维显示值换算（小黑板口径）：真实值 > 1200 时超出部分减半
///
/// `display = (real - 1200) / 2 + 1200` iff real > 1200，否则 display = real；
/// 整除向下取整。评分与运气分不受此换算影响。
pub fn display_status(real: i32) -> i32 {
    if real > DISPLAY_STATUS_THRESHOLD {
        (real - DISPLAY_STATUS_THRESHOLD) / 2 + DISPLAY_STATUS_THRESHOLD
    } else {
        real
    }
}

/// 五维数组显示值换算（逐维 [`display_status`]）
pub fn display_status_array(five: Array5) -> Array5 {
    let mut out = five;
    for v in out.iter_mut() {
        *v = display_status(*v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 显示值换算：阈值内原值、超阈值减半、数组分维独立换算
    #[test]
    fn test_display_status() {
        println!("1200 → {}", display_status(1200));
        assert_eq!(display_status(1199), 1199, "阈值内原值");
        assert_eq!(display_status(1200), 1200, "恰好阈值不减半");
        println!("3276 → {}", display_status(3276));
        assert_eq!(display_status(3276), 2238);
        assert_eq!(display_status(2326), 1763);
        assert_eq!(display_status(1702), 1451);
        assert_eq!(display_status(2084), 1642);
        let five: Array5 = [3276, 2326, 1702, 1194, 2084];
        let disp = display_status_array(five);
        println!("array {five:?} → {disp:?}");
        assert_eq!(disp, [2238, 1763, 1451, 1194, 1642], "分维独立换算");
    }

    /// 需要真实 gamedata（GAMECONSTANTS 查表口径），与项目测试同法：
    /// cwd 切到 workspace 根 + init_global
    #[test]
    fn test_final_score_and_rank() -> anyhow::Result<()> {
        let root = umasim::utils::get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        umasim::gamedata::init_global()?;
        let base = GameStatusBase {
            turn: 77,
            five_status: [1200, 1100, 1000, 900, 800],
            five_status_limit: [1500; 5],
            skill_pt: 500,
            skill_score: 0,
            total_hints: 10,
            ..Default::default()
        };
        let score = final_score(&base);
        let rank = rank_name(score);
        println!("final_score={score} rank={rank}");
        assert!(score > 0, "五维 + PT 折算应得正分");
        assert!(!rank.is_empty());
        // score_parts 口径交叉验证：pt 分量 = total_pt × pt_score_rate
        let cons = global!(GAMECONSTANTS);
        let total_pt = (500.0 + 10.0 * cons.hint_pt_rate).floor() as i32;
        let pt_part = (total_pt as f32 * cons.pt_score_rate) as i32;
        let five_part: i32 = (0..5)
            .map(|i| cons.status_final_score(base.five_status[i].min(base.five_status_limit[i])))
            .sum();
        println!("pt_part={pt_part} five_part={five_part} 合计={}", pt_part + five_part);
        assert_eq!(score, pt_part + five_part, "calc_score = pt + five（skill=0）");
        Ok(())
    }
}
