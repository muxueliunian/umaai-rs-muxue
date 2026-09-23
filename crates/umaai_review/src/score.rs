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
use umasim::{game::Uma, global, gamedata::GAMECONSTANTS};

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

#[cfg(test)]
mod tests {
    use super::*;

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
