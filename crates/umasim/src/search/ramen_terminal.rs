//! 拉面剧本的终局多维记录
//!
//! 维度选取围绕一个诊断目标：**属性控制能力**。顶级人类玩家的终局面板是五维
//! 同时贴近上限，而手写策略倾向把两维堆到顶、另外两维荒废。要把这件事量化，
//! 光有总分不够，得同时看到「每一维实际到了多少」「离上限还差多少」。
//!
//! # 为什么没有「溢出浪费」维度
//!
//! [`Uma::add_value`](crate::game::Uma::add_value) 每次加属性都当场
//! `.min(five_status_limit[i])` 截断，因此终局 `five_status` **不可能**超过上限，
//! `max(0, five_status - limit)` 恒为 0。真正的「被截掉多少训练收益」是状态层
//! 的观测能力，须在 `add_value` 内部于截断前累加，无法从终局值反推。
//! 本模块不提供恒零的伪指标。
//!
//! # ⚠ 维度已冻结（2026-08-26）
//!
//! 合作伙伴用这些维度做手写策略的前后对比。**增删、重命名或重排都会让他此前
//! 记录的读数不可比**，等于把历史诊断数据作废。
//! [`FROZEN_DIM_KEYS`] 与 `test_dim_keys_frozen` 是这条约定的守门：改动维度必须
//! 先与使用方约定，再同步更新那张表，让 diff 显式暴露出来。

use crate::{
    game::ramen::RamenGame,
    gamedata::GAMECONSTANTS,
    global,
    search::terminal::define_terminal_record
};


define_terminal_record! {
    /// 一次拉面 rollout 的终局事实
    ///
    /// 全部为未归一化原值，量纲保持人类可读；NN 所需的归一化定长张量属独立的
    /// 版本化编码步骤，不在此处混合。
    RamenTerminal,
    /// 拉面终局统计（按候选累加）
    RamenTerminalStats {
        speed_score => { key: "speed_score", label: "速度评分分量", unit: "score" },
        stamina_score => { key: "stamina_score", label: "耐力评分分量", unit: "score" },
        power_score => { key: "power_score", label: "力量评分分量", unit: "score" },
        guts_score => { key: "guts_score", label: "根性评分分量", unit: "score" },
        wisdom_score => { key: "wisdom_score", label: "智力评分分量", unit: "score" },
        skill_score => { key: "skill_score", label: "技能评分", unit: "score" },
        pt_score => { key: "pt_score", label: "PT 折算评分", unit: "score" },

        speed_final => { key: "speed_final", label: "最终速度", unit: "status" },
        stamina_final => { key: "stamina_final", label: "最终耐力", unit: "status" },
        power_final => { key: "power_final", label: "最终力量", unit: "status" },
        guts_final => { key: "guts_final", label: "最终根性", unit: "status" },
        wisdom_final => { key: "wisdom_final", label: "最终智力", unit: "status" },

        speed_headroom => { key: "speed_headroom", label: "速度距上限", unit: "status" },
        stamina_headroom => { key: "stamina_headroom", label: "耐力距上限", unit: "status" },
        power_headroom => { key: "power_headroom", label: "力量距上限", unit: "status" },
        guts_headroom => { key: "guts_headroom", label: "根性距上限", unit: "status" },
        wisdom_headroom => { key: "wisdom_headroom", label: "智力距上限", unit: "status" },

        scenario_pt_y1 => { key: "scenario_pt_y1", label: "第一年剧本 PT", unit: "pt" },
        scenario_pt_y2 => { key: "scenario_pt_y2", label: "第二年剧本 PT", unit: "pt" },
        scenario_pt_y3 => { key: "scenario_pt_y3", label: "第三年剧本 PT", unit: "pt" },

        rmj_ok_y1 => { key: "rmj_ok_y1", label: "第一年 RMJ 达成", unit: "flag" },
        rmj_ok_y2 => { key: "rmj_ok_y2", label: "第二年 RMJ 达成", unit: "flag" },
        rmj_ok_y3 => { key: "rmj_ok_y3", label: "第三年 RMJ 达成", unit: "flag" },

        status_gap_sum => { key: "status_gap_sum", label: "五维评分缺口之和", unit: "score" },
        status_gap_spread => { key: "status_gap_spread", label: "五维评分缺口极差", unit: "score" }
    }
}

/// 冻结的维度键与顺序（2026-08-26 起）
///
/// 与 [`RamenTerminalStats::visit`] 的遍历结果逐项对齐。**不要为了让测试变绿
/// 而顺手改这张表**——它存在的意义正是让维度变更无法悄悄发生。
pub const FROZEN_DIM_KEYS: [&str; 25] = [
    "speed_score",
    "stamina_score",
    "power_score",
    "guts_score",
    "wisdom_score",
    "skill_score",
    "pt_score",
    "speed_final",
    "stamina_final",
    "power_final",
    "guts_final",
    "wisdom_final",
    "speed_headroom",
    "stamina_headroom",
    "power_headroom",
    "guts_headroom",
    "wisdom_headroom",
    "scenario_pt_y1",
    "scenario_pt_y2",
    "scenario_pt_y3",
    "rmj_ok_y1",
    "rmj_ok_y2",
    "rmj_ok_y3",
    "status_gap_sum",
    "status_gap_spread"
];

impl RamenTerminal {
    /// 从拉面终局局面提取原始事实
    ///
    /// 评分七分量直接取 [`Uma::score_parts`](crate::game::Uma::score_parts)，与
    /// `calc_score()` 逐位可加回去，不另造一套平行算法。
    pub fn from_game(game: &RamenGame) -> Self {
        let cons = global!(GAMECONSTANTS);
        let uma = &game.uma;
        let parts = uma.score_parts();

        // 查表下标必须与 score_parts 同一套截断口径，否则七分量对不上总分
        let lookup = |v: i32| -> i32 { cons.status_final_score(v) };

        let mut headroom = [0.0f64; 5];
        let mut final_status = [0.0f64; 5];
        // 单维「评分缺口」= 把这一维补满到上限还能拿到的分数。
        //
        // 两个归约都是对局面**非线性**的量，必须在这里算完再平均：
        // - `sum`    还有多少分没吃到（能把「五维全废」和「只有一维废」分开）
        // - `spread` 极差，即属性控制能力本身。全员同等未完成 → 接近 0；
        //            两维堆爆、两维荒废 → 很大。
        //
        // 曾用过 `max`（最差那一维的缺口），但它答的是另一个问题：
        // 「五维全 400」与「四维满 + 一维 400」的 max 相同，恰好分不开
        // 本模块要诊断的那件事。且评分表是凸的，高段边际更陡，max 会把
        // 锅甩给快满的那一维。
        let mut gap_sum = 0.0f64;
        let mut gap_min = f64::MAX;
        let mut gap_max = 0.0f64;
        for i in 0..5 {
            // 与 score_parts / headroom 同一套截断口径，保证 final + headroom == limit
            let status = uma.five_status[i].min(uma.five_status_limit[i]).max(0);
            let limit = uma.five_status_limit[i].max(0);
            final_status[i] = status as f64;
            headroom[i] = (limit - status) as f64;
            let gap = (lookup(limit) - lookup(status)).max(0) as f64;
            gap_sum += gap;
            gap_min = gap_min.min(gap);
            gap_max = gap_max.max(gap);
        }
        let gap_spread = gap_max - gap_min;

        // 局末 live `scenario_pt` 恒为 0（turn 72–77 不再吃面），必须读逐年归档
        let yearly = game.ramen.yearly_scenario_pt;

        // 直接读规则层已经算完的结算结果，**不要**用阈值重算一遍。
        // `check_rmj` 的阈值来自 `RAMENDATA.ramen_success_pt`（可随数据更新），
        // 且第 3 年还有 ≥5000 的大成功分支；在这里复制一份判据等于给结算规则
        // 造第二数据源，数据一改就变成会说谎的仪表。
        let rmj_ok = |y: usize| -> f64 {
            if game.ramen.rmj_results.get(y).copied().unwrap_or(false) {
                1.0
            } else {
                0.0
            }
        };

        Self {
            speed_score: parts.five_status[0] as f64,
            stamina_score: parts.five_status[1] as f64,
            power_score: parts.five_status[2] as f64,
            guts_score: parts.five_status[3] as f64,
            wisdom_score: parts.five_status[4] as f64,
            skill_score: parts.skill as f64,
            pt_score: parts.pt as f64,

            speed_final: final_status[0],
            stamina_final: final_status[1],
            power_final: final_status[2],
            guts_final: final_status[3],
            wisdom_final: final_status[4],

            speed_headroom: headroom[0],
            stamina_headroom: headroom[1],
            power_headroom: headroom[2],
            guts_headroom: headroom[3],
            wisdom_headroom: headroom[4],

            scenario_pt_y1: yearly[0] as f64,
            scenario_pt_y2: yearly[1] as f64,
            scenario_pt_y3: yearly[2] as f64,

            rmj_ok_y1: rmj_ok(0),
            rmj_ok_y2: rmj_ok(1),
            rmj_ok_y3: rmj_ok(2),

            status_gap_sum: gap_sum,
            status_gap_spread: gap_spread
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::{
        game::ramen::rules::check_rmj,
        gamedata::init_global,
        search::{TerminalRecord, TerminalStats},
        utils::{Checks, get_workspace_root}
    };

    /// 载入游戏常量：查表分与上限都来自 gamedata，测试工作目录须为 workspace 根
    fn setup() -> Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        init_global()
    }

    /// 终局提取的四组不变量
    #[test]
    fn test_ramen_terminal_from_game() -> Result<()> {
        setup()?;
        let mut c = Checks::new();

        let mut game = RamenGame::default();
        game.uma.five_status_limit = [1600, 1500, 1400, 1400, 1500];
        // 速度贴顶、智力荒废：正是要诊断的那种面板
        game.uma.five_status = [1600, 1200, 1100, 900, 400];
        game.uma.skill_score = 3000;
        game.uma.skill_pt = 800;
        game.ramen.yearly_scenario_pt = [1600, 2900, 3500];
        // RMJ 结果由规则层写入，不在此处重算
        game.ramen.scenario_pt = 1600;
        check_rmj(&mut game.ramen, 0);
        game.ramen.scenario_pt = 2900;
        check_rmj(&mut game.ramen, 1);
        game.ramen.scenario_pt = 3500;
        check_rmj(&mut game.ramen, 2);

        let t = RamenTerminal::from_game(&game);

        // 七分量之和必须逐位等于 calc_score()，否则归因是假的
        let parts_sum =
            t.speed_score + t.stamina_score + t.power_score + t.guts_score + t.wisdom_score + t.skill_score + t.pt_score;
        println!("七分量之和 = {parts_sum}, calc_score() = {}", game.uma.calc_score());
        c.check(parts_sum as i32 == game.uma.calc_score(), "七分量之和逐位等于 calc_score()");

        // final + headroom == limit（两者共用同一套截断口径）
        let finals = [t.speed_final, t.stamina_final, t.power_final, t.guts_final, t.wisdom_final];
        let heads = [
            t.speed_headroom,
            t.stamina_headroom,
            t.power_headroom,
            t.guts_headroom,
            t.wisdom_headroom
        ];
        println!("最终五维 = {finals:?}");
        println!("距上限   = {heads:?}");
        let invariant = (0..5).all(|i| finals[i] + heads[i] == game.uma.five_status_limit[i] as f64);
        c.check(invariant, "final + headroom == limit");
        c.check(heads[0] == 0.0, "贴顶维 headroom 为 0");
        c.check(heads[4] == 1100.0, "荒废维 headroom 为 1100");

        // RMJ 必须与规则层结算结果一致，不是拿 PT 重算
        println!(
            "RMJ 记录 = [{}, {}, {}], 规则层 rmj_results = {:?}",
            t.rmj_ok_y1, t.rmj_ok_y2, t.rmj_ok_y3, game.ramen.rmj_results
        );
        let from_rules: Vec<f64> = game
            .ramen
            .rmj_results
            .iter()
            .map(|&ok| if ok { 1.0 } else { 0.0 })
            .collect();
        c.check(
            from_rules == vec![t.rmj_ok_y1, t.rmj_ok_y2, t.rmj_ok_y3],
            "rmj_ok_* 与规则层 rmj_results 一致"
        );

        // 缺口两维：一维贴顶一维荒废时，极差必须显著大于 0
        println!("缺口之和 = {}, 缺口极差 = {}", t.status_gap_sum, t.status_gap_spread);
        c.check(t.status_gap_sum > 0.0, "缺口之和 > 0");
        c.check(t.status_gap_spread > 0.0, "面板不均衡时缺口极差 > 0");

        c.finish()
    }

    /// 极差能分开「全员同等未完成」与「两维堆爆两维荒废」——这是 max 做不到的
    #[test]
    fn test_gap_spread_separates_balance() -> Result<()> {
        setup()?;
        let mut c = Checks::new();
        let limit = [1600, 1500, 1400, 1400, 1500];

        let build = |five: [i32; 5]| -> RamenTerminal {
            let mut g = RamenGame::default();
            g.uma.five_status_limit = limit;
            g.uma.five_status = five;
            RamenTerminal::from_game(&g)
        };

        // 均衡：五维同等未完成
        let balanced = build([1100, 1000, 900, 900, 1000]);
        // 失衡：两维贴顶、两维荒废，总缺口刻意造得接近
        let skewed = build([1600, 1500, 400, 400, 1500]);

        println!(
            "均衡 sum={:.0} spread={:.0}",
            balanced.status_gap_sum, balanced.status_gap_spread
        );
        println!("失衡 sum={:.0} spread={:.0}", skewed.status_gap_sum, skewed.status_gap_spread);

        c.check(
            skewed.status_gap_spread > balanced.status_gap_spread,
            "失衡面板的缺口极差大于均衡面板"
        );

        // 全员荒废 vs 单维荒废：sum 必须能分开（这正是 max 分不开的那一对）
        let all_low = build([400, 400, 400, 400, 400]);
        let one_low = build([1600, 1500, 1400, 1400, 400]);
        println!("全员荒废 sum={:.0}", all_low.status_gap_sum);
        println!("单维荒废 sum={:.0}", one_low.status_gap_sum);
        c.check(all_low.status_gap_sum > one_low.status_gap_sum, "缺口之和能分开全员荒废与单维荒废");

        c.finish()
    }

    /// 阈值均值化陷阱：达成率必须逐 rollout 归约后再平均
    #[test]
    fn test_threshold_must_be_reduced_per_rollout() -> Result<()> {
        setup()?;
        let mut c = Checks::new();
        let mut stats = RamenTerminalStats::default();

        // 两次 rollout：PT 分别为 1200 与 1840，均值 1520 已越过第 1 年阈值 1500，
        // 但真实达成率只有 50%
        for (pt, ok) in [(1200.0, 0.0), (1840.0, 1.0)] {
            let mut t = RamenTerminal::from_game(&RamenGame::default());
            t.scenario_pt_y1 = pt;
            t.rmj_ok_y1 = ok;
            t.accumulate_into(&mut stats);
        }

        println!("PT 均值 = {}", stats.scenario_pt_y1.mean());
        println!("达成率   = {}", stats.rmj_ok_y1.mean());
        c.check(stats.scenario_pt_y1.mean() == 1520.0, "PT 均值为 1520（越过阈值 1500）");
        c.check(stats.rmj_ok_y1.mean() == 0.5, "达成率为 0.5，而非从均值反推的 1.0");

        c.finish()
    }

    /// 维度冻结守门：键名与顺序必须与 [`FROZEN_DIM_KEYS`] 完全一致
    ///
    /// 这条测试红了不代表代码坏了，而是**有人动了对外契约**：先与使用方确认，
    /// 再同步 `FROZEN_DIM_KEYS`。
    #[test]
    fn test_dim_keys_frozen() -> Result<()> {
        let mut c = Checks::new();
        let stats = RamenTerminalStats::default();
        let mut keys: Vec<&'static str> = Vec::new();
        stats.visit(&mut |m| keys.push(m.key));

        println!("当前 {} 维, 冻结表 {} 维", keys.len(), FROZEN_DIM_KEYS.len());
        for (i, (now, frozen)) in keys.iter().zip(FROZEN_DIM_KEYS.iter()).enumerate() {
            if now != frozen {
                println!("  #{i} 不一致: 当前 {now} / 冻结 {frozen}");
            }
        }
        let added: Vec<&str> = keys.iter().filter(|k| !FROZEN_DIM_KEYS.contains(k)).copied().collect();
        let removed: Vec<&str> = FROZEN_DIM_KEYS.iter().filter(|k| !keys.contains(k)).copied().collect();
        println!("新增 = {added:?}");
        println!("移除 = {removed:?}");

        c.check(keys.as_slice() == FROZEN_DIM_KEYS.as_slice(), "维度键名与顺序与冻结表一致");
        c.finish()
    }

    /// 名称与数值同源生成：遍历应覆盖全部维度且不重键
    #[test]
    fn test_visit_covers_all_dims() -> Result<()> {
        let mut c = Checks::new();
        let stats = RamenTerminalStats::default();
        let mut keys = Vec::new();
        stats.visit(&mut |m| keys.push((m.key, m.label, m.unit)));

        for (k, l, u) in &keys {
            println!("  {k:24} {l:16} [{u}]");
        }
        println!("维度数 = {}", keys.len());

        c.check(keys.len() == 25, "维度数为 25");
        c.check(keys.len() == stats.dim_count(), "dim_count() 与 visit 次数一致");

        let mut sorted: Vec<&str> = keys.iter().map(|(k, _, _)| *k).collect();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        c.check(sorted.len() == before, "无重复键");

        // 量纲只允许这几种：诊断出口按量纲决定显示阈值
        let bad: Vec<&str> = keys
            .iter()
            .map(|(_, _, u)| *u)
            .filter(|u| !matches!(*u, "score" | "status" | "pt" | "flag"))
            .collect();
        println!("未知量纲 = {bad:?}");
        c.check(bad.is_empty(), "量纲全部在已知集合内");

        c.finish()
    }
}
