//! 仅接管外层 `RegionSelect` 的网络决策器（研究工具用）
//!
//! 客户端的网络决策已统一到 `ramen_trainer_policy`（见 [`crate::ramen_nn`]），本模块
//! **不再从 `GameConfig` 读决策来源配置**，只给 `bin/ramen_client_game_bench` 等研究入口
//! 提供「除地区外全部走搜索、外层三次 `RegionSelect` 交给网络」的决策器，模型由调用方
//! 自己加载（命令行 `--model`）。
//!
//! # 隔离边界
//!
//! 唯一的策略差异是**外层实际对局**的三次 `RegionSelect`（turn 2 / 23 / 47）。
//! 除此之外的一切——`Train` / `RamenSelect` / `SpecialSelect` /
//! `SuperRamenSelect`、事件选项、自选比赛守门——全部**原样转发**同一个
//! `RamenMctsTrainer` 实例。
//!
//! **搜索内部模拟的地区选择仍是手写**：rollout 基策由 `RamenMctsTrainer` 自己持有，
//! 本模块不接触。地区阶段候选数恒 > 1（第 1/2 年 C(5,3)=10，第 3 年 `all` 下
//! C(10,3)=120），因此正常整局**推理请求恰好 3 次**。

use anyhow::{Result, bail};
use umasim::{
    gamedata::{GameConfig, RamenRegionStrategy},
    trainer::RamenSearchStages
};

#[cfg(feature = "onnx")]
mod nn;
#[cfg(feature = "onnx")]
pub use nn::{CompareMode, RegionDecisionObserver, RegionNnTrainer};

/// 校验「网络完全接管地区」与相邻配置项的相容性
///
/// 下列两项必须同时成立，否则会出现「打印一套、执行另一套」：
///
/// 1. `ramen_search_stages` **不含** `region`：地区已被网络接管，搜索永远轮不到它；
/// 2. `ramen_region_strategy` **不是** `fixed`：`fixed` 下第 3 年只剩 1 个候选，
///    网络会被单候选短路，实际只决策 2 次而非 3 次——「整局恰好 3 次推理」这条
///    隔离证据随之失效。
///
/// ❗调用方必须传**命令行覆盖之后**的生效配置与生效阶段集，不能传文件里的原值。
///
/// # 错误
///
/// 上述任一条不成立时返回带修复建议的错误。
pub fn check_region_nn_applicable(cfg: &GameConfig, stages: RamenSearchStages) -> Result<()> {
    if stages.region_select {
        bail!(
            "配置冲突：网络已接管地区决策，但**生效**的搜索阶段集 {stages:?} 仍含 region\
             （配置文件 [mcts] ramen_search_stages = {file:?}）。地区不会再经过搜索，\
             请从 ramen_search_stages 里去掉 region",
            stages = stages,
            file = cfg.mcts.ramen_search_stages
        );
    }
    if matches!(cfg.ramen_region_strategy, RamenRegionStrategy::Fixed) {
        bail!(
            "配置冲突：网络接管地区与 ramen_region_strategy=\"fixed\" 不相容。\
             fixed 下第 3 年只枚举 1 个候选，网络会被单候选短路，实际只决策 2 次地区而非 3 次。\
             请把 ramen_region_strategy 改回 \"all\""
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use umasim::{gamedata::GameConfig, trainer::RamenSearchStages};

    use super::*;
    use crate::utils::Checks;

    /// 适用性校验：含 region 搜索 / fixed 候选各自报错，二者都不沾时通过
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_check_region_nn_applicable() -> Result<()> {
        let mut c = Checks::new();
        let cfg = GameConfig::default_for_init();
        let no_region = RamenSearchStages::parse("train,ramen")?;
        let with_region = RamenSearchStages::parse("train,ramen,region")?;

        let ok = check_region_nn_applicable(&cfg, no_region);
        println!("train,ramen + all → {ok:?}");
        c.check(ok.is_ok(), "不含 region 且 strategy=all 时通过");

        let e1 = check_region_nn_applicable(&cfg, with_region);
        println!("含 region → {e1:?}");
        c.check(
            e1.as_ref().is_err_and(|e| e.to_string().contains("ramen_search_stages")),
            "搜索阶段含 region 时报错并指向 ramen_search_stages"
        );

        let mut fixed = cfg.clone();
        fixed.ramen_region_strategy = RamenRegionStrategy::Fixed;
        let e2 = check_region_nn_applicable(&fixed, no_region);
        println!("fixed → {e2:?}");
        c.check(
            e2.as_ref().is_err_and(|e| e.to_string().contains("fixed")),
            "strategy=fixed 时报错"
        );
        c.finish()
    }

    /// 三年 `RegionSelect` 的候选必须全部解码成合法地区组合
    ///
    /// 候选数：第 1 年 C(5,3)=10、第 2 年 10、第 3 年 `all` 下 C(10,3)=120。
    /// 三个地区必须互不相同且落在该年的地区区间内——这就是「候选落格」的定义，
    /// 接管器在决策后会对选中项做同样的解码校验。
    ///
    /// ❗走**纯函数** `region_select_combos` 并把策略显式传成 `All`，不走
    /// `game.list_actions()`：后者从全局 `GAMECONFIG` 读 `ramen_region_strategy`，
    /// 而 `init_global*` 是幂等的（先到先得），用户把配置设成 `fixed` 时第 3 年只会
    /// 给 1 个候选，测试就会随用户配置变红；同一测试进程里还会和别的用例抢全局配置。
    /// 纯函数路径既不读用户配置，也不引入进程级全局竞争。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_candidates_decode_all_three_years() -> Result<()> {
        use std::env;

        use umasim::{
            game::ramen::{RamenAction, region_select_combos, rules::validate_region_selection},
            gamedata::init_global,
            utils::get_workspace_root
        };

        // `get_region_combinations` 读的是 RAMENDATA（**静态数据**，不是可调配置），
        // 所以这里只需要全局数据加载完成，与谁先初始化过 GAMECONFIG 无关。
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        for (year_idx, want_n) in [(0usize, 10usize), (1, 10), (2, 120)] {
            let combos = region_select_combos(year_idx, RamenRegionStrategy::All, None)?;
            let actions: Vec<RamenAction> = combos
                .iter()
                .map(|&r| RamenAction::no_ramen(umasim::game::ramen::Operation::RegionSelect(r)))
                .collect();
            let mut bad = 0usize;
            for (i, a) in actions.iter().enumerate() {
                // 走接管器**自己那条**解码路径，而不是再写一遍 match
                match RegionNnTrainer::region_of(&actions, i) {
                    Ok(r) if validate_region_selection(year_idx, &r) => {}
                    Ok(r) => {
                        bad += 1;
                        println!("第 {} 年候选 {i} 非法组合 {r:?}", year_idx + 1);
                    }
                    Err(e) => {
                        bad += 1;
                        println!("第 {} 年候选 {i} 解码失败（{a:?}）: {e}", year_idx + 1);
                    }
                }
            }
            println!("第 {} 年候选数 {} 非法 {bad}", year_idx + 1, actions.len());
            c.check(actions.len() == want_n, &format!("第 {} 年候选数 = {want_n}", year_idx + 1));
            c.check(bad == 0, &format!("第 {} 年全部候选合法且可解码", year_idx + 1));
        }
        c.finish()
    }
}
