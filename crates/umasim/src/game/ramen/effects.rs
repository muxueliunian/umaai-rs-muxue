//! 拉面杯效果计算
//!
//! 包含基础效果、地区效果、超级拉面效果的叠加计算。
//! 将所有剧本加成来源合并为统一的训练效果，再应用于训练数值计算。

use super::RamenGame;
use crate::{gamedata::ramen::RAMENDATA, global};

/// 拉面杯训练效果（合并所有来源的加成）
///
/// 由 `calc_ramen_training_effect` 根据当前游戏状态计算得出，
/// 包含所有生效的剧本加成词条的合并结果。
#[derive(Debug, Clone, Default)]
pub struct RamenTrainingEffect {
    /// 训练加成（百分比）
    pub xunlian: i32,
    /// 友情训练加成（百分比，仅友情训练时生效）
    pub youqing: i32,
    /// PT加成（百分比）
    pub pt_bonus: i32,
    /// 属性上限增加
    pub status_limit: i32,
    /// PT上限增加
    pub pt_limit: i32,
    /// 失败率下降（百分比）
    pub fail_rate_drop: i32,
    /// 羁绊增加
    pub friendship: i32,
    /// 得意率加成
    pub deyilv: i32,
    /// hint出现率加成
    pub hint: i32,
    /// hint_special标记
    pub hint_special: bool,
    /// 分身数量
    pub clone_count: i32
}

/// 把训练效果格式化为词条列表（非 0 才显示），如 `["训+23", "失败率-50", "上限+20"]`
///
/// 供吃面候选预览（`RamenGame::ramen_candidate_preview`）与吃面后效果展示
/// （`explain_ramen_info`）复用，保证两处口径一致。
pub fn format_ramen_effect_parts(eff: &RamenTrainingEffect) -> Vec<String> {
    let mut parts = vec![];
    if eff.xunlian != 0 {
        parts.push(format!("训+{}", eff.xunlian));
    }
    if eff.youqing != 0 {
        parts.push(format!("友情+{}", eff.youqing));
    }
    if eff.deyilv != 0 {
        parts.push(format!("得意+{}", eff.deyilv));
    }
    if eff.fail_rate_drop != 0 {
        parts.push(format!("失败率-{}", eff.fail_rate_drop));
    }
    if eff.friendship != 0 {
        parts.push(format!("羁绊+{}", eff.friendship));
    }
    if eff.status_limit != 0 {
        parts.push(format!("上限+{}", eff.status_limit));
    }
    if eff.pt_bonus != 0 {
        parts.push(format!("PT+{}", eff.pt_bonus));
    }
    if eff.pt_limit != 0 {
        parts.push(format!("PT上限+{}", eff.pt_limit));
    }
    if eff.hint != 0 {
        parts.push(format!("hint+{}", eff.hint));
    }
    if eff.clone_count != 0 {
        parts.push(format!("分身+{}", eff.clone_count));
    }
    if eff.hint_special {
        parts.push("hint全卡".to_string());
    }
    parts
}

/// 根据当前剧本PT查找对应的 `ramen_pt_effect` 档位
///
/// 从高到低查找第一个 `pt_min <= scenario_pt` 的档位。
pub fn find_pt_effect_tier(scenario_pt: i32) -> usize {
    let ramen_data = global!(RAMENDATA);
    let mut tier = 0;
    for (i, pe) in ramen_data.ramen_pt_effect.iter().enumerate() {
        if scenario_pt >= pe.pt_min {
            tier = i;
        }
    }
    tier
}

/// 计算地区词条加成档位
///
/// 每获得300点剧本PT提升一档，最高5档。
fn calc_region_bonus_tier(year_scenario_pt: i32) -> usize {
    (year_scenario_pt / 300).min(5) as usize
}

/// 计算超级拉面回合的效果（回合 72-77 自动生效）
///
/// 超级拉面期间：
/// - `ramen_pt_effect` 按最高档生效
/// - `ramen_basic_effect` 按最高档生效
/// - 第3年RMJ结算效果（rmj_results[2]）常驻生效
/// - `finals_effect.base` 效果生效
/// - `finals_effect.extra` 效果仅在支援卡种类 >= 4 时生效
/// - 地区效果不生效
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `year_idx`: 年份索引（0-2）
fn calc_finals_effect(game: &RamenGame, _year_idx: usize) -> RamenTrainingEffect {
    let ramen_data = global!(RAMENDATA);
    let mut effect = RamenTrainingEffect::default();

    // 1. ramen_pt_effect 按最高档生效（最后一档）
    let pt_effect = ramen_data.ramen_pt_effect.last().unwrap();
    effect.xunlian += pt_effect.xunlian;
    effect.deyilv += pt_effect.deyilv;
    effect.hint += pt_effect.hint;

    // 2. ramen_basic_effect 按最高档生效（最后一档）
    let basic = ramen_data.ramen_basic_effect.last().unwrap();
    effect.xunlian += basic.xunlian;
    effect.youqing += basic.youqing;
    effect.fail_rate_drop += basic.fail_rate_drop;
    effect.friendship += basic.friendship;
    effect.status_limit += basic.status_limit;
    effect.hint_special |= basic.hint_special;

    // 3. 第3年RMJ结算效果（rmj_results[2]）在URA期间生效
    if let Some(&success) = game.ramen.rmj_results.get(2) {
        let rmj_effect = if success {
            &ramen_data.ramen_success_effect[2]
        } else {
            &ramen_data.ramen_fail_effect[2]
        };
        effect.youqing += rmj_effect.youqing;
        effect.deyilv += rmj_effect.deyilv;
        effect.hint += rmj_effect.hint;
    }

    // 4. finals_effect.base 效果
    let finals = &ramen_data.finals_effect;
    effect.youqing += finals.base.youqing;

    // 5. finals_effect.extra 效果：支援卡种类 >= 4 时额外生效
    if game.deck_can_split {
        effect.pt_bonus += finals.extra.pt_bonus;
        effect.pt_limit += finals.extra.pt_limit;
        effect.clone_count += finals.extra.clone_count;
    }

    effect
}

/// 计算普通回合的效果（非超级拉面回合）
///
/// 普通回合效果来源：
/// - `ramen_pt_effect`：常驻生效（根据当前剧本PT决定档次）
/// - `ramen_success_effect` / `ramen_fail_effect`：RMJ结算后常驻生效
/// - `ramen_basic_effect`：仅吃面后生效
/// - `ramen_region_effect`：仅吃面后且在 `at_trains` 标注的训练位置生效
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
/// - `year_idx`: 年份索引（0-2）
fn calc_normal_effect(game: &RamenGame, train: usize, year_idx: usize) -> RamenTrainingEffect {
    let ramen_data = global!(RAMENDATA);
    let mut effect = RamenTrainingEffect::default();

    // 1. ramen_pt_effect（常驻生效）
    let pt_tier = find_pt_effect_tier(game.ramen.scenario_pt);
    let pt_effect = &ramen_data.ramen_pt_effect[pt_tier];
    effect.xunlian += pt_effect.xunlian;
    effect.deyilv += pt_effect.deyilv;
    effect.hint += pt_effect.hint;

    // 2. ramen_success_effect / ramen_fail_effect（RMJ结算后常驻生效）
    if year_idx >= 1 {
        // year_idx 1 使用 rmj_results[0]，year_idx 2 使用 rmj_results[1]
        let prev_idx = year_idx - 1;
        if let Some(&success) = game.ramen.rmj_results.get(prev_idx) {
            let rmj_effect = if success {
                &ramen_data.ramen_success_effect[prev_idx]
            } else {
                &ramen_data.ramen_fail_effect[prev_idx]
            };
            effect.youqing += rmj_effect.youqing;
            effect.deyilv += rmj_effect.deyilv;
            effect.hint += rmj_effect.hint;
        }
    }

    // 3. ramen_basic_effect（仅吃面后生效）
    let eating = game.ramen.current_ramen.is_some();
    if eating && year_idx < ramen_data.ramen_basic_effect.len() {
        let basic = &ramen_data.ramen_basic_effect[year_idx];
        effect.xunlian += basic.xunlian;
        effect.youqing += basic.youqing;
        effect.fail_rate_drop += basic.fail_rate_drop;
        effect.friendship += basic.friendship;
        effect.status_limit += basic.status_limit;
        effect.hint_special |= basic.hint_special;
    }

    // 4. ramen_region_effect（仅吃面后且在 at_trains 标注位置生效）
    if eating {
        if let Some(ramen_idx) = game.ramen.current_ramen {
            let region = &ramen_data.ramen_region_effect[ramen_idx];
            if region.at_trains.contains(&(train as i32)) {
                // 地区词条加成随当年剧本PT增加
                let bonus_tier = calc_region_bonus_tier(game.ramen.scenario_pt);
                let region_bonus = ramen_data.region_bonus.get(bonus_tier).copied().unwrap_or(0);
                effect.xunlian += region.xunlian;
                effect.youqing += region.youqing + region_bonus;
                effect.pt_bonus += region.pt_bonus + region_bonus;
            }
        }
    }

    effect
}

/// 计算拉面杯的训练效果
///
/// 根据当前游戏状态，合并所有生效的加成来源：
/// - 超级拉面回合（72-77）：调用 `calc_finals_effect`
/// - 普通回合：调用 `calc_normal_effect`
///
/// **重要**：非友情训练时 youqing 会被强制归零，调用方无需额外判断。
///
/// # 参数
/// - `game`: 拉面杯游戏状态
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
/// - `is_shining`: 是否友情训练（非友情训练时 youqing 视为 0）
pub fn calc_ramen_training_effect(game: &RamenGame, train: usize, is_shining: bool) -> RamenTrainingEffect {
    let super_ramen = game.is_super_ramen_turn();
    let year_idx = (game.current_year() - 1) as usize;

    let mut effect = if super_ramen {
        // 超级拉面回合：基础效果 + finals_effect，不享受地区效果
        calc_finals_effect(game, year_idx)
    } else {
        // 普通回合：PT常驻 + RMJ常驻 + 吃面基础 + 地区效果
        calc_normal_effect(game, train, year_idx)
    };

    // 非友情训练时，youqing 不生效（强制归零）
    if !is_shining {
        effect.youqing = 0;
    }

    effect
}

/// 计算当前回合生效的剧本得意率总加成
///
/// 按剧本原始规则：剧本得意率只和支援卡的得意率相加（参见 `ramen_memo_cn.md`）。
/// 本函数仅汇总**对训练分布生效**的剧本得意率来源：
/// - `ramen_pt_effect`：常驻生效
/// - `ramen_success_effect` / `ramen_fail_effect`：RMJ 结算后常驻
///
/// **不包含** `ramen_basic_effect`（全部为 0）和 `ramen_region_effect`（无 deyilv 字段）。
///
/// 用于 `RamenGame::deyilv`，与 `calc_finals_effect` / `calc_normal_effect` 中的
/// deyilv 计算保持一致（**超级拉面直接复用 `calc_finals_effect`**）。
///
/// # 参数
/// - `game`: 拉面杯游戏状态
///
/// # 返回
/// 当前回合的剧本得意率总加成（i32，可直接 + 到支援卡 deyilv 上）
pub fn calc_scenario_deyilv(game: &RamenGame) -> i32 {
    let ramen_data = global!(RAMENDATA);
    let year_idx = (game.current_year() - 1) as usize;

    if game.is_super_ramen_turn() {
        // 超级拉面：复用 calc_finals_effect（最后一档 pt + rmj_results[2]）
        calc_finals_effect(game, year_idx).deyilv
    } else {
        // 普通回合：pt_effect(当前档) + rmj_results[year-1]
        // 这里 calc_normal_effect 是训练位置相关的，单独算 deyilv 更直接
        let mut deyilv = 0;
        let pt_tier = find_pt_effect_tier(game.ramen.scenario_pt);
        deyilv += ramen_data.ramen_pt_effect[pt_tier].deyilv;
        if year_idx >= 1 {
            let prev_idx = year_idx - 1;
            if let Some(&success) = game.ramen.rmj_results.get(prev_idx) {
                let rmj_effect = if success {
                    &ramen_data.ramen_success_effect[prev_idx]
                } else {
                    &ramen_data.ramen_fail_effect[prev_idx]
                };
                deyilv += rmj_effect.deyilv;
            }
        }
        deyilv
    }
}

/// 应用拉面杯训练效果计算最终训练数值
///
/// 计算公式：
/// - 属性增加值 = lower_value * (100 + xunlian) / 100 * (100 + youqing) / 100
/// - PT增加值 = lower_value * (100 + xunlian) / 100 * (100 + youqing) / 100 * (100 + pt_bonus) / 100
///
/// 上层数值上限：
/// - 属性上限 = 100 + status_limit
/// - PT上限 = 100 + status_limit + pt_limit
///
/// # 参数
/// - `lower_value`: 下层数值（不计算剧本加成的基础训练数值，上限100）
/// - `effect`: 合并后的拉面杯训练效果
/// - `train`: 训练位置（0=速, 1=耐, 2=力, 3=根, 4=智）
///
/// # 返回
/// `(属性增加值, PT增加值)` - 包含下层数值和受上限约束的上层数值的最终训练数值
pub fn apply_ramen_training_value(lower_value: i32, effect: &RamenTrainingEffect, _train: usize) -> (i32, i32) {
    let lower = lower_value.min(100);

    // 计算上层数值
    let xunlian_mult = (100 + effect.xunlian) as f64 / 100.0;
    let youqing_mult = (100 + effect.youqing) as f64 / 100.0;
    let pt_bonus_mult = (100 + effect.pt_bonus) as f64 / 100.0;

    // 属性训练上层数值
    let status_upper_raw = (lower as f64 * xunlian_mult * youqing_mult) as i32 - lower;
    // PT训练上层数值
    let pt_upper_raw = (lower as f64 * xunlian_mult * youqing_mult * pt_bonus_mult) as i32 - lower;

    // 上层数值上限约束
    let status_limit = 100 + effect.status_limit;
    let pt_limit = 100 + effect.status_limit + effect.pt_limit;

    let status_upper = status_upper_raw.min(status_limit);
    let pt_upper = pt_upper_raw.min(pt_limit);

    // 调试日志：打印约束前后的 upper/lower 值（排查训练数值不对时使用）
    crate::diag!(
        "  apply_ramen_training_value: lower={} (raw={}) \
         xunlian={} youqing={} pt_bonus={} status_limit={} pt_limit={}\n    \
         属性: status_upper_raw={} -> status_limit={} -> status_upper={} (最终={})\n    \
         PT:   pt_upper_raw={} -> pt_limit={} -> pt_upper={} (最终={})",
        lower,
        lower_value,
        effect.xunlian,
        effect.youqing,
        effect.pt_bonus,
        effect.status_limit,
        effect.pt_limit,
        status_upper_raw,
        status_limit,
        status_upper,
        lower + status_upper,
        pt_upper_raw,
        pt_limit,
        pt_upper,
        lower + pt_upper,
    );

    (lower + status_upper, lower + pt_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::ramen::RamenState,
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    /// 创建一个用于测试的 RamenGame 实例
    fn make_test_game() -> RamenGame {
        RamenGame {
            ramen: RamenState {
                scenario_pt: 1000,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_calc_effect_pt_only() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 不吃面、非超级拉面回合、无RMJ结果
        // 仅有 ramen_pt_effect 常驻生效
        let mut game = make_test_game();
        game.base.turn = 5; // year 1

        let effect = calc_ramen_training_effect(&game, 0, false);
        println!("PT=1000, 不吃面, 非友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  deyilv={} hint={} fail_rate_drop={}",
            effect.deyilv, effect.hint, effect.fail_rate_drop
        );
        // pt_min=1000 的档位: xunlian=8, deyilv=63, hint=50
        println!("  => 期望: xunlian=8, deyilv=63, hint=50");

        Ok(())
    }

    #[test]
    fn test_calc_effect_with_eating() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let ramen_data = global!(RAMENDATA);

        // 吃面 + year 1 + 友情训练
        let mut game = make_test_game();
        game.base.turn = 5;
        game.ramen.current_ramen = Some(0); // 吃第一种地区拉面
        game.ramen.scenario_pt = 500;

        // 查看 region 0 的 at_trains
        let region0 = &ramen_data.ramen_region_effect[0];
        println!("region 0: name={} at_trains={:?}", region0.name, region0.at_trains);

        // 在 at_trains 包含的位置上测试
        let train_in_region = region0.at_trains[0] as usize;
        let effect = calc_ramen_training_effect(&game, train_in_region, true);
        println!("PT=500, 吃面region0, train={train_in_region}, 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  fail_rate_drop={} friendship={} status_limit={}",
            effect.fail_rate_drop, effect.friendship, effect.status_limit
        );

        // 在 at_trains 不包含的位置上测试
        let effect2 = calc_ramen_training_effect(&game, 4, true);
        println!("PT=500, 吃面region0, train=4(智), 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect2.xunlian, effect2.youqing, effect2.pt_bonus
        );

        Ok(())
    }

    #[test]
    fn test_calc_effect_rmj_success() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // year 2, RMJ year 1 成功
        let mut game = make_test_game();
        game.base.turn = 30;
        game.ramen.scenario_pt = 2000;
        game.ramen.rmj_results = vec![true];

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("year2, PT=2000, RMJ成功, 友情:");
        println!(
            "  xunlian={} youqing={} deyilv={} hint={}",
            effect.xunlian, effect.youqing, effect.deyilv, effect.hint
        );
        // pt_effect(PT=2000): xunlian=12, deyilv=68, hint=70
        // rmj_success[0]: youqing=5, deyilv=80, hint=30
        println!("  => 期望: xunlian=12, youqing=5, deyilv=148, hint=100");

        Ok(())
    }

    #[test]
    fn test_calc_effect_super_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 超级拉面回合
        let mut game = make_test_game();
        game.base.turn = 72;
        game.ramen.scenario_pt = 5000;

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("超级拉面, PT=5000, 友情:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  status_limit={} pt_limit={} clone_count={}",
            effect.status_limit, effect.pt_limit, effect.clone_count
        );
        // pt_effect(PT=5000): xunlian=20
        // basic(year3): xunlian=15, youqing=45, status_limit=40
        // finals base: youqing=150
        // finals extra (deck_can_split=false 默认): 不生效
        println!("  => 期望: xunlian=35, youqing=195, pt_bonus=0, status_limit=40");

        Ok(())
    }

    #[test]
    fn test_calc_effect_super_ramen_with_split() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 超级拉面回合 + deck_can_split = true
        let mut game = make_test_game();
        game.base.turn = 73;
        game.ramen.scenario_pt = 5000;
        game.deck_can_split = true;

        let effect = calc_ramen_training_effect(&game, 0, true);
        println!("超级拉面, PT=5000, 友情, deck_can_split=true:");
        println!(
            "  xunlian={} youqing={} pt_bonus={}",
            effect.xunlian, effect.youqing, effect.pt_bonus
        );
        println!(
            "  status_limit={} pt_limit={} clone_count={}",
            effect.status_limit, effect.pt_limit, effect.clone_count
        );
        // pt_effect(PT=5000): xunlian=20
        // basic(year3): xunlian=15, youqing=45, status_limit=40
        // finals base: youqing=150
        // finals extra: pt_bonus=100, pt_limit=100, clone_count=1
        println!("  => 期望: xunlian=35, youqing=195, pt_bonus=100, pt_limit=100, clone_count=1");

        Ok(())
    }

    #[test]
    fn test_calc_effect_non_shining() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 非友情训练时 youqing 应为 0
        let mut game = make_test_game();
        game.base.turn = 5;
        game.ramen.scenario_pt = 500;
        game.ramen.current_ramen = Some(5); // 吃面
        game.ramen.rmj_results = vec![true]; // 不影响 year 1

        let effect_shining = calc_ramen_training_effect(&game, 0, true);
        let effect_normal = calc_ramen_training_effect(&game, 0, false);
        println!("友情训练: youqing={}", effect_shining.youqing);
        println!("普通训练: youqing={}", effect_normal.youqing);
        println!("  => 普通训练 youqing 应为 0");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_status() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // 属性训练: lower=50, xunlian=20, youqing=10, pt_bonus=0
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 10,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(50, &effect, 0);
        // upper = 50 * 1.2 * 1.1 - 50 = 66 - 50 = 16
        // status = 50 + 16 = 66
        // pt = 50 + 16 = 66 (pt_bonus=0 时与属性相同)
        println!("lower=50, xunlian=20, youqing=10, pt_bonus=0:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=66, pt=66");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_pt() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // PT训练: lower=50, xunlian=20, youqing=10, pt_bonus=50
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 10,
            pt_bonus: 50,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(50, &effect, 0);
        // status upper = 50 * 1.2 * 1.1 - 50 = 16, status = 66
        // pt upper = 50 * 1.2 * 1.1 * 1.5 - 50 = 99 - 50 = 49, pt = 99
        println!("lower=50, xunlian=20, youqing=10, pt_bonus=50:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=66, pt=99");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_upper_limit() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // 上层数值超过上限时应被截断
        let effect = RamenTrainingEffect {
            xunlian: 100,
            youqing: 100,
            pt_bonus: 100,
            status_limit: 50,
            pt_limit: 100,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(80, &effect, 0);
        // status upper raw = 80 * 2.0 * 2.0 - 80 = 240, cap = 100+50=150
        // status = 80 + 150 = 230
        // pt upper raw = 80 * 2.0 * 2.0 * 2.0 - 80 = 560, cap = 100+50+100=250
        // pt = 80 + 250 = 330
        println!("lower=80, xunlian=100, youqing=100, pt_bonus=100, status_limit=50, pt_limit=100:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=230, pt=330");

        Ok(())
    }

    #[test]
    fn test_apply_training_value_lower_cap() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;

        // lower_value 超过 100 时应被截断到 100
        let effect = RamenTrainingEffect {
            xunlian: 20,
            youqing: 0,
            ..Default::default()
        };
        let (status_val, pt_val) = apply_ramen_training_value(150, &effect, 0);
        // lower = min(150, 100) = 100
        // upper = 100 * 1.2 - 100 = 20
        // status = pt = 120
        println!("lower=150(截断为100), xunlian=20:");
        println!("  status={status_val} pt={pt_val}");
        println!("  => 期望: status=120, pt=120");

        Ok(())
    }

    // ========== calc_scenario_deyilv 测试 ==========

    /// 普通回合：PT 1000 + 无 RMJ → 仅 pt_effect(PT=1000档) 的 deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_pt_only() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 5; // year 1
        game.ramen.scenario_pt = 1000;
        // rmj_results 为空（year 1）

        let deyilv = calc_scenario_deyilv(&game);
        // pt_min=1000 的档位: deyilv=63
        println!("year1, PT=1000, 无 RMJ: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 63);
        Ok(())
    }

    /// 普通回合：PT 1000 + RMJ 成功 → pt_effect + rmj_success[0].deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_with_rmj_success() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 30; // year 2
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect(PT=1000).deyilv = 63
        // rmj_success[0].deyilv = 80
        // 总计 = 63 + 80 = 143
        println!("year2, PT=1000, RMJ成功: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 143);
        Ok(())
    }

    /// 普通回合：RMJ 失败 → pt_effect + rmj_fail[year-1].deyilv
    #[test]
    fn test_calc_scenario_deyilv_normal_with_rmj_fail() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 30; // year 2
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![false]; // year 1 RMJ 失败

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect(PT=1000).deyilv = 63
        // rmj_fail[0].deyilv = 30
        println!("year2, PT=1000, RMJ失败: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 93); // 63 + 30
        Ok(())
    }

    /// 超级拉面：PT 5000 + RMJ 成功 → pt_effect(最后一档) + rmj_success[2].deyilv
    #[test]
    fn test_calc_scenario_deyilv_super_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 72; // 超级拉面回合（URA）
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, true]; // 前三年都成功

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect 最后一档(PT>=5000).deyilv = 80
        // rmj_success[2].deyilv = 250
        // 总计 = 80 + 250 = 330
        println!("超级拉面 turn=72, PT=5000, RMJ成功: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 330);
        Ok(())
    }

    /// 超级拉面：RMJ 失败 → pt_effect(最后一档) + rmj_fail[2].deyilv
    #[test]
    fn test_calc_scenario_deyilv_super_ramen_rmj_fail() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut game = make_test_game();
        game.base.turn = 73;
        game.ramen.scenario_pt = 5000;
        game.ramen.rmj_results = vec![true, true, false]; // year 3 RMJ 失败

        let deyilv = calc_scenario_deyilv(&game);
        // pt_effect 最后一档.deyilv = 80
        // rmj_fail[2].deyilv = 150
        println!("超级拉面 turn=73, PT=5000, RMJ失败: scenario_deyilv={deyilv}");
        assert_eq!(deyilv, 230); // 80 + 150
        Ok(())
    }
}
