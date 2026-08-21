//! 拉面杯核心规则
//!
//! 诀窍系统、做面/吃面、RMJ 结算等核心机制的纯函数实现。
//! 函数以 `RamenState` 或相关数据为参数，不直接修改游戏状态的其他部分。

use anyhow::Result;

use super::{FeelingType, RamenState};
use crate::{diag, gamedata::ramen::RAMENDATA, global};

/// 诀窍总上限
pub const FEELING_LIMIT: i32 = 10;
/// 诀窍槽上限
pub const GAUGE_LIMIT: i32 = 7;
/// 做面消耗的诀窍点数
pub const RAMEN_COST: i32 = 5;

// ========== 诀窍槽基础值分配 ==========

/// 根据年度三个配方的诀窍消耗比例，计算基础值分配到三种类型 (A/B/C) 的数量。
///
/// base_sum 固定为 10（只考虑新友人）。
/// 按配方总消耗 [A, B, C] 的比例分配，四舍五入后调整使总和等于 base_sum。
pub fn calc_gauge_base_distribution(selected_regions: &[usize; 3]) -> [i32; 3] {
    let ramen_data = global!(RAMENDATA);
    let base_sum = 10;

    // 累加三个配方的各类型消耗（年3地区复用年1配方，取模映射）
    let mut recipe_sum = [0i32; 3];
    for &region_idx in selected_regions {
        let feeling_idx = region_idx % ramen_data.region_feeling.len();
        let feeling = &ramen_data.region_feeling[feeling_idx];
        for j in 0..3 {
            recipe_sum[j] += feeling[j];
        }
    }

    // 按比例分配：先 floor，再逐个补给"已分配最少"的位置
    // 已分配相同时，优先给配方消耗量更大的位置
    // 特殊规则：消耗=1 的位置固定分配 1，且不允许任何位置分配为 0
    let mut result = [0i32; 3];
    let mut fixed = [false; 3];
    for i in 0..3 {
        if recipe_sum[i] == 1 {
            result[i] = 1;
            fixed[i] = true;
        }
    }
    let fixed_sum: i32 = result.iter().sum();
    let remaining = base_sum - fixed_sum;
    // 对未固定的位置按比例分配
    let unfixed_consumed: i32 = (0..3).filter(|&i| !fixed[i]).map(|i| recipe_sum[i]).sum();
    for i in 0..3 {
        if !fixed[i] && unfixed_consumed > 0 {
            let exact = recipe_sum[i] as f64 * remaining as f64 / unfixed_consumed as f64;
            result[i] = exact.floor() as i32;
        }
    }
    let mut diff = base_sum - result.iter().sum::<i32>();
    while diff > 0 {
        // 找已分配最小、配方消耗最大的未固定位置
        let mut best = None;
        for i in 0..3 {
            if fixed[i] {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    if result[i] < result[b] || (result[i] == result[b] && recipe_sum[i] > recipe_sum[b]) {
                        best = Some(i);
                    }
                }
            }
        }
        if let Some(b) = best {
            result[b] += 1;
            diff -= 1;
        } else {
            break;
        }
    }

    result
}

// ========== 诀窍槽操作 ==========

/// 向指定类型的诀窍槽增加数值，满 GAUGE_LIMIT 则清零并获得 1 个诀窍。
///
/// 无论溢出多少，都只能增加 1 个诀窍并清零，超出部分不保留。
/// 返回实际获得的诀窍数量（0 或 1）。
pub fn add_gauge(state: &mut RamenState, feeling_type: FeelingType, amount: i32) -> i32 {
    let idx = feeling_type as usize;
    state.feeling_slot[idx] += amount;
    if state.feeling_slot[idx] >= GAUGE_LIMIT {
        state.feeling_slot[idx] = 0;
        add_feeling(state, feeling_type, 1);
        1
    } else {
        0
    }
}

/// 向诀窍库存增加指定类型的诀窍点。
///
/// 超过总上限时，按获得顺序队列丢弃最早的诀窍。
pub fn add_feeling(state: &mut RamenState, feeling_type: FeelingType, count: i32) {
    let idx = feeling_type as usize;
    for _ in 0..count {
        state.feeling_stock[idx] += 1;
        state.feeling_queue.push(feeling_type);
        // 溢出丢弃
        while state.feeling_stock.iter().sum::<i32>() > FEELING_LIMIT {
            if let Some(oldest) = state.feeling_queue.first().cloned() {
                let oldest_idx = oldest as usize;
                if state.feeling_stock[oldest_idx] > 0 {
                    state.feeling_stock[oldest_idx] -= 1;
                }
                state.feeling_queue.remove(0);
            } else {
                break;
            }
        }
    }
}

// ========== 做面/吃面 ==========

/// 校验隐藏风味替换目标的合法性。
///
/// - `recipe`: 配方消耗 [A, B, C]
/// - `special_targets`: 每种类型用几个隐藏风味替换 [A, B, C]
///
/// 约束：
/// - 每个 `special_targets[i] >= 0` 且 `<= recipe[i]`
/// - `sum(special_targets) <= 2`（单次做面最多用 2 个隐藏风味）
fn validate_special_targets(recipe: &[i32; 3], special_targets: &[i32; 3]) -> Result<()> {
    let total: i32 = special_targets.iter().sum();
    if total > 2 {
        anyhow::bail!("隐藏风味使用总数不能超过 2，实际: {total}");
    }
    for i in 0..3 {
        if special_targets[i] < 0 {
            anyhow::bail!("special_targets[{i}] 不能为负: {}", special_targets[i]);
        }
        if special_targets[i] > recipe[i] {
            anyhow::bail!(
                "special_targets[{i}] 超过配方消耗: {} > {}",
                special_targets[i],
                recipe[i]
            );
        }
    }
    Ok(())
}

/// 计算隐藏风味替换后的实际诀窍消耗。
///
/// 返回 `[A, B, C]` 实际需要消耗的诀窍数量。
fn calc_net_recipe(recipe: &[i32; 3], special_targets: &[i32; 3]) -> [i32; 3] {
    [
        recipe[0] - special_targets[0],
        recipe[1] - special_targets[1],
        recipe[2] - special_targets[2]
    ]
}

/// 获取指定地区的配方。
///
/// # 错误
/// - `recipe_idx` 超出范围
/// - 配方总消耗不等于 `RAMEN_COST`
pub fn get_recipe(recipe_idx: usize) -> Result<&'static [i32; 3]> {
    let ramen_data = global!(RAMENDATA);
    // 年3地区(10-19)复用年1配方(0-9)，取模映射
    let feeling_idx = recipe_idx % ramen_data.region_feeling.len();
    let recipe = &ramen_data.region_feeling[feeling_idx];
    if recipe.iter().sum::<i32>() != RAMEN_COST {
        anyhow::bail!("配方总消耗不为 {RAMEN_COST}: idx={recipe_idx}, recipe={recipe:?}");
    }
    Ok(recipe)
}

/// 检查是否有足够的诀窍和隐藏风味做面。
///
/// 假设 `recipe` 和 `special_targets` 已经通过合法性校验。
/// 只检查当前资源是否足够。
///
/// - `recipe`: 配方消耗 [A, B, C]
/// - `special_targets`: 每种类型用几个隐藏风味替换 [A, B, C]
pub fn can_make_ramen(state: &RamenState, recipe: &[i32; 3], special_targets: &[i32; 3]) -> bool {
    let total_special: i32 = special_targets.iter().sum();
    if total_special > state.special_feeling {
        return false;
    }
    let net = calc_net_recipe(recipe, special_targets);
    (0..3).all(|i| state.feeling_stock[i] >= net[i])
}

/// 消耗诀窍做面，返回实际消耗的隐藏风味数量。
///
/// - `recipe_idx`: `region_feeling` 数组下标
/// - `special_targets`: 每种类型用几个隐藏风味替换 [A, B, C]，总和 <= 2
///
/// # 错误
/// - recipe_idx 无效或配方非法
/// - special_targets 不合法（负值、超过配方消耗、总和 > 2）
/// - 隐藏风味不足
/// - 诀窍库存不足
pub fn consume_for_ramen(state: &mut RamenState, recipe_idx: usize, special_targets: &[i32; 3]) -> Result<i32> {
    let recipe = get_recipe(recipe_idx)?;
    validate_special_targets(recipe, special_targets)?;
    let total_special: i32 = special_targets.iter().sum();
    if total_special > state.special_feeling {
        anyhow::bail!("隐藏风味不足: 需要 {total_special}，实际 {}", state.special_feeling);
    }
    let net = calc_net_recipe(recipe, special_targets);
    for i in 0..3 {
        if state.feeling_stock[i] < net[i] {
            anyhow::bail!("诀窍不足: 类型{i} 需要 {}，实际 {}", net[i], state.feeling_stock[i]);
        }
    }
    // 消耗前快照：便于排查"消耗大于库存"
    let before_stock = state.feeling_stock;
    let before_special = state.special_feeling;
    diag!(
        ">> 吃面消耗前: 配方={:?}, 隐藏风味替换={:?}, 净消耗={:?}, 库存 A={} B={} C={}, 隐藏风味={}",
        recipe,
        special_targets,
        net,
        before_stock[0],
        before_stock[1],
        before_stock[2],
        before_special,
    );
    // 消耗诀窍
    for i in 0..3 {
        state.feeling_stock[i] -= net[i];
    }
    // 同步更新 feeling_queue
    let mut remaining = net;
    state.feeling_queue.retain(|&ft| {
        let idx = ft as usize;
        if remaining[idx] > 0 {
            remaining[idx] -= 1;
            false
        } else {
            true
        }
    });
    state.special_feeling -= total_special;
    diag!(
        ">> 吃面消耗后: 库存 A={} B={} C={}, 隐藏风味={}",
        state.feeling_stock[0],
        state.feeling_stock[1],
        state.feeling_stock[2],
        state.special_feeling,
    );
    Ok(total_special)
}

/// 枚举给定当前库存和隐藏风味下，可用于制作指定面的所有合法 `special_targets`。
///
/// 返回按 `sum(t)` 升序排列的候选列表（最少隐藏风味优先）。
/// 调用方无需再过滤 `can_make_ramen`；返回空时该面不可做。
///
/// 生成算法：
/// 1. 算 `min_needed[i] = max(0, recipe[i] - feeling_stock[i])`（每类缺口）
/// 2. `need_sum = min_needed.iter().sum()`
/// 3. `budget = min(2, special_feeling) - need_sum`（剩余隐藏风味预算）
/// 4. 若 `budget < 0`，该面无法制作，返回空
/// 5. 枚举 `t[i] ∈ [min_needed[i], recipe[i]]`，`sum(t) ≤ need_sum + budget`
/// 6. 过滤 `can_make_ramen`，按 `sum(t)` 升序返回
///
/// 这样生成的候选数与玩家实际可选空间一致（库存紧张 1~6 种，全富余 9~10 种）。
pub fn list_special_targets_for(state: &RamenState, ramen_idx: usize) -> Result<Vec<[i32; 3]>> {
    let recipe = get_recipe(ramen_idx)?;
    let min_needed: [i32; 3] = [
        (recipe[0] - state.feeling_stock[0]).max(0),
        (recipe[1] - state.feeling_stock[1]).max(0),
        (recipe[2] - state.feeling_stock[2]).max(0)
    ];
    let need_sum: i32 = min_needed.iter().sum();
    let budget = 2.min(state.special_feeling) - need_sum;
    if budget < 0 {
        return Ok(Vec::new());
    }
    let total_cap = need_sum + budget;
    let mut out: Vec<[i32; 3]> = Vec::new();
    for t_a in min_needed[0]..=recipe[0] {
        if t_a > total_cap {
            break;
        }
        for t_b in min_needed[1]..=recipe[1] {
            if t_a + t_b > total_cap {
                break;
            }
            for t_c in min_needed[2]..=recipe[2] {
                let s = t_a + t_b + t_c;
                if s > total_cap {
                    break;
                }
                let t = [t_a, t_b, t_c];
                if can_make_ramen(state, recipe, &t) {
                    out.push(t);
                }
            }
        }
    }
    out.sort_by_key(|t| t.iter().sum::<i32>());
    Ok(out)
}

/// 计算吃面获得的剧本 PT。
///
/// - `year_idx`: 年份索引（0-2）
/// - `eat_count`: 当年内已吃面次数（第一面 eat_count=0）
///
/// 公式：`gain_pt_base[year] + gain_pt_delta[year] * min(eat_count, 5)`
///
/// # 错误
/// `year_idx` 超出范围时返回错误。
pub fn calc_ramen_pt_gain(year_idx: usize, eat_count: i32) -> Result<i32> {
    let ramen_data = global!(RAMENDATA);
    let base = ramen_data
        .gain_pt_base
        .get(year_idx)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("year_idx 越界: {year_idx}"))?;
    let delta = ramen_data
        .gain_pt_delta
        .get(year_idx)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("year_idx 越界: {year_idx}"))?;
    Ok(base + delta * eat_count.min(5))
}

// ========== RMJ 结算 ==========

/// RMJ 结算结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmjResult {
    /// 失败
    Fail,
    /// 成功
    Success,
    /// 大成功（第三年 pt >= 5000）
    GreatSuccess
}

impl RmjResult {
    /// 是否成功（包括大成功）
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success | Self::GreatSuccess)
    }

    /// 是否大成功
    pub fn is_great_success(&self) -> bool {
        matches!(self, Self::GreatSuccess)
    }
}

/// 执行 RMJ 结算，返回结算结果。
///
/// 比较当前剧本 PT 与阈值 `ramen_success_pt[year_idx]`。
/// 第三年（year_idx=2）PT >= 5000 为大成功。
/// 结果存入 `state.rmj_results`。
pub fn check_rmj(state: &mut RamenState, year_idx: usize) -> RmjResult {
    let ramen_data = global!(RAMENDATA);
    let threshold = ramen_data.ramen_success_pt.get(year_idx).copied().unwrap_or(i32::MAX);
    let result = if state.scenario_pt < threshold {
        RmjResult::Fail
    } else if year_idx == 2 && state.scenario_pt >= 5000 {
        RmjResult::GreatSuccess
    } else {
        RmjResult::Success
    };
    state.rmj_results.push(result.is_success());
    result
}

// ========== 地区选择 ==========

/// 各年份的地区 ID 范围 [start, end_inclusive]。
const REGION_RANGES: [(usize, usize); 3] = [(0, 4), (5, 9), (10, 19)];

/// 获取指定年份可选地区 ID 列表。
///
/// # 错误
/// `year_idx` 超出范围（0-2）时返回错误。
pub fn get_region_range(year_idx: usize) -> Result<Vec<usize>> {
    let &(start, end) = REGION_RANGES
        .get(year_idx)
        .ok_or_else(|| anyhow::anyhow!("year_idx 越界: {year_idx}"))?;
    Ok((start..=end).collect())
}

/// 生成指定年份的所有3地区组合（不考虑顺序）
///
/// 返回所有 C(n, 3) 的组合，每个组合为 `[usize; 3]`，已排序。
pub fn get_region_combinations(year_idx: usize) -> Result<Vec<[usize; 3]>> {
    let range = get_region_range(year_idx)?;
    let n = range.len();
    if n < 3 {
        anyhow::bail!("可选地区不足 3 个: year_idx={year_idx}, range={range:?}");
    }
    let mut combos = vec![];
    for i in 0..n - 2 {
        for j in i + 1..n - 1 {
            for k in j + 1..n {
                combos.push([range[i], range[j], range[k]]);
            }
        }
    }
    Ok(combos)
}

/// 验证地区选择是否合法。
///
/// 选择 3 个地区，必须在该年份范围内且互不重复。
pub fn validate_region_selection(year_idx: usize, selections: &[usize; 3]) -> bool {
    let Some(&(start, end)) = REGION_RANGES.get(year_idx) else {
        return false;
    };
    selections.iter().all(|&id| id >= start && id <= end)
        && selections[0] != selections[1]
        && selections[0] != selections[2]
        && selections[1] != selections[2]
}

// ========== 隐藏风味 ==========

/// 获取指定回合开始时获得的隐藏风味数量（仅考虑新友人卡 id: 30305）。
pub fn get_turn_special_feeling(turn: i32) -> i32 {
    match turn {
        2 | 24 | 36 | 48 | 60 => 2,
        37 | 38 | 39 | 61 | 62 | 63 => 1,
        _ => 0
    }
}

// ========== 地区词条加成 ==========

/// 根据当年累计剧本 PT 计算地区词条加成档位数值。
///
/// 每 300 点 PT 提升一档，最高第 5 档（1500+ PT）。
pub fn calc_region_bonus(scenario_pt: i32) -> i32 {
    const BONUS_TABLE: [i32; 6] = [0, 3, 5, 7, 9, 10];
    let tier = (scenario_pt / 300).min(5) as usize;
    BONUS_TABLE[tier]
}

// ========== 分身系统 ==========

/// 获取地区拉面分身的训练位置列表。
///
/// 返回 `at_trains` 字段的 clone。
/// 地区分身条件（id >= 5 且 card_type_count >= 4）应在游戏逻辑中判定。
///
/// # 错误
/// `region_id` 超出范围时返回错误。
pub fn get_region_clone_trains(region_id: usize) -> Result<Vec<i32>> {
    let ramen_data = global!(RAMENDATA);
    let effect = ramen_data
        .ramen_region_effect
        .get(region_id)
        .ok_or_else(|| anyhow::anyhow!("region_id 越界: {region_id}"))?;
    Ok(effect.at_trains.clone())
}

/// 获取超级拉面分身的训练范围选项。
///
/// 返回 `training_limit_options` 的 clone。
/// 超级拉面分身条件（card_type_count >= 4）应在游戏逻辑中判定。
pub fn get_super_ramen_clone_train_options() -> Result<Vec<Vec<i32>>> {
    let ramen_data = global!(RAMENDATA);
    Ok(ramen_data.finals_effect.training_limit_options.clone())
}

/// NPC 相关常量
///
/// 5 个固定 NPC 的 chara_id（美妙/怒涛/内恰/成田路/金镇）。
pub const NPC_CHARA_IDS: &[u32] = &[1022, 1058, 1060, 1077, 1120];

// ========== 训练诀窍槽加成 ==========

/// 计算某个训练类型的诀窍槽额外加成量。
///
/// 公式：`1 + 支援卡数量 + floor(NPC 数量 / 2)`
/// 支援卡数量不包括 NPC、记者和理事长。
pub fn calc_train_feeling_bonus(support_count: usize, npc_count: usize) -> i32 {
    (1 + support_count + npc_count / 2) as i32
}

/// 应用友情训练的诀窍槽加成（三种各 +2，上限 GAUGE_LIMIT）。
pub fn apply_friendship_gauge_bonus(state: &mut RamenState) {
    for i in 0..3 {
        state.feeling_slot[i] = (state.feeling_slot[i] + 2).min(GAUGE_LIMIT);
    }
}

/// 夏合宿"全 MAX"：三种槽都直接补到 `GAUGE_LIMIT`，溢出自动 +1 诀窍。
///
/// 按 `ramen_memo.md` 原始规则：带新友人(30305)时夏合宿"全習得ゲージMAX"。
/// 不区分基础值/训练加成/友情加成——所有类型一律填满。
fn fill_gauge_xiahesu_max(state: &mut RamenState) {
    for i in 0..3 {
        if let Ok(ft) = FeelingType::try_from(i as i32) {
            // 补差额即可：`add_gauge` 内部在达到上限时清零 +1 诀窍
            let need = GAUGE_LIMIT - state.feeling_slot[i];
            if need > 0 {
                add_gauge(state, ft, need);
            }
        }
    }
}

/// 处理训练后的诀窍槽填充：基础值 + 训练加成 + 友情加成。
///
/// - `base_dist`: 三种类型的基础分配量
/// - `train_type`: 本回合训练角标（A/B/C）
/// - `train_bonus`: 训练额外加成量
/// - `is_shining`: 是否为友情训练
/// - `is_xiahesu`: 是否处于夏合宿回合
///
/// 夏合宿特殊规则：无论基础值/训练加成/友情加成如何分配，三种槽一律直接填满
/// 到上限（参见 `fill_gauge_xiahesu_max`）。
pub fn fill_gauge_after_train(
    state: &mut RamenState, base_dist: &[i32; 3], train_type: FeelingType, train_bonus: i32, is_shining: bool,
    is_xiahesu: bool
) {
    if is_xiahesu {
        fill_gauge_xiahesu_max(state);
        return;
    }
    // 1. 基础值
    for i in 0..3 {
        if let Ok(ft) = FeelingType::try_from(i as i32) {
            add_gauge(state, ft, base_dist[i]);
        }
    }
    // 2. 训练角标加成
    add_gauge(state, train_type, train_bonus);
    // 3. 友情训练加成
    if is_shining {
        apply_friendship_gauge_bonus(state);
    }
}

/// 处理非训练动作（比赛/休息/外出/友人出行/治病）后的诀窍槽填充：仅基础值。
///
/// - `base_dist`: 三种类型的基础分配量
/// - `is_xiahesu`: 是否处于夏合宿回合
///
/// 夏合宿特殊规则：三种槽一律直接填满到上限（参见 `fill_gauge_xiahesu_max`）。
///
/// 注：治病的调用方不应使用本函数（按用户确认，治病不获得诀窍槽）。
pub fn fill_gauge_after_non_train(state: &mut RamenState, base_dist: &[i32; 3], is_xiahesu: bool) {
    if is_xiahesu {
        fill_gauge_xiahesu_max(state);
        return;
    }
    for i in 0..3 {
        if let Ok(ft) = FeelingType::try_from(i as i32) {
            add_gauge(state, ft, base_dist[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    #[test]
    fn test_gauge_base_distribution() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;
        // ramen_memo 中"使用新友人"(base_sum=10) 的全部算例
        // 地区索引: 札幌=0, 函馆=1, 新潟=2, 福岛=3, 东京=4, 中山=5, 中京=6, 京都=7, 小仓=9
        let cases: &[([usize; 3], &str)] = &[
            ([2, 3, 6], "新潟福島中京"),
            ([0, 3, 6], "札幌福島中京"),
            ([0, 3, 9], "札幌福島小倉"),
            ([0, 6, 9], "札幌中京小倉"),
            ([3, 7, 9], "福島京都小倉"),
            ([0, 3, 7], "札幌福島京都"),
            ([0, 6, 7], "札幌中京京都"),
            ([0, 1, 6], "札幌函館中京"),
            ([0, 4, 6], "札幌東京中京"),
            ([0, 5, 6], "札幌中山中京"),
            ([5, 6, 7], "中山中京京都")
        ];
        for &(regions, name) in cases {
            let dist = calc_gauge_base_distribution(&regions);
            let mut sorted = dist;
            sorted.sort_by(|a, b| b.cmp(a));
            let ramen_data = global!(RAMENDATA);
            let mut actual_sum = [0i32; 3];
            for &r in &regions {
                let f = &ramen_data.region_feeling[r];
                for j in 0..3 {
                    actual_sum[j] += f[j];
                }
            }
            println!("{name}: 配方 {:?} 分配 {:?} 降序 {:?}", actual_sum, dist, sorted);
        }
        Ok(())
    }

    #[test]
    fn test_feeling_overflow() {
        let mut state = RamenState::default();
        // 先加 5A, 3B, 2C，共 10 个，顺序为 [A,A,A,A,A,B,B,B,C,C]
        add_feeling(&mut state, FeelingType::A, 5);
        add_feeling(&mut state, FeelingType::B, 3);
        add_feeling(&mut state, FeelingType::C, 2);
        println!("初始状态:");
        println!(
            "  库存 A={} B={} C={}",
            state.feeling_stock[0], state.feeling_stock[1], state.feeling_stock[2]
        );
        println!("  总数 {}", state.feeling_stock.iter().sum::<i32>());
        println!("  队列 {:?}", state.feeling_queue);

        // 再加 1 个 B，总数将超上限 10，应丢弃队列最前面的 A
        println!("\n加 1 个 B (总数将超过上限 {}):", FEELING_LIMIT);
        add_feeling(&mut state, FeelingType::B, 1);
        println!(
            "  库存 A={} B={} C={}",
            state.feeling_stock[0], state.feeling_stock[1], state.feeling_stock[2]
        );
        println!("  总数 {}", state.feeling_stock.iter().sum::<i32>());
        println!("  队列 {:?}", state.feeling_queue);
        println!("  => 队列最前面是 A，丢弃 1 个 A → A=4 B=4 C=2");
    }

    #[test]
    fn test_gauge_overflow() {
        let mut state = RamenState::default();
        state.feeling_slot[0] = 5;
        println!("初始槽值 A={}", state.feeling_slot[0]);

        // 5+3=8 >= 7，溢出，清零，获得 1 个诀窍，超出部分不保留
        println!("\n诀窍槽 A +3 (5+3=8 >= 上限 {}):", GAUGE_LIMIT);
        let gained = add_gauge(&mut state, FeelingType::A, 3);
        println!("  溢出! 槽值清零 (超出的 1 点不保留)");
        println!("  获得诀窍 A +{gained}");
        println!("  槽值 A={}", state.feeling_slot[0]);
        println!("  库存 A={}", state.feeling_stock[0]);
    }

    #[test]
    fn test_train_feeling_bonus() {
        // 公式: 1 + 支援卡数量 + floor(NPC数量 / 2)
        let (sc, npc) = (2, 3);
        let bonus = calc_train_feeling_bonus(sc, npc);
        println!(
            "支援卡={sc} NPC={npc}: 1 + {sc} + {npc}/2 = 1 + {sc} + {} = {bonus}",
            npc / 2
        );

        let (sc, npc) = (4, 5);
        let bonus = calc_train_feeling_bonus(sc, npc);
        println!(
            "支援卡={sc} NPC={npc}: 1 + {sc} + {npc}/2 = 1 + {sc} + {} = {bonus}",
            npc / 2
        );
    }

    // ========== 做面/吃面测试 ==========

    #[test]
    fn test_can_make_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 获取配方
        let recipe = get_recipe(0).expect("札幌配方应存在");
        println!("札幌配方: {recipe:?}");

        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5];
        // 无隐藏风味，足够
        println!("库存={:?}, special={}", state.feeling_stock, state.special_feeling);
        println!(
            "不使用隐藏风味: can_make={}",
            can_make_ramen(&state, recipe, &[0, 0, 0])
        );

        // 库存不足的情况
        state.feeling_stock = [1, 5, 5];
        println!("\n库存={:?} => A不足", state.feeling_stock);
        println!("can_make={}", can_make_ramen(&state, recipe, &[0, 0, 0]));

        // 用隐藏风味弥补 A
        state.special_feeling = 1;
        println!(
            "special=1, 替换A: can_make={}",
            can_make_ramen(&state, recipe, &[1, 0, 0])
        );
        // 隐藏风味不够用
        state.special_feeling = 0;
        println!(
            "special=0, 替换A: can_make={}",
            can_make_ramen(&state, recipe, &[1, 0, 0])
        );

        // get_recipe 测试（年3地区通过取模映射到年1配方）
        assert!(get_recipe(0).is_ok());
        assert!(get_recipe(15).is_ok()); // 年3地区，映射到 region 5
        println!("get_recipe 测试通过");

        Ok(())
    }

    #[test]
    fn test_consume_for_ramen() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5];
        state.feeling_queue = vec![
            FeelingType::A,
            FeelingType::A,
            FeelingType::A,
            FeelingType::A,
            FeelingType::A,
            FeelingType::B,
            FeelingType::B,
            FeelingType::B,
            FeelingType::B,
            FeelingType::B,
            FeelingType::C,
            FeelingType::C,
            FeelingType::C,
            FeelingType::C,
            FeelingType::C,
        ];
        println!(
            "初始: stock={:?}, queue_len={}",
            state.feeling_stock,
            state.feeling_queue.len()
        );

        // 札幌 [2,2,1], 不使用隐藏风味
        let used = consume_for_ramen(&mut state, 0, &[0, 0, 0])?;
        println!(
            "消耗后: stock={:?}, queue_len={}, used_special={}",
            state.feeling_stock,
            state.feeling_queue.len(),
            used
        );

        // 使用隐藏风味：替换 1A + 1B
        state.special_feeling = 2;
        let used = consume_for_ramen(&mut state, 0, &[1, 1, 0])?;
        println!(
            "再做一次(替换1A+1B): stock={:?}, special={}, used_special={}",
            state.feeling_stock, state.special_feeling, used
        );

        // 验证手动选择替换：配方 [1,1,3] (东京=4)，替换 2C
        state.feeling_stock = [5, 5, 5];
        state.special_feeling = 2;
        state.feeling_queue = vec![FeelingType::A; 5]
            .into_iter()
            .chain(vec![FeelingType::B; 5])
            .chain(vec![FeelingType::C; 5])
            .collect();
        let used = consume_for_ramen(&mut state, 4, &[0, 0, 2])?;
        println!(
            "东京[1,1,3], 替换2C: stock={:?}, special={}, used_special={}",
            state.feeling_stock, state.special_feeling, used
        );
        // 预期: A=5-1=4, B=5-1=4, C=5-1=4, special=2-2=0

        Ok(())
    }

    #[test]
    fn test_consume_for_ramen_errors() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5];

        // 无效 recipe_idx
        let result = consume_for_ramen(&mut state, 999, &[0, 0, 0]);
        println!("无效 recipe_idx: {result:?}");

        // 超过配方消耗
        let result = consume_for_ramen(&mut state, 0, &[3, 0, 0]);
        println!("special_targets[0]=3 超过配方消耗: {result:?}");

        // 隐藏风味不足
        state.special_feeling = 0;
        let result = consume_for_ramen(&mut state, 0, &[1, 0, 0]);
        println!("隐藏风味不足: {result:?}");

        // 负值
        let result = consume_for_ramen(&mut state, 0, &[-1, 0, 0]);
        println!("负值: {result:?}");

        Ok(())
    }

    #[test]
    fn test_calc_ramen_pt_gain() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // gain_pt_base = [300, 400, 500], gain_pt_delta = [30, 40, 50]
        let ramen_data = global!(RAMENDATA);
        println!("gain_pt_base: {:?}", ramen_data.gain_pt_base);
        println!("gain_pt_delta: {:?}", ramen_data.gain_pt_delta);

        // 第1年第1面(eat_count=0): 300 + 30*0 = 300
        let pt = calc_ramen_pt_gain(0, 0)?;
        println!("第1年第1面: {pt}");

        // 第1年第4面(eat_count=3): 300 + 30*3 = 390
        let pt = calc_ramen_pt_gain(0, 3)?;
        println!("第1年第4面: {pt}");

        // 第1年第6面(eat_count=5, 已上限): 300 + 30*5 = 450
        let pt = calc_ramen_pt_gain(0, 5)?;
        println!("第1年第6面: {pt}");

        // 第1年 eat_count=10 超过5, 按5算: 300 + 30*5 = 450
        let pt = calc_ramen_pt_gain(0, 10)?;
        println!("第1年 eat_count=10: {pt}");

        // 第3年第1面: 500 + 50*0 = 500
        let pt = calc_ramen_pt_gain(2, 0)?;
        println!("第3年第1面: {pt}");

        // year_idx 越界
        let result = calc_ramen_pt_gain(3, 0);
        println!("year_idx=3 越界: {result:?}");
        assert!(result.is_err());

        Ok(())
    }

    // ========== RMJ 结算测试 ==========

    #[test]
    fn test_check_rmj() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let ramen_data = global!(RAMENDATA);
        println!("ramen_success_pt: {:?}", ramen_data.ramen_success_pt);

        // 第1年: 阈值1500
        let mut state = RamenState::default();
        state.scenario_pt = 1500;
        let result = check_rmj(&mut state, 0);
        println!("第1年 pt=1500: {result:?}");
        assert!(result.is_success());
        assert!(!result.is_great_success());

        state.scenario_pt = 1499;
        let result = check_rmj(&mut state, 0);
        println!("第1年 pt=1499: {result:?}");
        assert!(!result.is_success());

        // 第3年: 阈值3500
        state.scenario_pt = 3500;
        let result = check_rmj(&mut state, 2);
        println!("第3年 pt=3500: {result:?}");
        assert!(result.is_success());
        assert!(!result.is_great_success());

        // 第3年: >=5000为大成功
        state.scenario_pt = 5000;
        let result = check_rmj(&mut state, 2);
        println!("第3年 pt=5000 (大成功): {result:?}");
        assert!(result.is_success());
        assert!(result.is_great_success());

        // 第3年: 4999 只是普通成功
        state.scenario_pt = 4999;
        let result = check_rmj(&mut state, 2);
        println!("第3年 pt=4999: {result:?}");
        assert!(result.is_success());
        assert!(!result.is_great_success());

        println!("rmj_results: {:?}", state.rmj_results);
        Ok(())
    }

    // ========== 地区选择测试 ==========

    #[test]
    fn test_get_region_range() -> anyhow::Result<()> {
        let range0 = get_region_range(0)?;
        println!("第1年可选地区: {range0:?}");
        assert_eq!(range0, vec![0, 1, 2, 3, 4]);

        let range1 = get_region_range(1)?;
        println!("第2年可选地区: {range1:?}");
        assert_eq!(range1, vec![5, 6, 7, 8, 9]);

        let range2 = get_region_range(2)?;
        println!("第3年可选地区: {range2:?}");
        assert_eq!(range2.len(), 10);
        assert_eq!(range2[0], 10);
        assert_eq!(range2[9], 19);

        // year_idx 越界
        assert!(get_region_range(3).is_err());
        println!("year_idx 越界验证通过");

        Ok(())
    }

    #[test]
    fn test_validate_region_selection() {
        // 合法选择
        assert!(validate_region_selection(0, &[0, 2, 4]));
        assert!(validate_region_selection(1, &[5, 7, 9]));
        assert!(validate_region_selection(2, &[10, 15, 19]));
        println!("合法选择验证通过");

        // 超出范围
        assert!(!validate_region_selection(0, &[0, 2, 5]));
        assert!(!validate_region_selection(1, &[4, 7, 9]));
        println!("超出范围验证通过");

        // 重复选择
        assert!(!validate_region_selection(0, &[0, 0, 4]));
        assert!(!validate_region_selection(2, &[10, 15, 10]));
        println!("重复选择验证通过");

        // 无效年份
        assert!(!validate_region_selection(3, &[0, 1, 2]));
        println!("无效年份验证通过");
    }

    // ========== 隐藏风味测试 ==========

    #[test]
    fn test_get_turn_special_feeling() {
        // 固定回合获得2个
        for &turn in &[2, 24, 36, 48, 60] {
            let amount = get_turn_special_feeling(turn);
            println!("回合{turn}: 隐藏风味={amount}");
            assert_eq!(amount, 2);
        }
        // 固定回合获得1个
        for &turn in &[37, 38, 39, 61, 62, 63] {
            let amount = get_turn_special_feeling(turn);
            println!("回合{turn}: 隐藏风味={amount}");
            assert_eq!(amount, 1);
        }
        // 其他回合为0
        for &turn in &[0, 1, 10, 23, 35, 50, 70] {
            let amount = get_turn_special_feeling(turn);
            println!("回合{turn}: 隐藏风味={amount}");
            assert_eq!(amount, 0);
        }
    }

    // ========== 地区词条加成测试 ==========

    #[test]
    fn test_calc_region_bonus() {
        // 0-299: 档0, 加成0
        println!("PT=0: bonus={}", calc_region_bonus(0));
        assert_eq!(calc_region_bonus(0), 0);
        println!("PT=299: bonus={}", calc_region_bonus(299));
        assert_eq!(calc_region_bonus(299), 0);

        // 300-599: 档1, 加成3
        println!("PT=300: bonus={}", calc_region_bonus(300));
        assert_eq!(calc_region_bonus(300), 3);

        // 600-899: 档2, 加成5
        println!("PT=600: bonus={}", calc_region_bonus(600));
        assert_eq!(calc_region_bonus(600), 5);

        // 900-1199: 档3, 加成7
        println!("PT=1000: bonus={}", calc_region_bonus(1000));
        assert_eq!(calc_region_bonus(1000), 7);

        // 1200-1499: 档4, 加成9
        println!("PT=1200: bonus={}", calc_region_bonus(1200));
        assert_eq!(calc_region_bonus(1200), 9);

        // 1500+: 档5, 加成10
        println!("PT=1500: bonus={}", calc_region_bonus(1500));
        assert_eq!(calc_region_bonus(1500), 10);
        println!("PT=5000: bonus={}", calc_region_bonus(5000));
        assert_eq!(calc_region_bonus(5000), 10);
    }

    // ========== 分身系统测试 ==========

    #[test]
    fn test_get_region_clone_trains() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 地区 5 (中山): at_trains = [0,1,2,3,4]
        let trains = get_region_clone_trains(5)?;
        println!("地区5(中山) 分身位置: {trains:?}");
        assert_eq!(trains, vec![0, 1, 2, 3, 4]);

        // 地区 6 (中京): at_trains = [2,3]
        let trains = get_region_clone_trains(6)?;
        println!("地区6(中京) 分身位置: {trains:?}");
        assert_eq!(trains, vec![2, 3]);

        // id < 5 的地区 at_trains 只有1个位置
        let trains = get_region_clone_trains(0)?;
        println!("地区0(札幌) 分身位置: {trains:?}");
        assert_eq!(trains, vec![0]);

        // 无效 id 返回错误
        assert!(get_region_clone_trains(999).is_err());
        println!("id=999 越界验证通过");

        Ok(())
    }

    #[test]
    fn test_get_super_ramen_clone_train_options() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let options = get_super_ramen_clone_train_options()?;
        println!("超级拉面训练选项: {options:?}");
        assert_eq!(options.len(), 3);
        // 选项1: 速/耐/根/智 [0,1,3,4]
        assert_eq!(options[0], vec![0, 1, 3, 4]);
        // 选项2: 速/耐/力/智 [0,1,2,4]
        assert_eq!(options[1], vec![0, 1, 2, 4]);
        // 选项3: 速/力/根/智 [0,2,3,4]
        assert_eq!(options[2], vec![0, 2, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_npc_chara_ids() {
        println!("NPC chara_ids: {:?}", NPC_CHARA_IDS);
        assert_eq!(NPC_CHARA_IDS.len(), 5);
        assert_eq!(NPC_CHARA_IDS[0], 1022); // 美妙
        assert_eq!(NPC_CHARA_IDS[1], 1058); // 怒涛
        assert_eq!(NPC_CHARA_IDS[2], 1060); // 内恰
        assert_eq!(NPC_CHARA_IDS[3], 1077); // 成田路
        assert_eq!(NPC_CHARA_IDS[4], 1120); // 金镇
    }

    // ========== list_special_targets_for 测试 ==========

    /// 构造测试用 RamenState（直接写库存与特殊风味，避免依赖 add_feeling 的副作用）。
    fn make_state_for_targets(stock: [i32; 3], special: i32) -> RamenState {
        let mut s = RamenState::default();
        s.feeling_stock = stock;
        s.special_feeling = special;
        s
    }

    #[test]
    fn test_list_special_targets_full_stock_sapporo_9() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // 札幌 (idx=0) recipe = [2,2,1]，全富余 9 种
        let state = make_state_for_targets([5, 5, 5], 4);
        let targets = list_special_targets_for(&state, 0)?;
        println!("札幌全富余 special=4: {targets:?}");
        assert_eq!(targets.len(), 9);
        // sum 升序检查
        let sums: Vec<i32> = targets.iter().map(|t| t.iter().sum()).collect();
        let mut sorted = sums.clone();
        sorted.sort();
        assert_eq!(sums, sorted);
        // 验证包含 [0,0,0]
        assert!(targets.contains(&[0, 0, 0]));
        // 验证包含 [2,0,0]（替换 2 个 A）
        assert!(targets.contains(&[2, 0, 0]));
        Ok(())
    }

    #[test]
    fn test_list_special_targets_min_needed_a3b1c1() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // recipe[2] = [3,1,1]（用户例子 A3B1C1），库存 A 缺 1 个
        let state = make_state_for_targets([2, 5, 5], 4);
        let targets = list_special_targets_for(&state, 2)?;
        println!("A3B1C1 A缺1: {targets:?}");
        // min_needed=[1,0,0], need_sum=1, budget=1 → sum(t) ≤ 2
        // 4 种: [1,0,0] [1,1,0] [1,0,1] [2,0,0]
        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0], [1, 0, 0]);
        // 升序：sum 依次为 1, 2, 2, 2
        let sums: Vec<i32> = targets.iter().map(|t| t.iter().sum()).collect();
        assert_eq!(sums, vec![1, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_list_special_targets_impossible() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // A3B1C1，库存 A=0（缺 3 个），special=1：need_sum=3 > budget=1 → 不可做
        let state = make_state_for_targets([0, 5, 5], 1);
        let targets = list_special_targets_for(&state, 2)?;
        println!("A3B1C1 A=0 special=1 (不可做): {targets:?}");
        assert!(targets.is_empty());
        Ok(())
    }

    #[test]
    fn test_list_special_targets_no_special_feeling() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // 札幌 [2,2,1]，全够，special=0：仅 [0,0,0]
        let state = make_state_for_targets([5, 5, 5], 0);
        let targets = list_special_targets_for(&state, 0)?;
        println!("札幌 special=0: {targets:?}");
        assert_eq!(targets, vec![[0, 0, 0]]);
        Ok(())
    }

    #[test]
    fn test_list_special_targets_recipe_with_zero_dim() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // recipe[7] = [0, 3, 2]（含 0 维度），全富余
        let state = make_state_for_targets([5, 5, 5], 4);
        let targets = list_special_targets_for(&state, 7)?;
        println!("recipe [0,3,2] 全富余: {targets:?}");
        // tA 必须为 0；sum(t) ≤ 2
        // 候选：[0,0,0] [0,1,0] [0,0,1] [0,2,0] [0,1,1] [0,0,2]
        assert_eq!(targets.len(), 6);
        for t in &targets {
            assert_eq!(t[0], 0, "维度 A 必须为 0");
        }
        Ok(())
    }

    #[test]
    fn test_list_special_targets_sorted_ascending() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_global()?;
        let _ = crate::utils::init_test_logger("info");
        // 遍历所有合法 recipe_idx，检查排序
        let state = make_state_for_targets([5, 5, 5], 4);
        let ramen_data = global!(RAMENDATA);
        for idx in 0..ramen_data.region_feeling.len() {
            let targets = list_special_targets_for(&state, idx)?;
            let sums: Vec<i32> = targets.iter().map(|t| t.iter().sum()).collect();
            let mut sorted = sums.clone();
            sorted.sort();
            println!(
                "recipe[{idx}] {:?} 候选数={} sums={sums:?}",
                ramen_data.region_feeling[idx],
                targets.len()
            );
            assert_eq!(sums, sorted, "recipe[{idx}] 排序错误");
        }
        Ok(())
    }

    // ========== 夏合宿 + 训练 / 非训练 填充测试 ==========

    /// 夏合宿"全 MAX"：三种槽都补到 GAUGE_LIMIT，溢出自动 +1 诀窍
    #[test]
    fn test_fill_gauge_after_train_xiahesu_max() {
        let mut state = RamenState::default();
        // 初始槽值 [0, 3, 6]
        state.feeling_slot = [0, 3, 6];
        // 初始库存 [0, 0, 0]
        state.feeling_stock = [0, 0, 0];
        println!("初始: 槽 {:?} 库存 {:?}", state.feeling_slot, state.feeling_stock);

        // 夏合宿触发 fill_gauge_after_train
        fill_gauge_after_train(
            &mut state,
            &[5, 4, 1],
            FeelingType::A,
            4,
            false,
            true // is_xiahesu
        );

        println!(
            "夏合宿 fill_gauge_after_train 后: 槽 {:?} 库存 {:?}",
            state.feeling_slot, state.feeling_stock
        );

        // 三种槽都填到上限后清零，每种 +1 诀窍
        assert_eq!(state.feeling_slot, [0, 0, 0]);
        assert_eq!(state.feeling_stock, [1, 1, 1]);
    }

    /// 非夏合宿 fill_gauge_after_train：走原有"基础值+训练加成+友情加成"路径
    #[test]
    fn test_fill_gauge_after_train_normal() {
        let mut state = RamenState::default();
        state.feeling_slot = [0, 0, 0];
        state.feeling_stock = [0, 0, 0];

        // base_dist=[5,4,1], train_type=C, train_bonus=3, is_shining=false
        fill_gauge_after_train(
            &mut state,
            &[5, 4, 1],
            FeelingType::C,
            3,
            false,
            false // is_xiahesu = false
        );

        // A=5, B=4, C=1+3=4，都没有到上限
        assert_eq!(state.feeling_slot, [5, 4, 4]);
        assert_eq!(state.feeling_stock, [0, 0, 0]);

        // 友情训练：三种各 +2，上限 7
        let mut state2 = RamenState::default();
        state2.feeling_slot = [5, 4, 4];
        fill_gauge_after_train(
            &mut state2,
            &[0, 0, 0],
            FeelingType::A,
            0,
            true, // 友情训练
            false
        );
        assert_eq!(state2.feeling_slot, [7, 6, 6]);
    }

    /// 非训练动作 + 夏合宿：三种槽直接 +MAX
    #[test]
    fn test_fill_gauge_after_non_train_xiahesu() {
        let mut state = RamenState::default();
        state.feeling_slot = [1, 2, 3];
        state.feeling_stock = [0, 0, 0];

        // 即使 base_dist 全 0，夏合宿时也强制全 MAX
        fill_gauge_after_non_train(&mut state, &[0, 0, 0], true);

        println!(
            "夏合宿 fill_gauge_after_non_train 后: 槽 {:?} 库存 {:?}",
            state.feeling_slot, state.feeling_stock
        );
        assert_eq!(state.feeling_slot, [0, 0, 0]);
        assert_eq!(state.feeling_stock, [1, 1, 1]);
    }

    /// 非训练动作 + 非夏合宿：仅基础值填充
    #[test]
    fn test_fill_gauge_after_non_train_normal() {
        let mut state = RamenState::default();
        state.feeling_slot = [0, 0, 0];
        state.feeling_stock = [0, 0, 0];

        // base_dist=[5, 4, 1]
        fill_gauge_after_non_train(&mut state, &[5, 4, 1], false);

        println!(
            "非夏合宿 fill_gauge_after_non_train 后: 槽 {:?} 库存 {:?}",
            state.feeling_slot, state.feeling_stock
        );
        assert_eq!(state.feeling_slot, [5, 4, 1]);
        assert_eq!(state.feeling_stock, [0, 0, 0]);
    }

    /// 夏合宿 fill_gauge_after_non_train：部分槽已接近上限，验证自动溢出 +1 诀窍
    #[test]
    fn test_fill_gauge_after_non_train_xiahesu_partial() {
        let mut state = RamenState::default();
        state.feeling_slot = [5, 6, 7]; // C 已满
        state.feeling_stock = [2, 2, 2];

        fill_gauge_after_non_train(&mut state, &[0, 0, 0], true);

        // C 已满：add_gauge(0, 0) = 0，不会重复清零
        // A: 5+2=7 -> +1 诀窍 -> 0
        // B: 6+1=7 -> +1 诀窍 -> 0
        // C: 7+0=7 -> 不变（已满）
        assert_eq!(state.feeling_slot, [0, 0, 7]);
        assert_eq!(state.feeling_stock, [3, 3, 2]);
    }
}
