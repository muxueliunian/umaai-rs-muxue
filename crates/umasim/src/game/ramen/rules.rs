//! 拉面杯核心规则
//!
//! 诀窍系统、做面/吃面、RMJ 结算等核心机制的纯函数实现。
//! 函数以 `RamenState` 或相关数据为参数，不直接修改游戏状态的其他部分。

use super::{FeelingType, RamenState};
use crate::{gamedata::ramen::RAMENDATA, global};

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

    // 累加三个配方的各类型消耗
    let mut recipe_sum = [0i32; 3];
    for &region_idx in selected_regions {
        let feeling = &ramen_data.region_feeling[region_idx];
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
                    if result[i] < result[b]
                        || (result[i] == result[b] && recipe_sum[i] > recipe_sum[b])
                    {
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

/// 处理训练后的诀窍槽填充：基础值 + 训练加成 + 友情加成。
///
/// - `base_dist`: 三种类型的基础分配量
/// - `train_type`: 本回合训练角标（A/B/C）
/// - `train_bonus`: 训练额外加成量
/// - `is_shining`: 是否为友情训练
pub fn fill_gauge_after_train(
    state: &mut RamenState,
    base_dist: &[i32; 3],
    train_type: FeelingType,
    train_bonus: i32,
    is_shining: bool
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_logger}};

    #[test]
    fn test_gauge_base_distribution() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
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
            ([5, 6, 7], "中山中京京都"),
        ];
        for &(regions, name) in cases {
            let dist = calc_gauge_base_distribution(&regions);
            let mut sorted = dist;
            sorted.sort_by(|a, b| b.cmp(a));
            let ramen_data = global!(RAMENDATA);
            let mut actual_sum = [0i32; 3];
            for &r in &regions {
                let f = &ramen_data.region_feeling[r];
                for j in 0..3 { actual_sum[j] += f[j]; }
            }
            println!(
                "{name}: 配方 {:?} 分配 {:?} 降序 {:?}",
                actual_sum, dist, sorted
            );
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
        println!("  库存 A={} B={} C={}", state.feeling_stock[0], state.feeling_stock[1], state.feeling_stock[2]);
        println!("  总数 {}", state.feeling_stock.iter().sum::<i32>());
        println!("  队列 {:?}", state.feeling_queue);

        // 再加 1 个 B，总数将超上限 10，应丢弃队列最前面的 A
        println!("\n加 1 个 B (总数将超过上限 {}):", FEELING_LIMIT);
        add_feeling(&mut state, FeelingType::B, 1);
        println!("  库存 A={} B={} C={}", state.feeling_stock[0], state.feeling_stock[1], state.feeling_stock[2]);
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
        println!("支援卡={sc} NPC={npc}: 1 + {sc} + {npc}/2 = 1 + {sc} + {} = {bonus}", npc / 2);

        let (sc, npc) = (4, 5);
        let bonus = calc_train_feeling_bonus(sc, npc);
        println!("支援卡={sc} NPC={npc}: 1 + {sc} + {npc}/2 = 1 + {sc} + {} = {bonus}", npc / 2);
    }
}
