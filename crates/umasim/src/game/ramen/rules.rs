//! 拉面杯核心规则
//!
//! 诀窍系统、做面/吃面、RMJ 结算等核心机制的纯函数实现。
//! 函数以 `RamenState` 或相关数据为参数，不直接修改游戏状态的其他部分。

use super::{FeelingType, RamenState};
use crate::gamedata::ramen::RAMENDATA;

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
    let ramen_data = match RAMENDATA.get() {
        Some(d) => d,
        None => return [4, 3, 3] // fallback
    };
    let base_sum = 10;

    // 累加三个配方的各类型消耗
    let mut recipe_sum = [0i32; 3];
    for &region_idx in selected_regions {
        if let Some(feeling) = ramen_data.region_feeling.get(region_idx) {
            for j in 0..3 {
                recipe_sum[j] += feeling[j];
            }
        }
    }

    // 按比例分配，四舍五入
    let mut result = [0i32; 3];
    let total_consumed: i32 = recipe_sum.iter().sum();
    if total_consumed == 0 {
        // fallback: 均分
        return [4, 3, 3];
    }
    for i in 0..3 {
        result[i] = (recipe_sum[i] as f64 * base_sum as f64 / total_consumed as f64).round() as i32;
    }

    // 调整使总和等于 base_sum
    let diff = base_sum - result.iter().sum::<i32>();
    if diff != 0 {
        // 按消耗量从大到小调整
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&a, &b| recipe_sum[b].cmp(&recipe_sum[a]));
        for &i in &indices {
            if diff == 0 {
                break;
            }
            if diff > 0 {
                result[i] += 1;
            } else if result[i] > 0 {
                result[i] -= 1;
            }
        }
    }

    result
}

// ========== 诀窍槽操作 ==========

/// 向指定类型的诀窍槽增加数值，满 GAUGE_LIMIT 则清零并获得 1 个诀窍。
///
/// 返回实际获得的诀窍数量（0 或 1）。
pub fn add_gauge(state: &mut RamenState, feeling_type: FeelingType, amount: i32) -> i32 {
    let idx = feeling_type as usize;
    state.feeling_slot[idx] += amount;
    if state.feeling_slot[idx] >= GAUGE_LIMIT {
        state.feeling_slot[idx] %= GAUGE_LIMIT;
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

    #[test]
    fn test_gauge_base_distribution() {
        // 札幌[2,2,1] + 函馆[1,2,2] + 新潟[3,1,1] = [6,5,4], sum=15
        let dist = calc_gauge_base_distribution(&[0, 1, 2]);
        println!("[0,1,2] 札幌函館新潟 => {dist:?}, sum={}", dist.iter().sum::<i32>());
        assert_eq!(dist.iter().sum::<i32>(), 10);

        // 札幌[2,2,1] + 福岛[2,3,0] + 中京[3,2,0] = [7,7,1], sum=15
        let dist = calc_gauge_base_distribution(&[0, 3, 6]);
        println!("[0,3,6] 札幌福島中京 => {dist:?}, sum={}", dist.iter().sum::<i32>());
        assert_eq!(dist.iter().sum::<i32>(), 10);
    }

    #[test]
    fn test_feeling_overflow() {
        let mut state = RamenState::default();
        // 填满 10 个 A
        add_feeling(&mut state, FeelingType::A, 10);
        println!("初始: {:?}", state.feeling_stock);
        println!("队列: {:?}", state.feeling_queue);
        assert_eq!(state.feeling_stock.iter().sum::<i32>(), FEELING_LIMIT);

        // 再加 1 个 B，应该丢弃最早的 A
        add_feeling(&mut state, FeelingType::B, 1);
        println!("加B后: {:?}", state.feeling_stock);
        println!("队列: {:?}", state.feeling_queue);
        assert_eq!(state.feeling_stock.iter().sum::<i32>(), FEELING_LIMIT);
        assert_eq!(state.feeling_stock[1], 1); // B = 1
        assert_eq!(state.feeling_stock[0], 9); // A = 9（丢了一个）
    }

    #[test]
    fn test_gauge_overflow() {
        let mut state = RamenState::default();
        state.feeling_slot[0] = 5;
        let gained = add_gauge(&mut state, FeelingType::A, 3);
        println!("槽值: {:?}, 获得诀窍: {gained}", state.feeling_slot);
        println!("库存: {:?}", state.feeling_stock);
        assert_eq!(gained, 1);
        assert_eq!(state.feeling_slot[0], 1); // 5+3=8, 满7清零, 余1
        assert_eq!(state.feeling_stock[0], 1);
    }

    #[test]
    fn test_train_feeling_bonus() {
        // 支援卡=2, NPC=3 => 1+2+1=4
        let bonus = calc_train_feeling_bonus(2, 3);
        println!("支援卡2 NPC3 => bonus={bonus}");
        assert_eq!(bonus, 4);

        // 支援卡=4, NPC=5 => 1+4+2=7
        let bonus = calc_train_feeling_bonus(4, 5);
        println!("支援卡4 NPC5 => bonus={bonus}");
        assert_eq!(bonus, 7);
    }
}
