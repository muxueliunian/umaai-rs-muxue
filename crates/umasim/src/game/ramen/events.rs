//! 拉面杯剧本事件处理
//!
//! 包含友人事件链状态管理、隐藏风味分配和训练角标分配。
//!
//! 注意：合宿判断使用 `BaseGame::is_xiahesu()`，超级拉面/RMJ 回合判断使用 `RamenGame` 的方法。

use rand::{Rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::utils::Array5;

// ========== 友人事件链 ==========

/// 友人事件状态
///
/// 友人从第 2 回合开始出现在训练中，经历登场→解锁→出行的完整流程。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FriendEventState {
    /// 未出现
    #[default]
    NotAppear,
    /// 已登场（首次点击友人在的训练后）
    Appeared,
    /// 已解锁（回合开始随机触发解锁事件后）
    Unlocked,
    /// 出行中（已出行次数 0~4）
    Outing(u8),
    /// 完成所有出行（5 次出行结束）
    Complete
}

impl FriendEventState {
    /// 友人是否从本回合开始出现在训练中（回合 >= 2 且未完成）
    pub fn is_visible(&self, turn: i32) -> bool {
        turn >= 2 && *self != Self::Complete
    }

    /// 是否处于已登场或之后的状态（可用于判定"点击友人"事件）
    pub fn has_appeared(&self) -> bool {
        !matches!(self, Self::NotAppear)
    }

    /// 是否可以选择"友人出行"动作
    pub fn can_outing(&self) -> bool {
        matches!(self, Self::Unlocked | Self::Outing(_))
    }

    /// 处理首次选择友人在的训练，触发"友人登场"事件
    pub fn on_appear(&mut self) -> bool {
        if *self == Self::NotAppear {
            *self = Self::Appeared;
            true
        } else {
            false
        }
    }

    /// 处理回合开始时的"友人解锁"事件判定
    pub fn on_unlock(&mut self) -> bool {
        if *self == Self::Appeared {
            *self = Self::Unlocked;
            true
        } else {
            false
        }
    }

    /// 处理"友人出行"，返回是否成功执行
    pub fn on_outing(&mut self) -> bool {
        match *self {
            Self::Unlocked => {
                *self = Self::Outing(1);
                true
            }
            Self::Outing(n) if n < 5 => {
                if n + 1 >= 5 {
                    *self = Self::Complete;
                } else {
                    *self = Self::Outing(n + 1);
                }
                true
            }
            _ => false
        }
    }
}

// ========== 回合隐藏风味分配 ==========

/// 获取指定回合开始时获得的隐藏风味数量（简化为新友人情况）
///
/// 固定回合获得隐藏风味，其他回合返回 0。
/// 与 `rules::get_turn_special_feeling` 保持一致。
pub fn get_turn_special_feeling(turn: i32) -> i32 {
    match turn {
        2 | 24 | 36 | 48 | 60 => 2,
        37 | 38 | 39 | 61 | 62 | 63 => 1,
        _ => 0
    }
}

// ========== 训练角标分配 ==========

/// 随机分配本回合训练角标（A/B/C）
///
/// 每回合每个训练随机分配一个诀窍类型角标，该角标决定训练
/// 额外加成作用于哪个诀窍槽。
/// 每种诀窍类型至少出现1次（前3个位置打乱保证覆盖，后2个随机）。
///
/// # 返回值
/// `Array5`（`[i32; 5]`），分别对应速度/耐力/力量/根性/智力训练的诀窍类型角标（0=A, 1=B, 2=C）。
pub fn assign_train_feeling_type(rng: &mut impl Rng) -> Array5 {
    let mut result = [0i32; 5];
    // 前3个位置保证每种类型各出现1次
    let mut base = [0, 1, 2];
    base.shuffle(rng);
    result[0] = base[0];
    result[1] = base[1];
    result[2] = base[2];
    // 后2个位置随机
    result[3] = rng.random_range(0..3);
    result[4] = rng.random_range(0..3);
    result
}

// ========== 事件 ID 常量 ==========

/// 友人登场事件 ID
pub const EVENT_FRIEND_APPEAR: u32 = 830305101;
/// 点击友人事件 ID
pub const EVENT_FRIEND_CLICK: u32 = 830305102;
/// 友人解锁事件 ID
pub const EVENT_FRIEND_UNLOCK: u32 = 830305103;

// ========== 单元测试 ==========

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_friend_event_state_lifecycle() {
        let mut state = FriendEventState::default();
        println!("初始状态: {state:?}");
        assert_eq!(state, FriendEventState::NotAppear);

        // 未登场时不可出行
        println!("未登场时 can_outing: {}", state.can_outing());
        assert!(!state.can_outing());

        // 登场
        let triggered = state.on_appear();
        println!("触发登场: {triggered}, 状态: {state:?}");
        assert!(triggered);
        assert_eq!(state, FriendEventState::Appeared);
        assert!(state.has_appeared());

        // 重复登场不触发
        let triggered = state.on_appear();
        println!("重复登场: {triggered}");
        assert!(!triggered);

        // 解锁
        let triggered = state.on_unlock();
        println!("触发解锁: {triggered}, 状态: {state:?}");
        assert!(triggered);
        assert_eq!(state, FriendEventState::Unlocked);
        assert!(state.can_outing());

        // 出行 5 次
        for i in 1..=5 {
            let ok = state.on_outing();
            println!("第 {i} 次出行: {ok}, 状态: {state:?}");
            assert!(ok);
        }
        assert_eq!(state, FriendEventState::Complete);
        assert!(!state.can_outing());
    }

    #[test]
    fn test_friend_visibility() {
        println!("回合 1 未出现: {}", FriendEventState::NotAppear.is_visible(1));
        assert!(!FriendEventState::NotAppear.is_visible(1));

        println!("回合 2 未出现: {}", FriendEventState::NotAppear.is_visible(2));
        assert!(FriendEventState::NotAppear.is_visible(2));

        println!("回合 2 已登场: {}", FriendEventState::Appeared.is_visible(2));
        assert!(FriendEventState::Appeared.is_visible(2));

        println!("回合 5 已完成: {}", FriendEventState::Complete.is_visible(5));
        assert!(!FriendEventState::Complete.is_visible(5));
    }

    #[test]
    fn test_turn_special_feeling() {
        let expected: &[(i32, i32)] = &[
            (2, 2),
            (24, 2),
            (36, 2),
            (37, 1),
            (38, 1),
            (39, 1),
            (48, 2),
            (60, 2),
            (61, 1),
            (62, 1),
            (63, 1)
        ];
        for &(turn, amount) in expected {
            let result = get_turn_special_feeling(turn);
            println!("回合 {turn} 隐藏风味: {result}");
            assert_eq!(result, amount, "回合 {turn} 期望 {amount} 实际 {result}");
        }

        // 其他回合返回 0
        let zero_turns = [0, 1, 3, 10, 23, 35, 47, 59, 64, 72];
        for turn in zero_turns {
            let result = get_turn_special_feeling(turn);
            println!("回合 {turn} 隐藏风味: {result}");
            assert_eq!(result, 0, "回合 {turn} 期望 0 实际 {result}");
        }
    }

    #[test]
    fn test_assign_train_feeling_type() {
        let mut rng = StdRng::seed_from_u64(42);
        for round in 0..5 {
            let types = assign_train_feeling_type(&mut rng);
            println!("第 {round} 轮角标: {types:?}");
            for (i, &ft) in types.iter().enumerate() {
                assert!((0..3).contains(&ft), "训练 {i} 角标无效: {ft}");
            }
        }

        // 验证分布大致均匀（大量样本后 0/1/2 都应出现）
        let mut counts = [0u32; 3];
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..1000 {
            let types = assign_train_feeling_type(&mut rng);
            for ft in types {
                counts[ft as usize] += 1;
            }
        }
        println!(
            "1000 轮 x 5 训练角标分布: A={} B={} C={}",
            counts[0], counts[1], counts[2]
        );
        // 期望每种约 1667，允许较大偏差
        for &c in &counts {
            assert!(c > 1000, "分布过偏: {counts:?}");
            assert!(c < 2500, "分布过偏: {counts:?}");
        }
    }

    #[test]
    fn test_event_ids() {
        println!("友人登场: {EVENT_FRIEND_APPEAR}");
        println!("点击友人: {EVENT_FRIEND_CLICK}");
        println!("友人解锁: {EVENT_FRIEND_UNLOCK}");
        assert_eq!(EVENT_FRIEND_APPEAR, 830305101);
        assert_eq!(EVENT_FRIEND_CLICK, 830305102);
        assert_eq!(EVENT_FRIEND_UNLOCK, 830305103);
    }
}
