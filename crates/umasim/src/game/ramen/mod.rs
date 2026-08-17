//! 拉面杯剧本模块
//!
//! 拉面杯围绕诀窍（feeling）和拉面展开，核心机制包括：
//! - 三种诀窍（A/B/C）库存管理
//! - 拉面配方和制作
//! - 年度地区选择
//! - 超级拉面（72-77 回合自动生效）
//! - 组合动作（吃面 + 基础操作）

pub mod action;
pub mod effects;
pub mod events;
pub mod game;
pub mod policy;
pub mod rules;
pub mod state;

pub use action::*;
pub use state::*;

use enum_iterator::Sequence;
use int_enum::IntEnum;
use serde::{Deserialize, Serialize};

/// 拉面杯回合阶段
///
/// 拉面杯在普通回合的基础上增加了地区选择和超级拉面选择阶段。
/// 超级拉面选择初期固定为选项二，不做独立决策。
///
/// 可操作部分（Train）拆为三阶段状态机：
/// - `RamenSelect`：选择吃哪碗面（含不吃）→ 写入 `pending_ramen`
/// - `SpecialSelect`：选择隐藏风味用法（仅在 `pending_ramen` 非 None 时进入）→ 写入 `pending_special_targets`
/// - `Train`：选择基础操作（与现有 Operation 一致）
///
/// 每个阶段都是一次 `run_stage` 调用，由外部 `run_full_game` 按 stage 顺序驱动。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, Sequence)]
pub enum RamenStage {
    /// 1. 回合开始，随机事件
    #[default]
    Begin,
    /// 2. 分配人头前，随机事件
    Distribute,
    // --- 可操作部分（三阶段）
    /// 3a. 选择吃哪碗面（含不吃）
    RamenSelect,
    /// 3b. 选择隐藏风味用法（仅在选了面时进入；不吃面时短路跳过）
    SpecialSelect,
    /// 3c. 选择训练或比赛（含吃面执行）
    Train,
    /// 4. 回合后事件
    AfterTrain,
    // --- 特殊阶段
    /// 推进到下一回合（处理回合边界逻辑）
    NextTurn,
    /// 年度地区选择（回合 23/47/71 结束后，RMJ 结算后）
    RegionSelect,
    /// 超级拉面选择（第 71 回合结束后）
    SuperRamenSelect,
    /// 剧本结算（回合 23/47/71 结束时）
    Settlement
}

impl RamenStage {
    /// 获取回合内的下一个阶段，如果已到回合末尾则返回 None
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Begin => Some(Self::Distribute),
            Self::Distribute => Some(Self::RamenSelect),
            Self::RamenSelect => Some(Self::SpecialSelect),
            Self::SpecialSelect => Some(Self::Train),
            Self::Train => Some(Self::AfterTrain),
            Self::AfterTrain => Some(Self::NextTurn),
            // NextTurn 在 run_stage 中推进回合，回到 Begin 或特殊阶段
            Self::NextTurn => None,
            // 特殊阶段处理后回到 Begin
            Self::RegionSelect | Self::SuperRamenSelect | Self::Settlement => None
        }
    }
}

/// 诀窍类型
///
/// 拉面杯有三种诀窍类型，用于配方消耗和诀窍槽系统。
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, IntEnum)]
pub enum FeelingType {
    /// 诀窍 A
    A = 0,
    /// 诀窍 B
    B = 1,
    /// 诀窍 C
    C = 2
}

/// 训练类型（与 BaseAction 对应）
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, IntEnum)]
pub enum TrainingType {
    /// 速度
    Speed = 0,
    /// 耐力
    Stamina = 1,
    /// 力量
    Power = 2,
    /// 根性
    Guts = 3,
    /// 智力
    Wisdom = 4
}

/// 拉面基础操作（不含吃面决策）
///
/// 对应玩家在每个回合可选的非拉面操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// 训练（指定训练类型）
    Train(TrainingType),
    /// 比赛
    Race,
    /// 休息
    Rest,
    /// 普通外出
    NormalOuting,
    /// 友人出行
    FriendOuting,
    /// 治病
    Clinic,
    /// 地区选择（选择3个地区）
    RegionSelect([usize; 3]),
    /// 阶段阶段动作占位（仅承载本阶段决策，不执行任何 operation）
    ///
    /// 用于 `RamenSelect`/`SpecialSelect` 阶段的 `RamenAction`，这些阶段的决策
    /// 仅体现在 `ramen` 或 `special_targets` 字段上，不需要真正的基础操作。
    /// `apply` 看到此变体时直接切阶段、不执行任何操作。
    StageOnly,
}
