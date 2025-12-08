use std::ops::Deref;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use umasim::game::onsen::game::OnsenGame;

use crate::protocol::{GameStatus, GameStatusBase};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BathingStatus {
    pub ticket_num: i32,
    pub buff_remain_turn: i32,
    pub is_super_ready: bool
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnsenStatus {
    /// 当前挖掘的温泉
    pub current_onsen: usize,
    /// 温泉Buff信息
    pub bathing: BathingStatus,
    /// 温泉状态
    pub onsen_state: Vec<bool>,
    /// 当前每个温泉的剩余挖掘量
    pub dig_remain: Vec<[i32; 3]>,
    /// 已挖掘的温泉数
    pub dig_count: i32,
    /// 挖掘力加成
    pub dig_power: [i32; 3],
    /// 挖掘工具等级
    pub dig_level: [i32; 3],
    /// 挖掘花费的体力
    pub dig_vital_cost: i32,
    /// 是否需要选择温泉
    pub pending_selection: bool
}

/// 从小黑板接收的数据
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusOnsen {
    pub base_game: GameStatusBase,
    pub onsen: OnsenStatus
}

impl Deref for GameStatusOnsen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}

impl GameStatus for GameStatusOnsen {
    type Game = OnsenGame;

    fn scenario_id() -> u32 {
        12
    }

    fn into_game(self) -> Result<Self::Game> {
        todo!()
    }
}
