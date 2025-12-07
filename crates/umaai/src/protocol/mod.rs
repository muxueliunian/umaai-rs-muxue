use std::ops::Deref;

use serde::{Deserialize, Serialize};
use umasim::utils::Array5;

pub mod urafile;

/// 从小黑板接收的基础人头信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BasePersonStatus {
    /// 人头类型
    pub person_type: u32,
    /// 角色ID
    pub chara_id: u32,
    /// 羁绊
    pub friendship: i32,
    /// 是否叹号
    pub is_hint: bool
}

/// 从小黑板接收的数据的baseGame字段
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusBase {
    /// 剧本ID
    pub scenario_id: u32,
    /// 马娘ID
    pub uma_id: u32,
    /// 马娘星数
    pub uma_star: u32,
    /// 回合(0-77)
    pub turn: i32,
    /// 体力
    pub vital: i32,
    /// 最大体力
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    /// 当前属性。1200以上不减半
    pub five_status: Array5,
    /// 属性上限
    pub five_status_limit: Array5,
    /// 技能点
    pub skill_pt: i32,
    /// 已学习技能分数
    pub skill_score: i32,
    /// 训练设施等级
    pub train_level_count: Array5,
    /// PT系数
    pub pt_score_rate: f32,
    /// 失败率修正值
    pub failure_rate_bias: i32,
    /// 是否切者
    #[serde(rename = "isQieZhe")]
    pub is_qiezhe: bool,
    /// 是否爱娇
    #[serde(rename = "isAiJiao")]
    pub is_aijiao: bool,
    /// 是否正向思考
    pub is_positive_thinking: bool,
    /// 是否有休息心得
    pub is_refresh_mind: bool,
    /// 种马蓝因子数量
    #[serde(rename = "zhongMaBlueCount")]
    pub zhongma_blue_count: Array5,
    /// 是否比赛状态
    pub is_racing: bool,
    /// 卡组
    pub card_id: Vec<u32>,
    /// 人头
    pub persons: Vec<BasePersonStatus>,
    /// 人头分布
    pub person_distribution: Vec<Vec<i32>>,
    /// 是否锁定到某个训练
    pub locked_training_id: i32,
    /// 理事长羁绊
    #[serde(rename = "friendship_noncard_yayoi")]
    pub friendship_noncard_yayoi: i32,
    /// 记者羁绊
    #[serde(rename = "friendship_noncard_reporter")]
    pub friendship_noncard_reporter: i32,
    /// 友人解锁阶段
    #[serde(rename = "friend_stage")]
    pub friend_stage: i32,
    /// 友人出行阶段
    #[serde(rename = "friend_outgoingUsed")]
    pub friend_outgoing_used: i32,
    /// 回合状态
    #[serde(rename = "playing_state")]
    pub playing_state: i32
}

/// 从小黑板接收的数据
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusOnsen {
    pub base_game: GameStatusBase
}

impl Deref for GameStatusOnsen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}
