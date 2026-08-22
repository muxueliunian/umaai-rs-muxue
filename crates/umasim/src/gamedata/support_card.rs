use serde::{Deserialize, Serialize};

use crate::utils::Array6;

/// 支援卡数据 CardDB.json
/// 支援卡具体数值
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardValue {
    /// 友情
    #[serde(default, rename = "youQing")]
    pub youqing: f32,
    /// 干劲
    #[serde(default, rename = "ganJing")]
    pub ganjing: i32,
    /// 训练
    #[serde(default, rename = "xunLian")]
    pub xunlian: i32,
    /// 赛后
    #[serde(default, rename = "saiHou")]
    pub saihou: i32,
    /// 得意率
    #[serde(default, rename = "deYiLv")]
    pub deyilv: f32,
    /// 初始羁绊
    #[serde(default, rename = "initialJiBan")]
    pub initial_jiban: i32,
    /// 启发等级
    #[serde(default)]
    pub hint_level: i32,
    /// 启发概率
    #[serde(default)]
    pub hint_prob_increase: i32,
    /// 智训练体力恢复
    #[serde(default)]
    pub wiz_vital_bonus: i32,
    /// 失败率下降
    #[serde(default)]
    pub fail_rate_drop: f32,
    /// 体力消耗降低
    #[serde(default)]
    pub vital_cost_drop: f32,
    /// 事件效果提高
    #[serde(default)]
    pub event_effect_up: i32,
    /// 事件回复量提高
    #[serde(default)]
    pub event_recovery_amount_up: i32,
    /// 副属性
    pub bonus: Array6,
    /// 初始属性
    pub initial_bonus: Array6,
    /// 启发收益
    pub hint_bonus: Array6
}

/// 支援卡数据 CardDB.json
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupportCardData {
    /// 支援卡ID
    pub card_id: u32,
    /// 角色ID
    pub chara_id: u32,
    /// 卡名
    pub card_name: String,
    /// 全名
    pub full_name: String,
    /// 稀有度，123
    pub rarity: u32,
    /// 卡类型 0速1耐2力3根4智5友人6团队（对照 cardDB.json 实测：30305[友]=5，团队卡=6）
    pub card_type: i32,
    /// 数值
    pub card_value: Vec<CardValue>,
    /// 固有类型
    #[serde(default)]
    pub unique_effect_type: u32,
    /// 固有描述
    pub unique_effect_summary: Option<String>,
    /// 固有数值
    #[serde(default)]
    pub unique_effect_param: Vec<i32>
}

impl SupportCardData {
    pub fn short_name(&self) -> String {
        let parts: Vec<_> = self.card_name.split(']').collect();
        let left = parts[0];
        let right_short: String = parts[1].chars().take(2).collect();
        format!("{left}]{right_short}")
    }
}
