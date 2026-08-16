//! 拉面杯剧本数据

use std::sync::OnceLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::gamedata::{TrainingBasicTable, EventData, load_json};
use std::collections::HashMap;

/// 拉面基础效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenBasicEffect {
    /// 训练加成
    pub xunlian: i32,
    /// 友情训练加成
    pub youqing: i32,
    /// 得意率（本剧本无此效果）
    pub deyilv: i32,
    /// 失败率下降
    pub fail_rate_drop: i32,
    /// 羁绊增加
    pub friendship: i32,
    /// 属性和PT上限增加
    pub status_limit: i32,
    /// 仅第三年生效的特殊hint效果
    pub hint_special: bool,
}

/// 地区拉面效果
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegionEffect {
    /// 地区 ID
    pub id: usize,
    /// 地区名称
    pub name: String,
    /// 训练加成
    #[serde(default)]
    pub xunlian: i32,
    /// 友情训练加成
    #[serde(default)]
    pub youqing: i32,
    /// PT 加成
    #[serde(default)]
    pub pt_bonus: i32,
    /// 发动 Hint 数量
    #[serde(default)]
    pub hint_count: i32,
    /// 生效的训练位置
    #[serde(default)]
    pub at_trains: Vec<i32>
}

/// 拉面杯剧本数据
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenScenarioData {
    /// 剧本ID = 14
    pub scenario_id: i32,
    /// 链接角色ID
    pub link_chara_id: Vec<i32>,
    /// 训练基础值表格
    pub training_basic_value: TrainingBasicTable,
    /// 拉面基础效果（按年份）
    pub ramen_basic_effect: Vec<RamenBasicEffect>,
    /// 地区诀窍配方
    #[serde(default)]
    pub region_feeling: Vec<[i32; 3]>,
    /// 地区拉面效果
    #[serde(default)]
    pub ramen_region_effect: Vec<RegionEffect>,
    /// 剧本事件
    #[serde(default)]
    pub scenario_events: Vec<EventData>,
    /// 友人事件
    #[serde(default)]
    pub friend_events: HashMap<String, EventData>,
}

impl RamenScenarioData {
    /// 从 JSON 文件加载拉面杯剧本数据
    pub fn load() -> Result<Self> {
        load_json("gamedata/scenario_ramen.json")
    }

}

/// 全局拉面杯剧本数据
pub static RAMENDATA: OnceLock<RamenScenarioData> = OnceLock::new();

/// 初始化拉面杯剧本数据
pub fn init_ramen_data() -> Result<()> {
    RAMENDATA.set(RamenScenarioData::load()?).expect("ramen data");
    Ok(())
}
