//! 拉面杯剧本数据

use std::sync::OnceLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::gamedata::{TrainingBasicTable, load_json};

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
    pub jiban: i32,
    /// 属性和PT上限增加
    pub status_limit: i32,
    /// 仅第三年生效的特殊hint效果
    pub hint_special: bool,
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
    // 后续补充其他字段...
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
