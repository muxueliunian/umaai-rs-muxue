//! 拉面杯剧本数据

use std::sync::OnceLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::gamedata::{TrainingBasicTable, load_json};

/// 拉面杯剧本数据
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenScenarioData {
    /// 剧本ID = 14
    pub scenario_id: i32,
    /// 链接角色ID
    pub link_chara_id: Vec<i32>,
    /// 训练基础值表格
    pub training_basic_value: TrainingBasicTable,
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
