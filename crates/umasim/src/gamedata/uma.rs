use serde::{Deserialize, Serialize};

use crate::{gamedata::GAMECONSTANTS, global, utils::Array5};

/// 自由比赛区间数据
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreeRaceData {
    /// 开始回合(从0开始)
    pub start_turn: u32,
    // 结束回合
    pub end_turn: u32,
    /// 比赛次数
    pub count: u32,
    /// 比赛等级, 可选
    pub grade: Option<u32>,
    /// 比赛掩码，json里不存在，载入时计算
    #[serde(default)]
    pub mask: u64
}

impl FreeRaceData {
    /// 能打的比赛设为1，其他为0
    pub fn update_turn_mask(&mut self) {
        let mut ret = 0;
        let race_grades = &global!(GAMECONSTANTS).race_grades;
        for i in self.start_turn..=self.end_turn {
            if let Some(g) = &self.grade {
                if race_grades[i as usize] <= *g as i32 {
                    ret |= 1 << (i - 11);
                }
            } else {
                ret |= 1 << (i - 11);
            }
        }
        self.mask = ret;
    }
}

/// 马娘数据 UmaDB.json
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UmaData {
    /// 马娘ID
    pub game_id: u32,
    /// 星数
    pub star: u32,
    /// 名字
    pub name: String,
    /// 五维加成
    pub five_status_bonus: Array5,
    /// 初始五维
    pub five_status_initial: Array5,
    /// 比赛回合
    pub races: Vec<i32>,
    /// 自由比赛回合
    pub free_races: Vec<FreeRaceData>
}

impl UmaData {
    pub fn short_name(&self) -> &str {
        self.name.split("]").last().unwrap_or(&self.name)
    }

    /// 把比赛回合压缩进u64位段 对应11-71回合
    pub fn zip_races(&self) -> u64 {
        let mut ret = 0;
        for race in &self.races {
            ret |= 1 << (race - 11);
        }
        ret
    }
}
