use serde::{Deserialize, Serialize};
use umasim::{explain::Explain, gamedata::ActionValue, utils::Array6};

/// 从小黑板接收的事件信息
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct StoryStatus {
    /// 事件ID
    pub id: u32,
    /// 事件名
    pub name: String,
    /// 角色名
    pub trigger_name: String,
    /// 选项数据
    pub choices: Vec<Vec<StoryChoice>>
}

/// 事件选项数据
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct StoryChoice {
    /// 选项文本
    pub option: String,
    /// 成功/大成功效果
    #[serde(default)]
    pub success_effect: String,
    /// 失败/小成功效果
    #[serde(default)]
    pub failed_effect: String,
    /// 成功数值
    pub success_effect_value: Option<StoryEffectValue>,
    /// 失败数值
    pub failed_effect_value: Option<StoryEffectValue>
}

/// 事件数值数据
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct StoryEffectValue {
    /// 事件属性，分别为：速耐力根智，pt，hint等级，体力，羁绊，干劲
    pub values: Vec<i32>,
    /// Hint技能名
    pub skill_names: Vec<String>,
    /// 其他词条
    pub extras: Vec<String>,
    /// 状态名字，可选
    pub buff_name: Option<String>
}

impl StoryEffectValue {
    /// 属性，PT
    pub fn status_pt(&self) -> Array6 {
        self.values[0..6].try_into().expect("event status_pt")
    }
    /// Hint等级
    pub fn hint_level(&self) -> i32 {
        self.values[6]
    }
    /// 体力
    pub fn vital(&self) -> i32 {
        self.values[7]
    }
    /// 羁绊
    pub fn friendship(&self) -> i32 {
        self.values[8]
    }
    /// 干劲
    pub fn motivation(&self) -> i32 {
        self.values[9]
    }

    pub fn explain(&self) -> String {
        let mut line = String::new();
        if self.status_pt() != [0; 6] { 
            line += &format!("{} ", Explain::status_with_pt(&self.status_pt()));
        }
        if self.vital() != 0 {
            line += &format!("体力{}", self.vital());
        }
        if self.friendship() != 0 { 
            line += &format!("羁绊{} ", self.friendship());
        }
        if self.motivation() != 0 { 
            line += &format!("干劲{} ", self.motivation());
        }
        if self.hint_level() > 0 { 
            line += &format!("{:?} Hint+{} ", self.skill_names, self.hint_level());
        }
        if let Some(buff) = &self.buff_name { 
            line += &format!("获得状态->{buff} ");
        }
        line += &self.extras.join("/");
        line
    }
}

impl From<&StoryEffectValue> for ActionValue {
    fn from(value: &StoryEffectValue) -> Self {
        ActionValue {
            status_pt: value.status_pt(),
            hint_level: value.hint_level(),
            vital: value.vital(),
            friendship: value.friendship(),
            motivation: value.motivation(),
            max_vital: 0    // 暂无这个字段
        }
    }
}