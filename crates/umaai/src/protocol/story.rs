use anyhow::bail;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use umasim::{
    explain::Explain,
    gamedata::{ActionValue, EventData, GAMECONSTANTS},
    global,
    utils::Array6
};

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

impl StoryStatus {
    pub fn explain(&self) -> String {
        let mut lines = vec![];
        lines.push(
            format!("事件 #{} [{}]{}", self.id, self.trigger_name, self.name)
                .bright_yellow()
                .to_string()
        );
        for (i, ch) in self.choices.iter().enumerate() {
            if let Some(c) = ch.first() {
                lines.push(format!("选项 {}: {}", i + 1, c.explain()));
            }
        }
        lines.join("\n")
    }
}

impl TryFrom<&StoryStatus> for EventData {
    type Error = anyhow::Error;
    /// 暂时只考虑SuccessEffect数值
    fn try_from(value: &StoryStatus) -> anyhow::Result<Self> {
        let mut choices = vec![];
        for ch in &value.choices {
            if let Some(success_value) = ch.first().and_then(|x| x.success_effect_value.as_ref()) {
                choices.push(ActionValue::from(success_value))
            } else {
                bail!("success_effect_value is None. StoryStatus: {value:#?}");
            }
        }
        //log::info!("{choices:?}");
        Ok(EventData {
            id: value.id,
            name: value.name.clone(),
            //   prob: 100,
            //   choices,
            ..Default::default()
        })
    }
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

impl StoryChoice {
    pub fn explain(&self) -> String {
        let mut lines = vec![];
        lines.push(format!("- {}", self.option));
        let mut effect_line = ">> ".to_string();
        if let Some(value) = &self.success_effect_value {
            effect_line += &value.explain();
        }
        if let Some(value) = &self.failed_effect_value {
            effect_line += &format!(" / {}", value.explain());
        }
        lines.push(format!("{}", effect_line.cyan()));
        lines.join("\n")
    }

    /// 选项的默认成功概率(0.4)
    pub fn default_success_rate() -> f64 {
        global!(GAMECONSTANTS)
            .event_probs
            .get("default_success_event")
            .cloned()
            .unwrap_or(0.4)
    }
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
            max_vital: 0 // 暂无这个字段
        }
    }
}
