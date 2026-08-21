use std::collections::HashMap;

use anyhow::{Result, anyhow};
use int_enum::IntEnum;
use serde::{Deserialize, Serialize};

use crate::{
    explain::Explain,
    game::UmaFlags,
    gamedata::{GAMECONSTANTS, GAMEDATA},
    global,
    utils::Array6
};

/// 事件触发类型
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TriggerType {
    /// 随机事件，任意回合可触发
    Random {
        /// 最大触发次数, 0为无限
        #[serde(default)]
        max_time: u32
    },
    /// 代码生成的临时事件，不会触发仅列出（默认）
    #[default]
    Code,
    /// 固定回合触发
    Fixed {
        /// 触发回合列表
        turns: Vec<i32>
    }
}

/// 训练或事件数值
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionValue {
    /// 基础属性
    #[serde(default)]
    pub status_pt: Array6,
    /// 体力
    #[serde(default)]
    pub vital: i32,
    /// 最大体力
    #[serde(default)]
    pub max_vital: i32,
    /// 干劲
    #[serde(default)]
    pub motivation: i32,
    /// Hint等级
    #[serde(default)]
    pub hint_level: i32,
    /// 羁绊
    #[serde(default)]
    pub friendship: i32
}

impl ActionValue {
    pub fn explain(&self) -> String {
        let mut s = Explain::status_with_pt(&self.status_pt);
        if self.vital != 0 {
            s += &format!(" 体力{}", self.vital);
        }
        if self.max_vital != 0 {
            s += &format!(" 最大体力+{}", self.max_vital);
        }
        if self.friendship != 0 {
            s += &format!(" 羁绊+{}", self.friendship);
        }
        if self.motivation != 0 {
            s += &format!(" 干劲{}", self.motivation);
        }
        if self.hint_level != 0 {
            s += &format!(" Hint+{}", self.hint_level);
        }
        s
    }

    /// 对五维属性和pt进行映射, 例如全属性x3，用于事件效果提高时的计算
    pub fn map_status<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(i32) -> i32
    {
        for i in 0..6 {
            self.status_pt[i] = f(self.status_pt[i]);
        }
        self
    }
}

impl std::fmt::Display for ActionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.explain())
    }
}

/// 剧本事件信息，也用于临时生成一些固定事件如赛后
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventData {
    /// ID, 必须不同, 游戏内ID为9位数，自定义的位数更少
    pub id: u32,
    /// 名字
    pub name: String,
    /// 对应第几张卡或者理事长记者，计算时随机指定，不在数据里
    #[serde(default)]
    pub person_index: Option<i32>,
    /// 触发类型
    #[serde(default)]
    pub trigger: TriggerType,
    /// 选项是否可以选择, true -> 可以选择, false -> 自动选择
    #[serde(default)]
    pub player_select: bool,
    /// 属性奖励(随机改为平均) 速耐力根智pt，体力
    #[serde(default)]
    pub choices: Vec<Vec<EventChoice>>
}

/// 事件选项结果
#[repr(i32)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, IntEnum)]
pub enum ChoiceResult {
    #[default]
    Normal = 0,
    Success = 1,
    BigSuccess = 2,
    Fail = 3,
    BigFail = 4
}

impl std::fmt::Display for ChoiceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChoiceResult::Normal => write!(f, ""),
            ChoiceResult::Success => write!(f, "[成功]"),
            ChoiceResult::BigSuccess => write!(f, "[大成功]"),
            ChoiceResult::Fail => write!(f, "[失败]"),
            ChoiceResult::BigFail => write!(f, "[大失败]")
        }
    }
}

/// 剧本事件某一个可能结果的数值和出现概率
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EventChoice {
    /// 选项结果，可以转为[ChoiceResult]
    #[serde(default)]
    pub result: i32,
    /// 出现概率，0为默认，100为必定触发, 暂时不考虑根据属性改变的概率
    #[serde(default)]
    pub prob: i32,
    /// 效果
    #[serde(default)]
    pub value: ActionValue,
    /// 添加的状态
    #[serde(default)]
    pub add_flags: Option<UmaFlags>,
    /// 移除的状态
    #[serde(default)]
    pub remove_flags: Option<UmaFlags>
}

impl EventChoice {
    /// 从ActionValue和指定的ChoiceResult结果类型生成
    pub fn from_action_value(value: &ActionValue, result: ChoiceResult) -> Self {
        Self {
            result: result.into(),
            prob: 100,
            value: value.clone(),
            add_flags: None,
            remove_flags: None
        }
    }

    pub fn explain(&self) -> String {
        let mut words = vec![
            ChoiceResult::try_from(self.result).unwrap_or_default().to_string(),
            self.value.explain(),
        ];
        if let Some(flags) = &self.add_flags {
            words.push(format!("+{}", flags.explain()));
        }
        if let Some(flags) = &self.remove_flags {
            words.push(format!("-{}", flags.explain()));
        }
        if self.prob > 0 && self.prob < 100 {
            words.push(format!("({}%)", self.prob));
        }
        words.join(" ")
    }

    /// 对五维属性和pt进行映射, 例如全属性x3，用于事件效果提高时的计算
    pub fn map_status<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(i32) -> i32
    {
        self.value.map_status(f);
        self
    }
}

impl EventData {
    /// 解释事件内容，输出事件名称和每个选项组的效果
    pub fn explain(&self) -> String {
        let first = format!(
            "[{}] {} {}",
            self.id,
            self.name,
            if self.player_select { "[选择]-->  " } else { "" }
        );
        let mut lines = vec![first];
        for (i, group) in self.choices.iter().enumerate() {
            lines.push(format!("  选项{}: {}", i + 1, Explain::event_choice(group)));
        }
        lines.join("\n")
    }

    /// 返回第一个选项和可能性
    pub fn default_choice(&self) -> &EventChoice {
        &self.choices[0][0]
    }

    /// 对所有选项的五维属性和pt进行映射, 例如全属性x3，用于事件效果提高时的计算
    pub fn map_status<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(i32) -> i32
    {
        for choice_group in self.choices.iter_mut() {
            for choice in choice_group.iter_mut() {
                choice.map_status(&f);
            }
        }
        self
    }
    /// 红点属性事件
    pub fn hint_attr_event(train: usize, person_index: usize) -> Result<Self> {
        if train < 5 {
            let train_name = global!(GAMECONSTANTS).train_names[train].clone();
            let value = ActionValue {
                status_pt: global!(GAMECONSTANTS).hint_event_value[train],
                friendship: 5,
                ..Default::default()
            };
            let choice = EventChoice::from_action_value(&value, ChoiceResult::Normal);
            Ok(Self {
                id: 101,
                name: format!("Hint - {train_name}属性"),
                person_index: Some(person_index as i32),
                choices: vec![vec![choice]],
                ..Default::default()
            })
        } else {
            Err(anyhow!("train越界: {train}"))
        }
    }

    /// 红点技能事件
    pub fn hint_skill_event(hint_level: i32, person_index: usize) -> Self {
        let value = ActionValue {
            status_pt: [0, 0, 0, 0, 0, 0],
            hint_level,
            friendship: 5,
            ..Default::default()
        };
        let choice = EventChoice::from_action_value(&value, ChoiceResult::Normal);
        Self {
            id: 101,
            name: format!("Hint - 技能"),
            person_index: Some(person_index as i32),
            choices: vec![vec![choice]],
            ..Default::default()
        }
    }

    /// 加练事件
    pub fn extra_training_event(train: usize) -> Self {
        let mut ret = global!(GAMEDATA).events.system_events["extra_train"].clone();
        ret.choices[0][0].value.status_pt[train] = 5;
        ret
    }

    /// 从ActionValue生成基础事件
    pub fn from_action_value(id: u32, name: &str, value: &ActionValue) -> Self {
        let choice = EventChoice::from_action_value(value, ChoiceResult::Success);
        Self {
            id,
            name: name.to_string(),
            choices: vec![vec![choice]],
            ..Default::default()
        }
    }
}

/// 事件数据表
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventCollection {
    /// 剧本必发事件
    pub story_events: Vec<EventData>,
    /// 马娘正面事件
    pub uma_events: Vec<EventData>,
    /// 支援卡连续事件
    pub card_events: Vec<EventData>,
    /// 友人事件
    pub friend_events: HashMap<String, EventData>,
    /// 系统事件
    pub system_events: HashMap<String, EventData>
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::{GAMECONSTANTS, GameConstants, load_json},
        utils::get_workspace_root
    };

    /// 从workspace根目录的gamedata/events.json载入EventCollection，并分别explain各类事件
    ///
    /// 例如: cargo test -p umasim test_load_and_explain_all_events -- --nocapture
    #[test]
    fn test_load_and_explain_all_events() -> Result<()> {
        // 切换到workspace根目录，以便正确加载gamedata目录下的文件
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;

        // 初始化GAMECONSTANTS（幂等：已初始化则跳过），explain依赖train_names
        let _ = GAMECONSTANTS.set(GameConstants::load()?);

        let events: EventCollection = load_json("gamedata/events.json")?;

        println!("=== story_events ({} 条) ===", events.story_events.len());
        for e in &events.story_events {
            println!("{}", e.explain());
        }

        println!("=== uma_events ({} 条) ===", events.uma_events.len());
        for e in &events.uma_events {
            println!("{}", e.explain());
        }

        println!("=== card_events ({} 条) ===", events.card_events.len());
        for e in &events.card_events {
            println!("{}", e.explain());
        }

        println!("=== friend_events ({} 条) ===", events.friend_events.len());
        for e in events.friend_events.values() {
            println!("{}", e.explain());
        }

        println!("=== system_events ({} 条) ===", events.system_events.len());
        for e in events.system_events.values() {
            println!("{}", e.explain());
        }

        Ok(())
    }

    /// 从workspace根目录的gamedata/scenario_ramen.json载入拉面剧本事件，并显示内容
    ///
    /// 例如: cargo test -p umasim test_load_and_explain_ramen_events -- --nocapture
    #[test]
    fn test_load_and_explain_ramen_events() -> Result<()> {
        use crate::gamedata::ramen::RamenScenarioData;

        // 切换到workspace根目录，以便正确加载gamedata目录下的文件
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;

        // 初始化GAMECONSTANTS（幂等：已初始化则跳过），explain依赖train_names
        let _ = GAMECONSTANTS.set(GameConstants::load()?);

        let ramen_data = RamenScenarioData::load()?;

        println!("=== scenario_events ({} 条) ===", ramen_data.scenario_events.len());
        for e in &ramen_data.scenario_events {
            println!("{}", e.explain());
        }

        println!("\n=== friend_events ({} 条) ===", ramen_data.friend_events.len());
        for e in ramen_data.friend_events.values() {
            println!("{}", e.explain());
        }

        Ok(())
    }
}
