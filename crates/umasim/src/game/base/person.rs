use std::default::Default;

use serde::{Deserialize, Serialize};

use crate::{game::*, gamedata::GAMEDATA, global};

/// 训练人头信息（动态）
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BasePerson {
    /// 人头顺序。**不保证与卡组槽位同序**：base / onsen 是 0-5 支援卡、6 理事长，
    /// 但拉面是 0-4 训练卡、5 理事长、6 友人卡、7-11 NPC、12 记者。
    /// 需要由人头访问卡组时走 `Game::deck_index_of` 反查，不要拿本字段当卡组下标。
    pub person_index: i32,
    /// 人头类型
    pub person_type: PersonType,
    /// 得意训练类型，0-5:速耐力根智团 一部分npc也有 -1为没有
    pub train_type: i32,
    /// 角色ID
    pub chara_id: u32,
    /// 羁绊
    pub friendship: i32,
    /// 是否有叹号
    pub is_hint: bool,
    /// 支援卡信息
    pub card_id: Option<u32>
}

impl BasePerson {
    pub fn short_name(&self) -> String {
        let gamedata = global!(GAMEDATA);
        match self.person_type {
            PersonType::Npc => "[NPC]".to_string(),
            PersonType::Yayoi => "理事长".to_string(),
            PersonType::Reporter => "记者".to_string(),
            _ => {
                if let Some(Ok(support)) = self.card_id.map(|id| gamedata.get_card(id)) {
                    support.short_name()
                } else {
                    let short_chara_name: String = gamedata.get_chara_name(self.chara_id).chars().take(2).collect();
                    format!("[???]{short_chara_name}")
                }
            }
        }
    }

    pub fn explain(&self) -> String {
        let mut ret = self.short_name();
        if self.person_type != PersonType::Npc && self.friendship > 0 && self.friendship < 100 {
            ret = format!("{}{}", ret, self.friendship);
        }
        if self.is_hint {
            ret = format!("{}{ret}", "!");
        }
        ret
    }

    pub fn yayoi() -> Self {
        BasePerson {
            person_index: 6,
            person_type: PersonType::Yayoi,
            train_type: -1,
            chara_id: 9002,
            friendship: 0,
            is_hint: false,
            card_id: None
        }
    }

    pub fn reporter() -> Self {
        BasePerson {
            person_index: 7,
            person_type: PersonType::Reporter,
            train_type: -1,
            chara_id: 9003,
            friendship: 0,
            is_hint: false,
            card_id: None
        }
    }
}

impl Person for BasePerson {
    fn person_type(&self) -> PersonType {
        self.person_type
    }
    fn person_index(&self) -> i32 {
        self.person_index
    }
    fn train_type(&self) -> i32 {
        self.train_type
    }
    fn friendship(&self) -> i32 {
        self.friendship
    }
    fn set_hint(&mut self, hint: bool) {
        self.is_hint = hint;
    }
    fn hint(&self) -> bool {
        self.is_hint
    }
    fn card_id(&self) -> Option<u32> {
        self.card_id
    }
}

impl TryFrom<&SupportCard> for BasePerson {
    type Error = anyhow::Error;
    fn try_from(card: &SupportCard) -> Result<Self> {
        let person_type = match card.card_type {
            0..=4 => PersonType::Card,
            5 => PersonType::ScenarioCard,
            6 => PersonType::TeamCard,
            _ => PersonType::Card
        };
        Ok(BasePerson {
            person_index: 0,
            person_type,
            train_type: card.card_type,
            chara_id: card.data.chara_id,
            friendship: card.friendship,
            is_hint: false,
            card_id: Some(card.card_id)
        })
    }
}
