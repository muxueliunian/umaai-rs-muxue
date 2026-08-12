use std::{
    collections::BTreeMap, fmt::Display, num::NonZeroU32, sync::{Mutex, OnceLock}
};

use anyhow::{Result, anyhow};
use flexi_logger::LoggerHandle;
use hashbrown::HashMap;
use log::info;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use int_enum::IntEnum;
use crate::{
    explain::Explain,
    game::{UmaFlags, onsen::OnsenOrder},
    global,
    utils::{Array5, Array6}
};
pub mod event;
pub use event::*;
pub mod uma;
pub use uma::*;
pub mod support_card;
pub use support_card::*;
pub mod config;
pub use config::*;

pub mod onsen;
pub mod ramen;
#[derive(Clone, Debug)]
pub struct GameData {
    pub uma: BTreeMap<String, UmaData>,
    pub card: BTreeMap<String, SupportCardData>,
    pub text: BTreeMap<String, BTreeMap<String, String>>,
    pub events: EventCollection
}

pub fn load_json<T: DeserializeOwned>(path: &str) -> Result<T> {
    info!("载入数据 {path}");
    Ok(serde_json::from_str(&fs_err::read_to_string(path)?)?)
}

impl GameData {
    pub fn load() -> Result<Self> {
        let mut uma: BTreeMap<String, UmaData> = load_json("gamedata/umaDB.json")?;
        let card: BTreeMap<_, _> = load_json("gamedata/cardDB.json")?;
        let text = load_json("gamedata/text_data_dict.json")?;
        let events = load_json("gamedata/events.json")?;
        info!("载入 {} 马娘, {} 支援卡", uma.len(), card.len());
        // 处理free race mask
        for uma in uma.values_mut() {
            for f in uma.free_races.iter_mut() {
                f.update_turn_mask();
            }
        }
        Ok(Self { uma, card, text, events })
    }

    pub fn get_uma(&self, id: u32) -> Result<&UmaData> {
        self.uma
            .get(&id.to_string())
            .ok_or_else(|| anyhow!("未找到 id={id} 的马娘，需要更新数据"))
    }

    pub fn get_card(&self, id: u32) -> Result<&SupportCardData> {
        self.card
            .get(&id.to_string())
            .ok_or_else(|| anyhow!("未找到 id={id} 的支援卡，需要更新数据"))
    }

    pub fn get_chara_name(&self, chara_id: u32) -> &str {
        self.text["6"]
            .get(&chara_id.to_string())
            .map(|x| x.as_str())
            .unwrap_or("未知")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use anyhow::Result;

    use super::*;
    use crate::utils::{init_logger, make_table};

    #[test]
    fn test_uma_data() -> Result<()> {
        let uma_data: HashMap<String, UmaData> = serde_json::from_str(&fs_err::read_to_string("gamedata/umaDB.json")?)?;
        let umas: Vec<_> = uma_data.values().take(10).collect();
        println!("{}", make_table(&umas)?);
        Ok(())
    }

    #[test]
    fn test_support_data() -> Result<()> {
        let support_data: HashMap<String, SupportCardData> =
            serde_json::from_str(&fs_err::read_to_string("gamedata/cardDB.json")?)?;
        let cards: Vec<_> = support_data.values().skip(300).take(10).collect();
        println!("{:#?}", cards);
        Ok(())
    }

    #[test]
    fn test_consts() -> Result<()> {
        init_logger("test", "info")?;
        let consts = GameConstants::load()?;
        println!("{:?}", consts);

        println!("{}", consts.get_rank_name(63399));
        Ok(())
    }

    #[test]
    fn test_turn_mask() -> Result<()> {
        GAMECONSTANTS.set(GameConstants::load()?).expect("global constants");
        init_logger("test", "info")?;
        let mut free_race = FreeRaceData {
            start_turn: 24,
            end_turn: 47,
            count: 1,
            grade: Some(1),
            mask: 0
        };
        free_race.update_turn_mask(); // 只有G1会被标1
        println!("{:b}", free_race.mask); // 10111010000111110100000000000000000000
        Ok(())
    }
}

pub static GAMEDATA: OnceLock<GameData> = OnceLock::new();
pub static GAMECONSTANTS: OnceLock<GameConstants> = OnceLock::new();
pub static LOGGER: OnceLock<Mutex<LoggerHandle>> = OnceLock::new();

pub fn init_global() -> Result<()> {
    GAMECONSTANTS.set(GameConstants::load()?).expect("global constants");
    GAMEDATA.set(GameData::load()?).expect("global gamedata");
    onsen::init_onsen_data()?;
    ramen::init_ramen_data()?;
    Ok(())
}
