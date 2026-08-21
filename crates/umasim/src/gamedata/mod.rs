#[cfg(feature = "cli")]
use std::sync::Mutex;
use std::{collections::BTreeMap, sync::OnceLock};

use anyhow::{Result, anyhow};
#[cfg(feature = "cli")]
use flexi_logger::LoggerHandle;
use log::info;
use serde::de::DeserializeOwned;
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
    use crate::utils::get_workspace_root;
    #[cfg(feature = "cli")]
    use crate::utils::{init_test_logger, make_table};

    #[cfg(feature = "cli")]
    #[test]
    fn test_uma_data() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let uma_data: HashMap<String, UmaData> = serde_json::from_str(&fs_err::read_to_string("gamedata/umaDB.json")?)?;
        let umas: Vec<_> = uma_data.values().take(10).collect();
        println!("{}", make_table(&umas)?);
        Ok(())
    }

    #[test]
    fn test_support_data() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let support_data: HashMap<String, SupportCardData> =
            serde_json::from_str(&fs_err::read_to_string("gamedata/cardDB.json")?)?;
        let cards: Vec<_> = support_data.values().skip(300).take(10).collect();
        println!("{:#?}", cards);
        Ok(())
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_consts() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        let consts = GameConstants::load()?;
        println!("{:?}", consts);

        println!("{}", consts.get_rank_name(63399));
        Ok(())
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_turn_mask() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = GAMECONSTANTS.set(GameConstants::load()?);
        let _ = init_test_logger("info");
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
pub static GAMECONFIG: OnceLock<GameConfig> = OnceLock::new();
/// 全局 LoggerHandle（仅 cli feature 下编译）。
///
/// 仅 `utils::init_logger*` 写入，core-only 构建不使用，故 cfg gate。
/// core-only 消费者（如 .so/嵌入式）若需要日志，直接使用 `log` crate 自行配置即可。
#[cfg(feature = "cli")]
pub static LOGGER: OnceLock<Mutex<LoggerHandle>> = OnceLock::new();

/// 初始化全局游戏数据。
///
/// Phase 2 步骤 1（2026-08-19）：`mcts_turn_bonus` / `pt_favor_rate` / `race_grades`
/// 已从 `gamedata/constants.json` 迁出到 `gamedata/default_config.toml`（可由
/// `game_config.toml` 覆盖）。`init_global` 现在接受 `&GameConfig`，把用户可调项
/// 注入 `GAMECONSTANTS`，保持 `global!(GAMECONSTANTS).xxx` 引用点不变。
///
/// 旧的无参签名保留为便捷重载，但只能在已加载过 `GameConfig` 后调用（仅用兜底默认值）。
/// 入口应优先使用 `init_global_with_config(&GameConfig)`。
pub fn init_global() -> Result<()> {
    // 幂等：已初始化过则直接返回
    if GAMECONSTANTS.get().is_some() && GAMEDATA.get().is_some() {
        return Ok(());
    }
    // 兜底：使用 GameConfig::default() 提供的默认用户可调值
    init_global_with_config(&GameConfig::default_for_init())
}

/// 带 GameConfig 的初始化：注入用户可调项（mcts_turn_bonus / pt_favor_rate / race_grades）
/// 到 `GAMECONSTANTS`，供 `global!(GAMECONSTANTS)` 读取。
pub fn init_global_with_config(config: &GameConfig) -> Result<()> {
    // 幂等
    if GAMECONSTANTS.get().is_some() && GAMEDATA.get().is_some() {
        return Ok(());
    }
    // 加载 GameConstants（基础部分），再注入用户可调项
    let mut constants = GameConstants::load()?;
    // 注入用户可调项（来源 GameConfig，默认/用户覆盖）
    // 注意：若后续需要把 race_grades/mcts_turn_bonus 完全从 GameConstants 移除，
    // 改为新建全局或扩展其他结构。当前为了保持所有 `cons.race_grades` 等引用点不变，
    // 选择注入式实现（Phase 2 步骤 1 临时方案）。
    constants.race_grades = config.race_grades.clone();
    constants.mcts_turn_bonus = config.mcts_turn_bonus;
    constants.pt_favor_rate = config.pt_favor_rate;
    let _ = GAMECONSTANTS.set(constants);
    let _ = GAMEDATA.set(GameData::load()?);
    onsen::init_onsen_data()?;
    ramen::init_ramen_data()?;
    // 注入 GameConfig 副本（Phase 2 步骤 5：策略模块读取 PolicyConfig 使用）
    let _ = GAMECONFIG.set(config.clone());
    Ok(())
}
