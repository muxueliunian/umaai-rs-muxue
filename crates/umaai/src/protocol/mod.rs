use std::sync::Arc;

use anyhow::Result;
use log::warn;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use umasim::{
    game::{BaseGame, BasePerson, FriendOutState, FriendState, InheritInfo, SupportCard, TurnStage, Uma, UmaFlags},
    gamedata::{EventData, GAMEDATA},
    global,
    utils::{Array5, load_game_config}
};

pub mod onsen;
pub mod ramen;
pub mod story;
pub mod urafile;
pub use onsen::*;
pub use ramen::*;
pub use story::*;

/// 描述不同剧本的通信状态，需要能转为对应的Game结构
pub trait GameStatus: DeserializeOwned {
    type Game;

    fn scenario_id() -> u32;

    fn into_game(self) -> Result<Self::Game>;
}

/// 从小黑板接收的基础人头信息
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BasePersonStatus {
    /// 人头类型
    pub person_type: u32,
    /// 角色ID
    pub chara_id: u32,
    /// 羁绊
    pub friendship: i32,
    /// 是否叹号
    pub is_hint: bool
}

/// 从小黑板接收的数据的baseGame字段
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusBase {
    /// 剧本ID
    pub scenario_id: u32,
    /// 马娘ID
    pub uma_id: u32,
    /// 马娘星数
    pub uma_star: u32,
    /// 回合(0-77)
    pub turn: i32,
    /// 体力
    pub vital: i32,
    /// 最大体力
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    /// 当前属性。1200以上不减半
    pub five_status: Array5,
    /// 属性上限
    pub five_status_limit: Array5,
    /// 技能点
    pub skill_pt: i32,
    /// 已学习技能分数
    pub skill_score: i32,
    /// 总Hint等级
    pub total_hints: i32,
    /// 训练设施等级
    pub train_level_count: Array5,
    /// PT系数
    pub pt_score_rate: f32,
    /// 失败率修正值
    pub failure_rate_bias: i32,
    /// 是否生病
    pub is_ill: bool,
    /// 是否切者
    #[serde(rename = "isQieZhe")]
    pub is_qiezhe: bool,
    /// 是否爱娇
    #[serde(rename = "isAiJiao")]
    pub is_aijiao: bool,
    /// 是否正向思考
    pub is_positive_thinking: bool,
    /// 是否有休息心得
    pub is_refresh_mind: bool,
    /// 是否有幸运体质
    pub is_lucky: bool,
    /// 种马蓝因子数量
    #[serde(rename = "zhongMaBlueCount")]
    pub zhongma_blue_count: Array5,
    /// 是否生涯比赛状态
    pub is_racing: bool,
    /// 卡组
    pub card_id: Vec<u32>,
    /// 人头
    pub persons: Vec<BasePersonStatus>,
    /// 人头分布
    pub person_distribution: Vec<Vec<i32>>,
    /// 是否锁定到某个训练
    pub locked_training_id: i32,
    /// 理事长羁绊
    #[serde(rename = "friendship_noncard_yayoi")]
    pub friendship_noncard_yayoi: i32,
    /// 记者羁绊
    #[serde(rename = "friendship_noncard_reporter")]
    pub friendship_noncard_reporter: i32,
    /// 友人解锁阶段
    #[serde(rename = "friend_stage")]
    pub friend_stage: i32,
    /// 友人出行阶段
    #[serde(rename = "friend_outgoingUsed")]
    pub friend_outgoing_used: i32,
    /// 回合状态
    #[serde(rename = "playing_state")]
    pub playing_state: i32,
    /// 胜场信息
    #[serde(default)]
    pub race_history: Vec<i32>,
    /// 事件信息
    pub story: Option<StoryStatus>
}

impl GameStatusBase {
    pub fn parse_uma(&self) -> Result<Uma> {
        let data = global!(GAMEDATA).get_uma(self.uma_id)?;
        let flags = UmaFlags {
            ill: self.is_ill,
            lucky: self.is_lucky,
            qiezhe: self.is_qiezhe,
            aijiao: self.is_aijiao,
            good_trainer: self.failure_rate_bias > 0,
            bad_trainer: self.failure_rate_bias < 0,
            positive_thinking: self.is_positive_thinking,
            refresh_mind: self.is_refresh_mind as i32,
            ..Default::default()
        };

        let mut ret = Uma {
            uma_id: self.uma_id,
            vital: self.vital,
            max_vital: self.max_vital,
            motivation: self.motivation,
            five_status: self.five_status.clone(),
            five_status_bonus: data.five_status_bonus.clone(),
            five_status_limit: self.five_status_limit,
            skill_pt: self.skill_pt,
            skill_score: self.skill_score,
            total_hints: self.total_hints,
            race_bonus: 0,
            flags,
            career_races: data.zip_races(),
            win_races: 0
        };
        // 设置比赛状态
        for t in &self.race_history {
            ret.set_race(*t);
        }
        //if ret.win_races != 0 {
        //    info!("win_races: {:b}", ret.win_races);
        // }
        Ok(ret)
    }

    pub fn parse_friend(&self, scenario_friend_chara_id: u32) -> Result<FriendState> {
        for (index, id) in self.card_id.iter().enumerate() {
            let card = SupportCard::new(*id)?;
            if card.card_type >= 5 && card.data.chara_id == scenario_friend_chara_id {
                let friend_id = Some(*id);
                let friend_index = index;
                let mut friend = FriendState::new(friend_id, friend_index)?;
                friend.out_state = match self.friend_stage {
                    0 => FriendOutState::UnClicked,
                    1 => FriendOutState::BeforeUnlock,
                    2 => FriendOutState::AfterUnlock,
                    _ => FriendOutState::Away
                };
                for i in 0..self.friend_outgoing_used {
                    friend.out_used[i as usize] = true;
                }
                return Ok(friend);
            }
        }
        // 如果没找到友人
        warn!("没带剧本友人? AI可能无法正常工作。请检查卡组");
        Ok(FriendState::default())
    }

    pub fn parse_inherit(&self) -> Result<InheritInfo> {
        // 1. 先读取配置文件
        let game_config = load_game_config()?;
        Ok(InheritInfo {
            blue_count: self.zhongma_blue_count.clone(),
            extra_count: game_config.extra_count.clone()
        })
    }

    pub fn parse_basegame(&self, scenario_friend_chara_id: u32) -> Result<BaseGame> {
        let inherit = self.parse_inherit()?;
        let mut uma = self.parse_uma()?;
        // 检查是否有赛程信息
        if self.turn > 12 && self.race_history.is_empty() {
            warn!("未接收到胜场信息，自选比赛计算可能出错；需要更新小黑板插件");
        }
        let friend = self.parse_friend(scenario_friend_chara_id)?;
        let mut deck = vec![];
        let mut card_type_count = [0; 7];
        for (index, id) in self.card_id.iter().enumerate() {
            let mut card = SupportCard::new(*id)?;
            card.friendship = self.persons[index].friendship;
            uma.race_bonus += card.effect.saihou;
            if card.card_type < 7 {
                card_type_count[card.card_type as usize] += 1;
            }
            deck.push(card);
        }
        // 解析事件
        let mut unresolved_events = vec![];
        if let Some(story) = &self.story {
            log::info!("{}", story.explain());
            if let Ok(event) = EventData::try_from(story) {
                unresolved_events.push(event);
            } else {
                warn!("事件效果解析失败, 无法计算");
            }
        }

        Ok(BaseGame {
            turn: self.turn,
            stage: TurnStage::Train, // 随便列一个
            uma,
            deck,
            inherit: Arc::new(inherit),
            friend,
            train_level_count: self.train_level_count.clone(),
            distribution: self.person_distribution.clone(),
            card_type_count: Arc::new(card_type_count),
            unresolved_events,
            ..Default::default()
        })
    }
}

impl From<&BasePerson> for BasePersonStatus {
    fn from(person: &BasePerson) -> Self {
        Self {
            person_type: 0, // 没使用这个字段
            chara_id: person.chara_id,
            friendship: person.friendship,
            is_hint: person.is_hint
        }
    }
}

/// 反向转回GameStatusBase, persons和friend_没有设置
impl From<&BaseGame> for GameStatusBase {
    fn from(game: &BaseGame) -> Self {
        let failure_rate_bias = if game.uma.flags.good_trainer {
            2
        } else if game.uma.flags.bad_trainer {
            -2
        } else {
            0
        };
        let card_id = game.deck.iter().map(|sc| sc.card_id * 10 + sc.rank).collect();
        let friend_outgoing_used = game.friend.out_used.iter().filter(|x| **x).count() as i32;

        GameStatusBase {
            scenario_id: 0,
            uma_id: game.uma.uma_id,
            uma_star: 5,
            turn: game.turn,
            vital: game.uma.vital,
            max_vital: game.uma.max_vital,
            motivation: game.uma.motivation,
            five_status: game.uma.five_status.clone(),
            five_status_limit: game.uma.five_status_limit.clone(),
            skill_pt: game.uma.skill_pt,
            skill_score: game.uma.skill_score,
            total_hints: game.uma.total_hints,
            train_level_count: game.train_level_count.clone(),
            pt_score_rate: 2.0,
            failure_rate_bias,
            is_ill: game.uma.flags.ill,
            is_qiezhe: game.uma.flags.qiezhe,
            is_aijiao: game.uma.flags.aijiao,
            is_positive_thinking: game.uma.flags.positive_thinking,
            is_refresh_mind: game.uma.flags.refresh_mind > 0,
            is_lucky: game.uma.flags.lucky,
            zhongma_blue_count: game.inherit.blue_count.clone(),
            is_racing: game.uma.is_race_turn(game.turn),
            card_id,
            persons: vec![],
            person_distribution: game.distribution.clone(),
            locked_training_id: -1,
            friendship_noncard_reporter: 0,
            friendship_noncard_yayoi: 0,
            friend_stage: game.friend.out_state.to_int(),
            friend_outgoing_used,
            playing_state: 1,
            race_history: game.uma.list_races(),
            story: None
        }
    }
}

/// 解析后的剧本：用于 main loop 按 scenarioId 分发（避免一个 `Game` trait object
/// 处理两种剧本的复杂度——`G::Action` 是关联类型，trait object 路径受限）
///
/// **Step 6 现状**：温泉路径完整可用；拉面路径走 `GameStatusRamen::into_game`
/// 占位（构造基础 RamenGame，ramen 段增量字段覆写在 Step 7 落实）。
#[derive(Debug)]
pub enum ParsedGame {
    /// `scenarioId == 12`：温泉剧本
    Onsen(umasim::game::onsen::game::OnsenGame),
    /// `scenarioId == 14`：拉面剧本
    Ramen(umasim::game::ramen::RamenGame)
}

/// 从 `thisTurn.json` 内容读 `baseGame.scenarioId`（int）
///
/// 不走 `parse_game::<GameStatusOnsen>`（会消耗一次解析），用 `serde_json::Value`
/// 偷出 scenarioId 即可。
pub fn extract_scenario_id(contents: &str) -> Result<u32> {
    let value: serde_json::Value = serde_json::from_str(contents)?;
    value
        .get("baseGame")
        .and_then(|b| b.get("scenarioId"))
        .and_then(|s| s.as_u64())
        .map(|id| id as u32)
        .ok_or_else(|| anyhow::anyhow!("baseGame.scenarioId 缺失或类型错误"))
}

/// 按 `baseGame.scenarioId` 分发解析（详见 `umaai_air_redirector_integration.md` §3.4）
///
/// - `12` → `GameStatusOnsen` → `OnsenGame`
/// - `14` → `GameStatusRamen` → `RamenGame`（Step 6 占位；Step 7 完整覆写）
/// - 其它 → `Err`
pub fn parse_game_by_scenario(contents: &str) -> Result<ParsedGame> {
    use crate::protocol::urafile::parse_game;
    let scenario_id = extract_scenario_id(contents)?;
    match scenario_id {
        12 => parse_game::<GameStatusOnsen>(contents).map(ParsedGame::Onsen),
        14 => parse_game::<GameStatusRamen>(contents).map(ParsedGame::Ramen),
        other => Err(anyhow::anyhow!("不支持的 scenarioId: {other}（仅支持 12=温泉 / 14=拉面）"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小 thisTurn.json fixture（scenarioId=12 温泉）
    ///
    /// 注：`GameStatusBase` 用 `rename_all = "camelCase"`，但部分字段有
    /// `#[serde(rename = "...")]` 覆盖为 snake_case（friend_stage / playing_state /
    /// friendship_noncard_yayoi / friend_outgoingUsed 等）——fixture 必须保持原样。
    const FIXTURE_ONSEN: &str = r#"{
        "baseGame": {
            "scenarioId": 12,
            "umaId": 102601,
            "umaStar": 5,
            "turn": 0,
            "vital": 100,
            "maxVital": 120,
            "motivation": 4,
            "fiveStatus": [1000, 800, 800, 800, 800],
            "fiveStatusLimit": [1600, 1500, 1400, 1400, 1500],
            "skillPt": 0,
            "skillScore": 0,
            "totalHints": 0,
            "trainLevelCount": [1, 1, 1, 1, 1],
            "ptScoreRate": 2.0,
            "failureRateBias": 0,
            "isIll": false,
            "isQieZhe": false,
            "isAiJiao": false,
            "isPositiveThinking": false,
            "isRefreshMind": false,
            "isLucky": false,
            "zhongMaBlueCount": [0, 0, 0, 0, 0],
            "isRacing": false,
            "cardId": [],
            "persons": [],
            "personDistribution": [[], [], [], [], []],
            "lockedTrainingId": -1,
            "friendship_noncard_yayoi": 0,
            "friendship_noncard_reporter": 0,
            "friend_stage": 0,
            "friend_outgoingUsed": 0,
            "playing_state": 1,
            "raceHistory": [],
            "story": null
        },
        "onsen": {
            "currentOnsen": 0,
            "bathing": { "ticketNum": 5, "buffRemainTurn": 0, "isSuperReady": false },
            "onsenState": [],
            "digRemain": [],
            "digCount": 0,
            "digPower": [1, 1, 1],
            "digLevel": [1, 1, 1],
            "digVitalCost": 5,
            "pendingSelection": false
        }
    }"#;

    /// 最小 thisTurn.json fixture（scenarioId=14 拉面）—— ramen 段空（Step 6 占位）
    const FIXTURE_RAMEN: &str = r#"{
        "baseGame": {
            "scenarioId": 14,
            "umaId": 102601,
            "umaStar": 5,
            "turn": 0,
            "vital": 100,
            "maxVital": 120,
            "motivation": 4,
            "fiveStatus": [1000, 800, 800, 800, 800],
            "fiveStatusLimit": [1600, 1500, 1400, 1400, 1500],
            "skillPt": 0,
            "skillScore": 0,
            "totalHints": 0,
            "trainLevelCount": [1, 1, 1, 1, 1],
            "ptScoreRate": 2.0,
            "failureRateBias": 0,
            "isIll": false,
            "isQieZhe": false,
            "isAiJiao": false,
            "isPositiveThinking": false,
            "isRefreshMind": false,
            "isLucky": false,
            "zhongMaBlueCount": [0, 0, 0, 0, 0],
            "isRacing": false,
            "cardId": [302424, 302894, 303044, 302924, 303024, 303054],
            "persons": [],
            "personDistribution": [[], [], [], [], []],
            "lockedTrainingId": -1,
            "friendship_noncard_yayoi": 0,
            "friendship_noncard_reporter": 0,
            "friend_stage": 0,
            "friend_outgoingUsed": 0,
            "playing_state": 1,
            "raceHistory": [],
            "story": null
        },
        "ramen": {}
    }"#;

    /// extract_scenario_id：正常字段
    #[test]
    fn test_extract_scenario_id_ok() {
        assert_eq!(extract_scenario_id(FIXTURE_ONSEN).unwrap(), 12);
        assert_eq!(extract_scenario_id(FIXTURE_RAMEN).unwrap(), 14);
    }

    /// extract_scenario_id：缺 baseGame
    #[test]
    fn test_extract_scenario_id_missing_basegame() {
        let json = r#"{}"#;
        let r = extract_scenario_id(json);
        println!("缺 baseGame: is_err={}", r.is_err());
        assert!(r.is_err());
    }

    /// extract_scenario_id：缺 scenarioId
    #[test]
    fn test_extract_scenario_id_missing_field() {
        let json = r#"{"baseGame": {}}"#;
        let r = extract_scenario_id(json);
        println!("缺 scenarioId: is_err={}", r.is_err());
        assert!(r.is_err());
    }

    /// extract_scenario_id：scenarioId 是字符串而非数字
    #[test]
    fn test_extract_scenario_id_wrong_type() {
        let json = r#"{"baseGame": {"scenarioId": "12"}}"#;
        let r = extract_scenario_id(json);
        println!("scenarioId 类型错: is_err={}", r.is_err());
        assert!(r.is_err());
    }

    /// parse_game_by_scenario：onsen fixture 路由到 ParsedGame::Onsen
    #[test]
    fn test_parse_game_by_scenario_onsen() {
        // 跑前需要 init_global + cwd 是 workspace 根（gamedata/default_config.toml）
        let workspace_root = umasim::utils::get_workspace_root().unwrap();
        let _ = std::env::set_current_dir(workspace_root);
        let _ = umasim::gamedata::init_global();
        let r = parse_game_by_scenario(FIXTURE_ONSEN);
        println!("onsen fixture: is_ok={}", r.is_ok());
        match r {
            Ok(ParsedGame::Onsen(_)) => {}
            Ok(ParsedGame::Ramen(_)) => panic!("不应路由到 Ramen"),
            Err(e) => panic!("onsen fixture 不应报错: {e}")
        }
    }

    /// parse_game_by_scenario：ramen fixture 路由到 ParsedGame::Ramen（占位 newgame）
    #[test]
    fn test_parse_game_by_scenario_ramen() {
        let workspace_root = umasim::utils::get_workspace_root().unwrap();
        let _ = std::env::set_current_dir(workspace_root);
        let _ = umasim::gamedata::init_global();
        let r = parse_game_by_scenario(FIXTURE_RAMEN);
        println!("ramen fixture: is_ok={}", r.is_ok());
        match r {
            Ok(ParsedGame::Ramen(_)) => {}
            Ok(ParsedGame::Onsen(_)) => panic!("不应路由到 Onsen"),
            Err(e) => panic!("ramen fixture 不应报错: {e}")
        }
    }

    /// parse_game_by_scenario：未知 scenarioId 报错
    #[test]
    fn test_parse_game_by_scenario_unknown_id() {
        let json = r#"{"baseGame": {"scenarioId": 999}}"#;
        let r = parse_game_by_scenario(json);
        println!("scenarioId=999: is_err={}", r.is_err());
        assert!(r.is_err());
    }
}
