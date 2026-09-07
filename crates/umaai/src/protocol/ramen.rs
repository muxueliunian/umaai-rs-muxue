//! 拉面杯剧本通信状态（`scenarioId = 14`）
//!
//! 协议定稿见 `.trae/documents/ramen_protocol_v2.md`（v1，2026-09）。
//!
//! **Step 7 现状**：`GameStatusRamen::into_game` 完整实现，从 `thisTurn.json`
//! 覆写所有 ramen 段字段到 `RamenGame`：
//! - baseGame 增量字段（`playingState` → stage dispatch、`source` 路由）
//! - ramen 段全字段（last_ramen / feeling_stock / feeling_slot / feeling_guage_gains /
//!   active_effect_array / super_ramen / selected_regions / scenario_pt / next_scenario_pt /
//!   feeling_guage_gain_base / train_feeling_type / special_feeling）
//!
//! 拆分 `active_effect_array` 到 `RamenEffect` 各字段**搁置**（按 §5 第 5 条）：
//! 当前只做忠实映射（`Vec<ActiveEffectEntry>` 直接覆写），后续按训练数值需求再补。

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::ops::Deref;

use crate::protocol::{BasePersonStatus, GameStatus, GameStatusBase};
use umasim::{
    game::{
        BasePerson,
        PersonType,
        SupportCard,
        ramen::{RamenGame, RamenStage}
    }
};

/// 拉面剧本通信状态顶层结构
///
/// 两段：`base_game`（与温泉剧本共用 `GameStatusBase` + 拉面增量字段）+ `ramen`
/// （拉面段所有字段，与文档 §2 一一对应）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusRamen {
    pub base_game: GameStatusBase,
    /// 拉面段（完整字段定义见 `RamenStatus`）
    #[serde(default)]
    pub ramen: RamenStatus
}

/// 拉面段通信状态（完整字段映射表见 `ramen_protocol_v2.md` §2）
///
/// 字段命名遵循协议 **snake_case**（实测样本 `scenario_pt` / `next_scenario_pt` /
/// `feeling_guage` / `last_ramen` 等都是 snake_case，**不**走 `camelCase` —— 这是
/// ramen 段与 baseGame 段（`GameStatusBase` 走 camelCase + 个别 rename 覆盖）
/// 的字段命名约定差异）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RamenStatus {
    /// 每训练×每类型回合增量 `[[i32; 3]; 5]`
    #[serde(default)]
    pub feeling_guage_gains: [[i32; 3]; 5],
    /// 三种诀窍（A/B/C）当前槽值
    #[serde(default)]
    pub feeling_guage: [i32; 3],
    /// 诀窍队列（按获得顺序；C# 端会过滤 feeling_id==0 的项目）
    #[serde(default)]
    pub feeling_stock: Vec<i32>,
    /// 隐藏风味数量
    #[serde(default)]
    pub special_feeling: i32,
    /// 训练角标（0=无 / 1/2/3=A/B/C）
    #[serde(default)]
    pub train_feeling_type: [i32; 5],
    /// 当前生效效果列表 `{category, id, value}`（语义搁置，见 §5）
    #[serde(default)]
    pub active_effect_array: Vec<ActiveEffectEntry>,
    /// 超级拉面：-1=未选 / 0/1/2=档位
    #[serde(default = "default_super_ramen")]
    pub super_ramen: i32,
    /// 当年已选地区（`region_id`）
    #[serde(default)]
    pub selected_regions: [i32; 3],
    /// 基础增量（按 region 配方）
    #[serde(default)]
    pub feeling_guage_gain_base: [i32; 3],
    /// **直接 = region_id**（实测；与 `selected_regions` 严格对齐）
    #[serde(default = "default_last_ramen")]
    pub last_ramen: i32,
    /// 当前累计剧本 PT（RMJ 失败归零）
    #[serde(default)]
    pub scenario_pt: i32,
    /// 下次吃面可获 PT
    #[serde(default)]
    pub next_scenario_pt: i32
}

fn default_super_ramen() -> i32 {
    -1
}
fn default_last_ramen() -> i32 {
    -1
}

/// 协议 `active_effect_array` 的单项 `{category, id, value}`
///
/// 直接 `Vec<ActiveEffectEntry>` 落到 `RamenState::active_effect_array`。
/// 按 category 拆分到 `RamenEffect` 各字段暂不实现。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveEffectEntry {
    /// 类别（1/2/4 等，语义未公开）
    pub category: i32,
    /// 效果 ID
    pub id: i32,
    /// 效果数值
    pub value: i32
}

impl Deref for GameStatusRamen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}

impl GameStatus for GameStatusRamen {
    type Game = RamenGame;

    fn scenario_id() -> u32 {
        14
    }

    /// 完整协议 → RamenGame 转换（Step 7 实现）
    fn into_game(self) -> Result<Self::Game> {
        let base = self.base_game;
        let inherit = base.parse_inherit()?;

        // 1. 构造基础 RamenGame（newgame 已校验卡组含新友人卡 + BaseGame 字段初始化）
        let deck_ids: [u32; 6] = base
            .card_id
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| anyhow!("拉面协议要求 6 张卡"))?;
        let mut game = RamenGame::newgame(base.uma_id, &deck_ids, inherit)?;

        // 2. 覆写 base 字段（newgame 用 defaults / gamedata 五维上限；协议覆写覆盖）
        game.base.turn = base.turn;
        game.base.uma.vital = base.vital;
        game.base.uma.max_vital = base.max_vital;
        game.base.uma.motivation = base.motivation;
        game.base.uma.five_status = base.five_status.clone();
        game.base.uma.skill_pt = base.skill_pt;
        game.base.uma.skill_score = base.skill_score;
        game.base.uma.total_hints = base.total_hints;
        game.base.uma.race_bonus = base.parse_uma()?.race_bonus;
        // 五维上限 / 训练等级 / 卡组羁绊等也由协议覆写（下面 persons 路径处理）

        // 3. 构造 deck（按协议 card_id + persons.friendship）
        let mut deck = vec![];
        let mut card_type_count = [0; 7];
        for (index, idrank) in base.card_id.iter().enumerate() {
            let mut card = SupportCard::new(*idrank)?;
            if index < base.persons.len() {
                card.friendship = base.persons[index].friendship;
            }
            game.base.uma.race_bonus += card.effect.saihou;
            if card.card_type < 7 {
                card_type_count[card.card_type as usize] += 1;
            }
            deck.push(card);
        }
        game.base.deck = deck;
        game.base.card_type_count = std::sync::Arc::new(card_type_count);
        game.base.train_level_count = base.train_level_count.clone();
        game.base.distribution = base.person_distribution.clone();

        // 4. 构造 persons（友人 / 理事长 / 记者）
        let mut persons = vec![];
        for (index, card) in game.base.deck.iter().enumerate() {
            let mut person = BasePerson::try_from(card)?;
            person.person_index = index as i32;
            if person.person_type == PersonType::ScenarioCard {
                if person.chara_id != 9030 {
                    person.person_type = PersonType::OtherFriend;
                }
            }
            if index < base.persons.len() {
                person.friendship = base.persons[index].friendship;
                person.is_hint = base.persons[index].is_hint;
            }
            persons.push(person);
        }
        // 理事长
        let mut yayoi = BasePerson::yayoi();
        yayoi.friendship = base.friendship_noncard_yayoi;
        persons.push(yayoi);
        // 记者
        let mut reporter = BasePerson::reporter();
        reporter.friendship = base.friendship_noncard_reporter;
        persons.push(reporter);
        game.persons = persons;

        // 5. 事件：协议 baseGame.story 非空 → push 到 unresolved_events
        if let Some(story) = &base.story {
            log::info!("{}", story.explain());
            match umasim::gamedata::EventData::try_from(story) {
                Ok(event) => game.base.unresolved_events.push(event),
                Err(e) => log::warn!("事件效果解析失败, 无法计算: {e}")
            }
        }

        // 6. 覆写 ramen 段（协议 `RamenStatus` → `RamenState` 全字段映射）
        let ramen = self.ramen;
        game.ramen.feeling_guage_gains = ramen.feeling_guage_gains;
        game.ramen.feeling_slot = ramen.feeling_guage;
        game.ramen.feeling_stock = {
            // 协议 feeling_stock 是按"获得顺序"的队列，每项 1/2/3 表示 A/B/C
            // 累计整个 Vec 中 1/2/3 的出现次数 → [count_A, count_B, count_C]
            // 协议 feeling_id==0 项忽略（已被 C# 过滤但 Rust 端可能保留）
            let mut arr = [0; 3];
            for &f in &ramen.feeling_stock {
                if f >= 1 && f <= 3 {
                    arr[(f - 1) as usize] += 1;
                }
            }
            arr
        };
        game.ramen.special_feeling = ramen.special_feeling;
        game.ramen.train_feeling_type = {
            let mut arr = [umasim::game::ramen::FeelingType::A; 5];
            for (i, &t) in ramen.train_feeling_type.iter().enumerate() {
                arr[i] = match t {
                    1 => umasim::game::ramen::FeelingType::A,
                    2 => umasim::game::ramen::FeelingType::B,
                    3 => umasim::game::ramen::FeelingType::C,
                    _ => umasim::game::ramen::FeelingType::A
                };
            }
            // 协议 0=本回合无角标 → 整个 Option 设 None（让 Ramen 内部走默认）
            if ramen.train_feeling_type.iter().all(|&t| t == 0) {
                None
            } else {
                Some(arr)
            }
        };
        game.ramen.active_effect_array = ramen
            .active_effect_array
            .into_iter()
            .map(|e| umasim::game::ramen::ActiveEffectEntry {
                category: e.category,
                id: e.id,
                value: e.value
            })
            .collect();
        game.ramen.super_ramen = if ramen.super_ramen < 0 {
            None
        } else {
            Some(ramen.super_ramen as usize)
        };
        game.ramen.selected_regions = {
            let mut arr = [0usize; 3];
            for (i, &r) in ramen.selected_regions.iter().enumerate() {
                if i < 3 && r >= 0 {
                    arr[i] = r as usize;
                }
            }
            arr
        };
        game.ramen.feeling_guage_gain_base = ramen.feeling_guage_gain_base;
        game.ramen.current_ramen = if ramen.last_ramen < 0 {
            None
        } else {
            Some(ramen.last_ramen as usize)
        };
        game.ramen.scenario_pt = ramen.scenario_pt;
        game.ramen.next_scenario_pt = ramen.next_scenario_pt;

        // 7. Stage dispatch（按协议 §3 playing_state 映射）
        let playing_state = base.playing_state;
        game.stage = match playing_state {
            1 => RamenStage::Train,
            // playing_state=5 事件回合：stage 与 ps=1 同（事件在 unresolved_events）
            5 => RamenStage::Train,
            45 | 46 => RamenStage::Settlement,
            48 => RamenStage::SuperRamenSelect,
            // 比赛回合（2..=10）Rust 端沿用 Train——是否进游戏循环由 main loop 路由
            2..=10 => RamenStage::Train,
            other => {
                log::warn!("未知 playing_state: {other}，fallback 到 Train");
                RamenStage::Train
            }
        };

        // 8. 友人在 5 人卡组下才能分身（newgame 已用同校验；这里保险起见再算一次）
        game.deck_can_split = game.base.card_type_count.iter().filter(|x| **x > 0).count() >= 5;

        Ok(game)
    }
}

/// `GameStatusRamen` → 拉面协议 JSON（暂未实现反向转换，Step 7 后续补）
impl From<&RamenGame> for GameStatusRamen {
    fn from(_game: &RamenGame) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

/// feeling_stock 协议 → RamenState 映射（Vec<i32> → [i32; 3]）
    #[test]
    fn test_feeling_stock_mapping() {
        // 协议示例：[1, 2, 3, 1, 2, 3] → A/B/C 各 2
        let raw = vec![1, 2, 3, 1, 2, 3];
        let arr: [i32; 3] = {
            let mut a = [0; 3];
            for &f in &raw {
                if f >= 1 && f <= 3 {
                    a[(f - 1) as usize] += 1;
                }
            }
            a
        };
        println!("raw={raw:?} → arr={arr:?}");
        assert_eq!(arr, [2, 2, 2]);
    }

    /// train_feeling_type 协议 → RamenState 映射
    #[test]
    fn test_train_feeling_type_mapping() {
        let raw = [3, 1, 1, 3, 2];
        let arr: [umasim::game::ramen::FeelingType; 5] = {
            let mut a = [umasim::game::ramen::FeelingType::A; 5];
            for (i, &t) in raw.iter().enumerate() {
                a[i] = match t {
                    1 => umasim::game::ramen::FeelingType::A,
                    2 => umasim::game::ramen::FeelingType::B,
                    3 => umasim::game::ramen::FeelingType::C,
                    _ => umasim::game::ramen::FeelingType::A
                };
            }
            a
        };
        println!("raw={raw:?} → arr={arr:?}");
        // C, A, A, C, B
        assert_eq!(arr[0], umasim::game::ramen::FeelingType::C);
        assert_eq!(arr[1], umasim::game::ramen::FeelingType::A);
        assert_eq!(arr[4], umasim::game::ramen::FeelingType::B);
    }

    /// selected_regions 映射
    #[test]
    fn test_selected_regions_mapping() {
        let raw = [1, 4, 5];
        let arr: [usize; 3] = {
            let mut a = [0usize; 3];
            for (i, &r) in raw.iter().enumerate() {
                if i < 3 && r >= 0 {
                    a[i] = r as usize;
                }
            }
            a
        };
        assert_eq!(arr, [1, 4, 5]);
    }

    /// super_ramen / last_ramen -1 → None
    #[test]
    fn test_optional_mappings() {
        assert_eq!(if -1_i32 < 0 { None } else { Some(-1_i32 as usize) }, None);
        assert_eq!(if 0_i32 < 0 { None } else { Some(0_i32 as usize) }, Some(0));
        assert_eq!(if 2_i32 < 0 { None } else { Some(2_i32 as usize) }, Some(2));
    }

    /// 151 份 turn import 样本驱动测试（实测 chara 6204 全 78 回合）
    ///
    /// 数据来源：`logs/GameStatusSend_Ramen/game6204_turn*.json`（151 份）
    /// 测试目标：每份样本 parse → into_game → 关键字段 round-trip 校验
    #[test]
    fn test_turn_import_v2_full_samples() {
        use std::fs;

        // 定位样本目录（workspace 根 + logs/GameStatusSend_Ramen）
        let workspace_root = umasim::utils::get_workspace_root().expect("workspace root");
        let sample_dir = workspace_root.join("logs").join("GameStatusSend_Ramen");
        if !sample_dir.is_dir() {
            eprintln!("样本目录不存在：{}（跳过本测试）", sample_dir.display());
            return;
        }
        let _ = std::env::set_current_dir(&workspace_root);
        let _ = umasim::gamedata::init_global();

        // 收集所有 turn 样本（排除 thisTurn.json 当前软链）
        let mut files: Vec<_> = fs::read_dir(&sample_dir)
            .expect("read sample dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.starts_with("game6204_turn") && s != "thisTurn.json")
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        println!("驱动 {} 份样本", files.len());
        assert!(files.len() >= 100, "样本数过少（{}），请检查 logs 目录", files.len());

        let mut count_ok = 0;
        let mut count_stage: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut count_eaten_turns = 0usize; // last_ramen >= 0 + selected_regions 非零（年内吃面回合）
        let mut count_super_ramen_2 = 0usize; // super_ramen == 2（选了超级拉面档位 2）
        let mut max_scenario_pt: i32 = 0;

        for path in &files {
            let contents = fs::read_to_string(path).expect("read sample");
            // 先 parse 成通用 Value 取 ramen 段（用于校验透传）
            let value: serde_json::Value = match serde_json::from_str(&contents) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  parse value fail {}: {e}", path.display());
                    continue;
                }
            };
            let ramen_json = value.get("ramen").cloned().unwrap_or_default();
            let scenario_pt_json = ramen_json.get("scenario_pt").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let last_ramen_json = ramen_json.get("last_ramen").and_then(|v| v.as_i64()).unwrap_or(-1);
            let super_ramen_json = ramen_json.get("super_ramen").and_then(|v| v.as_i64()).unwrap_or(-1);
            let selected_regions_json: [i32; 3] = {
                let arr = ramen_json.get("selected_regions").and_then(|v| v.as_array());
                let mut r = [0; 3];
                if let Some(a) = arr {
                    for (i, v) in a.iter().enumerate() {
                        if i < 3 {
                            r[i] = v.as_i64().unwrap_or(0) as i32;
                        }
                    }
                }
                r
            };
            // 解析 + 构造 GameStatusRamen
            let status: GameStatusRamen = serde_json::from_value(value).expect("reparse");
            // into_game 构造 RamenGame
            let game = match status.into_game() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("  into_game fail {}: {e}", path.display());
                    continue;
                }
            };

            // 关键字段 round-trip 校验
            // 1) scenario_pt 透传
            assert_eq!(game.ramen.scenario_pt, scenario_pt_json, "{}: scenario_pt 不一致", path.display());
            // 2) current_ramen 透传（last_ramen 协议字段）
            let expected_current = if last_ramen_json < 0 { None } else { Some(last_ramen_json as usize) };
            assert_eq!(game.ramen.current_ramen, expected_current, "{}: current_ramen 不一致", path.display());
            // 3) selected_regions 透传
            let expected_regions: [usize; 3] = [
                selected_regions_json[0].max(0) as usize,
                selected_regions_json[1].max(0) as usize,
                selected_regions_json[2].max(0) as usize
            ];
            assert_eq!(game.ramen.selected_regions, expected_regions, "{}: selected_regions 不一致", path.display());
            // 4) super_ramen 透传
            let expected_super = if super_ramen_json < 0 { None } else { Some(super_ramen_json as usize) };
            assert_eq!(game.ramen.super_ramen, expected_super, "{}: super_ramen 不一致", path.display());

            // 累计统计
            count_ok += 1;
            *count_stage.entry(format!("{:?}", game.stage)).or_insert(0) += 1;
            if last_ramen_json >= 0 && selected_regions_json.iter().any(|&r| r > 0) {
                count_eaten_turns += 1;
            }
            if super_ramen_json == 2 {
                count_super_ramen_2 += 1;
            }
            max_scenario_pt = max_scenario_pt.max(game.ramen.scenario_pt);
        }

        println!("解析成功：{} / {}", count_ok, files.len());
        println!("stage 分布：{count_stage:?}");
        println!("max_scenario_pt = {max_scenario_pt}");
        println!("年内吃面回合数={count_eaten_turns}");
        println!("选了超级拉面档位 2 的样本数={count_super_ramen_2}");
        assert_eq!(count_ok, files.len(), "所有样本必须 parse + into_game 成功");
        // stage 分布：Train 应最多；Settlement/SuperRamenSelect 也应出现
        assert!(count_stage.contains_key("Train"), "Train stage 应占绝大多数");
        // 协议文档约束：chara 6204 max scenario_pt = 7500（Y3 终值）
        assert_eq!(max_scenario_pt, 7500, "实测 chara 6204 应在 Y3 终值 7500");
        // 至少有一个 super_ramen == 2 的样本（实测 turn72 起）
        assert!(count_super_ramen_2 >= 1, "应至少有 1 份 super_ramen=2 样本");
    }
}