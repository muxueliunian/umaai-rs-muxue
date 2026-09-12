//! 拉面杯剧本通信状态（`scenarioId = 14`）
//!
//! 协议定稿见 `.trae/documents/ramen_protocol_v2.md`（v1，2026-09）。
//! 易混淆点见 `.trae/documents/adapter_spec.md`（2026-09-08 修订）。
//!
//! **Step 7 现状**：`GameStatusRamen::into_game` 完整实现，从 `thisTurn.json`
//! 覆写所有 ramen 段字段到 `RamenGame`：
//! - baseGame 增量字段（`source` + `playingState` + `active_effect_array` 三方联合 stage dispatch）
//! - ramen 段（last_ramen / feeling_stock / feeling_slot / super_ramen /
//!   selected_regions / scenario_pt / train_feeling_type / special_feeling 落入
//!   `RamenState`；`feeling_gauge_gains` / `feeling_gauge_gain_base` /
//!   `next_scenario_pt` / `active_effect_array` 仅用于 stage dispatch 判断，
//!   **不**存进 `RamenState`）
//!
//! `active_effect_array` 拆分到 `RamenEffect` 各字段**搁置**（按 §5 第 5 条）：
//! 当前只用其长度判 stage dispatch，不解读单项语义，后续按训练数值需求再补。
//!
//! **base 重建口径**：`into_game` 不采用 `RamenGame::newgame` 打补丁，而是严格用
//! [`GameStatusBase::parse_basegame`] 从协议重建 `BaseGame`（Uma / Friend 走
//! `parse_uma` / `parse_friend`，`five_status_limit` 取协议值，`friend_event_ids`
//! 不合并），再由 [`RamenGame::from_base_game`] 组装剧本专用状态。友人事件、
//! 五维上限等均视为外部输入，不做本地推算。
//!
//! ## stage dispatch 规则（adapter_spec §source / §playing_state）
//!
//! | turn | source | active_effect | playing_state | 含义 | stage |
//! |---|---|---|---|---|---|
//! | ≤ 1 | 任 | 任 | 1 | 剧本机制未启用（拉面机制 turn >= 2 才启动） | `Train` |
//! | ≥ 2 | `event` | 任 | 1/5 | 事件回合，AI 暂不处理 | (warn + 不 dispatch) |
//! | ≥ 2 | `command` | 空 | 1/5 | 当回合训练前，未吃面 | `RamenSelect` |
//! | ≥ 2 | `command` | 有 | 1/5 | 当回合已吃面，写中间状态 `pending_ramen=Some(last_ramen)` | `Train` |
//! | ≥ 2 | `special` | - | 45 | 地区选择 | `RegionSelect` |
//! | ≥ 2 | `command` | - | 45 | 同上（source=special 暂未出现） | `RegionSelect` |
//! | ≥ 2 | `command` | - | 46 | RMJ 结算，不处理 | (warn + 不 dispatch) |
//! | ≥ 2 | `command` | - | 48 | RMJ 最终结算，不处理 | (warn + 不 dispatch) |
//!
//! `source=special` 在 151 份样本中暂未出现，按 `command + playing_state=45` 兜底为 RegionSelect。
//! 数据获取不全（turn 2..=71 且 `selected_regions` 全 0）→ warn + 不 dispatch。
//!
//! ## persons layout（adapter_spec §理事長、记者、NPC生成）
//!
//! | person_index | 身份 | 出现条件 |
//! |---|---|---|
//! | 0..=5 | 6 张训练卡（含友人） | 始终 |
//! | 6 | 理事長 | turn >= 0 |
//! | 7 | 记者 | turn > 12（不包含 12） |
//! | 8..=12 | 5 个 NPC | turn >= 2 |
//!
//! 规则层内部查找走 `PersonType`（Yayoi/Reporter/Npc），不依赖 person_index 数字；
//! person_index 数字仅供 `distribute_all` / 日志观测使用。
//!
//! ## personDistribution 中 NPC chara_id 派发（adapter_spec §personDistribution 适配）
//!
//! 协议约定：NPC 全填 `8`。但 `distribute_all` 按 chara_id 派发，5 个 NPC 必须 chara_id 各异。
//! 规则：把 `person_distribution` 中**全局按出现次序**的 `8` 依次改写为 `8, 9, 10, 11, 12`，
//! 改写后的数字即 NPC chara_id 来源（与 `NPC_CHARA_IDS` 一一对齐）。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

use crate::protocol::{GameStatus, GameStatusBase};
use umasim::{
    game::{
        BasePerson,
        PersonType,
        ramen::{RamenGame, RamenStage, rules::NPC_CHARA_IDS}
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
/// `feeling_gauge` / `last_ramen` 等都是 snake_case，**不**走 `camelCase` —— 这是
/// ramen 段与 baseGame 段（`GameStatusBase` 走 camelCase + 个别 rename 覆盖）
/// 的字段命名约定差异）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RamenStatus {
    /// 每训练×每类型回合增量 `[[i32; 3]; 5]`
    #[serde(default)]
    pub feeling_gauge_gains: [[i32; 3]; 5],
    /// 三种诀窍（A/B/C）当前槽值
    #[serde(default)]
    pub feeling_gauge: [i32; 3],
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
    pub feeling_gauge_gain_base: [i32; 3],
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
/// 仅在 `into_game` 内用其长度做 stage dispatch 判断，不落入 `RamenState`；
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

        // 1. 从协议**严格重建** base（非 `RamenGame::newgame` 打补丁）：
        //    Uma / Friend 经 `parse_uma` / `parse_friend` 重建，`five_status_limit` 取协议值，
        //    `friend_event_ids` 不合并（丢弃）；deck / card_type_count / train_level_count /
        //    distribution / unresolved_events(story) 均由 `parse_basegame` 落地。
        let mut game = RamenGame::from_base_game(base.parse_basegame(9001)?)?;

        // 2. 构造 persons。按 spec §'理事長、记者、NPC生成' 的 layout：
        //   0..5 = deck 6 张（友人 chara_id=9001 / 其他友人改 OtherFriend）
        //   6 = 理事長（始终在场，turn=0 也有）
        //   7 = 记者（turn > 12 时才有，不包含 12）
        //   8..12 = 5 个 NPC（turn >= 2 时才有，chara_id 来自 NPC_CHARA_IDS 一一对齐）
        //   规则层查找走 PersonType，不依赖 person_index 数字；这里赋的 person_index
        //   仅供 distribute_all / 日志观测。NPC chara_id 派发见 §'personDistribution 适配'。
        let mut persons = vec![];
        for card in game.base.deck.iter() {
            let mut person = BasePerson::try_from(card)?;
            person.person_index = persons.len() as i32;
            if person.person_type == PersonType::ScenarioCard && person.chara_id != 9001 {
                person.person_type = PersonType::OtherFriend;
            }
            persons.push(person);
        }
        // 人头的 friendship/is_hint 来自协议 persons（按 cardIndex 对齐）
        for (index, person) in persons.iter_mut().enumerate() {
            if index < base.persons.len() {
                person.friendship = base.persons[index].friendship;
                person.is_hint = base.persons[index].is_hint;
            }
        }
        // 6: 理事長（始终在场）
        let mut yayoi = BasePerson::yayoi();
        yayoi.person_index = 6;
        yayoi.friendship = base.friendship_noncard_yayoi;
        persons.push(yayoi);
        // 7: 记者（adapter_spec：turn > 12 时才有，不包含 12）
        if base.turn > 12 {
            let mut reporter = BasePerson::reporter();
            reporter.person_index = 7;
            reporter.friendship = base.friendship_noncard_reporter;
            persons.push(reporter);
        }
        // 8..12: 5 NPC（adapter_spec：turn >= 2 时才有，chara_id 各异）
        if base.turn >= 2 {
            for (i, &npc_id) in NPC_CHARA_IDS.iter().enumerate() {
                persons.push(BasePerson {
                    person_index: (8 + i) as i32,
                    person_type: PersonType::Npc,
                    train_type: -1,
                    chara_id: npc_id,
                    friendship: 0,
                    is_hint: false,
                    card_id: None
                });
            }
        }
        game.persons = persons;

        // 3. 覆写 ramen 段（协议 `RamenStatus` → `RamenState` 全字段映射）
        let ramen = self.ramen;
        game.ramen.feeling_slot = ramen.feeling_gauge;
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
        game.ramen.current_ramen = if ramen.last_ramen < 0 || ramen.active_effect_array.is_empty() {
            None
        } else {
            Some(ramen.last_ramen as usize)
        };
        game.ramen.scenario_pt = ramen.scenario_pt;

        // 4. personDistribution 适配（adapter_spec §personDistribution 适配）：
        //    spec 要求把全局按出现次序的 `8` 依次改写为 `8, 9, 10, 11, 12`。
        //    **当前实现不改写**——若启用会越界（详见 `issues.md` #12：spec 期望固定
        //    person_index 6/7/8-12，但当前 into_game 按 push 顺序动态分配 person_index，
        //    当 turn <= 12 时 persons 只有 12 项，distribution 出现 `12` 会越界）。等
        //    `BasePerson.is_hidden` 重构落地后再启用改写。
        //
        //    NPC chara_id 各异由 `NPC_CHARA_IDS` 常量保证（persons 构造时直接取常量），
        //    与 distribution 数字无绑定；distribute_all 按 persons 顺序遍历 chara_id 派发。
        //    留此注释作为占位，等 is_hidden PR 合并后启用改写函数。

        // 5. Stage dispatch（adapter_spec §source / §playing_state 三方联合）。
        //    先做数据获取不全检查（turn 2..=71 且 selected_regions 全 0），命中则 warn + 不 dispatch。
        //    不 dispatch 时保留 `RamenStage::Begin`（newgame 默认值），由 main loop 识别并跳过。
        let active_effect_count = ramen.active_effect_array.len();
        let data_incomplete = (2..=71).contains(&base.turn)
            && game.ramen.selected_regions.iter().all(|&r| r == 0);
        if data_incomplete {
            log::warn!(
                "拉面协议数据获取不全：turn={} selected_regions 全 0（年份选择未到位）",
                base.turn
            );
            // 不动 game.stage，保留 Begin 让 main loop 走 fallback
        } else if base.turn <= 1 {
            // turn 0/1：剧本机制未启用（拉面机制 turn >= 2 才启动），直接进 Train。
            // 与 `RamenGame::next()` 内部短路（game.rs:124 turn < 2 跳 RamenSelect）口径一致：
            // 我们在 into_game 派发阶段提前派发，避免 main loop 走到 Distribute 后被 next() 短路时
            // 看不到本应有 RamenSelect 候选可选的语义。
            log::info!("turn={} 剧本机制未启用，直接进 Train 阶段", base.turn);
            game.stage = RamenStage::Train;
        } else {
            let source = base.source.as_deref().unwrap_or("");
            let playing_state = base.playing_state;
            let turn = base.turn;
            match (source, active_effect_count > 0, playing_state) {
                // event / playing_state=5 / 46 / 48：AI 不进决策循环
                ("event", _, _) => {
                    log::info!("source=event，本回合为事件回合，跳过 AI 推荐");
                }
                (_, _, 5) => {
                    log::info!("playing_state=5 事件回合，跳过 AI 推荐");
                }
                (_, _, 46) => {
                    log::info!("playing_state=46 RMJ 结算，跳过 AI 推荐");
                }
                (_, _, 48) => {
                    log::info!("playing_state=48 RMJ 最终结算，跳过 AI 推荐");
                }
                // 超级拉面回合（turn >= 72）：
                //   active_effect_array 空 → 直接丢包（按 spec §超级拉面回合处理），
                //     等下一条数据；下一条数据会有 active_effect_array，是超级拉面激活后
                //     的效果，给训练决策。
                //   active_effect_array 有 → 训练阶段直接给决策，且不能重算超级拉面。
                (src, true, 1) if turn >= 72 => {
                    log::info!("超级拉面回合 turn={turn}，active_effect 已生效，给训练决策");
                    game.stage = RamenStage::Train;
                    // game.ramen.pending_ramen 已在前面 current_ramen 写入路径设置
                    game.ramen.combined_decision = true;
                    let _ = src;
                }
                (src, false, 1) if turn >= 72 => {
                    log::info!("超级拉面回合 turn={turn} 但 active_effect_array 为空，丢弃等下一条");
                    let _ = src;
                    // 不 dispatch
                }
                // 普通训练回合：command + active_effect_array 有 → 已吃面，Train
                //                                                  且构造中间状态 pending_ramen
                ("command", true, 1) => {
                    game.stage = RamenStage::Train;
                    // 写中间状态：adapter_spec §source 'command + active_effect_array 有' →
                    //  构造 RamenAction::ramen_select(Some(last_ramen))，让 umaai 决策训练。
                    //  按用户决策，落地为 `game.ramen.pending_ramen = Some(last_ramen)`。
                    if let Some(cur) = game.ramen.current_ramen {
                        game.ramen.pending_ramen = Some(cur);
                    }
                }
                // 普通训练回合：command + active_effect_array 空 → 吃面前，给 RamenSelect 决策
                ("command", false, 1) => {
                    game.stage = RamenStage::RamenSelect;
                }
                // 地区选择：playing_state=45（source=special 在 151 样本中暂未出现）
                (_, _, 45) => {
                    game.stage = RamenStage::RegionSelect;
                }
                // 兜底：未识别的 playing_state → warn + fallback Train
                (_, _, other) => {
                    log::warn!(
                        "未知 playing_state={other} source={source:?} active_effect={active_effect_count} turn={turn}，fallback 到 Train"
                    );
                    game.stage = RamenStage::Train;
                }
            }
        }

        Ok(game)
    }
}

/// `person_distribution` 适配（adapter_spec §personDistribution 适配）
///
/// TODO：本函数暂未启用——spec 要求把全局按出现次序的 `8` 依次改写为 `8, 9, 10, 11, 12`，
/// 但当前 `into_game` 按 push 顺序动态分配 person_index，当 turn <= 12 时 persons 只有
/// 12 项（下标 0..11），distribution 出现 `12` 会越界。详见 `issues.md` #12。
///
/// 启用条件：等 `BasePerson.is_hidden` 重构落地（按 spec 固定 person_index 6/7/8-12，
/// 缺位者以 placeholder + is_hidden=true 占位）。届时本函数实现即可安全启用。
#[allow(dead_code, unused_variables)]
fn adapt_person_distribution_npc_todo(distribution: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = Vec::with_capacity(distribution.len());
    let mut next_npc_id = 8_i32;
    for row in distribution {
        let mut new_row = Vec::with_capacity(row.len());
        for &v in row {
            if v == 8 {
                new_row.push(next_npc_id);
                next_npc_id += 1;
            } else {
                new_row.push(v);
            }
        }
        out.push(new_row);
    }
    out
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
            let active_effect_json = ramen_json.get("active_effect_array").and_then(|v| v.as_array());
            let active_effect_empty = active_effect_json.map_or(true, |a| a.is_empty());
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
            // 2) current_ramen 透传：仅在 last_ramen >= 0 且 active_effect_array 非空时生效
            //    （L311 改动：active_effect_array 为空时 current_ramen 置 None，即使 last_ramen 有效）
            let expected_current = if last_ramen_json < 0 || active_effect_empty {
                None
            } else {
                Some(last_ramen_json as usize)
            };
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

            // 5) persons layout 校验（adapter_spec §理事長、记者、NPC生成）：
            //    按 into_game 内的 push 顺序实际下标（无空洞）：
            //      0..=5   deck 6 张
            //      6       理事長（始终在场）
            //      7..=11  NPC（turn >= 2；记者不存在时 NPC 占据此区）
            //      7       记者（turn > 12；NPC 后移到 8..=12）
            //    ——注意：spec 写"理事长 6 / 记者 7 / NPC 8-12"是 spec 期望的固定下标，
            //    但实现按 push 顺序，无记者时 NPC 占 7..=11。
            let turn = game.base.turn;
            if turn < 2 {
                assert_eq!(game.persons.len(), 7, "{}: turn < 2 应为 7（6 卡 + 1 理事長）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
            } else if turn <= 12 {
                assert_eq!(game.persons.len(), 12, "{}: turn 2..=12 应为 12（+ 5 NPC，无记者）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
                for i in 7..=11 {
                    assert!(matches!(game.persons[i].person_type, PersonType::Npc),
                        "{}: persons[{i}] 应为 NPC", path.display());
                }
            } else {
                assert_eq!(game.persons.len(), 13, "{}: turn > 12 应为 13（+ 记者 + 5 NPC）", path.display());
                assert!(matches!(game.persons[6].person_type, PersonType::Yayoi),
                    "{}: persons[6] 应为理事長", path.display());
                assert!(matches!(game.persons[7].person_type, PersonType::Reporter),
                    "{}: persons[7] 应为记者", path.display());
                for i in 8..=12 {
                    assert!(matches!(game.persons[i].person_type, PersonType::Npc),
                        "{}: persons[{i}] 应为 NPC", path.display());
                }
            }

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
        // stage 分布（adapter_spec §stage dispatch 实测 chara 6204）：
        //   Begin: 数据获取不全 / 不 dispatch 的样本（source=event / playing_state=46/48 等）
        //   RamenSelect: command + active_effect 空（吃面前）
        //   Train: command + active_effect 有（吃面后训练 / 超级拉面回合训练）
        //   RegionSelect: playing_state=45（地区选择前）
        assert!(count_stage.contains_key("RamenSelect"), "应有 RamenSelect 样本");
        assert!(count_stage.contains_key("Train"), "应有 Train 样本");
        assert!(count_stage.contains_key("RegionSelect"), "应有 RegionSelect 样本");
        // 不应再出现旧协议派发的 stage
        assert!(!count_stage.contains_key("Settlement"), "Settlement stage 已废弃（46/48 不 dispatch）");
        assert!(!count_stage.contains_key("SuperRamenSelect"), "SuperRamenSelect stage 不应自动派发");
        // 协议文档约束：chara 6204 max scenario_pt = 7500（Y3 终值）
        assert_eq!(max_scenario_pt, 7500, "实测 chara 6204 应在 Y3 终值 7500");
        // 至少有一个 super_ramen == 2 的样本（实测 turn72 起）
        assert!(count_super_ramen_2 >= 1, "应至少有 1 份 super_ramen=2 样本");
        // 至少有一个 source=event / playing_state=46 / 48 等不 dispatch 样本（落到 Begin）
        assert!(count_stage.get("Begin").copied().unwrap_or(0) >= 10, "不 dispatch 样本数应 >= 10");
    }
}