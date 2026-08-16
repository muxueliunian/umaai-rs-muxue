//! 拉面杯游戏状态定义
//!
//! 包含 RamenGame（游戏主状态）、RamenState（拉面杯专用状态）和 RamenEffect（效果合并）。

use std::ops::{Deref, DerefMut};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{FeelingType, RamenStage};
use super::rules::NPC_CHARA_IDS;
use crate::game::{BaseGame, BasePerson, InheritInfo, PersonType};
use crate::game::traits::Game;

/// 拉面杯专用状态
///
/// 包含诀窍系统、拉面库存、剧本 Pt 和各种计数器。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenState {
    // ========== 诀窍系统 ==========
    /// 三种诀窍（A/B/C）库存数量，上限 10
    pub feeling_stock: [i32; 3],
    /// 三种诀窍（A/B/C）当前槽值，满 7 清零 + 1 诀窍
    pub feeling_slot: [i32; 3],
    /// 诀窍获得顺序队列（维护溢出时的丢弃顺序）
    pub feeling_queue: Vec<FeelingType>,

    // ========== 隐藏风味 ==========
    /// 隐藏风味（special_feeling）库存，上限 4
    pub special_feeling: i32,

    // ========== 地区拉面 ==========
    /// 当年已选择的三种地区拉面（ramen_region_effect 下标）
    pub selected_regions: [usize; 3],
    /// 当前回合使用的拉面（ramen_region_effect 下标，None 表示不吃面）
    pub current_ramen: Option<usize>,

    // ========== 剧本 Pt 和结算 ==========
    /// 剧本 Pt
    pub scenario_pt: i32,
    /// RMJ 结算结果（第几次结算的成功/失败状态）
    pub rmj_results: Vec<bool>,

    // ========== 超级拉面 ==========
    /// 超级拉面选择（选的是第几个训练限制选项，回合 >= 72 时自动生效）
    pub super_ramen: Option<usize>,

    // ========== 剧本计数器 ==========
    /// 当年吃面次数（每年重置，叠加增量上限 5 次）
    pub eat_count: i32,
    /// 诀窍角标分配（回合 2-71 时每个训练随机分配一个诀窍类型）
    pub train_feeling_type: Option<[FeelingType; 5]>
}

/// 拉面效果合并（基础效果 + 地区效果 + 超级拉面效果 + Pt常驻效果）
///
/// 字段对应剧本加成词条，参见 ramen_memo_cn 的"剧本加成"和"训练计算公式"。
/// 训练数值公式：
/// - 属性: lower_value * (100 + xunlian)/100 * (100 + youqing)/100
/// - PT: lower_value * (100 + xunlian)/100 * (100 + youqing)/100 * (100 + pt_bonus)/100
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RamenEffect {
    // ========== 基础效果 ==========
    /// 体力恢复
    pub vital: i32,
    /// 干劲提升
    pub motivation: i32,
    /// 赛后加成（来自超级拉面等）
    pub saihou: i32,

    // ========== 训练加成（百分比） ==========
    /// 训练加成（来自 Pt 效果、基础效果、地区效果，求和）
    pub xunlian: i32,
    /// 友情训练加成（仅友情训练时生效，非友情训练时视为 0）
    pub youqing: i32,
    /// PT 加成（来自地区效果、超级拉面额外效果）
    pub pt_bonus: i32,

    // ========== 上限与修正 ==========
    /// 属性上层数值上限增加（来自基础效果、超级拉面选项）
    pub train_limit: i32,
    /// PT 上层数值上限增加（来自超级拉面额外效果）
    pub pt_limit: i32,
    /// 失败率下降（百分比）
    /// 注意：当前 merge 采用简单求和，实际合并算法可能需要根据来源区分处理，待确认
    pub fail_rate_drop: f32,
    /// 羁绊增加（来自基础效果）
    pub friendship: i32,

    // ========== 特殊效果 ==========
    /// 得意率加成
    pub deyilv: i32,
    /// Hint 出现率加成（百分比，如 +30 表示基础 7.5% * 1.3）
    pub hint: i32,
    /// 分身数量（额外支援卡出现次数）
    pub clone: i32,
    /// hint_special: 支援卡类型>=4 时，除友人/团队卡外所有支援卡出现 Hint
    pub hint_special: bool
}

impl RamenEffect {
    /// 合并两个效果
    pub fn merge(&self, other: &RamenEffect) -> RamenEffect {
        RamenEffect {
            vital: self.vital + other.vital,
            motivation: self.motivation + other.motivation,
            saihou: self.saihou + other.saihou,
            xunlian: self.xunlian + other.xunlian,
            youqing: self.youqing + other.youqing,
            pt_bonus: self.pt_bonus + other.pt_bonus,
            train_limit: self.train_limit + other.train_limit,
            pt_limit: self.pt_limit + other.pt_limit,
            fail_rate_drop: self.fail_rate_drop + other.fail_rate_drop,
            friendship: self.friendship + other.friendship,
            deyilv: self.deyilv + other.deyilv,
            hint: self.hint + other.hint,
            clone: self.clone + other.clone,
            hint_special: self.hint_special || other.hint_special
        }
    }
}

/// 拉面杯游戏主状态
///
/// 包含 BaseGame 通用状态和拉面杯专用状态。
/// 通过 Deref 实现方便地访问 BaseGame 字段，但不直接依赖具体字段布局。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RamenGame {
    /// 基础游戏状态
    pub base: BaseGame,
    /// 回合阶段（覆盖 base.stage）
    pub stage: RamenStage,
    /// 人头列表
    pub persons: Vec<BasePerson>,
    /// 拉面杯专用状态
    pub ramen: RamenState,
    /// 当前生效的拉面效果（每回合重新计算）
    pub current_effect: RamenEffect,
    /// 是否能触发分身
    pub deck_can_split: bool
}

impl Deref for RamenGame {
    type Target = BaseGame;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for RamenGame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl RamenGame {
    /// 创建新的拉面杯游戏实例
    pub fn newgame(uma_id: u32, deck_ids: &[u32; 6], inherit: InheritInfo) -> Result<Self> {
        let mut ret = RamenGame {
            base: BaseGame::new(uma_id, deck_ids, inherit)?,
            stage: RamenStage::Begin,
            persons: vec![],
            ramen: RamenState::default(),
            current_effect: RamenEffect::default(),
            deck_can_split: false
        };
        // 上限规范化
        for i in 0..5 {
            ret.uma.five_status_limit[i] = ret.uma.five_status_limit[i].min(2800);
        }
        // 携带5种卡以上才能分身
        ret.deck_can_split = ret.card_type_count.iter().filter(|x| **x > 0).count() >= 5;
        // 初始化人头（Game trait 方法）
        Game::init_persons(&mut ret)?;
        Ok(ret)
    }

    /// 添加友人卡和NPC（第2回合开始）
    pub fn add_friend_and_npcs(&mut self) -> Result<()> {
        // 添加友人卡（card_type >= 5）
        let friend_persons: Vec<BasePerson> = self
            .deck
            .iter()
            .filter(|card| card.card_type >= 5)
            .map(|card| BasePerson::try_from(card))
            .collect::<Result<Vec<_>>>()?;
        for p in friend_persons {
            self.add_person(p);
        }
        // 添加5个NPC
        for &npc_id in NPC_CHARA_IDS {
            self.add_person(BasePerson {
                person_index: 0,
                person_type: PersonType::Npc,
                train_type: -1,
                chara_id: npc_id,
                friendship: 0,
                is_hint: false,
                card_id: None,
            });
        }
        Ok(())
    }

    /// 添加记者（第12回合开始）
    pub fn add_reporter(&mut self) {
        self.add_person(BasePerson::reporter());
    }

    /// 添加人头
    pub fn add_person(&mut self, mut person: BasePerson) {
        person.person_index = self.persons.len() as i32;
        self.persons.push(person);
    }

    /// 添加羁绊
    pub fn add_friendship(&mut self, person_index: usize, value: i32) {
        if person_index < self.persons.len() {
            let old_value = self.persons[person_index].friendship;
            let new_value = (self.persons[person_index].friendship + value).min(100);
            self.persons[person_index].friendship = new_value;
            if person_index < 6 {
                self.deck[person_index].friendship = new_value;
            }
            if old_value < 100 {
                log::info!(
                    "{} 羁绊+{} (={})",
                    self.persons[person_index].short_name(),
                    value,
                    new_value
                );
            }
        }
    }

    /// 是否为比赛回合
    pub fn is_race_turn(&self) -> bool {
        self.uma.is_race_turn(self.turn)
    }

    /// 获取当前年份（1-3）
    pub fn current_year(&self) -> i32 {
        if self.turn < 24 {
            1
        } else if self.turn < 48 {
            2
        } else {
            3
        }
    }

    /// 是否为超级拉面回合（72-77）
    pub fn is_super_ramen_turn(&self) -> bool {
        self.turn >= 72 && self.turn <= 77
    }

    /// 是否为 RMJ 结算回合
    pub fn is_rmj_turn(&self) -> bool {
        matches!(self.turn, 23 | 47 | 71)
    }
}
