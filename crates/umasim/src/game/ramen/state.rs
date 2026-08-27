//! 拉面杯游戏状态定义
//!
//! 包含 RamenGame（游戏主状态）、RamenState（拉面杯专用状态）和 RamenEffect（效果合并）。

use std::ops::{Deref, DerefMut};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{FeelingType, RamenStage, rules::NPC_CHARA_IDS};
use crate::{
    game::{BaseGame, BasePerson, InheritInfo, PersonType, traits::Game},
    gamedata::{GAMECONSTANTS, ramen::RAMENDATA},
    global,
    rng::{EventRng, SplitmixRng, StrategyRng, StreamTag, TurnFixedRng, derive_seed}
};

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
    /// 训练等级剧本加成（RMJ成功时+1，上限5）
    pub train_level_bonus: i32,

    // ========== 超级拉面 ==========
    /// 超级拉面选择（选的是第几个训练限制选项，回合 >= 72 时自动生效）
    pub super_ramen: Option<usize>,

    // ========== 剧本计数器 ==========
    /// 当年吃面次数（每年重置，叠加增量上限 5 次）
    pub eat_count: i32,
    /// 逐年归档的剧本 PT（下标 0/1/2 = 第 1/2/3 年）。
    ///
    /// 在每年 RMJ 结算回合（turn 23/47/71）将 live [`Self::scenario_pt`] **清零之前**写入。
    /// 局末 live 字段恒为 0（turn 72–77 不再吃面），观测出口必须读本数组。
    /// 不叫 peak：当前 PT 年内只增不减只是实现巧合，不是契约。
    #[serde(default)]
    pub yearly_scenario_pt: [i32; 3],
    /// 逐年归档的吃面次数（下标 0/1/2 = 第 1/2/3 年）。
    ///
    /// 写入时机与 [`Self::yearly_scenario_pt`] 相同。
    #[serde(default)]
    pub yearly_eat_count: [i32; 3],
    /// 逐年归档的地区选择（下标 0/1/2 = 第 1/2/3 年；每格三个地区 id）。
    ///
    /// 写入点在 `selected_regions` 赋值处。年份索引必须用
    /// [`Self::region_archive_year_idx`]（按回合硬编码），**不能**用 `current_year()`：
    /// turn 23 时 `current_year()` 仍为 1，但那一刻选的是**第 2 年**地区。
    /// 映射：turn 2 → 0，turn 23 → 1，turn 47 → 2。
    #[serde(default)]
    pub yearly_selected_regions: [[usize; 3]; 3],
    /// 观测用：当前年份下标（0/1/2），供诀窍流转埋点定位年份数组。
    ///
    /// 在 RMJ 归档（[`Self::archive_year_counters`]）时顺带推进（turn 71 后封顶 2），
    /// 初始 0。**纯观测**，不参与任何规则/策略逻辑。
    #[serde(default)]
    pub obs_year: usize,
    /// 观测用：逐年友情训练回合数（下标 0/1/2 = 第 1/2/3 年）。
    ///
    /// 写入点在 `fill_feeling_gauge`（`is_shining` 时累加）。**纯观测**。
    #[serde(default)]
    pub yearly_friend_turns: [i32; 3],
    /// 观测用：逐年诀窍获得数（槽满 [`GAUGE_LIMIT`] 清零 +1 的次数）。
    ///
    /// 写入点在 `add_gauge` 清零分支。**纯观测**。
    #[serde(default)]
    pub yearly_gauge_gain: [i32; 3],
    /// 观测用：逐年诀窍溢出数（库存超 [`FEELING_LIMIT`] 被丢弃）。
    ///
    /// 写入点在 `add_feeling` 丢弃分支。**纯观测**。
    #[serde(default)]
    pub yearly_gauge_overflow: [i32; 3],
    /// 诀窍角标分配（回合 2-71 时每个训练随机分配一个诀窍类型）
    pub train_feeling_type: Option<[FeelingType; 5]>,

    // ========== 缺席记录 ==========
    /// 本回合被判定为「不在」的全部人头下标（支援卡/友人/团队卡/理事长/记者；
    /// NPC 必定出现、永不在列）
    ///
    /// 由 `distribute_all` → `distribute_person` 判定不在时经
    /// [`crate::game::Game::record_absent_person`] 写入（匹配 vs 说「不在卡池」）。
    /// 每回合 `run_distribute` 的 `distribute_all` 调用前清空，不跨回合残留。
    /// 剧本侧按需按 [`PersonType`] 筛选（如只处理支援卡与友人/团队卡）。
    #[serde(default)]
    pub absent_cards: Vec<i32>,

    // ========== 三阶段决策 pending ==========
    /// 当前回合已选定的面（`RamenSelect` 阶段写入，`Train` 阶段消费）
    /// - None: 不吃面
    /// - Some(idx): 选定 `ramen_region_effect[idx]`
    pub pending_ramen: Option<usize>,
    /// 当前回合已选定的隐藏风味用法（`SpecialSelect` 阶段写入，`Train` 阶段消费）
    pub pending_special_targets: [i32; 3],
    /// 是否走"合并决策"路径（Trainer 在 `RamenSelect` 阶段一次性给出 ramen + targets）
    ///
    /// - true：`apply_combined_ramen_decision` 一次性写完两个 pending 字段；
    ///   `Game::next()` 在 RamenSelect 阶段看到此标记直接推 `Train`，跳过 `SpecialSelect`
    /// - false（默认）：走标准三阶段路径（next() 按 `pending_ramen` 决定 SpecialSelect / Train）
    ///
    /// 由 `clear_pending()` 一并清空，确保不跨回合残留。
    pub combined_decision: bool
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
    pub deck_can_split: bool,
    /// 规则层事件 RNG（可选）
    ///
    /// `Game::next()` 中的吃面效果落地（分身分配）与 RMJ 事件使用此 RNG；
    /// 为 `None` 时回退 `StdRng::from_os_rng()`（保持旧行为）。
    /// 用途：固定种子批量模拟时注入 seed rng，保证整局完全可复现
    /// （计划 §2-4 确定性要求；否则规则层随机性破坏基准对比与调参复现）。
    ///
    /// 注意：`Clone` 会复制 RNG 状态——MCTS 搜索复制状态时两个分支将共享后续
    /// 随机序列，属已知问题，搜索接入时需按分支重置（Phase 5+）。
    pub internal_rng: Option<StdRng>,
    /// 本局规则主种子（bench 局号派生，RNG Refactor Plan v2 §4.2）
    ///
    /// `None` 时规则层随机回退旧行为（用调用方传入的 rng）；`Some` 时
    /// 回合固定流 / 策略流按 `(rule_master, turn)` 派生（见 [`Self::reset_turn_streams`]）。
    pub rule_master: Option<u64>,
    /// 回合固定流（人头分布/角标/hint/回合开始事件）
    ///
    /// 与策略完全无关：同一种子、同一回合，任何策略看到的局面逐位相同。
    pub turn_fixed: Option<TurnFixedRng>,
    /// 策略流（训练成败/分身/吃面落地/策略触发事件）
    ///
    /// 仅 apply 真实动作时消耗；同一回合内 counter 从 0 计数。
    pub strategy: Option<StrategyRng>,
    /// 事件流（回合开始事件链：unlock 判定/事件生成/事件应用，v2 §4.3 三流）
    ///
    /// 事件的触发依赖事件历史（策略状态），但随机本身独立成轴——事件历史差异
    /// 只影响事件流自身，不污染局面流与策略流。
    pub event: Option<EventRng>
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

impl RamenState {
    /// 清空三阶段决策的 pending 字段
    ///
    /// 调用时机：
    /// - `Begin` 阶段开始时（清理上一回合残留）
    /// - `Train` 阶段结束后（防御性清理，避免 pending 跨回合保留）
    /// - `NextTurn` 阶段（回合边界清空）
    pub fn clear_pending(&mut self) {
        self.pending_ramen = None;
        self.pending_special_targets = [0, 0, 0];
        self.combined_decision = false;
    }

    /// 年度地区选择对应的归档下标（0-based）。
    ///
    /// **禁止**用 `current_year() - 1` 代替：地区选择发生在当年 RMJ 结算之后、
    /// `advance_turn` 之前，回合号仍停在年界，`current_year()` 还没跨年但选的已是
    /// 下一年的地区，故 turn 23/47 上两者差 1。
    pub fn region_archive_year_idx(turn: i32) -> Result<usize> {
        match turn {
            2 => Ok(0),
            23 => Ok(1),
            47 => Ok(2),
            _ => Err(anyhow!("非年度地区选择回合 turn={turn}，无法归档"))
        }
    }

    /// RMJ 结算回合对应的当年归档下标（0-based）。
    ///
    /// turn 23/47/71 → 0/1/2。与 [`Self::region_archive_year_idx`] 不同：
    /// 同一 turn 23，RMJ 归档的是第 1 年的 PT/吃面，地区归档的是第 2 年的选择。
    pub fn rmj_archive_year_idx(turn: i32) -> Result<usize> {
        match turn {
            23 => Ok(0),
            47 => Ok(1),
            71 => Ok(2),
            _ => Err(anyhow!("非 RMJ 结算回合 turn={turn}，无法归档"))
        }
    }

    /// 将 live `scenario_pt` / `eat_count` 写入对应年份。必须在清零之前调用。
    pub fn archive_year_counters(&mut self, year_idx: usize) -> Result<()> {
        *self
            .yearly_scenario_pt
            .get_mut(year_idx)
            .ok_or_else(|| anyhow!("yearly_scenario_pt 下标 {year_idx} 越界"))? = self.scenario_pt;
        *self
            .yearly_eat_count
            .get_mut(year_idx)
            .ok_or_else(|| anyhow!("yearly_eat_count 下标 {year_idx} 越界"))? = self.eat_count;
        // 观测：推进当前年份（turn 71 结算后封顶 2，覆盖超级拉面回合）
        self.obs_year = (year_idx + 1).min(2);
        Ok(())
    }

    /// 将本次地区选择写入逐年归档。
    ///
    /// `year_idx` 必须来自 [`Self::region_archive_year_idx`]，不能用 `current_year()`。
    pub fn archive_selected_regions(&mut self, year_idx: usize, regions: [usize; 3]) -> Result<()> {
        *self
            .yearly_selected_regions
            .get_mut(year_idx)
            .ok_or_else(|| anyhow!("yearly_selected_regions 下标 {year_idx} 越界"))? = regions;
        Ok(())
    }
}

impl RamenGame {
    /// 创建新的拉面杯游戏实例
    pub fn newgame(uma_id: u32, deck_ids: &[u32; 6], inherit: InheritInfo) -> Result<Self> {
        // 检测卡组是否携带新友人卡(card_id=30305，rank=1-4，即 idrank 303051-303054)
        // 注意：旧实现 `id / 10 == 30305` 会误判 rank=0（303050）和 rank=5-9（303055-303059）
        let has_new_friend = deck_ids.iter().any(|&idrank| {
            let rank = idrank % 10;
            idrank / 10 == 30305 && (1..=4).contains(&rank)
        });
        if !has_new_friend {
            anyhow::bail!("卡组未携带新友人卡(idrank=303051-303054，card_id=30305)，拉面杯模拟器仅支持新友人卡组");
        }
        let mut ret = RamenGame {
            base: BaseGame::new(uma_id, deck_ids, inherit)?,
            stage: RamenStage::Begin,
            persons: vec![],
            ramen: RamenState::default(),
            current_effect: RamenEffect::default(),
            deck_can_split: false,
            internal_rng: None,
            rule_master: None,
            turn_fixed: None,
            strategy: None,
            event: None
        };
        // 合并拉面杯剧本的友人事件 ID（base 已包含 global_events.friend_events 的 ID）
        // 让 apply_event 能正确识别 8303051xx 的友人事件并应用 friend.event_bonus / vital_bonus
        ret.base
            .friend_event_ids
            .extend(global!(RAMENDATA).friend_events.values().map(|e| e.id));
        // 五维属性上限：拉面杯剧本数据覆盖（Phase 2 步骤 1：从 constants.json 隔离到 scenario_ramen）
        // 若 scenario_ramen.json 未提供该字段，回退到全局默认值（防御）
        //
        // 注意：温泉剧本对应字段（onsen/game.rs:135）有 `min(2800)` 防御性 cap，
        // 但拉面剧本 speed 基础上限是 3100，`min(2800)` 会硬截断——把限高速玩家进 3100+
        // 区间到不可达。两剧本基值范围不同，不能共用同一 cap：拉面无防御需要。
        if let Some(limit) = global!(RAMENDATA).five_status_limit_base {
            ret.uma.five_status_limit = limit;
        } else {
            ret.uma.five_status_limit = global!(GAMECONSTANTS).five_status_limit_base;
        }
        // 携带4种卡以上才能分身
        ret.deck_can_split = ret.card_type_count.iter().filter(|x| **x > 0).count() >= 4;
        // 初始化人头（Game trait 方法）
        Game::init_persons(&mut ret)?;
        Ok(ret)
    }

    /// 添加友人卡和NPC（第2回合开始）
    pub fn add_friend_and_npcs(&mut self) -> Result<()> {
        // 添加友人卡（card_type >= 5），并更新 friend.person_index
        let friend_persons: Vec<BasePerson> = self
            .deck
            .iter()
            .filter(|card| card.card_type >= 5)
            .map(|card| BasePerson::try_from(card))
            .collect::<Result<Vec<_>>>()?;
        for p in friend_persons {
            let idx = self.persons.len();
            self.add_person(p);
            self.friend.person_index = idx;
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
                card_id: None
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

    /// 添加羁绊（NPC不增加羁绊）
    pub fn add_friendship(&mut self, person_index: usize, value: i32) {
        if person_index < self.persons.len() && self.persons[person_index].person_type != PersonType::Npc {
            // 人头下标 ≠ 卡组下标，回写卡组前先按 card_id 反查；
            // 无卡人头（理事长 / 记者）只记人头这一份羁绊
            let deck_index = Game::deck_index_of(self, person_index);
            let old_value = self.persons[person_index].friendship;
            let new_value = (self.persons[person_index].friendship + value).min(100);
            self.persons[person_index].friendship = new_value;
            if let Some(deck_index) = deck_index {
                self.deck[deck_index].friendship = new_value;
            }
            if old_value < 100 {
                crate::diag!(
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

    /// 注入规则层事件 RNG（固定种子复现用）
    ///
    /// 调用时机：`run_full_game` 之前。之后 `Game::next()` 中的规则层随机性
    /// （吃面分身分配、RMJ 事件）全部走此 RNG，同一 seed 的整局结果可完全复现。
    /// 注：这是旧机制的注入入口，规则层改造（v2 §7 步骤 3）完成后由
    /// [`Self::set_rule_master`] 取代。
    pub fn set_internal_rng(&mut self, rng: StdRng) {
        self.internal_rng = Some(rng);
    }

    /// 注入本局规则主种子（RNG Refactor Plan v2 §4.2）
    ///
    /// 调用时机：`run_full_game` 之前。之后每回合开始（`run_begin`）会按
    /// `(rule_master, turn)` 重置回合固定流与策略流，使同一 seed 的整局
    /// 规则随机完全可复现，且与策略选择无关。
    pub fn set_rule_master(&mut self, master: u64) {
        self.rule_master = Some(master);
        self.reset_turn_streams();
    }

    /// 按 `(rule_master, turn, tag)` 派生一条分身分配用的局部流
    ///
    /// 与从父流 fork 的做法（[`crate::rng::fork_local_stream`]）相比有两处好处：
    ///
    /// 1. **父流消耗为 0**，分身分配完全不推进策略流；
    /// 2. 结果与「本回合此前消耗了几次策略随机」**无关**。地区分身在吃面落地时执行，
    ///    父流 counter 取决于此前的动作与事件；从父流 fork 会让上游任何位移都改掉选卡。
    ///    对 MCTS 的 CRN 尤其重要：同一回合各候选动作走过不同路径后策略流消耗长度不同，
    ///    按 `(rule_master, turn)` 派生能让各候选抽到**同一份**分身随机性，配对方差削减更彻底。
    ///
    /// 未注入 `rule_master` 时返回 `None`，调用方回退到从父流 fork（保持旧路径可复现性契约）。
    pub(crate) fn clone_stream(&self, tag: u64) -> Option<SplitmixRng> {
        self.rule_master
            .map(|master| SplitmixRng::new(derive_seed(master, &[self.base.turn as u64, tag])))
    }

    /// 按当前 `(rule_master, turn)` 重置两条规则流（counter 归零）
    ///
    /// 回合固定流 master = `derive_seed(rule_master, [turn])`；
    /// 策略流 master = `derive_seed(rule_master, [turn, STRATEGY_TAG])`。
    /// 未注入 rule_master 时清空两条流（规则层回退旧行为）。
    /// 调用时机：`run_begin` 前半段（每次进入 `Begin`）。
    /// `BeginAfterRegionSelect` **不得**再调用，否则事件流被重置、随机语义已错。
    pub fn reset_turn_streams(&mut self) {
        match self.rule_master {
            Some(master) => {
                let turn = self.base.turn as u64;
                self.turn_fixed = Some(TurnFixedRng::new(derive_seed(master, &[turn])));
                self.strategy = Some(StrategyRng::new(derive_seed(master, &[
                    turn,
                    StreamTag::Strategy.tag()
                ])));
                self.event = Some(EventRng::new(derive_seed(master, &[turn, StreamTag::Event.tag()])));
            }
            None => {
                self.turn_fixed = None;
                self.strategy = None;
                self.event = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Checks;

    /// `current_year()` 在 turn 边界上的公式（与 [`RamenGame::current_year`] 保持一致）。
    fn year_at(turn: i32) -> i32 {
        if turn < 24 {
            1
        } else if turn < 48 {
            2
        } else {
            3
        }
    }

    /// 地区归档下标必须按回合硬编码，不能用 `current_year()-1`。
    #[test]
    fn test_region_archive_year_idx_not_current_year() -> Result<()> {
        let mut c = Checks::new();
        let cases = [(2, 0usize), (23, 1), (47, 2)];
        for (turn, want) in cases {
            let got = RamenState::region_archive_year_idx(turn);
            println!(
                "turn={turn}: region_idx={got:?} current_year()-1={}",
                year_at(turn) - 1
            );
            c.check(got.as_ref().ok() == Some(&want), &format!("turn {turn} 地区归档下标 = {want}"));
        }
        let trap_23 = (year_at(23) - 1) as usize;
        let trap_47 = (year_at(47) - 1) as usize;
        c.check(trap_23 == 0, "陷阱前提：turn 23 的 current_year()-1 仍是 0");
        c.check(trap_47 == 1, "陷阱前提：turn 47 的 current_year()-1 仍是 1");
        c.check(
            RamenState::region_archive_year_idx(23).ok() != Some(trap_23),
            "turn 23 地区归档不得等于 current_year()-1"
        );
        c.check(
            RamenState::region_archive_year_idx(47).ok() != Some(trap_47),
            "turn 47 地区归档不得等于 current_year()-1"
        );
        c.check(RamenState::region_archive_year_idx(0).is_err(), "非选面回合应报错");
        c.check(RamenState::rmj_archive_year_idx(23).ok() == Some(0), "turn 23 RMJ 归档第 1 年");
        c.check(RamenState::rmj_archive_year_idx(47).ok() == Some(1), "turn 47 RMJ 归档第 2 年");
        c.check(RamenState::rmj_archive_year_idx(71).ok() == Some(2), "turn 71 RMJ 归档第 3 年");
        c.check(
            RamenState::rmj_archive_year_idx(23).ok() != RamenState::region_archive_year_idx(23).ok(),
            "同一 turn 23：RMJ 归档年 ≠ 地区归档年"
        );
        c.finish()
    }
}
