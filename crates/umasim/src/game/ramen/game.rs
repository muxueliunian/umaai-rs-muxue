//! 拉面杯 Game trait 实现
//!
//! 实现回合推进、动作列表、事件处理、训练计算等核心游戏流程。
//!
//! 阶段流转设计：
//! - `RamenStage::next()`：负责回合内普通阶段流转（Begin → Distribute → Train → AfterTrain）
//! - `Game::next()`：负责跨阶段流转（AfterTrain → NextTurn → Begin/特殊阶段）

use anyhow::Result;
use comfy_table::{ColumnConstraint, Table, Width};
use log::{info, warn};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

use super::{RamenGame, RamenStage, RamenAction, Operation, FeelingType};
use super::rules::{self, get_turn_special_feeling};
use super::events::assign_train_feeling_type;
use super::effects::calc_ramen_training_effect;
use super::policy::fixed_super_ramen_selection;
use crate::game::{
    BasePerson, FriendOutState, PersonType,
    traits::{Game, Person, Trainer},
    uma::Uma,
};
use crate::gamedata::{
    ActionValue, EventData, GAMECONSTANTS,
    TriggerType, ramen::RAMENDATA,
};
use crate::global;
use crate::utils::{global_events, system_event, system_event_prob, AttributeArray};

impl Game for RamenGame {
    type Person = BasePerson;
    type Action = RamenAction;

    /// 初始化人头：开局仅加入非友人卡支援卡和理事长
    ///
    /// 友人卡、NPC和记者在后续回合动态添加（见 `run_stage` Begin 阶段）
    fn init_persons(&mut self) -> Result<()> {
        // 非友人卡支援卡（card_type < 5）
        let persons = self
            .deck
            .iter()
            .filter(|card| card.card_type < 5)
            .map(|card| BasePerson::try_from(card))
            .collect::<Result<Vec<_>>>()?;
        for p in persons {
            self.add_person(p);
        }
        // 理事长
        self.add_person(BasePerson::yayoi());
        Ok(())
    }

    fn turn(&self) -> i32 {
        self.base.turn
    }

    fn max_turn(&self) -> i32 {
        77
    }

    /// 阶段推进
    ///
    /// 回合内流转由 `RamenStage::next()` 处理（Begin → Distribute → Train → AfterTrain）。
    /// 本方法负责 AfterTrain → NextTurn 以及 NextTurn 的回合边界逻辑。
    fn next(&mut self) -> bool {
        // 回合内普通阶段：委托给 RamenStage::next()
        if let Some(next_stage) = self.stage.next() {
            self.stage = next_stage;
            return true;
        }

        // AfterTrain → NextTurn（RamenStage::next() 返回 None 时）
        if self.stage == RamenStage::AfterTrain {
            self.stage = RamenStage::NextTurn;
            return true;
        }

        // NextTurn：回合边界逻辑
        if self.stage == RamenStage::NextTurn {
            // 清除当前回合的吃面状态
            self.ramen.current_ramen = None;

            // RMJ 结算回合检查
            if self.is_rmj_turn() {
                let year_idx = (self.current_year() - 1) as usize;
                let result = rules::check_rmj(&mut self.ramen, year_idx);
                if result.is_success() {
                    self.ramen.train_level_bonus += 1;
                }
                info!("RMJ 结算: {:?} (PT={}) 训练等级加成={}", result, self.ramen.scenario_pt, self.ramen.train_level_bonus);
                self.ramen.eat_count = 0;
            }

            // 特殊阶段跳转（地区选择已移至 Begin 阶段）
            if self.base.turn == 71 {
                self.stage = RamenStage::SuperRamenSelect;
                return true;
            }

            // 推进到下一回合
            return self.advance_turn();
        }

        // 特殊阶段（SuperRamenSelect/Settlement）→ 推进到下一回合
        if matches!(
            self.stage,
            RamenStage::SuperRamenSelect | RamenStage::Settlement
        ) {
            return self.advance_turn();
        }

        false
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        match self.stage {
            RamenStage::Begin => self.run_begin(trainer, rng)?,
            RamenStage::Distribute => self.run_distribute(rng)?,
            RamenStage::Train => self.run_train(trainer, rng)?,
            RamenStage::AfterTrain => self.run_after_train(trainer, rng)?,
            RamenStage::NextTurn => {} // 回合推进逻辑在 next() 中处理
            RamenStage::RegionSelect => {} // 已移至 Begin 阶段
            RamenStage::SuperRamenSelect => self.run_super_ramen_select()?,
            RamenStage::Settlement => {} // RMJ 结算在 next() 中处理
        }
        Ok(())
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
        if self.is_race_turn() {
            Ok(vec![RamenAction::no_ramen(Operation::Race)])
        } else {
            let available_ramens = if self.base.turn >= 2 && !self.is_super_ramen_turn() {
                super::action::get_available_ramens(&self.ramen, &self.ramen.selected_regions)
            } else {
                vec![]
            };
            let can_friend_outing = self.friend.out_state == FriendOutState::AfterUnlock
                && self.base.turn < 72
                && !self.friend.out_used.iter().all(|used| *used);
            let is_ill = self.uma.flags.ill;
            Ok(super::action::list_all_actions(
                &available_ramens,
                can_friend_outing,
                is_ill,
            ))
        }
    }

    fn generate_events(&self, rng: &mut StdRng) -> Vec<EventData> {
        let mut events = vec![];
        let no_event_turns = &global!(GAMECONSTANTS).no_event_turns;

        // 剧本事件
        let ramen_data = global!(RAMENDATA);
        let story_events: Vec<EventData> = ramen_data
            .scenario_events
            .iter()
            .filter_map(|e| match &e.trigger {
                TriggerType::Random { .. } => Some(e.clone()),
                TriggerType::Code => None,
                TriggerType::Fixed { turns } => {
                    if turns.contains(&self.base.turn) {
                        Some(e.clone())
                    } else {
                        None
                    }
                }
            })
            .collect();
        if !story_events.is_empty() {
            return story_events;
        }

        if !no_event_turns.contains(&self.base.turn) {
            // 友人出门事件判定
            if self.friend.out_state == FriendOutState::BeforeUnlock {
                let friendship = self.persons[self.friend.person_index as usize].friendship;
                let out_prob = if friendship < 60 {
                    system_event_prob("friend_unlock_low")
                } else {
                    system_event_prob("friend_unlock_high")
                }
                .expect("friend_unlock_* prob key not found");
                if rng.random_bool(out_prob) {
                    let ramen_data = global!(RAMENDATA);
                    events.push(ramen_data.friend_events["out"].clone());
                    return events;
                }
            }
            // 一般随机事件
            let weights =
                WeightedIndex::new(global!(GAMECONSTANTS).get_event_distribution())
                    .expect("event weights");
            match weights.sample(rng) {
                0 => {
                    if let Some(event) = self.base.generate_card_event(
                        rng.random_range(0..6i32),
                        rng,
                    ) {
                        events.push(event);
                    }
                }
                1 => {
                    if let Some(event) =
                        self.base.random_select_event(&global_events().uma_events, rng)
                    {
                        events.push(event);
                    }
                }
                2 => {
                    if self.base.turn >= 12 {
                        events.push(
                            system_event("drop_motivation")
                                .expect("掉心情事件")
                                .clone(),
                        );
                    }
                }
                _ => {}
            }
        }
        events
    }

    fn apply_event(&mut self, event: &EventData, choice: usize, rng: &mut StdRng) -> Result<()> {
        if let Some(result) = self.base.apply_event(event, choice, rng) {
            if let Some(person_index) = &event.person_index && result.value.friendship != 0 {
                self.add_friendship(*person_index as usize, result.value.friendship);
            }
        }
        match event.id {
            4012 | 4013 => {
                let inherit_value = ActionValue {
                    status_pt: self.inherit.inherit(rng),
                    ..Default::default()
                };
                let inherit_limit = self.inherit.inherit_limit(rng);
                self.uma.add_value(&inherit_value);
                self.uma.five_status_limit.add_eq(&inherit_limit);
            }
            5007 => {
                if rng.random_bool(system_event_prob("qiezhe_normal")?) {
                    warn!(">> 获得【切者】");
                    self.uma.flags.qiezhe = true;
                }
            }
            super::events::EVENT_FRIEND_UNLOCK => {
                info!(">> 友人出行已解锁");
                self.friend.out_state = FriendOutState::AfterUnlock;
            }
            _ => {}
        }
        Ok(())
    }

    // ========== Getters ==========

    fn persons(&self) -> &[Self::Person] { &self.persons }
    fn persons_mut(&mut self) -> &mut [Self::Person] { &mut self.persons }
    fn absent_rate_drop(&self) -> i32 { self.base.absent_rate_drop }
    fn distribution(&self) -> &Vec<Vec<i32>> { &self.base.distribution }
    fn distribution_mut(&mut self) -> &mut Vec<Vec<i32>> { &mut self.base.distribution }
    fn uma(&self) -> &Uma { &self.uma }
    fn uma_mut(&mut self) -> &mut Uma { &mut self.uma }
    fn deck(&self) -> &Vec<crate::game::SupportCard> { &self.deck }

    fn deyilv(&mut self, person_index: i32) -> Result<f32> {
        if person_index < 6 {
            let (eff, lock) = self.deck[person_index as usize].calc_training_effect(self, 0)?;
            self.deck[person_index as usize].effect = eff.clone();
            if lock {
                self.deck[person_index as usize].is_locked = true;
            }
            Ok(eff.deyilv)
        } else {
            Ok(0.0)
        }
    }

    fn has_group_buff(&self) -> bool { self.friend.group_buff_turn > 0 }

    fn train_level(&self, train: usize) -> usize {
        if self.is_xiahesu() {
            5
        } else {
            let base = self.base.train_level_count[train] as usize / 4 + 1;
            (base + self.ramen.train_level_bonus as usize).min(5).max(1)
        }
    }

    fn training_basic_value(&self) -> &crate::gamedata::TrainingBasicTable {
        &global!(RAMENDATA).training_basic_value
    }

    fn explain_distribution(&self) -> Result<String> {
        let base_headers = vec!["速", "耐", "力", "根", "智"];
        // 剧本机制已开启 且 非URA回合 时显示诀窍角标
        let show_ramen = self.base.turn >= 2 && !self.is_super_ramen_turn();
        let headers: Vec<String> = base_headers.iter().enumerate().map(|(i, &h)| {
            if show_ramen {
                if let Some(types) = self.ramen.train_feeling_type {
                    format!("{}{:?}", h, types[i])
                } else {
                    h.to_string()
                }
            } else {
                h.to_string()
            }
        }).collect();
        let dist = &self.base.distribution;
        let mut rows = vec![];
        for i in 0..6 {
            let mut row = vec![];
            for train in 0..5 {
                if let Some(id) = dist[train].get(i) {
                    let mut text = self.persons[*id as usize].explain();
                    if self.is_shining_at(*id as usize, train) {
                        text = format!("+{text}+");
                    }
                    row.push(text);
                } else {
                    row.push("".to_string());
                }
            }
            rows.push(row);
        }
        let mut table = Table::new();
        table.set_header(headers.clone()).add_rows(rows).set_width(80);
        for col in table.column_iter_mut() {
            col.set_constraint(ColumnConstraint::Absolute(Width::Percentage(20)));
        }
        let mut lines = vec![table.to_string()];
        for train in 0..5 {
            let buffs = self.calc_training_buff(train)?;
            let fail_rate = self.calc_training_failure_rate(&buffs, train);
            let value = self.calc_training_value(&buffs, train)?;
            let is_shining = self.shining_count(train) > 0;
            let header = &headers[train];

            if !show_ramen {
                // 剧本机制未开启 或 URA回合：只显示基础训练数值和失败率
                if fail_rate > 0.0 {
                    lines.push(format!("{} {} 失败率: {}%", header, value.explain(), fail_rate));
                } else {
                    lines.push(format!("{} {}", header, value.explain()));
                }
            } else {
                // 普通回合：显示训练数值 + 失败率 + 诀窍槽明细
                let ramen_effect = calc_ramen_training_effect(self, train, is_shining);
                let effective_fail = (fail_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
                    .min(100.0).max(0.0);

                // 诀窍槽加成明细
                let support_count = dist[train].iter()
                    .filter(|&&p| p >= 0 && (p as usize) < self.persons.len()
                        && self.persons[p as usize].person_type == crate::game::PersonType::Card)
                    .count();
                let npc_count = dist[train].iter()
                    .filter(|&&p| p >= 0 && (p as usize) < self.persons.len()
                        && self.persons[p as usize].person_type == crate::game::PersonType::Npc)
                    .count();
                let train_feeling_bonus = super::rules::calc_train_feeling_bonus(support_count, npc_count);
                let base_dist = super::rules::calc_gauge_base_distribution(&self.ramen.selected_regions);
                let feeling_type = self.ramen.train_feeling_type.map(|types| types[train]);

                let gauge_a = base_dist[0] + if feeling_type == Some(super::FeelingType::A) { train_feeling_bonus } else { 0 } + if is_shining { 2 } else { 0 };
                let gauge_b = base_dist[1] + if feeling_type == Some(super::FeelingType::B) { train_feeling_bonus } else { 0 } + if is_shining { 2 } else { 0 };
                let gauge_c = base_dist[2] + if feeling_type == Some(super::FeelingType::C) { train_feeling_bonus } else { 0 } + if is_shining { 2 } else { 0 };

                let gauge_detail = format!("诀窍槽 A+{} B+{} C+{}", gauge_a, gauge_b, gauge_c);

                if effective_fail > 0.0 {
                    lines.push(format!("{} {} 失败率: {}% {}", header, value.explain(), effective_fail, gauge_detail));
                } else {
                    lines.push(format!("{} {} {}", header, value.explain(), gauge_detail));
                }
            }
        }
        Ok(lines.join("\n"))
    }

    fn calc_training_value(&self, buffs: &crate::game::CardTrainingEffect, train: usize) -> Result<ActionValue> {
        self.default_calc_training_value(buffs, train)
    }

    fn person_is_available(&self, person_index: usize) -> bool {
        match self.persons[person_index].person_type {
            PersonType::ScenarioCard => self.base.turn >= 2,
            PersonType::Reporter => self.base.turn >= 12,
            _ => true,
        }
    }

    fn distribute_hint(&mut self, rng: &mut StdRng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        let hint_bonus_pct = self.calc_hint_bonus_pct() as f64;
        let hint_probs: Vec<_> = self
            .deck()
            .iter()
            .map(|card| card.card_value().hint_prob_increase)
            .collect();
        for person in self.persons_mut() {
            if person.person_type() == PersonType::Card {
                let card_bonus = (100 + hint_probs[person.person_index() as usize]) as f64 / 100.0;
                let hint_prob = base_hint_rate * card_bonus * (1.0 + hint_bonus_pct / 100.0);
                person.set_hint(rng.random_bool(hint_prob));
            }
        }
        Ok(())
    }
}

// ========== 私有辅助方法 ==========

impl RamenGame {
    /// 推进到下一回合
    fn advance_turn(&mut self) -> bool {
        if self.base.turn < self.max_turn() {
            self.base.turn += 1;
            self.stage = RamenStage::Begin;
            if !self.check_free_race() {
                return false;
            }
            true
        } else {
            false
        }
    }

    /// Begin 阶段：动态人头管理、隐藏风味、事件处理
    fn run_begin<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        println!("-----------------------------------------");
        info!("{}", self.explain()?);
        // 显示拉面杯信息（剧本机制未开启或URA回合时简化显示）
        let ramen_info = self.explain_ramen_info();
        if !ramen_info.is_empty() {
            info!("{}", ramen_info);
        }

        // 动态人头管理
        self.manage_persons_on_turn_start()?;

        // 诀窍值初始化/重置（回合2/24/48），同时处理隐藏风味
        let initialized = matches!(self.base.turn, 2 | 24 | 48);
        if initialized {
            self.init_feeling_stocks();
            // 重新打印回合信息
            let ramen_info = self.explain_ramen_info();
            if !ramen_info.is_empty() {
                info!("{}", ramen_info);
            }
        }

        // 年度地区选择：回合2（第1年）、回合23（第2年）、回合47（第3年）
        match self.base.turn {
            2 => self.run_region_select(trainer, rng, 0)?,
            23 => self.run_region_select(trainer, rng, 1)?,
            47 => self.run_region_select(trainer, rng, 2)?,
            _ => {}
        }

        // 固定回合分配隐藏风味（初始化回合已由 init_feeling_stocks 处理，跳过）
        if !initialized {
            let special = get_turn_special_feeling(self.base.turn);
            if special > 0 {
                self.ramen.special_feeling = (self.ramen.special_feeling + special).min(4);
                info!("隐藏风味 +{} (={})", special, self.ramen.special_feeling);
                // 重新打印回合信息
                let ramen_info = self.explain_ramen_info();
                if !ramen_info.is_empty() {
                    info!("{}", ramen_info);
                }
            }
        }

        // 生成并执行回合前事件
        let events = self.generate_events(rng);
        self.add_mandatory_events();
        // 立即执行强制事件（如友人新年、友人结束）
        let mandatory = std::mem::take(&mut self.base.unresolved_events);
        for event in &mandatory {
            self.run_event(event, trainer, rng)?;
        }
        for event in &events {
            self.run_event(event, trainer, rng)?;
        }

        // 超级拉面回合自动效果
        if self.is_super_ramen_turn() {
            if let Some(sel) = self.ramen.super_ramen {
                let options = rules::get_super_ramen_clone_train_options()?;
                if let Some(_option_trains) = options.get(sel) {
                    info!("超级拉面回合自动生效 (选项 {})", sel + 1);
                }
            }
        }

        Ok(())
    }

    /// Distribute 阶段：分配人头和角标
    fn run_distribute(&mut self, rng: &mut StdRng) -> Result<()> {
        if self.is_race_turn() {
            self.reset_distribution();
        } else {
            let raw_types = assign_train_feeling_type(rng);
            let feelings: [FeelingType; 5] = raw_types.map(|v| {
                FeelingType::try_from(v).unwrap_or(FeelingType::A)
            });
            self.ramen.train_feeling_type = Some(feelings);
            self.distribute_all(rng)?;
            self.distribute_hint(rng)?;
            info!("训练:\n{}", self.explain_distribution()?);
        }
        Ok(())
    }

    /// Train 阶段：选择并执行动作
    fn run_train<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action(&actions[selection], rng)?;
        Ok(())
    }

    /// AfterTrain 阶段：处理后续事件
    fn run_after_train<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let after_events = std::mem::take(&mut self.base.unresolved_events);
        for event in &after_events {
            self.run_event(event, trainer, rng)?;
        }
        Ok(())
    }

    /// 年度地区选择（在 Begin 阶段调用，通过 Trainer 统一接口决策）
    ///
    /// `year_idx`: 0=第1年(地区0-4), 1=第2年(地区5-9), 2=第3年(地区10-19)
    fn run_region_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng, year_idx: usize) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let year = year_idx + 1;
        let combos = super::rules::get_region_combinations(year_idx)?;
        println!("==== 第{}年 地区选择 ({}种组合) ====", year, combos.len());
        for (i, combo) in combos.iter().enumerate() {
            let names: Vec<&str> = combo.iter().filter_map(|&idx| {
                ramen_data.ramen_region_effect.get(idx).map(|r| r.name.as_str())
            }).collect();
            info!("  {}: {}", i + 1, names.join(", "));
        }
        let actions: Vec<RamenAction> = combos.iter().map(|&c| RamenAction::no_ramen(Operation::RegionSelect(c))).collect();
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action(&actions[selection], rng)
    }

    /// SuperRamenSelect 阶段：超级拉面选择
    fn run_super_ramen_select(&mut self) -> Result<()> {
        let _option = fixed_super_ramen_selection()?;
        self.ramen.super_ramen = Some(1); // 选项二（索引 1）
        info!("超级拉面选择: 选项二");
        Ok(())
    }

    /// 初始化/重置诀窍值和隐藏诀窍（回合 2/24/48 开始时）
    ///
    /// 根据携带的友人卡类型决定初始化数量：
    /// - 新友人(30305)：每种诀窍=2，隐藏诀窍+=2
    /// - 旧友人(9001/9008)：每种诀窍=1，隐藏诀窍+=1
    /// - 无友人卡：不初始化
    fn init_feeling_stocks(&mut self) {
        // 查找友人卡
        let friend_card = self.deck.iter().find(|c| c.card_type >= 5);
        let Some(card) = friend_card else { return };

        let is_new_friend = card.card_id == 30305;
        let is_old_friend = matches!(card.data.chara_id, 9001 | 9008);
        let init_val = if is_new_friend { 2 } else if is_old_friend { 1 } else { 0 };
        if init_val == 0 { return; }

        self.ramen.feeling_stock = [init_val; 3];
        self.ramen.special_feeling = (self.ramen.special_feeling + init_val).min(4);
        self.ramen.feeling_queue.clear();
        for _ in 0..init_val {
            for ft in [super::FeelingType::A, super::FeelingType::B, super::FeelingType::C] {
                self.ramen.feeling_queue.push(ft);
            }
        }
        info!(
            ">> 诀窍初始化: 每种={} 隐藏+{} (={})",
            init_val, init_val, self.ramen.special_feeling
        );
    }

    /// 计算剧本 Hint 出现率加成百分比
    ///
    /// 来源：ramen_pt_effect.hint（常驻）+ ramen_success/fail_effect.hint（RMJ后）
    fn calc_hint_bonus_pct(&self) -> i32 {
        let ramen_data = global!(RAMENDATA);
        let year_idx = (self.current_year() - 1) as usize;

        // 1. ramen_pt_effect（常驻）
        let pt_tier = super::effects::find_pt_effect_tier(self.ramen.scenario_pt);
        let mut hint = ramen_data.ramen_pt_effect[pt_tier].hint;

        // 2. ramen_success/fail_effect（RMJ结算后）
        if year_idx >= 1 {
            let prev_idx = year_idx - 1;
            if let Some(&success) = self.ramen.rmj_results.get(prev_idx) {
                let rmj_effect = if success {
                    &ramen_data.ramen_success_effect[prev_idx]
                } else {
                    &ramen_data.ramen_fail_effect[prev_idx]
                };
                hint += rmj_effect.hint;
            }
        }
        hint
    }

    /// 动态人头管理：根据回合数添加友人卡、NPC和记者
    fn manage_persons_on_turn_start(&mut self) -> Result<()> {
        // 第2回合（turn==2）开始：添加友人卡和NPC
        if self.base.turn == 2 && !self.persons.iter().any(|p| p.person_type == PersonType::ScenarioCard) {
            self.add_friend_and_npcs()?;
            info!(">> 第2回合：添加友人卡和NPC，当前人头数 {}", self.persons.len());
        }
        // 第12回合（turn==12）开始：添加记者
        if self.base.turn == 12 && !self.persons.iter().any(|p| p.person_type == PersonType::Reporter) {
            self.add_reporter();
            info!(">> 第12回合：添加记者，当前人头数 {}", self.persons.len());
        }
        Ok(())
    }

    /// 格式化拉面杯剧本信息（用于回合开始时显示）
    ///
    /// 包含：当前拉面地域及效果、当前选择地区、诀窍库存和槽值、剧本PT及加成档位
    /// 剧本机制未开启时（回合 < 2）返回空字符串
    /// URA回合（72-77）不显示地区、诀窍槽、诀窍点
    pub fn explain_ramen_info(&self) -> String {
        // 剧本机制未开启时，不显示拉面杯信息
        if self.base.turn < 2 {
            return String::new();
        }

        let ramen_data = global!(RAMENDATA);
        let is_ura = self.is_super_ramen_turn();

        // 当前拉面
        let ramen_str = if let Some(idx) = self.ramen.current_ramen {
            if let Some(region) = ramen_data.ramen_region_effect.get(idx) {
                let eff = super::effects::calc_ramen_training_effect(self, 0, false);
                let mut parts = vec![];
                if eff.xunlian != 0 { parts.push(format!("训+{}", eff.xunlian)); }
                if eff.fail_rate_drop as i32 != 0 { parts.push(format!("失败率-{}", eff.fail_rate_drop as i32)); }
                if eff.friendship != 0 { parts.push(format!("羁绊+{}", eff.friendship)); }
                if eff.status_limit != 0 { parts.push(format!("上限+{}", eff.status_limit)); }
                if parts.is_empty() { region.name.clone() } else { format!("{}({})", region.name, parts.join(",")) }
            } else {
                "无".to_string()
            }
        } else {
            "无".to_string()
        };

        // URA回合：简化显示
        if is_ura {
            return format!("拉面: {}", ramen_str);
        }

        // 普通回合：完整显示
        // 当前选择地区
        let regions_str: Vec<String> = self.ramen.selected_regions.iter().filter_map(|&idx| {
            ramen_data.ramen_region_effect.get(idx).map(|r| r.name.clone())
        }).collect();

        // 诀窍库存和槽
        let stock = &self.ramen.feeling_stock;
        let slot = &self.ramen.feeling_slot;

        // 剧本PT加成档位
        let pt_tier = super::effects::find_pt_effect_tier(self.ramen.scenario_pt);
        let pt_effect = &ramen_data.ramen_pt_effect[pt_tier];
        let mut pt_parts = vec![];
        if pt_effect.xunlian != 0 { pt_parts.push(format!("训+{}", pt_effect.xunlian)); }
        if pt_effect.deyilv != 0 { pt_parts.push(format!("得意+{}", pt_effect.deyilv)); }
        if pt_effect.hint != 0 { pt_parts.push(format!("hint+{}", pt_effect.hint)); }

        // 基础诀窍槽加成
        let base_dist = super::rules::calc_gauge_base_distribution(&self.ramen.selected_regions);

        format!(
            "拉面: {} | 地区: {} | 诀窍 A{}/{} B{}/{} C{}/{} | 槽基础 {:?} | 隐藏诀窍 {} | PT{} [{}]",
            ramen_str,
            regions_str.join(","),
            stock[0], slot[0],
            stock[1], slot[1],
            stock[2], slot[2],
            base_dist,
            self.ramen.special_feeling,
            self.ramen.scenario_pt,
            if pt_parts.is_empty() { "无加成".to_string() } else { pt_parts.join(",") }
        )
    }

    /// 添加强制事件（友人事件、育成结束事件）
    fn add_mandatory_events(&mut self) {
        let ramen_data = global!(RAMENDATA);
        if self.friend.out_state == FriendOutState::AfterUnlock {
            if self.base.turn == 24 {
                self.base.unresolved_events.push(
                    ramen_data.friend_events["newyear"].clone(),
                );
            } else if self.base.turn == 77 {
                self.base.unresolved_events.push(
                    ramen_data.friend_events["end"].clone(),
                );
            }
        }
        if self.base.turn == 77 {
            self.base.unresolved_events.push(
                system_event("ending").expect("ending event").clone(),
            );
        }
    }
}

// ========== 测试 ==========

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::init_global,
        trainer::RandomTrainer,
        utils::{get_workspace_root, init_logger},
    };
    use rand::SeedableRng;

    #[test]
    fn test_ramen_game_newgame() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        // [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
        let game = RamenGame::newgame(
            101901,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            crate::game::InheritInfo {
                blue_count: [15, 3, 0, 0, 0],
                extra_count: [0, 30, 0, 0, 30, 30],
            },
        )?;
        println!("开局人头数: {}", game.persons.len());
        println!("{}", game.explain()?);

        let card_count = game.persons.iter().filter(|p| p.person_type == PersonType::Card).count();
        let yayoi_count = game.persons.iter().filter(|p| p.person_type == PersonType::Yayoi).count();
        let npc_count = game.persons.iter().filter(|p| p.person_type == PersonType::Npc).count();
        let reporter_count = game.persons.iter().filter(|p| p.person_type == PersonType::Reporter).count();
        let scenario_count = game.persons.iter().filter(|p| p.person_type == PersonType::ScenarioCard).count();

        println!("支援卡: {}, 理事长: {}, NPC: {}, 记者: {}, 友人卡: {}", 
                card_count, yayoi_count, npc_count, reporter_count, scenario_count);

        assert_eq!(yayoi_count, 1, "开局应该有1个理事长");
        assert_eq!(npc_count, 0, "开局不应该有NPC");
        assert_eq!(reporter_count, 0, "开局不应该有记者");
        assert_eq!(scenario_count, 0, "开局不应该有友人卡");

        Ok(())
    }

    #[test]
    fn test_ramen_game_full_loop() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        // [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
        let mut game = RamenGame::newgame(
            102601,
            &[302424, 302894, 303044, 302924, 303024, 303054],
            crate::game::InheritInfo {
                blue_count: [15, 3, 0, 0, 0],
                extra_count: [0, 30, 0, 0, 30, 30],
            },
        )?;

        let trainer = RandomTrainer;
        let mut rng = StdRng::seed_from_u64(42);

        println!("开始完整模拟...");
        game.run_full_game(&trainer, &mut rng)?;

        println!("育成结束!");
        println!("最终回合: {}", game.turn());
        println!("剧本PT: {}", game.ramen.scenario_pt);
        println!("RMJ结果: {:?}", game.ramen.rmj_results);
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!("超级拉面选择: {:?}", game.ramen.super_ramen);
        println!("诀窍库存: A={} B={} C={}", game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]);
        println!("隐藏风味: {}", game.ramen.special_feeling);
        let score = game.uma.calc_score();
        println!("评分: {} {}", global!(GAMECONSTANTS).get_rank_name(score), score);

        Ok(())
    }
}
