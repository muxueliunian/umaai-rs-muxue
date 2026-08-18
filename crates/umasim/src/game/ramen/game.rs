//! 拉面杯 Game trait 实现
//!
//! 实现回合推进、动作列表、事件处理、训练计算等核心游戏流程。
//!
//! 阶段流转设计：
//! - `RamenStage::next()`：负责回合内普通阶段流转（Begin → Distribute → Train → AfterTrain）
//! - `Game::next()`：负责跨阶段流转（AfterTrain → NextTurn → Begin/特殊阶段）

use anyhow::{Result, anyhow};
use comfy_table::{ColumnConstraint, Table, Width};
use log::{info, warn};
use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
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
        // RamenSelect 阶段：
        // - combined_decision=true（合并决策路径，由 apply_combined_ramen_decision 写入）→ 直接推 Train
        // - 否则按 pending_ramen 决定推 SpecialSelect（吃了面）还是 Train（不吃面）
        if self.stage == RamenStage::RamenSelect {
            let next_stage = if self.ramen.combined_decision {
                RamenStage::Train
            } else if self.ramen.pending_ramen.is_some() {
                RamenStage::SpecialSelect
            } else {
                RamenStage::Train
            };
            self.stage = next_stage;
            return true;
        }

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
            // 防御性清空 pending
            self.ramen.clear_pending();

            // RMJ 结算回合检查
            if self.is_rmj_turn() {
                let year_idx = (self.current_year() - 1) as usize;
                let result = rules::check_rmj(&mut self.ramen, year_idx);
                if result.is_success() {
                    self.ramen.train_level_bonus += 1;
                }
                info!("RMJ 结算: {:?} (PT={}) 训练等级加成={}", result, self.ramen.scenario_pt, self.ramen.train_level_bonus);
                self.ramen.eat_count = 0;
                // RMJ 事件立即 apply（在 turn=N 末触发，而非 turn=N+1 末）
                // 原因：push 到 unresolved_events 后会被 AfterTrain 阶段消费，
                // 而 AfterTrain 阶段在 turn=N 的 NextTurn 阶段之后才轮到 turn=N+1，
                // 会延迟一整个回合。
                // RMJ 事件没有 player_select=true，可以直接 apply 而不需 Trainer。
                // 事件 ID：401404(年1) / 401405(年2) / 401406(年3)，按 rmj_results[year_idx] 决定 result=2/1
                if let Some(event) = find_rmj_event(year_idx) {
                    let mut apply_rng = rand::rngs::StdRng::from_os_rng();
                    info!(
                        "+ 事件: #{} {} (回合 {} 末)",
                        event.id,
                        event.name,
                        self.base.turn + 1
                    );
                    if let Err(e) = self.apply_event(&event, 0, &mut apply_rng) {
                        log::warn!("RMJ 事件 #{} apply 失败: {e:?}", event.id);
                    }
                }
                // RMJ 结算后 scenario_pt 归零，下一年重新累计
                // 此时 rmj_results 已写入，下一年的 ramen_success_effect / ramen_fail_effect 已可读取
                let pt_before_reset = self.ramen.scenario_pt;
                self.ramen.scenario_pt = 0;
                info!(
                    "scenario_pt 已归零（结算前 PT={}，下年重新累计）",
                    pt_before_reset
                );
            }

            // 年度地区选择：回合23（第1年结束后）、回合47（第2年结束后）
            // RMJ 结算后选择下一年的地区
            match self.base.turn {
                23 | 47 => {
                    self.stage = RamenStage::RegionSelect;
                    return true;
                }
                _ => {}
            }

            // 特殊阶段跳转：超级拉面选择（回合71）
            if self.base.turn == 71 {
                self.stage = RamenStage::SuperRamenSelect;
                return true;
            }

            // 推进到下一回合
            return self.advance_turn();
        }

        // 特殊阶段（RegionSelect/SuperRamenSelect/Settlement）→ 推进到下一回合
        if matches!(
            self.stage,
            RamenStage::RegionSelect | RamenStage::SuperRamenSelect | RamenStage::Settlement
        ) {
            return self.advance_turn();
        }

        false
    }

    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        match self.stage {
            RamenStage::Begin => self.run_begin(trainer, rng)?,
            RamenStage::Distribute => self.run_distribute(rng)?,
            RamenStage::RamenSelect => self.run_ramen_select(trainer, rng)?,
            RamenStage::SpecialSelect => self.run_special_select(trainer, rng)?,
            RamenStage::Train => self.run_train(trainer, rng)?,
            RamenStage::AfterTrain => self.run_after_train(trainer, rng)?,
            RamenStage::NextTurn => {} // 回合推进逻辑在 next() 中处理
            RamenStage::RegionSelect => {
                // 回合2→第1年(year_idx=0)，回合23→第2年(year_idx=1)，回合47→第3年(year_idx=2)
                let year_idx = match self.base.turn {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => unreachable!("unexpected turn for RegionSelect: {}", self.base.turn),
                };
                self.run_region_select(trainer, rng, year_idx)?;
            }
            RamenStage::SuperRamenSelect => self.run_super_ramen_select()?,
            RamenStage::Settlement => {} // RMJ 结算在 next() 中处理
        }
        Ok(())
    }

    fn list_actions(&self) -> Result<Vec<Self::Action>> {
        // race_turn 短路：仅"比赛"一个动作，跳过 RamenSelect/SpecialSelect
        if self.is_race_turn() {
            return Ok(vec![RamenAction::no_ramen(Operation::Race)]);
        }

        // 公共判定：friend_outing / ill
        let can_friend_outing = self.friend.out_state == FriendOutState::AfterUnlock
            && self.base.turn < 72
            && !self.friend.out_used.iter().all(|used| *used);
        let is_ill = self.uma.flags.ill;

        // 按当前阶段返回候选动作
        match self.stage {
            RamenStage::RamenSelect => {
                // 拉面回合（turn >= 2 且非超级拉面回合）才有面可选；其他时段只显示"不吃"
                if self.base.turn >= 2 && !self.is_super_ramen_turn() {
                    Ok(super::action::list_ramen_select_actions(
                        &self.ramen,
                        &self.ramen.selected_regions,
                    ))
                } else {
                    Ok(vec![RamenAction::ramen_select(None)])
                }
            }
            RamenStage::SpecialSelect => {
                let ramen_idx = self
                    .ramen
                    .pending_ramen
                    .ok_or_else(|| anyhow::anyhow!("SpecialSelect 阶段要求 pending_ramen 已设置"))?;
                super::action::list_special_select_actions(&self.ramen, ramen_idx)
            }
            RamenStage::Train => Ok(super::action::list_train_actions(
                self.ramen.pending_ramen,
                self.ramen.pending_special_targets,
                can_friend_outing,
                is_ill,
                self.is_xiahesu(),
            )),
            // 其他阶段的 list_actions 保留旧行为（虽然外部不会在此阶段调）
            _ => {
                let available_ramens = if self.base.turn >= 2 && !self.is_super_ramen_turn() {
                    super::action::get_available_ramens(&self.ramen, &self.ramen.selected_regions)
                } else {
                    vec![]
                };
                Ok(super::action::list_all_actions(
                    &available_ramens,
                    can_friend_outing,
                    is_ill,
                    self.is_xiahesu(),
                ))
            }
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

        // 全局剧本事件（400000400 马娘登场 / 4009 经典年新年 / 4010 古马年新年 等）
        // 这些事件是 gamesystem 共享的（onsen/basic 也用），拉面杯需要按 Fixed 回合触发
        let global_story_events: Vec<EventData> = global_events()
            .story_events
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
        if !global_story_events.is_empty() {
            return global_story_events;
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
                    // 只从 Card 类型的人物中随机选择
                    let card_indices: Vec<i32> = self.persons.iter()
                        .enumerate()
                        .filter(|(_, p)| p.person_type == PersonType::Card)
                        .map(|(i, _)| i as i32)
                        .collect();
                    if let Some(&person_index) = card_indices.choose(rng) {
                        if let Some(event) = self.base.generate_card_event(
                            person_index,
                            rng,
                        ) {
                            events.push(event);
                        }
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
        // RMJ 事件特殊处理：根据 rmj_results[year_idx] 选择 result=2 或 result=1 的分支
        if let Some(year_idx) = rmj_event_year(event.id) {
            if let Some(choice_group) = event.choices.first() {
                if let Some(target) = select_rmj_choice_by_result(choice_group, self.ramen.rmj_results.get(year_idx).copied()) {
                    info!(
                        "RMJ 事件 #{} 应用 result={} 分支",
                        event.id,
                        target.result
                    );
                    self.base.uma.add_value(&target.value);
                } else {
                    warn!(
                        "RMJ 事件 #{} 无法匹配 result 分支（rmj_results[{}]={:?}），使用默认分支",
                        event.id,
                        year_idx,
                        self.ramen.rmj_results.get(year_idx)
                    );
                }
            }
            // 计数 +1（与 base.apply_event 行为一致）
            self.base.events.entry(event.id).and_modify(|x| *x += 1).or_insert(1);
            return Ok(());
        }

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
                self.uma.flags.refresh_mind = 1;
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
            // 卡得意率 + 剧本得意率总加成（参见 calc_scenario_deyilv）
            let scenario_deyilv = super::effects::calc_scenario_deyilv(self);
            Ok(eff.deyilv + scenario_deyilv as f32)
        } else {
            Ok(0.0)
        }
    }

    fn has_group_buff(&self) -> bool { self.friend.group_buff_turn > 0 }

    /// 重写闪耀判定
    ///
    /// 支援卡（含分身）：只能在本体的得意训练位置闪耀（train_type == train && friendship >= 80）
    /// 友人卡：有 group buff 时闪耀
    fn is_shining_at(&self, person_index: usize, train: usize) -> bool {
        if person_index >= self.persons.len() {
            return false;
        }
        let person = &self.persons[person_index];
        match person.person_type {
            // 支援卡（含分身）：只能在本体的得意训练位置闪耀
            PersonType::Card => {
                person.train_type == train as i32 && person.friendship >= 80
            },
            // 友人卡：有 group buff 时闪耀
            PersonType::ScenarioCard => self.has_group_buff(),
            // NPC、理事长、记者不能闪耀
            _ => false
        }
    }

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
                    if *id < 0 || *id as usize >= self.persons.len() {
                        row.push("".to_string());
                        continue;
                    }
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
            let base_value = self.calc_training_value(&buffs, train)?;
            let is_shining = self.shining_count(train) > 0;
            let header = &headers[train];

            if !show_ramen {
                // 剧本机制未开启 或 URA回合：只显示基础训练数值和失败率
                if fail_rate > 0.0 {
                    lines.push(format!("{} {} 失败率: {}%", header, base_value.explain(), fail_rate));
                } else {
                    lines.push(format!("{} {}", header, base_value.explain()));
                }
            } else {
                // 普通回合：显示训练数值（包含拉面效果）+ 失败率 + 诀窍槽明细
                // calc_training_value 内部已经完成两阶段计算（含拉面 buff），
                // 直接使用 status_pt[train] 和 status_pt[5] 即可
                let ramen_effect = calc_ramen_training_effect(self, train, is_shining);
                let effective_fail = (fail_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
                    .min(100.0).max(0.0);

                let value = ActionValue {
                    status_pt: base_value.status_pt,
                    vital: base_value.vital,
                    motivation: base_value.motivation,
                    ..Default::default()
                };

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
        if train > 5 {
            return Err(anyhow!("训练类型错误"));
        }
        // 两阶段计算：参考 OnsenGame 的实现
        // 1. 下层值：default_calc_training_value 应用卡 buff（友情/训练/干劲/人数/成长率），
        //    然后约束 status_pt 各元素 ≤ 100（剧本规则：下层不超过 100）
        let mut base_value = self.default_calc_training_value(buffs, train)?;
        for i in 0..6 {
            base_value.status_pt[i] = base_value.status_pt[i].min(100);
        }
        // 2. 拉面 buff：累乘到下层值上（不合并到 buffs，避免累乘 vs 加法混淆）
        let is_shining = self.shining_count(train) > 0;
        let ramen_effect = super::effects::calc_ramen_training_effect(self, train, is_shining);
        let xunlian_mult = (100 + ramen_effect.xunlian) as f64 / 100.0;
        let youqing_mult = (100 + ramen_effect.youqing) as f64 / 100.0;
        let pt_bonus_mult = (100 + ramen_effect.pt_bonus) as f64 / 100.0;
        let status_limit = 100 + ramen_effect.status_limit;
        let pt_limit = 100 + ramen_effect.status_limit + ramen_effect.pt_limit;
        // 3. 上层值：拉面 buff 带来的增量
        // - xunlian × youqing 对 status_pt[0..4]（5 个属性训练值，含副属性加成 buff.bonus）都生效
        // - pt_bonus 仅对 status_pt[5]（PT）单独生效
        for i in 0..5 {
            if base_value.status_pt[i] > 0 {
                let upper_raw = (base_value.status_pt[i] as f64 * xunlian_mult * youqing_mult) as i32
                    - base_value.status_pt[i];
                let upper = upper_raw.min(status_limit).max(0);
                base_value.status_pt[i] += upper;
            }
        }
        // PT 部分额外乘 pt_bonus
        let pt_upper_raw = (base_value.status_pt[5] as f64 * xunlian_mult * youqing_mult * pt_bonus_mult) as i32
            - base_value.status_pt[5];
        let pt_upper = pt_upper_raw.min(pt_limit).max(0);
        base_value.status_pt[5] += pt_upper;
        Ok(base_value)
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
    // ========== 合并决策接口（仅 RamenGame，不放 Game trait） ==========

    /// 合并决策候选列表：不吃面 + 每个面 × `list_special_targets_for` 候选 targets
    ///
    /// 是 [`super::action::list_combined_ramen_select_actions`] 在 `RamenGame` 上的便捷转发。
    /// 适用于 MctsTrainer / 在线搜索等需要"选面+选吃法"一次性决策的场景。
    ///
    /// 与 `Game::list_actions` 的区别：
    /// - `Game::list_actions` 按当前 stage 分发（三阶段路径下 RamenSelect 只返回面选择）
    /// - 本方法直接在 RamenSelect 阶段返回 ramen × targets 笛卡尔积
    pub fn list_combined_ramen_select_actions(&self) -> Vec<super::action::RamenAction> {
        super::action::list_combined_ramen_select_actions(&self.ramen, &self.ramen.selected_regions)
    }

    /// 应用合并决策：在 RamenSelect 阶段一次性给出 ramen + targets 决策
    ///
    /// 与标准三阶段路径不同：调用本方法后 `Game::next()` 会直接把 stage 推到 Train，
    /// 跳过 SpecialSelect（靠 `RamenState::combined_decision` 标记位判断）。
    ///
    /// # 参数
    /// - `ramen`：选面决策；`None` 表示不吃面（此时 `targets` 被强制为 `[0,0,0]`）
    /// - `targets`：隐藏风味替换目标；吃面时必须在 `list_special_targets_for` 给出的
    ///   合法 targets 列表中，否则报错
    ///
    /// # 行为
    /// 1. 校验 stage 与 targets 合法性
    /// 2. 写 `pending_ramen` + `pending_special_targets`
    /// 3. 设 `combined_decision = true`
    /// 4. **不直接设 stage**，交给 `Game::next()` 推进（避免后续 next 混乱）
    ///
    /// 必须在 `stage == RamenStage::RamenSelect` 时调用；其他阶段调用返回错误。
    pub fn apply_combined_ramen_decision(
        &mut self,
        ramen: Option<usize>,
        targets: [i32; 3],
    ) -> Result<()> {
        if self.stage != RamenStage::RamenSelect {
            anyhow::bail!(
                "apply_combined_ramen_decision: 仅在 RamenSelect 阶段可调用，当前 stage={:?}",
                self.stage
            );
        }

        // 不吃面强制 targets 全零
        let targets = match ramen {
            None => [0, 0, 0],
            Some(idx) => {
                // 校验 targets 是否合法
                let legal = super::rules::list_special_targets_for(&self.ramen, idx)?;
                if !legal.contains(&targets) {
                    anyhow::bail!(
                        "apply_combined_ramen_decision: targets {:?} 不在面 {} 的合法 targets 列表 {:?}",
                        targets,
                        idx,
                        legal
                    );
                }
                targets
            }
        };

        self.ramen.pending_ramen = ramen;
        self.ramen.pending_special_targets = targets;
        self.ramen.combined_decision = true;
        Ok(())
    }

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
        // 三阶段决策 pending 防御性清空（Train 阶段结束后已清，但再确保一次）
        self.ramen.clear_pending();

        info!("-----------------------------------------");
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

        // 第1年地区选择（回合2开始时）
        if self.base.turn == 2 {
            self.run_region_select(trainer, rng, 0)?;
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

        // 休息心得
        if self.uma.flags.refresh_mind > 0 {
            self.update_refresh_mind(rng);
        }

        // 生成回合前事件（含随机事件和强制事件）
        let mut events = self.generate_events(rng);
        self.add_mandatory_events(&mut events)?;
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
            // 应用 finals_effect.base 的 vital/motivation 恢复效果（每回合）
            // + saihou（赛后加成）一次性应用：仅在进入超级拉面第一回合（turn=72）+saihou，
            // 之后回合保留已生效值，不重复累加
            let ramen_data = global!(RAMENDATA);
            let finals_base = &ramen_data.finals_effect.base;
            let value = ActionValue {
                vital: finals_base.vital,
                motivation: finals_base.motivation,
                ..Default::default()
            };
            self.uma.add_value(&value);
            if self.base.turn == 72 {
                // 进入超级拉面第一回合时一次性加 saihou（之后回合不再累加）
                self.uma.race_bonus += finals_base.saihou;
                info!(
                    "超级拉面自动恢复: 体力+{}, 干劲+{}, 赛后+{}（一次性）",
                    finals_base.vital, finals_base.motivation, finals_base.saihou
                );
            } else {
                info!(
                    "超级拉面自动恢复: 体力+{}, 干劲+{}",
                    finals_base.vital, finals_base.motivation
                );
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

            // 超级拉面分身在 distribute_all 之后分配
            if self.is_super_ramen_turn() {
                super::action::RamenAction::distribute_super_ramen_clones(self, rng)?;
            }

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

    /// RamenSelect 阶段：选择吃哪碗面（含不吃）
    ///
    /// race_turn 时直接执行比赛，跳过 SpecialSelect/Train 阶段；
    /// 否则由 trainer 从候选面（不含/含至少一面）中选一个，apply 写 pending_ramen。
    fn run_ramen_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        // race_turn 短路：直接执行比赛，stage 切到 AfterTrain
        if self.is_race_turn() {
            let actions = self.list_actions()?;
            // actions 此时仅含 [no_ramen(Race)]，但 trainer 不必要再选；
            // 直接应用比赛行为（与旧行为兼容）。
            self.apply_action(&actions[0], rng)?;
            // race_turn 不进入 SpecialSelect/Train，直接跳到 AfterTrain
            self.stage = RamenStage::AfterTrain;
            // 立即处理 AfterTrain 阶段遗留的 unresolved_events（如 race_career）
            // 否则下次 next() 会跳过 run_after_train，直接到 NextTurn
            self.run_after_train(trainer, rng)?;
            self.stage = RamenStage::NextTurn;
            return Ok(());
        }

        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action(&actions[selection], rng)?;
        // apply 已根据 ramen None/Some 自动切到 Train 或 SpecialSelect
        Ok(())
    }

    /// SpecialSelect 阶段：选择隐藏风味用法
    ///
    /// 由 trainer 从 `list_special_targets_for` 候选中选一个，apply 写 pending_special_targets。
    fn run_special_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        let selection = trainer.select_action(self, &actions, rng)?;
        self.apply_action(&actions[selection], rng)?;
        // apply 已切到 Train
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

    /// 年度地区选择（在 NextTurn 阶段 RMJ 结算后调用，通过 Trainer 统一接口决策）
    ///
    /// `year_idx`: 0=第1年(地区0-4), 1=第2年(地区5-9), 2=第3年(地区10-19)
    fn run_region_select<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng, year_idx: usize) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let year = year_idx + 1;
        let combos = super::rules::get_region_combinations(year_idx)?;
        info!("==== 第{}年 地区选择 ({}种组合) ====", year, combos.len());
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
    /// - 无友人卡：每种诀窍=0，隐藏诀窍+=1
    fn init_feeling_stocks(&mut self) {
        // 查找友人卡
        let friend_card = self.deck.iter().find(|c| c.card_type >= 5);
        let init_val = match friend_card {
            Some(card) if card.card_id == 30305 => 2,       // 新友人
            Some(card) if matches!(card.data.chara_id, 9001 | 9008) => 1, // 旧友人
            _ => 0,                                           // 无友人卡
        };

        self.ramen.feeling_stock = [init_val; 3];
        // 无友人卡时仍获得1个隐藏风味
        let special_gain = if init_val > 0 { init_val } else { 1 };
        self.ramen.special_feeling = (self.ramen.special_feeling + special_gain).min(4);
        self.ramen.feeling_queue.clear();
        for _ in 0..init_val {
            for ft in [super::FeelingType::A, super::FeelingType::B, super::FeelingType::C] {
                self.ramen.feeling_queue.push(ft);
            }
        }
        info!(
            ">> 诀窍初始化: 每种={} 隐藏+{} (={})",
            init_val, special_gain, self.ramen.special_feeling
        );
    }

    /// 更新休息心得
    ///
    /// 当 refresh_mind > 0 时，每回合开始时体力+5，并根据概率判定是否结束。
    fn update_refresh_mind(&mut self, rng: &mut StdRng) {
        let t = self.uma.flags.refresh_mind as usize;
        if t > 0 {
            info!("休息心得已持续 {t} 回合 -->");
            self.uma.add_value(&ActionValue { vital: 5, ..Default::default() });
            self.uma.flags.refresh_mind += 1;
            let end_prob = global!(GAMECONSTANTS).group_buff_end_prob[t.min(6)];
            if rng.random_bool(end_prob) {
                info!(">> 休息心得结束");
                self.uma.flags.refresh_mind = 0;
            }
        }
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

    /// 格式化游戏状态（重写 BaseGame::explain，显示带剧本加成的训练等级）
    pub fn explain(&self) -> Result<String> {
        let mut lines = vec![];
        lines.push(format!(
            "回合: {}-{:?} 设施等级: {} 友人: {}",
            self.base.turn + 1,
            self.base.stage,
            crate::explain::Explain::train_level_count_with_bonus(
                &self.base.train_level_count,
                self.ramen.train_level_bonus
            ),
            self.base.friend.explain()
        ));
        lines.push(self.base.uma.explain()?);
        Ok(lines.join("\n"))
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
                // 计算地域效果生效的训练位置的效果
                let train = region.at_trains.first().copied().unwrap_or(0) as usize;
                let eff = super::effects::calc_ramen_training_effect(self, train, false);
                let mut parts = vec![];
                if eff.xunlian != 0 { parts.push(format!("训+{}", eff.xunlian)); }
                if eff.fail_rate_drop as i32 != 0 { parts.push(format!("失败率-{}", eff.fail_rate_drop as i32)); }
                if eff.friendship != 0 { parts.push(format!("羁绊+{}", eff.friendship)); }
                if eff.status_limit != 0 { parts.push(format!("上限+{}", eff.status_limit)); }
                if eff.pt_bonus != 0 { parts.push(format!("PT+{}", eff.pt_bonus)); }
                if parts.is_empty() { region.name.clone() } else { format!("{}({})", region.name, parts.join(",")) }
            } else {
                "无".to_string()
            }
        } else {
            "无".to_string()
        };

        // URA回合：显示超级拉面加成
        if is_ura {
            let eff = super::effects::calc_ramen_training_effect(self, 0, false);
            let mut parts = vec![];
            if eff.xunlian != 0 { parts.push(format!("训+{}", eff.xunlian)); }
            if eff.youqing != 0 { parts.push(format!("友情+{}", eff.youqing)); }
            if eff.deyilv != 0 { parts.push(format!("得意+{}", eff.deyilv)); }
            if eff.fail_rate_drop as i32 != 0 { parts.push(format!("失败率-{}", eff.fail_rate_drop as i32)); }
            if eff.friendship != 0 { parts.push(format!("羁绊+{}", eff.friendship)); }
            if eff.status_limit != 0 { parts.push(format!("上限+{}", eff.status_limit)); }
            if eff.pt_bonus != 0 { parts.push(format!("PT+{}", eff.pt_bonus)); }
            if eff.hint != 0 { parts.push(format!("hint+{}", eff.hint)); }
            if eff.clone_count != 0 { parts.push(format!("分身+{}", eff.clone_count)); }
            
            let mut result = format!("超级拉面回合");
            if !parts.is_empty() {
                result.push_str(&format!(" [{}]", parts.join(",")));
            }
            return result;
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

    /// 添加强制事件（友人新年事件）
    ///
    /// 仅同步处理回合**开始时**发生的事件（push 到 `events`，立即 `run_event`）：
    /// - `turn=24` 友人新年事件（友人解锁后才有）
    ///
    /// 回合**结束时**发生的事件改由本函数内部直接 push 到 `base.unresolved_events`，
    /// 由 AfterTrain 阶段执行：
    /// - `turn=48` 新年抽签 4011（`system_events["ticket"]`，按 prob 加权选 result 分支）
    /// - `turn=77` 友人结束事件 + 育成结束事件 5011 + 401407
    ///
    /// 注：友人结束事件原本 push 到 `events` 在 Begin 阶段立即执行，但用户需求是"育成结束时（77 回合末尾）"
    /// 触发，所以改为 push 到 `unresolved_events` 在 AfterTrain 阶段执行。
    fn add_mandatory_events(&mut self, events: &mut Vec<EventData>) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        if self.friend.out_state == FriendOutState::AfterUnlock {
            if self.base.turn == 24 {
                events.push(ramen_data.friend_events["newyear"].clone());
            } else if self.base.turn == 77 {
                // 77 回合末尾：友人结束事件
                self.base
                    .unresolved_events
                    .push(ramen_data.friend_events["end"].clone());
            }
        }
        // 48 回合结束：新年抽签 4011
        if self.base.turn == 48 {
            self.base
                .unresolved_events
                .push(system_event("ticket")?.clone());
        }
        // 77 回合结束：育成结束事件 5011（ending）和 401407
        if self.base.turn == 77 {
            self.base
                .unresolved_events
                .push(system_event("ending").expect("ending event").clone());
            if let Some(event) = find_scenario_event(401407) {
                self.base.unresolved_events.push(event);
            }
        }
        Ok(())
    }
}

/// 按年份查找对应的 RMJ 事件（401404 / 401405 / 401406）
///
/// 返回事件 clone，供 push 到 `unresolved_events`。
/// 不存在时返回 None（数据缺失或年份越界）。
fn find_rmj_event(year_idx: usize) -> Option<crate::gamedata::EventData> {
    let ramen_data = global!(RAMENDATA);
    let target_id = match year_idx {
        0 => 401404,
        1 => 401405,
        2 => 401406,
        _ => return None,
    };
    ramen_data
        .scenario_events
        .iter()
        .find(|e| e.id == target_id)
        .cloned()
}

/// 按 ID 在 scenario_events 中查找事件
///
/// 用于 push 未在 `add_mandatory_events` 处理的事件（如育成结束事件 401407）。
fn find_scenario_event(target_id: u32) -> Option<crate::gamedata::EventData> {
    let ramen_data = global!(RAMENDATA);
    ramen_data
        .scenario_events
        .iter()
        .find(|e| e.id == target_id)
        .cloned()
}

/// 判断事件 ID 是否为 RMJ 结算事件，若是则返回对应的年份索引（0/1/2）
///
/// 成功/失败分支选择见 `select_rmj_choice_by_result`。
fn rmj_event_year(event_id: u32) -> Option<usize> {
    match event_id {
        401404 => Some(0),
        401405 => Some(1),
        401406 => Some(2),
        _ => None,
    }
}

/// 按 RMJ 结算结果（success=true/false）选择对应 result 分支
///
/// - `choices` 通常是 RMJ 事件的 `choices[0]`（选项组），含 2 个分支：
///   - `result=2`：成功（含大成功）
///   - `result=1`：失败
/// - `is_success`：来自 `rmj_results[year_idx]`，true 表示 result=2 分支，false 表示 result=1 分支
///
/// 选择规则：
/// - 优先按 `result` 字段匹配（成功→2，失败→1）
/// - 若无 `result` 字段匹配，则回退到第 0 个分支（防御性）
fn select_rmj_choice_by_result(
    choices: &[crate::gamedata::EventChoice],
    is_success: Option<bool>,
) -> Option<&crate::gamedata::EventChoice> {
    if choices.is_empty() {
        return None;
    }
    let target_result = match is_success {
        Some(true) => 2,   // 成功 → result=2
        Some(false) => 1,  // 失败 → result=1
        None => return Some(&choices[0]), // 无结算结果时回退到第一个分支
    };
    choices.iter().find(|c| c.result == target_result).or(Some(&choices[0]))
}

// ========== 测试 ==========

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::{init_global, ActionValue, EventChoice},
        trainer::RandomTrainer,
        utils::{get_workspace_root, init_logger, disable_log, enable_log},
    };
    use rand::SeedableRng;
    use crate::game::ramen::events::assign_train_feeling_type;
    use crate::game::PersonType;

    // 测试用公共参数
    // [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30],
    };
    const TEST_UMA_ID: u32 = 102601;

    #[test]
    fn test_ramen_game_newgame() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
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
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;

        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();
        println!("随机种子: {:?}", rng);

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

    /// 静默测试游戏流程
    ///
    /// 关闭日志输出，仅输出育成配置和最终结果
    #[test]
    fn test_ramen_silent_loop() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "error");  // 只输出错误
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();

        println!("=== 静默测试 ===");
        println!("卡组: {:?}", TEST_DECK);
        println!("随机种子: {:?}", rng);

        // 关闭日志运行游戏
        disable_log();
        game.run_full_game(&trainer, &mut rng)?;
        enable_log();

        // 输出最终结果
        println!("\n=== 育成结果 ===");
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

    /// 训练参数分解日志专项测试
    ///
    /// 固定场景：回合 31（第二年，Lv=4），3 张速卡（杏目 id=0、洛林 id=3、里见 id=4）
    /// + 2 个 NPC 都在速训练位置，羁绊全部 100。然后分别在
    /// "不吃面"和"吃面 Some(5) 中京"两种情况下触发速训练，
    /// 输出 `explain_distribution` 和 `calc_train_params` 分解日志，
    /// 排查 issues.md「训练数值不对，尤其是友情加成」。
    #[test]
    fn test_train_param_decomposition() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到回合 31（避开 102601 的生涯比赛回合）
        game.base.turn = 31;
        game.add_friend_and_npcs()?; // persons[0..5]=支援卡，[6]=友人卡，[7..12]=5个NPC
        game.add_reporter(); // persons[12]=记者
        // 所有支援卡羁绊 = 100（确保都能闪耀）
        for i in 0..6 {
            game.persons[i].friendship = 100;
            game.deck[i].friendship = 100;
        }
        // 第二年参数
        game.ramen.scenario_pt = 2000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功
        // 训练次数全部 10，配合 train_level_bonus 让训练等级 = 4
        game.base.train_level_count = [10, 10, 10, 10, 10];
        game.ramen.train_level_bonus = 1;
        // 第 1 年地区选 [0, 6, 7]（札幌/中京/京都），便于 add_reporter 等流程
        game.ramen.selected_regions = [0, 6, 7];

        // 直接构造 distribution：3 张速卡 + 2 个 NPC 都在速训练位置
        game.base.distribution = vec![
            vec![0, 3, 4, 7, 8], // 速：杏目 + 洛林 + 里见 + NPC#1 + NPC#2
            vec![],              // 耐
            vec![],              // 力
            vec![],              // 根
            vec![],              // 智
        ];
        // 训练角标设为 [A, B, C, A, B]（无所谓，主要让 explain_distribution 不报错）
        game.ramen.train_feeling_type = Some([
            FeelingType::A,
            FeelingType::B,
            FeelingType::C,
            FeelingType::A,
            FeelingType::B,
        ]);

        use crate::game::traits::{ActionEnum, Game};
        let mut rng = StdRng::seed_from_u64(42);

        // 跳到 Train 阶段
        game.stage = crate::game::ramen::RamenStage::Train;

        // ============ 场景 A：不吃面、速训练 ============
        game.ramen.current_ramen = None;
        let actions = game.list_actions()?;
        let train_idx = actions
            .iter()
            .position(|a| {
                matches!(
                    a.as_base_action(),
                    Some(crate::game::BaseAction::Train(0))
                )
            })
            .expect("应能找到速训练动作");
        println!(
            "\n===== 场景 A：不吃面、速训练 =====\n{}",
            game.explain_distribution()?
        );
        game.apply_action(&actions[train_idx], &mut rng)?;

        // ============ 场景 B：吃面 Some(5) 中京、速训练 ============
        game.ramen.current_ramen = Some(5); // 中京 at_trains=[0,1,2,3,4], youqing=10
        let train_idx2 = actions
            .iter()
            .position(|a| {
                matches!(
                    a.as_base_action(),
                    Some(crate::game::BaseAction::Train(0))
                )
            })
            .expect("应能找到速训练动作");
        println!(
            "\n===== 场景 B：吃面 Some(5) 中京、速训练 =====\n{}",
            game.explain_distribution()?
        );
        game.apply_action(&actions[train_idx2], &mut rng)?;

        Ok(())
    }

    #[test]
    fn test_random_event_generation() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        // 创建游戏实例
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;

        // 使用随机种子
        let mut rng = StdRng::from_os_rng();
        println!("随机种子: {:?}", rng);

        // 模拟一整年（24回合）的事件生成
        println!("\n========== 模拟一整年（24回合）的事件生成 ==========");
        let mut total_events = 0;
        let mut event_counts = std::collections::HashMap::new();

        for turn in 1..=24 {
            game.base.turn = turn;
            let events = game.generate_events(&mut rng);

            println!("\n回合 {}: 生成 {} 个事件", turn, events.len());
            for (i, event) in events.iter().enumerate() {
                println!("  事件 {}: ID={}, 名称={}", i + 1, event.id, event.name);
                total_events += 1;
                *event_counts.entry(event.name.clone()).or_insert(0) += 1;
                // 更新事件计数（模拟 apply_event 的计数逻辑）
                *game.base.events.entry(event.id).or_insert(0) += 1;
            }

            if events.is_empty() {
                println!("  无事件触发");
            }
        }

        // 输出统计信息
        println!("\n========== 事件统计 ==========");
        println!("总事件数: {}", total_events);
        println!("平均每次回合事件数: {:.2}", total_events as f64 / 24.0);

        println!("\n事件类型统计:");
        let mut sorted_events: Vec<_> = event_counts.iter().collect();
        sorted_events.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in sorted_events {
            println!("  {}: {} 次", name, count);
        }

        // 验证事件生成逻辑
        println!("\n========== 事件分布验证 ==========");
        let event_dist = global!(GAMECONSTANTS).get_event_distribution();
        println!("事件分布配置: {:?}", event_dist);
        println!("说明: [支援卡事件, 马娘事件, 掉心情事件, 无事件]");

        Ok(())
    }

    /// 端到端训练数值测试：固定回合 30（第二年），分别打印不吃面 / 吃面 Some(5) 的训练信息
    ///
    /// 固定场景：
    /// - 回合 30（第二年），友人和全部 NPC 已解锁，记者已加入
    /// - feeling_stocks = [3, 3, 3]，地区选择 [5, 6, 7]，scenario_pt = 3000
    /// - rmj_results = [true]（第 1 年 RMJ 成功），所有支援卡羁绊设为 100
    /// - 随机产生 1 次训练分配，分别以 `current_ramen = None` 和 `current_ramen = Some(5)`
    ///   复用同一份分配，调用 `explain_distribution` 输出训练信息
    ///
    /// 主要观测点：
    /// 1. `is_shining_at` 判定（闪耀标记）是否符合"得意位置 + 羁绊 ≥ 80"
    /// 2. 不吃面 vs 吃面：吃面是否引入了 `basic_effect`（羁绊/失败率等）和
    ///    命中 `at_trains` 的 `region_effect`（xunlian/youqing/pt_bonus）
    /// 3. 拉面杯加成的累乘结果是否与公式一致
    #[test]
    fn test_random_distribution_training_value() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        // 1. 创建游戏并直接跳到回合 30（第二年）
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 30;
        // 2. 解锁友人和全部 NPC（person_is_available 要求 turn >= 2）
        game.add_friend_and_npcs()?;
        // 3. 加入记者（person_is_available 要求 turn >= 12）
        game.add_reporter();
        // 4. feeling_stocks = [3, 3, 3]
        game.ramen.feeling_stock = [3, 3, 3];
        // 5. 地区选择 [5, 6, 7]
        game.ramen.selected_regions = [5, 6, 7];
        // 6. scenario_pt = 3000
        game.ramen.scenario_pt = 3000;
        // 7. rmj_results = [true]（第 1 年 RMJ 成功 → 第 2 年常驻 ramen_success_effect[0]）
        game.ramen.rmj_results = vec![true];
        // 直接跳到回合 30 跳过了 RMJ 结算的 train_level_bonus += 1，
        // 这里手动 +1，使 Lv = 10/4 + 1 + 1 = 4
        game.ramen.train_level_bonus = 1;
        // 每个训练的点击次数设为 10，配合 bonus=1 使实际训练等级 = 4
        game.base.train_level_count = [10, 10, 10, 10, 10];
        // 8. 所有支援卡羁绊设为 100（顺手同步 persons / deck 两处）
        for i in 0..6 {
            game.persons[i].friendship = 100;
            game.deck[i].friendship = 100;
        }
        for p in game.persons.iter_mut() {
            if p.person_type == PersonType::Card {
                p.friendship = 100;
            }
        }

        let mut rng = StdRng::from_os_rng();
        println!("\n========== 端到端训练数值测试 ==========");
        println!("随机种子: {:?}", rng);

        // ========== 详细回合信息 ==========
        println!("\n----- 回合信息 -----");
        println!("回合: {} (第{}年)", game.base.turn, game.current_year());
        println!("地区选择: {:?}", game.ramen.selected_regions);
        println!(
            "地区词条: {}",
            game.ramen
                .selected_regions
                .iter()
                .map(|&i| {
                    let ramen_data = global!(RAMENDATA);
                    ramen_data.ramen_region_effect[i].name.clone()
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("剧本 PT: {}", game.ramen.scenario_pt);
        println!("RMJ 结果: {:?}", game.ramen.rmj_results);
        println!("训练等级加成: {}", game.ramen.train_level_bonus);
        println!("训练点击次数: {:?}", game.base.train_level_count);
        println!(
            "feeling_stocks: A={} B={} C={}",
            game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
        );
        println!("隐藏风味: {}", game.ramen.special_feeling);
        println!("人头总数: {}", game.persons.len());

        // ========== 支援卡羁绊概览 ==========
        println!("\n----- 支援卡羁绊 -----");
        for i in 0..6 {
            let p = &game.persons[i];
            println!(
                "  [#{}] {} 类型={} 羁绊={}",
                i,
                p.short_name(),
                p.train_type,
                p.friendship
            );
        }

        // ========== 随机分配 1 次（两个场景共用同一份分配） ==========
        let raw_types = assign_train_feeling_type(&mut rng);
        let feelings: [FeelingType; 5] = raw_types.map(|v| {
            FeelingType::try_from(v).unwrap_or(FeelingType::A)
        });
        game.ramen.train_feeling_type = Some(feelings);
        game.distribute_all(&mut rng)?;
        game.distribute_hint(&mut rng)?;

        // ========== 场景1：不吃面 ==========
        game.ramen.current_ramen = None;
        println!("\n========== 场景1：current_ramen = None（不吃面）==========");
        println!(
            "训练等级: 速={} 耐={} 力={} 根={} 智={}",
            game.train_level(0),
            game.train_level(1),
            game.train_level(2),
            game.train_level(3),
            game.train_level(4)
        );
        println!("\n{}", game.explain_distribution()?);

        // ========== 场景2：吃面 Some(5) ==========
        game.ramen.current_ramen = Some(5);
        let ramen_data = global!(RAMENDATA);
        let region = &ramen_data.ramen_region_effect[5];
        println!(
            "\n========== 场景2：current_ramen = Some(5) ==========\n        地区 {} xunlian={} youqing={} pt_bonus={} hint_count={} at_trains={:?}",
            region.name, region.xunlian, region.youqing, region.pt_bonus, region.hint_count, region.at_trains
        );
        println!(
            "训练等级: 速={} 耐={} 力={} 根={} 智={}",
            game.train_level(0),
            game.train_level(1),
            game.train_level(2),
            game.train_level(3),
            game.train_level(4)
        );
        println!("\n{}", game.explain_distribution()?);

        Ok(())
    }

    /// 验证 RamenGame::deyilv 返回"卡 deyilv + 剧本 deyilv 总加成"
    ///
    /// 关键点：
    /// - 普通回合：剧本 deyilv = pt_effect(当前档) + rmj_results[year-1] success/fail
    /// - 超级拉面：剧本 deyilv = pt_effect(最后一档) + rmj_results[2] success/fail
    /// - 调用方拿到这个值后，会作为 distribute_person 的训练位置权重加成
    #[test]
    fn test_ramen_deyilv_includes_scenario_bonus() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        // ========== 普通回合（year 2, PT=1000, RMJ 成功） ==========
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 30; // year 2
        game.add_friend_and_npcs()?; // person[0..5] 是支援卡
        game.ramen.scenario_pt = 1000;
        game.ramen.rmj_results = vec![true]; // year 1 RMJ 成功

        // 卡 deyilv 来自 calc_training_effect，剧本 deyilv = pt(1000档=63) + rmj_success[0]=80 = 143
        let person_idx = 0;
        let card_deyilv_only = game.deck[person_idx].effect.deyilv;
        let actual_deyilv = game.deyilv(person_idx as i32)?;
        println!(
            "year2, PT=1000, RMJ成功: card_deyilv_only={} 实际 deyilv={}",
            card_deyilv_only, actual_deyilv
        );
        // 期望：actual_deyilv = card_deyilv_only + 143
        assert_eq!(actual_deyilv, card_deyilv_only + 143.0);

        // ========== 超级拉面（turn=72, PT=5000, RMJ 都成功） ==========
        let mut game2 = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game2.base.turn = 72;
        game2.add_friend_and_npcs()?;
        game2.ramen.scenario_pt = 5000;
        game2.ramen.rmj_results = vec![true, true, true];

        let card_deyilv_only2 = game2.deck[person_idx].effect.deyilv;
        let actual_deyilv2 = game2.deyilv(person_idx as i32)?;
        println!(
            "超级拉面, PT=5000, RMJ都成功: card_deyilv_only={} 实际 deyilv={}",
            card_deyilv_only2, actual_deyilv2
        );
        // 期望：actual_deyilv = card_deyilv_only + (pt(5000档=80) + rmj_success[2]=250) = +330
        assert_eq!(actual_deyilv2, card_deyilv_only2 + 330.0);

        // ========== person_index >= 6 返回 0 ==========
        let actual = game2.deyilv(6)?;
        assert_eq!(actual, 0.0);
        println!("person_index >= 6: deyilv={actual}");

        Ok(())
    }

    /// 三阶段决策衔接测试
    ///
    /// 手动模拟回合 2 的 RamenSelect → SpecialSelect → Train 全流程，
    /// 验证阶段切换与 pending 字段在阶段间正确传递。
    #[test]
    fn test_three_stage_decision_flow() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到回合 2 强制 selected_regions 已设且库存足够
        game.base.turn = 2;
        // 直接给一个够库存的状态（手动跳过 RegionSelect 等阶段）
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2]; // 札幌、函馆、新潟

        // 把 stage 推进到 RamenSelect（手动 set，不经过真实流程）
        game.stage = RamenStage::RamenSelect;

        // ===== 阶段1：RamenSelect =====
        let actions = game.list_actions()?;
        println!("RamenSelect 阶段: {actions:#?}");
        assert!(actions.len() >= 1, "至少有'不吃面'候选");
        // 所有动作 operation 必须是 StageOnly
        for a in &actions {
            assert!(
                matches!(a.operation, Operation::StageOnly),
                "RamenSelect 阶段动作 operation 必须是 StageOnly"
            );
        }
        // 选第一个面（确保库存够）
        let pick_idx = actions
            .iter()
            .position(|a| a.ramen.is_some())
            .expect("至少有一个候选面");
        let ramen_idx = actions[pick_idx].ramen.expect("已 Some");
        game.apply_action(&actions[pick_idx], &mut StdRng::from_os_rng())?;

        // 验证 pending_ramen 已写
        assert_eq!(game.ramen.pending_ramen, Some(ramen_idx));
        println!("pending_ramen: {:?}", game.ramen.pending_ramen);
        // apply 不切 stage；外部 next() 决定推进
        assert!(matches!(game.stage, RamenStage::RamenSelect));

        // 推进 stage：模拟 Game::next() 行为
        let next_stage = if game.ramen.pending_ramen.is_some() {
            RamenStage::SpecialSelect
        } else {
            RamenStage::Train
        };
        game.stage = next_stage;

        // ===== 阶段2：SpecialSelect =====
        let actions = game.list_actions()?;
        println!(
            "SpecialSelect 阶段: {actions:#?}"
        );
        assert!(actions.len() >= 1, "至少有 1 个 targets 候选");
        for a in &actions {
            assert!(
                matches!(a.operation, Operation::StageOnly),
                "SpecialSelect 阶段动作 operation 必须是 StageOnly"
            );
            assert_eq!(a.ramen, Some(ramen_idx));
            assert!(a.special_targets.is_some(), "SpecialSelect 阶段动作应携带 special_targets");
        }

        // 选第一个 targets（按 sum 升序通常第一个是最小必要）
        let chosen_targets = actions[0].special_targets.expect("已 Some");
        game.apply_action(&actions[0], &mut StdRng::from_os_rng())?;

        // 验证 pending_special_targets 已写
        println!("pending_special_targets: {:?}", game.ramen.pending_special_targets);
        assert_eq!(game.ramen.pending_special_targets, chosen_targets);

        // 推进 stage
        game.stage = RamenStage::Train;

        // ===== 阶段3：Train =====
        let actions = game.list_actions()?;
        println!("Train 阶段: {actions:#?}");
        assert!(actions.len() >= 8);
        for a in &actions {
            assert_eq!(a.ramen, Some(ramen_idx), "Train 阶段动作 ramen 应携带 pending");
            assert_eq!(
                a.special_targets,
                Some(chosen_targets),
                "Train 阶段动作 special_targets 应携带 pending"
            );
            assert!(
                !matches!(a.operation, Operation::StageOnly),
                "Train 阶段动作 operation 不应是 StageOnly"
            );
        }

        // 验证 pending_targets 在 Train 阶段动作上携带，且每个 operation 都不再是 StageOnly
        // （不实际 apply 以避免触发 explain_distribution 依赖的 distribution 初始化）

        Ok(())
    }

    /// 合并决策路径端到端测试
    ///
    /// 验证：在 RamenSelect 阶段使用 `apply_combined_ramen_decision` 一次性给出
    /// ramen + targets 后，`Game::next()` 直接把 stage 推到 Train，跳过 SpecialSelect。
    /// 同时验证三阶段路径与合并路径在同一回合内互不干扰。
    #[test]
    fn test_combined_decision_path_skips_special_select() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];

        // 把 stage 推到 RamenSelect
        game.stage = RamenStage::RamenSelect;
        assert!(!game.ramen.combined_decision);

        // ===== 合并决策：选面 0 + targets=[1,0,0] =====
        let combined_actions = game.list_combined_ramen_select_actions();
        println!("合并决策候选数: {}", combined_actions.len());
        // 3 面全富余下：1(不吃) + 9(札幌) + 9(函馆) + 8(新潟) = 27
        assert!(
            combined_actions.len() >= 27,
            "3 面全富余应至少 27 个（实测 {}）",
            combined_actions.len()
        );

        let chosen = combined_actions
            .iter()
            .find(|a| a.ramen == Some(0) && a.special_targets == Some([1, 0, 0]))
            .copied()
            .expect("候选中应包含 面0 + [1,0,0]");

        // 应用合并决策
        game.apply_combined_ramen_decision(chosen.ramen, chosen.special_targets.unwrap())?;

        // 验证 pending 字段已写 + 标记位已设
        assert_eq!(game.ramen.pending_ramen, Some(0));
        assert_eq!(game.ramen.pending_special_targets, [1, 0, 0]);
        assert!(game.ramen.combined_decision, "combined_decision 应为 true");
        // stage 仍是 RamenSelect（不直接设 stage）
        assert!(matches!(game.stage, RamenStage::RamenSelect));

        // ===== Game::next() 推进：合并决策应直接推 Train，跳过 SpecialSelect =====
        game.next();
        println!("next() 后 stage: {:?}", game.stage);
        assert!(
            matches!(game.stage, RamenStage::Train),
            "合并决策路径应直接推 Train（跳过 SpecialSelect）"
        );

        // ===== 关键不变性：再次 next() 不应再推 SpecialSelect =====
        // （SpecialSelect 已被跳过；如果 next() 误推会出错）
        let prev_stage = game.stage.clone();
        // 不再调 next()（会推进到 AfterTrain）；只校验 stage 已是 Train

        // ===== clear_pending 后 combined_decision 应清空（回合边界语义） =====
        game.ramen.clear_pending();
        assert!(!game.ramen.combined_decision);
        assert_eq!(game.ramen.pending_ramen, None);
        assert_eq!(game.ramen.pending_special_targets, [0, 0, 0]);
        println!("clear_pending 后所有 pending 已清空（含 combined_decision）");

        // 防止 "unused" 警告
        let _ = prev_stage;

        Ok(())
    }

    /// 合并决策路径"不吃面"分支测试
    ///
    /// 验证 `apply_combined_ramen_decision(None, ...)` 强制 targets=[0,0,0] 且
    /// `Game::next()` 同样直接推 Train（与"三阶段不吃面"行为一致）。
    #[test]
    fn test_combined_decision_path_no_ramen() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;

        // 不吃面 + 任意 targets（应被强制成 [0,0,0]）
        game.apply_combined_ramen_decision(None, [2, 2, 2])?;
        assert_eq!(game.ramen.pending_ramen, None);
        assert_eq!(game.ramen.pending_special_targets, [0, 0, 0]);
        assert!(game.ramen.combined_decision);

        // next() 推到 Train
        game.next();
        assert!(matches!(game.stage, RamenStage::Train));

        Ok(())
    }

    /// 合并决策路径非法 targets 应报错
    #[test]
    fn test_combined_decision_invalid_targets_rejected() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;

        // 面 0 札幌 recipe=[2,2,1]，targets=[3,0,0] 不合法（t_a 超过 recipe[0]=2）
        let result = game.apply_combined_ramen_decision(Some(0), [3, 0, 0]);
        println!("非法 targets 应报错: {:?}", result.is_err());
        assert!(result.is_err(), "targets 越界应被拒绝");

        // pending 应未写入
        assert_eq!(game.ramen.pending_ramen, None);
        assert!(!game.ramen.combined_decision);

        Ok(())
    }

    /// 三阶段路径在 combined_decision=false 时行为不变（回归测试）
    ///
    /// 确认方案 E 不影响 HandwrittenTrainer 等走三阶段的 Trainer：
    /// RamenSelect → next() 仍按 pending_ramen 决定 SpecialSelect / Train。
    #[test]
    fn test_three_stage_path_unaffected_by_combined_flag() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 2;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;
        assert!(!game.ramen.combined_decision);

        // 走三阶段路径：选面 0 后 apply，写 pending_ramen
        let actions = game.list_actions()?;
        let pick = actions
            .iter()
            .position(|a| a.ramen == Some(0))
            .expect("应有面 0 候选");
        game.apply_action(&actions[pick], &mut StdRng::from_os_rng())?;

        // combined_decision 应保持 false（apply_action 走阶段阶段，不设标记）
        assert!(!game.ramen.combined_decision);
        assert_eq!(game.ramen.pending_ramen, Some(0));

        // next() 应推 SpecialSelect（标准三阶段路径）
        game.next();
        assert!(
            matches!(game.stage, RamenStage::SpecialSelect),
            "三阶段路径下 RamenSelect → SpecialSelect"
        );

        Ok(())
    }

    // ========== RMJ 结算事件 + 固定触发事件 测试 ==========

    /// 验证 `select_rmj_choice_by_result` 的分支选择逻辑
    #[test]
    fn test_select_rmj_choice_by_result() {
        let choices = vec![
            EventChoice {
                result: 2, // 成功
                value: ActionValue { status_pt: [10, 10, 10, 10, 10, 100], vital: 33, ..Default::default() },
                ..Default::default()
            },
            EventChoice {
                result: 1, // 失败
                value: ActionValue { status_pt: [5, 5, 5, 5, 5, 50], vital: 30, ..Default::default() },
                ..Default::default()
            },
        ];

        // 成功（rmj_results=true）→ result=2 分支
        let picked = select_rmj_choice_by_result(&choices, Some(true)).unwrap();
        println!("成功分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 2);
        assert_eq!(picked.value.status_pt[5], 100);

        // 失败（rmj_results=false）→ result=1 分支
        let picked = select_rmj_choice_by_result(&choices, Some(false)).unwrap();
        println!("失败分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 1);
        assert_eq!(picked.value.status_pt[5], 50);

        // 无结算结果 → 回退到第一个分支
        let picked = select_rmj_choice_by_result(&choices, None).unwrap();
        println!("无结果分支 result={}, value={:?}", picked.result, picked.value);
        assert_eq!(picked.result, 2);

        // 空 choices
        let picked = select_rmj_choice_by_result(&[], Some(true));
        assert!(picked.is_none());
        println!("空 choices 返回 None: {:?}", picked);
    }

    /// 验证 `rmj_event_year` 能正确返回年份索引
    #[test]
    fn test_rmj_event_year() {
        assert_eq!(rmj_event_year(401404), Some(0));
        assert_eq!(rmj_event_year(401405), Some(1));
        assert_eq!(rmj_event_year(401406), Some(2));
        assert_eq!(rmj_event_year(401407), None); // 育成结束事件不是 RMJ 事件
        assert_eq!(rmj_event_year(0), None);
        println!("rmj_event_year 映射验证通过");
    }

    /// 验证 RMJ 结算成功时，apply_event 选择 result=2 分支
    #[test]
    fn test_rmj_event_apply_success() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 设置 RMJ 成功状态
        game.ramen.rmj_results = vec![true];

        // 获取 401404 事件并 apply
        let event = find_rmj_event(0).expect("401404 事件应存在");
        let status_before = game.uma.five_status;
        let pt_before = game.uma.skill_pt;
        let vital_before = game.uma.vital;
        println!(
            "应用前: status={:?}, PT={}, vital={}",
            status_before, pt_before, vital_before
        );

        let mut rng = StdRng::seed_from_u64(42);
        game.apply_event(&event, 0, &mut rng)?;

        let status_after = game.uma.five_status;
        let pt_after = game.uma.skill_pt;
        let vital_after = game.uma.vital;
        println!(
            "应用后: status={:?}, PT={}, vital={}",
            status_after, pt_after, vital_after
        );

        // 成功分支应该：速+10, 耐+10, 力+10, 根+10, 智+10, pt+100, vital+33
        for i in 0..5 {
            assert_eq!(
                status_after[i] - status_before[i],
                10,
                "属性 {i} 增量应为 10"
            );
        }
        assert_eq!(pt_after - pt_before, 100);
        assert_eq!(vital_after - vital_before, 33);
        println!("RMJ 成功分支效果验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算失败时，apply_event 选择 result=1 分支
    #[test]
    fn test_rmj_event_apply_fail() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 设置 RMJ 失败状态
        game.ramen.rmj_results = vec![false];

        let event = find_rmj_event(0).expect("401404 事件应存在");
        let status_before = game.uma.five_status;
        let pt_before = game.uma.skill_pt;
        let vital_before = game.uma.vital;
        println!(
            "RMJ 失败前: status={:?}, PT={}, vital={}",
            status_before, pt_before, vital_before
        );

        let mut rng = StdRng::seed_from_u64(42);
        game.apply_event(&event, 0, &mut rng)?;

        let status_after = game.uma.five_status;
        let pt_after = game.uma.skill_pt;
        let vital_after = game.uma.vital;
        println!(
            "RMJ 失败后: status={:?}, PT={}, vital={}",
            status_after, pt_after, vital_after
        );

        // 失败分支应该：速+5, 耐+5, 力+5, 根+5, 智+5, pt+50, vital+30
        for i in 0..5 {
            assert_eq!(
                status_after[i] - status_before[i],
                5,
                "属性 {i} 增量应为 5"
            );
        }
        assert_eq!(pt_after - pt_before, 50);
        assert_eq!(vital_after - vital_before, 30);
        println!("RMJ 失败分支效果验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算后立即 apply 对应事件（在 turn=23 末触发，而非 turn=24 末）
    #[test]
    fn test_rmj_event_immediate_apply_at_turn_23() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 把 vital 调到 0 避免上限截断干扰
        game.uma.vital = 0;
        // 手动模拟 turn=23 RMJ 结算
        game.base.turn = 23;
        game.stage = RamenStage::NextTurn;

        // RMJ 结算前：unresolved 应该为空
        assert!(game.base.unresolved_events.is_empty());

        let pt_before = game.uma.skill_pt;
        let status_before = game.uma.five_status;

        // 触发 next() 中的 RMJ 结算逻辑
        // 注意：turn=23 的 RMJ 结算后会进入 RegionSelect 阶段（不是 advance_turn）
        game.next();
        println!("RMJ 结算后 turn={}, stage={:?}", game.base.turn, game.stage);

        // 验证 RMJ 已结算（rmj_results 写入）
        assert_eq!(game.ramen.rmj_results, vec![false], "默认 PT=0 < 1500 应失败");

        // turn=23 的 RMJ 结算后会进入 RegionSelect 阶段（地区选择是回合 23 末的特殊阶段）
        assert!(
            matches!(game.stage, RamenStage::RegionSelect),
            "RMJ 后应进入 RegionSelect 阶段（turn=23 末）"
        );

        // 验证 RMJ 失败分支已立即应用：pt 增加 50
        let pt_after = game.uma.skill_pt;
        println!("RMJ 结算前 PT={}, 结算后 PT={}", pt_before, pt_after);
        assert_eq!(pt_after - pt_before, 50, "RMJ 失败分支应加 50pt");

        // 验证 status[0] 增加 5（RMJ 失败分支）
        assert_eq!(game.uma.five_status[0] - status_before[0], 5);

        println!("RMJ 事件在 turn=23 末立即 apply 验证通过");

        Ok(())
    }

    /// 验证 RMJ 结算后 scenario_pt 归零，下一年重新累计
    #[test]
    fn test_scenario_pt_reset_after_rmj() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 模拟 turn=23 的 RMJ 结算：先设置 scenario_pt = 2500
        game.base.turn = 23;
        game.stage = RamenStage::NextTurn;
        game.ramen.scenario_pt = 2500;
        let pt_before = game.ramen.scenario_pt;
        println!("RMJ 结算前 scenario_pt = {}", pt_before);

        // 触发 next() 中的 RMJ 结算逻辑
        game.next();

        // 验证 scenario_pt 已归零
        assert_eq!(
            game.ramen.scenario_pt, 0,
            "RMJ 结算后 scenario_pt 应归零（实际 {}）",
            game.ramen.scenario_pt
        );
        println!("RMJ 结算后 scenario_pt = {}（归零成功）", game.ramen.scenario_pt);

        Ok(())
    }

    /// 验证 generate_events 在 turn=0 时返回 400000400 马娘登场事件
    #[test]
    fn test_generate_events_uma_debut() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        // turn=0 应触发马娘登场
        game.base.turn = 0;
        let events = game.generate_events(&mut rng);
        println!("turn=0 事件数: {}, IDs: {:?}", events.len(), events.iter().map(|e| e.id).collect::<Vec<_>>());
        assert!(!events.is_empty(), "turn=0 应有事件");
        assert_eq!(events[0].id, 400000400, "turn=0 第一个事件应是马娘登场");

        Ok(())
    }

    /// 验证 generate_events 在 turn=24 时返回 4009 经典年新年事件
    #[test]
    fn test_generate_events_classic_newyear() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        game.base.turn = 24;
        let events = game.generate_events(&mut rng);
        println!("turn=24 事件数: {}, IDs: {:?}", events.len(), events.iter().map(|e| e.id).collect::<Vec<_>>());
        assert!(!events.is_empty(), "turn=24 应有事件");
        assert_eq!(events[0].id, 4009, "turn=24 第一个事件应是经典年新年");

        Ok(())
    }

    /// 验证 generate_events 在 turn=48 时返回 4010 古马年新年事件
    #[test]
    fn test_generate_events_ancient_newyear() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(42);
        game.base.turn = 48;
        let events = game.generate_events(&mut rng);
        println!("turn=48 事件数: {}, IDs: {:?}", events.len(), events.iter().map(|e| e.id).collect::<Vec<_>>());
        assert!(!events.is_empty(), "turn=48 应有事件");
        assert_eq!(events[0].id, 4010, "turn=48 第一个事件应是古马年新年");

        Ok(())
    }

    /// 验证 add_mandatory_events 在 turn=48 时将 ticket(4011) push 到 unresolved_events
    #[test]
    fn test_add_mandatory_events_ticket_at_48() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 48;
        let mut events = vec![];
        game.add_mandatory_events(&mut events)?;
        // turn=48 没有友人解锁就没有友人事件
        println!("turn=48 同步事件数: {}, unresolved 数: {}", events.len(), game.base.unresolved_events.len());
        // 4011 (ticket) 应在 unresolved_events 中
        assert!(game.base.unresolved_events.iter().any(|e| e.id == 4011));
        println!("turn=48 ticket(4011) 已在 unresolved_events 中");

        Ok(())
    }

    /// 验证 add_mandatory_events 在 turn=77 时将 ending(5011) 和 401407 push 到 unresolved_events
    #[test]
    fn test_add_mandatory_events_ending_at_77() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 77;
        let mut events = vec![];
        game.add_mandatory_events(&mut events)?;
        println!("turn=77 同步事件数: {}, unresolved 数: {}", events.len(), game.base.unresolved_events.len());

        // ending(5011) 和 401407 应在 unresolved_events 中
        let unresolved_ids: Vec<u32> = game.base.unresolved_events.iter().map(|e| e.id).collect();
        println!("turn=77 unresolved_events IDs: {:?}", unresolved_ids);
        assert!(unresolved_ids.contains(&5011), "5011 应在 unresolved_events");
        assert!(unresolved_ids.contains(&401407), "401407 应在 unresolved_events");

        Ok(())
    }

    /// 验证超级拉面回合（turn=72-77）的 vital/motivation 每回合自动恢复
/// + saihou（赛后加成）仅 turn=72 一次性 +100（之后回合不重复累加）
    #[test]
    fn test_super_ramen_base_effect_vital_motivation() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 跳到 URA 第一个回合（turn=72）
        game.base.turn = 72;
        game.add_friend_and_npcs()?;
        // 设置 super_ramen 选项（必要条件之一）
        game.ramen.super_ramen = Some(1);

        // 清零关键字段以便观察增量
        game.uma.vital = 50;
        game.uma.motivation = 2;
        let race_bonus_before = game.uma.race_bonus;
        let vital_before = game.uma.vital;
        let motivation_before = game.uma.motivation;

        // 调用 run_begin（vital/motivation + race_bonus 一次性+100）
        let trainer = RandomTrainer;
        let mut rng = StdRng::from_os_rng();
        game.run_begin(&trainer, &mut rng)?;

        let race_bonus_after_run_begin = game.uma.race_bonus;
        let vital_after = game.uma.vital;
        let motivation_after = game.uma.motivation;
        println!(
            "超级拉面前: vital={}, motivation={}, race_bonus={}",
            vital_before, motivation_before, race_bonus_before
        );
        println!(
            "超级拉面 run_begin 后: vital={}, motivation={}, race_bonus={}",
            vital_after, motivation_after, race_bonus_after_run_begin
        );

        // 验证 turn=72：vital+20, motivation+1, race_bonus+100（一次性）
        assert_eq!(vital_after - vital_before, 20, "vital 应 +20");
        assert_eq!(motivation_after - motivation_before, 1, "motivation 应 +1");
        assert_eq!(
            race_bonus_after_run_begin - race_bonus_before,
            100,
            "turn=72 race_bonus 应一次性 +100"
        );

        println!("超级拉面 turn=72 一次性恢复 + vital/motivation 每回合恢复验证通过");

        Ok(())
    }

    /// 验证 saihou 仅在 turn=72 一次性 +100，turn=73-77 不再累加
    ///
    /// 模拟 turn=72-75 连续运行，观察 race_bonus 只在 turn=72 +100，后续回合不变。
    #[test]
    fn test_super_ramen_saihou_one_time_only() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_logger("test", "info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.ramen.super_ramen = Some(1);

        let race_bonus_initial = game.uma.race_bonus;
        println!("初始 race_bonus: {}", race_bonus_initial);

        let trainer = RandomTrainer;
        // 模拟连续多个 URA 回合（turn=72-75），观察 race_bonus 增量
        for turn in 72..=75 {
            game.base.turn = turn;
            // 重新设置 vital/motivation 以避免上限截断干扰
            game.uma.vital = 50;
            game.uma.motivation = 2;

            let race_bonus_before = game.uma.race_bonus;
            let mut rng = StdRng::from_os_rng();
            game.run_begin(&trainer, &mut rng)?;
            let race_bonus_after = game.uma.race_bonus;
            let expected_increment = if turn == 72 { 100 } else { 0 };
            println!(
                "turn={} 前 race_bonus={}, 后 race_bonus={}, 期望增量={}",
                turn, race_bonus_before, race_bonus_after, expected_increment
            );
            assert_eq!(
                race_bonus_after - race_bonus_before,
                expected_increment,
                "turn={} race_bonus 增量应={}",
                turn,
                expected_increment
            );
        }

        // 最终 race_bonus 应为 initial + 100（仅 turn=72 加了一次）
        assert_eq!(
            game.uma.race_bonus,
            race_bonus_initial + 100,
            "连续 4 回合 URA 后 race_bonus 仅 +100"
        );

        println!("saihou 一次性 +100（不跨回合累积）验证通过");

        Ok(())
    }
}
