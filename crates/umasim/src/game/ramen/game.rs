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
use super::policy::{fixed_region_selection, fixed_super_ramen_selection};
use crate::game::{
    BasePerson, FriendOutState, PersonType,
    traits::{Game, Trainer},
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
                info!("RMJ 结算: {:?} (PT={})", result, self.ramen.scenario_pt);
                self.ramen.eat_count = 0;
            }

            // 特殊阶段跳转
            if self.base.turn == 1 {
                self.stage = RamenStage::RegionSelect;
                return true;
            }
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
            RamenStage::Train => self.run_train(trainer, rng)?,
            RamenStage::AfterTrain => self.run_after_train(trainer, rng)?,
            RamenStage::NextTurn => {} // 回合推进逻辑在 next() 中处理
            RamenStage::RegionSelect => self.run_region_select()?,
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
                    events.push(global_events().friend_events["out"].clone());
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
            809050004 => {
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
            (self.base.train_level_count[train] as usize / 4 + 1).min(5).max(1)
        }
    }

    fn training_basic_value(&self) -> &crate::gamedata::TrainingBasicTable {
        &global!(RAMENDATA).training_basic_value
    }

    fn explain_distribution(&self) -> Result<String> {
        let headers = vec!["速", "耐", "力", "根", "智"];
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
            let ramen_effect = calc_ramen_training_effect(self, train, is_shining);
            let ramen_val = super::effects::apply_ramen_training_value(
                value.status_pt[train], &ramen_effect, train,
            );
            let effective_fail = (fail_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
                .min(100.0).max(0.0);
            if effective_fail > 0.0 {
                lines.push(format!("{} {} 拉面→{} 失败率: {}%", headers[train], value.explain(), ramen_val.0, effective_fail));
            } else {
                lines.push(format!("{} {} 拉面→{}", headers[train], value.explain(), ramen_val.0));
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

        // 动态人头管理
        self.manage_persons_on_turn_start()?;

        // 分配隐藏风味
        let special = get_turn_special_feeling(self.base.turn);
        if special > 0 {
            self.ramen.special_feeling = (self.ramen.special_feeling + special).min(4);
            info!("隐藏风味 +{} (={})", special, self.ramen.special_feeling);
        }

        // 生成并执行回合前事件
        let events = self.generate_events(rng);
        self.add_mandatory_events();
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

    /// RegionSelect 阶段：年度地区选择
    fn run_region_select(&mut self) -> Result<()> {
        let year_idx = (self.current_year() - 1) as usize;
        let selection = fixed_region_selection(year_idx)?;
        self.ramen.selected_regions = selection;
        info!("地区选择: {:?} (第 {} 年)", selection, self.current_year());
        Ok(())
    }

    /// SuperRamenSelect 阶段：超级拉面选择
    fn run_super_ramen_select(&mut self) -> Result<()> {
        let _option = fixed_super_ramen_selection()?;
        self.ramen.super_ramen = Some(1); // 选项二（索引 1）
        info!("超级拉面选择: 选项二");
        Ok(())
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

    /// 添加强制事件（友人事件、育成结束事件）
    fn add_mandatory_events(&mut self) {
        if self.friend.out_state == FriendOutState::AfterUnlock {
            if self.base.turn == 24 {
                self.base.unresolved_events.push(
                    global_events().friend_events["newyear"].clone(),
                );
            } else if self.base.turn == 77 {
                self.base.unresolved_events.push(
                    global_events().friend_events["end"].clone(),
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
