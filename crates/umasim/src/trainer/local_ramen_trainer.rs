//! 拉面杯策略：在现有 `RamenPolicy` 即时评分上增加受保护的长期收益修正。

use std::sync::Mutex;

use anyhow::Result;
use rand::prelude::StdRng;

use crate::{
    game::{
        FriendOutState, Game, Person, PersonType, Trainer,
        ramen::{
            Operation, RamenAction, RamenGame, RamenStage,
            policy::{RamenPolicy, RamenPolicyOutput}
        }
    },
    gamedata::{EventChoice, EventData}
};

#[derive(Debug, Clone)]
pub struct LocalRamenConfig {
    pub early_bond_value: f32,
    pub hint_bonus: f32,
    pub first_friend_click_value: f32,
    pub low_friend_bond_value: f32,
    pub active_friend_value: f32,
    pub high_fail_penalty: f32,
    pub feeling_overflow_threshold: i32,
    pub overflow_value: f32,
    pub rmj_urgency_margin: i32,
    pub rmj_urgency_bonus: f32,
    /// 长期价值修正最多允许放弃的现有策略即时训练分。
    pub max_base_score_sacrifice: f32
}

impl Default for LocalRamenConfig {
    fn default() -> Self {
        Self {
            early_bond_value: 8.0,
            hint_bonus: 6.0,
            first_friend_click_value: 75.0,
            low_friend_bond_value: 35.0,
            active_friend_value: 8.0,
            high_fail_penalty: 700.0,
            feeling_overflow_threshold: 10,
            overflow_value: 8.0,
            rmj_urgency_margin: 450,
            rmj_urgency_bonus: 60.0,
            max_base_score_sacrifice: 120.0
        }
    }
}

/// 基于现有手写策略、带长期收益修正和基础分保护的拉面杯训练员。
pub struct LocalRamenTrainer {
    policy: RamenPolicy,
    config: LocalRamenConfig,
    last_breakdown: Mutex<Option<String>>
}

impl Default for LocalRamenTrainer {
    fn default() -> Self {
        Self {
            policy: RamenPolicy::default(),
            config: LocalRamenConfig::default(),
            last_breakdown: Mutex::new(None)
        }
    }
}

impl LocalRamenTrainer {
    pub fn new() -> Self {
        Self::default()
    }

    fn choose(outputs: &[RamenPolicyOutput]) -> usize {
        outputs
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.score.total_cmp(&right.score).then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn stash(&self, outputs: &[RamenPolicyOutput]) {
        let text = outputs
            .iter()
            .enumerate()
            .map(|(index, output)| format!("#{index} {:.0}[{}]", output.score, output.reason))
            .collect::<Vec<_>>()
            .join(" | ");
        if let Ok(mut breakdown) = self.last_breakdown.lock() {
            *breakdown = Some(text);
        }
    }

    fn phase_scale(turn: i32) -> f32 {
        if turn < 24 { 1.0 } else if turn < 48 { 0.55 } else { 0.15 }
    }

    fn decide_train(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (guard_choice, mut outputs) = self.policy.decide_train(game, actions)?;
        if outputs.len() != actions.len() {
            return Ok((guard_choice, outputs));
        }
        let base_scores = outputs.iter().map(|output| output.score).collect::<Vec<_>>();
        let base_best = Self::choose(&outputs);
        let phase_scale = Self::phase_scale(game.turn());
        for (action, output) in actions.iter().zip(outputs.iter_mut()) {
            let Operation::Train(training_type) = action.operation else { continue };
            let training = training_type as usize;
            let people = game
                .distribution()
                .get(training)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&person| person >= 0 && (person as usize) < game.persons().len())
                .map(|person| person as usize);
            let mut long_term = 0.0;
            for person_index in people {
                let person = &game.persons()[person_index];
                match person.person_type() {
                    PersonType::ScenarioCard => {
                        long_term += match game.friend.out_state {
                            FriendOutState::UnClicked => self.config.first_friend_click_value,
                            _ if person.friendship() < 60 => self.config.low_friend_bond_value * phase_scale,
                            _ => self.config.active_friend_value
                        };
                    }
                    PersonType::Card if person.friendship() < 80 => {
                        let mut bond_gain: f32 = if game.uma.flags.aijiao { 9.0 } else { 7.0 };
                        if person.hint() { bond_gain += 5.0; }
                        bond_gain = bond_gain.min((80 - person.friendship()) as f32);
                        long_term += bond_gain * self.config.early_bond_value * phase_scale;
                        if person.hint() { long_term += self.config.hint_bonus; }
                    }
                    PersonType::Card if person.hint() => long_term += self.config.hint_bonus,
                    _ => {}
                }
            }
            output.score += long_term;
            output.add("local_long_term", long_term);
            let buffs = game.calc_training_buff(training)?;
            let failure_rate = game.calc_training_failure_rate(&buffs, training);
            if failure_rate > 15.0 {
                let penalty = -((failure_rate - 15.0) / 85.0).clamp(0.0, 1.0) * self.config.high_fail_penalty;
                output.score += penalty;
                output.add("local_high_fail_tail", penalty);
            }
        }
        let local_best = Self::choose(&outputs);
        let sacrifice = base_scores[base_best] - base_scores[local_best];
        let choice = if sacrifice <= self.config.max_base_score_sacrifice { local_best } else { base_best };
        if sacrifice > self.config.max_base_score_sacrifice {
            outputs[choice].reason.push_str(&format!(
                ";保护:长期修正牺牲基础分{sacrifice:.0}>上限{:.0}", self.config.max_base_score_sacrifice
            ));
        }
        Ok((choice, outputs))
    }

    fn decide_ramen(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        let (_, mut outputs) = self.policy.decide_ramen(game, actions)?;
        let stock: i32 = game.ramen.feeling_stock.iter().sum();
        let overflow = (stock - self.config.feeling_overflow_threshold).max(0) as f32;
        let year = (game.current_year() - 1).clamp(0, 2) as usize;
        let gap = [1500, 3000, 3500][year] - game.ramen.scenario_pt;
        for (action, output) in actions.iter().zip(outputs.iter_mut()) {
            if action.ramen.is_none() { continue; }
            let overflow_bonus = overflow * self.config.overflow_value;
            output.score += overflow_bonus;
            output.add("local_stock_overflow", overflow_bonus);
            if gap > 0 && gap <= self.config.rmj_urgency_margin {
                let closeness = 1.0 - gap as f32 / self.config.rmj_urgency_margin as f32;
                let urgency = self.config.rmj_urgency_bonus * (0.5 + 0.5 * closeness);
                output.score += urgency;
                output.add("local_rmj_urgency", urgency);
            }
        }
        Ok((Self::choose(&outputs), outputs))
    }
}

impl Trainer<RamenGame> for LocalRamenTrainer {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], _rng: &mut StdRng) -> Result<usize> {
        if actions.len() <= 1 { return Ok(0); }
        let (choice, outputs) = match game.stage {
            RamenStage::Train => self.decide_train(game, actions)?,
            RamenStage::RamenSelect => self.decide_ramen(game, actions)?,
            RamenStage::SpecialSelect => self.policy.decide_special(game, actions)?,
            RamenStage::RegionSelect => {
                let year = match game.turn() { 2 => 0, 23 => 1, 47 => 2, _ => 0 };
                self.policy.decide_region(game, year, actions)?
            }
            _ => (0, Vec::new())
        };
        self.stash(&outputs);
        Ok(choice)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let (choice, outputs) = self.policy.decide_event(game, choices)?;
        self.stash(&outputs);
        Ok(choice)
    }

    fn select_event_choice(&self, game: &RamenGame, _event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.select_choice(game, choices, rng)
    }

    fn last_breakdown(&self) -> Option<String> {
        self.last_breakdown.lock().ok().and_then(|breakdown| breakdown.clone())
    }
}
