//! 拉面杯动作定义
//!
//! RamenAction 是拉面杯的基本动作单位，采用分离决策模型：
//! - 阶段1：吃面决策（不吃面 / 吃面X / 吃面Y / 吃面Z）
//! - 阶段2：基础操作（训练/比赛/休息/外出/治病）
//!
//! 执行流程：
//! 1. 吃面处理（消耗诀窍、获得PT、触发分身分配）
//! 2. 基础操作执行（训练含拉面效果叠加）

use std::fmt::Display;

use anyhow::{Result, anyhow};
use log::{info, warn};
use rand::{Rng, rngs::StdRng, seq::IndexedRandom};
use serde::{Deserialize, Serialize};

use super::{Operation, TrainingType};
use super::rules::{consume_for_ramen, calc_ramen_pt_gain, fill_gauge_after_train};
use super::effects::{calc_ramen_training_effect, apply_ramen_training_value};
use crate::game::{ActionEnum, BaseAction, FriendOutState, PersonType};
use crate::game::traits::Game;
use crate::gamedata::{ActionValue, GAMECONSTANTS, ramen::RAMENDATA, EventData};
use crate::global;
use crate::utils::{global_events, system_event, system_event_prob};

/// 拉面杯动作
///
/// 包含吃面决策和基础操作两个部分。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamenAction {
    /// 吃面决策
    /// - None: 不吃面
    /// - Some(idx): 吃 ramen_region_effect[idx] 对应的地区拉面
    pub ramen: Option<usize>,
    /// 基础操作
    pub operation: Operation,
}

impl RamenAction {
    /// 创建不吃面 + 基础操作的动作
    pub fn no_ramen(operation: Operation) -> Self {
        Self {
            ramen: None,
            operation,
        }
    }

    /// 创建吃面 + 基础操作的动作
    pub fn with_ramen(ramen_idx: usize, operation: Operation) -> Self {
        Self {
            ramen: Some(ramen_idx),
            operation,
        }
    }

    /// 是否包含吃面决策
    pub fn is_eating_ramen(&self) -> bool {
        self.ramen.is_some()
    }

    /// 获取基础操作
    pub fn base_operation(&self) -> Operation {
        self.operation
    }
}

impl Display for RamenAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ramen_text = match self.ramen {
            Some(idx) => {
                let name = RAMENDATA
                    .get()
                    .and_then(|d| d.ramen_region_effect.get(idx))
                    .map(|r| r.name.as_str())
                    .unwrap_or("???");
                format!("吃面/{name} + ")
            }
            None => String::new(),
        };
        let op_text = match self.operation {
            Operation::Train(train) => {
                let names = &global!(GAMECONSTANTS).train_names;
                format!("{}训练", names[train as usize])
            }
            Operation::Race => "比赛".to_string(),
            Operation::Rest => "休息".to_string(),
            Operation::NormalOuting => "普通出行".to_string(),
            Operation::FriendOuting => "友人出行".to_string(),
            Operation::Clinic => "治病".to_string(),
            Operation::RegionSelect(regions) => {
                let ramen_data = RAMENDATA.get();
                let names: Vec<&str> = regions.iter().filter_map(|&idx| {
                    ramen_data.and_then(|d| d.ramen_region_effect.get(idx)).map(|r| r.name.as_str())
                }).collect();
                format!("地区[{}]", names.join(","))
            }
        };
        write!(f, "{ramen_text}{op_text}")
    }
}

/// 拉面杯动作的 Operation 到 BaseAction 的映射
impl Operation {
    /// 转换为基础动作类型
    pub fn to_base_action(&self) -> Option<BaseAction> {
        match self {
            Operation::Train(t) => Some(BaseAction::Train(*t as i32)),
            Operation::Race => Some(BaseAction::Race),
            Operation::Rest => Some(BaseAction::Sleep),
            Operation::NormalOuting => Some(BaseAction::NormalOuting),
            Operation::FriendOuting => Some(BaseAction::FriendOuting),
            Operation::Clinic => Some(BaseAction::Clinic),
            Operation::RegionSelect(_) => None,
        }
    }
}

/// 拉面杯动作的 Game trait 实现
///
/// 执行流程严格分离：
/// 1. 吃面处理（消耗诀窍、获得PT、触发分身分配）
/// 2. 基础操作执行
impl ActionEnum for RamenAction {
    type Game = super::RamenGame;

    fn apply(&self, game: &mut super::RamenGame, rng: &mut StdRng) -> Result<()> {
        // 阶段1：吃面处理（必须在训练前完成，因为分身会影响人头分布）
        if let Some(ramen_idx) = self.ramen {
            self.apply_ramen(game, ramen_idx, rng)?;
            // 吃面后、训练前，打印当前回合信息
            println!("---- 吃面后 ----");
            let ramen_info = game.explain_ramen_info();
            if !ramen_info.is_empty() {
                info!("{}", ramen_info);
            }
            if let Ok(dist_info) = game.explain_distribution() {
                info!("训练:\n{}", dist_info);
            }
        }

        // 阶段2：执行基础操作
        match self.operation {
            Operation::Train(train) => {
                // 拉面羁绊效果必须在训练前生效（影响闪耀判定）
                self.apply_ramen_friendship(game)?;
                self.do_train(game, train as usize, rng)?;
            }
            Operation::FriendOuting => {
                self.do_friend_outing(game)?;
            }
            Operation::RegionSelect(regions) => {
                game.ramen.selected_regions = regions;
                info!("地区选择: {:?} (第 {} 年)", regions, game.current_year());
            }
            op => {
                if let Some(base_action) = op.to_base_action() {
                    base_action.apply(&mut game.base, rng)?;
                }
            }
        }
        Ok(())
    }

    fn as_base_action(&self) -> Option<BaseAction> {
        self.operation.to_base_action()
    }
}

impl RamenAction {
    /// 阶段1：吃面处理
    ///
    /// 消耗诀窍、获得PT、设置当前拉面状态、触发分身分配。
    /// 分身必须在此阶段分配，因为会影响后续训练的人头分布。
    fn apply_ramen(
        &self,
        game: &mut super::RamenGame,
        ramen_idx: usize,
        rng: &mut StdRng,
    ) -> Result<()> {
        let _recipe = super::rules::get_recipe(ramen_idx)?;
        // 默认不使用隐藏风味（special_targets = [0,0,0]）
        let used_special = consume_for_ramen(&mut game.ramen, ramen_idx, &[0, 0, 0])?;
        game.ramen.current_ramen = Some(ramen_idx);

        // 计算并增加剧本PT
        let year_idx = (game.current_year() - 1) as usize;
        let pt_gain = calc_ramen_pt_gain(year_idx, game.ramen.eat_count)?;
        game.ramen.scenario_pt += pt_gain;
        game.ramen.eat_count += 1;

        info!(
            ">> 吃面[{}] PT+{} (总计{}), 消耗隐藏风味{}",
            ramen_idx, pt_gain, game.ramen.scenario_pt, used_special
        );

        // 分身分配（id >= 5 的地区拉面触发分身）
        self.distribute_clones(game, ramen_idx, rng)?;

        Ok(())
    }

    /// 分配地区拉面分身
    ///
    /// 地区拉面 id >= 5 时，会在指定训练位置分配额外的人头（分身）。
    /// 分身不计算得意率，不包含友人卡。
    ///
    /// 满员规则：
    /// - 每个训练位置最多5个人
    /// - 如果已满5人，分身会优先"挤"掉NPC
    /// - 如果已经包含5个非NPC的人物，则不能创建分身
    ///
    /// 分身分配逻辑：
    /// - 对于 at_trains 中的每个训练位置，随机选择一个支援卡分配分身
    fn distribute_clones(
        &self,
        game: &mut super::RamenGame,
        region_id: usize,
        rng: &mut StdRng,
    ) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let region = &ramen_data.ramen_region_effect[region_id];

        // 检查是否满足分身条件（id >= 5 且 card_type_count >= 4）
        if region_id < 5 || !game.deck_can_split {
            return Ok(());
        }

        let clone_trains = &region.at_trains;
        if clone_trains.is_empty() {
            return Ok(());
        }

        // 获取所有支援卡索引
        let card_indices: Vec<i32> = (0..6i32)
            .filter(|&i| game.persons[i as usize].person_type == PersonType::Card)
            .collect();
        if card_indices.is_empty() {
            return Ok(());
        }

        // 对于 at_trains 中的每个训练位置，随机选择一个不重复的支援卡分配分身
        for &train in clone_trains {
            let train = train as usize;
            if train >= 5 {
                continue;
            }

            // 获取当前训练位置已有的人员（包括本体和分身）
            let existing: std::collections::HashSet<i32> = game.base.distribution[train]
                .iter()
                .filter(|&&id| id >= 0)
                .copied()
                .collect();

            // 过滤掉已在该训练位置的支援卡
            let available: Vec<i32> = card_indices.iter()
                .filter(|&&idx| !existing.contains(&idx))
                .copied()
                .collect();

            if available.is_empty() {
                warn!(">> 分身失败: {}训练无可用支援卡（所有支援卡已在该位置）", 
                    global!(GAMECONSTANTS).train_names[train]);
                continue;
            }

            // 随机选择一个不重复的支援卡
            let person_idx = *available.choose(rng).unwrap();

            // 检查当前训练位置的人数
            let dist = &game.base.distribution[train];
            let non_npc_count = dist.iter()
                .filter(|&&id| id >= 0 && game.persons[id as usize].person_type != PersonType::Npc)
                .count();

            if non_npc_count >= 5 {
                // 已经有5个非NPC人物，不能创建分身
                warn!(">> 分身失败: {}训练已满5个非NPC人物，无法添加分身", 
                    global!(GAMECONSTANTS).train_names[train]);
                continue;
            }

            if dist.len() >= 5 {
                // 已满5人，尝试挤掉NPC
                if let Some(npc_pos) = dist.iter().position(|&id| {
                    id >= 0 && game.persons[id as usize].person_type == PersonType::Npc
                }) {
                    let removed_id = game.base.distribution[train].remove(npc_pos);
                    game.base.distribution[train].push(person_idx);
                    warn!(">> 分身挤掉NPC: {} -> {}训练 (挤掉{})", 
                        game.persons[person_idx as usize].short_name(), 
                        global!(GAMECONSTANTS).train_names[train],
                        game.persons[removed_id as usize].short_name()
                    );
                } else {
                    warn!(">> 分身失败: {}训练已满5人且无NPC可挤，无法添加分身", 
                        global!(GAMECONSTANTS).train_names[train]);
                }
            } else {
                // 未满5人，直接添加
                game.base.distribution[train].push(person_idx);
                info!(">> 分身: {} -> {}训练", 
                    game.persons[person_idx as usize].short_name(), 
                    global!(GAMECONSTANTS).train_names[train]);
            }
        }

        Ok(())
    }

    /// 超级拉面分身分配
    ///
    /// 触发条件：超级拉面回合且支援卡种类>=4
    /// - 每个支援卡（含友人卡）固定额外出现一次
    /// - 分配算法：出现的训练范围由`training_limit_options`指定
    /// - 随机选择训练位置，如果分配失败则重新随机
    /// - 特殊规则：同一训练不能存在相同卡的`Person`和分身
    pub fn distribute_super_ramen_clones(
        game: &mut super::RamenGame,
        rng: &mut StdRng,
    ) -> Result<()> {
        if !game.is_super_ramen_turn() || !game.deck_can_split {
            return Ok(());
        }

        let Some(sel) = game.ramen.super_ramen else {
            return Ok(());
        };
        let options = super::rules::get_super_ramen_clone_train_options()?;
        let Some(option_trains) = options.get(sel) else {
            return Ok(());
        };

        info!(">> 超级拉面分身分配 (选项 {})", sel + 1);

        // 获取所有支援卡索引（含友人卡，index 0-5）
        let card_indices: Vec<i32> = (0..6i32)
            .filter(|&i| {
                let person = &game.persons[i as usize];
                person.person_type == PersonType::Card || person.person_type == PersonType::ScenarioCard
            })
            .collect();

        if card_indices.is_empty() {
            return Ok(());
        }

        // 对每个支援卡，随机分配到一个训练位置，失败则重试
        for &person_idx in &card_indices {
            let mut success = false;
            let max_retries = option_trains.len() * 2; // 最多重试次数
            
            for _ in 0..max_retries {
                // 随机选择一个训练位置
                let &train = option_trains.choose(rng).unwrap();
                let train = train as usize;
                
                match Self::try_add_clone(game, person_idx, train) {
                    Ok(()) => {
                        success = true;
                        break;
                    }
                    Err(_) => continue, // 分配失败，重试
                }
            }
            
            if !success {
                warn!(">> 超级拉面分身失败: {} 无法分配到任何训练位置", 
                    game.persons[person_idx as usize].short_name());
            }
        }

        Ok(())
    }

    /// 尝试在指定训练位置添加分身
    ///
    /// 返回错误如果：
    /// - 已有5个非NPC人物
    /// - 已满5人且无NPC可挤
    fn try_add_clone(
        game: &mut super::RamenGame,
        person_idx: i32,
        train: usize,
    ) -> Result<()> {
        if train >= 5 {
            return Err(anyhow::anyhow!("训练位置越界: {}", train));
        }

        // 检查是否已有该人物的分身
        if game.base.distribution[train].contains(&person_idx) {
            return Err(anyhow::anyhow!("{}训练已有{}的分身", 
                global!(GAMECONSTANTS).train_names[train],
                game.persons[person_idx as usize].short_name()));
        }

        // 检查当前训练位置的人数
        let dist = &game.base.distribution[train];
        let non_npc_count = dist.iter()
            .filter(|&&id| id >= 0 && game.persons[id as usize].person_type != PersonType::Npc)
            .count();

        if non_npc_count >= 5 {
            return Err(anyhow::anyhow!("{}训练已满5个非NPC人物", 
                global!(GAMECONSTANTS).train_names[train]));
        }

        if dist.len() >= 5 {
            // 已满5人，尝试挤掉NPC
            if let Some(npc_pos) = dist.iter().position(|&id| {
                id >= 0 && game.persons[id as usize].person_type == PersonType::Npc
            }) {
                let removed_id = game.base.distribution[train].remove(npc_pos);
                game.base.distribution[train].push(person_idx);
                warn!(">> 超级拉面分身挤掉NPC: {} -> {}训练 (挤掉{})", 
                    game.persons[person_idx as usize].short_name(), 
                    global!(GAMECONSTANTS).train_names[train],
                    game.persons[removed_id as usize].short_name()
                );
            } else {
                return Err(anyhow::anyhow!("{}训练已满5人且无NPC可挤", 
                    global!(GAMECONSTANTS).train_names[train]));
            }
        } else {
            // 未满5人，直接添加
            game.base.distribution[train].push(person_idx);
            info!(">> 超级拉面分身: {} -> {}训练", 
                game.persons[person_idx as usize].short_name(), 
                global!(GAMECONSTANTS).train_names[train]);
        }

        Ok(())
    }

    /// 训练前应用拉面羁绊效果
    ///
    /// `ramen_basic_effect.friendship` 对卡组所有支援卡生效（含友人卡，不含理事长/记者/NPC）。
    /// 必须在训练前生效，因为羁绊值影响闪耀判定（friendship >= 80）。
    ///
    /// 生效条件：吃面回合（`current_ramen.is_some()`）或超级拉面回合（72-77）。
    fn apply_ramen_friendship(&self, game: &mut super::RamenGame) -> Result<()> {
        let eating = game.ramen.current_ramen.is_some();
        let super_ramen = game.is_super_ramen_turn();
        if !eating && !super_ramen {
            return Ok(());
        }
        let year_idx = (game.current_year() - 1) as usize;
        let ramen_data = global!(RAMENDATA);
        if let Some(basic) = ramen_data.ramen_basic_effect.get(year_idx) {
            if basic.friendship > 0 {
                for i in 0..game.persons.len() {
                    if matches!(game.persons[i].person_type, PersonType::Card | PersonType::ScenarioCard) {
                        game.add_friendship(i, basic.friendship);
                    }
                }
            }
        }
        Ok(())
    }

    /// 阶段2：执行训练（含拉面效果叠加）
    ///
    /// 流程：
    /// 1. 计算基础参数（buffs、失败率、拉面效果）
    /// 2. 判定成功/失败
    /// 3. 成功时应用训练值和后续事件
    fn do_train(
        &self,
        game: &mut super::RamenGame,
        train: usize,
        rng: &mut StdRng,
    ) -> Result<()> {
        if train >= 5 {
            return Err(anyhow!("训练类型越界: {train}"));
        }

        info!(
            ">> {}训练 等级 {}",
            global!(GAMECONSTANTS).train_names[train],
            game.train_level(train)
        );

        // 计算训练参数
        let params = self.calc_train_params(game, train)?;

        // 判定成功/失败
        if rng.random_bool(params.failure_rate as f64 / 100.0) {
            self.handle_train_failure(game, params.failure_rate, rng)?;
        } else {
            self.handle_train_success(game, train, &params, rng)?;
        }

        Ok(())
    }

    /// 计算训练参数（buffs、失败率、拉面效果）
    fn calc_train_params(
        &self,
        game: &super::RamenGame,
        train: usize,
    ) -> Result<TrainParams> {
        let buffs = game.calc_training_buff(train)?;
        let is_shining = game.shining_count(train) > 0;
        let ramen_effect = calc_ramen_training_effect(game, train, is_shining);

        // 基础失败率 + 拉面修正
        let base_failure_rate = game.calc_training_failure_rate(&buffs, train);
        let failure_rate = (base_failure_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
            .min(100.0)
            .max(0.0);

        Ok(TrainParams {
            buffs,
            is_shining,
            ramen_effect,
            failure_rate,
        })
    }

    /// 处理训练失败
    fn handle_train_failure(
        &self,
        game: &mut super::RamenGame,
        failure_rate: f32,
        rng: &mut StdRng,
    ) -> Result<()> {
        // 再判断一次，如果还失败就是大失败
        if rng.random_bool(failure_rate as f64 / 100.0) {
            warn!("训练大失败!");
            game.apply_event(system_event("training_fail_low")?, 0, rng)?;
            game.uma.flags.ill = true;
            game.uma.flags.bad_trainer = true;
        } else {
            warn!("训练失败!");
            game.apply_event(system_event("training_fail")?, 0, rng)?;
        }
        Ok(())
    }

    /// 处理训练成功
    fn handle_train_success(
        &self,
        game: &mut super::RamenGame,
        train: usize,
        params: &TrainParams,
        rng: &mut StdRng,
    ) -> Result<()> {
        // 计算并应用训练值（基础值 + 拉面效果同时生效）
        let base_value = game.calc_training_value(&params.buffs, train)?;
        let final_value = self.apply_ramen_to_train_value(base_value, train, params);
        game.uma.add_value(&final_value);

        // 增加训练次数
        game.base.train_level_count[train] += 1;

        // 处理羁绊和后续事件
        self.handle_post_train(game, train, rng)?;

        // 诀窍槽填充
        self.fill_feeling_gauge(game, train, params)?;

        Ok(())
    }

    /// 应用拉面效果到训练值
    ///
    /// 拉面效果和PT加成同时生效，一次性计算最终值。
    fn apply_ramen_to_train_value(
        &self,
        base_value: ActionValue,
        train: usize,
        params: &TrainParams,
    ) -> ActionValue {
        // 属性训练值：lower * (100+xunlian)/100 * (100+youqing)/100
        let (status_val, pt_val) = apply_ramen_training_value(
            base_value.status_pt[train],
            &params.ramen_effect,
            train,
        );

        // PT训练值：属性值 * (100+pt_bonus)/100
        let (_, final_pt) = apply_ramen_training_value(
            base_value.status_pt[5],
            &params.ramen_effect,
            train,
        );

        ActionValue {
            status_pt: {
                let mut arr = [0; 6];
                arr[train] = status_val;
                arr[5] = final_pt;
                arr
            },
            vital: base_value.vital,
            motivation: base_value.motivation,
            ..Default::default()
        }
    }

    /// 处理训练后的羁绊和事件
    fn handle_post_train(
        &self,
        game: &mut super::RamenGame,
        train: usize,
        rng: &mut StdRng,
    ) -> Result<()> {
        let friendship_bonus = if game.uma.flags.aijiao { 9 } else { 7 };
        let mut hint_persons = vec![];
        let mut friend_clicked = false;

        for person_index in game.distribution[train].clone() {
            if person_index < 0 {
                continue;
            }
            game.add_friendship(person_index as usize, friendship_bonus);
            if game.persons[person_index as usize].is_hint {
                hint_persons.push(person_index);
            }
            if game.persons[person_index as usize].person_type == PersonType::ScenarioCard {
                friend_clicked = true;
            }
        }

        // Hint 事件
        self.handle_hint_event(game, &hint_persons, rng)?;

        // 额外训练事件（非合宿）
        let extra_train_prob = system_event_prob("extra_train")?;
        if !game.is_xiahesu() && rng.random_bool(extra_train_prob as f64) {
            let mut event = EventData::extra_training_event(train);
            // 动态设置理事长索引
            if let Some(yayoi_index) = game.persons.iter()
                .position(|p| p.person_type == PersonType::Yayoi) {
                event.person_index = Some(yayoi_index as i32);
            }
            game.base.unresolved_events.push(event);
        }

        // 友人点击事件
        if friend_clicked {
            self.handle_friend_click(game, rng)?;
        }

        Ok(())
    }

    /// 处理 Hint 事件
    fn handle_hint_event(
        &self,
        game: &mut super::RamenGame,
        hint_persons: &[i32],
        rng: &mut StdRng,
    ) -> Result<()> {
        if let Some(&p) = hint_persons.choose(rng) {
            if p < 0 || p as usize >= game.persons.len() {
                return Ok(());
            }
            let person_index = p as usize;
            let attr_prob = system_event_prob("hint_attr")?;
            let hint_level = if person_index < 6 {
                1 + game.deck[person_index].card_value().hint_level
            } else {
                1
            };
            let mut hint_event = if rng.random_bool(attr_prob) {
                EventData::hint_attr_event(game.persons[person_index].train_type as usize, person_index)?
            } else {
                EventData::hint_skill_event(hint_level, person_index)
            };
            hint_event.name = format!("{} - {}", hint_event.name, game.deck[person_index].short_name());
            game.base.unresolved_events.push(hint_event);
        }
        Ok(())
    }

    /// 处理友人点击事件（使用拉面杯友人事件）
    fn handle_friend_click(
        &self,
        game: &mut super::RamenGame,
        _rng: &mut StdRng,
    ) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        match game.friend.out_state {
            FriendOutState::UnClicked => {
                game.friend.out_state = FriendOutState::BeforeUnlock;
                let mut event = ramen_data.friend_events["first"].clone();
                event.person_index = Some(game.friend.person_index as i32);
                game.base.unresolved_events.push(event);
            }
            _ => {
                let mut event = ramen_data.friend_events["click"].clone();
                event.person_index = Some(game.friend.person_index as i32);
                game.base.unresolved_events.push(event);
            }
        }
        Ok(())
    }

    /// 友人出行（使用拉面杯友人出行事件 + 增加隐藏风味）
    fn do_friend_outing(
        &self,
        game: &mut super::RamenGame,
    ) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let mut which = 0;
        while which < 5 && game.friend.out_used[which] {
            which += 1;
        }
        if which < 5 {
            info!(">> 友人出行 #{}", which + 1);
            let key = format!("outing{}", which + 1);
            let mut event = ramen_data.friend_events[&key].clone();
            event.person_index = Some(game.friend.person_index as i32);
            game.friend.out_used[which] = true;
            game.base.unresolved_events.push(event);

            // 友人出行后获得隐藏风味（新友人固定2个）
            let special = 2;
            game.ramen.special_feeling = (game.ramen.special_feeling + special).min(4);
            info!(">> 隐藏风味 +{} (={})", special, game.ramen.special_feeling);
            Ok(())
        } else {
            Err(anyhow!("友人出行越界: {which}"))
        }
    }

    /// 填充诀窍槽
    fn fill_feeling_gauge(
        &self,
        game: &mut super::RamenGame,
        train: usize,
        params: &TrainParams,
    ) -> Result<()> {
        if let Some(train_feelings) = game.ramen.train_feeling_type {
            let base_dist = super::rules::calc_gauge_base_distribution(&game.ramen.selected_regions);
            // 计算支援卡数量（包括分身，分身id < 0 但本体 id 在 0-5 范围）
            let support_count = game.distribution[train]
                .iter()
                .filter(|&&p| p != 6 && p != 7)  // 排除理事长和记者
                .count();
            let train_bonus = super::rules::calc_train_feeling_bonus(
                support_count,
                5, // 5 个 NPC
            );
            fill_gauge_after_train(
                &mut game.ramen,
                &base_dist,
                train_feelings[train],
                train_bonus,
                params.is_shining,
            );
        }
        Ok(())
    }
}

/// 训练参数（计算后缓存）
struct TrainParams {
    /// 支援卡 Buff
    buffs: crate::game::CardTrainingEffect,
    /// 是否友情训练
    is_shining: bool,
    /// 拉面训练效果
    ramen_effect: super::effects::RamenTrainingEffect,
    /// 失败率（百分比）
    failure_rate: f32,
}

/// 列出吃面选择（阶段1）
///
/// 返回所有可用的吃面选择：
/// - 不吃面
/// - 吃面X（如果诀窍足够）
/// - 吃面Y
/// - 吃面Z
///
/// # 参数
/// - `available_ramens`: 当前可以吃的面（诀窍足够）
pub fn list_ramen_choices(available_ramens: &[usize]) -> Vec<Option<usize>> {
    let mut choices = vec![None]; // 不吃面
    for &idx in available_ramens {
        choices.push(Some(idx));
    }
    choices
}

/// 列出所有基础操作（阶段2）
///
/// 返回所有可用的基础操作。
///
/// # 参数
/// - `can_friend_outing`: 是否可以选择友人出行
/// - `is_ill`: 是否生病
pub fn list_operations(can_friend_outing: bool, is_ill: bool) -> Vec<Operation> {
    let mut ops = vec![
        Operation::Train(TrainingType::Speed),
        Operation::Train(TrainingType::Stamina),
        Operation::Train(TrainingType::Power),
        Operation::Train(TrainingType::Guts),
        Operation::Train(TrainingType::Wisdom),
        Operation::Race,
        Operation::Rest,
        Operation::NormalOuting,
    ];
    if can_friend_outing {
        ops.push(Operation::FriendOuting);
    }
    if is_ill {
        ops.push(Operation::Clinic);
    }
    ops
}

/// 生成所有组合动作
///
/// 组合 = 吃面选择 × 基础操作。
///
/// # 参数
/// - `available_ramens`: 当前可以吃的面
/// - `can_friend_outing`: 是否可以选择友人出行
/// - `is_ill`: 是否生病
pub fn list_all_actions(
    available_ramens: &[usize],
    can_friend_outing: bool,
    is_ill: bool,
) -> Vec<RamenAction> {
    let ramen_choices = list_ramen_choices(available_ramens);
    let operations = list_operations(can_friend_outing, is_ill);

    let mut actions = Vec::new();
    for ramen in &ramen_choices {
        for &op in &operations {
            match ramen {
                None => actions.push(RamenAction::no_ramen(op)),
                Some(idx) => actions.push(RamenAction::with_ramen(*idx, op)),
            }
        }
    }
    actions
}

/// 获取当年可用的面（诀窍足够）
///
/// 返回可以吃的面的 ID 列表。
pub fn get_available_ramens(
    state: &super::RamenState,
    selected_regions: &[usize; 3],
) -> Vec<usize> {
    use super::rules::can_make_ramen;

    let mut available = Vec::new();
    for &region_id in selected_regions {
        if let Ok(recipe) = super::rules::get_recipe(region_id) {
            if can_make_ramen(state, recipe, &[0, 0, 0]) {
                available.push(region_id);
            }
        }
    }
    available
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::RamenState;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_logger}};

    #[test]
    fn test_ramen_action_display() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        let a1 = RamenAction::no_ramen(Operation::Train(TrainingType::Speed));
        println!("动作1: {a1}");

        let a2 = RamenAction::with_ramen(0, Operation::Train(TrainingType::Wisdom));
        println!("动作2: {a2}");

        let a3 = RamenAction::no_ramen(Operation::Race);
        println!("动作3: {a3}");

        let a4 = RamenAction::with_ramen(5, Operation::Rest);
        println!("动作4: {a4}");

        Ok(())
    }

    #[test]
    fn test_ramen_action_properties() {
        let a1 = RamenAction::no_ramen(Operation::Rest);
        assert!(!a1.is_eating_ramen());
        assert_eq!(a1.base_operation(), Operation::Rest);

        let a2 = RamenAction::with_ramen(5, Operation::Race);
        assert!(a2.is_eating_ramen());
        assert_eq!(a2.ramen, Some(5));
        assert_eq!(a2.base_operation(), Operation::Race);
    }

    #[test]
    fn test_list_ramen_choices() {
        // 无可用面
        let choices = list_ramen_choices(&[]);
        println!("无可用面: {choices:?}");
        assert_eq!(choices, vec![None]);

        // 有3种可用面
        let choices = list_ramen_choices(&[0, 1, 2]);
        println!("有3种可用面: {choices:?}");
        assert_eq!(choices, vec![None, Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_list_operations() {
        // 基础情况
        let ops = list_operations(false, false);
        println!("基础操作: {} 个", ops.len());
        assert_eq!(ops.len(), 8); // 5训练+比赛+休息+普通外出

        // 有友人出行和治病
        let ops = list_operations(true, true);
        println!("有友人+治病: {} 个", ops.len());
        assert_eq!(ops.len(), 10); // 5训练+比赛+休息+普通外出+友人出行+治病
    }

    #[test]
    fn test_list_all_actions() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        // 无可用面，无友人，无生病
        let actions = list_all_actions(&[], false, false);
        println!("无可用面: {} 个动作", actions.len());
        // 1吃面选择 * 8操作 = 8
        assert_eq!(actions.len(), 8);

        // 有3种可用面
        let actions = list_all_actions(&[0, 1, 2], false, false);
        println!("有3种可用面: {} 个动作", actions.len());
        // 4吃面选择 * 8操作 = 32
        assert_eq!(actions.len(), 32);

        // 有友人出行和治病，3种可用面
        let actions = list_all_actions(&[0, 1, 2], true, true);
        println!("有友人+治病+3种面: {} 个动作", actions.len());
        // 4吃面选择 * 10操作 = 40
        assert_eq!(actions.len(), 40);

        // 列出所有动作
        for (i, a) in actions.iter().enumerate() {
            println!("  {i:2}: {a}");
        }

        Ok(())
    }

    #[test]
    fn test_get_available_ramens() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        let selected_regions = [0, 1, 2];

        // 诀窍足够
        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5];
        let available = get_available_ramens(&state, &selected_regions);
        println!("诀窍足够: 可用面={available:?}");
        assert_eq!(available.len(), 3);

        // 诀窍不足
        state.feeling_stock = [0, 0, 0];
        let available = get_available_ramens(&state, &selected_regions);
        println!("诀窍不足: 可用面={available:?}");
        assert_eq!(available.len(), 0);

        Ok(())
    }
}
