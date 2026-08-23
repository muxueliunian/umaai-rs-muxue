use std::fmt::{Debug, Display};

use anyhow::{Result, anyhow};
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

use super::PersonType;
use crate::{
    diag,
    explain::Explain,
    game::{BaseAction, CardTrainingEffect, SupportCard, Uma},
    gamedata::{ActionValue, EventChoice, EventData, GAMECONSTANTS, TrainingBasicTable, TriggerType},
    global,
    output::{DecisionInfo, GameView}
};
// Game为核心特性，
// ActionEnum 执行动作，修改Game状态
// Trainer 选择动作
// 对事件的处理由Game自己进行

/// 训练人头特性，用于随机分配
pub trait Person: Debug + Clone + PartialEq + Default {
    /// person type getter
    fn person_type(&self) -> PersonType;

    /// person index getter
    fn person_index(&self) -> i32;

    /// train type getter
    fn train_type(&self) -> i32;

    /// friendship getter
    fn friendship(&self) -> i32;

    /// hint getter
    fn hint(&self) -> bool;

    /// hint setter
    fn set_hint(&mut self, hint: bool);

    /// 支援卡 ID getter。非支援卡人头（理事长 / 记者 / NPC）返回 `None`。
    ///
    /// 用于把人头反查回卡组槽位，见 [`Game::deck_index_of`]。
    fn card_id(&self) -> Option<u32>;

    /// provided: 是否为友人，团队，记者或者理事长
    fn is_friend(&self) -> bool {
        self.train_type() > 4 || matches!(self.person_type(), PersonType::Reporter | PersonType::Yayoi)
    }

    /// 是否为剧本友人
    fn is_scenario_card(&self) -> bool {
        self.person_type() == PersonType::ScenarioCard
    }
}

/// 会改变Game状态的主动选项
pub trait ActionEnum: Debug + Display + Clone + PartialEq {
    /// 操作的对象类型，不一定要实现Game Trait
    type Game;

    /// visitor，调用具体动作
    fn apply(&self, game: &mut Self::Game, rng: &mut impl Rng) -> Result<()>;

    /// 尝试转变为BaseAction以获取基础行动类型
    fn as_base_action(&self) -> Option<BaseAction> {
        None
    }
}

/// 游戏状态类型需要实现的Trait，不包括初始化
pub trait Game: Clone {
    type Person: Person;
    type Action: ActionEnum<Game = Self>;

    // 回合相关
    fn turn(&self) -> i32;
    /// 最大回合数 getter
    fn max_turn(&self) -> i32;
    /// 下一阶段。如果已经结束，返回false
    fn next(&mut self) -> bool;
    /// 模拟当前Stage
    fn run_stage<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()>;
    /// provided: 模拟到游戏结束
    fn run_full_game<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        self.run_stage(trainer, rng)?;
        while self.next() {
            self.run_stage(trainer, rng)?;
        }
        // 触发育成结束奖励
        self.on_simulation_end(trainer, rng)?;
        Ok(())
    }
    /// 育成结束时的处理（如最终奖励）
    /// 默认实现为空，由具体剧本覆盖
    fn on_simulation_end<T: Trainer<Self>>(&mut self, _trainer: &T, _rng: &mut StdRng) -> Result<()> {
        Ok(())
    }
    // 动作相关
    /// 获取当前可能的可控行动
    fn list_actions(&self) -> Result<Vec<Self::Action>>;
    /// 生成当前回合的事件
    fn generate_events(&self, rng: &mut impl Rng) -> Vec<EventData>;
    /// 应用事件效果，一些特殊事件需要用到rng和Result
    fn apply_event(&mut self, event: &EventData, choice: usize, rng: &mut impl Rng) -> Result<()>;
    /// 执行事件，如果有选项，交给Trainer去决定
    ///
    /// 决策策略：
    /// - `player_select = true`：调用 Trainer 在 `choices` 间选择（玩家决策）
    /// - `player_select = false`：直接选第 0 组选项，由 `apply_event` 按 prob/result 内部决定具体分支
    ///
    /// 注意：原判断条件 `event.choices.len() > 1` 改用 `event.player_select`，
    /// 以保证带 player_select=true 的事件无论选项数都交给 Trainer 决策，
    /// 而单选项的 RMJ/抽签等随机事件仍走 `apply_event` 内的 prob 加权逻辑。
    fn run_event<T: Trainer<Self>>(&mut self, event: &EventData, trainer: &T, rng: &mut StdRng) -> Result<()> {
        // 事件输出三段式：信息（事件名）→ 决策（选项 + 选择）→ 效果（apply 内部输出）
        diag!("【事件】#{} {}", event.id, event.name);
        if event.player_select && event.choices.len() > 1 {
            for (index, choice) in event.choices.iter().enumerate() {
                diag!("  选项 {}: {}", index + 1, Explain::event_choice(choice));
            }

            // 统一调用 select_event_choice（默认实现内部区分 chance/决策）
            let selection = trainer.select_event_choice(self, event, &event.choices, rng)?;
            if selection >= event.choices.len() {
                return Err(anyhow!(
                    "事件选项索引超出范围: selection={}, choices_len={}, event#{} {}",
                    selection,
                    event.choices.len(),
                    event.id,
                    event.name
                ));
            }
            diag!("  → 选择 选项 {}", selection + 1);
            self.apply_event(&event, selection, rng)
        } else {
            self.apply_event(&event, 0, rng)
        }
    }
    /// provided: 列出本回合触发的事件
    fn list_turn_events(&self, events: &[EventData]) -> Vec<EventData> {
        events
            .iter()
            .filter_map(|e| match &e.trigger {
                TriggerType::Random { .. } => Some(e.clone()),
                TriggerType::Code => None,
                TriggerType::Fixed { turns } => {
                    if turns.contains(&self.turn()) {
                        Some(e.clone())
                    } else {
                        None
                    }
                }
            })
            .collect()
    }
    /// provided: 执行指定动作
    fn apply_action(&mut self, action: &Self::Action, rng: &mut impl Rng) -> Result<()> {
        action.apply(self, rng)
    }
    /// provided: 列出动作，交给训练员判定并执行
    fn list_and_apply_action<T: Trainer<Self>>(&mut self, trainer: &T, rng: &mut StdRng) -> Result<()> {
        let actions = self.list_actions()?;
        if !actions.is_empty() {
            let selection = trainer.select_action(self, &actions, rng)?;
            self.apply_action(&actions[selection], rng)?;
        }
        Ok(())
    }
    // 人头分配相关
    /// persons getter
    fn persons(&self) -> &[Self::Person];
    /// persons mut
    fn persons_mut(&mut self) -> &mut [Self::Person];
    /// 初始化人头
    fn init_persons(&mut self) -> Result<()>;
    /// 已经初始化的人头是否能出现在训练（如记者）
    fn person_is_available(&self, person_index: usize) -> bool {
        match self.persons()[person_index].person_type() {
            PersonType::ScenarioCard => self.turn() >= 2,
            PersonType::Reporter => self.turn() >= 13,
            _ => true
        }
    }
    /// distribution getter
    fn distribution(&self) -> &Vec<Vec<i32>>;
    /// distribution mut
    fn distribution_mut(&mut self) -> &mut Vec<Vec<i32>>;
    /// absent_rate_drop getter
    fn absent_rate_drop(&self) -> i32;
    /// 计算得意率，同时修改卡片计算状态所以要mut
    fn deyilv(&mut self, person_index: i32) -> Result<f32>;
    /// 团队卡是否可以闪彩，不考虑多个团卡的情况
    fn has_group_buff(&self) -> bool;
    /// 显示分布信息
    fn explain_distribution(&self) -> Result<String>;
    /// 重置分布和叹号
    fn reset_distribution(&mut self) {
        self.distribution_mut().clear();
        for _ in 0..5 {
            self.distribution_mut().push(vec![]);
        }
        for p in self.persons_mut() {
            p.set_hint(false);
        }
    }
    /// 追加分配一个在persons里已经存在的人头, -1为不在
    /// 如果要新加角色 需要手动添加到persons里
    fn distribute_person(&mut self, person_index: i32, allow_absent: bool, rng: &mut impl Rng) -> Result<i32> {
        let person = self.persons()[person_index as usize].clone();
        let train_type = person.train_type() as usize;
        // 计算不在率
        let mut absent_rate = match person.person_type() {
            PersonType::Card => 50 - self.absent_rate_drop(),
            PersonType::Yayoi | PersonType::Reporter => 200,
            _ => 100 - self.absent_rate_drop()
        };
        if !allow_absent {
            absent_rate = 0;
        }
        // 计算得意率权重
        let mut weights = [100, 100, 100, 100, 100, absent_rate];
        let mut real_deyilv = 0;
        if train_type <= 4 {
            real_deyilv = self.deyilv(person_index)? as i32;
            weights[train_type] += real_deyilv;
        }
        let weights_sum = 500 + absent_rate + real_deyilv;
        // 先判断是否不在
        if rng.random_bool(absent_rate as f64 / weights_sum as f64) {
            Ok(-1)
        } else {
            let dist = WeightedIndex::new(&weights[0..5])?;
            // 尝试分配
            let d = self.distribution();
            let mut ok = false;
            let mut retries = 0;
            let mut train = 0;
            while !ok && retries < 10 {
                train = dist.sample(rng);
                retries += 1;
                // 不能多于5人或出现同样人头
                if d[train].len() >= 5 || d[train].contains(&person_index) {
                    continue;
                }
                // 每个训练只能出现一个友人
                if person.is_friend() && d[train].iter().any(|index| self.persons()[*index as usize].is_friend()) {
                    continue;
                }
                ok = true;
            }
            if !ok {
                diag!("分配角色#{person_index}失败");
                Ok(-1)
            } else {
                self.distribution_mut()[train as usize].push(person_index);
                Ok(train as i32)
            }
        }
    }
    /// 重新分配所有人头
    fn distribute_all(&mut self, rng: &mut impl Rng) -> Result<()> {
        let sequence = vec![
            PersonType::Yayoi,
            PersonType::Reporter,
            PersonType::ScenarioCard,
            PersonType::TeamCard,
            PersonType::Card,
            PersonType::Npc,
        ];
        self.reset_distribution();
        for ty in sequence {
            for i in 0..self.persons().len() {
                if self.persons()[i].person_type() == ty && self.person_is_available(i) {
                    self.distribute_person(i as i32, true, rng)?;
                }
            }
        }
        Ok(())
    }
    /// 分配Hint. 注意同一个卡的不同分身会同时触发Hint
    fn distribute_hint(&mut self, rng: &mut impl Rng) -> Result<()> {
        let base_hint_rate = global!(GAMECONSTANTS).base_hint_rate / 100.0;
        // 人头下标 ≠ 卡组下标：预抽 (card_id, hint 概率加成)，循环内按 card_id 反查。
        // 这里不能调 deck_index_of——它借 &self，与 persons_mut() 冲突。
        let hint_probs: Vec<(u32, i32)> = self
            .deck()
            .iter()
            .map(|card| (card.card_id, card.card_value().hint_prob_increase))
            .collect();
        for person in self.persons_mut() {
            if person.person_type() == PersonType::Card {
                let bonus = person
                    .card_id()
                    .and_then(|cid| hint_probs.iter().find(|(id, _)| *id == cid))
                    .map_or(0, |(_, bonus)| *bonus);
                let hint_prob = base_hint_rate * ((100 + bonus) as f64 / 100.0);
                person.set_hint(rng.random_bool(hint_prob as f64));
            }
        }
        Ok(())
    }
    // provided: 指定人头出现在训练中的位置
    fn at_trains(&self, person_index: i32) -> Vec<bool> {
        self.distribution()
            .iter()
            .map(|train| train.contains(&person_index))
            .collect()
    }
    /// provided: 指定人头如果在指定位置是否会闪彩 train 0-4 速耐力根智 >=5暂时不考虑  
    /// 非默认实现需要依赖于一部分剧本Buff，所以要在Game里判断
    fn is_shining_at(&self, person_index: usize, train: usize) -> bool {
        let person = &self.persons()[person_index];
        match person.person_type() {
            PersonType::Card => person.train_type() == train as i32 && person.friendship() >= 80,
            PersonType::TeamCard => self.has_group_buff(),
            // 默认实现中其他卡不能闪彩
            _ => false
        }
    }
    /// provided: 指定训练的彩圈个数
    fn shining_count(&self, train: usize) -> usize {
        // 防御：distribution 未初始化时返回 0
        self.distribution()
            .get(train)
            .map(|d| {
                d.iter()
                    .filter(|index| **index >= 0) // 过滤掉分身（负数）和空位（-1）
                    .filter(|index| self.is_shining_at(**index as usize, train))
                    .count()
            })
            .unwrap_or(0)
    }
    // 训练相关
    /// 设施等级 getter
    fn train_level(&self, train: usize) -> usize;
    /// 训练基础值表格（剧本特定）
    fn training_basic_value(&self) -> &TrainingBasicTable;
    /// uma getter
    fn uma(&self) -> &Uma;
    /// uma mut getter
    fn uma_mut(&mut self) -> &mut Uma;
    /// deck getter
    fn deck(&self) -> &Vec<SupportCard>;
    /// provided: 人头下标 → 卡组下标。
    ///
    /// `persons` 与 `deck` 是两个平行容器，下标**不保证**一一对应：拉面剧本的
    /// `init_persons` 只放 5 张训练卡再追加理事长，友人卡推迟到回合 2 才加入，
    /// 于是理事长落在人头 5、友人卡落在人头 6，而 `deck[5]` 是友人卡。
    /// 需要访问 `deck[..]` 时一律走本方法反查，不要拿人头下标直接索引卡组。
    ///
    /// 无卡人头（理事长 / 记者 / NPC）或下标越界返回 `None`。
    ///
    /// 前置条件：`deck` 内 `card_id` 唯一。同一张卡的不同突破共享 `card_id`
    /// （`SupportCard::new` 取 `idrank / 10`），若卡组里放了重复卡，这里会静默
    /// 命中第一张。现有构造路径均不做去重校验，暂由调用方保证。
    fn deck_index_of(&self, person_index: usize) -> Option<usize> {
        let cid = self.persons().get(person_index)?.card_id()?;
        self.deck().iter().position(|card| card.card_id == cid)
    }
    /// provided: 统计一个训练位上吃人数加成的人头数
    ///
    /// 训练值公式里的 `1 + 0.05 × 人数` 乘区。支援卡 / 剧本友人卡 / NPC /
    /// 其他友人 / 团队卡都计入，按 [`PersonType`] 排除理事长与记者。
    ///
    /// 分身不新建人头，而是把本体的 `person_index` 再放进另一个训练位，
    /// 因此这里**按占位逐个计数，不去重**——去重会让分身的加成凭空消失。
    ///
    /// 原实现写作 `p != 6 && p != 7`，那是温泉布局（卡 0-5、理事长 6、记者 7）
    /// 的下标常量。拉面的理事长在 5、友人卡在 6、记者在 12，硬编码下标全错。
    /// 温泉与 base 改判类型后结果逐位不变，详见 `deck_index_of` 的同类说明。
    ///
    /// 负数（`distribute_person` 的「不出现」哨兵）与越界下标一律不计。
    fn count_training_persons(&self, train: usize) -> usize {
        let Some(dist) = self.distribution().get(train) else {
            return 0;
        };
        let persons = self.persons();
        dist.iter()
            .filter(|&&p| p >= 0)
            .filter_map(|&p| persons.get(p as usize))
            .filter(|p| !matches!(p.person_type(), PersonType::Yayoi | PersonType::Reporter))
            .count()
    }
    /// provided: 计算来自支援卡的训练buff
    fn calc_training_buff(&self, train: usize) -> Result<CardTrainingEffect> {
        self.default_calc_training_buff(train)
    }

    /// 如果calc_training_buff被重写，仍然可以调用这里的默认方法
    fn default_calc_training_buff(&self, train: usize) -> Result<CardTrainingEffect> {
        let mut ret = CardTrainingEffect::default();
        if train >= 5 {
            return Err(anyhow!("训练类型错误: {train}"));
        }
        // 防御：distribution 未初始化时（如测试或 ground 阶段触发）返回空 buff
        let train_dist = self.distribution().get(train);
        if let Some(indices) = train_dist {
            for index in indices {
                // 负数为分身占位或空位
                if *index < 0 {
                    continue;
                }
                let person_index = *index as usize;
                // 人头下标 ≠ 卡组下标，必须按 card_id 反查；
                // 无卡人头（理事长 / 记者 / NPC）不贡献任何卡加成
                let Some(deck_index) = self.deck_index_of(person_index) else {
                    continue;
                };
                let card = &self.deck()[deck_index];
                let (mut effect, _) = card.calc_training_effect(self, train as i32)?;
                // 闪彩判定用人头下标，不能用卡组下标
                if !self.is_shining_at(person_index, train) {
                    effect.youqing = 0.0;
                }
                ret = ret.add(&effect);
            }
        }
        Ok(ret)
    }

    /// 可重写: 计算训练属性
    fn calc_training_value(&self, buffs: &CardTrainingEffect, train: usize) -> Result<ActionValue> {
        self.default_calc_training_value(buffs, train)
    }
    /// provided: 计算训练属性
    fn default_calc_training_value(&self, buffs: &CardTrainingEffect, train: usize) -> Result<ActionValue> {
        let train_level = self.train_level(train) - 1; // 返回1-5处理成0-4
        if train >= 5 {
            return Err(anyhow!("训练类型错误: {train}"));
        }
        // 人数, 包括NPC和分身, 排除掉理事长和记者
        // 按 PersonType 判定，不再硬编码人头下标（拉面布局与温泉不同）
        let person_count = self.count_training_persons(train);
        // 基础值
        let basic_value = &self.training_basic_value()[train][train_level];
        let basic_motivation = ((self.uma().motivation - 3) * 10) as f32;
        // 成长率
        let b = &self.uma().five_status_bonus;
        let status_bonus = [b[0], b[1], b[2], b[3], b[4], 0];
        let mut ret = ActionValue::default();
        // 副属性
        for i in 0..6 {
            if basic_value[i] > 0 {
                ret.status_pt[i] = basic_value[i] + buffs.bonus[i];
            }
        }
        ret.vital = basic_value[6];
        // 直接计算。假设buffs里已经算好中间加成
        for i in 0..6 {
            if basic_value[i] > 0 {
                let real_value = ret.status_pt[i] as f32
                    * (1.0 + 0.01 * buffs.youqing as f32)
                    * (1.0 + 0.01 * basic_motivation * (1.0 + 0.01 * buffs.ganjing as f32))
                    * (1.0 + 0.01 * buffs.xunlian as f32)
                    * (1.0 + 0.05 * person_count as f32)
                    * (1.0 + 0.01 * status_bonus[i] as f32);
                ret.status_pt[i] = real_value.floor() as i32;
                //diag!("Train: {train}, i: {i}, real: {real_value}, ret: {}", ret.status_pt[i]);
            }
        }
        // 智力回体
        if train == 4 && buffs.youqing > 0.0 {
            ret.vital += buffs.wiz_vital_bonus;
        }
        // 体力消耗降低
        if ret.vital < 0 {
            ret.vital = (ret.vital as f32 * (1.0 - 0.01 * buffs.vital_cost_drop)) as i32;
        }
        //diag!("Train: {train}, buffs: {}, basic_value: {basic_value:?}, status_bonus: {status_bonus:?}, ret: {ret:?}", buffs.explain());
        Ok(ret)
    }

    // 粗略拟合的训练失败率，二次函数 A*(x0-x)^2+B*(x0-x)
    fn calc_training_failure_rate(&self, buffs: &CardTrainingEffect, train: usize) -> f32 {
        let x0 = global!(GAMECONSTANTS).training_vital_threshold[train][self.train_level(train) - 1];
        let vital = self.uma().vital as f32;
        // 失败率修正
        let bias = if self.uma().flags.good_trainer {
            -2.0
        } else if self.uma().flags.bad_trainer {
            2.0
        } else {
            0.0
        };
        // 原始失败率最大99%
        let mut f = if vital < x0 {
            (100.0 - vital) * (x0 - vital) / 40.0
        } else {
            0.0
        }
        .min(99.0)
        .max(0.0);
        // 如果有不擅长练习，失败率可能达到100%
        f = (f * (100.0 - buffs.fail_rate_drop) / 100.0 + bias).min(100.0).max(0.0);
        f
    }

    /// provided: 游戏状态的结构化展示（Phase 3 / 阶段 4）
    ///
    /// 默认实现从 `Game` 公共接口（`turn()`/`max_turn()`/`uma()`）填充，
    /// `scenario` 字段留空——具体剧本 override 此方法以填入剧本名
    /// （如 `OnsenGame::view` 填 `"onsen"`）。
    ///
    /// 与 `explain()` 的关系：`explain()` 是开发者诊断快照（含 Array5
    /// 等多义性结构），`view()` 是面向下游消费者的结构化字段。两者并存。
    /// 设计依据：见 `.trae/documents/log_refactor_plan.md` §7.4。
    fn view(&self) -> GameView {
        let uma = self.uma();
        GameView {
            scenario: String::new(),
            turn: (self.turn() + 1).max(0) as u32,
            max_turn: self.max_turn().max(0) as u32,
            vital: uma.vital,
            max_vital: uma.max_vital,
            motivation: uma.motivation,
            skill_pt: uma.skill_pt,
            total_hints: uma.total_hints
        }
    }
}

pub trait Trainer<G: Game> {
    /// 选择动作
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize>;
    /// 选择事件选项（旧接口，保留向后兼容）
    fn select_choice(&self, game: &G, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize>;
    /// 选择事件选项（新接口，携带完整的 EventData）
    ///
    /// 暂时回退到 select_choice。
    fn select_event_choice(
        &self, game: &G, _event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.select_choice(game, choices, rng)
    }

    /// 上一次决策的附加上下文（Phase 3 阶段 1 占位实现）
    ///
    /// 默认返回 `None`：Trainer trait 保持只输出 `action_index`，
    /// 面向用户的 Trainer（MctsTrainer / HandwrittenTrainer）按需 override 此方法。
    /// 设计依据：见 `.trae/documents/log_refactor_plan.md` §5。
    fn last_decision(&self) -> Option<DecisionInfo> {
        None
    }

    /// 上一次决策的评分分解文本（开发调参格式，进决策日志 breakdown 列）
    ///
    /// 默认 `None`（随机等无打分的训练员不产生分解）；手写策略等按需 override。
    /// 与 [`last_decision`](Self::last_decision)（协议格式）分层：本方法面向调参日志，
    /// 内容可随意演进（候选数/维度分解/守门原因），不承诺 schema 稳定。
    fn last_breakdown(&self) -> Option<String> {
        None
    }
}
