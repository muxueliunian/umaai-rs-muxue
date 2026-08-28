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
use rand::{Rng, seq::IndexedRandom};
use serde::{Deserialize, Serialize};

use super::{
    Operation,
    TrainingType,
    effects::calc_ramen_training_effect,
    rules::{fill_gauge_after_non_train, fill_gauge_after_train}
};
use crate::{
    diag,
    game::{ActionEnum, BaseAction, FriendOutState, PersonType, traits::{Game, Person}},
    gamedata::{EventData, GAMECONFIG, GAMECONSTANTS, RamenRegionStrategy, ramen::RAMENDATA},
    global,
    rng::{CLONE_SUPER_TAG, fork_local_stream},
    utils::{system_event, system_event_prob}
};

/// 拉面杯动作
///
/// 三阶段选择的状态机下，`RamenAction` 复用单一结构承载每个阶段的决策：
/// - `RamenSelect` 阶段：仅 `ramen` 字段有意义，`operation = Operation::StageOnly`
/// - `SpecialSelect` 阶段：`ramen = pending`、`special_targets` 字段有意义，`operation = StageOnly`
/// - `Train` 阶段：`ramen`/`special_targets` 为 pending 拷贝，`operation` 为真实操作
///
/// 合并决策路径下，`combined_select` 构造的中间步骤动作同时承载 `ramen` 和 `special_targets`，
/// Trainer 在 `RamenSelect` 阶段一次性给出两个决策，由 `apply_combined_ramen_decision` 处理。
///
/// `apply` 按当前 `game.stage` 路由处理：中间步骤动作仅切阶段并写入 pending，
/// Train 阶段动作才真执行吃面 + 基础操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamenAction {
    /// 吃面决策
    /// - None: 不吃面
    /// - Some(idx): 吃 ramen_region_effect[idx] 对应的地区拉面
    pub ramen: Option<usize>,
    /// 隐藏风味替换目标
    /// - None: 不替换（吃面但不省诀窍，或中间步骤未确定）
    /// - Some([tA, tB, tC]): 替换各类 `t[i]` 个普通诀窍
    pub special_targets: Option<[i32; 3]>,
    /// 基础操作
    pub operation: Operation
}

impl RamenAction {
    /// 创建只承载基础操作的 Train 阶段动作
    ///
    /// 重构后，Train 阶段不再带 `ramen` 和 `special_targets` 字段（已由
    /// `ground_ramen_effects` 落地）。`ramen = None`、`special_targets = None`，
    /// 与 [`Self::no_ramen`] 语义一致，但命名更清晰。
    pub fn new(operation: Operation) -> Self {
        Self {
            ramen: None,
            special_targets: None,
            operation
        }
    }

    /// 创建不吃面 + 基础操作的动作（保留旧 API，等价于 [`Self::new`]）
    pub fn no_ramen(operation: Operation) -> Self {
        Self::new(operation)
    }

    /// 创建吃面 + 基础操作的动作（保留旧 API，用于合并决策 path 中间表示）
    pub fn with_ramen(ramen_idx: usize, operation: Operation) -> Self {
        Self {
            ramen: Some(ramen_idx),
            special_targets: None,
            operation
        }
    }

    /// 创建 `RamenSelect` 阶段动作（仅承载面选择，不含 operation）。
    ///
    /// `ramen_idx = None` 表示不吃面；否则为 `selected_regions` 中某一面。
    pub fn ramen_select(ramen_idx: Option<usize>) -> Self {
        Self {
            ramen: ramen_idx,
            special_targets: None,
            operation: Operation::StageOnly
        }
    }

    /// 创建 `SpecialSelect` 阶段动作（承载隐藏风味用法）。
    pub fn special_select(ramen_idx: usize, targets: [i32; 3]) -> Self {
        Self {
            ramen: Some(ramen_idx),
            special_targets: Some(targets),
            operation: Operation::StageOnly
        }
    }

    /// 创建"合并决策"阶段动作（一次承载 ramen + targets 两个决策）
    ///
    /// 约定：`ramen_idx = None` 时强制 `targets = [0,0,0]`（即"不吃面"在合并决策视角下
    /// 与"吃面 + 全零 targets"等价，但保持 ramen=None 以便 Trainer/日志清晰识别）。
    pub fn combined_select(ramen_idx: Option<usize>, targets: [i32; 3]) -> Self {
        let targets = if ramen_idx.is_none() { [0, 0, 0] } else { targets };
        Self {
            ramen: ramen_idx,
            special_targets: Some(targets),
            operation: Operation::StageOnly
        }
    }

    /// 创建 `SuperRamenSelect` 阶段动作（承载超级拉面选项下标）。
    ///
    /// `idx` 是 `finals_effect.training_limit_options` 的位置下标，不是地区拉面 ID。
    /// `ramen` / `special_targets` 保持空：这两个字段的语义不属于超级拉面。
    pub fn super_ramen_select(idx: usize) -> Self {
        Self {
            ramen: None,
            special_targets: None,
            operation: Operation::SuperRamenSelect(idx)
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

impl RamenAction {
    /// 吃面决策的可读文本（`"不吃面"` / `"吃面/中山-全"` / `"吃面/中山-全(替换Bx1)"`）
    ///
    /// 选择阶段（RamenSelect / SpecialSelect / 合并决策中间步骤）动作的呈现主体，
    /// 只表达"吃不吃面、怎么替换"，不携带基础操作。
    fn ramen_decision_text(&self) -> String {
        // 隐藏风味替换说明（如有）
        let targets_text = match self.special_targets {
            Some(t) if t.iter().any(|&x| x > 0) => {
                let mut parts = Vec::new();
                for (i, &n) in t.iter().enumerate() {
                    if n > 0 {
                        parts.push(format!("{}x{}", "ABC".chars().nth(i).unwrap(), n));
                    }
                }
                format!("(替换{})", parts.join("+"))
            }
            _ => String::new()
        };
        match self.ramen {
            Some(idx) => {
                let name = RAMENDATA
                    .get()
                    .and_then(|d| d.ramen_region_effect.get(idx))
                    .map(|r| r.name.as_str())
                    .unwrap_or("???");
                format!("吃面/{name}{targets_text}")
            }
            None => "不吃面".to_string()
        }
    }

    /// 基础操作的可读文本；选择阶段动作返回空串（此时只有吃面决策，无操作）
    fn operation_text(&self) -> String {
        match self.operation {
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
                let names: Vec<&str> = regions
                    .iter()
                    .filter_map(|&idx| {
                        ramen_data
                            .and_then(|d| d.ramen_region_effect.get(idx))
                            .map(|r| r.name.as_str())
                    })
                    .collect();
                format!("地区[{}]", names.join(","))
            }
            Operation::StageOnly => String::new(),
            Operation::SuperRamenSelect(idx) => format!("超级拉面选项 {}", idx + 1)
        }
    }
}

impl Display for RamenAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ramen_text = self.ramen_decision_text();
        let op_text = self.operation_text();
        if op_text.is_empty() {
            // 选择阶段动作（RamenSelect / SpecialSelect / 合并决策中间步骤）：
            // 只呈现吃面决策，无 `<下一步>` 之类的机器痕迹
            write!(f, "{ramen_text}")
        } else if self.ramen.is_some() {
            // 合并决策路径的 `with_ramen` 动作：吃面 + 基础操作
            write!(f, "{ramen_text} + {op_text}")
        } else {
            // Train 阶段动作：只呈现基础操作
            write!(f, "{op_text}")
        }
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
            Operation::StageOnly => None,
            Operation::SuperRamenSelect(_) => None
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

    fn apply(&self, game: &mut super::RamenGame, rng: &mut impl Rng) -> Result<()> {
        use super::RamenStage;
        match game.stage {
            RamenStage::RamenSelect => {
                // race_turn 短路：operation 非 StageOnly 表示 race turn 一体化执行
                if !matches!(self.operation, Operation::StageOnly) {
                    // 非中间步骤动作（如 race_turn 的 Race）：直接执行 operation 并跳到 AfterTrain
                    if let Some(base_action) = self.operation.to_base_action() {
                        base_action.apply(&mut game.base, rng)?;
                    }
                    game.stage = RamenStage::AfterTrain;
                    return Ok(());
                }
                // 中间步骤：仅承载面选择，写 pending；Game::next() 会按 pending_ramen 推 SpecialSelect 或 Train
                game.ramen.pending_ramen = self.ramen;
                game.ramen.pending_special_targets = [0, 0, 0];
                Ok(())
            }
            RamenStage::SpecialSelect => {
                // 中间步骤：仅承载隐藏风味用法，写 pending；Game::next() 推到 Train
                let targets = self
                    .special_targets
                    .ok_or_else(|| anyhow::anyhow!("SpecialSelect 阶段动作应携带 special_targets"))?;
                game.ramen.pending_special_targets = targets;
                Ok(())
            }
            RamenStage::Train => {
                // 拉面效果已在 SpecialSelect → Train 过渡时由 `ground_ramen_effects()` 全部落地
                // （消耗诀窍 / PT 增量 / current_ramen / 分身 / 羁绊 / 显示 buff+distribution）
                // 此处只负责执行 operation（训练/比赛/休息等）
                let is_xiahesu = game.is_xiahesu();

                match self.operation {
                    Operation::Train(train) => {
                        self.do_train(game, train as usize, rng)?;
                        // 训练分支：fill_feeling_gauge 已在 do_train 内统一处理（含 is_xiahesu）
                    }
                    Operation::FriendOuting => {
                        self.do_friend_outing(game)?;
                        self.fill_gauge_non_train(game, is_xiahesu)?;
                    }
                    Operation::RegionSelect(regions) => {
                        apply_region_selection(game, regions)?;
                    }
                    Operation::StageOnly => {
                        // Train 阶段不应收到 StageOnly 操作；若出现则忽略
                    }
                    Operation::Rest => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        // 夏合宿休息自动治病（等同 Clinic 效果）
                        if is_xiahesu {
                            game.uma.flags.ill = false;
                            game.uma.flags.bad_trainer = false;
                            diag!(">> 夏合宿休息：自动治病");
                        }
                        self.fill_gauge_non_train(game, is_xiahesu)?;
                    }
                    Operation::NormalOuting => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        self.fill_gauge_non_train(game, is_xiahesu)?;
                    }
                    Operation::Race => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        self.fill_gauge_non_train(game, is_xiahesu)?;
                    }
                    Operation::Clinic => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        // 按设计：治病动作不获得诀窍槽
                    }
                    Operation::SuperRamenSelect(_) => {
                        return Err(anyhow!(
                            "超级拉面选择只能在 SuperRamenSelect 阶段执行（当前 Train）"
                        ));
                    }
                }
                Ok(())
            }
            RamenStage::SuperRamenSelect => {
                // 只写字段，不切阶段、不推进回合；阶段推进一律留给 Game::next()
                match self.operation {
                    Operation::SuperRamenSelect(idx) => {
                        let n = super::rules::get_super_ramen_clone_train_options()?.len();
                        if idx >= n {
                            return Err(anyhow!("超级拉面选项下标越界: idx={idx}, 共 {n} 个"));
                        }
                        game.ramen.super_ramen = Some(idx);
                        Ok(())
                    }
                    other => Err(anyhow!("SuperRamenSelect 阶段收到非法动作: {other:?}"))
                }
            }
            // 其他阶段（如 RegionSelect）保持旧行为，按 operation 直接分发
            // 拉面效果已由 ground_ramen_effects() 在阶段过渡时落地，此处不再重复
            _ => {
                let is_xiahesu = game.is_xiahesu();
                match self.operation {
                    Operation::RegionSelect(regions) => {
                        apply_region_selection(game, regions)?;
                    }
                    Operation::StageOnly => {}
                    Operation::SuperRamenSelect(_) => {
                        return Err(anyhow!(
                            "超级拉面选择只能在 SuperRamenSelect 阶段执行（当前 {:?})",
                            game.stage
                        ));
                    }
                    Operation::Rest => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        if is_xiahesu {
                            game.uma.flags.ill = false;
                            game.uma.flags.bad_trainer = false;
                            diag!(">> 夏合宿休息：自动治病");
                        }
                        self.fill_gauge_non_train(game, is_xiahesu)?;
                    }
                    Operation::Clinic => {
                        if let Some(base_action) = self.operation.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                    }
                    op => {
                        if let Some(base_action) = op.to_base_action() {
                            base_action.apply(&mut game.base, rng)?;
                        }
                        // 其他非训练动作（比赛/普通外出/友人出行）按需补 fill_gauge
                        match op {
                            Operation::Race | Operation::NormalOuting | Operation::FriendOuting => {
                                self.fill_gauge_non_train(game, is_xiahesu)?;
                            }
                            _ => {}
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn as_base_action(&self) -> Option<BaseAction> {
        self.operation.to_base_action()
    }
}

impl RamenAction {
    /// 超级拉面分身分配
    ///
    /// 触发条件：超级拉面回合且支援卡种类>=4
    /// - 每个支援卡（含友人卡）固定额外出现一次
    /// - 分配算法：出现的训练范围由`training_limit_options`指定
    /// - 对每张卡先过滤出合法训练位再均匀抽一个；有合法位就一定放下
    /// - 友人卡**优先**分配（见实现内注释：这是实现选择，非规格）
    /// - 特殊规则：同一训练不能存在相同卡的`Person`和分身
    /// - 规则上真的放不下时**跳过该卡并告警，不返回 `Err`**：分身放不下在规则上允许，
    ///   中断育成会让整次搜索 rollout 作废，且只丢弃「无解局面」这一类盘面
    pub fn distribute_super_ramen_clones(game: &mut super::RamenGame, rng: &mut impl Rng) -> Result<()> {
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

        diag!(">> 超级拉面分身分配 (选项 {})", sel + 1);

        // 获取所有支援卡索引（含友人卡）
        // 人头下标 ≠ 卡组下标：拉面友人卡在人头 6，写死 0..6 会把它整个漏掉
        let mut card_indices: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| {
                let person = &game.persons[i as usize];
                person.person_type == PersonType::Card || person.person_type == PersonType::ScenarioCard
            })
            .collect();

        if card_indices.is_empty() {
            return Ok(());
        }

        // 友人卡优先分配。
        //
        // ⚠ 实现选择，不是规格——官服真实分配顺序未知。它服务于已确认的约束
        // 「支援卡必出分身」：友人合法位常只剩一格，排在最后会被普通卡分身填满而饿死。
        // 拿到回放证据后只需改这一行排序。注意这只消除饿死，没把贪心变成最大匹配。
        // 稳定排序：普通卡之间保持原有相对顺序。
        card_indices.sort_by_key(|&i| !game.persons[i as usize].is_friend());

        // 分身分配走局部流，算法内部抽多少次都不位移同回合后续的策略随机
        // （训练成败 / 休息 / 外出）。注入 rule_master 时按 `(rule_master, turn, TAG)` 派生，
        // **完全不消耗父流**，且与本回合此前消耗了几次无关；未注入时回退到从父流 fork
        // （消耗 1 次），保持旧路径的可复现性契约。
        let mut clone_rng = game
            .clone_stream(CLONE_SUPER_TAG)
            .unwrap_or_else(|| fork_local_stream(rng, CLONE_SUPER_TAG));

        for &person_idx in &card_indices {
            // 先过滤出合法位再抽，而不是「有放回随便抽、失败重试」：
            // 后者在只剩一格合法时有 (3/4)^8 ≈ 10% 的概率明明放得下却放弃。
            let legal: Vec<usize> = option_trains
                .iter()
                .filter_map(|&t| usize::try_from(t).ok())
                .filter(|&t| Self::can_place_clone(game, person_idx, t))
                .collect();

            match legal.choose(&mut clone_rng) {
                Some(&train) => Self::place_clone(game, person_idx, train, "超级拉面")?,
                None => {
                    // 规则上真的放不下（候选位各自被同卡 / 友人互斥 / 5 非 NPC 挡住）。
                    // 不返回 Err：分身放不下在规则上允许，中断育成会让整次 rollout 作废，
                    // 且只丢掉「无解局面」这一类盘面，给搜索样本引入系统性缺失偏差。
                    // 用 diag! 而非 log::warn!：规则层热路径的既有惯例是一律 diag!
                    // （不开 feature 时编译期删除），聚合式统计才走 warn!
                    // （见 `search/flat_search.rs` 的「本次搜索共 N 次 rollout 失败」）。
                    // 若将来要盯这件事的频率，应加到 bench 的聚合统计里，而不是逐次打日志。
                    diag!(
                        ">> 超级拉面分身无处可放: {} (候选训练位 {:?})",
                        game.persons[person_idx as usize].short_name(),
                        option_trains
                    );
                }
            }
        }

        Ok(())
    }

    /// 判定能否在 `train` 为 `person_idx` 放置一个分身
    ///
    /// 纯判定：不修改局面、不消耗随机数。两条分身路径（超级拉面 / 地区拉面）共用，
    /// 避免各写一份满员与挤 NPC 规则后逐渐漂移。
    ///
    /// 注意「已满 5 人但含 NPC」算**可放置**——由 [`Self::place_clone`] 挤掉 NPC。
    pub(crate) fn can_place_clone(game: &super::RamenGame, person_idx: i32, train: usize) -> bool {
        let Ok(idx) = usize::try_from(person_idx) else {
            return false;
        };
        let Some(person) = game.persons.get(idx) else {
            return false;
        };
        if train >= 5 {
            return false;
        }
        let Some(dist) = game.base.distribution.get(train) else {
            return false;
        };

        // 该训练已有该人物的本体或分身
        if dist.contains(&person_idx) {
            return false;
        }

        // 每个训练只能出现一个友人，分身同样受限：
        // `distribute_person` 对本体维护了这条不变式，分身走本路径，必须同样把关。
        // `is_friend()` 覆盖剧本友人卡、理事长与记者。
        if person.is_friend()
            && dist.iter().any(|&id| id >= 0 && game.persons[id as usize].is_friend())
        {
            return false;
        }

        // 已有 5 个非 NPC：满员且挤不出空间
        //
        // 与下面的「满 5 人且无 NPC 可挤」在可达状态下互为冗余（长度不变式 <= 5），
        // 留着是防御该不变式将来被破坏。⚠ 单独删掉不会有任何测试变红，别当成有守护的分支。
        let non_npc_count = dist
            .iter()
            .filter(|&&id| id >= 0 && game.persons[id as usize].person_type != PersonType::Npc)
            .count();
        if non_npc_count >= 5 {
            return false;
        }

        // 已满 5 人时必须有 NPC 可挤
        if dist.len() >= 5 {
            return dist
                .iter()
                .any(|&id| id >= 0 && game.persons[id as usize].person_type == PersonType::Npc);
        }

        true
    }

    /// 在 `train` 放置 `person_idx` 的分身
    ///
    /// 满 5 人时挤掉一个 NPC。`source` 仅用于日志区分来源（`"超级拉面"` / `"地区"`）。
    ///
    /// # 错误
    ///
    /// 仅在 [`Self::can_place_clone`] 为假、或为真却仍无法写入时返回 `Err`——那是不变式
    /// 被破坏，属编程错误。**不用于**表达「这张卡放不下」，后者由调用方按规则处理。
    pub(crate) fn place_clone(
        game: &mut super::RamenGame, person_idx: i32, train: usize, source: &str
    ) -> Result<()> {
        if !Self::can_place_clone(game, person_idx, train) {
            return Err(anyhow!("{source}分身落点非法: 人头 {person_idx} -> 训练位 {train}"));
        }

        if game.base.distribution[train].len() >= 5 {
            // 已满 5 人：挤掉 NPC（`can_place_clone` 已保证有 NPC 可挤）
            let npc_pos = game.base.distribution[train]
                .iter()
                .position(|&id| id >= 0 && game.persons[id as usize].person_type == PersonType::Npc)
                .ok_or_else(|| {
                    anyhow!("{source}分身: {}训练已满5人且无NPC可挤", global!(GAMECONSTANTS).train_names[train])
                })?;
            let removed_id = game.base.distribution[train].remove(npc_pos);
            game.base.distribution[train].push(person_idx);
            diag!(
                ">> {}分身挤掉NPC: {} -> {}训练 (挤掉{})",
                source,
                game.persons[person_idx as usize].short_name(),
                global!(GAMECONSTANTS).train_names[train],
                game.persons[removed_id as usize].short_name()
            );
        } else {
            game.base.distribution[train].push(person_idx);
            diag!(
                ">> {}分身: {} -> {}训练",
                source,
                game.persons[person_idx as usize].short_name(),
                global!(GAMECONSTANTS).train_names[train]
            );
        }

        Ok(())
    }

    /// 阶段2：执行训练（含拉面效果叠加）
    ///
    /// 流程：
    /// 1. 计算基础参数（buffs、失败率、拉面效果）
    /// 2. 判定成功/失败
    /// 3. 成功时应用训练值和后续事件
    fn do_train(&self, game: &mut super::RamenGame, train: usize, rng: &mut impl Rng) -> Result<()> {
        if train >= 5 {
            return Err(anyhow!("训练类型越界: {train}"));
        }

        diag!(
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
    ///
    /// 调试模式（INFO 日志级别）下输出每个支援卡的 youqing 原始值/闪耀状态/最终
    /// 值，方便排查"友情加成不对"问题（详见 issues.md）。
    fn calc_train_params(&self, game: &super::RamenGame, train: usize) -> Result<TrainParams> {
        let buffs = game.calc_training_buff(train)?;
        let is_shining = game.shining_count(train) > 0;
        let ramen_effect = calc_ramen_training_effect(game, train, is_shining);

        // 基础失败率 + 拉面修正
        let base_failure_rate = game.calc_training_failure_rate(&buffs, train);
        let failure_rate = (base_failure_rate * (100.0 - ramen_effect.fail_rate_drop as f32) / 100.0)
            .min(100.0)
            .max(0.0);

        // ========== 训练效果分解（暂时屏蔽，需要时删除下方 /* */ 块注释恢复）==========
        /*
        // 详细日志：训练参数逐项分解
        let train_name = global!(GAMECONSTANTS).train_names[train].clone();
        let shining_count = game.shining_count(train);
        diag!(
            "\n===== 训练参数分解 {}训练 =====\n\
             distribution={:?} shining_count={} is_shining={}",
            train_name,
            game.distribution[train],
            shining_count,
            is_shining,
        );

        // 逐个支援卡：原始 youqing vs 清零后的 youqing
        for &pidx in game.distribution[train].iter() {
            if pidx < 0 {
                continue;
            }
            let person = &game.persons[pidx as usize];
            let shining_at = game.is_shining_at(pidx as usize, train);
            if pidx < 6 {
                // 支援卡：重新计算卡效果原始值（不应用 youqing 清零），方便对比
                let (raw_effect, _) = game.deck[pidx as usize].calc_training_effect(game, train as i32)?;
                let raw_youqing = raw_effect.youqing;
                let final_youqing = if shining_at { raw_youqing } else { 0.0 };
                diag!(
                    "  [#{pidx}] {} 类型={} 羁绊={} shining_at={} raw_youqing={:.1} final_youqing={:.1} raw_xunlian={}",
                    person.short_name(),
                    person.train_type,
                    person.friendship,
                    shining_at,
                    raw_youqing,
                    final_youqing,
                    raw_effect.xunlian,
                );
            } else {
                // NPC / 记者 / 友人 等
                diag!(
                    "  [#{pidx}] {} 类型={:?} 羁绊={} shining_at={} (非支援卡，不贡献 youqing)",
                    person.short_name(),
                    person.person_type,
                    person.friendship,
                    shining_at,
                );
            }
        }

        diag!(
            "  → buffs 汇总: youqing={:.1} xunlian={} failure_rate_drop={:.1} ganjing={} deyilv={:.1}",
            buffs.youqing,
            buffs.xunlian,
            buffs.fail_rate_drop,
            buffs.ganjing,
            buffs.deyilv,
        );
        diag!(
            "  → ramen_effect: xunlian={} youqing={} deyilv={} hint={} fail_rate_drop={} pt_bonus={} status_limit={} friendship={}",
            ramen_effect.xunlian,
            ramen_effect.youqing,
            ramen_effect.deyilv,
            ramen_effect.hint,
            ramen_effect.fail_rate_drop,
            ramen_effect.pt_bonus,
            ramen_effect.status_limit,
            ramen_effect.friendship,
        );
        diag!(
            "  → failure_rate: base={:.2}% final={:.2}%\n",
            base_failure_rate,
            failure_rate,
        );
        */

        Ok(TrainParams {
            buffs,
            is_shining,
            failure_rate
        })
    }

    /// 处理训练失败
    fn handle_train_failure(&self, game: &mut super::RamenGame, failure_rate: f32, rng: &mut impl Rng) -> Result<()> {
        // 再判断一次，如果还失败就是大失败
        if rng.random_bool(failure_rate as f64 / 100.0) {
            diag!("训练大失败!");
            game.apply_event(system_event("training_fail_low")?, 0, rng)?;
            game.uma.flags.ill = true;
            game.uma.flags.bad_trainer = true;
        } else {
            diag!("训练失败!");
            game.apply_event(system_event("training_fail")?, 0, rng)?;
        }
        Ok(())
    }

    /// 处理训练成功
    fn handle_train_success(
        &self, game: &mut super::RamenGame, train: usize, params: &TrainParams, rng: &mut impl Rng
    ) -> Result<()> {
        // calc_training_value 内部已两阶段计算（含拉面 buff），直接使用结果
        let final_value = game.calc_training_value(&params.buffs, train)?;
        game.uma.add_value(&final_value);

        // 增加训练次数
        game.base.train_level_count[train] += 1;

        // 处理羁绊和后续事件
        self.handle_post_train(game, train, rng)?;

        // 诀窍槽填充（含夏合宿特殊判定）
        self.fill_feeling_gauge(game, train, params, game.is_xiahesu())?;

        Ok(())
    }

    /// 处理训练后的羁绊和事件
    fn handle_post_train(&self, game: &mut super::RamenGame, train: usize, rng: &mut impl Rng) -> Result<()> {
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
        self.handle_hint_event(game, train, &hint_persons, rng)?;

        // 额外训练事件（非合宿）
        let extra_train_prob = system_event_prob("extra_train")?;
        if !game.is_xiahesu() && rng.random_bool(extra_train_prob as f64) {
            let mut event = EventData::extra_training_event(train);
            // 动态设置理事长索引
            if let Some(yayoi_index) = game.persons.iter().position(|p| p.person_type == PersonType::Yayoi) {
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
    ///
    /// 行为：
    /// - 当 hint_special 生效（且 train 在当前回合 at_trains 中）时：依次触发 hint_persons 中
    ///   所有 PersonType::Card 的 hint 事件，每个支援卡触发 `1 + hint_count_bonus` 次
    /// - 否则：从 hint_persons 中随机选一个触发 `1 + hint_count_bonus` 次（保留温泉杯逻辑）
    fn handle_hint_event(
        &self, game: &mut super::RamenGame, train: usize, hint_persons: &[i32], rng: &mut impl Rng
    ) -> Result<()> {
        if hint_persons.is_empty() {
            return Ok(());
        }
        // 判断 hint_special 是否对当前 train 生效
        let hint_special_active = game.is_hint_special_active_for_train(train);
        if hint_special_active {
            // 依次触发 hint_persons 中所有 PersonType::Card 的 hint 事件
            for &p in hint_persons {
                if p < 0 || p as usize >= game.persons.len() {
                    continue;
                }
                let person_index = p as usize;
                if game.persons[person_index].person_type != PersonType::Card {
                    continue;
                }
                // 人头下标 ≠ 卡组下标，取 hint_count_bonus 前先反查
                let hint_count = 1 + game.deck_index_of(person_index).map_or(0, |di| game.deck[di].effect.hint_count_bonus);
                for _ in 0..hint_count {
                    self.push_hint_event(game, person_index, rng)?;
                }
            }
        } else if let Some(&p) = hint_persons.choose(rng) {
            if p < 0 || p as usize >= game.persons.len() {
                return Ok(());
            }
            let person_index = p as usize;
            // 人头下标 ≠ 卡组下标，取 hint_count_bonus 前先反查
            let hint_count = 1 + game.deck_index_of(person_index).map_or(0, |di| game.deck[di].effect.hint_count_bonus);
            for _ in 0..hint_count {
                self.push_hint_event(game, person_index, rng)?;
            }
        }
        Ok(())
    }

    /// 推送一个 Hint 事件到 unresolved_events
    ///
    /// 支援卡根据 hint_level / total_hints 上限决定属性事件还是技能事件；
    /// 非支援卡统一按 hint_level=1 处理。
    fn push_hint_event(&self, game: &mut super::RamenGame, person_index: usize, rng: &mut impl Rng) -> Result<()> {
        let attr_prob = system_event_prob("hint_attr")?;
        let max_hint = global!(GAMECONSTANTS).max_hint_per_card;
        // 人头下标 ≠ 卡组下标：反查得到卡组槽位才走支援卡分支，
        // 无卡人头（理事长 / 记者 / NPC）按 hint_level=1 处理，且名字取人头自己的
        if let Some(di) = game.deck_index_of(person_index) {
            // 支援卡 hint 等级上限：超过则只触发属性事件（不加技能）
            let hint_level = (1 + game.deck[di].card_value().hint_level)
                .min(5)
                .min(max_hint - game.deck[di].total_hints);
            let mut hint_event = if hint_level <= 0 || rng.random_bool(attr_prob) {
                EventData::hint_attr_event(game.persons[person_index].train_type as usize, person_index)?
            } else {
                game.deck[di].total_hints += hint_level;
                EventData::hint_skill_event(hint_level, person_index)
            };
            hint_event.name = format!("{} - {}", hint_event.name, game.deck[di].short_name());
            game.base.unresolved_events.push(hint_event);
        } else {
            let hint_level = 1;
            let mut hint_event = if rng.random_bool(attr_prob) {
                EventData::hint_attr_event(game.persons[person_index].train_type as usize, person_index)?
            } else {
                EventData::hint_skill_event(hint_level, person_index)
            };
            hint_event.name = format!("{} - {}", hint_event.name, game.persons[person_index].short_name());
            game.base.unresolved_events.push(hint_event);
        }
        Ok(())
    }

    /// 处理友人点击事件（使用拉面杯友人事件）
    fn handle_friend_click(&self, game: &mut super::RamenGame, _rng: &mut impl Rng) -> Result<()> {
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
    fn do_friend_outing(&self, game: &mut super::RamenGame) -> Result<()> {
        let ramen_data = global!(RAMENDATA);
        let mut which = 0;
        while which < 5 && game.friend.out_used[which] {
            which += 1;
        }
        if which < 5 {
            diag!(">> 友人出行 #{}", which + 1);
            let key = format!("outing{}", which + 1);
            let mut event = ramen_data.friend_events[&key].clone();
            event.person_index = Some(game.friend.person_index as i32);
            game.friend.out_used[which] = true;
            game.base.unresolved_events.push(event);

            // 友人出行后获得隐藏风味（新友人固定2个）
            let special = 2;
            game.ramen.special_feeling = (game.ramen.special_feeling + special).min(4);
            diag!(">> 隐藏风味 +{} (={})", special, game.ramen.special_feeling);
            Ok(())
        } else {
            Err(anyhow!("友人出行越界: {which}"))
        }
    }

    /// 填充诀窍槽
    fn fill_feeling_gauge(
        &self, game: &mut super::RamenGame, train: usize, params: &TrainParams, is_xiahesu: bool
    ) -> Result<()> {
        if let Some(train_feelings) = game.ramen.train_feeling_type {
            let base_dist = super::rules::calc_gauge_base_distribution(&game.ramen.selected_regions);
            // 支援卡数量：仅统计类型为 Card 的 person（分身是同一索引在分布中重复出现，自然计入）。
            // 不得用固定索引排除（旧布局 p!=6&&p!=7 会把 NPC/理事长/记者误算进来）。
            let support_count = game.distribution[train]
                .iter()
                .filter(|&&p| p >= 0 && game.persons[p as usize].person_type == PersonType::Card)
                .count();
            // NPC 数量 = 本训练位置实际分配的 Npc 人数（`ramen_memo_cn.md` 公式与算例；
            // 不是全局固定 5——NPC 随机分配且可能被分身挤掉，须按实际分布统计）
            let npc_count = game.distribution[train]
                .iter()
                .filter(|&&p| p >= 0 && game.persons[p as usize].person_type == PersonType::Npc)
                .count();
            let train_bonus = super::rules::calc_train_feeling_bonus(support_count, npc_count);
            fill_gauge_after_train(
                &mut game.ramen,
                &base_dist,
                train_feelings[train],
                train_bonus,
                params.is_shining,
                is_xiahesu
            );
            // 观测：友情训练回合数（纯采集，不影响逻辑）
            if params.is_shining {
                if let Some(slot) = game.ramen.yearly_friend_turns.get_mut(game.ramen.obs_year) {
                    *slot += 1;
                }
            }
        }
        Ok(())
    }

    /// 填充诀窍槽（非训练动作：比赛/休息/外出/友人出行）
    ///
    /// 仅按基础值填充；夏合宿时三种槽直接全 MAX。
    /// 复用 `calc_gauge_base_distribution` 保持与训练分支一致的基础分配。
    fn fill_gauge_non_train(&self, game: &mut super::RamenGame, is_xiahesu: bool) -> Result<()> {
        let base_dist = super::rules::calc_gauge_base_distribution(&game.ramen.selected_regions);
        fill_gauge_after_non_train(&mut game.ramen, &base_dist, is_xiahesu);
        Ok(())
    }
}

/// 训练参数（计算后缓存）
struct TrainParams {
    /// 支援卡 Buff
    buffs: crate::game::CardTrainingEffect,
    /// 是否友情训练
    is_shining: bool,
    /// 失败率（百分比）
    failure_rate: f32
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
/// # 参数
/// - `can_friend_outing`: 是否可以选择友人出行（基础判定，未叠加夏令营限制）
/// - `is_ill`: 是否生病
/// - `is_xiahesu`: 是否处于夏合宿回合
///
/// 夏合宿禁用普通外出/友人出行/治病（与 `BasicGame::list_actions` 一致，休息自动治病）。
/// 回合 0-12 不允许自选比赛（`can_race = turn > 12`，回合 11 为出道赛、回合 12 无可用比赛）。
pub fn list_operations(can_friend_outing: bool, is_ill: bool, is_xiahesu: bool, can_race: bool) -> Vec<Operation> {
    let mut ops = vec![
        Operation::Train(TrainingType::Speed),
        Operation::Train(TrainingType::Stamina),
        Operation::Train(TrainingType::Power),
        Operation::Train(TrainingType::Guts),
        Operation::Train(TrainingType::Wisdom),
    ];
    if can_race {
        ops.push(Operation::Race);
    }
    ops.push(Operation::Rest);
    if !is_xiahesu {
        ops.push(Operation::NormalOuting);
    }
    if can_friend_outing && !is_xiahesu {
        ops.push(Operation::FriendOuting);
    }
    if is_ill && !is_xiahesu {
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
/// - `can_friend_outing`: 是否可以选择友人出行（基础判定，未叠加夏令营限制）
/// - `is_ill`: 是否生病
/// - `is_xiahesu`: 是否处于夏合宿回合
/// - `can_race`: 是否允许比赛（回合 0-12 为 false，回合 13 起为 true）
pub fn list_all_actions(
    available_ramens: &[usize], can_friend_outing: bool, is_ill: bool, is_xiahesu: bool, can_race: bool
) -> Vec<RamenAction> {
    let ramen_choices = list_ramen_choices(available_ramens);
    let operations = list_operations(can_friend_outing, is_ill, is_xiahesu, can_race);

    let mut actions = Vec::new();
    for ramen in &ramen_choices {
        for &op in &operations {
            match ramen {
                None => actions.push(RamenAction::no_ramen(op)),
                Some(idx) => actions.push(RamenAction::with_ramen(*idx, op))
            }
        }
    }
    actions
}

/// 写入 live `selected_regions` 并按显式年份归档。
///
/// 归档下标见 [`super::RamenState::region_archive_year_idx`]：必须按回合硬编码，
/// **不能**用 `current_year()`。turn 23 仍属第一年，但选的是第二年地区。
fn apply_region_selection(game: &mut super::RamenGame, regions: [usize; 3]) -> Result<()> {
    game.ramen.selected_regions = regions;
    let year_idx = super::RamenState::region_archive_year_idx(game.base.turn)?;
    game.ramen.archive_selected_regions(year_idx, regions)?;
    diag!("地区选择: {:?} (第 {} 年，手写逻辑)", regions, year_idx + 1);
    Ok(())
}

// ========== 三阶段决策候选生成 ==========

/// `RamenSelect` 阶段的候选动作：不吃 + 候选面（`selected_regions` 中可做面）
///
/// 所有候选动作的 `operation = Operation::StageOnly`，`special_targets = None`。
pub fn list_ramen_select_actions(state: &super::RamenState, selected_regions: &[usize; 3]) -> Vec<RamenAction> {
    use super::rules::list_special_targets_for;

    let mut actions = vec![RamenAction::ramen_select(None)]; // 不吃面
    for &region_id in selected_regions {
        // 用隐藏风味可达（候选非空）即可选
        let ok = !list_special_targets_for(state, region_id)
            .map(|t| t.is_empty())
            .unwrap_or(true);
        if ok {
            actions.push(RamenAction::ramen_select(Some(region_id)));
        }
    }
    actions
}

/// `SpecialSelect` 阶段的候选动作：`list_special_targets_for` 生成的每个 targets 一个
///
/// 所有候选动作的 `ramen = Some(ramen_idx)`、`operation = Operation::StageOnly`。
pub fn list_special_select_actions(state: &super::RamenState, ramen_idx: usize) -> anyhow::Result<Vec<RamenAction>> {
    use super::rules::list_special_targets_for;

    let targets = list_special_targets_for(state, ramen_idx)?;
    Ok(targets
        .into_iter()
        .map(|t| RamenAction::special_select(ramen_idx, t))
        .collect())
}

/// `SuperRamenSelect` 阶段的候选动作：每个 `training_limit_options` 下标一个。
///
/// 数据源是纯位置下标，没有 option ID。所有候选的 `ramen` / `special_targets`
/// 均为空，`operation = SuperRamenSelect(idx)`。
pub fn list_super_ramen_select_actions() -> Result<Vec<RamenAction>> {
    let n = super::rules::get_super_ramen_clone_train_options()?.len();
    Ok((0..n).map(RamenAction::super_ramen_select).collect())
}

/// `RegionSelect` 阶段的候选动作
///
/// - 第 1/2 年：枚举该年全部组合（C(5,3)=10）
/// - 第 3 年 `ramen_region_strategy=all`：C(10,3)=120
/// - 第 3 年 `ramen_region_strategy=fixed`：**单候选直达**，不展开 120 组合
///
/// 年份下标见 [`super::RamenState::region_archive_year_idx`]（turn 2/23/47 → 0/1/2）。
pub fn list_region_select_actions(turn: i32) -> Result<Vec<RamenAction>> {
    let year_idx = super::RamenState::region_archive_year_idx(turn)?;
    let cfg = global!(GAMECONFIG);
    let combos = region_select_combos(year_idx, cfg.ramen_region_strategy, cfg.ramen_region_fixed.as_deref())?;
    Ok(combos
        .into_iter()
        .map(|c| RamenAction::no_ramen(Operation::RegionSelect(c)))
        .collect())
}

/// 地区候选组合的**纯函数**实现：策略与 fixed 表由调用方显式传入
///
/// 与 [`list_region_select_actions`] 拆开的理由是**可测性**：
/// `init_global_with_config` 是幂等的（globals 已初始化时直接返回 `Ok(())` 并
/// 丢弃传入 config），测试进程里只要有任何一条用例先调过 `init_global()`，
/// 后续想设置非默认策略的用例就会**静默地对着默认配置跑且不报错**。
/// 因此策略分支必须能脱离全局配置单独验证，否则守门测试是空转的。
///
/// 只有 `get_region_combinations` 仍读 `RAMENDATA`——那是静态数据不是可调配置。
pub fn region_select_combos(
    year_idx: usize, strategy: RamenRegionStrategy, fixed: Option<&[[usize; 3]]>
) -> Result<Vec<[usize; 3]>> {
    // fixed 仅第 3 年生效；第 1/2 年一律走全枚举
    if year_idx == 2 && matches!(strategy, RamenRegionStrategy::Fixed) {
        let fixed = fixed.ok_or_else(|| {
            anyhow!("ramen_region_strategy=fixed 但未设置 ramen_region_fixed（仅第3年需要，长度 = 1）")
        })?;
        let first = fixed
            .first()
            .ok_or_else(|| anyhow!("ramen_region_fixed 长度必须 = 1（仅第3年）"))?;
        return Ok(vec![*first]);
    }
    super::rules::get_region_combinations(year_idx)
}

/// `Train` 阶段的候选动作：所有 Operation ×（带 pending 的完整动作）
///
/// **重构后**：Train 阶段只承载基础操作（训练/比赛/休息等），不再带 `ramen` 和
/// `special_targets` 字段——这两个字段已在 `SpecialSelect → Train` 过渡由
/// [`RamenGame::ground_ramen_effects`] 落地，玩家在选训练前已看到完整 buff 和 distribution。
///
/// 三阶段流程：RamenSelect（`list_ramen_select_actions`）→ SpecialSelect
/// （`list_special_select_actions`）→ **过渡**（`ground_ramen_effects`）→ Train（本函数）。
///
/// # 参数
/// - `can_race`: 是否允许比赛（回合 0-12 为 false，回合 13 起为 true）
pub fn list_train_actions(can_friend_outing: bool, is_ill: bool, is_xiahesu: bool, can_race: bool) -> Vec<RamenAction> {
    let operations = list_operations(can_friend_outing, is_ill, is_xiahesu, can_race);
    operations.into_iter().map(RamenAction::new).collect()
}

/// 合并决策阶段的候选动作：不吃面 + 每个面 × `list_special_targets_for` 候选 targets
///
/// 返回所有 (ramen_idx, special_targets) 笛卡尔积的 `RamenAction`，每个的 `operation = StageOnly`。
///
/// 适用场景：在线搜索/未来 MctsTrainer 等需要"选面+选吃法"一次性决策的 Trainer。
/// 调用方使用本函数生成候选后，应通过 `RamenGame::apply_combined_ramen_decision`
/// （而非 `apply_action`）应用选中的合并决策。
///
/// 与三阶段路径的关系：
/// - 标准三阶段（HandwrittenTrainer 等）：RamenSelect → SpecialSelect → Train，分别调用
///   `list_ramen_select_actions` / `list_special_select_actions` / `list_train_actions`
/// - 合并路径（本函数）：RamenSelect 直接列出 ramen × targets 笛卡尔积，一次决策
///
/// 候选数估算：1（不吃）+ Σ 各面 `list_special_targets_for` 长度。
/// 库存紧张时每个面仅 1~6 种，全富余时 6~9 种。
/// 峰值在「全富余库存 + `special_feeling = 4`」下，按 `REGION_RANGES` 每年 C(n,3)
/// 穷举实测为 **28**（年 1 = 27 / 年 2 = 24 / 年 3 = 28；含不吃面）。
pub fn list_combined_ramen_select_actions(
    state: &super::RamenState, selected_regions: &[usize; 3]
) -> Vec<RamenAction> {
    use super::rules::list_special_targets_for;

    let mut actions = vec![RamenAction::combined_select(None, [0, 0, 0])]; // 不吃面
    for &region_id in selected_regions {
        // 任一 targets 候选非空即该面可选；与 `get_available_ramens` 判定一致
        let targets_vec = list_special_targets_for(state, region_id).unwrap_or_default();
        for t in targets_vec {
            actions.push(RamenAction::combined_select(Some(region_id), t));
        }
    }
    actions
}

/// 获取当年可用的面（存在合法 `special_targets` 即可选）。
///
/// 与之前用 `can_make_ramen(recipe, &[0,0,0])` 过滤不同：本函数委托给
/// [`super::rules::list_special_targets_for`]，允许"普通诀窍不够、用隐藏风味补缺口"的面
/// 也算作可选。例如库存 A=5 B=0 C=0、recipe=[2,2,1] 时，用 1 个隐藏风味替代 B 仍可做面。
///
/// 返回可以吃的面的 ID 列表。
pub fn get_available_ramens(state: &super::RamenState, selected_regions: &[usize; 3]) -> Vec<usize> {
    use super::rules::list_special_targets_for;

    let mut available = Vec::new();
    for &region_id in selected_regions {
        if !list_special_targets_for(state, region_id)
            .map(|t| t.is_empty())
            .unwrap_or(true)
        {
            available.push(region_id);
        }
    }
    available
}

#[cfg(test)]
mod tests {
    use super::{super::RamenState, *};
    use crate::{
        gamedata::init_global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    #[test]
    fn test_ramen_action_display() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let a1 = RamenAction::no_ramen(Operation::Train(TrainingType::Speed));
        println!("动作1: {a1}");

        let a2 = RamenAction::with_ramen(0, Operation::Train(TrainingType::Wisdom));
        println!("动作2: {a2}");

        let a3 = RamenAction::no_ramen(Operation::Race);
        println!("动作3: {a3}");

        let a4 = RamenAction::with_ramen(5, Operation::Rest);
        println!("动作4: {a4}");

        // 新增：含 special_targets 的动作 Display 应输出 (替...)
        let mut a5 = RamenAction::with_ramen(0, Operation::Train(TrainingType::Speed));
        a5.special_targets = Some([1, 1, 0]);
        println!("动作5: {a5}");

        // 占位阶段动作：StageOnly + ramen=Some
        let a6 = RamenAction::ramen_select(Some(0));
        println!("动作6 (RamenSelect 选面): {a6}");

        // 占位阶段动作：StageOnly + ramen=None（不吃面）
        let a7 = RamenAction::ramen_select(None);
        println!("动作7 (RamenSelect 不吃面): {a7}");

        // 占位阶段动作：StageOnly + special_targets
        let a8 = RamenAction::special_select(0, [1, 0, 0]);
        println!("动作8 (SpecialSelect): {a8}");

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
        // 基础情况（允许比赛）
        let ops = list_operations(false, false, false, true);
        println!("基础操作: {} 个", ops.len());
        assert_eq!(ops.len(), 8); // 5训练+比赛+休息+普通外出

        // 有友人出行和治病（允许比赛）
        let ops = list_operations(true, true, false, true);
        println!("有友人+治病: {} 个", ops.len());
        assert_eq!(ops.len(), 10); // 5训练+比赛+休息+普通外出+友人出行+治病

        // 夏合宿：禁用普通外出、友人出行、治病（允许比赛）
        let ops = list_operations(false, false, true, true);
        println!("夏合宿 无友人无生病: {} 个", ops.len());
        assert_eq!(ops.len(), 7); // 5训练+比赛+休息

        // 夏合宿 + 有友人 + 生病：仍只有 7 个（允许比赛）
        let ops = list_operations(true, true, true, true);
        println!("夏合宿 有友人+生病: {} 个", ops.len());
        assert_eq!(ops.len(), 7);

        // 回合 0-12 不允许比赛：少了"比赛"操作，少 1 个
        let ops = list_operations(false, false, false, false);
        println!("基础操作(禁赛): {} 个", ops.len());
        assert_eq!(ops.len(), 7); // 5训练+休息+普通外出
    }

    #[test]
    fn test_list_all_actions() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 无可用面，无友人，无生病，非夏合宿（允许比赛）
        let actions = list_all_actions(&[], false, false, false, true);
        println!("无可用面: {} 个动作", actions.len());
        // 1吃面选择 * 8操作 = 8
        assert_eq!(actions.len(), 8);

        // 有3种可用面（允许比赛）
        let actions = list_all_actions(&[0, 1, 2], false, false, false, true);
        println!("有3种可用面: {} 个动作", actions.len());
        // 4吃面选择 * 8操作 = 32
        assert_eq!(actions.len(), 32);

        // 有友人出行和治病，3种可用面（允许比赛）
        let actions = list_all_actions(&[0, 1, 2], true, true, false, true);
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
        let _ = init_test_logger("info");
        let _ = init_global();

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

        // 关键：用隐藏风味可达（库存紧但 hidden 够）应视为可选
        // 札幌 [2,2,1]，库存 A=5 B=0 C=0，special=1：min_needed=[0,2,0]，need_sum=2 = budget → 空（缺 2 不可用 1 个 hidden）
        // 改为库存 A=5 B=0 C=5, special=1: min_needed=[0,2,0], need_sum=2 > budget=1 → 空
        // 库存 A=5 B=1 C=5, special=2: min_needed=[0,1,0], need_sum=1, budget=2-1=1 → [0,1,0]（含 1 个 B 替换）
        let mut state = RamenState::default();
        state.feeling_stock = [5, 1, 5];
        state.special_feeling = 2;
        let available = get_available_ramens(&state, &selected_regions);
        println!("隐藏风味补 B 缺口: 可用面={available:?}");
        // 札幌 [2,2,1] 缺 B=1，用 1 个 hidden 补 → 可选
        assert!(available.contains(&0));

        Ok(())
    }

    // ========== 三阶段决策候选生成测试 ==========

    #[test]
    fn test_list_ramen_select_actions_full() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5]; // 全够
        state.special_feeling = 2;

        let actions = list_ramen_select_actions(&state, &[0, 1, 2]);
        println!("全富余 3 个面: {} 个动作", actions.len());
        // 不吃 + 3 面 = 4 个
        assert_eq!(actions.len(), 4);
        // 第一个动作一定是不吃面
        assert_eq!(actions[0].ramen, None);
        assert!(matches!(actions[0].operation, Operation::StageOnly));
        assert_eq!(actions[0].special_targets, None);

        // 其他动作各对应一个面
        for (i, a) in actions.iter().enumerate().skip(1) {
            assert_eq!(a.ramen, Some(i - 1 + 0)); // 0,1,2
            assert!(matches!(a.operation, Operation::StageOnly));
        }

        Ok(())
    }

    #[test]
    fn test_list_ramen_select_actions_no_available() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let state = RamenState::default(); // 库存全 0
        let actions = list_ramen_select_actions(&state, &[0, 1, 2]);
        println!("全空: {} 个动作", actions.len());
        // 仅"不吃面"一个候选
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].ramen, None);

        Ok(())
    }

    #[test]
    fn test_list_special_select_actions_uses_special_targets() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut state = RamenState::default();
        state.feeling_stock = [2, 5, 5]; // 札幌 [2,2,1] 缺 A=0 不缺，stock 够
        state.special_feeling = 2;

        let actions = list_special_select_actions(&state, 0)?;
        println!("札幌全富余 special=2: {actions:?}");
        // 9 种：1 (sum=0) + 3 (sum=1) + 5 (sum=2) = 9
        assert_eq!(actions.len(), 9);
        // 所有动作 ramen=Some(0)、operation=StageOnly
        for a in &actions {
            assert_eq!(a.ramen, Some(0));
            assert!(matches!(a.operation, Operation::StageOnly));
        }
        // 第一个应是 [0,0,0]（sum 升序）
        assert_eq!(actions[0].special_targets, Some([0, 0, 0]));

        Ok(())
    }

    #[test]
    fn test_list_train_actions_no_ramen_field() -> anyhow::Result<()> {
        // 重构后：Train 阶段动作不再带 ramen / special_targets（已由 ground_ramen_effects 落地）

        // 不吃面/吃面/夏合宿参数都不影响 candidates 数量和字段（允许比赛）
        let actions = list_train_actions(false, false, false, true);
        println!("Train 阶段候选: {actions:#?}");
        // 8 个 operation
        assert_eq!(actions.len(), 8);
        // 每个动作 ramen=None、special_targets=None（不再有 pending 字段）
        for a in &actions {
            assert_eq!(a.ramen, None);
            assert_eq!(a.special_targets, None);
        }

        // 有友人 + 治病（允许比赛）
        let actions = list_train_actions(true, true, false, true);
        assert_eq!(actions.len(), 10);

        // 夏合宿：禁用普通外出/友人/治病（允许比赛）
        let actions = list_train_actions(true, true, true, true);
        assert_eq!(actions.len(), 7);

        Ok(())
    }

    // ========== 合并决策候选生成测试 ==========

    #[test]
    fn test_combined_select_normalizes_targets_when_no_ramen() {
        // 不吃面时 targets 强制 [0,0,0]
        let a = RamenAction::combined_select(None, [1, 2, 3]);
        println!("不吃面 + 任意 targets: {a}");
        assert_eq!(a.ramen, None);
        assert_eq!(a.special_targets, Some([0, 0, 0]));
        assert!(matches!(a.operation, Operation::StageOnly));
    }

    #[test]
    fn test_combined_select_keeps_targets_when_eating() {
        // 吃面时 targets 保留
        let a = RamenAction::combined_select(Some(5), [1, 0, 1]);
        println!("吃面5 + targets=[1,0,1]: {a}");
        assert_eq!(a.ramen, Some(5));
        assert_eq!(a.special_targets, Some([1, 0, 1]));
        assert!(matches!(a.operation, Operation::StageOnly));
    }

    #[test]
    fn test_list_combined_ramen_select_actions_full() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5]; // 全够
        state.special_feeling = 2;

        let actions = list_combined_ramen_select_actions(&state, &[0, 1, 2]);
        println!("合并决策候选 (3面全富余): {} 个", actions.len());
        // 不吃面 1 + 札幌(2,2,1) 9 + 函馆(1,2,2) 9 + 新潟(3,1,1) 8 = 27
        // （新潟的 t_b/t_c 上限为 1，max sum=2 ≤ budget=2 时 [3,0,0] 已超 2 被排除）
        assert_eq!(actions.len(), 27);

        // 第一个一定是不吃面
        assert_eq!(actions[0].ramen, None);
        assert_eq!(actions[0].special_targets, Some([0, 0, 0]));

        // 不吃面动作的唯一性
        let no_ramen_count = actions.iter().filter(|a| a.ramen.is_none()).count();
        assert_eq!(no_ramen_count, 1);

        // 各面 targets 数：札幌 9、函馆 9、新潟 8
        let expected_per = [9usize, 9, 8];
        for (region, &expected) in [0usize, 1, 2].iter().zip(expected_per.iter()) {
            let count = actions.iter().filter(|a| a.ramen == Some(*region)).count();
            println!("面 {region} 候选数: {count}");
            assert_eq!(count, expected, "面 {region} 候选数应 = {expected}");
        }

        Ok(())
    }

    #[test]
    fn test_list_combined_ramen_select_actions_no_available() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        // 全空 + 无隐藏风味：所有面都不可做
        let state = RamenState::default();
        let actions = list_combined_ramen_select_actions(&state, &[0, 1, 2]);
        println!("全空候选: {actions:?}");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].ramen, None);

        Ok(())
    }

    /// 诀窍槽加成按**训练位置实际分配的 NPC 数**生效（`ramen_memo_cn.md` 公式与算例）
    ///
    /// 修复回归：旧实现硬编码 npc_count=5 或按固定索引过滤（`p != 6 && p != 7`），
    /// 与实际分布不符。本测试构造不同 NPC 数的速训练分布，直接调用
    /// `handle_train_success`（绕过失败率），验证槽增量
    /// = 基础分配 + (1 + 支援卡数 + floor(NPC数/2))，与 game.rs 显示层一致。
    #[test]
    fn test_train_gauge_uses_actual_npc_count() -> anyhow::Result<()> {
        use rand::{SeedableRng, rngs::StdRng};

        use crate::{
            game::{
                ramen::{FeelingType, RamenGame, RamenStage, rules::calc_gauge_base_distribution},
                traits::Game
            },
            gamedata::init_global,
            utils::{get_workspace_root, init_test_logger}
        };

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(&workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        // 第 2 回合后（已有友人卡 + 5 个 NPC：persons[0..5]=支援卡, [6]=友人, [7..12]=NPC）
        game.add_friend_and_npcs()?;
        game.ramen.selected_regions = [0, 6, 7];
        game.ramen.train_feeling_type = Some([
            FeelingType::A,
            FeelingType::B,
            FeelingType::C,
            FeelingType::A,
            FeelingType::B
        ]);
        game.stage = RamenStage::Train;
        let _rng = StdRng::seed_from_u64(42);
        let action = RamenAction::new(Operation::Train(TrainingType::Speed));
        let base_dist = calc_gauge_base_distribution(&game.ramen.selected_regions);
        let gauge_limit = crate::game::ramen::rules::GAUGE_LIMIT;
        println!("基础分配 base_dist = {base_dist:?}（槽上限 {gauge_limit}，超出会清零+1诀窍）");

        // 场景 A：0 张支援卡 + 2 个 NPC → 加成 1+0+floor(2/2)=2（不溢出，可精确验证）
        game.base.distribution = vec![
            vec![7, 8], // 速：0支援卡 + 2 NPC
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let params = TrainParams {
            buffs: game.calc_training_buff(0)?,
            is_shining: false,
            failure_rate: 0.0
        };
        game.ramen.feeling_slot = [0, 0, 0];
        action.fill_feeling_gauge(&mut game, 0, &params, false)?;
        let gain = game.ramen.feeling_slot[0];
        let expect_bonus = 1 + 0 + 2 / 2; // 1 + 支援卡0 + floor(2/2)=1
        let expected_gain = base_dist[0] + expect_bonus;
        println!(
            "场景A: 速训练 0支援卡+2NPC, 槽A={gain}, 期望 {expected_gain} = 基础{} + 加成{expect_bonus}",
            base_dist[0]
        );
        assert_eq!(gain, expected_gain, "2 个 NPC 时应按 floor(2/2)=1 计算");

        // 场景 B：0 张支援卡 + 4 个 NPC → 加成 1+0+floor(4/2)=3
        // 同支援卡数仅 NPC 数翻倍：若旧实现硬编码 5（floor(5/2)=2）则两场景加成相同，
        // 实际应按本训练位置 4 个 NPC → floor(4/2)=2 有差异。
        game.base.distribution = vec![
            vec![7, 8, 9, 10], // 速：0支援卡 + 4 NPC
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let params = TrainParams {
            buffs: game.calc_training_buff(0)?,
            is_shining: false,
            failure_rate: 0.0
        };
        game.ramen.feeling_slot = [0, 0, 0];
        action.fill_feeling_gauge(&mut game, 0, &params, false)?;
        let gain = game.ramen.feeling_slot[0];
        let expect_bonus = 1 + 0 + 4 / 2; // 1 + 支援卡0 + floor(4/2)=2
        let expected_gain = base_dist[0] + expect_bonus;
        println!(
            "场景B: 速训练 0支援卡+4NPC, 槽A={gain}, 期望 {expected_gain} = 基础{} + 加成{expect_bonus}",
            base_dist[0]
        );
        assert_eq!(
            gain, expected_gain,
            "4 个 NPC 时应按 floor(4/2)=2 计算（而非硬编码 5 个 NPC 的 floor(5/2)=2 恰好巧合相同）"
        );

        Ok(())
    }

    /// 回归：超级拉面分身必须包含友人卡，且分身同样受「每训练一个友人」约束
    ///
    /// `distribute_super_ramen_clones` 的候选收集原先写作 `(0..6i32)`，
    /// 那是温泉布局的卡组范围。拉面的 `init_persons` 只把 5 张非友人卡建成人头
    /// （占 0-4）再追加理事长（5），友人卡回合 2 才加入、落在人头 6，
    /// 于是 `0..6` 永远取不到它——filter 里的 `|| ScenarioCard` 是一句死条件，
    /// 友人卡分身从未生成过。规格见 `ramen_memo_cn.md`「分身来源：包含友人卡」。
    ///
    /// 友人卡分身补上之后，`try_add_clone` 必须同样把住 `distribute_person`
    /// 对本体维护的「每个训练只能出现一个友人」，否则会出现友人卡分身与
    /// 理事长 / 记者同格这种自然分配产生不出的局面。
    ///
    /// 落位是均匀随机（超级拉面分身不计算得意率），故只验不变式、不固化具体
    /// 训练位；拒绝分支靠多种子遍历覆盖，避免单一种子恰好绕开。
    #[test]
    fn test_super_ramen_clones_include_friend_card() -> anyhow::Result<()> {
        use rand::{SeedableRng, rngs::StdRng};

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        // [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
        const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        // 假失败率在「只剩一格合法」时是 (3/4)^8 ≈ 10%，32 个种子期望只丢 3 次、
        // 且旧断言只查「不该有什么」，会整片漏过去。提到 256 让漏放必然现形。
        const SEEDS: u64 = 256;

        let mut game = super::super::RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.base.turn = 72; // 超级拉面回合（72-77）
        game.deck_can_split = true;
        game.ramen.super_ramen = Some(1); // 选项二：速/耐/力/智 = [0,1,2,4]，根训练不在范围

        let support_indices: Vec<i32> = game
            .persons
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.person_type == PersonType::Card || p.person_type == PersonType::ScenarioCard
            })
            .map(|(i, _)| i as i32)
            .collect();
        let find = |ty: PersonType| -> anyhow::Result<i32> {
            game.persons
                .iter()
                .position(|p| p.person_type == ty)
                .map(|i| i as i32)
                .ok_or_else(|| anyhow::anyhow!("找不到人头类型 {ty:?}"))
        };
        let friend = find(PersonType::ScenarioCard)?;
        let yayoi = find(PersonType::Yayoi)?;
        let allowed = super::super::rules::get_super_ramen_clone_train_options()?;
        let allowed = allowed.get(1).ok_or_else(|| anyhow::anyhow!("缺少超级拉面选项二"))?.clone();
        println!("支援卡人头 = {support_indices:?}，友人卡 = {friend}，理事长 = {yayoi}，允许训练位 = {allowed:?}");

        let count_of = |g: &super::super::RamenGame, idx: i32| -> usize {
            g.base.distribution.iter().flatten().filter(|&&p| p == idx).count()
        };
        // 每个训练位最多一个友人（含理事长、记者）
        let friend_ok = |g: &super::super::RamenGame| -> bool {
            g.base.distribution.iter().all(|d| {
                d.iter().filter(|&&p| p >= 0 && g.persons[p as usize].is_friend()).count() <= 1
            })
        };

        let mut c = Checks::new();
        c.check(support_indices.len() == 6, "测试卡组应有 5 张训练卡 + 1 张友人卡");

        // 场景 A：空分布起步，出现的每一次都必然是分身
        let mut a_ok = (true, true, true, true);
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![]; 5];
            RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(seed))?;
            if seed == 0 {
                println!("场景 A（空分布，seed 0）: {:?}", game.base.distribution);
            }
            a_ok.0 &= game.base.distribution.iter().flatten().count() == 6;
            a_ok.1 &= count_of(&game, friend) == 1;
            a_ok.2 &= friend_ok(&game);
            a_ok.3 &= game
                .base
                .distribution
                .iter()
                .enumerate()
                .all(|(t, d)| d.is_empty() || allowed.contains(&(t as i32)));
        }
        c.check(a_ok.0, "空分布下每张支援卡各生成 1 个分身，合计 6 个占位");
        c.check(a_ok.1, "友人卡应生成 1 个分身（修复前恒为 0）");
        c.check(a_ok.2, "任一训练位友人不超过 1 个");
        c.check(a_ok.3, "分身只落在选项允许的训练位");

        // 场景 B：友人卡本体已在速训练，分身必须另开一格
        let mut b_ok = (true, true);
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![friend], vec![], vec![], vec![], vec![]];
            RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(seed))?;
            b_ok.0 &= count_of(&game, friend) == 2;
            b_ok.1 &= game.base.distribution[0].iter().filter(|&&p| p == friend).count() == 1;
        }
        c.check(b_ok.0, "友人卡本体 + 分身合计 2 个占位");
        c.check(b_ok.1, "同一训练不能同时存在本体与分身");

        // 场景 C：理事长占住允许范围内的三个训练位，友人分身只能去剩下那个
        //
        // 这是「只剩一格合法」(|L| = 1) 的局面。旧实现有放回抽 8 次，约 10% 的种子
        // 明明放得下却放弃；而旧断言只查「不得与理事长同格」「0/1/2 不出现友人」——
        // 友人分身**根本没被放出来**时这两条同样成立，测试照绿。
        // 下面第三条才是真正的守卫：必须**确实放出来**。
        let mut c_ok = (true, true, true);
        let mut c_lost = 0usize;
        for seed in 0..SEEDS {
            game.base.distribution = vec![vec![yayoi], vec![yayoi], vec![yayoi], vec![], vec![]];
            RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(seed))?;
            if seed == 0 {
                println!("场景 C（理事长占 0/1/2，seed 0）: {:?}", game.base.distribution);
            }
            c_ok.0 &= friend_ok(&game);
            c_ok.1 &= (0..3).all(|t| !game.base.distribution[t].contains(&friend));
            let placed = count_of(&game, friend) == 1;
            if !placed {
                c_lost += 1;
            }
            c_ok.2 &= placed;
        }
        println!("场景 C: {SEEDS} 个种子中友人分身漏放 {c_lost} 次（修复前期望约 10%）");
        c.check(c_ok.0, "友人卡分身不得与理事长同格");
        c.check(c_ok.1, "理事长占住的训练位不应出现友人卡分身");
        c.check(c_ok.2, "只剩一格合法时友人卡分身必须放出来（旧实现约 10% 漏放）");

        // 场景 D：非超级拉面回合应原样早退
        game.base.turn = 71;
        game.base.distribution = vec![vec![]; 5];
        RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(42))?;
        c.check(
            game.base.distribution.iter().flatten().count() == 0,
            "非超级拉面回合不应生成任何分身"
        );

        c.finish()
    }
    /// 回归：友人卡必须优先分配，否则普通卡会抢走它唯一的容量
    ///
    /// 分配是**在线贪心**，前面的卡改写后面的卡看到的合法集，而友人卡人头下标最大、
    /// 永远最后处理。本用例的盘面下友人分身只有智可去，训练卡分身先占满智就把友人饿死。
    /// 所以「洗牌 / 过滤合法位」只能消灭随机重试的假失败，消灭不了这个顺序问题。
    #[test]
    fn test_super_ramen_clones_friend_priority_beats_greedy_starvation() -> anyhow::Result<()> {
        use rand::{SeedableRng, rngs::StdRng};

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        const SEEDS: u64 = 256;

        let mut game = super::super::RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.add_reporter();
        game.base.turn = 72;
        game.ramen.super_ramen = Some(1); // 选项二：允许 [0,1,2,4]
        game.deck_can_split = true;

        let friend = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::ScenarioCard)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到友人卡"))?;
        let yayoi = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Yayoi)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到理事长"))?;
        let reporter = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Reporter)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到记者"))?;
        let cards: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| game.persons[i as usize].person_type == PersonType::Card)
            .collect();
        println!("友人卡={friend} 理事长={yayoi} 记者={reporter} 训练卡={cards:?}");

        let mut c = Checks::new();
        c.check(cards.len() == 5, "测试卡组应有 5 张训练卡");

        // 可解局面：友人优先后六张卡必须全部放下
        let mut all_placed = true;
        let mut friend_placed = true;
        let mut min_clones = usize::MAX;
        for seed in 0..SEEDS {
            game.base.distribution =
                vec![vec![friend], vec![yayoi], vec![reporter], vec![], vec![cards[0], cards[1]]];
            let before: usize = game.base.distribution.iter().flatten().count();
            RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(seed))?;
            let clones: usize = game.base.distribution.iter().flatten().count() - before;
            min_clones = min_clones.min(clones);
            all_placed &= clones == 6;
            friend_placed &=
                game.base.distribution.iter().flatten().filter(|&&p| p == friend).count() == 2;
            if seed == 0 {
                println!("可解反例（seed 0）: {:?}", game.base.distribution);
            }
        }
        println!("可解反例：{SEEDS} 个种子中最少放下 {min_clones} 个分身（应恒为 6）");
        c.check(all_placed, "可解局面下六张支援卡必须各放下一个分身");
        c.check(friend_placed, "友人卡本体 + 分身合计 2 个占位");

        // 真无解局面：智已被 5 张训练卡本体占满，另三个候选位各有一个友人。
        // 友人分身无处可去——这是约束本身的死局，任何分配算法都救不了，不是回归。
        // 此时仍必须：不 panic、不返回 Err、5 张训练卡照常拿到分身。
        let mut dead_ok = (true, true);
        for seed in 0..SEEDS {
            game.base.distribution =
                vec![vec![friend], vec![yayoi], vec![reporter], vec![], cards.clone()];
            let before: usize = game.base.distribution.iter().flatten().count();
            let r =
                RamenAction::distribute_super_ramen_clones(&mut game, &mut StdRng::seed_from_u64(seed));
            dead_ok.0 &= r.is_ok();
            let clones: usize = game.base.distribution.iter().flatten().count() - before;
            dead_ok.1 &= clones == 5;
            if seed == 0 {
                println!("真无解（seed 0）: {:?}", game.base.distribution);
            }
        }
        c.check(dead_ok.0, "真无解局面不得返回 Err（分身放不下在规则上允许，不该中断育成）");
        c.check(dead_ok.1, "真无解局面下 5 张训练卡仍各拿到 1 个分身，只有友人卡落空");

        c.finish()
    }

    /// 回归：分身分配与父随机流解耦
    ///
    /// 按 `(rule_master, turn, TAG)` 派生，故不消耗父流、也与本回合此前消耗过几次无关。
    /// 这条是为 MCTS 的 CRN 服务：从父流 fork 会让各候选（策略流消耗长度不同）的
    /// 分身随机性去相关。未注入 `rule_master` 的旧路径回退父流 fork，消耗恰好 1 次。
    #[test]
    fn test_super_ramen_clones_decoupled_from_parent_stream() -> anyhow::Result<()> {
        use rand::RngCore;

        use crate::rng::StrategyRng;

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };

        let mut game = super::super::RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;
        game.base.turn = 72;
        game.ramen.super_ramen = Some(1);
        game.deck_can_split = true;

        let yayoi = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::Yayoi)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到理事长"))?;

        let mut c = Checks::new();

        // ① 注入 rule_master：父流一次都不消耗，且合法集大小无关
        game.set_rule_master(0x5EED_1234);
        game.base.turn = 72; // set_rule_master 会重置回合流，turn 要在之后再确认
        let boards: [(&str, Vec<Vec<i32>>); 3] = [
            ("空分布（每张卡 4 个合法位）", vec![vec![]; 5]),
            (
                "理事长占三格（友人只剩一格）",
                vec![vec![yayoi], vec![yayoi], vec![yayoi], vec![], vec![]]
            ),
            ("常规盘面", vec![vec![0], vec![1], vec![], vec![], vec![]])
        ];
        let mut all_zero = true;
        for (name, board) in boards {
            game.base.distribution = board;
            let mut parent = StrategyRng::new(0xDEAD_BEEF);
            RamenAction::distribute_super_ramen_clones(&mut game, &mut parent)?;
            println!("父流消耗（已注入 rule_master）: {name} -> {} 次", parent.counter());
            all_zero &= parent.counter() == 0;
        }
        c.check(all_zero, "注入 rule_master 后分身分配完全不消耗父策略流");

        // ② 关键性质：父流此前消耗了几次，不影响分身结果
        let mut results = Vec::new();
        for pre_draws in [0usize, 1, 7, 33] {
            game.base.distribution = vec![vec![]; 5];
            let mut parent = StrategyRng::new(0xDEAD_BEEF);
            for _ in 0..pre_draws {
                let _ = parent.next_u64();
            }
            RamenAction::distribute_super_ramen_clones(&mut game, &mut parent)?;
            println!("父流预消耗 {pre_draws} 次 -> 分身分布 {:?}", game.base.distribution);
            results.push(game.base.distribution.clone());
        }
        c.check(
            results.windows(2).all(|w| w[0] == w[1]),
            "父流此前消耗多少次都不影响分身结果（CRN 对齐的前提）"
        );

        // ③ 未注入 rule_master 的旧路径：回退到从父流 fork，消耗恰好 1 次
        let mut legacy = super::super::RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        legacy.add_friend_and_npcs()?;
        legacy.base.turn = 72;
        legacy.ramen.super_ramen = Some(1);
        legacy.deck_can_split = true;
        legacy.base.distribution = vec![vec![]; 5];
        let mut parent = StrategyRng::new(0xDEAD_BEEF);
        RamenAction::distribute_super_ramen_clones(&mut legacy, &mut parent)?;
        println!("父流消耗（未注入 rule_master，回退 fork）: {} 次", parent.counter());
        c.check(parent.counter() == 1, "未注入 rule_master 时回退从父流 fork，消耗恰好 1 次");

        // ④ 早退路径两种情况都不消耗父流
        game.base.turn = 71;
        game.base.distribution = vec![vec![]; 5];
        let mut parent = StrategyRng::new(0xDEAD_BEEF);
        RamenAction::distribute_super_ramen_clones(&mut game, &mut parent)?;
        c.check(parent.counter() == 0, "非超级拉面回合早退，不应消耗父流");

        c.finish()
    }

    /// 回归：`can_place_clone` / `place_clone` 的满员与挤 NPC 分支
    ///
    /// 挤 NPC 与满员拒绝这两条分支改写前从未被执行过——旧用例里没有任何盘面含 NPC。
    /// 直接测原语而非经由 `distribute_super_ramen_clones`：后者命中满员位是概率性的
    /// （无人选中的概率 (3/4)^6 ≈ 17.8%），拿它测会得到随种子飘的测试。
    #[test]
    fn test_clone_placement_full_train_and_npc_eviction() -> anyhow::Result<()> {
        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
        const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };

        let mut game = super::super::RamenGame::newgame(102601, &TEST_DECK, TEST_INHERIT)?;
        game.add_friend_and_npcs()?;

        let npcs: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| game.persons[i as usize].person_type == PersonType::Npc)
            .collect();
        let cards: Vec<i32> = (0..game.persons.len() as i32)
            .filter(|&i| game.persons[i as usize].person_type == PersonType::Card)
            .collect();
        println!("NPC 人头={npcs:?}，训练卡人头={cards:?}");

        let mut c = Checks::new();
        c.check(npcs.len() >= 5 && cards.len() == 5, "测试局面应有 5 个 NPC 与 5 张训练卡");

        // ① 满 5 人但全是 NPC：可放置，且放置时挤掉一个 NPC
        game.base.distribution = vec![npcs[..5].to_vec(), vec![], vec![], vec![], vec![]];
        c.check(
            RamenAction::can_place_clone(&game, cards[0], 0),
            "满 5 人但含 NPC 时可放置（由 place_clone 挤掉 NPC）"
        );
        RamenAction::place_clone(&mut game, cards[0], 0, "测试")?;
        let d = &game.base.distribution[0];
        let npc_left = d.iter().filter(|&&p| game.persons[p as usize].person_type == PersonType::Npc).count();
        println!("挤 NPC 后速位: {d:?}（NPC 剩 {npc_left}）");
        c.check(d.len() == 5, "挤掉一个补一个，人数仍为 5");
        c.check(npc_left == 4, "恰好挤掉 1 个 NPC");
        c.check(d.contains(&cards[0]), "分身确实进入了该训练位");

        // ② 满 5 人且全是非 NPC：不可放置
        game.base.distribution = vec![cards.clone(), vec![], vec![], vec![], vec![]];
        c.check(
            !RamenAction::can_place_clone(&game, cards[0], 0),
            "该训练已有本体时不可放置"
        );
        let friend = game
            .persons
            .iter()
            .position(|p| p.person_type == PersonType::ScenarioCard)
            .map(|i| i as i32)
            .ok_or_else(|| anyhow!("找不到友人卡"))?;
        c.check(
            !RamenAction::can_place_clone(&game, friend, 0),
            "满 5 个非 NPC 时不可放置（注意：与「满 5 人且无 NPC 可挤」互为冗余，测不出是哪条生效）"
        );
        c.check(
            RamenAction::place_clone(&mut game, friend, 0, "测试").is_err(),
            "对非法落点调用 place_clone 必须返回 Err（不变式被破坏，属编程错误）"
        );

        // ③ 越界与非法人头下标
        c.check(!RamenAction::can_place_clone(&game, cards[0], 5), "训练位越界不可放置");
        c.check(!RamenAction::can_place_clone(&game, -1, 0), "负数人头下标不可放置（原实现会 panic）");
        c.check(
            !RamenAction::can_place_clone(&game, game.persons.len() as i32, 0),
            "越界人头下标不可放置"
        );

        c.finish()
    }

    /// 地区选择必须按显式 `year_idx` 归档，不能用 `current_year()-1`。
    ///
    /// 正向断言：turn 23 写入第 2 年格子，且第 1 年格子不被覆盖。
    #[test]
    fn test_region_select_archives_explicit_year_idx() -> anyhow::Result<()> {
        use crate::game::{
            ActionEnum,
            ramen::{RamenGame, RamenStage}
        };
        use rand::{SeedableRng, rngs::StdRng};

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit.clone())?;
        let mut rng = StdRng::seed_from_u64(1);
        let mut c = Checks::new();

        // 第 1 年：turn 2。生产路径已是 RegionSelect 阶段；Begin 的 `_` 分支仍能归档。
        game.base.turn = 2;
        game.stage = RamenStage::Begin;
        let y1 = [0usize, 1, 2];
        RamenAction::no_ramen(Operation::RegionSelect(y1)).apply(&mut game, &mut rng)?;
        println!("turn2 yearly={:?} live={:?}", game.ramen.yearly_selected_regions, game.ramen.selected_regions);
        c.check(game.ramen.yearly_selected_regions[0] == y1, "turn 2 归档到第 1 年");
        c.check(game.ramen.selected_regions == y1, "turn 2 live selected_regions");
        c.check(
            (game.current_year() - 1) as usize == 0,
            "turn 2 的 current_year()-1 碰巧也是 0"
        );

        // 第 2 年：turn 23 / RegionSelect。current_year() 仍是 1。
        game.base.turn = 23;
        game.stage = RamenStage::RegionSelect;
        let y2 = [5usize, 6, 7];
        RamenAction::no_ramen(Operation::RegionSelect(y2)).apply(&mut game, &mut rng)?;
        println!(
            "turn23 current_year()={} yearly={:?}",
            game.current_year(),
            game.ramen.yearly_selected_regions
        );
        c.check(game.current_year() == 1, "陷阱：turn 23 的 current_year() 仍是 1");
        c.check(
            RamenState::region_archive_year_idx(23)? == 1,
            "turn 23 显式 year_idx = 1"
        );
        c.check(game.ramen.yearly_selected_regions[1] == y2, "turn 23 归档到第 2 年");
        c.check(
            game.ramen.yearly_selected_regions[0] == y1,
            "turn 23 不得覆盖第 1 年（current_year()-1 会写到下标 0）"
        );

        // 第 3 年：turn 47
        game.base.turn = 47;
        game.stage = RamenStage::RegionSelect;
        let y3 = [10usize, 11, 12];
        RamenAction::no_ramen(Operation::RegionSelect(y3)).apply(&mut game, &mut rng)?;
        println!("turn47 current_year()={} yearly={:?}", game.current_year(), game.ramen.yearly_selected_regions);
        c.check(game.current_year() == 2, "陷阱：turn 47 的 current_year() 仍是 2");
        c.check(game.ramen.yearly_selected_regions[2] == y3, "turn 47 归档到第 3 年");
        c.check(game.ramen.yearly_selected_regions[1] == y2, "turn 47 不得覆盖第 2 年");

        // Train 阶段同一 helper 也必须按显式 year_idx 写（防某一处漏接）
        let mut game_train = RamenGame::newgame(102601, &deck, inherit)?;
        game_train.base.turn = 23;
        game_train.stage = RamenStage::Train;
        RamenAction::no_ramen(Operation::RegionSelect(y2)).apply(&mut game_train, &mut rng)?;
        c.check(
            game_train.ramen.yearly_selected_regions[1] == y2,
            "Train 阶段 turn 23 同样归档到第 2 年"
        );
        c.check(
            game_train.ramen.yearly_selected_regions[0] == [0, 0, 0],
            "Train 阶段 turn 23 不得写入第 1 年"
        );

        c.finish()
    }

    /// SuperRamenSelect：候选恰好 3 个；apply 只写字段；越界 / 错误阶段返回 Err
    #[test]
    fn test_super_ramen_select_list_and_apply() -> anyhow::Result<()> {
        use crate::game::{
            ActionEnum,
            ramen::{RamenGame, RamenStage}
        };
        use rand::{SeedableRng, rngs::StdRng};

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        let mut rng = StdRng::seed_from_u64(1);
        let mut c = Checks::new();

        game.base.turn = 71;
        game.stage = RamenStage::SuperRamenSelect;
        // 即使 turn 71 被标成比赛回合，候选也必须是 3 个超级拉面选项
        game.uma.career_races |= 1u64 << (71 - 11);

        let actions = crate::game::traits::Game::list_actions(&game)?;
        println!(
            "SuperRamenSelect list_actions (is_race_turn={}): {} 个 -> {:?}",
            game.is_race_turn(),
            actions.len(),
            actions.iter().map(|a| a.to_string()).collect::<Vec<_>>()
        );
        c.check(game.is_race_turn(), "本测试强制 turn 71 为比赛回合");
        c.check(actions.len() == 3, "恰好 3 个超级拉面候选");
        c.check(
            actions.iter().all(|a| matches!(a.operation, Operation::SuperRamenSelect(_))),
            "候选全部是 SuperRamenSelect"
        );
        c.check(
            actions.iter().all(|a| a.ramen.is_none() && a.special_targets.is_none()),
            "不复用 ramen / special_targets 字段"
        );
        c.check(actions[0].to_string() == "超级拉面选项 1", "Display 选项 1");
        c.check(actions[1].to_string() == "超级拉面选项 2", "Display 选项 2");
        c.check(actions[2].to_string() == "超级拉面选项 3", "Display 选项 3");
        c.check(actions[0].as_base_action().is_none(), "to_base_action 为 None");

        let turn_before = game.turn();
        let stage_before = game.stage.clone();
        actions[1].apply(&mut game, &mut rng)?;
        println!("apply 选项二后 super_ramen={:?} stage={:?} turn={}", game.ramen.super_ramen, game.stage, game.turn());
        c.check(game.ramen.super_ramen == Some(1), "apply 写入选项二");
        c.check(game.stage == stage_before, "apply 不切阶段");
        c.check(game.turn() == turn_before, "apply 不推进回合");

        let oob = RamenAction::super_ramen_select(99);
        let oob_err = oob.apply(&mut game, &mut rng);
        println!("越界 apply: {oob_err:?}");
        c.check(oob_err.is_err(), "越界下标返回 Err");

        game.stage = RamenStage::Train;
        let wrong_stage = RamenAction::super_ramen_select(1).apply(&mut game, &mut rng);
        println!("错误阶段 Train apply: {wrong_stage:?}");
        c.check(wrong_stage.is_err(), "错误阶段返回 Err，不静默忽略");

        game.stage = RamenStage::Begin;
        let wrong_begin = RamenAction::super_ramen_select(1).apply(&mut game, &mut rng);
        println!("错误阶段 Begin apply: {wrong_begin:?}");
        c.check(wrong_begin.is_err(), "Begin 阶段同样返回 Err");

        c.finish()
    }

}
