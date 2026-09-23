//! 外层 `RegionSelect` 的地区网络决策器（`onnx` feature）
//!
//! 包着一个 [`RamenMctsTrainer`]：除外层地区选择外的一切调用原样转发。地区选择按
//! [`RegionNnRole`] 二选一：网络执行（`region_nn`），或既有逻辑执行、网络只给参考
//! （`region_nn_hint`）。

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering}
};

use anyhow::{Result, anyhow, bail};
use rand::prelude::StdRng;
use serde_json::{Map, Value, json};
use umasim::{
    game::{
        Trainer,
        ramen::{Operation, RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo,
        decision::{REGION_NN_HINT_KEY, SOURCE_REGION_NN}
    },
    trainer::{RamenMctsTrainer, RamenNnTrainer, ramen_handwritten_trainer::ramen_effective_stage}
};

/// 网络在外层地区选择中的角色
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionNnRole {
    /// 网络的推荐就是执行推荐（`region_nn`），不再做地区搜索
    Execute,
    /// 既有逻辑照常执行，网络的推荐只作参考挂在决策上（`region_nn_hint`）
    Hint
}

/// `RamenMctsTrainer` + 外层 `RegionSelect` 的地区网络
///
/// 搜索内部模拟的地区选择由 [`RamenMctsTrainer`] 自己的手写 rollout 策略负责，
/// 本壳不接触。
pub struct RegionNnTrainer {
    /// 做除地区外全部决策的搜索训练员（`Hint` 下地区也由它执行）
    mcts: RamenMctsTrainer,
    /// 外层地区选择使用的网络
    nn: RamenNnTrainer,
    /// 网络的角色
    role: RegionNnRole,
    /// 最近一次决策是否为外层地区选择
    ///
    /// 置位时 [`Trainer::last_decision`] 改读 [`Self::region_info`]。任何一次转发给
    /// 内部搜索训练员的调用都会清位。用 `AtomicBool` 是因为 `Trainer` 的方法都取 `&self`。
    region_last: AtomicBool,
    /// 最近一次外层地区选择的协议摘要（`region_last` 为真时有效）
    region_info: Mutex<Option<DecisionInfo>>
}

impl RegionNnTrainer {
    /// 装配地区网络决策器
    pub fn new(mcts: RamenMctsTrainer, nn: RamenNnTrainer, role: RegionNnRole) -> Self {
        Self {
            mcts,
            nn,
            role,
            region_last: AtomicBool::new(false),
            region_info: Mutex::new(None)
        }
    }

    /// 网络的角色
    pub fn role(&self) -> RegionNnRole {
        self.role
    }

    /// 取出候选 `i` 的三个地区下标
    ///
    /// # 错误
    ///
    /// 下标越界，或该候选不是 `RegionSelect` 时报错。
    pub fn region_of(actions: &[RamenAction], i: usize) -> Result<[usize; 3]> {
        match actions.get(i).map(|a| a.operation) {
            Some(Operation::RegionSelect(r)) => Ok(r),
            other => bail!("地区阶段候选 {i} 不是 RegionSelect：{other:?}")
        }
    }

    /// 为一次无搜索评分的地区决策合成协议摘要
    ///
    /// 形状与 `scenario::ramen::fallback_decision` 一致（候选描述完整、评分为空）。
    fn plain_region_info(actions: &[RamenAction], picked: usize) -> DecisionInfo {
        DecisionInfo {
            action_index: picked,
            score: 0.0,
            decision_kind: "region_select".to_string(),
            candidate_scores: Vec::new(),
            candidate_descriptions: actions.iter().map(ToString::to_string).collect(),
            candidate_n: Vec::new(),
            scenario_extra: None
        }
    }

    /// 把网络的参考推荐挂到决策的 `scenario_extra.region_nn_hint`
    ///
    /// 只记录网络选了什么、是否与执行推荐一致，不改任何评分字段。
    fn attach_hint(mut info: DecisionInfo, actions: &[RamenAction], nn_idx: usize, picked: usize) -> DecisionInfo {
        let payload = json!({
            "choice": actions.get(nn_idx).map(ToString::to_string),
            "same_as_executed": nn_idx == picked
        });
        match info.scenario_extra {
            Some(Value::Object(ref mut map)) => {
                map.insert(REGION_NN_HINT_KEY.to_string(), payload);
            }
            _ => {
                let mut map = Map::new();
                map.insert(REGION_NN_HINT_KEY.to_string(), payload);
                info.scenario_extra = Some(Value::Object(map));
            }
        }
        info
    }

    /// 记下本次地区决策的摘要，并把 `last_decision` 切到它上面
    ///
    /// # 错误
    ///
    /// 摘要锁被毒化时报错，不把上一条摘要当成本次结果发出去。
    fn store_region_info(&self, info: DecisionInfo) -> Result<()> {
        *self
            .region_info
            .lock()
            .map_err(|_| anyhow!("地区决策摘要锁被毒化"))? = Some(info);
        self.region_last.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// 解除地区标记（转发给内部搜索训练员之前调用）
    fn clear_region_flag(&self) {
        self.region_last.store(false, Ordering::Relaxed);
    }
}

impl Trainer<RamenGame> for RegionNnTrainer {
    /// 外层地区选择按角色处理，其余阶段原样转发搜索训练员
    ///
    /// # 错误
    ///
    /// 网络推理失败、候选不是地区组合，或转发的搜索训练员报错时原样返回，
    /// 不会静默退回既有逻辑。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        self.clear_region_flag();
        if ramen_effective_stage(game, actions) != RamenStage::RegionSelect {
            return self.mcts.select_action(game, actions, rng);
        }
        match self.role {
            RegionNnRole::Execute => {
                let picked = self.nn.select_action(game, actions, rng)?;
                let _ = Self::region_of(actions, picked)?;
                let info = Self::plain_region_info(actions, picked).with_source(SOURCE_REGION_NN);
                self.store_region_info(info)?;
                Ok(picked)
            }
            RegionNnRole::Hint => {
                // 执行侧吃真实 rng，与 `handwritten` 逐字一致；网络在副本上算，不推进随机流
                let picked = self.mcts.select_action(game, actions, rng)?;
                let mut hint_rng = rng.clone();
                let nn_idx = self.nn.select_action(game, actions, &mut hint_rng)?;
                let _ = Self::region_of(actions, nn_idx)?;
                // 既有逻辑走过地区搜索时沿用它自己的摘要，否则与手写 fallback 同形
                let base = self
                    .mcts
                    .last_decision()
                    .unwrap_or_else(|| Self::plain_region_info(actions, picked));
                self.store_region_info(Self::attach_hint(base, actions, nn_idx, picked))?;
                Ok(picked)
            }
        }
    }

    /// 事件选项转发搜索训练员
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_region_flag();
        self.mcts.select_choice(game, choices, rng)
    }

    /// 事件选项（新接口）转发搜索训练员
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_region_flag();
        self.mcts.select_event_choice(game, event, choices, rng)
    }

    /// 地区决策后返回本次地区决策的摘要，其余返回内部搜索训练员的摘要
    fn last_decision(&self) -> Option<DecisionInfo> {
        match self.region_last.load(Ordering::Relaxed) {
            true => self.region_info.lock().ok()?.clone(),
            false => self.mcts.last_decision()
        }
    }

    /// 网络执行的地区决策没有搜索分解；其余情况透传内部搜索训练员
    fn last_breakdown(&self) -> Option<String> {
        match (self.region_last.load(Ordering::Relaxed), self.role) {
            (true, RegionNnRole::Execute) => None,
            _ => self.mcts.last_breakdown()
        }
    }
}
