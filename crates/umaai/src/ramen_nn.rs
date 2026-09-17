//! 客户端「整局无搜索 NN」的最小包装层（`ramen_trainer_policy = "nn"`，`onnx` feature）
//!
//! [`RamenNnTrainer`] 本身**不实现** [`Trainer::last_decision`]：它是给批量采集与
//! benchmark 用的，那些入口不需要协议摘要。客户端需要——没有摘要，
//! `calc_ramen_training` 里除地区 / 比赛回合 / `RamenSelect` 之外的阶段
//! （最常见的就是普通训练回合）会一条结果都不输出，屏幕全空。
//!
//! 本模块只补这一件事：把一次动作决策的**下标、完整候选描述、阶段、真实来源**记下来，
//! 由 [`Trainer::last_decision`] 交给既有输出链路。
//!
//! # 不伪造评分
//!
//! `candidate_scores` / `candidate_n` 恒空、`score` 恒 0。policy logits 是「教师在这个
//! 局面上更可能选谁」的相对量，**不是终局分**，把它填进评分字段会让下游的
//! luck baseline、「期望评分」行、`action_luck` 全部读出一个没有量纲的数。上层按
//! `candidate_scores` 是否为空路由，因此这一路天然不挂 luck。
//!
//! # 来源照实标注
//!
//! 一次动作决策有四个出口，[`LabeledPrep`] 把它们分开，本层原样翻译成来源标签，
//! **不另做一遍判定**、也不为了标来源再跑一次推理：
//!
//! | 出口 | 来源标签 | 跑推理？ |
//! |---|---|---|
//! | 网络 argmax | `ramen_nn` | 是 |
//! | 自选比赛硬守门 | `ramen_race_gate` | 否 |
//! | 唯一候选 | `ramen_single_candidate` | 否 |
//! | `SpecialSelect` 整阶段转手写 | `ramen_handwritten_stage` | 否 |
//!
//! ❗**不是「完全没有手写逻辑」**：事件选项与友人事件仍由 [`RamenNnTrainer`] 内部的
//! 手写策略处理（choice 头没训练），自选比赛硬守门也是手写规则。「纯 NN」指的是
//! **动作决策**这条线。

use std::sync::Mutex;

use anyhow::{Result, anyhow};
use rand::prelude::StdRng;
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo,
        decision::{
            SOURCE_RAMEN_HANDWRITTEN_STAGE, SOURCE_RAMEN_NN, SOURCE_RAMEN_RACE_GATE,
            SOURCE_RAMEN_SINGLE_CANDIDATE
        }
    },
    trainer::{LabeledPrep, RamenNnTrainer, ramen_handwritten_trainer::ramen_effective_stage}
};

use crate::scenario::ramen::ramen_stage_kind;

/// 整局直接走网络的客户端决策器（**不跑任何搜索**）
///
/// 除了记录 [`Trainer::last_decision`] 所需的摘要，本层不改变 [`RamenNnTrainer`]
/// 的任何行为：动作走同一条 `prepare → infer → resolve`，事件选项原样转发。
pub struct WholeGameNnTrainer {
    /// 真正做决策的网络训练员
    nn: RamenNnTrainer,
    /// 最近一次**动作决策**的协议摘要
    ///
    /// 每次动作决策开头先清空：这样任何早退（推理失败、候选落格失败）都不会把上一步的
    /// 摘要留在槽里被当成本次结果。事件选项转发同样清空——事件不是动作决策，
    /// 把上一步的动作摘要挂在它后面就是串了来源。
    last: Mutex<Option<DecisionInfo>>
}

impl WholeGameNnTrainer {
    /// 包装一个已加载好的网络训练员
    pub fn new(nn: RamenNnTrainer) -> Self {
        Self {
            nn,
            last: Mutex::new(None)
        }
    }

    /// 清空动作摘要槽
    ///
    /// 锁被毒化时也照清（取 `into_inner`）：这里写的是「没有结果」，比留着旧结果安全。
    fn clear_last(&self) {
        let mut slot = match self.last.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = None;
    }

    /// 写入本次动作决策的摘要
    ///
    /// # 错误
    ///
    /// 摘要锁被毒化时报错——宁可让这一局停下，也不要把上一条摘要当成本次结果发出去。
    fn store_last(&self, info: DecisionInfo) -> Result<()> {
        *self
            .last
            .lock()
            .map_err(|_| anyhow!("整局网络决策摘要锁被毒化"))? = Some(info);
        Ok(())
    }

    /// 为一次无搜索的动作决策合成协议摘要
    ///
    /// 形状与 `scenario::ramen::fallback_decision` 合成的那条对齐（候选描述完整、
    /// 评分为空），额外带上来源标签与正确的 `decision_kind`。
    fn decision_info(
        game: &RamenGame, actions: &[RamenAction], picked: usize, source: &str
    ) -> DecisionInfo {
        let stage = ramen_effective_stage(game, actions);
        DecisionInfo {
            action_index: picked,
            score: 0.0,
            decision_kind: ramen_stage_kind(stage).to_string(),
            candidate_scores: Vec::new(),
            candidate_descriptions: actions.iter().map(ToString::to_string).collect(),
            candidate_n: Vec::new(),
            scenario_extra: None
        }
        .with_source(source)
    }
}

impl Trainer<RamenGame> for WholeGameNnTrainer {
    /// 直接走网络（或守门 / 单候选短路），**不跑搜索**，并记下本次决策的摘要
    ///
    /// # 错误
    ///
    /// 候选为空、特征编码失败、推理失败、任一候选无法落格，或摘要锁被毒化时报错
    /// ——**任何一种都不会静默回退成手写或搜索**。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        // 先清空：本次若中途报错，上一步的摘要不得留在槽里
        self.clear_last();
        let (picked, source) = match self.nn.prepare_decision_labeled(game, actions, rng)? {
            LabeledPrep::RaceGate(i) => (i, SOURCE_RAMEN_RACE_GATE),
            LabeledPrep::HandwrittenStage(i) => (i, SOURCE_RAMEN_HANDWRITTEN_STAGE),
            LabeledPrep::SingleCandidate(i) => (i, SOURCE_RAMEN_SINGLE_CANDIDATE),
            LabeledPrep::NeedsInference(features) => {
                let out = self.nn.infer_features(features)?;
                (self.nn.resolve_decision(game, actions, &out.policy)?, SOURCE_RAMEN_NN)
            }
        };
        self.store_last(Self::decision_info(game, actions, picked, source))?;
        Ok(picked)
    }

    /// 事件选项（旧接口）原样转发给网络训练员内部的手写策略
    ///
    /// 转发前清空动作摘要：事件不是动作决策，留着上一步的摘要会让它被当成本回合的结果。
    ///
    /// # 错误
    ///
    /// 内部策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_last();
        self.nn.select_choice(game, choices, rng)
    }

    /// 事件选项（新接口）——同 [`Self::select_choice`]
    ///
    /// # 错误
    ///
    /// 内部策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_last();
        self.nn.select_event_choice(game, event, choices, rng)
    }

    /// 本次动作决策自己的摘要；没有动作决策（或刚转发过事件）时为 `None`
    fn last_decision(&self) -> Option<DecisionInfo> {
        self.last.lock().ok()?.clone()
    }

    /// 恒为 `None`：无搜索就没有候选评分分解，透传手写策略的分解会挂错理由
    fn last_breakdown(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use umasim::game::ramen::RamenStage;

    use super::*;
    use crate::utils::Checks;

    /// 四个来源标签互不相同、都不为空
    ///
    /// 这是「来源必须真实」的最小守门：`select_action` 的 match 一旦把两个出口写成同一个
    /// 标签，屏幕上就分不出「网络算的」和「守门顶上的」。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_source_labels_are_distinct() -> Result<()> {
        let mut c = Checks::new();
        let labels = [
            SOURCE_RAMEN_NN,
            SOURCE_RAMEN_RACE_GATE,
            SOURCE_RAMEN_SINGLE_CANDIDATE,
            SOURCE_RAMEN_HANDWRITTEN_STAGE
        ];
        for l in labels {
            println!("  来源标签 {l}");
            c.check(!l.is_empty(), &format!("{l} 非空"));
        }
        let mut sorted = labels.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        c.check(sorted.len() == labels.len(), "四个来源标签互不相同");
        c.finish()
    }

    /// 客户端会真实派发的四个阶段都能映射出非空 `decision_kind`
    ///
    /// `decision_kind` 是 AIRedirector 分发 partial decision 的依据；映射漏一个，
    /// 下游就收到空串。❗`SuperRamenSelect` **不在此列**——客户端根本不派发它
    /// （`calc_ramen_training` 把该阶段的候选直接置空），见模块 §阶段边界。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_client_stage_kinds_are_named() -> Result<()> {
        let mut c = Checks::new();
        for (stage, want) in [
            (RamenStage::Train, "train"),
            (RamenStage::RamenSelect, "ramen_select"),
            (RamenStage::SpecialSelect, "special_select"),
            (RamenStage::RegionSelect, "region_select")
        ] {
            let got = ramen_stage_kind(stage.clone());
            println!("  {stage:?} → {got}");
            c.check(got == want, &format!("{stage:?} 的 decision_kind 是 {want}"));
        }
        c.finish()
    }
}
