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
//!
//! # 对照模式（`ramen_trainer_policy = "nn_compare"`）
//!
//! 挂上 [`WholeCompare`] 后，每个**多候选**动作决策先让 MCTS 在随机流副本上算一遍，
//! 再照常走网络；屏幕上两条推荐都打印。执行推荐恒为网络那条，MCTS 那一跑的理由与
//! verbose 日志被静音（它不是执行侧，留着会被当成本次推荐的理由）。
//! MCTS 的 `RamenSelect` 合并搜索会缓存下一步 `SpecialSelect` 的 targets；实际执行的
//! 吃面与 MCTS 选的不同时丢弃该缓存，下一步由 MCTS 按实际局面重新搜索。

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use colored::Colorize;
use rand::prelude::StdRng;
use serde_json::{Map, Value, json};
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo,
        decision::{
            SOURCE_RAMEN_HANDWRITTEN_STAGE, SOURCE_RAMEN_NN, SOURCE_RAMEN_RACE_GATE,
            SOURCE_RAMEN_SINGLE_CANDIDATE
        }
    },
    trainer::{LabeledPrep, RamenMctsTrainer, RamenNnTrainer, ramen_handwritten_trainer::ramen_effective_stage}
};

use crate::{decision::ReasonGate, scenario::ramen::ramen_stage_kind};

/// 整局对照模式的 MCTS 一侧（`ramen_trainer_policy = "nn_compare"`）
///
/// 只用于显示与记录，**不参与执行**：执行推荐恒为网络那条。
pub struct WholeCompare {
    /// 对照用的搜索训练员（主程序按既有路径构造，搜索参数取自 `[mcts]`）
    pub mcts: RamenMctsTrainer,
    /// 是否把对照行直接打到 stdout
    ///
    /// ❗human 模式为真；`--json` 模式**必须**为假，否则会污染严格 JSON 流。
    pub print: bool,
    /// 内部 [`RamenMctsTrainer`] 实际挂着的理由门（`None` = 不做理由隔离）
    pub reason_gate: Option<Arc<ReasonGate>>
}

/// MCTS 一侧在同一局面下的选择
struct ComparePick {
    /// 选中的候选下标（在传入的 `actions` 全表里）
    idx: usize,
    /// 这一侧是否真搜索过（含 `SpecialSelect` 命中合并搜索缓存）；假表示落在手写基策上
    searched: bool
}

impl ComparePick {
    /// 屏幕与协议里给这一侧用的名字
    fn label(&self) -> &'static str {
        match self.searched {
            true => "MCTS",
            false => "手写策略"
        }
    }
}

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
    last: Mutex<Option<DecisionInfo>>,
    /// 可选的 MCTS 对照（`None` = 纯网络，与对照模式引入前逐字一致）
    compare: Option<WholeCompare>
}

impl WholeGameNnTrainer {
    /// 包装一个已加载好的网络训练员
    pub fn new(nn: RamenNnTrainer) -> Self {
        Self {
            nn,
            last: Mutex::new(None),
            compare: None
        }
    }

    /// 打开 MCTS 对照（`nn_compare`）：每个多候选动作决策多跑一次搜索
    pub fn with_compare(mut self, compare: WholeCompare) -> Self {
        self.compare = Some(compare);
        self
    }

    /// 是否开着 MCTS 对照
    pub fn is_compare(&self) -> bool {
        self.compare.is_some()
    }

    /// 在随机流**副本**上让 MCTS 算一遍当前局面（外层 `rng` 不被推进）
    ///
    /// 那一跑的理由数据与 verbose 日志都被静音：它不是执行侧。两个守卫都是 RAII，
    /// 中途 `?` 早退照样恢复。
    ///
    /// # 错误
    ///
    /// 搜索报错时原样返回——不静默跳过对照。
    fn run_compare(
        cmp: &WholeCompare, game: &RamenGame, actions: &[RamenAction], rng: &StdRng
    ) -> Result<ComparePick> {
        let mut rng_copy = rng.clone();
        let mute = cmp.reason_gate.as_ref().map(|g| g.mute());
        let hush = cmp.mcts.quiet_scope();
        let hits_before = cmp.mcts.combined_cache_hits();
        let idx = cmp.mcts.select_action(game, actions, &mut rng_copy)?;
        let cache_hit = cmp.mcts.combined_cache_hits() > hits_before;
        let searched = cache_hit
            || cmp
                .mcts
                .last_decision()
                .is_some_and(|i| !i.candidate_scores.is_empty());
        drop(hush);
        drop(mute);
        Ok(ComparePick { idx, searched })
    }

    /// 屏幕上的对照行：两条推荐各自是什么
    fn compare_line(actions: &[RamenAction], nn_idx: usize, base: &ComparePick) -> String {
        let text = |i: usize| actions.get(i).map(ToString::to_string).unwrap_or_default();
        format!("{} → {} ｜ 神经网络 → {}", base.label(), text(base.idx), text(nn_idx))
    }

    /// 把对照结果挂到决策的 `scenario_extra.trainer_compare`
    ///
    /// 只记录两侧各选了什么，不伪造任何评分。下标一律是传入 `actions` 全表的下标。
    fn attach_compare(
        mut info: DecisionInfo, actions: &[RamenAction], nn_idx: usize, base: &ComparePick
    ) -> DecisionInfo {
        let payload = json!({
            "index_space": "actions",
            "nn_index": nn_idx,
            "nn_choice": actions.get(nn_idx).map(ToString::to_string),
            "mcts_index": base.idx,
            "mcts_choice": actions.get(base.idx).map(ToString::to_string),
            "mcts_kind": match base.searched { true => "search", false => "handwritten" }
        });
        match info.scenario_extra {
            Some(Value::Object(ref mut map)) => {
                map.insert("trainer_compare".to_string(), payload);
            }
            _ => {
                let mut map = Map::new();
                map.insert("trainer_compare".to_string(), payload);
                info.scenario_extra = Some(Value::Object(map));
            }
        }
        info
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
        // 对照侧必须在网络推理**之前**取随机流快照：两侧看到同一个随机状态
        let base = match &self.compare {
            Some(cmp) if actions.len() > 1 => Some(Self::run_compare(cmp, game, actions, rng)?),
            Some(cmp) => {
                // 单候选不做对照；但若是 SpecialSelect，MCTS 缓存的 targets 已无人消费，丢弃
                if game.stage == RamenStage::SpecialSelect {
                    cmp.mcts.discard_pending_combined();
                }
                None
            }
            None => None
        };
        let (picked, source) = match self.nn.prepare_decision_labeled(game, actions, rng)? {
            LabeledPrep::RaceGate(i) => (i, SOURCE_RAMEN_RACE_GATE),
            LabeledPrep::HandwrittenStage(i) => (i, SOURCE_RAMEN_HANDWRITTEN_STAGE),
            LabeledPrep::SingleCandidate(i) => (i, SOURCE_RAMEN_SINGLE_CANDIDATE),
            LabeledPrep::NeedsInference(features) => {
                let out = self.nn.infer_features(features)?;
                (self.nn.resolve_decision(game, actions, &out.policy)?, SOURCE_RAMEN_NN)
            }
        };
        let mut info = Self::decision_info(game, actions, picked, source);
        if let (Some(cmp), Some(base)) = (&self.compare, &base) {
            // 实际执行的吃面与 MCTS 选的不同：MCTS 缓存的 SpecialSelect targets 属于没走的分支
            if game.stage == RamenStage::RamenSelect && base.idx != picked {
                cmp.mcts.discard_pending_combined();
            }
            if cmp.print {
                println!("{}", Self::compare_line(actions, picked, base).bright_cyan());
            }
            info = Self::attach_compare(info, actions, picked, base);
        }
        self.store_last(info)?;
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

    /// 记录每次动作决策摘要的转发壳（测试内用）
    struct Recorder<'a> {
        /// 被测的整局网络训练员
        inner: &'a WholeGameNnTrainer,
        /// 每次多候选动作决策的（阶段, 对照载荷）
        seen: Mutex<Vec<(RamenStage, Option<Value>)>>
    }

    impl Trainer<RamenGame> for Recorder<'_> {
        fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
            let idx = self.inner.select_action(game, actions, rng)?;
            if actions.len() > 1 {
                let extra = self
                    .inner
                    .last_decision()
                    .and_then(|i| i.scenario_extra)
                    .and_then(|v| v.get("trainer_compare").cloned());
                self.seen
                    .lock()
                    .map_err(|_| anyhow!("记录锁被毒化"))?
                    .push((game.stage.clone(), extra));
            }
            Ok(idx)
        }

        fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
            self.inner.select_choice(game, choices, rng)
        }

        fn select_event_choice(
            &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_event_choice(game, event, choices, rng)
        }
    }

    /// 整局对照模式跑完整局：不报错、执行侧恒为网络、每个多候选决策都带对照载荷
    ///
    /// 重点覆盖 MCTS 合并搜索缓存：实际执行的吃面与 MCTS 选的不同时必须丢弃缓存，
    /// 否则下一步 `SpecialSelect` 会报「缓存未命中」让整局中止。本用例要求这条路径
    /// 至少真实发生一次，否则声明零覆盖。模型不在版本库里；缺模型时跳过。
    ///
    /// # 错误
    ///
    /// 整局报错或任一观测未通过时返回错误。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_whole_nn_compare_full_game() -> Result<()> {
        use std::{env, path::Path};

        use rand::SeedableRng;
        use umasim::{
            game::{Game, InheritInfo},
            gamedata::init_global,
            search::SearchConfig,
            trainer::{RamenSearchStages, SpecialSelectMode},
            utils::get_workspace_root
        };

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let model = Path::new("saved_models/arms/ens_R67mix60k_g123.onnx");
        if !model.is_file() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 「整局对照模式跑完整局」这条本次**零覆盖**，绿色不代表它还正确。");
            return c.finish();
        }
        let mcts = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::parse("train,ramen")?)
            .verbose(false);
        let nn = RamenNnTrainer::load(model)?
            .with_race_shield(true)
            .with_special_mode(SpecialSelectMode::Canonical);
        let trainer = WholeGameNnTrainer::new(nn).with_compare(WholeCompare {
            mcts,
            print: false,
            reason_gate: None
        });
        let rec = Recorder {
            inner: &trainer,
            seen: Mutex::new(Vec::new())
        };
        let mut game = RamenGame::newgame(
            106402,
            &[303174, 303124, 303044, 302894, 303064, 303054],
            InheritInfo {
                blue_count: [15, 0, 0, 0, 3],
                extra_count: [0, 10, 20, 40, 40, 40]
            }
        )?;
        let mut rng = StdRng::seed_from_u64(20260920);
        let run = game.run_full_game(&rec, &mut rng);
        println!("整局结果: {run:?}，终局评分 {}", game.uma.calc_score());
        c.check(run.is_ok(), "对照模式整局跑完不报错");

        let seen = rec.seen.lock().map_err(|_| anyhow!("记录锁被毒化"))?;
        let with_payload = seen.iter().filter(|(_, p)| p.is_some()).count();
        println!("多候选动作决策 {} 次，带对照载荷 {with_payload} 次", seen.len());
        c.check(!seen.is_empty() && with_payload == seen.len(), "每个多候选动作决策都带对照载荷");
        let ramen_diff = seen
            .iter()
            .filter(|(st, p)| {
                *st == RamenStage::RamenSelect
                    && p.as_ref().is_some_and(|v| v.get("nn_index") != v.get("mcts_index"))
            })
            .count();
        let specials = seen.iter().filter(|(st, _)| *st == RamenStage::SpecialSelect).count();
        println!("RamenSelect 上两侧不同 {ramen_diff} 次；SpecialSelect 多候选决策 {specials} 次");
        if ramen_diff == 0 {
            println!("❗❗ 本局 RamenSelect 两侧从未分歧，「丢弃缓存」路径**零覆盖**，绿色不代表它正确。");
        }
        let mut shown = 0;
        for (st, p) in seen.iter() {
            if let Some(v) = p {
                if shown < 6 && v.get("nn_index") != v.get("mcts_index") {
                    println!("  {st:?}: MCTS({}) → {} ｜ 神经网络 → {}", v["mcts_kind"], v["mcts_choice"], v["nn_choice"]);
                    shown += 1;
                }
            }
        }
        c.finish()
    }
}
