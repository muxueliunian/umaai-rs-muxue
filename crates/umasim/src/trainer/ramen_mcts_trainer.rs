//! 拉面杯 MCTS 训练员
//!
//! 用扁平蒙特卡洛搜索（[`FlatSearch<RamenGame>`]）替换手写策略的部分决策点，
//! 其余决策点仍走 [`RamenHandwrittenTrainer`]。
//!
//! # 为什么不复用 `MctsTrainer`
//!
//! `MctsTrainer` 只 `impl Trainer<OnsenGame>`，且字段与温泉强耦合
//! （`HandwrittenEvaluator` 只实现了 `Evaluator<OnsenGame>`、`OnsenAction::Dig`
//! 特判）。把它泛型化要连带掀开 `umaai` 的调用签名，代价远大于另写一个薄壳。
//! 搜索核心 [`FlatSearch`] 本身已泛型化，拉面侧缺的只是最外层这一层。
//!
//! # 阶段门控
//!
//! 一局约 170 个决策点（实测单局：Train 69 / RamenSelect 61 / SpecialSelect 25 /
//! Event 15 / RegionSelect 3），全搜代价高。[`RamenSearchStages`] 允许只搜指定阶段，
//! 未选中的阶段直接转发给手写策略。这样既能压预算，也能单独测量
//! 「只搜 Train」/「只搜 RamenSelect」各自的边际收益。
//!
//! # 事件选项不走搜索
//!
//! [`Trainer::select_choice`] / [`Trainer::select_event_choice`] 的候选不来自
//! [`Game::list_actions`]，通用 rollout 入口 `apply_action` 吃不下，一律转发手写策略。
//!
//! # 合并动作搜索（`use_combined_ramen_select`）
//!
//! 打开时 `RamenSelect` 用 `list_combined_ramen_select_actions` 一次搜
//! `(ramen, targets)`，再把最优 `ramen` 映射回三阶段候选下标；紧随其后的
//! `SpecialSelect` 直接返回缓存的 targets，不再搜索。
//!
//! 这会改变对外层 rng 的消耗：`FlatSearch::search` 每次恰好消耗一次
//! `next_u64`。三阶段路径在 RamenSelect + SpecialSelect 各搜一次（2 次），
//! 合并路径只在 RamenSelect 搜一次（1 次）。随机序列整体位移，拉面基线作废。
//! 这是预期行为，不是 bug。关闭本开关即退回改动前的三阶段分别搜。

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering}
};

use anyhow::{Result, anyhow, bail};
use log::info;
use rand::prelude::StdRng;

use super::RamenHandwrittenTrainer;
use crate::{
    game::{
        Game, Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData},
    search::{FlatSearch, SearchConfig, SearchOutput}
};

/// 搜索哪些阶段的门控开关
///
/// 字段对应 [`RamenStage`] 中会产生多候选的阶段。未列出的阶段
/// （`Begin` / `Distribute` / `AfterTrain` / `NextTurn` / `Settlement`）
/// 不产生真正的选择空间，无需门控。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamenSearchStages {
    /// 训练/比赛选择（决策点最多，约占一局的 45%）
    pub train: bool,
    /// 吃哪碗面（约占 27%）
    pub ramen_select: bool,
    /// 隐藏风味用法
    pub special_select: bool,
    /// 年度地区选择
    ///
    /// **只覆盖第 2/3 年，且这是有意的**：第 1 年在 `run_begin` 的 turn 2 中途内联
    /// 调用，`game.stage` 仍是 `Begin`，不在阶段入口上；从那里开搜会让 rollout 的
    /// `apply → next()` 跳过 `run_begin` 后半段。第 1 年一律转发手写策略打分
    /// （手写侧已由 `ramen_effective_stage` 修好，不再恒选候选 0）。
    pub region_select: bool,
    /// 超级拉面选择
    ///
    /// **当前恒为死开关**：`run_super_ramen_select` 不接 trainer，固定选项二
    /// （`ramen/game.rs`）。保留字段是为了上游哪天把它交回 trainer 时不用改签名。
    pub super_ramen_select: bool
}

impl RamenSearchStages {
    /// 全部阶段都搜
    pub fn all() -> Self {
        Self {
            train: true,
            ramen_select: true,
            special_select: true,
            region_select: true,
            super_ramen_select: true
        }
    }

    /// 一个阶段都不搜（等价于纯手写策略，用于对照组）
    pub fn none() -> Self {
        Self {
            train: false,
            ramen_select: false,
            special_select: false,
            region_select: false,
            super_ramen_select: false
        }
    }

    /// 只搜训练阶段
    pub fn train_only() -> Self {
        Self {
            train: true,
            ..Self::none()
        }
    }

    /// 只搜吃面阶段
    pub fn ramen_only() -> Self {
        Self {
            ramen_select: true,
            ..Self::none()
        }
    }

    /// 解析逗号分隔的阶段名（CLI 用）
    ///
    /// 可用名：`all` / `none` / `train` / `ramen` / `special` / `region` / `super`。
    /// 例：`"train,ramen"`。
    ///
    /// # 三条严格性约定
    ///
    /// 这些输入直接决定实验分组，静默接受歧义输入会让对照组悄悄退化成纯手写策略，
    /// 是最难发现的一类错，故一律 `Err`：
    ///
    /// - 未知阶段名
    /// - 空串 / 只有逗号（否则静默得到 `none`）
    /// - `all` / `none` 与其他名混用（否则 `train,none` 与 `none,train` 结果不同）
    pub fn parse(spec: &str) -> Result<Self> {
        let names: Vec<&str> = spec.split(',').map(str::trim).filter(|n| !n.is_empty()).collect();
        if names.is_empty() {
            anyhow::bail!("搜索阶段为空（要表达「不搜索」请显式写 none）");
        }
        if names.iter().any(|n| matches!(*n, "all" | "none")) {
            if names.len() > 1 {
                anyhow::bail!("all / none 必须单独使用，不能与其他阶段名混用: {spec}");
            }
            return Ok(if names[0] == "all" { Self::all() } else { Self::none() });
        }
        let mut stages = Self::none();
        for name in names {
            match name {
                "train" => stages.train = true,
                "ramen" => stages.ramen_select = true,
                "special" => stages.special_select = true,
                "region" => stages.region_select = true,
                "super" => stages.super_ramen_select = true,
                other => {
                    anyhow::bail!("未知搜索阶段: {other}（可用 all/none/train/ramen/special/region/super）")
                }
            }
        }
        Ok(stages)
    }

    /// 该阶段是否应走搜索
    ///
    /// 取引用而非按值：`RamenStage` 未实现 `Copy`（上游类型，不在本次改动范围内）。
    pub fn contains(&self, stage: &RamenStage) -> bool {
        match stage {
            RamenStage::Train => self.train,
            RamenStage::RamenSelect => self.ramen_select,
            RamenStage::SpecialSelect => self.special_select,
            RamenStage::RegionSelect => self.region_select,
            RamenStage::SuperRamenSelect => self.super_ramen_select,
            _ => false
        }
    }
}

impl Default for RamenSearchStages {
    fn default() -> Self {
        Self::all()
    }
}

/// 最优动作的取分口径
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamenSelection {
    /// 结算评分（`calc_score`）
    Score,
    /// 计入 PT 偏好的评分（`calc_score_with_pt_favor`）
    Pt
}

/// 拉面杯 MCTS 训练员
///
/// 被门控选中的阶段走 [`FlatSearch`]，其余转发给内置的
/// [`RamenHandwrittenTrainer`]。搜索的 rollout 基策同样是手写策略
/// （由 `FlatSearchGame::default_rollout_trainer` 提供），因此本训练员
/// 是「手写策略 + 搜索」的严格叠加：门控全关时行为与纯手写策略一致。
pub struct RamenMctsTrainer {
    /// 扁平搜索器
    pub search: FlatSearch<RamenGame>,
    /// 未搜索阶段与事件选项的回退策略
    pub fallback: RamenHandwrittenTrainer,
    /// 搜索哪些阶段
    pub stages: RamenSearchStages,
    /// 取分口径
    pub selection: RamenSelection,
    /// 是否输出每步决策日志
    pub verbose: bool,
    /// `RamenSelect` 是否用合并动作（ramen + targets 一次决策）搜索
    ///
    /// 打开时 `SpecialSelect` 不再是独立决策点：`RamenSelect` 的搜索结果里
    /// 已经含 targets，`SpecialSelect` 直接返回缓存值。
    /// 关闭时退回三阶段分别搜（改动前行为）。
    ///
    /// **RNG 消耗会变**：`FlatSearch::search` 对外层 rng 恰好消耗一次 `next_u64`。
    /// 打开后 SpecialSelect 零消耗，随机序列整体位移，拉面基线作废。这是预期的。
    pub use_combined_ramen_select: bool,
    /// 最近一次搜索决策的候选统计文本（供 `LoggingTrainer` 写入决策日志）
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`。
    last_breakdown: Mutex<Option<String>>,
    /// 本训练员真正走过搜索的决策次数（转发给手写策略的不计）
    ///
    /// `Trainer::select_action` 只有 `&self`，故用原子量。用途是让「门控是否生效」
    /// 可观测：只看分数无法区分「搜索没提分」与「门控写错、根本没搜」。
    searched: AtomicUsize,
    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    ///
    /// 与 [`Self::searched`] 同理，用原子量是因为 `select_action` 只有 `&self`。
    /// 用途是钉住「缓存检查必须在门控早退之前」：若它被挪到早退之后，
    /// `special_select` 门控关闭时合并搜索选出的 targets 会被**静默丢弃**、
    /// 改由手写策略另选，而分数上看不出来——本计数器归零才看得见。
    combined_cache_hits: AtomicUsize,
    /// `RamenSelect` 合并搜索选出的 targets，供紧随其后的 `SpecialSelect` 复用
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`
    /// （与既有 `last_breakdown` 同理）。
    pending_combined_targets: Mutex<Option<[i32; 3]>>
}

impl RamenMctsTrainer {
    /// 用指定搜索配置创建（默认搜全部阶段、按 `score` 口径取最优、打开合并动作搜索）
    pub fn new(config: SearchConfig) -> Self {
        Self {
            search: FlatSearch::<RamenGame>::new(config),
            fallback: RamenHandwrittenTrainer::new(),
            stages: RamenSearchStages::all(),
            selection: RamenSelection::Score,
            verbose: false,
            use_combined_ramen_select: true,
            last_breakdown: Mutex::new(None),
            searched: AtomicUsize::new(0),
            combined_cache_hits: AtomicUsize::new(0),
            pending_combined_targets: Mutex::new(None)
        }
    }

    /// 本训练员真正走过搜索的决策次数
    pub fn searched_count(&self) -> usize {
        self.searched.load(Ordering::Relaxed)
    }

    /// 设置搜索阶段门控
    pub fn with_stages(mut self, stages: RamenSearchStages) -> Self {
        self.stages = stages;
        self
    }

    /// 设置取分口径
    pub fn with_selection(mut self, selection: RamenSelection) -> Self {
        self.selection = selection;
        self
    }

    /// 设置是否输出每步决策日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// `SpecialSelect` 直接命中合并搜索缓存的次数
    pub fn combined_cache_hits(&self) -> usize {
        self.combined_cache_hits.load(Ordering::Relaxed)
    }

    /// 设置 `RamenSelect` 是否走合并动作搜索
    pub fn with_combined_ramen_select(mut self, on: bool) -> Self {
        self.use_combined_ramen_select = on;
        self
    }

    /// 获取搜索配置
    pub fn config(&self) -> &SearchConfig {
        self.search.config()
    }

    /// 缓存本次搜索的候选统计（次数 / 均分 / 标准差 / PT 均分）
    fn stash_search_breakdown(&self, output: &SearchOutput<RamenAction>) {
        let text = output
            .actions
            .iter()
            .zip(output.action_results.iter())
            .enumerate()
            .map(|(i, (action, (res, res_pt)))| {
                format!(
                    "#{i} {action} n={} mean={:.0} sd={:.0} pt={:.0}",
                    res.count(),
                    res.mean(),
                    res.stdev(),
                    res_pt.mean()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = Some(text);
        }
    }

    /// 清空本次缓存（转发给手写策略时用，避免读到上一条搜索的陈旧文本）
    fn clear_breakdown(&self) {
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = None;
        }
    }

    /// 取出并清空合并搜索缓存的 targets
    fn take_pending_combined_targets(&self) -> Option<[i32; 3]> {
        match self.pending_combined_targets.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take()
        }
    }

    /// 写入合并搜索缓存的 targets（`None` 表示不吃面或不缓存）
    fn store_pending_combined_targets(&self, targets: Option<[i32; 3]>) {
        let mut slot = match self.pending_combined_targets.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner()
        };
        *slot = targets;
    }
}

impl Default for RamenMctsTrainer {
    fn default() -> Self {
        Self::new(SearchConfig::default())
    }
}

impl Trainer<RamenGame> for RamenMctsTrainer {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        // (A) SpecialSelect 命中缓存 —— 必须放在早退判断之前。
        // 候选可能只有 1 个，或 stages.special_select 关着，这两种情况都要消费缓存，
        // 否则会污染下一回合的 SpecialSelect。
        if game.stage == RamenStage::SpecialSelect {
            if let Some(t) = self.take_pending_combined_targets() {
                match actions.iter().position(|a| a.special_targets == Some(t)) {
                    Some(idx) => {
                        self.combined_cache_hits.fetch_add(1, Ordering::Relaxed);
                        self.clear_breakdown();
                        return Ok(idx);
                    }
                    None => {
                        bail!(
                            "SpecialSelect 缓存未命中: 缓存 targets={t:?}，实际候选=[{}]",
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.special_targets))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
        }

        // (C) RamenSelect 每次做决策时先无条件清一次，防止上一回合遗留
        if game.stage == RamenStage::RamenSelect {
            self.store_pending_combined_targets(None);
        }

        // 单候选无选择空间，跑搜索纯属浪费预算
        // 门控**必须**用未经纠正的 `game.stage`
        //
        // 第 1 年地区选择由 `run_begin` 在 turn 2 中途内联调用，此时 `game.stage`
        // 仍是 `Begin`。若按 `ramen_effective_stage` 把它纠正成 `RegionSelect` 去开搜索，
        // `simulate_common` 的 `apply → while next()` 会从 Begin 直接跳到 Distribute，
        // **跳过 `run_begin` 后半段**（隐藏风味分配、refresh_mind、回合开始事件链），
        // rollout 评的是一个不存在的局面。`sampler.rs` 对根局面的约束同理。
        // 纠正只用于把决策转发给手写策略打分（`fallback` 内部会做）。
        if actions.len() <= 1 || !self.stages.contains(&game.stage) {
            self.clear_breakdown();
            return self.fallback.select_action(game, actions, rng);
        }

        // (B) RamenSelect 走合并搜索（排除 race_turn：那边 list_actions 是比赛动作）
        if self.use_combined_ramen_select && game.stage == RamenStage::RamenSelect && !game.is_race_turn()
        {
            let combined = game.list_combined_ramen_select_actions();
            if combined.len() > 1 {
                self.searched.fetch_add(1, Ordering::Relaxed);
                let output = self.search.search(game, &combined, rng)?;
                let idx = match self.selection {
                    RamenSelection::Score => output.best_action_idx,
                    RamenSelection::Pt => output.best_action_pt_idx()
                };
                let best = combined
                    .get(idx)
                    .ok_or_else(|| anyhow!("合并搜索最优下标 {idx} 超出候选数 {}", combined.len()))?;
                // 不吃面时 next() 会直接推到 Train，不会有 SpecialSelect；留缓存会污染下一回合
                if best.ramen.is_none() {
                    self.store_pending_combined_targets(None);
                } else {
                    self.store_pending_combined_targets(best.special_targets);
                }
                self.stash_search_breakdown(&output);
                if self.verbose {
                    let (res, _) = &output.action_results[idx];
                    info!(
                        "[MCTS][回合 {}] 阶段 {:?} 合并 {} 候选 -> combined#{idx} {} (mean={:.0} n={})",
                        game.turn(),
                        game.stage,
                        combined.len(),
                        best,
                        res.mean(),
                        res.count()
                    );
                }
                match actions.iter().position(|a| a.ramen == best.ramen) {
                    Some(three_idx) => return Ok(three_idx),
                    None => {
                        bail!(
                            "RamenSelect 合并搜索结果在三阶段候选中找不到: best.ramen={:?}，实际候选=[{}]",
                            best.ramen,
                            actions
                                .iter()
                                .map(|a| format!("{:?}", a.ramen))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
            // combined.len() <= 1：不走合并，落回原逻辑
        }

        self.searched.fetch_add(1, Ordering::Relaxed);
        let output = self.search.search(game, actions, rng)?;
        let idx = match self.selection {
            RamenSelection::Score => output.best_action_idx,
            RamenSelection::Pt => output.best_action_pt_idx()
        };
        self.stash_search_breakdown(&output);
        if self.verbose {
            let (res, _) = &output.action_results[idx];
            info!(
                "[MCTS][回合 {}] 阶段 {:?} {} 候选 -> #{idx} {} (mean={:.0} n={})",
                game.turn(),
                game.stage,
                actions.len(),
                actions[idx],
                res.mean(),
                res.count()
            );
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_breakdown();
        self.fallback.select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_breakdown();
        self.fallback.select_event_choice(game, event, choices, rng)
    }

    /// 搜索决策返回候选统计；转发决策返回手写策略自己的分解
    fn last_breakdown(&self) -> Option<String> {
        match self.last_breakdown.lock().ok().and_then(|slot| slot.clone()) {
            Some(text) => Some(text),
            None => self.fallback.last_breakdown()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::{GAMECONSTANTS, init_global},
        global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 准备一局固定种子的拉面局面
    fn setup(seed: u64) -> Result<(RamenGame, StdRng)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let (decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        Ok((game, decision_rng))
    }

    /// 阶段门控字符串解析
    #[test]
    fn test_search_stages_parse() -> Result<()> {
        let mut c = Checks::new();
        let s = RamenSearchStages::parse("train,ramen")?;
        println!("parse(train,ramen) = {s:?}");
        c.check(
            s.train && s.ramen_select && !s.special_select && !s.region_select && !s.super_ramen_select,
            "只有 train / ramen_select 为真"
        );
        c.check(s.contains(&RamenStage::Train), "Train 命中");
        c.check(!s.contains(&RamenStage::SpecialSelect), "SpecialSelect 不命中");
        c.check(!RamenSearchStages::all().contains(&RamenStage::Begin), "Begin 永不命中");

        // 四类必须报错的输入：静默接受会让实验对照组悄悄退化成纯手写策略
        c.check(RamenSearchStages::parse("train,bogus").is_err(), "未知阶段名报错");
        c.check(RamenSearchStages::parse("").is_err(), "空串报错（不静默当 none）");
        c.check(RamenSearchStages::parse(" , ").is_err(), "只有逗号报错");
        c.check(RamenSearchStages::parse("train,none").is_err(), "none 与其他名混用报错");
        c.check(RamenSearchStages::parse("all,train").is_err(), "all 与其他名混用报错");
        c.check(RamenSearchStages::parse("all")?.train, "单独 all 有效");
        c.check(!RamenSearchStages::parse("none")?.train, "单独 none 有效");
        c.finish()
    }

    /// 第 1 年地区选择：手写策略必须打分，MCTS 门控必须**不**搜
    ///
    /// 两个方向都要钉住：
    /// - `ramen_effective_stage` 把 `Begin` + RegionSelect 动作纠正为 `RegionSelect`，
    ///   否则手写策略落到默认分支恒选候选 0（第 1 年从未经过打分）；
    /// - MCTS 门控**不得**用纠正后的阶段，否则会在 `run_begin` 中途开搜，
    ///   rollout 的 `apply -> next()` 会跳过 `run_begin` 后半段。
    #[test]
    fn test_year1_region_scored_but_not_searched() -> Result<()> {
        use crate::{
            game::ramen::{Operation, rules::get_region_combinations},
            trainer::ramen_handwritten_trainer::ramen_effective_stage
        };

        let mut c = Checks::new();
        let (game, _rng) = setup(42)?;
        // 第 1 年地区选择在 run_begin 内部触发、外部观察不到，
        // 故直接构造该决策点的候选集来验证两个判定
        let combos = get_region_combinations(0)?;
        let actions: Vec<RamenAction> = combos
            .iter()
            .map(|&combo| RamenAction::no_ramen(Operation::RegionSelect(combo)))
            .collect();
        println!("game.stage={:?} 第 1 年地区候选={}", game.stage, actions.len());

        let eff = ramen_effective_stage(&game, &actions);
        println!("ramen_effective_stage = {eff:?}");
        c.check(eff == RamenStage::RegionSelect, "有效阶段纠正为 RegionSelect（手写据此打分）");
        c.check(game.stage == RamenStage::Begin, "raw game.stage 仍是 Begin");

        let gate = RamenSearchStages::all();
        c.check(
            !gate.contains(&game.stage),
            "MCTS 门控用 raw stage，第 1 年不进搜索（否则 rollout 跳过 run_begin 后半）"
        );
        c.check(gate.contains(&eff), "纠正后的阶段本身被门控覆盖（说明上一条不是巧合）");
        c.finish()
    }

    /// 门控全关时必须与纯手写策略**逐位一致**
    ///
    /// 这是实验的对照组正确性前提：若两者不一致，说明 MCTS 壳自己额外消耗了
    /// 随机流或改了决策，后续「搜索提分多少」的差值就无从归因。
    #[test]
    fn test_stages_none_matches_handwritten() -> Result<()> {
        let seed = 42;

        let (mut game_hw, mut rng_hw) = setup(seed)?;
        game_hw.run_full_game(&RamenHandwrittenTrainer::new(), &mut rng_hw)?;
        let score_hw = game_hw.uma.calc_score();

        let (mut game_mcts, mut rng_mcts) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(8))
            .with_stages(RamenSearchStages::none());
        game_mcts.run_full_game(&trainer, &mut rng_mcts)?;
        let score_mcts = game_mcts.uma.calc_score();

        let mut c = Checks::new();
        println!("手写={score_hw} / MCTS(stages=none)={score_mcts}");
        c.check(score_hw == score_mcts, "门控全关 == 纯手写策略");
        println!("  五维 {:?} vs {:?}", game_hw.uma.five_status, game_mcts.uma.five_status);
        c.check(game_hw.uma.five_status == game_mcts.uma.five_status, "五维一致");
        c.check(game_hw.uma.skill_pt == game_mcts.uma.skill_pt, "技能点一致");
        c.check(game_hw.ramen.scenario_pt == game_mcts.ramen.scenario_pt, "剧本 PT 一致");
        c.check(trainer.searched_count() == 0, "门控全关时一次搜索都没发生");
        c.finish()
    }

    /// 只搜训练阶段跑通整局（小预算冒烟）
    #[test]
    fn test_mcts_train_only_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(RamenSearchStages::train_only());
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let score = game.uma.calc_score();
        println!(
            "MCTS(train, search_n=4) 整局: 回合={} 评分={} ({}) 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            global!(GAMECONSTANTS).get_rank_name(score)
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        // 「末次决策有 breakdown」几乎恒真（末步必是转发、回落到手写分解），
        // 改为统计整局真正走过搜索的次数——这才是门控生效的证据
        println!("  整局走搜索的决策数={}", trainer.searched_count());
        c.check(trainer.searched_count() > 0, "确实走过搜索");
        c.check(trainer.searched_count() <= 80, "只搜 Train（约 69 个点），没有蔓延到其他阶段");
        c.finish()
    }

    /// rollout 的根动作必须走策略流，不能走通用 `apply_action`
    ///
    /// 真实对局中 `run_train` 用 `apply_action_with_strategy`（优先用局面内策略流），
    /// 而旧 `simulate_common` 直接 `apply_action(action, rng)`。本测试扫过整局所有
    /// 多候选 Train 决策点，统计两条路径跑到终局的分数有多少个点不同——
    /// 若一个都不同不了，说明该修复是空操作，需要重新评估。
    #[test]
    fn test_root_action_uses_strategy_stream() -> Result<()> {
        use rand::SeedableRng;

        use crate::search::FlatSearchGame;

        let (mut game, mut rng) = setup(42)?;
        let hw = RamenHandwrittenTrainer::new();
        let seed = 12345u64;
        let (mut checked, mut differ) = (0usize, 0usize);
        let mut first_diff = None;

        while game.next() {
            if matches!(game.stage, RamenStage::Train) {
                let actions = game.list_actions()?;
                if actions.len() > 1 {
                    // 同一个动作、同一个种子，两条 apply 路径各自跑到终局
                    let mut scores = [0i32; 2];
                    for (k, score) in scores.iter_mut().enumerate() {
                        let mut g = game.fork_for_rollout(seed);
                        let mut r = StdRng::seed_from_u64(seed);
                        if k == 0 {
                            g.apply_action(&actions[0], &mut r)?;
                        } else {
                            g.apply_root_action(&actions[0], &mut r)?;
                        }
                        while g.next() {
                            g.run_stage(&hw, &mut r)?;
                        }
                        *score = g.uma.calc_score();
                    }
                    checked += 1;
                    if scores[0] != scores[1] {
                        differ += 1;
                        first_diff.get_or_insert((game.turn(), scores[0], scores[1]));
                    }
                }
            }
            game.run_stage(&hw, &mut rng)?;
        }

        println!("扫过 {checked} 个多候选 Train 决策点，其中 {differ} 个两条路径终局分数不同");
        if let Some((turn, a, b)) = first_diff {
            println!("  首个差异: 回合 {turn} 通用={a} 策略流={b}");
        }
        let mut c = Checks::new();
        c.check(differ > 0, "修复非空操作（至少一个 Train 决策点两条路径结果不同）");
        c.finish()
    }

    /// 同种子两次整局结果一致（搜索层的 CRN 种子由传入 rng 派生）
    #[test]
    fn test_mcts_reproducible() -> Result<()> {
        let seed = 7;
        let mut scores = Vec::new();
        for _ in 0..2 {
            let (mut game, mut rng) = setup(seed)?;
            let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
                .with_stages(RamenSearchStages::train_only());
            game.run_full_game(&trainer, &mut rng)?;
            scores.push(game.uma.calc_score());
        }
        let mut c = Checks::new();
        println!("两次评分: {scores:?}");
        c.check(scores[0] == scores[1], "可复现");
        c.finish()
    }

    /// 吃面 + 隐藏风味两阶段都搜（P1.2 / P1.3 对照与测量用）
    fn ramen_and_special_stages() -> RamenSearchStages {
        RamenSearchStages {
            ramen_select: true,
            special_select: true,
            ..RamenSearchStages::none()
        }
    }

    /// 硬性验收 1 的对照尺子：`use_combined_ramen_select = false` 必须与改动前逐位相同
    ///
    /// 改动前（字段尚不存在、等价于三阶段分别搜）实测：
    /// 评分=55153 五维=[2958, 1742, 2200, 866, 1112] skill_pt=7390 scenario_pt=0 searched_count=46
    #[test]
    fn test_combined_gate_off_full_game() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(false);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let searched = trainer.searched_count();
        println!(
            "gate-off 整局: 回合={} 评分={} 五维={:?} skill_pt={} scenario_pt={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            game.uma.five_status,
            game.uma.skill_pt,
            game.ramen.scenario_pt,
            searched
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        // 2026-08-25 更新：不在判定与得意率解耦 + 地区分身缺席优先，模拟数值变化，基准重抓
        c.check(score == 56916, "评分与改动前逐位相同");
        c.check(
            game.uma.five_status == [2958, 2150, 2200, 1091, 706],
            "五维与改动前逐位相同"
        );
        c.check(game.uma.skill_pt == 7685, "技能点与改动前逐位相同");
        c.check(game.ramen.scenario_pt == 0, "剧本 PT 与改动前逐位相同");
        c.check(searched == 43, "searched_count 与改动前逐位相同");
        c.finish()
    }

    /// 默认打开合并搜索；链式 setter 能关掉
    #[test]
    fn test_combined_default_on() -> Result<()> {
        // `RamenMctsTrainer::default()` 会构造 `HandwrittenEvaluator`，后者
        // `load_onsen_order().expect(..)` 依赖工作目录与全局数据；不初始化则本测试
        // 只在别的测试先跑过时才碰巧通过（顺序依赖）。
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let t = RamenMctsTrainer::default();
        println!("default use_combined_ramen_select = {}", t.use_combined_ramen_select);
        let mut c = Checks::new();
        c.check(t.use_combined_ramen_select, "new()/default 默认打开");
        let t2 = t.with_combined_ramen_select(false);
        println!(
            "after with_combined_ramen_select(false) = {}",
            t2.use_combined_ramen_select
        );
        c.check(!t2.use_combined_ramen_select, "setter 关闭");
        c.finish()
    }

    /// 只统计不干预的包装训练员：记录各阶段调用次数，以及其中真正走过搜索的次数
    struct CountingTrainer {
        /// 被包装的 MCTS 训练员
        inner: RamenMctsTrainer,
        /// `RamenSelect` 的 `select_action` 调用次数
        ramen_select_calls: AtomicUsize,
        /// `SpecialSelect` 的 `select_action` 调用次数
        special_select_calls: AtomicUsize,
        /// `RamenSelect` 中真正走过搜索的次数
        ramen_select_searches: AtomicUsize,
        /// `SpecialSelect` 中真正走过搜索的次数
        special_select_searches: AtomicUsize
    }

    impl CountingTrainer {
        /// 包装一个已构造好的 `RamenMctsTrainer`
        fn wrap(inner: RamenMctsTrainer) -> Self {
            Self {
                inner,
                ramen_select_calls: AtomicUsize::new(0),
                special_select_calls: AtomicUsize::new(0),
                ramen_select_searches: AtomicUsize::new(0),
                special_select_searches: AtomicUsize::new(0)
            }
        }
    }

    impl Trainer<RamenGame> for CountingTrainer {
        fn select_action(
            &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
        ) -> Result<usize> {
            let before = self.inner.searched_count();
            let idx = self.inner.select_action(game, actions, rng)?;
            let did_search = self.inner.searched_count() > before;
            match game.stage {
                RamenStage::RamenSelect => {
                    self.ramen_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.ramen_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                RamenStage::SpecialSelect => {
                    self.special_select_calls.fetch_add(1, Ordering::Relaxed);
                    if did_search {
                        self.special_select_searches.fetch_add(1, Ordering::Relaxed);
                    }
                }
                _ => {}
            }
            Ok(idx)
        }

        fn select_choice(
            &self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_choice(game, choices, rng)
        }

        fn select_event_choice(
            &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            self.inner.select_event_choice(game, event, choices, rng)
        }

        fn last_breakdown(&self) -> Option<String> {
            self.inner.last_breakdown()
        }
    }

    /// 硬性验收 2：合并开启时 SpecialSelect 全程不再被搜
    #[test]
    fn test_combined_on_skips_special_search() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let inner = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(ramen_and_special_stages())
            .with_combined_ramen_select(true);
        let trainer = CountingTrainer::wrap(inner);
        let start = std::time::Instant::now();
        game.run_full_game(&trainer, &mut rng)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let score = game.uma.calc_score();
        let ramen_calls = trainer.ramen_select_calls.load(Ordering::Relaxed);
        let special_calls = trainer.special_select_calls.load(Ordering::Relaxed);
        let ramen_searches = trainer.ramen_select_searches.load(Ordering::Relaxed);
        let special_searches = trainer.special_select_searches.load(Ordering::Relaxed);
        let searched = trainer.inner.searched_count();
        println!(
            "gate-on 整局: 回合={} 评分={} searched_count={} 耗时={elapsed:.0}ms",
            game.turn(),
            score,
            searched
        );
        println!(
            "  RamenSelect 调用={ramen_calls} 搜索={ramen_searches} / SpecialSelect 调用={special_calls} 搜索={special_searches}"
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        c.check(ramen_calls > 0, "RamenSelect 被调用过");
        c.check(ramen_searches > 0, "RamenSelect 走过搜索");
        c.check(special_calls > 0, "SpecialSelect 被调用过（缓存命中路径）");
        c.check(special_searches == 0, "SpecialSelect 从未触发搜索");
        c.check(searched == ramen_searches, "整局 searched_count 全部来自 RamenSelect");
        c.finish()
    }

    /// 只搜 `ramen`、不搜 `special` 时，合并搜索选出的 targets 仍必须被采用
    ///
    /// 钉「缓存检查必须在门控早退之前」：挪到早退之后，targets 会被静默丢弃、
    /// 改由手写策略另选，分数上看不出来，只有 `combined_cache_hits()` 归零才暴露。
    #[test]
    fn test_combined_cache_used_when_special_gate_off() -> Result<()> {
        let seed = 42;
        let (mut game, mut rng) = setup(seed)?;
        let stages = RamenSearchStages {
            ramen_select: true,
            ..RamenSearchStages::none()
        };
        let trainer = RamenMctsTrainer::new(SearchConfig::default().with_search_n(4).with_ucb(false))
            .with_stages(stages)
            .with_combined_ramen_select(true);
        game.run_full_game(&trainer, &mut rng)?;
        let hits = trainer.combined_cache_hits();
        println!(
            "special 门控关: 回合={} 评分={} searched_count={} combined_cache_hits={hits}",
            game.turn(),
            game.uma.calc_score(),
            trainer.searched_count()
        );
        let mut c = Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(trainer.searched_count() > 0, "RamenSelect 走过合并搜索");
        c.check(hits > 0, "SpecialSelect 必须命中合并缓存（门控关也要用）");
        c.finish()
    }
}
