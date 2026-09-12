//! 拉面杯手写策略训练员（测试壳）
//!
//! 直接使用 [`RamenPolicy`](crate::game::ramen::policy::RamenPolicy) 的确定性打分
//! 选择动作，不经过 MCTS 搜索。用途（计划 §2）：跑完整局验证策略效果 + 作为 MCTS
//! rollout 基策的调参载体，本身不是交付主体。
//!
//! 与旧 `HandwrittenTrainer`（温泉杯）无架构耦合：本实现直接在拉面杯规则层
//! （`RamenGame` / `RamenAction` / `policy.rs`）上重新实现。
//!
//! 每次决策后把各候选的评分分解（`RamenPolicyOutput::breakdown`）缓存，
//! 供 `LoggingTrainer` 写入决策日志 breakdown 列（调参用，见 `Trainer::last_breakdown`）。

use std::sync::Mutex;

use anyhow::Result;
use log::info;
use rand::prelude::StdRng;

use crate::{
    game::{
        Game,
        Trainer,
        ramen::{Operation, RamenAction, RamenGame, RamenStage, policy::RamenPolicy, policy::RamenPolicyOutput}
    },
    gamedata::{EventChoice, EventData},
    output::DecisionInfo as DecisionInfoProto
};

/// 决策所属的**有效阶段**（按动作类型纠正 `game.stage`）
///
/// 第 1 年地区选择已是独立的 [`RamenStage::RegionSelect`]（turn 2 在 `Begin`
/// 前半段之后），不再发生在 `Begin` 内部。本函数保留作防御性兼容：若有人
/// 仍把 `RegionSelect` 动作塞进其它阶段，按 [`Operation`] 纠正，避免落到
/// `select_action` 的 `_ => (0, vec![])` 恒选候选 0。
///
/// 超级拉面同样按动作类型纠正：缺分支会把选项二静默变成选项一。
///
/// 判定用 `any` 而非 `all`：地区选择的候选集是同构的，命中一个即可；
/// 用 `all` 会在候选集混入其他动作时静默退回默认分支。
pub fn ramen_effective_stage(game: &RamenGame, actions: &[RamenAction]) -> RamenStage {
    if actions.iter().any(|a| matches!(a.operation, Operation::RegionSelect(_))) {
        return RamenStage::RegionSelect;
    }
    if actions.iter().any(|a| matches!(a.operation, Operation::SuperRamenSelect(_))) {
        return RamenStage::SuperRamenSelect;
    }
    game.stage.clone()
}

/// 拉面杯手写策略训练员
pub struct RamenHandwrittenTrainer {
    /// 策略核心（参数化配置 + 各阶段打分）
    pub policy: RamenPolicy,
    /// 是否输出每步决策日志（整局跑批时建议关闭）
    pub verbose: bool,
    /// 最近一次决策的评分分解文本（供 LoggingTrainer 提取进决策日志）
    ///
    /// 用 `Mutex` 而非 `RefCell`：搜索层要求 `Trainer: Sync`（rayon 跨线程共享同一个
    /// rollout 决策器），`RefCell` 会让整个 `FlatSearch<RamenGame>` 失去 `Sync`。
    /// 单局日志场景无竞争，加锁开销可忽略。
    last_breakdown: Mutex<Option<String>>,
    /// 上一次决策的协议字段（选中下标 + 各候选评分），供 [`Trainer::last_decision`](crate::game::Trainer::last_decision) 读取
    ///
    /// 独立于原因文本采集写入；早退路径清成 `None`，不暴露上一决策的数据。
    last_decision_summary: Mutex<Option<LastDecisionSummary>>
}

/// 手写策略上一次决策的最小摘要
///
/// 2026-09 扩展：新增 `descriptions` 字段——AIRedirector 仅靠 `action_index` 数字
/// 无法映射拉面组合动作名，必须挂描述才能展示。`descriptions` 与 `outputs`
/// 严格同长同序（同源于 `select_action` 的 `actions` 入参）。
#[derive(Debug, Clone)]
struct LastDecisionSummary {
    /// `select_action` 返回的下标（caller 视角，与传入 `actions` 数组一致）
    chosen_idx: usize,
    /// 各候选的 `RamenPolicyOutput`（与传入 `actions` 数组严格同序；长度不一致视为异常）
    outputs: Vec<RamenPolicyOutput>,
    /// 各候选的可读描述（与 `outputs` 严格同长同序，按 `actions` 入参顺序）
    descriptions: Vec<String>
}

impl RamenHandwrittenTrainer {
    /// 创建默认配置的手写策略训练员
    pub fn new() -> Self {
        Self {
            policy: RamenPolicy::default(),
            verbose: false,
            last_breakdown: Mutex::new(None),
            last_decision_summary: Mutex::new(None)
        }
    }

    /// 创建 rollout 专用实例：关闭评分分解和原因文本采集，保留协议摘要。
    pub fn for_rollout() -> Self {
        let mut trainer = Self::new();
        trainer.policy.collect_details = false;
        trainer
    }

    /// 使用指定策略核心创建
    pub fn with_policy(policy: RamenPolicy) -> Self {
        Self {
            policy,
            verbose: false,
            last_breakdown: Mutex::new(None),
            last_decision_summary: Mutex::new(None)
        }
    }

    /// 速度特化配置
    pub fn speed_build() -> Self {
        Self::with_policy(RamenPolicy::speed_build())
    }

    /// 设置是否输出每步决策日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 缓存本次决策的评分分解（各候选 `score + reason` 摘要）
    fn stash_breakdown(&self, outputs: &[RamenPolicyOutput]) {
        if !self.policy.collect_details {
            return;
        }
        let text = outputs
            .iter()
            .enumerate()
            .map(|(i, out)| format!("#{i} {:.0}[{}]", out.score, out.reason))
            .collect::<Vec<_>>()
            .join(" | ");
        // 锁中毒说明别处 panic 过；此处只是调试文本，静默跳过而非把育成流程一起带崩
        if let Ok(mut slot) = self.last_breakdown.lock() {
            *slot = Some(text);
        }
    }

    /// 缓存本次决策的协议摘要（各候选 score + 选中下标），供 `last_decision` 读取
    ///
    /// 与 [`Self::stash_breakdown`] 解耦：rollout 路径关闭 breakdown 采集时
    /// 仍须保留协议摘要——后者是协议层契约，不该被 rollout 性能优化关掉。
    ///
    /// 2026-09 扩展：`actions` 入参新增——把候选描述缓存到 `descriptions`，
    /// 让 `last_decision` 输出给 AIRedirector 映射动作名。
    fn stash_decision_summary(
        &self, idx: usize, outputs: &[RamenPolicyOutput], actions: &[<crate::game::ramen::RamenGame as crate::game::Game>::Action]
    ) {
        let descriptions: Vec<String> = actions.iter().map(|a| a.to_string()).collect();
        if let Ok(mut slot) = self.last_decision_summary.lock() {
            *slot = Some(LastDecisionSummary {
                chosen_idx: idx,
                outputs: outputs.to_vec(),
                descriptions
            });
        }
    }

    /// 清空 `last_decision_summary`（早退 / 转发 fallback 时调用）
    ///
    /// 与 [`Self::clear_breakdown`] 配套——单候选走早退时不该让 caller 看到
    /// 上一次决策的协议数据。
    fn clear_decision_summary(&self) {
        if let Ok(mut slot) = self.last_decision_summary.lock() {
            *slot = None;
        }
    }
}

impl Default for RamenHandwrittenTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer<RamenGame> for RamenHandwrittenTrainer {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], _rng: &mut StdRng
    ) -> Result<usize> {
        // 单个候选直接返回（无选择空间）
        if actions.len() <= 1 {
            if self.policy.collect_details {
                if let Ok(mut slot) = self.last_breakdown.lock() {
                    *slot = Some(format!("仅1候选: {}", actions[0]));
                }
            }
            self.clear_decision_summary();
            return Ok(0);
        }
        let (idx, outputs) = match ramen_effective_stage(game, actions) {
            RamenStage::RamenSelect => self.policy.decide_ramen(game, actions)?,
            RamenStage::SpecialSelect => self.policy.decide_special(game, actions)?,
            RamenStage::Train => self.policy.decide_train(game, actions)?,
            // 地区选择：第 1/2/3 年分别在 turn 2/23/47 触发（第 3 年 fixed 为单候选，走上方早退）
            // turn → year_idx 与 `region_archive_year_idx` 一致，拆 Begin 后 turn 号不变
            RamenStage::RegionSelect => {
                let year_idx = match game.turn() {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => 0
                };
                self.policy.decide_region(game, year_idx, actions)?
            }
            // 查找 SuperRamenSelect(1) 的候选位置，不是硬编码返回下标 1
            RamenStage::SuperRamenSelect => self.policy.decide_super_ramen(game, actions)?,
            // 其他阶段（Begin/Distribute/AfterTrain 等）不应有多个候选
            _ => (0, vec![])
        };
        self.stash_breakdown(&outputs);
        self.stash_decision_summary(idx, &outputs, actions);
        if self.verbose {
            info!(
                "[手写][回合 {}] 阶段 {:?} 选择: {}",
                game.turn(),
                game.stage,
                actions.get(idx).map(|a| a.to_string()).unwrap_or_default()
            );
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let (idx, outputs) = self.policy.decide_event(game, choices)?;
        self.stash_breakdown(&outputs);
        // 事件选择：actions 在 trait 里类型为 `&[Vec<EventChoice>]`，与 RamenAction 不可转——这里
        // 不挂候选描述（事件选择阶段对 AIRed 端不展示候选名）。stash_decision_summary
        // 第三个参数改用空 Vec 兜底，避免 protocol 字段错位。
        self.stash_decision_summary(idx, &outputs, &[]);
        if self.verbose {
            info!("[手写][回合 {}] 事件选择: {}", game.turn(), idx + 1);
        }
        Ok(idx)
    }

    fn select_event_choice(
        &self, game: &RamenGame, _event: &EventData, choices: &[Vec<EventChoice>], _rng: &mut StdRng
    ) -> Result<usize> {
        let (idx, outputs) = self.policy.decide_event(game, choices)?;
        self.stash_breakdown(&outputs);
        // 同 select_choice：事件选择阶段不挂候选描述（与 RamenAction 类型不可转）
        self.stash_decision_summary(idx, &outputs, &[]);
        if self.verbose {
            info!("[手写][回合 {}] 事件选择: {}", game.turn(), idx + 1);
        }
        Ok(idx)
    }

    fn last_breakdown(&self) -> Option<String> {
        self.last_breakdown.lock().ok().and_then(|slot| slot.clone())
    }

    /// 上一次手写策略决策的协议格式
    ///
    /// 候选评分按 score 降序截断到 `reason_max_display`（手写策略没有局数概念，
    /// `candidate_n` 留空）。`reason` 暂留空（用户拍板"其他剧本暂留空"——拉面 MCTS
    /// 走 [`crate::output::reason`] 的维度差分析；手写策略的 reason 输出语义尚未
    /// 拍板，留待后续步骤）。
    ///
    /// `score_breakdown` 直接挂中选者的 `RamenPolicyOutput::breakdown` 调参维度。
    fn last_decision(&self) -> Option<DecisionInfoProto> {
        let summary = self.last_decision_summary.lock().ok()?.clone()?;
        if summary.outputs.is_empty() || summary.chosen_idx >= summary.outputs.len() {
            return None;
        }
        let chosen = &summary.outputs[summary.chosen_idx];
        // 手写策略不属于搜索层，`SearchConfig::reason_max_display` 与手写 reason 显示语义无关，
        // 这里硬编码 5：与 `SearchConfig::default()` 同口径，保持屏幕显示与协议层一致。
        let max_n = 5usize;

        // 按 score 降序排序并截断；手写策略默认按该字段硬编码 5（不读 SearchConfig）。
        // 不读 SearchConfig：手写策略不属于搜索层，调参配置改的是搜索预算，与 reason 显示无关。
        let mut indexed: Vec<(usize, f32)> = summary.outputs.iter().enumerate().map(|(i, o)| (i, o.score)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let ordered: Vec<(usize, f32)> = if indexed.len() <= max_n {
            indexed
        } else {
            indexed.truncate(max_n);
            if indexed.iter().any(|(i, _)| *i == summary.chosen_idx) {
                indexed
            } else {
                let mut v = vec![(summary.chosen_idx, chosen.score)];
                v.extend(indexed);
                v
            }
        };

        let action_index = ordered.iter().position(|(i, _)| *i == summary.chosen_idx).unwrap_or(0);

        // 2026-09 简化：`score_breakdown` 字段已从 DecisionInfo 删除——
        // breakdown 数据改走 `last_breakdown()` 方法（`LoggingTrainer` 调参日志仍用）
        //
        // 候选描述按 ordered 顺序取（与 candidate_scores 严格同长同序同截断）——
        // 拉面组合动作靠此字段让 AIRedirector 映射动作名
        let candidate_descriptions: Vec<String> = ordered
            .iter()
            .map(|(i, _)| summary.descriptions[*i].clone())
            .collect();
        Some(DecisionInfoProto {
            action_index,
            score: chosen.score,
            // decision_kind 由 main.rs 外部填——trainer 不感知 stage
            decision_kind: String::new(),
            candidate_scores: ordered.iter().map(|(_, s)| *s).collect(),
            candidate_descriptions,
            candidate_n: vec![],
            scenario_extra: None
        })
    }
}
#[cfg(test)]
mod tests {
    //use rand::SeedableRng;

    use super::*;
    use crate::{
        game::ramen::RamenGame,
        gamedata::{GAMECONSTANTS, init_global},
        global,
        utils::{get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 完整 77 回合跑通（固定种子可复现），输出关键结局指标
    #[test]
    fn test_handwritten_full_game() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let seed: u64 = 42;
        let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let trainer = RamenHandwrittenTrainer::new();
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        game.run_full_game(&trainer, &mut decision_rng)?;

        let score = game.uma.calc_score();
        let rank = global!(GAMECONSTANTS).get_rank_name(score);
        println!(
            "手写策略完整局: 回合={} 评分={} ({}) RMJ={:?} 吃面={} 五维={:?} super_ramen={:?}",
            game.turn(),
            score,
            rank,
            game.ramen.rmj_results,
            game.ramen.eat_count,
            game.uma.five_status,
            game.ramen.super_ramen
        );
        let mut c = crate::utils::Checks::new();
        c.check(game.turn() == 77, "跑满 77 回合");
        c.check(score > 0, "评分为正");
        c.check(game.ramen.super_ramen == Some(1), "手写回退必须钉死选项二");
        c.finish()
    }

    /// 确定性：同 seed 两次整局，事件选择与动作选择均一致（决策序列可复现）
    #[test]
    fn test_handwritten_reproducible() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let seed: u64 = 7;
        let mut scores = Vec::new();
        for _ in 0..2 {
            let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
            let trainer = RamenHandwrittenTrainer::new();
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.set_rule_master(rule_master);
            game.run_full_game(&trainer, &mut decision_rng)?;
            scores.push(game.uma.calc_score());
        }
        println!("两次评分: {:?}", scores);
        assert_eq!(scores[0], scores[1]);
        Ok(())
    }

    /// last_decision 保留候选评分和描述；rollout 关闭原因文本后决策与协议输出一致。
    ///
    /// 手写策略没有局数概念（`RamenPolicyOutput` 无 count 字段），`candidate_n` 必须为空。
    #[test]
    fn test_handwritten_last_decision() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let seed: u64 = 42;
        let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let trainer = RamenHandwrittenTrainer::new();
        let rollout = RamenHandwrittenTrainer::for_rollout();
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);

        // 跑一局，每步决策都查 last_decision：协议字段格式守门
        let mut decisions = 0usize;
        let mut had_breakdown = 0usize;
        let mut same_choices = true;
        let mut same_protocol = true;
        while game.next() {
            let actions = game.list_actions()?;
            let mut rollout_rng = decision_rng.clone();
            let idx = trainer.select_action(&game, &actions, &mut decision_rng)?;
            same_choices &= rollout.select_action(&game, &actions, &mut rollout_rng)? == idx;
            let info = trainer.last_decision();
            same_protocol &= info.as_ref() == rollout.last_decision().as_ref();
            had_breakdown += usize::from(trainer.last_breakdown().is_some());
            if let Some(info) = info {
                decisions += 1;
                let mut c = crate::utils::Checks::new();
                c.check(!info.candidate_scores.is_empty(), "候选评分非空");
                c.check(
                    info.candidate_n.is_empty(),
                    "手写策略 candidate_n 必须留空（无局数概念）"
                );
                c.check(info.action_index < info.candidate_scores.len(), "选中下标在截断后范围内");
                c.check(info.candidate_scores.len() <= 5, "候选评分截断到 5");
                c.finish()?;
            }
            game.run_stage(&trainer, &mut decision_rng)?;
        }

        let mut c = crate::utils::Checks::new();
        println!("手写策略决策 {decisions} 次");
        c.check(decisions > 50, "整局绝大多数决策点都有 last_decision");
        c.check(same_choices, "关闭原因文本后每个决策点的动作选择一致");
        c.check(same_protocol, "关闭原因文本后每个决策点的协议输出完整且一致");
        c.check(had_breakdown > 0, "普通实例保留评分说明供决策日志使用");
        c.check(rollout.last_breakdown().is_none(), "rollout 实例不采集原因文本");
        c.finish()
    }
}
