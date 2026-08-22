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
    gamedata::{EventChoice, EventData}
};

/// 决策所属的**有效阶段**（按动作类型纠正 `game.stage`）
///
/// 第 1 年地区选择由 `run_begin` 在 turn 2 内联调用 `run_region_select`，
/// 此时 `game.stage` 仍是 [`RamenStage::Begin`]。只按 `game.stage` 分派的训练员
/// 会把它落到默认分支、恒选候选 0——第 1 年的地区**从未经过策略打分**。
/// 因此一律按动作携带的 [`Operation`] 判定，`game.stage` 只作为回退。
///
/// 判定用 `any` 而非 `all`：地区选择的候选集是同构的，命中一个即可；
/// 用 `all` 会在候选集混入其他动作时静默退回默认分支。
pub fn ramen_effective_stage(game: &RamenGame, actions: &[RamenAction]) -> RamenStage {
    if actions.iter().any(|a| matches!(a.operation, Operation::RegionSelect(_))) {
        return RamenStage::RegionSelect;
    }
    game.stage.clone()
}

/// 拉面杯手写策略训练员
pub struct RamenHandwrittenTrainer {
    /// 策略核心（参数化配置 + 各阶段打分）
    pub policy: RamenPolicy,
    /// 是否输出每步决策日志（整局跑批时建议关闭）
    pub verbose: bool,
    /// 是否采集评分分解文本（供 `LoggingTrainer` 取用）
    ///
    /// 作为**搜索的 rollout 基策**时必须关掉：`stash_breakdown` 每次决策都无条件
    /// `format!` 出全候选分解并锁同一把 `Mutex`，而所有 rayon 线程共享同一个
    /// rollout trainer。单次 rollout 约 170 次决策 × 24 线程 = 高频锁争用，
    /// 而 rollout 内部的分解文本没有任何消费者。不影响分数，只影响吞吐。
    pub collect_breakdown: bool,
    /// 最近一次决策的评分分解文本（供 LoggingTrainer 提取进决策日志）
    ///
    /// 用 `Mutex` 而非 `RefCell`：搜索层要求 `Trainer: Sync`（rayon 跨线程共享同一个
    /// rollout 决策器），`RefCell` 会让整个 `FlatSearch<RamenGame>` 失去 `Sync`。
    /// 单局日志场景无竞争，加锁开销可忽略。
    last_breakdown: Mutex<Option<String>>
}

impl RamenHandwrittenTrainer {
    /// 创建默认配置的手写策略训练员
    pub fn new() -> Self {
        Self {
            policy: RamenPolicy::default(),
            verbose: false,
            collect_breakdown: true,
            last_breakdown: Mutex::new(None)
        }
    }

    /// 创建 rollout 专用实例（关闭分解采集，见 [`collect_breakdown`](Self::collect_breakdown)）
    pub fn for_rollout() -> Self {
        Self {
            collect_breakdown: false,
            ..Self::new()
        }
    }

    /// 使用指定策略核心创建
    pub fn with_policy(policy: RamenPolicy) -> Self {
        Self {
            policy,
            verbose: false,
            collect_breakdown: true,
            last_breakdown: Mutex::new(None)
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
        if !self.collect_breakdown {
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
            if self.collect_breakdown {
                if let Ok(mut slot) = self.last_breakdown.lock() {
                    *slot = Some(format!("仅1候选: {}", actions[0]));
                }
            }
            return Ok(0);
        }
        let (idx, outputs) = match ramen_effective_stage(game, actions) {
            RamenStage::RamenSelect => self.policy.decide_ramen(game, actions)?,
            RamenStage::SpecialSelect => self.policy.decide_special(game, actions)?,
            RamenStage::Train => self.policy.decide_train(game, actions)?,
            // 地区选择：第 1/2/3 年分别在 turn 2/23/47 触发（第 3 年 fixed 策略不走 trainer）
            // 第 1 年的 `game.stage` 是 Begin，靠 `ramen_effective_stage` 纠正到这里
            RamenStage::RegionSelect => {
                let year_idx = match game.turn() {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => 0
                };
                self.policy.decide_region(game, year_idx, actions)?
            }
            // 其他阶段（Begin/Distribute/AfterTrain 等）不应有多个候选
            _ => (0, vec![])
        };
        self.stash_breakdown(&outputs);
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
        if self.verbose {
            info!("[手写][回合 {}] 事件选择: {}", game.turn(), idx + 1);
        }
        Ok(idx)
    }

    fn last_breakdown(&self) -> Option<String> {
        self.last_breakdown.lock().ok().and_then(|slot| slot.clone())
    }
}
#[cfg(test)]
mod tests {
    use rand::SeedableRng;

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
            "手写策略完整局: 回合={} 评分={} ({}) RMJ={:?} 吃面={} 五维={:?}",
            game.turn(),
            score,
            rank,
            game.ramen.rmj_results,
            game.ramen.eat_count,
            game.uma.five_status,
        );
        assert_eq!(game.turn(), 77);
        assert!(score > 0);
        Ok(())
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
}
