//! 拉面杯手写策略训练员（测试壳）
//!
//! 直接使用 [`RamenPolicy`](crate::game::ramen::policy::RamenPolicy) 的确定性打分
//! 选择动作，不经过 MCTS 搜索。用途（计划 §2）：跑完整局验证策略效果 + 作为 MCTS
//! rollout 基策的调参载体，本身不是交付主体。
//!
//! 与旧 `HandwrittenTrainer`（温泉杯）无架构耦合：本实现直接在拉面杯规则层
//! （`RamenGame` / `RamenAction` / `policy.rs`）上重新实现。

use anyhow::Result;
use log::info;
use rand::prelude::StdRng;

use crate::{
    game::{
        Game,
        Trainer,
        ramen::{RamenGame, RamenStage, policy::RamenPolicy}
    },
    gamedata::{EventChoice, EventData}
};

/// 拉面杯手写策略训练员
pub struct RamenHandwrittenTrainer {
    /// 策略核心（参数化配置 + 各阶段打分）
    pub policy: RamenPolicy,
    /// 是否输出每步决策日志（整局跑批时建议关闭）
    pub verbose: bool
}

impl RamenHandwrittenTrainer {
    /// 创建默认配置的手写策略训练员
    pub fn new() -> Self {
        Self {
            policy: RamenPolicy::default(),
            verbose: false
        }
    }

    /// 使用指定策略核心创建
    pub fn with_policy(policy: RamenPolicy) -> Self {
        Self { policy, verbose: false }
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
            return Ok(0);
        }
        let idx = match game.stage {
            RamenStage::RamenSelect => self.policy.select_ramen(game, actions)?,
            RamenStage::SpecialSelect => self.policy.select_special(game, actions)?,
            RamenStage::Train => self.policy.select_train(game, actions)?,
            // 地区选择：第 1/2/3 年分别在 turn 2/23/47 触发（第 3 年 fixed 策略不走 trainer）
            RamenStage::RegionSelect => {
                let year_idx = match game.turn() {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => 0
                };
                self.policy.select_region(game, year_idx, actions)?
            }
            // 其他阶段（Begin/Distribute/AfterTrain 等）不应有多个候选
            _ => 0
        };
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
        let idx = self.policy.select_event(game, choices)?;
        if self.verbose {
            info!("[手写][回合 {}] 事件选择: {}", game.turn(), idx + 1);
        }
        Ok(idx)
    }

    fn select_event_choice(
        &self, game: &RamenGame, _event: &EventData, choices: &[Vec<EventChoice>], _rng: &mut StdRng
    ) -> Result<usize> {
        let idx = self.policy.select_event(game, choices)?;
        if self.verbose {
            info!("[手写][回合 {}] 事件选择: {}", game.turn(), idx + 1);
        }
        Ok(idx)
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
        let mut decision_rng = StdRng::seed_from_u64(seed);
        let rule_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let trainer = RamenHandwrittenTrainer::new();
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_internal_rng(rule_rng);
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
            let mut decision_rng = StdRng::seed_from_u64(seed);
            let rule_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
            let trainer = RamenHandwrittenTrainer::new();
            let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
            game.set_internal_rng(rule_rng);
            game.run_full_game(&trainer, &mut decision_rng)?;
            scores.push(game.uma.calc_score());
        }
        println!("两次评分: {:?}", scores);
        assert_eq!(scores[0], scores[1]);
        Ok(())
    }
}
