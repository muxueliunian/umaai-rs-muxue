//! 决策日志包装训练员
//!
//! 包装任意 [`Trainer<RamenGame>`]，在每次选择动作/事件选项前后计时并记录
//! 一条 [`DecisionLogRow`]，用于开发调参（基准对比、决策轨迹回放）。
//!
//! 记录开关：构造后调用 [`LoggingTrainer::set_logging`]（或 bench 传参）控制；
//! rollout 海量场景默认应关闭（计划 §4：决策日志默认关闭）。

use std::{cell::RefCell, time::Instant};

use anyhow::Result;
use rand::prelude::StdRng;

use crate::{
    game::{
        ramen::{Operation, RamenGame, RamenStage},
        traits::{Game, Trainer}
    },
    gamedata::{EventChoice, EventData},
    output::decision_log::{DecisionLog, DecisionLogRow}
};

/// 决策日志包装训练员
pub struct LoggingTrainer<T> {
    /// 被包装的决策器
    inner: T,
    /// 决策日志（内部可变：`Trainer` trait 只给 `&self`）
    log: RefCell<DecisionLog>,
    /// 本局种子（写入每条记录）
    seed: u64,
    /// 是否记录（默认开；bench 之外可按需关闭）
    logging: bool
}

impl<T> LoggingTrainer<T> {
    /// 创建包装器：默认开启决策日志
    pub fn new(inner: T, seed: u64) -> Self {
        Self {
            inner,
            log: RefCell::new(DecisionLog::new()),
            seed,
            logging: true
        }
    }

    /// 设置是否记录决策日志
    pub fn set_logging(&mut self, on: bool) {
        self.logging = on;
    }

    /// 取出全部记录，清空日志（用于按局分段收集）
    pub fn take_records(&self) -> DecisionLog {
        std::mem::take(&mut *self.log.borrow_mut())
    }
}

impl<T: Trainer<RamenGame>> Trainer<RamenGame> for LoggingTrainer<T> {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        let start = Instant::now();
        let idx = self.inner.select_action(game, actions, rng)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        if self.logging {
            // 阶段标记：地区选择动作在 Begin/turn2（第 1 年）或 RegionSelect 阶段发生，
            // 统一按动作类型判定为 RegionSelect，便于调参按阶段分组
            let is_region_select = actions
                .get(idx)
                .is_some_and(|a| matches!(a.operation, Operation::RegionSelect(_)));
            let row = DecisionLogRow {
                seed: self.seed,
                turn: game.turn(),
                stage: if is_region_select {
                    "RegionSelect".to_string()
                } else {
                    match &game.stage {
                        RamenStage::RegionSelect => "RegionSelect".to_string(),
                        other => format!("{other:?}")
                    }
                },
                candidates: actions.len(),
                action_index: idx,
                action_desc: actions.get(idx).map(|a| a.to_string()).unwrap_or_default(),
                elapsed_us,
                score_breakdown: self.inner.last_breakdown()
            };
            self.log.borrow_mut().record(row);
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        let start = Instant::now();
        let idx = self.inner.select_choice(game, choices, rng)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        if self.logging {
            let explain = choices
                .iter()
                .map(|c| {
                    c.iter()
                        .map(|e| e.explain().replace(',', "；"))
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect::<Vec<_>>()
                .join(" / ");
            let row = DecisionLogRow {
                seed: self.seed,
                turn: game.turn(),
                stage: "Event".to_string(),
                candidates: choices.len(),
                action_index: idx,
                action_desc: explain,
                elapsed_us,
                score_breakdown: None
            };
            self.log.borrow_mut().record(row);
        }
        Ok(idx)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        let start = Instant::now();
        let idx = self.inner.select_event_choice(game, event, choices, rng)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        if self.logging {
            let explain = choices
                .iter()
                .map(|c| {
                    c.iter()
                        .map(|e| e.explain().replace(',', "；"))
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect::<Vec<_>>()
                .join(" / ");
            let row = DecisionLogRow {
                seed: self.seed,
                turn: game.turn(),
                stage: "Event".to_string(),
                candidates: choices.len(),
                action_index: idx,
                action_desc: format!("事件#{} {}: {}", event.id, event.name, explain),
                elapsed_us,
                score_breakdown: None
            };
            self.log.borrow_mut().record(row);
        }
        Ok(idx)
    }
}
#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::{
        gamedata::init_global,
        trainer::RandomTrainer,
        utils::{get_workspace_root, init_test_logger}
    };

    // 与 game.rs 测试公共参数一致（避免跨文件依赖私有常量）
    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 完整 77 回合跑一局（固定 seed），返回决策序列
    fn run_full(seed: u64) -> Result<Vec<(i32, String, usize)>> {
        let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(seed, 0);
        let trainer = LoggingTrainer::new(RandomTrainer, seed);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        game.run_full_game(&trainer, &mut decision_rng)?;
        let log = trainer.take_records();
        println!(
            "seed={} score={} 决策记录 {} 条",
            seed,
            game.uma.calc_score(),
            log.rows.len()
        );
        Ok(log
            .rows
            .iter()
            .map(|r| (r.turn, r.stage.clone(), r.action_index))
            .collect())
    }

    #[test]
    fn test_logging_trainer_records_full_game() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let (mut decision_rng, rule_master) = crate::bench::seeded_rngs(42, 0);
        let trainer = LoggingTrainer::new(RandomTrainer, 42);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        game.run_full_game(&trainer, &mut decision_rng)?;

        let log = trainer.take_records();
        println!("完整局决策记录: {} 条", log.rows.len());
        for r in &log.rows {
            println!(
                "  turn={} stage={} candidates={} idx={} elapsed={}us",
                r.turn, r.stage, r.candidates, r.action_index, r.elapsed_us
            );
        }
        // 记录非空 + 阶段覆盖（三阶段 + 事件 + 地区选择均应在日志中出现）
        println!("记录条数: {}", log.rows.len());
        println!("阶段集合: {:?}", {
            let mut stages: Vec<_> = log.rows.iter().map(|r| r.stage.as_str()).collect();
            stages.sort();
            stages.dedup();
            stages
        });
        assert!(!log.rows.is_empty());
        let stages: std::collections::HashSet<&str> = log.rows.iter().map(|r| r.stage.as_str()).collect();
        for expect in ["RamenSelect", "SpecialSelect", "Train", "Event"] {
            println!("包含阶段 {expect}: {}", stages.contains(expect));
            assert!(stages.contains(expect));
        }
        Ok(())
    }

    /// 固定种子可复现性：同 seed 两次整局，决策序列与最终评分完全一致
    #[test]
    fn test_reproducible_same_seed() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let seed: u64 = 12345;
        let first = run_full(seed)?;
        let second = run_full(seed)?;
        println!("第一次决策数: {}", first.len());
        println!("第二次决策数: {}", second.len());
        let same_seq = first == second;
        println!("决策序列 (turn,stage,index) 完全一致: {same_seq}");
        assert!(same_seq);
        Ok(())
    }
}
