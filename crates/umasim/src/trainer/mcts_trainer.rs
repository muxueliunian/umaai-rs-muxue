//! MCTS 训练员模块
//!
//! 使用蒙特卡洛搜索算法进行动作选择。
//! 通过模拟大量后续局面来评估每个动作的价值。

use anyhow::Result;
use log::info;
use rand::rngs::StdRng;

use crate::{
    game::{Trainer, onsen::game::OnsenGame, onsen::action::OnsenAction},
    gamedata::ActionValue,
    search::{Evaluator, HandwrittenEvaluator, MonteCarloSearch, SearchParam},
};

// ============================================================================
// 温泉选择顺序（主流攻略）
// ============================================================================

/// 推荐温泉选择顺序（索引对应 onsen_info 数组）
///
/// 基础温泉(index=0)自动获得，不需要挖掘
///
/// 顺序说明：
/// 1. 疾驰之泉(1) - 速度/力量友情
/// 2. 坚忍之泉(2) - 耐力/根性友情
/// 3. 明晰之泉(3) - 比赛+30%
/// 4. 健壮古泉(6) - 体力消耗-10%
/// 5. 刚足古泉(5) - 力量/根性友情
/// 6. 秘汤汤驹(8) - 分身效果
/// 7. 骏闪古泉(4) - Hint+100%
/// 8. 传说秘泉(9) - 比赛+80%（⭐特殊：第三年9月强制选择）
/// 9. 天翔古泉(7) - 比赛+60%
const RECOMMENDED_ONSEN_ORDER: &[usize] = &[
    1,  // 疾驰之泉
    2,  // 坚忍之泉
    3,  // 明晰之泉
    6,  // 健壮古泉
    5,  // 刚足古泉
    8,  // 秘汤汤驹
    4,  // 骏闪古泉
    9,  // 传说秘泉
    7,  // 天翔古泉
];

/// 传说秘泉的强制选择回合（第三年9月 = 回合65）
const LEGEND_ONSEN_FORCE_TURN: i32 = 65;

/// 传说秘泉的索引
const LEGEND_ONSEN_INDEX: usize = 9;

/// MCTS 训练员
#[derive(Debug, Clone)]
pub struct MctsTrainer<E: Evaluator<OnsenGame> = HandwrittenEvaluator> {
    /// 搜索参数
    pub param: SearchParam,
    /// 评估器
    pub evaluator: E,
    /// 是否输出详细日志
    pub verbose: bool,
}

impl MctsTrainer<HandwrittenEvaluator> {
    /// 创建默认 MCTS 训练员
    pub fn new() -> Self {
        Self {
            param: SearchParam::default(),
            evaluator: HandwrittenEvaluator::new(),
            verbose: false,
        }
    }

    /// 使用指定搜索参数创建
    pub fn with_param(param: SearchParam) -> Self {
        Self {
            param,
            evaluator: HandwrittenEvaluator::new(),
            verbose: false,
        }
    }

    /// 快速搜索配置（用于测试）
    pub fn fast() -> Self {
        Self {
            param: SearchParam::fast(),
            evaluator: HandwrittenEvaluator::new(),
            verbose: false,
        }
    }
}

impl<E: Evaluator<OnsenGame>> MctsTrainer<E> {
    /// 使用自定义评估器创建
    pub fn with_evaluator(evaluator: E) -> Self {
        Self {
            param: SearchParam::default(),
            evaluator,
            verbose: false,
        }
    }

    /// 设置搜索参数
    pub fn param(mut self, param: SearchParam) -> Self {
        self.param = param;
        self
    }

    /// 设置是否输出详细日志
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 按主流攻略顺序选择温泉
    ///
    /// # 选择逻辑
    /// 1. 检查动作列表是否全为 Dig 动作
    /// 2. 特殊处理：第三年9月（回合65+）强制选择传说秘泉（如果可选）
    /// 3. 按推荐顺序遍历，选择第一个可选的温泉
    fn select_onsen_by_order(
        &self,
        game: &OnsenGame,
        actions: &[<OnsenGame as crate::game::Game>::Action],
    ) -> Option<usize> {
        // 检查是否全为 Dig 动作
        let dig_indices: Vec<_> = actions.iter().enumerate()
            .filter_map(|(i, a)| {
                if let OnsenAction::Dig(idx) = a {
                    Some((i, *idx as usize))
                } else {
                    None
                }
            })
            .collect();

        // 如果不是全为 Dig 动作，返回 None（由 MCTS 处理）
        if dig_indices.len() != actions.len() || dig_indices.is_empty() {
            return None;
        }

        // 特殊处理：第三年9月强制选择传说秘泉
        if game.turn >= LEGEND_ONSEN_FORCE_TURN {
            if let Some((action_idx, _)) = dig_indices.iter().find(|(_, onsen_idx)| *onsen_idx == LEGEND_ONSEN_INDEX) {
                return Some(*action_idx);
            }
        }

        // 按推荐顺序选择
        for &recommended_idx in RECOMMENDED_ONSEN_ORDER {
            if let Some((action_idx, _)) = dig_indices.iter().find(|(_, onsen_idx)| *onsen_idx == recommended_idx) {
                return Some(*action_idx);
            }
        }

        // 如果推荐列表都不可选，选择第一个可选的
        dig_indices.first().map(|(action_idx, _)| *action_idx)
    }
}

impl Default for MctsTrainer<HandwrittenEvaluator> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Evaluator<OnsenGame> + Clone + Send + Sync> Trainer<OnsenGame> for MctsTrainer<E> {
    fn select_action(
        &self,
        game: &OnsenGame,
        actions: &[<OnsenGame as crate::game::Game>::Action],
        rng: &mut StdRng,
    ) -> Result<usize> {
        if actions.len() <= 1 {
            return Ok(0);
        }

        // 硬编码规则：温泉选择（按主流攻略顺序，不进行 MCTS 搜索）
        if let Some(idx) = self.select_onsen_by_order(game, actions) {
            if self.verbose {
                info!("[回合 {}] 温泉选择: {}", game.turn, actions[idx]);
            }
            return Ok(idx);
        }

        if self.verbose {
            info!(
                "MCTS 开始搜索 [回合 {}]，可选动作数: {}",
                game.turn,
                actions.len()
            );
        }

        let mut search = MonteCarloSearch::new(self.param.clone());
        let result = search.run_search(game, actions, &self.evaluator, rng)?;

        if self.verbose {
            info!("MCTS 选择: {} - {}", result, actions[result]);
        }

        Ok(result)
    }

    fn select_choice(
        &self,
        game: &OnsenGame,
        choices: &[ActionValue],
        _rng: &mut StdRng,
    ) -> Result<usize> {
        let best = (0..choices.len())
            .max_by(|&i, &j| {
                let vi = self.evaluator.evaluate_choice(game, i);
                let vj = self.evaluator.evaluate_choice(game, j);
                vi.partial_cmp(&vj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        if self.verbose {
            info!("MCTS 选择选项: {} - {:?}", best, choices.get(best));
        }

        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_trainer_new() {
        let trainer = MctsTrainer::new();
        assert!(!trainer.verbose);
        assert_eq!(trainer.param.search_single_max, 256);
    }

    #[test]
    fn test_mcts_trainer_with_param() {
        let param = SearchParam::fast();
        let trainer = MctsTrainer::with_param(param.clone());
        assert_eq!(trainer.param.search_single_max, param.search_single_max);
    }

    #[test]
    fn test_mcts_trainer_builder() {
        let trainer = MctsTrainer::new()
            .param(SearchParam::fast())
            .verbose(true);
        assert!(trainer.verbose);
    }
}

