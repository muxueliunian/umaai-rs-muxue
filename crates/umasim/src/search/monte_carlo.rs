//! 蒙特卡洛搜索核心实现
//!
//! 基于扁平化蒙特卡洛采样的决策算法。
//! 不同于传统 MCTS 树搜索，本实现使用扁平数组存储动作结果，
//! 通过 UCT-like 公式动态分配搜索预算。

use std::cmp::Ordering;
use std::marker::PhantomData;

use anyhow::Result;
use log::{debug, info, trace};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::game::{Game, Trainer};
use crate::gamedata::ActionValue;

use super::{Evaluator, SearchParam, SearchResult, ValueOutput};

/// 预期搜索标准差（经验值，用于 UCT 公式）
const EXPECTED_SEARCH_STDEV: f64 = 2200.0;

/// 将 Evaluator 包装为 Trainer，用于模拟中的 run_stage 调用
pub struct EvaluatorTrainerAdapter<'e, G: Game, E: Evaluator<G>> {
    evaluator: &'e E,
    _marker: PhantomData<G>,
}

impl<'e, G: Game, E: Evaluator<G>> EvaluatorTrainerAdapter<'e, G, E> {
    pub fn new(evaluator: &'e E) -> Self {
        Self {
            evaluator,
            _marker: PhantomData,
        }
    }
}

impl<G: Game, E: Evaluator<G>> Trainer<G> for EvaluatorTrainerAdapter<'_, G, E>
where
    G::Action: PartialEq,
{
    fn select_action(
        &self,
        game: &G,
        actions: &[G::Action],
        rng: &mut StdRng,
    ) -> Result<usize> {
        // 尝试使用评估器选择动作
        if let Some(action) = self.evaluator.select_action(game, rng) {
            if let Some(idx) = actions.iter().position(|a| *a == action) {
                return Ok(idx);
            }
        }
        // 如果评估器返回的动作不在列表中，尝试使用 select_action_from_list
        // 这主要用于温泉选择场景，传入的 actions 只包含 Dig 动作
        let idx = self.evaluator.select_action_from_list(game, actions, rng);
        Ok(idx)
    }

    fn select_choice(
        &self,
        game: &G,
        choices: &[ActionValue],
        _rng: &mut StdRng,
    ) -> Result<usize> {
        let best = (0..choices.len())
            .max_by(|&i, &j| {
                let vi = self.evaluator.evaluate_choice(game, i);
                let vj = self.evaluator.evaluate_choice(game, j);
                vi.partial_cmp(&vj).unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);
        Ok(best)
    }
}

/// 蒙特卡洛搜索器
pub struct MonteCarloSearch<G: Game> {
    /// 搜索参数
    pub param: SearchParam,
    /// 每个动作的搜索结果（扁平数组）
    action_results: Vec<SearchResult>,
    /// 总搜索次数
    total_n: usize,
    _phantom: PhantomData<G>,
}

impl<G> MonteCarloSearch<G>
where
    G: Game + Clone + Send + Sync,
    G::Action: Clone + Send + Sync + PartialEq,
    G::Person: Send + Sync,
{
    /// 创建新的搜索器
    pub fn new(param: SearchParam) -> Self {
        Self {
            param,
            action_results: Vec::new(),
            total_n: 0,
            _phantom: PhantomData,
        }
    }

    /// 使用默认参数创建搜索器
    pub fn default_param() -> Self {
        Self::new(SearchParam::default())
    }

    /// 使用快速参数创建搜索器（用于测试）
    pub fn fast() -> Self {
        Self::new(SearchParam::fast())
    }

    /// 重置搜索状态
    fn reset(&mut self, action_count: usize) {
        self.action_results.clear();
        for _ in 0..action_count {
            self.action_results.push(SearchResult::new_legal());
        }
        self.total_n = 0;
    }

    /// 计算激进因子
    fn calc_radical_factor(&self, current_turn: i32, max_turn: i32) -> f64 {
        let remain = (max_turn - current_turn).max(0) as f64;
        let total = max_turn as f64;
        (remain / total).sqrt() * self.param.max_radical_factor
    }

    /// 主搜索入口
    pub fn run_search<E: Evaluator<G> + Sync>(
        &mut self,
        game: &G,
        actions: &[G::Action],
        evaluator: &E,
        rng: &mut StdRng,
    ) -> Result<usize> {
        if actions.is_empty() {
            anyhow::bail!("动作列表为空");
        }
        if actions.len() == 1 {
            return Ok(0);
        }

        self.param.validate()?;
        self.reset(actions.len());

        let radical_factor = self.calc_radical_factor(game.turn(), game.max_turn());
        debug!(
            "MCTS 搜索开始: {} 个动作, 激进因子={:.3}",
            actions.len(),
            radical_factor
        );

        // 阶段1: 初始采样
        for (idx, action) in actions.iter().enumerate() {
            let values = self.search_single_action(
                game, action, evaluator, self.param.search_group_size, rng,
            );
            self.action_results[idx].add_results(&values);
            self.total_n += values.len();
            trace!(
                "初始采样动作 {}: {} 次, 均值={:.1}",
                idx, values.len(), self.action_results[idx].mean()
            );
        }

        // 阶段2: 迭代预算分配
        while self.total_n < self.param.search_total_max {
            let action_idx = self.select_action_to_search(radical_factor);
            if self.action_results[action_idx].num >= self.param.search_single_max {
                if self.action_results.iter().all(|r| r.num >= self.param.search_single_max) {
                    break;
                }
                continue;
            }

            let values = self.search_single_action(
                game, &actions[action_idx], evaluator, self.param.search_group_size, rng,
            );
            self.action_results[action_idx].add_results(&values);
            self.total_n += values.len();
        }

        // 阶段3: 选择最优动作
        let best_idx = self.select_best_action(radical_factor);
        let best_result = &mut self.action_results[best_idx];
        let best_value = best_result.get_weighted_mean_score(radical_factor);

        info!(
            "MCTS 选择动作 {}: {} (总搜索 {} 次, 均值={:.1}, 标准差={:.1})",
            best_idx, actions[best_idx], self.total_n, best_value.score_mean, best_value.score_stdev
        );

        Ok(best_idx)
    }

    /// UCT-like 选择下一个分配预算的动作
    fn select_action_to_search(&mut self, radical_factor: f64) -> usize {
        let sqrt_total = (self.total_n as f64).sqrt();
        let cpuct = self.param.search_cpuct;
        let single_max = self.param.search_single_max;

        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (idx, result) in self.action_results.iter_mut().enumerate() {
            if !result.is_legal || result.num >= single_max {
                continue;
            }
            let value = result.get_weighted_mean_score(radical_factor).value;
            let n = result.num.max(1) as f64;
            let exploration = cpuct * EXPECTED_SEARCH_STDEV * sqrt_total / n;
            let search_value = value + exploration;

            if search_value > best_value {
                best_value = search_value;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// 选择最优动作
    fn select_best_action(&mut self, radical_factor: f64) -> usize {
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (idx, result) in self.action_results.iter_mut().enumerate() {
            if !result.is_legal || result.num == 0 {
                continue;
            }
            let value = result.get_weighted_mean_score(radical_factor).value;

            if value > best_value {
                best_value = value;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// 并行批量模拟单个动作
    fn search_single_action<E: Evaluator<G> + Sync>(
        &self,
        game: &G,
        action: &G::Action,
        evaluator: &E,
        search_n: usize,
        rng: &mut StdRng,
    ) -> Vec<ValueOutput> {
        let seeds: Vec<u64> = (0..search_n).map(|_| rng.random()).collect();

        seeds
            .par_iter()
            .map(|&seed| {
                let mut local_game = game.clone();
                let mut local_rng = StdRng::seed_from_u64(seed);
                self.simulate_once(&mut local_game, action, evaluator, &mut local_rng)
            })
            .collect()
    }

    /// 单次模拟
    fn simulate_once<E: Evaluator<G>>(
        &self,
        sim_game: &mut G,
        action: &G::Action,
        evaluator: &E,
        rng: &mut StdRng,
    ) -> ValueOutput {
        if sim_game.apply_action(action, rng).is_err() {
            return ValueOutput::ILLEGAL;
        }

        let adapter = EvaluatorTrainerAdapter::new(evaluator);

        for _depth in 0..self.param.max_depth {
            if !sim_game.next() {
                break;
            }
            if sim_game.turn() > sim_game.max_turn() {
                break;
            }
            if sim_game.run_stage(&adapter, rng).is_err() {
                break;
            }
        }

        evaluator.evaluate(sim_game)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::onsen::game::OnsenGame;

    #[test]
    fn test_monte_carlo_search_new() {
        let search: MonteCarloSearch<OnsenGame> = MonteCarloSearch::new(SearchParam::fast());
        assert_eq!(search.param.search_single_max, 32);
        assert_eq!(search.total_n, 0);
    }

    #[test]
    fn test_radical_factor_calculation() {
        let search: MonteCarloSearch<OnsenGame> = MonteCarloSearch::new(SearchParam::default());

        let rf_early = search.calc_radical_factor(1, 78);
        assert!(rf_early > 0.9, "初期激进因子应接近 1.0: {}", rf_early);

        let rf_mid = search.calc_radical_factor(39, 78);
        assert!(rf_mid < rf_early, "中期激进因子应小于初期");
        assert!(rf_mid > 0.5, "中期激进因子应大于 0.5: {}", rf_mid);

        let rf_late = search.calc_radical_factor(77, 78);
        assert!(rf_late < 0.2, "末期激进因子应接近 0: {}", rf_late);
    }
}

