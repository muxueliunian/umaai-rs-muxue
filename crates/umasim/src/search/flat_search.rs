//! 扁平蒙特卡洛搜索
//!
//! 对每个合法动作执行多次模拟，统计分数分布，选择最优动作。
//! 支持两种搜索策略：
//! - 均匀分配：每个动作平均分配搜索次数（并行化）
//! - UCB 分配：根据 UCB 公式动态分配搜索资源（C++ UmaAi 风格）

use anyhow::Result;
use log::{debug, warn};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::{
    config::{SearchConfig, TOTAL_TURN},
    result::{ActionResult, SearchOutput},
    seeds::RolloutSeeds
};
#[cfg(feature = "onnx")]
use crate::neural::{ThreadLocalNeuralNetLeafEvaluator, ThreadLocalNeuralNetLeafStatsSnapshot};
use crate::{
    game::{
        Game,
        onsen::{action::OnsenAction, game::OnsenGame}
    },
    gamedata::EventChoice,
    neural::{Evaluator, HandwrittenEvaluator, ValueOutput}
};

#[derive(Clone)]
enum LeafEvaluator {
    Handwritten,
    #[cfg(feature = "onnx")]
    NeuralNet(ThreadLocalNeuralNetLeafEvaluator)
}

impl LeafEvaluator {
    fn name(&self) -> &'static str {
        match self {
            LeafEvaluator::Handwritten => "handwritten",
            #[cfg(feature = "onnx")]
            LeafEvaluator::NeuralNet(_) => "nn"
        }
    }

    fn evaluate(&self, rollout_evaluator: &HandwrittenEvaluator, game: &OnsenGame) -> ValueOutput {
        match self {
            LeafEvaluator::Handwritten => rollout_evaluator.evaluate(game),
            #[cfg(feature = "onnx")]
            LeafEvaluator::NeuralNet(nn) => nn.evaluate(game)
        }
    }
}

/// 扁平蒙特卡洛搜索
///
/// 使用手写逻辑进行模拟，统计各动作的分数分布。
#[derive(Clone)]
pub struct FlatSearch {
    /// 手写评估器（用于模拟）
    rollout_evaluator: HandwrittenEvaluator,

    /// leaf eval 评估器（用于 max_depth>0 截断估值）
    leaf_evaluator: LeafEvaluator,

    /// 搜索配置
    config: SearchConfig,

    /// E4：leaf eval 微批大小（仅在 max_depth>0 && leaf_eval=nn 时生效）
    rollout_batch_size: usize
}

impl FlatSearch {
    /// 创建搜索器
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rollout_evaluator: HandwrittenEvaluator::new(),
            leaf_evaluator: LeafEvaluator::Handwritten,
            config,
            rollout_batch_size: 1
        }
    }

    /// 创建默认搜索器
    pub fn default_search() -> Self {
        Self::new(SearchConfig::default())
    }

    /// 设置 leaf eval 为神经网络（用于 max_depth>0 截断估值）
    ///
    /// 仅在 `onnx` feature 下可用；core-only 构建调用会编译错误（编译器提示）。
    #[cfg(feature = "onnx")]
    pub fn with_leaf_evaluator_nn(mut self, model_path: impl Into<String>) -> Self {
        self.leaf_evaluator = LeafEvaluator::NeuralNet(ThreadLocalNeuralNetLeafEvaluator::new(model_path));
        self
    }

    /// 强制 leaf eval 回退为 handwritten（默认）
    pub fn with_leaf_evaluator_handwritten(mut self) -> Self {
        self.leaf_evaluator = LeafEvaluator::Handwritten;
        self
    }

    /// 设置 leaf eval 微批大小（仅 nn leaf 生效）
    pub fn with_rollout_batch_size(mut self, batch_size: usize) -> Self {
        self.rollout_batch_size = batch_size.max(1).min(1024);
        self
    }

    /// 获取配置
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// E4 调试：获取 leaf NN 推理统计（仅当 leaf evaluator 为 nn 时存在）
    #[cfg(feature = "onnx")]
    pub fn leaf_nn_stats(&self) -> Option<ThreadLocalNeuralNetLeafStatsSnapshot> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn.stats()),
            _ => None
        }
    }

    fn use_parallel_simulation(&self) -> bool {
        // E4.3：leaf eval 使用 thread_local 模型后，可安全恢复 Rayon 并行
        true
    }

    #[cfg(feature = "onnx")]
    fn leaf_nn(&self) -> Option<&ThreadLocalNeuralNetLeafEvaluator> {
        match &self.leaf_evaluator {
            LeafEvaluator::NeuralNet(nn) => Some(nn),
            _ => None
        }
    }

    /// 执行搜索
    ///
    /// 根据配置选择搜索策略：
    /// - use_ucb = true: UCB 动态分配
    /// - use_ucb = false: 均匀分配（并行化）
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// 搜索输出，包含各动作的分数分布和最优动作
    pub fn search(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<SearchOutput> {
        if actions.is_empty() {
            anyhow::bail!("没有可用动作");
        }

        // 计算激进度因子（C++ 风格，无随机性）
        let radical_factor = self.compute_radical_factor(game.turn as usize);

        debug!(
            "[回合 {}] 开始搜索: {} 个动作, search_n={}, max_depth={}, leaf_eval={}, radical_factor={:.1}, ucb={}",
            game.turn,
            actions.len(),
            self.config.search_n,
            self.config.max_depth,
            self.leaf_evaluator.name(),
            radical_factor,
            self.config.use_ucb
        );

        // 本次搜索的 rollout 种子表：所有候选共享，由传入 rng 派生（可复现性入口）
        let seeds = RolloutSeeds::from_rng(rng);
        debug!("[回合 {}] 搜索根种子 = {:#018x}", game.turn, seeds.root());

        // 根据配置选择搜索策略
        let action_results = if self.config.use_ucb {
            self.search_ucb(game, &actions, radical_factor, &seeds)?
        } else {
            self.search_uniform(game, &actions, &seeds)?
        };

        // 某候选一次都没跑成功时其统计全是空的，继续用下去等于拿垃圾数据排序
        for (i, (result, _)) in action_results.iter().enumerate() {
            if result.count() == 0 {
                anyhow::bail!("候选动作 {i} 的全部 rollout 均失败，搜索结果不可用");
            }
        }

        Ok(SearchOutput::new(actions.to_vec(), action_results, radical_factor))
    }

    /// 计算激进度因子
    ///
    /// 使用 C++ UmaAi 的固定公式，不使用随机性：
    /// radical_factor = (剩余回合 / 总回合)^0.5 * 最大激进度

    fn compute_radical_factor(&self, turn: usize) -> f64 {
        let remain_turns = (TOTAL_TURN.saturating_sub(turn)) as f64;
        let factor = (remain_turns / TOTAL_TURN as f64).powf(0.5);
        factor * self.config.radical_factor_max
    }

    /// 均匀分配搜索（并行化）
    ///
    /// 每个动作平均分配 `search_n` 次搜索，使用 Rayon 并行化。
    ///
    /// 所有候选的第 j 次 rollout 共用 `seeds.seed_at(j)`（CRN 载体，见
    /// [`RolloutSeeds`]），故并行粒度不影响结果：每次 rollout 的随机流由
    /// 工作项序号唯一决定，与线程调度无关。
    ///
    /// 注：此处按候选并行，并行度上限即候选数（≤10）。改为按
    /// `(候选, rollout)` 扁平并行可提升吞吐且结果位级不变，
    /// 但现有粒度可能另有原因，留作后续性能对照实验。
    fn search_uniform(
        &self, game: &OnsenGame, actions: &[OnsenAction], seeds: &RolloutSeeds
    ) -> Result<Vec<(ActionResult, ActionResult)>> {
        let n = self.config.search_n;
        let run = |action: &OnsenAction| -> Result<(ActionResult, ActionResult, usize)> {
            let mut result = ActionResult::new();
            let mut result_pt = ActionResult::new();
            // offset=0：均匀分配下每个候选都从 rollout 0 开始，天然完全配对
            let failed = self.simulate_many(game, action, n, seeds, 0, &mut result, &mut result_pt)?;
            Ok((result, result_pt, failed))
        };

        let collected: Vec<(ActionResult, ActionResult, usize)> = if self.use_parallel_simulation() {
            actions.par_iter().map(run).collect::<Result<Vec<_>>>()?
        } else {
            actions.iter().map(run).collect::<Result<Vec<_>>>()?
        };

        Ok(Self::split_failures(collected, "均匀分配"))
    }

    /// 拆出失败计数并汇总告警，返回各候选的 (score, pt) 统计
    ///
    /// rollout 失败会让该候选的样本数少于计划值，静默丢弃会把「跑失败」
    /// 混同于「跑出来分低」。此处不中断搜索（避免偶发失败拖垮实时通道层），
    /// 但必须在日志里留下痕迹。
    fn split_failures(
        collected: Vec<(ActionResult, ActionResult, usize)>, stage: &str
    ) -> Vec<(ActionResult, ActionResult)> {
        let total_failed: usize = collected.iter().map(|(_, _, f)| f).sum();
        if total_failed > 0 {
            warn!("[搜索][{stage}] {total_failed} 次 rollout 失败，对应候选的样本数少于计划值");
        }
        collected.into_iter().map(|(r, r_pt, _)| (r, r_pt)).collect()
    }

    /// UCB 动态分配搜索
    ///
    /// 使用 UCB 公式动态分配搜索资源，好的动作获得更多搜索次数。
    /// UCB 决策是串行的，但每组模拟内部使用 Rayon 并行化。
    ///
    /// # UCB 公式
    /// search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn search_ucb(
        &self, game: &OnsenGame, actions: &[OnsenAction], radical_factor: f64, seeds: &RolloutSeeds
    ) -> Result<Vec<(ActionResult, ActionResult)>> {
        let num_actions = actions.len();
        let mut action_results: Vec<(ActionResult, ActionResult)> = vec![Default::default(); num_actions];
        let group_size = self.config.search_group_size;
        anyhow::ensure!(group_size > 0, "search_group_size 不能为 0（UCB 分配会死循环）");
        let use_parallel = self.use_parallel_simulation();

        // 各候选**已计划**的 rollout 次数（≠ 已成功次数）
        //
        // 种子偏移必须用计划次数而非 `ActionResult::count()`：后者会因 rollout 失败
        // 而少计，导致同一 rollout 序号在不同候选上错位，破坏配对。
        let mut planned = vec![0usize; num_actions];

        // 第一阶段：每个动作先搜一组（并行）
        let run_initial = |action: &OnsenAction| -> Result<(ActionResult, ActionResult, usize)> {
            let mut result = ActionResult::new();
            let mut result_pt = ActionResult::new();
            let failed = self.simulate_many(game, action, group_size, seeds, 0, &mut result, &mut result_pt)?;
            Ok((result, result_pt, failed))
        };
        let initial: Vec<(ActionResult, ActionResult, usize)> = if use_parallel {
            actions.par_iter().map(run_initial).collect::<Result<Vec<_>>>()?
        } else {
            actions.iter().map(run_initial).collect::<Result<Vec<_>>>()?
        };

        // 合并初始结果
        for (i, result) in Self::split_failures(initial, "UCB 首组").into_iter().enumerate() {
            action_results[i] = result;
            planned[i] = group_size;
        }

        let mut total_n = (group_size * num_actions) as f64;

        // 第二阶段：UCB 动态分配
        loop {
            // 检查是否有动作达到 search_n
            let max_count = action_results.iter().map(|r| r.0.count()).max().unwrap_or(0);
            if max_count >= self.config.search_n as u32 {
                break;
            }

            // 使用 UCB 公式选择下一个要搜索的动作
            let best_action_idx = self.select_ucb_action(&action_results, radical_factor, total_n);

            // 对选中的动作搜索一组（并行）
            let action = &actions[best_action_idx];
            // E4：nn leaf 时，rollout 收集 leaf features -> infer_batch -> 写入结果
            // 仅 onnx feature 下启用 nn leaf 微批；core-only 构建走 handwritten 默认路径
            #[cfg(feature = "onnx")]
            if self.config.max_depth > 0 && self.leaf_nn().is_some() && self.rollout_batch_size > 1 {
                let nn = self.leaf_nn().expect("nn");

                // 每次 rollout 由工作项序号自行播种：结果与线程调度无关
                let offset = planned[best_action_idx];
                let run_leaf = |k: usize| -> Option<SimOutcome> {
                    let mut rng = StdRng::seed_from_u64(seeds.seed_at(offset + k));
                    match self.simulate_until_terminal_or_leaf(game, action, &mut rng) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            debug!("[搜索][UCB nn leaf] rollout {} 失败: {e}", offset + k);
                            None
                        }
                    }
                };
                let outcomes: Vec<_> = if use_parallel {
                    (0..group_size).into_par_iter().filter_map(run_leaf).collect()
                } else {
                    (0..group_size).filter_map(run_leaf).collect()
                };
                if outcomes.len() < group_size {
                    warn!(
                        "[搜索][UCB nn leaf] {} 次 rollout 失败，样本数少于计划值",
                        group_size - outcomes.len()
                    );
                }

                let mut leaf_features: Vec<f32> = Vec::new();
                let mut leaf_pt_bias: Vec<f64> = Vec::new();

                for o in outcomes {
                    match o {
                        SimOutcome::Terminal { score, score_pt } => {
                            action_results[best_action_idx].0.add(score);
                            action_results[best_action_idx].1.add(score_pt);
                        }
                        SimOutcome::Leaf { features, pt_bias } => {
                            leaf_features.extend_from_slice(&features);
                            leaf_pt_bias.push(pt_bias);
                        }
                    }
                }

                if !leaf_pt_bias.is_empty() {
                    let leaf_n = leaf_pt_bias.len();
                    match nn.evaluate_features_batch(&leaf_features, leaf_n) {
                        Ok(values) => {
                            for (i, v) in values.into_iter().enumerate() {
                                let score_mean = v.score_mean;
                                action_results[best_action_idx].0.add(score_mean);
                                action_results[best_action_idx].1.add(score_mean + leaf_pt_bias[i]);
                            }
                        }
                        Err(e) => {
                            log::warn!("[NN][leaf] infer_batch 失败，回退逐样本（性能受限）: {e}");
                            for i in 0..leaf_n {
                                let start = i * 1121;
                                let end = start + 1121;
                                if let Ok(v) = nn.evaluate_features_batch(&leaf_features[start..end], 1) {
                                    let score_mean = v[0].score_mean;
                                    action_results[best_action_idx].0.add(score_mean);
                                    action_results[best_action_idx].1.add(score_mean + leaf_pt_bias[i]);
                                }
                            }
                        }
                    }
                }
                // nn leaf 微批路径已完成本组搜索，跳过默认循环
                planned[best_action_idx] += group_size;
                total_n += group_size as f64;
                continue;
            }
            // 默认循环：handwritten 评估（onnx 关闭时也走这条）
            //
            // 该候选已计划 offset 次，本组取 seeds[offset..offset+group_size]。
            // 两个候选因而在 0..min(n_a, n_b) 上完全配对，多出的部分为 unpaired，
            // 这是 CRN 在不等样本数下的标准做法。
            let offset = planned[best_action_idx];
            let run_one = |k: usize| -> Option<(f64, f64)> {
                let mut rng = StdRng::seed_from_u64(seeds.seed_at(offset + k));
                match self.simulate(game, action, &mut rng) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        debug!("[搜索][UCB] rollout {} 失败: {e}", offset + k);
                        None
                    }
                }
            };
            let scores: Vec<_> = if use_parallel {
                (0..group_size).into_par_iter().filter_map(run_one).collect()
            } else {
                (0..group_size).filter_map(run_one).collect()
            };
            if scores.len() < group_size {
                warn!(
                    "[搜索][UCB] {} 次 rollout 失败，样本数少于计划值",
                    group_size - scores.len()
                );
            }

            for score in scores {
                action_results[best_action_idx].0.add(score.0);
                action_results[best_action_idx].1.add(score.1);
            }

            planned[best_action_idx] += group_size;
            total_n += group_size as f64;
        }

        Ok(action_results)
    }

    /// 使用 UCB 公式选择下一个要搜索的动作
    ///
    /// UCB 公式: search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn select_ucb_action(
        &self, action_results: &[(ActionResult, ActionResult)], radical_factor: f64, total_n: f64
    ) -> usize {
        let sqrt_total = total_n.sqrt();
        let cpuct = self.config.search_cpuct;
        let expected_stdev = self.config.expected_search_stdev;

        let mut best_idx = 0;
        let mut best_search_value = f64::NEG_INFINITY;

        for (i, result) in action_results.iter().enumerate() {
            let n = result.0.count() as f64;
            if n == 0.0 {
                // 未搜索的动作优先级最高
                return i;
            }

            let value = result.0.weighted_mean(radical_factor);
            // UCB 公式：value 越高或搜索次数越少，search_value 越高
            let delta = cpuct * expected_stdev * sqrt_total / n;
            let search_value = value + delta;
            //println!("#{i} score: {value:.0}, ucb: {delta:.0}, sqrt_total: {sqrt_total:.0}, n: {n}");
            if search_value > best_search_value {
                best_search_value = search_value;
                best_idx = i;
            }
        }
        // println!("best: #{best_idx}");
        // println!("--------------------");
        best_idx
    }

    /// 模拟单个动作到终局
    ///
    /// 从当前状态开始，执行指定动作，然后用手写逻辑走到游戏结束。
    ///
    /// # 参数
    /// - `game`: 当前游戏状态
    /// - `action`: 要模拟的动作
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// 最终分数
    fn simulate(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<(f64, f64)> {
        if matches!(action, OnsenAction::Dig(_)) {
            self.simulate_onsen_select(game, action, rng)
        } else if matches!(action, OnsenAction::Upgrade(_)) {
            self.simulate_dig_upgrade(game, action, rng)
        } else {
            // 克隆游戏状态
            let mut sim_game = game.clone();
            let trainer_hw = SimulationTrainer {
                evaluator: &self.rollout_evaluator
            };

            // 执行初始动作
            sim_game.apply_action(action, rng)?;

            // max_depth==0：保持旧行为，rollout 跑到终局
            if self.config.max_depth == 0 {
                while sim_game.next() {
                    sim_game.run_stage(&trainer_hw, rng)?;
                }
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }

            // max_depth>0：按 turn 截断；未终局则 leaf eval 估值
            let start_turn = sim_game.turn;
            let max_depth = self.config.max_depth as i32;
            let mut finished = false;

            loop {
                if !sim_game.next() {
                    finished = true;
                    break;
                }
                sim_game.run_stage(&trainer_hw, rng)?;
                if (sim_game.turn - start_turn) >= max_depth {
                    break;
                }
            }

            if finished {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }
            // 有些情况下（例如在达到 max_depth 的同一轮刚好走到终局），可能还未通过 next() 触发 finished。
            // 用 turn>=max_turn 兜底判定终局，并确保 on_simulation_end 被触发，避免漏算最终奖励。
            if sim_game.turn >= sim_game.max_turn() {
                sim_game.on_simulation_end(&trainer_hw, rng)?;
                return Ok((
                    sim_game.uma().calc_score() as f64,
                    sim_game.uma().calc_score_with_pt_favor() as f64
                ));
            }

            // 未终局：leaf eval（scoreMean）；PT 口径用“当前 pt_bias”近似对齐
            let v = self.leaf_evaluator.evaluate(&self.rollout_evaluator, &sim_game);
            let score_mean = v.score_mean;
            let current_score = sim_game.uma().calc_score() as f64;
            let current_pt_score = sim_game.uma().calc_score_with_pt_favor() as f64;
            let pt_bias = current_pt_score - current_score;
            Ok((score_mean, score_mean + pt_bias))
        }
    }

    /// 对同一候选连续跑 `n` 次 rollout
    ///
    /// 第 k 次取 `seeds.seed_at(offset + k)` 播种，`offset` 为该候选**已计划**的次数。
    /// 返回失败次数（不中断搜索，由调用方汇总告警）。
    fn simulate_many(
        &self, game: &OnsenGame, action: &OnsenAction, n: usize, seeds: &RolloutSeeds, offset: usize,
        result: &mut ActionResult, result_pt: &mut ActionResult
    ) -> Result<usize> {
        // 仅 nn leaf + max_depth>0 才走微批；否则保持旧行为（handwritten 默认循环）
        // onnx feature 关闭时直接走默认循环，避免对 leaf_nn 的引用。
        #[cfg(feature = "onnx")]
        if self.config.max_depth > 0 && self.leaf_nn().is_some() && self.rollout_batch_size > 1 {
            let nn = self.leaf_nn().expect("nn");
            let mut pending_features: Vec<f32> = Vec::with_capacity(self.rollout_batch_size * 1121);
            let mut pending_pt_bias: Vec<f64> = Vec::with_capacity(self.rollout_batch_size);

            let mut failed = 0usize;
            for k in 0..n {
                let mut rng = StdRng::seed_from_u64(seeds.seed_at(offset + k));
                let outcome = match self.simulate_until_terminal_or_leaf(game, action, &mut rng) {
                    Ok(v) => v,
                    Err(e) => {
                        debug!("[搜索][nn leaf] rollout {} 失败: {e}", offset + k);
                        failed += 1;
                        continue;
                    }
                };
                match outcome {
                    SimOutcome::Terminal { score, score_pt } => {
                        result.add(score);
                        result_pt.add(score_pt);
                    }
                    SimOutcome::Leaf { features, pt_bias } => {
                        pending_features.extend_from_slice(&features);
                        pending_pt_bias.push(pt_bias);
                        if pending_pt_bias.len() >= self.rollout_batch_size {
                            let leaf_n = pending_pt_bias.len();
                            let values = nn.evaluate_features_batch(&pending_features, leaf_n)?;
                            for (i, v) in values.into_iter().enumerate() {
                                let score_mean = v.score_mean;
                                result.add(score_mean);
                                result_pt.add(score_mean + pending_pt_bias[i]);
                            }
                            pending_features.clear();
                            pending_pt_bias.clear();
                        }
                    }
                }
            }

            if !pending_pt_bias.is_empty() {
                let leaf_n = pending_pt_bias.len();
                let values = nn.evaluate_features_batch(&pending_features, leaf_n)?;
                for (i, v) in values.into_iter().enumerate() {
                    let score_mean = v.score_mean;
                    result.add(score_mean);
                    result_pt.add(score_mean + pending_pt_bias[i]);
                }
                pending_features.clear();
                pending_pt_bias.clear();
            }
            return Ok(failed);
        }
        // 默认循环：handwritten 评估（onnx 关闭时也走这条）
        let mut failed = 0usize;
        for k in 0..n {
            let mut rng = StdRng::seed_from_u64(seeds.seed_at(offset + k));
            match self.simulate(game, action, &mut rng) {
                Ok(score) => {
                    result.add(score.0);
                    result_pt.add(score.1);
                }
                Err(e) => {
                    debug!("[搜索] rollout {} 失败: {e}", offset + k);
                    failed += 1;
                }
            }
        }
        Ok(failed)
    }

    #[cfg(feature = "onnx")]
    fn simulate_until_terminal_or_leaf(
        &self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng
    ) -> Result<SimOutcome> {
        // Dig/Upgrade 目前仍走完整模拟（未对齐 max_depth）；这里直接复用现有路径，视为 Terminal
        if matches!(action, OnsenAction::Dig(_)) {
            let (s, pt) = self.simulate_onsen_select(game, action, rng)?;
            return Ok(SimOutcome::Terminal { score: s, score_pt: pt });
        }
        if matches!(action, OnsenAction::Upgrade(_)) {
            let (s, pt) = self.simulate_dig_upgrade(game, action, rng)?;
            return Ok(SimOutcome::Terminal { score: s, score_pt: pt });
        }

        // 克隆游戏状态
        let mut sim_game = game.clone();
        let trainer_hw = SimulationTrainer {
            evaluator: &self.rollout_evaluator
        };

        // 执行初始动作
        sim_game.apply_action(action, rng)?;

        // max_depth==0：保持旧行为，rollout 跑到终局
        if self.config.max_depth == 0 {
            while sim_game.next() {
                sim_game.run_stage(&trainer_hw, rng)?;
            }
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64
            });
        }

        // max_depth>0：按 turn 截断；未终局则返回 leaf features（不在这里做推理）
        let start_turn = sim_game.turn;
        let max_depth = self.config.max_depth as i32;
        let mut finished = false;

        loop {
            if !sim_game.next() {
                finished = true;
                break;
            }
            sim_game.run_stage(&trainer_hw, rng)?;
            if (sim_game.turn - start_turn) >= max_depth {
                break;
            }
        }

        if finished || sim_game.turn >= sim_game.max_turn() {
            sim_game.on_simulation_end(&trainer_hw, rng)?;
            return Ok(SimOutcome::Terminal {
                score: sim_game.uma().calc_score() as f64,
                score_pt: sim_game.uma().calc_score_with_pt_favor() as f64
            });
        }

        let current_score = sim_game.uma().calc_score() as f64;
        let current_pt_score = sim_game.uma().calc_score_with_pt_favor() as f64;
        let pt_bias = current_pt_score - current_score;
        let features = sim_game.extract_nn_features(None);

        Ok(SimOutcome::Leaf { features, pt_bias })
    }

    /// 模拟选择温泉. 因为没有做成单独的阶段，所以单独处理
    pub fn simulate_onsen_select(
        &self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng
    ) -> Result<(f64, f64)> {
        let mut sim_game = game.clone();
        let mut best_score = (0.0, 0.0);

        sim_game.apply_action(action, rng)?;
        for i in sim_game.get_upgradeable_equipment() {
            let score = self.simulate_dig_upgrade(&sim_game, &OnsenAction::Upgrade(i as i32), rng)?;
            if score.0 > best_score.0 {
                best_score = score;
            }
        }
        Ok(best_score)
    }

    /// 模拟升级挖掘装备
    pub fn simulate_dig_upgrade(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> Result<(f64, f64)> {
        let mut sim_game = game.clone();
        sim_game.apply_action(action, rng)?;
        sim_game.pending_selection = false;
        // 去除pending_selection状态后就可以正常模拟了。
        let trainer_hw = SimulationTrainer {
            evaluator: &self.rollout_evaluator
        };
        while sim_game.next() {
            sim_game.run_stage(&trainer_hw, rng)?;
        }
        sim_game.on_simulation_end(&trainer_hw, rng)?;
        Ok((
            sim_game.uma().calc_score() as f64,
            sim_game.uma().calc_score_with_pt_favor() as f64
        ))
    }
}

#[cfg(feature = "onnx")]
enum SimOutcome {
    Terminal { score: f64, score_pt: f64 },
    Leaf { features: Vec<f32>, pt_bias: f64 }
}

/// 模拟用训练员
///
/// 包装 HandwrittenEvaluator，实现 Trainer trait。
struct SimulationTrainer<'a> {
    evaluator: &'a HandwrittenEvaluator
}

impl<'a> crate::game::Trainer<OnsenGame> for SimulationTrainer<'a> {
    fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng) -> Result<usize> {
        // 只有一个动作时直接返回
        if actions.len() <= 1 {
            return Ok(0);
        }

        // 检查是否是温泉选择场景（所有动作都是 Dig）
        let all_dig = actions.iter().all(|a| matches!(a, OnsenAction::Dig(_)));
        if all_dig {
            return Ok(self.evaluator.select_onsen_index(game, actions));
        }

        // 检查是否是装备升级场景
        let all_upgrade = actions.iter().all(|a| matches!(a, OnsenAction::Upgrade(_)));
        if all_upgrade {
            return Ok(self.evaluator.select_upgrade_action(game, actions));
        }

        // 使用 HandwrittenEvaluator 的 select_action 逻辑
        let selected_action = self.evaluator.select_action(game, rng);
        let idx = match &selected_action {
            Some(action) => actions.iter().position(|a| *a == action.selection).unwrap_or(0),
            None => 0
        };

        Ok(idx)
    }

    fn select_choice(&self, game: &OnsenGame, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        // 使用 HandwrittenEvaluator 的 evaluate_choice 逻辑
        let mut best_idx = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (i, _choice) in choices.iter().enumerate() {
            let value = self.evaluator.evaluate_choice(game, i);
            if value > best_value {
                best_value = value;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }
}

// 说明：E6 的“rollout 动作走 NN”已回退；rollout 全程固定使用 SimulationTrainer(HandwrittenEvaluator)。

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rand::SeedableRng;

    use super::*;
    use crate::{
        game::{InheritInfo, Trainer},
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    /// 单个候选的统计摘要（回归比对的最小单元）
    ///
    /// 只取样本数与均值：直方图整体比对过于笨重，而这两项已足以暴露
    /// 随机流错位——种子一变，均值必然漂移。
    #[derive(Debug, Clone, PartialEq)]
    struct ActionDigest {
        /// 成功样本数
        n: u32,
        /// 分数均值
        mean: f64
    }

    /// 在首个多候选决策点捕获搜索结果，然后固定选 0 号动作
    ///
    /// 直接从外部构造搜索根局面会得到不合法状态（`next()` 空推进会跳过阶段初始化），
    /// 故通过真实的 `run_stage` → `Trainer::select_action` 路径取根节点。
    struct CapturingTrainer {
        /// 被测搜索器
        search: FlatSearch,
        /// 搜索入口种子
        seed: u64,
        /// 捕获到的各候选统计（`None` 表示尚未捕获）
        captured: RefCell<Option<Vec<ActionDigest>>>,
        /// 是否反转候选顺序后再搜索（用于顺序无关性回归）
        reverse: bool
    }

    impl CapturingTrainer {
        /// 构造捕获用 trainer
        fn new(config: SearchConfig, seed: u64, reverse: bool) -> Self {
            Self {
                search: FlatSearch::new(config),
                seed,
                captured: RefCell::new(None),
                reverse
            }
        }

        /// 取出捕获结果
        fn take(&self) -> Result<Vec<ActionDigest>> {
            self.captured
                .borrow_mut()
                .take()
                .ok_or_else(|| anyhow::anyhow!("整局结束仍未遇到多候选决策点"))
        }
    }

    impl Trainer<OnsenGame> for CapturingTrainer {
        fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> Result<usize> {
            if self.captured.borrow().is_none() && actions.len() >= 2 {
                let mut owned = actions.to_vec();
                if self.reverse {
                    owned.reverse();
                }
                let mut rng = StdRng::seed_from_u64(self.seed);
                let out = self.search.search(game, &owned, &mut rng)?;
                let mut digest: Vec<ActionDigest> = out
                    .action_results
                    .iter()
                    .map(|r| ActionDigest {
                        n: r.0.count(),
                        mean: r.0.mean()
                    })
                    .collect();
                // 统一回正序，使正/逆序两次运行的结果可直接逐项比对
                if self.reverse {
                    digest.reverse();
                }
                *self.captured.borrow_mut() = Some(digest);
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &OnsenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 回归基准专用配置
    ///
    /// 强制 `use_ucb=false`：UCB 的样本分配依赖分数，代码一改样本数就变，
    /// 无法作为「改动前后输出一致」的尺子。均匀分配下每个候选固定 `search_n` 次。
    fn regression_config() -> SearchConfig {
        SearchConfig::default().with_search_n(16).with_ucb(false)
    }

    /// 跑一局到首个多候选决策点，返回该点的搜索统计
    fn capture(seed: u64, reverse: bool) -> Result<Vec<ActionDigest>> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = OnsenGame::newgame(102601, &deck, inherit)?;

        let trainer = CapturingTrainer::new(regression_config(), seed, reverse);
        // 局面推进本身用固定种子，保证根局面在各次运行间一致
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&trainer, &mut rng)?;
            if trainer.captured.borrow().is_some() {
                break;
            }
        }
        trainer.take()
    }

    /// 回归 1：同一 seed 两次搜索必须完全一致
    ///
    /// 这是泛型化改造的护栏——没有它，`FlatSearch<G>` 改坏了也无从发现。
    /// 注：仓库测试规范一般要求用 `println` 而非 `assert`，回归基准是刻意的例外：
    /// 只打印不断言，回归就形同虚设。
    #[test]
    fn test_search_reproducible_same_seed() -> Result<()> {
        let a = capture(42, false)?;
        let b = capture(42, false)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "同一 seed 两次搜索结果必须完全一致");
        Ok(())
    }

    /// 回归 2：不同 seed 必须给出不同结果（否则种子根本没接进 rollout）
    #[test]
    fn test_search_seed_actually_used() -> Result<()> {
        let a = capture(42, false)?;
        let b = capture(4242, false)?;
        println!("seed=42   : {a:?}");
        println!("seed=4242 : {b:?}");
        assert_ne!(a, b, "换 seed 必须改变搜索结果，否则种子未接入 rollout");
        Ok(())
    }

    /// 回归 3：候选顺序重排后，各动作统计量按动作对齐后不变
    ///
    /// 专抓「候选索引混进种子派生」——一旦 `seed_at` 吃了候选下标，
    /// 重排 actions 就会让同一动作拿到不同随机流，本测试立刻失败。
    #[test]
    fn test_search_invariant_to_action_order() -> Result<()> {
        let normal = capture(42, false)?;
        let reversed = capture(42, true)?;
        for (i, (a, b)) in normal.iter().zip(reversed.iter()).enumerate() {
            println!("动作 {i}: 正序 n={} mean={:.6} | 逆序 n={} mean={:.6}", a.n, a.mean, b.n, b.mean);
        }
        assert_eq!(normal, reversed, "各动作统计量不应随候选顺序变化");
        Ok(())
    }
}
