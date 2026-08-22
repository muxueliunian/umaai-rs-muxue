//! 扁平蒙特卡洛搜索
//!
//! 对每个合法动作执行多次模拟，统计分数分布，选择最优动作。
//! 支持两种搜索策略：
//! - 均匀分配：每个动作平均分配搜索次数（并行化）
//! - UCB 分配：根据 UCB 公式动态分配搜索资源（C++ UmaAi 风格）

use anyhow::{Result, anyhow, bail, ensure};
use log::{debug, warn};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::{
    config::{SearchConfig, TOTAL_TURN},
    result::{ActionResult, SearchOutput},
    searchable::{FlatSearchGame, SearchScore},
    seeds::RolloutSeeds
};
#[cfg(feature = "onnx")]
use crate::neural::{ThreadLocalNeuralNetLeafEvaluator, ThreadLocalNeuralNetLeafStatsSnapshot};
use crate::{
    game::{
        Game,
        onsen::{OnsenTurnStage, action::OnsenAction, game::OnsenGame},
        ramen::{RamenAction, RamenGame}
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
pub struct FlatSearch<G: FlatSearchGame = OnsenGame>
where
    G::Action: Send + Sync + Clone
{
    /// 手写评估器（温泉 rollout 与 leaf 估值用）
    ///
    /// 仅温泉路径使用：`HandwrittenEvaluator` 只 impl 了 `Evaluator<OnsenGame>`。
    /// 拉面走 `G::RolloutTrainer`，该字段闲置。Phase 1.4 保留此不对称以免掀开
    /// `umaai` 的 `with_leaf_evaluator_handwritten()` 调用签名。
    rollout_evaluator: HandwrittenEvaluator,

    /// leaf eval 评估器（用于 max_depth>0 截断估值；温泉专用）
    leaf_evaluator: LeafEvaluator,

    /// rollout 决策器（由剧本指定）
    rollout_trainer: G::RolloutTrainer,

    /// 搜索配置
    config: SearchConfig,

    /// E4：leaf eval 微批大小（仅在 max_depth>0 && leaf_eval=nn 时生效）
    rollout_batch_size: usize
}

impl<G: FlatSearchGame> FlatSearch<G>
where
    G::Action: Send + Sync + Clone
{
    /// 创建搜索器
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rollout_evaluator: HandwrittenEvaluator::new(),
            leaf_evaluator: LeafEvaluator::Handwritten,
            rollout_trainer: G::default_rollout_trainer(),
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

    /// 通用搜索内核
    ///
    /// 负责候选分配（均匀 / UCB）、CRN 种子、统计与并行调度；
    /// 「一次 rollout 怎么跑」由调用方以闭包注入，使剧本特判（如温泉
    /// `Dig`/`Upgrade`）留在各自的具体 impl 里，不进公共 trait。
    ///
    /// # 为什么用闭包而非 trait 钩子
    ///
    /// 泛型 `impl<G> FlatSearch<G>` 内部调用 `self.simulate()` 时，方法解析只会
    /// 找到泛型版本，**永远不会**落到更具体的 `impl FlatSearch<OnsenGame>`。
    /// 若把特判留作具体 impl 的同名方法，它会静默变成死代码、温泉 `Dig`/`Upgrade`
    /// 改走通用路径——编译通过但行为改变。闭包注入从根上避免这个陷阱。
    pub fn search_with<F>(
        &self, game: &G, actions: &[G::Action], rng: &mut StdRng, rollout: F
    ) -> Result<SearchOutput<G::Action>>
    where
        F: Fn(&G, &G::Action, u64) -> Result<SearchScore> + Sync
    {
        if actions.is_empty() {
            bail!("没有可用动作");
        }
        ensure!(
            G::SUPPORTS_TRUNCATED_LEAF || self.config.max_depth == 0,
            "本剧本没有 leaf 估值器，max_depth 必须为 0（当前 {}）",
            self.config.max_depth
        );

        // 计算激进度因子（C++ 风格，无随机性）
        let radical_factor = self.compute_radical_factor(game.turn() as usize);

        // 本次搜索的 rollout 种子表：所有候选共享，由传入 rng 派生（可复现性入口）
        let seeds = RolloutSeeds::from_rng(rng);
        debug!(
            "[回合 {}] 开始搜索: {} 个动作, search_n={}, max_depth={}, radical_factor={:.1}, ucb={}, 根种子={:#018x}",
            game.turn(),
            actions.len(),
            self.config.search_n,
            self.config.max_depth,
            radical_factor,
            self.config.use_ucb,
            seeds.root()
        );

        let action_results = if self.config.use_ucb {
            self.search_ucb(game, actions, radical_factor, &seeds, &rollout)?
        } else {
            self.search_uniform(game, actions, &seeds, &rollout)?
        };

        // 某候选一次都没跑成功时其统计全是空的，继续用下去等于拿垃圾数据排序
        for (i, (result, _)) in action_results.iter().enumerate() {
            if result.count() == 0 {
                bail!("候选动作 {i} 的全部 rollout 均失败，搜索结果不可用");
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

    /// 按当前 `(回合, 阶段)` 重新播种 rollout 随机流（外挂 CRN）
    ///
    /// **仅规则层未改造的剧本（onsen）使用**：onsen 的规则随机仍走传入的
    /// `&mut StdRng`，靠本方法按 `(rollout 种子, 回合, 阶段)` 重播种对齐各候选；
    /// 拉面规则层已由无状态流接管（RNG Refactor Plan v2 §5.2，`fork_for_rollout`
    /// 注入 rule_master），其 rollout 路径不再调用本方法。
    pub fn reseed_for_stage(&self, rng: &mut StdRng, rollout_seed: u64, game: &G) {
        if !self.config.crn_stage_reseed {
            return;
        }
        *rng = StdRng::seed_from_u64(RolloutSeeds::stage_seed(rollout_seed, game.turn(), game.crn_stage_key()));
    }

    /// rollout 决策器
    pub fn rollout_trainer(&self) -> &G::RolloutTrainer {
        &self.rollout_trainer
    }

    /// 通用单次 rollout：执行动作后跑到终局
    ///
    /// 只处理 `max_depth == 0`；截断估值需要 leaf 估值器，属剧本专属能力。
    /// 分支一律经 [`FlatSearchGame::fork_for_rollout`] 建立，不得直接 `clone()`
    /// ——那会漏掉剧本内部随机流的重置。
    pub fn simulate_common(&self, game: &G, action: &G::Action, seed: u64) -> Result<SearchScore> {
        let rng = &mut StdRng::seed_from_u64(seed);
        let mut sim_game = game.fork_for_rollout(seed);
        sim_game.apply_action(action, rng)?;
        while sim_game.next() {
            sim_game.run_stage(&self.rollout_trainer, rng)?;
        }
        sim_game.on_simulation_end(&self.rollout_trainer, rng)?;
        Ok(sim_game.search_score())
    }

    /// 均匀分配搜索（并行化）
    ///
    /// 每个动作平均分配 `search_n` 次搜索。所有候选的第 j 次 rollout 共用
    /// `seeds.seed_at(j)`（CRN 载体），故并行粒度不影响结果。
    ///
    /// 注：此处按候选并行，并行度上限即候选数（≤10）。改为按 `(候选, rollout)`
    /// 扁平并行可提升吞吐且结果位级不变，留作后续性能对照实验。
    fn search_uniform<F>(
        &self, game: &G, actions: &[G::Action], seeds: &RolloutSeeds, rollout: &F
    ) -> Result<Vec<(ActionResult, ActionResult)>>
    where
        F: Fn(&G, &G::Action, u64) -> Result<SearchScore> + Sync
    {
        let n = self.config.search_n;
        let run = |action: &G::Action| -> Result<(ActionResult, ActionResult, usize)> {
            let mut result = ActionResult::new();
            let mut result_pt = ActionResult::new();
            // offset=0：均匀分配下每个候选都从 rollout 0 开始，天然完全配对
            let failed = self.simulate_many(game, action, n, seeds, 0, &mut result, &mut result_pt, rollout)?;
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

    /// 对同一候选连续跑 `n` 次 rollout
    ///
    /// 第 k 次取 `seeds.seed_at(offset + k)` 播种，`offset` 为该候选**已计划**的次数。
    /// 返回失败次数（不中断搜索，由调用方汇总告警）。
    #[allow(clippy::too_many_arguments)]
    fn simulate_many<F>(
        &self, game: &G, action: &G::Action, n: usize, seeds: &RolloutSeeds, offset: usize,
        result: &mut ActionResult, result_pt: &mut ActionResult, rollout: &F
    ) -> Result<usize>
    where
        F: Fn(&G, &G::Action, u64) -> Result<SearchScore> + Sync
    {
        let mut failed = 0usize;
        for k in 0..n {
            match rollout(game, action, seeds.seed_at(offset + k)) {
                Ok(v) => {
                    result.add(v.score);
                    result_pt.add(v.score_pt);
                }
                Err(e) => {
                    debug!("[搜索] rollout {} 失败: {e}", offset + k);
                    failed += 1;
                }
            }
        }
        Ok(failed)
    }

    /// UCB 动态分配搜索
    ///
    /// 使用 UCB 公式动态分配搜索资源，好的动作获得更多搜索次数。
    /// UCB 决策是串行的，但每组模拟内部使用 Rayon 并行化。
    ///
    /// # UCB 公式
    /// search_value = value + cpuct * expected_stdev * sqrt(total_n) / n
    fn search_ucb<F>(
        &self, game: &G, actions: &[G::Action], radical_factor: f64, seeds: &RolloutSeeds, rollout: &F
    ) -> Result<Vec<(ActionResult, ActionResult)>>
    where
        F: Fn(&G, &G::Action, u64) -> Result<SearchScore> + Sync
    {
        let num_actions = actions.len();
        let mut action_results: Vec<(ActionResult, ActionResult)> = vec![Default::default(); num_actions];
        let group_size = self.config.search_group_size;
        ensure!(group_size > 0, "search_group_size 不能为 0（UCB 分配会死循环）");
        let use_parallel = self.use_parallel_simulation();

        // 各候选**已计划**的 rollout 次数（≠ 已成功次数）
        //
        // 种子偏移必须用计划次数而非 `ActionResult::count()`：后者会因 rollout 失败
        // 而少计，导致同一 rollout 序号在不同候选上错位，破坏配对。
        let mut planned = vec![0usize; num_actions];

        // 第一阶段：每个动作先搜一组（并行）
        let run_initial = |action: &G::Action| -> Result<(ActionResult, ActionResult, usize)> {
            let mut result = ActionResult::new();
            let mut result_pt = ActionResult::new();
            let failed =
                self.simulate_many(game, action, group_size, seeds, 0, &mut result, &mut result_pt, rollout)?;
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
            // 终止判据用**已计划**次数，不能用 ActionResult::count()（成功次数）
            //
            // 用成功次数会在 rollout 稳定失败时死循环：失败只增加 planned、不增加 count，
            // 于是 count 永远达不到 search_n，而末尾的「零样本」检查在本函数返回之后
            // 才执行，根本到不了。
            let max_planned = planned.iter().copied().max().unwrap_or(0);
            if max_planned >= self.config.search_n {
                break;
            }

            // 使用 UCB 公式选择下一个要搜索的动作
            let best_action_idx = self.select_ucb_action(&action_results, radical_factor, total_n);
            let action = &actions[best_action_idx];

            // 该候选已计划 offset 次，本组取 seeds[offset..offset+group_size]。
            // 两个候选因而在 0..min(n_a, n_b) 上完全配对，多出的部分为 unpaired，
            // 这是 CRN 在不等样本数下的标准做法。
            let offset = planned[best_action_idx];
            let run_one = |k: usize| -> Option<SearchScore> {
                match rollout(game, action, seeds.seed_at(offset + k)) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        debug!("[搜索][UCB] rollout {} 失败: {e}", offset + k);
                        None
                    }
                }
            };
            let scores: Vec<SearchScore> = if use_parallel {
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

            for v in scores {
                action_results[best_action_idx].0.add(v.score);
                action_results[best_action_idx].1.add(v.score_pt);
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

}

impl FlatSearch<OnsenGame> {
    /// 温泉根节点搜索
    ///
    /// 走通用内核 [`FlatSearch::search_with`]，但注入温泉专属的 rollout 分发：
    /// `Dig`/`Upgrade` 不是回合阶段，而是嵌套的 `pending_selection` 选择，
    /// 必须走各自的特判路径，不能落到 `apply_action -> while next` 的通用流程。
    ///
    /// 保持既有对外签名不变，`MctsTrainer` 与 `umaai` 通道层无需改动。
    pub fn search(
        &self, game: &OnsenGame, actions: &[OnsenAction], rng: &mut StdRng
    ) -> Result<SearchOutput<OnsenAction>> {
        self.search_with(game, actions, rng, |game, action, seed| {
            let (score, score_pt) = self.simulate(game, action, seed)?;
            Ok(SearchScore { score, score_pt })
        })
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
    /// 单次 rollout
    ///
    /// `seed` 为该次 rollout 的种子（由 [`RolloutSeeds::seed_at`] 给出，所有候选共享）。
    /// 开启 [`SearchConfig::crn_stage_reseed`] 时，每进入一个阶段会按
    /// `(seed, 回合, 阶段)` 重新播种，使各候选在同一阶段抽到同一份随机性。
    fn simulate(&self, game: &OnsenGame, action: &OnsenAction, seed: u64) -> Result<(f64, f64)> {
        let rng = &mut StdRng::seed_from_u64(seed);
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
                    self.reseed_for_stage(rng, seed, &sim_game);
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
                self.reseed_for_stage(rng, seed, &sim_game);
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


    #[cfg(feature = "onnx")]
    /// 单次 rollout，跑到终局或 `max_depth` 截断处（NN leaf 微批路径用）
    ///
    /// `seed` 语义与 [`Self::simulate`] 一致：同一 rollout 序号在所有候选上共享，
    /// 且按 `(回合, 阶段)` 重播种，使本路径与 [`Self::simulate`] 的 CRN 行为一致。
    fn simulate_until_terminal_or_leaf(&self, game: &OnsenGame, action: &OnsenAction, seed: u64) -> Result<SimOutcome> {
        let rng = &mut StdRng::seed_from_u64(seed);
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
                self.reseed_for_stage(rng, seed, &sim_game);
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
            self.reseed_for_stage(rng, seed, &sim_game);
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

impl FlatSearch<RamenGame> {
    /// 拉面根节点搜索
    ///
    /// 全部候选走通用 rollout：拉面的 `RamenSelect` / `SpecialSelect` / `Train`
    /// 已是正规回合阶段（`run_stage` -> `list_actions` -> `select_action`），
    /// 不像温泉 `Dig`/`Upgrade` 那样需要特判。
    ///
    /// # Phase 1 范围限制
    ///
    /// 只接受 [`Game::list_actions`] 产出的**标准分阶段动作**。
    /// `list_combined_ramen_select_actions()` 的合并动作**不可**传入：
    /// 通用 `apply_action` 在 `RamenSelect` 只写 `pending_ramen`、清零
    /// `special_targets`，**不设 `combined_decision`**，会静默丢掉隐藏风味。
    /// 合并动作需走 `apply_combined_ramen_decision`，留待 Phase 2。
    pub fn search(
        &self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng
    ) -> Result<SearchOutput<RamenAction>> {
        self.search_with(game, actions, rng, |game, action, seed| self.simulate_common(game, action, seed))
    }
}

/// 回合阶段编号（种子派生用）
///
/// 显式 match 而非依赖枚举判别值：`OnsenTurnStage` 的变体顺序若调整，
/// 这里会编译报错提醒同步，而不是静默改变所有历史种子。
fn stage_id(stage: &OnsenTurnStage) -> u64 {
    match stage {
        OnsenTurnStage::Begin => 0,
        OnsenTurnStage::Distribute => 1,
        OnsenTurnStage::Bathing => 2,
        OnsenTurnStage::Train => 3,
        OnsenTurnStage::AfterTrain => 4
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
        game::{
            InheritInfo, Trainer,
            ramen::{RamenAction, RamenGame}
        },
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
                .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
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

    /// 捕获首个多候选决策点的**局面本身**（CRN 测量需要直接调 `simulate`）
    struct RootCapture {
        /// 捕获到的 (根局面, 候选表)
        got: RefCell<Option<(OnsenGame, Vec<OnsenAction>)>>
    }

    impl Trainer<OnsenGame> for RootCapture {
        fn select_action(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> Result<usize> {
            if self.got.borrow().is_none() && actions.len() >= 2 {
                *self.got.borrow_mut() = Some((game.clone(), actions.to_vec()));
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &OnsenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 取首个多候选决策点的根局面与候选表
    fn root_state() -> Result<(OnsenGame, Vec<OnsenAction>)> {
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
        let cap = RootCapture { got: RefCell::new(None) };
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&cap, &mut rng)?;
            if cap.got.borrow().is_some() {
                break;
            }
        }
        cap.got
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
    }

    /// 样本均值
    fn mean_of(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len().max(1) as f64
    }

    /// 样本方差（无偏）
    fn var_of(xs: &[f64]) -> f64 {
        let m = mean_of(xs);
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len().saturating_sub(1)).max(1) as f64
    }

    /// Pearson 相关系数
    fn corr_of(a: &[f64], b: &[f64]) -> f64 {
        let (ma, mb) = (mean_of(a), mean_of(b));
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let da: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>().sqrt();
        let db: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>().sqrt();
        if da <= 0.0 || db <= 0.0 { 0.0 } else { cov / (da * db) }
    }

    /// 回归基准专用配置
    ///
    /// 强制 `use_ucb=false`：UCB 的样本分配依赖分数，代码一改样本数就变，
    /// 无法作为「改动前后输出一致」的尺子。均匀分配下每个候选固定 `search_n` 次。
    fn regression_config() -> SearchConfig {
        SearchConfig::default().with_search_n(16).with_ucb(false)
    }

    /// 回归基准配置（UCB 路径）
    ///
    /// `use_ucb` 默认为 `true`，是活跃入口实际走的路径，必须单独覆盖。
    /// `group_size` 取小值以免单次搜索过久。
    fn regression_config_ucb() -> SearchConfig {
        SearchConfig::default()
            .with_search_n(16)
            .with_ucb(true)
            .with_search_group_size(4)
    }

    /// 跑一局到首个多候选决策点，返回该点的搜索统计（均匀分配）
    fn capture(seed: u64, reverse: bool) -> Result<Vec<ActionDigest>> {
        capture_with(regression_config(), seed, reverse)
    }

    /// 同 [`capture`]，但可指定搜索配置
    fn capture_with(config: SearchConfig, seed: u64, reverse: bool) -> Result<Vec<ActionDigest>> {
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

        let trainer = CapturingTrainer::new(config, seed, reverse);
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

    /// 回归 4：UCB 路径下同一 seed 两次搜索必须一致
    ///
    /// `use_ucb` 默认为 `true`，是 `umaai` / `umasim` 活跃入口实际走的路径。
    /// 回归 1~3 写死 `use_ucb=false`（UCB 分配依赖分数，不适合当泛型化的尺子），
    /// 但可复现性本身在 UCB 下同样必须成立，故单独覆盖。
    #[test]
    fn test_search_ucb_reproducible() -> Result<()> {
        let a = capture_with(regression_config_ucb(), 42, false)?;
        let b = capture_with(regression_config_ucb(), 42, false)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "UCB 路径下同一 seed 两次搜索结果必须一致");
        Ok(())
    }

    /// UCB 路径的候选顺序敏感性（诊断用，不断言）
    ///
    /// UCB 按分数动态分配预算且平局时取索引最小者，故候选重排**可能**改变各动作
    /// 拿到的样本数——这是算法固有性质，不是可复现性缺陷。本测试只打印差异供观察；
    /// 泛型化的顺序无关性护栏由均匀分配的回归 3 承担。
    #[test]
    fn test_search_ucb_order_sensitivity() -> Result<()> {
        let normal = capture_with(regression_config_ucb(), 42, false)?;
        let reversed = capture_with(regression_config_ucb(), 42, true)?;
        let mut diff = 0usize;
        for (i, (a, b)) in normal.iter().zip(reversed.iter()).enumerate() {
            let same = a == b;
            if !same {
                diff += 1;
            }
            println!(
                "动作 {i}: 正序 n={} mean={:.6} | 逆序 n={} mean={:.6} | {}",
                a.n,
                a.mean,
                b.n,
                b.mean,
                if same { "一致" } else { "不同" }
            );
        }
        println!("UCB 下候选重排导致 {diff}/{} 个动作统计不同", normal.len());
        Ok(())
    }

    // ========== 拉面根节点冒烟（Phase 1.4 验收） ==========

    /// 捕获拉面首个多候选决策点的局面与候选表
    struct RamenRootCapture {
        /// 捕获到的 (根局面, 候选表)
        got: RefCell<Option<(RamenGame, Vec<RamenAction>)>>
    }

    impl Trainer<RamenGame> for RamenRootCapture {
        fn select_action(&self, game: &RamenGame, actions: &[RamenAction], _rng: &mut StdRng) -> Result<usize> {
            // 避开第 3 年地区选择：该阶段最多 120 个候选，冒烟不必付这个代价
            let skip = matches!(game.stage, crate::game::ramen::RamenStage::RegionSelect);
            if self.got.borrow().is_none() && actions.len() >= 2 && !skip {
                *self.got.borrow_mut() = Some((game.clone(), actions.to_vec()));
            }
            Ok(0)
        }

        fn select_choice(&self, _game: &RamenGame, _choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
            Ok(0)
        }
    }

    /// 取拉面首个多候选决策点
    fn ramen_root() -> Result<(RamenGame, Vec<RamenAction>)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let inherit = InheritInfo {
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        let mut game = RamenGame::newgame(102601, &deck, inherit)?;
        let cap = RamenRootCapture { got: RefCell::new(None) };
        let mut rng = StdRng::seed_from_u64(20260822);
        while game.next() {
            game.run_stage(&cap, &mut rng)?;
            if cap.got.borrow().is_some() {
                break;
            }
        }
        cap.got
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("整局结束仍未遇到多候选决策点"))
    }

    /// 跑一次拉面根节点搜索，返回各候选统计
    fn ramen_search(seed: u64) -> Result<Vec<ActionDigest>> {
        let (game, actions) = ramen_root()?;
        let search: FlatSearch<RamenGame> = FlatSearch::new(regression_config());
        let mut rng = StdRng::seed_from_u64(seed);
        let out = search.search(&game, &actions, &mut rng)?;
        Ok(out
            .action_results
            .iter()
            .map(|r| ActionDigest {
                n: r.0.count(),
                mean: r.0.mean()
            })
            .collect())
    }

    /// Phase 1.4 验收 1：拉面能跑通根节点搜索，且固定种子可复现
    #[test]
    fn test_ramen_root_search_reproducible() -> Result<()> {
        let (game, actions) = ramen_root()?;
        println!("拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个", game.turn(), game.stage, actions.len());

        let a = ramen_search(42)?;
        let b = ramen_search(42)?;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            println!("动作 {i}: n={} mean={:.6} | n={} mean={:.6}", x.n, x.mean, y.n, y.mean);
        }
        assert_eq!(a, b, "拉面根节点搜索必须可复现");
        Ok(())
    }

    /// Phase 1.4 验收 2：换种子必须改变结果
    ///
    /// 拉面的关键在于规则层第二条随机流 `internal_rng`——若 `fork_for_rollout`
    /// 没有注入它，`take_internal_rng()` 会回退 `from_os_rng()`，
    /// 届时验收 1 会失败（同种子两次不一致）。两条测试合起来才能证明两条流都受控。
    #[test]
    fn test_ramen_root_search_seed_used() -> Result<()> {
        let a = ramen_search(42)?;
        let b = ramen_search(4242)?;
        println!("seed=42   : {a:?}");
        println!("seed=4242 : {b:?}");
        assert_ne!(a, b, "换 seed 必须改变拉面搜索结果");
        Ok(())
    }

    /// CRN 收益实测（拉面 / 目标剧本）
    ///
    /// onsen 上测得 1.31x → 3.65x，但拉面三阶段决策使状态分叉更剧烈，
    /// 相关性预计更弱。计划中明确要求「在目标剧本上重测后才可据此下调 search_n」，
    /// 本测试即该重测。
    ///
    /// `cargo test --release -p umasim --lib test_crn_pairing_gain_ramen -- --ignored --nocapture`
    #[test]
    #[ignore = "CRN 收益测量，耗时较长，按需手动运行"]
    fn test_crn_pairing_gain_ramen() -> Result<()> {
        /// 每候选 rollout 次数
        const ROLLOUTS: usize = 200;

        let (game, actions) = ramen_root()?;
        println!("拉面根局面: 回合 {} 阶段 {:?}，候选 {} 个
", game.turn(), game.stage, actions.len());

        for reseed in [false, true] {
            let cfg = SearchConfig::default().with_crn_stage_reseed(reseed);
            let search: FlatSearch<RamenGame> = FlatSearch::new(cfg);
            let seeds = RolloutSeeds::from_root(20260822);

            let mut scores: Vec<Vec<f64>> = Vec::with_capacity(actions.len());
            let mut failed_total = 0usize;
            for action in &actions {
                let col: Vec<Option<f64>> = (0..ROLLOUTS)
                    .into_par_iter()
                    .map(|j| search.simulate_common(&game, action, seeds.seed_at(j)).ok().map(|v| v.score))
                    .collect();
                failed_total += col.iter().filter(|x| x.is_none()).count();
                scores.push(col.into_iter().flatten().collect());
            }
            if failed_total > 0 {
                println!("  ⚠ 共 {failed_total} 次 rollout 失败，配对可能错位");
            }

            let label = if reseed { "开启按阶段重播种" } else { "仅共享起始种子" };
            let mut corrs = Vec::new();
            let mut gains = Vec::new();
            for a in 0..scores.len() {
                for b in (a + 1)..scores.len() {
                    let (xa, xb) = (&scores[a], &scores[b]);
                    let n = xa.len().min(xb.len());
                    if n < 2 {
                        continue;
                    }
                    let diff: Vec<f64> = (0..n).map(|j| xa[j] - xb[j]).collect();
                    let indep = var_of(&xa[..n]) + var_of(&xb[..n]);
                    let paired = var_of(&diff);
                    if paired > 0.0 {
                        gains.push(indep / paired);
                    }
                    corrs.push(corr_of(&xa[..n], &xb[..n]));
                }
            }
            println!(
                "===== {label} =====
候选对数 {} | 平均 corr = {:.4} | 平均等效倍率 = {:.2}x | 区间 [{:.2}, {:.2}]
",
                corrs.len(),
                mean_of(&corrs),
                mean_of(&gains),
                gains.iter().cloned().fold(f64::INFINITY, f64::min),
                gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
        }
        Ok(())
    }

    /// CRN 收益实测：配对相关系数与等效样本倍率
    ///
    /// 对同一根局面，各候选在 rollout 序号 j 上共享种子，逐 j 收集分数，
    /// 再按候选两两计算：
    ///
    /// - `corr`：配对相关系数。越高说明「同一份未来随机性」共享得越充分。
    /// - `倍率`：`(Var_a + Var_b) / Var(X_a - X_b)`。独立抽样时分母等于分子，
    ///   倍率为 1；配对生效则分母变小、倍率 > 1，等价于把 `search_n` 放大同样倍数。
    ///
    /// 关闭 / 开启 `crn_stage_reseed` 各测一次，差值即按阶段重播种的真实收益。
    /// 计划中「等效 4–10 倍」为无实测支撑的预期值，以本测试输出为准。
    ///
    /// 耗时较长（每配置 候选数 × ROLLOUTS 次整局 rollout），故标记 ignore：
    /// `cargo test --release -p umasim --lib test_crn_pairing_gain -- --ignored --nocapture`
    #[test]
    #[ignore = "CRN 收益测量，耗时较长，按需手动运行"]
    fn test_crn_pairing_gain() -> Result<()> {
        /// 每候选 rollout 次数
        const ROLLOUTS: usize = 200;

        let (game, actions) = root_state()?;
        println!("根局面: 回合 {} 阶段 {:?}，候选 {} 个
", game.turn, game.stage, actions.len());

        for reseed in [false, true] {
            let cfg = SearchConfig::default().with_crn_stage_reseed(reseed);
            let search = FlatSearch::new(cfg);
            let seeds = RolloutSeeds::from_root(20260822);

            // scores[候选][rollout 序号]
            let mut scores: Vec<Vec<f64>> = Vec::with_capacity(actions.len());
            for (i, action) in actions.iter().enumerate() {
                let col: Vec<(usize, Result<(f64, f64)>)> = (0..ROLLOUTS)
                    .into_par_iter()
                    .map(|j| (j, search.simulate(&game, action, seeds.seed_at(j))))
                    .collect();
                let mut ok_scores = Vec::with_capacity(col.len());
                let mut first_err: Option<String> = None;
                for (_, r) in col {
                    match r {
                        Ok(v) => ok_scores.push(v.0),
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e.to_string());
                            }
                        }
                    }
                }
                // 失败会让配对错位（不同候选丢掉的 j 不同），必须先确认没有失败
                println!(
                    "  候选 {i}: 成功 {}/{ROLLOUTS}{}",
                    ok_scores.len(),
                    first_err.map(|e| format!("，首个失败: {e}")).unwrap_or_default()
                );
                scores.push(ok_scores);
            }

            let label = if reseed { "开启按阶段重播种" } else { "仅共享起始种子" };
            println!("===== {label} =====");

            let mut corrs = Vec::new();
            let mut gains = Vec::new();
            for a in 0..scores.len() {
                for b in (a + 1)..scores.len() {
                    let (xa, xb) = (&scores[a], &scores[b]);
                    let n = xa.len().min(xb.len());
                    if n < 2 {
                        continue;
                    }
                    let diff: Vec<f64> = (0..n).map(|j| xa[j] - xb[j]).collect();
                    let indep = var_of(&xa[..n]) + var_of(&xb[..n]);
                    let paired = var_of(&diff);
                    let gain = if paired > 0.0 { indep / paired } else { f64::NAN };
                    corrs.push(corr_of(&xa[..n], &xb[..n]));
                    gains.push(gain);
                }
            }
            println!(
                "候选对数 {} | 平均 corr = {:.4} | 平均等效倍率 = {:.2}x | 倍率区间 [{:.2}, {:.2}]",
                corrs.len(),
                mean_of(&corrs),
                mean_of(&gains),
                gains.iter().cloned().fold(f64::INFINITY, f64::min),
                gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
            println!();
        }
        Ok(())
    }
}
