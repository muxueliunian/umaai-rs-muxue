//! 神经网络评估器
//!
//! 使用 ONNX 模型进行策略和价值评估。
//!
//! # 输入输出维度
//! - 输入：1121 维特征向量 (Global 587 + Card 89*6)
//! - 输出：61 维 (Policy 50 + Choice 8 + Value 3)
//!
//! # Value 反归一化
//! - scoreMean = VALUE_MEAN + VALUE_SCALE * output[58]
//! - scoreStdev = STDEV_SCALE * abs(output[59])
//! - value = VALUE_MEAN + VALUE_SCALE * output[60]

use std::sync::Arc;

use anyhow::{Context, Result};
use rand::{Rng, rngs::StdRng};
use tract_onnx::prelude::*;

use super::{Evaluator, ValueOutput};
use crate::game::{
    ActionScore, Game, onsen::{action::OnsenAction, game::OnsenGame}
};

// ============================================================================
// 常量定义（与 Python config.py 一致）
// ============================================================================

/// 输入维度
const INPUT_DIM: usize = 1121;

/// 输出维度
const OUTPUT_DIM: usize = 61;

/// Policy 输出维度
const POLICY_DIM: usize = 50;

/// Choice 输出维度
const CHOICE_DIM: usize = 8;

/// Value 反归一化参数 - 均值
const VALUE_MEAN: f64 = 58000.0;

/// Value 反归一化参数 - 缩放
const VALUE_SCALE: f64 = 300.0;

/// Stdev 反归一化参数 - 缩放
const STDEV_SCALE: f64 = 150.0;

// ============================================================================
// 类型别名
// ============================================================================

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ============================================================================
// NeuralNetEvaluator
// ============================================================================

/// 神经网络评估器
///
/// 使用 ONNX 模型进行策略和价值评估。
#[derive(Clone)]
pub struct NeuralNetEvaluator {
    /// ONNX 模型（使用 Arc 共享，因为 SimplePlan 不可克隆）
    model: Arc<OnnxModel>
}

impl NeuralNetEvaluator {
    /// 从 ONNX 文件加载模型
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    ///
    /// # 返回
    /// 加载成功返回 NeuralNetEvaluator，失败返回错误
    pub fn load(model_path: &str) -> Result<Self> {
        log::info!("加载 ONNX 模型: {}", model_path);

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("无法读取 ONNX 模型文件")?
            .into_optimized()
            .context("模型优化失败")?
            .into_runnable()
            .context("模型转换失败")?;

        log::info!("ONNX 模型加载成功");

        Ok(Self { model: Arc::new(model) })
    }

    /// 执行神经网络推理
    ///
    /// # 参数
    /// - `features`: 1121 维输入特征
    ///
    /// # 返回
    /// 61 维输出向量
    pub fn infer(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != INPUT_DIM {
            anyhow::bail!("输入维度错误: 期望 {}, 实际 {}", INPUT_DIM, features.len());
        }

        // 创建输入张量 [1, 1121]
        let input =
            tract_ndarray::Array2::from_shape_vec((1, INPUT_DIM), features.to_vec()).context("创建输入张量失败")?;

        // 运行推理
        let output = self.model.run(tvec!(input.into_tvalue())).context("推理失败")?;

        // 提取输出
        let output_tensor = output[0].to_array_view::<f32>().context("提取输出张量失败")?;

        let result: Vec<f32> = output_tensor.iter().copied().collect();

        if result.len() != OUTPUT_DIM {
            anyhow::bail!("输出维度错误: 期望 {}, 实际 {}", OUTPUT_DIM, result.len());
        }

        Ok(result)
    }

    /// 从输出中提取 Policy 概率分布
    fn extract_policy(&self, output: &[f32]) -> Vec<f32> {
        output[0..POLICY_DIM].to_vec()
    }

    /// 从输出中提取 Choice 概率分布
    fn extract_choice(&self, output: &[f32]) -> Vec<f32> {
        output[POLICY_DIM..POLICY_DIM + CHOICE_DIM].to_vec()
    }

    /// 从输出中提取 Value（反归一化）
    fn extract_value(&self, output: &[f32]) -> ValueOutput {
        let score_mean = VALUE_MEAN + VALUE_SCALE * output[POLICY_DIM + CHOICE_DIM] as f64;
        let score_stdev = STDEV_SCALE * output[POLICY_DIM + CHOICE_DIM + 1] as f64;
        // output[POLICY_DIM + CHOICE_DIM + 2] 是 value，但我们使用 score_mean 作为主要评估值
        ValueOutput::new(score_mean, score_stdev.abs())
    }

    /// 根据 Policy logits 采样选择动作索引
    ///
    /// 注意：神经网络输出的是 logits（可为负数），不能直接当作概率使用。
    /// 这里对合法动作做 softmax，再按概率采样。
    fn sample_action_index(&self, logits: &[f32], legal_mask: &[bool], rng: &mut StdRng) -> usize {
        // 找到合法动作中最大的 logit（softmax 数值稳定）
        let mut max_logit = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if i < legal_mask.len() && legal_mask[i] && v > max_logit {
                max_logit = v;
            }
        }

        // 没有任何合法动作，回退到第一个合法动作
        if !max_logit.is_finite() {
            return legal_mask.iter().position(|&x| x).unwrap_or(0);
        }

        // 计算 softmax 权重（只对合法动作赋值）
        let mut weights: Vec<f64> = vec![0.0; logits.len()];
        let mut sum: f64 = 0.0;
        for (i, &v) in logits.iter().enumerate() {
            if i < legal_mask.len() && legal_mask[i] {
                let w = ((v - max_logit) as f64).exp();
                weights[i] = w;
                sum += w;
            }
        }

        if sum <= 0.0 || !sum.is_finite() {
            // 数值异常时回退到最后一个合法动作（保持确定性）
            return legal_mask.iter().rposition(|&x| x).unwrap_or(0);
        }

        // 采样：在 [0, sum) 上采样，再落到累计权重区间
        let r: f64 = rng.random::<f64>() * sum;
        let mut acc = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if r <= acc {
                return i;
            }
        }

        // 理论上不会走到这里，兜底返回最后一个合法动作
        legal_mask.iter().rposition(|&x| x).unwrap_or(0)
    }

    /// 将动作转换为全局索引
    ///
    /// 与 sample_collector::action_to_global_index 保持一致
    fn action_to_global_index(action: &OnsenAction) -> Option<usize> {
        match action {
            OnsenAction::Train(t) => Some(*t as usize),
            OnsenAction::Sleep => Some(5),
            OnsenAction::NormalOuting => Some(6),
            OnsenAction::FriendOuting => Some(7),
            OnsenAction::Race => Some(8),
            OnsenAction::Clinic => Some(9),
            OnsenAction::PR => Some(10),
            OnsenAction::Dig(idx) => Some(11 + *idx as usize),
            OnsenAction::Upgrade(idx) => Some(21 + *idx as usize),
            OnsenAction::UseTicket(is_super) => Some(if *is_super { 25 } else { 24 })
        }
    }
}

// ============================================================================
// Evaluator trait 实现
// ============================================================================

impl Evaluator<OnsenGame> for NeuralNetEvaluator {
    fn select_action(&self, game: &OnsenGame, rng: &mut StdRng) -> Option<ActionScore<OnsenAction>> {
        // 获取可选动作列表
        let actions = game.list_actions().ok()?;
        if actions.is_empty() {
            return None;
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        let output = match self.infer(&features) {
            Ok(o) => o,
            Err(e) => {
                log::warn!("神经网络推理失败: {}", e);
                // 回退到随机选择
                return Some(ActionScore::new(actions[0].clone(), actions, vec![]));
            }
        };

        // 提取 Policy
        let policy = self.extract_policy(&output);

        // 构建合法动作掩码（使用全局动作索引）
        let mut legal_mask = vec![false; POLICY_DIM];
        for action in &actions {
            if let Some(idx) = Self::action_to_global_index(action) {
                if idx < POLICY_DIM {
                    legal_mask[idx] = true;
                }
            }
        }

        // 采样选择全局动作索引
        let global_idx = self.sample_action_index(&policy, &legal_mask, rng);

        // 找到对应的动作
        for action in &actions {
            if let Some(idx) = Self::action_to_global_index(action) {
                if idx == global_idx {
                    return Some(ActionScore::new(action.clone(), actions, vec![]));
                }
            }
        }

        // 如果没找到，返回第一个动作
        Some(ActionScore::new(actions[0].clone(), actions, vec![]))
    }

    fn evaluate(&self, game: &OnsenGame) -> ValueOutput {
        // 如果游戏结束，返回实际分数
        if game.turn >= game.max_turn() {
            let score = game.uma.calc_score() as f64;
            return ValueOutput::new(score, 0.0);
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        match self.infer(&features) {
            Ok(output) => self.extract_value(&output),
            Err(e) => {
                log::warn!("神经网络推理失败: {}", e);
                // 回退到简单评估
                let score = game.uma.calc_score() as f64;
                let progress = game.turn as f64 / game.max_turn() as f64;
                let stdev = 500.0 * (1.0 - progress) + 100.0;
                ValueOutput::new(score, stdev)
            }
        }
    }

    fn evaluate_choice(&self, game: &OnsenGame, choice_index: usize) -> f64 {
        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        match self.infer(&features) {
            Ok(output) => {
                let choice = self.extract_choice(&output);
                if choice_index < choice.len() {
                    choice[choice_index] as f64
                } else {
                    0.0
                }
            }
            Err(_) => {
                // 默认选择第一个
                if choice_index == 0 { 1.0 } else { 0.0 }
            }
        }
    }

    fn select_action_from_list(&self, game: &OnsenGame, actions: &[OnsenAction], _rng: &mut StdRng) -> usize {
        if actions.is_empty() {
            return 0;
        }

        // 提取特征
        let features = game.extract_nn_features(None);

        // 推理
        let output = match self.infer(&features) {
            Ok(o) => o,
            Err(_) => return 0
        };

        // 提取 Policy
        let policy = self.extract_policy(&output);

        // 找到给定动作列表中 Policy 值最大的动作（使用全局索引）
        let mut best_idx = 0;
        let mut best_value = f32::NEG_INFINITY;

        for (action_idx, action) in actions.iter().enumerate() {
            if let Some(global_idx) = Self::action_to_global_index(action) {
                if global_idx < policy.len() {
                    let value = policy[global_idx];
                    if value > best_value {
                        best_value = value;
                        best_idx = action_idx;
                    }
                }
            }
        }

        best_idx
    }
}

// ============================================================================
// Send + Sync 实现
// ============================================================================

// NeuralNetEvaluator 通过 Arc 共享模型，是线程安全的
unsafe impl Send for NeuralNetEvaluator {}
unsafe impl Sync for NeuralNetEvaluator {}
