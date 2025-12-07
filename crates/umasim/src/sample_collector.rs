/// 样本收集器模块
/// 
/// 用于在模拟过程中收集训练数据
/// 每回合记录游戏状态、选择的动作、事件选项等信息
/// 游戏结束后根据最终分数生成训练样本

use crate::training_sample::{TrainingSample, NN_INPUT_DIM};

/// 单回合数据
#[derive(Debug, Clone)]
pub struct TurnData {
    /// 590 维特征向量
    pub features: Vec<f32>,
    /// 选择的动作索引
    pub action_idx: usize,
    /// 可选动作数量
    pub num_actions: usize,
    /// 事件选项索引（如果有）
    pub choice_idx: Option<usize>,
    /// 事件选项数量
    pub num_choices: usize,
}

/// 样本收集器
/// 
/// 在游戏过程中收集每回合的决策数据
/// 游戏结束后生成训练样本
#[derive(Debug, Clone)]
pub struct SampleCollector {
    /// 每回合的数据
    turn_data: Vec<TurnData>,
    /// 最终分数
    final_score: i32,
    /// 是否已完成
    is_finished: bool,
}

impl SampleCollector {
    /// 创建新的样本收集器
    pub fn new() -> Self {
        Self {
            turn_data: Vec::with_capacity(78), // 预分配 78 回合
            final_score: 0,
            is_finished: false,
        }
    }

    /// 记录一个回合的动作选择
    /// 
    /// # 参数
    /// - `features`: 当前游戏状态的特征向量（590 维）
    /// - `action_idx`: 选择的动作索引
    /// - `num_actions`: 可选动作数量
    pub fn record_turn(&mut self, features: Vec<f32>, action_idx: usize, num_actions: usize) {
        debug_assert_eq!(features.len(), NN_INPUT_DIM, "特征维度必须是 {}", NN_INPUT_DIM);
        debug_assert!(action_idx < num_actions, "动作索引超出范围");
        
        self.turn_data.push(TurnData {
            features,
            action_idx,
            num_actions,
            choice_idx: None,
            num_choices: 0,
        });
    }

    /// 记录事件选项选择
    /// 
    /// 在 `record_turn` 之后调用，为当前回合添加事件选项信息
    /// 
    /// # 参数
    /// - `choice_idx`: 选择的事件选项索引
    /// - `num_choices`: 可选事件选项数量
    pub fn record_choice(&mut self, choice_idx: usize, num_choices: usize) {
        if let Some(last) = self.turn_data.last_mut() {
            debug_assert!(choice_idx < num_choices, "选项索引超出范围");
            last.choice_idx = Some(choice_idx);
            last.num_choices = num_choices;
        }
    }

    /// 设置最终分数
    pub fn set_final_score(&mut self, score: i32) {
        self.final_score = score;
        self.is_finished = true;
    }

    /// 获取最终分数
    pub fn final_score(&self) -> i32 {
        self.final_score
    }

    /// 获取回合数
    pub fn num_turns(&self) -> usize {
        self.turn_data.len()
    }

    /// 是否已完成
    pub fn is_finished(&self) -> bool {
        self.is_finished
    }

    /// 生成训练样本
    /// 
    /// 将收集的回合数据转换为训练样本
    /// 每个回合生成一个样本，使用最终分数作为 value target
    /// 
    /// # 返回
    /// 训练样本列表，每个回合一个样本
    pub fn finalize(self) -> Vec<TrainingSample> {
        if !self.is_finished {
            log::warn!("SampleCollector 未设置最终分数，使用默认值 0");
        }

        let final_score = self.final_score as f32;
        
        self.turn_data.into_iter().map(|turn| {
            // Policy target: one-hot 编码选择的动作
            let mut policy_target = vec![0.0_f32; 50];
            if turn.action_idx < 50 {
                policy_target[turn.action_idx] = 1.0;
            }

            // Choice target: one-hot 编码选择的事件选项
            let mut choice_target = vec![0.0_f32; 5];
            if let Some(idx) = turn.choice_idx {
                if idx < 5 {
                    choice_target[idx] = 1.0;
                }
            }

            // Value target: [scoreMean, scoreStdev, value]
            // 使用最终分数作为均值，固定标准差 500
            let value_target = vec![
                final_score / 1000.0,  // 归一化分数
                0.5,                    // 标准差 (500 / 1000)
                final_score / 1000.0,  // 价值
            ];

            TrainingSample::new(
                turn.features,
                policy_target,
                choice_target,
                value_target,
            )
        }).collect()
    }
}

impl Default for SampleCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 游戏样本（包含所有回合的样本和最终分数）
///
/// 用于批量收集时按分数排序
#[derive(Debug, Clone)]
pub struct GameSample {
    /// 最终分数
    pub final_score: i32,
    /// 所有回合的训练样本
    pub samples: Vec<TrainingSample>,
}

impl GameSample {
    /// 从 SampleCollector 创建
    pub fn from_collector(collector: SampleCollector) -> Self {
        let final_score = collector.final_score();
        let samples = collector.finalize();
        Self { final_score, samples }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_collector_basic() {
        let mut collector = SampleCollector::new();

        // 记录 3 个回合
        for i in 0..3 {
            let features = vec![i as f32; NN_INPUT_DIM];
            collector.record_turn(features, i % 5, 10);
        }

        assert_eq!(collector.num_turns(), 3);
        assert!(!collector.is_finished());

        // 设置最终分数
        collector.set_final_score(15000);
        assert!(collector.is_finished());
        assert_eq!(collector.final_score(), 15000);
    }

    #[test]
    fn test_sample_collector_with_choices() {
        let mut collector = SampleCollector::new();

        // 第一回合：只有动作
        collector.record_turn(vec![0.0; NN_INPUT_DIM], 2, 10);

        // 第二回合：有事件选项
        collector.record_turn(vec![1.0; NN_INPUT_DIM], 3, 10);
        collector.record_choice(1, 3);

        collector.set_final_score(12000);

        let samples = collector.finalize();
        assert_eq!(samples.len(), 2);

        // 检查第一个样本（无事件选项）
        assert_eq!(samples[0].policy_target[2], 1.0);
        assert_eq!(samples[0].choice_target.iter().sum::<f32>(), 0.0);

        // 检查第二个样本（有事件选项）
        assert_eq!(samples[1].policy_target[3], 1.0);
        assert_eq!(samples[1].choice_target[1], 1.0);
    }

    #[test]
    fn test_game_sample() {
        let mut collector = SampleCollector::new();
        collector.record_turn(vec![0.0; NN_INPUT_DIM], 0, 5);
        collector.record_turn(vec![0.0; NN_INPUT_DIM], 1, 5);
        collector.set_final_score(18000);

        let game_sample = GameSample::from_collector(collector);
        assert_eq!(game_sample.final_score, 18000);
        assert_eq!(game_sample.samples.len(), 2);
    }
}

