//! 神经网络评估器模块
//!
//! - [`Evaluator`]：评估器 trait
//! - [`HandwrittenEvaluator`]：手写启发式评估器（用于数据收集）
//! - [`NeuralNetEvaluator`]：神经网络评估器（**仅 `onnx` feature 下可用**）
//! - [`RandomEvaluator`]：随机评估器（基准测试）
//! - [`ValueOutput`]：评估器输出值
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use umasim::neural::{HandwrittenEvaluator, ValueOutput, Evaluator};
//! let evaluator = HandwrittenEvaluator::new();
//! let action = evaluator.select_action(&game, &mut rng);
//! ```

mod evaluator;
mod handwritten_evaluator;
#[cfg(feature = "onnx")]
mod neural_net_evaluator;
mod value_output;

// 公开导出
pub use evaluator::{Evaluator, RandomEvaluator};
pub use handwritten_evaluator::HandwrittenEvaluator;
#[cfg(feature = "onnx")]
pub use neural_net_evaluator::{
    NeuralNetEvaluator,
    ThreadLocalNeuralNetLeafEvaluator,
    ThreadLocalNeuralNetLeafStatsSnapshot
};
pub use value_output::ValueOutput;
