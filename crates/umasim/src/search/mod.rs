//! 蒙特卡洛搜索模块
//!
//! 提供基于蒙特卡洛采样的决策支持功能。
//!
//! # 模块结构
//!
//! - [`SearchParam`]: 搜索参数配置
//! - [`ValueOutput`]: 评估器输出值
//! - [`SearchResult`]: 搜索结果聚合
//! - [`Evaluator`]: 评估器 trait
//! - [`HandwrittenEvaluator`]: 手写启发式评估器
//! - [`RandomEvaluator`]: 随机评估器（基准测试）
//! - [`MonteCarloSearch`]: 蒙特卡洛搜索核心
//! - [`EvaluatorTrainerAdapter`]: 评估器到训练员的适配器
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use umasim::search::{SearchParam, ValueOutput, SearchResult, HandwrittenEvaluator, MonteCarloSearch};
//!
//! let param = SearchParam::default();
//! let evaluator = HandwrittenEvaluator::new();
//! let value = ValueOutput::new(1000.0, 200.0);
//! let mut result = SearchResult::new_legal();
//! result.add_result(value);
//!
//! // MCTS 搜索
//! let mut search = MonteCarloSearch::new(param);
//! // let best_idx = search.run_search(&game, &actions, &evaluator, &mut rng)?;
//! ```

mod search_param;
mod value_output;
mod search_result;
mod evaluator;
mod handwritten_evaluator;
mod monte_carlo;

// 公开导出
pub use search_param::SearchParam;
pub use value_output::ValueOutput;
pub use search_result::SearchResult;
pub use evaluator::{Evaluator, RandomEvaluator};
pub use handwritten_evaluator::HandwrittenEvaluator;
pub use monte_carlo::{MonteCarloSearch, EvaluatorTrainerAdapter};

