//! 搜索模块
//!
//! 提供扁平蒙特卡洛搜索，用于生成高质量训练数据。
//!
//! # 模块结构
//! - `config`: 搜索配置
//! - `result`: 搜索结果（分数分布统计）
//! - `flat_search`: 扁平蒙特卡洛搜索实现
//! - `seeds`: rollout 种子派生（可复现性与 CRN 的载体）
//! - `searchable`: 剧本适配层（搜索所需、`Game` 未覆盖的能力）
//! - `terminal`: 终局多维记录（不参与排序的观测量）
//! - `ramen_terminal`: 拉面剧本的终局维度定义
//! - `ramen_leaf`: 拉面截断续跑与叶估值的结果类型

mod config;
mod flat_search;
mod ramen_leaf;
mod ramen_terminal;
mod result;
mod searchable;
pub(crate) mod seeds;
pub(crate) mod terminal;

pub use config::SearchConfig;
pub use flat_search::FlatSearch;
pub use ramen_leaf::{
    LeafVerdict, RamenLeafFact, RamenLeafRecord, RamenLeafStage, RamenLeafStats, RamenLeafValuer,
    is_decision_stage, leaf_safe_stage, leaf_verdict
};
pub use ramen_terminal::{FROZEN_DIM_KEYS, RamenTerminal, RamenTerminalStats};
pub use result::{ActionResult, OrderedRollouts, SearchOutput, SelectObjective};

/// 拉面搜索输出
///
/// 拉面是首个接入终局多维记录的剧本，故第二个类型参数不再是默认的
/// [`NoTerminalStats`]。别名放在模块层而非 `result.rs`：搜索结果类型本身
/// 不该反向依赖某个具体剧本的维度定义。
pub type RamenSearchOutput = SearchOutput<crate::game::ramen::RamenAction, RamenTerminalStats>;

/// 拉面**截断**搜索输出
///
/// 与 [`RamenSearchOutput`] 只差第二个类型参数：统计里多了叶那几维，且终局 25 维
/// 只吃真实终局样本。类型不同是刻意的——想把截断结果当成完整终局结果用，会在编译期
/// 就被挡住。
pub type RamenLeafSearchOutput = SearchOutput<crate::game::ramen::RamenAction, RamenLeafStats>;
pub use searchable::{FlatSearchGame, RolloutHost, SearchScore};
pub use seeds::RolloutSeeds;
pub use terminal::{
    MomentResult, NamedMetricRef, NoTerminal, NoTerminalStats, RolloutOutcome, TerminalRecord, TerminalStats
};
