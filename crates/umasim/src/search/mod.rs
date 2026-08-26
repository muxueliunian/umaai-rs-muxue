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

mod config;
mod flat_search;
mod ramen_terminal;
mod result;
mod searchable;
pub(crate) mod seeds;
pub(crate) mod terminal;

pub use config::SearchConfig;
pub use flat_search::FlatSearch;
pub use ramen_terminal::{FROZEN_DIM_KEYS, RamenTerminal, RamenTerminalStats};
pub use result::{ActionResult, SearchOutput};

/// 拉面搜索输出
///
/// 拉面是首个接入终局多维记录的剧本，故第二个类型参数不再是默认的
/// [`NoTerminalStats`]。别名放在模块层而非 `result.rs`：搜索结果类型本身
/// 不该反向依赖某个具体剧本的维度定义。
pub type RamenSearchOutput = SearchOutput<crate::game::ramen::RamenAction, RamenTerminalStats>;
pub use searchable::{FlatSearchGame, RolloutHost, SearchScore};
pub use seeds::RolloutSeeds;
pub use terminal::{
    MomentResult, NamedMetricRef, NoTerminal, NoTerminalStats, RolloutOutcome, TerminalRecord, TerminalStats
};
