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

mod config;
mod flat_search;
mod result;
mod searchable;
mod seeds;

pub use config::SearchConfig;
pub use flat_search::FlatSearch;
pub use result::{ActionResult, SearchOutput};
pub use searchable::{FlatSearchGame, RolloutHost, SearchScore};
pub use seeds::{InternalSeed, RolloutSeeds};
