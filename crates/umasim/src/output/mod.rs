//! 结构化输出与决策信息模块（Phase 3 / 阶段 1 骨架）
//!
//! 提供三种子系统：
//!
//! - [`decision`] — `DecisionInfo`：AI 决策输出标准格式，多下游（Android/MCP/WebSocket）共享
//! - [`diagnostic`] — [`diag!`] 宏：编译期可裁剪的诊断日志，`#[cfg(feature = "diag")]` 关闭时为 no-op
//! - [`view`] — `GameView`：面向用户/AI 的游戏状态结构化展示，字段定义留到阶段 4 完善
//!
//! 与现有 `log::info!` 业务日志的关系：
//!
//! - 业务日志（`Trainer` / 决策层）继续用 `log::info!/warn!/error!`，始终输出，永不裁剪
//! - 规则层日志（`Game` / `Action` 层）在阶段 2 迁移到 [`diag!`]，可通过 `diag` feature 编译期裁剪

pub mod decision;
pub mod decision_log;
pub mod diagnostic;
pub mod reason;
pub mod turn_flow;
pub mod view;

pub use decision::DecisionInfo;
pub use decision_log::{DecisionLog, DecisionLogRow};
pub use reason::{DecisionReasonData, DecisionReasonSink, LogJsonSink, NoopSink};
pub use turn_flow::{RecordingTrainer, TurnDecision};
pub use view::GameView;
// 注意：`#[macro_export]` 标记的 `diag!` 宏已经在 crate 根全局可见（`umasim::diag!` 可直接调用），
// 不能再用 `pub use diagnostic::diag;` 重新导出——宏不是普通 item，不支持 re-export。
