//! 可裁剪的诊断日志宏（编译期 feature gate）
//!
//! ## 行为
//!
//! - `diag` feature **开启**（默认）：宏展开为 [`log::info!`]，target 固定为 `"diagnostic"`，
//!   便于按 spec 单独关闭规则层日志（如 `init_logger(app, "info,diagnostic=off")`）。
//! - `diag` feature **关闭**：宏展开为空，连 `format_args!` 都不执行，**零运行时开销**。
//!
//! ## 业务日志的关系
//!
//! - **业务日志**（决策层 / Trainer）继续用 [`log::info!`] / [`log::warn!`] / [`log::error!`]，
//!   这些**永不被裁剪**，必须始终输出。
//! - **规则层日志**（Game / Action 层）在阶段 2 起迁移到此宏，可在 `umaai` 等 AI 助手
//!   binary 中通过 `default-features = false` 完全消除。
//!
//! ## 用法
//!
//! ```ignore
//! use umasim::diag;
//! diag!("回合 {} 触发事件 {}", game.turn, event.id);
//! diag!("候选数 {}", actions.len());
//! ```
//!
//! 设计依据：见 `.trae/documents/log_refactor_plan.md` §3.1、§4.2、§7.1。

/// 可裁剪的诊断日志宏
///
/// - `feature = "diag"` 开启：编译为 `log::info!(target: "diagnostic", ...)`
/// - `feature = "diag"` 关闭：宏体被 `#[cfg]` 整个剔除，**不**调用 `format_args!`，不产任何代码
#[macro_export]
macro_rules! diag {
    ($($arg:tt)*) => {
        #[cfg(feature = "diag")]
        ::log::info!(target: "diagnostic", $($arg)*);
    };
}

#[cfg(test)]
mod tests {
    /// 在 feature 开启时，`diag!` 必须展开为真实的 `log::info!` 调用
    ///
    /// 测试本身无法验证"feature 关闭时宏被消除"——这需要跨 crate 编译对比，
    /// 由阶段 6 通过 `cargo bloat` 验证。此处只验证 feature 开启下宏可用。
    #[cfg(feature = "diag")]
    #[test]
    fn test_diag_expands_to_info() {
        // 调用不应 panic（log facade 的 no-op logger 允许无 logger handle）
        crate::diag!("diagnostic 测试: {}", 42);
    }

    /// 在 feature 关闭时，本模块仍需能编译（宏的 cfg 内部被消除，调用方代码也走 cfg 关路径）
    #[cfg(not(feature = "diag"))]
    #[test]
    fn test_diag_is_noop_when_feature_off() {
        // 即便没装任何 log handler，宏也不应展开出任何逻辑
        crate::diag!("feature off, {}", "should be no-op");
    }
}
