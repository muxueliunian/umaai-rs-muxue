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

//! ## 运行时开关（MCTS rollout 静默）
//!
//! 除编译期裁剪外，本模块还提供**进程级运行时开关**（[`enabled`] /
//! [`set_enabled`] / [`DiagGuard`]），用于在 MCTS 的 rayon rollout 流程中
//! 临时屏蔽全部诊断输出，同时保留 rollout 之外（真实局、决策层）的诊断。
//!
//! - 开关机制**不受** `diag` feature 门控：no-diag build 下同样编译
//!   （`FlatSearch::search_with_terminal` 的挂点无需 cfg 分支，此时开关无效果也无开销）。
//! - [`diag!`] 宏展开为 `if enabled() { log::info!(...) }`；cfg explain 块
//!   同样以 `if enabled()` 包裹，rollout 期间连 `format_args!` 与 comfy-table
//!   构造都一并跳过。
//! - 开关为全局 `AtomicBool`（默认开启）：rollout 跑在 rayon worker 线程上，
//!   thread_local 不可行，必须跨线程可见。内存序用 `Relaxed`——这是"尽力而为"
//!   的抑制语义，最坏情况只是边界处多/少输出几条日志，不影响正确性。
//! - **已知局限**：umasim 多局并行（`simulation_count > 1`）时，任一局的搜索
//!   都会全局关闭开关，顺带抑制其他局真实回合的 diag 输出。diag 的价值场景
//!   是单局观察，多局批量跑输出本就交错，此局限被接受并在此明示。
//!
//! 设计依据：见 `.trae/documents/archive/log_refactor_plan.md` §4.2、§7.5（阶段 5
//! 设想的 `output.diagnostic.set_enabled(false)` 即本机制）。

use std::sync::atomic::{AtomicBool, Ordering};

/// 诊断输出运行时开关（进程级，默认开启）
static DIAG_ENABLED: AtomicBool = AtomicBool::new(true);

/// 查询诊断输出当前是否开启
///
/// [`diag!`] 宏与 cfg explain 块在每次输出前调用本方法；
/// `diag` feature 关闭时宏体被整个剔除，本方法不会被宏路径调用。
#[inline]
#[must_use]
pub fn enabled() -> bool {
    DIAG_ENABLED.load(Ordering::Relaxed)
}

/// 设置诊断输出开关（进程级，跨线程可见）
///
/// 直接调用请优先考虑 [`DiagGuard`]：guard 以 RAII 方式恢复进入前的开关值，
/// 天然免疫 rollout 提前返回（`Err`/`?`）导致的开关泄漏。
pub fn set_enabled(on: bool) {
    DIAG_ENABLED.store(on, Ordering::Relaxed);
}

/// 诊断输出抑制 guard（RAII）
///
/// 创建时关闭诊断输出，[`Drop`](std::ops::Drop) 时恢复**进入前的值**（栈式
/// 嵌套安全：内层 guard 提前 drop 不会误开外层的抑制状态）。
///
/// 用法（见 `FlatSearch::search_with_terminal`）：
///
/// ```ignore
/// let _diag_guard = DiagGuard::suppress();
/// // rayon rollout 期间 diag! 与 explain 块全部静默
/// // guard drop 时自动恢复，Err 提前返回同样安全
/// ```
#[derive(Debug)]
pub struct DiagGuard {
    /// 进入 guard 前的开关值（Drop 时恢复）
    prev: bool
}

impl DiagGuard {
    /// 关闭诊断输出并返回恢复 guard
    #[must_use = "guard 被 drop 前诊断输出一直关闭，请绑定到变量以覆盖整个作用域"]
    pub fn suppress() -> Self {
        let prev = enabled();
        set_enabled(false);
        Self { prev }
    }
}

impl std::ops::Drop for DiagGuard {
    fn drop(&mut self) {
        set_enabled(self.prev);
    }
}

/// 可裁剪的诊断日志宏
///
/// - `feature = "diag"` 开启：`if enabled() { log::info!(target: "diagnostic", ...) }`
///   ——运行时开关关闭（如 MCTS rollout 期间）时连 `format_args!` 都不执行
/// - `feature = "diag"` 关闭：宏体被 `#[cfg]` 整个剔除，**不**调用 `format_args!`，不产任何代码
#[macro_export]
macro_rules! diag {
    ($($arg:tt)*) => {
        #[cfg(feature = "diag")]
        if $crate::output::diagnostic::enabled() {
            ::log::info!(target: "diagnostic", $($arg)*);
        }
    };
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{DiagGuard, enabled, set_enabled};

    /// 全局开关测试互斥锁：开关是进程级状态，相关测试必须串行执行
    static SWITCH_LOCK: Mutex<()> = Mutex::new(());

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

    /// guard 抑制后 enabled() 必须为 false，drop 后必须恢复进入前的值
    #[test]
    fn test_guard_suppress_and_restore() {
        // unwrap：测试代码，锁中毒只可能来自其他测试 panic，无恢复意义
        let _serial = SWITCH_LOCK.lock().unwrap();
        // 入口断言开关处于默认开启态（前序测试若污染状态在此立即暴露）
        assert!(enabled(), "前置条件：诊断开关默认开启");
        {
            let _guard = DiagGuard::suppress();
            println!("guard 作用域内 enabled = {}", enabled());
            assert!(!enabled(), "guard 作用域内诊断输出必须被抑制");
        }
        println!("guard drop 后 enabled = {}", enabled());
        assert!(enabled(), "guard drop 后必须恢复开启");
    }

    /// guard 恢复的是**进入前的值**而非硬编码 true（栈式嵌套安全）
    ///
    /// 场景：外层已手动关闭时，内层 guard drop 不应把开关误开。
    #[test]
    fn test_guard_restores_prev_value() {
        // unwrap：测试代码，锁中毒只可能来自其他测试 panic，无恢复意义
        let _serial = SWITCH_LOCK.lock().unwrap();
        set_enabled(false);
        {
            let _guard = DiagGuard::suppress();
            assert!(!enabled(), "内层 guard 作用域内仍为关闭");
        }
        println!("外层关闭时 guard drop 后 enabled = {}", enabled());
        assert!(!enabled(), "guard 应恢复进入前的 false，而非硬编码 true");
        set_enabled(true);
        assert!(enabled(), "清理：恢复默认开启");
    }

    /// 嵌套 guard：内层 drop 只恢复到内层进入前的状态，外层抑制继续保持
    #[test]
    fn test_nested_guards() {
        // unwrap：测试代码，锁中毒只可能来自其他测试 panic，无恢复意义
        let _serial = SWITCH_LOCK.lock().unwrap();
        {
            let _outer = DiagGuard::suppress();
            {
                let _inner = DiagGuard::suppress();
                assert!(!enabled());
            }
            println!("内层 drop 后（外层仍持有） enabled = {}", enabled());
            assert!(!enabled(), "内层 drop 后外层的抑制必须继续保持");
        }
        assert!(enabled(), "全部 guard drop 后恢复开启");
    }

    /// 直接调用 diag! 在开关关闭时必须静默（不 panic 即可——log facade 无 handler 时本就丢弃）
    #[cfg(feature = "diag")]
    #[test]
    fn test_diag_silent_when_disabled() {
        // unwrap：测试代码，锁中毒只可能来自其他测试 panic，无恢复意义
        let _serial = SWITCH_LOCK.lock().unwrap();
        let _guard = DiagGuard::suppress();
        // 开关关闭：宏展开的 if enabled() 分支不执行，调用应无任何副作用
        crate::diag!("被抑制的日志 {}", 42);
        assert!(!enabled());
    }
}
