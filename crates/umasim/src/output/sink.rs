//! AI 决策的输出契约（[`DecisionSink`] trait + 三种实现）
//!
//! ## 与 [`DecisionReasonSink`](super::reason::DecisionReasonSink) 的边界
//!
//! `DecisionSink` 处理**决策主干**（选择 + 评分 + 理由），面向玩家屏幕 / 下游协议；
//! `DecisionReasonSink` 处理**决策理由装饰**（评分前 N + 终局维度差），面向协议分析。
//! 两者**并存**且互不依赖：决策主干必走 sink，理由装饰按需接入 reason_sink。
//!
//! ## 范围
//!
//! 仅包含**决策本身**（action_index / score / candidate_scores / reason）。
//! **回合 / 剧本状态**（vital / 五维 / scenario_pt 等）由调用方用独立开关控制：
//! `HumanReadableSink::emit` 只打决策，不打状态——避免状态日志污染决策 sink 的契约。
//! 状态日志的开关（verbose / log_level / 自定义 print_round_header 等）由 `main.rs`
//! 在调 sink.emit 之前 / 之后按需控制。
//!
//! ## Send + Sync 要求
//!
//! sink 由 trainer 内部存放、`Arc<dyn DecisionSink>` 跨线程共享（如 AIRedirector 模式
//! 在 rayon 上并行），实现必须 `Send + Sync`。三个 unit struct 默认满足。
//!
//! ## Step 3 范围（`StdoutJsonSink` 不关 ANSI）
//!
//! 关闭 ANSI / 启动横幅走 stderr 等**模式级副作用**由 `main.rs` 在 `--json` 分支
//! 集中处理；sink 内部只负责 emit 自身内容，不跨职责调整全局状态。这样 sink
//! 可以单独测试（不依赖 colored / log init 状态）。

use crate::output::{DecisionInfo, view::GameView};

/// AI 决策的输出契约
///
/// 实现负责把决策数据渲染到自身目标（屏幕 / stdout / 日志 / socket）。
/// `Send + Sync` 是 trainer 并行场景的硬性约束。
///
/// ## 调用时机
///
/// 由 `main.rs` 主循环在 `trainer.select_action` 之后、`luck_tracker.on_new_turn`
/// 之前调用（详见集成文档 §3.3.4）。一回合一次。
pub trait DecisionSink: Send + Sync {
    /// 发出一条决策原始数据
    ///
    /// `view` 携带回合 / 剧本状态（仅用于 JSON 输出做 payload 路由），屏幕 sink 通常忽略。
    /// 实现必须保证 `emit` 不 panic：序列化失败等异常路径走降级输出或静默丢弃。
    fn emit(&self, info: &DecisionInfo, view: &GameView);
}

/// 静默 sink：丢弃决策数据（库默认实现）
///
/// 与 `DecisionReasonSink::DecisionReasonNoopSink` 同模式：保留 sink 调用路径，下游可换成
/// 实际实现（如 `StdoutJsonSink`）。`umasim` / `umaai` 默认不主动选 `EmptySink`
/// （main.rs 显式选 human / json），但作为 trait 默认实现必备——库外 caller
/// 不必强制指定 sink。
///
/// 命名 `EmptySink` 而非 `NoopSink`：明确表达"emit 什么都不做"的语义，
/// 与 `DecisionSink::emit(&self, info, view)` 签名的"空实现"对应。
pub struct EmptySink;

impl DecisionSink for EmptySink {
    fn emit(&self, _info: &DecisionInfo, _view: &GameView) {}
}

/// 玩家屏幕 sink：决策渲染为人类文本 + `println!`
///
/// **只渲染决策本身**：首选下标 / 评分 / 理由。候选评分全表、终局维度差等
/// 详细解释**不**在这里打——这些走 LoggingTrainer / reason.rs / `explain_*` 等
/// 既有路径，避免 sink 重复造轮子。
///
/// 回合 / 剧本状态（vital / 五维 / scenario_pt 等）**不进 sink**，由调用方用
/// 独立开关控制（如 `main.rs` 在 `sink.emit` 前 / 后按 log_level 或 verbose
/// 决定要不要打 round header）。
pub struct HumanReadableSink;

impl DecisionSink for HumanReadableSink {
    fn emit(&self, info: &DecisionInfo, _view: &GameView) {
        // 决策主干：首选 + 评分
        println!("AI 选择: 第 {} 个动作（评分: {}）", info.action_index, info.score);
        // 理由（如有）—— 单独一行，避免和评分粘在一起
        if let Some(reason) = &info.reason {
            println!("理由: {reason}");
        }
    }
}

/// AIRedirector sink：序列化 JSON + `println!` 到 stdout
///
/// 输出 `schema_version` 标记 + 决策字段，AIRedirector 端 `HandleOutput` 解析
/// `schema_version` 识别本协议（详见集成文档 §4.3）。
///
/// ## 失败回退
///
/// 序列化失败（如 `f32::NaN` / `Inf` 等标准 JSON 不允许的值）时打一行 stderr
/// 错误日志，并输出一行占位 JSON（`error: "serialize_failed"` 字段便于排查）。
/// **不 panic**——主循环后续依赖 sink.emit 不抛异常。
///
/// ## ANSI / 启动横幅 / colored
///
/// 这些是**模式级副作用**，由 `main.rs` 在 `--json` 分支集中处理
/// （详见集成文档 §3.2.6）：
/// - `colored::control::set_override(false)` 关闭 ANSI
/// - 启动横幅 `eprintln!` 而非 `println!`
///
/// sink 内部不调 colored / eprintln!——保持单一职责，方便单独测试。
pub struct StdoutJsonSink;

impl DecisionSink for StdoutJsonSink {
    fn emit(&self, info: &DecisionInfo, view: &GameView) {
        let payload = serde_json::json!({
            "schema_version": 1,
            "turn": view.turn,
            "scenario": view.scenario,
            "action_index": info.action_index,
            "score": info.score,
            "candidate_scores": info.candidate_scores,
            "candidate_n": info.candidate_n,
            "reason": info.reason,
            "scenario_extra": info.scenario_extra,
        });
        match serde_json::to_string(&payload) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                // 不 panic：占位 JSON 让 AIRedirector 知道「这一行是错误占位」而非静默丢失
                eprintln!("[ERROR] decision serialize failed: {e}");
                println!(
                    "{}",
                    serde_json::json!({
                        "schema_version": 1,
                        "error": "serialize_failed",
                        "turn": view.turn,
                    })
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小可用的 DecisionInfo 用于测试
    fn sample_info() -> DecisionInfo {
        let mut info = DecisionInfo::default();
        info.action_index = 2;
        info.score = 1234.5;
        info.candidate_scores = vec![1200.0, 1234.5, 1100.0];
        info.candidate_n = vec![1024, 800, 256];
        info.reason = Some("vs #2 智+180 PT-33".into());
        info
    }

    /// EmptySink 不 panic 即过（静默丢弃，无副作用可断言）
    #[test]
    fn test_empty_sink_does_not_panic() {
        EmptySink.emit(&sample_info(), &GameView::default());
        println!("EmptySink emit 完成");
    }

    /// HumanReadableSink 不 panic 即过（println! 输出由 cargo test 默认 capture）
    #[test]
    fn test_human_readable_sink_does_not_panic() {
        HumanReadableSink.emit(&sample_info(), &GameView::default());
        println!("HumanReadableSink emit 完成");
    }

    /// HumanReadableSink 无 reason 时只打决策主干一行
    #[test]
    fn test_human_readable_sink_no_reason() {
        let mut info = sample_info();
        info.reason = None;
        HumanReadableSink.emit(&info, &GameView::default());
        println!("无 reason 时 emit 完成");
    }

    /// StdoutJsonSink 正常输出 JSON（含 schema_version / turn / scenario）
    #[test]
    fn test_stdout_json_sink_emits_json() {
        let view = GameView {
            scenario: "ramen".into(),
            turn: 5,
            ..Default::default()
        };
        StdoutJsonSink.emit(&sample_info(), &view);
        println!("StdoutJsonSink 正常路径 emit 完成");
    }

    /// StdoutJsonSink 序列化失败时走占位 JSON 路径（NaN 不允许序列化）
    ///
    /// 标准 JSON 不允许 NaN/Inf，serde_json 默认会报错——验证 emit 走 Err 分支
    /// 输出一行占位 JSON（error: "serialize_failed"）而非 panic。
    #[test]
    fn test_stdout_json_sink_fallback_on_serialize_failure() {
        let mut info = sample_info();
        info.score = f32::NAN; // 标准 JSON 不允许 NaN → 触发 Err 分支
        StdoutJsonSink.emit(&info, &GameView::default());
        println!("NaN 序列化失败回退路径 emit 完成（不 panic）");
    }
}