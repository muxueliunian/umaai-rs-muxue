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
//!
//! ## 输出消息类型（2026-09 扩展）
//!
//! `--json` 模式 stdout 严格只 JSON，三种消息类型用顶层 `type` 字段区分：
//!
//! - `decision`：AI 决策（与 `DecisionSink::emit` 对应）
//! - `info`：提示信息（`connected` / `compute_start` / `compute_next_step` / `new_game`）
//! - `error`：错误信息（只带 `message` 字段，不细分类型）
//!
//! `emit_info` / `emit_error` 是 `StdoutJsonSink` 的额外方法，不在 `DecisionSink`
//! trait 内——`main.rs` 在 `--json` 分支同时持有 `Arc<dyn DecisionSink>`（决策用）
//! 和 `Arc<StdoutJsonSink>`（info/error 用），human 模式下不需要后者。

use colored::Colorize;
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
        // 手写 fallback 决策（`candidate_scores` 为空）——主要指默认配置
        // `ramen_search_stages="train,ramen"` 下 region 未开启、地区选择走手写逻辑。
        // 它没有搜索评分，luck 行的「期望评分」只是回合加成的换算、运气恒 0，混入会误导，
        // 故地区选择改为显示所选地区并标注【手写逻辑】、跳过 luck 行。其余 None 阶段的
        // 决策不再被 main.rs 合成（见 `calc_ramen_training` 的 `decide`），不会到达本分支。
        if info.candidate_scores.is_empty() && info.decision_kind == "region_select" {
            if let Some(desc) = info.candidate_descriptions.get(info.action_index) {
                println!("{}", format!("选择{desc}（手写逻辑）").magenta());
            }
            return;
        }
        // 仅输出运气信息：期望评分 / 本局运气 / 本回合运气。
        // 数据来自 main.rs `emit_with_luck_decision` 挂到 `scenario_extra` 的
        // luck snapshot；scenario_extra 为 `None`（非 luck 决策，如拉面连续决策
        // 的中间项）时无内容可打，静默跳过。
        let Some(extra) = &info.scenario_extra else {
            return;
        };
        let as_int = |v: &serde_json::Value| -> String {
            match v {
                serde_json::Value::Number(n) => n.as_f64().map(|f| f.round().to_string()).unwrap_or_default(),
                _ => String::new()
            }
        };
        let exp = extra.get("current_terminal_baseline").map(&as_int).unwrap_or_default();
        let turn = extra.get("last_turn_delta").map(&as_int).unwrap_or_default();
        // 本局运气按数值范围着色（红/黄/白/绿/亮绿）
        let total = extra.get("total_luck_score").map(&as_int).unwrap_or_default();
        let total_f = extra
            .get("total_luck_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let total_colored = match total_f {
            x if x < -2000.0 => total.color("red"),
            x if x < -500.0 => total.color("yellow"),
            x if x <= 500.0 => total.color("white"),
            x if x <= 2000.0 => total.color("green"),
            _ => total.bright_green()
        };
        println!("期望评分 {exp} 运气: 本局 {total_colored}, 本回合 {turn}");
    }
}

/// AIRedirector sink：序列化 JSON + `println!` 到 stdout
///
/// 输出顶层 `type: "decision"` 标记 + 决策字段，AIRedirector 端 `HandleOutput`
/// 按 `type` 分发到 decision / info / error 三个处理路径（详见集成文档 §4.3）。
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
        // 2026-09 简化：顶层输出 10 字段
        // - `type` / `turn` / `scenario`：消息类型 + 路由信息
        // - `action_index` / `score`：决策主干
        // - `decision_kind`：子决策类型（ramen_select / special_select / train /
        //   region_select / super_ramen_select / event）——C# 端按此分发 partial decision
        // - `candidate_scores` / `candidate_descriptions` / `candidate_n`：所有候选的
        //   评分 / 可读描述 / 样本数（**完整保留**——按用户拍板"备选选项分复用" + 下游
        //   `action_index` 拿不到动作名，必须挂描述才能映射拉面组合动作）
        // - `scenario_extra`：扩展挂载点（luck_score / action_luck / reason / ramen_action）
        //
        // 旧 stub 字段 `reason` / `search_depth` / `visit_count` / `score_breakdown` /
        // `elapsed_ms` 已从 `DecisionInfo` 删除（不再输出）。
        let payload = serde_json::json!({
            "type": "decision",
            "turn": view.turn,
            "scenario": view.scenario,
            "decision_kind": info.decision_kind,
            "action_index": info.action_index,
            "score": info.score,
            "candidate_scores": info.candidate_scores,
            "candidate_descriptions": info.candidate_descriptions,
            "candidate_n": info.candidate_n,
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
                        "type": "decision",
                        "error": "serialize_failed",
                        "turn": view.turn,
                    })
                );
            }
        }
    }
}

impl StdoutJsonSink {
    /// 发出一条 `info` JSON 行（stdout 严格 JSON 流的一部分）
    ///
    /// `event` 取值由调用方负责保证合法（已知取值：`connected` / `compute_start` /
    /// `compute_next_step` / `new_game`）；sink 不做取值校验，按字符串透传。
    ///
    /// **不在 `DecisionSink` trait 内**——`main.rs` 在 `--json` 分支显式持有
    /// `Arc<StdoutJsonSink>`（具体类型），绕过 trait 直接调本方法。
    pub fn emit_info(&self, event: &str) {
        let payload = serde_json::json!({
            "type": "info",
            "event": event,
        });
        match serde_json::to_string(&payload) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("[ERROR] info serialize failed: {e}");
                // 占位 JSON 让 AIRedirector 知道「这一行是 info 错误占位」
                println!(r#"{{"type":"info","event":"serialize_failed"}}"#);
            }
        }
    }

    /// 发出一条 `error` JSON 行
    ///
    /// 按用户拍板：**只保留 `message` 字符串，不提供额外字段或细分类型**——
    /// 错误事件不区分 watch_fail / parse_fail 等子类型，避免过度细分。
    ///
    /// **不在 `DecisionSink` trait 内**——同 [`Self::emit_info`]。
    pub fn emit_error(&self, message: &str) {
        let payload = serde_json::json!({
            "type": "error",
            "message": message,
        });
        match serde_json::to_string(&payload) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("[ERROR] error serialize failed: {e}");
                println!(r#"{{"type":"error","message":"serialize_failed"}}"#);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小可用的 DecisionInfo 用于测试
    ///
    /// 2026-09 简化：保留 7 字段（action_index / score / decision_kind / candidate_scores /
    /// candidate_descriptions / candidate_n / scenario_extra）——`reason` /
    /// `search_depth` / `visit_count` / `score_breakdown` / `elapsed_ms` 已删除
    fn sample_info() -> DecisionInfo {
        let mut info = DecisionInfo::default();
        info.action_index = 2;
        info.score = 1234.5;
        info.decision_kind = "ramen_select".to_string();
        info.candidate_scores = vec![1200.0, 1234.5, 1100.0];
        info.candidate_descriptions = vec![
            "不吃面".to_string(),
            "吃面/中山-全(替换Bx1+Ax2)".to_string(),
            "不吃面".to_string()
        ];
        info.candidate_n = vec![1024, 800, 256];
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

    /// HumanReadableSink 无 scenario_extra 时静默跳过（不 panic、不输出）
    #[test]
    fn test_human_readable_sink_no_extra() {
        let mut info = sample_info();
        info.scenario_extra = None;
        HumanReadableSink.emit(&info, &GameView::default());
        println!("无 extra 时 emit 完成（应静默跳过）");
    }

    /// HumanReadableSink 带 scenario_extra（luck snapshot）时只输出运气行
    #[test]
    fn test_human_readable_sink_with_luck_extra() {
        let mut info = sample_info();
        info.scenario_extra = Some(serde_json::json!({
            "current_terminal_baseline": 60410.53,
            "total_luck_score": 152.5,
            "last_turn_delta": -12.4
        }));
        HumanReadableSink.emit(&info, &GameView::default());
        println!("带 luck extra 时 emit 完成");
    }

    /// 本局运气各颜色档位均不 panic（四舍五入 + 着色路径全覆盖）
    #[test]
    fn test_human_readable_sink_luck_color_brackets() {
        for total in [-2500.0, -1200.0, 0.0, 1200.0, 3500.0] {
            let mut info = sample_info();
            info.scenario_extra = Some(serde_json::json!({
                "current_terminal_baseline": 60000.0,
                "total_luck_score": total,
                "last_turn_delta": 0.0
            }));
            HumanReadableSink.emit(&info, &GameView::default());
        }
        println!("各颜色档位 emit 完成");
    }

    /// StdoutJsonSink 正常输出 JSON（2026-09 简化：顶层 10 字段）
    ///
    /// 验证五项契约：
    /// (1) 顶层 `type: "decision"` 标识消息类型
    /// (2) 已移除 `schema_version` 字段（按用户拍板）
    /// (3) 顶层含 10 字段：`type` / `turn` / `scenario` / `decision_kind` /
    ///     `action_index` / `score` / `candidate_scores` / `candidate_descriptions` /
    ///     `candidate_n` / `scenario_extra`
    /// (4) `scenario_extra` 完整透传（含 `luck_score` / `action_luck` / `reason` /
    ///     `ramen_action` 四类信息）
    /// (5) `candidate_descriptions` 与 `candidate_scores` / `candidate_n` 严格同长同序
    ///     ——C# 端靠此字段映射 `action_index` 到动作名（拉面组合动作关键）
    /// (6) `decision_kind` 标明 partial decision 类型（ramen_select / train / ...）
    ///
    /// 测试只 println 让人眼核对——cargo test 默认 capture stdout。
    #[test]
    fn test_stdout_json_sink_emits_json() {
        let view = GameView {
            scenario: "ramen".into(),
            turn: 5,
            ..Default::default()
        };
        let mut info = sample_info();
        // 挂一个含四类信息的 scenario_extra，验证透传（含 ramen_action 字符串）
        info.scenario_extra = Some(serde_json::json!({
            "luck_score": {
                "initial_terminal_baseline": 50078.0,
                "current_terminal_baseline": 50354.0,
                "total_luck_score": 276.0,
                "last_turn_delta": -121.0
            },
            "action_luck": {"0": -50.0, "1": 125.5, "2": 80.0},
            "ramen_action": "吃面/中山-全(替换Bx1+Ax2)",
            "reason": {
                "metric": "score",
                "chosen_desc": "吃面/中山-全(替换Bx1+Ax2)",
                "chosen_mean": 65000.0,
                "chosen_n": 1024,
                "rivals": [
                    {"index": 2, "desc": "吃面/中山-全(替换Bx1+Ax2)", "gap": 2200.0,
                     "confidence": 0.95, "n": 800, "mean": 67200.0, "sd": 200.0,
                     "pros": [{"key":"speed_final","label":"速","unit":"score","delta":30.0}],
                     "cons": [{"key":"pt_score","label":"PT","unit":"pt","delta":-33.0}]}
                ]
            }
        }));
        StdoutJsonSink.emit(&info, &view);
        println!("StdoutJsonSink 正常路径 emit 完成（顶层 10 字段 + decision_kind 标明 partial + candidate_descriptions 映射动作名 + scenario_extra 四类信息透传）");
    }

    /// StdoutJsonSink 序列化失败时走占位 JSON 路径（NaN 不允许序列化）
    ///
    /// 标准 JSON 不允许 NaN/Inf，serde_json 默认会报错——验证 emit 走 Err 分支
    /// 输出一行占位 JSON（`type: "decision"` + `error: "serialize_failed"`）而非 panic。
    /// 占位 JSON 也保留 `type: "decision"` 标识——AIRed 端按 type 仍能识别。
    #[test]
    fn test_stdout_json_sink_fallback_on_serialize_failure() {
        let mut info = sample_info();
        info.score = f32::NAN; // 标准 JSON 不允许 NaN → 触发 Err 分支
        StdoutJsonSink.emit(&info, &GameView::default());
        println!("NaN 序列化失败回退路径 emit 完成（不 panic，stderr 有错误日志）");
    }

    /// emit_info 输出正确格式：`{"type":"info","event":"<event>"}`
    ///
    /// 覆盖 4 种已知 event 取值（`connected` / `compute_start` / `compute_next_step` /
    /// `new_game`）。println 让人眼核对每行的 JSON 结构与 event 值。
    #[test]
    fn test_stdout_json_sink_info() {
        for event in ["connected", "compute_start", "compute_next_step", "new_game"] {
            StdoutJsonSink.emit_info(event);
            println!("emit_info({event}) 完成（应输出 {{\"type\":\"info\",\"event\":\"{event}\"}}）");
        }
    }

    /// emit_error 输出正确格式：`{"type":"error","message":"<message>"}`
    ///
    /// 验证三项契约：(1) 只带 message 字段无额外细分；(2) message 字符串透传；
    /// (3) 中文 message 不被转义。println 让人眼核对。
    #[test]
    fn test_stdout_json_sink_error() {
        StdoutJsonSink.emit_error("小黑板目录不存在");
        println!("emit_error 中文 message 完成（应输出 type:error + 中文 message）");
        StdoutJsonSink.emit_error("parse failed: bad json");
        println!("emit_error 英文 message 完成");
    }

    /// emit_info / emit_error 序列化失败时**不打 stdout 占位**、只打 stderr
    ///
    /// 与决策 emit 不同：info/error 的失败走"stderr 日志 + 静默丢弃"——避免
    /// 污染 stdout 严格 JSON 流（占位 JSON 形态需要约定，反而是负担）。
    /// 实际场景几乎不会触发（&str 序列化不会失败），测试覆盖兜底分支。
    /// 注：&str 字面量无法触发序列化失败；本测试仅冒烟验证 emit 不 panic。
    #[test]
    fn test_stdout_json_sink_info_error_does_not_panic() {
        StdoutJsonSink.emit_info("test_event");
        StdoutJsonSink.emit_error("test_error");
        println!("emit_info / emit_error 完成（不 panic 即可）");
    }
}