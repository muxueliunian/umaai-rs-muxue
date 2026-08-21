//! 用户/AI 视角的游戏状态结构化展示
//!
//! ## 与 `explain()` 的边界
//!
//! - [`explain()`](crate::explain) 是**开发者诊断快照**，用于排查 `Array5` 等多义性结构
//!   ——面向内部，逐字段铺平，含调试用裸数据
//! - 本 `GameView::view()` 是**面向用户/AI 的结构化展示**，纯函数形式，所有字段
//!   `Serialize`/`Deserialize` 派生后可直接 JSON 输出给下游（Android/MCP/WebSocket）
//!
//! 两者**并存**：诊断用 `explain()`，下游用 `view()`。
//! 设计依据：见 `.trae/documents/log_refactor_plan.md` §7.4。

use serde::{Deserialize, Serialize};

/// 用户/AI 视角的游戏状态展示
///
/// 字段均来自 `Game` trait 的公共接口（`turn()`/`max_turn()`/`uma()`），可在默认
/// `Game::view()` 中填充。具体剧本可 override `view()` 以填充 `scenario` 等剧本特有字段。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GameView {
    /// 剧本标识（`"base"` / `"onsen"` / `"ramen"` 等）
    ///
    /// 默认实现留空，剧本 Game 在 `Game::view()` override 中填充
    /// （如 `"ramen"`）。下游用此字段做 JSON payload 的路由分发。
    pub scenario: String,

    /// 当前回合数（人类视角，1-based；`Game::turn()` 是 0-based，view() 加 1）
    pub turn: u32,

    /// 总回合数
    pub max_turn: u32,

    /// 当前体力
    pub vital: i32,

    /// 体力上限
    pub max_vital: i32,

    /// 干劲值（与 [`crate::game::Uma::motivation`] 同口径）
    pub motivation: i32,

    /// 已累计技能点（PT）
    pub skill_pt: i32,

    /// 已累计 Hint 数
    pub total_hints: i32
}

impl GameView {
    /// 构造一个最小可用的 `GameView`，仅指定剧本名
    pub fn with_scenario(scenario: impl Into<String>) -> Self {
        Self {
            scenario: scenario.into(),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_view() {
        let view = GameView::default();
        assert_eq!(view.scenario, "");
        assert_eq!(view.turn, 0);
        assert_eq!(view.max_turn, 0);
        assert_eq!(view.vital, 0);
        assert_eq!(view.max_vital, 0);
        assert_eq!(view.motivation, 0);
        assert_eq!(view.skill_pt, 0);
        assert_eq!(view.total_hints, 0);
    }

    #[test]
    fn test_with_scenario_only_sets_scenario() {
        let view = GameView::with_scenario("ramen");
        assert_eq!(view.scenario, "ramen");
        assert_eq!(view.turn, 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let view = GameView {
            scenario: "onsen".into(),
            turn: 12,
            max_turn: 78,
            vital: 85,
            max_vital: 120,
            motivation: 4,
            skill_pt: 3500,
            total_hints: 18
        };
        let json = serde_json::to_string(&view).expect("serialize");
        let back: GameView = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(view, back);
    }

    /// `GameView` 必须能作为 `serde_json::Value` 直接挂载进 `DecisionInfo::scenario_extra`
    #[test]
    fn test_view_as_serde_json_value() {
        let view = GameView {
            scenario: "ramen".into(),
            turn: 5,
            max_turn: 78,
            vital: 70,
            max_vital: 100,
            motivation: 3,
            skill_pt: 1200,
            total_hints: 7
        };
        let v = serde_json::to_value(&view).expect("to_value");
        assert_eq!(v["scenario"], "ramen");
        assert_eq!(v["turn"], 5);
        assert_eq!(v["skill_pt"], 1200);
    }
}
