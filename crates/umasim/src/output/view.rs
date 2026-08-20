//! 用户/AI 视角的游戏状态结构化展示
//!
//! ## 与 `explain()` 的边界
//!
//! - [`explain()`](crate::explain) 是**开发者诊断快照**，用于排查 `Array5` 等多义性结构
//!   ——面向内部，逐字段铺平，含调试用裸数据
//! - 本 `GameView` 是**面向用户/AI 的结构化展示**，纯函数形式，字段定义留到阶段 4 完善
//!
//! 两者**并存**：本期仅占位，详细字段定义推迟到阶段 4（见
//! `log_refactor_plan.md` §7.4）。

use serde::{Deserialize, Serialize};

/// 用户/AI 视角的游戏状态展示
///
/// 详细字段定义留到阶段 4 完善。占位结构仅含通用字段的最小集合。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GameView {
    /// 剧本名（如 `"ramen"` / `"onsen"` / `"base"`），便于下游分类
    pub scenario: String,

    /// 当前回合数（人类视角，从 1 开始）
    pub turn: u32,

    /// 当前体力
    pub vital: i32,

    /// 当前干劲
    pub motivation: i32,
}

impl GameView {
    /// 构造一个最小可用的 `GameView`
    pub fn new(scenario: impl Into<String>) -> Self {
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
        assert_eq!(view.vital, 0);
        assert_eq!(view.motivation, 0);
    }

    #[test]
    fn test_new_sets_scenario() {
        let view = GameView::new("ramen");
        assert_eq!(view.scenario, "ramen");
    }

    #[test]
    fn test_serde_roundtrip() {
        let view = GameView {
            scenario: "onsen".into(),
            turn: 12,
            vital: 85,
            motivation: 4,
        };
        let json = serde_json::to_string(&view).expect("serialize");
        let back: GameView = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(view, back);
    }
}
