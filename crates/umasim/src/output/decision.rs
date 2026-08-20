//! AI 决策输出标准格式
//!
//! 多个下游（Android/MCP/WebSocket）共享同一结构。Trainer trait
//! 仅输出 `action_index`；附加的决策上下文（候选评分、耗时、搜索深度等）
//! 通过 [`Trainer::last_decision`](crate::game::Trainer::last_decision)
//! 旁路暴露，便于面向用户的 Trainer（MCTS、手写策略）逐步实现。
//!
//! 设计原则：
//!
//! - 不强制任何字段非空；调用方按需填充
//! - `Serialize`/`Deserialize` 双派生，便于 JSON / bincode 互通
//! - 剧本特有扩展字段用 `serde_json::Value`，避免在此结构内堆叠剧本 enum

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// AI 决策输出标准格式
///
/// 与 `Trainer` trait 分离。Trainer 接口保持只输出 `action_index`，
/// 额外上下文通过 [`Trainer::last_decision`](crate::game::Trainer::last_decision) 提供。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionInfo {
    /// 选中的动作索引（在传入候选列表中的位置）
    pub action_index: usize,

    /// 选中动作的评分（按 Trainer 内部约定的口径，如手写加权均分 / MCTS 搜索均分）
    pub score: f32,

    /// 所有候选动作的评分，按 `actions` 顺序排列
    ///
    /// 与 [`Self::action_index`] 等长，便于下游展示"为什么选 A 不选 B"。
    pub candidate_scores: Vec<f32>,

    /// 决策原因（手写逻辑说明 / MCTS 解释 / 自定义文案）
    pub reason: Option<String>,

    /// 决策耗时（毫秒）
    pub elapsed_ms: Option<u64>,

    /// 搜索相关（MCTS 适用）：实际搜索深度
    pub search_depth: Option<u32>,

    /// 搜索相关（MCTS 适用）：本次决策访问节点总数（visit count 总和）
    pub visit_count: Option<u32>,

    /// 评分细节（手写策略适用）
    ///
    /// key 为评分维度名（如 `"speed_bonus"`、`"pt_value"`），value 为对应分值。
    /// MCTS 场景通常不用此字段，用 [`Self::candidate_scores`] 表达。
    pub score_breakdown: Option<HashMap<String, f32>>,

    /// 剧本相关扩展字段
    ///
    /// 设计上保留最大灵活性：剧本特有信息（如拉面杯的"地区拉面"、温泉剧本的"温泉顺序"）
    /// 以 `serde_json::Value` 形式挂载，避免 [`DecisionInfo`] 被剧本 enum 渗透。
    /// 调用方需要时 `serde_json::to_value(&self).unwrap()` 整体序列化。
    pub scenario_extra: Option<serde_json::Value>,
}

impl DecisionInfo {
    /// 构造一个最小可用的 `DecisionInfo`（仅含 action_index）
    pub fn from_index(action_index: usize) -> Self {
        Self {
            action_index,
            ..Self::default()
        }
    }

    /// 构造含选中评分的 `DecisionInfo`
    pub fn from_index_and_score(action_index: usize, score: f32) -> Self {
        Self {
            action_index,
            score,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_zero_index() {
        let info = DecisionInfo::default();
        assert_eq!(info.action_index, 0);
        assert_eq!(info.score, 0.0);
        assert!(info.candidate_scores.is_empty());
        assert!(info.reason.is_none());
        assert!(info.elapsed_ms.is_none());
        assert!(info.search_depth.is_none());
        assert!(info.visit_count.is_none());
        assert!(info.score_breakdown.is_none());
        assert!(info.scenario_extra.is_none());
    }

    #[test]
    fn test_from_index_minimal() {
        let info = DecisionInfo::from_index(3);
        assert_eq!(info.action_index, 3);
        assert_eq!(info.score, 0.0);
    }

    #[test]
    fn test_from_index_and_score() {
        let info = DecisionInfo::from_index_and_score(2, 1234.5);
        assert_eq!(info.action_index, 2);
        assert!((info.score - 1234.5).abs() < 1e-6);
    }

    #[test]
    fn test_serde_roundtrip_minimal() {
        let info = DecisionInfo::default();
        let json = serde_json::to_string(&info).expect("serialize");
        let back: DecisionInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    #[test]
    fn test_serde_roundtrip_full() {
        let mut info = DecisionInfo::from_index_and_score(2, 1500.75);
        info.candidate_scores = vec![100.0, 200.0, 1500.75, 300.0];
        info.reason = Some("MCTS 选中评分最高动作".into());
        info.elapsed_ms = Some(128);
        info.search_depth = Some(8);
        info.visit_count = Some(4096);
        let mut breakdown = HashMap::new();
        breakdown.insert("speed".into(), 800.0);
        breakdown.insert("stamina".into(), 700.5);
        info.score_breakdown = Some(breakdown);
        info.scenario_extra = Some(serde_json::json!({
            "scenario": "ramen",
            "feeling_type": "A"
        }));

        let json = serde_json::to_string(&info).expect("serialize");
        let back: DecisionInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    #[test]
    fn test_serde_json_value_conversion() {
        // 验证文档 4.1 的写法：直接 serde_json::to_value 即可
        let info = DecisionInfo {
            action_index: 1,
            score: 42.0,
            candidate_scores: vec![10.0, 42.0, 30.0],
            reason: None,
            elapsed_ms: Some(7),
            search_depth: None,
            visit_count: None,
            score_breakdown: None,
            scenario_extra: None,
        };
        let v = serde_json::to_value(&info).expect("to_value");
        assert_eq!(v["action_index"], 1);
        assert_eq!(v["score"], 42.0);
        assert!(v["candidate_scores"].is_array());
    }
}
