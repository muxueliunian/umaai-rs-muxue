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
//!
//! ## 2026-09 简化
//!
//! 旧版 stub 字段（`reason` / `search_depth` / `visit_count` / `score_breakdown` /
//! `elapsed_ms`）已删除——JSON 输出不再暴露这些字段。**保留** `candidate_scores`
//! / `candidate_n`（按用户拍板"备选选项分复用"，下游需要展示"为什么选 A 不选 B"）。
//! `scenario_extra` 承载 luck_score / action_luck / reason 三类剧本特化信息。

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

    /// 决策子类型标识（**新增 2026-09**，按用户拍板"区分 partial decision"）
    ///
    /// partial decision 是拉面剧本的常态——三阶段（吃面 + 隐藏诀窍 + 训练）会拆成
    /// 多次 select_action，每次只描述当前子阶段。C# 端必须能区分：
    /// - `ramen_select`：吃哪个面（**不含**隐藏诀窍——三阶段路径下独立）
    /// - `special_select`：吃哪个面 + 隐藏诀窍用法（基于 RamenSelect 选择）
    /// - `train`：训练 / 比赛 / 休息 / 外出 / 治病
    /// - `region_select`：年度地区选择
    /// - `super_ramen_select`：超级拉面选择（71 回合后）
    /// - `event`：事件选择
    ///
    /// main.rs `calc_ramen_training` 在每次 select_action 前 snapshot `game.stage` 并填这个字段。
    /// onsen 路径填 `"train"` / `"event"`。
    pub decision_kind: String,

    /// 所有候选动作的评分，按 `actions` 顺序排列（**完整保留**）
    ///
    /// 与 [`Self::action_index`] 等长，便于下游展示"为什么选 A 不选 B"。
    /// 按用户 2026-09 拍板：备选选项分是下游必要信息，**不**从 struct 删除。
    pub candidate_scores: Vec<f32>,

    /// 各候选动作的可读描述（**新增 2026-09**，按用户拍板"备选选项名下游不可缺"）
    ///
    /// 与 [`Self::candidate_scores`] / [`Self::candidate_n`] **严格同长同序同截断**。
    /// C# 端 `action_index` 拿到的是数字（候选下标），**没有这个字段无法映射动作名**
    /// ——尤其拉面组合动作（吃面配方 + 特殊目标 + 操作三阶段）名字极长，
    /// 数字索引本身没有语义。下游必须按 `candidate_descriptions[i]` 取名。
    ///
    /// 实现路径：onsen 从 `SearchOutput.actions[i].to_string()` 取；
    /// 拉面 MCTS 从 `RamenSearchOutput.actions[i].to_string()` 取并在 `LastSearchSummary`
    /// 缓存；手写策略从 `actions[i].to_string()` 取并在 `LastDecisionSummary` 缓存。
    pub candidate_descriptions: Vec<String>,

    /// 各候选的 rollout 样本数（**完整保留**，与 [`Self::candidate_scores`] 严格同长同序同截断）
    ///
    /// UCB 下各候选跑数可能悬殊，手写策略 / 随机等无局数概念的 trainer 留空
    /// （`Vec::default()` 即 `vec![]`）。下游可用此字段算候选置信度 / luck baseline。
    pub candidate_n: Vec<u32>,

    /// 剧本相关扩展字段（**JSON 顶层唯一额外信息出口**）
    ///
    /// 承载四类信息（按 trainer 是否支持灵活挂载）：
    /// - `luck_score`：本局 + 本回合运气分（[`crate::luck_score::LuckScoreSnapshot`]）
    /// - `action_luck`：每候选"选项后运气分"（按局数加权 T(n,action_i) − T(n)）
    /// - `reason`：human mode reason 输出所需信息（[`crate::output::DecisionReasonData`]）
    /// - `ramen_action`：选中动作的 to_string（仅 ramen 剧本挂——`RamenAction::to_string()`
    ///   已含吃面 + 隐藏诀窍 + 操作三阶段信息，按用户拍板"AIRed 端只显示不解析"）
    ///
    /// 不在本结构内堆叠剧本 enum——以 `serde_json::Value` 形式挂载，调用方按需解析。
    pub scenario_extra: Option<serde_json::Value>
}

/// 决策来源标签：**整局网络直接决策**（`ramen_trainer_policy = "nn"`）
///
/// 与 [`SOURCE_REGION_NN`] 的区别是接管面：那个只接管地区，这个接管整局所有动作决策
/// （事件选项仍走手写）。两者都**没有搜索评分**，渲染端按本标签区分文案。
///
/// ❗只在**真的跑了一次网络推理**的那一步用本标签。同一模式下还有两类不经推理就定案的
/// 步骤，各有自己的标签：[`SOURCE_RAMEN_RACE_GATE`]、[`SOURCE_RAMEN_SINGLE_CANDIDATE`]。
pub const SOURCE_RAMEN_NN: &str = "ramen_nn";

/// 决策来源标签：**自选比赛硬守门**命中（整局网络模式下）
///
/// 守门是硬性义务而非价值权衡：区间内剩余可比赛回合已不够补齐缺口时，无视 policy
/// 直接选「比赛」。这一步**没有跑推理**，不能标成 [`SOURCE_RAMEN_NN`]。
pub const SOURCE_RAMEN_RACE_GATE: &str = "ramen_race_gate";

/// 决策来源标签：**唯一候选**直接定案（整局网络模式下）
///
/// 候选只有一个时 argmax 的结果与 policy 无关，整次推理被省掉。这一步同样**没有跑推理**。
pub const SOURCE_RAMEN_SINGLE_CANDIDATE: &str = "ramen_single_candidate";

/// 决策来源标签：`SpecialSelect` 整阶段按配置交给手写策略（整局网络模式下）
///
/// 只在 `SpecialSelectMode::Handwritten` 口径下出现；客户端默认取 `Canonical`，
/// 因此正常不会看到。留着是为了让来源标签覆盖 `prepare_decision` 的**全部**出口，
/// 不出现「无标签」的空洞。
pub const SOURCE_RAMEN_HANDWRITTEN_STAGE: &str = "ramen_handwritten_stage";

/// 决策来源标签：**地区搜索**做出的地区选择
///
/// 只在对照模式 `ramen_region_policy = "mcts_compare"` 且 `ramen_search_stages` 含
/// `region` 时出现：执行推荐就是那条真搜索决策本身，**带完整候选评分**。与
/// [`SOURCE_REGION_HANDWRITTEN`] 的区别正是「这一侧到底搜没搜」。
pub const SOURCE_REGION_SEARCH: &str = "region_search";

/// 决策来源标签：**既有装配**（手写地区基策）做出的地区选择
///
/// 只在对照模式（`ramen_region_policy = "mcts_compare"` 且未开 region 搜索）下
/// 出现：此时执行推荐取自手写基策，屏幕必须照实说是手写，而不是含糊的「无搜索评分」。
pub const SOURCE_REGION_HANDWRITTEN: &str = "region_handwritten";

/// 决策来源标签：**仅接管外层地区选择**的神经网络
///
/// 见 `umaai::region`。用常量而不是散落的字符串字面量，避免写端与读端拼错。
pub const SOURCE_REGION_NN: &str = "region_nn";

impl DecisionInfo {
    /// [`Self::scenario_extra`] 里承载**决策来源**的键名
    ///
    /// 走既有的 `scenario_extra` 出口而不是新增顶层字段：下游按
    /// `serde_json::Value` 解析，多一个键向后兼容。
    pub const SOURCE_KEY: &'static str = "decision_source";

    /// 标注本条决策由谁做出
    ///
    /// ❗只标注**来源**，不伪造任何评分：`score` / `candidate_scores` /
    /// `candidate_n` 一律保持原样。上层按 `candidate_scores` 是否为空决定要不要
    /// 走 luck 挂载，因此本方法**不改变 luck baseline 口径**。
    ///
    /// `scenario_extra` 已是 JSON 对象时就地插入键；为 `None` 时新建一个只含该键
    /// 的对象。（`scenario_extra` 在本项目里恒为对象或 `None`，不存在第三种形态。）
    pub fn with_source(mut self, label: &str) -> Self {
        let value = serde_json::Value::String(label.to_string());
        match self.scenario_extra {
            Some(serde_json::Value::Object(ref mut map)) => {
                map.insert(Self::SOURCE_KEY.to_string(), value);
            }
            _ => {
                let mut map = serde_json::Map::new();
                map.insert(Self::SOURCE_KEY.to_string(), value);
                self.scenario_extra = Some(serde_json::Value::Object(map));
            }
        }
        self
    }

    /// 读取决策来源标签；**未标注时为 `None`**
    ///
    /// ❗`None` 的含义是「来源未知」，**不等于**「手写」。渲染端据此走中性文案，
    /// 不得把「没有搜索评分」当成「手写逻辑」。
    pub fn source_label(&self) -> Option<&str> {
        self.scenario_extra
            .as_ref()?
            .get(Self::SOURCE_KEY)?
            .as_str()
    }

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

    /// 简化后字段集：7 个（action_index / score / decision_kind / candidate_scores /
    /// candidate_descriptions / candidate_n / scenario_extra）——旧 stub 字段
    /// （reason / search_depth / visit_count / score_breakdown / elapsed_ms）已删
    #[test]
    fn test_default_is_zero_index() {
        let info = DecisionInfo::default();
        assert_eq!(info.action_index, 0);
        assert_eq!(info.score, 0.0);
        assert_eq!(info.decision_kind, "", "默认空字符串（由 caller 填）");
        assert!(info.candidate_scores.is_empty());
        assert!(info.candidate_descriptions.is_empty());
        assert!(info.candidate_n.is_empty(), "默认无局数概念");
        assert!(info.scenario_extra.is_none());
    }

    /// 来源标签：写入 / 读出 / 未标注为 None，且不动任何评分字段
    #[test]
    fn test_source_label_roundtrip() {
        let bare = DecisionInfo::from_index(3);
        assert_eq!(bare.source_label(), None, "未标注时来源为 None（≠ 手写）");

        let tagged = DecisionInfo {
            action_index: 3,
            candidate_descriptions: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            ..DecisionInfo::default()
        }
        .with_source(SOURCE_REGION_NN);
        assert_eq!(tagged.source_label(), Some(SOURCE_REGION_NN));
        assert!(tagged.candidate_scores.is_empty(), "标注来源不伪造搜索评分");
        assert_eq!(tagged.score, 0.0, "标注来源不改 score");

        // 已有 scenario_extra 时就地插入，不丢原有键
        let merged = DecisionInfo {
            scenario_extra: Some(serde_json::json!({"ramen_action": "吃面/札幌"})),
            ..DecisionInfo::default()
        }
        .with_source(SOURCE_REGION_NN);
        let extra = merged.scenario_extra.as_ref().expect("scenario_extra");
        assert_eq!(extra.get("ramen_action").and_then(|v| v.as_str()), Some("吃面/札幌"));
        assert_eq!(merged.source_label(), Some(SOURCE_REGION_NN));
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
    fn test_serde_roundtrip_with_scenario_extra() {
        // 验证 scenario_extra（含 luck_score / action_luck / reason / ramen_action）
        // + candidate_descriptions + decision_kind 能正确序列化往返
        let info = DecisionInfo {
            action_index: 2,
            score: 1500.75,
            decision_kind: "ramen_select".to_string(),
            candidate_scores: vec![100.0, 200.0, 1500.75, 300.0],
            candidate_descriptions: vec![
                "不吃面".to_string(),
                "吃面/札幌".to_string(),
                "吃面/中山-全(替换Bx1+Ax2)".to_string(),
                "吃面/千叶".to_string()
            ],
            candidate_n: vec![100, 200, 1500, 300],
            scenario_extra: Some(serde_json::json!({
                "scenario": "ramen",
                "luck_score": {
                    "initial_terminal_baseline": 50078.0,
                    "current_terminal_baseline": 50354.0,
                    "total_luck_score": 276.0,
                    "last_turn_delta": -121.0
                },
                "action_luck": {"0": -50.0, "1": 125.5},
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
                         "cons": []}
                    ]
                }
            }))
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let back: DecisionInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    /// 顶层 JSON 不输出 5 个已删 stub 字段——保留 7 字段（含 decision_kind /
    /// candidate_descriptions）
    #[test]
    fn test_json_top_level_omits_stub_fields() {
        let info = DecisionInfo {
            action_index: 1,
            score: 42.0,
            decision_kind: "ramen_select".to_string(),
            candidate_scores: vec![10.0, 42.0, 30.0],
            candidate_descriptions: vec![
                "不吃面".to_string(),
                "吃面/中山-全(替换Bx1+Ax2)".to_string(),
                "不吃面".to_string()
            ],
            candidate_n: vec![100, 200, 50],
            scenario_extra: None
        };
        let v = serde_json::to_value(&info).expect("to_value");
        assert_eq!(v["action_index"], 1);
        assert_eq!(v["score"], 42.0);
        assert_eq!(v["decision_kind"], "ramen_select", "decision_kind 顶层保留");
        assert!(v["candidate_scores"].is_array(), "candidate_scores 保留");
        assert!(v["candidate_descriptions"].is_array(), "candidate_descriptions 保留");
        assert!(v["candidate_n"].is_array(), "candidate_n 保留");
        // 顶层不应再有这些已删字段
        assert!(v.get("reason").is_none(), "reason 已从 DecisionInfo 删除");
        assert!(v.get("search_depth").is_none(), "search_depth 已删除");
        assert!(v.get("visit_count").is_none(), "visit_count 已删除");
        assert!(v.get("score_breakdown").is_none(), "score_breakdown 已删除");
        assert!(v.get("elapsed_ms").is_none(), "elapsed_ms 已删除");
        // 候选描述数组与 scores / n 严格同长
        assert_eq!(
            v["candidate_descriptions"].as_array().unwrap().len(),
            v["candidate_scores"].as_array().unwrap().len(),
            "candidate_descriptions 与 candidate_scores 严格同长"
        );
    }
}
