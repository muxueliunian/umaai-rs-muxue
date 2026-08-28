//! 决策理由：原始数据 + 数据驱动的可读描述
//!
//! 分层依据（`handwritten_base_policy_plan.md` §4 输出分层 / 上游重构需求 §4.1
//! `explanation` 契约）：**原始数据与可读文字面向不同场景，走不同出口**。
//!
//! ```text
//! analyze_narrow_win（纯函数，门限先行懒计算）
//!   ├─ DecisionReasonData（原始数据，Serialize，schema 稳定）
//!   │    └─ DecisionReasonSink 接口发出
//!   │         └─ umasim 默认实现 [`NoopSink`]（静默丢弃——屏幕只要可读文字）；
//!   │            下游程序可实现自己的通道（[`LogJsonSink`] 接日志 / 文件 / socket）
//!   └─ render_reason_lines（可读文字，数据驱动渲染）
//!        └─ info! 直接上屏（措辞固定：中选 / 简称维度，子项条数 REASON_TOP_DIMS）
//! ```
//!
//! 两者都**不是** [`DecisionInfo`](crate::output::decision::DecisionInfo) 协议
//! 格式：协议 schema 冻结留待协议接入时从本结构映射，不被屏幕日志需求绑架。
//!
//! 输出频率由**分差门限**控制：只有存在与选中者分差 < `threshold` 的候选
//! （险胜局）才产生理由；悬殊局返回 `None` 完全静默——门限判断在终局维度
//! 分析**之前**，不浪费悬殊局的分析开销。`threshold = 0` 恒无接近组，等价禁用。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    global,
    gamedata::GAMECONSTANTS,
    search::{ActionResult, RamenSearchOutput, RamenTerminalStats, TerminalStats}
};

/// 理由分析的比较口径
///
/// 与 `RamenMctsTrainer::selection` 对齐：理由必须解释"实际为什么这么选"。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasonMetric {
    /// 结算评分口径（`action_results.0`）
    #[serde(rename = "score")]
    Score,
    /// PT 偏好口径（`action_results.1`）
    #[serde(rename = "pt")]
    Pt
}

impl ReasonMetric {
    /// 从候选统计中按口径取均值
    fn mean_of(self, pair: &(ActionResult, ActionResult)) -> f64 {
        match self {
            Self::Score => pair.0.mean(),
            Self::Pt => pair.1.mean()
        }
    }

    /// 从候选统计中按口径取样本数
    fn count_of(self, pair: &(ActionResult, ActionResult)) -> u32 {
        match self {
            Self::Score => pair.0.count(),
            Self::Pt => pair.1.count()
        }
    }

    /// 从候选统计中按口径取标准差
    fn stdev_of(self, pair: &(ActionResult, ActionResult)) -> f64 {
        match self {
            Self::Score => pair.0.stdev(),
            Self::Pt => pair.1.stdev()
        }
    }

    /// 口径的机器可读名（JSON 字段值）
    fn as_str(self) -> &'static str {
        match self {
            Self::Score => "score",
            Self::Pt => "pt"
        }
    }
}

/// 单个终局维度的差值（接近候选相对选中者）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimDelta {
    /// 稳定机器键（与 `[终局差异]` 日志一致，可对照）
    pub key: String,
    /// 人类可读标签（理由渲染用白名单简称：速/耐/力/根/智/PT）
    pub label: String,
    /// 量纲（`score` / `status` / `pt` / `flag`）
    pub unit: String,
    /// 差值 = 该候选均值 − 选中者均值；正值表示该候选在此维度占优
    pub delta: f64
}

/// 一个未中选候选（评分前 N 内）与选中者的对比（原始数据）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RivalReason {
    /// 候选下标（在本次搜索候选表中的位置）
    pub index: usize,
    /// 候选动作的可读文本
    pub desc: String,
    /// 分差 = 候选均值 − 选中者均值；**可为正**（算术均值口径与 weighted_mean
    /// 选择口径不同，允许未中选候选均值更高，如实显示不消除）
    pub gap: f64,
    /// 分差显著度置信（0~1，1 = 完全可分）；正态近似 Φ(|z|)
    pub confidence: f64,
    /// 候选的 rollout 样本数
    pub n: u32,
    /// 候选的口径均值
    pub mean: f64,
    /// 候选的口径标准差
    pub sd: f64,
    /// 该候选占优的终局维度（按 |delta| 降序，截断到 `REASON_TOP_DIMS`）
    pub pros: Vec<DimDelta>,
    /// 该候选占劣的终局维度（按 |delta| 降序，截断到 `REASON_TOP_DIMS`）
    pub cons: Vec<DimDelta>
}

/// 决策理由原始数据（下游程序消费的稳定结构）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionReasonData {
    /// 决策回合（0-based，与 `Game::turn()` 一致）
    pub turn: i32,
    /// 比较口径（`"score"` / `"pt"`）
    pub metric: String,
    /// 本次生效的分差门限（下游可复现"为什么这回合算险胜"）
    pub threshold: f64,
    /// 本次生效的最多显示选项数（rivals 即评分降序前 N 中排除选中者）
    pub max_display: usize,
    /// 选中候选下标（`select_action` 实际返回者，非 `best_action_idx`）
    pub chosen_index: usize,
    /// 选中候选的可读文本
    pub chosen_desc: String,
    /// 选中者的口径均值
    pub chosen_mean: f64,
    /// 选中者的样本数
    pub chosen_n: u32,
    /// 评分降序前 N 中的未中选候选（按评分降序排列；其余候选已直接排除）
    pub rivals: Vec<RivalReason>
}

/// 决策理由原始数据出口
///
/// 原始数据面向**下游程序**（分析/展示/协议），不应与人类日志耦合。
/// umasim 默认挂 [`NoopSink`]（屏幕只需要可读文字）；需要原始数据的下游
/// 程序可实现本 trait（或直接用 [`LogJsonSink`]）接入自己的通道。
/// 实现需 `Send + Sync`：trainer 会被 rayon 多局并行共享。
pub trait DecisionReasonSink: Send + Sync {
    /// 发出一条决策理由原始数据
    fn emit(&self, reason: &DecisionReasonData);
}

/// 静默出口：丢弃原始数据（umasim 默认——屏幕只需要可读文字）
///
/// 保留 sink 调用路径：下游接入时通过 trainer 的 `with_reason_sink`
/// 换成实际实现即可，核心层无需改动。
pub struct NoopSink;

impl DecisionReasonSink for NoopSink {
    fn emit(&self, _reason: &DecisionReasonData) {}
}

/// JSON 日志出口：序列化写入日志（target = `"decision_reason"`）
///
/// 供需要原始数据的宿主/下游显式启用；决策层输出，不受 `diag` 运行时
/// 开关管辖（那是规则层 rollout 的静默机制）。
pub struct LogJsonSink;

impl DecisionReasonSink for LogJsonSink {
    fn emit(&self, reason: &DecisionReasonData) {
        match serde_json::to_string(reason) {
            Ok(json) => log::info!(target: "decision_reason", "{json}"),
            Err(e) => log::warn!("决策理由 JSON 序列化失败: {e}")
        }
    }
}

/// 分差显著度置信：正态近似 Φ(|z|)，`Φ(z) ≈ 0.5·(1 + tanh(0.798·z))`
///
/// Welch 近似 `z = |gap| / √(sd₁²/n₁ + sd₂²/n₂)`；合成标准误为 0 时
/// （样本过少或分布完全一致）退化为边界值：分差非零 → 1.0，分差为零 → 0.5。
fn gap_confidence(gap: f64, sd_a: f64, n_a: u32, sd_b: f64, n_b: u32) -> f64 {
    let se = (sd_a * sd_a / n_a.max(1) as f64 + sd_b * sd_b / n_b.max(1) as f64).sqrt();
    if se <= 0.0 {
        return if gap == 0.0 { 0.5 } else { 1.0 };
    }
    let z = gap.abs() / se;
    0.5 * (1.0 + (0.798_0 * z).tanh())
}

/// 屏幕 ✓✗ 子项的维度白名单（key → 玩家向简称）：只解释**五维最终值与 PT**
///
/// 简称（速/耐/力/根/智/PT）仅用于理由渲染；`ramen_terminal` 的完整 label
/// （"最终速度"等）仍供 `[终局差异]` 等其他日志使用，此处不回写。
/// 白名单外的 19 维（评分分量/距上限/RMJ 达成/缺口合成）对玩家决策帮助
/// 有限，不进入理由分析；调参者看 `[终局差异]`（debug 级）与原始数据 sink。
const PLAYER_DIM_WHITELIST: [(&str, &str); 6] = [
    ("speed_final", "速"),
    ("stamina_final", "耐"),
    ("power_final", "力"),
    ("guts_final", "根"),
    ("wisdom_final", "智"),
    ("pt_score", "PT")
];

/// 每个候选最多显示的优势/劣势维度数（原风格配置项，风格层删除后固化为常量）
const REASON_TOP_DIMS: usize = 2;

/// 显著维度差值：评分前 N 未中选候选相对选中者的终局维度对比
///
/// 只在 [`PLAYER_DIM_WHITELIST`] 内取显著维度（label 用白名单简称），阈值与
/// `[终局差异]` 日志一致（flag ±2% 达成率、其余 ±1.0），各按 |delta| 降序
/// 截断到 `top_dims`。
///
/// `pt_score` 特例：其值 = `total_pt × pt_score_rate`（常数系数），展示时
/// 除回系数还原成 **PT 点数**（unit 记为 `"pt"`），玩家读到的就是"这个选项
/// 最终多拿多少 PT"，与五维同一可读粒度。
fn dim_deltas(chosen: &RamenTerminalStats, rival: &RamenTerminalStats, top_dims: usize) -> (Vec<DimDelta>, Vec<DimDelta>) {
    let mut base: HashMap<&'static str, (&'static str, &'static str, f64)> = HashMap::new();
    chosen.visit(&mut |m| {
        base.insert(m.key, (m.key, m.label, m.result.mean()));
    });
    let pt_rate = global!(GAMECONSTANTS).pt_score_rate as f64;
    let mut pros = Vec::new();
    let mut cons = Vec::new();
    rival.visit(&mut |m| {
        // 白名单外直接排除：不显示、不分析
        let Some((_, short)) = PLAYER_DIM_WHITELIST.iter().find(|&&(k, _)| k == m.key) else {
            return;
        };
        let Some((key, _, chosen_mean)) = base.get(m.key) else {
            return;
        };
        let mut delta = m.result.mean() - chosen_mean;
        // pt_score 除回折算系数，还原 PT 点数（系数来自 GAMECONSTANTS，
        // 非正时放弃换算、保持评分口径原值）
        let unit = if *key == "pt_score" && pt_rate > 0.0 {
            delta /= pt_rate;
            "pt"
        } else {
            m.unit
        };
        // 与 log_terminal_breakdown 相同的可见性阈值，避免噪声维度刷屏
        let visible = match unit {
            "flag" => delta.abs() >= 0.02,
            _ => delta.abs() >= 1.0
        };
        if !visible {
            return;
        }
        let dim = DimDelta {
            key: (*key).to_string(),
            label: (*short).to_string(),
            unit: unit.to_string(),
            delta
        };        if delta > 0.0 {
            pros.push(dim);
        } else {
            cons.push(dim);
        }
    });
    // sort_by 稳定排序：同 |delta| 保持 visit（字段声明）顺序
    pros.sort_by(|a, b| b.delta.abs().total_cmp(&a.delta.abs()));
    cons.sort_by(|a, b| b.delta.abs().total_cmp(&a.delta.abs()));
    pros.truncate(top_dims);
    cons.truncate(top_dims);
    (pros, cons)
}

/// 险胜分析：门限先行，悬殊局（无接近候选）返回 `None`
///
/// - `metric`：比较口径，应与 trainer 的 selection 一致
/// - `threshold`：分差门限，仅作**输出触发**；`0` 等价禁用
/// - `max_display`：最多显示/分析的选项数——触发后，全部候选按口径评分
///   降序只取前 N 个（一般含中选者），其余**直接排除**：不显示内容、
///   不做原因分析。中选者固定由调用方先行单独显示，不在本集合内。
///
/// 终局维度分析（较贵）只对进入前 N 的未中选候选执行（懒计算）。
pub fn analyze_narrow_win(
    turn: i32, metric: ReasonMetric, threshold: f64, max_display: usize, chosen: usize,
    output: &RamenSearchOutput
) -> Option<DecisionReasonData> {
    if output.actions.is_empty() || chosen >= output.actions.len() {
        return None;
    }
    let chosen_mean = metric.mean_of(&output.action_results[chosen]);
    // 第一遍：全部候选的口径评分，门限触发判断 + 排序
    let mut order: Vec<(usize, f64)> = output
        .action_results
        .iter()
        .enumerate()
        .map(|(i, pair)| (i, metric.mean_of(pair)))
        .collect();
    // 门限触发：存在与中选者接近的候选（险胜局）；悬殊局静默
    let near_exists = order
        .iter()
        .any(|&(i, mean)| i != chosen && (mean - chosen_mean).abs() < threshold);
    if !near_exists {
        return None;
    }
    // 评分降序取前 N（N 含中选者；中选者不在前 N 的极端情况不做特殊处理）
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    order.truncate(max_display.max(1));
    let mut rivals: Vec<RivalReason> = order
        .into_iter()
        .filter(|&(i, _)| i != chosen)
        .map(|(i, mean)| RivalReason {
            index: i,
            desc: output.actions[i].to_string(),
            gap: mean - chosen_mean,
            confidence: 0.0,
            n: metric.count_of(&output.action_results[i]),
            mean,
            sd: metric.stdev_of(&output.action_results[i]),
            pros: Vec::new(),
            cons: Vec::new()
        })
        .collect();
    if rivals.is_empty() {
        return None;
    }
    // 第二遍：门限已通过，对进入前 N 的候选补置信度与终局维度差
    let chosen_n = metric.count_of(&output.action_results[chosen]);
    let chosen_sd = metric.stdev_of(&output.action_results[chosen]);
    let chosen_stats = output.terminal_results.get(chosen);
    for rival in &mut rivals {
        rival.confidence = gap_confidence(rival.gap, chosen_sd, chosen_n, rival.sd, rival.n);
        if let (Some(base), Some(other)) = (chosen_stats, output.terminal_results.get(rival.index)) {
            let (pros, cons) = dim_deltas(base, other, REASON_TOP_DIMS);
            rival.pros = pros;
            rival.cons = cons;
        }
    }
    Some(DecisionReasonData {
        turn,
        metric: metric.as_str().to_string(),
        threshold,
        max_display: max_display.max(1),
        chosen_index: chosen,
        chosen_desc: output.actions[chosen].to_string(),
        chosen_mean,
        chosen_n,
        rivals
    })
}

/// 维度差值的渲染片段：`智+180` / `PT-33`
fn render_dim(dim: &DimDelta) -> String {
    let name = dim.label.as_str();
    match dim.unit.as_str() {
        "flag" => format!("{name}{:+.0}%", dim.delta * 100.0),
        _ => format!("{name}{:+.0}", dim.delta)
    }
}

/// 未中选候选的分差片段：带符号分数（正 = 该候选均值更高，如实显示）
fn render_gap(gap: f64) -> String {
    format!("{gap:+.0}")
}

/// 渲染可读文字（数据驱动，每行一条，直接供 `info!` 上屏）
///
/// 行 1 = 选中候选（只显示选项内容，无分数无置信度）；行 2.. = 评分前 N 中
/// 的未中选候选：`±分差 （优势子项, ✗劣势子项）`，帮助调参者判断
/// "险胜选对了吗"。置信度只在原始数据里，不上屏。
pub fn render_reason_lines(d: &DecisionReasonData) -> Vec<String> {
    let mut lines = vec![format!(
        "[回合 {}][决策理由] 中选: {}",
        d.turn + 1,
        d.chosen_desc
    )];
    for r in &d.rivals {
        // 维度子项：优势打钩、劣势打叉，按 |delta| 降序
        let mut dims: Vec<String> = r.pros.iter().map(render_dim).collect();
        dims.extend(r.cons.iter().map(render_dim));
        let detail = if dims.is_empty() {
            String::new()
        } else {
            format!(" （{}）", dims.join(", "))
        };
        lines.push(format!(
            "[回合 {}][决策理由] #{} {}: {}{}",
            d.turn + 1,
            r.index,
            r.desc,
            render_gap(r.gap),
            detail
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造指定样本序列的统计
    fn stats_of(values: &[f64]) -> ActionResult {
        let mut s = ActionResult::new();
        for v in values {
            s.add(*v);
        }
        s
    }

    /// 构造指定维度样本的终局统计（其余维度为空统计，均值 0）
    ///
    /// 只填 `wisdom_final`（数值类）与 `pt_score`（PT 折算，白名单内唯一
    /// 需要系数还原的维度）便于断言；空统计维度均值 0，差值 0 不进 pros/cons。
    fn terminal_of(wisdom_samples: &[f64], pt_samples: &[f64]) -> RamenTerminalStats {
        let mut t = RamenTerminalStats::default();
        for v in wisdom_samples {
            t.wisdom_final.add(*v);
        }
        for v in pt_samples {
            t.pt_score.add(*v);
        }
        t
    }

    /// 构造指定均值的 8 样本序列（固定零和偏移模式，均值精确、方差非零）
    fn samples_of(mean: f64) -> ActionResult {
        let offsets = [-1000.0, 0.0, 500.0, -500.0, 1000.0, 0.0, -250.0, 250.0];
        stats_of(&offsets.iter().map(|o| mean + o).collect::<Vec<_>>())
    }

    /// 构造 3 候选搜索输出：选中 0（均分 65000），#1 接近（差 +250），#2 悬殊（差 +2200）
    ///
    /// 终局维度：#1 相对选中 智力 +200、RMJ 达成率 −5%（一优一劣）。
    /// 动作文本生成依赖 `GAMECONSTANTS`（训练名），数据加载以 workspace 根为
    /// 工作目录（与 flat_search 等既有测试同一惯例）。
    fn sample_output() -> RamenSearchOutput {
        // expect：测试代码；目录/数据缺失时测试本就无意义
        let root = crate::utils::get_workspace_root().expect("定位 workspace 根");
        std::env::set_current_dir(root).expect("切换工作目录");
        crate::gamedata::init_global().expect("初始化全局数据");
        let actions = vec![
            crate::game::ramen::RamenAction::no_ramen(crate::game::ramen::Operation::Train(
                crate::game::ramen::TrainingType::Speed
            )),
            crate::game::ramen::RamenAction::no_ramen(crate::game::ramen::Operation::Train(
                crate::game::ramen::TrainingType::Wisdom
            )),
            crate::game::ramen::RamenAction::no_ramen(crate::game::ramen::Operation::Rest)
        ];
        // 注意：中选 #0 的均值**最低**——正是"允许未选项显示分数更高"的口径差异场景
        let pairs: Vec<(ActionResult, ActionResult)> = vec![
            (samples_of(65000.0), samples_of(60000.0)),
            (samples_of(65250.0), samples_of(60100.0)),
            (samples_of(67200.0), samples_of(61000.0))
        ];
        let terminals = vec![
            terminal_of(&[3000.0; 8], &[880.0; 8]),
            terminal_of(&[3200.0; 8], &[814.0; 8]),
            terminal_of(&[2800.0; 8], &[660.0; 8])
        ];
        RamenSearchOutput::with_terminals(actions, pairs, terminals, 10.0)
    }

    /// 门限先行：悬殊局返回 None，不产生理由数据
    #[test]
    fn test_landslide_silent() {
        // 门限 200：#1 差 250、#2 差 2000 均不在门限内 → 悬殊静默
        let out = analyze_narrow_win(22, ReasonMetric::Score, 200.0, 5, 0, &sample_output());
        println!("门限 200: {out:?}");
        assert!(out.is_none(), "无接近候选时应返回 None");
    }

    /// 险胜局：rivals = 评分降序前 N 的未中选（按评分降序；允许分差为正）
    #[test]
    fn test_narrow_win_analysis() {
        let data = analyze_narrow_win(22, ReasonMetric::Score, 300.0, 5, 0, &sample_output())
            .expect("存在差 250 的接近候选，不应为 None");
        println!("chosen = {} ({})", data.chosen_index, data.chosen_desc);
        for r in &data.rivals {
            println!("#{} gap={:+.0} conf={:.2}", r.index, r.gap, r.confidence);
        }
        // 评分降序：#2(67200) > #1(65250)；中选 #0(65000) 不在 rivals
        assert_eq!(data.rivals.len(), 2, "max_display=5 下前 N 含全部 3 候选，rivals = 2 个未中选");
        assert_eq!(data.rivals[0].index, 2, "评分最高的 #2 排第一");
        assert!((data.rivals[0].gap - 2200.0).abs() < 1e-6, "#2 分差 +2200（允许未选项更高）");
        assert_eq!(data.rivals[1].index, 1);
        assert!((data.rivals[1].gap - 250.0).abs() < 1e-6, "#1 分差 +250");
        assert!((data.rivals[1].confidence - 0.5).abs() > 0.01, "非零标准差下置信度不应恰为 0.5");
        // 终局维度：#1 智力 +200（占优）、PT -33（占劣；pt_score 差 -66 / 系数 2.0）
        let r = &data.rivals[1];
        assert_eq!(r.pros.len(), 1, "pros 应恰含 wisdom_final");
        assert_eq!(r.pros[0].key, "wisdom_final");
        assert_eq!(r.pros[0].label, "智", "label 用白名单简称");
        assert!((r.pros[0].delta - 200.0).abs() < 1e-6);
        assert_eq!(r.cons.len(), 1, "cons 应恰含 pt_score（rmj 等已在白名单外）");
        assert_eq!(r.cons[0].key, "pt_score");
        assert_eq!(r.cons[0].label, "PT", "pt_score 展示时还原为 PT 点数");
        assert!((r.cons[0].delta + 33.0).abs() < 1e-6, "PT 差 = -66 评分 / pt_score_rate 2.0");
        // 原始数据可序列化（下游契约）
        let json = serde_json::to_string(&data).expect("JSON 序列化");
        println!("json = {json}");
        assert!(json.contains("\"metric\":\"score\""));
        assert!(json.contains("\"threshold\":300.0"));
        assert!(json.contains("\"max_display\":5"));
    }

    /// PT 口径：分析跟随所选口径而非默认 score 口径
    #[test]
    fn test_pt_metric() {
        let data = analyze_narrow_win(22, ReasonMetric::Pt, 300.0, 5, 0, &sample_output())
            .expect("PT 口径下 #1 差 100 仍应触发");
        println!("metric = {}", data.metric);
        for r in &data.rivals {
            println!("#{} gap={:+.0}", r.index, r.gap);
        }
        assert_eq!(data.metric, "pt");
        // PT 口径：#2(61000) gap=+1000 排第一，#1(60100) gap=+100 随后
        assert_eq!(data.rivals[0].index, 2);
        assert!((data.rivals[0].gap - 1000.0).abs() < 1e-6);
        assert!((data.rivals[1].gap - 100.0).abs() < 1e-6);
    }

    /// 最多显示选项数：评分降序前 N 截断，N 之外不显示不分析
    #[test]
    fn test_max_display_truncate() {
        // 一般情形（N 含中选者）：max_display=3 = 全部候选，rivals = 2 个未中选
        let data = analyze_narrow_win(22, ReasonMetric::Score, 300.0, 3, 0, &sample_output())
            .expect("#1 差 250 < 300，门限触发");
        println!("max_display=3 rivals = {}", data.rivals.len());
        assert_eq!(data.rivals.len(), 2);
        assert_eq!(data.max_display, 3);
        // N 截断：max_display=2 时前 2 = #2/#1（本样例中选者均值最低、不在前 N，
        // 属约定的"不考虑"情形——行为为前 N 中全部非中选者都显示）
        let narrow = analyze_narrow_win(22, ReasonMetric::Score, 300.0, 2, 0, &sample_output())
            .expect("门限触发");
        println!("max_display=2 rivals = {}", narrow.rivals.len());
        assert_eq!(narrow.rivals.len(), 2, "前 2 = #2/#1 均非中选，全部显示");
        // 门限仍作触发器：无接近候选时即使 N 很大也静默
        let silent = analyze_narrow_win(22, ReasonMetric::Score, 200.0, 5, 0, &sample_output());
        assert!(silent.is_none(), "门限未触发（最小分差 250 > 200）应静默");
    }

    /// 可读渲染：中选行只显内容 + 未中选 ±分差与 ✓✗ 子项
    #[test]
    fn test_render_lines() {
        let data = analyze_narrow_win(22, ReasonMetric::Score, 300.0, 5, 0, &sample_output()).unwrap();
        let lines = render_reason_lines(&data);
        println!("--- 渲染输出 ---");
        for l in &lines {
            println!("{l}");
        }
        // 中选 1 行 + 未中选 2 行（#2、#1）
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("中选"), "中选行含风格 chosen_word");
        assert!(!lines[0].contains("置信"), "中选行不显示置信度");
        assert!(lines[0].contains(&data.chosen_desc), "中选行只显示选项内容");
        assert!(lines[1].contains("#2"), "第二行是评分最高的未中选 #2");
        assert!(lines[1].contains("+2200"), "分差带符号显示（允许为正）");
        assert!(lines[2].contains("#1") && lines[2].contains("+250"));
        assert!(lines[2].contains("智+200"), "优势子项（简称）");
        assert!(lines[2].contains("PT-33"), "劣势子项且 PT 已还原点数");
        assert!(!lines.iter().any(|l| l.contains("置信")), "置信度不再上屏（只在原始数据）");
    }

    /// 置信度边界：合成标准误为 0 时按分差退化到 1.0 / 0.5
    #[test]
    fn test_gap_confidence_degenerate() {
        println!("零误差零分差: {}", gap_confidence(0.0, 0.0, 8, 0.0, 8));
        println!("零误差有分差: {}", gap_confidence(100.0, 0.0, 8, 0.0, 8));
        assert!((gap_confidence(0.0, 0.0, 8, 0.0, 8) - 0.5).abs() < 1e-9);
        assert!((gap_confidence(100.0, 0.0, 8, 0.0, 8) - 1.0).abs() < 1e-9);
    }

    /// NoopSink 静默丢弃：emit 后无 panic 无输出（原始数据出口的 umasim 默认）
    #[test]
    fn test_noop_sink() {
        let data = DecisionReasonData {
            turn: 0,
            metric: "score".to_string(),
            threshold: 300.0,
            max_display: 5,
            chosen_index: 0,
            chosen_desc: "测试".to_string(),
            chosen_mean: 65000.0,
            chosen_n: 8,
            rivals: vec![]
        };
        NoopSink.emit(&data);
        println!("NoopSink emit 完成（静默）");
    }
}
