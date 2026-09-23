//! decisions.csv 解析 → digest.decisions / coverage / luck（文档 §3.5、§5.1、§5.3）
//!
//! - **按表头名取值**，不硬编码列序（在线记录器列序 = `record::detail_header`）
//! - 锚点语义（§5.3）：决策点 = 每回合最后一份 `outcome=calc` 的快照；
//!   同回合 `seq` 递增是阶段推进不是行动结果
//! - chain_len 推断：同 `file` 的行**连续出现**成一组链（在线留空、离线重放
//!   填写；§5.3 实测「file 变化次数 = 唯一 file 数」→ 推断可靠）
//! - 运气分（§6.2）：`turn_delta` 按**回合合计**聚合（同回合多段 Δ 的单段
//!   会误导）；`flagged_turns` 伪波动标记留 M4 里程碑

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;

/// 候选项（cand{N}_desc / _score / _n；rank = 列序 + 1）
#[derive(Debug, Clone, Serialize)]
pub struct Cand {
    pub rank: usize,
    pub desc: String,
    pub score: Option<f64>,
    pub n: Option<u64>,
    /// 与最优候选的分差（cand1_score − 本候选 score）
    pub gap_to_best: Option<f64>,
}

/// 选中项
#[derive(Debug, Clone, Default, Serialize)]
pub struct Chosen {
    pub idx: Option<usize>,
    pub desc: String,
    pub action_luck: Option<f64>,
}

/// 一行决策明细（digest decisions 条目 + 解析中间字段）
#[derive(Debug, Clone, Serialize)]
pub struct DecRow {
    pub file: String,
    pub turn: u32,
    pub seq: u32,
    pub stage: String,
    #[serde(rename = "kind")]
    pub decision_kind: String,
    pub candidates: Vec<Cand>,
    pub chosen: Chosen,
    pub t_n_raw: Option<f64>,
    pub t_n_display: Option<f64>,
    pub total_luck: Option<f64>,
    pub turn_delta: Option<f64>,
    /// 同 file 连续行数（链式决策长度；单行 = 1）
    pub chain_len: usize,
    // —— 以下为解析中间字段，不落 digest ——
    /// 行类型（calc / skip / no_emit）
    #[serde(skip)]
    pub outcome: String,
    /// skip 原因（与 classify_begin_reason 同口径）
    #[serde(skip)]
    pub reason: String,
    /// 链内序号（0 基）
    #[serde(skip)]
    pub step: usize,
}

/// coverage（§3.5）：skip 按原因分组 / no_emit / unparsed / parse_error
#[derive(Debug, Default, Clone, Serialize)]
pub struct Coverage {
    pub skip: SkipCoverage,
    pub no_emit: u64,
    /// 包内 unparsed 快照数（pack 提供）
    pub unparsed: u64,
    /// 快照反序列化失败数（timeline 提供）
    pub parse_error: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SkipCoverage {
    pub by_reason: BTreeMap<String, u64>,
}

/// 伪波动标记回合（§6.7：年界 / 继承 / RMJ——归因时**降级或跳过**）
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FlaggedTurn {
    pub turn: u32,
    /// 标记原因（`year_boundary(24)` / `inherit` / `rmj_settle(46)` 等）
    pub reason: String,
}

/// luck 块（§3.5 / §6.2）
#[derive(Debug, Default, Clone, Serialize)]
pub struct LuckBlock {
    /// 运气分序列（每条带 total_luck 的决策行）
    pub series: Vec<LuckPoint>,
    /// 回合合计 Δ 最大的回合（top 5）
    pub top_gain: Vec<TurnDelta>,
    /// 回合合计 Δ 最小的回合（top 5）
    pub top_loss: Vec<TurnDelta>,
    /// 单段 turn_delta 统计
    pub raw_delta_stats: DeltaStats,
    /// 伪波动标记回合（年界 / 继承 / RMJ；由 `checks::flagged_turns` 填充）
    pub flagged_turns: Vec<FlaggedTurn>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LuckPoint {
    pub turn: u32,
    pub seq: u32,
    pub total_luck: f64,
}

/// 回合合计 Δ（§6.2「回合合计是基本观察单位」）
#[derive(Debug, Clone, Serialize)]
pub struct TurnDelta {
    pub turn: u32,
    /// 回合内各段 Δ 之和
    pub delta: f64,
    /// 同回合各段（RamenSelect / Train / RegionSelect 等决策点各自的 Δ）
    pub segments: Vec<f64>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct DeltaStats {
    pub n: u64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

/// decisions.csv 解析结果
#[derive(Debug, Default)]
pub struct DecisionsResult {
    /// 决策行（`outcome=calc`；skip / no_emit 只进 coverage）
    pub rows: Vec<DecRow>,
    pub coverage: Coverage,
    pub luck: LuckBlock,
}

/// 解析 decisions.csv 原文（`None` / 空文本 → 空结果，coverage 全零）
pub fn parse(csv_text: Option<&str>) -> Result<DecisionsResult> {
    let Some(text) = csv_text.filter(|t| !t.trim().is_empty()) else {
        return Ok(DecisionsResult::default());
    };
    let mut rdr = csv::ReaderBuilder::new().from_reader(text.as_bytes());
    let headers = rdr.headers()?.clone();

    // 按表头名取列号（不硬编码列序）
    let col = |name: &str| headers.iter().position(|h| h == name);
    let c_file = col("file");
    let c_turn = col("turn");
    let c_seq = col("seq");
    let c_stage = col("stage");
    let c_outcome = col("outcome");
    let c_reason = col("reason");
    let c_kind = col("decision_kind");
    let c_chosen_idx = col("chosen_idx");
    let c_chosen_desc = col("chosen_desc");
    let c_chosen_luck = col("chosen_action_luck");
    let c_t_raw = col("t_n_raw");
    let c_t_disp = col("t_n_display");
    let c_total = col("total_luck");
    let c_delta = col("turn_delta");
    let cand_cols: Vec<(usize, Option<usize>, Option<usize>)> = (1..=5)
        .map(|i| {
            (
                col(&format!("cand{i}_desc")).unwrap_or(usize::MAX),
                col(&format!("cand{i}_score")),
                col(&format!("cand{i}_n")),
            )
        })
        .collect();

    let opt_str = |rec: &csv::StringRecord, i: Option<usize>| {
        i.and_then(|i| rec.get(i)).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    };
    let opt_f64 = |rec: &csv::StringRecord, i: Option<usize>| {
        opt_str(rec, i).and_then(|s| s.parse::<f64>().ok())
    };
    let opt_u32 = |rec: &csv::StringRecord, i: Option<usize>| {
        opt_str(rec, i).and_then(|s| s.parse::<u32>().ok())
    };

    let mut all_rows: Vec<DecRow> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let candidates = cand_cols
            .iter()
            .enumerate()
            .filter(|(_, (desc_i, _, _))| {
                *desc_i != usize::MAX
                    && opt_str(&rec, Some(*desc_i)).is_some()
            })
            .map(|(rank0, (desc_i, score_i, n_i))| {
                let desc = opt_str(&rec, Some(*desc_i)).unwrap_or_default();
                let score = opt_f64(&rec, *score_i);
                let n = opt_str(&rec, *n_i).and_then(|s| s.parse::<u64>().ok());
                // gap_to_best：与 cand1 的分差（cand1 = 最优）
                let best = opt_f64(&rec, cand_cols[0].1);
                let gap = match (best, score) {
                    (Some(b), Some(s)) => Some(b - s),
                    _ => None,
                };
                Cand { rank: rank0 + 1, desc, score, n, gap_to_best: gap }
            })
            .collect();
        all_rows.push(DecRow {
            file: opt_str(&rec, c_file).unwrap_or_default(),
            turn: opt_u32(&rec, c_turn).unwrap_or(0),
            seq: opt_u32(&rec, c_seq).unwrap_or(0),
            stage: opt_str(&rec, c_stage).unwrap_or_default(),
            decision_kind: opt_str(&rec, c_kind).unwrap_or_default(),
            candidates,
            chosen: Chosen {
                idx: opt_str(&rec, c_chosen_idx).and_then(|s| s.parse::<usize>().ok()),
                desc: opt_str(&rec, c_chosen_desc).unwrap_or_default(),
                action_luck: opt_f64(&rec, c_chosen_luck),
            },
            t_n_raw: opt_f64(&rec, c_t_raw),
            t_n_display: opt_f64(&rec, c_t_disp),
            total_luck: opt_f64(&rec, c_total),
            turn_delta: opt_f64(&rec, c_delta),
            chain_len: 1,
            outcome: opt_str(&rec, c_outcome).unwrap_or_default(),
            reason: opt_str(&rec, c_reason).unwrap_or_default(),
            step: 0,
        });
    }

    // 链推断：同 file 连续行成组（file 变化次数 = 唯一 file 数，§5.3 实测）
    let mut pos = 0usize;
    while pos < all_rows.len() {
        let mut end = pos;
        while end + 1 < all_rows.len() && all_rows[end + 1].file == all_rows[pos].file {
            end += 1;
        }
        let len = end - pos + 1;
        for k in pos..=end {
            all_rows[k].chain_len = len;
            all_rows[k].step = k - pos;
        }
        pos = end + 1;
    }

    // coverage
    let mut coverage = Coverage::default();
    for r in &all_rows {
        match r.outcome.as_str() {
            "skip" => {
                let key = if r.reason.is_empty() { "unlabeled".to_string() } else { r.reason.clone() };
                *coverage.skip.by_reason.entry(key).or_default() += 1;
            }
            "no_emit" => coverage.no_emit += 1,
            _ => {}
        }
    }

    // 决策行（calc）
    let rows: Vec<DecRow> = all_rows.into_iter().filter(|r| r.outcome == "calc").collect();

    // luck 聚合
    let luck = build_luck(&rows);

    Ok(DecisionsResult { rows, coverage, luck })
}

/// luck 块聚合（§6.2：回合合计为基本观察单位）
fn build_luck(rows: &[DecRow]) -> LuckBlock {
    // series：每条带 total_luck 的决策行
    let series = rows
        .iter()
        .filter_map(|r| r.total_luck.map(|v| LuckPoint { turn: r.turn, seq: r.seq, total_luck: v }))
        .collect();

    // 回合合计 Δ（含段明细）
    let mut by_turn: BTreeMap<u32, (f64, Vec<f64>)> = BTreeMap::new();
    for r in rows {
        if let Some(d) = r.turn_delta {
            let e = by_turn.entry(r.turn).or_default();
            e.0 += d;
            e.1.push(d);
        }
    }
    let turn_deltas: Vec<TurnDelta> = by_turn
        .into_iter()
        .map(|(turn, (delta, segments))| TurnDelta { turn, delta, segments })
        .collect();
    let mut top_gain = turn_deltas.clone();
    top_gain.sort_by(|a, b| b.delta.partial_cmp(&a.delta).unwrap_or(std::cmp::Ordering::Equal));
    top_gain.truncate(5);
    let mut top_loss = turn_deltas.clone();
    top_loss.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap_or(std::cmp::Ordering::Equal));
    top_loss.truncate(5);

    // 单段统计
    let deltas: Vec<f64> = rows.iter().filter_map(|r| r.turn_delta).collect();
    let stats = if deltas.is_empty() {
        DeltaStats::default()
    } else {
        let n = deltas.len() as u64;
        let mean = deltas.iter().sum::<f64>() / n as f64;
        let var = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n as f64;
        DeltaStats {
            n,
            mean,
            std: var.sqrt(),
            min: deltas.iter().cloned().fold(f64::INFINITY, f64::min),
            max: deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    };

    LuckBlock { series, top_gain, top_loss, raw_delta_stats: stats, flagged_turns: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试 CSV：真实表头（record::detail_header 同序）+ 按名填列
    fn make_csv(rows: &[(&str, &str, &str, &str, &str, &str, &str, &str, &str, &str)]) -> String {
        // (file, turn, seq, stage, outcome, reason, kind, cand1_score, total_luck, turn_delta)
        let header = umaai::decision::record::detail_header().join(",");
        let mut out = header.clone();
        out.push('\n');
        for (file, turn, seq, stage, outcome, reason, kind, c1s, total, delta) in rows {
            let headers: Vec<&str> = header.split(',').collect();
            let mut cells: Vec<String> = vec![String::new(); headers.len()];
            let set = |cells: &mut Vec<String>, name: &str, v: &str| {
                if let Some(i) = headers.iter().position(|h| *h == name) {
                    cells[i] = v.to_string();
                }
            };
            set(&mut cells, "file", file);
            set(&mut cells, "turn", turn);
            set(&mut cells, "seq", seq);
            set(&mut cells, "stage", stage);
            set(&mut cells, "outcome", outcome);
            set(&mut cells, "reason", reason);
            set(&mut cells, "decision_kind", kind);
            set(&mut cells, "cand1_desc", "首选");
            set(&mut cells, "cand2_desc", "次选");
            set(&mut cells, "cand1_score", c1s);
            set(&mut cells, "cand2_score", "100.00");
            set(&mut cells, "cand1_n", "64");
            set(&mut cells, "chosen_idx", "0");
            set(&mut cells, "chosen_desc", "首选");
            set(&mut cells, "total_luck", total);
            set(&mut cells, "turn_delta", delta);
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        out
    }

    /// 解析 + 锚点 + coverage + 链推断 + luck 聚合
    #[test]
    fn test_parse_decisions() {
        let csv = make_csv(&[
            // 同 file 两行连续 = 链长 2（RamenSelect → Train）
            ("f1.json", "5", "0", "RamenSelect", "calc", "", "ramen", "200.00", "10.00", "1190.30"),
            ("f1.json", "5", "1", "Train", "calc", "", "train", "200.00", "-5.00", "-928.60"),
            // skip 行（独立 file，链长 1，只进 coverage）
            ("f2.json", "6", "0", "Begin", "skip", "event", "", "", "", ""),
            // no_emit 行
            ("f3.json", "7", "0", "Begin", "no_emit", "no_decision", "", "", "", ""),
            // 另一回合（链长 1）
            ("f4.json", "8", "0", "Train", "calc", "", "train", "300.00", "2.00", "-612.80"),
        ]);
        let r = parse(Some(&csv)).unwrap();
        println!(
            "rows={} skip={:?} no_emit={} series={} top_gain={:?} top_loss={:?} stats={:?}",
            r.rows.len(),
            r.coverage.skip.by_reason,
            r.coverage.no_emit,
            r.luck.series.len(),
            r.luck.top_gain.iter().map(|t| (t.turn, t.delta)).collect::<Vec<_>>(),
            r.luck.top_loss.iter().map(|t| (t.turn, t.delta)).collect::<Vec<_>>(),
            r.luck.raw_delta_stats
        );
        assert_eq!(r.rows.len(), 3, "只保留 calc 行");
        // 链推断：f1 两行 chain_len=2
        assert_eq!(r.rows[0].chain_len, 2);
        assert_eq!(r.rows[0].step, 0);
        assert_eq!(r.rows[1].chain_len, 2);
        assert_eq!(r.rows[1].step, 1);
        assert_eq!(r.rows[2].chain_len, 1);
        // 候选与 gap
        assert_eq!(r.rows[0].candidates.len(), 2);
        assert_eq!(r.rows[0].candidates[0].gap_to_best, Some(0.0));
        assert_eq!(r.rows[0].candidates[1].gap_to_best, Some(100.0));
        // coverage
        assert_eq!(r.coverage.skip.by_reason.get("event"), Some(&1));
        assert_eq!(r.coverage.no_emit, 1);
        // luck：回合合计 turn5 = 1190.30 + (-928.60)
        let t5 = r.luck.top_gain.iter().find(|t| t.turn == 5).unwrap();
        println!("turn5 合计: {}（段 {:?}）", t5.delta, t5.segments);
        assert!((t5.delta - 261.7).abs() < 1e-6);
        assert_eq!(t5.segments.len(), 2);
        // top_gain 排序：turn5 (261.7) > turn8 (-612.8)
        assert_eq!(r.luck.top_gain[0].turn, 5);
        assert_eq!(r.luck.top_loss[0].turn, 8);
        // series：3 条 calc 行都有 total_luck
        assert_eq!(r.luck.series.len(), 3);
        // stats：3 段
        assert_eq!(r.luck.raw_delta_stats.n, 3);
    }

    /// None / 空文本 → 空结果
    #[test]
    fn test_parse_empty() {
        let r = parse(None).unwrap();
        println!("None → rows={} series={}", r.rows.len(), r.luck.series.len());
        assert!(r.rows.is_empty() && r.luck.series.is_empty());
        let r = parse(Some("   ")).unwrap();
        println!("空白 → rows={}", r.rows.len());
        assert!(r.rows.is_empty());
    }
}
