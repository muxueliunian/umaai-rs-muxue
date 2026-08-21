//! 决策日志（开发调参格式，非协议格式）
//!
//! 与 [`DecisionInfo`](crate::output::decision::DecisionInfo) 的分工（计划 §4）：
//!
//! | 类型 | 记录对象 | 消费方 | 粒度 | 演进 |
//! |---|---|---|---|---|
//! | `DecisionInfo` | 决策结果 + 概要理由（协议格式） | 下游协议 | 每次决策，schema 稳定 | 冻结 |
//! | `DecisionLog` | 决策者怎么想的：候选、选中动作、耗时 | 开发调参 | rollout 成千上万条/局，默认关闭 | 随意演进 |
//!
//! 本模块是**开发格式**：CSV 落盘到 `logs/`，字段可随时增删；
//! `score_breakdown` 为预留列（手写策略接入后填充各评分维度分解）。

use std::{fmt::Write as _, path::Path};

use anyhow::Result;
use fs_err as fs;

/// 一次决策的记录（决策日志一行）
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionLogRow {
    /// 本局种子（第 i 局 = 基础种子 + i）
    pub seed: u64,
    /// 回合（0-based，与 `Game::turn()` 一致）
    pub turn: i32,
    /// 决策阶段（RamenSelect / SpecialSelect / Train / RegionSelect / Event）
    pub stage: String,
    /// 候选动作数
    pub candidates: usize,
    /// 选中的候选索引
    pub action_index: usize,
    /// 选中动作的展示文本
    pub action_desc: String,
    /// 决策耗时（微秒）
    pub elapsed_us: u64,
    /// 各候选评分分解（手写策略填充；随机基线为空）
    pub score_breakdown: Option<String>
}

/// CSV 列名（与 [`DecisionLogRow`] 字段一一对应）
const CSV_HEADER: &str = "seed,turn,stage,candidates,action_index,action_desc,elapsed_us,score_breakdown";

/// 决策日志集合（每次决策追加一行）
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionLog {
    /// 全部决策记录
    pub rows: Vec<DecisionLogRow>
}

impl DecisionLog {
    /// 创建一个空的决策日志
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一条决策记录
    pub fn record(&mut self, row: DecisionLogRow) {
        self.rows.push(row);
    }

    /// 序列化为 CSV 文本（含表头）
    pub fn to_csv(&self) -> String {
        let mut out = String::from(CSV_HEADER);
        out.push('\n');
        for row in &self.rows {
            let _ = writeln!(out, "{}", row.to_csv_row());
        }
        out
    }

    /// 落盘为 CSV（自动补目录；父目录不存在时创建）
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, self.to_csv())?;
        Ok(())
    }
}

impl DecisionLogRow {
    /// 序列化为 CSV 行（不含换行）
    pub fn to_csv_row(&self) -> String {
        let mut cols = vec![
            self.seed.to_string(),
            self.turn.to_string(),
            csv_escape(&self.stage),
            self.candidates.to_string(),
            self.action_index.to_string(),
            csv_escape(&self.action_desc),
            self.elapsed_us.to_string(),
        ];
        let breakdown = self.score_breakdown.as_deref().map(csv_escape).unwrap_or_default();
        cols.push(breakdown);
        cols.join(",")
    }
}

/// CSV 字段转义：含逗号/引号/换行时用双引号包裹，内部 `"` 翻倍
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::get_workspace_root;

    #[test]
    fn test_csv_escape() {
        let plain = csv_escape("不吃面");
        println!("普通字段: {plain:?}");
        assert_eq!(plain, "不吃面");

        let with_comma = csv_escape("吃面/札幌, 速度");
        println!("含逗号字段: {with_comma:?}");
        assert_eq!(with_comma, "\"吃面/札幌, 速度\"");

        let with_quote = csv_escape("说\"你好\"");
        println!("含引号字段: {with_quote:?}");
        assert_eq!(with_quote, "\"说\"\"你好\"\"\"");

        let with_newline = csv_escape("第一行\n第二行");
        println!("含换行字段: {with_newline:?}");
        assert_eq!(with_newline, "\"第一行\n第二行\"");
    }

    #[test]
    fn test_csv_row_and_header() {
        let row = DecisionLogRow {
            seed: 42,
            turn: 5,
            stage: "RamenSelect".into(),
            candidates: 4,
            action_index: 2,
            action_desc: "吃面/新潟, 速度".into(),
            elapsed_us: 123,
            score_breakdown: None
        };
        let line = row.to_csv_row();
        println!("单行 CSV: {line}");
        assert_eq!(line, "42,5,RamenSelect,4,2,\"吃面/新潟, 速度\",123,");

        let mut log = DecisionLog::new();
        log.record(row);
        let csv = log.to_csv();
        println!("完整 CSV:\n{csv}");
        assert!(csv.starts_with(CSV_HEADER));
        assert_eq!(csv.lines().count(), 2);
    }

    #[test]
    fn test_empty_log() {
        let log = DecisionLog::new();
        let csv = log.to_csv();
        println!("空日志 CSV:\n{csv}");
        assert_eq!(csv, format!("{CSV_HEADER}\n"));
        assert_eq!(log.rows.len(), 0);
    }

    #[test]
    fn test_save_to_roundtrip() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        let path = workspace_root.join("logs/test_decision_log_tmp.csv");

        let mut log = DecisionLog::new();
        log.record(DecisionLogRow {
            seed: 1,
            turn: 0,
            stage: "Train".into(),
            candidates: 7,
            action_index: 0,
            action_desc: "速度训练".into(),
            elapsed_us: 5,
            score_breakdown: Some("speed=100".into())
        });
        log.save_to(&path)?;

        let content = fs::read_to_string(&path)?;
        println!("落盘内容:\n{content}");
        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("speed=100"));

        fs::remove_file(&path)?;
        Ok(())
    }
}
