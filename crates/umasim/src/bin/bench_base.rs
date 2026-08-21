//! 基准测试 bin：固定种子批量跑批 + 决策日志 + 基线分布
//!
//! 对应手写策略计划 §8 主线第 1 步「先立地基」：没有基线无法量化改进。
//! 本 bin 产出 RandomTrainer 的基线分布（分数/PT/RMJ/耗时），
//! 并可选落盘每局决策轨迹（开发调参格式，见 `output::decision_log`）。
//!
//! 运行设施（固定种子双 RNG、单局运行、统计、CSV）复用 [`umasim::bench`]。
//!
//! # 用法（Release）
//!
//! ```text
//! cargo run --release --bin bench_base -- [--runs N] [--seed S] [--log] [--out DIR]
//! ```
//!
//! 参数缺省时读取 workspace 根目录 `bench_config.toml`（不存在则用内置默认，与
//! `bench_config.toml` 一致）。固定种子下结果完全可复现：
//! 决策 RNG 与规则层 RNG（`RamenGame::set_internal_rng`）分别由 seed 派生。
//!
//! # 产出（默认 `logs/`）
//!
//! - `bench_base_results.csv`：每局一行（seed、分数、rank、五维、PT、RMJ、吃面数、耗时）
//! - `bench_base_decision_<seed>.csv`：仅 `--log` 时，每局一份决策轨迹
//! - 汇总打印：分数分布（mean/median/min/max/std）、RMJ 成功年数、按阶段分组的决策耗时、吞吐

use std::time::Instant;

use anyhow::{Context, Result};
use lexopt::Arg;
use serde::Deserialize;
use umasim::bench;
use umasim::game::InheritInfo;
use umasim::gamedata::init_global_with_config;
use umasim::output::decision_log::DecisionLogRow;
use umasim::trainer::{LoggingTrainer, RamenHandwrittenTrainer, RandomTrainer};
use umasim::utils::{get_workspace_root, load_game_config};

/// bench_config.toml 的配置项（CLI 参数可覆盖同名项）
#[derive(Debug, Clone, Deserialize)]
struct BenchConfig {
    /// 马娘 ID
    uma: u32,
    /// 卡组（6 张支援卡 ID）
    cards: [u32; 6],
    /// 种马蓝因子个数
    blue_count: [i32; 5],
    /// 种马额外属性
    extra_count: [i32; 6],
    /// 批量局数
    runs: usize,
    /// 基础种子（第 i 局 = seed + i）
    seed: u64,
    /// 输出目录（相对 workspace 根）
    out_dir: String,
    /// 是否落盘决策日志
    decision_log: bool,
    /// 训练员: "random"（基线）| "handwritten"（手写策略）
    trainer: String,
}

/// 内置默认值（与 bench_config.toml 保持一致；文件缺失时使用）
impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            uma: 102601,
            cards: [302424, 302894, 303044, 302924, 303024, 303054],
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40],
            runs: 20,
            seed: 42,
            out_dir: "logs".to_string(),
            decision_log: false,
            trainer: "random".to_string(),
        }
    }
}

/// results CSV 表头
const RESULTS_HEADER: [&str; 13] = [
    "seed",
    "score",
    "rank",
    "speed",
    "stamina",
    "power",
    "guts",
    "wisdom",
    "skill_pt",
    "scenario_pt",
    "rmj_ok",
    "eat_count",
    "elapsed_ms",
];

/// 解析 CLI 参数（`--key value` 或 `--key=value`），覆盖 bench 配置
fn apply_cli(mut cfg: BenchConfig) -> Result<BenchConfig> {
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("runs") => cfg.runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("seed") => cfg.seed = bench::parse_value(&mut parser, "seed")?,
            Arg::Long("log") => cfg.decision_log = true,
            Arg::Long("out") => cfg.out_dir = bench::parse_value(&mut parser, "out")?,
            Arg::Long("trainer") => cfg.trainer = bench::parse_value(&mut parser, "trainer")?,
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: bench_base [--runs N] [--seed S] [--log] [--out DIR] [--trainer random|handwritten]\n\
                     缺省参数读取 workspace 根 bench_config.toml"
                );
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("未知参数: {other:?}（可用 --help 查看用法）");
            }
        }
    }
    Ok(cfg)
}

/// 读取 bench_config.toml（workspace 根）；缺失时用内置默认并提示
fn load_bench_config(workspace_root: &std::path::Path) -> Result<BenchConfig> {
    let path = workspace_root.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 bench_config.toml 失败: {}", path.display()))?;
        let cfg: BenchConfig =
            toml::from_str(&text).with_context(|| format!("解析 bench_config.toml 失败: {}", path.display()))?;
        Ok(cfg)
    } else {
        println!("提示: 未找到 bench_config.toml，使用内置默认参数");
        Ok(BenchConfig::default())
    }
}

/// 单局结果转 CSV 行（不含表头）
fn outcome_to_row(outcome: &bench::GameOutcome) -> Vec<String> {
    vec![
        outcome.seed.to_string(),
        outcome.score.to_string(),
        outcome.rank.clone(),
        outcome.five_status[0].to_string(),
        outcome.five_status[1].to_string(),
        outcome.five_status[2].to_string(),
        outcome.five_status[3].to_string(),
        outcome.five_status[4].to_string(),
        outcome.skill_pt.to_string(),
        outcome.scenario_pt.to_string(),
        outcome.rmj_ok.to_string(),
        outcome.eat_count.to_string(),
        format!("{:.3}", outcome.elapsed_ms),
    ]
}

/// 按决策阶段分组统计耗时（mean us / max us / 次数），按阶段名排序
fn summarize_decision_times(rows: &[DecisionLogRow]) -> Vec<(String, f64, u64, usize)> {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<String, (u128, u64, usize)> = BTreeMap::new();
    for r in rows {
        let e = acc.entry(r.stage.clone()).or_insert((0, 0, 0));
        e.0 += r.elapsed_us as u128;
        e.1 = e.1.max(r.elapsed_us);
        e.2 += 1;
    }
    acc.into_iter()
        .map(|(k, (sum, max, n))| (k, sum as f64 / n.max(1) as f64, max, n))
        .collect()
}

fn main() -> Result<()> {
    // 切换到 workspace 根（bench_config.toml / logs / gamedata 相对路径依赖）
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;

    let cfg = apply_cli(load_bench_config(&workspace_root)?)?;

    // 初始化全局数据（注入用户可调项：race_grades / mcts_turn_bonus 等）
    let game_config = load_game_config()?;
    init_global_with_config(&game_config)?;

    let out_dir = workspace_root.join(&cfg.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let inherit = InheritInfo {
        blue_count: cfg.blue_count,
        extra_count: cfg.extra_count,
    };

    println!(
        "===== bench_base: uma={} cards={:?} runs={} base_seed={} trainer={} =====",
        cfg.uma, cfg.cards, cfg.runs, cfg.seed, cfg.trainer
    );

    let mut results = Vec::with_capacity(cfg.runs);
    let mut all_rows: Vec<DecisionLogRow> = Vec::new();
    for i in 0..cfg.runs {
        let seed = cfg.seed + i as u64;
        // 构造训练员（LoggingTrainer 包装；决策日志默认开启，由 --log 决定是否落盘）
        let (outcome, log) = match cfg.trainer.as_str() {
            "random" => {
                let trainer = LoggingTrainer::new(RandomTrainer, seed);
                let outcome = bench::run_seeded(cfg.uma, &cfg.cards, &inherit, seed, &trainer)?;
                (outcome, trainer.take_records())
            }
            "handwritten" => {
                let trainer = LoggingTrainer::new(RamenHandwrittenTrainer::new(), seed);
                let outcome = bench::run_seeded(cfg.uma, &cfg.cards, &inherit, seed, &trainer)?;
                (outcome, trainer.take_records())
            }
            other => anyhow::bail!("未知 trainer: {other}（可选 random / handwritten）"),
        };
        println!(
            "  [#{:02}] seed={} score={} ({}) PT={} RMJ={}/3 耗时={:.3}ms",
            i + 1,
            outcome.seed,
            outcome.score,
            outcome.rank,
            outcome.scenario_pt,
            outcome.rmj_ok,
            outcome.elapsed_ms,
        );
        if cfg.decision_log {
            log.save_to(&out_dir.join(format!("bench_base_decision_{seed}.csv")))?;
        }
        all_rows.extend(log.rows);
        results.push(outcome);
    }

    // ===== 落盘结果 CSV =====
    let results_path = out_dir.join("bench_base_results.csv");
    let rows: Vec<Vec<String>> = results.iter().map(outcome_to_row).collect();
    bench::write_csv(&results_path, &RESULTS_HEADER, &rows)?;
    println!("\n结果已写入: {}", results_path.display());

    // ===== 汇总 =====
    let scores: Vec<f64> = results.iter().map(|r| r.score as f64).collect();
    let stats = bench::summarize(&scores);
    let rmj_mean = results.iter().map(|r| r.rmj_ok as f64).sum::<f64>() / results.len().max(1) as f64;
    let elapsed_ms: Vec<f64> = results.iter().map(|r| r.elapsed_ms).collect();
    let total_ms = elapsed_ms.iter().sum::<f64>();
    let throughput = cfg.runs as f64 / (total_ms / 1000.0).max(1e-9);
    let elapsed_stats = bench::summarize(&elapsed_ms);

    println!("\n===== 汇总 (runs={}, base_seed={}) =====", cfg.runs, cfg.seed);
    println!(
        "分数: mean={:.0} median={:.0} min={:.0} max={:.0} std={:.0}",
        stats.mean, stats.median, stats.min, stats.max, stats.std
    );
    println!("RMJ 成功年数: 平均 {rmj_mean:.2}/3");
    println!("决策耗时 (mean/max us, 次数):");
    for (stage, m, max_us, n) in summarize_decision_times(&all_rows) {
        println!("  {stage:<14} {m:>8.1} {max_us:>8} {n:>6}");
    }
    println!(
        "整局耗时: mean {:.3}ms, max {:.3}ms, 吞吐 {:.1} 局/s",
        elapsed_stats.mean, elapsed_stats.max, throughput
    );
    Ok(())
}
