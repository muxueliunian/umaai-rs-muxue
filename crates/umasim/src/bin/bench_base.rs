//! 基准测试 bin：固定种子批量跑批 + 决策日志 + 基线分布
//!
//! 对应手写策略计划 §8 主线第 1 步「先立地基」：没有基线无法量化改进。
//! 本 bin 产出 RandomTrainer 的基线分布（分数/PT/RMJ/耗时），
//! 并可选落盘每局决策轨迹（开发调参格式，见 `output::decision_log`）。
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
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use umasim::game::ramen::RamenGame;
use umasim::game::{Game, InheritInfo, Trainer};
use umasim::gamedata::{GAMECONSTANTS, init_global_with_config};
use umasim::global;
use umasim::output::decision_log::{DecisionLog, DecisionLogRow};
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

/// 单局结果（results CSV 一行）
struct GameResult {
    /// 本局种子
    seed: u64,
    /// 结算评分
    score: i32,
    /// 评分等级
    rank: String,
    /// 五维终值
    five_status: [i32; 5],
    /// 技能点
    skill_pt: i32,
    /// 剧本 PT
    scenario_pt: i32,
    /// RMJ 成功年数（0-3）
    rmj_ok: usize,
    /// 当年吃面次数
    eat_count: i32,
    /// 整局耗时（毫秒，浮点）
    elapsed_ms: f64,
}

impl GameResult {
    /// 序列化为 CSV 行（不含表头）
    fn to_csv_row(&self) -> String {
        let fs = self.five_status;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.3}",
            self.seed,
            self.score,
            csv_escape(&self.rank),
            fs[0],
            fs[1],
            fs[2],
            fs[3],
            fs[4],
            self.skill_pt,
            self.scenario_pt,
            self.rmj_ok,
            self.eat_count,
            self.elapsed_ms,
        )
    }
}

/// results CSV 表头
const RESULTS_HEADER: &str =
    "seed,score,rank,speed,stamina,power,guts,wisdom,skill_pt,scenario_pt,rmj_ok,eat_count,elapsed_ms";

/// CSV 字段转义（与 `output::decision_log` 同规则）
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// 解析 CLI 参数（`--key value` 或 `--flag`），覆盖 bench 配置
fn apply_cli(mut cfg: BenchConfig, args: &[String]) -> Result<BenchConfig> {
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--runs" => {
                cfg.runs = parse_arg(args, &mut i, "--runs")?;
            }
            "--seed" => {
                cfg.seed = parse_arg(args, &mut i, "--seed")?;
            }
            "--log" => {
                cfg.decision_log = true;
            }
            "--out" => {
                cfg.out_dir = parse_arg(args, &mut i, "--out")?;
            }
            "--trainer" => {
                cfg.trainer = parse_arg(args, &mut i, "--trainer")?;
            }
            "--help" | "-h" => {
                println!(
                    "用法: bench_base [--runs N] [--seed S] [--log] [--out DIR] [--trainer random|handwritten]\n\
                     缺省参数读取 workspace 根 bench_config.toml"
                );
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("未知参数: {other}（可用 --help 查看用法）");
            }
        }
        i += 1;
    }
    Ok(cfg)
}

/// 读取 `--key value` 的 value（并推进索引）
fn parse_arg<T: std::str::FromStr>(args: &[String], i: &mut usize, key: &str) -> Result<T> {
    *i += 1;
    let val = args
        .get(*i)
        .ok_or_else(|| anyhow::anyhow!("参数 {key} 缺少值"))?;
    val.parse()
        .map_err(|_| anyhow::anyhow!("参数 {key} 的值无效: {val}"))
}

/// 读取 bench_config.toml（workspace 根）；缺失时用内置默认并提示
fn load_bench_config(workspace_root: &std::path::Path) -> Result<BenchConfig> {
    let path = workspace_root.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 bench_config.toml 失败: {}", path.display()))?;
        let cfg: BenchConfig = toml::from_str(&text)
            .with_context(|| format!("解析 bench_config.toml 失败: {}", path.display()))?;
        Ok(cfg)
    } else {
        println!("提示: 未找到 bench_config.toml，使用内置默认参数");
        Ok(BenchConfig::default())
    }
}

/// 跑一局（固定 seed），返回结果与决策日志
fn run_once<T: Trainer<RamenGame>>(
    cfg: &BenchConfig,
    seed: u64,
    inherit: &InheritInfo,
    trainer: &LoggingTrainer<T>,
) -> Result<(GameResult, DecisionLog)> {
    // 决策 RNG 与规则层 RNG 从同一 seed 分裂派生，两轮同 seed 跑批结果完全一致
    let mut decision_rng = StdRng::seed_from_u64(seed);
    let rule_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);

    let mut game = RamenGame::newgame(cfg.uma, &cfg.cards, inherit.clone())?;
    game.set_internal_rng(rule_rng);

    let start = Instant::now();
    game.run_full_game(trainer, &mut decision_rng)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let score = game.uma.calc_score();
    let rank = global!(GAMECONSTANTS).get_rank_name(score);
    let result = GameResult {
        seed,
        score,
        rank,
        five_status: game.uma.five_status,
        skill_pt: game.uma.skill_pt,
        scenario_pt: game.ramen.scenario_pt,
        rmj_ok: game.ramen.rmj_results.iter().filter(|&&ok| ok).count(),
        eat_count: game.ramen.eat_count,
        elapsed_ms,
    };
    Ok((result, trainer.take_records()))
}

/// 汇总：基本统计（min/max/mean/median/std）
fn summarize(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let variance = values
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    (min, max, mean, median, std)
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

    let args: Vec<String> = std::env::args().collect();
    let cfg = apply_cli(load_bench_config(&workspace_root)?, &args)?;

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
        let (result, log) = match cfg.trainer.as_str() {
            "random" => run_once(
                &cfg,
                seed,
                &inherit,
                &LoggingTrainer::new(RandomTrainer, seed),
            )?,
            "handwritten" => run_once(
                &cfg,
                seed,
                &inherit,
                &LoggingTrainer::new(RamenHandwrittenTrainer::new(), seed),
            )?,
            other => anyhow::bail!("未知 trainer: {other}（可选 random / handwritten）"),
        };
        println!(
            "  [#{:02}] seed={} score={} ({}) PT={} RMJ={}/3 耗时={:.3}ms",
            i + 1,
            result.seed,
            result.score,
            result.rank,
            result.scenario_pt,
            result.rmj_ok,
            result.elapsed_ms,
        );
        if cfg.decision_log {
            log.save_to(&out_dir.join(format!("bench_base_decision_{seed}.csv")))?;
        }
        all_rows.extend(log.rows);
        results.push(result);
    }

    // ===== 落盘结果 CSV =====
    let results_path = out_dir.join("bench_base_results.csv");
    let mut csv = String::from(RESULTS_HEADER);
    csv.push('\n');
    for r in &results {
        csv.push_str(&r.to_csv_row());
        csv.push('\n');
    }
    std::fs::write(&results_path, csv)?;
    println!("\n结果已写入: {}", results_path.display());

    // ===== 汇总 =====
    let scores: Vec<f64> = results.iter().map(|r| r.score as f64).collect();
    let (min, max, mean, median, std) = summarize(&scores);
    let rmj_mean =
        results.iter().map(|r| r.rmj_ok as f64).sum::<f64>() / results.len().max(1) as f64;
    let elapsed_ms: Vec<f64> = results.iter().map(|r| r.elapsed_ms).collect();
    let total_ms = elapsed_ms.iter().sum::<f64>();
    let throughput_ind = cfg.runs as f64 / (total_ms / 1000.0).max(1e-9);

    println!("\n===== 汇总 (runs={}, base_seed={}) =====", cfg.runs, cfg.seed);
    println!(
        "分数: mean={mean:.0} median={median:.0} min={min:.0} max={max:.0} std={std:.0}"
    );
    println!("RMJ 成功年数: 平均 {rmj_mean:.2}/3");
    println!("决策耗时 (mean/max us, 次数):");
    for (stage, m, max_us, n) in summarize_decision_times(&all_rows) {
        println!("  {stage:<14} {m:>8.1} {max_us:>8} {n:>6}");
    }
    println!(
        "整局耗时: mean {:.3}ms, max {:.3}ms, 吞吐 {:.1} 局/s",
        total_ms / cfg.runs.max(1) as f64,
        max_of(&elapsed_ms),
        throughput_ind,
    );
    Ok(())
}

/// 一组 f64 的最大值（空序列返回 0）
fn max_of(values: &[f64]) -> f64 {
    values.iter().cloned().fold(0.0, f64::max)
}