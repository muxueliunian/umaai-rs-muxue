//! 采样空间基准：在第一代教师数据的**同一分布**上测某个策略的均分
//!
//! # 为什么不能用 `bench_base`
//!
//! `bench_config.toml` 的马娘是 `102601` 美浦波旁，它**不在**
//! [`SamplingSpace::gen1`] 定的 7 马娘名单里，且 `freeRaces = []` 没有自选比赛要求。
//! 计划文档 §2.3 已记明：`51168 / 50833 / 50872 / 51001` 这一系列手写策略基线
//! 全部测自该马娘，**不能用作第一代网络的验收门槛**——网络是在 7 马娘 × 525 种
//! (马娘, 卡组) 上训练的，拿一个训练分布外的马娘去比，结论没有意义。
//!
//! 本 bin 直接遍历 [`SamplingSpace::gen1`] 的全部计划，每个计划跑若干整局，
//! 给出与教师数据同分布的均分。它同时是第一代网络的验收口径：把 `--trainer`
//! 换成网络策略、其余参数不动，两个数字才可比。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p umasim --bin ramen_space_bench -- \
//!     --trainer handwritten --runs-per-plan 8
//! ```

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use rayon::prelude::*;
use umasim::{
    bench::{self, GameOutcome},
    gamedata::init_global_with_config,
    sampler::{DeckPlan, SamplingSpace, gen1_inherit},
    trainer::{LoggingTrainer, RandomTrainer, RecommendedRamenTrainer},
    utils::{get_workspace_root, load_game_config}
};
#[cfg(feature = "onnx")]
use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

/// 基准参数
#[derive(Parser, Debug)]
#[command(about = "在第一代采样空间（7 马娘 × 525 卡组组合）上测策略均分")]
struct BenchArgs {
    /// 策略：`handwritten`（手写规则）/ `random`（随机基线）/ `nn`（ONNX 网络，需 `--model`）
    #[arg(long, default_value = "handwritten")]
    trainer: String,

    /// ONNX 模型路径；`--trainer nn` 时必填
    #[arg(long)]
    model: Option<PathBuf>,

    /// 每个计划跑几局
    #[arg(long, default_value_t = 8)]
    runs_per_plan: u64,

    /// 基础种子。每局的随机世界由 `derive_seed(seed + plan * 1000003, [run_idx])` 决定
    ///
    /// ❗**不要用相邻的基种子跑多批当作独立样本**。`derive_seed` 是「XOR 后
    /// splitmix64」，而 `base ^ r == base + r` 在低位无进位时成立，所以
    /// `--seed 61444 --runs-per-plan 8` 与 `--seed 61445 --runs-per-plan 8`
    /// 会大面积撞上同一批世界：实测三个相邻基种子跑出的 12600 局里只有 5248 个
    /// 唯一世界（3152 个重复 3 次），重复局分数完全相同，白白虚增样本量、低估标准误。
    ///
    /// 正确做法是**固定一个基种子，用 [`BenchArgs::run_offset`] 切分世界空间**：
    /// 同一 `seed` 下不同 `run_idx` 必然给出不同世界（splitmix64 对不同输入单射），
    /// 且跨计划也不会撞（计划间基种子相差 1000003 的倍数，远大于 `run_idx` 的位宽）。
    #[arg(long, default_value_t = 61444)]
    seed: u64,

    /// 局号起点；本次跑 `run_idx ∈ [run_offset, run_offset + runs_per_plan)`
    ///
    /// 用来切出**互不重叠**的世界子集，实现「选择集 / 验收集分离」：
    /// 例如选择集用 `--run-offset 0 --runs-per-plan 1`（525 局，约 1 分钟），
    /// 验收集用 `--run-offset 8 --runs-per-plan 24`（12600 局）。
    /// 两者由构造保证零重叠，因此在选择集上挑 checkpoint 不会污染验收集的无偏性。
    #[arg(long, default_value_t = 0)]
    run_offset: u64,

    /// 只跑前 N 个计划（调试用，默认全跑）
    #[arg(long)]
    plans: Option<usize>,

    /// 把逐局结果写成 CSV
    #[arg(long)]
    csv: Option<PathBuf>,

    /// 关闭网络策略的自选比赛硬守门（纯网络，仅供研究守门能否移除；不作为验收口径）
    #[arg(long)]
    no_race_shield: bool,

    /// `SpecialSelect` 阶段的推理口径：`canonical`（还原到联合决策根，默认）/
    /// `raw`（历史行为，存在训练—部署语义错位）/ `handwritten`（该阶段交给手写，对照组）
    #[arg(long, default_value = "canonical")]
    special_mode: String
}

/// 解析 `--special-mode`（仅 onnx 下有意义）
///
/// # 错误
///
/// 未知取值时报错——静默回退到默认会让对照组静静地变成实验组。
#[cfg(feature = "onnx")]
fn parse_special_mode(s: &str) -> Result<SpecialSelectMode> {
    match s {
        "canonical" => Ok(SpecialSelectMode::Canonical),
        "raw" => Ok(SpecialSelectMode::Raw),
        "handwritten" => Ok(SpecialSelectMode::Handwritten),
        other => bail!("未知 --special-mode: {other}（可选 canonical / raw / handwritten）")
    }
}

/// 一组分数的汇总统计
#[derive(Debug, Clone, Copy)]
struct ScoreStats {
    /// 局数
    games: usize,
    /// 均分
    mean: f64,
    /// 样本标准差（`n-1` 分母）
    stdev: f64,
    /// 均值的标准误
    stderr: f64
}

impl ScoreStats {
    /// 从分数序列汇总
    ///
    /// # 错误
    ///
    /// 序列为空时报错——空集的均分没有意义，静默返回 0 会被误读成「跑了但很差」。
    fn from_scores(scores: &[f64]) -> Result<Self> {
        let games = scores.len();
        if games == 0 {
            bail!("没有可汇总的局");
        }
        let mean = scores.iter().sum::<f64>() / games as f64;
        let stdev = if games < 2 {
            0.0
        } else {
            let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (games - 1) as f64;
            var.sqrt()
        };
        Ok(Self {
            games,
            mean,
            stdev,
            stderr: if games == 0 { 0.0 } else { stdev / (games as f64).sqrt() }
        })
    }
}

/// 单个计划的全部对局结果
struct PlanResult {
    /// 计划下标，用于稳定排序
    plan_index: usize,
    /// 该计划的逐局结果
    outcomes: Vec<GameOutcome>
}

/// 本进程选定的策略；`nn` 变体持有已加载的模型（Arc 共享，不每局重载）
#[derive(Clone)]
enum SelectedTrainer {
    /// 随机基线
    Random,
    /// 手写规则
    Handwritten,
    /// 神经网络策略
    #[cfg(feature = "onnx")]
    Nn(RamenNnTrainer)
}

/// 按命令行构造策略；`nn` 在此处加载一次模型
///
/// # 错误
///
/// 未知策略名、缺少 `--model`、未启用 `onnx` feature，或模型加载失败时报错。
fn select_trainer(args: &BenchArgs) -> Result<SelectedTrainer> {
    match args.trainer.as_str() {
        "random" => Ok(SelectedTrainer::Random),
        "handwritten" => Ok(SelectedTrainer::Handwritten),
        "nn" => {
            #[cfg(feature = "onnx")]
            {
                let path = args
                    .model
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--trainer nn 需要同时给出 --model <onnx 路径>"))?;
                Ok(SelectedTrainer::Nn(
                    RamenNnTrainer::load(path)?
                        .with_race_shield(!args.no_race_shield)
                        .with_special_mode(parse_special_mode(&args.special_mode)?)
                ))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = (&args.model, &args.special_mode);
                bail!("--trainer nn 需要编译 feature onnx（cargo build --release --features onnx --bin ramen_space_bench）")
            }
        }
        other => bail!("未知 trainer: {other}（可选 random / handwritten / nn）")
    }
}

/// 跑一个计划的全部对局
///
/// # 错误
///
/// 任一局报错时报错。
fn run_plan(plan: &DeckPlan, plan_index: usize, args: &BenchArgs, kind: &SelectedTrainer) -> Result<PlanResult> {
    let inherit = gen1_inherit();
    // 每个计划用互不重叠的种子段，避免不同计划共用同一批随机世界
    let base_seed = args.seed.wrapping_add((plan_index as u64).wrapping_mul(1_000_003));
    let mut outcomes = Vec::with_capacity(args.runs_per_plan as usize);
    let run_end = args
        .run_offset
        .checked_add(args.runs_per_plan)
        .context("run_offset + runs_per_plan 溢出 u64")?;
    for run_idx in args.run_offset..run_end {
        let outcome = match kind {
            SelectedTrainer::Random => {
                let t = LoggingTrainer::new(RandomTrainer, base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            SelectedTrainer::Handwritten => {
                let t = LoggingTrainer::new(RecommendedRamenTrainer::new(), base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            #[cfg(feature = "onnx")]
            SelectedTrainer::Nn(nn) => {
                let t = LoggingTrainer::new(nn.clone(), base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
        };
        outcomes.push(outcome);
    }
    Ok(PlanResult { plan_index, outcomes })
}

/// 按某个键分组汇总并打印
///
/// # 错误
///
/// 任一组为空时报错。
fn print_grouped(title: &str, groups: &BTreeMap<String, Vec<f64>>) -> Result<()> {
    println!("\n{title}");
    println!("  {:<28} {:>6} {:>10} {:>9} {:>8}", "分组", "局数", "均分", "标准差", "标准误");
    for (key, scores) in groups {
        let s = ScoreStats::from_scores(scores)?;
        println!("  {:<28} {:>6} {:>10.0} {:>9.0} {:>8.1}", key, s.games, s.mean, s.stdev, s.stderr);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = BenchArgs::parse();

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    init_global_with_config(&load_game_config()?)?;
    let kind = select_trainer(&args)?;

    let space = SamplingSpace::gen1()?;
    let all_plans = space.plans();
    let plans = match args.plans {
        Some(n) => &all_plans[..n.min(all_plans.len())],
        None => all_plans
    };
    println!(
        "采样空间基准：{} 个计划 × {} 局（局号 {}..{}）= {} 局，策略 = {}",
        plans.len(),
        args.runs_per_plan,
        args.run_offset,
        args.run_offset + args.runs_per_plan,
        plans.len() as u64 * args.runs_per_plan,
        if args.trainer == "nn" {
            format!(
                "nn[{}]{}",
                args.special_mode,
                if args.no_race_shield { "(无守门)" } else { "" }
            )
        } else {
            args.trainer.clone()
        }
    );

    println!("  基种子 {}", args.seed);

    let start = std::time::Instant::now();
    let mut results: Vec<PlanResult> = plans
        .par_iter()
        .enumerate()
        .map(|(i, plan)| run_plan(plan, i, &args, &kind))
        .collect::<Result<Vec<_>>>()?;
    results.sort_by_key(|r| r.plan_index);
    let elapsed = start.elapsed().as_secs_f64();

    let mut all: Vec<f64> = Vec::new();
    let mut by_shape: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut by_uma: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut free_race_fail = 0usize;
    let mut rows: Vec<Vec<String>> = Vec::new();
    for r in &results {
        let plan = &plans[r.plan_index];
        for o in &r.outcomes {
            let score = f64::from(o.score);
            all.push(score);
            by_shape.entry(plan.shape.to_string()).or_default().push(score);
            by_uma.entry(format!("{}", plan.uma)).or_default().push(score);
            if !o.free_race_ok {
                free_race_fail += 1;
            }
            if args.csv.is_some() {
                rows.push(bench::outcome_to_row(plan.shape, o));
            }
        }
    }

    let overall = ScoreStats::from_scores(&all)?;
    print_grouped("按卡组构成", &by_shape)?;
    print_grouped("按马娘", &by_uma)?;
    println!("\n合计");
    println!("  局数        {}", overall.games);
    println!("  均分        {:.0}", overall.mean);
    println!("  标准差      {:.0}", overall.stdev);
    println!("  均值标准误  {:.1}", overall.stderr);
    println!("  自选比赛未达标  {} 局（{:.2}%）", free_race_fail, 100.0 * free_race_fail as f64 / all.len() as f64);
    println!("  耗时        {elapsed:.1} s");

    if let Some(path) = &args.csv {
        bench::write_csv(path, &bench::RESULTS_HEADER, &rows)?;
        println!("  逐局 CSV    {}", path.display());
    }
    Ok(())
}
