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
//! ❗`ramen_region_strategy` 由本 bin 强制为 `all`（与 `bench_base`、
//! `ramen_teacher_collect` 一致），不跟随 `game_config.toml`。跟随 toml 会让
//! 有人为手动模式改成 `fixed` 时，基准静默换成另一个分布，此前记下的全部均分
//! 基线一并作废，而输出里看不出任何区别。
//!
//! # 分布外模式
//!
//! 给出 `--shape` 后切换到 [`SamplingSpace::custom`]：只枚举该构成，并可用
//! `--extra-card` 往卡池里补卡。用途是检验网络对**未训练卡组流派**的泛化，
//! 例如「2 速 1 耐 2 智」——池内只有一张智力卡，必须补一张才组得出来。
//!
//! ❗分布外模式的分数**不能**与默认口径的数字直接比较：卡组空间换了，
//! 计划数变了，逐计划的种子段也随之不同。要下结论必须在同一模式下
//! 跑手写基线做配对参照。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p umasim --bin ramen_space_bench -- \
//!     --trainer handwritten --runs-per-plan 8
//!
//! cargo run --release -p umasim --features onnx --bin ramen_space_bench -- \
//!     --shape 2,1,0,0,2 --extra-card 303064 --trainer nn --model model.onnx
//! ```

use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rayon::prelude::*;
use umasim::{
    bench::{self, GameOutcome},
    gamedata::{RamenRegionStrategy, init_global_with_config},
    sampler::{DeckPlan, DeckShape, SamplingSpace, gen1_inherit},
    trainer::{LoggingTrainer, RandomTrainer, RecommendedRamenTrainer},
    utils::{get_workspace_root, load_game_config}
};
#[cfg(feature = "onnx")]
use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

/// 基准参数
#[derive(Parser, Debug)]
#[command(about = "在第一代采样空间（7 马娘 × 525 卡组组合）上测策略均分；--shape 可切到分布外空间")]
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
    special_mode: String,

    /// 卡组构成 `速,耐,力,根,智`，五项合计为 5（友人卡固定 1 张，不计入）
    ///
    /// 给出即切到**分布外**空间：只枚举这一种构成，用于检验网络对未训练卡组流派
    /// 的泛化。不给时用第一代的 3 种构成，与教师数据同分布。
    #[arg(long)]
    shape: Option<String>,

    /// 追加进卡池的支援卡 idrank（6 位 = 5 位卡 ID + 突破等级），可重复给出
    ///
    /// 只在有 `--shape` 时允许：默认口径必须锁死在训练分布的 11 张卡上，
    /// 否则「同分布均分」这个名字就不成立了。
    #[arg(long)]
    extra_card: Vec<u32>
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

/// 解析 `--shape` 的 `速,耐,力,根,智`
///
/// # 错误
///
/// 项数不是 5、某项不是非负整数，或五项合计不为 5 时报错。合计不为 5 必须报错
/// 而不是补齐：拉面杯的普通卡位恒为 5 张，猜用户想补哪一类只会静默跑错构成。
fn parse_shape(text: &str) -> Result<[usize; 5]> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    ensure!(parts.len() == 5, "--shape 需要 5 个数字（速,耐,力,根,智），实得 {}", parts.len());
    let mut counts = [0usize; 5];
    for (i, part) in parts.iter().enumerate() {
        counts[i] = part
            .parse::<usize>()
            .with_context(|| format!("--shape 第 {} 项 `{part}` 不是非负整数", i + 1))?;
    }
    let total: usize = counts.iter().sum();
    ensure!(total == 5, "--shape 五项合计必须为 5（友人卡固定 1 张不计入），实得 {total}");
    Ok(counts)
}

/// 把构成计数格式化成 `2速1耐2智1友`，与 `GEN1_SHAPES` 的命名习惯一致
fn format_shape_name(counts: &[usize; 5]) -> String {
    const TYPE_NAMES: [&str; 5] = ["速", "耐", "力", "根", "智"];
    let mut name = String::new();
    for (i, &n) in counts.iter().enumerate() {
        if n > 0 {
            name.push_str(&n.to_string());
            name.push_str(TYPE_NAMES[i]);
        }
    }
    name.push_str("1友");
    name
}

/// 按命令行构造采样空间：默认第一代，给了 `--shape` 则走分布外
///
/// # 错误
///
/// `--shape` 解析失败、只给 `--extra-card` 不给 `--shape`，或空间枚举失败时报错。
fn build_space(args: &BenchArgs) -> Result<SamplingSpace> {
    let Some(text) = &args.shape else {
        ensure!(
            args.extra_card.is_empty(),
            "--extra-card 必须与 --shape 同用：默认口径要与教师数据同分布，不能私自扩卡池"
        );
        return SamplingSpace::gen1();
    };
    let counts = parse_shape(text)?;
    // `DeckShape::name` 要求 'static。CLI 参数活到进程结束，泄漏一个短字符串
    // 换来 CSV 与分组统计里显示真实构成名，比塞一个占位常量更有用。
    let name: &'static str = Box::leak(format_shape_name(&counts).into_boxed_str());
    SamplingSpace::custom(&args.extra_card, DeckShape { counts, name })
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
    // Y3 地区选择必须交回策略（All 枚举），与 `bench_base` / `ramen_teacher_collect`
    // 同一条前提：教师数据就是在 all 下采的，基准若跟着 toml 走 fixed，测的不再是
    // 同一个分布，而此前记下的全部均分基线会被静默作废。
    //
    // 只改 strategy，不碰 `ramen_region_fixed`：后者仅在 `strategy == Fixed` 时被读
    // （`action.rs::region_select_combos` 与 `game.rs` 的 Y3 分支各一处，都带该守卫），
    // 在 All 下清空它是空操作，写出来只会让人误以为它参与了结果。
    let mut game_config = load_game_config()?;
    if game_config.ramen_region_strategy != RamenRegionStrategy::All {
        println!(
            "已将 ramen_region_strategy 从 {:?} 强制改为 All（基准须与教师数据同分布）",
            game_config.ramen_region_strategy
        );
        game_config.ramen_region_strategy = RamenRegionStrategy::All;
    }
    init_global_with_config(&game_config)?;
    let kind = select_trainer(&args)?;

    let space = build_space(&args)?;
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
    match &args.shape {
        None => println!("  空间   第一代（与教师数据同分布）"),
        Some(text) => println!(
            "  空间   ❗分布外：构成 {}，追加卡 {:?}——分数不可与默认口径直接比较",
            text, args.extra_card
        )
    }

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
