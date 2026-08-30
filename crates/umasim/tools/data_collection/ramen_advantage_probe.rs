//! on-policy 配对 advantage 探针：在**被评估策略自己的轨迹**上测单步优势
//!
//! # 为什么需要它
//!
//! 性能差分恒等式要求状态来自被评估策略的占用分布：
//!
//! ```text
//! J(π) − J(H) = Σ_t E_{s_t ~ d_t^π} [ Q_t^H(s, π(s)) − Q_t^H(s, H(s)) ]
//! ```
//!
//! 训练侧的 `expected_regret` 算在「手写 roll-in + ε 扰动 + 随机截断」的分布上，
//! 既不是 `d^H` 也不是 `d^π`。2026-08-31 已实测到该指标与闭环**反向**：离线
//! regret 全线更好的模型，配对闭环输 287 分。本 bin 提供替代的排序依据。
//!
//! # 做法
//!
//! 用 `--rollin` 指定的策略跑完整局。每个决策点同时问网络与手写要动作：
//! 两者一致则该点对上式贡献恒为 0，直接跳过不搜索；不一致的点进蓄水池，
//! 每局等概率留下 `--points-per-game` 个。局末对留下的点做**配对 rollout**
//! （两个动作共享同一张 CRN 种子表，rollout 基策为手写），得到
//!
//! ```text
//! Â^H(s, π(s)) = Q̂^H(s, π(s)) − Q̂^H(s, H(s))
//! ```
//!
//! 单局估计 = `分歧点总数 / 抽样点数 × Σ Â`，是该局 advantage 总和的无偏估计
//! （蓄水池抽样即不放回简单随机抽样）。
//!
//! # 两种 roll-in 的用法
//!
//! - `--rollin nn`：状态来自网络自己的轨迹，估计的就是 `J(π) − J(H)`，
//!   可直接与 `ramen_space_bench` 的配对闭环差值对照，用来验收探针本身。
//! - `--rollin handwritten`：状态来自手写轨迹，只给出恒等式的**首项**。
//!   两个数字的差就是占用分布错配的大小。
//!
//! # 用法
//!
//! ```text
//! cargo run --release --features onnx -p umasim --bin ramen_advantage_probe -- \
//!     --model saved_models/ramen_v3/model.onnx --points-per-game 8 --rollouts 256
//! ```

use std::{cell::RefCell, collections::BTreeMap, path::PathBuf, rc::Rc};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
#[cfg(feature = "onnx")]
use umasim::trainer::RamenNnTrainer;
use umasim::{
    bench,
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData, init_global_with_config},
    rng::splitmix64,
    sampler::{DeckPlan, SamplingSpace, gen1_inherit},
    search::{FlatSearch, RolloutSeeds, SearchConfig},
    trainer::{LoggingTrainer, RecommendedRamenTrainer, ramen_handwritten_trainer::ramen_effective_stage},
    utils::{get_workspace_root, load_game_config}
};

/// 探针参数
#[derive(Parser, Debug)]
#[command(about = "在策略自己的轨迹上测单步 advantage（配对 rollout，rollout 基策为手写）")]
struct ProbeArgs {
    /// ONNX 模型路径（被评估策略 π）
    #[arg(long)]
    model: PathBuf,

    /// 轨迹由谁驱动：`nn`（on-policy，估计 J(π)−J(H)）/ `handwritten`（只给恒等式首项）
    #[arg(long, default_value = "nn")]
    rollin: String,

    /// 关闭网络策略的自选比赛硬守门（与 `ramen_space_bench` 同名开关一致）
    #[arg(long)]
    no_race_shield: bool,

    /// 每个计划跑几局
    #[arg(long, default_value_t = 1)]
    runs_per_plan: u64,

    /// 基础种子；与 `ramen_space_bench` 同口径，便于和闭环结果对照
    #[arg(long, default_value_t = 61444)]
    seed: u64,

    /// 每局最多抽几个分歧决策点做配对 rollout
    #[arg(long, default_value_t = 8)]
    points_per_game: usize,

    /// 每个动作的配对 rollout 次数
    #[arg(long, default_value_t = 256)]
    rollouts: usize,

    /// 抽样与 CRN 的根种子（与 `--seed` 分开，便于固定轨迹只换探针随机性）
    #[arg(long, default_value_t = 0x5EED_A0FF_1CE0_0001)]
    probe_seed: u64,

    /// 只跑前 N 个计划（调试用，默认全跑）
    #[arg(long)]
    plans: Option<usize>,

    /// 把逐探针点写成 CSV
    #[arg(long)]
    csv: Option<PathBuf>
}

/// 逐探针点 CSV 表头
const POINT_HEADER: [&str; 11] = [
    "plan_index",
    "run_idx",
    "turn",
    "year",
    "stage",
    "pi_action",
    "h_action",
    "pi_mean",
    "h_mean",
    "advantage",
    "weight"
];

/// 一个待测的探针点：网络与手写在此处选了不同动作
struct ProbePoint {
    /// 决策点局面（配对 rollout 的根）
    game: RamenGame,
    /// 被评估策略选的动作
    pi_action: RamenAction,
    /// 手写策略选的动作
    h_action: RamenAction,
    /// 所在回合
    turn: i32,
    /// 所在年份（1/2/3）
    year: i32,
    /// 有效阶段（按动作类型纠正过）
    stage: RamenStage
}

/// 一局之内的探针状态（`Trainer` 只给 `&self`，故用内部可变）
struct ProbeState {
    /// 决策点总数
    total_decisions: usize,
    /// 网络与手写选择不同的决策点数
    diff_decisions: usize,
    /// 蓄水池：等概率保留的分歧点
    reservoir: Vec<ProbePoint>
}

/// 决策拦截器：驱动轨迹的同时收集分歧点
///
/// 谁驱动轨迹由 `pi_drives` 决定；**不驱动的那一方用私有 RNG 提问**，
/// 以免多消耗决策随机流、改变轨迹本身。
#[cfg(feature = "onnx")]
struct ProbeTrainer {
    /// 被评估策略（网络）
    pi: RamenNnTrainer,
    /// 参考策略（手写），同时也是 rollout 基策
    href: RecommendedRamenTrainer,
    /// 轨迹是否由网络驱动
    pi_drives: bool,
    /// 提问用的私有 RNG（不参与轨迹推进）
    aside_rng: RefCell<StdRng>,
    /// 蓄水池抽样用的 RNG
    sample_rng: RefCell<StdRng>,
    /// 每局保留的分歧点上限
    capacity: usize,
    /// 收集状态（局末由外层取走）
    state: Rc<RefCell<ProbeState>>
}

#[cfg(feature = "onnx")]
impl ProbeTrainer {
    /// 把一个分歧点按蓄水池规则纳入样本
    ///
    /// 前 `capacity` 个直接进池；此后第 `j`（从 1 计）个分歧点以 `capacity/j`
    /// 的概率顶掉池中随机一个，得到不放回简单随机样本。
    fn offer(&self, point: ProbePoint) {
        if self.capacity == 0 {
            return;
        }
        let mut state = self.state.borrow_mut();
        if state.reservoir.len() < self.capacity {
            state.reservoir.push(point);
            return;
        }
        // 调用前 `diff_decisions` 已把本点计入，故它就是「至今见过的分歧点数」
        let seen = state.diff_decisions;
        let roll = self.sample_rng.borrow_mut().random_range(0..seen);
        if roll < self.capacity {
            state.reservoir[roll] = point;
        }
    }
}

#[cfg(feature = "onnx")]
impl Trainer<RamenGame> for ProbeTrainer {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        ensure!(!actions.is_empty(), "候选动作为空");
        let (pi_idx, h_idx) = if self.pi_drives {
            let pi_idx = self.pi.select_action(game, actions, rng)?;
            let h_idx = self
                .href
                .select_action(game, actions, &mut self.aside_rng.borrow_mut())?;
            (pi_idx, h_idx)
        } else {
            let h_idx = self.href.select_action(game, actions, rng)?;
            let pi_idx = self.pi.select_action(game, actions, &mut self.aside_rng.borrow_mut())?;
            (pi_idx, h_idx)
        };
        let driving = if self.pi_drives { pi_idx } else { h_idx };

        {
            let mut state = self.state.borrow_mut();
            state.total_decisions += 1;
            if actions[pi_idx] == actions[h_idx] {
                // 两策略选同一动作，该点对恒等式贡献恒为 0，不必搜索
                return Ok(driving);
            }
            state.diff_decisions += 1;
        }
        self.offer(ProbePoint {
            game: game.clone(),
            pi_action: actions[pi_idx].clone(),
            h_action: actions[h_idx].clone(),
            turn: game.base.turn,
            year: game.current_year(),
            stage: ramen_effective_stage(game, actions)
        });
        Ok(driving)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        // 事件选项两侧同源（网络 choice 头未训练，推理时也是委托手写），恒等式上贡献为 0
        self.href.select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.href.select_event_choice(game, event, choices, rng)
    }
}

/// 一个探针点的测量结果
#[derive(Debug, Clone)]
struct PointResult {
    /// 计划下标
    plan_index: usize,
    /// 局内序号
    run_idx: u64,
    /// 回合
    turn: i32,
    /// 年份
    year: i32,
    /// 阶段名
    stage: String,
    /// 网络动作文本
    pi_action: String,
    /// 手写动作文本
    h_action: String,
    /// 网络动作的配对均分
    pi_mean: f64,
    /// 手写动作的配对均分
    h_mean: f64,
    /// 单点 advantage
    advantage: f64,
    /// 该点的抽样权重（分歧点总数 / 抽样点数）
    weight: f64
}

/// 一局的汇总
#[derive(Debug, Clone)]
struct GameResult {
    /// 计划下标
    plan_index: usize,
    /// 该局终局分（roll-in 策略的轨迹分）
    score: f64,
    /// 决策点总数
    total_decisions: usize,
    /// 分歧点总数
    diff_decisions: usize,
    /// 实际测了几个点
    probed: usize,
    /// 本局 advantage 总和的估计
    adv_total: f64,
    /// 按「年份 阶段」分解的贡献
    by_stratum: Vec<(String, f64)>
}

/// 对一个探针点做配对 rollout
///
/// 两个动作共享同一张 CRN 种子表（`RolloutSeeds::seed_at(j)`），
/// 这是配对方差削减的载体，不得按动作派生种子。
///
/// # 错误
///
/// `rollouts` 非正、或任一 rollout 失败时报错——静默丢点会让权重与实际测量数脱节。
fn measure_point(
    search: &FlatSearch<RamenGame>, point: &ProbePoint, rollouts: usize, root_seed: u64
) -> Result<(f64, f64)> {
    ensure!(rollouts > 0, "--rollouts 必须为正");
    let seeds = RolloutSeeds::from_root(root_seed);
    let mut pi_sum = 0.0;
    let mut h_sum = 0.0;
    for j in 0..rollouts {
        let seed = seeds.seed_at(j);
        pi_sum += search.simulate_common(&point.game, &point.pi_action, seed)?.score;
        h_sum += search.simulate_common(&point.game, &point.h_action, seed)?.score;
    }
    let n = rollouts as f64;
    Ok((pi_sum / n, h_sum / n))
}

/// 跑一个计划的全部对局并测量其探针点
///
/// # 错误
///
/// 任一局、任一次 rollout 失败，或探针状态无法独占取回时报错。
#[cfg(feature = "onnx")]
fn run_plan(
    plan: &DeckPlan, plan_index: usize, args: &ProbeArgs, nn: &RamenNnTrainer, search: &FlatSearch<RamenGame>
) -> Result<(Vec<GameResult>, Vec<PointResult>)> {
    let inherit = gen1_inherit();
    // 与 `ramen_space_bench` 完全一致的种子派生，保证轨迹与闭环基准逐局可对齐
    let base_seed = args.seed.wrapping_add((plan_index as u64).wrapping_mul(1_000_003));
    let mut games = Vec::with_capacity(args.runs_per_plan as usize);
    let mut points = Vec::new();

    for run_idx in 0..args.runs_per_plan {
        let stream = splitmix64(
            args.probe_seed
                .wrapping_add((plan_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add(run_idx.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        );
        let state = Rc::new(RefCell::new(ProbeState {
            total_decisions: 0,
            diff_decisions: 0,
            reservoir: Vec::with_capacity(args.points_per_game)
        }));
        let probe = ProbeTrainer {
            pi: nn.clone(),
            href: RecommendedRamenTrainer::new(),
            pi_drives: args.rollin == "nn",
            aside_rng: RefCell::new(StdRng::seed_from_u64(splitmix64(stream ^ 0xA5A5_A5A5_A5A5_A5A5))),
            sample_rng: RefCell::new(StdRng::seed_from_u64(splitmix64(stream ^ 0x1234_5678_9ABC_DEF0))),
            capacity: args.points_per_game,
            state: Rc::clone(&state)
        };
        let mut logging = LoggingTrainer::new(probe, base_seed + run_idx);
        logging.set_logging(false);
        let outcome = bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &logging)?;
        drop(logging);

        let taken = Rc::try_unwrap(state)
            .map_err(|_| anyhow!("探针状态仍被引用，无法取出"))?
            .into_inner();
        let probed = taken.reservoir.len();
        // 不放回简单随机抽样的总量估计：Σ_sample × (N / n)
        let weight = if probed == 0 {
            0.0
        } else {
            taken.diff_decisions as f64 / probed as f64
        };

        let mut adv_total = 0.0;
        let mut by_stratum: BTreeMap<String, f64> = BTreeMap::new();
        for (k, point) in taken.reservoir.iter().enumerate() {
            let root_seed =
                splitmix64(stream.wrapping_add((k as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let (pi_mean, h_mean) = measure_point(search, point, args.rollouts, root_seed)?;
            let advantage = pi_mean - h_mean;
            adv_total += advantage * weight;
            let key = format!("Y{} {:?}", point.year, point.stage);
            *by_stratum.entry(key).or_insert(0.0) += advantage * weight;
            points.push(PointResult {
                plan_index,
                run_idx,
                turn: point.turn,
                year: point.year,
                stage: format!("{:?}", point.stage),
                pi_action: point.pi_action.to_string(),
                h_action: point.h_action.to_string(),
                pi_mean,
                h_mean,
                advantage,
                weight
            });
        }

        games.push(GameResult {
            plan_index,
            score: f64::from(outcome.score),
            total_decisions: taken.total_decisions,
            diff_decisions: taken.diff_decisions,
            probed,
            adv_total,
            by_stratum: by_stratum.into_iter().collect()
        });
    }
    Ok((games, points))
}

/// 均值与均值标准误
///
/// # 错误
///
/// 序列为空时报错——空集的均值没有意义。
fn mean_stderr(xs: &[f64]) -> Result<(f64, f64)> {
    ensure!(!xs.is_empty(), "没有可汇总的样本");
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return Ok((mean, 0.0));
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok((mean, (var / n).sqrt()))
}

#[cfg(not(feature = "onnx"))]
fn main() -> Result<()> {
    bail!("ramen_advantage_probe 需要编译 feature onnx：cargo run --release --features onnx --bin ramen_advantage_probe")
}

#[cfg(feature = "onnx")]
fn main() -> Result<()> {
    let args = ProbeArgs::parse();
    ensure!(args.rollouts > 0, "--rollouts 必须为正");
    ensure!(args.points_per_game > 0, "--points-per-game 必须为正");
    if !matches!(args.rollin.as_str(), "nn" | "handwritten") {
        bail!("未知 --rollin: {}（可选 nn / handwritten）", args.rollin);
    }

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    init_global_with_config(&load_game_config()?)?;

    let nn = RamenNnTrainer::load(&args.model)?.with_race_shield(!args.no_race_shield);
    // rollout 基策取自 `FlatSearchGame::default_rollout_trainer`（拉面 = 手写推荐策略），
    // 与教师采集同源。`max_depth` 显式置 0：拉面不支持截断估值。
    // `use_ucb` 与 `simulate_common` 无关，置 false 只为配置意图清晰。
    let search = FlatSearch::<RamenGame>::new(SearchConfig::default().with_max_depth(0).with_ucb(false));

    let space = SamplingSpace::gen1()?;
    let all_plans = space.plans();
    let plans = match args.plans {
        Some(n) => &all_plans[..n.min(all_plans.len())],
        None => all_plans
    };
    println!(
        "advantage 探针：{} 个计划 × {} 局，roll-in = {}，每局抽 {} 个分歧点 × {} 次配对 rollout",
        plans.len(),
        args.runs_per_plan,
        args.rollin,
        args.points_per_game,
        args.rollouts
    );
    println!(
        "模型 {}{}",
        args.model.display(),
        if args.no_race_shield { "（无守门）" } else { "" }
    );

    let start = std::time::Instant::now();
    let collected: Vec<(Vec<GameResult>, Vec<PointResult>)> = plans
        .par_iter()
        .enumerate()
        .map(|(i, plan)| run_plan(plan, i, &args, &nn, &search))
        .collect::<Result<Vec<_>>>()?;
    let elapsed = start.elapsed().as_secs_f64();

    let mut games: Vec<GameResult> = Vec::new();
    let mut points: Vec<PointResult> = Vec::new();
    for (g, p) in collected {
        games.extend(g);
        points.extend(p);
    }
    games.sort_by_key(|g| g.plan_index);

    let adv: Vec<f64> = games.iter().map(|g| g.adv_total).collect();
    let scores: Vec<f64> = games.iter().map(|g| g.score).collect();
    let (adv_mean, adv_se) = mean_stderr(&adv)?;
    let (score_mean, score_se) = mean_stderr(&scores)?;
    let total_dec: usize = games.iter().map(|g| g.total_decisions).sum();
    let diff_dec: usize = games.iter().map(|g| g.diff_decisions).sum();
    let probed: usize = games.iter().map(|g| g.probed).sum();

    // 分层贡献按「出现过该分层的局」平均：未抽到该分层的局贡献未被观测，
    // 计 0 会把均值系统性拉向零。分层数字只作归因参考，合计以 `adv_total` 为准。
    let mut by_stratum: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for g in &games {
        for (k, v) in &g.by_stratum {
            by_stratum.entry(k.clone()).or_default().push(*v);
        }
    }

    println!("\n按「年份 阶段」分解（只统计抽到过该分层的局）");
    println!("  {:<28} {:>7} {:>12} {:>10}", "分层", "局数", "平均贡献", "标准误");
    for (key, vs) in &by_stratum {
        let (m, se) = mean_stderr(vs)?;
        println!("  {:<28} {:>7} {:>12.1} {:>10.1}", key, vs.len(), m, se);
    }

    println!("\n合计");
    println!("  局数               {}", games.len());
    println!(
        "  决策点             {}（其中分歧 {}，占 {:.1}%）",
        total_dec,
        diff_dec,
        100.0 * diff_dec as f64 / total_dec.max(1) as f64
    );
    println!("  实测点             {}", probed);
    println!("  轨迹均分           {:.0} ± {:.0}", score_mean, score_se);
    println!("  Â 估计 J(π)−J(H)   {:.1} ± {:.1}", adv_mean, adv_se);
    println!(
        "  95% CI             [{:.1}, {:.1}]",
        adv_mean - 1.96 * adv_se,
        adv_mean + 1.96 * adv_se
    );
    println!("  耗时               {elapsed:.1} s");

    if let Some(path) = &args.csv {
        let rows: Vec<Vec<String>> = points
            .iter()
            .map(|p| {
                vec![
                    p.plan_index.to_string(),
                    p.run_idx.to_string(),
                    p.turn.to_string(),
                    p.year.to_string(),
                    p.stage.clone(),
                    p.pi_action.clone(),
                    p.h_action.clone(),
                    format!("{:.2}", p.pi_mean),
                    format!("{:.2}", p.h_mean),
                    format!("{:.2}", p.advantage),
                    format!("{:.4}", p.weight)
                ]
            })
            .collect();
        bench::write_csv(path, &POINT_HEADER, &rows)?;
        println!("  逐点 CSV           {}", path.display());
    }
    Ok(())
}
