//! MCTS 训练员 vs 手写逻辑整局配对基准（闭环口径）
//!
//! 回答「在相同随机局面下，MCTS 训练员比正式推荐手写策略强多少」：
//! 对同一 `(build, 基础种子, 局号)` 分别用 [`RamenMctsTrainer`] 与
//! [`RecommendedRamenTrainer`] 跑完整局。两局通过 [`bench::seeded_rngs`]
//! 注入**同一个规则主种子**（[`PairRow::rule_seed`]），随机未来逐位一致
//! （CRN 配对），因此逐局差值 `Δ = 评分_mcts − 评分_handwritten`
//! 几乎全部来自策略选择差异，而不是运气。
//!
//! # 与 bench_base 的差异
//!
//! `bench_base --trainer mcts|handwritten` 是各策略独立跑批、无配对；
//! 本 bin 强制同种子配对，消除随机世界漂移，专门量化「训练员相对
//! 手写的期望提升」。手写侧决策确定性（不消耗随机流），MCTS 侧搜索
//! 消耗各自的决策流，但两局规则层面对的未来一致，配对成立。
//!
//! # 口径
//!
//! - 手写 = 正式推荐策略 [`RecommendedRamenTrainer::new()`]（bench
//!   "handwritten" 档语义）。
//! - MCTS = 生产训练员 [`RamenMctsTrainer`]；其门控全关时与手写策略
//!   **逐位一致**（`ramen_mcts_trainer::test_stages_none_matches_recommended`
//!   钉死），故本基准测出的差值可干净归因于「搜索」。
//! - 评分 = 终局 `calc_score()`（[`GameOutcome::score`]）。
//!
//! # 搜索参数 = 生产实际值（默认）
//!
//! 默认**不覆盖**任何 MCTS 参数：`SearchConfig::new_game_config(&game_config)`
//! + `game_config.mcts.ramen_search_stages` —— 与 umaai 在线运行
//! （`main.rs` 同款构造）完全一致。本机生产值见 `game_config.toml` /
//! `gamedata/default_config.toml` 的 `[mcts]` 段（如 `search_n=8192`、
//! stages `train,ramen,region`、UCB 开、`radical_factor_max=1.4`）。
//! `--search-n` 等参数仅为对照实验提供覆盖入口，覆盖后结果**不再代表
//! 生产训练员**，请勿与生产档数据混比。
//!
//! # 用法（Release，从 workspace 根运行）
//!
//! ```text
//! # 生产实际配置扫测（默认）
//! cargo run --release --bin ramen_mcts_pair_bench -- \
//!     --seeds 61444,42,7 --runs 3 --out logs/mcts_pair.csv
//!
//! # 指定 build、打折预算做对照（非生产档，勿与生产档混比）
//! cargo run --release --bin ramen_mcts_pair_bench -- \
//!     --builds speed,wisdom --search-n 1024 --search-stages train,ramen
//! ```
//!
//! # 产出
//!
//! - 配对 CSV（每局一行：两策略的评分/五维/PT/RMJ/自选比赛/耗时 + Δ）
//! - 汇总打印：全局与按 build 的 Δ 均值、标准误、95% CI、胜负局数
//!
//! # 可复现性与耗时
//!
//! 同一参数下结果完全确定（决策 RNG 与规则流均由种子派生；`elapsed_ms`
//! 属运行耗时，允许波动）。**注意生产档耗时长**：`search_n=8192` +
//! `region` 门控下 MCTS 整局可达分钟级（第 2/3 年地区单点 120 候选），
//! 扫测前先用 `--runs 1` 与单 build 探时间。

use std::{env, sync::Arc};

use anyhow::{Context, Result, bail, ensure};
use lexopt::Arg;
use rayon::prelude::*;
use serde::Deserialize;
use umasim::{
    bench::{self, CardPickOpts, GameOutcome, load_player_builds},
    game::InheritInfo,
    gamedata::{RamenRegionStrategy, init_global_with_config},
    search::{RamenLeafValuer, SearchConfig, SelectObjective},
    trainer::{
        LoggingTrainer, RamenLeafCounters, RamenMctsTrainer, RamenSearchStages, RecommendedRamenTrainer
    },
    utils::{get_workspace_root, load_game_config}
};

/// 三条对照臂
///
/// ❗**A → B 同时换了评分轴与聚合方式两项**，这个差值不得称为「纯聚合效应」。
/// 只有 B → C 用于评估叶替换路径：那一步只换「rollout 的尾巴怎么收」，
/// 评分轴、聚合方式、搜索阶段、候选、UCB 与预算全部保持一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// 上游原路径：PT 轴 + rank 加权，完整续跑到真实终局
    A,
    /// 完整续跑，改用结算分轴 + 普通均值
    B,
    /// 手写续跑到 H 个回合边界后取 CPU 叶估值，同样是结算分轴 + 普通均值
    C
}

impl Arm {
    /// 解析 CLI 取值
    ///
    /// # 错误
    ///
    /// 名字不认识时报错。
    fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "a" | "upstream" | "pt-rank" => Ok(Arm::A),
            "b" | "full-mean" => Ok(Arm::B),
            "c" | "leaf-mean" => Ok(Arm::C),
            other => bail!("未知实验臂 {other}（可用 a / b / c）")
        }
    }

    /// 进 CSV 与日志的短名
    fn tag(self) -> &'static str {
        match self {
            Arm::A => "a-pt-rank-full",
            Arm::B => "b-score-mean-full",
            Arm::C => "c-score-mean-leaf"
        }
    }

    /// 本臂的选优口径
    fn objective(self) -> SelectObjective {
        match self {
            Arm::A => SelectObjective::PtRank,
            Arm::B | Arm::C => SelectObjective::ScoreMean
        }
    }
}

/// 一条实验臂的完整装配
///
/// 三臂除本结构列出的差异外，卡组、马娘、继承、外层地区策略、搜索阶段、候选、
/// UCB 与预算完全一致，且配对局共享同一 `rule_master`。
struct ArmSetup {
    /// 哪一条臂
    arm: Arm,
    /// 叶截断深度 H（跨越的**回合边界**数；0 = 完整续跑）
    leaf_h: usize,
    /// 叶估值器（只有 C 臂有）
    valuer: Option<Arc<dyn RamenLeafValuer>>
}

impl ArmSetup {
    /// 按本臂装配一个 MCTS 训练员
    fn trainer(&self, search: &SearchConfig, stages: &RamenSearchStages) -> RamenMctsTrainer {
        let mut t = RamenMctsTrainer::new(search.clone())
            .with_stages(*stages)
            .with_objective(self.arm.objective())
            .with_leaf_horizon(self.leaf_h);
        if let Some(v) = &self.valuer {
            t = t.with_leaf_valuer(Arc::clone(v));
        }
        t
    }
}

/// 基准参数（`None` = 用生产实际值，不覆盖）
#[derive(Debug, Clone)]
struct BenchArgs {
    /// 每 (build, seed) 的配对局数
    runs: usize,
    /// 独立扫测的种子列表
    seeds: Vec<u64>,
    /// build 名过滤（None = bench_config.toml 全部 player_builds）
    builds: Option<Vec<String>>,
    /// 覆盖 MCTS 每候选 rollout 数（None = game_config 实际值）
    search_n: Option<usize>,
    /// 覆盖 MCTS 搜索阶段门控（None = game_config 实际值）
    search_stages: Option<String>,
    /// 覆盖 MCTS UCB 开关（None = game_config 实际值）
    search_ucb: Option<bool>,
    /// 覆盖 MCTS 激进度上限（None = game_config 实际值）
    radical_factor_max: Option<f64>,
    /// 实验臂（默认 A = 上游原路径）
    arm: Arm,
    /// C 臂的叶截断深度 H（跨越的回合边界数）
    leaf_h: usize,
    /// C 臂的叶估值模型路径
    leaf_model: Option<String>,
    /// rayon 线程数上限（None = game_config `collector.threads`）
    ///
    /// 冒烟时用它把占用压下来：默认值是按满负荷采集调的，在交互机器上跑小规模冒烟
    /// 会把整台机器占满。**不改配置文件**，只覆盖本次运行。
    threads: Option<usize>,
    /// 配对 CSV 输出路径（相对 workspace 根）
    out: String
}

impl Default for BenchArgs {
    /// 扫测默认：三种子 × 3 局；搜索参数全部走生产实际值
    fn default() -> Self {
        Self {
            runs: 3,
            seeds: vec![61444, 42, 7],
            builds: None,
            search_n: None,
            search_stages: None,
            search_ucb: None,
            radical_factor_max: None,
            arm: Arm::A,
            leaf_h: 8,
            leaf_model: None,
            threads: None,
            out: "logs/ramen_mcts_pair_bench.csv".to_string()
        }
    }
}

/// 解析 CLI 参数（`--key value` 或 `--key=value`），覆盖基准参数
fn apply_cli(mut args: BenchArgs) -> Result<BenchArgs> {
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("runs") => args.runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("seeds") => {
                let text: String = bench::parse_value(&mut parser, "seeds")?;
                args.seeds = text
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u64>().with_context(|| format!("种子无效: {s}")))
                    .collect::<Result<Vec<_>>>()?;
            }
            Arg::Long("builds") => {
                let text: String = bench::parse_value(&mut parser, "builds")?;
                args.builds = Some(
                    text.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                );
            }
            Arg::Long("search-n") => args.search_n = Some(bench::parse_value(&mut parser, "search-n")?),
            Arg::Long("search-stages") => args.search_stages = Some(bench::parse_value(&mut parser, "search-stages")?),
            Arg::Long("search-ucb") => args.search_ucb = Some(bench::parse_value(&mut parser, "search-ucb")?),
            Arg::Long("radical-factor") => {
                args.radical_factor_max = Some(bench::parse_value(&mut parser, "radical-factor")?)
            }
            Arg::Long("arm") => {
                let text: String = bench::parse_value(&mut parser, "arm")?;
                args.arm = Arm::parse(&text)?;
            }
            Arg::Long("leaf-h") => args.leaf_h = bench::parse_value(&mut parser, "leaf-h")?,
            Arg::Long("leaf-model") => args.leaf_model = Some(bench::parse_value(&mut parser, "leaf-model")?),
            Arg::Long("threads") => args.threads = Some(bench::parse_value(&mut parser, "threads")?),
            Arg::Long("out") => args.out = bench::parse_value(&mut parser, "out")?,
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: ramen_mcts_pair_bench [--runs N] [--seeds S1,S2,...] [--builds b1,b2,...]
\n      MCTS 参数默认=生产实际值（game_config [mcts] + ramen_search_stages）；
\n      覆盖实验: [--search-n N] [--search-stages train,ramen,...] [--search-ucb true|false]
\n                [--radical-factor F] [--out PATH]
\n      运行环境: [--threads N]（rayon 线程上限，默认取 game_config collector.threads）
\n      对照臂:   [--arm a|b|c] [--leaf-h H] [--leaf-model PATH]
\n        a = 上游原路径（PT 轴 + rank 加权，完整续跑，默认）
\n        b = 完整续跑，结算分轴 + 普通均值
\n        c = 手写续跑到 H 个回合边界后取 CPU 叶估值，结算分轴 + 普通均值
\n        ❗a→b 同时换了评分轴与聚合方式两项，不得称为「纯聚合效应」；
\n          只有 b→c 用于评估叶替换路径。c 需要 onnx 构建与 --leaf-model。
\n  同 (build, seed, 局号) 下用 MCTS 训练员与正式推荐手写策略各跑整局，
\n  配对差 Δ = 评分_mcts − 评分_handwritten，逐局落 CSV 并汇总（均值/SE/95%CI）。"
                );
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("未知参数: {other:?}（可用 --help 查看用法）");
            }
        }
    }
    Ok(args)
}

/// bench_config.toml 的基准基础字段（build 卡组生成用；CLI 不覆盖）
#[derive(Debug, Clone, Deserialize)]
struct BenchBaseCfg {
    /// 马娘 ID
    uma: u32,
    /// 固定友人卡 idrank
    friend: u32,
    /// 种马蓝因子个数
    blue_count: [i32; 5],
    /// 种马额外属性
    extra_count: [i32; 6]
}

impl Default for BenchBaseCfg {
    /// 与 bench_base 的内置默认一致（bench_config.toml 缺失时兜底）
    fn default() -> Self {
        Self {
            uma: 102601,
            friend: 303054,
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40]
        }
    }
}

/// 读取 bench_config.toml 的基础字段；缺失时用内置默认
fn load_bench_base(workspace_root: &std::path::Path) -> Result<BenchBaseCfg> {
    let path = workspace_root.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 bench_config.toml 失败: {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("解析 bench_config.toml 失败: {}", path.display()))
    } else {
        println!("提示: 未找到 bench_config.toml，使用内置默认参数");
        Ok(BenchBaseCfg::default())
    }
}

/// 一对配对局的完整结果（两策略共用同一规则主种子）
#[derive(Debug, Clone)]
struct PairRow {
    /// build 名
    build: String,
    /// 基础种子
    seed: u64,
    /// 局号（第 i 局 = seed 下的第 i 局）
    run: u64,
    /// 规则主种子（两策略必须相同，是配对成立的证据）
    rule_seed: u64,
    /// 手写策略整局结果
    hand: GameOutcome,
    /// MCTS 训练员整局结果
    mcts: GameOutcome,
    /// 本局用的实验臂
    arm: Arm,
    /// 本局的叶截断深度（0 = 完整续跑）
    leaf_h: usize,
    /// 本局 MCTS 侧的续跑分账（真实终局 / 叶 / 顺延）
    counters: RamenLeafCounters
}

impl PairRow {
    /// 配对差（MCTS − 手写）
    fn delta(&self) -> i32 {
        self.mcts.score - self.hand.score
    }
}

/// 跑一对配对局：同 (seed, run_idx) 下手写与 MCTS 各跑整局
///
/// 两局通过 [`bench::seeded_rngs`] 得到同一 `rule_master`，随机未来逐位一致；
/// 返回前校验两局 `rule_seed` 相等，防止将来种子派生改动后配对静默失效。
///
/// # 错误
///
/// 任一局失败、或两局规则主种子不一致时报错。
fn run_pair(
    uma: u32,
    deck: &[u32; 6],
    inherit: &InheritInfo,
    seed: u64,
    run_idx: u64,
    search: &SearchConfig,
    stages: &RamenSearchStages,
    arm: &ArmSetup
) -> Result<PairRow> {
    let log_seed = seed + run_idx;

    let mut hand_log = LoggingTrainer::new(RecommendedRamenTrainer::new(), log_seed);
    hand_log.set_logging(false);
    let hand = bench::run_seeded(uma, deck, inherit, seed, run_idx, &hand_log)?;
    drop(hand_log);

    let mut mcts_log = LoggingTrainer::new(arm.trainer(search, stages), log_seed);
    mcts_log.set_logging(false);
    let mcts = bench::run_seeded(uma, deck, inherit, seed, run_idx, &mcts_log)?;
    // 续跑分账要在训练员析构之前取：真实终局与叶预测分开数
    let counters = mcts_log.inner().leaf_counters();
    drop(mcts_log);

    ensure!(
        hand.seed == mcts.seed,
        "配对失败: 两策略规则主种子不一致 {} != {}（seed={seed} run={run_idx}）",
        hand.seed,
        mcts.seed
    );
    Ok(PairRow {
        build: String::new(),
        seed,
        run: run_idx,
        rule_seed: hand.seed,
        hand,
        mcts,
        arm: arm.arm,
        leaf_h: arm.leaf_h,
        counters
    })
}

/// 配对 CSV 表头（21 列）
///
/// 末尾 4 列是对照臂与续跑分账：同一份 CSV 里混着不同臂的行时，光看分数分不出
/// 是哪一臂跑的。
const PAIR_HEADER: [&str; 21] = [
    "build",
    "seed",
    "run",
    "rule_seed",
    "hand_score",
    "mcts_score",
    "delta",
    "hand_five",
    "mcts_five",
    "hand_pt",
    "mcts_pt",
    "hand_rmj",
    "mcts_rmj",
    "hand_free_race_ok",
    "mcts_free_race_ok",
    "hand_elapsed_ms",
    "mcts_elapsed_ms",
    "arm",
    "leaf_h",
    "rollout_terminal",
    "rollout_leaf"
];

/// 配对结果转 CSV 行（不含表头）
fn pair_to_row(build: &str, row: &PairRow) -> Vec<String> {
    // 五维用 `/` 连接、PT 取三年剧本 PT 合计（与 bench 系列口径一致）
    let join = |s: &[i32; 5]| s.iter().map(i32::to_string).collect::<Vec<_>>().join("/");
    let pt = |o: &GameOutcome| o.yearly_scenario_pt.iter().sum::<i32>();
    vec![
        build.to_string(),
        row.seed.to_string(),
        row.run.to_string(),
        row.rule_seed.to_string(),
        row.hand.score.to_string(),
        row.mcts.score.to_string(),
        row.delta().to_string(),
        join(&row.hand.five_status),
        join(&row.mcts.five_status),
        pt(&row.hand).to_string(),
        pt(&row.mcts).to_string(),
        row.hand.rmj_ok.to_string(),
        row.mcts.rmj_ok.to_string(),
        u8::from(row.hand.free_race_ok).to_string(),
        u8::from(row.mcts.free_race_ok).to_string(),
        format!("{:.3}", row.hand.elapsed_ms),
        format!("{:.3}", row.mcts.elapsed_ms),
        row.arm.tag().to_string(),
        row.leaf_h.to_string(),
        row.counters.terminal.to_string(),
        row.counters.leaf.to_string()
    ]
}

/// 一组配对差的统计：均值 / 样本标准误 / t / 95% CI / 胜负平局数
#[derive(Debug, Clone)]
struct DeltaStats {
    /// 样本数
    n: usize,
    /// Δ 均值
    mean: f64,
    /// 样本标准误（无偏，n≥2）
    se: f64,
    /// t = mean / se（se=0 时记 0）
    t: f64,
    /// 95% CI 下界 / 上界
    ci: (f64, f64),
    /// Δ>0 / Δ==0 / Δ<0 局数
    wins: usize,
    ties: usize,
    losses: usize
}

/// 从配对差序列聚合统计
///
/// # 错误
///
/// 样本为空时报错——空集没有统计意义。
fn delta_stats(deltas: &[f64]) -> Result<DeltaStats> {
    ensure!(!deltas.is_empty(), "没有可汇总的配对局");
    let n = deltas.len();
    let mean = deltas.iter().sum::<f64>() / n as f64;
    let se = if n < 2 {
        0.0
    } else {
        let var = deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        (var / n as f64).sqrt()
    };
    let t = if se > 0.0 { mean / se } else { 0.0 };
    let (wins, ties, losses) = deltas.iter().fold((0, 0, 0), |(w, t, l), &d| {
        if d > 0.0 {
            (w + 1, t, l)
        } else if d == 0.0 {
            (w, t + 1, l)
        } else {
            (w, t, l + 1)
        }
    });
    Ok(DeltaStats {
        n,
        mean,
        se,
        t,
        ci: (mean - 1.96 * se, mean + 1.96 * se),
        wins,
        ties,
        losses
    })
}

/// 打印一份统计
fn print_stats(label: &str, s: &DeltaStats) {
    println!(
        "{label:<16} n={:>3} Δ={:>8.1}±{:<7.1} t={:>5.1} 95%CI=[{:>8.1},{:>8.1}] 胜/平/负={}/{}/{}",
        s.n,
        s.mean,
        s.se,
        s.t,
        s.ci.0,
        s.ci.1,
        s.wins,
        s.ties,
        s.losses
    );
}

fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    env::set_current_dir(&workspace_root).with_context(|| "切换到 workspace 根失败")?;

    let args = apply_cli(BenchArgs::default())?;
    ensure!(args.runs > 0, "--runs 必须为正");
    ensure!(!args.seeds.is_empty(), "--seeds 不能为空");

    // 全局数据与线程池初始化（与 bench_base 同口径：基准不跟随手动模式地区策略）
    let mut game_config = load_game_config()?;
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;
    let threads = args.threads.unwrap_or(game_config.collector.threads);
    ensure!(threads > 0, "--threads 必须为正");
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;

    // 搜索参数 = 生产实际值（game_config [mcts]），CLI 仅覆盖显式给出的项
    let mut search = SearchConfig::new_game_config(&game_config);
    if let Some(n) = args.search_n {
        search = search.with_search_n(n);
    }
    if let Some(ucb) = args.search_ucb {
        search = search.with_ucb(ucb);
    }
    if let Some(rf) = args.radical_factor_max {
        search = search.with_radical_factor_max(rf);
    }
    let stages = match &args.search_stages {
        Some(spec) => RamenSearchStages::parse(spec)?,
        None => RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?
    };
    let arm = build_arm(&args)?;

    // 卡组来源：bench_config.toml [player_builds]（可 --builds 过滤）
    let builds = load_player_builds()?;
    let base = load_bench_base(&workspace_root)?;
    let jobs: Vec<(String, [u32; 6])> = builds
        .into_iter()
        .filter(|b| match &args.builds {
            Some(want) => want.contains(&b.name),
            None => true
        })
        .map(|b| {
            let deck = b.make_deck(&CardPickOpts::default(), base.friend)?;
            Ok((b.name(), deck))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!jobs.is_empty(), "--builds 过滤后没有可跑的 build");

    let uma = base.uma;
    let inherit = InheritInfo {
        blue_count: base.blue_count,
        extra_count: base.extra_count
    };

    println!(
        "===== ramen_mcts_pair_bench: uma={} builds={} seeds={:?} runs={} threads={} =====",
        uma,
        jobs.len(),
        args.seeds,
        args.runs,
        threads
    );
    println!(
        "  MCTS 生效参数（默认=生产实际值）: search_n={} stages={} ucb={} radical={} group_size={} cpuct={} expected_stdev={}",
        search.search_n,
        args.search_stages.as_deref().unwrap_or(&game_config.mcts.ramen_search_stages),
        search.use_ucb,
        search.radical_factor_max,
        search.search_group_size,
        search.search_cpuct,
        search.expected_search_stdev
    );
    // 生效口径必须显式打出来：评分轴、聚合方式、截断深度、估值器与 pt_favor_rate
    // 每一个都能悄悄改变结果，事后从分数上分不出是哪一个变了。
    println!(
        "  对照臂 = {} | {}",
        arm.arm.tag(),
        arm.trainer(&search, &stages).effective_setup()
    );
    println!(
        "  pt_favor_rate={} | 地区策略={:?}（与 bench_base 同口径，不跟随手动模式）",
        game_config.pt_favor_rate, game_config.ramen_region_strategy
    );
    if arm.arm != Arm::A {
        println!(
            "  ❗A→{} 同时换了**评分轴**与**聚合方式**两项，该差值不得称为「纯聚合效应」；只有 B→C 用于评估叶替换路径。",
            arm.arm.tag()
        );
    }

    // 并行跑全部 (build, seed, run) 配对局；任一对失败即整体报错（CSV 不落半截）
    // 外层 par_iter 闭包是 Fn、内层 move 闭包需重复持有参数：捕获引用（Copy）而非 owned 值
    let search_ref = &search;
    let stages_ref = &stages;
    let inherit_ref = &inherit;
    let arm_ref = &arm;
    let rows: Vec<PairRow> = jobs
        .par_iter()
        .flat_map_iter(|(name, deck)| {
            args.seeds.iter().flat_map(move |&seed| {
                (0..args.runs).map(move |run_idx| {
                    let row =
                        run_pair(uma, deck, inherit_ref, seed, run_idx as u64, search_ref, stages_ref, arm_ref)
                        .with_context(|| format!("build={name} seed={seed} run={run_idx}"))?;
                    Ok(PairRow {
                        build: name.clone(),
                        ..row
                    })
                })
            })
        })
        .collect::<Result<Vec<_>>>()?;
    println!("配对完成: {} 局 × 2 策略", rows.len());

    // 落盘配对 CSV
    let out_path = workspace_root.join(&args.out);
    let rows_csv: Vec<Vec<String>> = rows.iter().map(|r| pair_to_row(&r.build, r)).collect();
    bench::write_csv(&out_path, &PAIR_HEADER, &rows_csv)?;
    println!("配对 CSV 已写入: {}", out_path.display());

    // 汇总：全局 + 按 build（Δ = 评分_mcts − 评分_handwritten）
    let deltas: Vec<f64> = rows.iter().map(|r| r.delta() as f64).collect();
    let global = delta_stats(&deltas)?;
    println!("\n===== MCTS vs 手写（Δ = 评分_mcts − 评分_handwritten）=====");
    print_stats("全局", &global);
    let mut by_build: Vec<(String, Vec<f64>)> = Vec::new();
    for row in &rows {
        match by_build.iter_mut().find(|(name, _)| name == &row.build) {
            Some((_, xs)) => xs.push(row.delta() as f64),
            None => by_build.push((row.build.clone(), vec![row.delta() as f64]))
        }
    }
    for (name, xs) in &by_build {
        print_stats(name, &delta_stats(xs)?);
    }

    // 表现提示（MCTS 侧是主要耗时）
    let mcts_ms: Vec<f64> = rows.iter().map(|r| r.mcts.elapsed_ms).collect();
    let hand_ms: Vec<f64> = rows.iter().map(|r| r.hand.elapsed_ms).collect();
    let m_mean = mcts_ms.iter().sum::<f64>() / mcts_ms.len().max(1) as f64;
    let h_mean = hand_ms.iter().sum::<f64>() / hand_ms.len().max(1) as f64;
    println!(
        "\n整局耗时均值: 手写 {h_mean:.1}ms / MCTS {m_mean:.0}ms（{} 局）",
        rows.len(),
    );

    // 续跑分账：真实终局与叶预测分开报，不合成一个「完成数」
    let term: u64 = rows.iter().map(|r| r.counters.terminal).sum();
    let leaf: u64 = rows.iter().map(|r| r.counters.leaf).sum();
    let defer: u64 = rows.iter().map(|r| r.counters.deferrals).sum();
    println!(
        "续跑分账（{} 臂，H={}）: 真实终局 {term} 条 / 叶预测 {leaf} 条 / 边界顺延 {defer} 次",
        arm.arm.tag(),
        arm.leaf_h
    );
    if let Some(v) = &arm.valuer {
        println!("叶请求数（{}）: {}", v.label(), v.leaf_requests());
    }
    println!("提示：冒烟只证明接线与基本正确性，不据此宣称提分或提速。");
    Ok(())
}

/// 按 CLI 装配实验臂（并在这里把「半截配置」挡住）
///
/// # 错误
///
/// C 臂缺 `--leaf-model`、`--leaf-h` 为 0、非 C 臂却给了叶参数，或当前构建没有推理
/// 后端时报错。宁可拒绝也不要静默降级成别的臂——那会让整批结果的口径说不清。
fn build_arm(args: &BenchArgs) -> Result<ArmSetup> {
    if args.arm != Arm::C {
        ensure!(
            args.leaf_model.is_none(),
            "--leaf-model 只被 --arm c 消费；{} 臂下它会被静默忽略，故直接拒绝",
            args.arm.tag()
        );
        return Ok(ArmSetup { arm: args.arm, leaf_h: 0, valuer: None });
    }
    ensure!(args.leaf_h > 0, "--arm c 需要 --leaf-h > 0（H = 跨越的回合边界数）");
    let path = args
        .leaf_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--arm c 需要 --leaf-model PATH（叶估值用的模型）"))?;
    build_leaf_valuer(path).map(|valuer| ArmSetup {
        arm: Arm::C,
        leaf_h: args.leaf_h,
        valuer: Some(valuer)
    })
}

/// 加载 CPU 叶估值器（需要 `onnx` feature）
///
/// # 错误
///
/// 模型加载失败时报错。
#[cfg(feature = "onnx")]
fn build_leaf_valuer(path: &str) -> Result<Arc<dyn RamenLeafValuer>> {
    use umasim::trainer::{RamenNnLeafValuer, RamenNnTrainer};

    let nn = Arc::new(RamenNnTrainer::load(std::path::Path::new(path))?);
    println!("  叶估值模型已加载: {path}（后端 = tract CPU）");
    Ok(Arc::new(RamenNnLeafValuer::new(nn)))
}

/// 没有推理后端时明确拒绝，而不是悄悄退回别的臂
///
/// # 错误
///
/// 恒报错。
#[cfg(not(feature = "onnx"))]
fn build_leaf_valuer(_path: &str) -> Result<Arc<dyn RamenLeafValuer>> {
    bail!("--arm c 需要推理后端：请用 --features onnx 重新构建本 bin")
}

#[cfg(test)]
mod tests {
    use super::*;
    use umasim::{
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: InheritInfo = InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 准备固定种子的搜索参数（小预算冒烟档：不读 game_config，避免依赖文件配置）
    fn smoke_search() -> SearchConfig {
        SearchConfig::default().with_search_n(32).with_ucb(false)
    }

    /// 上游原路径那一臂（既有用例的默认装配）
    fn arm_a() -> ArmSetup {
        ArmSetup { arm: Arm::A, leaf_h: 0, valuer: None }
    }

    /// 臂名解析、短名与口径映射
    ///
    /// # 错误
    ///
    /// 任一映射不符时返回错误。
    #[test]
    fn test_arm_parse_and_objective() -> Result<()> {
        for (text, want) in [
            ("a", Arm::A),
            ("A", Arm::A),
            ("upstream", Arm::A),
            ("b", Arm::B),
            ("full-mean", Arm::B),
            ("c", Arm::C),
            ("leaf-mean", Arm::C)
        ] {
            let got = Arm::parse(text)?;
            println!("parse({text}) = {got:?} tag={}", got.tag());
            ensure!(got == want, "{} 解析成 {:?}", text, want);
        }
        ensure!(Arm::parse("d").is_err(), "不认识的臂名直接报错，不悄悄当成默认臂");
        ensure!(Arm::A.objective() == SelectObjective::PtRank,
            "A 臂用上游口径：PT 轴 + rank 加权"
        );
        ensure!(Arm::B.objective() == SelectObjective::ScoreMean
                && Arm::C.objective() == SelectObjective::ScoreMean,
            "B / C 臂都用结算分轴 + 普通均值（两臂口径一致，差别只在叶替换）"
        );
        let tags = [Arm::A.tag(), Arm::B.tag(), Arm::C.tag()];
        ensure!(tags.iter().collect::<std::collections::HashSet<_>>().len() == 3,
            "三个臂名互不相同（CSV 里分得开）"
        );
        Ok(())
    }

    /// 半截装配一律拒绝，不静默降级成别的臂
    ///
    /// # 错误
    ///
    /// 任一半截配置被放行时返回错误。
    #[test]
    fn test_build_arm_rejects_half_configs() -> Result<()> {
        let base = BenchArgs::default();

        let mut a_with_model = base.clone();
        a_with_model.arm = Arm::A;
        a_with_model.leaf_model = Some("x.onnx".to_string());
        let e = build_arm(&a_with_model).err().map(|e| format!("{e:#}")).unwrap_or_default();
        println!("A 臂 + --leaf-model -> {e}");
        ensure!(e.contains("--leaf-model"), "非 C 臂给了叶模型：直接拒绝而不是静默忽略");

        let mut c_no_model = base.clone();
        c_no_model.arm = Arm::C;
        c_no_model.leaf_model = None;
        let e2 = build_arm(&c_no_model).err().map(|e| format!("{e:#}")).unwrap_or_default();
        println!("C 臂缺 --leaf-model -> {e2}");
        ensure!(e2.contains("--leaf-model"), "C 臂缺模型：报错指名缺的是哪个参数");

        let mut c_zero_h = base.clone();
        c_zero_h.arm = Arm::C;
        c_zero_h.leaf_h = 0;
        c_zero_h.leaf_model = Some("x.onnx".to_string());
        let e3 = build_arm(&c_zero_h).err().map(|e| format!("{e:#}")).unwrap_or_default();
        println!("C 臂 --leaf-h 0 -> {e3}");
        ensure!(e3.contains("--leaf-h"), "C 臂 H=0：报错指名 H 的含义是回合边界数");

        // A / B 臂不需要任何叶参数，必须装得起来且不带估值器
        for arm in [Arm::A, Arm::B] {
            let mut ok = base.clone();
            ok.arm = arm;
            ok.leaf_model = None;
            let setup = build_arm(&ok)?;
            println!("{} 臂装配: leaf_h={} 估值器={}", arm.tag(), setup.leaf_h, setup.valuer.is_some());
            ensure!(setup.leaf_h == 0, "{} 臂不截断", arm.tag());
            ensure!(setup.valuer.is_none(), "{} 臂不带叶估值器", arm.tag());
        }
        Ok(())
    }

    /// A 臂装配必须与上游原路径逐位一致
    ///
    /// 新增的 `--arm` 开关默认值若把行为改掉，这条会红。
    ///
    /// # 错误
    ///
    /// A 臂结果与直接用 `RamenMctsTrainer::new(...).with_stages(...)` 不一致时返回错误。
    #[test]
    fn test_arm_a_matches_upstream_assembly() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        env::set_current_dir(&workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let stages = RamenSearchStages::parse("train,ramen")?;
        let search = smoke_search();

        let via_arm = run_pair(TEST_UMA_ID, &TEST_DECK, &TEST_INHERIT, 42, 0, &search, &stages, &arm_a())?;
        // 直接按上游装配跑一局 MCTS 侧
        let mut log = LoggingTrainer::new(
            RamenMctsTrainer::new(search.clone()).with_stages(stages),
            42
        );
        log.set_logging(false);
        let upstream = bench::run_seeded(TEST_UMA_ID, &TEST_DECK, &TEST_INHERIT, 42, 0, &log)?;
        println!("A 臂 mcts={} | 上游装配 mcts={}", via_arm.mcts.score, upstream.score);
        ensure!(via_arm.mcts.score == upstream.score, "A 臂与上游原装配逐位一致");
        println!("A 臂续跑分账: {:?}", via_arm.counters);
        ensure!(
            via_arm.counters.leaf == 0 && via_arm.counters.deferrals == 0,
            "A 臂一次叶估值都没发生"
        );
        ensure!(
            via_arm.counters.terminal > 0,
            "A 臂照样记下了真实终局 rollout 数（三臂报告要对齐，不能只有 C 臂有数）"
        );
        Ok(())
    }

    /// 冒烟：配对联成立（两策略 rule_seed 一致）+ 可复现（同参两次 Δ 相同）+ 结果合法
    ///
    /// 用小预算 search_n（不是生产档）只为快速验证工具链：配对守卫、可复现性、
    /// CSV 结构。生产档数值评估请用 bin 本体跑。
    #[test]
    fn test_pair_reproducible_and_paired() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        env::set_current_dir(&workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let stages = RamenSearchStages::parse("train,ramen")?;
        let search = smoke_search();

        let mut last: Option<(i32, i32, u64)> = None;
        for _ in 0..2 {
            // 同一 (seed, run) 跑两次：可复现性是「同参同局两次逐位一致」，
            // 不是不同局号比——不同 run 的规则主种子本来就不同
            let row = run_pair(TEST_UMA_ID, &TEST_DECK, &TEST_INHERIT, 42, 0, &search, &stages, &arm_a())?;
            println!(
                "rule_seed={} hand={} mcts={} Δ={}",
                row.rule_seed,
                row.hand.score,
                row.mcts.score,
                row.delta()
            );
            ensure!(row.hand.score > 0 && row.mcts.score > 0, "两策略评分均为正");
            ensure!(row.hand.seed == row.mcts.seed, "两策略规则主种子一致（配对成立）");
            ensure!(row.rule_seed == row.hand.seed, "rule_seed 与手写局一致");
            match last {
                Some((h, m, r)) => {
                    ensure!(h == row.hand.score, "手写局逐位可复现");
                    ensure!(m == row.mcts.score, "MCTS 局逐位可复现");
                    ensure!(r == row.rule_seed, "规则主种子逐位可复现");
                }
                None => last = Some((row.hand.score, row.mcts.score, row.rule_seed))
            }
        }
        Ok(())
    }

    /// CSV 行列结构：表头与数据行严格同长
    #[test]
    fn test_csv_row_matches_header() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        env::set_current_dir(&workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let stages = RamenSearchStages::parse("train,ramen")?;
        let search = smoke_search();
        let row = run_pair(TEST_UMA_ID, &TEST_DECK, &TEST_INHERIT, 42, 0, &search, &stages, &arm_a())?;
        let csv = pair_to_row("speed", &row);
        println!(
            "CSV {} 列: {:?}",
            csv.len(),
            csv.iter().take(7).cloned().collect::<Vec<_>>()
        );
        ensure!(csv.len() == PAIR_HEADER.len(), "数据行列数 == 表头列数");
        ensure!(csv[6] == row.delta().to_string(), "delta 列与计算一致");
        ensure!(
            csv.iter().all(|cell| !cell.is_empty()),
            "无空单元格（配对局必须完整）"
        );
        Ok(())
    }
}