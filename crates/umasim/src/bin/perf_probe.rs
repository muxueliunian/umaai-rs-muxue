//! CPU 耗时探针：固定根搜索 + 手写策略整局耗时（跨 commit 波动监测量具）
//!
//! 两个子命令：
//!
//! - `perf_probe root --turn 32|60 [...]`：用手写策略（[`RecommendedRamenTrainer`]）
//!   从固定种子正常推进育成到目标回合的 **Train 决策点**，在那里用生产搜索参数
//!   （`SearchConfig::new_game_config` 继承 `game_config.toml` 的 [mcts] 段）做一次
//!   固定搜索种子的整根搜索，输出候选数 / 总 rollout 数 / 最优动作 / 最优均值 /
//!   根局面真实评分（`calc_score`，不乘 pt 偏好）/ 耗时。这是「MCTS 单决策点
//!   搜索成本」的最稳定量具：局面与种子逐次一致，三轮中位数可跨 commit 对比
//!   （方法与 `perf_profiling.md` 固定根探针一致）。`root_score` 是两版评分区别的
//!   确定性标记：同种子下只随策略/评分公式变化。
//!
//! - `perf_probe whole --runs N [...]`：手写策略整局耗时批次（`bench::run_seeded`），
//!   输出 mean/median/min/max/std 与评分统计。手写策略是 MCTS rollout 的真正热点
//!   （`perf_profiling.md` §3.1），整局耗时是「代码改动是否让 rollout 变慢」的
//!   廉价快速信号。
//!
//! 输出为**单行式 stdout**（`ROOT label=.. turn=.. ...` / `WHOLE label=.. ...`），
//! 供 `scripts/bench_commit_compare.py` 解析；也可以独立使用：
//!
//! ```text
//! cargo run --release --bin perf_probe -- root --turn 60 --rounds 3
//! cargo run --release --bin perf_probe -- whole --runs 100
//! ```
//!
//! 与 `bench_base` 同口径：从 workspace 根读取 `bench_config.toml` 的
//! uma/friend/继承因子/种子默认值，`game_config.toml` 决定搜索参数与线程数；
//! 所有结果在固定种子下逐位可复现（唯一不变量是机器耗时本身）。

use anyhow::{Context, Result, bail};
use lexopt::Arg;
use rand::{rngs::StdRng, SeedableRng};
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use std::time::Instant;
use umasim::{
    bench::{self, CardPickOpts, load_player_builds},
    game::{
        Game, InheritInfo,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{RamenRegionStrategy, init_global_with_config},
    search::{FlatSearch, RamenSearchOutput, SearchConfig},
    trainer::{LoggingTrainer, RecommendedRamenTrainer},
    utils::{get_workspace_root, load_game_config}
};

/// bench_config.toml 中与本探针相关的默认值（serde 只取这些字段，其余忽略）。
#[derive(Debug, Clone, Deserialize)]
struct BenchDefaults {
    #[serde(default = "default_uma")]
    uma: u32,
    #[serde(default = "default_friend")]
    friend: u32,
    #[serde(default = "default_blue")]
    blue_count: [i32; 5],
    #[serde(default = "default_extra")]
    extra_count: [i32; 6],
    #[serde(default = "default_seed")]
    seed: u64
}

impl Default for BenchDefaults {
    fn default() -> Self {
        Self {
            uma: default_uma(),
            friend: default_friend(),
            blue_count: default_blue(),
            extra_count: default_extra(),
            seed: default_seed()
        }
    }
}

fn default_uma() -> u32 {
    102601
}

fn default_friend() -> u32 {
    303054
}

fn default_blue() -> [i32; 5] {
    [15, 0, 0, 0, 3]
}

fn default_extra() -> [i32; 6] {
    [10, 10, 20, 20, 20, 40]
}

fn default_seed() -> u64 {
    61444
}

/// 读取 workspace 根 `bench_config.toml` 中的默认值（缺失/解析失败时用内置默认）。
fn load_bench_defaults() -> Result<BenchDefaults> {
    let path = get_workspace_root()?.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 bench_config.toml 失败: {}", path.display()))?;
        Ok(toml::from_str(&text).unwrap_or_else(|_| {
            eprintln!("警告: bench_config.toml 解析失败（{}），使用内置默认值", path.display());
            BenchDefaults::default()
        }))
    } else {
        Ok(BenchDefaults::default())
    }
}

/// `--deck` 覆盖串解析：`"id1,id2,id3,id4,id5[,friend]"`（idrank，逗号分隔）。
fn parse_deck(s: &str, friend: u32) -> Result<[u32; 6]> {
    let v: Vec<u32> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(
        v.len() == 5 || v.len() == 6,
        "--deck 需要 5 个支援卡 idrank（友人可省略）或 6 个含友人，收到 {} 个: {s}",
        v.len()
    );
    let mut deck = [0u32; 6];
    deck[..5].copy_from_slice(&v[..5]);
    deck[5] = if v.len() == 6 { v[5] } else { friend };
    Ok(deck)
}

/// 解析卡组来源：`--deck` 优先（自定义卡组），否则取 preset builds 中的指定 build
/// （默认 speed，与 `perf_profiling.md` 固定根探针同口径）。
fn resolve_deck(build: &str, deck_override: Option<&str>, friend: u32) -> Result<[u32; 6]> {
    if let Some(ds) = deck_override {
        return parse_deck(ds, friend);
    }
    let pick = CardPickOpts::default();
    let builds = load_player_builds()?;
    let b = builds
        .iter()
        .find(|b| b.name() == build)
        .ok_or_else(|| {
            let names = builds.iter().map(|b| b.name()).collect::<Vec<_>>();
            anyhow::anyhow!("build 不存在: {build}（可选: {}）", names.join(", "))
        })?;
    Ok(b.make_deck(&pick, friend)?)
}

/// 手写策略从固定种子推进到目标回合的 Train 决策点，返回该局面与候选动作。
///
/// 推进过程与 `bench::run_seeded` 完全同源（`seeded_rngs(seed, 0)` + 逐阶段
/// `run_stage`），因此只要策略本身没变，任何 commit 上到达的根局面**逐位一致**；
/// 若策略改动改变了到达该回合的路径，根局面会不同——那本身就是被监测的波动。
fn advance_to_train_root(
    trainer: &RecommendedRamenTrainer,
    uma: u32,
    deck: &[u32; 6],
    inherit: &InheritInfo,
    base_seed: u64,
    target_turn: i32
) -> Result<(RamenGame, Vec<RamenAction>)> {
    let (mut decision_rng, rule_master) = bench::seeded_rngs(base_seed, 0);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    game.run_stage(trainer, &mut decision_rng)?;
    let mut guard = 0usize;
    while game.next() {
        guard += 1;
        if guard > 1000 {
            bail!("推进超过 1000 步仍未到达目标训练根（turn={target_turn}）");
        }
        // 阶段只会前进，回合越过目标就不可能再遇到它
        if game.turn() > target_turn {
            bail!(
                "回合越过目标（target={target_turn}，当前 turn={} stage={:?}）——该回合不是训练决策点",
                game.turn(),
                game.stage
            );
        }
        if game.turn() == target_turn && game.stage == RamenStage::Train {
            let actions = game.list_actions()?;
            if actions.len() > 1 {
                return Ok((game.clone(), actions));
            }
        }
        game.run_stage(trainer, &mut decision_rng)?;
    }
    bail!("推进到终局仍未遇到 turn={target_turn} 的 Train 决策点（候选需 >1）")
}

/// 在固定根上做一次整根搜索，返回 (候选数, 总 rollout 数, 最优下标, 最优均值, 耗时 ms)。
fn run_root_search(
    search: &FlatSearch<RamenGame>,
    game: &RamenGame,
    actions: &[RamenAction],
    search_seed: u64,
    round: u64
) -> Result<(usize, u64, usize, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(search_seed.wrapping_add(round));
    let start = Instant::now();
    let output: RamenSearchOutput = search.search(game, actions, &mut rng)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let searched: u64 = output
        .action_results
        .iter()
        .map(|(res, _)| res.count() as u64)
        .sum();
    let best = output.best_action_pt_idx();
    let best_mean = output.action_results[best].0.mean();
    Ok((actions.len(), searched, best, best_mean, elapsed_ms))
}

/// `root` 子命令：固定 Train 根搜索耗时。
#[derive(Debug, Clone)]
struct RootOpts {
    uma: u32,
    friend: u32,
    blue_count: [i32; 5],
    extra_count: [i32; 6],
    seed: u64,
    build: String,
    deck: Option<String>,
    turn: i32,
    rounds: usize,
    search_seed: u64,
    label: String
}

/// `whole` 子命令：手写策略整局耗时批次。
#[derive(Debug, Clone)]
struct WholeOpts {
    uma: u32,
    friend: u32,
    blue_count: [i32; 5],
    extra_count: [i32; 6],
    seed: u64,
    build: String,
    deck: Option<String>,
    runs: usize,
    label: String
}

fn cmd_root(opts: RootOpts, search_cfg: SearchConfig) -> Result<()> {
    let inherit = InheritInfo {
        blue_count: opts.blue_count,
        extra_count: opts.extra_count
    };
    let deck = resolve_deck(&opts.build, opts.deck.as_deref(), opts.friend)?;
    println!(
        "INFO label={} uma={} turn={} search_n={} rounds={} seed={} search_seed={} deck={:?}",
        opts.label, opts.uma, opts.turn, search_cfg.search_n, opts.rounds, opts.seed, opts.search_seed, deck
    );
    let trainer = RecommendedRamenTrainer::new();
    let (game, actions) =
        advance_to_train_root(&trainer, opts.uma, &deck, &inherit, opts.seed, opts.turn)?;
    // 根局面的真实累计评分：同种子下只随策略/评分公式变化，是「两版评分区别」的确定性标记
    let root_score = game.uma.calc_score();
    let search = FlatSearch::<RamenGame>::new(search_cfg.clone());
    println!(
        "INFO  根局面: turn={} stage=Train 候选 {} 个 score={} | group_size={} expected_stdev={}",
        game.turn(),
        actions.len(),
        root_score,
        search_cfg.search_group_size,
        search_cfg.expected_search_stdev
    );

    let mut times = Vec::with_capacity(opts.rounds);
    for r in 0..opts.rounds {
        let (candidates, searched, best, best_mean, ms) =
            run_root_search(&search, &game, &actions, opts.search_seed, r as u64)?;
        times.push(ms);
        println!(
            "ROOT label={} turn={} round={} candidates={} searched={} best={} best_mean={:.3} root_score={} elapsed_ms={:.3}",
            opts.label, opts.turn, r + 1, candidates, searched, best, best_mean, root_score, ms
        );
    }
    if opts.rounds > 1 {
        let s = bench::summarize(&times);
        println!(
            "ROOT_MEDIAN label={} turn={} rounds={} median_ms={:.3} mean_ms={:.3} min_ms={:.3} max_ms={:.3} std_ms={:.3}",
            opts.label,
            opts.turn,
            opts.rounds,
            s.median,
            s.mean,
            s.min,
            s.max,
            s.std
        );
    }
    Ok(())
}

fn cmd_whole(opts: WholeOpts) -> Result<()> {
    let inherit = InheritInfo {
        blue_count: opts.blue_count,
        extra_count: opts.extra_count
    };
    let deck = resolve_deck(&opts.build, opts.deck.as_deref(), opts.friend)?;
    let trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), 0);
    let mut times = Vec::with_capacity(opts.runs);
    let mut scores = Vec::with_capacity(opts.runs);
    for i in 0..opts.runs {
        let outcome = bench::run_seeded(opts.uma, &deck, &inherit, opts.seed, i as u64, &trainer)?;
        times.push(outcome.elapsed_ms);
        scores.push(outcome.score as f64);
    }
    let t = bench::summarize(&times);
    let s = bench::summarize(&scores);
    println!(
        "WHOLE label={} runs={} mean_ms={:.3} median_ms={:.3} min_ms={:.3} max_ms={:.3} std_ms={:.3} score_mean={:.1} score_min={:.0} score_max={:.0}",
        opts.label, opts.runs, t.mean, t.median, t.min, t.max, t.std, s.mean, s.min, s.max
    );
    Ok(())
}

fn print_help() {
    println!(
        "用法: perf_probe <root|whole> [选项]
\n  root   固定 Train 根搜索耗时（每轮一次整根搜索；--rounds 多轮出中位数）
\n  whole  手写策略整局耗时批次（--runs N 局）
\n通用选项:
\n  --uma U            马娘 ID（默认读 bench_config.toml）
\n  --friend F         友人卡 idrank（默认读 bench_config.toml / 303054）
\n  --seed S           育成推进基础种子（默认读 bench_config.toml / 61444）
\n  --build NAME       preset build 名（默认 speed，与固定根探针同口径）
\n  --deck \"id1,..,id5[,friend]\"   自定义卡组（覆盖 --build）
\n  --label TEXT       输出行标签（跨 commit 脚本用版本名标记）
\nroot 专用:
\n  --turn T           目标回合（默认 60；32 也是常用固定根）
\n  --search-n N       每候选 rollout 数（默认取 game_config.toml 的 mcts.search_n）
\n  --rounds R         轮数（默认 1；>1 时另输出中位数行）
\n  --search-seed S    搜索种子（默认 61444，每轮 +round）
\nwhole 专用:
\n  --runs N           局数（默认 100）
\n  --help, -h         显示本帮助"
    );
}

fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;

    let defaults = load_bench_defaults()?;
    let mut name = String::new();
    // 公共 CLI 覆盖（跨子命令共享）
    let mut uma = defaults.uma;
    let mut friend = defaults.friend;
    let mut seed = defaults.seed;
    let mut build = "speed".to_string();
    let mut deck: Option<String> = None;
    let mut label = String::new();
    // root 专用
    let mut turn: i32 = 60;
    let mut search_n: Option<usize> = None;
    let mut rounds: usize = 1;
    let mut search_seed: u64 = 61444;
    // whole 专用
    let mut runs: usize = 100;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Value(v) if name.is_empty() => {
                name = v.to_string_lossy().into_owned();
            }
            Arg::Value(v) => bail!("多余的位置参数: {v:?}（用法: perf_probe <root|whole> ...）"),
            Arg::Long("uma") => uma = bench::parse_value(&mut parser, "uma")?,
            Arg::Long("friend") => friend = bench::parse_value(&mut parser, "friend")?,
            Arg::Long("seed") => seed = bench::parse_value(&mut parser, "seed")?,
            Arg::Long("build") => build = bench::parse_value(&mut parser, "build")?,
            Arg::Long("deck") => deck = Some(bench::parse_value(&mut parser, "deck")?),
            Arg::Long("label") => label = bench::parse_value(&mut parser, "label")?,
            Arg::Long("turn") => turn = bench::parse_value(&mut parser, "turn")?,
            Arg::Long("search-n") => search_n = Some(bench::parse_value(&mut parser, "search-n")?),
            Arg::Long("rounds") => rounds = bench::parse_value(&mut parser, "rounds")?,
            Arg::Long("search-seed") => search_seed = bench::parse_value(&mut parser, "search-seed")?,
            Arg::Long("runs") => runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("help") | Arg::Short('h') => {
                print_help();
                return Ok(());
            }
            other => bail!("未知参数: {other:?}（--help 查看用法）"),
        }
    }

    // 与 bench_base 同口径：默认全部场景使用策略，region 交回策略（All）
    let mut game_config = load_game_config()?;
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;
    ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;
    let base_cfg = SearchConfig::new_game_config(&game_config);
    let search_n = search_n.unwrap_or(base_cfg.search_n);
    let search_cfg = base_cfg.with_search_n(search_n).with_max_depth(0);

    match name.as_str() {
        "root" => cmd_root(
            RootOpts {
                uma,
                friend,
                blue_count: defaults.blue_count,
                extra_count: defaults.extra_count,
                seed,
                build,
                deck,
                turn,
                rounds,
                search_seed,
                label
            },
            search_cfg
        ),
        "whole" => cmd_whole(WholeOpts {
            uma,
            friend,
            blue_count: defaults.blue_count,
            extra_count: defaults.extra_count,
            seed,
            build,
            deck,
            runs,
            label
        }),
        _ => bail!("未知子命令: {name:?}（可用 root / whole，--help 查看用法）")
    }
}