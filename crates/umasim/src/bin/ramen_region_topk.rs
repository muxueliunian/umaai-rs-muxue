//! 拉面杯 region 决策点 top-K 候选统计 dump
//!
//! 回答「MCTS 排名靠前的地区选项均值差很小，但它们的方差分布是否相同」。
//! 复用 perf_probe 思路：手写策略推进到指定回合的 **RegionSelect** 决策点，
//! 在该点跑一次 FlatSearch，把所有候选（或 top-K）按 `mean` 排序后输出
//! `(original_idx, description, count, mean, stdev, weighted_mean, was_chosen)`，
//! 落 CSV + 打印汇总（top1-top2 mean gap / top-K stdev 分布 / mean 排序 vs
//! radical 加权排序的差异）。
//!
//! # 用法（Release，从 workspace 根运行）
//!
//! ```text
//! # 默认：turn=2（10 候选，便宜）/ 3 seed × 2 build × 1 run = 6 个 region 点
//! cargo run --release --bin ramen_region_topk
//!
//! # 扩展到 turn=23（120 候选，单点约 15-20s）
//! cargo run --release --bin ramen_region_topk -- --turns 2,23
//! ```
//!
//! # 性能注意
//!
//! - turn=2：10 候选 × search_n ≈ <2 秒/点（含整局推进）
//! - turn=23/47：120 候选 × search_n=8192 ≈ 15-20 秒/点（受 CPU/线程数影响）
//! - 生产 search_n 默认取 game_config [mcts] 实际值（与在线构造同款）
//! - `--search-n` 覆盖后不再代表生产训练员
//!
//! # 口径
//!
//! - 评分轴：mean 来自 `ActionResult::mean()`（原始 mean 口径，不带 radical_factor）
//! - PT 排序：`best_action_pt_idx` 走 `weighted_mean(radical_factor=1.4)`，
//!   与生产 `RamenMctsTrainer` 选择一致
//! - 推进策略：手写 `RecommendedRamenTrainer`（与 MCTS rollout 基策同源，
//!   保证 region 决策点状态与生产一致）

use std::env;

use anyhow::{Context, Result, ensure, bail};
use lexopt::Arg;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use umasim::{
    bench::{self, CardPickOpts, load_player_builds},
    game::{Game, InheritInfo, ramen::{RamenAction, RamenGame, RamenStage}},
    gamedata::{RamenRegionStrategy, init_global_with_config},
    search::{FlatSearch, SearchConfig},
    trainer::RecommendedRamenTrainer,
    utils::{get_workspace_root, load_game_config}
};

/// CLI 参数
#[derive(Debug, Clone)]
struct Args {
    /// RegionSelect 回合列表（默认 2；已知 2/23/47 是三年地区选择回合）
    turns: Vec<i32>,
    /// 基准种子列表
    seeds: Vec<u64>,
    /// build 名过滤（默认 speed）
    builds: Vec<String>,
    /// 每 (turn, seed, build) 重复次数
    runs: usize,
    /// dump 排名前 K 个候选
    top_k: usize,
    /// 覆盖 search_n（None = game_config 实际值）
    search_n: Option<usize>,
    /// 输出 CSV 路径
    out: String
}

impl Default for Args {
    fn default() -> Self {
        Self {
            turns: vec![2],
            seeds: vec![61444, 42, 7],
            builds: vec!["speed".to_string()],
            runs: 1,
            top_k: 5,
            search_n: None,
            out: "logs/ramen_region_topk.csv".to_string()
        }
    }
}

/// 解析 CLI 参数
fn apply_cli(mut a: Args) -> Result<Args> {
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("turns") => {
                let t: String = bench::parse_value(&mut parser, "turns")?;
                a.turns = t
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<i32>().with_context(|| format!("回合无效: {s}")))
                    .collect::<Result<Vec<_>>>()?;
            }
            Arg::Long("seeds") => {
                let t: String = bench::parse_value(&mut parser, "seeds")?;
                a.seeds = t
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u64>().with_context(|| format!("种子无效: {s}")))
                    .collect::<Result<Vec<_>>>()?;
            }
            Arg::Long("builds") => {
                let t: String = bench::parse_value(&mut parser, "builds")?;
                a.builds = t
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            Arg::Long("runs") => a.runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("top-k") => a.top_k = bench::parse_value(&mut parser, "top-k")?,
            Arg::Long("search-n") => a.search_n = Some(bench::parse_value(&mut parser, "search-n")?),
            Arg::Long("out") => a.out = bench::parse_value(&mut parser, "out")?,
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: ramen_region_topk [--turns 2,23,47] [--seeds S1,S2,...]
\n                              [--builds b1,b2,...] [--runs N] [--top-k K]
\n                              [--search-n N] [--out PATH]
\n  手写策略推进到指定回合的 RegionSelect 决策点 → FlatSearch 一次 →
\n  输出 top-K 候选的 (idx, description, count, mean, stdev, weighted_mean, was_chosen)，
\n  落 CSV + 打印汇总（top1-top2 mean gap / stdev 分布 / mean vs radical 排序差异）。"
                );
                std::process::exit(0);
            }
            other => bail!("未知参数: {other:?}（--help 查看用法）"),
        }
    }
    ensure!(!a.turns.is_empty(), "--turns 不能为空");
    ensure!(!a.seeds.is_empty(), "--seeds 不能为空");
    ensure!(!a.builds.is_empty(), "--builds 不能为空");
    ensure!(a.runs > 0, "--runs 必须为正");
    ensure!(a.top_k > 0, "--top-k 必须为正");
    Ok(a)
}

/// bench_config.toml 的基础字段
#[derive(Debug, Clone, Deserialize)]
struct BenchBase {
    uma: u32,
    friend: u32,
    blue_count: [i32; 5],
    extra_count: [i32; 6]
}

impl Default for BenchBase {
    fn default() -> Self {
        Self {
            uma: 102601,
            friend: 303054,
            blue_count: [15, 0, 0, 0, 3],
            extra_count: [10, 10, 20, 20, 20, 40]
        }
    }
}

fn load_bench_base(root: &std::path::Path) -> Result<BenchBase> {
    let path = root.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)?;
        Ok(toml::from_str(&text).with_context(|| format!("解析 {} 失败", path.display()))?)
    } else {
        Ok(BenchBase::default())
    }
}

/// 手写策略从固定种子推进到 `target_turn` 的 RegionSelect 决策点
///
/// 与 `perf_probe::advance_to_train_root` 同思路：仅阶段判定改为
/// `game.stage == RegionSelect && game.turn() == target_turn`，候选 >1 才视为有效
/// （region 阶段候选数固定 = C(5,3)=10 或 C(10,3)=120，正常情况下无单候选早退）。
fn advance_to_region_root(
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
            bail!("推进超过 1000 步仍未到达目标 region 根（turn={target_turn}）");
        }
        if game.turn() > target_turn {
            bail!(
                "回合越过目标（target={target_turn}，当前 turn={} stage={:?}）",
                game.turn(),
                game.stage
            );
        }
        if game.turn() == target_turn && game.stage == RamenStage::RegionSelect {
            let actions = game.list_actions()?;
            if actions.len() > 1 {
                return Ok((game.clone(), actions));
            }
        }
        game.run_stage(trainer, &mut decision_rng)?;
    }
    bail!("推进到终局仍未遇到 turn={target_turn} 的 RegionSelect 决策点（候选需 >1）")
}

/// 一次 region 决策点的搜索结果（每个候选 1 行）
#[derive(Debug, Clone)]
struct CandRow {
    /// 基准种子
    seed: u64,
    /// 局号（重复运行用）
    run: u64,
    /// build 名
    build: String,
    /// region 回合
    turn: i32,
    /// 年份（1/2/3，按 turn 推算）
    year: i32,
    /// 总候选数
    candidates_total: usize,
    /// 实际总 rollout 数
    searched_total: u64,
    /// 按 mean 降序的排名（1=top1）
    rank: usize,
    /// 原始 action 下标
    original_idx: usize,
    /// 候选描述（地区三元组等）
    description: String,
    /// 该候选的 rollout 次数
    count: u32,
    /// mean 轴
    mean: f64,
    /// 标准差（mean 轴）
    stdev: f64,
    /// radical 加权均值（与生产选择同款）
    weighted_mean: f64,
    /// 是否被生产选择（按 score_pt / radical 加权）选中
    was_chosen: bool
}

/// CSV 表头
const HEADER: [&str; 15] = [
    "seed",
    "run",
    "build",
    "turn",
    "year",
    "candidates_total",
    "searched_total",
    "rank",
    "original_idx",
    "description",
    "count",
    "mean",
    "stdev",
    "weighted_mean",
    "was_chosen"
];

fn row_to_csv(r: &CandRow) -> Vec<String> {
    vec![
        r.seed.to_string(),
        r.run.to_string(),
        r.build.clone(),
        r.turn.to_string(),
        r.year.to_string(),
        r.candidates_total.to_string(),
        r.searched_total.to_string(),
        r.rank.to_string(),
        r.original_idx.to_string(),
        r.description.clone(),
        r.count.to_string(),
        format!("{:.3}", r.mean),
        format!("{:.3}", r.stdev),
        format!("{:.3}", r.weighted_mean),
        u8::from(r.was_chosen).to_string()
    ]
}

/// 一次 region 决策点的搜索 → 一组 CandRow（按 mean 降序前 K）
fn dump_region_point(
    search: &FlatSearch<RamenGame>,
    game: &RamenGame,
    actions: &[RamenAction],
    seed: u64,
    run: u64,
    build: &str,
    turn: i32,
    top_k: usize,
    search_seed: u64
) -> Result<Vec<CandRow>> {
    let mut rng = StdRng::seed_from_u64(search_seed.wrapping_add(run));
    let output = search.search(game, actions, &mut rng)?;
    let chosen = output.best_action_pt_idx();
    let searched_total: u64 = output.action_results.iter().map(|(r, _)| r.count() as u64).sum();
    let candidates_total = actions.len();
    let year = if turn <= 23 { 1 } else if turn <= 47 { 2 } else { 3 };

    // 按 mean 降序排：取 (original_idx, mean) 对，排序，记录 mean_rank
    let mut indexed: Vec<(usize, f64)> = output
        .action_results
        .iter()
        .enumerate()
        .map(|(i, (r, _))| (i, r.mean()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = top_k.min(indexed.len());
    indexed
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(rank, (idx, _))| {
            let (res, res_pt) = &output.action_results[idx];
            Ok(CandRow {
                seed,
                run,
                build: build.to_string(),
                turn,
                year,
                candidates_total,
                searched_total,
                rank: rank + 1,
                original_idx: idx,
                description: actions[idx].to_string(),
                count: res.count(),
                mean: res.mean(),
                stdev: res.stdev(),
                weighted_mean: res_pt.weighted_mean(output.radical_factor),
                was_chosen: idx == chosen
            })
        })
        .collect()
}

fn year_of(turn: i32) -> i32 {
    if turn <= 23 {
        1
    } else if turn <= 46 {
        2
    } else {
        3
    }
}

fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    env::set_current_dir(&workspace_root)?;

    let args = apply_cli(Args::default())?;

    let mut game_config = load_game_config()?;
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;

    let base = load_bench_base(&workspace_root)?;
    let inherit = InheritInfo {
        blue_count: base.blue_count,
        extra_count: base.extra_count
    };

    let builds_all = load_player_builds()?;
    let pick = CardPickOpts::default();
    let mut jobs: Vec<(String, [u32; 6])> = Vec::new();
    for want in &args.builds {
        let b = builds_all
            .iter()
            .find(|b| &b.name == want)
            .ok_or_else(|| anyhow::anyhow!("build 不存在: {want}"))?;
        let deck = b.make_deck(&pick, base.friend)?;
        jobs.push((b.name(), deck));
    }

    // 搜索参数：默认 = 生产实际值（game_config [mcts]）
    let mut search_cfg = SearchConfig::new_game_config(&game_config).with_max_depth(0);
    if let Some(n) = args.search_n {
        search_cfg = search_cfg.with_search_n(n);
    }
    let search = FlatSearch::<RamenGame>::new(search_cfg.clone());
    let search_seed: u64 = 61444;

    println!(
        "===== ramen_region_topk: turns={:?} seeds={:?} builds={:?} runs={} top_k={} =====",
        args.turns, args.seeds, args.builds, args.runs, args.top_k
    );
    println!(
        "  MCTS 生效参数: search_n={} ucb={} radical={} group_size={} cpuct={}",
        search_cfg.search_n,
        search_cfg.use_ucb,
        search_cfg.radical_factor_max,
        search_cfg.search_group_size,
        search_cfg.search_cpuct
    );

    // 推进 + 搜索 + dump（顺序跑：每点 0.5-20s，没必要并行）
    let trainer = RecommendedRamenTrainer::new();
    let mut all_rows: Vec<CandRow> = Vec::new();
    for turn in &args.turns {
        for seed in &args.seeds {
            for (build_name, deck) in &jobs {
                for run in 0..args.runs {
                    let (game, actions) = advance_to_region_root(
                        &trainer,
                        base.uma,
                        deck,
                        &inherit,
                        *seed,
                        *turn
                    )
                    .with_context(|| format!("build={build_name} seed={seed} turn={turn}"))?;
                    println!(
                        "  turn={} year={} seed={seed} build={build_name} run={run}: 候选 {} 个 → 搜索...",
                        turn,
                        year_of(*turn),
                        actions.len()
                    );
                    let rows = dump_region_point(
                        &search,
                        &game,
                        &actions,
                        *seed,
                        run as u64,
                        build_name,
                        *turn,
                        args.top_k,
                        search_seed
                    )?;
                    // 单点摘要
                    if let (Some(top1), Some(top2)) = (rows.first(), rows.get(1)) {
                        println!(
                            "    top1 mean={:.0} stdev={:.0} | top2 mean={:.0} stdev={:.0} | Δmean={:.1} chosen_idx={} chosen_wmean={:.0}",
                            top1.mean,
                            top1.stdev,
                            top2.mean,
                            top2.stdev,
                            top1.mean - top2.mean,
                            rows.iter().find(|r| r.was_chosen).map(|r| r.original_idx).unwrap_or(0),
                            rows.iter().find(|r| r.was_chosen).map(|r| r.weighted_mean).unwrap_or(0.0)
                        );
                    }
                    all_rows.extend(rows);
                }
            }
        }
    }

    // 落 CSV
    let out_path = workspace_root.join(&args.out);
    let csv_rows: Vec<Vec<String>> = all_rows.iter().map(row_to_csv).collect();
    bench::write_csv(&out_path, &HEADER, &csv_rows)?;
    println!("\nCSV 已写入: {}", out_path.display());
    println!("共 {} 行（每个 region 点 × top_k）", all_rows.len());

    // 汇总：按 (turn, build) 分组的 top1 vs top2 mean gap + top-K stdev 中位数
    println!("\n===== 汇总（按 turn / build）=====");
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(i32, String), Vec<&CandRow>> = BTreeMap::new();
    for r in &all_rows {
        groups.entry((r.turn, r.build.clone())).or_default().push(r);
    }
    for ((turn, build), rs) in &groups {
        // 按 (seed, run) 聚合 → 每点取 top1 与 top2
        let mut points: BTreeMap<(u64, u64), Vec<&CandRow>> = BTreeMap::new();
        for r in rs {
            points.entry((r.seed, r.run)).or_default().push(*r);
        }
        let mut gaps = Vec::new();
        let mut stdevs_top1 = Vec::new();
        let mut stdevs_top2 = Vec::new();
        let mut rankswap = 0usize;
        let mut points_n = 0usize;
        for (_key, mut pt) in points {
            pt.sort_by_key(|r| r.rank);
            points_n += 1;
            if let (Some(t1), Some(t2)) = (pt.first(), pt.get(1)) {
                gaps.push(t1.mean - t2.mean);
                stdevs_top1.push(t1.stdev);
                stdevs_top2.push(t2.stdev);
                // 排序差异：mean_rank vs weighted_rank
                let mut sorted_by_w: Vec<&CandRow> = pt.clone();
                sorted_by_w.sort_by(|a, b| {
                    b.weighted_mean.partial_cmp(&a.weighted_mean).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (rank_pos, r) in sorted_by_w.iter().enumerate() {
                    if r.rank != rank_pos + 1 {
                        rankswap += 1;
                        break;
                    }
                }
            }
        }
        let med = |xs: &[f64]| -> f64 {
            if xs.is_empty() {
                0.0
            } else {
                let mut s = xs.to_vec();
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            }
        };
        let stdev_top1 = med(&stdevs_top1);
        let stdev_top2 = med(&stdevs_top2);
        let gap_med = med(&gaps);
        println!(
            "  Y{}/{}:  n={} median(top1.top2 Δmean)={:.1} median(top1 stdev)={:.0} median(top2 stdev)={:.0} mean 排序 vs radical 排序 有差异的点={}",
            year_of(*turn),
            build,
            points_n,
            gap_med,
            stdev_top1,
            stdev_top2,
            rankswap
        );
    }
    Ok(())
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

    /// 准备小预算搜索参数（不读 game_config）
    fn smoke_search() -> SearchConfig {
        SearchConfig::default().with_search_n(32).with_ucb(false)
    }

    /// 冒烟：推进到 turn 2 RegionSelect + dump top-K，验证工具链工作
    #[test]
    fn test_region_dump_smoke() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        env::set_current_dir(&workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let trainer = RecommendedRamenTrainer::new();
        let (game, actions) = advance_to_region_root(&trainer, TEST_UMA_ID, &TEST_DECK, &TEST_INHERIT, 42, 2)?;
        println!("turn={} stage={:?} 候选 {} 个", game.turn(), game.stage, actions.len());
        ensure!(game.stage == RamenStage::RegionSelect, "已到 RegionSelect 阶段");
        ensure!(actions.len() > 1, "region 候选 >1");

        let search = FlatSearch::<RamenGame>::new(smoke_search());
        let rows = dump_region_point(&search, &game, &actions, 42, 0, "test", 2, 5, 61444)?;
        println!("dump {} 行: {:?}", rows.len(), rows.iter().map(|r| (r.rank, r.original_idx, r.mean, r.stdev)).collect::<Vec<_>>());
        ensure!(!rows.is_empty(), "dump 出 top-K 非空");
        ensure!(rows[0].rank == 1, "首行 rank=1");
        ensure!(rows.len() == 5.min(actions.len()), "行数 == min(top_k, 候选数)");
        // stdev 非负（count>0 时）
        for r in &rows {
            ensure!(r.stdev >= 0.0, "stdev 非负");
            ensure!(r.count > 0, "候选至少 1 次 rollout");
        }
        Ok(())
    }

    /// CSV 行列结构与表头同长
    #[test]
    fn test_csv_row_matches_header() -> Result<()> {
        let row = CandRow {
            seed: 42,
            run: 0,
            build: "speed".to_string(),
            turn: 2,
            year: 1,
            candidates_total: 10,
            searched_total: 320,
            rank: 1,
            original_idx: 3,
            description: "test".to_string(),
            count: 32,
            mean: 12345.6,
            stdev: 789.0,
            weighted_mean: 13500.0,
            was_chosen: true
        };
        let csv = row_to_csv(&row);
        ensure!(csv.len() == HEADER.len(), "数据行列数 == 表头列数");
        ensure!(csv[7] == "1", "rank 列");
        ensure!(csv[8] == "3", "original_idx 列");
        ensure!(csv[9] == "test", "description 列");
        Ok(())
    }
}