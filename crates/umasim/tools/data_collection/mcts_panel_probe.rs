//! MCTS 面板探针（speed / stamina 两个主流配卡，跨局并行）
//!
//! 为什么另写一个而不用 `bench_base --trainer mcts`：
//! `bench_base` 逐局串行，搜索内部只按候选动作并行（实测 24 核上有效并行度仅 7.7），
//! search_n=1024 时单局 ~1050s、20 局要 5.5 小时。本工具把「局」也放进 rayon，
//! 核利用率打满后同样 20 局约 100 分钟。
//!
//! 跑法：`MP_SEARCH_N=1024 MP_RF=0 MP_RUNS=10 cargo run --release --bin mcts_panel_probe`
//!
//! 环境变量：
//! - `MP_SEARCH_N` 每候选 rollout 数（缺省 1024）
//! - `MP_RF`       radical_factor_max（缺省 0）
//! - `MP_RUNS`     每个 build 的局数（缺省 10）
//! - `MP_SEED`     基础种子（缺省 61444，与 bench_config 一致）
//! - `MP_UCB`      是否开 UCB 自适应分配（缺省 0）
//! - `MP_GROUP`    UCB 每轮追加批量（缺省 32）。**必须远小于 search_n**：
//!                 `flat_search.rs:479` 会把 group 夹到 `min(group, search_n)`，
//!                 group >= search_n 时首轮就满足终止判据，自适应轮数为零、
//!                 结果与均匀分配逐位相同。
//! - `MP_STAGES`   搜索阶段（缺省 train,ramen）
//! - `MP_OUT`      CSV 输出目录（缺省 logs/mcts_panel）

use std::{env, sync::Mutex, time::Instant};

use anyhow::Result;
use rayon::prelude::*;
use umasim::{
    bench::{self, CardPickOpts, RESULTS_HEADER, load_player_builds, outcome_to_row},
    game::InheritInfo,
    gamedata::{RamenRegionStrategy, init_global_with_config},
    search::SearchConfig,
    trainer::{LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RamenSelection},
    utils::{get_workspace_root, load_game_config}
};

fn ev<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() -> Result<()> {
    let search_n: usize = ev("MP_SEARCH_N", 1024);
    let rf: f64 = ev("MP_RF", 0.0);
    let runs: u64 = ev("MP_RUNS", 10);
    let seed: u64 = ev("MP_SEED", 61444);
    let ucb: bool = ev::<u32>("MP_UCB", 0) > 0;
    let group: usize = ev("MP_GROUP", 32);
    let stages_s: String = ev("MP_STAGES", "train,ramen".to_string());
    let out_dir_s: String = ev("MP_OUT", "logs/mcts_panel".to_string());

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    let mut game_config = load_game_config()?;
    // 与 bench_base 对齐：基准测的是策略决策，Y3 地区必须交回策略枚举
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;

    // 只取两个主流配卡：3速1耐1智1友 / 2速2耐1智1友
    let targets = ["speed", "stamina"];
    let builds: Vec<_> =
        load_player_builds()?.into_iter().filter(|b| targets.contains(&b.name().as_str())).collect();
    anyhow::ensure!(builds.len() == 2, "未找到 speed / stamina build，实得 {}", builds.len());

    let pick = CardPickOpts::default();
    let friend: u32 = 303054;
    let decks: Vec<[u32; 6]> =
        builds.iter().map(|b| b.make_deck(&pick, friend)).collect::<Result<_>>()?;
    let inherit = InheritInfo { blue_count: [15, 0, 0, 0, 3], extra_count: [10, 10, 20, 20, 20, 40] };

    let search_config = SearchConfig::default()
        .with_search_n(search_n)
        .with_max_depth(0) // 拉面无 leaf 估值器，只能跑到终局
        .with_ucb(ucb)
        .with_radical_factor_max(rf);
    let search_config = SearchConfig { search_group_size: group, ..search_config };
    anyhow::ensure!(
        !ucb || group < search_n,
        "UCB 下 group({group}) 必须小于 search_n({search_n})，否则自适应轮数为零（见 flat_search.rs:479）"
    );
    let stages = RamenSearchStages::parse(&stages_s)?;

    println!(
        "mcts_panel_probe: search_n={search_n}/候选 rf={rf} stages={stages_s} runs={runs}/build \
         seed={seed} builds={targets:?} 线程={}",
        rayon::current_num_threads()
    );

    // (build_idx, run_idx) 全部摊平后一起并行：外层并行才能把 24 核吃满
    let jobs: Vec<(usize, u64)> =
        (0..builds.len()).flat_map(|bi| (0..runs).map(move |r| (bi, r))).collect();
    let total_jobs = jobs.len();
    let done = Mutex::new(0usize);
    let started = Instant::now();

    let mut outcomes: Vec<(usize, u64, bench::GameOutcome)> = jobs
        .into_par_iter()
        .map(|(bi, run_idx)| {
            let mcts = RamenMctsTrainer::new(search_config.clone())
                .with_stages(stages)
                .with_selection(RamenSelection::Score);
            let mut trainer = LoggingTrainer::new(mcts, seed + run_idx);
            trainer.set_logging(false);
            let o = bench::run_seeded(102601, &decks[bi], &inherit, seed, run_idx, &trainer)?;
            let mut d = done.lock().unwrap();
            *d += 1;
            println!(
                "  [{}/{}] {} #{:02} score={} 五维={:?} PT={} 累计耗时={:.0}s",
                *d,
                total_jobs,
                builds[bi].name(),
                run_idx + 1,
                o.score,
                o.five_status,
                o.skill_pt,
                started.elapsed().as_secs_f64()
            );
            Ok::<_, anyhow::Error>((bi, run_idx, o))
        })
        .collect::<Result<Vec<_>>>()?;
    outcomes.sort_by_key(|(bi, r, _)| (*bi, *r));

    // CSV：与 bench_base 同字段，便于用同一套脚本分析
    let out_dir = workspace_root.join(&out_dir_s);
    std::fs::create_dir_all(&out_dir)?;
    let mut csv = RESULTS_HEADER.join(",");
    csv.push('\n');
    for (bi, _, o) in &outcomes {
        csv.push_str(&outcome_to_row(&builds[*bi].name(), o).join(","));
        csv.push('\n');
    }
    let path = out_dir.join("mcts_panel_results.csv");
    std::fs::write(&path, csv)?;

    println!("\n===== 汇总 =====");
    for (bi, build) in builds.iter().enumerate() {
        let rows: Vec<_> = outcomes.iter().filter(|(b, _, _)| *b == bi).map(|(_, _, o)| o).collect();
        let n = rows.len() as f64;
        let mean = |f: &dyn Fn(&bench::GameOutcome) -> f64| rows.iter().map(|o| f(o)).sum::<f64>() / n;
        let five: Vec<f64> = (0..5).map(|i| mean(&|o| o.five_status[i] as f64)).collect();
        println!(
            "{:<9} n={} score={:.0} 五维=[{:.0} {:.0} {:.0} {:.0} {:.0}] 合计={:.0} PT={:.0} RMJ={:.2}",
            build.name(),
            rows.len(),
            mean(&|o| o.score as f64),
            five[0],
            five[1],
            five[2],
            five[3],
            five[4],
            five.iter().sum::<f64>(),
            mean(&|o| o.skill_pt as f64),
            mean(&|o| o.rmj_ok as f64)
        );
    }
    println!("CSV: {}", path.display());
    println!("总耗时: {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}
