//! MCTS rollout 切换 [`RecommendedRamenTrainer`] 后的快速验证
//!
//! 对照组：
//! - pure recommended：单 REC trainer 跑局（搜索关闭 / MCTS(stages=none) 等价）
//! - mcts train_only：MCTS 开启 train 阶段搜索，rollout/fallback 都是 REC
//! - mcts full：MCTS 全阶段开启搜索
//!
//! 输出每档的均值/标准差/最小/最大，便于快速看出切到 REC rollout 后搜索排序
//! 是否真的比手写策略提分。

use std::{collections::BTreeMap, env};

use anyhow::Result;
use umasim::{
    bench, game::{InheritInfo, ramen::RamenGame}, gamedata::init_global_with_config, search::{SearchConfig}, trainer::{
        LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RamenSelection, RecommendedRamenTrainer
    }, utils::{get_workspace_root, load_game_config}
};

const UMA: u32 = 102_601;
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};

fn run_one<T>(label: &'static str, make_trainer: impl Fn(u64) -> LoggingTrainer<T>, runs: u64, base_seed: u64)
    -> Result<(f64, f64, i32, i32)>
where
    T: umasim::game::Trainer<RamenGame> + Send + Sync
{
    let mut scores = Vec::with_capacity(runs as usize);
    let mut total_ms = 0.0;
    for run_idx in 0..runs {
        let trainer = make_trainer(run_idx);
        let outcome = bench::run_seeded(UMA, &DECK, &INHERIT, base_seed, run_idx, &trainer)?;
        scores.push(outcome.score);
        total_ms += outcome.elapsed_ms;
    }
    let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / scores.len() as f64;
    let stdev = {
        let var = scores.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        var.sqrt()
    };
    let min = *scores.iter().min().unwrap();
    let max = *scores.iter().max().unwrap();
    println!(
        "{label:30} mean={mean:8.0} std={stdev:6.0} min={min:6} max={max:6} | per_game_ms={:.1}",
        total_ms / runs as f64
    );
    Ok((mean, stdev, min, max))
}

fn main() -> Result<()> {
    let runs: u64 = env::var("MCTS_VERIFY_RUNS").unwrap_or_else(|_| "30".into()).parse()?;
    let base_seed: u64 = env::var("MCTS_VERIFY_SEED").unwrap_or_else(|_| "61444".into()).parse()?;

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(workspace_root)?;
    init_global_with_config(&load_game_config()?)?;

    println!("MCTS rollout switch verify: runs={runs} base_seed={base_seed} build=speed");

    // Pure REC
    let (mean_rec, _, _, _) = run_one("RecommendedRamenTrainer", |seed| {
        LoggingTrainer::new(RecommendedRamenTrainer::new(), seed)
    }, runs, base_seed)?;

    // MCTS stages=none（应是纯 REC 逐位等价）
    let search_cfg = SearchConfig::default().with_search_n(8);
    let (mean_mcts_none, _, _, _) = run_one("MCTS stages=none", |seed| {
        LoggingTrainer::new(
            RamenMctsTrainer::new(search_cfg.clone()).with_stages(RamenSearchStages::none()),
            seed
        )
    }, runs, base_seed)?;

    // MCTS train_only（rollout = REC, fallback = REC）
    let (mean_mcts_train, _, _, _) = run_one("MCTS train_only (search_n=128)", |seed| {
        LoggingTrainer::new(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(128).with_ucb(false))
                .with_stages(RamenSearchStages::train_only()),
            seed
        )
    }, runs, base_seed)?;

    // MCTS train+ramen
    let (mean_mcts_tr, _, _, _) = run_one("MCTS train+ramen (search_n=128)", |seed| {
        LoggingTrainer::new(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(128).with_ucb(false))
                .with_stages(RamenSearchStages {
                    train: true,
                    ramen_select: true,
                    ..RamenSearchStages::none()
                }),
            seed
        )
    }, runs, base_seed)?;

    println!("\n========== 提升幅度 ==========");
    println!("MCTS(none) vs REC:    {:+.0}", mean_mcts_none - mean_rec);
    println!("MCTS(train) vs REC:   {:+.0}", mean_mcts_train - mean_rec);
    println!("MCTS(train+ramen) vs REC: {:+.0}", mean_mcts_tr - mean_rec);

    Ok(())
}