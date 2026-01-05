//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::time::Instant;

use anyhow::{Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use umasim::{
    game::{Game, InheritInfo, onsen::game::OnsenGame},
    gamedata::{GAMECONSTANTS, GameConfig, init_global},
    global,
    trainer::*,
    utils::init_logger
};
use umasim::search::SearchConfig;

/// 单次模拟结果
struct SimulationResult {
    score: i32,
    pt: i32,
    explain: String
}

/// 运行 OnsenGame（单次），返回模拟结果
fn run_onsen_once(
    trainer_mcts: MctsTrainer, uma: u32, cards: &[u32; 6], inherit: InheritInfo
) -> Result<SimulationResult> {
    let trainer_hand = HandwrittenTrainer::new().verbose(true);
    let mut rng = StdRng::from_os_rng();

    let mut game = OnsenGame::newgame(uma, cards, inherit)?;
    // 使用2个trainer完成一局游戏
    game.run_stage(&trainer_mcts, &mut rng)?;
    while game.next() {
        if game.pending_selection || !game.list_actions().unwrap_or_default().is_empty() {
            let mut game2 = game.clone();
            game2.run_stage(&trainer_hand, &mut rng)?;
        }
        game.run_stage(&trainer_mcts, &mut rng)?;        
    }
    game.on_simulation_end(&trainer_mcts, &mut rng)?;

    info!("育成结束！");

    let score = game.uma.calc_score();
    let pt = game.uma.skill_pt;
    let explain = game.explain()?;

    Ok(SimulationResult { score, pt, explain })
}

/// 打印多次模拟的统计结果
fn print_simulation_stats(results: &[SimulationResult], elapsed: std::time::Duration) {
    if results.is_empty() {
        return;
    }

    // 找到最高分和最低分
    let (best_idx, best) = results.iter().enumerate().max_by_key(|(_, r)| r.score).unwrap();
    let (worst_idx, worst) = results.iter().enumerate().min_by_key(|(_, r)| r.score).unwrap();

    // 计算平均分
    let avg_score = results.iter().map(|r| r.score as f64).sum::<f64>() / results.len() as f64;
    let avg_pt = results.iter().map(|r| r.pt as f64).sum::<f64>() / results.len() as f64;

    // 打印分隔线
    println!("\n{}", "=".repeat(60));
    println!("多次模拟统计 (共 {} 次)", results.len());
    println!("{}", "=".repeat(60));

    // 打印最高分结算面板
    println!("\n【最高分】第 {} 次模拟:", best_idx + 1);
    println!("{}", best.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(best.score),
        best.score,
        best.pt
    );

    // 打印最低分结算面板
    println!("\n【最低分】第 {} 次模拟:", worst_idx + 1);
    println!("{}", worst.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(worst.score),
        worst.score,
        worst.pt
    );

    // 打印统计摘要
    println!("\n{}", "-".repeat(60));
    println!(
        "平均评分: {} {:.0}, 平均PT+Hint: {:.0}",
        global!(GAMECONSTANTS).get_rank_name(avg_score as i32),
        avg_score,
        avg_pt
    );
    println!("总耗时: {:?}, 平均单次: {:?}", elapsed, elapsed / results.len() as u32);
    println!("{}", "=".repeat(60));
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 先读取配置文件
    let config_file = fs_err::read_to_string("game_config.toml")?;
    let game_config: GameConfig = toml::from_str(&config_file)?;

    // 2. 根据配置初始化日志
    init_logger("analyzer", &game_config.log_level)?;

    // 3. 再初始化全局数据
    init_global()?;

    //let simulation_count = game_config.simulation_count.max(1);
    let simulation_count = 1;
    // 开始计时
    let start = Instant::now();

    let inherit = InheritInfo {
        blue_count: game_config.blue_count.clone(),
        extra_count: game_config.extra_count.clone()
    };

    let search_config = SearchConfig::default()
        .with_search_n(game_config.mcts.search_n)
        .with_radical_factor_max(game_config.mcts.radical_factor_max)
        .with_max_depth(game_config.mcts.max_depth)
        .with_policy_delta(game_config.mcts.policy_delta)
        // UCB 参数
        .with_ucb(game_config.mcts.use_ucb)
        .with_search_group_size(game_config.mcts.search_group_size)
        .with_search_cpuct(game_config.mcts.search_cpuct)
        .with_expected_search_stdev(game_config.mcts.expected_search_stdev);
    info!("search_config = {search_config:?}");
    let mut trainer_mcts = MctsTrainer::new(search_config).verbose(true);

    let sim_result = run_onsen_once(trainer_mcts, game_config.uma, &game_config.cards, inherit)?;

    println!("{}", sim_result.explain);
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(sim_result.score),
        sim_result.score,
        sim_result.pt
    );
    Ok(())
}
