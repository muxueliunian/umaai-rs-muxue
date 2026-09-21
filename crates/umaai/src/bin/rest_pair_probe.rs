//! 单回合诊断：对比「含不吃面」与「屏蔽不吃面」合并候选的搜索均值。
//!
//! 用法：`cargo run --release --bin rest_pair_probe -- <turn.json> [--search-n 8192]`
//!
//! 输出：两组的每个候选 mean / count / 选中动作。
//! 目的：验证 MCTS 在高体力合宿回合偏好"不吃面"的根因——是合并候选里的"不吃面"
//! 真优于任何吃面组合，还是被低估/高估。

use std::{fs, path::PathBuf, process::ExitCode, time::Instant};

use anyhow::{Result, anyhow};
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{Game, Trainer, ramen::RamenAction, ramen::RamenStage},
    gamedata::init_global_with_config,
    search::SearchConfig,
    trainer::RamenMctsTrainer,
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

use umaai::protocol::{ParsedGame, parse_game_by_scenario};

struct CliArgs {
    json_path: PathBuf,
    search_n: Option<usize>
}

fn parse_cli() -> Result<CliArgs> {
    let mut positional = Vec::new();
    let mut search_n = None;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        if a == "--search-n" {
            search_n = Some(iter.next().ok_or_else(|| anyhow!("--search-n 需要值"))?.parse()?);
        } else if a.starts_with("--search-n=") {
            search_n = Some(a["--search-n=".len()..].parse()?);
        } else {
            positional.push(a);
        }
    }
    match positional.as_slice() {
        [path] => Ok(CliArgs { json_path: PathBuf::from(path), search_n }),
        _ => Err(anyhow!("用法: rest_pair_probe <turn.json> [--search-n N]"))
    }
}

fn ensure_workspace_cwd() -> Result<()> {
    let root = get_workspace_root()?;
    std::env::set_current_dir(&root)?;
    Ok(())
}

/// 在指定候选集上跑 FlatSearch，打印 mean / n / 选中
fn run_search(trainer: &mut RamenMctsTrainer, game: &mut umasim::game::ramen::RamenGame,
              candidates: Vec<RamenAction>, rng: &mut StdRng, tag: &str) -> Result<()> {
    let t0 = Instant::now();
    let output = trainer.search.search(game, &candidates, rng)?;
    let elapsed = t0.elapsed().as_millis();
    println!("\n=== {} ({} 候选, {} ms) ===", tag, candidates.len(), elapsed);
    let mut rows: Vec<(usize, f64, u32, String)> = output
        .action_results
        .iter()
        .enumerate()
        .map(|(i, (r, _rpt))| (i, r.mean(), r.count(), candidates.get(i).map(|a| a.to_string()).unwrap_or_default()))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let best_idx = output.best_action_pt_idx();
    for (i, mean, n, desc) in &rows {
        let mark = if *i == best_idx { " <-- 选中" } else { "" };
        println!("  #{:>2} mean={:>10.2}  n={:>5}  {}{}", i, mean, n, desc, mark);
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[rest_pair_probe] 错误: {e:?}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<()> {
    let cli = parse_cli()?;
    ensure_workspace_cwd()?;
    let game_config = load_game_config()?;
    init_logger_stdout("rest_pair_probe", &game_config.log_level)?;
    init_global_with_config(&game_config)?;
    let contents = fs::read_to_string(&cli.json_path)?;
    println!("载入 JSON：{}", cli.json_path.display());
    let mut game = match parse_game_by_scenario(&contents) {
        Ok(ParsedGame::Ramen { game, .. }) => game,
        _ => return Err(anyhow!("需要 scenarioId=14 的拉面 JSON")),
    };
    println!(
        "回合 {} stage={:?} vital={}/{} scenario_pt={} regions={:?} special_feeling={}",
        game.turn(), game.stage, game.uma().vital, game.uma().max_vital,
        game.ramen.scenario_pt, game.ramen.selected_regions, game.ramen.special_feeling
    );
    if game.stage != RamenStage::RamenSelect {
        return Err(anyhow!("当前 stage={:?}，本工具要求 RamenSelect", game.stage));
    }

    let mut mcts_config = SearchConfig::new_game_config(&game_config);
    if let Some(n) = cli.search_n { mcts_config.search_n = n; }
    println!("search_n={}", mcts_config.search_n);

    let mut trainer = RamenMctsTrainer::new(mcts_config);
    let mut rng = StdRng::from_os_rng();

    // A) 完整合并候选（含"不吃面"）
    let combined_full = game.list_combined_ramen_select_actions();
    let mut game_a = game.clone();
    run_search(&mut trainer, &mut game_a, combined_full.clone(), &mut rng, "A. 完整候选（含不吃面）")?;

    // B) 屏蔽掉"不吃面"——只看吃面候选
    let only_eat: Vec<RamenAction> = combined_full.into_iter().filter(|a| a.ramen.is_some()).collect();
    let mut game_b = game.clone();
    run_search(&mut trainer, &mut game_b, only_eat, &mut rng, "B. 仅吃面候选（屏蔽不吃面）")?;

    Ok(())
}
