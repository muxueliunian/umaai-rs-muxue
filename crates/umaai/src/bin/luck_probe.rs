//! 快照状态整局模拟探针（诊断工具）
//!
//! ## 用途
//!
//! 从 `--file` 指定的协议快照（`thisTurn.json` 系列）重建 `RamenGame`，用全手写策略
//! （`stages` 全关，等价 MCTS rollin 的 leaf 策略；`search_n` 无关）跑完整局，打印
//! 年度切换段（turn 22..=30）的 `scenario_pt` / 五维轨迹与终局结果。
//!
//! 典型用途：定位**两份快照之间的期望落差**（如年度 RMJ 结算段的运气分骤降）。
//! 可用 `--bonus` / `--rmj` / `--pt` 覆盖起始状态做对照实验，量化各项状态的影响：
//!
//! ```text
//! # 现状（协议重建） vs 补齐 RMJ 派生状态 vs 归零新年窗口遗留 PT
//! cargo run --release --bin luck_probe -- --file logs/SendGameStatusPlugin/game7075_turn24_2.json \
//!     --seed 100 --rollouts 40 --bonus 1 --rmj 1 --pt 0
//! ```
//!
//! ## 参数
//! - `--file`：快照路径（拉面剧本）
//! - `--seed`：起始随机种子（`--rollouts N` 时按 `seed+i` 依次取）
//! - `--rollouts`：重复整局模拟次数（取均值用）
//! - `--bonus` / `--rmj` / `--pt`：覆盖起始 `train_level_bonus` / `rmj_results`（逗号分隔布尔）/ `scenario_pt`（-1 = 不覆盖）

use std::{fs, path::PathBuf};

use anyhow::{Result, anyhow};
use lexopt::{Arg, ValueExt};
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{Game, Trainer, ramen::RamenGame},
    gamedata::init_global_with_config,
    search::SearchConfig,
    trainer::{RamenMctsTrainer, RamenSearchStages},
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

use umaai::protocol::{ParsedGame, parse_game_by_scenario};

struct Cli {
    file: PathBuf,
    seed: u64,
    rollouts: usize,
    /// 起始状态强行设置的 `train_level_bonus`（默认 -1 = 不动）
    bonus: i32,
    /// 起始状态强行设置的 `rmj_results`（空 = 不动；`y1s,y1f` 形式按年追加）
    rmj: Option<String>,
    /// 起始状态强行设置的 `scenario_pt`（默认 -1 = 不动）
    pt: i32
}

fn parse_cli() -> Result<Cli> {
    let mut cli = Cli { file: PathBuf::new(), seed: 42, rollouts: 1, bonus: -1, rmj: None, pt: -1 };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("file") => cli.file = PathBuf::from(parser.value()?.string()?),
            Arg::Long("seed") => cli.seed = parser.value()?.parse()?,
            Arg::Long("rollouts") => cli.rollouts = parser.value()?.parse()?,
            Arg::Long("bonus") => cli.bonus = parser.value()?.parse()?,
            Arg::Long("rmj") => cli.rmj = Some(parser.value()?.string()?),
            Arg::Long("pt") => cli.pt = parser.value()?.parse()?,
            _ => return Err(arg.unexpected().into())
        }
    }
    if cli.file.as_os_str().is_empty() {
        return Err(anyhow!("用法: luck-probe --file <json> [--seed N] [--rollouts N]"));
    }
    Ok(cli)
}

fn all_off_stages() -> RamenSearchStages {
    RamenSearchStages {
        train: false,
        ramen_select: false,
        special_select: false,
        region_select: false,
        super_ramen_select: false
    }
}

fn run_one(game: &mut RamenGame, trainer: &RamenMctsTrainer, rng: &mut StdRng, label: &str) -> Result<()> {
    // 自己驱动 run_stage + next，打印年度切换段轨迹（turn 22..=30）
    let mut guard = 0usize;
    loop {
        let turn = game.turn();
        if (22..=30).contains(&turn) {
            println!(
                "  {label} turn={turn:<3} stage={:?} scenario_pt={:>4} 五维={:?} skill_pt={} vital={}/{}",
                game.stage,
                game.ramen.scenario_pt,
                game.uma().five_status,
                game.uma().skill_pt,
                game.uma().vital,
                game.uma().max_vital
            );
        }
        game.run_stage(trainer, rng)?;
        if !game.next() {
            break;
        }
        guard += 1;
        if guard > 1200 {
            return Err(anyhow!("推进超 120 步仍未结束，疑似死循环"));
        }
    }
    game.on_simulation_end(trainer, rng)?;
    println!(
        "  {label} 终局: 回合={} 评分={:.0} 五维={:?} skill_pt={} scenario_pt={}",
        game.turn(),
        game.uma().calc_score(),
        game.uma().five_status,
        game.uma().skill_pt,
        game.ramen.scenario_pt
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[luck-probe] 错误: {e:?}");
            ExitCode::FAILURE
        }
    }
}

use std::process::ExitCode;

fn run() -> Result<()> {
    let cli = parse_cli()?;
    let ws_root = get_workspace_root()?;
    std::env::set_current_dir(&ws_root)?;
    let game_config = load_game_config()?;
    init_logger_stdout("luck_probe", "error")?;
    init_global_with_config(&game_config)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;

    // 全手写 fallback（rollin leaf 同策略），预算无关
    let trainer = RamenMctsTrainer::new(SearchConfig::default())
        .with_stages(all_off_stages())
        .verbose(false);

    let contents = fs::read_to_string(&cli.file)
        .map_err(|e| anyhow!("读取 {} 失败: {e}", cli.file.display()))?;
    let game = match parse_game_by_scenario(&contents)? {
        ParsedGame::Ramen { game, .. } => game,
        ParsedGame::Onsen(_) => return Err(anyhow!("仅支持拉面快照"))
    };
    println!(
        "起始状态: {}  turn={} stage={:?} scenario_pt={} 五维={:?} seed={}",
        cli.file.display(),
        game.turn(),
        game.stage,
        game.ramen.scenario_pt,
        game.uma().five_status,
        cli.seed
    );
    for i in 0..cli.rollouts {
        let (mut g, mut rng) = (game.clone(), StdRng::seed_from_u64(cli.seed + i as u64));
        if cli.bonus >= 0 {
            g.ramen.train_level_bonus = cli.bonus;
            println!("  [强制] train_level_bonus = {}", cli.bonus);
        }
        if cli.pt >= 0 {
            g.ramen.scenario_pt = cli.pt;
            println!("  [强制] scenario_pt = {}", cli.pt);
        }
        if let Some(ref spec) = cli.rmj {
            let results: Vec<bool> = spec.split(',').map(|x| x.trim() != "0").collect();
            println!("  [强制] rmj_results = {results:?}");
            g.ramen.rmj_results = results;
        }
        run_one(&mut g, &trainer, &mut rng, &format!("#{i}"))?;
    }
    Ok(())
}