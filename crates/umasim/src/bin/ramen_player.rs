//! 拉面杯玩家手动决策记录器
//!
//! 玩家用 `ManualTrainer::new()`（Interactive 模式，inquire 终端交互）手动跑一局，
//! 同时复用 [`umasim::bench`] 的 `GameOutcome` / CSV 输出，与 bench_base 统计口径一致。
//!
//! 用途：收集真实玩家的年度吃面次数、隐藏风味使用、地区选择、训练分布等决策模式，
//! 为手写策略调优提供"手动基准"。
//!
//! # 用法
//!
//! ```text
//! cargo run --release --bin ramen_player -- --build <name> [--seed N] [--out DIR]
//! ```
//!
//! 必选 `--build`：从 workspace 根目录 `bench_config.toml` 的 `[player_builds]` 段
//! 取对应 build 名（如 `speed` / `wisdom` / `sta0_wis2`），自动从 bench 代表卡池
//! 生成 6 张卡（5 张普通卡 + 1 张友人卡）；其他参数与 bench_base 一致。
//!
//! 跑完输出：
//! - `logs/ramen_player_<build>_<seed>.csv`：与 bench_base 同款 30+ 字段 GameOutcome 行
//! - `logs/ramen_player_decisions_<build>_<seed>.csv`：玩家每回合决策（候选 + 选择）
//!
//! 仅在 cli + diag feature 下编译。
#![cfg(all(feature = "cli", feature = "diag"))]

use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use lexopt::Arg;
use umasim::{
    bench::{
        self, CardPickOpts, GameOutcome, RESULTS_HEADER, outcome_to_row, seeded_rngs, write_csv
    },
    game::{Game, InheritInfo, ramen::RamenGame},
    gamedata::{GAMECONSTANTS, GAMEDATA, init_global_with_config},
    global,
    output::{RecordingTrainer, decision_log::DecisionLog},
    trainer::ManualTrainer,
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

/// 与 bench_config.toml 默认值对齐
const DEFAULT_UMA: u32 = 102601;
const DEFAULT_FRIEND: u32 = 303054;
const DEFAULT_BLUE_COUNT: [i32; 5] = [15, 0, 0, 0, 3];
const DEFAULT_EXTRA_COUNT: [i32; 6] = [10, 10, 20, 20, 20, 40];
const DEFAULT_SEED: u64 = 61444;

/// CLI 参数
struct Args {
    /// build 名（必填），从 bench_config.toml 的 player_builds 段读取
    build: String,
    /// 马娘 ID（默认 102601；与 bench_base / game_config.toml 同源）
    uma: u32,
    /// 基础种子（默认 61444；同 bench_base）
    seed: u64,
    /// 输出目录（默认 "logs"）
    out_dir: String
}

fn parse_args() -> Result<Args> {
    let mut build: Option<String> = None;
    let mut uma = DEFAULT_UMA;
    let mut seed = DEFAULT_SEED;
    let mut out_dir = "logs".to_string();
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("build") => build = Some(bench::parse_value(&mut parser, "build")?),
            Arg::Long("uma") => uma = bench::parse_value(&mut parser, "uma")?,
            Arg::Long("seed") => seed = bench::parse_value(&mut parser, "seed")?,
            Arg::Long("out") => out_dir = bench::parse_value(&mut parser, "out")?,
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: ramen_player --build <name> [--uma ID] [--seed N] [--out DIR]\n\
                     --build    必填，build 名从 bench_config.toml 的 [player_builds] 段读取（如 speed / wisdom / sta0_wis2）\n\
                     --uma      可选，马娘 ID（默认 {DEFAULT_UMA}）\n\
                     --seed     可选，基础种子（默认 {DEFAULT_SEED}）\n\
                     --out      可选，输出目录（默认 logs）"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("未知参数: {other:?}（可用 --help 查看用法）")
        }
    }
    let Some(build) = build else {
        anyhow::bail!("必填 --build <name>（如 speed / wisdom / sta0_wis2）");
    };
    Ok(Args { build, uma, seed, out_dir })
}

fn main() -> Result<()> {
    // 工作目录到 workspace 根（游戏数据 + bench_config.toml 都按相对路径定位）
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到 workspace 根失败: {}", workspace_root.display()))?;

    let args = parse_args()?;

    // ===== 初始化全局数据 =====
    // 输出 diag 日志到 stdout：与 ramen_manual 同款，玩家能看见训练/事件/吃面落地的诊断信息
    let game_config = load_game_config()?;
    init_logger_stdout("ramen_player", &game_config.log_level)?;
    init_global_with_config(&game_config)?;
    let data = global!(GAMEDATA);

    // ===== 找 build，按代表卡池拼卡组 =====
    let builds = bench::load_player_builds()?;
    let build = builds
        .iter()
        .find(|b| b.name() == args.build)
        .ok_or_else(|| {
            let names: Vec<_> = builds.iter().map(|b| b.name()).collect();
            anyhow!("build {:?} 不存在；可选: {:?}", args.build, names)
        })?;
    let pick = CardPickOpts::default();
    let reps = bench::select_representatives(&pick)?;
    let deck = build.build_deck(&reps.picked, DEFAULT_FRIEND)?;
    let card_desc_fn = |id: u32| -> String {
        data.get_card(id / 10)
            .map(|c| format!("{} {}", id, c.card_name))
            .unwrap_or_else(|_| id.to_string())
    };
    let deck_desc: Vec<String> = deck.iter().map(|id| card_desc_fn(*id)).collect();

    let inherit = InheritInfo {
        blue_count: DEFAULT_BLUE_COUNT,
        extra_count: DEFAULT_EXTRA_COUNT
    };

    // ===== 跑一局：固定种子、玩家手动决策 =====
    let (mut rng, rule_master) = seeded_rngs(args.seed, 0);

    println!("╔══════════════════════════════════════════════╗");
    println!("║       拉面杯 玩家手动决策 → CSV 记录器          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    let uma_name = data.get_uma(args.uma)?.name.clone();
    println!("马娘: {} (id={})", uma_name, args.uma);
    println!(
        "build: {} ({})",
        build.name(),
        {
            let cs = &build.counts;
            const N: [&str; 5] = ["速", "耐", "力", "根", "智"];
            (0..5)
                .filter(|&i| cs[i] > 0)
                .map(|i| format!("{}{}", cs[i], N[i]))
                .collect::<Vec<_>>()
                .join("+")
        }
    );
    println!("卡组: [{}]", deck_desc.join(", "));
    println!("继承: blue={:?} extra={:?}", inherit.blue_count, inherit.extra_count);
    println!("种子: {rule_master}（base_seed={}, run_idx=0）", args.seed);
    println!(
        "CSV: logs/ramen_player_{}_{}_{}.csv（与 bench_base 同字段）",
        build.name(),
        args.uma,
        rule_master
    );
    println!();
    println!("提示：每回合弹出 inquire 候选菜单；上下键移动，回车确认");
    println!("      Ctrl+C 中断（已跑回合数据不会被保存）");
    println!();

    let mut game = RamenGame::newgame(args.uma, &deck, inherit.clone())?;
    game.set_rule_master(rule_master);

    // RecordingTrainer 包裹 ManualTrainer：
    //   - verbose=true：每回合实时打印候选列表（含选项名亮黄、内联预览）与选择确认
    //     ——与 ramen_manual 同款，给玩家完整的局面信息做决策
    //   - log：全部决策记录到内部 Vec，供赛后导出
    // 真实玩家交互模式（与 ramen_manual 同款，inquire 真实终端菜单）
    let mut trainer = RecordingTrainer::new(ManualTrainer::new());
    trainer.verbose = true;

    println!("=== 开局 ===");
    println!("{}", game.explain()?);
    println!();

    let start = Instant::now();
    game.run_full_game(&trainer, &mut rng)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!();
    println!("=== 育成结束 ===");

    // ===== 构造 GameOutcome（与 bench_base 同一口径） =====
    let score = game.uma.calc_score();
    let outcome = GameOutcome {
        seed: rule_master,
        score,
        rank: global!(GAMECONSTANTS).get_rank_name(score),
        five_status: game.uma.five_status,
        skill_pt: game.uma.skill_pt,
        yearly_scenario_pt: game.ramen.yearly_scenario_pt,
        rmj_ok: game.ramen.rmj_results.iter().filter(|&&ok| ok).count(),
        yearly_eat_count: game.ramen.yearly_eat_count,
        yearly_selected_regions: game.ramen.yearly_selected_regions,
        yearly_friend_turns: game.ramen.yearly_friend_turns,
        yearly_gauge_gain: game.ramen.yearly_gauge_gain,
        yearly_gauge_overflow: game.ramen.yearly_gauge_overflow,
        friend_all: game.friend.out_used.iter().all(|used| *used),
        free_race_ok: game.uma.all_free_races_done()?,
        elapsed_ms
    };

    // ===== 写 CSV =====
    let out_dir_path = workspace_root.join(&args.out_dir);
    std::fs::create_dir_all(&out_dir_path)?;
    let csv_path = out_dir_path.join(format!(
        "ramen_player_{}_{}_{}.csv",
        build.name(),
        args.uma,
        rule_master
    ));
    let row = outcome_to_row(&build.name(), &outcome);
    write_csv(&csv_path, &RESULTS_HEADER, std::slice::from_ref(&row))?;

    // ===== 决策轨迹（与 bench_base --log 同款字段） =====
    let decisions: Vec<_> = trainer.log.borrow().iter().cloned().collect();
    if !decisions.is_empty() {
        // 把 RecordingTrainer 的 TurnDecision 转成 bench_base 兼容的 DecisionLogRow
        let mut log = DecisionLog::new();
        for d in &decisions {
            // RecordingTrainer 的 candidates 是选项名 Vec；DecisionLogRow 没有 Vec<String>
            // ——把候选拍平成单列 `"1.候选1 2.候选2 3.候选3"`
            let candidates_desc = d
                .candidates
                .iter()
                .enumerate()
                .map(|(i, n)| format!("{}.{}", i + 1, n))
                .collect::<Vec<_>>()
                .join(" | ");
            // recipe + detail 合并进 action_desc 后段
            let detail_tail = match (d.candidate_recipes.first().is_some(), d.candidate_details.first().is_some()) {
                (true, _) | (_, true) if !d.candidate_recipes.is_empty() || !d.candidate_details.is_empty() => {
                    let mut parts = Vec::new();
                    if !d.candidate_recipes.is_empty() {
                        parts.push(format!("recipes={}", d.candidate_recipes.join("/")));
                    }
                    if !d.candidate_details.is_empty() {
                        parts.push(format!("details={}", d.candidate_details.join("/")));
                    }
                    format!(" [{}]", parts.join(", "))
                }
                _ => String::new()
            };
            use umasim::output::decision_log::DecisionLogRow;
            log.rows.push(DecisionLogRow {
                seed: args.seed,
                turn: d.turn,
                stage: d.stage.clone(),
                candidates: d.candidates.len(),
                action_index: d.selected,
                action_desc: format!("{}{}", d.selected_desc, detail_tail),
                elapsed_us: 0,
                score_breakdown: Some(candidates_desc)
            });
        }
        let log_path = out_dir_path.join(format!(
            "ramen_player_decisions_{}_{}_{}.csv",
            build.name(),
            args.uma,
            rule_master
        ));
        log.save_to(&log_path)?;
        println!();
        println!("CSV 已写入: {}", csv_path.display());
        println!("决策轨迹:   {} ({} 项)", log_path.display(), log.rows.len());
    } else {
        println!();
        println!("CSV 已写入: {}", csv_path.display());
        println!("（未记录到决策——可手动 --no-verbose 排查）");
    }

    // ===== 简要统计 / 屏幕反馈 =====
    println!();
    println!(
        "评分: {} {}  五维={:?}  skill_pt={}",
        outcome.rank, outcome.score, outcome.five_status, outcome.skill_pt
    );
    println!(
        "逐年 PT:        {:?} / {:?} / {:?}",
        outcome.yearly_scenario_pt[0],
        outcome.yearly_scenario_pt[1],
        outcome.yearly_scenario_pt[2]
    );
    println!(
        "逐年 吃面次数:  {:?} / {:?} / {:?}  （合计 {}）",
        outcome.yearly_eat_count[0],
        outcome.yearly_eat_count[1],
        outcome.yearly_eat_count[2],
        outcome.yearly_eat_count.iter().sum::<i32>()
    );
    println!(
        "RMJ 成功: {}/3   自选比赛达标: {}   友人出行全完成: {}",
        outcome.rmj_ok, outcome.free_race_ok, outcome.friend_all
    );
    println!(
        "逐年 地区: Y1={:?}  Y2={:?}  Y3={:?}",
        outcome.yearly_selected_regions[0],
        outcome.yearly_selected_regions[1],
        outcome.yearly_selected_regions[2]
    );
    println!(
        "逐年 诀窍溢出(feeling_stock): {:?} / {:?} / {:?}",
        outcome.yearly_gauge_overflow[0],
        outcome.yearly_gauge_overflow[1],
        outcome.yearly_gauge_overflow[2]
    );
    println!("耗时: {:.0}ms", elapsed_ms);

    Ok(())
}
