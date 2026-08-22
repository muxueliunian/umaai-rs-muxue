//! 拉面杯玩家手动测试程序
//!
//! 用 `ManualTrainer::new()`（Interactive 模式，inquire 真实终端交互）启动一局拉面杯，
//! 让玩家手动选择每个动作与事件选项，用于验证机制 / 调试回合逻辑 / 体验完整流程。
//!
//! 启动：`cargo run --bin ramen_manual --release`
//!
//! 配置：读取 `game_config.toml`（参考 `gamedata/default_config.toml`），仅使用
//! `log_level` / `uma` / `cards` / `extra_count` 字段；强制 `scenario = "ramen"`、
//! `trainer = "manual"`（不一致报错）。随机种子每次从系统熵源生成（密码学种子，
//! 不打印、不复现）。
//!
//! 仅在 cli + diag feature 下编译（`required-features`，文件级 cfg gate 防误用）。
#![cfg(all(feature = "cli", feature = "diag"))]

use std::time::Instant;

use anyhow::{Result, anyhow};
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{Game, InheritInfo, ramen::RamenGame},
    gamedata::{GAMECONSTANTS, init_global_with_config},
    global,
    output::RecordingTrainer,
    trainer::ManualTrainer,
    utils::{init_logger_stdout, load_game_config}
};

fn main() -> Result<()> {
    // 1. 加载 game_config.toml（与 default_config.toml 合并）
    //    注意：本程序依赖从 workspace 根目录运行（与 umasim 主程序一致），
    //    否则找不到 `gamedata/default_config.toml` 和 `game_config.toml`
    let game_config = load_game_config()?;

    // 3. 校验固定字段（scenario / trainer）
    if game_config.scenario != "ramen" {
        return Err(anyhow!(
            "ramen_manual 要求 scenario = \"ramen\"，当前 game_config.toml 中为 {:?}\n\
            请修改 game_config.toml：scenario = \"ramen\"",
            game_config.scenario
        ));
    }
    if game_config.trainer != "manual" {
        return Err(anyhow!(
            "ramen_manual 要求 trainer = \"manual\"，当前 game_config.toml 中为 {:?}\n\
            请修改 game_config.toml：trainer = \"manual\"",
            game_config.trainer
        ));
    }

    // 4. 日志输出到 stdout（玩家场景）：
    // - 日志与 println! 一起显示，玩家直接看到训练/事件信息
    // - inquire 默认从 /dev/tty 读取，与 stdout 日志互不干扰
    // - 不写文件，需要持久化日志可用 shell 重定向: `cargo run --bin ramen_manual --release 2>&1 | tee ramen.log`
    init_logger_stdout("ramen_manual", &game_config.log_level)?;
    init_global_with_config(&game_config)?;

    // 5. 提取配置（只关心我们支持的字段，其他字段忽略）
    let uma_id = game_config.uma;
    let deck = game_config.cards;
    let inherit = InheritInfo {
        blue_count: game_config.blue_count,
        extra_count: game_config.extra_count
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║        拉面杯 ManualTrainer 玩家测试          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("马娘: {}", uma_id);
    println!("卡组: {:?}", deck);
    println!("继承: blue={:?} extra={:?}", inherit.blue_count, inherit.extra_count);
    println!("日志: {}", game_config.log_level);
    println!();
    println!("提示：每次操作都会弹出 inquire 选择菜单");
    println!("      上下键移动，回车确认，Ctrl+C 中断");
    println!();

    let mut rng = StdRng::from_os_rng();
    let mut game = RamenGame::newgame(uma_id, &deck, inherit)?;
    // RecordingTrainer 包装 ManualTrainer：verbose 实时输出候选列表（选项名亮黄 +
    // 内联预览白色）与选择确认，同时记录全部决策供后续分析
    let mut trainer = RecordingTrainer::new(ManualTrainer::new());
    trainer.verbose = true;

    println!("=== 开局状态 ===");
    println!("{}", game.explain()?);
    println!();

    let start = Instant::now();
    info!("开始 ManualTrainer 手动模拟...");
    game.run_full_game(&trainer, &mut rng)?;
    let elapsed = start.elapsed();

    println!();
    println!("╔══════════════════════════════════════════════╗");
    println!("║              育成结束！                       ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("最终回合: {} (max_turn={})", game.turn(), game.max_turn());
    println!("剧本PT: {}", game.ramen.scenario_pt);
    println!("RMJ结果: {:?}", game.ramen.rmj_results);
    println!("地区选择: {:?}", game.ramen.selected_regions);
    println!("超级拉面选择: {:?}", game.ramen.super_ramen);
    println!(
        "诀窍库存: A={} B={} C={}",
        game.ramen.feeling_stock[0], game.ramen.feeling_stock[1], game.ramen.feeling_stock[2]
    );
    println!("隐藏风味: {}", game.ramen.special_feeling);

    let score = game.uma.calc_score();
    let pt = game.uma.total_pt();
    println!(
        "评分: {} {}, PT: {}",
        global!(GAMECONSTANTS).get_rank_name(score),
        score,
        pt
    );
    println!("耗时: {:?}", elapsed);

    Ok(())
}
