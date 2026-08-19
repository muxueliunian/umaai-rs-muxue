//! 拉面杯玩家手动测试程序
//!
//! 用 `ManualTrainer::new()`（Interactive 模式，inquire 真实终端交互）启动一局拉面杯，
//! 让玩家手动选择每一个动作和事件选项，用于：
//! - 验证游戏机制实际表现
//! - 调试特定的回合/事件逻辑
//! - 体验完整的拉面杯流程
//!
//! # 启动方式
//!
//! ```bash
//! cargo run --bin ramen_manual --release
//! ```
//!
//! # 配置
//!
//! 启动时会读取 `game_config.toml`（参考 `gamedata/default_config.toml`）。
//! 本程序**只使用以下字段**（其他字段会被忽略或必须保持固定值）：
//!
//! - `log_level`：日志级别（`"info"` / `"debug"` / `"off"` 等）
//! - `uma`：马娘 ID
//! - `cards`：6 张支援卡 ID
//! - `extra_count`：种马额外属性 `[速度, 耐力, 力量, 根性, 智力, 技能点]`
//!
//! 本程序**强制要求**以下字段为固定值（不一致会报错）：
//!
//! - `scenario = "ramen"`
//! - `trainer = "manual"`
//!
//! 随机种子在 `SEED` 常量中定义（不放入 config）。

use std::time::Instant;

use anyhow::{Result, anyhow};
use log::info;
use rand::{SeedableRng, rngs::StdRng};

use umasim::{
    game::{
        Game, InheritInfo,
        ramen::RamenGame,
    },
    gamedata::{GAMECONSTANTS, init_global},
    global,
    trainer::ManualTrainer,
    utils::{init_logger_stdout, load_game_config},
};

/// 随机种子（固定种子便于复现）
const SEED: u64 = 20240816;

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
    init_global()?;

    // 5. 提取配置（只关心我们支持的字段，其他字段忽略）
    let uma_id = game_config.uma;
    let deck = game_config.cards;
    let inherit = InheritInfo {
        blue_count: game_config.blue_count,
        extra_count: game_config.extra_count,
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║        拉面杯 ManualTrainer 玩家测试          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("马娘: {}", uma_id);
    println!("卡组: {:?}", deck);
    println!("继承: blue={:?} extra={:?}", inherit.blue_count, inherit.extra_count);
    println!("种子: {}", SEED);
    println!("日志: {}", game_config.log_level);
    println!();
    println!("提示：每次操作都会弹出 inquire 选择菜单");
    println!("      上下键移动，回车确认，Ctrl+C 中断");
    println!();

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut game = RamenGame::newgame(uma_id, &deck, inherit)?;
    let trainer = ManualTrainer::new();

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
        game.ramen.feeling_stock[0],
        game.ramen.feeling_stock[1],
        game.ramen.feeling_stock[2]
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