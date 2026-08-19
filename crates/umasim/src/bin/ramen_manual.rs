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
//! - 卡组（6 张支援卡 ID）：直接修改 `DECK` 常量
//! - 马娘 ID：直接修改 `UMA_ID` 常量
//! - 继承属性：直接修改 `INHERIT` 常量
//! - 随机种子：直接修改 `SEED` 常量

use std::time::Instant;

use anyhow::Result;
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
    utils::{check_working_dir, init_logger_stdout},
};

// ========== 测试用配置（按需修改） ==========

/// 马娘 ID（默认：102601 美浦波旁）
const UMA_ID: u32 = 102601;

/// 测试用卡组（与单元测试一致）
/// [速]杏目, [智]青春永驻, [耐]名将怒涛, [速]洛林军歌, [速]里见光钻, [友]骏川手纲
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];

/// 继承属性
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 3, 0, 0, 0],
    extra_count: [0, 30, 0, 0, 30, 30],
};

/// 随机种子（固定种子便于复现）
const SEED: u64 = 20240816;

fn main() -> Result<()> {
    // 切换到工作目录（确保 gamedata 路径正确）
    check_working_dir()?;
    // 日志输出到 stdout（玩家场景）：
    // - 日志与 println! 一起显示，玩家直接看到训练/事件信息
    // - inquire 默认从 /dev/tty 读取，与 stdout 日志互不干扰
    // - 不写文件，需要持久化日志可用 shell 重定向: `cargo run --bin ramen_manual --release 2>&1 | tee ramen.log`
    init_logger_stdout("ramen_manual", "info")?;
    init_global()?;

    println!("╔══════════════════════════════════════════════╗");
    println!("║        拉面杯 ManualTrainer 玩家测试          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("马娘: {}", UMA_ID);
    println!("卡组: {:?}", DECK);
    println!("种子: {}", SEED);
    println!();
    println!("提示：每次操作都会弹出 inquire 选择菜单");
    println!("      上下键移动，回车确认，Ctrl+C 中断");
    println!();

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut game = RamenGame::newgame(UMA_ID, &DECK, INHERIT)?;
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