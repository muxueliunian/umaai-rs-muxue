//! 拉面杯单回合诊断工具（Step 7 验证）
//!
//! ## 用途
//!
//! 用 CLI 参数指定单个 ramen `thisTurn.json`（scenarioId=14），走完整协议 → RamenGame
//! 转换 → 用 `game_config.toml` 默认 `SearchConfig` 构造 `RamenMctsTrainer` → 在当前
//! `stage` 跑一次 `select_action` → 打印 human-readable 回合状态与 MCTS 决策。
//!
//! 验证目标：
//!
//! 1. **协议 → RamenGame 转换正确性**：`into_game()` 后关键字段（`scenario_pt` /
//!    `current_ramen` / `selected_regions` / `super_ramen` / `feeling_slot` 等）与
//!    协议 JSON 一致。
//! 2. **MCTS 主路径可工作**：`RamenMctsTrainer::select_action` 在当前 stage 跑通，
//!    `last_decision` 返回合法的 `DecisionInfo`（含 `candidate_scores` / `candidate_n`）。
//! 3. **单回合决策可视化**：让用户直观看到 AI 推荐了什么（候选评分、中选下标、
//!    终局维度差 reason）。
//!
//! ## 用法
//!
//! ```text
//! cargo run --release --bin ramen-turn-inspect -- logs/GameStatusSend_Ramen/game6204_turn12.json
//! ```
//!
//! ## 与 `test_turn_import_v2_full_samples` 的差异
//!
//! | 维度 | `test_turn_import_v2_full_samples`（unit test） | 本 binary |
//! |---|---|---|
//! | 样本规模 | 151 份批量 | 单 JSON |
//! | 跑 MCTS | 不跑（只验证 `into_game` 字段） | 跑 `select_action` |
//! | 输出 | 仅 round-trip 字段断言 | human-readable 回合状态 + 决策 |
//! | 用途 | 防回归（CI 友好） | 单回合人工排查（开发友好） |
//!
//! ## 设计原则
//!
//! - **不引入新 CLI 解析器**：用 `std::env::args()` 第一个位置参数作为 JSON 路径。
//!   集成文档 §3.2.3 规定不引入新 CLI 参数，但 diag binary 不在「AI 主流程」边界内，
//!   按 `ramen_manual` / `ramen_player` 的先例开独立 `bin/` 文件即可。
//! - **MCTS 预算走 game_config.toml 默认值**：与 main.rs / `ramen_manual` 一致，
//!   调预算改 `[mcts]` 段即可。
//! - **不污染 stdout JSON 流**：本工具永远走 stdout（人类可读），不存在
//!   `--json` 分流的需要。
//! - **依赖最小**：只用 `umasim::trainer::RamenMctsTrainer` + 通道层
//!   `parse_game_by_scenario` + `umasim::utils::{init_logger_stdout, load_game_config}`。
//! - **cwd 必须 workspace 根**：与 `ramen_manual` 同约束，方便 `gamedata/default_config.toml`
//!   与 `game_config.toml` 的相对路径解析。

use std::{fs, path::PathBuf, process::ExitCode, time::Instant};

use anyhow::{Result, anyhow};
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{Game, Trainer, ramen::RamenGame},
    gamedata::init_global_with_config,
    output::DecisionInfo,
    search::SearchConfig,
    trainer::RamenMctsTrainer,
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

use umaai::protocol::{ParsedGame, parse_game_by_scenario};

/// CLI 参数：第一个位置参数为 JSON 路径
///
/// 不解析 flag（保持依赖最小）；错误参数走 `print_usage_and_exit`。
struct CliArgs {
    json_path: PathBuf
}

fn parse_cli() -> Result<CliArgs> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&str> = args.iter().map(String::as_str).filter(|a| !a.starts_with('-')).collect();
    match positional.as_slice() {
        [path] => Ok(CliArgs { json_path: PathBuf::from(path) }),
        _ => Err(anyhow!("用法: ramen-turn-inspect <path/to/thisTurn.json>"))
    }
}

fn print_usage_and_exit() -> ExitCode {
    eprintln!("用法: ramen-turn-inspect <path/to/thisTurn.json>");
    eprintln!();
    eprintln!("参数:");
    eprintln!("  <path>    拉面剧本 thisTurn.json 路径（scenarioId=14）");
    eprintln!();
    eprintln!("示例:");
    eprintln!("  cargo run --release --bin ramen-turn-inspect -- logs/GameStatusSend_Ramen/game6204_turn12.json");
    ExitCode::from(2)
}

/// 切到 workspace 根（与 `ramen_manual` 同约束：`load_game_config` 走相对路径）
///
/// `get_workspace_root` 已经在 `umasim::utils` 暴露。
fn ensure_workspace_cwd() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    Ok(())
}

/// 打印 human-readable 回合状态（baseGame + ramen 段 + persons layout）
///
/// 输出形如：
/// ```text
/// === 回合状态 ===
/// turn=12/77  vital=85/120  motivation=4  skill_pt=1500  hints=3
/// scenario_pt=4200  current_ramen=Some(3)  selected_regions=[3, 7, 12]
/// super_ramen=None  feeling_slot=[2, 1, 0]  feeling_stock=[4, 3, 2]  special_feeling=1
/// train_feeling_type=Some([A, B, A, C, B])
/// stage=Settlement   (协议 playing_state 已 dispatch)
/// deck=[302424, 302894, 303044, 302924, 303024, 303054]
/// uma_id=100603  five_status=[320, 248, 268, 214, 182]
/// persons (13) = [
///   0: Card 速小栗.., 1: Card 速东海.., 2: Card 根黄.., 3: Card 智.., 4: Card 速杏..,
///   5: ScenarioCard 友駿川 (9001),
///   6: Yayoi,
///   7: Npc 1022, 8: Npc 1058, 9: Npc 1060, 10: Npc 1077, 11: Npc 1120,
///   12: Reporter
/// ]
/// deck_can_split = true
/// ```
fn print_turn_state(game: &RamenGame) {
    let ramen = &game.ramen;
    println!("=== 回合状态 ===");
    println!("turn={}/{}  vital={}/{}  motivation={}  skill_pt={}  hints={}",
        game.turn(), game.max_turn(),
        game.uma().vital, game.uma().max_vital,
        game.uma().motivation, game.uma().skill_pt, game.uma().total_hints);
    println!("scenario_pt={}  current_ramen={:?}  selected_regions={:?}",
        ramen.scenario_pt, ramen.current_ramen, ramen.selected_regions);
    println!("super_ramen={:?}  feeling_slot={:?}  feeling_stock={:?}  special_feeling={}",
        ramen.super_ramen, ramen.feeling_slot, ramen.feeling_stock, ramen.special_feeling);
    println!("train_feeling_type={:?}", ramen.train_feeling_type);
    println!("stage={:?}   (协议 playing_state 已 dispatch)", game.stage);
    println!("deck={:?}", game.base.deck.iter().map(|c| c.card_id * 10 + c.rank).collect::<Vec<_>>());
    println!("uma_id={}  five_status={:?}", game.uma().uma_id, game.uma().five_status);
    // persons layout 摘要（按运行时 layout：0..5 训练卡+友人 / 6 理事长 / 7..11 NPC / 12 记者）
    println!("persons ({}) = [", game.persons.len());
    for (i, p) in game.persons.iter().enumerate() {
        println!("  {}: {:?} chara_id={} friendship={} hint={}",
            i, p.person_type, p.chara_id, p.friendship, p.is_hint);
    }
    println!("]");
    println!("deck_can_split = {}", game.deck_can_split);
    println!();
}

/// 候选评分可视化（按分降序编号 `#1` / `#2` / ...，高亮中选）
///
/// 不直接复用 `output::reason::render_reason_lines`：那是面向终端的窄胜理由渲染，
/// 这里要展示的是**所有**候选的评分与局数。中选用 `*` 标记，其余用 `#N`。
fn print_candidates(actions_len: usize, info: Option<&DecisionInfo>) {
    println!("=== MCTS 候选评分（按分降序） ===");
    if actions_len == 0 {
        println!("（无候选——可能当前 stage 无选择空间，如 Begin / Distribute）");
        println!();
        return;
    }
    println!("总候选数 = {}", actions_len);
    match info {
        Some(info) => {
            println!("中选下标（截断后） = {}", info.action_index);
            println!("选中评分 = {:.2}", info.score);
            println!("候选评分（截断到 top-{}） = {:?}",
                info.candidate_scores.len(), info.candidate_scores);
            if !info.candidate_n.is_empty() {
                println!("候选局数（与 scores 同序同截断） = {:?}", info.candidate_n);
            }
            // 2026-09 简化：`reason` / `elapsed_ms` 已从 DecisionInfo 删除——
            // reason 改由 scenario_extra.reason.rivals[] 承载（main.rs 接线）
            if let Some(extra) = &info.scenario_extra {
                println!("scenario_extra 键 = {:?}", extra.as_object().map(|o| o.keys().collect::<Vec<_>>()));
            }
        }
        None => {
            println!("（trainer.last_decision() = None——可能 fallback 到 RecommendedRamenTrainer 或单候选短路）");
        }
    }
    println!();
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ramen-turn-inspect] 错误: {e:?}");
            print_usage_and_exit()
        }
    }
}

fn run() -> Result<()> {
    // 1. CLI 解析
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(_) => {
            print_usage_and_exit();
            std::process::exit(2);
        }
    };

    // 2. 切到 workspace 根（load_game_config 走相对路径）
    ensure_workspace_cwd()?;

    // 3. 先初始化全局数据（必须**早于** parse_game_by_scenario：
    //    `RamenGame::newgame` 内部读 `global!(RAMENDATA).status_limit_base()`，
    //    未初始化直接 panic。main.rs 的做法同样：init_global_with_config 在 main loop 前）
    let game_config = load_game_config()?;
    init_logger_stdout("ramen_turn_inspect", &game_config.log_level)?;
    init_global_with_config(&game_config)?;

    // 4. 读 JSON 文件
    let contents = fs::read_to_string(&cli.json_path)
        .map_err(|e| anyhow!("读取 {} 失败: {e}", cli.json_path.display()))?;
    println!("=== 载入 JSON：{} ===", cli.json_path.display());
    println!("文件长度 = {} bytes", contents.len());
    println!();

    // 5. 走完整 parse_game_by_scenario → GameStatusRamen::into_game 构造 RamenGame
    let game = match parse_game_by_scenario(&contents) {
        Ok(ParsedGame::Ramen { game, .. }) => game,
        Ok(ParsedGame::Onsen(_)) => {
            return Err(anyhow!("当前 JSON 是 onsen 剧本（scenarioId=12），本工具只接 scenarioId=14 拉面"));
        }
        Err(e) => return Err(anyhow!("解析失败: {e}")),
    };

    // 6. 打印回合状态
    print_turn_state(&game);

    // 7. 用 game_config.toml 默认 SearchConfig 构造 RamenMctsTrainer
    //    与 main.rs / ramen_manual 行为一致——预算走 [mcts] 段，handwritten leaf
    //    （不依赖 onnx feature）。
    let mcts_config = SearchConfig::new_game_config(&game_config);
    let trainer = RamenMctsTrainer::new(mcts_config).verbose(true);

    // 8. 跑 MCTS：列候选 → select_action → 取 last_decision
    let actions = game.list_actions()?;
    println!("=== 跑 MCTS ===");
    println!("当前 stage = {:?}", game.stage);
    println!("list_actions 长度 = {}", actions.len());
    println!("search config: search_n={}  selection={:?}",
        trainer.search.config().search_n,
        trainer.selection);

    let mut rng = StdRng::from_os_rng();
    let t0 = Instant::now();
    let chosen_idx = if actions.is_empty() {
        println!("（无候选，跳过 select_action）");
        usize::MAX // 哨兵：方便下面打印「未选择」
    } else {
        let idx = trainer.select_action(&game, &actions, &mut rng)?;
        let elapsed_ms = t0.elapsed().as_millis() as u64;
        println!("select_action 选中 = {}  elapsed = {} ms", idx, elapsed_ms);
        idx
    };
    println!("searched_count = {}", trainer.searched_count());
    println!("combined_cache_hits = {}", trainer.combined_cache_hits());
    println!();

    // 9. 打印候选评分（last_decision 仅在真正走过 MCTS 时返回 Some）
    let info = trainer.last_decision();
    print_candidates(actions.len(), info.as_ref());

    // 10. 打印选中动作的可读名（仅在有候选且 trainer.last_decision 一致时打印）
    if chosen_idx != usize::MAX && chosen_idx < actions.len() {
        println!("=== 选中动作 ===");
        println!("action[{}] = {}", chosen_idx, actions[chosen_idx]);
    } else if chosen_idx != usize::MAX {
        println!("=== 选中动作 ===");
        println!("action[{}] = (索引异常，超出 actions 长度 {})", chosen_idx, actions.len());
    }

    Ok(())
}
