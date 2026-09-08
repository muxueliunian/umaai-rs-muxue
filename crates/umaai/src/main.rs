//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::{
    sync::{Arc, Mutex},
    time::Instant
};

use anyhow::Result;
use colored::Colorize;
use lexopt::prelude::*;
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use text_to_ascii_art::to_art;
use umasim::{
    game::{
        Game,
        Trainer,
        onsen::{OnsenTurnStage, action::OnsenAction, game::OnsenGame},
        ramen::{RamenGame, RamenStage}
    },
    gamedata::init_global_with_config,
    neural::Evaluator,
    output::{DecisionInfo, DecisionSink, HumanReadableSink, StdoutJsonSink},
    search::SearchConfig,
    trainer::{MctsTrainer, RamenMctsTrainer},
    utils::{check_working_dir, init_logger, load_game_config}
};

use crate::{
    luck_score::LuckScoreTracker,
    protocol::urafile::UraFileWatcher,
    utils::{SAVED_GAME, hotkey_handler}
};

pub mod luck_score;
pub mod protocol;
pub mod utils;

/// CLI 参数
///
/// `--json`：stdout 严格只 JSON（AIRedirector 模式）；启动横幅 / 日志 / 状态
/// 走 stderr，避免污染 JSON 流。**不引入新命令行参数**——除 `--json` 模式开关
/// 外，所有可调项走 `game_config.toml` / `default_config.toml`（详见集成文档 §3.2.3）。
#[derive(Default)]
struct Args {
    /// `--json` 模式：stdout 仅 JSON，供 AIRedirector 抓取
    json: bool,
    /// `--help` / `-h` 模式：打印用法并退出 0
    help: bool
}

/// 解析 CLI 参数（lexopt 与项目惯例一致——umasim 主 bin 全部用 lexopt）
///
/// 未知参数通过 `arg.unexpected()` 转为 `Err`，不静默接受歧义输入。
fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Long("json") => args.json = true,
            Short('h') | Long("help") => args.help = true,
            _ => return Err(arg.unexpected().into())
        }
    }
    Ok(args)
}

/// 打印 `--help` 输出（两个 sink 都走 stdout 没问题——`-h` 与 `--json` 互斥）
fn print_help_and_exit() -> ! {
    println!("umaai-rs — UmaAI decision engine");
    println!();
    println!("用法: umaai [--json]");
    println!();
    println!("选项:");
    println!("  --json    stdout 严格只 JSON（AIRedirector 模式：启动横幅 / 日志走 stderr）");
    println!("  -h, --help  打印本帮助");
    std::process::exit(0);
}

pub fn run_evaluate<G, E>(game: &G, evaluator: &E, rng: &mut StdRng) -> Result<()>
where
    G: Game + Serialize,
    G::Action: Serialize,
    E: Evaluator<G>
{
    let t = Instant::now();
    let score = evaluator.evaluate(&game);
    if let Some(action) = evaluator.select_action(&game, rng) {
        info!(
            "{}",
            format!(
                "AI选择: {action:?}, 均分: {}, 标准差: {}, Time: {:?}",
                score.score_mean as i64,
                score.score_stdev as i64,
                t.elapsed()
            )
            .bright_green()
        );
    }
    Ok(())
}

/// 训练模式
pub fn calc_onsen_training(trainer: &MctsTrainer, game: &mut OnsenGame, rng: &mut StdRng) -> Result<()> {
    println!("{}", game.explain_distribution()?);
    info!("{}", "正在计算...".bright_black());
    if game.pending_selection {
        // 是温泉选择状态
        let actions = game.list_actions_onsen_select();
        let onsen = trainer.select_action(game, &actions, rng)?;
        // 前进一步选择升级
        game.apply_action(&actions[onsen], rng)?;
        let upgradeable = game.get_upgradeable_equipment();
        if !upgradeable.is_empty() {
            let actions = upgradeable
                .iter()
                .map(|x| OnsenAction::Upgrade(*x as i32))
                .collect::<Vec<_>>();
            trainer.select_action(game, &actions, rng)?;
        }
    } else {
        // 如果被解析成 Bathing 但没有温泉券合buff，就直接跳过到 Train
        if game.stage == OnsenTurnStage::Bathing && game.bathing.ticket_num == 0 && game.bathing.buff_remain_turn == 0 {
            game.next();
        }

        let actions = game.list_actions()?;
        if actions.is_empty() {
            return Ok(());
        }
        let action_idx = trainer.select_action(game, &actions, rng)?;
        let action = actions[action_idx].clone();

        // 选择温泉券时需要继续给出训练推荐
        if game.stage == OnsenTurnStage::Bathing {
            // 日志控制说明：旧实现曾用 `disable_log()/enable_log()` 临时抑制温泉券期间
            // 的训练搜索日志。Phase 3 后规则层日志已通过 `diag` feature 编译期裁剪
            // （搜索 rollout 默认不产生 `info!` / `diag!`），无需运行期切换。这里直接
            // 走完整搜索流程，日志静默由 diag 特性保证。
            if action == OnsenAction::UseTicket(true) {
                game.do_use_ticket(rng)?;
            }
            game.next();

            info!("{}", "正在计算训练...".bright_black());
            let actions = game.list_actions()?;
            if !actions.is_empty() {
                let _action_idx = trainer.select_action(game, &actions, rng)?;
                //let action = actions[action_idx].clone();
            }
        }
    }
    println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    Ok(())
}

/// 事件模式
pub fn calc_onsen_event(trainer: &MctsTrainer, game: &OnsenGame, rng: &mut StdRng) -> Result<()> {
    if let Some(event) = game.unresolved_events.first() {
        let _selection = trainer.select_event_choice(game, event, &event.choices, rng)?;
        println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    }
    Ok(())
}

/// 拉面训练：跑本回合所有 stage 直到推到 NextTurn / Settlement / SuperRamenSelect
///
/// **Step 7 接入**（参照 onsen `calc_onsen_training` 模式）：
/// - 循环 `game.run_stage(&trainer, rng)` 让 trainer 在各 stage（RamenSelect /
///   SpecialSelect / Train / RegionSelect / Begin 等）出决策
/// - 退出条件：stage 推到 NextTurn（回合边界）/ Settlement（RMJ 结算，等下一条 JSON）
///   / SuperRamenSelect（超级拉面选择，等下一条 JSON）
/// - 特别处理「turn 2/23/47 的首次 RegionSelect」：`RamenGame::next()` 在 turn=2 的
///   `Begin` 之后会自动推进到 `RegionSelect`（adapter_spec §UmaAI 需要复合决策 + 项目
///   注释 §'Begin → RegionSelect → BeginAfterRegionSelect'），本函数不需要额外触发
///   —— 只要不提前退出，`run_stage` 会把整个阶段链跑完。
///
/// 异常退出（事件回合、不 dispatch 的样本）由 `parse_game_by_scenario` 前的 caller 检查
/// `game.stage`：若 stage 是 Begin（newgame 默认），说明 into_game 没 dispatch，
/// 主循环不应调用本函数。
pub fn calc_ramen_training(trainer: &RamenMctsTrainer, game: &mut RamenGame, rng: &mut StdRng) -> Result<()> {
    // 防御性保护：最多循环 32 次避免 stage 流转卡死
    const MAX_STAGE_LOOP: usize = 32;
    for _ in 0..MAX_STAGE_LOOP {
        match game.stage {
            RamenStage::NextTurn | RamenStage::Settlement | RamenStage::SuperRamenSelect => {
                // 回合边界 / RMJ 结算 / 超级拉面选择 —— 等下一条 JSON
                break;
            }
            _ => {
                // 其余 stage 全部交给 run_stage：内部已处理 select_action + apply_action + next()
                game.run_stage(trainer, rng)?;
            }
        }
    }
    println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    Ok(())
}

/// 把 trainer 的 last_decision 喂给 sink：先挂 luck score 字段，再 emit
///
/// **Step 5 改造**：从原 `emit_decision` 升级——每回合不再"select_action → 立即 emit"，
/// 而是把多次 select_action 的 last_decision 收集起来，由主循环在 calc_onsen_*
/// 完成后**统一调一次本函数**：
///
/// 1. 取 `trainer.last_decision()`（最后一次 select_action 的数据）
/// 2. 算 T(n) baseline（按局数加权：Σ score × n / Σ n，与 onsen `update_score` 同口径）
/// 3. `tracker.on_new_turn(chara_id, t_n_baseline)` 更新 / 切局检测
/// 4. 挂 `tracker.snapshot()` + 每候选 `action_luck` 到 `info.scenario_extra`
/// 5. `sink.emit(&info, &game.view())`
///
/// `GameView::view()` 由 Game trait 默认实现填充；onsen scenario 字段留空待 Step 6/7。
///
/// **Step 7 改造**：拆出 `emit_with_luck_decision` 接收 `Option<DecisionInfo>`，
/// 让拉面分支（`RamenMctsTrainer` 等其他 trainer）也能复用 luck score 挂载逻辑，
/// 不必为每个 trainer 单独写一份。
fn emit_with_luck<G: Game>(
    trainer: &MctsTrainer, game: &G, sink: &Arc<dyn DecisionSink>, tracker: &mut LuckScoreTracker, chara_id: u64
) {
    emit_with_luck_decision(trainer.last_decision(), game, sink, tracker, chara_id);
}

/// 把已提取的 `DecisionInfo` 喂给 sink：挂 luck score 字段 + emit。
///
/// 与 [`emit_with_luck`] 区别在于**不依赖具体 trainer 类型**——只要 trainer 实现了
/// `Trainer<G>` 并返回 `DecisionInfo` 即可。拉面分支（`RamenMctsTrainer` 等）走这里。
fn emit_with_luck_decision<G: Game>(
    last_decision: Option<DecisionInfo>, game: &G, sink: &Arc<dyn DecisionSink>, tracker: &mut LuckScoreTracker, chara_id: u64
) {
    let Some(mut info) = last_decision else {
        return;
    };
    // T(n) baseline：按局数加权（手写 / 早期早退时 candidate_n 为空 → 退化为按候选数等权）
    let t_n_baseline: f64 = if info.candidate_n.is_empty() {
        if info.candidate_scores.is_empty() {
            0.0
        } else {
            info.candidate_scores.iter().map(|&s| s as f64).sum::<f64>()
                / info.candidate_scores.len() as f64
        }
    } else {
        let total_n: u32 = info.candidate_n.iter().sum();
        if total_n == 0 {
            info.candidate_scores.iter().map(|&s| s as f64).sum::<f64>()
                / info.candidate_scores.len().max(1) as f64
        } else {
            info.candidate_scores
                .iter()
                .zip(info.candidate_n.iter())
                .map(|(&s, &n)| (s as f64) * (n as f64))
                .sum::<f64>()
                / total_n as f64
        }
    };

    let _turn_delta = tracker.on_new_turn(chara_id, t_n_baseline);

    // 每候选 action_luck：T(n, action_i) - T(n)（AIRedirector 关心，玩家模式跳过）
    let action_luck = serde_json::json!(
        info.candidate_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, (s as f64) - t_n_baseline))
            .collect::<std::collections::HashMap<usize, f64>>()
    );

    // 挂载 scenario_extra：snapshot + action_luck
    let extra = match serde_json::to_value(tracker.snapshot()) {
        Ok(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("action_luck".into(), action_luck);
            }
            Some(v)
        }
        Err(_) => None
    };
    info.scenario_extra = extra;

    sink.emit(&info, &game.view());
}

/// 实际的主函数
async fn main_guard() -> Result<()> {
    let args = parse_args()?;
    if args.help {
        print_help_and_exit();
    }

    // sink 选择必须在 colored::set_override 之前——后者是全局副作用
    let sink: Arc<dyn DecisionSink> = if args.json {
        // JSON 模式关闭 ANSI：colored 即使 --no-color 也可能输出 ANSI reset，
        // 影响 AIRedirector 解析。详见集成文档 §3.2.6 第 3 条。
        colored::control::set_override(false);
        Arc::new(StdoutJsonSink)
    } else {
        Arc::new(HumanReadableSink)
    };

    // 启动横幅走 stderr（避免污染 JSON 模式的 stdout 流）
    eprintln!("{}", to_art("UMAAI 0.26".to_string(), "small", 0, 1, 0).expect("here"));
    // 0. 运行前检查（Windows terminal 检测暂时注释掉——非 Windows 平台跳过，
    //    避免误报；Step 5 之后视需要再决定是否启用）
    // check_windows_terminal()?;
    if !fs_err::exists("game_config.toml")? {
        check_working_dir()?;
    }
    // 1. 先读取配置文件
    let game_config = load_game_config()?;
    let mcts_config = SearchConfig::new_game_config(&game_config);
    // 2. 根据配置初始化日志，设置工作线程
    init_logger("umaai", &game_config.log_level)?;
    init_global_with_config(&game_config)?;
    info!(
        "{}",
        format!("工作线程数: {}", game_config.collector.threads).bright_yellow()
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;
    //info!("search_config = {mcts_config:?}");

    // 3. 再初始化全局数据
    init_global_with_config(&game_config)?;

    // ctrl-s handler —— **延迟到 watcher 启动成功之后** spawn。`hotkey_handler`
    // 是无限循环（loop），如果 watcher init 失败走 early return，runtime drop
    // 时会等这个 task 结束 → hang，cargo run 卡住不退出。

    let mut rng = StdRng::from_os_rng();

    // 神经网络训练员
    //let model_path = "saved_models/onsen_v1/model.onnx";
    //let evaluator =
    //NeuralNetEvaluator::load(model_path).map_err(|e| anyhow!("错误: 无法加载神经网络模型 {model_path}: {e:?}"))?;

    // MCTS训练员
    let mut trainer = MctsTrainer::new(mcts_config).verbose(true);
    trainer.mcts_onsen = game_config.mcts_selected_onsen;
    // 这个设置在AI模式下不生效
    trainer.mcts_selection = "score".to_string();

    // 拉面 MCTS 训练员（与 onsen 的 MctsTrainer 强耦合 OnsenGame 不同；拉面用
    // RamenMctsTrainer 绑 RamenGame，独立构造。stages 走 game_config.mcts.ramen_search_stages，
    // 与 umasim/src/main.rs 拉面路径口径一致。
    let ramen_mcts_config = SearchConfig::new_game_config(&game_config);
    let ramen_stages = umasim::trainer::RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?;
    let ramen_trainer = RamenMctsTrainer::new(ramen_mcts_config)
        .with_stages(ramen_stages)
        .verbose(true);

    // Phase 4 feature 拆分后，onnx 评估器路径已 cfg gate 到 `onnx` feature。
    // 当前通道层不依赖 onnx（不需要 tract-onnx 巨大依赖链），强制走 MctsTrainer
    // 默认的 handwritten leaf eval（FlatSearch::new() 默认就是 Handwritten）。
    // 后续若恢复 nn leaf，可在此处重新启用 cfg(feature = "onnx") 分支。
    let _rollout_evaluator = game_config.mcts.rollout_evaluator.as_str();
    let _neuralnet_model_path = game_config.neuralnet_model_path.as_str();
    let _max_depth = game_config.mcts.max_depth;
    // 始终强制 handwritten（保持与原 "handwritten" 分支一致的行为）
    trainer.search = trainer.search.with_leaf_evaluator_handwritten();

    // E4：leaf eval 微批大小（batch=1 等价于逐样本推理；batch>1 才会启用 infer_batch）
    trainer.search = trainer
        .search
        .with_rollout_batch_size(game_config.mcts.rollout_batch_size);

    // 开始检测文件——init 失败时优雅退出（不 panic）：路径无效 / notify 失败都打 warn + return Ok(())
    let mut watcher = match UraFileWatcher::init() {
        Ok(w) => w,
        Err(e) => {
            log::warn!("UraFileWatcher init 失败: {e}，main 不进入 watch loop，程序正常退出（exit 0）");
            return Ok(());
        }
    };

    // watcher 启动成功后才 spawn hotkey_handler（见上方注释——避免失败路径 hang）
    tokio::spawn(async move {
        hotkey_handler().await;
    });

    // Luck score 跟踪器（Step 5）：每回合 baseline 累加 + 切局检测，snapshot
    // 挂在 DecisionInfo::scenario_extra 下发给 AIRedirector。
    let mut luck_tracker = LuckScoreTracker::new();

    loop {
        let contents = watcher.watch("thisTurn.json")?;
        // Step 6：按 baseGame.scenarioId 分发（12=温泉 / 14=拉面）。拉面侧 AI 主流程
        // 在 Step 7 接入——这里只解析 + 打 warn，AI 不出推荐。
        match crate::protocol::parse_game_by_scenario(&contents) {
            Ok(crate::protocol::ParsedGame::Onsen(mut game)) => {
                let mut is_newgame = false;
                // 保存一份到全局
                {
                    if let Some(mutex) = SAVED_GAME.get() {
                        let mut saved = mutex.lock().expect("saved game");
                        // 如果当前游戏不是下一轮，则打印当前游戏配置
                        if !game.is_next_of(&saved) {
                            is_newgame = true;
                        }
                        *saved = game.clone();
                    } else {
                        SAVED_GAME
                            .set(Mutex::new(game.clone()))
                            .expect("SAVED_GAME already initialized");
                        is_newgame = true;
                    }
                }
                if is_newgame {
                    trainer.print_newgame_config(&game);
                    eprintln!("{}", format!("温泉顺序: {:?}", game_config.onsen_order).bright_yellow());
                    eprintln!("{}", "------------------------------".bright_yellow())
                }

                // 切局检测：新对局起始时重置 tracker（让 total_luck 归零）
                let chara_id = game.uma().uma_id as u64;
                if is_newgame {
                    luck_tracker = LuckScoreTracker::new();
                }

                if !game.unresolved_events.is_empty() {
                    calc_onsen_event(&trainer, &game, &mut rng)?;
                } else {
                    calc_onsen_training(&trainer, &mut game, &mut rng)?;
                }

                // 回合决策完成后统一 emit（带 luck score 挂载）—— 见 emit_with_luck 注释
                emit_with_luck(&trainer, &game, &sink, &mut luck_tracker, chara_id);
            }
            Ok(crate::protocol::ParsedGame::Ramen { mut game, single_mode_chara_id }) => {
                // Step 7：拉面 AI 主流程接入（参照 onsen `Ok(ParsedGame::Onsen(..))` 路径）。
                // `into_game` 已经按 (source, active_effect, playing_state, turn) 完成 stage dispatch：
                //   - 不 dispatch 的样本（source=event / playing_state=5/46/48 / 数据获取不全）
                //     stage 仍为 Begin（newgame 默认值），跳过本分支
                //   - 已 dispatch 的样本按 RamenSelect / Train / RegionSelect 走主流程
                //
                // 注意：`SAVED_GAME` 是 OnsenGame（`ctrl-s` 玩家调试保存用），拉面侧
                // 暂不写入——避免 onsen / ramen 类型冲突；切局检测改用 single_mode_chara_id。
                if game.stage == RamenStage::Begin {
                    // into_game 没 dispatch（事件 / 结算 / 数据不全），等下一条 JSON
                    continue;
                }

                // 切局检测：`single_mode_chara_id` 变化 → 新一局开始。
                // C# 端 single_mode_chara_id 单调递增，同 uma_id 重复训练也能识别新局。
                // 协议字段缺失（None）时退化到 uma_id 兜底（旧 json / 测试 fixture）。
                let chara_id = single_mode_chara_id
                    .unwrap_or_else(|| game.uma().uma_id as u64);
                if luck_tracker.last_single_mode_id() != Some(chara_id) {
                    eprintln!("{}", "---- 拉面新一局 ----".bright_yellow());
                    luck_tracker = LuckScoreTracker::new();
                }

                // 跑本回合所有 stage 直到推到 NextTurn / Settlement / SuperRamenSelect
                calc_ramen_training(&ramen_trainer, &mut game, &mut rng)?;

                // 回合决策完成后统一 emit（带 luck score 挂载）
                emit_with_luck_decision(
                    ramen_trainer.last_decision(),
                    &game,
                    &sink,
                    &mut luck_tracker,
                    chara_id,
                );
            }
            Err(e) => {
                println!("{}", format!("解析回合信息出错: {e}").red());
                println!("----------");
            }
        }
    }
}

/// 出错时按 Enter 暂停（仅发布版，CI / 开发默认不阻塞 stdin）
///
/// 与 `release-pause` feature 联动：
/// - `cargo build --release`（默认）：开发 / CI 路径，**不暂停**——避免 stdin 在
///   自动化场景里 hang，且让 cargo run 时 Ctrl-C 后立即退出方便调试。
/// - `cargo build --release --features release-pause`：发布给用户的二进制，
///   启动后暂停"按 Enter 退出"，让用户看清错误信息。
#[cfg(feature = "release-pause")]
fn pause_on_exit() {
    eprintln!("\n按 Enter 退出...");
    let _ = std::io::stdin().read_line(&mut String::new());
}

#[cfg(not(feature = "release-pause"))]
fn pause_on_exit() {
    // 开发 / CI 默认 no-op；详见 fn pause_on_exit 文档
}

#[tokio::main]
async fn main() -> Result<()> {
    match main_guard().await {
        Ok(_) => {}
        Err(e) => {
            println!("{}", "UmaAI 出现错误，即将退出:".red());
            println!("{}", "-----------------------------------".red());
            println!("{}", format!("{e:?}").red());
            pause_on_exit();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path, sync::mpsc};

    use anyhow::Result;
    use colored::Colorize;
    use lexopt::prelude::*;
    use log::info;
    use notify::{Event, RecursiveMode, Watcher};
    use umasim::{gamedata::init_global, utils::init_logger};

    use super::Args;
    use crate::protocol::{
        GameStatusOnsen,
        urafile::{UraFileWatcher, parse_game}
    };

    /// 把 lexopt::Parser + 解析逻辑包成一个 helper（与 `parse_args` 同结构，
    /// 但用 `from_iter` 喂手工 vec 避免依赖真实 env arg）
    fn parse_from<I: IntoIterator<Item = String>>(args: I) -> anyhow::Result<Args> {
        let mut out = Args::default();
        let mut parser = lexopt::Parser::from_iter(args);
        while let Some(arg) = parser.next()? {
            match arg {
                Long("json") => out.json = true,
                Short('h') | Long("help") => out.help = true,
                _ => return Err(arg.unexpected().into())
            }
        }
        Ok(out)
    }

    /// 空参数列表 → 默认 `Args { json: false, help: false }`
    #[test]
    fn test_parse_args_default() -> Result<()> {
        let args = parse_from(vec!["umaai".to_string()])?;
        assert!(!args.json, "默认 json=false");
        assert!(!args.help, "默认 help=false");
        Ok(())
    }

    /// `--json` 解析为 `json=true`
    #[test]
    fn test_parse_args_json() -> Result<()> {
        let args = parse_from(vec!["umaai".to_string(), "--json".to_string()])?;
        assert!(args.json, "--json 触发");
        assert!(!args.help, "help 仍为 false");
        Ok(())
    }

    /// 未知参数 → `Err`，不静默接受歧义输入
    #[test]
    fn test_parse_args_unknown_rejects() {
        let result = parse_from(vec!["umaai".to_string(), "--bogus".to_string()]);
        println!("--bogus 解析: is_err={}", result.is_err());
        assert!(result.is_err(), "未知参数必须报错");
    }

    #[tokio::test]
    async fn test_watch() -> Result<()> {
        let local_app_path = env::var("LOCALAPPDATA")?;
        let urafile_path = format!("{local_app_path}/UmamusumeResponseAnalyzer/PluginData/SendGameStatusPlugin/");

        let (tx, rx) = mpsc::channel::<notify::Result<Event>>();
        let mut watcher = notify::recommended_watcher(tx)?;
        println!("{urafile_path}");
        watcher.watch(Path::new(&urafile_path), RecursiveMode::NonRecursive)?;
        loop {
            let event = rx.recv()??;
            println!("{event:?}");
        }
    }

    #[test]
    fn test_urafile() -> Result<()> {
        // 2. 根据配置初始化日志
        init_logger("test", "info")?;

        // 3. 再初始化全局数据
        init_global()?;
        let mut watcher = UraFileWatcher::init()?;
        loop {
            let contents = watcher.watch("thisTurn.json")?;
            match parse_game::<GameStatusOnsen>(&contents) {
                Ok(game) => {
                    info!("{}", game.explain_distribution()?);
                    println!("----------");
                }
                Err(e) => {
                    println!("{}", format!("解析回合信息出错: {e}").red());
                    println!("----------");
                }
            }
        }
    }
}
