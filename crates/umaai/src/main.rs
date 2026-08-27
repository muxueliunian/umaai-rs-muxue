//! umaai-rs - Rewrite UmaAI in Rust
//!
//! author: curran
use std::{sync::Mutex, time::Instant};

use anyhow::Result;
use colored::Colorize;
use log::info;
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use text_to_ascii_art::to_art;
use umasim::{
    game::{
        Game,
        Trainer,
        onsen::{OnsenTurnStage, action::OnsenAction, game::OnsenGame}
    },
    gamedata::init_global_with_config,
    neural::Evaluator,
    search::SearchConfig,
    trainer::MctsTrainer,
    utils::{check_windows_terminal, check_working_dir, init_logger, load_game_config, pause}
};

use crate::{
    protocol::{
        GameStatusOnsen,
        urafile::{UraFileWatcher, parse_game}
    },
    utils::{SAVED_GAME, hotkey_handler}
};

pub mod protocol;
pub mod utils;

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

/// 实际的主函数
async fn main_guard() -> Result<()> {
    println!("{}", to_art("UMAAI 0.26".to_string(), "small", 0, 1, 0).expect("here"));
    // 0. 运行前检查
    check_windows_terminal()?;
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

    // ctrl-s handler
    tokio::spawn(async move {
        hotkey_handler().await;
    });

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

    // Phase 4 feature 拆分后，onnx 评估器路径已 cfg gate 到 `onnx` feature。
    // 当前通道层不依赖 onnx（不需要 tract-onnx 巨大依赖链），强制走 MctsTrainer
    // 默认的 handwritten leaf eval（FlatSearch::new() 默认就是 Handwritten）。
    // 后续若恢复 nn leaf，可在此处重新启用 cfg(feature = "onnx") 分支。
    let _rollout_evaluator = game_config.mcts.rollout_evaluator.as_str();
    let _neuralnet_model_path = game_config.neuralnet_model_path.as_str();
    let _max_depth = game_config.mcts.max_depth;
    // 始终强制 handwritten（保持与原 "handwritten" 分支一致的行为）
    trainer.search = trainer.search.with_leaf_evaluator_handwritten();
    /*
    // 原始 leaf eval 开关（已注释，等 onnx feature 真正启用时再恢复）：
    match game_config.mcts.rollout_evaluator.as_str() {
        "handwritten" => {
            trainer.search = trainer.search.with_leaf_evaluator_handwritten();
        }
        "nn" => {
            if game_config.mcts.max_depth == 0 {
                println!(
                    "警告: mcts.rollout_evaluator=\"nn\" 但 mcts.max_depth=0，leaf eval 不会被使用（等价于旧路径）"
                );
            }
            if game_config.mcts_selection == "pt" && game_config.mcts.max_depth > 0 {
                return Err(anyhow!(
                    "E4 验收约束：mcts.rollout_evaluator=\"nn\" 且 max_depth>0 时禁止 mcts_selection=\"pt\"；请改为 \"score\""
                ));
            }

            let model_path = game_config.neuralnet_model_path.as_str();
            if !Path::new(model_path).exists() {
                return Err(anyhow!("mcts.rollout_evaluator=\"nn\" 但模型文件不存在: {model_path}"));
            }
            // 先验证模型可加载（避免"以为开了 NN 实际没开"的伪对照）
            let _ = NeuralNetEvaluator::load(model_path)?;
            trainer.search = trainer.search.with_leaf_evaluator_nn(model_path.to_string());
        }
        other => {
            return Err(anyhow!(
                "未知 mcts.rollout_evaluator=\"{other}\"（仅支持 \"handwritten\" | \"nn\"）"
            ));
        }
    }
    */

    // E4：leaf eval 微批大小（batch=1 等价于逐样本推理；batch>1 才会启用 infer_batch）
    trainer.search = trainer
        .search
        .with_rollout_batch_size(game_config.mcts.rollout_batch_size);

    // 开始检测文件
    let mut watcher = UraFileWatcher::init()?;
    loop {
        let contents = watcher.watch("thisTurn.json")?;
        let mut is_newgame = false;
        match parse_game::<GameStatusOnsen>(&contents) {
            Ok(mut game) => {
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
                    println!("{}", format!("温泉顺序: {:?}", game_config.onsen_order).bright_yellow());
                    println!("{}", "------------------------------".bright_yellow())
                }

                if !game.unresolved_events.is_empty() {
                    calc_onsen_event(&trainer, &game, &mut rng)?;
                } else {
                    calc_onsen_training(&trainer, &mut game, &mut rng)?;
                }
            }
            Err(e) => {
                println!("{}", format!("解析回合信息出错: {e}").red());
                println!("----------");
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    match main_guard().await {
        Ok(_) => {}
        Err(e) => {
            println!("{}", "UmaAI 出现错误，即将退出:".red());
            println!("{}", "-----------------------------------".red());
            println!("{}", format!("{e:?}").red());
            pause().expect("pause");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path, sync::mpsc};

    use anyhow::Result;
    use colored::Colorize;
    use log::info;
    use notify::{Event, RecursiveMode, Watcher};
    use umasim::{gamedata::init_global, utils::init_logger};

    use crate::protocol::{
        GameStatusOnsen,
        urafile::{UraFileWatcher, parse_game}
    };

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
