//! 温泉（`scenarioId=12`）回合处理：newgame 检测、事件/训练分发与 luck 挂载 emit。

use std::sync::{Arc, Mutex};

use anyhow::Result;
use colored::Colorize;
use log::info;
use rand::rngs::StdRng;
use umasim::{
    game::{
        Game,
        Trainer,
        onsen::game::OnsenGame
    },
    gamedata::GameConfig,
    output::DecisionSink,
    trainer::MctsTrainer
};

use crate::{
    decision::{emit_with_luck, LuckScoreTracker},
    utils::SAVED_GAME
};

/// 处理温泉剧本的一个回合快照（`thisTurn.json` → `ParsedGame::Onsen`）
///
/// 承接原 main watch loop 的 onsen 分支：写库 / newgame 检测、切局重置 luck、
/// 事件或训练分发、统一 `emit_with_luck` 挂载输出。
pub fn process_onsen(
    mut game: OnsenGame,
    trainer: &mut MctsTrainer,
    sink: &Arc<dyn DecisionSink>,
    luck_tracker: &mut LuckScoreTracker,
    rng: &mut StdRng,
    json_mode: bool,
    emit_info: &dyn Fn(&str),
    game_config: &GameConfig
) -> Result<()> {
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
        // 检测到新一局：通知 AIRed 重置 UI 状态
        emit_info("new_game");
        trainer.print_newgame_config(&game);
        eprintln!("{}", format!("温泉顺序: {:?}", game_config.onsen_order).bright_yellow());
        eprintln!("{}", "------------------------------".bright_yellow())
    }

    // 切局检测：新对局起始时重置 tracker（让 total_luck 归零）
    let chara_id = game.uma().uma_id as u64;
    if is_newgame {
        *luck_tracker = LuckScoreTracker::new();
    }

    if !game.unresolved_events.is_empty() {
        calc_onsen_event(trainer, &game, rng, json_mode)?;
    } else {
        calc_onsen_training(trainer, &mut game, rng, json_mode)?;
    }

    // 回合决策完成后统一 emit（带 luck score 挂载）
    // decision_kind 标明 partial decision 类型：onsen 路径下要么是 train 要么是 event
    let onsen_kind = if !game.unresolved_events.is_empty() { "event" } else { "train" };
    emit_with_luck(trainer, &game, sink, luck_tracker, chara_id, onsen_kind);

    // 计算完成：通知下游 watcher 进入阻塞状态
    eprintln!("计算完成，等待新数据...");
    Ok(())
}

/// 训练模式
///
/// 仅对当前阶段的候选列表调 `trainer.select_action` 出推荐，**不**调
/// `apply_action` / `next()` —— 与拉面侧的修复一致：AI 不推进游戏状态，
/// 下次 watch 收到 JSON 后主循环从零重建 game 再算。
///
/// `json_mode`：true 时跳过 F2 提示与训练分布的屏幕打印（这些输出仅供人类调试）。
pub fn calc_onsen_training(trainer: &MctsTrainer, game: &mut OnsenGame, rng: &mut StdRng, json_mode: bool) -> Result<()> {
    if !json_mode {
        println!("{}", game.explain_distribution()?);
        info!("{}", "正在计算...".bright_black());
    }
    if game.pending_selection {
        // 温泉选择状态：列出候选 + 选一次（升级是独立的下一阶段决策，本次不下发）
        let actions = game.list_actions_onsen_select();
        if !actions.is_empty() {
            let _ = trainer.select_action(game, &actions, rng)?;
        }
    } else {
        let actions = game.list_actions()?;
        if !actions.is_empty() {
            let _ = trainer.select_action(game, &actions, rng)?;
        }
    }
    if !json_mode {
        println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    }
    Ok(())
}

/// 事件模式
pub fn calc_onsen_event(trainer: &MctsTrainer, game: &OnsenGame, rng: &mut StdRng, json_mode: bool) -> Result<()> {
    if let Some(event) = game.unresolved_events.first() {
        let _selection = trainer.select_event_choice(game, event, &event.choices, rng)?;
        if !json_mode {
            println!("{}", "[按 F2 保存当前回合状态]".bright_black());
        }
    }
    Ok(())
}