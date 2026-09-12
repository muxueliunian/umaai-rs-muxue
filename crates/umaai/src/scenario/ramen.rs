//! 拉面（`scenarioId=14`）回合处理：切局检测、链式决策计算与 luck 挂载 emit。

use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;
use rand::rngs::StdRng;
use umasim::{
    game::{
        Game,
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    output::{
        DecisionInfo,
        DecisionSink,
        GameView,
        reason::render_reason_lines
    },
    trainer::RamenMctsTrainer
};

use crate::decision::{emit_with_luck_decision, LastReasonSink, LuckScoreTracker};

/// 处理拉面剧本的一个回合快照（`thisTurn.json` → `ParsedGame::Ramen`）
///
/// 承接原 main watch loop 的 ramen 分支：`Begin`（未 dispatch）早退、切局检测、
/// human 屏幕打印、链式决策 emit 与 luck 挂载。
///
/// 注意：`single_mode_chara_id` 为切局键（`None` 时退化用 uma_id）。
pub fn process_ramen(
    mut game: RamenGame,
    single_mode_chara_id: Option<u64>,
    trainer: &RamenMctsTrainer,
    reason_slot: &LastReasonSink,
    sink: &Arc<dyn DecisionSink>,
    luck_tracker: &mut LuckScoreTracker,
    rng: &mut StdRng,
    json_mode: bool,
    emit_info: &dyn Fn(&str)
) -> Result<()> {
    if game.stage == RamenStage::Begin {
        // into_game 没 dispatch（事件 / 结算 / 数据不全），等下一条 JSON
        return Ok(());
    }

    // 切局检测：`single_mode_chara_id` 变化 → 新一局开始。
    // C# 端 single_mode_chara_id 单调递增，同 uma_id 重复训练也能识别新局。
    // 协议字段缺失（None）时退化到 uma_id 兜底（旧 json / 测试 fixture）。
    let chara_id = single_mode_chara_id
        .unwrap_or_else(|| game.uma().uma_id as u64);
    if luck_tracker.last_single_mode_id() != Some(chara_id) {
        // 检测到新一局：通知 AIRed 重置 UI 状态
        emit_info("new_game");
        eprintln!("{}", "---- 拉面: 育成开始 ----".bright_yellow());
        *luck_tracker = LuckScoreTracker::new();
    }

    // 屏幕侧（human mode）按需求在收到并解析回合数据后**立即**显示：
    // 马娘状态 / 剧本信息 / 训练分布——后续才进入推理（calc_ramen_training）。
    if !json_mode {
        if let Ok(status) = game.explain() {
            println!("{status}");
        }
        let script_info = game.explain_ramen_info();
        if !script_info.is_empty() {
            println!("{script_info}");
        }
        if let Ok(dist_info) = game.explain_distribution() {
            println!("{dist_info}");
        }
    }
    eprintln!("AI计算中...");
    // 连续决策：返回链式决策（一个快照对应多个决策时逐个 emit）
    // 中间决策（除最后一个）用各自捕获的 view 直接 emit（不触 luck），
    // 最后一个决策走完整 luck 挂载（baseline 每回合只更新一次）。
    //
    // 注意：链式决策的"计算后续动作"通知（eprintln + compute_next_step）
    // 已在 calc_ramen_training 内部、决策#2 真正执行前发出——这里不再重复。
    let chain = calc_ramen_training(trainer, &mut game, rng, json_mode, reason_slot, emit_info)?;
    let last_index = chain.len().saturating_sub(1);
    for (_, (info, view)) in chain.iter().take(last_index).enumerate() {
        sink.emit(info, view);
    }
    if !chain.is_empty() {
        // 拉面 MCTS 路径：从 LastReasonSink 缓存取 DecisionReasonData 挂到
        // scenario_extra.reason，让 AIRedirector 拿到 human mode reason
        // 所需信息（metric / chosen_desc / chosen_mean / rivals[]）。
        // 链式决策**最后一步**才消费 reason_slot，避免中间决策漏挂。
        //
        // decision_kind 用最后一步决策的 stage——主链通常是 train（拉面
        // 决策落地后的训练阶段）。ramen_action 同样用最后一步决策的
        // candidate_descriptions[action_index]（to_string 形式，含
        // 训练名 + 之前已经 ground 的吃面效果）。
        let last_info = chain.last().expect("non-empty chain").0.clone();
        let last_kind = last_info.decision_kind.clone();
        let ramen_action_text = last_info
            .candidate_descriptions
            .get(last_info.action_index)
            .cloned();

        // 手写 fallback 决策（`candidate_scores` 为空，如默认配置下 region
        // 未开时的地区选择）：没有真正的搜索评分，走 luck 挂载只会以 baseline=0
        // 污染 luck tracker（后续回合运气全被算错），且 sink 打印的「期望评分」
        // 只是回合加成换算、运气恒 0 会误导。故直接 emit（不触 luck）；
        // HumanReadableSink 会为该决策打印「选择...（手写逻辑）」。搜索决策
        // （常见 train/ramen_select）仍走完整 luck 挂载。
        if last_info.candidate_scores.is_empty() {
            sink.emit(&last_info, &game.view());
        } else {
            emit_with_luck_decision(
                Some(last_info),
                &game,
                sink,
                luck_tracker,
                chara_id,
                reason_slot.take().as_ref(),
                &last_kind,
                ramen_action_text.as_deref(),
            );
        }
    }

    // 计算完成：通知下游 watcher 进入阻塞状态
    eprintln!("计算完成，等待新数据...");
    Ok(())
}

/// 拉面训练：当前阶段出推荐，并在**两个特定场景**连续出下一个决策
///
///**设计原则**：
///- watch 收到一次 `thisTurn.json` 只代表"当前回合、当前阶段"的快照，AI 基于本次
///  快照出推荐（select_action）。**仅解决"一个快照对应两个决策"的场景**，其余
///  情况下**不**改 game（下次 watch 收到新 JSON → 主循环重建 game 从零计算）。
///- 定向连续决策（类似 onsen 的"选完温泉券后继续给训练推荐"）：
///  1. `RamenSelect` 选**不吃面**：不吃面没有真实操作产生新 JSON，手动
///     `apply_action` + `next()` 推进到 `Train`，再给训练决策。
///  2. `Train` 且 turn == 1（仅剧本机制启动前的第 1 回合）：训练决策后下一屏是
///     回合 2 的地区选择（同样无新 JSON），跨过 `NextTurn` 推进到 `RegionSelect`，
///     再给地区决策；到达 RegionSelect 后**立即停**，不继续向下级联。
///- 其它所有阶段维持单决策：AI 不推进游戏状态，玩家执行后由 C# 发新 JSON。
///
/// 返回链式决策 `Vec<(DecisionInfo, GameView)>`（每个决策附带其**作出时**的
/// `GameView`，保证中间决策行的 `turn`/`scenario` 正确）；由 call 方逐个 emit。
pub fn calc_ramen_training(
    trainer: &RamenMctsTrainer, game: &mut RamenGame, rng: &mut StdRng, json_mode: bool, reason_slot: &LastReasonSink,
    emit_info: &dyn Fn(&str)
) -> Result<Vec<(DecisionInfo, GameView)>> {
    // 链式决策收集：每次 select_action 捕获 DecisionInfo + 该阶段 view
    let mut out: Vec<(DecisionInfo, GameView)> = Vec::new();
    let mut any_decision = false;

    {
        // 对当前阶段做一次决策：捕获决策与其阶段 view，返回选中的动作
        // （g / out / rng 走参数，避免闭包长期独占借用与下方直接使用冲突；仅捕获共享 trainer）
        //
        // 2026-09 扩展：snapshot select_action 前的 stage 填到 info.decision_kind——
        // 让 AIRedirector 端按 partial decision 类型分发。trainer 不感知 stage，
        // 由"发起决策的 umaai"统一管理。
        let decide =
            |g: &mut RamenGame, out: &mut Vec<(DecisionInfo, GameView)>, rng: &mut StdRng| -> Result<Option<RamenAction>> {
                let before_stage = g.stage.clone();
                let actions = match g.stage {
                    RamenStage::NextTurn | RamenStage::Settlement | RamenStage::SuperRamenSelect => {
                        // 回合边界 / RMJ 结算 / 超级拉面选择 —— 等下一条 JSON，AI 不出推荐
                        Vec::new()
                    }
                    _ => g.list_actions()?
                };
                if actions.is_empty() {
                    return Ok(None);
                }
                let idx = trainer.select_action(g, &actions, rng)?;
                let chosen = actions[idx].clone();
                let view = g.view();
                // `last_decision()` 仅对真正走过 MCTS 搜索的阶段返回 `Some`；其它（门控
                // 关闭的 `region`、合并 RamenSelect 路径、单候选等）返回 `None`。
                // 仅以下场景需合成一条输出（手写 fallback）——否则该决策没有结果可 emit：
                // 1) 地区选择（门控关闭，手写策略）——最初"无结果"的问题；
                // 2) **比赛回合**：`is_race_turn()` 下落 `Train`，list_actions 只有"比赛"
                //    一个固定动作，trainer 因单候选直接落 fallback、不搜索，`last_decision()`
                //    为 `None`，不合成的话 calc_ramen_training 返回空、屏幕上无策略输出。
                // 其余 None 阶段保持旧行为（决策仍返回但**不**合成、不 emit）。
                let mut info = match trainer.last_decision() {
                    Some(info) => Some(info),
                    None if before_stage == RamenStage::RegionSelect
                        || (before_stage == RamenStage::Train && g.is_race_turn()) =>
                    {
                        Some(fallback_decision(&actions, idx, &before_stage))
                    }
                    None => None,
                };
                if let Some(mut info) = info.take() {
                    info.decision_kind = ramen_stage_kind(before_stage).to_string();
                    out.push((info, view));
                }
                Ok(Some(chosen))
            };

        if let Some(chosen) = decide(game, &mut out, rng)? {
            any_decision = true;
            let before_stage = game.stage.clone();
            let before_turn = game.turn();
            // 定向连续决策判定：仅两个场景在决策#1 后继续给下一个决策
            let need_continue = (before_stage == RamenStage::RamenSelect && !chosen.is_eating_ramen())
                || (before_stage == RamenStage::Train && before_turn == 1);

            if need_continue {
                // 应用决策#1 并推进一个阶段（RamenSelect 不吃 → Train；Train(turn==1) → AfterTrain）
                game.apply_action(&chosen, rng)?;
                if game.next() {
                    // 逐阶段推进直到下一决策点（或真正需要等新 JSON 的结算 / 超级拉面阶段）
                    const MAX_STAGE_LOOP: usize = 32;
                    for _ in 0..MAX_STAGE_LOOP {
                        match game.stage {
                            // RMJ 结算 / 超级拉面选择：等新 JSON，不再续
                            RamenStage::Settlement | RamenStage::SuperRamenSelect => break,
                            // 到达决策点：给出决策#2，随后停止（定向，不再向下级联）
                            RamenStage::RamenSelect
                            | RamenStage::SpecialSelect
                            | RamenStage::Train
                            | RamenStage::RegionSelect => {
                                // 连续决策的**第 2 个决策**前先通知下游 "AI 还在算这一回合"：
                                // 必须在真正执行决策#2（select_action，MCTS 可能耗时数秒）
                                // **之前**打出屏幕并 emit——首决策前已在 watch loop 入口发射过
                                // compute_start，不需要重复。
                                eprintln!("计算后续动作...");
                                emit_info("compute_next_step");
                                let _ = decide(game, &mut out, rng)?;
                                break;
                            }
                            // 自动阶段（Begin / BeginAfterRegionSelect / Distribute / AfterTrain / NextTurn）：
                            // 交给 umasim 的 run_stage 执行载荷，再用 next() 推进到下一阶段
                            _ => {
                                game.run_stage(trainer, rng)?;
                                if !game.next() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // 屏幕侧（human mode）输出推理结果（回合头部打印由 call 方在调用前完成）
    if !json_mode {
        if any_decision {
            if let Some(data) = reason_slot.take() {
                for line in render_reason_lines(&data) {
                    println!("{line}");
                }
            }
        }
        println!("{}", "[按 F2 保存当前回合状态]".bright_black());
    }
    Ok(out)
}

/// RamenStage → decision_kind 字符串映射
///
/// 拉面模块在 calc_ramen_training 内部 snapshot stage 填这个字段——trainer 不关心，
/// 由"发起决策的 umaai"统一管理（按用户拍板）。
fn ramen_stage_kind(stage: RamenStage) -> &'static str {
    match stage {
        RamenStage::Begin => "begin",
        RamenStage::Distribute => "distribute",
        RamenStage::RamenSelect => "ramen_select",
        RamenStage::SpecialSelect => "special_select",
        RamenStage::Train => "train",
        RamenStage::AfterTrain => "after_train",
        RamenStage::NextTurn => "next_turn",
        RamenStage::RegionSelect => "region_select",
        RamenStage::SuperRamenSelect => "super_ramen_select",
        RamenStage::Settlement => "settlement",
        RamenStage::BeginAfterRegionSelect => "begin_after_region_select"
    }
}

/// 为 MCTS 手写 fallback 阶段的决策合成一条最小 `DecisionInfo`
///
/// [`Trainer::last_decision`] 只在**真正走过 MCTS 搜索**时返回 `Some`；门控关闭的阶段
/// （如默认配置 `ramen_search_stages="train,ramen"` 下未开启的 `region`）落入手写
/// fallback，`last_decision()` 为 `None`，但手写策略确实作出了选择——导致该阶段
/// 没有任何决策结果输出。这里按本次候选列表与选中下标合成一条无搜索评分的决策信息，
/// 保证 region_select 等阶段也有结果可 emit（candidate_scores 为空，luck baseline 退化按等权）。
fn fallback_decision(actions: &[RamenAction], chosen_idx: usize, before_stage: &RamenStage) -> DecisionInfo {
    DecisionInfo {
        action_index: chosen_idx,
        score: 0.0,
        decision_kind: ramen_stage_kind(before_stage.clone()).to_string(),
        candidate_scores: Vec::new(),
        candidate_descriptions: actions.iter().map(|a| a.to_string()).collect(),
        candidate_n: Vec::new(),
        scenario_extra: None
    }
}