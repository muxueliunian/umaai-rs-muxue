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
    }
};

use crate::decision::{emit_with_luck_decision, LastReasonSink, LuckScoreTracker};

/// 处理拉面剧本的一个回合快照（`thisTurn.json` → `ParsedGame::Ramen`）
///
/// 承接原 main watch loop 的 ramen 分支：`Begin`（未 dispatch）早退、切局检测、
/// human 屏幕打印、链式决策 emit 与 luck 挂载。
///
/// 注意：`single_mode_chara_id` 为切局键（`None` 时退化用 uma_id）。
pub fn process_ramen<T: Trainer<RamenGame>>(
    mut game: RamenGame,
    single_mode_chara_id: Option<u64>,
    trainer: &T,
    reason_slot: &LastReasonSink,
    sink: &Arc<dyn DecisionSink>,
    luck_tracker: &mut LuckScoreTracker,
    rng: &mut StdRng,
    json_mode: bool,
    emit_info: &dyn Fn(&str)
) -> Result<()> {
    if game.stage == RamenStage::Begin {
        // into_game 没 dispatch（事件 / 结算 / 数据不全），等下一条 JSON。
        // 本轮无决策，但仍要收尾 compute_done——保证每轮 JSON 流
        // `compute_start → compute_done` 成对，C# 端"计算中"状态不会悬挂。
        emit_info("compute_done");
        return Ok(());
    }

    // 切局检测：`single_mode_chara_id` 变化 → 新一局开始。
    // C# 端 single_mode_chara_id 单调递增，同 uma_id 重复训练也能识别新局。
    // 协议字段缺失（None）时退化到 uma_id 兜底（旧 json / 测试 fixture）。
    //
    // ❗**身份登记走 `begin_game`，不依赖运气数值那条路**：`last_single_mode_id`
    // 以前只在 `on_new_turn` 里写，而那条路只有**带搜索评分**的决策才会走。整局网络模式
    // （`ramen_trainer_policy = "nn"`）一次搜索都不跑，于是同一局的每一帧都被判成新局，
    // 屏幕反复打印「育成开始」、下游反复重置 UI。`begin_game` 只登记身份、不造假 baseline，
    // 有评分的那条路行为不变（见 [`LuckScoreTracker::begin_game`]）。
    let chara_id = single_mode_chara_id
        .unwrap_or_else(|| game.uma().uma_id as u64);
    if luck_tracker.begin_game(chara_id) {
        // 检测到新一局：通知 AIRed 重置 UI 状态
        emit_info("new_game");
        eprintln!("{}", "---- 拉面: 育成开始 ----".bright_yellow());
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
    // 连续决策：链式决策（一个快照对应多个决策）。
    // 中间决策（除最后一个）已在 calc_ramen_training 内部、决策#2 计算前
    // 立即 emit（不触 luck）——保证 JSON 流顺序为 decision#1 →
    // compute_next_step → decision#2；此处 `chain` 只剩末项，由下方走
    // 完整 luck 挂载（baseline 每回合只更新一次）。
    let chain = calc_ramen_training(trainer, &mut game, rng, json_mode, reason_slot, emit_info, sink)?;
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

        // **无搜索评分**的决策（`candidate_scores` 为空）。这一类比原先以为的多：
        // 默认配置下 region 未开时的地区选择、**比赛回合单候选**、RamenSelect 单候选短路，
        // 以及网络接管的地区决策与整局网络模式下的**每一步**动作决策。
        // 它们没有真正的搜索评分，走 luck 挂载只会以 baseline=0 污染 luck tracker
        // （后续回合运气全被算错），且 sink 打印的「期望评分」只是回合加成换算、
        // 运气恒 0 会误导。故直接 emit（不触 luck）；`HumanReadableSink` 会为带来源标签
        // 的决策打印「选择…（<来源>）」，**照实说**是谁做的，不再一律印成手写。
        // 搜索决策（常见 train/ramen_select，以及 mcts_compare 下真搜过的地区）仍走
        // 完整 luck 挂载。
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
    emit_info("compute_done");
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
/// 返回链式决策 `Vec<(DecisionInfo, GameView)>`。
///
/// **emit 时机**：链式决策的**中间项**（决策#1）在函数内部、决策#2 真正执行前
/// （`compute_next_step` 通知前）经 `sink` 立即 emit（不触 luck）——保证 JSON 流
/// 顺序为 `decision#1 → compute_next_step → decision#2`，下游不会先收到
/// "还在计算"通知而以为本回合没有结果。**末项**（决策#2，或非链式场景的决策#1）
/// 留在返回值中，由 call 方走完整 luck 挂载；每个决策附带其**作出时**的
/// `GameView`，保证决策行的 `turn`/`scenario` 正确。
pub fn calc_ramen_training<T: Trainer<RamenGame>>(
    trainer: &T, game: &mut RamenGame, rng: &mut StdRng, json_mode: bool, reason_slot: &LastReasonSink,
    emit_info: &dyn Fn(&str), sink: &Arc<dyn DecisionSink>
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
                // 关闭的 `region`、单候选等）返回 `None`。
                // 仅以下场景需合成一条输出（手写 fallback）——否则该决策没有结果可 emit：
                // 1) 地区选择（门控关闭，手写策略）——最初"无结果"的问题；
                // 2) **比赛回合**：`is_race_turn()` 下落 `Train`，list_actions 只有"比赛"
                //    一个固定动作，trainer 因单候选直接落 fallback、不搜索，`last_decision()`
                //    为 `None`，不合成的话 calc_ramen_training 返回空、屏幕上无策略输出。
                // 3) **RamenSelect 决策（兜底）**：合并搜索路径 2026-09 起按面聚合后暴露
                //    `last_decision()`；仅当回落三阶段逻辑（合并候选 ≤ 1 / 单候选短路）
                //    仍为 `None` 时才在此合成（吃面不链式时同样需要决策行）。
                //    其余 None 阶段保持旧行为（决策仍返回但**不**合成、不 emit）。
                let mut info = match trainer.last_decision() {
                    Some(info) => Some(info),
                    None if before_stage == RamenStage::RegionSelect
                        || (before_stage == RamenStage::Train && g.is_race_turn())
                        || before_stage == RamenStage::RamenSelect =>
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
                // 决策#1 是链式决策的**中间项**：先把它的结果 emit 出去（下游拿到
                // 即时反馈），再从 `out` 移除——否则它要等本函数返回后才由 call 方
                // emit，而 `compute_next_step` 已在下面决策#2 前发出，JSON 顺序变成
                // `compute_next_step` 先于任何决策结果到达，下游会误以为本回合没算。
                // 末项（决策#2）仍留在 `out` 返回，由 call 方走完整 luck 挂载。
                // 注：RamenSelect 决策已在上方 decide 合成（见合成条件 3），
                // `out` 通常有内容；其它 None 阶段不合成时这里无输出，与旧行为一致。
                if let Some((info, view)) = out.first() {
                    sink.emit(info, view);
                }
                out.clear();
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
///
/// 整局网络模式的包装层（`crate::ramen_nn`）也要填这个字段，故提升到 crate 可见：
/// 两处必须是**同一份**映射，各写一遍迟早会出现「同一阶段两个名字」。
pub(crate) fn ramen_stage_kind(stage: RamenStage) -> &'static str {
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

/// 为 `last_decision()` 为 `None` 的阶段合成一条最小 `DecisionInfo`
///
/// [`Trainer::last_decision`] 只在**真正走过 MCTS 搜索**时返回 `Some`；门控关闭的阶段
/// （如默认配置 `ramen_search_stages="train,ramen"` 下未开启的 `region`）落入手写
/// fallback，合并搜索的 `RamenSelect` 路径（吃面 / 不吃面）也会清掉 last_summary——
/// 这些情况 `last_decision()` 均为 `None`，但策略确实作出了选择，导致该阶段没有任何
/// 决策结果输出。这里按本次候选列表与选中下标合成一条无搜索评分的决策信息，保证
/// region_select / ramen_select 等阶段也有结果可 emit（candidate_scores 为空，
/// luck baseline 退化按等权）。
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


#[cfg(test)]
mod tests {
    use std::{env, sync::Mutex};

    use rand::SeedableRng;
    use umasim::{
        game::InheritInfo,
        gamedata::init_global,
        search::SearchConfig,
        trainer::{RamenMctsTrainer, RamenSearchStages},
        utils::get_workspace_root
    };

    use super::*;
    use crate::{decision::LuckScoreTracker, utils::Checks};

    /// 把 `decision` / `info` 两路输出按发生顺序记进一个流，用来核对 JSON 流次序
    ///
    /// 只记 `type:kind` 标签，不比任何指纹——次序与完整性看的就是这串标签本身。
    #[derive(Default)]
    struct EventLog {
        events: Mutex<Vec<String>>,
        /// **真正 emit 出去**的决策原件（按顺序）
        ///
        /// 与标签流并存：标签流看次序，原件看内容。挂在 sink 上而不是读
        /// `trainer.last_decision()`——后者没经过 `process_ramen` 的 luck 挂载，
        /// 「luck 挂载会不会把 `scenario_extra` 里的东西冲掉」这类问题只有原件能回答。
        infos: Mutex<Vec<DecisionInfo>>
    }

    impl EventLog {
        /// 追加一条事件
        fn push(&self, s: String) {
            self.events.lock().expect("事件流锁").push(s);
        }

        /// 追加一条 emit 出去的决策原件
        fn push_info(&self, info: DecisionInfo) {
            self.infos.lock().expect("决策原件锁").push(info);
        }

        /// 按发生顺序取出事件标签
        fn take(&self) -> Vec<String> {
            self.events.lock().expect("事件流锁").clone()
        }

        /// 按发生顺序取出 emit 出去的决策原件
        fn decisions(&self) -> Vec<DecisionInfo> {
            self.infos.lock().expect("决策原件锁").clone()
        }
    }

    /// 记录 `DecisionSink::emit` 的 sink（决策行进同一个事件流）
    struct RecordingSink(Arc<EventLog>);

    impl DecisionSink for RecordingSink {
        fn emit(&self, info: &DecisionInfo, _view: &GameView) {
            self.0.push_info(info.clone());
            // `src` 是决策来源标签（未标注记 `-`）；`reason` 记 scenario_extra 里
            // 有没有搜索理由——地区决策挂上它就说明把上一步搜索的理由带过来了。
            let has_reason = info
                .scenario_extra
                .as_ref()
                .is_some_and(|v| v.get("reason").is_some());
            self.0.push(format!(
                "decision:{}(scores={},cands={},idx={},src={},reason={has_reason})",
                info.decision_kind,
                info.candidate_scores.len(),
                info.candidate_descriptions.len(),
                info.action_index,
                info.source_label().unwrap_or("-")
            ));
        }
    }

    /// 把搜索压到最小的训练员（本测试只看输出次序，不看棋力）
    fn small_trainer() -> Result<RamenMctsTrainer> {
        Ok(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
                .with_stages(RamenSearchStages::parse("train,ramen")?)
                .verbose(false)
        )
    }

    /// 建一局标准拉面（卡组含新友人卡，`newgame` 会校验）
    fn new_game() -> Result<RamenGame> {
        RamenGame::newgame(
            101901,
            &[303124, 303114, 303084, 303094, 303064, 303054],
            InheritInfo {
                blue_count: [15, 0, 3, 0, 0],
                extra_count: [0, 40, 40, 20, 20, 40]
            }
        )
    }

    /// 把新局自然推进到 turn 1 的 `Train` 阶段（链式决策的触发点之一）
    ///
    /// # 错误
    ///
    /// 推进途中报错、提前终局，或超过步数上限仍没到达目标阶段时报错。
    fn advance_to_turn1_train(
        mut game: RamenGame, trainer: &RamenMctsTrainer, rng: &mut StdRng
    ) -> Result<RamenGame> {
        const MAX_STEPS: usize = 400;
        for _ in 0..MAX_STEPS {
            if game.turn() == 1 && game.stage == RamenStage::Train {
                return Ok(game);
            }
            if !game.next() {
                anyhow::bail!("推进到 turn 1 Train 之前本局就结束了");
            }
            if game.turn() == 1 && game.stage == RamenStage::Train {
                return Ok(game);
            }
            game.run_stage(trainer, rng)?;
        }
        anyhow::bail!("{MAX_STEPS} 步内未到达 turn 1 的 Train 阶段")
    }

    /// 地区回合（turn 2）：恰好一条 `region_select` 决策，不触发链式，`compute_done` 收尾
    ///
    /// 覆盖上游 e5cdd64 之后的输出契约：每轮 `compute_start → … → compute_done` 成对，
    /// 地区决策**不遗漏、不重复**；它走 `fallback_decision`，`candidate_scores` 为空
    /// —— 即没有把上一步搜索的理由挂到这次地区决策上。
    #[test]
    fn test_region_turn_emits_one_decision_and_compute_done() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let mut game = new_game()?;
        game.base.turn = 2;
        game.stage = RamenStage::RegionSelect;
        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);

        process_ramen(game, Some(1), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let decisions: Vec<_> = ev.iter().filter(|e| e.starts_with("decision:")).collect();
        c.check(decisions.len() == 1, &format!("恰好 1 条决策（实际 {}）", decisions.len()));
        c.check(
            decisions.first().is_some_and(|d| d.starts_with("decision:region_select")),
            "该决策的 decision_kind 是 region_select"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("scores=0")),
            "地区决策不带搜索评分（没有把上一步搜索的理由挂过来）"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("cands=10")),
            "第 1 年 10 个候选全部进 candidate_descriptions"
        );
        c.check(
            decisions.first().is_some_and(|d| d.contains("reason=false")),
            "地区决策的 scenario_extra 里没有搜索理由"
        );
        c.check(
            !ev.iter().any(|e| e == "info:compute_next_step"),
            "地区回合不触发链式决策，无 compute_next_step"
        );
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "恰好一条 compute_done 收尾"
        );
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 是最后一条");
        c.finish()
    }

    /// **无搜索评分的连续帧不得反复报新局**：同 ID 两帧 + 换 ID 一帧 → 恰好 2 次 `new_game`
    ///
    /// review#2 的回归。旧实现把「育成身份」挂在运气数值那条路上（`last_single_mode_id`
    /// 只在 `on_new_turn` 里写），而那条路只有**带搜索评分**的决策才会走。于是无搜索的
    /// 模式（整局网络，以及这里用来复现的「地区未开搜索」）下同一局每一帧都被判成新局：
    /// 屏幕反复打印「育成开始」，按协议处理 `new_game` 的客户端反复重置 UI。
    ///
    /// ❗本用例**故意不依赖模型**：用默认装配 + `ramen_search_stages="train,ramen"` 跑地区帧，
    /// 决策同样没有候选评分、同样不走 luck 挂载，复现条件与整局网络一致，但任何机器都能跑。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_unscored_frames_do_not_repeat_new_game() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        // ❗三帧共用同一个 tracker（真实主循环也是如此），否则观测不到「跨帧记住身份」
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260915);

        // 同一局连喂两帧，再换一局喂一帧
        for chara_id in [7u64, 7, 8] {
            let mut game = new_game()?;
            game.base.turn = 2;
            game.stage = RamenStage::RegionSelect;
            process_ramen(
                game, Some(chara_id), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info
            )?;
        }

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let new_games = ev.iter().filter(|e| *e == "info:new_game").count();
        println!("new_game 次数 = {new_games}（期望 2）");
        c.check(new_games == 2, &format!("恰好 2 次 new_game（实际 {new_games}）"));
        c.check(
            ev.first().is_some_and(|e| e == "info:new_game"),
            "第 1 帧报新局"
        );
        // 第 2 次 new_game 必须落在第 3 帧那一段里：它前面应当已经有 2 条 compute_done
        let second_pos = ev
            .iter()
            .enumerate()
            .filter(|(_, e)| **e == "info:new_game")
            .map(|(i, _)| i)
            .nth(1);
        let done_before = second_pos
            .map(|p| ev[..p].iter().filter(|e| **e == "info:compute_done").count())
            .unwrap_or(0);
        println!("第 2 次 new_game 之前的 compute_done 数 = {done_before}（期望 2）");
        c.check(done_before == 2, "第 2 次 new_game 出现在换 ID 的那一帧，不是第 2 帧");
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 3,
            "三帧各有一条 compute_done 收尾"
        );
        c.check(tracker.last_single_mode_id() == Some(8), "tracker 记住了最后一局的身份");
        c.finish()
    }

    /// **网络接管**的地区回合：走真实 `process_ramen` 与真实输出路径
    ///
    /// review#1 的回归：地区决策没有搜索评分，旧实现的渲染分支因此把它印成
    /// 「（手写逻辑）」。本测试用**真正的** [`RegionNnTrainer`] 跑一遍
    /// `process_ramen`，核对经过实际输出逻辑之后：
    ///
    /// 1. 恰好一条 `region_select` 决策，`compute_done` 完整收尾；
    /// 2. 该决策的来源标签是 `region_nn`（**不会**被渲染成手写）；
    /// 3. 没有搜索评分、也没有挂上一步搜索的理由；
    /// 4. 选中的候选解码成合法的三地区组合。
    ///
    /// 上面那条手写路径的测试用的是 `RamenMctsTrainer`，**不能**当作本路径已覆盖。
    ///
    /// 模型不在版本库里；缺模型时跳过并显式声明零覆盖。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_nn_turn_is_not_labelled_handwritten() -> Result<()> {
        use std::path::Path;

        use crate::region::RegionNnTrainer;
        use umasim::{
            game::ramen::{Operation, rules::validate_region_selection},
            output::decision::SOURCE_REGION_NN,
            trainer::{RamenNnTrainer, SpecialSelectMode}
        };

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let model = Path::new("saved_models/arms/ens_G2mix_g123.onnx");
        if !model.is_file() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 「NN 地区决策不被误标手写」这条本次**零覆盖**，绿色不代表它还正确。");
            return c.finish();
        }

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let mut game = new_game()?;
        game.base.turn = 2;
        game.stage = RamenStage::RegionSelect;
        let actions = game.list_actions()?;
        let trainer = RegionNnTrainer::new(
            small_trainer()?,
            RamenNnTrainer::load(model)?
                .with_race_shield(true)
                .with_special_mode(SpecialSelectMode::Canonical)
        );
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);

        process_ramen(game, Some(3), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let decisions: Vec<_> = ev.iter().filter(|e| e.starts_with("decision:")).collect();
        c.check(decisions.len() == 1, &format!("恰好 1 条决策（实际 {}）", decisions.len()));
        let first = decisions.first().map(|s| s.as_str()).unwrap_or_default();
        c.check(first.starts_with("decision:region_select"), "decision_kind 是 region_select");
        c.check(
            first.contains(&format!("src={SOURCE_REGION_NN}")),
            "来源标签是 region_nn（渲染端不会印成「手写逻辑」）"
        );
        c.check(first.contains("scores=0"), "不伪造搜索评分");
        c.check(first.contains("cands=10"), "第 1 年 10 个候选全部进 candidate_descriptions");
        c.check(first.contains("reason=false"), "没有把上一步搜索的理由挂到这次地区决策上");
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "恰好一条 compute_done 收尾"
        );
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 是最后一条");

        // 选中的候选必须解码成合法地区组合
        let idx = first
            .split("idx=")
            .nth(1)
            .and_then(|t| t.split(',').next())
            .and_then(|t| t.parse::<usize>().ok());
        match idx.and_then(|i| actions.get(i)).map(|a| a.operation) {
            Some(Operation::RegionSelect(r)) => {
                println!("选中地区组合 {r:?}");
                c.check(validate_region_selection(0, &r), "选中的三地区组合合法（第 1 年区间、互不相同）");
            }
            other => c.check(false, &format!("选中候选不是 RegionSelect：{other:?}"))
        }
        c.finish()
    }

    /// **对照模式**：两条推荐都算出来，执行哪条由模式决定，对照结果不丢
    ///
    /// 四种装配各跑一遍真实 `process_ramen`（含 luck 挂载那一段），核对**真正 emit
    /// 出去的那条决策**：
    ///
    /// | 装配 | 执行侧 | 来源 | 候选评分 |
    /// |---|---|---|---|
    /// | `nn_compare` + 不搜 region | 网络 | `region_nn` | 无 |
    /// | `nn_compare` + **搜** region | 网络 | `region_nn` | 无 |
    /// | `mcts_compare` + 不搜 region | 手写基策 | `region_handwritten` | 无 |
    /// | `mcts_compare` + **搜** region | 那次真搜索本身 | `region_search` | **有** |
    ///
    /// 四条都必须带上 `scenario_extra.region_compare`——最后一条是 review 点：它走
    /// luck 挂载，旧实现在那里整个重建 `scenario_extra`，对照结果与来源标签会凭空消失。
    ///
    /// ❗本用例用的是**与主程序一致**的理由接线：`LastReasonSink` 外套 `ReasonGate`，
    /// 门同时交给搜索训练员与接管器。
    ///
    /// 模型不在版本库里；缺模型时跳过并显式声明零覆盖。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_compare_modes_execute_the_declared_side() -> Result<()> {
        use std::path::Path;

        use serde_json::Value;

        use crate::{
            decision::ReasonGate,
            region::{CompareMode, RegionNnTrainer}
        };
        use umasim::{
            output::decision::{SOURCE_REGION_HANDWRITTEN, SOURCE_REGION_NN, SOURCE_REGION_SEARCH},
            trainer::{RamenNnTrainer, SpecialSelectMode}
        };

        /// 一次对照跑留下的全部可观测物
        struct CompareRun {
            /// 决策标签行（次序 / 形状）
            line: String,
            /// **真正 emit 出去**的那条决策原件
            info: Option<DecisionInfo>,
            /// 跑完之后理由槽里是否还留着理由
            reason_left: bool
        }

        /// 从 emit 出去的原件里取 `scenario_extra.region_compare`
        fn compare_of(info: &Option<DecisionInfo>) -> Option<Value> {
            info.as_ref()?
                .scenario_extra
                .as_ref()?
                .get("region_compare")
                .cloned()
        }

        /// 取对照载荷里的一个字符串字段（缺失时为空串）
        fn field(v: &Option<Value>, key: &str) -> String {
            v.as_ref()
                .and_then(|x| x.get(key))
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string()
        }

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let model = Path::new("saved_models/arms/ens_G2mix_g123.onnx");
        if !model.is_file() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 「对照模式执行声明的那一侧」这条本次**零覆盖**，绿色不代表它还正确。");
            return c.finish();
        }
        let load_nn = || -> Result<RamenNnTrainer> {
            Ok(RamenNnTrainer::load(model)?
                .with_race_shield(true)
                .with_special_mode(SpecialSelectMode::Canonical))
        };
        // `gated=false` 复现「没接理由门」的旧行为，用来证明理由隔离那条观测不是空跑
        let run = |nn_primary: bool, stages: &str, gated: bool| -> Result<CompareRun> {
            let log = Arc::new(EventLog::default());
            let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
            let log_info = Arc::clone(&log);
            let emit_info = move |e: &str| log_info.push(format!("info:{e}"));
            // ❗与 main.rs 同一条接线：LastReasonSink 外套 ReasonGate
            let reason_slot = LastReasonSink::new();
            let gate = ReasonGate::new(reason_slot.clone());
            let mcts = RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
                .with_stages(RamenSearchStages::parse(stages)?)
                .verbose(false)
                .with_reason_sink(gate.clone());
            let mut trainer = RegionNnTrainer::new(mcts, load_nn()?).with_compare(CompareMode {
                nn_primary,
                // 本用例按 json_mode=true 跑：对照行**不得**上 stdout（那是严格 JSON 流）
                print: false
            });
            if gated {
                trainer = trainer.with_reason_gate(gate.clone());
            }
            let mut game = new_game()?;
            game.base.turn = 2;
            game.stage = RamenStage::RegionSelect;
            let mut tracker = LuckScoreTracker::new();
            let mut rng = StdRng::seed_from_u64(20260915);
            process_ramen(game, Some(7), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;
            let ev = log.take();
            for e in &ev {
                println!("  {e}");
            }
            Ok(CompareRun {
                line: ev
                    .iter()
                    .find(|e| e.starts_with("decision:"))
                    .cloned()
                    .unwrap_or_default(),
                info: log.decisions().into_iter().next(),
                reason_left: reason_slot.take().is_some()
            })
        };

        // 1) nn_compare + 不搜 region：执行网络，参照侧是手写基策
        println!("-- nn_compare / stages=train,ramen --");
        let r1 = run(true, "train,ramen", true)?;
        c.check(r1.line.contains(&format!("src={SOURCE_REGION_NN}")), "nn_compare 执行网络那条");
        c.check(r1.line.contains("scores=0"), "nn_compare：网络那条不伪造搜索评分");
        let cmp1 = compare_of(&r1.info);
        println!("region_compare = {cmp1:?}");
        c.check(cmp1.is_some(), "对照结果挂在 emit 出去的 scenario_extra.region_compare 上");
        c.check(field(&cmp1, "baseline_kind") == "handwritten", "不搜 region 时对照侧记为 handwritten");
        c.check(field(&cmp1, "executed") == "nn", "executed 记为 nn");
        c.check(
            cmp1.as_ref()
                .and_then(|v| v.get("nn_index"))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                == r1.info.as_ref().map(|i| i.action_index),
            "执行下标就是网络那条"
        );

        // 2) nn_compare + **搜** region：参照侧是真搜索，它的理由不得落到执行推荐上
        println!("-- nn_compare / stages=train,ramen,region（理由隔离）--");
        let r2 = run(true, "train,ramen,region", true)?;
        let cmp2 = compare_of(&r2.info);
        println!("region_compare = {cmp2:?}");
        c.check(
            r2.line.contains(&format!("src={SOURCE_REGION_NN}")),
            "nn_compare + 搜 region：仍执行网络那条"
        );
        c.check(r2.line.contains("scores=0"), "网络那条不借用参照搜索的评分");
        c.check(r2.line.contains("reason=false"), "网络那条不挂参照搜索的理由");
        c.check(field(&cmp2, "baseline_kind") == "search", "参照侧照实记为 search（它确实搜了）");
        c.check(!r2.reason_left, "跑完后理由槽是空的：参照搜索的理由被挡在门外");
        // 对照组：拆掉理由门，同一配置应当**留下**理由——否则上面那条观测是空跑
        let ungated = run(true, "train,ramen,region", false)?;
        println!("拆掉理由门后理由槽非空 = {}", ungated.reason_left);
        match ungated.reason_left {
            true => c.check(!r2.reason_left, "理由隔离确实起了作用（有门 → 空，无门 → 非空）"),
            false => {
                println!("❗❗ 本次搜索没有产出理由（search_n=2 下 analyze_narrow_win 可能返回 None）");
                println!("❗❗ 「理由隔离」这条本次**零覆盖**，绿色不代表它还正确。");
            }
        }

        // 3) mcts_compare + 不搜 region：执行手写基策，来源必须照实说是手写
        println!("-- mcts_compare / stages=train,ramen --");
        let r3 = run(false, "train,ramen", true)?;
        c.check(
            r3.line.contains(&format!("src={SOURCE_REGION_HANDWRITTEN}")),
            "mcts_compare + 不搜 region：来源标为 region_handwritten"
        );
        c.check(r3.line.contains("scores=0"), "手写基策那条没有搜索评分");
        let cmp3 = compare_of(&r3.info);
        println!("region_compare = {cmp3:?}");
        c.check(field(&cmp3, "executed") == "baseline", "executed 记为 baseline");

        // 4) mcts_compare + **搜** region：执行的就是那条搜索，对照结果必须扛过 luck 挂载
        println!("-- mcts_compare / stages=train,ramen,region --");
        let r4 = run(false, "train,ramen,region", true)?;
        println!("{}", r4.line);
        c.check(!r4.line.contains("scores=0"), "开了 region 搜索时执行的那条带候选评分");
        c.check(r4.line.starts_with("decision:region_select"), "仍然是一条 region_select 决策");
        c.check(
            r4.line.contains(&format!("src={SOURCE_REGION_SEARCH}")),
            "来源标为 region_search（既不是手写也不是网络）"
        );
        let cmp4 = compare_of(&r4.info);
        println!("region_compare = {cmp4:?}");
        c.check(cmp4.is_some(), "❗对照结果扛过了 luck 挂载（旧实现会在这里整个丢掉）");
        c.check(field(&cmp4, "baseline_kind") == "search", "执行侧照实记为 search");
        c.check(field(&cmp4, "executed") == "baseline", "executed 记为 baseline");
        c.check(field(&cmp4, "index_space") == "actions", "载荷写明下标空间是候选全表");
        c.check(
            r4.info
                .as_ref()
                .and_then(|i| i.scenario_extra.as_ref())
                .and_then(|v| v.get("total_luck_score"))
                .is_some(),
            "luck 快照照样挂上了（合并而不是二选一）"
        );
        c.finish()
    }

    /// **整局无搜索 NN**：链式的两步都出结果、都标了真实来源、都不伪造评分
    ///
    /// 走 turn 1 的 `Train`——它会链式带出第 1 年的地区决策，一次覆盖两个阶段：
    ///
    /// 1. 恰好 2 条决策（`train` + `region_select`），`compute_next_step` 夹在中间；
    /// 2. 两条都**没有**候选评分，也没有挂任何搜索理由（本模式一次搜索都不跑）；
    /// 3. 两条都带来源标签，且标签落在本模式的已知集合内；
    /// 4. 候选描述完整（下游靠它把 `action_index` 映射成动作名）。
    ///
    /// 模型不在版本库里；缺模型时跳过并显式声明零覆盖。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_whole_game_nn_labels_every_step() -> Result<()> {
        use std::path::Path;

        use crate::ramen_nn::WholeGameNnTrainer;
        use umasim::{
            output::decision::{
                SOURCE_RAMEN_HANDWRITTEN_STAGE, SOURCE_RAMEN_NN, SOURCE_RAMEN_RACE_GATE,
                SOURCE_RAMEN_SINGLE_CANDIDATE
            },
            trainer::{RamenNnTrainer, SpecialSelectMode}
        };

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        let model = Path::new("saved_models/arms/ens_G2mix_g123.onnx");
        if !model.is_file() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 「整局 NN 每一步都有结果且来源真实」这条本次**零覆盖**，绿色不代表它还正确。");
            return c.finish();
        }

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let trainer = WholeGameNnTrainer::new(
            RamenNnTrainer::load(model)?
                .with_race_shield(true)
                .with_special_mode(SpecialSelectMode::Canonical)
        );
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260915);
        // ❗必须**自然推进**到 turn 1 的 Train（直接摆 stage 会留下空的训练分布）。
        // 推进用小搜索训练员，只是把局面走到那一帧，与本用例要观测的决策无关。
        let game = advance_to_turn1_train(new_game()?, &small_trainer()?, &mut rng)?;

        process_ramen(game, Some(11), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let decisions: Vec<_> = ev.iter().filter(|e| e.starts_with("decision:")).collect();
        c.check(decisions.len() == 2, &format!("链式共 2 条决策（实际 {}）", decisions.len()));
        c.check(
            decisions.first().is_some_and(|d| d.starts_with("decision:train")),
            "第 1 条是 train（普通训练回合也有结果，不再是空屏）"
        );
        c.check(
            decisions.get(1).is_some_and(|d| d.starts_with("decision:region_select")),
            "第 2 条是链式带出的第 1 年地区决策"
        );
        c.check(
            ev.iter().any(|e| e == "info:compute_next_step"),
            "链式通知照常发出（链式决策没被本模式破坏）"
        );
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "恰好一条 compute_done 收尾"
        );

        let known = [
            SOURCE_RAMEN_NN,
            SOURCE_RAMEN_RACE_GATE,
            SOURCE_RAMEN_SINGLE_CANDIDATE,
            SOURCE_RAMEN_HANDWRITTEN_STAGE
        ];
        for (i, info) in log.decisions().iter().enumerate() {
            let src = info.source_label().unwrap_or("-").to_string();
            println!(
                "  #{i} kind={} src={src} scores={} cands={} idx={}",
                info.decision_kind,
                info.candidate_scores.len(),
                info.candidate_descriptions.len(),
                info.action_index
            );
            c.check(info.candidate_scores.is_empty(), &format!("#{i} 不伪造搜索评分"));
            c.check(info.candidate_n.is_empty(), &format!("#{i} 不伪造 rollout 局数"));
            c.check(
                (info.score - 0.0f32).abs() < f32::EPSILON,
                &format!("#{i} score 保持 0（policy logits 不是终局分）")
            );
            c.check(known.contains(&src.as_str()), &format!("#{i} 来源 {src} 在本模式的已知集合内"));
            c.check(
                info.action_index < info.candidate_descriptions.len(),
                &format!("#{i} 选中下标落在候选描述表内（下游能映射出动作名）")
            );
            c.check(
                info.scenario_extra
                    .as_ref()
                    .and_then(|v| v.get("reason"))
                    .is_none(),
                &format!("#{i} 没有挂任何搜索理由（本模式一次搜索都不跑）")
            );
        }
        c.check(reason_slot.take().is_none(), "理由槽自始至终是空的");
        c.finish()
    }

    /// 链式决策（turn 1 的 Train）：`decision#1 → compute_next_step → … → compute_done`
    ///
    /// 这是上游 e5cdd64 修正的次序：中间决策必须**先于** `compute_next_step` 到达，
    /// 下游才不会以为本回合没算。
    #[test]
    fn test_chained_decision_order() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);
        // ❗必须**自然推进**到 turn 1 的 Train：直接摆 stage 会留下空的训练分布，
        // 搜索一展开就越界。这里按 umasim 的阶段机推进，等价于客户端收到那一帧快照。
        let game = advance_to_turn1_train(new_game()?, &trainer, &mut rng)?;

        process_ramen(game, Some(2), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        for e in &ev {
            println!("  {e}");
        }
        let pos = |needle: &str| ev.iter().position(|e| e.starts_with(needle));
        let first_decision = pos("decision:");
        let next_step = pos("info:compute_next_step");
        let done = pos("info:compute_done");
        let decisions = ev.iter().filter(|e| e.starts_with("decision:")).count();
        c.check(decisions == 2, &format!("链式共 2 条决策（实际 {decisions}）"));
        c.check(next_step.is_some(), "发出了 compute_next_step");
        c.check(
            matches!((first_decision, next_step), (Some(a), Some(b)) if a < b),
            "决策#1 先于 compute_next_step"
        );
        c.check(
            matches!((next_step, done), (Some(b), Some(d)) if b < d),
            "compute_next_step 先于 compute_done"
        );
        c.check(
            ev.iter().filter(|e| *e == "info:compute_done").count() == 1,
            "compute_done 只发一次"
        );
        c.check(ev.last().is_some_and(|e| e == "info:compute_done"), "compute_done 是最后一条");
        c.finish()
    }

    /// `Begin` 早退回合也补 `compute_done`（上游 e5cdd64 的成对保证）
    #[test]
    fn test_begin_stage_still_emits_compute_done() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();

        let log = Arc::new(EventLog::default());
        let sink: Arc<dyn DecisionSink> = Arc::new(RecordingSink(Arc::clone(&log)));
        let log_info = Arc::clone(&log);
        let emit_info = move |e: &str| log_info.push(format!("info:{e}"));

        let mut game = new_game()?;
        game.stage = RamenStage::Begin;
        let trainer = small_trainer()?;
        let reason_slot = LastReasonSink::new();
        let mut tracker = LuckScoreTracker::new();
        let mut rng = StdRng::seed_from_u64(20260914);

        process_ramen(game, Some(3), &trainer, &reason_slot, &sink, &mut tracker, &mut rng, true, &emit_info)?;

        let ev = log.take();
        println!("{ev:?}");
        c.check(ev == vec!["info:compute_done".to_string()], "Begin 早退只发 compute_done");
        c.finish()
    }
}
