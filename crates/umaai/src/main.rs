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
        onsen::{game::OnsenGame},
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{GAMECONSTANTS, init_global_with_config},
    global,
    neural::Evaluator,
    output::{
        DecisionInfo,
        DecisionSink,
        GameView,
        HumanReadableSink,
        StdoutJsonSink,
        reason::{DecisionReasonData, DecisionReasonSink, render_reason_lines}
    },
    search::SearchConfig,
    trainer::{MctsTrainer, RamenMctsTrainer},
    utils::{check_working_dir, init_logger, load_game_config}
};

use crate::{
    luck_score::LuckScoreTracker,
    protocol::urafile::UraFileWatcher,
    utils::SAVED_GAME
};

pub mod luck_score;
pub mod protocol;
pub mod utils;

/// 缓存最近一次决策理由的 sink（每回合覆写）
///
/// 接到 `RamenMctsTrainer::reason_sink`：把 `DecisionReasonData` 缓存到内部
/// `Mutex<Option<…>>`，由 main 在 human mode 下取出后调 [`render_reason_lines`]
/// `println!` 到屏幕。`emit_decision_reason` 内部的 `info!` 调用**已被 trainer
/// `.verbose(false)` 关闭**，避免双打印；同时也避开 umaai 默认关闭日志的现状。
pub struct LastReasonSink {
    inner: Mutex<Option<DecisionReasonData>>
}

impl LastReasonSink {
    fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(None) })
    }

    fn take(&self) -> Option<DecisionReasonData> {
        // 取走副本，留 None 给下一次覆写
        self.inner.lock().expect("reason sink").take()
    }
}

impl DecisionReasonSink for LastReasonSink {
    fn emit(&self, reason: &DecisionReasonData) {
        *self.inner.lock().expect("reason sink") = Some(reason.clone());
    }
}

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
/// `GameView`，保证中间决策行的 `turn`/`scenario` 正确）；由主循环逐个 emit。
pub fn calc_ramen_training(
    trainer: &RamenMctsTrainer, game: &mut RamenGame, rng: &mut StdRng, json_mode: bool, reason_slot: &LastReasonSink
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
                // 仅地区选择（手写 fallback）需要合成一条输出——这是最初"无结果"的问题；
                // 其余 None 阶段保持旧行为（决策仍返回但**不**合成、不 emit）。
                let mut info = match trainer.last_decision() {
                    Some(info) => Some(info),
                    None if before_stage == RamenStage::RegionSelect => {
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
    // 屏幕侧（human mode）输出推理结果（回合头部打印由 main 在调用前完成）
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
    trainer: &MctsTrainer, game: &G, sink: &Arc<dyn DecisionSink>, tracker: &mut LuckScoreTracker, chara_id: u64,
    decision_kind: &str
) {
    // onsen 路径没有 reason_sink，传 None——scenario_extra.reason 不挂
    emit_with_luck_decision(trainer.last_decision(), game, sink, tracker, chara_id, None, decision_kind, None);
}

/// 把已提取的 `DecisionInfo` 喂给 sink：挂 luck score 字段 + emit。
///
/// 与 [`emit_with_luck`] 区别在于**不依赖具体 trainer 类型**——只要 trainer 实现了
/// `Trainer<G>` 并返回 `DecisionInfo` 即可。拉面分支（`RamenMctsTrainer` 等）走这里。
///
/// **2026-09 扩展**：
/// - `reason_data`：拉面 MCTS 路径从 `LastReasonSink.take()` 取 `DecisionReasonData`，
///   挂到 `scenario_extra.reason` 让 AIRedirector 拿到完整 human mode reason 信息
///   （metric / chosen_desc / chosen_mean / chosen_n / rivals[]）。其他 trainer 传 `None`。
/// - `decision_kind`：由 main.rs 外部传入（按用户拍板"由发起决策的umaai从外部保存状态"）——
///   标明这条决策属于哪种（"ramen_select" / "special_select" / "train" / "region_select" /
///   "super_ramen_select" / "event"）。C# 端按此字段分发 partial decision。
/// - `ramen_action`：仅 ramen 路径传 `Some(&str)`——`RamenAction::to_string()` 的结果，
///   含吃面 + 隐藏诀窍 + 操作三阶段信息（按用户拍板"AIRed 端只显示不解析"）。
fn emit_with_luck_decision<G: Game>(
    last_decision: Option<DecisionInfo>, game: &G, sink: &Arc<dyn DecisionSink>,
    tracker: &mut LuckScoreTracker, chara_id: u64,
    reason_data: Option<&DecisionReasonData>,
    decision_kind: &str,
    ramen_action: Option<&str>
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

    let _turn_delta = tracker.on_new_turn(
        chara_id,
        t_n_baseline,
        game.turn(),
        game.max_turn(),
        global!(GAMECONSTANTS).mcts_turn_bonus,
    );

    // 每候选 action_luck：T(n, action_i) - T(n)（AIRedirector 关心，玩家模式跳过）
    let action_luck = serde_json::json!(
        info.candidate_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, (s as f64) - t_n_baseline))
            .collect::<std::collections::HashMap<usize, f64>>()
    );

    // 顶层 decision_kind（外部传入——按用户拍板"由发起决策的 umaai 从外部保存状态"）
    info.decision_kind = decision_kind.to_string();

    // 挂载 scenario_extra：snapshot + action_luck（必挂）+ reason（仅拉面 MCTS）+
    // ramen_action（仅 ramen 路径——按用户拍板"对吃面情况要输出隐藏诀窍用法"，
    // 这里直接存 to_string 字符串，AIRed 端只显示不解析）
    let extra = match serde_json::to_value(tracker.snapshot()) {
        Ok(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("action_luck".into(), action_luck);
                // reason：拉面 MCTS 路径挂，其他 trainer 不挂——按 trainer 支持度灵活
                if let Some(data) = reason_data {
                    if let Ok(reason_v) = serde_json::to_value(data) {
                        obj.insert("reason".into(), reason_v);
                    }
                }
                // ramen_action：仅 ramen 路径填（"吃面/X(替换Ax1+Bx2)" 等）
                if let Some(action_text) = ramen_action {
                    obj.insert("ramen_action".into(), action_text.into());
                }
            }
            Some(v)
        }
        Err(_) => None
    };
    info.scenario_extra = extra;

    sink.emit(&info, &game.view());
}

/// RamenStage → decision_kind 字符串映射
///
/// main.rs 在 calc_ramen_training 内部 snapshot stage 填这个字段——trainer 不关心，
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

/// 实际的主函数
async fn main_guard() -> Result<()> {
    let args = parse_args()?;
    if args.help {
        print_help_and_exit();
    }

    // sink 选择必须在 colored::set_override 之前——后者是全局副作用
    //
    // `--json` 分支额外保留 `StdoutJsonSink` 的具体类型句柄（`json_sink`）：
    // `DecisionSink` trait 只覆盖决策 emit（info/error 不在内）。`emit_info` /
    // `emit_error` 是 `StdoutJsonSink` 的额外方法，main 在 watch loop 的各触发点
    // 显式调——human 模式下 `json_sink` 为 `None`，闭包 no-op。
    let json_sink: Option<Arc<StdoutJsonSink>>;
    let sink: Arc<dyn DecisionSink> = if args.json {
        // JSON 模式关闭 ANSI：colored 即使 --no-color 也可能输出 ANSI reset，
        // 影响 AIRedirector 解析。详见集成文档 §3.2.6 第 3 条。
        colored::control::set_override(false);
        let js = Arc::new(StdoutJsonSink);
        json_sink = Some(js.clone());
        js
    } else {
        json_sink = None;
        Arc::new(HumanReadableSink)
    };
    let json_mode = args.json;

    // info / error 发射器闭包：human 模式 no-op；json 模式转发到 StdoutJsonSink
    // （stdout 严格只 JSON——不再走 eprintln/println 污染流）。闭包按 Fn 借用
    // json_sink，可在 watch loop 内反复调用。
    let emit_info = |event: &str| {
        if let Some(ref js) = json_sink {
            js.emit_info(event);
        }
    };
    let emit_error = |message: &str| {
        if let Some(ref js) = json_sink {
            js.emit_error(message);
        }
    };

    // 启动横幅走 stderr（避免污染 JSON 模式的 stdout 流）
    eprintln!("{}", to_art("Ramen-AI".to_string(), "small", 0, 1, 0).expect("here"));
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
    //
    // verbose=false：关闭 trainer 内部 `info!("[回合 X] 首选...")` 的 `log::info!` 上屏
    // （避免与下方 human mode 下手动调 `render_reason_lines` 双打印，且
    // umaai 默认关 log，trainer 走 info! 看不到）。DecisionReasonData 通过
    // `with_reason_sink(LastReasonSink)` 缓存到 `reason_slot`。
    let ramen_mcts_config = SearchConfig::new_game_config(&game_config);
    let ramen_stages = umasim::trainer::RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?;
    let reason_slot = LastReasonSink::new();
    let ramen_trainer = RamenMctsTrainer::new(ramen_mcts_config)
        .with_stages(ramen_stages)
        .verbose(true)
        .with_reason_sink(reason_slot.clone());

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
        Ok(w) => {
            // watcher 就绪：通知 AIRed 子进程已连接并进入监听状态
            emit_info("connected");
            w
        }
        Err(e) => {
            // watcher init 失败：json 模式发 error 行；human 模式保留原 warn 日志
            emit_error(&format!("watcher 初始化失败: {e}"));
            log::warn!("UraFileWatcher init 失败: {e}，main 不进入 watch loop，程序正常退出（exit 0）");
            return Ok(());
        }
    };

    // watcher 启动成功后才 spawn hotkey_handler（见上方注释——避免失败路径 hang）
    //
    // **临时停用**（2026-09 重构期）：ctrl-s 保存当前回合状态功能暂不可用，
    // 同时热键循环里无限 poll crossterm event 在玩家无 stdin 场景（容器 / CI / AIRedirector 接管）
    // 会持续占用 tokio worker 配额，且功能本身在 AI 通道下无意义。后续重构时再恢复，
    // 或改为「玩家调试入口受 feature gate / 命令行参数控制」。
    //
    // tokio::spawn(async move {
    //     hotkey_handler().await;
    // });

    // Luck score 跟踪器（Step 5）：每回合 baseline 累加 + 切局检测，snapshot
    // 挂在 DecisionInfo::scenario_extra 下发给 AIRedirector。
    let mut luck_tracker = LuckScoreTracker::new();

    loop {
        let contents = watcher.watch("thisTurn.json")?;
        // 收到一份新 JSON：通知 AIRed "开始计算本回合"
        emit_info("compute_start");
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
                    // 检测到新一局：通知 AIRed 重置 UI 状态
                    emit_info("new_game");
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
                    calc_onsen_event(&trainer, &game, &mut rng, json_mode)?;
                } else {
                    calc_onsen_training(&trainer, &mut game, &mut rng, json_mode)?;
                }

                // 回合决策完成后统一 emit（带 luck score 挂载）—— 见 emit_with_luck 注释
                // decision_kind 标明 partial decision 类型：onsen 路径下要么是 train 要么是 event
                let onsen_kind = if !game.unresolved_events.is_empty() { "event" } else { "train" };
                emit_with_luck(&trainer, &game, &sink, &mut luck_tracker, chara_id, onsen_kind);

                // 计算完成：通知下游 watcher 进入阻塞状态
                eprintln!("计算完成，等待新数据...");
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
                    // 检测到新一局：通知 AIRed 重置 UI 状态
                    emit_info("new_game");
                    eprintln!("{}", "---- 拉面: 育成开始 ----".bright_yellow());
                    luck_tracker = LuckScoreTracker::new();
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
                let chain = calc_ramen_training(&ramen_trainer, &mut game, &mut rng, json_mode, &reason_slot)?;
                let last_index = chain.len().saturating_sub(1);
                for (i, (info, view)) in chain.iter().take(last_index).enumerate() {
                    // 链式决策的**第 2 个及之后**的决策前发 compute_next_step：
                    // 通知 AIRed "AI 还在算这一回合"——首决策前已在 watch loop 入口
                    // 发过 compute_start，不需要重复。
                    if i > 0 {
                        eprintln!("计算后续动作...");
                        emit_info("compute_next_step");
                    }
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
                            &sink,
                            &mut luck_tracker,
                            chara_id,
                            reason_slot.take().as_ref(),
                            &last_kind,
                            ramen_action_text.as_deref(),
                        );
                    }
                }

                // 计算完成：通知下游 watcher 进入阻塞状态
                eprintln!("计算完成，等待新数据...");
            }
            Err(e) => {
                // json 模式：发 error JSON 行（不再用 println 污染 stdout 严格 JSON 流）
                // human 模式：保留原 println 红色提示，玩家可见
                emit_error(&format!("解析回合信息出错: {e}"));
                if !json_mode {
                    println!("{}", format!("解析回合信息出错: {e}").red());
                    println!("----------");
                }
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
