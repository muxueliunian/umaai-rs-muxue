//! 拉面杯回合信息输出（Agent 对话文本流风格）
//!
//! 目标：把 `ramen_manual` 的屏幕输出从「散落的 diag 日志 + 终端表格」整理为
//! 面向 AI 助手 / 玩家的叙事化文本流，按四类信息分节呈现：
//!
//! - **局面（Situation）**：马娘状态 / 回合基本信息 / 剧本状态 / 训练分布表 / 训练明细
//! - **候选（Options）**：交给 Trainer 选择的 Action 列表
//! - **选择（Action）**：Trainer 选中的具体单个项目
//! - **效果（Effect）**：训练 / 比赛 / 事件的结果
//!
//! 本模块同时提供 [`RecordingTrainer`]（记录每次决策的候选与选择），供测试与
//! `ramen_manual` 复用；测试 `test_turn_output_baseline` 用固定种子跑一个完整
//! 回合并输出四类信息，作为后续格式调整的基线。

use std::cell::RefCell;

use anyhow::Result;
use colored::Colorize;
use rand::prelude::StdRng;

use crate::{
    game::{
        ActionEnum,
        BaseAction,
        Game,
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData}
};

/// 一次决策的记录（局面/候选/选择 三类的中间载体）
#[derive(Debug, Clone)]
pub struct TurnDecision {
    /// 回合（0-based，与 `Game::turn()` 一致）
    pub turn: i32,
    /// 决策阶段（RamenSelect / SpecialSelect / Train / RegionSelect / 事件）
    pub stage: String,
    /// 全部候选的**选项名**文本（按传入顺序，渲染时亮黄显示）
    pub candidates: Vec<String>,
    /// 与 [`Self::candidates`] 等长的**内联效果预览**（渲染时默认色；无预览为空串）
    pub candidate_details: Vec<String>,
    /// 与 [`Self::candidates`] 等长的**配方信息**（RamenSelect 吃面候选的原始诀窍
    /// 配方，渲染时 cyan；无配方为空串）
    pub candidate_recipes: Vec<String>,
    /// 选中的候选索引
    pub selected: usize,
    /// 选中候选的可读文本（选项名）
    pub selected_desc: String
}

/// 记录型包装训练员
///
/// 包装任意 [`Trainer<RamenGame>`]，在每次 `select_action` / `select_event_choice`
/// 时把候选列表与选中项记入 [`Self::log`]，供上层输出"候选 / 选择"两节文本。
/// 决策本身委托给被包装的 inner trainer，不改变任何行为。
///
/// `verbose = true` 时为**实时输出模式**（ramen_manual 等交互场景）：
/// 决策前先打印候选列表（选项名亮黄 + 内联预览白色），决策后打印选择确认；
/// `log` 仍照常记录，测试可复用。
pub struct RecordingTrainer<T> {
    /// 被包装的真实训练员
    pub inner: T,
    /// 全部决策记录（按发生顺序）
    pub log: RefCell<Vec<TurnDecision>>,
    /// 是否实时打印候选列表与选择（默认 false：仅记录）
    pub verbose: bool
}

impl<T> RecordingTrainer<T> {
    /// 包装一个训练员并开始记录
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            log: RefCell::new(Vec::new()),
            verbose: false
        }
    }

    /// 追加一条决策记录
    fn record(
        &self, game: &RamenGame, stage: &str, candidates: Vec<String>, candidate_details: Vec<String>,
        candidate_recipes: Vec<String>, selected: usize
    ) {
        let selected_desc = candidates.get(selected).cloned().unwrap_or_default();
        self.log.borrow_mut().push(TurnDecision {
            turn: game.turn(),
            stage: stage.to_string(),
            candidates,
            candidate_details,
            candidate_recipes,
            selected,
            selected_desc
        });
    }

    /// 候选的可读文本（选项名 + 配方 + 内联效果预览）：
    /// - Train 阶段训练候选：`(速训练, "", 速60 力15 39pt 体力-22 诀窍槽 A+6 B+5 C+8)`
    /// - RamenSelect 阶段吃面候选：`(吃面/中山-全, 配方A2B0C3, (训+20,友情+50,...))`
    /// - RegionSelect 阶段地区候选：`(札幌-速, 配方A2B2C1, youqing50 pt_bonus50 训练位[速])`
    /// - 其他动作：`(Display 文本, "", "")`
    fn candidate_text(&self, game: &RamenGame, a: &RamenAction) -> Result<(String, String, String)> {
        match game.stage {
            RamenStage::Train => {
                if let Some(BaseAction::Train(train)) = a.as_base_action() {
                    let text = game.train_candidate_preview(train as usize)?;
                    let (name, detail) = Self::split_preview(&text);
                    return Ok((name, String::new(), detail));
                }
            }
            RamenStage::RamenSelect => {
                if let Some(region_idx) = a.ramen {
                    let text = game.ramen_candidate_preview(region_idx)?;
                    let (name, detail) = Self::split_preview(&text);
                    // 当前拉面的原始诀窍配方（如 A2 B0 C3）
                    let recipe = crate::gamedata::ramen::RAMENDATA
                        .get()
                        .and_then(|data| data.region_feeling.get(region_idx))
                        .map(|&[fa, fb, fc]| format!("配方A{fa}B{fb}C{fc}"))
                        .unwrap_or_default();
                    return Ok((name, recipe, detail));
                }
            }
            _ => {}
        }
        Ok((a.to_string(), String::new(), String::new()))
    }

    /// 实时输出模式的候选列表渲染（选项名亮黄、内联预览白色）
    pub fn render_candidates(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<String> {
        let mut lines = vec![];
        lines.push(format!(
            "== 候选 [回合 {} · {:?}] {} 个 ==",
            game.turn() + 1,
            game.stage,
            actions.len()
        ));
        for (i, a) in actions.iter().enumerate() {
            let (name, recipe, detail) = self.candidate_text(game, a)?;
            let mut parts = vec![format!("  {}. {}", i + 1, name.bright_yellow())];
            if !recipe.is_empty() {
                parts.push(recipe.cyan().to_string());
            }
            if !detail.is_empty() {
                parts.push(detail);
            }
            lines.push(parts.join(" "));
        }
        Ok(lines.join("\n"))
    }

    /// 把 `"选项名 效果预览..."` 拆成 `(选项名, 效果预览)`；无空格时预览为空
    fn split_preview(text: &str) -> (String, String) {
        match text.split_once(' ') {
            Some((name, detail)) => (name.to_string(), detail.to_string()),
            None => (text.to_string(), String::new())
        }
    }

    /// 当前已记录的决策数（用于取"本回合新增"区间）
    pub fn log_len(&self) -> usize {
        self.log.borrow().len()
    }

    /// 取 `start` 之后（含）的决策记录切片
    pub fn decisions_from(&self, start: usize) -> Vec<TurnDecision> {
        self.log.borrow().iter().skip(start).cloned().collect()
    }
}

impl<T: Trainer<RamenGame>> Trainer<RamenGame> for RecordingTrainer<T> {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        // 实时模式：按阶段分摊，避免每回合重复打三次相同信息
        // - RamenSelect（回合第一阶段，Distribute 完成后的全状态）：
        //     打马娘状态 + 剧本状态 + 训练明细 + 候选
        // - SpecialSelect（buff 未应用，训练值与 RamenSelect 阶段等价）：只打候选
        // - Train（吃面已 ground_ramen_effects，buff 已应用）：重打训练明细 + 候选
        if self.verbose {
            match game.stage {
                RamenStage::RamenSelect => {
                    // Distribute 完成（回合第一阶段，吃面与否决定前）
                    // ——打马娘状态 + 剧本状态 + 训练明细，玩家可对比"不吃面"和"吃面"训练值
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
                RamenStage::Train => {
                    // 训练阶段：玩家已选了面或决定不吃。仅当玩家选了面（current_ramen 已落地）
                    // 时训练明细与 RamenSelect 阶段不同——重打印；不吃面时不重复打。
                    if game.ramen.current_ramen.is_some() {
                        if let Ok(dist_info) = game.explain_distribution() {
                            println!("{dist_info}");
                        }
                    }
                }
                RamenStage::SpecialSelect => {
                    // 吃面 buff 还未 ground（仍 pending），训练值与 RamenSelect 等价，跳过
                }
                _ => {}
            }
            println!("{}", self.render_candidates(game, actions)?);
        }
        let idx = self.inner.select_action(game, actions, rng)?;
        if self.verbose {
            if let Some(a) = actions.get(idx) {
                println!("→ 选择: {}\n", a.to_string().bright_yellow());
            }
        }
        let pairs = actions
            .iter()
            .map(|a| self.candidate_text(game, a))
            .collect::<Result<Vec<_>>>()?;
        let candidates: Vec<String> = pairs.iter().map(|(name, _, _)| name.clone()).collect();
        let candidate_recipes: Vec<String> = pairs.iter().map(|(_, recipe, _)| recipe.clone()).collect();
        let candidate_details: Vec<String> = pairs.iter().map(|(_, _, detail)| detail.clone()).collect();
        let stage = format!("{:?}", game.stage);
        self.record(game, &stage, candidates, candidate_details, candidate_recipes, idx);
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        if self.verbose {
            println!("== 候选 [回合 {} · 事件] {} 个 ==", game.turn() + 1, choices.len());
            for (i, x) in choices.iter().enumerate() {
                let text = x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | ");
                println!("  {}. {}", i + 1, text.bright_yellow());
            }
        }
        let idx = self.inner.select_choice(game, choices, rng)?;
        if self.verbose {
            let text = choices[idx].iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | ");
            println!("→ 选择: {}\n", text.bright_yellow());
        }
        let candidates: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        let candidate_details = vec![String::new(); candidates.len()];
        let candidate_recipes = vec![String::new(); candidates.len()];
        self.record(game, "事件", candidates, candidate_details, candidate_recipes, idx);
        Ok(idx)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        if self.verbose {
            println!(
                "== 候选 [回合 {} · 事件#{}] {} 个 ==",
                game.turn() + 1,
                event.id,
                choices.len()
            );
            for (i, x) in choices.iter().enumerate() {
                let text = x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | ");
                println!("  {}. {}", i + 1, text.bright_yellow());
            }
        }
        let idx = self.inner.select_event_choice(game, event, choices, rng)?;
        if self.verbose {
            let text = choices[idx].iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | ");
            println!("→ 选择: {}\n", text.bright_yellow());
        }
        let candidates: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        let candidate_details = vec![String::new(); candidates.len()];
        let candidate_recipes = vec![String::new(); candidates.len()];
        let stage = format!("事件#{}({:?})", event.id, game.stage);
        self.record(game, &stage, candidates, candidate_details, candidate_recipes, idx);
        Ok(idx)
    }
}

/// 渲染"局面"节文本（回合开始时的静态信息：马娘状态 + 回合信息 + 剧本状态）
///
/// 组合 `Game::explain()` / [`RamenGame::explain_ramen_info`]。训练分布表不属于
/// 回合开始局面——它在 Distribute 阶段人头分配后才生成，由调用方在
/// `Game::explain_distribution()` 输出（见测试 `test_turn_output_baseline`）。
pub fn render_turn_situation(game: &RamenGame) -> Result<String> {
    let mut lines = vec![];
    lines.push(game.explain()?);
    let ramen_info = game.explain_ramen_info();
    if !ramen_info.is_empty() {
        lines.push(ramen_info);
    }
    Ok(lines.join("\n"))
}

/// 渲染一次决策的"候选 + 选择"两节文本
///
/// 候选的**选项名**亮黄显示，内联效果预览与选中标记保持默认色；无色环境
/// （no-color / 非 tty）由 colored 自动退化为纯文本。
pub fn render_decision(d: &TurnDecision) -> String {
    // SpecialSelect 阶段仅 1 个候选且无隐藏风味替换（0 替换）时，玩家没有真实
    // 选择空间，用提示代替"1 个候选"列表
    if d.stage.starts_with("SpecialSelect") && d.candidates.len() == 1 && !d.candidates[0].contains("(替换") {
        return format!(
            "== 决策 [回合 {} · SpecialSelect] 无隐藏风味可选，自动 0 替换 ==",
            d.turn + 1
        )
        .bright_yellow()
        .to_string();
    }
    let mut lines = vec![];
    lines.push(format!(
        "== 决策 [回合 {} · {}] 候选 {} 个 ==",
        d.turn + 1,
        d.stage,
        d.candidates.len()
    ));
    for (i, c) in d.candidates.iter().enumerate() {
        let mark = if i == d.selected { "  ⇐ 选中" } else { "" };
        let recipe = d.candidate_recipes.get(i).cloned().unwrap_or_default();
        let detail = d.candidate_details.get(i).cloned().unwrap_or_default();
        let mut parts = vec![format!("  {}. {}", i + 1, c.bright_yellow())];
        if !recipe.is_empty() {
            parts.push(recipe.cyan().to_string());
        }
        if !detail.is_empty() {
            parts.push(detail);
        }
        let line = parts.join(" ");
        lines.push(if mark.is_empty() {
            line
        } else {
            format!("{line} {mark}")
        });
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bench::seeded_rngs,
        game::{InheritInfo, ramen::RamenStage, traits::Person},
        gamedata::init_global,
        trainer::RamenHandwrittenTrainer,
        utils::{get_workspace_root, init_test_logger}
    };

    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: InheritInfo = InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };
    const TEST_UMA_ID: u32 = 102601;

    /// 体力文字着色验证：体力 <35 红、<50 黄、>=50 亮绿（分段上色，无警示行）
    ///
    /// 强制启用颜色后输出三种档位的 `Game::explain()` 文本，肉眼核对体力片段
    /// 的 ANSI 码（`\x1b[31m` 红 / `\x1b[33m` 黄 / `\x1b[92m` 亮绿）。
    #[test]
    fn test_vital_color() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        colored::control::set_override(true); // 管道下强制颜色

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.uma.vital = 30;
        println!("体力30:\n{}\n", game.explain()?);
        game.uma.vital = 45;
        println!("体力45:\n{}\n", game.explain()?);
        game.uma.vital = 60;
        println!("体力60:\n{}\n", game.explain()?);

        colored::control::unset_override();
        Ok(())
    }

    /// 训练分布表人物着色验证：彩圈亮绿（去加号）、hint 感叹号亮黄（名字不变）、友人绿
    ///
    /// 构造速训练位置 [杏目(hint+彩圈), 友人, NPC] 的分布，强制颜色输出分布表，
    /// 核对 ANSI 码：彩圈 `\x1b[92m`、hint `\x1b[93m`、友人 `\x1b[32m`、NPC 无色。
    #[test]
    fn test_distribution_colors() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        colored::control::set_override(true); // 管道下强制颜色

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.base.turn = 31;
        game.add_friend_and_npcs()?; // persons[6] = 友人
        game.add_reporter(); // persons[12] = 记者
        // 速训练位置：杏目(hint+彩圈) + 友人 + NPC
        game.base.distribution = vec![vec![0, 6, 7], vec![], vec![], vec![], vec![]];
        game.persons[0].set_hint(true);
        game.persons[0].friendship = 100; // 彩圈条件：train_type 匹配 + 羁绊>=80
        game.ramen.selected_regions = [0, 1, 2];
        game.ramen.train_feeling_type = Some([crate::game::ramen::FeelingType::A; 5]);
        println!("{}", game.explain_distribution()?);

        colored::control::unset_override();
        Ok(())
    }

    /// 输出一个完整回合（回合 31）的四类信息基线
    ///
    /// 用固定种子真实跑到回合 31 开始（turn=31, stage=Begin；避开 102601 的生涯
    /// 比赛回合——回合 30 附近常有比赛，比赛回合无吃面/训练决策，不适合做样例），
    /// 前 30 回合以 error 级别静默跑过；第 31 回合切回 info 并分节输出：
    /// 局面（回合开始马娘/剧本状态 + Distribute 后的训练分布表）→ 逐条决策
    /// （候选/选择，RecordingTrainer 记录）→ 效果（规则层 diag 日志自然呈现）。
    #[test]
    fn test_turn_output_baseline() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error"); // 前 30 回合静默跑过
        let _ = init_global();

        let seed = 42u64;
        let (mut decision_rng, rule_master) = seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        let trainer = RecordingTrainer::new(RamenHandwrittenTrainer::default());

        // 真实跑到回合 31 开始（turn=31, stage=Begin）
        loop {
            if game.turn() == 31 && game.stage == RamenStage::Begin {
                break;
            }
            game.run_stage(&trainer, &mut decision_rng)?;
            if !game.next() {
                break;
            }
        }
        if game.turn() != 31 || game.stage != RamenStage::Begin {
            anyhow::bail!("应停在回合 31 开始，实际 turn={} stage={:?}", game.turn(), game.stage);
        }

        // 切回 info：第 31 回合的规则层 diag（效果）可见。Core-only 测试没有
        // flexi_logger/LOGGER，保持静默即可；测试验证的是流程与渲染，不依赖日志后端。
        #[cfg(feature = "cli")]
        if let Some(logger) = crate::gamedata::LOGGER.get() {
            let handle = logger.lock().map_err(|_| anyhow::anyhow!("LOGGER 锁中毒"))?;
            let spec = flexi_logger::LogSpecification::try_from("info")?;
            handle.set_new_spec(spec);
        }

        println!("╔════════════════════════════════════════════╗");
        println!(
            "║  完整回合信息基线（第 {} 回合 · turn={}）   ║",
            game.turn() + 1,
            game.turn()
        );
        println!("╚════════════════════════════════════════════╝");
        println!();
        println!("【局面 · 回合开始】");
        println!("{}", render_turn_situation(&game)?);
        println!();

        // 手动跑第 31 回合，按阶段输出
        let log_start = trainer.log_len();
        loop {
            let stage_before = game.stage.clone();
            if stage_before == RamenStage::NextTurn {
                break;
            }
            game.run_stage(&trainer, &mut decision_rng)?;
            // Distribute 后输出分布表（此时人头已分配）
            if stage_before == RamenStage::Distribute {
                println!("【局面 · 训练分布】");
                println!("{}", game.explain_distribution()?);
                println!();
            }
            if !game.next() {
                break;
            }
        }

        // 候选 / 选择：本回合全部决策
        let decisions = trainer.decisions_from(log_start);
        println!("【候选 / 选择】（{} 条决策）", decisions.len());
        for d in &decisions {
            println!("{}", render_decision(d));
        }
        println!();
        println!("回合结束: turn={} stage={:?}", game.turn(), game.stage);
        Ok(())
    }

    /// verbose 实时输出模式演示：包装 ManualTrainer（mock 全选第一），
    /// 跑前 3 个回合观察决策前的候选列表（亮黄选项名 + 白色内联预览）与选择确认
    #[test]
    fn test_verbose_demo() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let seed = 42u64;
        let (mut decision_rng, rule_master) = seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        let mut trainer = RecordingTrainer::new(crate::trainer::ManualTrainer::with_mock_inputs(vec![]));
        trainer.verbose = true;

        println!("===== verbose 实时输出演示（前 3 回合）=====");
        // 正确推进 3 个回合（turn 0/1/2），到 turn 3 的 Begin 前停止
        while game.turn() < 3 {
            game.run_stage(&trainer, &mut decision_rng)?;
            if !game.next() {
                break;
            }
        }
        println!("已记录 {} 条决策", trainer.log_len());
        Ok(())
    }
}
