use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use anyhow::Result;
#[cfg(feature = "cli")]
use inquire::Select;
use log::info;
use rand::{Rng, prelude::StdRng, seq::SliceRandom};

use crate::{
    game::{ActionEnum, BaseAction, Game, Trainer},
    gamedata::EventChoice
};

// 导出手写逻辑训练员、数据收集训练员、神经网络训练员和 MCTS 训练员
//pub mod collector_trainer;
pub mod handwritten_trainer;
pub mod local_ramen_trainer;
pub mod logging_trainer;
pub mod mcts_trainer;
pub mod ramen_handwritten_trainer;
//pub mod mean_filter_collector_trainer;
//pub mod neural_net_trainer;

//pub use collector_trainer::CollectorTrainer;
pub use handwritten_trainer::HandwrittenTrainer;
pub use local_ramen_trainer::LocalRamenTrainer;
pub use logging_trainer::LoggingTrainer;
pub use mcts_trainer::MctsTrainer;
pub use ramen_handwritten_trainer::RamenHandwrittenTrainer;
//pub use mean_filter_collector_trainer::MeanFilterCollectorTrainer;
//pub use neural_net_trainer::NeuralNetTrainer;

/// 猴子训练师
pub struct RandomTrainer;

impl<G: Game> Trainer<G> for RandomTrainer {
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize> {
        let mut random_index: Vec<_> = (0..actions.len()).collect();
        let mut ret = None;
        random_index.shuffle(rng);
        for i in &random_index {
            // 优先休息，回心情，训练。都不满足就随机选择
            if game.uma().vital < 45 {
                if actions[*i].as_base_action() == Some(BaseAction::Sleep) {
                    ret = Some(*i);
                    break;
                }
            } else if game.uma().motivation < 5 {
                if matches!(
                    actions[*i].as_base_action(),
                    Some(BaseAction::NormalOuting) | Some(BaseAction::FriendOuting)
                ) {
                    ret = Some(*i);
                    break;
                }
            } else {
                if matches!(actions[*i].as_base_action(), Some(BaseAction::Train(_))) {
                    ret = Some(*i);
                    break;
                }
            }
        }
        // 没有基础动作候选时（拉面杯三阶段决策中中间步骤动作全为 None）：
        // 优先选有"实质内容"的候选（RamenAction 专属：ramen 非 None 或 special_targets 含非零值），
        // 避免误选"占位"动作（如 SpecialSelect 阶段默认生成的 [0,0,0]）。
        if ret.is_none() {
            for i in &random_index {
                if let Some(ra) = any_ramen_action(&actions[*i]) {
                    if ra.ramen.is_some() || ra.special_targets.is_some_and(|t| t.iter().any(|&x| x > 0)) {
                        ret = Some(*i);
                        break;
                    }
                }
            }
        }
        // 如果没有找到匹配的动作，随机选择一个
        let ret = ret.unwrap_or(random_index[0]);
        info!("吗喽训练员选择：{:?}", actions[ret]);
        Ok(ret)
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        let ret = rng.random_range(0..choices.len());
        let explain: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        info!("当前选项: {}, 随机选择选项 {}", explain.join(" / "), ret + 1);
        Ok(ret)
    }
}

/// 若 `action` 是 `RamenAction` 则返回其引用（用于在不耦合泛型 `Action` 的前提下读取拉面杯特有字段）。
///
/// 拉面杯的三阶段决策中，中间步骤动作（如 `RamenSelect`/`SpecialSelect`）的
/// `as_base_action()` 返回 `None`，且 RamenAction 字段（`ramen`/`special_targets`）承载决策。
/// RandomTrainer 在没有基础动作候选时，优先选这些字段"有内容"的动作，
/// 避免误选"占位候选"导致后续阶段库存不足。
fn any_ramen_action<A>(_action: &A) -> Option<&crate::game::ramen::RamenAction> {
    None
}

/// 手动训练师
///
/// 默认通过 `inquire` 让玩家在终端中手动选择动作/事件选项。
///
/// 同时支持 **mock 输入队列**：通过 `with_mock_inputs` 注入预定义的用户输入序列，
/// 队列非空时优先消费（取队首后弹出），队列空时回退到"选第一个候选"。
/// 这一机制仅用于自动化测试，真实玩家场景下应使用 `new()`。
pub struct ManualTrainer {
    /// mock 输入队列（自动化测试用）
    pub mock_inputs: Rc<RefCell<VecDeque<String>>>,
    /// mock 队列耗尽时的回退模式
    /// - `Interactive`（默认）：回退到 inquire（真实玩家模式）
    /// - `PickFirst`：选第一个候选（自动化测试模式，避免阻塞）
    pub fallback: FallbackMode
}

/// mock 输入队列耗尽时的回退策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    /// 回退到 inquire（真实玩家）
    Interactive,
    /// 自动选第一个候选（自动化测试）
    PickFirst
}

impl Default for ManualTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualTrainer {
    /// 创建一个空的 ManualTrainer（真实玩家模式，回退到 inquire）
    pub fn new() -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(VecDeque::new())),
            fallback: FallbackMode::Interactive
        }
    }

    /// 创建一个带 mock 输入队列的 ManualTrainer（仅用于自动化测试）
    ///
    /// 队列中的字符串会按顺序作为玩家输入消费：
    /// - 优先从队列中读取用户输入并匹配候选
    /// - 队列耗尽后回退到 `PickFirst`（选第一个候选），保证测试流程不阻塞
    ///
    /// 真实玩家场景请使用 `new()`。
    pub fn with_mock_inputs(inputs: Vec<String>) -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(inputs.into_iter().collect())),
            fallback: FallbackMode::PickFirst
        }
    }

    /// 弹出队首输入（若队列非空），否则返回 None
    fn pop_mock_input(&self) -> Option<String> {
        self.mock_inputs.borrow_mut().pop_front()
    }

    /// 处理 mock 输入回退逻辑（仅 PickFirst 模式，Interactive 模式由调用方处理）
    fn fallback_pick_first(&self, len: usize, item_desc: &str) -> Result<usize> {
        if len == 0 {
            return Err(anyhow::anyhow!("{item_desc} 候选为空"));
        }
        Ok(0)
    }
}

impl<G: Game> Trainer<G> for ManualTrainer {
    fn select_action(&self, _game: &G, actions: &[<G as Game>::Action], _rng: &mut StdRng) -> Result<usize> {
        // 优先消费 mock 输入
        if let Some(input) = self.pop_mock_input() {
            return actions
                .iter()
                .position(|x| x.to_string() == input)
                .ok_or_else(|| anyhow::anyhow!("mock 输入未匹配到候选动作: {input}"));
        }
        match self.fallback {
            FallbackMode::PickFirst => self.fallback_pick_first(actions.len(), "动作"),
            #[cfg(feature = "cli")]
            FallbackMode::Interactive => {
                //  println!("{actions:#?}");
                let selected = Select::new("请选择:", actions.to_vec())
                    .with_page_size(actions.len())
                    .prompt()?;
                actions
                    .iter()
                    .position(|x| *x == selected)
                    .ok_or_else(|| anyhow::anyhow!("未找到该动作: {selected}"))
            }
            #[cfg(not(feature = "cli"))]
            FallbackMode::Interactive => Err(anyhow::anyhow!(
                "ManualTrainer::Interactive 需要 cli feature（inquire 终端交互）；\
                     当前编译未启用 cli，请改用 ManualTrainer::with_mock_inputs(..)"
            ))
        }
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let explain: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
        // 优先消费 mock 输入
        if let Some(input) = self.pop_mock_input() {
            return explain
                .iter()
                .position(|x| x == &input)
                .ok_or_else(|| anyhow::anyhow!("mock 输入未匹配到候选选项: {input}"));
        }
        match self.fallback {
            FallbackMode::PickFirst => self.fallback_pick_first(explain.len(), "事件选项"),
            #[cfg(feature = "cli")]
            FallbackMode::Interactive => {
                let selected = Select::new("请选择:", explain.clone()).prompt()?;
                explain
                    .iter()
                    .position(|x| x == &selected)
                    .ok_or_else(|| anyhow::anyhow!("未找到该选项: {selected}"))
            }
            #[cfg(not(feature = "cli"))]
            FallbackMode::Interactive => Err(anyhow::anyhow!(
                "ManualTrainer::Interactive 需要 cli feature（inquire 终端交互）；\
                     当前编译未启用 cli，请改用 ManualTrainer::with_mock_inputs(..)"
            ))
        }
    }
}
