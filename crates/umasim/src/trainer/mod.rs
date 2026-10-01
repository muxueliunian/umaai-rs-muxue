use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use anyhow::Result;
#[cfg(feature = "cli")]
use inquire::Select;
use log::info;
use rand::{Rng, prelude::StdRng, seq::SliceRandom};

use crate::{
    game::{
        ActionEnum, BaseAction, Game, Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData},
    output::DecisionInfo
};

pub mod handwritten_trainer;
pub mod local_ramen_trainer;
pub mod logging_trainer;
pub mod mcts_trainer;
pub mod ramen_handwritten_trainer;
pub mod ramen_mcts_trainer;
#[cfg(feature = "onnx")]
pub mod ramen_nn_trainer;
// 模块本身不门控：守门测试必须在默认 feature 下运行（见模块文档）。
// 收成 `pub(crate)` 会在默认 feature 下触发 dead_code 警告——唯一的调用方
// `ramen_nn_trainer` 挂在 onnx 上，故保持 `pub`，只让顶层 re-export 跟着 onnx 走
pub mod ramen_special_root;
//pub mod mean_filter_collector_trainer;
//pub mod neural_net_trainer;

pub use handwritten_trainer::HandwrittenTrainer;
pub use local_ramen_trainer::{LocalRamenTrainer, RecommendedRamenTrainer};
pub use logging_trainer::LoggingTrainer;
pub use mcts_trainer::MctsTrainer;
pub use ramen_handwritten_trainer::RamenHandwrittenTrainer;
pub use ramen_mcts_trainer::{RamenMctsTrainer, RamenSearchStages};
#[cfg(feature = "onnx")]
pub use ramen_nn_trainer::{NnPick, NnVia, RamenNnTrainer, SpecialSelectMode};
// 只有网络策略用得上它，故 re-export 跟着 onnx 走
#[cfg(feature = "onnx")]
pub use ramen_special_root::canonical_ramen_select_root;
//pub use mean_filter_collector_trainer::MeanFilterCollectorTrainer;
//pub use neural_net_trainer::NeuralNetTrainer;

/// 拉面搜索的 rollout 基策：手写推荐策略或神经网络
///
/// 存在的理由：[`FlatSearchGame::RolloutTrainer`] 是**关联类型**，多开一个类型
/// 参数会把 `FlatSearch` / `RamenMctsTrainer` / `umaai` 的调用签名全部掀开
/// （见 `search/searchable.rs` 的设计说明）。用枚举把「rollout 走哪个基策」
/// 变成运行时开关，既有默认行为不变——生产路径恒为 [`Self::Handwritten`]。
///
/// [`Self::Nn`] 是实验档：rollout **每一步**都要编码局面并跑一次网络推理，
/// 一次搜索的推理次数是「rollout 条数 × 剩余决策点数」的乘积级，生产
/// `search_n` 下不可行；只用于小预算量测，见 `tools/data_collection/nn_rollout_probe.rs`。
///
/// [`FlatSearchGame::RolloutTrainer`]: crate::search::FlatSearchGame::RolloutTrainer
pub enum RamenRolloutTrainer {
    /// 正式推荐手写策略（生产默认，经 `for_rollout()` 构造）
    Handwritten(RecommendedRamenTrainer),
    /// 神经网络策略（实验档，模型用 `Arc` 共享，克隆不重载）
    #[cfg(feature = "onnx")]
    Nn(RamenNnTrainer)
}

impl RamenRolloutTrainer {
    /// 手写推荐策略变体（与 `RamenGame::default_rollout_trainer` 同源）
    pub fn handwritten() -> Self {
        Self::Handwritten(RecommendedRamenTrainer::for_rollout())
    }

    /// 神经网络变体
    #[cfg(feature = "onnx")]
    pub fn nn(trainer: RamenNnTrainer) -> Self {
        Self::Nn(trainer)
    }
}

impl Trainer<RamenGame> for RamenRolloutTrainer {
    /// 转发给选定的基策
    ///
    /// # 错误
    ///
    /// 基策报错时原样返回——rollout 出错必须让搜索停下，而不是换个策略接着跑。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_action(game, actions, rng),
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.select_action(game, actions, rng)
        }
    }

    /// 事件选项转发（网络策略内部同样转交手写策略）
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_choice(game, choices, rng),
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.select_choice(game, choices, rng)
        }
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_event_choice(game, event, choices, rng),
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.select_event_choice(game, event, choices, rng)
        }
    }

    fn last_decision(&self) -> Option<DecisionInfo> {
        match self {
            Self::Handwritten(t) => t.last_decision(),
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.last_decision()
        }
    }

    fn last_breakdown(&self) -> Option<String> {
        match self {
            Self::Handwritten(t) => t.last_breakdown(),
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.last_breakdown()
        }
    }
}

/// 猴子训练师
pub struct RandomTrainer;

impl<G: Game> Trainer<G> for RandomTrainer {
    fn select_action(&self, game: &G, actions: &[<G as Game>::Action], rng: &mut StdRng) -> Result<usize> {
        let mut random_index: Vec<_> = (0..actions.len()).collect();
        let mut ret = None;
        random_index.shuffle(rng);
        for i in &random_index {
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
            } else if matches!(actions[*i].as_base_action(), Some(BaseAction::Train(_))) {
                ret = Some(*i);
                break;
            }
        }
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

fn any_ramen_action<A>(_action: &A) -> Option<&crate::game::ramen::RamenAction> {
    None
}

pub struct ManualTrainer {
    pub mock_inputs: Rc<RefCell<VecDeque<String>>>,
    pub fallback: FallbackMode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackMode {
    Interactive,
    PickFirst
}

impl Default for ManualTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualTrainer {
    pub fn new() -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(VecDeque::new())),
            fallback: FallbackMode::Interactive
        }
    }
    pub fn with_mock_inputs(inputs: Vec<String>) -> Self {
        Self {
            mock_inputs: Rc::new(RefCell::new(inputs.into_iter().collect())),
            fallback: FallbackMode::PickFirst
        }
    }
    fn pop_mock_input(&self) -> Option<String> {
        self.mock_inputs.borrow_mut().pop_front()
    }
    fn fallback_pick_first(&self, len: usize, item_desc: &str) -> Result<usize> {
        if len == 0 {
            return Err(anyhow::anyhow!("{item_desc} 候选为空"));
        }
        Ok(0)
    }
}

impl<G: Game> Trainer<G> for ManualTrainer {
    fn select_action(&self, _game: &G, actions: &[<G as Game>::Action], _rng: &mut StdRng) -> Result<usize> {
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
                "ManualTrainer::Interactive 需要 cli feature；请改用 with_mock_inputs"
            ))
        }
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let explain: Vec<String> = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect();
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
                "ManualTrainer::Interactive 需要 cli feature；请改用 with_mock_inputs"
            ))
        }
    }
}
