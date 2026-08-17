use anyhow::Result;
use inquire::Select;
use log::info;
use rand::{Rng, prelude::StdRng, seq::SliceRandom};

use crate::{
    game::{ActionEnum, BaseAction, Game, Trainer}, gamedata::{ActionValue, EventChoice}
};

// 导出手写逻辑训练员、数据收集训练员、神经网络训练员和 MCTS 训练员
//pub mod collector_trainer;
pub mod handwritten_trainer;
pub mod mcts_trainer;
//pub mod mean_filter_collector_trainer;
//pub mod neural_net_trainer;

//pub use collector_trainer::CollectorTrainer;
pub use handwritten_trainer::HandwrittenTrainer;
pub use mcts_trainer::MctsTrainer;
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
        // 没有基础动作候选时（拉面杯三阶段决策中阶段阶段动作全为 None）：
        // 优先选有"实质内容"的候选（RamenAction 专属：ramen 非 None 或 special_targets 含非零值），
        // 避免误选"占位"动作（如 SpecialSelect 阶段默认生成的 [0,0,0]）。
        if ret.is_none() {
            for i in &random_index {
                if let Some(ra) = any_ramen_action(&actions[*i]) {
                    if ra.ramen.is_some()
                        || ra.special_targets.is_some_and(|t| t.iter().any(|&x| x > 0))
                    {
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
        info!("当前选项: {:?}, 随机选择选项 {}", choices, ret + 1);
        Ok(ret)
    }
}

/// 若 `action` 是 `RamenAction` 则返回其引用（用于在不耦合泛型 `Action` 的前提下读取拉面杯特有字段）。
///
/// 拉面杯的三阶段决策中，阶段阶段动作（如 `RamenSelect`/`SpecialSelect`）的
/// `as_base_action()` 返回 `None`，且 RamenAction 字段（`ramen`/`special_targets`）承载决策。
/// RandomTrainer 在没有基础动作候选时，优先选这些字段"有内容"的动作，
/// 避免误选"占位候选"导致后续阶段库存不足。
fn any_ramen_action<A>(_action: &A) -> Option<&crate::game::ramen::RamenAction> {
    None
}

/// 手动训练师
pub struct ManualTrainer;

impl<G: Game> Trainer<G> for ManualTrainer {
    fn select_action(&self, _game: &G, actions: &[<G as Game>::Action], _rng: &mut StdRng) -> Result<usize> {
        let selected = Select::new("请选择:", actions.to_vec())
            .with_page_size(actions.len())
            .prompt()?;
        actions
            .iter()
            .position(|x| *x == selected)
            .ok_or_else(|| anyhow::anyhow!("未找到该动作: {selected}"))
    }

    fn select_choice(&self, _game: &G, choices: &[Vec<EventChoice>], _rng: &mut StdRng) -> Result<usize> {
        let explain = choices
            .iter()
            .map(|x| x.iter().map(|y| y.explain()).collect::<Vec<_>>().join(" | "))
            .collect::<Vec<_>>();
        let selected = Select::new("请选择:", explain.clone()).prompt()?;
        explain
            .iter()
            .position(|x| *x == selected)
            .ok_or_else(|| anyhow::anyhow!("未找到该选项: {selected}"))
    }
}
