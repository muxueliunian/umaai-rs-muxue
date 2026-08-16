//! 拉面杯动作定义
//!
//! RamenAction 是拉面杯的基本动作单位，采用分离决策模型：
//! - 阶段1：吃面决策（不吃面 / 吃面X / 吃面Y / 吃面Z）
//! - 阶段2：基础操作（训练/比赛/休息/外出/治病）
//!
//! 分离决策可以大幅减少动作空间，因为吃面后的分身是随机的，无法提前感知。

use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{Operation, TrainingType};
use crate::gamedata::{GAMECONSTANTS, ramen::RAMENDATA};
use crate::global;

/// 拉面杯动作
///
/// 包含吃面决策和基础操作两个部分。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamenAction {
    /// 吃面决策
    /// - None: 不吃面
    /// - Some(idx): 吃 ramen_region_effect[idx] 对应的地区拉面
    pub ramen: Option<usize>,
    /// 基础操作
    pub operation: Operation,
}

impl RamenAction {
    /// 创建不吃面 + 基础操作的动作
    pub fn no_ramen(operation: Operation) -> Self {
        Self {
            ramen: None,
            operation,
        }
    }

    /// 创建吃面 + 基础操作的动作
    pub fn with_ramen(ramen_idx: usize, operation: Operation) -> Self {
        Self {
            ramen: Some(ramen_idx),
            operation,
        }
    }

    /// 是否包含吃面决策
    pub fn is_eating_ramen(&self) -> bool {
        self.ramen.is_some()
    }

    /// 获取基础操作
    pub fn base_operation(&self) -> Operation {
        self.operation
    }
}

impl Display for RamenAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ramen_text = match self.ramen {
            Some(idx) => {
                let name = RAMENDATA
                    .get()
                    .and_then(|d| d.ramen_region_effect.get(idx))
                    .map(|r| r.name.as_str())
                    .unwrap_or("???");
                format!("吃面/{name} + ")
            }
            None => String::new(),
        };
        let op_text = match self.operation {
            Operation::Train(train) => {
                let names = &global!(GAMECONSTANTS).train_names;
                format!("{}训练", names[train as usize])
            }
            Operation::Race => "比赛".to_string(),
            Operation::Rest => "休息".to_string(),
            Operation::NormalOuting => "普通出行".to_string(),
            Operation::FriendOuting => "友人出行".to_string(),
            Operation::Clinic => "治病".to_string(),
        };
        write!(f, "{ramen_text}{op_text}")
    }
}

/// 列出吃面选择（阶段1）
///
/// 返回所有可用的吃面选择：
/// - 不吃面
/// - 吃面X（如果诀窍足够）
/// - 吃面Y
/// - 吃面Z
///
/// # 参数
/// - `available_ramens`: 当前可以吃的面（诀窍足够）
pub fn list_ramen_choices(available_ramens: &[usize]) -> Vec<Option<usize>> {
    let mut choices = vec![None]; // 不吃面
    for &idx in available_ramens {
        choices.push(Some(idx));
    }
    choices
}

/// 列出所有基础操作（阶段2）
///
/// 返回所有可用的基础操作。
///
/// # 参数
/// - `can_friend_outing`: 是否可以选择友人出行
/// - `is_ill`: 是否生病
pub fn list_operations(can_friend_outing: bool, is_ill: bool) -> Vec<Operation> {
    let mut ops = vec![
        Operation::Train(TrainingType::Speed),
        Operation::Train(TrainingType::Stamina),
        Operation::Train(TrainingType::Power),
        Operation::Train(TrainingType::Guts),
        Operation::Train(TrainingType::Wisdom),
        Operation::Race,
        Operation::Rest,
        Operation::NormalOuting,
    ];
    if can_friend_outing {
        ops.push(Operation::FriendOuting);
    }
    if is_ill {
        ops.push(Operation::Clinic);
    }
    ops
}

/// 生成所有组合动作
///
/// 组合 = 吃面选择 × 基础操作。
///
/// # 参数
/// - `available_ramens`: 当前可以吃的面
/// - `can_friend_outing`: 是否可以选择友人出行
/// - `is_ill`: 是否生病
pub fn list_all_actions(
    available_ramens: &[usize],
    can_friend_outing: bool,
    is_ill: bool,
) -> Vec<RamenAction> {
    let ramen_choices = list_ramen_choices(available_ramens);
    let operations = list_operations(can_friend_outing, is_ill);

    let mut actions = Vec::new();
    for ramen in &ramen_choices {
        for &op in &operations {
            match ramen {
                None => actions.push(RamenAction::no_ramen(op)),
                Some(idx) => actions.push(RamenAction::with_ramen(*idx, op)),
            }
        }
    }
    actions
}

/// 获取当年可用的面（诀窍足够）
///
/// 返回可以吃的面的 ID 列表。
pub fn get_available_ramens(
    state: &super::RamenState,
    selected_regions: &[usize; 3],
) -> Vec<usize> {
    use super::rules::can_make_ramen;

    let mut available = Vec::new();
    for &region_id in selected_regions {
        if let Ok(recipe) = super::rules::get_recipe(region_id) {
            if can_make_ramen(state, recipe, &[0, 0, 0]) {
                available.push(region_id);
            }
        }
    }
    available
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::RamenState;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_logger}};

    #[test]
    fn test_ramen_action_display() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        let a1 = RamenAction::no_ramen(Operation::Train(TrainingType::Speed));
        println!("动作1: {a1}");

        let a2 = RamenAction::with_ramen(0, Operation::Train(TrainingType::Wisdom));
        println!("动作2: {a2}");

        let a3 = RamenAction::no_ramen(Operation::Race);
        println!("动作3: {a3}");

        let a4 = RamenAction::with_ramen(5, Operation::Rest);
        println!("动作4: {a4}");

        Ok(())
    }

    #[test]
    fn test_ramen_action_properties() {
        let a1 = RamenAction::no_ramen(Operation::Rest);
        assert!(!a1.is_eating_ramen());
        assert_eq!(a1.base_operation(), Operation::Rest);

        let a2 = RamenAction::with_ramen(5, Operation::Race);
        assert!(a2.is_eating_ramen());
        assert_eq!(a2.ramen, Some(5));
        assert_eq!(a2.base_operation(), Operation::Race);
    }

    #[test]
    fn test_list_ramen_choices() {
        // 无可用面
        let choices = list_ramen_choices(&[]);
        println!("无可用面: {choices:?}");
        assert_eq!(choices, vec![None]);

        // 有3种可用面
        let choices = list_ramen_choices(&[0, 1, 2]);
        println!("有3种可用面: {choices:?}");
        assert_eq!(choices, vec![None, Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_list_operations() {
        // 基础情况
        let ops = list_operations(false, false);
        println!("基础操作: {} 个", ops.len());
        assert_eq!(ops.len(), 8); // 5训练+比赛+休息+普通外出

        // 有友人出行和治病
        let ops = list_operations(true, true);
        println!("有友人+治病: {} 个", ops.len());
        assert_eq!(ops.len(), 10); // 5训练+比赛+休息+普通外出+友人出行+治病
    }

    #[test]
    fn test_list_all_actions() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        // 无可用面，无友人，无生病
        let actions = list_all_actions(&[], false, false);
        println!("无可用面: {} 个动作", actions.len());
        // 1吃面选择 * 8操作 = 8
        assert_eq!(actions.len(), 8);

        // 有3种可用面
        let actions = list_all_actions(&[0, 1, 2], false, false);
        println!("有3种可用面: {} 个动作", actions.len());
        // 4吃面选择 * 8操作 = 32
        assert_eq!(actions.len(), 32);

        // 有友人出行和治病，3种可用面
        let actions = list_all_actions(&[0, 1, 2], true, true);
        println!("有友人+治病+3种面: {} 个动作", actions.len());
        // 4吃面选择 * 10操作 = 40
        assert_eq!(actions.len(), 40);

        // 列出所有动作
        for (i, a) in actions.iter().enumerate() {
            println!("  {i:2}: {a}");
        }

        Ok(())
    }

    #[test]
    fn test_get_available_ramens() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        let selected_regions = [0, 1, 2];

        // 诀窍足够
        let mut state = RamenState::default();
        state.feeling_stock = [5, 5, 5];
        let available = get_available_ramens(&state, &selected_regions);
        println!("诀窍足够: 可用面={available:?}");
        assert_eq!(available.len(), 3);

        // 诀窍不足
        state.feeling_stock = [0, 0, 0];
        let available = get_available_ramens(&state, &selected_regions);
        println!("诀窍不足: 可用面={available:?}");
        assert_eq!(available.len(), 0);

        Ok(())
    }
}
