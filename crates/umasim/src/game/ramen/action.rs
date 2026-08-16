//! 拉面杯动作定义
//!
//! RamenAction 是拉面杯的基本动作单位，采用组合动作模型：
//! "吃面/不吃面 + 基础操作" 统一表达为一个动作。

use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Operation;
use crate::gamedata::{GAMECONSTANTS, ramen::RAMENDATA};
use crate::global;

/// 拉面杯组合动作
///
/// 每个动作由两部分组成：
/// - `ramen`: 是否吃面以及吃哪种面（Option<usize>，Some 为 ramen_region_effect 数组下标）
/// - `operation`: 基础操作（训练/比赛/休息/外出/治病）
///
/// 决策应视为完整组合，规则层生成所有合法组合，不提前过滤。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamenAction {
    /// 是否吃面以及吃哪种面
    /// - None: 不吃面
    /// - Some(idx): 吃 ramen_region_effect[idx] 对应的地区拉面
    pub ramen: Option<usize>,
    /// 基础操作
    pub operation: Operation
}

impl RamenAction {
    /// 创建不吃面 + 基础操作的组合动作
    pub fn no_ramen(operation: Operation) -> Self {
        Self {
            ramen: None,
            operation
        }
    }

    /// 创建吃面 + 基础操作的组合动作
    pub fn with_ramen(ramen_idx: usize, operation: Operation) -> Self {
        Self {
            ramen: Some(ramen_idx),
            operation
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
            None => String::new()
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
            Operation::Clinic => "治病".to_string()
        };
        write!(f, "{ramen_text}{op_text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_logger}};

    #[test]
    fn test_ramen_action_display() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_logger("test", "info")?;
        init_global()?;

        let a1 = RamenAction::no_ramen(Operation::Train(super::super::TrainingType::Speed));
        println!("动作1: {a1}");

        let a2 = RamenAction::with_ramen(0, Operation::Train(super::super::TrainingType::Wisdom));
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
}
