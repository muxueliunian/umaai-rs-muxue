//! 拉面杯手写策略
//!
//! 包含地区选择策略和超级拉面选择策略。
//! 地区选择初期使用固定顺序策略，超级拉面初期固定为选项二。

use anyhow::Result;

use super::rules::{get_region_range, get_super_ramen_clone_train_options};

/// 地区选择策略：固定顺序
///
/// 每年从可选地区中按固定顺序选择前 3 个。
/// - 第 1 年（year_idx=0）：选择 [0, 1, 2]（札幌、函馆、新潟）
/// - 第 2 年（year_idx=1）：选择 [5, 6, 7]（中山、中京、京都）
/// - 第 3 年（year_idx=2）：选择 [10, 11, 12]（札幌、函馆、新潟）
pub fn fixed_region_selection(year_idx: usize) -> Result<[usize; 3]> {
    let range = get_region_range(year_idx)?;
    if range.len() < 3 {
        anyhow::bail!("可选地区不足 3 个: year_idx={year_idx}, range={range:?}");
    }
    Ok([range[0], range[1], range[2]])
}

/// 超级拉面选择策略：固定选项二
///
/// 初期固定选择 `training_limit_options` 的第二个选项（索引 1）。
/// 返回选项对应的训练位置列表。
pub fn fixed_super_ramen_selection() -> Result<Vec<i32>> {
    let options = get_super_ramen_clone_train_options()?;
    if options.len() < 2 {
        anyhow::bail!("超级拉面选项不足 2 个");
    }
    Ok(options[1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_test_logger}};

    #[test]
    fn test_fixed_region_selection() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 第 1 年：选择 [0, 1, 2]
        let sel = fixed_region_selection(0)?;
        println!("第1年固定选择: {sel:?}");
        assert_eq!(sel, [0, 1, 2]);

        // 第 2 年：选择 [5, 6, 7]
        let sel = fixed_region_selection(1)?;
        println!("第2年固定选择: {sel:?}");
        assert_eq!(sel, [5, 6, 7]);

        // 第 3 年：选择 [10, 11, 12]
        let sel = fixed_region_selection(2)?;
        println!("第3年固定选择: {sel:?}");
        assert_eq!(sel, [10, 11, 12]);

        // 无效 year_idx
        assert!(fixed_region_selection(3).is_err());
        println!("无效 year_idx 验证通过");

        Ok(())
    }

    #[test]
    fn test_fixed_super_ramen_selection() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let sel = fixed_super_ramen_selection()?;
        println!("超级拉面固定选择(选项二): {sel:?}");
        // 选项2: 速/耐/力/智 [0,1,2,4]
        assert_eq!(sel, vec![0, 1, 2, 4]);

        Ok(())
    }
}
