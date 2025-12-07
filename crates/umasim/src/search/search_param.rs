//! 蒙特卡洛搜索参数配置
//!
//! 包含所有可调节的搜索超参数，支持默认值和自定义配置。

use serde::{Deserialize, Serialize};

/// 蒙特卡洛搜索参数
///
/// # 参数说明
///
/// - `search_single_max`: 单个动作最大搜索次数，防止对某个动作过度搜索
/// - `search_total_max`: 总搜索次数上限，控制计算预算
/// - `search_group_size`: 每批分配的搜索次数
/// - `search_cpuct`: UCT 公式中的探索常数，越大越倾向探索
/// - `max_depth`: 模拟深度限制，超过后使用评估函数估值
/// - `max_radical_factor`: 最大激进因子，用于调节风险偏好
/// - `thread_num`: 线程数，0 表示使用 CPU 核心数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParam {
    /// 单个动作最大搜索次数
    pub search_single_max: usize,
    /// 总搜索次数上限（0 表示无限制）
    pub search_total_max: usize,
    /// 批量分配大小
    pub search_group_size: usize,
    /// 探索常数 (UCT 的 c 值)
    pub search_cpuct: f64,
    /// 模拟深度限制
    pub max_depth: usize,
    /// 最大激进因子
    pub max_radical_factor: f64,
    /// 线程数（0 表示使用 CPU 核心数）
    pub thread_num: usize,
}

impl Default for SearchParam {
    fn default() -> Self {
        Self {
            search_single_max: 256,
            search_total_max: 2048,
            search_group_size: 32,
            search_cpuct: 1.5,
            max_depth: 78, // 温泉剧本最大回合数
            max_radical_factor: 1.0,
            thread_num: 0, // 自动检测
        }
    }
}

impl SearchParam {
    /// 快速搜索配置（用于测试）
    pub fn fast() -> Self {
        Self {
            search_single_max: 32,
            search_total_max: 256,
            search_group_size: 16,
            ..Default::default()
        }
    }

    /// 深度搜索配置（用于正式对局）
    pub fn deep() -> Self {
        Self {
            search_single_max: 512, 
            search_total_max: 8192,
            search_group_size: 64,
            ..Default::default()
        }
    }

    /// 验证参数合法性
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.search_single_max == 0 {
            anyhow::bail!("search_single_max 必须 > 0");
        }
        if self.search_group_size == 0 {
            anyhow::bail!("search_group_size 必须 > 0");
        }
        if self.search_group_size > self.search_single_max {
            anyhow::bail!("search_group_size 不能大于 search_single_max");
        }
        if self.search_cpuct <= 0.0 {
            anyhow::bail!("search_cpuct 必须 > 0");
        }
        if self.max_depth == 0 {
            anyhow::bail!("max_depth 必须 > 0");
        }
        Ok(())
    }

    /// 获取实际线程数
    pub fn actual_thread_num(&self) -> usize {
        if self.thread_num == 0 {
            rayon::current_num_threads()
        } else {
            self.thread_num
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_param() {
        let param = SearchParam::default();
        assert!(param.validate().is_ok());
    }

    #[test]
    fn test_fast_param() {
        let param = SearchParam::fast();
        assert!(param.validate().is_ok());
        assert!(param.search_single_max < SearchParam::default().search_single_max);
    }

    #[test]
    fn test_deep_param() {
        let param = SearchParam::deep();
        assert!(param.validate().is_ok());
        assert!(param.search_single_max > SearchParam::default().search_single_max);
    }

    #[test]
    fn test_invalid_param_single_max_zero() {
        let mut param = SearchParam::default();
        param.search_single_max = 0;
        assert!(param.validate().is_err());
    }

    #[test]
    fn test_invalid_param_group_size_zero() {
        let mut param = SearchParam::default();
        param.search_group_size = 0;
        assert!(param.validate().is_err());
    }

    #[test]
    fn test_invalid_param_group_size_too_large() {
        let mut param = SearchParam::default();
        param.search_group_size = param.search_single_max + 1;
        assert!(param.validate().is_err());
    }

    #[test]
    fn test_invalid_param_cpuct_zero() {
        let mut param = SearchParam::default();
        param.search_cpuct = 0.0;
        assert!(param.validate().is_err());
    }

    #[test]
    fn test_invalid_param_max_depth_zero() {
        let mut param = SearchParam::default();
        param.max_depth = 0;
        assert!(param.validate().is_err());
    }

    #[test]
    fn test_actual_thread_num() {
        let param = SearchParam::default();
        assert!(param.actual_thread_num() > 0);

        let mut param2 = SearchParam::default();
        param2.thread_num = 4;
        assert_eq!(param2.actual_thread_num(), 4);
    }
}

