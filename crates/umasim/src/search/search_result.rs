//! 搜索结果聚合
//!
//! 收集单个动作的多次模拟结果，计算统计量。

use super::ValueOutput;

/// 单个动作的搜索结果聚合器
///
/// 使用在线算法累积均值和方差，避免存储所有样本。
///
/// # 方差计算公式
///
/// 使用 Welford 在线算法的简化版本：
/// - `mean = score_sum / n`
/// - `variance = (score_sq_sum / n) - mean^2`
/// - `stdev = sqrt(variance)`
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 是否为合法动作
    pub is_legal: bool,
    /// 已搜索次数
    pub num: usize,
    /// 分数累加和
    score_sum: f64,
    /// 分数平方累加和（用于计算方差）
    score_sq_sum: f64,
    /// 最后一次计算的结果（缓存）
    last_calculate: ValueOutput,
    /// 缓存是否有效
    cache_valid: bool,
}

impl Default for SearchResult {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchResult {
    /// 创建新的搜索结果
    pub fn new() -> Self {
        Self {
            is_legal: false,
            num: 0,
            score_sum: 0.0,
            score_sq_sum: 0.0,
            last_calculate: ValueOutput::default(),
            cache_valid: false,
        }
    }

    /// 创建合法动作的搜索结果
    pub fn new_legal() -> Self {
        let mut r = Self::new();
        r.is_legal = true;
        r
    }

    /// 清空搜索结果
    pub fn clear(&mut self) {
        self.num = 0;
        self.score_sum = 0.0;
        self.score_sq_sum = 0.0;
        self.cache_valid = false;
    }

    /// 添加一个模拟结果
    pub fn add_result(&mut self, value: ValueOutput) {
        self.num += 1;
        self.score_sum += value.score_mean;
        self.score_sq_sum += value.score_mean * value.score_mean;
        self.cache_valid = false;
    }

    /// 批量添加模拟结果
    pub fn add_results(&mut self, values: &[ValueOutput]) {
        for v in values {
            self.add_result(*v);
        }
    }

    /// 获取均值
    pub fn mean(&self) -> f64 {
        if self.num == 0 { 0.0 } else { self.score_sum / self.num as f64 }
    }

    /// 获取标准差
    pub fn stdev(&self) -> f64 {
        if self.num < 2 {
            0.0
        } else {
            let mean = self.mean();
            let variance = (self.score_sq_sum / self.num as f64) - mean * mean;
            variance.max(0.0).sqrt()
        }
    }

    /// 获取加权后的平均分数
    pub fn get_weighted_mean_score(&mut self, radical_factor: f64) -> ValueOutput {
        if self.num == 0 {
            return ValueOutput::ILLEGAL;
        }

        if !self.cache_valid {
            let mean = self.mean();
            let stdev = self.stdev();
            self.last_calculate = ValueOutput {
                score_mean: mean,
                score_stdev: stdev,
                value: mean + radical_factor * stdev,
            };
            self.cache_valid = true;
        } else {
            self.last_calculate.value =
                self.last_calculate.score_mean + radical_factor * self.last_calculate.score_stdev;
        }

        self.last_calculate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let r = SearchResult::new();
        assert!(!r.is_legal);
        assert_eq!(r.num, 0);
    }

    #[test]
    fn test_new_legal() {
        let r = SearchResult::new_legal();
        assert!(r.is_legal);
        assert_eq!(r.num, 0);
    }

    #[test]
    fn test_empty_result() {
        let mut r = SearchResult::new_legal();
        assert!(r.get_weighted_mean_score(0.0).is_illegal());
    }

    #[test]
    fn test_single_result() {
        let mut r = SearchResult::new_legal();
        r.add_result(ValueOutput::new(1000.0, 0.0));
        assert_eq!(r.num, 1);
        assert_eq!(r.mean(), 1000.0);
        assert_eq!(r.stdev(), 0.0);
    }

    #[test]
    fn test_multiple_results() {
        let mut r = SearchResult::new_legal();
        r.add_result(ValueOutput::new(900.0, 0.0));
        r.add_result(ValueOutput::new(1100.0, 0.0));
        assert_eq!(r.num, 2);
        assert_eq!(r.mean(), 1000.0);
        assert!((r.stdev() - 100.0).abs() < 1.0);
    }
}

