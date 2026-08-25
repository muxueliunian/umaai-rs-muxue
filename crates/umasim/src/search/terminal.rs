//! 终局多维记录
//!
//! 搜索排序只用 [`SearchScore`](super::SearchScore) 的两个标量；本模块负责**不参与排序**
//! 的终局观测量，把「选速度，因为均分 56712」变成「因为最终智力高 300、第三年剧本 PT 多 187」。
//!
//! # 分层
//!
//! ```text
//! TerminalRecord  一次 rollout 的原始事实（Copy，栈上定长）
//!        │
//!        └─ TerminalStats  按候选累加的矩统计（均值/标准差）
//!                 │
//!                 ├─ 诊断出口：visit() 带名称与量纲遍历
//!                 └─ NN 出口：单独的版本化编码步骤（尚未实现）
//! ```
//!
//! # 两条硬约束
//!
//! 1. **不复用 [`ActionResult`](super::ActionResult)**：它每个实例分配 `vec![0; 100000]`
//!    直方图（约 400 KB），20 余维乘候选数会把热路径打穿；而 `weighted_mean` 对
//!    「距上限还差多少」这类量纲毫无意义。这里用无直方图的 [`MomentResult`]。
//!
//! 2. **非线性量必须在 rollout 内部算完**：逐维 `sum`/`sum_sq` 只能重建线性量。
//!    「PT 是否达到 1500」「五维中最差的那个缺口」都是对局面非线性的归约——
//!    `mean(pt) = 1520` 可以是「一半 1200 + 一半 1840」，达成率其实只有 50%。
//!    事后从各维均值重建会得到一块**会说谎的仪表**。

use std::fmt::Debug;

/// 不含直方图的标量矩统计
///
/// 与 [`ActionResult`](super::ActionResult) 的分工：后者伺候两条参与排序的评分口径
/// （需要直方图算 `weighted_mean`），本类型伺候纯观测维度，`Copy` 且零堆分配。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MomentResult {
    /// 样本数
    num: u32,
    /// 样本总和
    pub sum: f64,
    /// 样本平方和
    pub sum_sq: f64
}

impl MomentResult {
    /// 加入一个样本
    pub fn add(&mut self, value: f64) {
        self.num += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// 样本数
    pub fn count(&self) -> u32 {
        self.num
    }

    /// 算术平均值；无样本时返回 0
    pub fn mean(&self) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        self.sum / self.num as f64
    }

    /// 样本标准差（贝塞尔校正）；样本数 ≤ 1 时返回 0
    ///
    /// 浮点消去可能让方差算出极小负值，故取 `max(0.0)` 后再开方。
    pub fn stdev(&self) -> f64 {
        if self.num <= 1 {
            return 0.0;
        }
        let n = self.num as f64;
        let variance = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        variance.max(0.0).sqrt()
    }
}

/// 一个已与名称、标签、量纲绑定的统计维度引用
///
/// 消费方拿到的是「名字和数值已经配好对」的整体，不存在按下标去平行名表里
/// 取名字这一步——那正是维度增删后静默错位的来源。
#[derive(Debug, Clone, Copy)]
pub struct NamedMetricRef<'a> {
    /// 稳定的机器可读键（用于 CSV 表头、NN 编码器取字段）
    pub key: &'static str,
    /// 人类可读标签
    pub label: &'static str,
    /// 量纲（`score` / `status` / `pt` / `flag`）
    pub unit: &'static str,
    /// 对应统计值
    pub result: &'a MomentResult
}

/// 一个剧本的终局统计集合（按候选累加的结果）
pub trait TerminalStats: Debug + Clone + Default + Send + Sync + 'static {
    /// 逐项访问「名称与数值已绑定」的统计维度
    fn visit(&self, visitor: &mut dyn FnMut(NamedMetricRef<'_>));

    /// 维度个数
    fn dim_count(&self) -> usize {
        let mut n = 0;
        self.visit(&mut |_| n += 1);
        n
    }
}

/// 一次 rollout 产生的定长终局记录
///
/// `Copy` + 固定字段：热路径上不允许每次 rollout 做堆分配。
pub trait TerminalRecord: Debug + Clone + Copy + Send + Sync + 'static {
    /// 对应的逐候选累加器
    type Stats: TerminalStats;

    /// 把本次终局事实并入统计
    fn accumulate_into(&self, stats: &mut Self::Stats);
}

/// 空终局记录（温泉及一切尚未接入观测的剧本）
///
/// ZST：单态化后 [`RolloutOutcome`] 与原 `SearchScore` 同尺寸，累加是空操作。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoTerminal;

/// 空终局统计
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoTerminalStats;

impl TerminalRecord for NoTerminal {
    type Stats = NoTerminalStats;

    fn accumulate_into(&self, _stats: &mut Self::Stats) {}
}

impl TerminalStats for NoTerminalStats {
    fn visit(&self, _visitor: &mut dyn FnMut(NamedMetricRef<'_>)) {}
}

/// 一次 rollout 的主评分与终局记录
///
/// 终局局面只存在于 rollout 内部，跑完即弃，故多维事实必须在这里随评分一起
/// 返回——事后无法补取。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutOutcome<T> {
    /// 参与搜索排序的两条评分口径
    pub score: super::SearchScore,
    /// 不参与排序的终局观测记录
    pub terminal: T
}

/// 从单一字段清单同时生成终局记录、累加器与名称绑定
///
/// 字段名、键、标签、量纲写在同一处，因而不存在「平行名表与数值顺序错位」这类
/// 静默 bug：增删或重排维度只改这一份声明。
///
/// 注意它守不住的东西：`speed_score` 是否误取了力量的值，属于领域映射错误，
/// 只能靠 `from_game` 的单元测试兜底。
macro_rules! define_terminal_record {
    (
        $(#[$record_meta:meta])*
        $record:ident,
        $(#[$stats_meta:meta])*
        $stats:ident {
            $(
                $field:ident => { key: $key:literal, label: $label:literal, unit: $unit:literal }
            ),+ $(,)?
        }
    ) => {
        $(#[$record_meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $record {
            $(
                #[doc = $label]
                pub $field: f64
            ),+
        }

        $(#[$stats_meta])*
        #[derive(Debug, Clone, Copy, Default, PartialEq)]
        pub struct $stats {
            $(
                #[doc = $label]
                pub $field: $crate::search::MomentResult
            ),+
        }

        impl $crate::search::TerminalRecord for $record {
            type Stats = $stats;

            fn accumulate_into(&self, stats: &mut Self::Stats) {
                $(stats.$field.add(self.$field);)+
            }
        }

        impl $crate::search::TerminalStats for $stats {
            fn visit(&self, visitor: &mut dyn FnMut($crate::search::NamedMetricRef<'_>)) {
                $(
                    visitor($crate::search::NamedMetricRef {
                        key: $key,
                        label: $label,
                        unit: $unit,
                        result: &self.$field
                    });
                )+
            }
        }
    };
}

pub(crate) use define_terminal_record;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_result() {
        let mut m = MomentResult::default();
        println!("空统计: count={}, mean={}, stdev={}", m.count(), m.mean(), m.stdev());

        for v in [10.0, 20.0, 30.0, 40.0] {
            m.add(v);
        }
        // 均值 25，样本标准差 = sqrt(500/3) ≈ 12.9099
        println!(
            "四样本: count={}, sum={}, mean={:.4}, stdev={:.4}",
            m.count(),
            m.sum,
            m.mean(),
            m.stdev()
        );

        // 单样本方差无定义，约定返回 0 而不是 NaN
        let mut one = MomentResult::default();
        one.add(7.0);
        println!("单样本: mean={}, stdev={}", one.mean(), one.stdev());

        // 全等样本：浮点消去可能给出负方差，必须被 max(0.0) 吃掉
        let mut same = MomentResult::default();
        for _ in 0..64 {
            same.add(56712.0);
        }
        println!("全等样本: mean={}, stdev={} (期望 0，不得为 NaN)", same.mean(), same.stdev());
        println!("stdev 是否有限: {}", same.stdev().is_finite());
    }

    #[test]
    fn test_no_terminal_is_zst() {
        use std::mem::size_of;

        println!("NoTerminal 尺寸: {}", size_of::<NoTerminal>());
        println!("NoTerminalStats 尺寸: {}", size_of::<NoTerminalStats>());
        println!("SearchScore 尺寸: {}", size_of::<super::super::SearchScore>());
        println!(
            "RolloutOutcome<NoTerminal> 尺寸: {}（应与 SearchScore 相同）",
            size_of::<RolloutOutcome<NoTerminal>>()
        );

        let mut stats = NoTerminalStats;
        NoTerminal.accumulate_into(&mut stats);
        println!("空累加维度数: {}", stats.dim_count());
    }
}
