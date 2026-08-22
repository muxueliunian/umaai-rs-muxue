//! 搜索 rollout 种子派生
//!
//! 一次搜索内的所有候选动作共享同一张 rollout 种子表，这是公共随机数
//! （CRN, Common Random Numbers）的载体：候选 i 的第 j 次 rollout 一律取
//! `seed_at(j)`，使候选之间的分数差异尽量只来自动作本身，而非随机流不同。
//!
//! # 为什么候选索引不能参与派生
//!
//! CRN 的方差削减来自配对样本的协方差项：
//! `Var(X_a - X_b) = Var(X_a) + Var(X_b) - 2 Cov(X_a, X_b)`。
//! 若种子按 `hash(root, 候选, rollout)` 派生，候选间协方差归零，
//! 比较方差退回独立抽样——那不是 CRN，只是可复现的独立抽样。
//! 故 [`RolloutSeeds::seed_at`] **只吃 rollout 序号**。
//!
//! # 尚未实现的部分
//!
//! 共享种子只保证各候选 rollout 的**起始流位置**相同。候选一经执行状态立刻分叉，
//! 后续随机数消耗长度不同，第 t+1 回合抽的未必是同一件事。真正的对齐需要
//! 按 `(turn, stage)` 重播种（计划 Phase 1.3），本模块只提供其种子来源。
//! 在那之前，**不应对外宣称已实现 CRN**。

use rand::{RngCore, rngs::StdRng};

/// SplitMix64 的 gamma 增量常数（黄金比例）
///
/// 与 [`crate::bench::seeded_rngs`] 同源，保持全仓库种子派生常数一致。
/// 具体取值无关紧要，关键是固定不变——可复现性依赖它在代码演进中保持稳定。
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer 的两个混淆常数
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// 见 [`MIX_A`]
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// 一次搜索内所有候选共享的 rollout 种子表
///
/// 按需计算而非预分配数组：UCB 分配下 rollout 序号的上界不确定
/// （`search_n` 会被 `search_group_size` 打超，例如 `search_n=1000` +
/// `group_size=256` 实际会到 1024），预分配容易算错长度。
#[derive(Debug, Clone, Copy)]
pub struct RolloutSeeds {
    /// 本次搜索的根种子
    root: u64
}

impl RolloutSeeds {
    /// 从搜索入口 RNG 抽取根种子
    ///
    /// 只抽一次，使外层（如 `MctsTrainer` 的整局种子）能罩住整次搜索。
    pub fn from_rng(rng: &mut StdRng) -> Self {
        Self { root: rng.next_u64() }
    }

    /// 用指定根种子构造（测试与回归基准用）
    pub fn from_root(root: u64) -> Self {
        Self { root }
    }

    /// 本次搜索的根种子
    pub fn root(&self) -> u64 {
        self.root
    }

    /// 第 `rollout` 次 rollout 的种子（SplitMix64 派生）
    ///
    /// **不吃候选索引**，理由见模块文档。同一 `rollout` 序号在所有候选上返回同一值。
    pub fn seed_at(&self, rollout: usize) -> u64 {
        let mut z = self
            .root
            .wrapping_add((rollout as u64).wrapping_add(1).wrapping_mul(GOLDEN_GAMMA));
        z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    /// 同一根种子下 `seed_at` 必须确定：可复现性的最底层保证
    #[test]
    fn test_seed_at_deterministic() {
        let a = RolloutSeeds::from_root(42);
        let b = RolloutSeeds::from_root(42);
        for j in [0usize, 1, 7, 255, 1023, 12287] {
            println!("rollout {j}: {:#018x}", a.seed_at(j));
            assert_eq!(a.seed_at(j), b.seed_at(j), "同根种子同序号必须一致");
        }
    }

    /// 不同 rollout 序号必须给出不同种子（否则所有 rollout 退化成同一局）
    #[test]
    fn test_seed_at_distinct_per_rollout() {
        let seeds = RolloutSeeds::from_root(42);
        let got: Vec<u64> = (0..1024).map(|j| seeds.seed_at(j)).collect();
        let mut uniq = got.clone();
        uniq.sort_unstable();
        uniq.dedup();
        println!("1024 个序号产出 {} 个不同种子", uniq.len());
        assert_eq!(uniq.len(), got.len(), "同一根种子下各 rollout 序号不得碰撞");
    }

    /// 不同根种子必须给出不同序列（否则换 seed 跑批等于没换）
    #[test]
    fn test_distinct_root_distinct_sequence() {
        let a = RolloutSeeds::from_root(42);
        let b = RolloutSeeds::from_root(43);
        let same = (0..256).filter(|&j| a.seed_at(j) == b.seed_at(j)).count();
        println!("root=42 与 root=43 在前 256 个序号上的碰撞数: {same}");
        assert_eq!(same, 0, "不同根种子不应产生相同序列");
    }

    /// `from_rng` 由入口 RNG 决定，故入口种子固定时根种子也固定
    #[test]
    fn test_from_rng_follows_entry_seed() {
        let mut rng1 = StdRng::seed_from_u64(7);
        let mut rng2 = StdRng::seed_from_u64(7);
        let s1 = RolloutSeeds::from_rng(&mut rng1);
        let s2 = RolloutSeeds::from_rng(&mut rng2);
        println!("root = {:#018x}", s1.root());
        assert_eq!(s1.root(), s2.root(), "入口种子相同则根种子相同");
    }
}
