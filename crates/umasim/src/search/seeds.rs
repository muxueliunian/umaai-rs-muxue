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
        splitmix64(
            self.root
                .wrapping_add((rollout as u64).wrapping_add(1).wrapping_mul(GOLDEN_GAMMA))
        )
    }

    /// 由 rollout 种子再派生「该 rollout 在指定 `(回合, 阶段)` 上的随机流种子」
    ///
    /// 这是把「共享起始种子」升级为**真 CRN** 的关键：候选执行后消耗的随机数个数不同，
    /// 顺序流会就此错位；每进入一个阶段就按 `(rollout 种子, 回合, 阶段)` 重新播种，
    /// 则无论此前消耗多少，各候选在同一 `(回合, 阶段)` 上抽到的都是同一份随机性。
    ///
    /// 对齐的正是 CRN 的大头——下一回合的人头分配与事件抽签。
    ///
    /// 不吃候选索引，理由同 [`Self::seed_at`]。
    pub fn stage_seed(rollout_seed: u64, turn: i32, stage: u64) -> u64 {
        // 回合可能为负（未初始化局面），转 u64 前先做无符号重解释，避免符号扩展碰撞
        let turn_bits = (turn as i64) as u64;
        let mixed = rollout_seed
            .wrapping_add(turn_bits.wrapping_add(1).wrapping_mul(GOLDEN_GAMMA))
            .wrapping_add(stage.wrapping_add(1).wrapping_mul(MIX_A));
        splitmix64(mixed)
    }
}

/// 规则层内部随机流的种子（与搜索主 RNG 分频道）
///
/// 必须与 rollout 主种子**不同值**：两个 `StdRng::seed_from_u64(同一值)` 是同一条流，
/// 直接复用会让规则层与决策层抽到相同序列。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InternalSeed(u64);

impl InternalSeed {
    /// 由 rollout 种子派生规则层种子
    pub fn derive(rollout_seed: u64) -> Self {
        Self(splitmix64(rollout_seed ^ INTERNAL_STREAM_TAG))
    }

    /// 取出种子值
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// 规则层随机流的频道标记（任取的固定常数，只需与主流区分开）
const INTERNAL_STREAM_TAG: u64 = 0x5265_616C_5F52_4E47;

/// SplitMix64 finalizer
///
/// 纯终混合，**不含** `seed += gamma` 那一步——调用方自行决定如何构造输入。
/// `crate::sampler` 复用同一份实现，避免两处出现同名但行为不同的函数。
pub(crate) fn splitmix64(seed: u64) -> u64 {
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
    z ^ (z >> 31)
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

    /// 阶段派生必须确定，且 (回合, 阶段) 任一不同即给出不同种子
    #[test]
    fn test_stage_seed_distinct_per_turn_and_stage() {
        let base = RolloutSeeds::from_root(42).seed_at(3);
        let mut seen = Vec::new();
        for turn in 0..78i32 {
            for stage in 0..5u64 {
                seen.push(RolloutSeeds::stage_seed(base, turn, stage));
            }
        }
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        println!("78 回合 × 5 阶段 = {total} 个组合，产出 {} 个不同种子", seen.len());
        assert_eq!(seen.len(), total, "(回合, 阶段) 组合不得碰撞");

        // 确定性
        let a = RolloutSeeds::stage_seed(base, 12, 3);
        let b = RolloutSeeds::stage_seed(base, 12, 3);
        assert_eq!(a, b, "同参数必须给出同种子");
    }

    /// 不同 rollout 的同一 (回合, 阶段) 必须是不同随机流
    ///
    /// 否则所有 rollout 在该阶段会抽到完全一样的结果，方差直接塌掉。
    #[test]
    fn test_stage_seed_distinct_per_rollout() {
        let seeds = RolloutSeeds::from_root(42);
        let got: Vec<u64> = (0..512)
            .map(|j| RolloutSeeds::stage_seed(seeds.seed_at(j), 20, 3))
            .collect();
        let mut uniq = got.clone();
        uniq.sort_unstable();
        uniq.dedup();
        println!("512 个 rollout 在 (回合 20, 阶段 3) 上产出 {} 个不同种子", uniq.len());
        assert_eq!(uniq.len(), got.len(), "不同 rollout 在同一阶段不得共用随机流");
    }

    /// 规则层种子必须与 rollout 主种子不同值，且随 rollout 变化
    #[test]
    fn test_internal_seed_separated_from_main() {
        let seeds = RolloutSeeds::from_root(42);
        let mut collisions = 0;
        let mut got = Vec::new();
        for j in 0..256 {
            let main = seeds.seed_at(j);
            let internal = InternalSeed::derive(main);
            if internal.get() == main {
                collisions += 1;
            }
            got.push(internal.get());
        }
        println!("规则层种子与主种子相同的次数: {collisions}");
        assert_eq!(collisions, 0, "规则层种子不得与主种子同值（同值即同一条流）");

        got.sort_unstable();
        got.dedup();
        assert_eq!(got.len(), 256, "不同 rollout 的规则层种子不得碰撞");
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
