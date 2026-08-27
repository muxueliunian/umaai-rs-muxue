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
//! # CRN 对齐的承担者（RNG Refactor Plan v2 §5.2）
//!
//! - **拉面（规则层接管）**：rollout 分支经 [`crate::search::FlatSearchGame::fork_for_rollout`]
//!   注入 `rule_master = seed_at(j)`，规则层每回合按 `(rule_master, turn)` 派生
//!   回合固定流/策略流——同一轮内各候选面对逐位一致的随机未来，无需按阶段重播种。
//! - **温泉（外挂 CRN 保留）**：onsen 规则层未改造，仍由 [`RolloutSeeds::stage_seed`]
//!   按 `(rollout 种子, 回合, 阶段)` 重播种对齐（`crn_stage_reseed` 开关）。
//!
//! `InternalSeed` 已随拉面接驳退役（规则层直接注入 rollout 种子，无需分频道派生）。

use crate::rng::splitmix64;

/// SplitMix64 的 gamma 增量常数（黄金比例）
///
/// 与 [`crate::bench::seeded_rngs`]、[`crate::rng::GAMMA`] 同源，保持全仓库
/// 种子派生常数一致。具体取值无关紧要，关键是固定不变——可复现性依赖它在
/// 代码演进中保持稳定。
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer 混淆常数 A（仅 [`RolloutSeeds::stage_seed`] 使用，冻结不可改）
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;

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
    pub fn from_rng(rng: &mut impl rand::Rng) -> Self {
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
    /// **仅温泉（外挂 CRN）使用**：onsen 规则层未改造，靠每进入一个阶段按
    /// `(rollout 种子, 回合, 阶段)` 重新播种，使各候选在同一 `(回合, 阶段)` 上
    /// 抽到同一份随机性。拉面规则层已由无状态流接管，不再调用本方法。
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

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

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
