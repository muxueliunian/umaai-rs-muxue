//! 随机数受控重构原语（RNG Refactor Plan v2 §4）
//!
//! 提供无状态随机流 [`SplitmixRng`]、类型隔离的两条规则流
//! [`TurnFixedRng`] / [`StrategyRng`]、种子派生 [`derive_seed`] 与流标记 [`StreamTag`]，
//! 是拉面杯"回合固定随机与策略完全解耦"的底层实现：
//!
//! - 第 N 次随机 = `splitmix64(master + N·GAMMA)`，纯函数，不依赖此前消耗次数；
//! - 流值语义：Clone 即独立实例（counter 各自推进），MCTS 克隆局面的隔离基础；
//! - 两条规则流类型不同：任何随机点接错流直接编译不过（v2 强制措施）。

use rand::RngCore;

/// SplitMix64 黄金比例 gamma（标准常数，冻结不可改）
//
// 与 `seeded_rngs` 历史约定的 `0x9E37_79B9_7F4A_7C15` 同源（黄金比例二进制表示）。
// 派生常数一旦发布即不可变（可复现性契约，见 rng_refactor_plan_v2 §8）。
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer 混淆常数 A（冻结不可改）
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// SplitMix64 finalizer 混淆常数 B（冻结不可改）
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// 决策流派生标记（bench 局号派生用，冻结不可改）
pub const DECISION_TAG: u64 = 0xD3C1_51A8_0000_0001;
/// 策略流派生标记（回合策略流 master 用，冻结不可改）
pub const STRATEGY_TAG: u64 = 0x5374_7261_0000_0002;
/// 探测流派生标记（MCTS rollout CRN 骨架，冻结不可改）
pub const PROBE_TAG: u64 = 0x5072_6F62_0000_0003;
/// 事件流派生标记（回合开始事件链，冻结不可改）
pub const EVENT_TAG: u64 = 0x4576_656E_7400_0004;
/// 超级拉面分身局部流派生标记（冻结不可改）
pub const CLONE_SUPER_TAG: u64 = 0x436C_6F6E_0000_0005;
/// 地区拉面分身局部流派生标记（冻结不可改）
pub const CLONE_REGION_TAG: u64 = 0x436C_6F6E_0000_0006;

/// SplitMix64 输出混合（finalizer），全仓库唯一权威哈希实现
//
// 纯终混合，**不含** `seed += gamma` 那一步——调用方自行决定如何构造输入。
// 从 `search/seeds.rs` 的 `pub(crate)` 版提升而来（行为逐位不变，seeds.rs 测试守护）。
pub fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
    z ^ (z >> 31)
}

/// 种子派生：各分量异或后单次 splitmix64 哈希
//
// 与 rng_refactor_plan_v2 §4.2 公式逐字一致（如 `splitmix64(base ^ i ^ TAG)`）。
// 是 bench 局号、回合流 master 与 rollout 骨架的统一派生入口。
pub fn derive_seed(base: u64, parts: &[u64]) -> u64 {
    let mut x = base;
    for &p in parts {
        x ^= p;
    }
    splitmix64(x)
}

/// 从父流取 1 个随机字，派生一条一次性的局部随机流
///
/// 用于把「消耗次数随局面变化」的子算法与父流隔离：父流**恒定消耗 1 次**，
/// 子算法内部无论抽多少次都只推进局部流，因此改子算法不再位移父流后续的随机点。
///
/// 分身分配（[`CLONE_SUPER_TAG`] / [`CLONE_REGION_TAG`]）是首个用例：改分配算法
/// 不会再让同回合后续的训练成败、休息 / 外出结果整体错位。
///
/// `tag` 必须是冻结常量，且不同用途取不同值——同 tag 同父流值会得到同一条局部流。
///
/// # 已知耦合（与 v2 §4.2 的派生形式的差别）
///
/// v2 的流 master 派生形如 `splitmix64(rule_master ^ turn ^ TAG)`，**不依赖此前消耗次数**。
/// 本函数用的是父流在派生点的**当前取值**，因此局部流内容依赖父流此前消耗了几次：
/// 它只隔离下游（子算法内部怎么改都不影响父流后续），**隔离不了上游**。
///
/// 分身分配已改走 `RamenState::clone_stream` 的 `(rule_master, turn, tag)` 派生绕开
/// 这条限制（父流消耗降到 0）；本函数留给未注入 rule_master 的回退路径。
pub fn fork_local_stream(parent: &mut impl RngCore, tag: u64) -> SplitmixRng {
    SplitmixRng::new(derive_seed(parent.next_u64(), &[tag]))
}

/// 流标记：随机流类别（冻结不可改）
//
// - [`StreamTag::TurnFixed`]：回合固定流（人头分布/角标/hint）
// - [`StreamTag::Strategy`]：策略流（训练成败/分身/吃面落地/策略触发事件）
// - [`StreamTag::Event`]：事件流（回合开始事件链：unlock 判定/事件生成/事件应用）
// - [`StreamTag::Probe`]：探测流（MCTS rollout 公共随机数骨架，预留）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamTag {
    /// 回合固定流
    TurnFixed,
    /// 策略流
    Strategy,
    /// 事件流（回合开始事件链）
    Event,
    /// 探测流（MCTS 预留）
    Probe
}

impl StreamTag {
    /// 流标记数值（参与种子派生异或，冻结不可改）
    pub const fn tag(self) -> u64 {
        match self {
            Self::TurnFixed => 0,
            Self::Strategy => STRATEGY_TAG,
            Self::Event => EVENT_TAG,
            Self::Probe => PROBE_TAG
        }
    }
}

/// 无状态随机流
//
// 第 N 次随机 = `splitmix64(master + N·GAMMA)`，counter 从 0 起每次自增；
// 与"此前消耗多少次"无关——Clone 即获得独立实例（counter 各自推进）。
//
// # 用法
// - 回合开始：`SplitmixRng::new(master)`（counter 归零）
// - MCTS rollout：`rng.reset(master_k)` 注入本轮公共骨架
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitmixRng {
    /// 流主种子（本流随机序列的确定性来源）
    master: u64,
    /// 本回合内已消费次数（第 N 次随机 = N）
    counter: u64
}

impl SplitmixRng {
    /// 创建主种子为 `master`、计数为 0 的新流
    pub fn new(master: u64) -> Self {
        Self { master, counter: 0 }
    }

    /// 当前主种子（测试观测用）
    pub fn master(&self) -> u64 {
        self.master
    }

    /// 当前计数（= 下次随机的 N，测试观测用）
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// 重置为主种子 `master`、计数归零（回合切换 / rollout 注入）
    pub fn reset(&mut self, master: u64) {
        self.master = master;
        self.counter = 0;
    }
}

impl RngCore for SplitmixRng {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        let v = splitmix64(self.master.wrapping_add(self.counter.wrapping_mul(GAMMA)));
        self.counter = self.counter.wrapping_add(1);
        v
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

/// 回合固定流（v2 类型隔离：与 [`StrategyRng`] 接错直接编译不过）
//
// 随机点：人头分布、训练角标、hint 分配、回合事件生成（含回合开始事件的应用结果）。
// 每回合从 `(rule_master, turn)` 派生 master，counter 从 0 计数——与策略完全无关。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnFixedRng(SplitmixRng);

impl TurnFixedRng {
    /// 创建主种子为 `master` 的新回合固定流（counter 归零）
    pub fn new(master: u64) -> Self {
        Self(SplitmixRng::new(master))
    }

    /// 当前主种子（测试观测用）
    pub fn master(&self) -> u64 {
        self.0.master()
    }

    /// 当前计数（= 下次随机的 N，测试观测用）
    pub fn counter(&self) -> u64 {
        self.0.counter()
    }

    /// 重置为主种子 `master`、计数归零（回合切换）
    pub fn reset(&mut self, master: u64) {
        self.0.reset(master);
    }
}

impl RngCore for TurnFixedRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
}

/// 策略流（v2 类型隔离：与 [`TurnFixedRng`] 接错直接编译不过）
//
// 随机点：训练成功率/大失败、分身分配、比赛结果、吃面效果落地、休息/外出结果、
// 策略触发事件的应用结果。从 `(rule_master, turn, STRATEGY_TAG)` 派生 master；
// 仅 apply 真实动作时消耗。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyRng(SplitmixRng);

impl StrategyRng {
    /// 创建主种子为 `master` 的新策略流（counter 归零）
    pub fn new(master: u64) -> Self {
        Self(SplitmixRng::new(master))
    }

    /// 当前主种子（测试观测用）
    pub fn master(&self) -> u64 {
        self.0.master()
    }

    /// 当前计数（= 下次随机的 N，测试观测用）
    pub fn counter(&self) -> u64 {
        self.0.counter()
    }

    /// 重置为主种子 `master`、计数归零（回合切换）
    pub fn reset(&mut self, master: u64) {
        self.0.reset(master);
    }
}

impl RngCore for StrategyRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
}

/// 事件流（v2 §4.3 三流：第三轴）
//
// 随机点：回合开始事件链（友人解锁判定、事件生成、事件应用）。
// 事件的**是否触发**依赖事件历史（max_time / 卡事件 8001-8003 连续触发，策略状态），
// 但其随机本身与策略/局面无关——独立成轴后，事件历史的差异只影响事件流自身，
// 不污染局面流（角标/分布/hint）与策略流（训练/分身/比赛）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventRng(SplitmixRng);

impl EventRng {
    /// 创建主种子为 `master` 的新事件流（counter 归零）
    pub fn new(master: u64) -> Self {
        Self(SplitmixRng::new(master))
    }

    /// 当前主种子（测试观测用）
    pub fn master(&self) -> u64 {
        self.0.master()
    }

    /// 当前计数（= 下次随机的 N，测试观测用）
    pub fn counter(&self) -> u64 {
        self.0.counter()
    }

    /// 重置为主种子 `master`、计数归零（回合切换）
    pub fn reset(&mut self, master: u64) {
        self.0.reset(master);
    }
}

impl RngCore for EventRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /// 层1：同一 (master, n) 两次计算同值（确定性）
    #[test]
    fn test_deterministic() {
        let mut a = SplitmixRng::new(42);
        let mut b = SplitmixRng::new(42);
        let mut all_same = true;
        for n in 0..8 {
            let va = a.next_u64();
            let vb = b.next_u64();
            all_same &= va == vb;
            println!("master=42 n={n}: {va:#018x} == {vb:#018x}");
        }
        println!("8 次取值全部相同: {all_same}");
    }

    /// 层1：不同 master 序列不同（无相关性）
    #[test]
    fn test_masters_differ() {
        let mut a = SplitmixRng::new(1);
        let mut b = SplitmixRng::new(2);
        let sa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let same = sa.iter().zip(&sb).filter(|(x, y)| x == y).count();
        println!("master=1 前8值: {sa:?}");
        println!("master=2 前8值: {sb:?}");
        println!("同位相同个数: {same}/8");
    }

    /// 层1：消费 k 次后从第 k+1 次继续（无状态）
    #[test]
    fn test_counter_continues() {
        let mut rng = SplitmixRng::new(7);
        let first = rng.next_u64();
        let second = rng.next_u64();
        let mut skip = SplitmixRng::new(7);
        let _ = skip.next_u64();
        let after_skip = skip.next_u64();
        println!("n=0: {first:#018x}");
        println!("n=1: {second:#018x}");
        println!(
            "消费1次后第2次: {after_skip:#018x} == 直接第2次: {}",
            second == after_skip
        );
    }

    /// 层1：Clone 值语义——克隆后各自推进互不影响（MCTS 隔离原子验证）
    #[test]
    fn test_clone_independent() {
        let mut a = SplitmixRng::new(99);
        let mut b = a;
        let (va1, vb1) = (a.next_u64(), b.next_u64());
        let (va2, vb2) = (a.next_u64(), b.next_u64());
        println!("a: {va1:#018x} {va2:#018x}");
        println!("b: {vb1:#018x} {vb2:#018x}");
        println!("同位相等: {} {}", va1 == vb1, va2 == vb2);
        println!("a 计数={} b 计数={}", a.counter(), b.counter());
    }

    /// 层1：不同 StreamTag 派生序列不重叠（流间隔离）
    #[test]
    fn test_stream_tags_isolated() {
        let base = 12345u64;
        let mut fixed = SplitmixRng::new(derive_seed(base, &[7]));
        let mut strategy = SplitmixRng::new(derive_seed(base, &[7, StreamTag::Strategy.tag()]));
        let sf: Vec<u64> = (0..8).map(|_| fixed.next_u64()).collect();
        let ss: Vec<u64> = (0..8).map(|_| strategy.next_u64()).collect();
        let overlap = sf.iter().filter(|x| ss.contains(x)).count();
        println!("固定流: {sf:?}");
        println!("策略流: {ss:?}");
        println!("序列重叠数: {overlap}");
    }

    /// 层1（v2 新增）：加法派生防撞车——stream(master,n) != stream(master^k, n^k)
    #[test]
    fn test_additive_no_xor_collision() {
        // XOR 派生下 stream(A, n) == stream(A^k, n^k)；加法派生应无此性质
        let mut a = SplitmixRng::new(0);
        let mut b = SplitmixRng::new(1);
        let _ = b.next_u64(); // b 消费 1 次 → 若 XOR 派生，a 第 0 次 == b 第 1 次
        let va0 = a.next_u64();
        let vb1 = b.next_u64();
        println!("master=0 n=0: {va0:#018x}");
        println!("master=1 n=1: {vb1:#018x}");
        println!("撞车（应 false）: {}", va0 == vb1);
    }

    /// 层1：TurnFixedRng / StrategyRng 类型隔离流可正常消费
    #[test]
    fn test_typed_streams_work() {
        let mut fixed = TurnFixedRng::new(42);
        let mut strategy = StrategyRng::new(42);
        let f0 = fixed.next_u64();
        let s0 = strategy.next_u64();
        println!(
            "回合固定流[0]: {f0:#018x} (master={:#x} counter={})",
            fixed.master(),
            fixed.counter() - 1
        );
        println!(
            "策略流[0]: {s0:#018x} (master={:#x} counter={})",
            strategy.master(),
            strategy.counter() - 1
        );
        fixed.reset(7);
        let f1 = fixed.next_u64();
        println!("固定流 reset(7) 后[0]: {f1:#018x} (counter={})", fixed.counter() - 1);
    }
    /// 局部流派生：父流恰好推进 1 次，不同 tag 得到互不相同的序列
    ///
    /// 层 1 原语自测（plan v2 要求每个 rng 原语有单测）。拉面侧那两个测试测的是
    /// 「父流消耗 = 1」这个**调用方**性质，测不到局部流本身是否确定、是否按 tag 分离。
    #[test]
    fn test_fork_local_stream() {
        const TAG_A: u64 = CLONE_SUPER_TAG;
        const TAG_B: u64 = CLONE_REGION_TAG;

        // ① 父流恰好推进 1 次
        let mut parent = SplitmixRng::new(0x1234_5678);
        let before = parent.counter();
        let mut local = fork_local_stream(&mut parent, TAG_A);
        println!("父流 counter: {before} -> {}", parent.counter());
        assert_eq!(parent.counter() - before, 1);

        // ② 局部流独立计数，抽多少次都不回流父流
        for _ in 0..20 {
            let _ = local.next_u64();
        }
        println!("局部流抽 20 次后，父流 counter 仍为 {}", parent.counter());
        assert_eq!(parent.counter() - before, 1);
        assert_eq!(local.counter(), 20);

        // ③ 确定性：同一父流状态 + 同一 tag => 同一局部流
        let mut p1 = SplitmixRng::new(0xABCD);
        let mut p2 = SplitmixRng::new(0xABCD);
        let mut l1 = fork_local_stream(&mut p1, TAG_A);
        let mut l2 = fork_local_stream(&mut p2, TAG_A);
        let (a, b) = (l1.next_u64(), l2.next_u64());
        println!("确定性: {a:#018x} vs {b:#018x}");
        assert_eq!(a, b);

        // ④ tag 分离：同一父流取值配不同 tag => 不同序列
        let mut p3 = SplitmixRng::new(0xABCD);
        let mut l3 = fork_local_stream(&mut p3, TAG_B);
        let d = l3.next_u64();
        println!("tag 分离: TAG_A={a:#018x} TAG_B={d:#018x}");
        assert_ne!(a, d);

        // ⑤ 父流位置分离：父流 counter 不同 => 局部流不同
        //    这也是本原语的已知耦合：局部流依赖父流在派生点的取值，
        //    因此它只隔离「下游」，隔离不了「上游此前消耗了几次」。
        let mut p4 = SplitmixRng::new(0xABCD);
        let _ = p4.next_u64();
        let mut l4 = fork_local_stream(&mut p4, TAG_A);
        let e = l4.next_u64();
        println!("父流位置分离: counter=0 时 {a:#018x}，counter=1 时 {e:#018x}");
        assert_ne!(a, e);
    }

}
