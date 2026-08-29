//! 拉面杯神经网络 **policy 头布局**（NN 管线 Phase 3 前置）
//!
//! 本模块只做一件事：把一个 [`RamenAction`] 映射到 policy 向量的**固定格位**，
//! 并保证这个映射**永不随局面改变**。
//!
//! # 为什么它必须单独成模块并被测试钉死
//!
//! 教师数据一旦落盘，10 万条样本的 policy 目标就是按这里的格位填的。
//! 格位含义改一个字，全部数据作废、搜索要重跑十几个机时。
//! 因此这里的每个常量都不是「实现细节」，而是**数据格式的一部分**。
//!
//! 与之对照，**不能**用 `list_*_actions()` 的返回顺序当格位：那些列表的长度与顺序
//! 随库存、夏合宿、疾病而变，同一个下标在两条样本里会指向不同动作。
//!
//! # 布局
//!
//! | 区间 | 宽度 | 含义 |
//! |---|---|---|
//! | `[0, 1)` | 1 | 不吃面 |
//! | `[1, 201)` | 200 | 吃面：20 个地区 × 10 种万能风味用法 |
//! | `[201, 211)` | 10 | 本回合基础操作（5 训练 + 比赛/休息/外出/友人外出/就医） |
//! | `[211, 214)` | 3 | 超级拉面训练范围三选一 |
//! | `[214, 234)` | 20 | 地区选择：每个地区一个分，取 top-3 |
//!
//! # 三个设计决定（2026-08-28 用户拍板，不可逆）
//!
//! 1. **吃面按地区 ID 编，不按槽位编。**
//!    `selected_regions` 虽然总是升序（见 `rules::get_region_combinations`），
//!    但「第 1 碗」在不同局里是不同的面，槽位编码学不到「15 号中山-速力智好不好」
//!    这种跨局稳定的知识。地区 ID 按年份天然分区
//!    （`rules::REGION_RANGES` = `[(0,4), (5,9), (10,19)]`），
//!    所以 20 维在任一决策点上都只有 5 或 10 个合法，掩码后并不稀疏。
//!
//! 2. **吃面与万能风味用法合成一个联合格，不拆成两个边缘头。**
//!    搜索本来就把两者合并成单个候选一起评（见 `RamenMctsTrainer::use_combined_ramen_select`），
//!    拆开会丢掉「这碗面只在某种用法下才值」的相关性。
//!
//! 3. **地区选择纳入第一代。**
//!    手写策略的地区逻辑是启发式且疑似有误，正是搜索最可能赢的地方。
//!    代价见 [`REGION_SELECT_BASE`] 的说明。
//!
//! # 万能风味的格位为何恒为 10 个
//!
//! `rules::list_special_targets_for` 里 `total_cap = min(2, special_feeling)`，
//! 因此 `sum(t) <= 2` **恒成立**，与配方、库存、吃哪碗面都无关。
//! 于是「和不超过 2 的三元组」这个集合是固定的 10 个，见 [`TRIPLES`]。
//!
//! 注意其中只有一部分是真正的决策：`min_needed`（配方缺口）是**强制**补的，
//! 自由度只在富余部分——即「现在花万能风味、把库存诀窍省给后面的面」。

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use super::{Operation, RamenStage, TrainingType, action::RamenAction};

// ========== 布局常量 ==========

/// 地区总数（`ramen_region_effect` 的条数）
///
/// 按年份分区：第 1 年 0-4、第 2 年 5-9、第 3 年 10-19。
pub const REGION_NUM: usize = 20;

/// 万能风味用法的种类数（和不超过 2 的三元组个数）
pub const TRIPLE_NUM: usize = 10;

/// 基础操作的种类数
pub const TRAIN_OP_NUM: usize = 10;

/// 超级拉面训练范围选项数（`finals_effect.training_limit_options` 长度）
pub const SUPER_NUM: usize = 3;

/// 「不吃面」的格位
pub const EAT_NONE: usize = 0;

/// 吃面段起点：`EAT_BASE + region_id * TRIPLE_NUM + triple_id`
pub const EAT_BASE: usize = EAT_NONE + 1;

/// 基础操作段起点
pub const TRAIN_BASE: usize = EAT_BASE + REGION_NUM * TRIPLE_NUM;

/// 超级拉面段起点
pub const SUPER_BASE: usize = TRAIN_BASE + TRAIN_OP_NUM;

/// 地区选择段起点
///
/// 每个地区一个分，动作是「取 top-3」而不是「选一个组合」。
///
/// 不用 C(10,3)=120 的组合头，理由有两条：组合格极稀疏（一局只出现 3 次），
/// 且组合下标学不到「地区本身好不好」这个可跨年迁移的量。
///
/// ⚠ 采数据时第 3 年必须跑 `ramen_region_strategy = all`（120 个候选组合）
/// 而不是 `fixed`，否则该决策点只有单候选、搜索给不出分布。
/// 一条第 3 年 `RegionSelect` 样本的搜索成本约为普通样本的 12 倍。
pub const REGION_SELECT_BASE: usize = SUPER_BASE + SUPER_NUM;

/// policy 向量总维度
pub const POLICY_DIM: usize = REGION_SELECT_BASE + REGION_NUM;

/// 万能风味用法的**规范全序**：和不超过 2 的全部三元组，字典序升序
///
/// 顺序与 `rules::list_special_targets_for` 的三层嵌套循环（`t_a` 外、`t_c` 内）一致。
/// **此表是数据格式的一部分，任何情况下都不得重排。**
pub const TRIPLES: [[i32; 3]; TRIPLE_NUM] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, 0],
    [0, 1, 1],
    [0, 2, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [2, 0, 0]
];

// ========== 动作 → 格位 ==========

/// 一个动作占用的 policy 格位
///
/// `RegionSelect` 一次选 3 个地区，故不是单格；其余阶段都是单格。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicySlots {
    /// 单格（吃面 / 基础操作 / 超级拉面）
    One(usize),
    /// 三格（地区选择，取 top-3）
    Three([usize; 3])
}

impl PolicySlots {
    /// 展平成下标切片视图，便于统一遍历
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::One(i) => std::slice::from_ref(i),
            Self::Three(a) => a.as_slice()
        }
    }
}

/// 万能风味三元组 → 格位偏移
///
/// # 错误
///
/// 三元组含负数、或和大于 2 时报错——这两种都说明调用方拿到的不是
/// `list_special_targets_for` 的产物，静默接受会把错误标签写进数据集。
pub fn triple_id(t: [i32; 3]) -> Result<usize> {
    ensure!(t.iter().all(|&x| x >= 0), "万能风味三元组不得含负数: {t:?}");
    let sum: i32 = t.iter().sum();
    ensure!(sum <= 2, "万能风味三元组之和恒 <= 2，实得 {sum}: {t:?}");
    TRIPLES
        .iter()
        .position(|x| *x == t)
        .ok_or_else(|| anyhow::anyhow!("三元组不在规范表内: {t:?}"))
}

/// 格位偏移 → 万能风味三元组（[`triple_id`] 的逆）
///
/// # 错误
///
/// 下标越界时报错。
pub fn triple_of(id: usize) -> Result<[i32; 3]> {
    TRIPLES
        .get(id)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("triple_id 越界: {id} >= {TRIPLE_NUM}"))
}

/// 吃面动作 → 格位
///
/// `region` 为 `None` 表示不吃面。此时 `targets` 允许两种形态，二者等价：
///
/// - `None` —— 三阶段路径下 `RamenSelect` 阶段的「不吃面」；
/// - `Some([0, 0, 0])` —— 合并决策路径下的「不吃面」。
///   这是 [`RamenAction::combined_select`] 明文约定的形态
///   （`ramen_idx = None` 时强制 `targets = [0,0,0]`，保留 `ramen = None`
///   是为了让 Trainer 与日志能识别）。教师数据走的正是合并路径，
///   若这里不接受它，每个 `RamenSelect` 决策点的「不吃面」候选都会落格失败。
///
/// `targets` 非零却不吃面才是真的不一致，仍然报错。
///
/// # 错误
///
/// 地区 ID 越界、或「不吃面却带了非零万能风味用法」时报错。
pub fn eat_index(region: Option<usize>, targets: Option<[i32; 3]>) -> Result<usize> {
    match (region, targets) {
        (None, None) => Ok(EAT_NONE),
        (None, Some([0, 0, 0])) => Ok(EAT_NONE),
        (None, Some(t)) => bail!("不吃面却带了非零万能风味用法: {t:?}"),
        (Some(rid), t) => {
            ensure!(rid < REGION_NUM, "地区 ID 越界: {rid} >= {REGION_NUM}");
            // 合并动作在 RamenSelect 阶段就定下用法；三阶段路径下 RamenSelect
            // 这一步尚未定，按规范三元组的零元 [0,0,0] 占位。
            let tid = triple_id(t.unwrap_or([0, 0, 0]))?;
            Ok(EAT_BASE + rid * TRIPLE_NUM + tid)
        }
    }
}

/// 基础操作 → 格位
///
/// # 错误
///
/// `StageOnly` / `RegionSelect` / `SuperRamenSelect` 不属于本段，传入报错——
/// 它们各有自己的段，走 [`slots_of`] 分派。
pub fn train_index(op: &Operation) -> Result<usize> {
    let off = match op {
        Operation::Train(TrainingType::Speed) => 0,
        Operation::Train(TrainingType::Stamina) => 1,
        Operation::Train(TrainingType::Power) => 2,
        Operation::Train(TrainingType::Guts) => 3,
        Operation::Train(TrainingType::Wisdom) => 4,
        Operation::Race => 5,
        Operation::Rest => 6,
        Operation::NormalOuting => 7,
        Operation::FriendOuting => 8,
        Operation::Clinic => 9,
        other => bail!("基础操作段不接受 {other:?}")
    };
    Ok(TRAIN_BASE + off)
}

/// 超级拉面选项下标 → 格位
///
/// # 错误
///
/// 下标越界时报错。
pub fn super_index(idx: usize) -> Result<usize> {
    ensure!(idx < SUPER_NUM, "超级拉面选项越界: {idx} >= {SUPER_NUM}");
    Ok(SUPER_BASE + idx)
}

/// 地区 ID → 地区选择段的格位
///
/// # 错误
///
/// 地区 ID 越界时报错。
pub fn region_index(rid: usize) -> Result<usize> {
    ensure!(rid < REGION_NUM, "地区 ID 越界: {rid} >= {REGION_NUM}");
    Ok(REGION_SELECT_BASE + rid)
}

/// 按阶段把一个动作分派到它的格位
///
/// 这是导出教师样本时的唯一入口——不要在别处自己拼偏移量。
///
/// # 错误
///
/// 阶段与动作不匹配（例如 `Train` 阶段拿到 `RegionSelect` 操作）时报错。
/// 这类不一致若被静默接受，会在数据集里留下无法追查的错标签。
pub fn slots_of(stage: RamenStage, act: &RamenAction) -> Result<PolicySlots> {
    match stage {
        RamenStage::RamenSelect | RamenStage::SpecialSelect => {
            Ok(PolicySlots::One(eat_index(act.ramen, act.special_targets)?))
        }
        RamenStage::Train => Ok(PolicySlots::One(train_index(&act.operation)?)),
        RamenStage::SuperRamenSelect => match act.operation {
            Operation::SuperRamenSelect(i) => Ok(PolicySlots::One(super_index(i)?)),
            ref other => bail!("SuperRamenSelect 阶段拿到非法操作: {other:?}")
        },
        RamenStage::RegionSelect => match act.operation {
            Operation::RegionSelect(ids) => {
                let mut out = [0usize; 3];
                for (dst, &rid) in out.iter_mut().zip(ids.iter()) {
                    *dst = region_index(rid)?;
                }
                ensure!(
                    out[0] != out[1] && out[1] != out[2] && out[0] != out[2],
                    "地区选择出现重复: {ids:?}"
                );
                Ok(PolicySlots::Three(out))
            }
            ref other => bail!("RegionSelect 阶段拿到非法操作: {other:?}")
        },
        other => bail!("阶段 {other:?} 不是可采样的决策点")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{Checks, init_test_logger};

    /// 布局首尾相接、不重叠、总和等于 `POLICY_DIM`
    ///
    /// 这是「改了某段宽度却忘了改后续偏移」的唯一防线。
    #[test]
    fn test_layout_contiguous() -> Result<()> {
        let _ = init_test_logger("error");
        let segs = [
            ("不吃面", EAT_NONE, 1),
            ("吃面", EAT_BASE, REGION_NUM * TRIPLE_NUM),
            ("基础操作", TRAIN_BASE, TRAIN_OP_NUM),
            ("超级拉面", SUPER_BASE, SUPER_NUM),
            ("地区选择", REGION_SELECT_BASE, REGION_NUM)
        ];
        let mut c = Checks::new();
        let mut cur = 0usize;
        for (name, base, width) in segs {
            println!("  {name:8} [{base:3}, {:3})  宽 {width}", base + width);
            c.check(base == cur, &format!("{name} 段紧接前一段"));
            cur = base + width;
        }
        println!("总维度 = {cur}（常量 POLICY_DIM = {POLICY_DIM}）");
        c.check(cur == POLICY_DIM, "各段宽度之和等于 POLICY_DIM");
        c.check(POLICY_DIM == 234, "POLICY_DIM 与落盘格式冻结值一致");
        c.finish()
    }

    /// `TRIPLES` 恰好是「和 <= 2 的非负三元组」全集，且无重复
    #[test]
    fn test_triples_are_exactly_sum_le_2() -> Result<()> {
        let _ = init_test_logger("error");
        let mut want: Vec<[i32; 3]> = Vec::new();
        for a in 0..=2 {
            for b in 0..=2 - a {
                for c in 0..=2 - a - b {
                    want.push([a, b, c]);
                }
            }
        }
        println!("枚举得 {} 个三元组，常量表 {} 个", want.len(), TRIPLES.len());
        let mut c = Checks::new();
        c.check(want.len() == TRIPLE_NUM, "枚举个数等于 TRIPLE_NUM");
        c.check(want.as_slice() == TRIPLES.as_slice(), "顺序与常量表逐位相同");
        c.finish()
    }

    /// `triple_id` 与 `triple_of` 互为逆
    #[test]
    fn test_triple_roundtrip() -> Result<()> {
        let _ = init_test_logger("error");
        let mut c = Checks::new();
        for (i, t) in TRIPLES.iter().enumerate() {
            let back = triple_id(*t)?;
            println!("  {i} <-> {t:?}");
            c.check(back == i, &format!("triple_id({t:?}) == {i}"));
            c.check(triple_of(i)? == *t, &format!("triple_of({i}) == {t:?}"));
        }
        c.check(triple_id([1, 1, 1]).is_err(), "和为 3 的三元组被拒绝");
        c.check(triple_id([-1, 0, 0]).is_err(), "含负数的三元组被拒绝");
        c.check(triple_of(TRIPLE_NUM).is_err(), "越界 triple_id 被拒绝");
        c.finish()
    }

    /// 吃面格位互不碰撞，且全部落在吃面段内
    #[test]
    fn test_eat_index_unique_and_in_range() -> Result<()> {
        let _ = init_test_logger("error");
        let mut seen = vec![false; POLICY_DIM];
        let mut c = Checks::new();
        let none = eat_index(None, None)?;
        seen[none] = true;
        c.check(none == EAT_NONE, "不吃面落在 EAT_NONE");
        let mut dup = 0usize;
        for rid in 0..REGION_NUM {
            for t in TRIPLES.iter() {
                let i = eat_index(Some(rid), Some(*t))?;
                if seen[i] {
                    dup += 1;
                }
                seen[i] = true;
                if i < EAT_BASE || i >= TRAIN_BASE {
                    println!("  ⚠ 越段: rid={rid} t={t:?} -> {i}");
                    dup += 1;
                }
            }
        }
        println!("吃面格位共 {} 个，碰撞/越段 {dup} 次", REGION_NUM * TRIPLE_NUM + 1);
        c.check(dup == 0, "吃面格位无碰撞且不越段");
        c.check(eat_index(Some(REGION_NUM), Some([0, 0, 0])).is_err(), "越界地区被拒绝");
        c.check(eat_index(None, Some([1, 0, 0])).is_err(), "不吃面却带非零用法被拒绝");
        c.check(
            eat_index(None, Some([0, 0, 0]))? == EAT_NONE,
            "合并决策形态的不吃面（None + [0,0,0]）等价于 EAT_NONE"
        );
        c.finish()
    }

    /// 基础操作十个种类互异且落在本段内
    #[test]
    fn test_train_index_covers_ten_ops() -> Result<()> {
        let _ = init_test_logger("error");
        let ops = [
            Operation::Train(TrainingType::Speed),
            Operation::Train(TrainingType::Stamina),
            Operation::Train(TrainingType::Power),
            Operation::Train(TrainingType::Guts),
            Operation::Train(TrainingType::Wisdom),
            Operation::Race,
            Operation::Rest,
            Operation::NormalOuting,
            Operation::FriendOuting,
            Operation::Clinic
        ];
        let mut got: Vec<usize> = Vec::new();
        for op in ops.iter() {
            got.push(train_index(op)?);
        }
        println!("基础操作格位: {got:?}");
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let mut c = Checks::new();
        c.check(sorted.len() == TRAIN_OP_NUM, "十个操作格位互异");
        c.check(
            got.iter().all(|&i| (TRAIN_BASE..SUPER_BASE).contains(&i)),
            "全部落在基础操作段内"
        );
        c.check(train_index(&Operation::StageOnly).is_err(), "StageOnly 被拒绝");
        c.check(
            train_index(&Operation::SuperRamenSelect(0)).is_err(),
            "SuperRamenSelect 不走基础操作段"
        );
        c.finish()
    }

    /// `slots_of` 按阶段分派正确，且拒绝阶段/动作不匹配
    #[test]
    fn test_slots_of_dispatch() -> Result<()> {
        let _ = init_test_logger("error");
        let mut c = Checks::new();

        let eat = RamenAction::special_select(7, [1, 0, 0]);
        let s = slots_of(RamenStage::RamenSelect, &eat)?;
        println!("吃面(地区7, [1,0,0]) -> {s:?}");
        c.check(s == PolicySlots::One(EAT_BASE + 7 * TRIPLE_NUM + 6), "吃面格位正确");

        let tr = RamenAction::new(Operation::Train(TrainingType::Wisdom));
        println!("训练(智) -> {:?}", slots_of(RamenStage::Train, &tr)?);
        c.check(
            slots_of(RamenStage::Train, &tr)? == PolicySlots::One(TRAIN_BASE + 4),
            "训练格位正确"
        );

        let rg = RamenAction::new(Operation::RegionSelect([10, 15, 19]));
        let s = slots_of(RamenStage::RegionSelect, &rg)?;
        println!("地区选择[10,15,19] -> {s:?}");
        c.check(
            s == PolicySlots::Three([
                REGION_SELECT_BASE + 10,
                REGION_SELECT_BASE + 15,
                REGION_SELECT_BASE + 19
            ]),
            "地区选择三格正确"
        );
        c.check(s.as_slice().len() == 3, "as_slice 展平为三格");

        c.check(
            slots_of(RamenStage::Train, &rg).is_err(),
            "Train 阶段拿到 RegionSelect 操作被拒绝"
        );
        c.check(
            slots_of(RamenStage::RegionSelect, &tr).is_err(),
            "RegionSelect 阶段拿到训练操作被拒绝"
        );
        c.check(
            slots_of(RamenStage::Begin, &tr).is_err(),
            "非决策点阶段被拒绝"
        );
        c.check(
            slots_of(
                RamenStage::RegionSelect,
                &RamenAction::new(Operation::RegionSelect([10, 10, 11]))
            )
            .is_err(),
            "地区重复被拒绝"
        );
        c.finish()
    }

    // ========== 与规则层的耦合回归 ==========
    //
    // 上面 6 个测试都是**自洽**的：只拿本文件的常量互相验证，一次都没碰
    // `super::rules` 与真实动作生成器。因此 `list_special_targets_for` 的
    // `total_cap` 一旦放宽、或动作生成器多出一类新动作，上面全绿，
    // 而 10 万条样本会静默错位——正是模块头说要防的那件事。
    // 下面两个测试补的就是这条防线。

    /// 把一个格位归到它所属的段（仅用于测试报告）
    fn seg_name(slot: usize) -> &'static str {
        match slot {
            s if s == EAT_NONE => "不吃面",
            s if s < TRAIN_BASE => "吃面",
            s if s < SUPER_BASE => "基础操作",
            s if s < REGION_SELECT_BASE => "超级拉面",
            _ => "地区选择"
        }
    }

    /// 真实候选动作必须全部能落格，且同一决策点内格位不碰撞
    ///
    /// 用 [`crate::sampler`] 抽真实局面，把每个决策点的**每个真实候选**过一遍
    /// [`slots_of`]。除 `list_actions` 给出的三阶段形态外，`RamenSelect` 点额外
    /// 覆盖 [`super::game::RamenGame::list_combined_ramen_select_actions`] 的合并形态
    /// ——教师数据实际走的是后者（搜索开 `use_combined_ramen_select`）。
    ///
    /// `RegionSelect` 不做碰撞检查：top-3 编码下不同组合本就共享单地区格位，
    /// 那是设计而非缺陷。
    #[test]
    fn test_real_actions_map_to_slots() -> Result<()> {
        use std::collections::{BTreeMap, BTreeSet};

        use crate::{
            gamedata::init_global,
            sampler::{SamplerConfig, SamplingSpace, sample_position},
            utils::get_workspace_root
        };

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let space = SamplingSpace::gen1()?;
        let cfg = SamplerConfig::default();
        let mut c = Checks::new();
        let mut positions = 0usize;
        let mut acts = 0usize;
        let mut combined = 0usize;
        let mut seg_hits: BTreeMap<&str, usize> = BTreeMap::new();
        let mut stages: BTreeMap<String, usize> = BTreeMap::new();
        let mut bad: Vec<String> = Vec::new();

        for index in 0..300u64 {
            let Some(pos) = sample_position(&space, &cfg, index)?.into_captured() else {
                continue;
            };
            positions += 1;
            *stages.entry(format!("{:?}", pos.stage)).or_default() += 1;
            let skip_dup = pos.stage == RamenStage::RegionSelect;

            let mut seen: BTreeSet<usize> = BTreeSet::new();
            for act in pos.actions.iter() {
                acts += 1;
                match slots_of(pos.stage.clone(), act) {
                    Ok(slots) => {
                        for &s in slots.as_slice() {
                            *seg_hits.entry(seg_name(s)).or_default() += 1;
                            if !seen.insert(s) && !skip_dup {
                                bad.push(format!("index={index} 格位 {s} 在同一决策点重复"));
                            }
                        }
                    }
                    Err(e) => bad.push(format!("index={index} {:?} 落格失败: {e}", pos.stage))
                }
            }

            if pos.stage == RamenStage::RamenSelect {
                let mut seen_c: BTreeSet<usize> = BTreeSet::new();
                for act in pos.game.list_combined_ramen_select_actions() {
                    combined += 1;
                    match slots_of(pos.stage.clone(), &act) {
                        Ok(slots) => {
                            for &s in slots.as_slice() {
                                *seg_hits.entry(seg_name(s)).or_default() += 1;
                                if !seen_c.insert(s) {
                                    bad.push(format!("index={index} 合并动作格位 {s} 重复"));
                                }
                            }
                        }
                        Err(e) => bad.push(format!("index={index} 合并动作落格失败: {e}"))
                    }
                }
            }
        }

        println!("采样 300 次，捕获 {positions} 个决策点");
        println!("阶段分布: {stages:?}");
        println!("三阶段候选 {acts} 个，合并候选 {combined} 个");
        println!("命中的段: {seg_hits:?}");
        for line in bad.iter().take(20) {
            println!("  ⚠ {line}");
        }
        c.check(positions > 0, "应至少捕获一个决策点");
        c.check(bad.is_empty(), &format!("真实候选全部正确落格（异常 {} 条）", bad.len()));
        c.finish()
    }

    /// `list_special_targets_for` 的产物必须全部落在 [`TRIPLES`] 内
    ///
    /// 钉死的是 `rules.rs` 的 `total_cap = min(2, special_feeling)` 这条不变量。
    /// 它一旦放宽（比如允许和为 3），[`TRIPLE_NUM`] = 10 个格位就装不下，
    /// 而只对照常量表自身的 [`test_triples_are_exactly_sum_le_2`] 发现不了。
    #[test]
    fn test_rules_special_targets_stay_in_schema() -> Result<()> {
        use crate::{
            gamedata::init_global,
            game::ramen::rules::list_special_targets_for,
            sampler::{SamplerConfig, SamplingSpace, sample_position},
            utils::get_workspace_root
        };

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let space = SamplingSpace::gen1()?;
        let cfg = SamplerConfig::default();
        let mut c = Checks::new();
        let mut lists = 0usize;
        let mut triples = 0usize;
        let mut max_len = 0usize;
        let mut errs = 0usize;
        let mut bad: Vec<String> = Vec::new();

        for index in 0..200u64 {
            let Some(pos) = sample_position(&space, &cfg, index)?.into_captured() else {
                continue;
            };
            for rid in 0..REGION_NUM {
                let got = match list_special_targets_for(&pos.game.ramen, rid) {
                    Ok(v) => v,
                    Err(_) => {
                        errs += 1;
                        continue;
                    }
                };
                lists += 1;
                max_len = max_len.max(got.len());
                if got.len() > TRIPLE_NUM {
                    bad.push(format!("index={index} 地区 {rid} 返回 {} 个用法，超出 TRIPLE_NUM", got.len()));
                }
                for t in got.iter() {
                    triples += 1;
                    if let Err(e) = triple_id(*t) {
                        bad.push(format!("index={index} 地区 {rid} 用法 {t:?} 不在 schema 内: {e}"));
                    }
                }
            }
        }

        println!("扫描 {lists} 张用法表，共 {triples} 个三元组，单表最长 {max_len} / {TRIPLE_NUM}");
        println!("get_recipe 报错 {errs} 次（地区无配方，非缺陷）");
        for line in bad.iter().take(20) {
            println!("  ⚠ {line}");
        }
        c.check(lists > 0, "应至少扫到一张用法表");
        c.check(bad.is_empty(), &format!("规则层产物全部在 schema 内（异常 {} 条）", bad.len()));
        c.finish()
    }
}
