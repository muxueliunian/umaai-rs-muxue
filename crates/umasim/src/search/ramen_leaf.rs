//! 拉面截断续跑与叶估值的结果类型
//!
//! # 截断深度 H 的定义
//!
//! H 表示从根回合起**跨越多少个回合边界**——不是推理次数，也不是动作数。历史记法
//! 「H8」指的就是 8 个回合边界。目标回合 = 根回合 + H，根回合取自**执行根动作之前**
//! 的局面（根动作本身可能推进阶段）。
//!
//! # 为什么叶结果要和真实终局分成两种
//!
//! 真实终局手里有 25 维事实（五维、PT、达成率……）；叶估值手里只有网络预测的一个
//! 标量。把后者塞进 [`RamenTerminal`] 需要给那 25 维编造数值，之后所有读那张表的地方
//! （决策理由、终局差异日志、训练数据）都会把编出来的数当成事实。因此这里用枚举把
//! 两者分开，[`RamenLeafStats`] 里**只有真实终局**能并入终局维度。
//!
//! 同理，叶预测没有 PT 分量：[`TerminalRecord::has_pt_axis`] 返回 `false`，PT 累加器
//! 收不到样本，`best_idx_by(PtRank)` 会在截断臂上直接报错而不是给个错答案。
//!
//! # value 的语义
//!
//! `value[0]` 学的是「有限样本教师选一步、随后手写续跑」的交叉拟合普通均值，**不是**
//! 完整 NN 或持续重搜索的价值。用它替掉尾巴，评估的是「叶替换路径」的效应，
//! **不能**说成隔离出了纯降方差。

use crate::{
    game::ramen::{RamenGame, RamenStage},
    search::{
        MomentResult, NamedMetricRef, RamenTerminal, RamenTerminalStats, TerminalRecord, TerminalStats
    }
};

/// 该阶段是不是一个决策点
///
/// 与 `RamenGame::list_actions` 认的那五个阶段一致。
pub fn is_decision_stage(stage: &RamenStage) -> bool {
    matches!(
        stage,
        RamenStage::Train
            | RamenStage::RamenSelect
            | RamenStage::SpecialSelect
            | RamenStage::RegionSelect
            | RamenStage::SuperRamenSelect
    )
}

/// 叶估值可以落在的阶段
///
/// 独立的小枚举而不是直接存 [`RamenStage`]：
///
/// 1. [`RamenLeafFact`] 必须是 `Copy`（定长栈上记录，热路径不许堆分配），而
///    `RamenStage` 带 `Serialize` / `Sequence` 一串派生、不是 `Copy`；
/// 2. 更要紧的是，这样「叶只可能落在安全阶段」成了**类型层面**的事实，不靠调用方
///    自觉——构造 `RamenLeafFact` 就必须先拿到一个 [`Self::from_stage`] 的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamenLeafStage {
    /// 训练 / 比赛选择
    Train,
    /// 吃面选择
    RamenSelect,
    /// 年度地区选择
    RegionSelect,
    /// 超级拉面选择
    SuperRamenSelect
}

impl RamenLeafStage {
    /// 安全阶段 → 对应变体；其余（含 `SpecialSelect`）返回 `None`
    pub fn from_stage(stage: &RamenStage) -> Option<Self> {
        match stage {
            RamenStage::Train => Some(Self::Train),
            RamenStage::RamenSelect => Some(Self::RamenSelect),
            RamenStage::RegionSelect => Some(Self::RegionSelect),
            RamenStage::SuperRamenSelect => Some(Self::SuperRamenSelect),
            _ => None
        }
    }

    /// 进日志与 CSV 的短名
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "Train",
            Self::RamenSelect => "RamenSelect",
            Self::RegionSelect => "RegionSelect",
            Self::SuperRamenSelect => "SuperRamenSelect"
        }
    }
}

/// 该阶段能否**安全地**做叶估值
///
/// 判据是「特征编码器喂给模型的那份局面，与训练样本的含义一致」，不是「代码跑得通」：
///
/// - `SpecialSelect` **排除**。`SpecialSelectMode::Canonical` 下喂进去的是
///   `canonical_ramen_select_root` 还原出来的局面，其阶段 one-hot 落在 `RamenSelect`
///   槽、与上一拍逐位相同；在这里取 value 等于把上一拍的预测再读一遍，不是一个新的
///   叶估值。到达目标回合却撞上它时记一次**顺延**，正常推进到下一个安全边界。
/// - 其余四个决策阶段各有自己的阶段 one-hot。
/// - 非决策阶段与 `Settlement` 一律不估值。
pub fn leaf_safe_stage(stage: &RamenStage) -> bool {
    RamenLeafStage::from_stage(stage).is_some()
}

/// 在一个阶段边界上该做什么
///
/// 抽成独立判据而不是写死在 rollout 循环里，是为了让它**可以单独测**——尤其是
/// [`Self::Defer`]：当前阶段顺序下，一个回合里 `RamenSelect` 必定排在 `SpecialSelect`
/// 之前，所以从合法根出发的真实 rollout **走不到**那条分支（叶总是先落在
/// `RamenSelect`）。它是一道守门，不是热路径；把它留成内联代码就只能靠读代码确认。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafVerdict {
    /// 在这个安全阶段估值
    Evaluate(RamenLeafStage),
    /// 到了目标回合，但阶段不安全：记一次顺延，正常推进
    Defer,
    /// 还没到目标回合，或这里根本不是决策点：正常推进
    Continue
}

/// 判断当前边界该估值、该顺延，还是继续推进
///
/// `target_turn` 为根回合 + H。判据**独立于**候选数与守门：单候选、自选比赛守门命中
/// 照样算决策点，因为它们在训练样本意义上就是决策点。
pub fn leaf_verdict(turn: i32, target_turn: i32, stage: &RamenStage) -> LeafVerdict {
    if turn < target_turn || !is_decision_stage(stage) {
        return LeafVerdict::Continue;
    }
    match RamenLeafStage::from_stage(stage) {
        Some(s) => LeafVerdict::Evaluate(s),
        None => LeafVerdict::Defer
    }
}

/// 把一个**非终局**局面估成结算分量纲的标量
///
/// 独立成 trait 而不是直接依赖具体网络：搜索层只需要「给我一个数」，装配、模型加载与
/// 反归一化属于 trainer 层的事。CPU 实现见
/// [`RamenNnLeafValuer`](crate::trainer::RamenNnLeafValuer)。
pub trait RamenLeafValuer: Send + Sync {
    /// 估一个叶局面，返回**已反归一化**的结算分预测
    ///
    /// 实现必须保证：反归一化恰好做一次、不叠加当前分、不取整，且返回值有限。
    ///
    /// # 错误
    ///
    /// 编码、推理、形状校验或有限性检查失败时报错——**不得**静默退回手写估值，
    /// 那会把「推理坏了」混同成「这条路线分低」。
    fn value_of(&self, game: &RamenGame) -> anyhow::Result<f64>;

    /// 本估值器至今发出的**叶**请求数
    ///
    /// 与 policy 决策的推理请求分开计数：混在一起就说不清叶路径到底花了多少推理。
    fn leaf_requests(&self) -> u64;

    /// 简短标签（进日志与 CSV，用于区分对照臂）
    fn label(&self) -> String;
}

/// 一次叶估值的事实
///
/// 刻意**不含**任何终局维度：叶处没有五维、没有 PT、没有达成率。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamenLeafFact {
    /// 估值发生的回合
    pub turn: i32,
    /// 估值发生的阶段（类型上只可能是安全阶段）
    pub stage: RamenLeafStage,
    /// 反归一化后的结算分预测
    pub value: f64,
    /// 本条轨迹到达目标回合后，因阶段不安全而顺延的次数
    pub deferrals: u32
}

/// 一次截断 rollout 的结果：真实终局，或叶预测
///
/// 两者在**类型上**就分得开——这正是「不能把预测值塞进 `RolloutOutcome<RamenTerminal>`」
/// 那条约束的落点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RamenLeafRecord {
    /// 跑到了真实终局：带完整 25 维事实，评分是**真实** `search_score()`
    Terminal(RamenTerminal),
    /// 在安全叶边界停下：只有一个预测标量
    Leaf(RamenLeafFact)
}

/// 截断臂按候选累加的统计
///
/// [`Self::terminal`] **只**吃真实终局样本，故它的 `count()` 是该候选跑到终局的次数，
/// 一般小于总 rollout 数——这不是 bug，是「叶样本不进真实终局统计」的直接体现。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RamenLeafStats {
    /// 真实终局的 25 维统计（叶样本**不**并入）
    pub terminal: RamenTerminalStats,
    /// 本次是不是叶（1 = 叶，0 = 真实终局）；其 `count()` 即**总**样本数
    pub leaf_flag: MomentResult,
    /// 叶所在回合（只有叶样本并入）
    pub leaf_turn: MomentResult,
    /// 顺延次数（只有叶样本并入）
    pub leaf_deferrals: MomentResult,
    /// 叶预测值（只有叶样本并入）
    pub leaf_value: MomentResult
}

impl RamenLeafStats {
    /// 总样本数（真实终局 + 叶）
    pub fn total_count(&self) -> u32 {
        self.leaf_flag.count()
    }

    /// 叶预测样本数
    pub fn leaf_count(&self) -> u32 {
        self.leaf_turn.count()
    }

    /// 跑到真实终局的样本数
    pub fn terminal_count(&self) -> u32 {
        self.total_count() - self.leaf_count()
    }

    /// 顺延次数合计
    pub fn deferral_sum(&self) -> f64 {
        self.leaf_deferrals.sum
    }
}

impl TerminalRecord for RamenLeafRecord {
    type Stats = RamenLeafStats;

    fn accumulate_into(&self, stats: &mut Self::Stats) {
        match self {
            RamenLeafRecord::Terminal(t) => {
                t.accumulate_into(&mut stats.terminal);
                stats.leaf_flag.add(0.0);
            }
            RamenLeafRecord::Leaf(f) => {
                // ❗**不**碰 stats.terminal：叶处那 25 维不存在，补 0 等于伪造事实
                stats.leaf_flag.add(1.0);
                stats.leaf_turn.add(f64::from(f.turn));
                stats.leaf_deferrals.add(f64::from(f.deferrals));
                stats.leaf_value.add(f.value);
            }
        }
    }

    /// 叶预测没有 PT 分量
    fn has_pt_axis(&self) -> bool {
        matches!(self, RamenLeafRecord::Terminal(_))
    }
}

impl TerminalStats for RamenLeafStats {
    fn visit(&self, visitor: &mut dyn FnMut(NamedMetricRef<'_>)) {
        // 终局那 25 维原样转发：键名与非截断路径一致，下游表头不分叉
        self.terminal.visit(visitor);
        for (key, label, unit, result) in [
            ("leaf_rate", "叶预测占比", "flag", &self.leaf_flag),
            ("leaf_turn", "叶所在回合", "turn", &self.leaf_turn),
            ("leaf_deferrals", "叶边界顺延次数", "count", &self.leaf_deferrals),
            ("leaf_value", "叶预测值", "score", &self.leaf_value)
        ] {
            visitor(NamedMetricRef { key, label, unit, result });
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::{search::FROZEN_DIM_KEYS, utils::Checks};

    /// 叶样本只进叶那几维，真实终局只进终局那 25 维
    #[test]
    fn test_leaf_samples_stay_out_of_terminal_stats() -> Result<()> {
        let mut c = Checks::new();
        let mut stats = RamenLeafStats::default();

        let leaf = RamenLeafRecord::Leaf(RamenLeafFact {
            turn: 24,
            stage: RamenLeafStage::Train,
            value: 68000.0,
            deferrals: 1
        });
        for _ in 0..3 {
            leaf.accumulate_into(&mut stats);
        }
        println!(
            "只加 3 个叶样本后：总 {} / 叶 {} / 终局 {} / 终局首维样本数 {}",
            stats.total_count(),
            stats.leaf_count(),
            stats.terminal_count(),
            stats.terminal.speed_score.count()
        );
        c.check(stats.total_count() == 3, "总样本数 3");
        c.check(stats.leaf_count() == 3, "叶样本数 3");
        c.check(stats.terminal_count() == 0, "真实终局样本数 0");
        c.check(
            stats.terminal.speed_score.count() == 0,
            "终局 25 维一个样本都没收到（叶不伪造终局事实）"
        );
        c.check(stats.deferral_sum() == 3.0, "顺延次数按条累加");
        c.check(!leaf.has_pt_axis(), "叶样本自报没有 PT 分量");
        c.check(stats.leaf_value.mean() == 68000.0, "叶预测值进了 leaf_value 维");
        c.finish()
    }

    /// `visit` 把终局 25 维与叶 4 维一起报出来，且键名不撞车
    #[test]
    fn test_visit_covers_terminal_and_leaf_dims() -> Result<()> {
        let mut c = Checks::new();
        let stats = RamenLeafStats::default();
        let mut keys: Vec<&'static str> = Vec::new();
        stats.visit(&mut |m| keys.push(m.key));
        println!("维度 {} 个: {keys:?}", keys.len());
        c.check(
            keys.len() == FROZEN_DIM_KEYS.len() + 4,
            &format!("维度数 = 冻结 {} + 叶 4", FROZEN_DIM_KEYS.len())
        );
        c.check(
            keys[..FROZEN_DIM_KEYS.len()] == FROZEN_DIM_KEYS,
            "前 25 维与冻结表逐项同名同序（下游表头不分叉）"
        );
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        c.check(sorted.len() == keys.len(), "没有重名维度");
        c.finish()
    }

    /// 三种边界判定各走一遍，`Defer` 尤其要有真覆盖
    ///
    /// 真实 rollout 走不到 `Defer`（同一回合里 `RamenSelect` 排在 `SpecialSelect` 之前，
    /// 叶总是先落在前者），所以只能在这里单独钉住它的语义。
    ///
    /// # 错误
    ///
    /// 任一判定不符时返回错误。
    #[test]
    fn test_leaf_verdict_covers_all_three_branches() -> Result<()> {
        let mut c = Checks::new();
        let target = 20;
        for (turn, stage, want, why) in [
            (19, RamenStage::Train, LeafVerdict::Continue, "没到目标回合：继续推进"),
            (20, RamenStage::Distribute, LeafVerdict::Continue, "到了目标回合但不是决策点：继续推进"),
            (20, RamenStage::Settlement, LeafVerdict::Continue, "结算阶段不估值"),
            (20, RamenStage::SpecialSelect, LeafVerdict::Defer, "❗到了目标回合但阶段不安全：顺延"),
            (25, RamenStage::SpecialSelect, LeafVerdict::Defer, "过了目标回合仍撞上不安全阶段：仍顺延"),
            (
                20,
                RamenStage::Train,
                LeafVerdict::Evaluate(RamenLeafStage::Train),
                "到了目标回合且阶段安全：估值"
            ),
            (
                47,
                RamenStage::RegionSelect,
                LeafVerdict::Evaluate(RamenLeafStage::RegionSelect),
                "跨年度地区选择同样是安全边界"
            ),
            (
                71,
                RamenStage::SuperRamenSelect,
                LeafVerdict::Evaluate(RamenLeafStage::SuperRamenSelect),
                "超级拉面选择同样是安全边界"
            )
        ] {
            let got = leaf_verdict(turn, target, &stage);
            println!("turn={turn} target={target} {stage:?} -> {got:?}");
            c.check(got == want, &format!("{why}（{stage:?} @ {turn}）"));
        }
        c.finish()
    }

    /// 安全叶边界的判据：`SpecialSelect` 必须被排除
    #[test]
    fn test_leaf_safe_stage_excludes_special_select() -> Result<()> {
        let mut c = Checks::new();
        for st in [
            RamenStage::Train,
            RamenStage::RamenSelect,
            RamenStage::RegionSelect,
            RamenStage::SuperRamenSelect
        ] {
            println!("{st:?}: 决策点={} 可估值={}", is_decision_stage(&st), leaf_safe_stage(&st));
            c.check(leaf_safe_stage(&st), &format!("{st:?} 是安全叶边界"));
            c.check(is_decision_stage(&st), &format!("{st:?} 是决策点"));
            c.check(
                RamenLeafStage::from_stage(&st).is_some_and(|x| format!("{st:?}") == x.as_str()),
                &format!("{st:?} 的叶阶段短名与阶段名一致")
            );
        }
        println!(
            "SpecialSelect: 决策点={} 可估值={}",
            is_decision_stage(&RamenStage::SpecialSelect),
            leaf_safe_stage(&RamenStage::SpecialSelect)
        );
        c.check(
            is_decision_stage(&RamenStage::SpecialSelect),
            "SpecialSelect 是决策点（所以会撞上目标回合）"
        );
        c.check(
            !leaf_safe_stage(&RamenStage::SpecialSelect),
            "❗但**不是**安全叶边界：canonical 还原后阶段 one-hot 与上一拍重复"
        );
        for st in [
            RamenStage::Begin,
            RamenStage::Distribute,
            RamenStage::AfterTrain,
            RamenStage::NextTurn,
            RamenStage::Settlement,
            RamenStage::BeginAfterRegionSelect
        ] {
            c.check(!leaf_safe_stage(&st), &format!("{st:?} 不估值"));
            c.check(!is_decision_stage(&st), &format!("{st:?} 不是决策点"));
        }
        c.finish()
    }
}
