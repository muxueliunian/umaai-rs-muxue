//! 扁平搜索的剧本适配层
//!
//! [`Game`] 负责规则推进；本模块的 [`FlatSearchGame`] 只补搜索额外需要、
//! 且**不能安全给默认值**的能力，避免把搜索关注点塞进已经很大的 `Game` trait。
//!
//! # 设计约束
//!
//! 所有方法**一律不给默认实现**（除终局取分外，它对所有剧本同构）。
//! 理由：新增剧本时漏实现会编译失败，而不是静默退化——
//! [`FlatSearchGame::fork_for_rollout`] 若被漏掉，拉面会每次 rollout 各摸一次
//! OS 随机数，可复现性与 CRN 一起失效，且**不会有任何报错**。

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;

use crate::game::{
    Game,
    Trainer,
    onsen::{OnsenTurnStage, game::OnsenGame},
    ramen::{Operation, RamenGame, RamenStage}
};

/// 一次 rollout 的两种终局评分口径
///
/// 新类型而非裸 `(f64, f64)`：两个分数量纲相同、极易写反。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchScore {
    /// 结算评分
    pub score: f64,
    /// 计入 PT 偏好的评分
    pub score_pt: f64
}

/// 可被 [`FlatSearch`](super::FlatSearch) 驱动的剧本
///
/// # 实现须知
///
/// 任何**不由传入 `&mut StdRng` 驱动**的随机流，必须在 [`Self::fork_for_rollout`]
/// 中重置，否则搜索层的种子控制不到它。
pub trait FlatSearchGame: Game + Clone + Send + Sync
where
    Self::Action: Send + Sync + Clone
{
    /// 本剧本 rollout 阶段使用的决策器
    ///
    /// 用关联类型而非 `FlatSearch<G, T>` 的第二个类型参数：每个剧本现阶段只有一种
    /// 基策，多一个参数会把 `MctsTrainer` 与 `umaai` 的调用签名全部掀开。
    type RolloutTrainer: Trainer<Self> + Send + Sync;

    /// 是否支持 `max_depth > 0` 的截断估值
    ///
    /// 截断后需要 leaf 估值器，目前只有 onsen 有（`HandwrittenEvaluator` /
    /// NN）。拉面为 `false`，搜索入口会在开跑前直接拒绝而不是给出错误标签。
    const SUPPORTS_TRUNCATED_LEAF: bool;

    /// 构造默认 rollout 决策器
    fn default_rollout_trainer() -> Self::RolloutTrainer;

    /// CRN 种子派生用的阶段编号（onsen 外挂重播种用，v2 §5.2）
    ///
    /// 必须显式 `match`，**不得依赖枚举判别值**：变体顺序若调整，显式 match 会
    /// 编译报错提醒同步，而判别值会静默改变所有历史种子。
    /// 注：拉面规则层已由无状态流接管，不再调用本方法（保留实现仅为满足 trait）。
    fn crn_stage_key(&self) -> u64;

    /// 按剧本的**真实对局路径**执行 rollout 的根动作
    ///
    /// 必须与 `run_stage` 内执行动作的方式逐字一致。拉面的
    /// `run_train` / `run_ramen_select` / `run_special_select` / `run_region_select`
    /// 走的是 `apply_action_with_strategy`（优先用局面内的策略流），而不是
    /// `Game::apply_action`；rollout 若直接调后者，被评估的那一个动作会用上
    /// 另一条随机流，且策略流的 counter 不前进，后续回合的策略随机整体偏移——
    /// 搜索排序会被系统性污染，且不报任何错。
    ///
    /// 无默认实现：漏实现要编译失败，而不是静默退化。
    fn apply_root_action(&mut self, action: &Self::Action, rng: &mut StdRng) -> Result<()>;

    /// 为一次 rollout 建立分支，并初始化剧本内部随机流
    ///
    /// 通用搜索**只能**经由本方法建分支，不得直接 `clone()`——把「克隆」与
    /// 「重置内部 RNG」绑成一个不可分割的操作，是防止遗漏的唯一可靠手段。
    fn fork_for_rollout(&self, rollout_seed: u64) -> Self;

    /// 当前状态的两种终局评分
    ///
    /// 唯一给默认实现的方法：两个剧本都经 [`Game::uma`] 取分，逻辑同构。
    fn search_score(&self) -> SearchScore {
        SearchScore {
            score: self.uma().calc_score() as f64,
            score_pt: self.uma().calc_score_with_pt_favor() as f64
        }
    }
}

impl FlatSearchGame for OnsenGame {
    type RolloutTrainer = crate::trainer::HandwrittenTrainer;

    const SUPPORTS_TRUNCATED_LEAF: bool = true;

    fn default_rollout_trainer() -> Self::RolloutTrainer {
        crate::trainer::HandwrittenTrainer::new()
    }

    fn crn_stage_key(&self) -> u64 {
        match self.stage {
            OnsenTurnStage::Begin => 0,
            OnsenTurnStage::Distribute => 1,
            OnsenTurnStage::Bathing => 2,
            OnsenTurnStage::Train => 3,
            OnsenTurnStage::AfterTrain => 4
        }
    }

    /// 温泉的 `run_stage` 直接调 `Game::apply_action`，无策略流分支
    fn apply_root_action(&mut self, action: &Self::Action, rng: &mut StdRng) -> Result<()> {
        self.apply_action(action, rng)
    }

    /// 温泉没有规则层内部 RNG（`game/onsen/` 下无 `internal_rng`），
    /// 全部随机性走传入的 `&mut StdRng`，故只需克隆。
    fn fork_for_rollout(&self, _rollout_seed: u64) -> Self {
        self.clone()
    }
}

impl FlatSearchGame for RamenGame {
    /// rollout 基策 = 正式推荐手写策略
    ///
    /// 2026-08-27 切换：原用 `RamenHandwrittenTrainer`（纯 RamenPolicy，缺平衡/吃面联动/
    /// 体力门限等机制），切到 [`RecommendedRamenTrainer`] 后搜索评分与正式手写策略对齐，
    /// 排序结果更有意义；门控全关时与纯推荐策略逐位等价。决策开销 ×6.36（RamenSelect
    /// 预演主导），单局 ×2.10，搜索预算需相应调小或 train_only。
    type RolloutTrainer = crate::trainer::RecommendedRamenTrainer;

    /// 拉面暂无 leaf 估值器，Phase 1 只允许跑到终局
    const SUPPORTS_TRUNCATED_LEAF: bool = false;

    /// rollout 专用实例：三份年的 breakdown 全部关闭
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        crate::trainer::RecommendedRamenTrainer::for_rollout()
    }

    /// 拉面 stage key（保留实现仅为满足 trait；规则层接管后不再被调用）
    ///
    /// 0–9 与既有变体一一对应，不得重排。[`RamenStage::BeginAfterRegionSelect`] 用新键 10。
    fn crn_stage_key(&self) -> u64 {
        match self.stage {
            RamenStage::Begin => 0,
            RamenStage::Distribute => 1,
            RamenStage::RamenSelect => 2,
            RamenStage::SpecialSelect => 3,
            RamenStage::Train => 4,
            RamenStage::AfterTrain => 5,
            RamenStage::NextTurn => 6,
            RamenStage::RegionSelect => 7,
            RamenStage::SuperRamenSelect => 8,
            RamenStage::Settlement => 9,
            RamenStage::BeginAfterRegionSelect => 10
        }
    }

    /// 与 `run_train` / `run_ramen_select` / `run_special_select` / `run_region_select`
    /// 保持一致：走策略流而非传入的决策 rng。
    ///
    /// 例外：`RamenSelect` + `StageOnly` + `special_targets.is_some()` 视为合并动作，
    /// 走 [`RamenGame::apply_combined_ramen_decision`]（`apply_action` 会丢掉 targets）。
    /// race_turn 一体化动作（`operation` 非 `StageOnly`）仍走策略流。
    fn apply_root_action(&mut self, action: &Self::Action, rng: &mut StdRng) -> Result<()> {
        // 合并动作判别：RamenSelect 阶段 + StageOnly + 携带 special_targets
        if self.stage == RamenStage::RamenSelect
            && matches!(action.operation, Operation::StageOnly)
            && action.special_targets.is_some()
        {
            let targets = action
                .special_targets
                .ok_or_else(|| anyhow!("合并动作应携带 special_targets，但为 None"))?;
            return self.apply_combined_ramen_decision(action.ramen, targets);
        }
        self.apply_action_with_strategy(action, rng)
    }

    /// 拉面规则层为无状态流（RNG Refactor Plan v2 §5.2）
    ///
    /// rollout 分支 = 克隆局面 + 注入 `rule_master = rollout 种子`：规则层
    /// 每回合按 `(rule_master, turn)` 派生固定流/策略流，所有候选共享同一
    /// rollout 种子时面对逐位一致的随机未来——CRN 对齐由规则层自身承担，
    /// 搜索层不再需要按阶段重播种。
    fn fork_for_rollout(&self, rollout_seed: u64) -> Self {
        let mut game = self.clone();
        game.set_rule_master(rollout_seed);
        game
    }
}

/// 跑完一次 rollout 到终局（供剧本特判路径回调）
///
/// 温泉的 `Dig`/`Upgrade` 特判需要「继续把这局跑完」的能力，但那属于搜索侧逻辑。
/// 该 trait 只暴露这一条最小能力，不把整套 rollout 协议开放给剧本。
pub trait RolloutHost<G: FlatSearchGame>
where
    G::Action: Send + Sync + Clone
{
    /// 从给定局面跑到终局，返回终局评分
    fn rollout_to_end(&self, game: &G, rng: &mut StdRng) -> Result<SearchScore>;
}
