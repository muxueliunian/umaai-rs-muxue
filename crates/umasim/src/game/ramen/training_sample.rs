//! 拉面杯教师样本容器（NN 管线 Phase 3）
//!
//! 本模块只定义**一条教师样本在内存里的形状**：定长特征 + 元信息 + 变长候选表。
//! 它与另外三件事的关系必须先说清楚，否则很容易被当成别的东西：
//!
//! - **动作 ID 空间已冻结**，在 [`super::policy_schema`]（234 格）。本模块直接复用
//!   它的 [`PolicySlots`]，一律不自己拼偏移量。
//! - **落盘格式尚未冻结**。末尾的 [`RamenSampleBatch`] 只是 pilot 期能用的 bincode
//!   落盘，正式格式（mmap 友好的 CSR 三件套）另议——**不要把它当契约**。
//! - **标签不在这里**。policy / value 目标是按某组 `radical_factor` / `policy_delta`
//!   离线生成的可再生 sidecar，本容器存的是生成标签所需的**原始素材**。
//!
//! # 为什么按 rollout 序号存定长槽位
//!
//! 搜索用 CRN：同一个 `rollout_index` 在各候选之间共享随机种子。把顺序展平进直方图
//! （现行 [`crate::search::ActionResult`] 就是这么做的）会永久丢掉配对信息，
//! 而配对信息是离线做 cross-fitting、对抗 winner's curse 的唯一来源。
//! 既然已决定落盘每候选的原始分，有序与无序占的磁盘一样大，所以现在就存有序的。
//!
//! **失败的 rollout 不能简单跳过**：一旦跳过，其后所有 rollout 在候选间就整体错位，
//! CRN 配对当场失效。因此每个候选都带一份 [`ValidMask`]，失败的槽位留空并标记无效；
//! 采集方也可以选择「任一 rollout 失败就丢弃整个根局面」，两种做法本容器都支持。
//!
//! # 精度
//!
//! rollout 分按 `f32` 存（同为 4 字节，2²⁴ 以内对整数无损），而 `n` / `sum` / `sum_sq`
//! 由**原始 `f64`** 累加，因此均值与标准差不受存储降精度影响。

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};

use super::{
    RamenGame,
    RamenStage,
    features::INPUT_DIM,
    policy_schema::{POLICY_DIM, PolicySlots}
};
use crate::game::Game;

/// 容器格式版本
///
/// 语义变更（字段含义、格位口径、精度约定）必须递增，纯新增可选字段不必。
pub const SAMPLE_FORMAT_VERSION: u32 = 1;

// ============================================================================
// 阶段编码
// ============================================================================

/// 决策点阶段 → **稳定编码**
///
/// 这是数据格式的一部分：`RamenStage` 是普通枚举，往中间插一个变体就会改掉
/// bincode 判别值，因此这里显式写死映射，不依赖枚举顺序。
///
/// # 错误
///
/// 传入非决策点阶段时报错——那种样本本就不该被采集。
pub fn stage_code(stage: &RamenStage) -> Result<u8> {
    Ok(match stage {
        RamenStage::RamenSelect => 0,
        RamenStage::SpecialSelect => 1,
        RamenStage::Train => 2,
        RamenStage::SuperRamenSelect => 3,
        RamenStage::RegionSelect => 4,
        other => bail!("阶段 {other:?} 不是可采样的决策点")
    })
}

/// 稳定编码 → 决策点阶段（[`stage_code`] 的逆）
///
/// # 错误
///
/// 编码未定义时报错。
pub fn stage_of_code(code: u8) -> Result<RamenStage> {
    Ok(match code {
        0 => RamenStage::RamenSelect,
        1 => RamenStage::SpecialSelect,
        2 => RamenStage::Train,
        3 => RamenStage::SuperRamenSelect,
        4 => RamenStage::RegionSelect,
        other => bail!("未定义的阶段编码: {other}")
    })
}

// ============================================================================
// 有效性位图
// ============================================================================

/// 按 rollout 序号的有效性位图
///
/// 位为 1 表示该序号上的 rollout 成功、分数可用；为 0 表示该次 rollout 失败，
/// 对应槽位的分数是占位的 `0.0`，**不得参与统计**。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidMask {
    /// 位数（= rollout 次数）
    len: usize,
    /// 位存储，低位在前；尾部未用位恒为 0
    words: Vec<u64>
}

impl ValidMask {
    /// 新建全部有效的位图
    pub fn all_valid(len: usize) -> Self {
        let mut words = vec![u64::MAX; len.div_ceil(64)];
        // 清掉尾部未用位，否则 count_valid 会多算
        let rem = len % 64;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last = (1u64 << rem) - 1;
            }
        }
        Self { len, words }
    }

    /// 设置某个序号的有效性
    ///
    /// # 错误
    ///
    /// 序号越界时报错。
    pub fn set(&mut self, idx: usize, valid: bool) -> Result<()> {
        ensure!(idx < self.len, "rollout 序号越界: {idx} >= {}", self.len);
        let (w, b) = (idx / 64, idx % 64);
        if valid {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
        Ok(())
    }

    /// 查询某个序号是否有效（越界一律视为无效）
    pub fn is_valid(&self, idx: usize) -> bool {
        idx < self.len && (self.words[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// 位数
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 有效位个数
    pub fn count_valid(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 内部一致性：字数正确、尾部未用位为 0
    fn check(&self) -> Result<()> {
        ensure!(
            self.words.len() == self.len.div_ceil(64),
            "位图字数与长度不符: {} 字 vs {} 位",
            self.words.len(),
            self.len
        );
        let rem = self.len % 64;
        if rem != 0 {
            let last = self.words.last().copied().unwrap_or(0);
            ensure!(last >> rem == 0, "位图尾部未用位不为 0");
        }
        Ok(())
    }
}

// ============================================================================
// 候选
// ============================================================================

/// 一个候选动作的完整 rollout 记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamenCandidate {
    /// 该动作占用的 policy 格位（地区选择为三格）
    pub slots: PolicySlots,
    /// 按 rollout 序号定长的分数槽位；失败槽位为 `0.0`，须配合 [`Self::valid`] 读
    pub scores: Vec<f32>,
    /// 有效性位图
    pub valid: ValidMask,
    /// 成功的 rollout 次数
    pub n: u32,
    /// 原始 `f64` 分数之和
    pub sum: f64,
    /// 原始 `f64` 分数平方之和
    pub sum_sq: f64
}

impl RamenCandidate {
    /// 从**按 rollout 序号排好**的结果构造
    ///
    /// `rollouts[i]` 为 `None` 表示第 i 次 rollout 失败。调用方必须保证下标就是
    /// 搜索里的 `rollout_index`，否则跨候选的 CRN 配对会错位。
    ///
    /// # 错误
    ///
    /// rollout 为空、格位越界、或分数不是有限值时报错。
    pub fn from_rollouts(slots: PolicySlots, rollouts: &[Option<f64>]) -> Result<Self> {
        ensure!(!rollouts.is_empty(), "候选至少要有一次 rollout");
        for &s in slots.as_slice() {
            ensure!(s < POLICY_DIM, "policy 格位越界: {s} >= {POLICY_DIM}");
        }
        let mut scores = vec![0.0f32; rollouts.len()];
        let mut valid = ValidMask::all_valid(rollouts.len());
        let (mut n, mut sum, mut sum_sq) = (0u32, 0.0f64, 0.0f64);
        for (i, r) in rollouts.iter().enumerate() {
            match r {
                Some(v) => {
                    ensure!(v.is_finite(), "rollout {i} 的分数不是有限值: {v}");
                    scores[i] = *v as f32;
                    n += 1;
                    sum += v;
                    sum_sq += v * v;
                }
                None => valid.set(i, false)?
            }
        }
        Ok(Self {
            slots,
            scores,
            valid,
            n,
            sum,
            sum_sq
        })
    }

    /// rollout 槽位总数（含失败的）
    pub fn rollouts(&self) -> usize {
        self.scores.len()
    }

    /// 失败次数
    pub fn failed(&self) -> usize {
        self.rollouts() - self.n as usize
    }

    /// 均值；无有效样本时为 0，与 [`crate::search::ActionResult::mean`] 同口径
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.sum / self.n as f64
    }

    /// 样本标准差（除以 `n - 1`），与 [`crate::search::ActionResult::stdev`] 同口径
    pub fn stdev(&self) -> f64 {
        if self.n <= 1 {
            return 0.0;
        }
        let n = self.n as f64;
        ((self.sum_sq - self.sum * self.sum / n) / (n - 1.0)).max(0.0).sqrt()
    }

    /// 内部一致性检查
    ///
    /// # 错误
    ///
    /// 长度不匹配、位图损坏、或 `n` 与位图不符时报错。
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.scores.len() == self.valid.len(),
            "分数槽位数 {} 与位图长度 {} 不符",
            self.scores.len(),
            self.valid.len()
        );
        self.valid.check()?;
        ensure!(
            self.n as usize == self.valid.count_valid(),
            "n = {} 与位图有效位数 {} 不符",
            self.n,
            self.valid.count_valid()
        );
        for &s in self.slots.as_slice() {
            ensure!(s < POLICY_DIM, "policy 格位越界: {s} >= {POLICY_DIM}");
        }
        Ok(())
    }
}

// ============================================================================
// 元信息与样本
// ============================================================================

/// 每条样本的定长元信息
///
/// 牌组与马娘不入元信息：它们由 [`crate::sampler::SamplingSpace`] 与 `index`
/// 确定性地还原，重复存反而给了两处可以不一致的余地。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamenSampleMeta {
    /// 采样工作项序号（配合采样空间可完全还原局面）
    pub index: u64,
    /// 决策点所在回合
    pub turn: i32,
    /// 决策点阶段的稳定编码，见 [`stage_code`]
    pub stage: u8,
    /// 搜索根种子（CRN 的起点，用于复现与划分 fold）
    pub root_seed: u64
}

impl RamenSampleMeta {
    /// 构造元信息
    ///
    /// # 错误
    ///
    /// 阶段不是决策点时报错。
    pub fn new(index: u64, turn: i32, stage: &RamenStage, root_seed: u64) -> Result<Self> {
        Ok(Self {
            index,
            turn,
            stage: stage_code(stage)?,
            root_seed
        })
    }

    /// 还原阶段
    ///
    /// # 错误
    ///
    /// 编码未定义时报错。
    pub fn stage(&self) -> Result<RamenStage> {
        stage_of_code(self.stage)
    }
}

/// 一条拉面杯教师样本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamenTrainingSample {
    /// 容器格式版本
    pub format_version: u32,
    /// 元信息
    pub meta: RamenSampleMeta,
    /// 定长输入特征，长度恒为 [`INPUT_DIM`]
    pub features: Vec<f32>,
    /// 变长候选表；每个候选的 rollout 槽位数必须一致
    pub candidates: Vec<RamenCandidate>
}

impl RamenTrainingSample {
    /// 构造并校验一条样本
    ///
    /// # 错误
    ///
    /// 特征维度不符、特征含非有限值、候选表为空、候选间 rollout 数不一致、
    /// 或任一候选自身不一致时报错。这些都是会静默污染数据集的错误，不能放过。
    pub fn new(meta: RamenSampleMeta, features: Vec<f32>, candidates: Vec<RamenCandidate>) -> Result<Self> {
        ensure!(
            features.len() == INPUT_DIM,
            "特征维度必须是 {INPUT_DIM}，实得 {}",
            features.len()
        );
        if let Some(i) = features.iter().position(|v| !v.is_finite()) {
            bail!("特征第 {i} 维不是有限值: {}", features[i]);
        }
        ensure!(!candidates.is_empty(), "候选表不得为空");
        let rollouts = candidates[0].rollouts();
        for (i, c) in candidates.iter().enumerate() {
            c.check().with_context(|| format!("候选 {i} 不一致"))?;
            ensure!(
                c.rollouts() == rollouts,
                "候选 {i} 的 rollout 数 {} 与候选 0 的 {rollouts} 不一致——CRN 配对要求逐次对齐",
                c.rollouts()
            );
        }
        meta.stage()?;
        Ok(Self {
            format_version: SAMPLE_FORMAT_VERSION,
            meta,
            features,
            candidates
        })
    }

    /// 每候选的 rollout 槽位数
    pub fn rollouts(&self) -> usize {
        self.candidates.first().map(|c| c.rollouts()).unwrap_or(0)
    }

    /// 动作合法掩码：本决策点上被搜索评过的 policy 格位置 1
    ///
    /// 地区选择下多个组合共享单地区格位，掩码因此只表示「该地区出现在某个候选里」。
    pub fn legal_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0f32; POLICY_DIM];
        for c in self.candidates.iter() {
            for &s in c.slots.as_slice() {
                mask[s] = 1.0;
            }
        }
        mask
    }
}

impl crate::search::RamenSearchOutput {
    /// 把一次搜索的结果导出成教师样本
    ///
    /// 只打包原始素材：定长特征、元信息、按 rollout 序号对齐的候选分。
    /// 不算 policy / value 标签——那些是按某组 `radical_factor` / `policy_delta`
    /// 离线生成的可再生 sidecar。
    ///
    /// 必须先开启 [`crate::search::SearchConfig::record_ordered_rollouts`]，
    /// 不能用直方图回填；直方图已经丢掉跨候选的 CRN 配对顺序。
    ///
    /// # 错误
    ///
    /// - `ordered_rollouts` 为 `None`（未开 `SearchConfig::record_ordered_rollouts`）
    /// - `per_candidate` 长度与 `actions` 不一致
    /// - 特征编码、格位分派、候选构造或样本校验失败
    pub fn export_ramen_sample(&self, game: &RamenGame, stage: &RamenStage, index: u64) -> Result<RamenTrainingSample> {
        let ordered = self.ordered_rollouts.as_ref().ok_or_else(|| {
            anyhow!("未记录有序 rollout：请开启 SearchConfig::record_ordered_rollouts 后再导出教师样本")
        })?;
        ensure!(
            ordered.per_candidate.len() == self.actions.len(),
            "ordered_rollouts.per_candidate 长度 {} 与 actions 长度 {} 不一致",
            ordered.per_candidate.len(),
            self.actions.len()
        );

        let features = super::features::encode(game)?;
        let mut candidates = Vec::with_capacity(self.actions.len());
        for (action, rollouts) in self.actions.iter().zip(ordered.per_candidate.iter()) {
            let slots = super::policy_schema::slots_of(stage.clone(), action)?;
            candidates.push(RamenCandidate::from_rollouts(slots, rollouts)?);
        }

        let meta = RamenSampleMeta::new(index, game.turn(), stage, ordered.root_seed)?;
        RamenTrainingSample::new(meta, features, candidates)
    }
}

// ============================================================================
// 批次落盘（pilot 用，格式未冻结）
// ============================================================================

/// 样本批次
///
/// ⚠ 这里的 bincode 落盘是 **pilot 期的临时手段**，正式教师数据要用 mmap 友好的
/// CSR 布局（features / offsets / candidates 分文件）。不要有代码依赖本结构的字节布局。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RamenSampleBatch {
    /// 样本列表
    pub samples: Vec<RamenTrainingSample>
}

impl RamenSampleBatch {
    /// 新建空批次
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一条样本
    pub fn push(&mut self, sample: RamenTrainingSample) {
        self.samples.push(sample);
    }

    /// 样本条数
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 写入二进制文件
    ///
    /// # 错误
    ///
    /// 创建文件或序列化失败时报错。
    pub fn save_binary(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("创建样本文件失败: {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).with_context(|| format!("序列化样本失败: {}", path.display()))?;
        Ok(())
    }

    /// 读取二进制文件
    ///
    /// # 错误
    ///
    /// 打开文件、反序列化、或版本不匹配时报错。
    pub fn load_binary(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("打开样本文件失败: {}", path.display()))?;
        let batch: Self = bincode::deserialize_from(BufReader::new(file))
            .with_context(|| format!("反序列化样本失败: {}", path.display()))?;
        for (i, s) in batch.samples.iter().enumerate() {
            ensure!(
                s.format_version == SAMPLE_FORMAT_VERSION,
                "样本 {i} 的格式版本是 {}，当前是 {SAMPLE_FORMAT_VERSION}",
                s.format_version
            );
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{Checks, init_test_logger};

    /// 造一组确定性的假分数（不依赖 rng，测试之间可复现）
    fn fake_scores(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                50000.0 + (s >> 40) as f64 / 100.0
            })
            .collect()
    }

    /// 阶段编码与 [`RamenStage`] 互为逆，且拒绝非决策点
    #[test]
    fn test_stage_code_roundtrip() -> Result<()> {
        let _ = init_test_logger("error");
        let stages = [
            RamenStage::RamenSelect,
            RamenStage::SpecialSelect,
            RamenStage::Train,
            RamenStage::SuperRamenSelect,
            RamenStage::RegionSelect
        ];
        let mut c = Checks::new();
        let mut codes: Vec<u8> = Vec::new();
        for st in stages.iter() {
            let code = stage_code(st)?;
            println!("  {st:?} -> {code}");
            codes.push(code);
            c.check(&stage_of_code(code)? == st, &format!("{st:?} 编码可逆"));
        }
        codes.sort_unstable();
        codes.dedup();
        c.check(codes.len() == stages.len(), "五个决策点阶段编码互异");
        c.check(stage_code(&RamenStage::Begin).is_err(), "非决策点阶段被拒绝");
        c.check(stage_of_code(5).is_err(), "未定义编码被拒绝");
        c.finish()
    }

    /// 位图的尾部未用位必须清零，否则 `count_valid` 会多算
    #[test]
    fn test_valid_mask_tail_bits() -> Result<()> {
        let _ = init_test_logger("error");
        let mut c = Checks::new();
        for len in [1usize, 63, 64, 65, 100, 1024] {
            let mut m = ValidMask::all_valid(len);
            let full = m.count_valid();
            m.set(0, false)?;
            m.set(len - 1, false)?;
            let after = m.count_valid();
            println!("  len={len:4} 全有效={full:4} 去掉首尾后={after:4}");
            c.check(full == len, &format!("len={len} 全有效计数等于长度"));
            c.check(
                after == if len == 1 { 0 } else { len - 2 },
                &format!("len={len} 置无效后计数正确")
            );
            c.check(!m.is_valid(len - 1) && !m.is_valid(len), &format!("len={len} 越界与失败位均为无效"));
            c.check(m.check().is_ok(), &format!("len={len} 位图自洽"));
        }
        let mut m = ValidMask::all_valid(8);
        c.check(m.set(8, false).is_err(), "越界置位被拒绝");
        c.finish()
    }

    /// 候选统计必须与 [`ActionResult`] 逐位同口径
    ///
    /// 这是本容器唯一会被拿来和历史基线对比的地方：均值/标准差若与搜索层算得不一样，
    /// 离线标签与线上搜索就不是同一把尺子。
    #[test]
    fn test_candidate_stats_match_action_result() -> Result<()> {
        use crate::search::ActionResult;

        let _ = init_test_logger("error");
        let raw = fake_scores(256, 0xC0FFEE);
        // 第 3、100、255 次 rollout 记为失败
        let failed = [3usize, 100, 255];
        let rollouts: Vec<Option<f64>> = raw
            .iter()
            .enumerate()
            .map(|(i, v)| (!failed.contains(&i)).then_some(*v))
            .collect();

        let cand = RamenCandidate::from_rollouts(PolicySlots::One(7), &rollouts)?;
        let mut ar = ActionResult::new();
        for r in rollouts.iter().flatten() {
            ar.add(*r);
        }

        println!(
            "  槽位 {} 成功 {} 失败 {}；mean={:.6} / {:.6}，stdev={:.6} / {:.6}",
            cand.rollouts(),
            cand.n,
            cand.failed(),
            cand.mean(),
            ar.mean(),
            cand.stdev(),
            ar.stdev()
        );
        let mut c = Checks::new();
        c.check(cand.rollouts() == 256, "槽位数等于 rollout 次数（含失败）");
        c.check(cand.n as usize == 253 && cand.failed() == 3, "成功/失败计数正确");
        c.check((cand.mean() - ar.mean()).abs() < 1e-9, "均值与 ActionResult 一致");
        c.check((cand.stdev() - ar.stdev()).abs() < 1e-9, "标准差与 ActionResult 一致");
        for &i in failed.iter() {
            c.check(!cand.valid.is_valid(i) && cand.scores[i] == 0.0, &format!("第 {i} 次为失败槽位"));
        }
        c.check(cand.valid.is_valid(4), "未失败的序号仍有效");
        // 存 f32 不影响统计，但槽位本身允许有相对误差
        let drift = (cand.scores[4] as f64 - raw[4]).abs() / raw[4];
        println!("  f32 槽位相对误差 {drift:.3e}");
        c.check(drift < 1e-6, "f32 槽位相对误差在 1e-6 内");
        c.check(cand.check().is_ok(), "候选自洽");
        c.check(
            RamenCandidate::from_rollouts(PolicySlots::One(POLICY_DIM), &rollouts).is_err(),
            "越界格位被拒绝"
        );
        c.check(
            RamenCandidate::from_rollouts(PolicySlots::One(0), &[Some(f64::NAN)]).is_err(),
            "非有限分数被拒绝"
        );
        c.check(RamenCandidate::from_rollouts(PolicySlots::One(0), &[]).is_err(), "空 rollout 被拒绝");
        c.finish()
    }

    /// 样本构造必须拦下会静默污染数据集的四类输入
    #[test]
    fn test_sample_rejects_bad_input() -> Result<()> {
        let _ = init_test_logger("error");
        let meta = RamenSampleMeta::new(42, 23, &RamenStage::RegionSelect, 0xABCD)?;
        let ok_cand = |slot: usize, n: usize| -> Result<RamenCandidate> {
            let r: Vec<Option<f64>> = fake_scores(n, slot as u64 + 1).into_iter().map(Some).collect();
            RamenCandidate::from_rollouts(PolicySlots::One(slot), &r)
        };
        let feats = vec![0.5f32; INPUT_DIM];
        let mut c = Checks::new();

        let good = RamenTrainingSample::new(meta, feats.clone(), vec![ok_cand(1, 16)?, ok_cand(2, 16)?])?;
        println!("  正常样本：{} 候选 × {} rollout", good.candidates.len(), good.rollouts());
        c.check(good.format_version == SAMPLE_FORMAT_VERSION, "版本号写入正确");
        c.check(good.meta.stage()? == RamenStage::RegionSelect, "阶段可还原");
        let mask = good.legal_mask();
        println!("  合法掩码置 1 的格位数 {}", mask.iter().filter(|v| **v > 0.0).count());
        c.check(mask.len() == POLICY_DIM, "掩码维度为 POLICY_DIM");
        c.check(mask[1] == 1.0 && mask[2] == 1.0 && mask[0] == 0.0, "掩码只在候选格位上置 1");

        c.check(
            RamenTrainingSample::new(meta, vec![0.0; INPUT_DIM - 1], vec![ok_cand(1, 16)?]).is_err(),
            "特征维度不符被拒绝"
        );
        let mut nan_feats = feats.clone();
        nan_feats[100] = f32::NAN;
        c.check(
            RamenTrainingSample::new(meta, nan_feats, vec![ok_cand(1, 16)?]).is_err(),
            "特征含 NaN 被拒绝"
        );
        c.check(
            RamenTrainingSample::new(meta, feats.clone(), vec![]).is_err(),
            "空候选表被拒绝"
        );
        c.check(
            RamenTrainingSample::new(meta, feats.clone(), vec![ok_cand(1, 16)?, ok_cand(2, 15)?]).is_err(),
            "候选间 rollout 数不一致被拒绝——CRN 配对要求逐次对齐"
        );

        let mut broken = ok_cand(3, 16)?;
        broken.n += 1;
        c.check(
            RamenTrainingSample::new(meta, feats, vec![broken]).is_err(),
            "n 与位图不符被拒绝"
        );
        c.finish()
    }

    /// 批次落盘可往返（pilot 用，非冻结格式）
    #[test]
    fn test_batch_binary_roundtrip() -> Result<()> {
        use crate::utils::get_workspace_root;

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let meta = RamenSampleMeta::new(7, 11, &RamenStage::Train, 0x1234_5678)?;
        let r: Vec<Option<f64>> = fake_scores(32, 9).into_iter().map(Some).collect();
        let mut batch = RamenSampleBatch::new();
        batch.push(RamenTrainingSample::new(
            meta,
            vec![0.25f32; INPUT_DIM],
            vec![
                RamenCandidate::from_rollouts(PolicySlots::One(201), &r)?,
                RamenCandidate::from_rollouts(PolicySlots::Three([214, 215, 216]), &r)?
            ]
        )?);

        let dir = get_workspace_root()?.join("target").join("tmp");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("ramen_sample_roundtrip.bin");
        batch.save_binary(&path)?;
        let size = std::fs::metadata(&path)?.len();
        let back = RamenSampleBatch::load_binary(&path)?;
        let _ = std::fs::remove_file(&path);

        let a = &batch.samples[0];
        let b = &back.samples[0];
        println!("  落盘 {size} 字节，{} 条样本", back.len());
        let mut c = Checks::new();
        c.check(back.len() == 1, "样本条数一致");
        c.check(b.meta == a.meta, "元信息一致");
        c.check(b.features == a.features, "特征逐位一致");
        c.check(b.candidates.len() == 2, "候选个数一致");
        c.check(b.candidates[0].scores == a.candidates[0].scores, "rollout 分逐位一致");
        c.check(b.candidates[0].valid == a.candidates[0].valid, "位图一致");
        c.check(
            b.candidates[1].slots == PolicySlots::Three([214, 215, 216]),
            "地区选择的三格位一致"
        );
        c.check(b.candidates[1].sum == a.candidates[1].sum, "sum 逐位一致");
        c.finish()
    }

    /// 用真实局面的真实候选装样本
    ///
    /// 前面几个测试都只拿本模块的常量自洽验证；这个测试把
    /// [`crate::sampler`] 抽到的真实决策点、[`super::features::encode`] 的真实特征、
    /// [`super::policy_schema::slots_of`] 的真实格位串起来走一遍，
    /// 防的是「容器自身没问题、但接不上生产数据」。
    #[test]
    fn test_build_sample_from_real_position() -> Result<()> {
        use crate::{
            game::ramen::{features::encode, policy_schema::slots_of},
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
        let mut built = 0usize;
        let mut bad: Vec<String> = Vec::new();

        for index in 0..40u64 {
            let Some(pos) = sample_position(&space, &cfg, index)?.into_captured() else {
                continue;
            };
            let feats = encode(&pos.game)?;
            let mut cands = Vec::new();
            for (i, act) in pos.actions.iter().enumerate() {
                let slots = slots_of(pos.stage.clone(), act)?;
                let r: Vec<Option<f64>> = fake_scores(8, index * 1000 + i as u64)
                    .into_iter()
                    .enumerate()
                    // 让第 0 号候选的第 5 次 rollout 失败，走一遍失败路径
                    .map(|(k, v)| (!(i == 0 && k == 5)).then_some(v))
                    .collect();
                cands.push(RamenCandidate::from_rollouts(slots, &r)?);
            }
            match RamenTrainingSample::new(
                RamenSampleMeta::new(index, pos.turn, &pos.stage, pos.spec.seed)?,
                feats,
                cands
            ) {
                Ok(s) => {
                    built += 1;
                    if built == 1 {
                        println!(
                            "  首条：index={} turn={} stage={:?} 候选 {} 个，掩码置 1 {} 格",
                            s.meta.index,
                            s.meta.turn,
                            s.meta.stage()?,
                            s.candidates.len(),
                            s.legal_mask().iter().filter(|v| **v > 0.0).count()
                        );
                    }
                    if s.candidates[0].failed() != 1 {
                        bad.push(format!("index={index} 失败槽位没被记下"));
                    }
                }
                Err(e) => bad.push(format!("index={index} 装样本失败: {e}"))
            }
        }

        println!("  40 次采样共装出 {built} 条样本，异常 {} 条", bad.len());
        for line in bad.iter().take(10) {
            println!("  ⚠ {line}");
        }
        c.check(built > 0, "至少装出一条样本");
        c.check(bad.is_empty(), "真实局面全部能装进容器");
        c.finish()
    }

    /// 真实搜索走通 `search -> export_ramen_sample`
    ///
    /// 必须开启 `record_ordered_rollouts` 才能导出；关开关时应直接报错，
    /// 不能退化成用直方图凑一份（那会丢掉 CRN 配对顺序）。
    #[test]
    fn test_export_ramen_sample_from_real_search() -> Result<()> {
        use crate::{
            gamedata::init_global,
            sampler::{SamplerConfig, SamplingSpace, sample_position},
            search::{FlatSearch, SearchConfig},
            utils::get_workspace_root
        };

        std::env::set_current_dir(get_workspace_root()?)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let space = SamplingSpace::gen1()?;
        let cfg = SamplerConfig::default();
        let search_n = 8;
        let search: FlatSearch<RamenGame> = FlatSearch::new(
            SearchConfig::default()
                .with_search_n(search_n)
                .with_ucb(false)
                .with_record_ordered_rollouts(true)
        );
        let search_off: FlatSearch<RamenGame> = FlatSearch::new(
            SearchConfig::default().with_search_n(search_n).with_ucb(false)
        );

        let mut picked = None;
        for index in 0..40u64 {
            let Some(pos) = sample_position(&space, &cfg, index)?.into_captured() else {
                continue;
            };
            if pos.actions.len() > 16 {
                println!(
                    "  index={index} stage={:?} 候选 {} 个，跳过以免测试过慢",
                    pos.stage,
                    pos.actions.len()
                );
                continue;
            }
            picked = Some((index, pos));
            break;
        }
        let (index, pos) = picked.ok_or_else(|| anyhow!("40 次采样未抽到合适的决策点"))?;
        println!(
            "  抽到 index={index} turn={} stage={:?} 候选 {} 个",
            pos.turn,
            pos.stage,
            pos.actions.len()
        );

        let mut rng = pos.decision_rng.clone();
        let output = search.search(&pos.game, &pos.actions, &mut rng)?;
        let sample = output.export_ramen_sample(&pos.game, &pos.stage, index)?;

        let ones = sample.legal_mask().iter().filter(|v| **v > 0.0).count();
        println!(
            "  导出：{} 候选 × {} rollout，掩码置 1 {} 格，root_seed={:#018x}",
            sample.candidates.len(),
            sample.rollouts(),
            ones,
            sample.meta.root_seed
        );

        let mut c = Checks::new();
        c.check(
            sample.candidates.len() == output.actions.len(),
            "候选数等于 actions.len()"
        );
        c.check(
            sample.candidates.iter().all(|x| x.rollouts() == search_n),
            "每候选 rollout 槽位数等于 search_n"
        );
        c.check(ones > 0 && ones <= POLICY_DIM, "合法掩码置 1 的格位数合理");
        c.check(sample.meta.stage()? == pos.stage, "meta.stage 与传入阶段一致");
        c.check(sample.meta.index == index, "meta.index 与采样序号一致");
        c.check(sample.meta.turn == pos.turn, "meta.turn 与局面回合一致");
        let root = output.ordered_rollouts.as_ref().map(|o| o.root_seed);
        c.check(root == Some(sample.meta.root_seed), "meta.root_seed 取自 ordered_rollouts");
        c.check(sample.features.len() == INPUT_DIM, "特征维度为 INPUT_DIM");

        let mut rng_off = pos.decision_rng.clone();
        let out_off = search_off.search(&pos.game, &pos.actions, &mut rng_off)?;
        match out_off.export_ramen_sample(&pos.game, &pos.stage, index) {
            Ok(_) => c.check(false, "开关关闭时应返回 Err"),
            Err(e) => {
                println!("  开关关闭错误: {e}");
                c.check(
                    e.to_string().contains("record_ordered_rollouts"),
                    "错误信息提示开启 SearchConfig::record_ordered_rollouts"
                );
            }
        }
        c.finish()
    }
}
