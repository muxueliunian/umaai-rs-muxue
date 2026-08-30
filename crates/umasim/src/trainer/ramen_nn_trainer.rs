//! 拉面杯神经网络训练员
//!
//! 把已训练的 ONNX 模型接到 [`Trainer<RamenGame>`]：编码 754 维特征、跑模型、
//! 按冻结的 policy 格位表给当前候选打分并 argmax。choice 头未训练，事件选项
//! 委托给 [`RecommendedRamenTrainer`]。
//!
//! 本模块仅在 `onnx` feature 下编译。

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow, bail, ensure};
use rand::rngs::StdRng;
use serde::Deserialize;
use tract_onnx::prelude::*;

use crate::{
    game::{
        Trainer,
        ramen::{
            RamenAction, RamenGame, RamenStage,
            features::{self, encode},
            policy::free_race_gate_index,
            policy_schema::{POLICY_DIM, PolicySlots, slots_of},
            rules::list_special_targets_for
        }
    },
    gamedata::{EventChoice, EventData}
};

use super::{RecommendedRamenTrainer, ramen_handwritten_trainer::ramen_effective_stage};

/// ONNX 可运行图（与温泉评估器同一套 tract 类型）
type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// choice 头宽度（未训练，推理时丢弃）
const CHOICE_DIM: usize = 8;

/// value 头宽度：mean / stdev / 高分位
const VALUE_DIM: usize = 3;

/// 自选比赛硬守门的宽裕度，与 `RamenPolicyConfig::race_gate_slack` 的正式取值一致
const RACE_GATE_SLACK: u32 = 1;

/// 模型总输出维度：policy + choice + value
const OUTPUT_DIM: usize = POLICY_DIM + CHOICE_DIM + VALUE_DIM;

/// 模型旁 JSON 的顶层字段（其余键忽略）
#[derive(Debug, Deserialize)]
struct ModelMetaJson {
    /// 特征输入维度
    input_dim: usize,
    /// 模型输出维度
    output_dim: usize,
    /// 三路价值反归一化常数
    value_normalization: ValueNormJson
}

/// JSON 里的 `value_normalization` 对象
#[derive(Debug, Deserialize)]
struct ValueNormJson {
    /// 各路中心
    center: [f64; 3],
    /// 各路尺度
    scale: [f64; 3]
}

/// 三路价值反归一化常数
///
/// 反归一化公式为 `center[i] + scale[i] * output[i]`；stdev 那一路（下标 1）
/// 再截到非负。常数必须从模型旁 JSON 读取，禁止硬编码。
#[derive(Debug, Clone, Copy)]
pub struct RamenValueNorm {
    /// `[mean, stdev, high]` 的中心
    pub center: [f64; 3],
    /// `[mean, stdev, high]` 的尺度
    pub scale: [f64; 3]
}

impl RamenValueNorm {
    /// 把模型输出的三路归一化 value 还原到分数量纲
    ///
    /// # 错误
    ///
    /// `raw` 长度不是 3、或含非有限值时报错。
    pub fn denormalize(&self, raw: &[f32]) -> Result<RamenNnValue> {
        ensure!(raw.len() == VALUE_DIM, "value 头长度应为 {VALUE_DIM}，实得 {}", raw.len());
        let mut out = [0.0f64; VALUE_DIM];
        for (i, &x) in raw.iter().enumerate() {
            ensure!(x.is_finite(), "value 头第 {i} 路不是有限值: {x}");
            out[i] = self.center[i] + self.scale[i] * f64::from(x);
        }
        Ok(RamenNnValue {
            mean: out[0],
            stdev: out[1].max(0.0),
            high: out[2]
        })
    }
}

/// 反归一化后的三路价值
#[derive(Debug, Clone, Copy)]
pub struct RamenNnValue {
    /// 期望终局分（分数量纲）
    pub mean: f64,
    /// 样本标准差（已截到非负）
    pub stdev: f64,
    /// 高分位积分（训练侧 rf=1.4）
    pub high: f64
}

/// 一次推理的切片结果
#[derive(Debug, Clone)]
pub struct RamenNnOutput {
    /// policy logits，长度恒为 [`POLICY_DIM`]
    pub policy: Vec<f32>,
    /// 反归一化后的三路价值
    pub value: RamenNnValue
}

/// 单个候选动作在 policy 头上的得分
#[derive(Debug, Clone, Copy)]
pub struct ActionLogit {
    /// 该候选在 `actions` 切片中的下标
    pub index: usize,
    /// 映射到格位后的 logit（RegionSelect 为三格之和，RamenSelect 吃面为用法 max）
    pub logit: f32
}

/// 拉面杯神经网络训练员
///
/// 模型用 [`Arc`] 共享，整进程加载一次即可；事件选项走内部的手写策略。
#[derive(Clone)]
pub struct RamenNnTrainer {
    /// 可运行 ONNX 图
    model: Arc<OnnxModel>,
    /// 从模型旁 JSON 读出的反归一化常数
    value_norm: RamenValueNorm,
    /// choice 头未训练，事件选项全部转交给手写策略
    fallback: Arc<RecommendedRamenTrainer>,
    /// 是否启用自选比赛硬守门（见 [`Self::with_race_shield`]）
    race_shield: bool
}

impl RamenNnTrainer {
    /// 从 ONNX 文件加载模型，并读取同路径旁的 `<model>.json` 反归一化常数
    ///
    /// `model_path` 为 `foo.onnx` 时，元数据路径为 `foo.onnx.json`。
    ///
    /// # 错误
    ///
    /// - 模型文件无法读取、优化或转为可运行图
    /// - 旁路 JSON 缺失、无法解析，或缺少 `input_dim` / `output_dim` / `value_normalization`
    /// - JSON 中的维度与 [`features::INPUT_DIM`] / [`POLICY_DIM`] / 245 不符
    pub fn load(model_path: &Path) -> Result<Self> {
        ensure!(model_path.is_file(), "ONNX 模型不存在: {}", model_path.display());
        let json_path = {
            let mut s = model_path.as_os_str().to_os_string();
            s.push(".json");
            std::path::PathBuf::from(s)
        };
        ensure!(json_path.is_file(), "模型元数据不存在: {}", json_path.display());

        let meta_text = std::fs::read_to_string(&json_path)
            .with_context(|| format!("读取模型元数据失败: {}", json_path.display()))?;
        let meta: ModelMetaJson = serde_json::from_str(&meta_text)
            .with_context(|| format!("解析模型元数据失败: {}", json_path.display()))?;

        ensure!(
            meta.input_dim == features::INPUT_DIM,
            "模型 JSON input_dim={}，与特征编码 INPUT_DIM={} 不符",
            meta.input_dim,
            features::INPUT_DIM
        );
        ensure!(
            POLICY_DIM == 234,
            "policy_schema::POLICY_DIM 已变为 {POLICY_DIM}，与冻结契约 234 不符"
        );
        ensure!(
            meta.output_dim == OUTPUT_DIM,
            "模型 JSON output_dim={}，与契约 {OUTPUT_DIM}（policy {POLICY_DIM} + choice {CHOICE_DIM} + value {VALUE_DIM}）不符",
            meta.output_dim
        );
        for (i, &x) in meta.value_normalization.center.iter().enumerate() {
            ensure!(x.is_finite(), "value_normalization.center[{i}] 不是有限值: {x}");
        }
        for (i, &x) in meta.value_normalization.scale.iter().enumerate() {
            ensure!(x.is_finite(), "value_normalization.scale[{i}] 不是有限值: {x}");
            ensure!(x != 0.0, "value_normalization.scale[{i}] 为 0，无法反归一化");
        }

        log::info!("加载拉面杯 ONNX 模型: {}", model_path.display());
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("无法读取 ONNX 模型文件")?
            .into_optimized()
            .context("模型优化失败")?
            .into_runnable()
            .context("模型转换失败")?;
        log::info!("拉面杯 ONNX 模型加载成功");

        Ok(Self {
            model: Arc::new(model),
            value_norm: RamenValueNorm {
                center: meta.value_normalization.center,
                scale: meta.value_normalization.scale
            },
            fallback: Arc::new(RecommendedRamenTrainer::for_rollout()),
            race_shield: true
        })
    }

    /// 编码局面并跑一次推理
    ///
    /// # 错误
    ///
    /// 特征编码失败、输入输出维度不符、或 tract 推理失败时报错。
    pub fn infer(&self, game: &RamenGame) -> Result<RamenNnOutput> {
        let features = encode(game)?;
        ensure!(
            features.len() == features::INPUT_DIM,
            "特征长度 {} 与 INPUT_DIM={} 不符",
            features.len(),
            features::INPUT_DIM
        );
        let input = tract_ndarray::Array2::from_shape_vec((1, features::INPUT_DIM), features)
            .context("创建输入张量失败")?;
        let output = self.model.run(tvec!(input.into_tvalue())).context("推理失败")?;
        let output_tensor = output[0].to_array_view::<f32>().context("提取输出张量失败")?;
        let raw: Vec<f32> = output_tensor.iter().copied().collect();
        ensure!(
            raw.len() == OUTPUT_DIM,
            "模型输出长度 {} 与契约 {OUTPUT_DIM} 不符",
            raw.len()
        );
        let policy = raw[..POLICY_DIM].to_vec();
        let value = self.value_norm.denormalize(&raw[POLICY_DIM + CHOICE_DIM..])?;
        Ok(RamenNnOutput { policy, value })
    }

    /// 开关自选比赛硬守门（默认开启）
    ///
    /// 守门只在 `Train` 阶段生效：区间内剩余可比赛回合已不够补齐缺口时，无视 policy
    /// logit 直接选「比赛」。这不是价值权衡而是硬性义务——自选比赛不达标由
    /// `BaseGame::check_free_race` 直接判定育成失败，且教师数据在此处几乎没有信号
    /// （整个 12k 样本里 `remain == need` 的严格截止局面只有 3 条），网络无法从中学到
    /// 接近 100% 可靠的规则。判定逻辑与手写策略共用 [`free_race_gate_index`]。
    ///
    /// 关闭后为**纯网络**策略，仅供研究「守门能否移除」，不可用于生产验收。
    pub fn with_race_shield(mut self, on: bool) -> Self {
        self.race_shield = on;
        self
    }

    /// 按当前阶段把每个候选映射到 policy logit
    ///
    /// # 错误
    ///
    /// 任一候选无法落格、格位越界、或阶段不是决策点时报错——不静默跳过。
    pub fn score_actions(
        &self, game: &RamenGame, actions: &[RamenAction], policy: &[f32]
    ) -> Result<Vec<ActionLogit>> {
        ensure!(
            policy.len() == POLICY_DIM,
            "policy 长度 {} 与 POLICY_DIM={POLICY_DIM} 不符",
            policy.len()
        );
        let stage = ramen_effective_stage(game, actions);
        let mut out = Vec::with_capacity(actions.len());
        for (index, action) in actions.iter().enumerate() {
            let logit = score_one(game, stage.clone(), action, policy)?;
            ensure!(logit.is_finite(), "候选 {index} 的 logit 不是有限值: {logit}");
            out.push(ActionLogit { index, logit });
        }
        Ok(out)
    }
}

/// 从 policy 向量取一个格位的 logit
///
/// # 错误
///
/// 格位越界时报错。
fn logit_at(policy: &[f32], slot: usize) -> Result<f32> {
    policy
        .get(slot)
        .copied()
        .ok_or_else(|| anyhow!("格位 {slot} 越出 policy 长度 {}", policy.len()))
}

/// 把 `slots_of` 的单格结果读成 logit
///
/// # 错误
///
/// 得到三格、或格位越界时报错。
fn one_slot_logit(stage: RamenStage, action: &RamenAction, policy: &[f32]) -> Result<f32> {
    match slots_of(stage.clone(), action)? {
        PolicySlots::One(i) => logit_at(policy, i),
        PolicySlots::Three(a) => bail!("阶段 {stage:?} 期望单格，得到三格 {a:?}")
    }
}

/// 给一个候选打分
///
/// # 错误
///
/// 阶段/动作无法落格，或 RamenSelect 某碗面没有合法风味用法时报错。
fn score_one(game: &RamenGame, stage: RamenStage, action: &RamenAction, policy: &[f32]) -> Result<f32> {
    match stage {
        RamenStage::Train | RamenStage::SuperRamenSelect => one_slot_logit(stage, action, policy),
        RamenStage::RegionSelect => match slots_of(RamenStage::RegionSelect, action)? {
            PolicySlots::Three(ids) => {
                let mut sum = 0.0f32;
                for slot in ids {
                    sum += logit_at(policy, slot)?;
                }
                Ok(sum)
            }
            PolicySlots::One(i) => bail!("RegionSelect 期望三格，得到单格 {i}")
        },
        RamenStage::RamenSelect => match action.ramen {
            None => one_slot_logit(RamenStage::RamenSelect, action, policy),
            Some(rid) => {
                let targets = list_special_targets_for(&game.ramen, rid)?;
                ensure!(
                    !targets.is_empty(),
                    "地区 {rid} 没有合法风味用法，无法给 RamenSelect 候选打分"
                );
                let mut best = f32::NEG_INFINITY;
                for t in targets {
                    let combined = RamenAction::combined_select(Some(rid), t);
                    let s = one_slot_logit(RamenStage::RamenSelect, &combined, policy)?;
                    if s > best {
                        best = s;
                    }
                }
                Ok(best)
            }
        },
        RamenStage::SpecialSelect => {
            let region = game
                .ramen
                .pending_ramen
                .ok_or_else(|| anyhow!("SpecialSelect 阶段 pending_ramen 为空"))?;
            match action.ramen {
                Some(r) => ensure!(
                    r == region,
                    "SpecialSelect 候选面 {r} 与 pending_ramen {region} 不一致"
                ),
                None => bail!("SpecialSelect 候选 ramen 为空")
            }
            let targets = action
                .special_targets
                .ok_or_else(|| anyhow!("SpecialSelect 候选缺少 special_targets"))?;
            let combined = RamenAction::combined_select(Some(region), targets);
            one_slot_logit(RamenStage::RamenSelect, &combined, policy)
        }
        other => bail!("阶段 {other:?} 不是可映射的决策点")
    }
}

/// 在已打分的候选里取 logit 最大者；并列取更小下标
///
/// # 错误
///
/// 候选为空时报错。
fn argmax_logit(scores: &[ActionLogit]) -> Result<usize> {
    let best = scores
        .iter()
        .max_by(|a, b| match a.logit.total_cmp(&b.logit) {
            std::cmp::Ordering::Equal => b.index.cmp(&a.index),
            other => other
        })
        .ok_or_else(|| anyhow!("候选动作为空，无法 argmax"))?;
    Ok(best.index)
}

impl Trainer<RamenGame> for RamenNnTrainer {
    /// 编码局面、跑模型，按格位映射 argmax 选动作
    ///
    /// # 错误
    ///
    /// 推理失败、任一候选无法落格、或候选为空时报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], _rng: &mut StdRng) -> Result<usize> {
        ensure!(!actions.is_empty(), "候选动作为空");
        // 自选比赛硬守门优先于网络输出：不达标直接育成失败，不是可权衡的价值项
        if self.race_shield && ramen_effective_stage(game, actions) == RamenStage::Train {
            if let Some(idx) = free_race_gate_index(game, actions, RACE_GATE_SLACK) {
                return Ok(idx);
            }
        }
        let out = self.infer(game)?;
        let scores = self.score_actions(game, actions, &out.policy)?;
        argmax_logit(&scores)
    }

    /// 事件选项委托给手写策略（choice 头未训练）
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.fallback.select_choice(game, choices, rng)
    }

    /// 事件选项（含友人事件特例）全部转交手写策略
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.fallback.select_event_choice(game, event, choices, rng)
    }

    fn last_breakdown(&self) -> Option<String> {
        self.fallback.last_breakdown()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, bail};
    use rand::rngs::StdRng;

    use super::*;
    use crate::{
        game::{
            Game, InheritInfo,
            ramen::{RamenGame, RamenStage}
        },
        gamedata::init_global,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: InheritInfo = InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 把开局局面推进到第一个真正的决策阶段
    ///
    /// # 错误
    ///
    /// 推进中规则层报错，或转完仍不是决策点时报错。
    fn advance_to_decision(game: &mut RamenGame, trainer: &RamenNnTrainer, rng: &mut StdRng) -> Result<()> {
        for _ in 0..16 {
            match game.stage {
                RamenStage::Train
                | RamenStage::RamenSelect
                | RamenStage::SpecialSelect
                | RamenStage::RegionSelect
                | RamenStage::SuperRamenSelect => return Ok(()),
                _ => {
                    game.run_stage(trainer, rng)?;
                    if !game.next() {
                        bail!("开局推进后游戏已结束，未到达决策阶段");
                    }
                }
            }
        }
        bail!("开局推进 16 步仍未到达决策阶段，当前 {:?}", game.stage)
    }

    /// 加载 pilot 模型，在开局第一决策点跑一次 select_action
    #[test]
    fn test_ramen_nn_select_action_opening() -> Result<()> {
        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let model_path = root.join("saved_models").join("ramen_pilot").join("model.onnx");
        println!("模型路径: {}", model_path.display());
        let trainer = RamenNnTrainer::load(&model_path)?;

        let (mut rng, rule_master) = crate::bench::seeded_rngs(42, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        advance_to_decision(&mut game, &trainer, &mut rng)?;

        let actions = game.list_actions()?;
        println!("阶段: {:?}  回合: {}  候选数: {}", game.stage, game.turn(), actions.len());
        let out = trainer.infer(&game)?;
        println!(
            "value 反归一化: mean={:.1} stdev={:.1} high={:.1}",
            out.value.mean, out.value.stdev, out.value.high
        );
        let scores = trainer.score_actions(&game, &actions, &out.policy)?;
        for s in &scores {
            println!("  候选 {:>2}  logit={:>10.4}  {}", s.index, s.logit, actions[s.index]);
        }
        let idx = trainer.select_action(&game, &actions, &mut rng)?;
        println!("选中下标: {idx}  {}", actions[idx]);

        let mut c = Checks::new();
        c.check(!actions.is_empty(), "开局决策点应有候选");
        c.check(idx < actions.len(), "选中下标在候选范围内");
        c.check(
            (50_000.0..70_000.0).contains(&out.value.mean),
            "value.mean 落在 5 万–7 万分数量纲"
        );
        c.check(out.value.stdev >= 0.0 && out.value.stdev.is_finite(), "value.stdev 非负且有限");
        c.check(scores.iter().all(|s| s.logit.is_finite()), "各候选 logit 均为有限值");
        c.finish()
    }
}
