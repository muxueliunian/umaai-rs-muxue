//! 拉面杯神经网络训练员
//!
//! 把已训练的 ONNX 模型接到 [`Trainer<RamenGame>`]：编码 754 维特征、跑模型、
//! 按冻结的 policy 格位表给当前候选打分并 argmax。choice 头未训练，事件选项
//! 委托给 [`RecommendedRamenTrainer`]。
//!
//! 本模块仅在 `onnx` feature 下编译。

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering}
    }
};

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
            policy::{RamenPolicyConfig, free_race_gate_index},
            policy_schema::{POLICY_DIM, PolicySlots, slots_of},
            rules::list_special_targets_for
        }
    },
    gamedata::{EventChoice, EventData}
};

use super::{
    RecommendedRamenTrainer, ramen_handwritten_trainer::ramen_effective_stage,
    ramen_special_root::canonical_ramen_select_root
};

/// 进程内累计的推理请求数
///
/// 用来实测**当前 CPU 搜索的聚合推理吞吐**（请求数 / 墙钟时间）。这是评估
/// 「换 GPU 批量推理能带来多少加速」的唯一合法基准量：搜索本来就在多个
/// rayon 线程上并行推理，拿 GPU 满批吞吐去比单线程 tract 微基准会高估收益。
///
/// 计数在 [`RamenNnTrainer::infer`] 里做，故与模型实例无关、整进程累计；
/// 单次推理约 1.4 ms，一次 `Relaxed` 自增的代价可以忽略。
static INFER_REQUESTS: AtomicU64 = AtomicU64::new(0);

/// 读取进程内累计的推理请求数
pub fn infer_request_count() -> u64 {
    INFER_REQUESTS.load(Ordering::Relaxed)
}

/// ONNX 可运行图（与温泉评估器同一套 tract 类型）
type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// choice 头宽度（未训练，推理时丢弃）
const CHOICE_DIM: usize = 8;

/// value 头宽度：mean / stdev / 高分位
const VALUE_DIM: usize = 3;

/// 自选比赛硬守门的宽裕度
///
/// 单源取自 [`RamenPolicyConfig::race_gate_slack`] 的正式默认值，**不再硬编码**：
/// 手写策略与网络策略必须用同一个宽裕度，否则日后调这个参数时两边会静默分叉。
/// 该 `Default` 全为常量字面量，release 下会被常量折叠。
fn race_gate_slack() -> u32 {
    RamenPolicyConfig::default().race_gate_slack
}

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

/// `SpecialSelect` 阶段的推理口径
///
/// 教师在 `RamenSelect` 根上搜的是联合动作（面 × 隐藏风味用法），policy 格位
/// `[1,201)` 也是联合格，而训练集里 `SpecialSelect` 阶段样本数为 **0**。
/// 真实对局却把决策拆成两拍，第二拍的阶段 one-hot 在全部训练样本里恒为 0。
/// 本枚举把「第二拍读哪个状态」做成显式对照，便于量测该错位值多少分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialSelectMode {
    /// 直接用 `SpecialSelect` 局面推理（存在训练—部署语义错位）
    Raw,
    /// 先经 [`canonical_ramen_select_root`] 还原到联合决策根再推理
    Canonical,
    /// 该阶段整个交给手写策略（对照组，用于给该阶段的可恢复上限定界）
    Handwritten
}

/// 把 ONNX 文件编译成**固定 batch** 的可运行图
///
/// 导出的模型第 0 维是符号 `batch`，tract 在维度未知时拿不到形状特化，
/// 优化后的图明显更慢。实测（`ens_d3`，本机单线程）：
///
/// | 输入形状 | µs/次 |
/// |---|---|
/// | 符号 `['batch', 754]` | 2647 |
/// | 固定 `[1, 754]` | 1447 |
///
/// 白拿 1.83×，且**输出逐位不变**（见 `test_fixed_shape_matches_symbolic`）。
///
/// # 错误
///
/// 文件读不出、输入形状与 [`features::INPUT_DIM`] 不符、优化或转换失败，
/// 或**图的输出契约**与 [`validate_output_contract`] 不符时报错。
fn build_runnable(model_path: &Path, batch: usize) -> Result<OnnxModel> {
    ensure!(batch >= 1, "batch 必须 >= 1，实得 {batch}");
    let plan = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("无法读取 ONNX 模型文件")?
        .with_input_fact(0, f32::fact([batch, features::INPUT_DIM]).into())
        .context("固定输入形状失败")?
        .into_optimized()
        .context("模型优化失败")?
        .into_runnable()
        .context("模型转换失败")?;
    validate_output_contract(&plan, batch)
        .with_context(|| format!("ONNX 图输出契约校验失败: {}", model_path.display()))?;
    Ok(plan)
}

/// 校验**图本身**的输出契约：单输出、f32、形状 `[batch, OUTPUT_DIM]`
///
/// 旁车 JSON 里的 `input_dim` / `output_dim` 是**模型作者的声明**，与图实际长什么样
/// 是两件事：旁车写对、图导错（少一个头、多一个输出、dtype 不是 f32）时，旧实现要到
/// **第一次真实决策**才在 `infer_features` 的长度检查里炸。那时候一局已经跑了一半，
/// 客户端也已经对用户宣称网络生效。这里把它提到加载阶段。
///
/// 只读 tract 优化后图的 fact，**不跑任何推理**——因此不会额外计进
/// [`infer_request_count`]，也不会污染「整局恰好 3 次推理」这条隔离证据。
///
/// 输入形状已由 [`build_runnable`] 固定成 `[batch, INPUT_DIM]`，因此这里的输出形状
/// 必然是具体值；拿不到具体值本身就说明图没能被特化，同样报错。
///
/// # 错误
///
/// 输出个数不为 1、元素类型不是 f32、形状不是 `[batch, OUTPUT_DIM]`，或输出 fact
/// 读不出 / 不是具体形状时报错。
fn validate_output_contract(plan: &OnnxModel, batch: usize) -> Result<()> {
    let graph = plan.model();
    ensure!(
        graph.outputs.len() == 1,
        "ONNX 图输出个数为 {}，契约要求恰好 1 个（policy {POLICY_DIM} + choice {CHOICE_DIM} + value {VALUE_DIM} 拼成的单张量）",
        graph.outputs.len()
    );
    let fact = graph.output_fact(0).context("读取 ONNX 图输出 fact 失败")?;
    ensure!(
        fact.datum_type == f32::datum_type(),
        "ONNX 图输出元素类型为 {:?}，契约要求 f32",
        fact.datum_type
    );
    let shape = fact
        .shape
        .as_concrete()
        .ok_or_else(|| anyhow!("ONNX 图输出形状未特化为具体值: {:?}", fact.shape))?;
    ensure!(
        shape == [batch, OUTPUT_DIM],
        "ONNX 图输出形状为 {shape:?}，契约要求 [{batch}, {OUTPUT_DIM}]"
    );
    Ok(())
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
    race_shield: bool,
    /// `SpecialSelect` 阶段的推理口径（见 [`Self::with_special_mode`]）
    special_mode: SpecialSelectMode
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
        let model = build_runnable(model_path, 1)?;
        log::info!("拉面杯 ONNX 模型加载成功");

        Ok(Self {
            model: Arc::new(model),
            value_norm: RamenValueNorm {
                center: meta.value_normalization.center,
                scale: meta.value_normalization.scale
            },
            fallback: Arc::new(RecommendedRamenTrainer::for_rollout()),
            race_shield: true,
            special_mode: SpecialSelectMode::Canonical
        })
    }

    /// 编码局面并跑一次推理
    ///
    /// # 错误
    ///
    /// 特征编码失败、输入输出维度不符、或 tract 推理失败时报错。
    pub fn infer(&self, game: &RamenGame) -> Result<RamenNnOutput> {
        self.infer_features(encode(game)?)
    }

    /// 对**已编码**的特征跑一次推理
    ///
    /// 从 [`Self::infer`] 里拆出来：批量调度器与 CPU/GPU 对拍都需要「先拿到输入、
    /// 稍后再推理」，若各自重写一份编码就会与生产路径分叉。生产路径同样走这里，
    /// 保证三条路用的是同一份实现。
    ///
    /// # 错误
    ///
    /// 输入输出维度不符或 tract 推理失败时报错。
    pub fn infer_features(&self, features: Vec<f32>) -> Result<RamenNnOutput> {
        INFER_REQUESTS.fetch_add(1, Ordering::Relaxed);
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

    /// 本模型旁车里的三路 value 反归一化常数
    ///
    /// 存在的理由：GPU 侧车返回的是**已按成员 0 尺度重归一化**的 value，解码必须用
    /// 与 policy **同一个模型**的常数。把常数从这里取出来，批量调度器就不必自己再读
    /// 一遍旁车 JSON，也就不会与 [`Self::infer_features`] 的解码口径分叉。
    pub fn value_norm(&self) -> RamenValueNorm {
        self.value_norm
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

    /// 选择 `SpecialSelect` 阶段的推理口径（默认 [`SpecialSelectMode::Canonical`]）
    ///
    /// 默认取 `Canonical` 而非保持历史行为：`Raw` 让网络在一个**训练集中从未出现**
    /// 的阶段 one-hot 上推理（实测差异位为特征下标 8/9/140/143，其中 SpecialSelect
    /// 位在全部 55733 条样本里恒为 0），输出属外推。`Raw` 保留仅供 A/B 对照，
    /// `Handwritten` 用于给该阶段的可恢复上限定界。
    pub fn with_special_mode(mut self, mode: SpecialSelectMode) -> Self {
        self.special_mode = mode;
        self
    }

    /// 决策点的「推理前」一步：守门、口径选择、特征编码、单候选收敛
    ///
    /// 返回 [`DecisionPrep::Resolved`] 表示这一步根本不需要网络。`rng` 只在
    /// `SpecialSelect` 转交手写策略时用到。
    ///
    /// 三类不需要网络的情形：自选比赛硬守门命中、`SpecialSelect` 取
    /// [`SpecialSelectMode::Handwritten`] 口径、**候选只有一个**（见函数体末尾）。
    ///
    /// 本函数是 CPU 直连、rollout 与批量后端**共用的唯一决策准备入口**，
    /// 三条路因此不会各自漂移。
    ///
    /// # 错误
    ///
    /// 候选为空、特征编码失败、[`SpecialSelectMode::Canonical`] 下联合决策根
    /// 还原失败（阶段不对 / `pending_ramen` 为空），或单候选落格检查失败时报错。
    pub fn prepare_decision(
        &self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng
    ) -> Result<DecisionPrep> {
        Ok(self.prepare_decision_labeled(game, actions, rng)?.into())
    }

    /// 同 [`Self::prepare_decision`]，但**保留「为什么不需要推理」**
    ///
    /// [`DecisionPrep::Resolved`] 把三种完全不同的出口压成同一个下标：自选比赛硬守门、
    /// `SpecialSelect` 整阶段转手写、唯一候选收敛。客户端要在屏幕与协议上**照实**标注
    /// 来源，就不能拿一个下标反推，更不能为了标来源再跑一遍守门判定——那会变成第二份
    /// 判定逻辑。于是把判定结果原样带出来，[`Self::prepare_decision`] 退化成本函数的
    /// 一层映射，两条路因此不会漂移。
    ///
    /// # 错误
    ///
    /// 与 [`Self::prepare_decision`] 完全一致：候选为空、特征编码失败、
    /// [`SpecialSelectMode::Canonical`] 下联合决策根还原失败，或单候选落格检查失败时报错。
    pub fn prepare_decision_labeled(
        &self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng
    ) -> Result<LabeledPrep> {
        ensure!(!actions.is_empty(), "候选动作为空");
        let stage = ramen_effective_stage(game, actions);
        // 自选比赛硬守门优先于网络输出：不达标直接育成失败，不是可权衡的价值项
        if self.race_shield && stage == RamenStage::Train {
            if let Some(idx) = free_race_gate_index(game, actions, race_gate_slack()) {
                return Ok(LabeledPrep::RaceGate(idx));
            }
        }
        // SpecialSelect 是联合决策的第二拍，推理状态由 special_mode 决定；
        // 候选合法性与打分一律基于**原局面**，只有喂给模型的那一份被还原
        let prep = if stage == RamenStage::SpecialSelect {
            match self.special_mode {
                SpecialSelectMode::Handwritten => {
                    LabeledPrep::HandwrittenStage(self.fallback.select_action(game, actions, rng)?)
                }
                SpecialSelectMode::Canonical => {
                    LabeledPrep::NeedsInference(encode(&canonical_ramen_select_root(game)?)?)
                }
                SpecialSelectMode::Raw => LabeledPrep::NeedsInference(encode(game)?)
            }
        } else {
            LabeledPrep::NeedsInference(encode(game)?)
        };

        // 单候选：唯一候选必然中选，policy 取任何值都改不了 argmax 的结果，
        // 于是整次网络往返可以省掉。
        // （这类请求的占比只在若干固定根上量过，**不是整局比例**，见实验记录。）
        //
        // ❗只把 [`DecisionPrep::NeedsInference`] 收敛成 `Resolved`，**不碰任何
        // 已经是 `Resolved` 的分支**：自选比赛守门在上面就地返回；`SpecialSelect`
        // 的 `Handwritten` 口径要走 `fallback.select_action`，**那条路会消耗随机流**，
        // 抢在它前面短路会改变 RNG 序列。
        //
        // ❗仍然走完上面全部守门与还原（含 `canonical_ramen_select_root` 的阶段校验
        // 与 `encode`），并用零 policy 调一次 [`Self::score_actions`] 把**候选落格**
        // 检查留住。
        //
        // 等价性的准确表述：**正常模型与合法局面下，动作与随机流不变**。
        // 它**不**保留依赖真实推理的错误行为——推理失败、模型输出非有限值等，
        // 在这条路径上不会再被触发。
        if actions.len() == 1 && matches!(prep, LabeledPrep::NeedsInference(_)) {
            self.score_actions(game, actions, &[0.0f32; POLICY_DIM])?;
            return Ok(LabeledPrep::SingleCandidate(0));
        }
        Ok(prep)
    }

    /// 决策点的「推理后」一步：按候选打分取赢家
    ///
    /// 打分基于**原局面** `game`，与 [`Self::prepare_decision`] 是否做过 canonical
    /// 还原无关——还原只作用于喂给模型的那一份输入。
    ///
    /// # 错误
    ///
    /// 任一候选无法落格、格位越界、或候选为空时报错。
    pub fn resolve_decision(&self, game: &RamenGame, actions: &[RamenAction], policy: &[f32]) -> Result<usize> {
        let scores = self.score_actions(game, actions, policy)?;
        argmax_logit(&scores)
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

/// 一个决策点在推理之前的准备结果
///
/// 把 `select_action` 的「推理前」与「推理后」切开，是批量调度的前提：调度器要
/// 在推理点挂起 rollout，就必须先能单独拿到「这一步要喂给模型的输入」，等批量
/// 推理回来再单独执行「按候选打分取赢家」。
///
/// 切口刻意放在这里而不是更深处：自选比赛守门与 `SpecialSelect` 的手写口径都是
/// **不经过网络**的分支，若让调度器自己判断这些条件，就会出现第二份判定逻辑。
#[derive(Debug, Clone)]
pub enum DecisionPrep {
    /// 无需推理即可定案：守门命中，或该阶段按配置转交手写策略
    Resolved(usize),
    /// 需要一次网络推理，`features` 是已编码好的模型输入
    NeedsInference(Vec<f32>)
}

/// 一个决策点在推理之前的准备结果，**带上「为什么」**
///
/// [`DecisionPrep`] 面向批量调度器，它只关心「要不要推理」，三种不推理的出口压成一个
/// [`DecisionPrep::Resolved`] 就够了。客户端要把来源照实写进屏幕与协议，需要的信息更细，
/// 于是由 [`RamenNnTrainer::prepare_decision_labeled`] 直接给出本枚举，
/// [`DecisionPrep`] 退化成它的一层 [`From`] 映射——判定逻辑只有一份。
#[derive(Debug, Clone)]
pub enum LabeledPrep {
    /// 自选比赛硬守门命中：无视 policy 直接选「比赛」，**没有跑推理**
    RaceGate(usize),
    /// `SpecialSelect` 整阶段按 [`SpecialSelectMode::Handwritten`] 口径转交手写策略
    ///
    /// ❗这条路会**消耗随机流**（手写策略内部取随机数），与另外两种不推理的出口不同。
    HandwrittenStage(usize),
    /// 候选只有一个：argmax 的结果与 policy 无关，整次推理省掉
    SingleCandidate(usize),
    /// 需要一次网络推理，`features` 是已编码好的模型输入
    NeedsInference(Vec<f32>)
}

impl From<LabeledPrep> for DecisionPrep {
    /// 丢掉「为什么」，只保留「要不要推理」
    fn from(value: LabeledPrep) -> Self {
        match value {
            LabeledPrep::RaceGate(i) | LabeledPrep::HandwrittenStage(i) | LabeledPrep::SingleCandidate(i) => {
                Self::Resolved(i)
            }
            LabeledPrep::NeedsInference(f) => Self::NeedsInference(f)
        }
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
    /// 推理失败、任一候选无法落格、候选为空，或 [`SpecialSelectMode::Canonical`] 下
    /// 联合决策根还原失败（阶段不对 / `pending_ramen` 为空）时报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self.prepare_decision(game, actions, rng)? {
            DecisionPrep::Resolved(idx) => Ok(idx),
            DecisionPrep::NeedsInference(features) => {
                let out = self.infer_features(features)?;
                self.resolve_decision(game, actions, &out.policy)
            }
        }
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
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        game::{
            Game, InheritInfo,
            ramen::{RamenGame, RamenStage}
        },
        gamedata::init_global,
        utils::{Checks, cleanup_test_dir, get_workspace_root, init_test_logger, unique_test_dir}
    };

    /// 最小 ONNX fixture 构造器（只够表达「输入 → 一两个输出」的图）
    ///
    /// 用途是覆盖**负向**契约：旁车 JSON 声明正确、但图本身导错。这类模型不可能
    /// 从正式权重里改出来（改 JSON 只能模拟旁车错，模拟不了图错），也不该往仓库里
    /// 塞二进制；所以按 ONNX 的 protobuf 线格式**当场生成**，完全可复现。
    ///
    /// 只用到 protobuf 的两种 wire type：varint（0）与 length-delimited（2）。
    /// 字段号取自 ONNX 的 `onnx.proto`：
    /// `ModelProto{1:ir_version, 2:producer_name, 7:graph, 8:opset_import}`、
    /// `GraphProto{1:node, 2:name, 5:initializer, 11:input, 12:output}`、
    /// `NodeProto{1:input, 2:output, 3:name, 4:op_type}`、
    /// `ValueInfoProto{1:name, 2:type}`、`TypeProto{1:tensor_type}`、
    /// `TypeProto.Tensor{1:elem_type, 2:shape}`、`TensorShapeProto{1:dim}`、
    /// `Dimension{1:dim_value}`、`TensorProto{1:dims, 2:data_type, 8:name, 9:raw_data}`、
    /// `OperatorSetIdProto{1:domain, 2:version}`。
    mod onnx_fixture {
        /// 追加一个 protobuf varint
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let b = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(b);
                    return;
                }
                out.push(b | 0x80);
            }
        }

        /// 追加一个 `字段号 + wire type` 标签
        fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
            varint(u64::from((field << 3) | wire), out);
        }

        /// 追加一个 varint 字段
        fn put_varint(field: u32, v: u64, out: &mut Vec<u8>) {
            tag(field, 0, out);
            varint(v, out);
        }

        /// 追加一个 length-delimited 字段（字符串 / 字节串 / 嵌套消息）
        fn put_bytes(field: u32, v: &[u8], out: &mut Vec<u8>) {
            tag(field, 2, out);
            varint(v.len() as u64, out);
            out.extend_from_slice(v);
        }

        /// `TypeProto`：元素类型恒为 FLOAT(1)，形状为给定的具体维度
        fn type_proto(dims: &[usize]) -> Vec<u8> {
            let mut shape = Vec::new();
            for &d in dims {
                let mut dim = Vec::new();
                put_varint(1, d as u64, &mut dim);
                put_bytes(1, &dim, &mut shape);
            }
            let mut tensor = Vec::new();
            put_varint(1, 1, &mut tensor);
            put_bytes(2, &shape, &mut tensor);
            let mut ty = Vec::new();
            put_bytes(1, &tensor, &mut ty);
            ty
        }

        /// `ValueInfoProto`
        fn value_info(name: &str, dims: &[usize]) -> Vec<u8> {
            let mut v = Vec::new();
            put_bytes(1, name.as_bytes(), &mut v);
            put_bytes(2, &type_proto(dims), &mut v);
            v
        }

        /// 单输入单输出的 `NodeProto`
        fn node(op: &str, name: &str, input: &str, output: &str) -> Vec<u8> {
            let mut n = Vec::new();
            put_bytes(1, input.as_bytes(), &mut n);
            put_bytes(2, output.as_bytes(), &mut n);
            put_bytes(3, name.as_bytes(), &mut n);
            put_bytes(4, op.as_bytes(), &mut n);
            n
        }

        /// 双输入单输出的 `NodeProto`（MatMul 用）
        fn node2(op: &str, name: &str, a: &str, b: &str, output: &str) -> Vec<u8> {
            let mut n = Vec::new();
            put_bytes(1, a.as_bytes(), &mut n);
            put_bytes(1, b.as_bytes(), &mut n);
            put_bytes(2, output.as_bytes(), &mut n);
            put_bytes(3, name.as_bytes(), &mut n);
            put_bytes(4, op.as_bytes(), &mut n);
            n
        }

        /// 全 0 的 f32 `TensorProto` initializer
        fn zero_initializer(name: &str, rows: usize, cols: usize) -> Vec<u8> {
            let mut t = Vec::new();
            put_varint(1, rows as u64, &mut t);
            put_varint(1, cols as u64, &mut t);
            put_varint(2, 1, &mut t);
            put_bytes(8, name.as_bytes(), &mut t);
            put_bytes(9, &vec![0u8; rows * cols * 4], &mut t);
            t
        }

        /// 把 `GraphProto` 包成完整 `ModelProto`
        fn wrap_model(graph: Vec<u8>) -> Vec<u8> {
            let mut opset = Vec::new();
            put_bytes(1, b"", &mut opset);
            put_varint(2, 13, &mut opset);
            let mut m = Vec::new();
            put_varint(1, 7, &mut m);
            put_bytes(2, b"umaai-test-fixture", &mut m);
            put_bytes(7, &graph, &mut m);
            put_bytes(8, &opset, &mut m);
            m
        }

        /// 输出形状 = 输入形状的图（`Identity`）：输出维度**错**的负向 fixture
        pub fn identity_model(dim: usize) -> Vec<u8> {
            let mut g = Vec::new();
            put_bytes(1, &node("Identity", "n0", "X", "Y"), &mut g);
            put_bytes(2, b"identity", &mut g);
            put_bytes(11, &value_info("X", &[1, dim]), &mut g);
            put_bytes(12, &value_info("Y", &[1, dim]), &mut g);
            wrap_model(g)
        }

        /// 两个输出的图：输出**个数**错的负向 fixture
        pub fn two_output_model(dim: usize) -> Vec<u8> {
            let mut g = Vec::new();
            put_bytes(1, &node("Identity", "n0", "X", "Y"), &mut g);
            put_bytes(1, &node("Identity", "n1", "X", "Z"), &mut g);
            put_bytes(2, b"two_outputs", &mut g);
            put_bytes(11, &value_info("X", &[1, dim]), &mut g);
            put_bytes(12, &value_info("Y", &[1, dim]), &mut g);
            put_bytes(12, &value_info("Z", &[1, dim]), &mut g);
            wrap_model(g)
        }

        /// `X[1,in] @ W[in,out]` 的图，W 全 0：输出契约**正确**的正向 fixture
        pub fn matmul_model(input_dim: usize, output_dim: usize) -> Vec<u8> {
            let mut g = Vec::new();
            put_bytes(1, &node2("MatMul", "n0", "X", "W", "Y"), &mut g);
            put_bytes(2, b"matmul", &mut g);
            put_bytes(5, &zero_initializer("W", input_dim, output_dim), &mut g);
            put_bytes(11, &value_info("X", &[1, input_dim]), &mut g);
            put_bytes(12, &value_info("Y", &[1, output_dim]), &mut g);
            wrap_model(g)
        }
    }

    /// 与冻结契约一致的旁车 JSON 文本（`input_dim` / `output_dim` 都**声明正确**）
    fn valid_sidecar_json() -> String {
        format!(
            r#"{{"input_dim":{},"output_dim":{},"value_normalization":{{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}}}"#,
            features::INPUT_DIM,
            OUTPUT_DIM
        )
    }

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: InheritInfo = InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// [`LabeledPrep`] 到 [`DecisionPrep`] 的降级映射逐项正确
    ///
    /// 三种「不需要推理」的出口必须全部落进 `Resolved` 且**下标原样带过**——映射写错
    /// 会让批量后端拿到一个不同的动作；反过来，客户端靠 `LabeledPrep` 区分来源，
    /// 三者塌成一个变体就分不出「网络算的」和「守门顶上的」。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_labeled_prep_downgrades_to_decision_prep() -> Result<()> {
        let mut c = Checks::new();
        for (labeled, want_idx) in [
            (LabeledPrep::RaceGate(3), 3usize),
            (LabeledPrep::HandwrittenStage(1), 1),
            (LabeledPrep::SingleCandidate(0), 0)
        ] {
            let tag = format!("{labeled:?}");
            let down: DecisionPrep = labeled.into();
            println!("  {tag} → {down:?}");
            c.check(
                matches!(down, DecisionPrep::Resolved(i) if i == want_idx),
                &format!("{tag} 降级成 Resolved({want_idx})")
            );
        }
        let feats = vec![0.0f32; features::INPUT_DIM];
        let down: DecisionPrep = LabeledPrep::NeedsInference(feats.clone()).into();
        c.check(
            matches!(&down, DecisionPrep::NeedsInference(f) if f.len() == feats.len()),
            "NeedsInference 原样带过，特征长度不变"
        );
        c.finish()
    }

    /// **旁车声明正确、ONNX 图输出错**时，必须在 `load` 就报错
    ///
    /// review#2：旧实现只校验旁车 JSON 自称的维度，图的输出个数 / 类型 / 形状完全
    /// 没看，错误要等到第一次真实决策才在 `infer_features` 里爆出来——那时一局已经
    /// 跑了一半、客户端也已经向用户宣称网络生效。
    ///
    /// 三个 fixture 都**当场生成**，不依赖未入库的正式权重；旁车 JSON 一律写成
    /// 与冻结契约一致的正确值，所以报错只可能来自图本身。
    ///
    /// 正向 fixture（`MatMul` 到 `[1,245]`）同时证明这条校验**不会误杀**合法图。
    #[test]
    fn test_load_rejects_wrong_graph_output_contract() -> Result<()> {
        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let mut c = Checks::new();
        let dir = unique_test_dir("nn_graph_contract")?;
        println!("fixture 目录: {}", dir.display());

        let cases: [(&str, Vec<u8>, &str); 3] = [
            (
                "wrong_dim",
                onnx_fixture::identity_model(features::INPUT_DIM),
                "形状"
            ),
            (
                "two_outputs",
                onnx_fixture::two_output_model(features::INPUT_DIM),
                "输出个数"
            ),
            (
                "good",
                onnx_fixture::matmul_model(features::INPUT_DIM, OUTPUT_DIM),
                ""
            )
        ];

        for (name, bytes, want) in cases {
            let model = dir.join(format!("{name}.onnx"));
            let sidecar = dir.join(format!("{name}.onnx.json"));
            std::fs::write(&model, &bytes)?;
            std::fs::write(&sidecar, valid_sidecar_json())?;
            let got = RamenNnTrainer::load(&model);
            match (want.is_empty(), got) {
                (false, Err(e)) => {
                    let msg = format!("{e:#}");
                    println!("{name} → {msg}");
                    c.check(msg.contains(want), &format!("{name}: 加载阶段报错且提到「{want}」"));
                }
                (false, Ok(_)) => {
                    c.check(false, &format!("{name}: 图输出不合契约却加载成功"));
                }
                (true, Ok(_)) => {
                    println!("{name} → 加载成功");
                    c.check(true, &format!("{name}: 合法图（输出 [1,{OUTPUT_DIM}]）正常加载，校验不误杀"));
                }
                (true, Err(e)) => {
                    println!("{name} → {e:#}");
                    c.check(false, &format!("{name}: 合法图被误杀"));
                }
            }
        }

        cleanup_test_dir(&dir)?;
        c.check(!dir.exists(), "本次 fixture 目录已清理（且清理前核对过在 target/test-tmp 之下）");
        c.finish()
    }

    /// 校验图输出契约时**不额外产生推理请求**
    ///
    /// 「整局恰好 3 次推理」是地区接入的隔离证据，加载期偷跑一次推理会把它污染成 4。
    #[test]
    fn test_output_contract_check_costs_no_inference() -> Result<()> {
        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let mut c = Checks::new();
        let dir = unique_test_dir("nn_graph_contract_cost")?;
        let model = dir.join("good.onnx");
        std::fs::write(&model, onnx_fixture::matmul_model(features::INPUT_DIM, OUTPUT_DIM))?;
        std::fs::write(dir.join("good.onnx.json"), valid_sidecar_json())?;

        let before = infer_request_count();
        let loaded = RamenNnTrainer::load(&model);
        let after = infer_request_count();
        println!("加载前后推理请求数: {before} → {after}");
        c.check(loaded.is_ok(), "合法 fixture 加载成功");
        c.check(after == before, "加载（含图输出契约校验）不计入任何推理请求");
        cleanup_test_dir(&dir)?;
        c.finish()
    }

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
    ///
    /// `saved_models/` 在 `.gitignore` 里，模型不随仓库分发。缺模型时本测试
    /// **跳过而不是失败**——否则任何没跑过训练管线的机器上
    /// `cargo test --features onnx` 都会红，而红的原因与代码无关。
    #[test]
    fn test_ramen_nn_select_action_opening() -> Result<()> {
        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let model_path = root.join("saved_models").join("ramen_pilot").join("model.onnx");
        println!("模型路径: {}", model_path.display());
        if !model_path.is_file() {
            println!("跳过：模型不存在（saved_models 不入库，需先跑 scripts/ramen_nn 导出）");
            return Ok(());
        }
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

    /// 单候选决策点直接定案，不再交给推理
    ///
    /// 两条断言：
    /// 1. 多候选仍返回 [`DecisionPrep::NeedsInference`]；
    /// 2. 同一局面同一阶段、候选切到只剩 1 个时返回 `Resolved(0)`，
    ///    且该路径**不消耗随机流**（用 rng 克隆体的下一个 u64 做指纹，前后一致）。
    ///
    /// ❗**本测试只证明分支返回正确，不证明省下了推理**：`prepare_decision` 改动前
    /// 本来也不执行推理，真正的省是「`Resolved` 不会再进 `infer_features`」，那要在
    /// `select_action` 或整根对拍上量。
    ///
    /// ❗[`infer_request_count`] 是**进程级全局计数**，`cargo test` 默认并行跑，
    /// 别的测试可能在读取前后递增它。故这里**只打印不断言**——把它写成确定性断言
    /// 会做出一个随并行调度变红的测试。
    ///
    /// ❗本测试钉的是「正常模型 + 合法局面下动作与随机流不变」。它**不**覆盖
    /// 「推理失败 / 模型输出非有限值」这类依赖真实推理的错误行为——单候选路径上
    /// 那些检查本就不会再触发。
    #[test]
    fn test_single_candidate_skips_inference() -> Result<()> {
        use rand::RngCore;

        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let model_path = root.join("saved_models").join("ramen_pilot").join("model.onnx");
        if !model_path.is_file() {
            println!("跳过：模型不存在（saved_models 不入库）");
            return Ok(());
        }
        let trainer = RamenNnTrainer::load(&model_path)?;

        let (mut rng, rule_master) = crate::bench::seeded_rngs(42, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        advance_to_decision(&mut game, &trainer, &mut rng)?;

        let actions = game.list_actions()?;
        println!("阶段 {:?}  回合 {}  候选数 {}", game.stage, game.turn(), actions.len());

        let mut c = Checks::new();

        // (1) 多候选：仍需推理
        if actions.len() > 1 {
            let multi = trainer.prepare_decision(&game, &actions, &mut rng)?;
            println!("多候选 prepare_decision -> {}", prep_name(&multi));
            c.check(
                matches!(multi, DecisionPrep::NeedsInference(_)),
                "多候选决策点应仍返回 NeedsInference"
            );
        } else {
            println!("本决策点只有 1 个候选，跳过多候选那一条");
        }

        // (2)(3) 单候选：直接定案 + 不动请求计数 + 不消耗随机流
        let before = infer_request_count();
        let probe_before = rng.clone().next_u64();
        let single = trainer.prepare_decision(&game, &actions[..1], &mut rng)?;
        let probe_after = rng.clone().next_u64();
        let after = infer_request_count();
        println!(
            "单候选 prepare_decision -> {}  请求计数 {} -> {}  rng 指纹 {:#018x} -> {:#018x}",
            prep_name(&single),
            before,
            after,
            probe_before,
            probe_after
        );
        c.check(
            matches!(single, DecisionPrep::Resolved(0)),
            "单候选应直接定案为 Resolved(0)"
        );
        c.check(probe_after == probe_before, "单候选路径不应消耗随机流");
        // ❗请求计数只作观察：全局静态 + 并行测试，差值不是确定量，不能断言
        println!("  （观察）全局推理请求计数 {before} -> {after}，并行下不可作断言");
        c.finish()
    }

    /// 给 [`DecisionPrep`] 一个可打印的短名（测试输出用）
    fn prep_name(p: &DecisionPrep) -> String {
        match p {
            DecisionPrep::Resolved(i) => format!("Resolved({i})"),
            DecisionPrep::NeedsInference(f) => format!("NeedsInference(特征 {} 维)", f.len())
        }
    }

    /// 固定输入形状后，输出必须与符号 batch 图**逐位一致**
    ///
    /// [`build_runnable`] 把第 0 维从符号 `batch` 钉成 1 换来 1.83× 提速，
    /// 但 tract 的形状特化会改变算子选择与融合方式。若输出哪怕只差一个 ulp，
    /// argmax 就可能在打平处翻面，**此前记录的全部网络闭环分静默作废**，
    /// 而分数上只表现为「好像有点飘」。因此这里逐位比对，不设容差。
    ///
    /// 覆盖真实轨迹上的多个局面，而不是零输入：形状特化的差异往往只在特定
    /// 数值区间显形。
    #[test]
    fn test_fixed_shape_matches_symbolic() -> Result<()> {
        use tract_ndarray::Array2;

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut c = Checks::new();
        let path = std::path::Path::new("saved_models/dagger/ens_d3.onnx");
        if !path.is_file() {
            println!("跳过：本机没有 {}", path.display());
            return c.finish();
        }

        let symbolic = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        let fixed = build_runnable(path, 1)?;

        // 沿真实轨迹取局面：用网络自己往前走，覆盖各个阶段
        let trainer = RamenNnTrainer::load(path)?;
        let mut rng = StdRng::seed_from_u64(42);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut checked = 0usize;
        let mut max_abs_diff = 0.0f32;
        let mut stages = Vec::new();
        while game.next() && checked < 24 {
            let feats = encode(&game)?;
            let input = Array2::<f32>::from_shape_vec((1, features::INPUT_DIM), feats)?;
            let a = symbolic.run(tvec!(input.clone().into_tvalue()))?;
            let b = fixed.run(tvec!(input.into_tvalue()))?;
            let va = a[0].to_array_view::<f32>()?;
            let vb = b[0].to_array_view::<f32>()?;
            for (x, y) in va.iter().zip(vb.iter()) {
                max_abs_diff = max_abs_diff.max((x - y).abs());
            }
            stages.push(format!("{:?}", game.stage));
            checked += 1;
            game.run_stage(&trainer, &mut rng)?;
        }
        println!("比对 {checked} 个局面，阶段：{stages:?}");
        println!("最大逐元素绝对差 = {max_abs_diff:e}");
        c.check(checked >= 8, "至少覆盖 8 个局面");
        c.check(max_abs_diff == 0.0, "固定形状与符号 batch 输出逐位一致");
        c.finish()
    }


    /// 推理成本微基准：符号 batch vs 固定 batch，以及 batch 规模的吞吐曲线
    ///
    /// 要回答的问题是「132× 的成本到底花在哪」：
    /// 1. **特征编码**占多少——若编码是大头，换推理后端不会有收益；
    /// 2. **符号 batch 维**代价多少——[`RamenNnTrainer::load`] 直接 `into_optimized()`
    ///    一个 `['batch', 754]` 的图，tract 在维度未知时拿不到形状特化，
    ///    固定成 `[1, 754]` 可能白拿一大截；
    /// 3. **批量的边际收益**——决定「把 512 条 rollout 改成锁步批量推理」这项
    ///    结构性改造值不值得做，以及 GPU 后端的上限在哪。
    ///
    /// 仅微基准用途，`#[ignore]` 手动执行：
    /// `cargo test --release --lib --features onnx -- --ignored --nocapture bench_infer`
    #[test]
    #[ignore]
    #[allow(clippy::unwrap_used)]
    fn bench_infer_batch_scaling() -> Result<()> {
        use std::time::Instant;

        use tract_ndarray::Array2;

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let path = std::path::Path::new("saved_models/dagger/ens_d3.onnx");
        let mut c = Checks::new();
        if !path.is_file() {
            println!("跳过：本机没有 {}", path.display());
            return c.finish();
        }

        // 取一个真实局面用于编码基准
        let trainer = RamenNnTrainer::load(path)?;
        let mut rng = StdRng::seed_from_u64(42);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        advance_to_decision(&mut game, &trainer, &mut rng)?;

        // --- 1. 特征编码 ---
        let n_enc = 20_000;
        let t0 = Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..n_enc {
            let f = encode(&game)?;
            sink += f[0];
        }
        let enc_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n_enc);
        println!("特征编码        {enc_us:>8.1} µs/次 (sink={sink:.3})");

        // --- 2. 符号 batch（当前 load 的做法） ---
        let dynamic = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        let one = Array2::<f32>::zeros((1, features::INPUT_DIM));
        let n_run = 2_000;
        let t0 = Instant::now();
        for _ in 0..n_run {
            let _ = dynamic.run(tvec!(one.clone().into_tvalue()))?;
        }
        let dyn_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n_run);
        println!("符号 batch b=1  {dyn_us:>8.1} µs/次");

        // --- 3. 固定 batch 的吞吐曲线 ---
        println!("{:<16}{:>12}{:>14}{:>10}", "固定 batch", "µs/批", "µs/样本", "相对 b=1");
        let mut per_sample_at_1 = 0.0f64;
        for &b in &[1usize, 8, 32, 128, 512] {
            let fixed = tract_onnx::onnx()
                .model_for_path(path)?
                .with_input_fact(0, f32::fact([b, features::INPUT_DIM]).into())?
                .into_optimized()?
                .into_runnable()?;
            let input = Array2::<f32>::zeros((b, features::INPUT_DIM));
            // 批越大单次越贵，样本总数大致持平即可
            let reps = (16_384 / b).max(4);
            let t0 = Instant::now();
            for _ in 0..reps {
                let _ = fixed.run(tvec!(input.clone().into_tvalue()))?;
            }
            let per_batch_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let per_sample_us = per_batch_us / b as f64;
            if b == 1 {
                per_sample_at_1 = per_sample_us;
            }
            println!(
                "{:<16}{:>12.1}{:>14.2}{:>10.2}x",
                b,
                per_batch_us,
                per_sample_us,
                per_sample_at_1 / per_sample_us
            );
        }

        // --- 4. 集成 vs 单成员：ens_d3 是 3 个模型的算术平均，成本理应约 3 倍 ---
        println!("\n{:<28}{:>12}", "模型（固定 b=1）", "µs/次");
        for name in ["ens_d3.onnx", "d_s1.onnx"] {
            let one_path = std::path::Path::new("saved_models/dagger").join(name);
            if !one_path.is_file() {
                println!("{name:<28}{:>12}", "缺文件");
                continue;
            }
            let m = tract_onnx::onnx()
                .model_for_path(&one_path)?
                .with_input_fact(0, f32::fact([1, features::INPUT_DIM]).into())?
                .into_optimized()?
                .into_runnable()?;
            let input = Array2::<f32>::zeros((1, features::INPUT_DIM));
            let t0 = Instant::now();
            for _ in 0..2_000 {
                let _ = m.run(tvec!(input.clone().into_tvalue()))?;
            }
            println!("{name:<28}{:>12.1}", t0.elapsed().as_secs_f64() * 1e6 / 2_000.0);
        }

        println!(
            "\n参考：手写策略单次决策约 {:.1} µs（由 3.5 s / 局 与约 2.4 万次 rollout 决策粗估）",
            3.5e6 / 24_000.0
        );
        c.check(enc_us > 0.0, "编码基准跑通");
        c.finish()
    }
}
