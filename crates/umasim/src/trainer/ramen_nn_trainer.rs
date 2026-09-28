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
    gamedata::{EventChoice, EventData},
    search::RamenLeafValuer
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
    special_mode: SpecialSelectMode,
    /// **本实例**发出的推理请求数（见 [`Self::request_count`]）
    ///
    /// 与进程级的 [`infer_request_count`] 并存而不是取代它：全局量用来看整进程的
    /// 总成本，实例量用来写**确定性断言**——`cargo test` 默认并行，全局量的前后差
    /// 会被别的测试污染，断言它等于某个数就是在造一个随调度变红的测试。
    ///
    /// 放在 [`Arc`] 里：本结构 `#[derive(Clone)]`，克隆体与本体指的是同一个模型，
    /// 计数也该是同一份。
    requests: Arc<AtomicU64>
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
            special_mode: SpecialSelectMode::Canonical,
            requests: Arc::new(AtomicU64::new(0))
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
        self.requests.fetch_add(1, Ordering::Relaxed);
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

    /// **本实例**至今发出的推理请求数
    ///
    /// 与进程级 [`infer_request_count`] 的区别见 [`Self::requests`]：写断言用这个，
    /// 看整进程总成本用那个。
    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// 本模型旁车里的三路 value 反归一化常数
    ///
    /// 存在的理由：叶估值与外部批量后端拿到的都是**归一化**的 value，解码必须用与
    /// policy **同一个模型**的常数。把常数从这里取出来，调用方就不必自己再读一遍
    /// 旁车 JSON，也就不会与 [`Self::infer_features`] 的解码口径分叉。
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

/// CPU 叶估值器：用 [`RamenNnTrainer`] 的 value 头给**非终局**局面打一个结算分预测
///
/// # 取哪一路、怎么解码
///
/// 取 `value[0]`（三路里的 `mean`），由 [`RamenValueNorm::denormalize`] **恰好反归一化
/// 一次**。不叠加当前分、不取整、不与 `stdev` / `high` 混用。
///
/// ❗`value[0]` 学的是「有限样本教师选一步、随后手写续跑」的交叉拟合普通均值，**不是**
/// 完整 NN 或持续重搜索的价值。用它替掉 rollout 的尾巴，量的是「叶替换路径」的效应。
///
/// # 计数分两本账
///
/// [`Self::leaf_requests`] 只数**叶**请求。底层 [`RamenNnTrainer::infer`] 同时会推高
/// 该模型实例的总请求数，因此 `policy 请求数 = nn.request_count() - leaf_requests()`
/// ——两者混在一起就说不清叶路径到底花了多少推理。
///
/// # 为什么持 `Arc`
///
/// 一次搜索会在多个 rayon 线程上跑成百上千条 rollout，模型必须共享；按需加载会把加载
/// 代价乘以 rollout 数。
pub struct RamenNnLeafValuer {
    /// 共享的模型实例
    nn: Arc<RamenNnTrainer>,
    /// 本估值器发出的**叶**请求数（rollout 并行，故用原子量）
    leaf_requests: Arc<AtomicU64>
}

impl RamenNnLeafValuer {
    /// 用一个已加载的模型建叶估值器
    pub fn new(nn: Arc<RamenNnTrainer>) -> Self {
        Self { nn, leaf_requests: Arc::new(AtomicU64::new(0)) }
    }

    /// 底层模型（供调用方读总请求数、打印旁车常量等）
    pub fn model(&self) -> &Arc<RamenNnTrainer> {
        &self.nn
    }
}

impl RamenLeafValuer for RamenNnLeafValuer {
    /// 编码 → 推理 → 取 `value[0]`
    ///
    /// # 错误
    ///
    /// 编码、推理、输出契约或有限性检查失败时原样上抛——**不**退回手写估值，
    /// 否则「推理坏了」会被记成「这条路线分低」。
    fn value_of(&self, game: &RamenGame) -> Result<f64> {
        self.leaf_requests.fetch_add(1, Ordering::Relaxed);
        let out = self.nn.infer(game)?;
        let v = out.value.mean;
        // `denormalize` 查的是**反归一化前**的 raw 有限性；center/scale 再离谱也可能把
        // 有限的 raw 推成 inf，故这里再查一次反归一化**之后**的值。
        ensure!(
            v.is_finite(),
            "叶预测反归一化后不是有限值: {v}（center[0]={} scale[0]={}）",
            self.nn.value_norm().center[0],
            self.nn.value_norm().scale[0]
        );
        Ok(v)
    }

    fn leaf_requests(&self) -> u64 {
        self.leaf_requests.load(Ordering::Relaxed)
    }

    fn label(&self) -> String {
        "cpu-nn-value0".to_string()
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

    /// 最小 ONNX 模型生成器：workspace 根的 `testsupport/onnx_fixture.rs`
    ///
    /// `umasim` 与 `umaai` 两侧**共用同一份**生成器，靠 `#[path]` 各自引入一次：
    /// 不复制代码、不新增依赖、不做成生产公开 API，也不进入正式构建。
    #[path = "../../../../../../testsupport/onnx_fixture.rs"]
    mod onnx_fixture;


    /// 与冻结契约一致的旁车 JSON 文本（`input_dim` / `output_dim` 都**声明正确**）
    fn valid_sidecar_json() -> String {
        format!(
            r#"{{"input_dim":{},"output_dim":{},"value_normalization":{{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}}}"#,
            features::INPUT_DIM,
            OUTPUT_DIM
        )
    }

    /// 指定 `value` 三路取值的旁车 JSON
    fn sidecar_json(center: [f64; 3], scale: [f64; 3]) -> String {
        format!(
            r#"{{"input_dim":{},"output_dim":{},"value_normalization":{{"center":[{},{},{}],"scale":[{},{},{}]}}}}"#,
            features::INPUT_DIM,
            OUTPUT_DIM,
            center[0],
            center[1],
            center[2],
            scale[0],
            scale[1],
            scale[2]
        )
    }

    /// 造一个输出恒等于指定向量的 fixture 模型，并写好旁车
    ///
    /// 返回 `(临时目录, 模型路径)`；调用方负责 `cleanup_test_dir`。
    ///
    /// # 错误
    ///
    /// 目录创建、模型生成或写盘失败时返回。
    fn fixture_with_value(
        tag: &str, raw_value: [f32; VALUE_DIM], center: [f64; 3], scale: [f64; 3]
    ) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
        let dir = unique_test_dir(tag)?;
        let model = dir.join("v.onnx");
        let mut logits = vec![0.0f32; OUTPUT_DIM];
        for (i, v) in raw_value.iter().enumerate() {
            logits[POLICY_DIM + CHOICE_DIM + i] = *v;
        }
        std::fs::write(&model, onnx_fixture::const_logits_model(features::INPUT_DIM, &logits)?)?;
        std::fs::write(dir.join("v.onnx.json"), sidecar_json(center, scale))?;
        Ok((dir, model))
    }

    /// 叶估值器取 `value[0]`，**恰好**反归一化一次，不叠加当前分、不取整
    ///
    /// 用生成式 fixture 把网络输出钉死成一个已知常数，于是期望值可以手算：
    /// `center[0] + scale[0] * raw`。真实权重下这个数是多少属于模型质量问题，
    /// 与解码正确性是两件事。
    ///
    /// # 错误
    ///
    /// 解码结果与手算不符，或计数分账不对时返回错误。
    #[test]
    fn test_leaf_valuer_decodes_value0_once() -> Result<()> {
        use crate::search::RamenLeafValuer;

        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let mut c = Checks::new();

        // raw = [2.5, 9.0, 9.0]，center/scale 取互不相同、且带小数的值：
        // 三路写反哪一路都会露馅，期望值 2251.5 还能顺带钉住「不取整」。
        let (dir, model) = fixture_with_value(
            "leaf_value_decode",
            [2.5, 9.0, 9.0],
            [1000.25, -7.0, -7.0],
            [500.5, 3.0, 3.0]
        )?;
        let nn = Arc::new(RamenNnTrainer::load(&model)?);
        let valuer = RamenNnLeafValuer::new(Arc::clone(&nn));

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(20260922);
        advance_to_decision(&mut game, &nn, &mut rng)?;

        let want = 1000.25 + 500.5 * 2.5;
        let got = valuer.value_of(&game)?;
        println!("叶估值 = {got}（手算 center[0] + scale[0]*raw = {want}）");
        println!("当前局面分 = {}", game.uma.calc_score());
        c.check((got - want).abs() < 1e-9, "取的是 value[0]，且反归一化恰好一次");
        c.check(
            (got - want).abs() < 1e-9 && got != want + f64::from(game.uma.calc_score()),
            "没有叠加当前分"
        );
        c.check(got.fract() != 0.0, "没有取整：小数部分原样保留（期望值 2251.5）");

        // 计数分两本账：叶请求单独记，模型实例的总请求数把 policy 与 leaf 都算上
        let leaf_before = valuer.leaf_requests();
        let total_before = nn.request_count();
        let actions = game.list_actions()?;
        let _ = nn.select_action(&game, &actions, &mut rng.clone())?;
        let leaf_mid = valuer.leaf_requests();
        let total_mid = nn.request_count();
        let _ = valuer.value_of(&game)?;
        println!(
            "叶请求 {leaf_before} -> {leaf_mid} -> {}；实例总请求 {total_before} -> {total_mid} -> {}",
            valuer.leaf_requests(),
            nn.request_count()
        );
        c.check(leaf_mid == leaf_before, "一次 policy 决策**不**计入叶请求");
        c.check(total_mid > total_before, "但它确实计入了实例总请求（证明上一条不是空的）");
        c.check(valuer.leaf_requests() == leaf_mid + 1, "一次叶估值只记一笔叶请求");
        c.check(valuer.label() == "cpu-nn-value0", "估值器标签写明了口径");

        cleanup_test_dir(&dir)?;
        c.finish()
    }

    /// 反归一化**之后**的非有限值必须被拦下
    ///
    /// `RamenValueNorm::denormalize` 查的是反归一化**前**的 raw；raw 有限、
    /// `center + scale * raw` 却溢出成 `inf` 的情形只有在解码之后才看得见。
    ///
    /// # 错误
    ///
    /// 没有报错时返回错误。
    #[test]
    fn test_leaf_valuer_rejects_non_finite_after_denorm() -> Result<()> {
        use crate::search::RamenLeafValuer;

        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let mut c = Checks::new();

        // raw = 1e38（f32 有限），scale = 1e300 → 乘出来在 f64 里也溢出
        let (dir, model) =
            fixture_with_value("leaf_value_overflow", [1e38, 0.0, 0.0], [0.0, 0.0, 0.0], [1e300, 1.0, 1.0])?;
        let nn = Arc::new(RamenNnTrainer::load(&model)?);
        let valuer = RamenNnLeafValuer::new(Arc::clone(&nn));

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        let mut rng = StdRng::seed_from_u64(20260922);
        advance_to_decision(&mut game, &nn, &mut rng)?;

        // 先证明 raw 那一关确实放行了——否则这条测的就不是「反归一化之后」
        let raw_ok = nn.infer(&game).map(|o| o.value.mean);
        println!("反归一化后的 value[0] = {raw_ok:?}");
        c.check(
            raw_ok.as_ref().is_ok_and(|v| !v.is_finite()),
            "raw 有限，denormalize 放行，但乘完之后已经不是有限值"
        );

        let e = valuer.value_of(&game).err().map(|e| format!("{e:#}")).unwrap_or_default();
        println!("叶估值器 -> {e}");
        c.check(e.contains("反归一化后"), "叶估值器在解码**之后**再查一次有限性并报错");

        cleanup_test_dir(&dir)?;
        c.finish()
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

        // ❗断言走**实例**计数：进程级 `infer_request_count()` 会被并行跑的别的测试
        // 推高，拿它的前后差做断言等于造一个随调度变红的测试（`-j` 控制的是构建并发，
        // 不是测试串行）。
        let loaded = RamenNnTrainer::load(&model);
        c.check(loaded.is_ok(), "合法 fixture 加载成功");
        match &loaded {
            Ok(t) => {
                println!("加载后本实例推理请求数: {}", t.request_count());
                c.check(t.request_count() == 0, "加载（含图输出契约校验）不计入任何推理请求");
                // 再证一次这个计数**不是恒零**，否则上一条断言是空的
                let feats = vec![0.0f32; features::INPUT_DIM];
                let _ = t.infer_features(feats)?;
                println!("跑一次推理后: {}", t.request_count());
                c.check(t.request_count() == 1, "跑一次推理后实例计数为 1（证明上条断言非空）");
            }
            Err(e) => println!("加载失败，跳过实例计数检查: {e:#}")
        }
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

    /// 加载真实 pilot 模型，在开局第一决策点跑一次 select_action
    ///
    /// ❗**真实模型集成测试，默认 `#[ignore]`。** 它断言 `value.mean` 落在 5 万–7 万
    /// 分数量纲，这是**训练出来的权重**才有的性质，生成式 fixture（全零权重）顶不上。
    ///
    /// 以前它写成「缺模型就 `return Ok(())`」：在没有 `saved_models/` 的机器上会**绿着
    /// 通过而零覆盖**，把「没验」记成「验过了」。现在改成默认不跑、显式跑就必须有模型：
    /// `cargo test -p umasim --release --lib --features onnx -- --ignored select_action_opening`
    ///
    /// # 错误
    ///
    /// 显式运行而模型不存在时**报错**，不再静默通过。
    #[test]
    #[ignore = "需要真实 pilot 模型（saved_models 不入库）；显式 --ignored 运行"]
    fn test_ramen_nn_select_action_opening() -> Result<()> {
        let root = get_workspace_root()?;
        std::env::set_current_dir(&root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let model_path = root.join("saved_models").join("ramen_pilot").join("model.onnx");
        println!("模型路径: {}", model_path.display());
        ensure!(
            model_path.is_file(),
            "本测试需要真实模型 {}，但它不存在；本测试是显式运行的，不会静默跳过",
            model_path.display()
        );
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
    /// ❗进程级的 [`infer_request_count`] 只打印不断言（并行下不是确定量）；
    /// 确定性断言走 [`RamenNnTrainer::request_count`] 这个**实例**计数。
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

        // ❗用**当场生成**的 fixture，不依赖未入库的正式权重：本测试钉的是
        // 「单候选走 Resolved、不推理、不动随机流」这条分支行为，与权重数值无关。
        // 以前它依赖 saved_models/ramen_pilot/model.onnx，在没有模型的机器上
        // 绿着通过而零覆盖。
        let dir = unique_test_dir("single_candidate_skips_inference")?;
        let model_path = dir.join("fixture.onnx");
        fs_err::write(&model_path, onnx_fixture::matmul_model(features::INPUT_DIM, OUTPUT_DIM))?;
        fs_err::write(dir.join("fixture.onnx.json"), valid_sidecar_json())?;
        let trainer = RamenNnTrainer::load(&model_path)?;

        let (mut rng, rule_master) = crate::bench::seeded_rngs(42, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        advance_to_decision(&mut game, &trainer, &mut rng)?;

        let actions = game.list_actions()?;
        println!("阶段 {:?}  回合 {}  候选数 {}", game.stage, game.turn(), actions.len());

        let mut c = Checks::new();

        c.check(actions.len() > 1, "本决策点应是多候选（否则下面的对照没有意义）");

        // (1) 准备阶段的分支：多候选仍返回 NeedsInference，单候选直接 Resolved(0)
        let multi_prep = trainer.prepare_decision(&game, &actions, &mut rng)?;
        println!("多候选 prepare_decision -> {}", prep_name(&multi_prep));
        c.check(
            matches!(multi_prep, DecisionPrep::NeedsInference(_)),
            "多候选决策点应仍返回 NeedsInference"
        );
        let single_prep = trainer.prepare_decision(&game, &actions[..1], &mut rng)?;
        println!("单候选 prepare_decision -> {}", prep_name(&single_prep));
        c.check(
            matches!(single_prep, DecisionPrep::Resolved(0)),
            "单候选应直接定案为 Resolved(0)"
        );

        // (2) **真正的决策路径**：`select_action`
        //
        // ❗只测 `prepare_decision` 证明不了「省下了推理」——那个方法本来就不推理。
        // 省的是「`Resolved` 不会再进 `infer_features`」，只有走完整的 `select_action`
        // 才看得到。下面用**实例**请求计数（确定量）分别量多候选与单候选两条路。
        let multi_before = trainer.request_count();
        let multi_probe_before = rng.clone().next_u64();
        let multi_idx = trainer.select_action(&game, &actions, &mut rng)?;
        let multi_probe_after = rng.clone().next_u64();
        let multi_after = trainer.request_count();
        println!(
            "多候选 select_action -> {multi_idx}  实例请求 {multi_before} -> {multi_after}  \
             rng {multi_probe_before:#018x} -> {multi_probe_after:#018x}"
        );
        c.check(multi_idx < actions.len(), "多候选返回的下标在候选范围内");
        c.check(
            multi_after == multi_before + 1,
            "多候选走完整决策路径时**确实发生了一次推理**（实例计数 +1）"
        );

        let single_before = trainer.request_count();
        let global_before = infer_request_count();
        let probe_before = rng.clone().next_u64();
        let single_idx = trainer.select_action(&game, &actions[..1], &mut rng)?;
        let probe_after = rng.clone().next_u64();
        let global_after = infer_request_count();
        let single_after = trainer.request_count();
        println!(
            "单候选 select_action -> {single_idx}  实例请求 {single_before} -> {single_after}  \
             rng {probe_before:#018x} -> {probe_after:#018x}"
        );
        println!("  （观察）全局推理请求计数 {global_before} -> {global_after}，并行下不可作断言");
        c.check(single_idx == 0, "单候选 select_action 返回唯一候选的下标 0");
        c.check(
            single_after == single_before,
            "单候选走完整决策路径时**一次推理都没发出**（实例计数不变）"
        );
        c.check(probe_after == probe_before, "单候选路径不消耗随机流");
        cleanup_test_dir(&dir)?;
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
    /// ❗**真实模型集成测试，默认 `#[ignore]`。** 生成式 fixture 的权重是全零，
    /// 两条路的输出都是 0，逐位相等**恒成立**，证明不了形状特化没改数值。只有真实
    /// 权重才有区分度。以前写成「缺模型就跳过并返回成功」，在没有模型的机器上是
    /// 绿着的零覆盖。显式运行：
    /// `cargo test -p umasim --release --lib --features onnx -- --ignored fixed_shape`
    ///
    /// # 错误
    ///
    /// 显式运行而模型不存在时**报错**，不再静默通过。
    #[test]
    #[ignore = "需要真实模型 saved_models/dagger/ens_d3.onnx；显式 --ignored 运行"]
    fn test_fixed_shape_matches_symbolic() -> Result<()> {
        use tract_ndarray::Array2;

        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut c = Checks::new();
        let path = std::path::Path::new("saved_models/dagger/ens_d3.onnx");
        ensure!(
            path.is_file(),
            "本测试需要真实模型 {}，但它不存在；本测试是显式运行的，不会静默跳过",
            path.display()
        );

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


}
