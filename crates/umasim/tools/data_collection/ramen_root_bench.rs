//! 固定根实验入口：CPU/GPU 动作对拍、rollout trace、两种 CPU 并行基线
//!
//! # 为什么是「固定根」而不是整局闭环
//!
//! 整局闭环要回答的是「`greedy(Q^NN)` 是否强于 `greedy(Q^手写)`」，那是几十小时
//! 量级的验收。本工具只回答工程问题：**批量 GPU 推理是否与 CPU 得到同一个动作、
//! 每条 rollout 要多少次推理、按 rollout 并行能把 CPU 自己提到多快**。三个问题
//! 共用同一批根、候选与种子表，故做成同一个入口的三种模式，改参数不必重新编译。
//!
//! # 汇总一律走生产路径
//!
//! 曾经在这里自己写 `sum/n → argmax`，那**不是**教师口径（教师用 rank 加权均值
//! 与 radical factor）。现在改成：并行粒度只决定「rollout 在哪跑」，跑完的原始结果
//! 存进备忘表，再交给 [`FlatSearch::search_with`] 做官方汇总与排序。两种粒度因此
//! 共用同一份汇总实现，不会各自漂移。
//!
//! # 对拍为什么不落盘重放
//!
//! 保留同一份原始快照就已经保证两端比较的是同一个决策，不需要「重建局面 + 恢复
//! RNG + 重走轨迹」那套回放系统。
//!
//! ❗compare 模式带快照克隆与批量同步开销，**只用于正确性与逻辑长度分析，不作性能
//! 数字**；性能取自两个 CPU 基线模式。

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Instant
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{from_str as json_from_str, to_string_pretty as json_to_string_pretty};
use rand::rngs::StdRng;
use rayon::prelude::*;
use rand::{RngCore, SeedableRng};
use std::cell::Cell;
use umasim::{
    bench::seeded_rngs,
    game::{
        Game, InheritInfo, Trainer,
        ramen::{RamenAction, RamenGame, RamenStage, RamenState, features::encode}
    },
    exp_config::{EffectiveSearchFacts, ScoringOverride, report_effective},
    gamedata::{EventChoice, EventData, RamenRegionStrategy, init_global_with_config},
    sampler::{SamplingSpace, gen1_inherit, space_from_cli, space_version_by_name},
    search::{
        FlatSearch, FlatSearchGame, RamenBatchRollout, RamenBatchTable, RamenTerminal, RolloutOutcome, RolloutSeeds,
        SearchConfig, SearchScore
    },
    trainer::{
        DecisionPrep, RolloutInferSink, RolloutInferSnapshot, RamenMctsTrainer, RamenNnTrainer, RamenRolloutTrainer,
        RamenSearchStages, RamenSelection, RecommendedRamenTrainer, SpecialSelectMode
    },
    utils::{get_workspace_root, load_game_config}
};

/// 模型输出里 policy 段的长度（与 `RamenNnTrainer` 契约一致）
const POLICY_DIM: usize = 234;
/// 侧车一次返回的每行长度
const OUTPUT_DIM: usize = 245;
/// 模型输入维度
const INPUT_DIM: usize = 754;

/// 一条 rollout 的身份
#[derive(Debug, Clone, Copy)]
struct RolloutCtx {
    /// 候选下标
    candidate: usize,
    /// rollout 序号 `j`（CRN 载体，跨候选共享种子）
    j: usize,
    /// 该 rollout 内已发生的推理次数
    seq: u32
}

thread_local! {
    /// 当前线程正在跑的 rollout 身份
    ///
    /// 录制上下文绑定在**实验入口**而不是搜索内核：入口本来就知道 `(候选, j)`，
    /// 从这里透进去不必改通用搜索的接口。每条 rollout 在一个 rayon 任务里从头跑到尾，
    /// 故线程局部量足以定位。
    static ROLLOUT_CTX: RefCell<Option<RolloutCtx>> = const { RefCell::new(None) };

    /// 本线程上一次推理返回的时刻，用于算决策间隔
    static LAST_INFER_END: RefCell<Option<Instant>> = const { RefCell::new(None) };
}

/// 并行粒度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Granularity {
    /// 与生产 `search_uniform` 同构：按候选并行，候选内串行跑 `j`
    Candidate,
    /// 按 `(候选, j)` 扁平并行
    Flat
}

/// 运行模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    /// CPU 推进 + 批量 GPU 对拍（同时产出 trace）
    Compare,
    /// CPU 基线：按候选并行
    CpuCandidate,
    /// CPU 基线：按 `(候选, j)` 扁平并行
    ///
    /// 公平基线的必要性：只跟「按候选并行」比，会把「解除 CPU 并行瓶颈」的收益
    /// 也算到 GPU 头上。候选数少时（吃面阶段常见 2–4 个）这一项相当可观。
    CpuFlat,
    /// GPU 波次驱动：可暂停的 rollout + 确定性补位 + 批量推理
    GpuWave,
    /// 常驻侧车复用：同一侧车连续跑「根 A → 根 B → 根 A」，比较两次 A
    SidecarReuse,
    /// 整局冒烟：一局完整的波次驱动搜索，验证接入
    ///
    /// ❗只验证「接得通」，**回答不了策略优劣**——那要配对实验。
    GameSmoke,
    /// 正式接入一致性：**生产教师**在 CPU / GPU 两种后端下的逐决策比较
    TeacherConsistency,
    /// 正式配置整局：生产教师 + 波次后端，实测耗时与利用率
    TeacherGame,
    /// 决策点扫描：用冻结策略走一局，落下每个决策点的回合/阶段/候选数
    ///
    /// 用于把预登记里的「Train-早 / RegionSelect-Y2」这类配额**在看结果之前**
    /// 解析成具体的 `(root_turn, root_stage)`，不涉及任何搜索。
    RootScan,
    /// leaf value 截断试点：同一个根上跑 full 与若干截断深度
    LeafPilot,
    /// value 解码对拍：同一批真实局面上比较 CPU(ONNX 集成) 与 GPU(侧车三成员) 的 value
    ValueCheck
}

/// 固定根实验参数
#[derive(Parser, Debug)]
#[command(about = "固定根：CPU/GPU 对拍、rollout trace、CPU 并行基线")]
struct RootArgs {
    /// 运行模式
    #[arg(long, value_enum)]
    mode: Mode,

    /// 每个候选的 rollout 条数
    #[arg(long, default_value_t = 128)]
    search_n: usize,

    /// 采样空间里的计划下标（决定马娘与卡组）
    #[arg(long, default_value_t = 0)]
    plan_index: usize,

    /// 建局基种子
    #[arg(long, default_value_t = 61444)]
    seed: u64,

    /// 局号（与 `ramen_space_bench` 的 `run_idx` 同义）
    #[arg(long, default_value_t = 1200)]
    run_idx: u64,

    /// 把根推进到该回合（含）后停在第一个决策点
    #[arg(long, default_value_t = 12)]
    root_turn: i32,

    /// 只在指定阶段取根；不给则取到达的第一个决策阶段
    #[arg(long)]
    root_stage: Option<String>,

    /// 复用测试里第二个根的回合（仅 sidecar-reuse）
    #[arg(long, default_value_t = 6)]
    root_b_turn: i32,

    /// 复用测试里第二个根的阶段（仅 sidecar-reuse）
    #[arg(long)]
    root_b_stage: Option<String>,

    /// 教师的激进度上限（与正式口径一致）
    #[arg(long, default_value_t = 1.4)]
    radical_factor_max: f64,

    /// 显式固定 `pt_favor_rate`（进终局 `score_pt`）
    ///
    /// 本入口选动作固定走 [`RamenSelection::Score`]，该倍率不参与选动作；不给时由
    /// [`ScoringOverride`] 钉死为 [`umasim::exp_config::PINNED_PT_FAVOR_RATE`]，
    /// **不回落 `game_config.toml`**。仍然记录，是因为上游 `70550cd` 后它进
    /// `search_score().score_pt`，而 `pt_favor_rate = 1` 与 Score 轴**并不等价**
    /// （`score_pt` 不含 Hint 折算、`score` 含）。
    #[arg(long)]
    pt_favor_rate: Option<f32>,

    /// rollout 基策用的 ONNX 模型（CPU 侧）
    #[arg(long)]
    rollout_model: PathBuf,

    /// rayon worker 数；不给则用默认（逻辑核数）
    #[arg(long)]
    workers: Option<usize>,

    /// 侧车脚本路径（仅 compare 模式）
    #[arg(long)]
    sidecar: Option<PathBuf>,

    /// 侧车用的 checkpoint（可重复，仅 compare 模式）
    #[arg(long)]
    checkpoint: Vec<PathBuf>,

    /// python 解释器
    #[arg(long, default_value = "python")]
    python: String,

    /// 侧车后端张量的物理批尺寸（恒定，不足补零）
    ///
    /// 开 `--adaptive-batch` 时本值变成**上限**：侧车按它分配缓冲区，实际前向行数
    /// 由每根选出的档位决定。
    #[arg(long, default_value_t = 512)]
    batch: usize,

    /// 按根选择物理批档位（见 [`plan_batch`]），默认关闭
    ///
    /// ❗关闭时一个控制消息都不发，线上字节流与改动前完全一致。
    #[arg(long, default_value_t = false)]
    adaptive_batch: bool,

    /// 让侧车把整段集成前向捕成 CUDA Graph 后重放（仅波次后端），默认关闭
    ///
    /// 只改前向的执行调度，不动模型、精度、集成口径与候选评分。图按固定行数捕获，
    /// 因此与 `--adaptive-batch` 互斥，由 [`check_switch_support`] 在建局前拦下。
    #[arg(long, default_value_t = false)]
    sidecar_graph: bool,

    /// 同轨迹 `RamenSelect` → `SpecialSelect` 的 policy 复用（仅波次后端），默认关闭
    ///
    /// 命中口径见 [`cache_probe`]：同一条轨迹、同一回合、上一拍是 `RamenSelect`、
    /// 当前是 `SpecialSelect`，且实际编码的 754 个 f32 **位模式**逐一相同且全部有限。
    #[arg(long, default_value_t = false)]
    policy_cache: bool,

    /// 逐决策 trace 输出 CSV（仅 compare 模式）
    #[arg(long)]
    trace: Option<PathBuf>,

    /// 逐 rollout 汇总输出 CSV（含零推理轨迹）
    #[arg(long)]
    rollout_csv: Option<PathBuf>,

    /// 逐 rollout 原始结果输出 CSV（用于跨模式逐位比较）
    #[arg(long)]
    raw_csv: Option<PathBuf>,

    /// 用手写 rollout 基策（配对实验的另一臂），不接网络也不接侧车
    ///
    /// 用来**分臂**估算配对实验预算：两臂成本不同，不能都按 NN 计价。
    #[arg(long, default_value_t = false)]
    handwritten_rollout: bool,

    /// 整局逐步决策输出 CSV（仅 teacher-game）
    ///
    /// 落的是**实际值**：回合、阶段、候选原顺序、选中下标、决策后 RNG 探针取值。
    /// 两臂各落一份后可直接逐行比较。
    #[arg(long)]
    steps_csv: Option<PathBuf>,

    /// 逐决策等价性输出 CSV
    ///
    /// 落的是**实际值**：合法候选的原顺序、选中动作、完整输入特征。
    /// ❗体积随 `候选数 × n × 决策数` 线性增长，详细对拍请用小 `n`，与性能测量分开跑。
    #[arg(long)]
    decision_csv: Option<PathBuf>,

    /// 具名采样空间版本（如 `gen2_v1`）；不给则用第一代默认空间
    #[arg(long)]
    space_version: Option<String>,

    /// 建根时推进用的策略
    ///
    /// `handwritten`（默认，既有行为）或 `nn`（用 `--rollout-model` 的纯网络策略）。
    #[arg(long, default_value = "handwritten")]
    root_policy: String,

    /// 整局搜索的叶深度路由（只被 [`Mode::GameSmoke`] 消费）
    ///
    /// `uniform`（默认，**既有行为**：全局用 `--leaf-h` 给的那一个深度，不给就是 full）
    /// 或 `hybrid-y3-full`（第三年 `RegionSelect` 完整续跑 + 原 rf，其余根按 `--leaf-h` 截断 + mean）。
    #[arg(long, default_value = "uniform")]
    game_route: String,

    /// 整局逐根路由记录输出 CSV（只被 [`Mode::GameSmoke`] 消费）
    ///
    /// 落的是**实际值**：每个走了搜索的根的路由判定、地区年份、实际聚合目标与实际 rf、
    /// 两类请求数与两类结局条数。
    #[arg(long)]
    route_csv: Option<PathBuf>,

    /// 截断深度清单，可重复；每项是**正整数** H（跨越 H 个 turn 边界）
    ///
    /// 只被 [`Mode::LeafPilot`] 消费。给了却用在别的模式上会在建局前报错，
    /// 不静默忽略。full 臂恒定参与，不需要在这里写。
    #[arg(long)]
    leaf_h: Vec<i32>,

    /// 截断模式的根目标
    ///
    /// 只接受 `mean`。存在的理由：普通配置里的 `--radical-factor-max` 有默认值，
    /// CLI 分不清「用户没提」与「用户显式要求叶结果也用 rf」。这里给一个**显式**
    /// 目标参数，写别的值直接拒绝，而不是假装读过一个实际没用的参数。
    #[arg(long, default_value = "mean")]
    leaf_objective: String,

    /// 完整参考臂的列数（selection bank + audit bank）
    #[arg(long, default_value_t = 512)]
    full_n: usize,

    /// 用于**选择**的前缀列数；也是各截断臂的 n
    #[arg(long, default_value_t = 256)]
    select_n: usize,

    /// 根清单 JSON（[`Mode::LeafPilot`]）；不给则只跑单根 CLI 参数指定的那一个
    #[arg(long)]
    roots_file: Option<PathBuf>,

    /// leaf 实验输出目录（[`Mode::RootScan`] / [`Mode::LeafPilot`]）
    #[arg(long)]
    leaf_out: Option<PathBuf>,

    /// [`Mode::ValueCheck`] 取多少个真实决策局面做对拍
    #[arg(long, default_value_t = 64)]
    value_rows: usize,

    /// 叶状态续跑诊断：记下叶估值后**用同一份局面与同一个 RNG** 继续跑到终局
    ///
    /// 于是同一条 rollout 同时给出「网络预测」与「真实终局」，可算逐条配对残差。
    /// ❗额外续跑会让截断臂的请求数与耗时**不再代表截断的成本**，
    /// 本模式只用于正确性与偏差诊断，性能数字必须另跑一轮不带本开关的。
    #[arg(long, default_value_t = false)]
    leaf_diag: bool
}

/// 侧车就绪握手的标记行
///
/// 侧车必须在**模型加载与预热都完成之后**才写这一行：否则首个 `infer` 会把权重
/// 加载、CUDA 上下文建立与首次 kernel 编译一并算进推理耗时。
const SIDECAR_READY_MARK: &str = "[sidecar] ready";

/// 排空线程保留的诊断行数上限
const SIDECAR_DIAG_LINES: usize = 64;

/// 控制消息的哨兵值，占数据头的第一个 u32
///
/// 数据头 `(rows, valid)` 的含义**保持不变**：`rows` 恒在 `1..=物理批` 内，取不到
/// `u32::MAX`。于是控制消息与数据消息天然可分，不需要重新解释旧字段。
const CTRL_MAGIC: u32 = u32::MAX;

/// 控制操作码：设置本根的物理张量批尺寸
const OP_SET_BATCH: u32 = 1;

/// 本工具要求的侧车协议版本（就绪行里的 `proto=`）
const SIDECAR_PROTO: u64 = 2;

/// 物理批档位表
const BATCH_TIERS: [usize; 3] = [256, 512, 1024];

/// 本轮两个优化开关在各模式下**是否真的会被消费**
///
/// 照实现逐条核对，不按印象写：
/// - [`Mode::GpuWave`]、[`Mode::SidecarReuse`]、[`Mode::TeacherConsistency`]、[`Mode::TeacherGame`]
///   都把两个开关交给波次后端；[`Mode::GameSmoke`] 经 [`WaveTrainer`] 同样**已接线**。
/// - [`Mode::Compare`] 只走 [`CompareSink`] 的固定批，不起波次调度；
///   [`Mode::CpuCandidate`] / [`Mode::CpuFlat`] 根本不起侧车。
fn mode_consumes_wave_switches(mode: Mode) -> bool {
    matches!(
        mode,
        Mode::GpuWave
            | Mode::SidecarReuse
            | Mode::GameSmoke
            | Mode::TeacherConsistency
            | Mode::TeacherGame
            | Mode::LeafPilot
            | Mode::ValueCheck
    )
}

/// 检查优化开关会不会被**静默忽略**
///
/// 调用点在建局、模型加载与侧车启动**之前**：宁可一上来就指名参数报错，也不要跑完
/// 才发现开关压根没生效、却把结果当成「优化后」的数字。
///
/// # 错误
///
/// 开关落在不消费它的模式上、与 `--handwritten-rollout` 同时给出，或
/// `--sidecar-graph` 与 `--adaptive-batch` 同时给出时报错。
fn check_leaf_support(args: &RootArgs) -> Result<Vec<LeafDepth>> {
    if !matches!(args.mode, Mode::LeafPilot) {
        ensure!(
            !args.leaf_diag,
            "--leaf-diag 只被 leaf-pilot 模式消费；{:?} 模式下直接拒绝",
            args.mode
        );
    }
    if matches!(args.mode, Mode::ValueCheck) {
        ensure!(args.value_rows > 0, "--value-rows 必须为正");
        ensure!(args.leaf_h.is_empty(), "value-check 不消费 --leaf-h，直接拒绝");
        return Ok(Vec::new());
    }
    if matches!(args.mode, Mode::GameSmoke) && !args.leaf_h.is_empty() {
        // 整局冒烟只接**一个**深度：一局只有一条轨迹，给两个深度没有意义，
        // 静默取第一个会让日志与实际跑的东西对不上，所以直接拒绝。
        ensure!(
            args.leaf_h.len() == 1,
            "game-smoke 只接受一个 --leaf-h，收到 {} 个",
            args.leaf_h.len()
        );
        ensure!(
            args.leaf_objective == "mean",
            "截断模式的根目标只支持 mean（实际 rf=0），收到 --leaf-objective {}",
            args.leaf_objective
        );
        let h = args.leaf_h[0];
        ensure!(h > 0, "截断深度必须为正整数，收到 {h}");
        ensure!(args.roots_file.is_none(), "game-smoke 不消费 --roots-file，直接拒绝");
        return Ok(vec![LeafDepth::Turns(h)]);
    }
    if !matches!(args.mode, Mode::LeafPilot) {
        ensure!(
            args.leaf_h.is_empty(),
            "--leaf-h 只被 leaf-pilot 模式消费；{:?} 模式下它会被静默忽略，故直接拒绝",
            args.mode
        );
        ensure!(
            args.roots_file.is_none(),
            "--roots-file 只被 leaf-pilot 模式消费；{:?} 模式下直接拒绝",
            args.mode
        );
        return Ok(Vec::new());
    }
    ensure!(
        args.leaf_objective == "mean",
        "截断模式的根目标只支持 mean（实际 rf=0），收到 --leaf-objective {}。\n\
         本轮不为叶结果启用非零 rf：叶预测与提前终局必须按同一个均值目标聚合。",
        args.leaf_objective
    );
    ensure!(
        !args.handwritten_rollout,
        "--handwritten-rollout 与 leaf 截断互斥：手写 rollout 不产生 value"
    );
    ensure!(args.select_n > 0, "--select-n 必须为正");
    ensure!(
        args.select_n <= args.full_n,
        "--select-n {} 不能大于 --full-n {}：选择列必须是完整参考的前缀",
        args.select_n,
        args.full_n
    );
    let mut depths = Vec::new();
    for &h in &args.leaf_h {
        ensure!(h > 0, "截断深度必须为正整数，收到 {h}");
        let d = LeafDepth::Turns(h);
        ensure!(!depths.contains(&d), "截断深度 {h} 重复给出");
        depths.push(d);
    }
    ensure!(!depths.is_empty(), "leaf-pilot 至少要给一个 --leaf-h");
    Ok(depths)
}

/// 解析并校验整局路由，在读模型与启侧车**之前**拒绝无效组合
///
/// 与 [`check_leaf_support`] 分开的理由：那个函数回答「跑哪些深度」，
/// 这个函数回答「整局里每个根怎么挑深度」。两者在非 `game-smoke` 模式下都必须
/// 显式拒绝，而不是静默忽略。
///
/// # 错误
///
/// 未知路由名、`hybrid-y3-full` 少给或错给 `--leaf-h`、在非 `game-smoke` 模式下
/// 给了 `--game-route` / `--route-csv` 时报错。
fn check_game_route(args: &RootArgs, depths: &[LeafDepth]) -> Result<LeafRoute> {
    if !matches!(args.mode, Mode::GameSmoke) {
        ensure!(
            args.game_route == "uniform",
            "--game-route 只被 game-smoke 模式消费；{:?} 模式下它会被静默忽略，故直接拒绝",
            args.mode
        );
        ensure!(
            args.route_csv.is_none(),
            "--route-csv 只被 game-smoke 模式消费；{:?} 模式下直接拒绝",
            args.mode
        );
        // 其它模式不走整局路由，这个值不会被消费
        return Ok(LeafRoute::Uniform(LeafDepth::Full));
    }
    match args.game_route.as_str() {
        // 既有行为：给了 --leaf-h 就全局截断，没给就全局 full
        "uniform" => Ok(LeafRoute::Uniform(depths.first().copied().unwrap_or(LeafDepth::Full))),
        "hybrid-y3-full" => {
            let d = depths
                .first()
                .copied()
                .ok_or_else(|| anyhow!("--game-route hybrid-y3-full 必须同时给一个 --leaf-h（普通根的截断深度）"))?;
            match d {
                LeafDepth::Turns(h) => Ok(LeafRoute::HybridY3Full(h)),
                LeafDepth::Full => bail!("--game-route hybrid-y3-full 的普通根深度不能是 full")
            }
        }
        other => bail!("未知 --game-route {other}：只支持 uniform 或 hybrid-y3-full")
    }
}

/// 检查优化开关会不会被**静默忽略**（原有实现，仅扩展模式表）
fn check_switch_support(
    mode: Mode, cache_on: bool, adaptive: bool, graph: bool, handwritten: bool
) -> Result<()> {
    // 图按固定行数捕获，换档要重捕；本轮不做这件事，所以直接拒绝而不是悄悄沿用旧图
    ensure!(
        !(graph && adaptive),
        "--sidecar-graph 与 --adaptive-batch 同时给出：图按固定物理批捕获，换档必须重捕；本轮不把两者耦合，请固定 --batch 后只给 --sidecar-graph"
    );
    let mut on: Vec<&str> = Vec::new();
    if cache_on {
        on.push("--policy-cache");
    }
    if adaptive {
        on.push("--adaptive-batch");
    }
    if graph {
        on.push("--sidecar-graph");
    }
    if on.is_empty() {
        return Ok(());
    }
    let names = on.join(" / ");
    ensure!(
        mode_consumes_wave_switches(mode),
        "{names} 只对波次（GPU）后端生效，模式 {mode:?} 不会消费它；\
         支持的模式：gpu-wave / sidecar-reuse / game-smoke / teacher-consistency / teacher-game"
    );
    ensure!(
        !(mode == Mode::TeacherGame && handwritten),
        "{names} 与 --handwritten-rollout 同时给出：手写臂不接网络、不起侧车，开关不会生效；\
         去掉其中之一再跑"
    );
    Ok(())
}

/// 批利用率的显示文本；**零推理格位的根给 N/A，不给 0%**
///
/// `served / slots` 在 `slots == 0` 时不是「利用率 0%」，而是**没有意义**：那种根一次
/// 推理都没发过。把它当 0% 会压低最低填充率，让统计说谎。
fn fill_text(served: usize, slots: usize) -> String {
    if slots == 0 {
        "N/A（零推理格位）".to_string()
    } else {
        format!("{:.1}%", 100.0 * served as f64 / slots as f64)
    }
}

/// 从侧车就绪行里取一个 `key=value` 的字符串字段
///
/// 直接读**实际写出来的字段**，不对整行做任何指纹比较。
fn banner_word<'a>(banner: &'a str, key: &str) -> Option<&'a str> {
    banner.split_whitespace().find_map(|tok| tok.strip_prefix(key))
}

/// 从侧车就绪行里取一个 `key=value` 的整数字段
///
/// 直接读**实际写出来的字段**，不对整行做任何指纹比较。
fn banner_u64(banner: &str, key: &str) -> Option<u64> {
    banner
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix(key))
        .and_then(|v| v.parse::<u64>().ok())
}

/// 按本根的总 rollout 数选物理批档位
///
/// 规则：取**不超过总 rollout 数**的最大档位；总数低于最小档时仍用最小档（尾部补零）；
/// 任何情况下不超过侧车分配的上限 `max_batch`。
///
/// ❗这是待测启发式，不是最优解——上一轮实测同一档位在不同根上有快有慢。
fn plan_batch(adaptive: bool, max_batch: usize, total_rollouts: usize) -> usize {
    if !adaptive {
        return max_batch;
    }
    let mut pick = BATCH_TIERS[0].min(max_batch).max(1);
    for t in BATCH_TIERS {
        if t <= max_batch && t <= total_rollouts && t > pick {
            pick = t;
        }
    }
    pick
}

/// 常驻侧车的客户端
struct Sidecar {
    /// 子进程句柄；`Drop` 时关闭 stdin 让侧车自行退出
    child: Child,
    /// 侧车启动时分配的最大物理批尺寸；运行期不变，不为换档重启进程或重载模型
    max_batch: usize,
    /// 当前生效的物理张量批尺寸（`<= max_batch`）
    ///
    /// ❗与调度器的「活跃轨迹容量」是**两个概念**：容量决定一波最多挂起多少条 rollout，
    /// 本字段决定 GPU 前向实际算多少行。二者数值可以相同，含义不同。
    active_batch: usize,
    /// 档位切换次数
    batch_switches: usize,
    /// 档位切换（含侧车首次见到该档位时的预热）累计墙钟，**单列，不进稳态计时**
    batch_setup_s: f64,
    /// 排空线程保留的最近若干行 stderr
    diag: Arc<Mutex<VecDeque<String>>>,
    /// stderr 排空线程；侧车退出后读到 EOF 自行结束
    ///
    /// ❗**必须持续排空**：只持有不读会让 stderr 管道缓冲区填满，侧车写日志时
    /// 阻塞在写上，我方则永远等不到 stdout，形成死锁。
    drain: Option<JoinHandle<()>>,
    /// 进程启动到就绪的墙钟（模型加载 + 预热），**单独报告，不进性能窗口**
    startup_s: f64,
    /// 侧车自报的就绪行（含 device 与数值配置）
    banner: String
}

impl Sidecar {
    /// 启动侧车并等待就绪握手
    ///
    /// # 错误
    ///
    /// 缺参数、进程启动失败，或侧车在写出就绪标记前退出时报错。
    fn start(args: &RootArgs) -> Result<Self> {
        let script = args
            .sidecar
            .as_ref()
            .ok_or_else(|| anyhow!("compare 模式需要 --sidecar"))?;
        ensure!(!args.checkpoint.is_empty(), "compare 模式需要至少一个 --checkpoint");
        let mut cmd = Command::new(&args.python);
        cmd.arg(script);
        for ck in &args.checkpoint {
            cmd.arg("--checkpoint").arg(ck);
        }
        cmd.arg("--batch").arg(args.batch.to_string());
        if args.sidecar_graph {
            cmd.arg("--cuda-graph");
        }
        // 验收口径：不给 --tf32，走严格 FP32
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let t0 = Instant::now();
        let mut child = cmd.spawn().with_context(|| format!("启动侧车失败: {}", script.display()))?;
        let raw = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("侧车 stderr 不可用"))?;
        let mut stderr = BufReader::new(raw);

        // 阻塞等待就绪：在此之前不发任何请求，模型加载与预热因而不会混进推理计时
        let mut banner = String::new();
        loop {
            let mut line = String::new();
            let read = stderr.read_line(&mut line).context("读侧车 stderr 失败")?;
            if read == 0 {
                let _ = child.kill();
                bail!("侧车在写出就绪标记前退出；已收到的输出：\n{banner}");
            }
            banner.push_str(&line);
            if line.contains(SIDECAR_READY_MARK) {
                banner = line.trim_end().to_string();
                break;
            }
        }
        // 握手之后交给排空线程：标准库线程即可，不引入新依赖
        let diag: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        let sink = Arc::clone(&diag);
        let drain = std::thread::Builder::new()
            .name("sidecar-stderr".to_string())
            .spawn(move || {
                let mut rdr = stderr;
                let mut line = String::new();
                loop {
                    line.clear();
                    match rdr.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            if let Ok(mut q) = sink.lock() {
                                if q.len() == SIDECAR_DIAG_LINES {
                                    q.pop_front();
                                }
                                q.push_back(line.trim_end().to_string());
                            }
                        }
                    }
                }
            })
            .context("启动侧车 stderr 排空线程失败")?;

        let mut sc = Self {
            child,
            max_batch: args.batch,
            active_batch: args.batch,
            batch_switches: 0,
            batch_setup_s: 0.0,
            diag,
            drain: Some(drain),
            startup_s: t0.elapsed().as_secs_f64(),
            banner
        };
        if args.sidecar_graph {
            // 不能只看我们传了什么：必须由侧车自报已开图，否则开关就是被静默吞掉
            let got = banner_word(&sc.banner, "graph=").ok_or_else(|| {
                anyhow!("侧车就绪行没有 graph= 字段，无法确认 CUDA Graph 已开：{}", sc.banner)
            })?;
            ensure!(
                got == "on",
                "已给 --sidecar-graph，但侧车自报 graph={got}；换支持该开关的侧车，或去掉开关"
            );
        }
        if args.adaptive_batch {
            // 两端同步校验：协议版本与侧车**实际分配**的缓冲区尺寸都读自就绪行
            let proto = banner_u64(&sc.banner, "proto=")
                .ok_or_else(|| anyhow!("侧车就绪行没有 proto= 字段，无法开启自适应批：{}", sc.banner))?;
            ensure!(proto == SIDECAR_PROTO, "侧车协议 {proto} 与本工具要求的 {SIDECAR_PROTO} 不符");
            let max_b = banner_u64(&sc.banner, "maxB=")
                .ok_or_else(|| anyhow!("侧车就绪行没有 maxB= 字段：{}", sc.banner))?;
            ensure!(
                max_b as usize == args.batch,
                "侧车分配的缓冲区 {max_b} 与 --batch {} 不符",
                args.batch
            );
            // 会用到的档位在启动窗口里全部预热完；之后换档只剩一次控制往返
            for tier in BATCH_TIERS {
                if tier <= args.batch {
                    sc.set_batch(tier)?;
                }
            }
            sc.set_batch(args.batch)?;
            sc.startup_s = t0.elapsed().as_secs_f64();
            sc.batch_switches = 0;
            sc.batch_setup_s = 0.0;
        }
        Ok(sc)
    }

    /// 切换侧车的物理张量批尺寸（同一进程、同一份模型，不重启不重载）
    ///
    /// 走**显式控制消息**而不是重新解释数据头；侧车首次见到某个档位会先预热再应答，
    /// 故本函数耗时单列在 `batch_setup_s`，不进稳态计时。
    ///
    /// # 错误
    ///
    /// 越界、管道读写失败或回执与请求值不符时报错。
    fn set_batch(&mut self, p: usize) -> Result<()> {
        ensure!(0 < p && p <= self.max_batch, "物理批 {p} 越界（侧车上限 {}）", self.max_batch);
        let t0 = Instant::now();
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("侧车 stdin 已关闭"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(12);
        buf.extend_from_slice(&CTRL_MAGIC.to_le_bytes());
        buf.extend_from_slice(&OP_SET_BATCH.to_le_bytes());
        buf.extend_from_slice(&(p as u32).to_le_bytes());
        stdin.write_all(&buf).context("写侧车控制消息失败")?;
        stdin.flush().context("刷新侧车 stdin 失败")?;
        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("侧车 stdout 已关闭"))?;
        let mut ack = [0u8; 4];
        let read_err = stdout.read_exact(&mut ack).err();
        if let Some(e) = read_err {
            bail!("读侧车控制回执失败: {e}\n侧车最近的 stderr：\n{}", self.diagnostics());
        }
        let got = u32::from_le_bytes(ack) as usize;
        ensure!(got == p, "侧车回执的物理批 {got} 与请求 {p} 不符");
        self.active_batch = p;
        self.batch_switches += 1;
        self.batch_setup_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// 需要时才切档；已经在目标档位上则**一个字节都不发**
    ///
    /// # 错误
    ///
    /// 切档失败时报错。
    fn ensure_batch(&mut self, p: usize) -> Result<()> {
        if p == self.active_batch {
            return Ok(());
        }
        self.set_batch(p)
    }

    /// 侧车最近的 stderr 输出，用于给通信错误补上下文
    fn diagnostics(&self) -> String {
        match self.diag.lock() {
            Ok(q) if q.is_empty() => "（侧车 stderr 无后续输出）".to_string(),
            Ok(q) => q.iter().cloned().collect::<Vec<_>>().join("\n"),
            Err(_) => "（诊断锁被毒化）".to_string()
        }
    }

    /// 送一批有效行，取回同样行数的输出
    ///
    /// # 错误
    ///
    /// 行数越界、管道读写失败或返回字节数不符时报错。
    fn infer(&mut self, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        ensure!(!rows.is_empty(), "不得向侧车发送空推理请求");
        ensure!(
            rows.len() <= self.active_batch,
            "一批 {} 行超过当前物理批尺寸 {}",
            rows.len(),
            self.active_batch
        );
        let valid = rows.len() as u32;
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("侧车 stdin 已关闭"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(8 + rows.len() * INPUT_DIM * 4);
        buf.extend_from_slice(&valid.to_le_bytes());
        buf.extend_from_slice(&valid.to_le_bytes());
        for r in rows {
            ensure!(r.len() == INPUT_DIM, "输入行长 {} 与 {INPUT_DIM} 不符", r.len());
            for v in r {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        stdin.write_all(&buf).context("写侧车 stdin 失败")?;
        stdin.flush().context("刷新侧车 stdin 失败")?;

        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("侧车 stdout 已关闭"))?;
        let mut out = vec![0u8; rows.len() * OUTPUT_DIM * 4];
        // 先结束对 child 的可变借用，再取诊断（诊断要不可变借 self）
        let read_err = stdout.read_exact(&mut out).err();
        if let Some(e) = read_err {
            bail!("读侧车输出失败: {e}\n侧车最近的 stderr：\n{}", self.diagnostics());
        }
        Ok(out
            .chunks_exact(OUTPUT_DIM * 4)
            .map(|row| {
                row.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
            .collect())
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // 关掉 stdin，侧车读到 EOF 自行退出；杀进程会丢掉它的 stderr
        drop(self.child.stdin.take());
        let _ = self.child.wait();
        // 侧车退出后 stderr 读到 EOF，排空线程自行结束
        if let Some(h) = self.drain.take() {
            let _ = h.join();
        }
    }
}

/// 一条对拍记录
struct CompareRow {
    /// rollout 身份
    ctx: RolloutCtx,
    /// 决策发生的回合
    turn: i32,
    /// 决策发生的阶段
    stage: String,
    /// 合法候选数
    n_actions: usize,
    /// 决策 RNG 探针：决策 RNG 克隆体的下一个 u64 **实际取值**（不是内容指纹）
    rng_probe: u64,
    /// CPU 侧赢家下标
    cpu_winner: usize,
    /// GPU 侧赢家下标
    gpu_winner: usize,
    /// CPU 侧候选打分的 top1−top2 分差
    cpu_margin: f32,
    /// 两端**候选分数**的最大绝对差
    ///
    /// ❗不是 policy 单格误差：RegionSelect 要三格求和、吃面动作有自己的映射，
    /// 单格误差不能通用地当作赢家稳定阈值。这里两端都过一遍 `score_actions`。
    action_delta: f32,
    /// 「上一次推理返回 → 本次进入 prepare」的间隔
    ///
    /// ❗含规则推进、特征编码与调度残余，**不是纯规则推进耗时**。
    gap_us: f64
}

/// 一条 rollout 的汇总
struct RolloutRow {
    /// 候选下标
    candidate: usize,
    /// rollout 序号
    j: usize,
    /// 该 rollout 实际发给侧车的推理请求数（可能为 0；缓存命中不计）
    requests: u32,
    /// 该 rollout 的逻辑网络决策数（缓存命中也计）
    ///
    /// 不开缓存时与 `requests` 恒等；开缓存后两者的差就是复用掉的请求。
    decisions: u32
}

/// 截断实验里一条 rollout 的结果（**带类型标记**）
///
/// 与 [`RawCell`] 分开的理由：`RawCell` 恒是真实终局，两个分数都有值；本类型要同时
/// 承载叶预测，缺的维度按**缺失**处理，不填 0、不伪造 [`RamenTerminal`]。
#[derive(Debug, Clone)]
struct ArmCell {
    /// 候选下标
    candidate: usize,
    /// rollout 序号 `j`
    j: usize,
    /// 该 rollout 的种子
    seed: u64,
    /// 结果类型：`terminal` / `leaf`
    kind: &'static str,
    /// 进入均值聚合的那一个标量（终局 `score` 或叶 `value[0]`）
    value: f64,
    /// PT 口径评分；**叶没有**，故为 `None`（不是 0）
    score_pt: Option<f64>,
    /// 诊断模式下，**同一条轨迹**从叶状态续跑到终局的真实分；否则 `None`
    ///
    /// ❗这是真实终局统计，与 `value`（网络预测）语义不同，列名与含义都不得混用。
    paired_terminal: Option<f64>,
    /// 叶所在回合；终局为 `None`
    leaf_turn: Option<i32>,
    /// 叶所在阶段；终局为 `None`
    leaf_stage: Option<String>,
    /// 相对根回合实际跨越的 turn 数；终局为 `None`
    turn_delta: Option<i32>,
    /// 到达目标回合后的顺延次数；终局为 `None`
    deferrals: Option<u32>
}

/// 一条 rollout 的原始结果
struct RawCell {
    /// 候选下标
    candidate: usize,
    /// rollout 序号
    j: usize,
    /// 该 rollout 的种子（`seed_at(j)`，不吃候选下标）
    seed: u64,
    /// 终局评分
    score: f64,
    /// 计入 PT 偏好的终局评分
    score_pt: f64
}

/// 一次网络决策的完整记录
///
/// 用于两条执行路径（CPU 串行 rollout / GPU 波次驱动）之间的**逐决策等价性**比较。
/// ❗一律落**实际值**，不落哈希：指纹相同只说明字节相同，不同则看不出差在哪。
struct DecisionRow {
    /// 候选下标
    candidate: usize,
    /// rollout 序号 `j`
    j: usize,
    /// 该 rollout 内的网络决策序号（从 0 起）
    seq: u32,
    /// 决策发生的回合
    turn: i32,
    /// 决策发生的阶段
    stage: String,
    /// 合法候选，保持内核给出的**原顺序**
    actions: Vec<String>,
    /// 选中的候选下标
    chosen: usize,
    /// 该决策的完整输入特征
    features: Vec<f32>
}

/// 把动作渲染成不含分隔符的可比较文本
///
/// 不用 `Debug`：它会带逗号，落进 CSV 需要再转义。这里只取参与身份的三个字段。
fn action_repr(a: &RamenAction) -> String {
    let ramen = a.ramen.map_or_else(|| "-".to_string(), |x| x.to_string());
    let targets = a
        .special_targets
        .map_or_else(|| "-".to_string(), |t| format!("{}+{}+{}", t[0], t[1], t[2]));
    format!("{ramen}/{targets}/{:?}", a.operation).replace([',', '|'], ";")
}

/// CPU 路径的逐决策记录端
#[derive(Default)]
struct DecisionLogSink {
    /// 已记录的决策
    rows: Mutex<Vec<DecisionRow>>
}

impl RolloutInferSink for DecisionLogSink {
    fn on_inferred(&self, snap: RolloutInferSnapshot) -> Result<()> {
        let ctx = ROLLOUT_CTX.with(|c| {
            let mut c = c.borrow_mut();
            let ctx = c
                .as_mut()
                .ok_or_else(|| anyhow!("rollout 上下文缺失：录制必须由实验入口绑定"))?;
            let taken = *ctx;
            ctx.seq += 1;
            Ok::<_, anyhow::Error>(taken)
        })?;
        let mut rows = self.rows.lock().map_err(|_| anyhow!("决策记录锁被毒化"))?;
        rows.push(DecisionRow {
            candidate: ctx.candidate,
            j: ctx.j,
            seq: ctx.seq,
            turn: snap.turn,
            stage: format!("{:?}", snap.stage),
            actions: snap.actions.iter().map(action_repr).collect(),
            chosen: snap.cpu_winner,
            features: snap.features
        });
        Ok(())
    }

    fn on_resolved(&self, _turn: i32, _stage: RamenStage, _winner: usize) -> Result<()> {
        // 未经网络的决策不进逐决策比较：波次路径同样只在网络决策处暂停。
        //
        // ❗因此本比较的覆盖范围仅限**网络决策点**：守门、转交手写的决策与事件选项
        // 的完整序列**没有被直接核验**。不要反过来推「上游分歧必然在下一次网络输入上
        // 暴露」——特征编码未必覆盖完整状态，两条路径也可能重新汇合，
        // 而最后一次网络请求之后根本没有后续输入可查。
        Ok(())
    }
}

/// 攒批并当场对拍的接收端
struct CompareSink {
    /// 待对拍的快照（连同其 rollout 身份与间隔）
    pending: Mutex<Vec<(RolloutInferSnapshot, RolloutCtx, f64)>>,
    /// 侧车（一问一答，故整体加锁）
    sidecar: Mutex<Sidecar>,
    /// 用于复用 `score_actions` / `resolve_decision` 的 CPU 侧策略
    nn: Arc<RamenNnTrainer>,
    /// 物理批尺寸
    batch: usize,
    /// 对拍结果
    rows: Mutex<Vec<CompareRow>>,
    /// 未经网络即定案的决策点计数（守门 / 转交手写）
    resolved: Mutex<usize>
}

impl CompareSink {
    /// 对一批快照做 GPU 推理与赢家比较
    ///
    /// # 错误
    ///
    /// 侧车通信失败、返回行数不符或打分失败时报错。
    fn compare_batch(&self, batch: Vec<(RolloutInferSnapshot, RolloutCtx, f64)>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let features: Vec<Vec<f32>> = batch.iter().map(|(s, _, _)| s.features.clone()).collect();
        let outs = {
            let mut sc = self.sidecar.lock().map_err(|_| anyhow!("侧车锁被毒化"))?;
            sc.infer(&features)?
        };
        ensure!(outs.len() == batch.len(), "侧车返回 {} 行，期望 {}", outs.len(), batch.len());

        let mut rows = self.rows.lock().map_err(|_| anyhow!("结果锁被毒化"))?;
        for ((snap, ctx, gap_us), out) in batch.iter().zip(outs.iter()) {
            let gpu_policy = &out[..POLICY_DIM];
            let gpu_winner = self.nn.resolve_decision(&snap.game, &snap.actions, gpu_policy)?;
            // 两端都过 score_actions：三格求和与吃面映射的影响自动包含在内
            let cpu_scores: Vec<f32> = self
                .nn
                .score_actions(&snap.game, &snap.actions, &snap.cpu_policy)?
                .iter()
                .map(|s| s.logit)
                .collect();
            let gpu_scores: Vec<f32> = self
                .nn
                .score_actions(&snap.game, &snap.actions, gpu_policy)?
                .iter()
                .map(|s| s.logit)
                .collect();
            let action_delta = cpu_scores
                .iter()
                .zip(gpu_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mut sorted = cpu_scores.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let cpu_margin = if sorted.len() >= 2 { sorted[0] - sorted[1] } else { f32::INFINITY };
            rows.push(CompareRow {
                ctx: *ctx,
                turn: snap.turn,
                stage: format!("{:?}", snap.stage),
                n_actions: snap.actions.len(),
                rng_probe: snap.rng_probe,
                cpu_winner: snap.cpu_winner,
                gpu_winner,
                cpu_margin,
                action_delta,
                gap_us: *gap_us
            });
        }
        Ok(())
    }

    /// 排空残批
    ///
    /// # 错误
    ///
    /// 锁被毒化或对拍失败时报错。
    fn drain(&self) -> Result<()> {
        let rest = {
            let mut pending = self.pending.lock().map_err(|_| anyhow!("快照锁被毒化"))?;
            std::mem::take(&mut *pending)
        };
        self.compare_batch(rest)
    }
}

impl RolloutInferSink for CompareSink {
    fn on_inferred(&self, snap: RolloutInferSnapshot) -> Result<()> {
        let ctx = ROLLOUT_CTX.with(|c| {
            let mut c = c.borrow_mut();
            let ctx = c
                .as_mut()
                .ok_or_else(|| anyhow!("rollout 上下文缺失：录制必须由实验入口绑定"))?;
            let taken = *ctx;
            ctx.seq += 1;
            Ok::<_, anyhow::Error>(taken)
        })?;
        let gap_us = LAST_INFER_END.with(|last| match last.borrow_mut().replace(snap.infer_end) {
            Some(p) if snap.prep_start > p => snap.prep_start.duration_since(p).as_secs_f64() * 1e6,
            _ => 0.0
        });

        // ❗加入、判满、取走必须在**同一次持锁**内完成：
        // 分成两次持锁的话，另一个 worker 可以在中间插入第 513 行，
        // 取走时就会超过物理批尺寸，侧车入口直接报错。
        let full = {
            let mut pending = self.pending.lock().map_err(|_| anyhow!("快照锁被毒化"))?;
            pending.push((snap, ctx, gap_us));
            if pending.len() >= self.batch {
                Some(std::mem::take(&mut *pending))
            } else {
                None
            }
        };
        if let Some(batch) = full {
            self.compare_batch(batch)?;
        }
        Ok(())
    }

    fn on_resolved(&self, _turn: i32, _stage: RamenStage, _winner: usize) -> Result<()> {
        let mut r = self.resolved.lock().map_err(|_| anyhow!("计数锁被毒化"))?;
        *r += 1;
        Ok(())
    }
}

/// 把一局推进到指定回合的第一个决策点，作为固定根
///
/// # 错误
///
/// 建局失败、推进中规则层报错，或到终局仍未命中目标时报错。
fn build_root(
    args: &RootArgs, uma: u32, deck: &[u32; 6], inherit: &InheritInfo, root_turn: i32, root_stage: Option<&str>
) -> Result<(RamenGame, StdRng)> {
    build_root_with(args, uma, deck, inherit, root_turn, root_stage, &RootGuide::Handwritten(
        RecommendedRamenTrainer::for_rollout()
    ), args.run_idx)
}

/// 建根时用哪套策略推进
///
/// 做成枚举而不是 `&dyn Trainer`：[`Game::run_stage`] 是泛型方法，trait object
/// 过不去。两个变体都**原样转发**，不在这里复制任何决策逻辑。
enum RootGuide {
    /// 手写推荐策略（既有行为）
    Handwritten(RecommendedRamenTrainer),
    /// 冻结的纯网络策略
    Nn(Arc<RamenNnTrainer>)
}

impl Trainer<RamenGame> for RootGuide {
    /// 原样转发给所选策略
    ///
    /// # 错误
    ///
    /// 被转发的策略报错时原样返回。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self {
            RootGuide::Handwritten(t) => t.select_action(game, actions, rng),
            RootGuide::Nn(t) => t.select_action(game, actions, rng)
        }
    }

    /// 原样转发给所选策略
    ///
    /// # 错误
    ///
    /// 被转发的策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        match self {
            RootGuide::Handwritten(t) => t.select_choice(game, choices, rng),
            RootGuide::Nn(t) => t.select_choice(game, choices, rng)
        }
    }

    /// 原样转发给所选策略
    ///
    /// # 错误
    ///
    /// 被转发的策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        match self {
            RootGuide::Handwritten(t) => t.select_event_choice(game, event, choices, rng),
            RootGuide::Nn(t) => t.select_event_choice(game, event, choices, rng)
        }
    }
}

/// 建根：用指定策略推进到目标根
///
/// 与 [`build_root`] 同一套阶段判据，只是把推进策略与局号参数化。
///
/// # 错误
///
/// 建局失败、规则层报错，或推进到终局仍未命中目标根时报错。
fn build_root_with(
    args: &RootArgs, uma: u32, deck: &[u32; 6], inherit: &InheritInfo, root_turn: i32, root_stage: Option<&str>,
    guide: &RootGuide, run_idx: u64
) -> Result<(RamenGame, StdRng)> {
    let (mut rng, rule_master) = seeded_rngs(args.seed, run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let want = root_stage;
    while game.next() {
        let is_decision = matches!(
            game.stage,
            RamenStage::Train
                | RamenStage::RamenSelect
                | RamenStage::SpecialSelect
                | RamenStage::RegionSelect
                | RamenStage::SuperRamenSelect
        );
        let stage_ok = match want {
            Some(w) => format!("{:?}", game.stage).eq_ignore_ascii_case(w),
            None => true
        };
        if is_decision && stage_ok && game.turn() >= root_turn {
            return Ok((game, rng));
        }
        game.run_stage(guide, &mut rng)?;
    }
    bail!("推进到终局仍未命中目标根（回合 >= {root_turn}，阶段 {want:?}）")
}

/// 扫描一局里的**全部**决策点
///
/// 只走局、不搜索：把「Train-早 / RegionSelect-Y2」这类配额解析成具体
/// `(root_turn, root_stage)`，在看任何搜索结果之前完成。
///
/// # 错误
///
/// 建局或规则推进失败时报错。
fn scan_decisions(
    args: &RootArgs, uma: u32, deck: &[u32; 6], inherit: &InheritInfo, guide: &RootGuide, run_idx: u64
) -> Result<Vec<DecisionPoint>> {
    let (mut rng, rule_master) = seeded_rngs(args.seed, run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let mut out = Vec::new();
    let mut seq = 0usize;
    while game.next() {
        if is_decision_stage(&game.stage) {
            let actions = game.list_actions()?;
            out.push(DecisionPoint {
                seq,
                turn: game.turn(),
                stage: format!("{:?}", game.stage),
                candidates: actions.len()
            });
            seq += 1;
        }
        game.run_stage(guide, &mut rng)?;
    }
    Ok(out)
}

/// 一局里的一个决策点（[`Mode::RootScan`] 的产物）
#[derive(Debug, Clone, Serialize)]
struct DecisionPoint {
    /// 局内决策序号（从 0 开始）
    seq: usize,
    /// 回合
    turn: i32,
    /// 阶段
    stage: String,
    /// 该决策点的合法候选数
    candidates: usize
}

/// 逐字段比较两批原始结果，返回（键不一致条数, 值不同条数, 最大绝对差）
///
/// 两批都已按 `(候选, j)` 排好序，故按下标对齐即可；键本身也参与比较。
/// ❗不做哈希：不同的时候要能看出差在哪一条、差多少。
fn diff_cells(a: &[RawCell], b: &[RawCell]) -> (usize, usize, f64) {
    if a.len() != b.len() {
        return (a.len().abs_diff(b.len()), 0, f64::NAN);
    }
    let mut key_bad = 0usize;
    let mut val_bad = 0usize;
    let mut max_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.candidate != y.candidate || x.j != y.j || x.seed != y.seed {
            key_bad += 1;
            continue;
        }
        if x.score != y.score || x.score_pt != y.score_pt {
            val_bad += 1;
            max_abs = max_abs.max((x.score - y.score).abs()).max((x.score_pt - y.score_pt).abs());
        }
    }
    (key_bad, val_bad, max_abs)
}

/// 常驻侧车复用测试：同一侧车连续跑「根 A → 根 B → 根 A」
///
/// 两次 A 之间夹一个形状不同的根 B；若侧车有请求残留或状态串扰，
/// 第二次 A 会与第一次不同。
///
/// # 错误
///
/// 建根、波次驱动或官方汇总失败时报错。
fn run_sidecar_reuse(
    args: &RootArgs, search: &FlatSearch<RamenGame>, nn: &RamenNnTrainer, uma: u32, deck: &[u32; 6],
    inherit: &InheritInfo
) -> Result<()> {
    let (root_a, rng_a) = build_root(args, uma, deck, inherit, args.root_turn, args.root_stage.as_deref())?;
    let (root_b, rng_b) = build_root(args, uma, deck, inherit, args.root_b_turn, args.root_b_stage.as_deref())?;
    let acts_a = root_a.list_actions()?;
    let acts_b = root_b.list_actions()?;
    println!(
        "  根 A  t{} {:?} 候选 {}    根 B  t{} {:?} 候选 {}",
        root_a.turn(),
        root_a.stage,
        acts_a.len(),
        root_b.turn(),
        root_b.stage,
        acts_b.len()
    );

    let mut sc = Sidecar::start(args)?;
    println!("  侧车就绪 {:.1} s | {}（此后不再重启）", sc.startup_s, sc.banner);

    let mut once = |game: &RamenGame, actions: &[RamenAction], rng: &StdRng, label: &str| -> Result<Vec<RawCell>> {
        let mut rng = rng.clone();
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let batch = plan_batch(args.adaptive_batch, args.batch, actions.len() * args.search_n);
        let t0 = Instant::now();
        // 本模式不做截断：`LeafDepth::Full` 保持既有行为逐字不变
        let out = run_gpu_wave(
            game,
            actions,
            args.search_n,
            &seeds,
            nn,
            &mut sc,
            batch,
            false,
            args.policy_cache,
            LeafDepth::Full,
            false
        )?;
        let (cells, st) = (out.raw_cells()?, &out.stats);
        let (best, _) = official_result(search, game, actions, &cells, &mut rng)?;
        println!(
            "  [{label}] {:.1} s 档位 B{} 波次 {} 请求 {} 利用率 {} 最优候选 {best}",
            t0.elapsed().as_secs_f64(),
            st.batch,
            st.waves,
            st.served,
            fill_text(st.served, st.slots)
        );
        println!("        复用 {}", st.cache.line());
        Ok(cells)
    };

    let a1 = once(&root_a, &acts_a, &rng_a, "A 第一次")?;
    let b = once(&root_b, &acts_b, &rng_b, "B")?;
    let a2 = once(&root_a, &acts_a, &rng_a, "A 第二次")?;
    ensure!(!b.is_empty(), "根 B 没有产出结果");

    let (key_bad, val_bad, max_abs) = diff_cells(&a1, &a2);
    println!("  两次 A 逐字段比较：{} 条 vs {} 条", a1.len(), a2.len());
    println!("    键不一致 {key_bad} 条，值不同 {val_bad} 条，最大绝对差 {max_abs:.3e}");
    ensure!(key_bad == 0 && val_bad == 0, "侧车复用后两次 A 的结果不一致：存在请求残留或状态串扰");
    println!("  ✅ 同一侧车跨根复用未出现请求残留或状态串扰");
    Ok(())
}

/// 正式接入一致性：生产教师在 CPU / GPU 两种后端下打同一局，逐步比较
///
/// ❗用**少量样本**（小 `search_n`）：这是接线一致性检查，不是性能测量。
///
/// # 错误
///
/// 任一臂失败时报错；两臂决策不一致时报错。
fn run_teacher_consistency(
    args: &RootArgs, config: &SearchConfig, nn: &Arc<RamenNnTrainer>, uma: u32, deck: &[u32; 6],
    inherit: &InheritInfo
) -> Result<()> {
    println!("  正式接入一致性：生产教师，search_n={}（少量样本）", args.search_n);

    let t_cpu = build_teacher(config.clone(), Some(Arc::clone(nn)), None);
    let (steps_cpu, score_cpu, wall_cpu) = play_with_teacher(args, uma, deck, inherit, t_cpu)?;
    println!("  [CPU 后端] {:.1} s，决策 {} 步，终局 {:.3}", wall_cpu, steps_cpu.len(), score_cpu.score);

    let sidecar = Sidecar::start(args)?;
    println!("  侧车就绪 {:.1} s | {}", sidecar.startup_s, sidecar.banner);
    let backend = Arc::new(WaveBackend {
        nn: Arc::clone(nn),
        sidecar: Mutex::new(sidecar),
        max_batch: args.batch,
        adaptive: args.adaptive_batch,
        cache_on: args.policy_cache,
        stats: Mutex::new(GameStats::default())
    });
    let t_gpu = build_teacher(config.clone(), Some(Arc::clone(nn)), Some(Arc::clone(&backend) as Arc<dyn RamenBatchRollout>));
    let (steps_gpu, score_gpu, wall_gpu) = play_with_teacher(args, uma, deck, inherit, t_gpu)?;
    println!("  [GPU 后端] {:.1} s，决策 {} 步，终局 {:.3}", wall_gpu, steps_gpu.len(), score_gpu.score);

    let bad = diff_steps(&steps_cpu, &steps_gpu);
    println!("  逐步比较：{} 步中 {} 步不一致", steps_cpu.len(), bad);
    ensure!(bad == 0, "生产教师在两种后端下的决策不一致");
    ensure!(
        score_cpu.score == score_gpu.score && score_cpu.score_pt == score_gpu.score_pt,
        "终局评分不一致：{:?} vs {:?}",
        score_cpu,
        score_gpu
    );
    println!("  ✅ 逐步动作、候选顺序、RNG 探针与终局评分全部相同");
    Ok(())
}

/// 正式配置整局：生产教师 + 波次后端
///
/// # 错误
///
/// 建局、规则推进或后端失败时报错。
/// 打印**实际执行搜索的那个** `FlatSearch` 持有的配置
///
/// 取自教师内部的搜索器而非 CLI 参数：`build_teacher` 会重建搜索器，
/// 打印 CLI 值证明不了内核最终生效的是什么。
fn print_effective_config(teacher: &RamenMctsTrainer) {
    let c = teacher.search.config();
    println!(
        "  生效搜索配置 search_n={} use_ucb={} radical_factor_max={}（读自教师内部的 FlatSearch）",
        c.search_n, c.use_ucb, c.radical_factor_max
    );
}

fn run_teacher_game(
    args: &RootArgs, config: &SearchConfig, nn: &Arc<RamenNnTrainer>, uma: u32, deck: &[u32; 6],
    inherit: &InheritInfo, model_load_s: f64
) -> Result<()> {
    println!("  模型加载    {model_load_s:.1} s（不计入整局墙钟）");
    if args.handwritten_rollout {
        // 配对实验的手写臂：不接网络、不接侧车，其余口径完全相同
        let teacher = build_teacher(config.clone(), None, None);
        print_effective_config(&teacher);
        let (steps, score, wall) = play_with_teacher(args, uma, deck, inherit, teacher)?;
        println!("  手写 rollout 基策（配对实验另一臂）");
        println!("  整局墙钟    {wall:.1} s");
        println!("  教师决策    {} 步", steps.len());
        println!("  终局评分    score={:.3} score_pt={:.3}", score.score, score.score_pt);
        println!("  终局全精度  score={:.17e} score_pt={:.17e}", score.score, score.score_pt);
        if let Some(p) = args.steps_csv.as_ref() {
            write_steps_csv(p, &steps)?;
            println!("  逐步决策    {} 步 → {}", steps.len(), p.display());
        }
        return Ok(());
    }
    let sidecar = Sidecar::start(args)?;
    println!("  侧车就绪 {:.1} s | {}（整局复用，不计入整局墙钟）", sidecar.startup_s, sidecar.banner);
    let backend = Arc::new(WaveBackend {
        nn: Arc::clone(nn),
        sidecar: Mutex::new(sidecar),
        max_batch: args.batch,
        adaptive: args.adaptive_batch,
        cache_on: args.policy_cache,
        stats: Mutex::new(GameStats::default())
    });
    let teacher = build_teacher(
        config.clone(),
        Some(Arc::clone(nn)),
        Some(Arc::clone(&backend) as Arc<dyn RamenBatchRollout>)
    );
    print_effective_config(&teacher);
    let (steps, score, wall) = play_with_teacher(args, uma, deck, inherit, teacher)?;

    let st = backend.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
    println!("  整局墙钟    {wall:.1} s（其中后端 {:.1} s）", st.search_s);
    println!("  教师决策    {} 步（其中触发搜索 {} 次）", steps.len(), st.searched);
    println!("  平均候选数  {:.1}", st.candidates as f64 / st.searched.max(1) as f64);
    println!(
        "  请求 / 格位 {} / {}（整局批利用率 {}）",
        st.served,
        st.slots,
        fill_text(st.served, st.slots)
    );
    println!(
        "  ❗单次搜索最低批利用率 {}；零请求根 {} 个（不参与该统计）",
        st.min_fill_text(),
        st.zero_request_roots
    );
    println!(
        "  物理批档位  {}（上限 {}，自适应 {}）；换档与档位预热累计 {:.2} s（含在整局墙钟内，照实单列）",
        tier_line(&st.tiers),
        args.batch,
        if args.adaptive_batch { "开" } else { "关" },
        st.setup_s
    );
    println!(
        "  policy 复用 {} | {}",
        if args.policy_cache { "开" } else { "关" },
        st.cache.line()
    );
    println!("  波次        {}", st.waves);
    println!("  rollout 总数 {}", st.rollouts);
    println!(
        "  后端拆分    CPU 段 {:.1} s | 侧车往返 {:.1} s（轨迹自报 CPU 累加 {:.1} s，跨 worker，不可加进墙钟）",
        st.cpu_wall_s, st.gpu_wall_s, st.cpu_in_traj_s
    );
    println!("  ❗「侧车往返」是 infer() 整段：含序列化/解码、两次管道、主机↔设备拷贝与前向，未再拆分");
    println!("  终局评分    score={:.3} score_pt={:.3}", score.score, score.score_pt);
    println!("  终局全精度  score={:.17e} score_pt={:.17e}", score.score, score.score_pt);
    if let Some(p) = args.steps_csv.as_ref() {
        write_steps_csv(p, &steps)?;
        println!("  逐步决策    {} 步 → {}", steps.len(), p.display());
    }
    println!("  ❗单局不回答策略优劣；且**手写臂不按 NN 成本计价**，预算要分臂估。");
    Ok(())
}

/// 整局冒烟：一局完整的波次驱动搜索
///
/// 建局与随机流走 [`seeded_rngs`]，与 `ramen_space_bench` 同一条路径；
/// 每个决策点的 CRN 与官方汇总口径都不变。
///
/// # 错误
///
/// 建局、规则推进、波次驱动或官方汇总失败时报错。
fn run_game_smoke(
    args: &RootArgs, search: &FlatSearch<RamenGame>, nn: &RamenNnTrainer, uma: u32, deck: &[u32; 6],
    inherit: &InheritInfo, game_route: LeafRoute
) -> Result<()> {
    println!("  整局路由   {}", game_route.tag());
    println!("             {}", game_route.describe());
    if let Some(p) = &args.route_csv {
        ensure!(!p.exists(), "逐根路由 CSV 已存在，拒绝覆盖：{}", p.display());
    }
    let sidecar = Sidecar::start(args)?;
    println!("  侧车就绪 {:.1} s | {}（整局复用）", sidecar.startup_s, sidecar.banner);

    let build = Instant::now();
    let (mut rng, rule_master) = seeded_rngs(args.seed, args.run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let build_s = build.elapsed().as_secs_f64();

    let trainer = WaveTrainer {
        nn,
        search,
        sidecar: Mutex::new(sidecar),
        max_batch: args.batch,
        adaptive: args.adaptive_batch,
        cache_on: args.policy_cache,
        n: args.search_n,
        fallback: RecommendedRamenTrainer::for_rollout(),
        route: game_route,
        stats: Mutex::new(GameStats::default())
    };

    let play = Instant::now();
    while game.next() {
        game.run_stage(&trainer, &mut rng)?;
    }
    game.on_simulation_end(&trainer, &mut rng)?;
    let play_s = play.elapsed().as_secs_f64();
    let score = game.search_score();

    let st = trainer.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
    println!("  建局        {build_s:.3} s");
    println!("  整局墙钟    {play_s:.1} s（其中搜索 {:.1} s）", st.search_s);
    println!("  搜索决策点  {}（候选唯一直接返回 {} 个）", st.searched, st.trivial);
    println!(
        "  路由        完整续跑+原 rf {} 根 / 截断+mean {} 根",
        st.route_full, st.route_trunc
    );
    for line in route_breakdown(&st.route_rows) {
        println!("    {line}");
    }
    println!(
        "  叶请求 {} / 叶结果 {} / 顺延 {}",
        st.leaf_requests, st.leaf_results, st.deferrals
    );
    println!("  平均候选数  {:.1}", st.candidates as f64 / st.searched.max(1) as f64);
    println!(
        "  请求 / 格位 {} / {}（整局批利用率 {}）",
        st.served,
        st.slots,
        fill_text(st.served, st.slots)
    );
    println!(
        "  ❗单点最低批利用率 {}；零请求根 {} 个（不参与该统计）",
        st.min_fill_text(),
        st.zero_request_roots
    );
    println!(
        "  物理批档位  {}（上限 {}，自适应 {}）；换档与预热累计 {:.2} s",
        tier_line(&st.tiers),
        args.batch,
        if args.adaptive_batch { "开" } else { "关" },
        st.setup_s
    );
    println!(
        "  policy 复用 {} | {}",
        if args.policy_cache { "开" } else { "关" },
        st.cache.line()
    );
    println!("  波次        {}", st.waves);
    println!("  终局评分    score={:.3} score_pt={:.3}", score.score, score.score_pt);
    println!("  ❗冒烟只验证接入，回答不了 greedy(Q^NN) vs greedy(Q^手写)");
    if let Some(p) = &args.route_csv {
        write_route_csv(p, &st.route_rows)?;
        println!("  逐根路由    {}（{} 行）", p.display(), st.route_rows.len());
    }
    Ok(())
}

/// 按「阶段 + 地区年份 → 路由」汇总逐根记录
///
/// 输出顺序固定（按 key 字典序），便于两臂逐行比较。
fn route_breakdown(rows: &[RouteRow]) -> Vec<String> {
    let mut agg: HashMap<String, (usize, usize, usize)> = HashMap::new();
    for r in rows {
        let key = match r.region_year_idx {
            Some(y) => format!("{}(Y{})->{}", r.stage, y + 1, r.route),
            None => format!("{}->{}", r.stage, r.route)
        };
        let e = agg.entry(key).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += r.candidates;
        e.2 += r.leaf_requests;
    }
    let mut keys: Vec<&String> = agg.keys().collect();
    keys.sort();
    keys.iter()
        .filter_map(|k| {
            agg.get(*k)
                .map(|(n, cand, leaf)| format!("{k}: {n} 根 | 候选合计 {cand} | 叶请求 {leaf}"))
        })
        .collect()
}

/// 落逐根路由记录
///
/// # 错误
///
/// 文件已存在或写入失败时报错：证据文件不就地覆盖。
fn write_route_csv(path: &Path, rows: &[RouteRow]) -> Result<()> {
    ensure!(!path.exists(), "逐根路由 CSV 已存在，拒绝覆盖：{}", path.display());
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir).with_context(|| format!("建目录失败: {}", dir.display()))?;
        }
    }
    let mut out = String::from(
        "seq,turn,stage,region_year_idx,candidates,route,depth,objective,rf_actual,policy_requests,leaf_requests,terminal_results,leaf_results,wall_s\n"
    );
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6}\n",
            r.seq,
            r.turn,
            r.stage,
            r.region_year_idx.map(|y| y.to_string()).unwrap_or_default(),
            r.candidates,
            r.route,
            r.depth,
            r.objective,
            r.rf_actual,
            r.policy_requests,
            r.leaf_requests,
            r.terminal_results,
            r.leaf_results,
            r.wall_s
        ));
    }
    fs::write(path, out).with_context(|| format!("写逐根路由 CSV 失败: {}", path.display()))?;
    Ok(())
}

/// 按给定粒度跑完固定根的全部 rollout，返回原始结果与逐 rollout 汇总
///
/// 两种粒度都用 `seeds.seed_at(j)`（**不吃候选下标**），故 CRN 配对性质相同，
/// 原始结果应当逐位一致——这正是跨模式比较要验证的。
///
/// # 错误
///
/// 任一 rollout 失败时上抛：静默丢样本会让候选样本数悄悄变少。
fn collect_rollouts(
    search: &FlatSearch<RamenGame>, game: &RamenGame, actions: &[RamenAction], n: usize, seeds: &RolloutSeeds,
    granularity: Granularity, record_ctx: bool
) -> Result<(Vec<RawCell>, Vec<RolloutRow>)> {
    let one = |candidate: usize, j: usize| -> Result<(RawCell, RolloutRow)> {
        if record_ctx {
            ROLLOUT_CTX.with(|c| {
                *c.borrow_mut() = Some(RolloutCtx {
                    candidate,
                    j,
                    seq: 0
                })
            });
            LAST_INFER_END.with(|l| *l.borrow_mut() = None);
        }
        let seed = seeds.seed_at(j);
        let res = search.simulate_common(game, &actions[candidate], seed);
        // 请求数在错误路径上也要取：失败的 rollout 同样占用过批次
        let requests = if record_ctx {
            ROLLOUT_CTX.with(|c| {
                let taken = c.borrow().map(|x| x.seq).unwrap_or(0);
                *c.borrow_mut() = None;
                taken
            })
        } else {
            0
        };
        let s = res?;
        Ok((
            RawCell {
                candidate,
                j,
                seed,
                score: s.score,
                score_pt: s.score_pt
            },
            RolloutRow {
                candidate,
                j,
                requests,
                decisions: requests
            }
        ))
    };

    let mut out: Vec<(RawCell, RolloutRow)> = match granularity {
        Granularity::Candidate => (0..actions.len())
            .into_par_iter()
            .map(|c| (0..n).map(|j| one(c, j)).collect::<Result<Vec<_>>>())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect(),
        Granularity::Flat => {
            let tasks: Vec<(usize, usize)> = (0..actions.len()).flat_map(|c| (0..n).map(move |j| (c, j))).collect();
            tasks.par_iter().map(|&(c, j)| one(c, j)).collect::<Result<Vec<_>>>()?
        }
    };
    // 并行完成顺序不得影响写回顺序：先按 (候选, j) 排好再汇总
    out.sort_by_key(|(r, _)| (r.candidate, r.j));
    Ok(out.into_iter().unzip())
}

/// 一条轨迹上可复用的 policy
///
/// ❗只存**实际送去推理的那 754 个 f32** 与模型返回的 policy 段：不存动作下标、
/// 不存候选分数、不存合法掩码。复用的只是「同一份输入 ⇒ 同一份输出」这一条。
struct PolicyCache {
    /// 写入时的回合
    turn: i32,
    /// 写入时送去推理的完整输入
    features: Vec<f32>,
    /// 模型返回的 policy 段
    policy: Vec<f32>
}

/// 不能复用的原因
///
/// ❗`NoPrev` 合并了「本轨迹还没有过决策」与「上一拍不是 `RamenSelect` 推理决策」
/// 两种情形：缓存只在后者之后写入，其余一律让它过期，因此在这里看不出区别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheMiss {
    /// 没有可用的前一拍
    NoPrev,
    /// 回合不同
    TurnDiff,
    /// 输入有任何一个 f32 位模式不同（含长度不同）
    FeatDiff,
    /// 输入或缓存的 policy 含非有限值
    NonFinite
}

/// 缓存判定结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheProbe {
    /// 本决策不在复用口径内（当前阶段不是 `SpecialSelect`）
    NotEligible,
    /// 可复用
    Hit,
    /// 在口径内但不能复用
    Miss(CacheMiss)
}

/// 判定这一次决策能否复用上一拍的 policy
///
/// 口径极窄：同一条轨迹（调用方保证）、同一回合、上一拍是 `RamenSelect` 推理决策
/// （由写入时机保证）、当前是 `SpecialSelect`，且**实际编码的全部 754 个 f32 位模式
/// 逐一相同**。不用容差、不用哈希；含非有限值一律按未命中处理，走原推理与错误路径。
fn cache_probe(cache: Option<&PolicyCache>, turn: i32, special: bool, feats: &[f32]) -> CacheProbe {
    if !special {
        return CacheProbe::NotEligible;
    }
    let Some(c) = cache else {
        return CacheProbe::Miss(CacheMiss::NoPrev);
    };
    if c.turn != turn {
        return CacheProbe::Miss(CacheMiss::TurnDiff);
    }
    if c.features.len() != feats.len() || c.features.iter().zip(feats).any(|(a, b)| a.to_bits() != b.to_bits()) {
        return CacheProbe::Miss(CacheMiss::FeatDiff);
    }
    if !feats.iter().all(|v| v.is_finite()) || !c.policy.iter().all(|v| v.is_finite()) {
        return CacheProbe::Miss(CacheMiss::NonFinite);
    }
    CacheProbe::Hit
}

/// 缓存计数
#[derive(Default, Clone, Copy)]
struct CacheStats {
    /// 落在复用口径内的决策数（当前阶段是 `SpecialSelect` 的推理决策）
    eligible: usize,
    /// 命中并复用的决策数
    hits: usize,
    /// 未命中：没有可用前一拍
    no_prev: usize,
    /// 未命中：回合不同
    turn_diff: usize,
    /// 未命中：输入不同
    feat_diff: usize,
    /// 未命中：含非有限值
    nonfinite: usize
}

impl CacheStats {
    /// 记一次未命中
    fn note(&mut self, r: CacheMiss) {
        match r {
            CacheMiss::NoPrev => self.no_prev += 1,
            CacheMiss::TurnDiff => self.turn_diff += 1,
            CacheMiss::FeatDiff => self.feat_diff += 1,
            CacheMiss::NonFinite => self.nonfinite += 1
        }
    }

    /// 合并另一份计数
    fn merge(&mut self, o: &Self) {
        self.eligible += o.eligible;
        self.hits += o.hits;
        self.no_prev += o.no_prev;
        self.turn_diff += o.turn_diff;
        self.feat_diff += o.feat_diff;
        self.nonfinite += o.nonfinite;
    }

    /// 一行摘要
    fn line(&self) -> String {
        format!(
            "口径内 {} 命中 {} | 未命中 无前项 {} / 换回合 {} / 输入不同 {} / 非有限 {}",
            self.eligible, self.hits, self.no_prev, self.turn_diff, self.feat_diff, self.nonfinite
        )
    }
}

// ============================================================================
// leaf value 截断实验（隔离段，默认关闭）
//
// 实现的是 `.trae/documents/ramen_leaf_value_opus_task_0916.md`。关键约束：
// **只在显式给出深度时生效**，[`LeafDepth::Full`] 下全部既有模式逐字保持原行为。
// ============================================================================

/// 模型输出里 value 段的起点（`policy 234 + choice 8`）
///
/// 与 [`OUTPUT_DIM`] 一起构成 `[242, 245)` 这一段；三路依次是
/// 期望终局分 / 目标分布标准差 / rf=1.4 高分位。本实验**只用第一路**。
const VALUE_OFF: usize = 242;

/// `advance` 单次调用的推进步数上限
///
/// 纯保护值：一局最多 78 回合、每回合阶段数有限，正常轨迹远达不到。
/// 越界说明阶段机出现了不推进的环，宁可响亮报错也不要空转。
const MAX_ADVANCE_STEPS: u32 = 100_000;

/// 一条 rollout 的截断深度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafDepth {
    /// 不截断：跑到真实终局（既有行为）
    Full,
    /// 跨越 `H` 个 turn 边界后，在第一个**安全的**决策前边界做 leaf 估值
    Turns(i32)
}

impl LeafDepth {
    /// 目标回合 = 根回合 + H；`Full` 没有目标回合
    fn target_turn(self, root_turn: i32) -> Option<i32> {
        match self {
            LeafDepth::Full => None,
            LeafDepth::Turns(h) => Some(root_turn + h)
        }
    }

    /// 本深度下的聚合目标说明（进日志，避免把配置里的 rf 误当成实际口径）
    fn objective(self) -> &'static str {
        match self {
            LeafDepth::Full => "weighted_mean(radical_factor)（原 rf 口径）",
            LeafDepth::Turns(_) => "mean（此模式 rf 不参与聚合，实际 rf=0）"
        }
    }

    /// 进 CSV 的**短**目标名（`objective()` 那句人话里有逗号，不能进 CSV 字段）
    fn objective_tag(self) -> &'static str {
        match self {
            LeafDepth::Full => "rf_weighted",
            LeafDepth::Turns(_) => "mean"
        }
    }

    /// CLI / 日志里的短名
    fn tag(self) -> String {
        match self {
            LeafDepth::Full => "full".to_string(),
            LeafDepth::Turns(h) => format!("h{h}")
        }
    }
}

/// 整局搜索里每个**根**该用哪种叶深度
///
/// 存在的理由：上一轮的固定根试点显示，截断的代价几乎全部集中在第三年的地区选择根，
/// 而那恰恰是加速比最低的一类根。[`LeafRoute::HybridY3Full`] 把这一类根退回完整续跑，
/// 其余根仍走截断。
///
/// ❗[`LeafRoute::Uniform`] **逐字保持既有行为**：`Uniform(LeafDepth::Full)` 就是本工具
/// 改动之前的整局路径，`Uniform(LeafDepth::Turns(h))` 就是上一轮 T4 的纯截断臂。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafRoute {
    /// 全局单一深度（既有行为）
    Uniform(LeafDepth),
    /// 保守混合：第三年 `RegionSelect` 走 [`LeafDepth::Full`] + 原 rf，其余走 `Turns(h)` + mean
    HybridY3Full(i32)
}

/// 一个根的路由判定结果（进日志与逐根 CSV）
#[derive(Debug, Clone, Copy)]
struct RouteDecision {
    /// 本根实际使用的叶深度
    depth: LeafDepth,
    /// 路由标签（`uniform` / `y3_region_full` / `other_trunc`）
    tag: &'static str,
    /// 地区选择的年份归档下标（0/1/2）；非地区根为 `None`
    region_year_idx: Option<usize>
}

impl LeafRoute {
    /// 本路由在当前根上选哪种深度
    ///
    /// 判定**只看当前搜索根**：rollout 途中经过第三年地区不改路由，也不启动嵌套搜索。
    /// 年份用生产口径 [`RamenState::region_archive_year_idx`]（`turn 2/23/47 → 0/1/2`），
    /// **不用**「候选数恰好 120」这种替代条件，也不用 `current_year()-1`（生产文档明令禁止）。
    ///
    /// # 错误
    ///
    /// `RegionSelect` 出现在非 2/23/47 回合时上抛：这说明规则层与本判定的前提不一致，
    /// 宁可失败也不猜年份。
    fn decide(self, game: &RamenGame) -> Result<RouteDecision> {
        let region_year_idx = if matches!(game.stage, RamenStage::RegionSelect) {
            Some(RamenState::region_archive_year_idx(game.turn())?)
        } else {
            None
        };
        Ok(match self {
            LeafRoute::Uniform(d) => RouteDecision {
                depth: d,
                tag: "uniform",
                region_year_idx
            },
            LeafRoute::HybridY3Full(h) => {
                if region_year_idx == Some(2) {
                    RouteDecision {
                        depth: LeafDepth::Full,
                        tag: "y3_region_full",
                        region_year_idx
                    }
                } else {
                    RouteDecision {
                        depth: LeafDepth::Turns(h),
                        tag: "other_trunc",
                        region_year_idx
                    }
                }
            }
        })
    }

    /// 日志里的短名
    fn tag(self) -> String {
        match self {
            LeafRoute::Uniform(d) => format!("uniform:{}", d.tag()),
            LeafRoute::HybridY3Full(h) => format!("hybrid_y3_full:h{h}")
        }
    }

    /// 路由说明（一句话进日志，避免把「配置里的 rf」误当成实际口径）
    fn describe(self) -> String {
        match self {
            LeafRoute::Uniform(d) => format!("全局单一深度 {}；根目标 {}", d.tag(), d.objective()),
            LeafRoute::HybridY3Full(h) => format!(
                "第三年 RegionSelect → full + 原 rf；其余根 → h{h} + mean（rf 不参与聚合）"
            )
        }
    }
}

/// 一次侧车请求的**用途**
///
/// 存在的理由：同一个波次里可能同时有「要 policy 去执行一个动作」和「只要 value
/// 就地收尾」两种行。不显式标出来，value 响应就会被当成一次动作执行。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqKind {
    /// 要 policy logits，回来后执行挂起的那一个决策
    PolicyDecision,
    /// 要叶估值，回来后**不执行任何动作**，就地收尾
    LeafValue
}

/// 一次叶估值的记录
///
/// 与结局类型分开：诊断模式下同一条 rollout **既**有叶估值**又**跑到了真实终局，
/// 两者必须并存才能算逐条配对残差。
#[derive(Debug, Clone)]
struct LeafNote {
    /// 反归一化到普通分数量纲的 `value[0]`
    value: f64,
    /// 实际做 leaf 的回合
    turn: i32,
    /// 实际做 leaf 的阶段
    stage: String,
    /// 相对根回合实际跨越的 turn 数
    turn_delta: i32,
    /// 到达目标回合后因阶段不安全而顺延的次数
    deferrals: u32
}

/// 一条 rollout 的**结局类型**
///
/// 刻意做成枚举而不是「给 leaf 补一份假的终局记录」：叶预测没有 `score_pt`，
/// 也没有五维终局事实，缺的维度按**缺失**处理。
#[derive(Debug, Clone)]
enum TrajOutcome {
    /// 跑到了真实终局：评分与多维记录都是真的
    Terminal {
        /// 真实终局评分（两条口径）
        score: SearchScore,
        /// 真实终局多维记录
        terminal: RamenTerminal
    },
    /// 在叶边界被截断：只有一个**网络预测**的期望分
    Leaf(LeafNote)
}

impl TrajOutcome {
    /// 进入根部聚合的那一个标量
    ///
    /// 终局取普通 `score`（**不是** `score_pt`），叶取反归一化后的 `value[0]`。
    /// 两类样本在截断模式内按同一个均值目标聚合，不分权重。
    fn objective_value(&self) -> f64 {
        match self {
            TrajOutcome::Terminal { score, .. } => score.score,
            TrajOutcome::Leaf(note) => note.value
        }
    }

    /// 结果类型标记（进 CSV，绝不让叶伪装成终局）
    fn kind(&self) -> &'static str {
        match self {
            TrajOutcome::Terminal { .. } => "terminal",
            TrajOutcome::Leaf(_) => "leaf"
        }
    }
}

/// 该阶段能否**安全地**做 leaf 估值
///
/// 判据是「特征编码器与训练样本含义一致」，不是「代码跑得通」：
/// - `SpecialSelect` **排除**。`SpecialSelectMode::Canonical` 下喂给模型的是
///   `canonical_ramen_select_root` 还原出来的局面，其 stage one-hot 落在
///   `RamenSelect` 槽、与上一拍逐位相同；在这里取 value 等于重复上一拍的预测，
///   不是一个新的叶估值。
/// - 其余四个决策阶段各自有自己的 stage one-hot。
/// - 非决策阶段与 `Settlement` 一律不估值，继续正常推进。
fn leaf_safe_stage(stage: &RamenStage) -> bool {
    matches!(
        stage,
        RamenStage::Train | RamenStage::RamenSelect | RamenStage::RegionSelect | RamenStage::SuperRamenSelect
    )
}

/// 一条 rollout 的推进阶段
///
/// ❗必须区分「还没调 `next()`」与「`next()` 已返回 true、等待执行本阶段」：
/// 恢复时重复调用 `next()` 会跳过阶段或重复触发效果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// 需要调用 `next()`
    NeedNext,
    /// `next()` 已返回 true，当前阶段尚未执行
    RunStage,
    /// 已终局并结算
    Done
}

/// 一条可暂停 / 可恢复的 rollout
///
/// 身份、RNG、计时状态全部**随轨迹保存**，不放线程局部量：薄状态机恢复后可能换
/// worker，线程局部量承载身份会错位。
struct Traj {
    /// 候选下标
    candidate: usize,
    /// rollout 序号 `j`
    j: usize,
    /// 该 rollout 的局面
    game: RamenGame,
    /// 该 rollout 的决策 RNG
    rng: StdRng,
    /// 已完成的**逻辑网络决策**数（缓存命中也计）
    ///
    /// ❗对拍靠它对齐：开缓存后真实请求数会减少，用请求数当序号会让两份 trace 错位。
    seq: u32,
    /// 实际发给侧车的推理请求数（缓存命中**不**计）
    infers: u32,
    /// 是否启用同轨迹 policy 复用
    cache_on: bool,
    /// 上一拍可复用的 policy；**属于本轨迹**，随 `Traj` 生灭，不跨轨迹共享
    cache: Option<PolicyCache>,
    /// 本轨迹的复用计数
    cstat: CacheStats,
    /// 推进阶段
    phase: Phase,
    /// 结局（`Done` 后才有）：真实终局或叶估值
    outcome: Option<TrajOutcome>,
    /// 本次挂起待推理的输入（`RunStage` 且需网络时有值）
    request: Option<Vec<f32>>,
    /// 挂起请求的用途；`request` 为 `None` 时无意义
    req_kind: ReqKind,
    /// 本轨迹的截断深度
    leaf: LeafDepth,
    /// 目标回合 = **应用根动作之前**的回合 + H（`Full` 时为 `None`）
    leaf_target: Option<i32>,
    /// 根回合（应用根动作之前），用于算实际跨越的 turn 数
    root_turn: i32,
    /// 到达目标回合后因阶段不安全而顺延的次数
    deferrals: u32,
    /// 已记下的叶估值（诊断模式下会与真实终局并存）
    leaf_note: Option<LeafNote>,
    /// 叶边界上 `prepare_decision` 本来会返回 `Resolved` 的次数（仅诊断模式统计）
    ///
    /// 用来**实测**「leaf 判定独立于 policy 的 `NeedsInference`」：单候选收敛、
    /// 自选比赛守门命中时 policy 不发请求，叶估值仍然必须照发。
    leaf_at_resolved: u32,
    /// 诊断模式：记下叶估值后**用同一份局面与同一个 RNG** 继续跑到终局
    ///
    /// ❗不重新播种、不重建局面：续跑的是叶状态自己的克隆体延续，
    /// 因此前缀与 full 臂逐位相同，终局也应当与 full 臂同 `(候选, j)` 完全一致。
    /// 该模式只用于正确性与偏差诊断，**额外续跑不计入截断臂的性能数字**。
    diag: bool,
    /// `advance` 的推进步数计数（上限保护）
    steps: u32,
    /// 累计的 CPU 推进耗时（不含推理与调度等待）
    cpu_ns: u128,
    /// 逐决策记录（未开启时恒空）
    rows: Vec<DecisionRow>
}

/// 一条已终局 rollout 的完成记录
///
/// ❗存在的理由是**释放局面**：`Traj` 内含完整 [`RamenGame`]，若把终局轨迹原样
/// 留到整根结束，同时驻留的局面数等于「候选数 × n」。2048 条尚可，
/// 120 候选 × 512 = 61,440 份就会耗尽内存。终局时立刻转成本记录，`Traj` 随即释放。
struct TrajDone {
    /// 候选下标
    candidate: usize,
    /// rollout 序号 `j`
    j: usize,
    /// 该 rollout 的种子（`seed_at(j)`，不吃候选下标）
    seed: u64,
    /// 结局：真实终局或叶估值
    outcome: TrajOutcome,
    /// 叶估值记录；诊断模式下与真实终局并存
    leaf_note: Option<LeafNote>,
    /// 叶边界上 policy 本来会 `Resolved` 的次数（仅诊断模式）
    leaf_at_resolved: u32,
    /// 该 rollout 实际发给侧车的推理请求数（缓存命中不计，含 leaf 请求）
    requests: u32,
    /// 其中 leaf 估值请求数（0 或 1）
    leaf_requests: u32,
    /// 该 rollout 的逻辑网络决策数（缓存命中也计）
    decisions: u32,
    /// 本轨迹的复用计数
    cstat: CacheStats,
    /// 该 rollout 自报的 CPU 推进耗时
    cpu_ns: u128,
    /// 逐决策记录（未开启时恒空）
    rows: Vec<DecisionRow>
}

/// 消费预选答案的转发器
///
/// 规则层照旧执行动作；只有「该选哪个」这一步改由调度器事先备好。
/// 进入 `run_stage` **之前**答案就已就绪，故不需要在规则调用栈中途挂起。
struct PreselectTrainer<'a> {
    /// 网络策略（用它的三步接口，保证与生产同源）
    nn: &'a RamenNnTrainer,
    /// 事件选项出口
    handwritten: RecommendedRamenTrainer,
    /// 待消费的 policy；`None` 表示本阶段不该需要网络
    pending: RefCell<Option<Vec<f32>>>,
    /// 调度器送去推理时用的输入，用于核验「答案对得上这个决策」
    expect: Option<Vec<f32>>,
    /// 答案是否已被消费
    consumed: Cell<bool>,
    /// 开启后记下本次决策的实际候选、顺序与选中项（逐决策等价性用）
    record: RefCell<Option<(Vec<String>, usize, i32, String)>>,
    /// 是否记录
    recording: bool
}

impl<'a> PreselectTrainer<'a> {
    /// 构造
    fn new(nn: &'a RamenNnTrainer, pending: Option<Vec<f32>>, expect: Option<Vec<f32>>) -> Self {
        Self {
            nn,
            handwritten: RecommendedRamenTrainer::for_rollout(),
            pending: RefCell::new(pending),
            expect,
            consumed: Cell::new(false),
            record: RefCell::new(None),
            recording: false
        }
    }

    /// 开启逐决策记录
    fn recording(mut self) -> Self {
        self.recording = true;
        self
    }

    /// 核验答案「恰好消费一次」
    ///
    /// # 错误
    ///
    /// 备了答案却没被消费（说明阶段判断错位）时报错。
    fn check_consumed(&self, expected: bool) -> Result<()> {
        ensure!(
            self.consumed.get() == expected,
            "预选答案消费状态不符：期望 {expected}，实际 {}",
            self.consumed.get()
        );
        Ok(())
    }
}

impl Trainer<RamenGame> for PreselectTrainer<'_> {
    /// 走与生产同一套三步；到推理那一步改用事先备好的 policy
    ///
    /// # 错误
    ///
    /// 需要网络却没有备答案、备的答案与本决策的输入对不上，或打分失败时报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self.nn.prepare_decision(game, actions, rng)? {
            DecisionPrep::Resolved(idx) => Ok(idx),
            DecisionPrep::NeedsInference(features) => {
                let policy = self
                    .pending
                    .borrow_mut()
                    .take()
                    .ok_or_else(|| anyhow!("调度器没有为该决策准备答案：阶段判断与实际推进不一致"))?;
                if let Some(expect) = self.expect.as_ref() {
                    ensure!(
                        expect == &features,
                        "预选答案与实际决策输入不匹配：送去推理的局面不是正在决策的这个"
                    );
                }
                self.consumed.set(true);
                let chosen = self.nn.resolve_decision(game, actions, &policy)?;
                if self.recording {
                    // 在**规则层实际使用的**候选表上取值，不另行 list_actions
                    *self.record.borrow_mut() = Some((
                        actions.iter().map(action_repr).collect(),
                        chosen,
                        game.turn(),
                        format!("{:?}", game.stage)
                    ));
                }
                Ok(chosen)
            }
        }
    }

    /// 事件选项转交手写（与生产 rollout 基策一致）
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.handwritten.select_choice(game, choices, rng)
    }

    /// 事件选项（含友人事件特例）转交手写
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.handwritten.select_event_choice(game, event, choices, rng)
    }
}

/// 该阶段是否是决策点
fn is_decision_stage(stage: &RamenStage) -> bool {
    matches!(
        stage,
        RamenStage::Train
            | RamenStage::RamenSelect
            | RamenStage::SpecialSelect
            | RamenStage::RegionSelect
            | RamenStage::SuperRamenSelect
    )
}

impl Traj {
    /// 建一条 rollout 并执行根动作（**恰好一次**）
    ///
    /// # 错误
    ///
    /// 根动作应用失败时报错。
    fn start(
        candidate: usize, j: usize, game: &RamenGame, action: &RamenAction, seed: u64, cache_on: bool,
        leaf: LeafDepth, diag: bool
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        // ❗根回合取自**应用根动作之前**的局面：根动作本身可能推进阶段
        let root_turn = game.turn();
        let mut sim = game.fork_for_rollout(seed);
        sim.apply_root_action(action, &mut rng)?;
        Ok(Self {
            candidate,
            j,
            game: sim,
            rng,
            seq: 0,
            infers: 0,
            cache_on,
            cache: None,
            cstat: CacheStats::default(),
            phase: Phase::NeedNext,
            outcome: None,
            request: None,
            req_kind: ReqKind::PolicyDecision,
            leaf,
            leaf_target: leaf.target_turn(root_turn),
            root_turn,
            deferrals: 0,
            leaf_note: None,
            leaf_at_resolved: 0,
            diag,
            steps: 0,
            cpu_ns: 0,
            rows: Vec::new()
        })
    }

    /// 推进到「下一个需要网络的决策」或终局
    ///
    /// 挂起点落在**已有的阶段边界**上：只在即将进入某个决策阶段、且该阶段确实需要
    /// 网络时才停下，规则调用栈里没有半执行的现场。
    ///
    /// # 错误
    ///
    /// 规则层报错，或不需要网络的阶段却消费了预选答案时报错。
    fn advance(&mut self, nn: &RamenNnTrainer, record: bool) -> Result<()> {
        let t0 = Instant::now();
        loop {
            self.steps += 1;
            ensure!(
                self.steps <= MAX_ADVANCE_STEPS,
                "轨迹推进超过 {MAX_ADVANCE_STEPS} 步仍未终局或挂起：阶段机可能不推进"
            );
            match self.phase {
                Phase::Done => break,
                Phase::NeedNext => {
                    if self.game.next() {
                        self.phase = Phase::RunStage;
                    } else {
                        let t = PreselectTrainer::new(nn, None, None);
                        // 终局优先于截断：结算只做一次，取的是**真实**终局评分
                        self.game.on_simulation_end(&t, &mut self.rng)?;
                        let score = self.game.search_score();
                        // 终局多维记录必须在这里取：`finish` 之后局面就没了
                        let terminal = RamenTerminal::from_game(&self.game);
                        self.outcome = Some(TrajOutcome::Terminal { score, terminal });
                        self.phase = Phase::Done;
                        break;
                    }
                }
                Phase::RunStage => {
                    // ❗leaf 判定独立于 policy 的 `NeedsInference`：单候选收敛、自选比赛
                    // 守门、缓存命中都不免除叶估值。判定放在 policy 分支**之前**。
                    if let Some(target) = self.leaf_target {
                        if is_decision_stage(&self.game.stage) && self.game.turn() >= target {
                            if leaf_safe_stage(&self.game.stage) {
                                let actions = self.game.list_actions()?;
                                ensure!(!actions.is_empty(), "叶边界上候选为空，不估值");
                                if self.diag {
                                    // 只在诊断模式下多跑一次判定：用 RNG 的**克隆体**探测，
                                    // 不改生产随机流。生产计时臂不付这份开销。
                                    let mut probe = self.rng.clone();
                                    if matches!(
                                        nn.prepare_decision(&self.game, &actions, &mut probe)?,
                                        DecisionPrep::Resolved(_)
                                    ) {
                                        self.leaf_at_resolved += 1;
                                    }
                                }
                                let f = encode(&self.game)?;
                                ensure!(
                                    f.len() == INPUT_DIM,
                                    "叶特征长度 {} 与 INPUT_DIM={INPUT_DIM} 不符",
                                    f.len()
                                );
                                self.request = Some(f);
                                self.req_kind = ReqKind::LeafValue;
                                break;
                            }
                            // 阶段不安全（当前只有 `SpecialSelect`）：记一次延期，正常推进
                            self.deferrals += 1;
                        }
                    }
                    if is_decision_stage(&self.game.stage) {
                        let actions = self.game.list_actions()?;
                        // 用 RNG 的**克隆体**探测：探测不得改变生产随机流
                        let mut probe = self.rng.clone();
                        if let DecisionPrep::NeedsInference(f) =
                            nn.prepare_decision(&self.game, &actions, &mut probe)?
                        {
                            // 只在**真正进入 NeedsInference 之后**才查缓存：自选比赛守门、
                            // 单候选收敛、SpecialSelect 手写口径都在 `prepare_decision` 里
                            // 就地返回，这里不复制第二份判定逻辑。
                            if self.cache_on {
                                let special = self.game.stage == RamenStage::SpecialSelect;
                                // 先取判定结果再 match：`self.cache` 的借用不能跨进各臂，
                                // 臂里要按 `&mut self` 执行决策
                                let verdict = cache_probe(self.cache.as_ref(), self.game.turn(), special, &f);
                                match verdict {
                                    CacheProbe::Hit => {
                                        let policy = self
                                            .cache
                                            .as_ref()
                                            .map(|c| c.policy.clone())
                                            .ok_or_else(|| anyhow!("缓存判定命中却取不到 policy"))?;
                                        self.cstat.eligible += 1;
                                        self.cstat.hits += 1;
                                        // 仍走原 `run_stage`：候选表、打分与动作落地一律重算
                                        self.run_decision(nn, policy, f, record)?;
                                        continue;
                                    }
                                    CacheProbe::Miss(r) => {
                                        self.cstat.eligible += 1;
                                        self.cstat.note(r);
                                    }
                                    CacheProbe::NotEligible => {}
                                }
                            }
                            self.request = Some(f);
                            self.req_kind = ReqKind::PolicyDecision;
                            break;
                        }
                    }
                    // 任何一次阶段执行都先让缓存过期：只有紧接 `RamenSelect` 的那一拍能复用
                    self.cache = None;
                    let t = PreselectTrainer::new(nn, None, None);
                    self.game.run_stage(&t, &mut self.rng)?;
                    t.check_consumed(false)?;
                    self.phase = Phase::NeedNext;
                }
            }
        }
        self.cpu_ns += t0.elapsed().as_nanos();
        Ok(())
    }

    /// 用批量推理回来的**整行输出**收掉挂起的那一步
    ///
    /// 两种用途分流：
    /// - [`ReqKind::PolicyDecision`]：取 `[..POLICY_DIM]` 执行挂起的那一个决策阶段。
    /// - [`ReqKind::LeafValue`]：取 `[VALUE_OFF..]` 解码一次，就地收尾。
    ///   **不调用 `run_decision`**，不消耗动作、不再推进轨迹。
    ///
    /// # 错误
    ///
    /// 当前并非挂起在决策点、行长不符、value 非有限、规则层报错，
    /// 或答案未被恰好消费一次时报错。
    fn resume(&mut self, nn: &RamenNnTrainer, out: &[f32], record: bool) -> Result<()> {
        let expect = self
            .request
            .take()
            .ok_or_else(|| anyhow!("该 rollout 并未挂起在决策点"))?;
        ensure!(
            out.len() == OUTPUT_DIM,
            "侧车返回行长 {} 与契约 {OUTPUT_DIM} 不符",
            out.len()
        );
        let t0 = Instant::now();
        self.infers += 1;
        match self.req_kind {
            ReqKind::PolicyDecision => {
                self.run_decision(nn, out[..POLICY_DIM].to_vec(), expect, record)?;
            }
            ReqKind::LeafValue => {
                ensure!(self.phase == Phase::RunStage, "叶估值返回时阶段状态不是 RunStage");
                ensure!(self.leaf_note.is_none(), "同一条 rollout 记了两次叶估值");
                // 侧车已把三路 value 换算到「成员 0 尺度」的归一化空间；这里**只解码一次**
                let value = nn.value_norm().denormalize(&out[VALUE_OFF..])?;
                ensure!(value.mean.is_finite(), "反归一化后的叶估值不是有限值: {}", value.mean);
                let turn = self.game.turn();
                let note = LeafNote {
                    value: value.mean,
                    turn,
                    stage: format!("{:?}", self.game.stage),
                    turn_delta: turn - self.root_turn,
                    deferrals: self.deferrals
                };
                self.leaf_note = Some(note.clone());
                // 关掉目标回合：无论哪条路都不再触发第二次叶估值
                self.leaf_target = None;
                if self.diag {
                    // 诊断：**不执行任何动作、不碰 RNG、不重新播种**，
                    // 只是把这一步退回「尚未决策」，让 `advance` 按原路继续到终局。
                    // 于是本条 rollout 的前缀与 full 臂逐位相同。
                    self.phase = Phase::RunStage;
                } else {
                    self.outcome = Some(TrajOutcome::Leaf(note));
                    // 局面到此为止：不执行动作、不推进、`finish` 随即释放它
                    self.phase = Phase::Done;
                }
            }
        }
        self.cpu_ns += t0.elapsed().as_nanos();
        Ok(())
    }

    /// 用一份 policy 执行挂起的那一个决策阶段
    ///
    /// 真实推理与缓存复用**共用本函数**：两条路都重新走 `prepare_decision` 与
    /// 当前候选表上的 `resolve_decision`，动作落地与「答案恰好消费一次」检查不变。
    /// 不计时——调用方各自把耗时记进自己的窗口，避免重复累加。
    ///
    /// # 错误
    ///
    /// 当前并非挂起在决策点、规则层报错，或答案未被恰好消费一次时报错。
    fn run_decision(&mut self, nn: &RamenNnTrainer, policy: Vec<f32>, expect: Vec<f32>, record: bool) -> Result<()> {
        ensure!(self.phase == Phase::RunStage, "恢复时阶段状态不是 RunStage");
        let turn_now = self.game.turn();
        // 只有「本拍是 RamenSelect 的推理决策」才值得留给下一拍；其余一律让缓存过期
        let keep = self.cache_on && self.game.stage == RamenStage::RamenSelect;
        self.cache = None;
        let saved = keep.then(|| (expect.clone(), policy.clone()));
        let features = record.then(|| expect.clone());
        let t = PreselectTrainer::new(nn, Some(policy), Some(expect));
        let t = if record { t.recording() } else { t };
        self.game.run_stage(&t, &mut self.rng)?;
        t.check_consumed(true)?;
        if let Some(features) = features {
            let (actions, chosen, turn, stage) = t
                .record
                .borrow_mut()
                .take()
                .ok_or_else(|| anyhow!("开启了记录却没有拿到决策明细"))?;
            self.rows.push(DecisionRow {
                candidate: self.candidate,
                j: self.j,
                seq: self.seq,
                turn,
                stage,
                actions,
                chosen,
                features
            });
        }
        if let Some((f, p)) = saved {
            self.cache = Some(PolicyCache {
                turn: turn_now,
                features: f,
                policy: p
            });
        }
        self.phase = Phase::NeedNext;
        self.seq += 1;
        Ok(())
    }

    /// 终局轨迹转成完成记录，**消费自身**以释放其持有的完整局面
    ///
    /// # 错误
    ///
    /// 尚未终局或缺终局评分时报错。
    fn finish(self, seed: u64) -> Result<TrajDone> {
        ensure!(self.phase == Phase::Done, "轨迹尚未收尾，不能转成完成记录");
        let outcome = self.outcome.ok_or_else(|| anyhow!("已收尾的轨迹缺少结局记录"))?;
        // `Full` 深度下**不允许**出现叶估值：出现了说明深度参数漏传
        if matches!(self.leaf, LeafDepth::Full) {
            ensure!(
                matches!(outcome, TrajOutcome::Terminal { .. }) && self.leaf_note.is_none(),
                "未开启截断却收到叶结果：深度参数传递有误"
            );
        }
        // 非诊断模式下，叶估值与「叶结局」必须同进同出
        if !self.diag {
            ensure!(
                self.leaf_note.is_some() == matches!(outcome, TrajOutcome::Leaf(_)),
                "叶估值与结局类型不一致：记了叶却没以叶收尾（或反之）"
            );
        }
        let leaf_requests = u32::from(self.leaf_note.is_some());
        Ok(TrajDone {
            leaf_at_resolved: self.leaf_at_resolved,
            candidate: self.candidate,
            j: self.j,
            seed,
            outcome,
            leaf_note: self.leaf_note,
            requests: self.infers,
            leaf_requests,
            decisions: self.seq,
            cstat: self.cstat,
            cpu_ns: self.cpu_ns,
            rows: self.rows
        })
        // self 在此处丢弃：`game` 与 `rng` 随之释放
    }
}

/// 波次驱动的一次固定根搜索
///
/// 语义与离线模拟一致：任务队列按 `(候选, j)` 固定顺序；最多 `B` 条活跃轨迹；
/// 每个波次每条存活轨迹恰好贡献 1 个请求；跑完的槽位按固定顺序补位。
/// 批组成完全由**逻辑波次与固定顺序**决定，worker 到达顺序与超时都不参与。
///
/// # 错误
///
/// 轨迹推进、侧车通信或恢复失败时报错。
fn run_gpu_wave(
    game: &RamenGame, actions: &[RamenAction], n: usize, seeds: &RolloutSeeds, nn: &RamenNnTrainer,
    sidecar: &mut Sidecar, batch: usize, record: bool, cache_on: bool, leaf: LeafDepth, diag: bool
) -> Result<WaveOut> {
    // `batch` 同时是**调度器的活跃轨迹容量**与**GPU 物理张量行数**：本根内固定，
    // 数值相同但含义不同，改其一必须同时想清楚另一个。
    let setup_before = sidecar.batch_setup_s;
    // 换档与该档位的首次预热放在波次循环之外，不进稳态计时
    sidecar.ensure_batch(batch)?;
    let setup_s = sidecar.batch_setup_s - setup_before;
    let total_start = Instant::now();
    let queue: Vec<(usize, usize)> = (0..actions.len()).flat_map(|c| (0..n).map(move |j| (c, j))).collect();
    let mut qi = 0usize;
    let mut active: Vec<Traj> = Vec::with_capacity(batch);
    // 只装完成记录，不装 `Traj`：终局局面在这里就已经被释放
    let mut done: Vec<TrajDone> = Vec::with_capacity(queue.len());
    let mut stats = WaveStats::default();

    // 装满第一批（**计入**波次循环耗时：CPU 基线同样包含建分支）
    while qi < queue.len() && active.len() < batch {
        let (c, j) = queue[qi];
        qi += 1;
        active.push(Traj::start(c, j, game, &actions[c], seeds.seed_at(j), cache_on, leaf, diag)?);
    }

    while !active.is_empty() {
        // 1) CPU 并行推进到各自的下一个请求或终局
        let adv = Instant::now();
        active.par_iter_mut().try_for_each(|t| t.advance(nn, record))?;
        stats.cpu_wall_s += adv.elapsed().as_secs_f64();

        // 2) 收走已终局的，按固定顺序补位
        let mut still: Vec<Traj> = Vec::with_capacity(active.len());
        for t in active.drain(..) {
            if t.phase == Phase::Done {
                let seed = seeds.seed_at(t.j);
                done.push(t.finish(seed)?);
            } else {
                still.push(t);
            }
        }
        active = still;
        while active.len() < batch && qi < queue.len() {
            let (c, j) = queue[qi];
            qi += 1;
            let mut t = Traj::start(c, j, game, &actions[c], seeds.seed_at(j), cache_on, leaf, diag)?;
            let adv = Instant::now();
            t.advance(nn, record)?;
            stats.cpu_wall_s += adv.elapsed().as_secs_f64();
            if t.phase == Phase::Done {
                done.push(t.finish(seeds.seed_at(j))?);
            } else {
                active.push(t);
            }
        }
        if active.is_empty() {
            break;
        }

        // 3) 一个波次：每条存活轨迹恰好一个请求，补零到物理批尺寸
        let rows: Vec<Vec<f32>> = active
            .iter()
            .map(|t| {
                t.request
                    .clone()
                    .ok_or_else(|| anyhow!("活跃轨迹缺少挂起的请求"))
            })
            .collect::<Result<Vec<_>>>()?;
        // 缓存命中在 `advance` 里就地消化，活跃轨迹必然各挂着一个请求；
        // 真出现「有活跃轨迹却没有待推理行」就是调度错位，宁可响亮报错也不发空请求
        ensure!(!rows.is_empty(), "存在活跃轨迹却没有任何待推理行");
        stats.waves += 1;
        stats.served += rows.len();
        // 物理格位按**本根实际生效的档位**累加
        stats.slots += batch;
        let infer = Instant::now();
        let outs = sidecar.infer(&rows)?;
        stats.gpu_wall_s += infer.elapsed().as_secs_f64();
        ensure!(outs.len() == active.len(), "侧车返回 {} 行，期望 {}", outs.len(), active.len());

        // 4) 用返回结果恢复各自那一步
        let res = Instant::now();
        active
            .par_iter_mut()
            .zip(outs.into_par_iter())
            .try_for_each(|(t, out)| t.resume(nn, &out, record))?;
        stats.cpu_wall_s += res.elapsed().as_secs_f64();
    }
    ensure!(
        done.len() == queue.len(),
        "完成记录 {} 条与任务队列 {} 条不符",
        done.len(),
        queue.len()
    );

    // 并行完成顺序不得影响写回顺序
    done.sort_by_key(|t| (t.candidate, t.j));
    // 每个候选的完成条数必须恰好是 n：总数守门查不出「某候选多、另一候选少」
    let mut per_candidate = vec![0usize; actions.len()];
    for t in &done {
        let slot = per_candidate
            .get_mut(t.candidate)
            .ok_or_else(|| anyhow!("完成记录的候选下标 {} 越界", t.candidate))?;
        *slot += 1;
    }
    for (c, got) in per_candidate.iter().enumerate() {
        ensure!(*got == n, "候选 {c} 完成 {got} 条，期望 {n} 条");
    }
    let mut rollout_rows = Vec::with_capacity(done.len());
    let mut decisions = Vec::new();
    for t in &done {
        rollout_rows.push(RolloutRow {
            candidate: t.candidate,
            j: t.j,
            requests: t.requests,
            decisions: t.decisions
        });
        stats.cache.merge(&t.cstat);
        stats.cpu_in_traj_s += t.cpu_ns as f64 / 1e9;
        stats.leaf_results += t.leaf_requests as usize;
        stats.leaf_requests += t.leaf_requests as usize;
        stats.diag_continued += usize::from(t.leaf_note.is_some() && matches!(t.outcome, TrajOutcome::Terminal { .. }));
        stats.leaf_at_resolved += t.leaf_at_resolved as usize;
        stats.policy_requests += (t.requests - t.leaf_requests) as usize;
    }
    for t in &mut done {
        decisions.append(&mut t.rows);
    }
    stats.total_wall_s = total_start.elapsed().as_secs_f64();
    stats.batch = batch;
    stats.setup_s = setup_s;
    stats.terminal_results = done
        .iter()
        .filter(|t| matches!(t.outcome, TrajOutcome::Terminal { .. }))
        .count();
    stats.per_candidate = per_candidate;
    Ok(WaveOut {
        done,
        rollout_rows,
        decisions,
        stats
    })
}

/// 一次波次驱动搜索的全部产物
///
/// 拆成结构体而不是继续加元组分量：叶截断打开后，「真实终局原始表」与「查表备忘表」
/// 都可能**不存在**，用取值方法显式报错比塞一张空表安全。
struct WaveOut {
    /// 逐 rollout 的完成记录（已按 `(候选, j)` 排序）
    done: Vec<TrajDone>,
    /// 逐 rollout 的请求/决策计数
    rollout_rows: Vec<RolloutRow>,
    /// 逐决策记录（未开启时为空）
    decisions: Vec<DecisionRow>,
    /// 耗时与利用率
    stats: WaveStats
}

impl WaveOut {
    /// 取真实终局的原始结果表（既有模式用）
    ///
    /// # 错误
    ///
    /// 出现任何叶截断结果时报错：叶预测不是终局分，不得混进这张表。
    fn raw_cells(&self) -> Result<Vec<RawCell>> {
        self.done
            .iter()
            .map(|t| match &t.outcome {
                TrajOutcome::Terminal { score, .. } => Ok(RawCell {
                    candidate: t.candidate,
                    j: t.j,
                    seed: t.seed,
                    score: score.score,
                    score_pt: score.score_pt
                }),
                TrajOutcome::Leaf { .. } => Err(anyhow!(
                    "候选 {} 第 {} 条是叶结果，不能当终局原始分使用",
                    t.candidate,
                    t.j
                ))
            })
            .collect()
    }

    /// 取搜索内核查表用的备忘表（只装真实终局）
    ///
    /// # 错误
    ///
    /// 出现叶结果，或同一候选内 seed 重复导致表项被覆盖时报错。
    fn batch_table(&self) -> Result<RamenBatchTable> {
        let mut table: RamenBatchTable = HashMap::with_capacity(self.done.len());
        for t in &self.done {
            match &t.outcome {
                TrajOutcome::Terminal { score, terminal } => {
                    table.insert(
                        (t.candidate, t.seed),
                        RolloutOutcome {
                            score: *score,
                            terminal: *terminal
                        }
                    );
                }
                TrajOutcome::Leaf { .. } => {
                    bail!("候选 {} 第 {} 条是叶结果，不能进搜索内核的备忘表", t.candidate, t.j)
                }
            }
        }
        // 查表键是 `(候选, seed)`：同一候选内出现重复 seed 会让表项被覆盖而条数变少，
        // 而完成记录仍然是满的。官方汇总走原始表、内核查表走本表，两边都要守。
        for (c, got) in self.stats.per_candidate.iter().enumerate() {
            let in_table = table.keys().filter(|(cand, _)| *cand == c).count();
            ensure!(in_table == *got, "候选 {c} 查表项 {in_table} 条与完成记录 {got} 条不符（seed 重复？）");
        }
        Ok(table)
    }

    /// 取截断实验用的臂结果（终局与叶统一带类型标记）
    fn arm_cells(&self) -> Vec<ArmCell> {
        self.done
            .iter()
            .map(|t| {
                // 诊断模式：本条既有叶估值又跑到了真实终局。聚合仍用**叶估值**
                // （否则测的就不是截断臂），真实终局单独记在 `paired_terminal`。
                let (kind, value, paired) = match (&t.leaf_note, &t.outcome) {
                    (Some(note), TrajOutcome::Terminal { score, .. }) => {
                        ("leaf", note.value, Some(score.score))
                    }
                    _ => (t.outcome.kind(), t.outcome.objective_value(), None)
                };
                let score_pt = match (&t.leaf_note, &t.outcome) {
                    (None, TrajOutcome::Terminal { score, .. }) => Some(score.score_pt),
                    _ => None
                };
                let (leaf_turn, leaf_stage, turn_delta, deferrals) = match &t.leaf_note {
                    Some(n) => (Some(n.turn), Some(n.stage.clone()), Some(n.turn_delta), Some(n.deferrals)),
                    None => (None, None, None, None)
                };
                ArmCell {
                    candidate: t.candidate,
                    j: t.j,
                    seed: t.seed,
                    kind,
                    value,
                    score_pt,
                    paired_terminal: paired,
                    leaf_turn,
                    leaf_stage,
                    turn_delta,
                    deferrals
                }
            })
            .collect()
    }
}

/// 把波次驱动接进生产教师的批量后端
///
/// 侧车由本结构持有并跨决策点复用；请求串行（整锁），不做跨局共享队列。
struct WaveBackend {
    /// 网络策略（rollout 基策）
    nn: Arc<RamenNnTrainer>,
    /// 常驻侧车
    sidecar: Mutex<Sidecar>,
    /// 侧车分配的物理批上限；固定档时它就是恒定档位
    max_batch: usize,
    /// 是否按根选择物理批档位
    adaptive: bool,
    /// 是否启用同轨迹 policy 复用
    cache_on: bool,
    /// 累计统计
    stats: Mutex<GameStats>
}

impl RamenBatchRollout for WaveBackend {
    fn precompute(
        &self, game: &RamenGame, actions: &[RamenAction], seeds: &RolloutSeeds, n: usize
    ) -> Result<RamenBatchTable> {
        let t0 = Instant::now();
        let batch = plan_batch(self.adaptive, self.max_batch, actions.len() * n);
        let out = {
            let mut sc = self.sidecar.lock().map_err(|_| anyhow!("侧车锁被毒化"))?;
            // 生产教师后端不接截断：`LeafDepth::Full` 保持既有行为逐字不变
            run_gpu_wave(game, actions, n, seeds, &self.nn, &mut sc, batch, false, self.cache_on, LeafDepth::Full, false)?
        };
        let table = out.batch_table()?;
        let wave = &out.stats;
        let mut st = self.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
        *st.tiers.entry(wave.batch).or_insert(0) += 1;
        st.setup_s += wave.setup_s;
        st.cache.merge(&wave.cache);
        st.note_fill(wave.served, wave.slots);
        st.searched += 1;
        st.candidates += actions.len();
        st.waves += wave.waves;
        st.served += wave.served;
        st.slots += wave.slots;
        st.rollouts += actions.len() * n;
        st.cpu_wall_s += wave.cpu_wall_s;
        st.gpu_wall_s += wave.gpu_wall_s;
        st.cpu_in_traj_s += wave.cpu_in_traj_s;
        st.search_s += t0.elapsed().as_secs_f64();
        Ok(table)
    }
}

/// 波次驱动的耗时与利用率统计
#[derive(Default)]
struct WaveStats {
    /// 波次数
    waves: usize,
    /// 其中用途为 policy 决策的请求数
    policy_requests: usize,
    /// 其中用途为叶估值的请求数
    leaf_requests: usize,
    /// 叶截断收尾的 rollout 条数
    leaf_results: usize,
    /// 真实终局收尾的 rollout 条数
    terminal_results: usize,
    /// 诊断模式下「记了叶又续跑到真实终局」的条数
    diag_continued: usize,
    /// 诊断模式下，叶边界上 policy 本来会 `Resolved`（不发请求）的条数
    leaf_at_resolved: usize,
    /// 各候选实际完成条数（供备忘表守门复核）
    per_candidate: Vec<usize>,
    /// 实际服务的请求数
    served: usize,
    /// 占用的物理批格位数（波次 × B）
    slots: usize,
    /// 波次循环墙钟：从建第一条轨迹到全部完成记录整理完毕
    ///
    /// ❗**不含**官方汇总，也**不含**侧车启动与预热；统一口径的耗时在 `main` 里量。
    total_wall_s: f64,
    /// CPU 推进与恢复的墙钟（并行段整体计时）
    cpu_wall_s: f64,
    /// 侧车往返墙钟
    gpu_wall_s: f64,
    /// 各轨迹自报的 CPU 耗时之和（跨 worker 累加，**不可**直接加进端到端墙钟）
    cpu_in_traj_s: f64,
    /// 本根实际生效的物理批档位
    batch: usize,
    /// 换档与该档位首次预热的墙钟，**单列，不在 `total_wall_s` 内**
    setup_s: f64,
    /// 同轨迹 policy 复用计数
    cache: CacheStats
}

/// 整局里一个根的路由与实际口径记录（落 `--route-csv`）
///
/// 落的是**实际值**：路由判定、阶段与地区年份、实际聚合目标、实际生效的 rf、
/// 以及该根真正发出的两类请求与两类结局条数。
#[derive(Debug, Clone)]
struct RouteRow {
    /// 决策序号（只数走了搜索的根）
    seq: usize,
    /// 回合
    turn: i32,
    /// 阶段
    stage: String,
    /// 地区年份归档下标；非地区根留空
    region_year_idx: Option<usize>,
    /// 候选数
    candidates: usize,
    /// 路由标签
    route: &'static str,
    /// 实际叶深度短名
    depth: String,
    /// 实际聚合目标
    objective: &'static str,
    /// 实际生效的 rf（mean 口径恒为 0.0）
    rf_actual: f64,
    /// 本根 policy 请求数
    policy_requests: usize,
    /// 本根叶请求数
    leaf_requests: usize,
    /// 本根真实终局条数
    terminal_results: usize,
    /// 本根叶结局条数
    leaf_results: usize,
    /// 本根搜索墙钟
    wall_s: f64
}

/// 整局冒烟的累计统计
#[derive(Default)]
struct GameStats {
    /// 走了搜索的决策点数
    searched: usize,
    /// 走完整续跑 + 原 rf 的根数
    route_full: usize,
    /// 走截断 + mean 的根数
    route_trunc: usize,
    /// 逐根路由记录
    route_rows: Vec<RouteRow>,
    /// 整局累计的叶估值请求数（`LeafDepth::Full` 下恒为 0）
    leaf_requests: usize,
    /// 整局累计的叶结果条数
    leaf_results: usize,
    /// 整局累计的叶边界顺延次数
    deferrals: usize,
    /// 转交回退策略的决策点数（候选唯一）
    trivial: usize,
    /// 累计候选数
    candidates: usize,
    /// 累计波次
    waves: usize,
    /// 累计服务请求
    served: usize,
    /// 累计物理格位（波次 × B）
    slots: usize,
    /// 累计搜索耗时
    search_s: f64,
    /// 单个决策点的最低批利用率，**只统计真正占过推理格位的根**
    min_fill: f64,
    /// 参与最低批利用率统计的根数（`slots > 0`）
    fill_roots: usize,
    /// 一次推理都没发过的根数（`slots == 0`），单列，不参与最低填充率
    zero_request_roots: usize,
    /// 累计 rollout 条数（各决策点 候选数 × n 之和）
    rollouts: usize,
    /// 累计 CPU 段墙钟（并行推进 + 恢复，按并行段整体计时）
    cpu_wall_s: f64,
    /// 累计侧车往返墙钟
    ///
    /// ❗这是 `infer()` 整段：含 Rust 序列化/解码、两次管道传输、Python 侧全部工作、
    /// 主机↔设备拷贝与前向。**内部没有进一步拆分**，不要当成纯 GPU 前向。
    gpu_wall_s: f64,
    /// 各轨迹自报 CPU 耗时之和（跨 worker 累加，**不可**加进端到端墙钟）
    cpu_in_traj_s: f64,
    /// 换档与档位首次预热累计墙钟（整局里无法挪到窗口外，故**照实单列**）
    setup_s: f64,
    /// 各物理批档位的使用次数
    tiers: HashMap<usize, usize>,
    /// 同轨迹 policy 复用计数
    cache: CacheStats
}

impl GameStats {
    /// 记一个根的批利用率
    ///
    /// 零推理格位的根单独计数，**不**参与最低填充率：它压根没占过 GPU 格位，
    /// 拿它的「0%」去拉低最低值等于污染统计。
    fn note_fill(&mut self, served: usize, slots: usize) {
        if slots == 0 {
            self.zero_request_roots += 1;
            return;
        }
        let fill = served as f64 / slots as f64;
        self.min_fill = if self.fill_roots == 0 { fill } else { self.min_fill.min(fill) };
        self.fill_roots += 1;
    }

    /// 最低批利用率的显示文本；没有任何根占过格位时给 N/A
    fn min_fill_text(&self) -> String {
        if self.fill_roots == 0 {
            "N/A（没有任何根发出过推理）".to_string()
        } else {
            format!("{:.1}%（取自 {} 个有推理格位的根）", 100.0 * self.min_fill, self.fill_roots)
        }
    }
}

/// 把档位使用情况排成一行
fn tier_line(tiers: &HashMap<usize, usize>) -> String {
    let mut v: Vec<(usize, usize)> = tiers.iter().map(|(k, n)| (*k, *n)).collect();
    v.sort_unstable();
    v.iter()
        .map(|(b, n)| format!("B{b}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// 整局冒烟用的搜索策略
///
/// 每个决策点用波次驱动跑完全部 rollout，再交 [`official_result`] 走生产汇总。
/// 侧车由本结构持有并**跨决策点复用**；请求保持串行（整锁），不设计多局共享队列。
///
/// ❗这是**接入冒烟**，不是生产教师：没有合并动作、没有阶段开关、没有超级拉面
/// 平局处理。它能回答「接得通吗」，回答不了「策略谁强」。
struct WaveTrainer<'a> {
    /// 网络策略
    nn: &'a RamenNnTrainer,
    /// 只用它的官方汇总
    search: &'a FlatSearch<RamenGame>,
    /// 常驻侧车（跨决策点复用）
    sidecar: Mutex<Sidecar>,
    /// 侧车分配的物理批上限；固定档时它就是恒定档位
    max_batch: usize,
    /// 是否按根选择物理批档位
    adaptive: bool,
    /// 是否启用同轨迹 policy 复用
    cache_on: bool,
    /// 每候选 rollout 条数
    n: usize,
    /// 非搜索出口
    fallback: RecommendedRamenTrainer,
    /// 整局搜索的叶深度**路由**
    ///
    /// [`LeafRoute::Uniform`] 逐字保持既有行为；[`LeafRoute::HybridY3Full`] 只把
    /// 第三年 `RegionSelect` 退回完整续跑 + 原 rf，其余根仍按 mean 聚合。
    route: LeafRoute,
    /// 累计统计
    stats: Mutex<GameStats>
}

impl Trainer<RamenGame> for WaveTrainer<'_> {
    /// 决策点走波次搜索 + 官方汇总；候选唯一时直接返回
    ///
    /// # 错误
    ///
    /// 波次驱动、侧车通信或官方汇总失败时报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        if actions.len() <= 1 {
            // 无从选起：不动 rng，也不占批
            let mut st = self.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
            st.trivial += 1;
            return Ok(0);
        }
        let t0 = Instant::now();
        // ❗路由只看**当前根**：rollout 途中经过第三年地区不改路由、不起嵌套搜索
        let route = self.route.decide(game)?;
        // 预跑与官方汇总共用同一张种子表：clone 派生，原 rng 留给 search_with
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let batch = plan_batch(self.adaptive, self.max_batch, actions.len() * self.n);
        let out = {
            let mut sc = self.sidecar.lock().map_err(|_| anyhow!("侧车锁被毒化"))?;
            run_gpu_wave(
                game,
                actions,
                self.n,
                &seeds,
                self.nn,
                &mut sc,
                batch,
                false,
                self.cache_on,
                route.depth,
                false
            )?
        };
        let wave = &out.stats;
        // ❗两条路必须分开：`Full` 仍走生产的 rf 加权汇总，一个字都不改；
        // 截断走 mean 选择器（rf 不参与聚合），且 `raw_cells` 会因叶结果直接报错。
        let mut leaf_deferrals = 0usize;
        let (best, rf_actual) = match route.depth {
            LeafDepth::Full => {
                let cells = out.raw_cells()?;
                let (b, _, rf, _) = official_result_full(self.search, game, actions, &cells, rng)?;
                (b, rf)
            }
            LeafDepth::Turns(_) => {
                let cells = out.arm_cells();
                leaf_deferrals = cells.iter().filter_map(|c| c.deferrals).sum::<u32>() as usize;
                // ❗这个 0.0 只是**本根**的实际聚合口径，不写回 `self.search`，
                // 也不写回任何公共配置：下一个走 full 的根仍按原 rf 汇总。
                (mean_select(&cells, actions.len(), 0, self.n)?.0, 0.0)
            }
        };
        let mut st = self.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
        match route.depth {
            LeafDepth::Full => st.route_full += 1,
            LeafDepth::Turns(_) => st.route_trunc += 1
        }
        let seq = st.searched;
        st.route_rows.push(RouteRow {
            seq,
            turn: game.turn(),
            stage: format!("{:?}", game.stage),
            region_year_idx: route.region_year_idx,
            candidates: actions.len(),
            route: route.tag,
            depth: route.depth.tag(),
            objective: route.depth.objective_tag(),
            rf_actual,
            policy_requests: wave.policy_requests,
            leaf_requests: wave.leaf_requests,
            terminal_results: wave.terminal_results,
            leaf_results: wave.leaf_results,
            wall_s: t0.elapsed().as_secs_f64()
        });
        *st.tiers.entry(wave.batch).or_insert(0) += 1;
        st.setup_s += wave.setup_s;
        st.cache.merge(&wave.cache);
        st.note_fill(wave.served, wave.slots);
        st.searched += 1;
        st.leaf_requests += wave.leaf_requests;
        st.leaf_results += wave.leaf_results;
        st.deferrals += leaf_deferrals;
        st.candidates += actions.len();
        st.waves += wave.waves;
        st.served += wave.served;
        st.slots += wave.slots;
        st.rollouts += actions.len() * self.n;
        st.cpu_wall_s += wave.cpu_wall_s;
        st.gpu_wall_s += wave.gpu_wall_s;
        st.cpu_in_traj_s += wave.cpu_in_traj_s;
        st.search_s += t0.elapsed().as_secs_f64();
        Ok(best)
    }

    /// 事件选项转交手写（与生产 rollout 基策一致）
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.fallback.select_choice(game, choices, rng)
    }

    /// 事件选项（含友人事件特例）转交手写
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.fallback.select_event_choice(game, event, choices, rng)
    }
}

/// 生产教师的一步决策记录（正式接入一致性用）
struct TeacherStep {
    /// 决策序号
    seq: usize,
    /// 回合
    turn: i32,
    /// 阶段
    stage: String,
    /// 合法候选，保持原顺序
    actions: Vec<String>,
    /// 选中的候选下标
    chosen: usize,
    /// 决策后 RNG 的非破坏性探针（检查 RNG 消耗是否一致）
    rng_probe: u64
}

/// 记录生产教师逐步决策的包装器
///
/// 只做记录与转发：合并动作、阶段门控、平局处理、RNG 消耗**全部由内层
/// [`RamenMctsTrainer`] 自己决定**，本包装器不参与。
struct RecordingTeacher {
    /// 生产教师
    inner: RamenMctsTrainer,
    /// 逐步记录
    steps: Mutex<Vec<TeacherStep>>
}

impl Trainer<RamenGame> for RecordingTeacher {
    /// 转发给生产教师，并记录该步的候选、选择与 RNG 指纹
    ///
    /// # 错误
    ///
    /// 生产教师报错或记录锁被毒化时报错。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        let chosen = self.inner.select_action(game, actions, rng)?;
        let mut steps = self.steps.lock().map_err(|_| anyhow!("记录锁被毒化"))?;
        let seq = steps.len();
        steps.push(TeacherStep {
            seq,
            turn: game.turn(),
            stage: format!("{:?}", game.stage),
            actions: actions.iter().map(action_repr).collect(),
            chosen,
            // 克隆体探针：不消耗生产随机流
            rng_probe: rng.clone().next_u64()
        });
        Ok(chosen)
    }

    /// 转发给生产教师
    ///
    /// # 错误
    ///
    /// 生产教师报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.inner.select_choice(game, choices, rng)
    }

    /// 转发给生产教师
    ///
    /// # 错误
    ///
    /// 生产教师报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.inner.select_event_choice(game, event, choices, rng)
    }
}

/// 按生产口径搭一台教师
///
/// 与 `ramen_space_bench` 同一条构造链：`RamenSearchStages::all()` +
/// `RamenSelection::Score` + `with_nn_rollout`。**合并动作、阶段门控、平局处理
/// 与 RNG 消耗一律沿用生产实现**，本工具不再自写简化版。
fn build_teacher(
    config: SearchConfig, nn: Option<Arc<RamenNnTrainer>>, backend: Option<Arc<dyn RamenBatchRollout>>
) -> RamenMctsTrainer {
    let mcts = RamenMctsTrainer::new(config.clone())
        .with_stages(RamenSearchStages::all())
        .with_selection(RamenSelection::Score);
    // 不给 nn 就是手写 rollout 基策：配对实验的另一臂
    let mut mcts = match nn {
        Some(nn) => mcts.with_nn_rollout(nn, None),
        None => mcts
    };
    if let Some(b) = backend {
        // `search` 是公开字段；取出来挂上后端再放回，避免改动生产 trainer
        let search = std::mem::replace(&mut mcts.search, FlatSearch::<RamenGame>::new(config));
        mcts.search = search.with_batch_rollout(b);
    }
    mcts
}

/// 用给定教师打完一局，返回逐步记录与终局评分
///
/// 建局与随机流走 [`seeded_rngs`]，与 `ramen_space_bench` 是同一个函数。
///
/// ❗**但传进去的基种子口径不同，两个工具的同一个 `--seed` 不是同一批世界**：
/// `ramen_space_bench::run_plan` 用 `seed + plan_index × 1_000_003`
/// （让各计划的种子段互不重叠），本工具**直接用 `args.seed`**。
/// 于是只有 `--plan-index 0`（偏移量为 0）时两边恰好对齐，其余计划的同名 seed
/// 指向完全不同的世界。
///
/// ⇒ 跨这两个入口做比较前，必须先对齐**有效基种子**并比对实际世界字段；
/// 只在 plan 0 上验证过「两边一致」不能推广到其他计划。
///
/// # 错误
///
/// 建局或规则推进失败时报错。
fn play_with_teacher(
    args: &RootArgs, uma: u32, deck: &[u32; 6], inherit: &InheritInfo, teacher: RamenMctsTrainer
) -> Result<(Vec<TeacherStep>, SearchScore, f64)> {
    let (mut rng, rule_master) = seeded_rngs(args.seed, args.run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let rec = RecordingTeacher {
        inner: teacher,
        steps: Mutex::new(Vec::new())
    };
    let t0 = Instant::now();
    while game.next() {
        game.run_stage(&rec, &mut rng)?;
    }
    game.on_simulation_end(&rec, &mut rng)?;
    // 终局结果提取计入墙钟：统一边界是「最后一次官方汇总与终局结果提取完成」
    let score = game.search_score();
    let wall = t0.elapsed().as_secs_f64();
    let steps = rec.steps.into_inner().map_err(|_| anyhow!("记录锁被毒化"))?;
    Ok((steps, score, wall))
}

/// 把生产教师的逐步决策落成 CSV（**实际值**，不落哈希）
///
/// # 错误
///
/// 创建或写文件失败时报错。
fn write_steps_csv(path: &PathBuf, steps: &[TeacherStep]) -> Result<()> {
    let f = std::fs::File::create(path).with_context(|| format!("创建 steps_csv 失败: {}", path.display()))?;
    let mut f = std::io::BufWriter::new(f);
    writeln!(f, "seq,turn,stage,n_actions,actions,chosen,rng_probe")?;
    for st in steps {
        writeln!(
            f,
            "{},{},{},{},{},{},{:#018x}",
            st.seq,
            st.turn,
            st.stage,
            st.actions.len(),
            st.actions.join("|"),
            st.chosen,
            st.rng_probe
        )?;
    }
    Ok(())
}

/// 逐字段比较两条教师决策序列
///
/// 返回不同的步数；差异明细直接打印，**不做哈希**。
fn diff_steps(a: &[TeacherStep], b: &[TeacherStep]) -> usize {
    if a.len() != b.len() {
        println!("    ❗决策步数不同：{} vs {}", a.len(), b.len());
    }
    let mut bad = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let mut why = Vec::new();
        if x.turn != y.turn {
            why.push(format!("turn {} vs {}", x.turn, y.turn));
        }
        if x.stage != y.stage {
            why.push(format!("stage {} vs {}", x.stage, y.stage));
        }
        if x.actions != y.actions {
            why.push(format!("候选 {} 个 vs {} 个（或顺序不同）", x.actions.len(), y.actions.len()));
        }
        if x.chosen != y.chosen {
            why.push(format!("选中 {} vs {}", x.chosen, y.chosen));
        }
        if x.rng_probe != y.rng_probe {
            why.push(format!("RNG 探针 {:#018x} vs {:#018x}", x.rng_probe, y.rng_probe));
        }
        if !why.is_empty() {
            bad += 1;
            if bad <= 10 {
                println!("    第 {} 步 t{} {}: {}", x.seq, x.turn, x.stage, why.join("; "));
            }
        }
    }
    bad + a.len().abs_diff(b.len())
}

/// 用生产汇总路径给出官方搜索结果
///
/// 并行粒度只决定 rollout 在哪跑；排序口径（rank 加权均值 + radical factor）
/// 一律由 [`FlatSearch::search_with`] 决定。这里把已算好的结果做成备忘表，
/// 交给它按生产实现汇总。
///
/// # 错误
///
/// 备忘表缺项（说明预跑的种子与内核派生的不一致）或内核报错时报错。
fn official_result(
    search: &FlatSearch<RamenGame>, game: &RamenGame, actions: &[RamenAction], cells: &[RawCell], rng: &mut StdRng
) -> Result<(usize, Vec<f64>)> {
    let (best, ranked, _rf, _means) = official_result_full(search, game, actions, cells, rng)?;
    Ok((best, ranked))
}

/// 同 [`official_result`]，另外带出**本次实际生效的 rf** 与各候选的普通均值
///
/// 实际 rf 必须原样带出来：生产 rf 随回合缩放，`radical_factor_max` 只是上限，
/// 报告里写死 1.4 就是错的。普通均值一并带出，供均值选择器与生产 rf=0 对拍。
///
/// # 错误
///
/// 与 [`official_result`] 完全一致。
fn official_result_full(
    search: &FlatSearch<RamenGame>, game: &RamenGame, actions: &[RamenAction], cells: &[RawCell], rng: &mut StdRng
) -> Result<(usize, Vec<f64>, f64, Vec<f64>)> {
    let mut table: HashMap<(usize, u64), SearchScore> = HashMap::new();
    for c in cells {
        table.insert(
            (c.candidate, c.seed),
            SearchScore {
                score: c.score,
                score_pt: c.score_pt
            }
        );
    }
    let expected = cells.len() / actions.len();
    let out = search.search_with(game, actions, rng, |_g, a, seed| {
        let idx = actions
            .iter()
            .position(|x| x == a)
            .ok_or_else(|| anyhow!("汇总时找不到候选下标"))?;
        table
            .get(&(idx, seed))
            .copied()
            .ok_or_else(|| anyhow!("备忘表缺 (候选 {idx}, 种子 {seed:#018x})：预跑与内核的种子派生不一致"))
    })?;
    // 守门：均匀分配下每个候选都必须恰好消费 n 条。计数不等于 n 说明汇总口径不对
    // （例如 UCB 被打开），这时排序值看着正常却不可比。
    for (i, (a, _)) in out.action_results.iter().enumerate() {
        ensure!(
            a.count() as usize == expected,
            "候选 {i} 的汇总计数 {} 与预期 {expected} 不符：汇总口径不是均匀分配",
            a.count()
        );
    }
    // 报告排序键本身（rank 加权均值），不是普通均值——普通均值不是教师的判据
    let ranked = out
        .action_results
        .iter()
        .map(|(a, _)| a.weighted_mean(out.radical_factor))
        .collect();
    let means = out.action_results.iter().map(|(a, _)| a.mean()).collect();
    Ok((out.best_action_idx, ranked, out.radical_factor, means))
}

// ============================================================================
// leaf value 截断试点（只被 [`Mode::LeafPilot`] 使用）
// ============================================================================

/// 预登记里的一个根规格
#[derive(Debug, Clone, Deserialize, Serialize)]
struct RootSpec {
    /// 根标识（进文件名与报告）
    id: String,
    /// 来源局标识（G1..G9）
    game: String,
    /// 采样空间里的计划下标
    plan_index: usize,
    /// 局号（世界）
    run_idx: u64,
    /// 目标回合
    root_turn: i32,
    /// 目标阶段名
    root_stage: String,
    /// 预登记里的配额名
    quota: String
}

/// 截断模式专用的均值选择器（rf **不参与**聚合）
///
/// 并列规则**逐字复刻**生产 `SearchOutput::with_terminals`：那里用
/// `max_by(partial_cmp)`，Rust 的 `max_by` 在相等时保留**后一个**元素，
/// 所以这里也必须用 `max_by`，不能写成 `>` 比较（那会变成前一个胜出）。
///
/// ❗刻意**不**走 `ActionResult`：那条路会把每个分数落进 10 万槽整数直方图，
/// 叶预测是浮点，落格要取整。`weighted_mean(0)` 恰好绕开直方图直接返回
/// `sum/num`，但本函数不依赖那个巧合。累加顺序按 `j` 升序，与生产逐条 `push`
/// 的顺序一致，故真实终局样本上两者应当逐位相同。
///
/// # 错误
///
/// 候选数为 0、候选下标越界、估值非有限，或某候选在选定列区间内条数不等于期望时报错。
fn mean_select(cells: &[ArmCell], candidates: usize, j_lo: usize, j_hi: usize) -> Result<(usize, Vec<f64>)> {
    let grid = validate_cells(cells, candidates, j_lo, j_hi)?;
    let width = j_hi - j_lo;
    // 按 `j` 升序逐条累加：浮点加法不满足结合律，运算顺序必须与生产逐条 `push` 一致
    let means: Vec<f64> = (0..candidates)
        .map(|cand| {
            let mut acc = 0.0f64;
            for col in 0..width {
                acc += grid[cand * width + col];
            }
            acc / width as f64
        })
        .collect();
    let best = means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow!("均值表为空"))?;
    Ok((best, means))
}

/// 逐条核对一批结果的**身份完整性**
///
/// 只数条数是不够的：「候选 A 的第 3 列出现两次、第 5 列缺失」条数照样对得上，
/// 均值却已经错了。本函数在选定列区间内核对三件事：
///
/// 1. 每个 `(候选, j)` **恰好出现一次**（无重复、无缺失）；
/// 2. 同一个 `j` 在**所有候选**上共享同一个种子——这是 CRN 配对的前提，
///    错位会让「配对差」变成两批不同世界的差；
/// 3. 每个估值都是有限值。
///
/// 校验通过后返回按 `[候选 * 宽度 + 列]` 排好的估值网格，供聚合按固定顺序累加。
///
/// # 错误
///
/// 候选数为 0、列区间非法、下标越界、重复列、缺列、种子错位或非有限值时报错。
fn validate_cells(cells: &[ArmCell], candidates: usize, j_lo: usize, j_hi: usize) -> Result<Vec<f64>> {
    ensure!(candidates > 0, "候选数为 0");
    ensure!(j_lo < j_hi, "列区间 [{j_lo}, {j_hi}) 非法");
    let width = j_hi - j_lo;
    let mut seen = vec![0u32; candidates * width];
    let mut grid = vec![f64::NAN; candidates * width];
    let mut seed_of_j: Vec<Option<u64>> = vec![None; width];
    for c in cells {
        if c.j < j_lo || c.j >= j_hi {
            continue;
        }
        ensure!(
            c.candidate < candidates,
            "结果里的候选下标 {} 越界（共 {candidates} 个）",
            c.candidate
        );
        let col = c.j - j_lo;
        let k = c.candidate * width + col;
        seen[k] += 1;
        ensure!(
            seen[k] == 1,
            "候选 {} 第 {} 列出现 {} 次：同一 (候选, j) 只能有一条",
            c.candidate,
            c.j,
            seen[k]
        );
        match seed_of_j[col] {
            None => seed_of_j[col] = Some(c.seed),
            Some(prev) => ensure!(
                prev == c.seed,
                "第 {} 列在候选 {} 上的种子 {:#018x} 与其他候选的 {:#018x} 不同：CRN 配对已错位",
                c.j,
                c.candidate,
                c.seed,
                prev
            )
        }
        ensure!(
            c.value.is_finite(),
            "候选 {} 第 {} 列的估值不是有限值: {}",
            c.candidate,
            c.j,
            c.value
        );
        grid[k] = c.value;
    }
    for cand in 0..candidates {
        for j in j_lo..j_hi {
            ensure!(
                seen[cand * width + (j - j_lo)] == 1,
                "候选 {cand} 在列 [{j_lo}, {j_hi}) 内缺第 {j} 列"
            );
        }
    }
    Ok(grid)
}

/// 独立评估列上的**真实终局**均值
///
/// # 错误
///
/// 区间内混进叶结果（说明评估列被截断污染）、条数不符或估值非有限时报错。
fn audit_means(cells: &[ArmCell], candidates: usize, j_lo: usize, j_hi: usize) -> Result<Vec<f64>> {
    for c in cells {
        if c.j >= j_lo && c.j < j_hi {
            ensure!(
                c.kind == "terminal",
                "评估列 [{j_lo}, {j_hi}) 内出现叶结果（候选 {}, j {}）：audit 必须全是真实终局",
                c.candidate,
                c.j
            );
        }
    }
    let (_, means) = mean_select(cells, candidates, j_lo, j_hi)?;
    Ok(means)
}

/// 两个**已固定**候选在同一批列上的配对差
///
/// 返回 `(配对差均值, 配对差标准误, 配对列数)`。按列 CRN 逐列求差再算方差，
/// **不**用两个候选各自的绝对方差之和代替——那会高估噪声。
///
/// # 错误
///
/// 某列缺一边、或区间内条数不符时报错。
fn paired_diff(cells: &[ArmCell], a: usize, b: usize, j_lo: usize, j_hi: usize) -> Result<(f64, f64, usize)> {
    if a == b {
        return Ok((0.0, 0.0, j_hi - j_lo));
    }
    let mut va: HashMap<usize, f64> = HashMap::new();
    let mut vb: HashMap<usize, f64> = HashMap::new();
    for c in cells {
        if c.j < j_lo || c.j >= j_hi {
            continue;
        }
        if c.candidate == a {
            va.insert(c.j, c.value);
        } else if c.candidate == b {
            vb.insert(c.j, c.value);
        }
    }
    let mut diffs = Vec::with_capacity(j_hi - j_lo);
    for j in j_lo..j_hi {
        let x = va.get(&j).ok_or_else(|| anyhow!("候选 {a} 缺第 {j} 列"))?;
        let y = vb.get(&j).ok_or_else(|| anyhow!("候选 {b} 缺第 {j} 列"))?;
        diffs.push(x - y);
    }
    let n = diffs.len();
    ensure!(n > 1, "配对列数 {n} 不足以估方差");
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n as f64 - 1.0);
    Ok((mean, (var / n as f64).sqrt(), n))
}

/// 诊断模式下的逐条配对残差统计
///
/// 只有 `--leaf-diag` 打开时才有值。参与配对的是「同一条 rollout 既给了叶估值、
/// 又从叶状态用同一个 RNG 续跑到了真实终局」的那些条。
#[derive(Debug, Clone, Serialize)]
struct DiagStats {
    /// 参与配对的条数
    paired: usize,
    /// 与 full 臂同 `(候选, j)` 的终局分**逐位相同**的条数
    matches_full: usize,
    /// 与 full 臂不一致的条数（前缀与 RNG 都相同 ⇒ 应恒为 0）
    mismatches_full: usize,
    /// 在 full 臂里找不到对应 `(候选, j)` 的条数
    missing_in_full: usize,
    /// 逐条偏差 `叶估值 − 真实终局` 的均值
    bias_mean: f64,
    /// 逐条偏差的样本标准差
    bias_sd: f64,
    /// 各候选的平均偏差
    bias_by_candidate: Vec<f64>,
    /// **去掉全根共同偏移**后各候选的相对偏差（`bias_by_candidate − bias_mean`）
    ///
    /// 这一项才是排序关心的量：全体同减一个常数不改变 argmax。
    relative_bias: Vec<f64>,
    /// 相对偏差的最大绝对值
    relative_bias_max_abs: f64,
    /// 同一批列上各候选的**真实终局**均值；没有配对样本的候选为 `None`
    paired_terminal_means: Vec<Option<f64>>,
    /// 各候选的配对条数
    pairs_by_candidate: Vec<usize>,
    /// 按真实终局均值选出的候选：「叶若完美」本臂应当选的那个
    ///
    /// 只有**全部候选都有配对样本**时才给值：部分候选缺样本时各候选不可比，
    /// 这时宁可留空也不拿不完整的集合冒充 oracle。
    oracle_best: Option<usize>
}

/// 算诊断模式的配对残差
///
/// `full_cells` 用于核对续跑终局与 full 臂**同 `(候选, j)`** 是否逐位相同：
/// 两边的前缀、种子与策略都一样，不同就是接线错了。
///
/// 返回 `None` 表示本臂**一条配对样本都没有**——当 `H` 超出剩余回合时全部 rollout
/// 走真实终局，这是正确行为而不是错误。
///
/// # 错误
///
/// 候选下标越界时报错。
fn diag_stats(cells: &[ArmCell], full_cells: &[ArmCell], candidates: usize) -> Result<Option<DiagStats>> {
    let mut full_by_key: HashMap<(usize, usize), f64> = HashMap::with_capacity(full_cells.len());
    for c in full_cells {
        if c.kind == "terminal" {
            full_by_key.insert((c.candidate, c.j), c.value);
        }
    }
    let mut bias = Vec::new();
    let mut sum_bias = vec![0.0f64; candidates];
    let mut sum_term = vec![0.0f64; candidates];
    let mut cnt = vec![0usize; candidates];
    let mut matches_full = 0usize;
    let mut mismatches_full = 0usize;
    let mut missing_in_full = 0usize;
    for c in cells {
        let Some(tt) = c.paired_terminal else {
            continue;
        };
        ensure!(c.candidate < candidates, "候选下标 {} 越界", c.candidate);
        match full_by_key.get(&(c.candidate, c.j)) {
            Some(f) => {
                if f.to_bits() == tt.to_bits() {
                    matches_full += 1;
                } else {
                    mismatches_full += 1;
                }
            }
            None => missing_in_full += 1
        }
        let d = c.value - tt;
        bias.push(d);
        sum_bias[c.candidate] += d;
        sum_term[c.candidate] += tt;
        cnt[c.candidate] += 1;
    }
    if bias.is_empty() {
        return Ok(None);
    }
    let n = bias.len() as f64;
    let bias_mean = bias.iter().sum::<f64>() / n;
    let bias_sd = if bias.len() > 1 {
        (bias.iter().map(|d| (d - bias_mean) * (d - bias_mean)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    // 没有配对样本的候选留 0 偏差并在 `pairs_by_candidate` 里显式记 0 条，不伪造均值
    let bias_by_candidate: Vec<f64> = sum_bias
        .iter()
        .zip(cnt.iter())
        .map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 })
        .collect();
    let relative_bias: Vec<f64> = bias_by_candidate
        .iter()
        .zip(cnt.iter())
        .map(|(b, c)| if *c == 0 { 0.0 } else { b - bias_mean })
        .collect();
    let relative_bias_max_abs = relative_bias
        .iter()
        .zip(cnt.iter())
        .filter(|(_, c)| **c > 0)
        .fold(0.0f64, |m, (x, _)| m.max(x.abs()));
    let paired_terminal_means: Vec<Option<f64>> = sum_term
        .iter()
        .zip(cnt.iter())
        .map(|(s, c)| (*c > 0).then(|| s / *c as f64))
        .collect();
    // 只有所有候选都有配对样本时 oracle 才可比
    let oracle_best = if cnt.iter().all(|c| *c > 0) {
        paired_terminal_means
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.map(|v| (i, v)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
    } else {
        None
    };
    Ok(Some(DiagStats {
        paired: bias.len(),
        matches_full,
        mismatches_full,
        missing_in_full,
        bias_mean,
        bias_sd,
        bias_by_candidate,
        relative_bias,
        relative_bias_max_abs,
        paired_terminal_means,
        pairs_by_candidate: cnt,
        oracle_best
    }))
}

/// 一臂在一个根上的产出摘要
#[derive(Debug, Clone, Serialize)]
struct ArmSummary {
    /// 臂名（full / h4 / h8）
    arm: String,
    /// 本臂的列数
    n: usize,
    /// 本臂的实际聚合目标
    objective: String,
    /// policy 决策请求数
    policy_requests: usize,
    /// 叶估值请求数
    leaf_requests: usize,
    /// 两类请求之和（应与侧车实际服务数一致）
    total_requests: usize,
    /// 侧车实际服务的行数（核对用）
    served: usize,
    /// 真实终局收尾条数
    terminal_results: usize,
    /// 叶截断收尾条数
    leaf_results: usize,
    /// 波次数
    waves: usize,
    /// 物理格位数
    slots: usize,
    /// 波次循环墙钟
    wall_s: f64,
    /// 其中侧车往返
    gpu_s: f64,
    /// 其中 CPU 并行段
    cpu_s: f64,
    /// policy 缓存口径内条数
    cache_eligible: usize,
    /// policy 缓存命中条数
    cache_hits: usize,
    /// 叶所在回合的直方图（仅截断臂）
    leaf_turn_hist: Vec<(i32, usize)>,
    /// 叶所在阶段的直方图（仅截断臂）
    leaf_stage_hist: Vec<(String, usize)>,
    /// 实际跨越 turn 数的直方图（仅截断臂）
    turn_delta_hist: Vec<(i32, usize)>,
    /// 顺延次数合计（仅截断臂）
    deferrals: usize,
    /// 诊断模式下「记了叶又续跑到真实终局」的条数
    diag_continued: usize,
    /// 诊断模式下，叶边界上 policy 本来会 `Resolved`（不发请求）却仍然发了叶请求的条数
    leaf_at_resolved: usize,
    /// 诊断模式下的配对残差统计；未开 `--leaf-diag` 时为 `None`
    diag: Option<DiagStats>,
    /// 本臂选中的候选下标
    best: usize,
    /// 各候选的聚合值
    values: Vec<f64>
}

/// 把一个直方图 map 排成稳定顺序
fn hist_sorted<K: Ord + Clone>(m: &HashMap<K, usize>) -> Vec<(K, usize)> {
    let mut v: Vec<(K, usize)> = m.iter().map(|(k, c)| (k.clone(), *c)).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// 落一臂的逐 rollout 原始结果
///
/// 列里带**类型标记**与缺失语义：叶没有 `score_pt` 与终局五维，写空而不是 0。
///
/// # 错误
///
/// 写文件失败时报错。
fn write_arm_csv(path: &PathBuf, cells: &[ArmCell]) -> Result<()> {
    let mut buf = String::with_capacity(cells.len() * 72 + 128);
    buf.push_str("candidate,j,seed,kind,value,score_pt,paired_terminal,leaf_turn,leaf_stage,turn_delta,deferrals\n");
    for c in cells {
        buf.push_str(&format!(
            "{},{},{:#018x},{},{},{},{},{},{},{},{}\n",
            c.candidate,
            c.j,
            c.seed,
            c.kind,
            c.value,
            c.score_pt.map(|x| x.to_string()).unwrap_or_default(),
            c.paired_terminal.map(|x| x.to_string()).unwrap_or_default(),
            c.leaf_turn.map(|x| x.to_string()).unwrap_or_default(),
            c.leaf_stage.clone().unwrap_or_default(),
            c.turn_delta.map(|x| x.to_string()).unwrap_or_default(),
            c.deferrals.map(|x| x.to_string()).unwrap_or_default()
        ));
    }
    fs::write(path, buf).with_context(|| format!("写 {} 失败", path.display()))?;
    Ok(())
}

/// 从一臂的结果汇总出摘要
fn summarize_arm(
    arm: &str, n: usize, depth: LeafDepth, out: &WaveOut, cells: &[ArmCell], best: usize, values: Vec<f64>
) -> ArmSummary {
    let mut turn_hist: HashMap<i32, usize> = HashMap::new();
    let mut stage_hist: HashMap<String, usize> = HashMap::new();
    let mut delta_hist: HashMap<i32, usize> = HashMap::new();
    let mut deferrals = 0usize;
    for c in cells {
        if let Some(t) = c.leaf_turn {
            *turn_hist.entry(t).or_insert(0) += 1;
        }
        if let Some(st) = c.leaf_stage.as_ref() {
            *stage_hist.entry(st.clone()).or_insert(0) += 1;
        }
        if let Some(d) = c.turn_delta {
            *delta_hist.entry(d).or_insert(0) += 1;
        }
        deferrals += c.deferrals.unwrap_or(0) as usize;
    }
    let st = &out.stats;
    ArmSummary {
        arm: arm.to_string(),
        n,
        objective: depth.objective().to_string(),
        policy_requests: st.policy_requests,
        leaf_requests: st.leaf_requests,
        total_requests: st.policy_requests + st.leaf_requests,
        served: st.served,
        terminal_results: st.terminal_results,
        leaf_results: st.leaf_results,
        waves: st.waves,
        slots: st.slots,
        wall_s: st.total_wall_s,
        gpu_s: st.gpu_wall_s,
        cpu_s: st.cpu_wall_s,
        cache_eligible: st.cache.eligible,
        cache_hits: st.cache.hits,
        leaf_turn_hist: hist_sorted(&turn_hist),
        leaf_stage_hist: hist_sorted(&stage_hist),
        turn_delta_hist: hist_sorted(&delta_hist),
        deferrals,
        diag_continued: st.diag_continued,
        leaf_at_resolved: st.leaf_at_resolved,
        diag: None,
        best,
        values
    }
}


/// 生产 rf=0 与本工具均值选择器的对拍
#[derive(Debug, Clone, Serialize)]
struct MeanCheck {
    /// 生产路径在 `radical_factor_max=0` 下的选中下标
    prod_best: usize,
    /// 本工具均值选择器的选中下标
    bench_best: usize,
    /// 两边各候选均值的最大绝对差
    max_abs_diff: f64,
    /// 两边各候选均值是否**逐位**相同
    bitwise_equal: bool
}

/// 一臂在独立评估列上的配对损失
#[derive(Debug, Clone, Serialize)]
struct AuditLoss {
    /// 臂名
    arm: String,
    /// 本臂选中的候选
    best: usize,
    /// `audit_mean(full-rf 选) − audit_mean(本臂选)`；允许为负，不裁 0
    vs_full_rf: f64,
    /// 上一项的配对标准误（按列 CRN 逐列求差）
    vs_full_rf_se: f64,
    /// `audit_mean(full-mean 选) − audit_mean(本臂选)`
    vs_full_mean: f64,
    /// 上一项的配对标准误
    vs_full_mean_se: f64,
    /// 选择列上的观察 regret（❗有选择偏差，不是金标准）
    sel_regret_observed: f64,
    /// 是否与 full-rf 选同一个候选
    top1_same_as_full_rf: bool,
    /// 是否与离线 full-mean 选同一个候选
    top1_same_as_full_mean: bool
}

/// 一个根上的全部结果
#[derive(Debug, Clone, Serialize)]
struct RootSummary {
    /// 根规格（来自预登记）
    spec: RootSpec,
    /// 马娘
    uma: u32,
    /// 卡组
    deck: Vec<u32>,
    /// 构成名
    shape: String,
    /// 根的实际回合
    turn: i32,
    /// 根的实际阶段
    stage: String,
    /// 根上的合法候选数
    candidates: usize,
    /// 候选的**原顺序**文本
    actions: Vec<String>,
    /// 本次搜索的根种子（CRN 起点）
    root_seed: u64,
    /// 完整参考列数
    full_n: usize,
    /// 选择列数
    select_n: usize,
    /// 本根**实际生效**的 rf（不是上限）
    radical_factor_actual: f64,
    /// 配置里的 rf 上限
    radical_factor_max: f64,
    /// 均值选择器与生产 rf=0 的对拍
    mean_selector_check: MeanCheck,
    /// full-rf 在选择列上选中的候选
    full_rf_best: usize,
    /// 离线 full-mean 在选择列上选中的候选
    full_mean_best: usize,
    /// 独立评估列上各候选的真实终局均值
    audit_means: Vec<f64>,
    /// 各臂摘要
    arms: Vec<ArmSummary>,
    /// 各臂的 audit 配对损失
    audit_losses: Vec<AuditLoss>
}

/// 沿一局真实推进采集前 `k` 个决策局面的 754 维特征
///
/// 用真实局面而不是合成向量：合成向量测不出编码器与训练分布的量纲问题。
///
/// # 错误
///
/// 建局、推进或编码失败时报错。
fn collect_feature_rows(
    args: &RootArgs, uma: u32, deck: &[u32; 6], inherit: &InheritInfo, guide: &RootGuide, run_idx: u64, k: usize
) -> Result<Vec<Vec<f32>>> {
    let (mut rng, rule_master) = seeded_rngs(args.seed, run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let mut rows = Vec::with_capacity(k);
    while game.next() {
        if rows.len() < k && is_decision_stage(&game.stage) && leaf_safe_stage(&game.stage) {
            let f = encode(&game)?;
            ensure!(f.len() == INPUT_DIM, "特征长度 {} 与 INPUT_DIM={INPUT_DIM} 不符", f.len());
            rows.push(f);
        }
        game.run_stage(guide, &mut rng)?;
    }
    ensure!(!rows.is_empty(), "一局里没采到任何可用于 value 的决策局面");
    Ok(rows)
}

/// value 解码对拍：CPU（tract 跑集成 ONNX）对 GPU（侧车跑三个 checkpoint）
///
/// 归一化空间的误差与反归一化后的**绝对分数**误差**分开报告**：两者相差一个
/// `scale[0] ≈ 4240` 的因子，混称「1e-4」会把 0.4 分的差说成 1e-4 分。
///
/// # 错误
///
/// 计划越界、建局失败、推理失败、行数/形状不符或出现非有限值时报错。
fn run_value_check(
    args: &RootArgs, nn: &Arc<RamenNnTrainer>, space: &SamplingSpace, inherit: &InheritInfo, guide: &RootGuide
) -> Result<()> {
    let plans = space.plans();
    let plan = plans
        .get(args.plan_index)
        .ok_or_else(|| anyhow!("计划下标 {} 越界（共 {} 个）", args.plan_index, plans.len()))?;
    let rows = collect_feature_rows(args, plan.uma, &plan.deck, inherit, guide, args.run_idx, args.value_rows)?;
    println!(
        "value 对拍：plan{} run{} 取到 {} 个真实决策局面（上限 {}）",
        args.plan_index,
        args.run_idx,
        rows.len(),
        args.value_rows
    );
    let norm = nn.value_norm();
    println!(
        "  成员 0 常数 center[0]={} scale[0]={}",
        norm.center[0], norm.scale[0]
    );

    // CPU：tract 跑集成 ONNX，走生产的同一条解码路径
    let cpu: Vec<f64> = rows
        .iter()
        .map(|r| Ok(nn.infer_features(r.clone())?.value.mean))
        .collect::<Result<Vec<_>>>()?;

    // GPU：侧车跑三个 checkpoint，返回已按成员 0 尺度重归一化的 value
    let mut sc = Sidecar::start(args)?;
    sc.ensure_batch(args.batch)?;
    println!("  侧车就绪 {:.1} s | {}", sc.startup_s, sc.banner);
    let outs = sc.infer(&rows)?;
    ensure!(outs.len() == rows.len(), "侧车返回 {} 行，期望 {}", outs.len(), rows.len());

    let mut max_norm_err = 0.0f64;
    let mut max_score_err = 0.0f64;
    let mut max_policy_err = 0.0f32;
    let mut nonfinite = 0usize;
    let cpu_policy: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| Ok(nn.infer_features(r.clone())?.policy))
        .collect::<Result<Vec<_>>>()?;
    for (i, out) in outs.iter().enumerate() {
        ensure!(out.len() == OUTPUT_DIM, "第 {i} 行长 {} 与契约 {OUTPUT_DIM} 不符", out.len());
        if !out.iter().all(|x| x.is_finite()) {
            nonfinite += 1;
            continue;
        }
        let gpu_value = norm.denormalize(&out[VALUE_OFF..])?;
        // CPU 侧的归一化值由精确逆变换还原，避免第二次解码引入新口径
        let cpu_norm = (cpu[i] - norm.center[0]) / norm.scale[0];
        let gpu_norm = f64::from(out[VALUE_OFF]);
        max_norm_err = max_norm_err.max((cpu_norm - gpu_norm).abs());
        max_score_err = max_score_err.max((cpu[i] - gpu_value.mean).abs());
        for (a, b) in cpu_policy[i].iter().zip(out[..POLICY_DIM].iter()) {
            max_policy_err = max_policy_err.max((a - b).abs());
        }
    }
    ensure!(nonfinite == 0, "{nonfinite} 行侧车输出含非有限值");
    let cpu_min = cpu.iter().cloned().fold(f64::INFINITY, f64::min);
    let cpu_max = cpu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cpu_mean = cpu.iter().sum::<f64>() / cpu.len() as f64;
    println!("  CPU value.mean 区间 [{cpu_min:.1}, {cpu_max:.1}]，均值 {cpu_mean:.1}");
    println!("  归一化空间最大绝对差 {max_norm_err:.3e}（阈值 1e-4）");
    println!("  反归一化后**绝对分数**最大绝对差 {max_score_err:.4} 分（单独报告，不与上一行混称）");
    println!("  policy logits 最大绝对差 {max_policy_err:.3e}（同批次顺带核对）");
    ensure!(
        max_norm_err < 1e-4,
        "归一化 value 最大绝对差 {max_norm_err:.3e} 超过 1e-4"
    );
    Ok(())
}

/// 按 CLI 组出采样空间
///
/// # 错误
///
/// 版本名未注册，或构成解析失败时报错。
fn build_space(args: &RootArgs) -> Result<SamplingSpace> {
    match args.space_version.as_deref() {
        Some(name) => SamplingSpace::from_version(space_version_by_name(name)?),
        None => space_from_cli(None, &[])
    }
}

/// 决策点扫描模式：把 9 局的全部决策点落盘
///
/// # 错误
///
/// 计划下标越界、建局或推进失败、写文件失败时报错。
fn run_root_scan(args: &RootArgs, space: &SamplingSpace, inherit: &InheritInfo, guide: &RootGuide) -> Result<()> {
    let out_dir = args
        .leaf_out
        .as_ref()
        .ok_or_else(|| anyhow!("root-scan 需要 --leaf-out 指定输出目录"))?;
    fs::create_dir_all(out_dir).with_context(|| format!("建目录 {} 失败", out_dir.display()))?;
    let plans = space.plans();
    let plan = plans
        .get(args.plan_index)
        .ok_or_else(|| anyhow!("计划下标 {} 越界（共 {} 个）", args.plan_index, plans.len()))?;
    let t0 = Instant::now();
    let points = scan_decisions(args, plan.uma, &plan.deck, inherit, guide, args.run_idx)?;
    let mut buf = String::from("plan_index,run_idx,uma,shape,seq,turn,stage,candidates\n");
    for p in &points {
        buf.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            args.plan_index, args.run_idx, plan.uma, plan.shape, p.seq, p.turn, p.stage, p.candidates
        ));
    }
    let path = out_dir.join(format!("scan_plan{}_run{}.csv", args.plan_index, args.run_idx));
    ensure!(!path.exists(), "输出文件 {} 已存在，拒绝覆盖", path.display());
    fs::write(&path, buf).with_context(|| format!("写 {} 失败", path.display()))?;
    println!(
        "决策点扫描 plan{} run{} uma={} shape={} → {} 个决策点，{:.1} s，落 {}",
        args.plan_index,
        args.run_idx,
        plan.uma,
        plan.shape,
        points.len(),
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

/// leaf 试点：同一个根上跑 full 与若干截断深度
///
/// 各臂**克隆同一个根**、共用同一张种子表；full 臂跑 `--full-n` 列，
/// 前 `--select-n` 列用于选择，其余列作为独立评估列。
///
/// # 错误
///
/// 根清单读失败、计划越界、建根失败、波次驱动失败、聚合口径不符或写文件失败时报错。
fn run_leaf_pilot(
    args: &RootArgs, nn: &Arc<RamenNnTrainer>, depths: &[LeafDepth], space: &SamplingSpace, inherit: &InheritInfo,
    guide: &RootGuide
) -> Result<()> {
    let out_dir = args
        .leaf_out
        .as_ref()
        .ok_or_else(|| anyhow!("leaf-pilot 需要 --leaf-out 指定输出目录"))?;
    // ❗拒绝覆盖既有证据：重跑必须换新编号/子目录，失败日志原样保留
    ensure!(
        !out_dir.exists(),
        "输出目录 {} 已存在，拒绝覆盖既有证据；请换一个新编号或子目录",
        out_dir.display()
    );
    let raw_dir = out_dir.join("raw");
    fs::create_dir_all(&raw_dir).with_context(|| format!("建目录 {} 失败", raw_dir.display()))?;

    let specs: Vec<RootSpec> = match args.roots_file.as_ref() {
        Some(p) => {
            let text = fs::read_to_string(p).with_context(|| format!("读 {} 失败", p.display()))?;
            json_from_str(&text).with_context(|| format!("解析 {} 失败", p.display()))?
        }
        None => vec![RootSpec {
            id: format!("single_plan{}_run{}", args.plan_index, args.run_idx),
            game: "-".to_string(),
            plan_index: args.plan_index,
            run_idx: args.run_idx,
            root_turn: args.root_turn,
            root_stage: args.root_stage.clone().unwrap_or_default(),
            quota: "-".to_string()
        }]
    };
    ensure!(!specs.is_empty(), "根清单为空");
    // 根 ID 直接进文件名：重名会让后写的那份**静默覆盖**前一份
    let mut ids: HashSet<&str> = HashSet::with_capacity(specs.len());
    for spec in &specs {
        ensure!(!spec.id.is_empty(), "根清单里存在空 id");
        ensure!(
            spec.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "根 id {} 含会影响文件名的字符，只允许 ASCII 字母数字与 _ -",
            spec.id
        );
        ensure!(ids.insert(spec.id.as_str()), "根 id {} 重复：产物文件名会互相覆盖", spec.id);
    }
    println!(
        "leaf 试点：{} 个根 × (full n={} + {} 个截断臂 n={})，选择列 [0,{})，评估列 [{},{})",
        specs.len(),
        args.full_n,
        depths.len(),
        args.select_n,
        args.select_n,
        args.select_n,
        args.full_n
    );
    println!("  ❗截断臂的聚合目标是 mean：配置里的 radical_factor_max={} 在该模式下**不参与聚合**", args.radical_factor_max);

    // 选择器：两台只用于汇总的搜索（rollout 由闭包供给，不跑模拟）
    let cfg_sel = SearchConfig {
        search_n: args.select_n,
        radical_factor_max: args.radical_factor_max,
        use_ucb: false,
        ..SearchConfig::default()
    };
    let cfg_mean = SearchConfig {
        radical_factor_max: 0.0,
        ..cfg_sel.clone()
    };
    ensure!(!cfg_sel.use_ucb, "截断实验要求均匀分配，use_ucb 必须为 false");
    let search_rf = FlatSearch::<RamenGame>::new(cfg_sel);
    let search_mean = FlatSearch::<RamenGame>::new(cfg_mean);

    let mut sc = Sidecar::start(args)?;
    sc.ensure_batch(args.batch)?;
    println!("  侧车就绪 {:.1} s | {}", sc.startup_s, sc.banner);

    let plans = space.plans();
    let mut summaries: Vec<RootSummary> = Vec::with_capacity(specs.len());
    for spec in &specs {
        let plan = plans
            .get(spec.plan_index)
            .ok_or_else(|| anyhow!("根 {} 的计划下标 {} 越界", spec.id, spec.plan_index))?;
        let stage_want = (!spec.root_stage.is_empty()).then_some(spec.root_stage.as_str());
        let build_t = Instant::now();
        let (root, rng0) =
            build_root_with(args, plan.uma, &plan.deck, inherit, spec.root_turn, stage_want, guide, spec.run_idx)?;
        let actions = root.list_actions()?;
        ensure!(!actions.is_empty(), "根 {} 上没有合法候选", spec.id);
        let seeds = RolloutSeeds::from_rng(&mut rng0.clone());
        println!(
            "\n[{}] {} plan{} run{} uma={} shape={} → t{} {:?} 候选 {}（建根 {:.1} s）",
            spec.id,
            spec.quota,
            spec.plan_index,
            spec.run_idx,
            plan.uma,
            plan.shape,
            root.turn(),
            root.stage,
            actions.len(),
            build_t.elapsed().as_secs_f64()
        );

        // ---- full 臂：完整续跑，原 rf 行为
        let full = run_gpu_wave(
            &root,
            &actions,
            args.full_n,
            &seeds,
            nn,
            &mut sc,
            args.batch,
            false,
            args.policy_cache,
            LeafDepth::Full,
            false
        )?;
        let full_cells = full.arm_cells();
        let raw_all = full.raw_cells()?;
        let raw_sel: Vec<RawCell> = raw_all.into_iter().filter(|c| c.j < args.select_n).collect();
        let (best_rf, ranked_rf, rf_actual, _) =
            official_result_full(&search_rf, &root, &actions, &raw_sel, &mut rng0.clone())?;
        let (best_prod_mean, _, rf_zero, prod_means) =
            official_result_full(&search_mean, &root, &actions, &raw_sel, &mut rng0.clone())?;
        ensure!(rf_zero == 0.0, "rf=0 对照臂的实际 rf 是 {rf_zero}，不是 0");
        // 用**真实终局样本**证明本工具的均值选择器与生产 rf=0 一致
        let (best_bench_mean, bench_means) = mean_select(&full_cells, actions.len(), 0, args.select_n)?;
        let max_abs_diff = prod_means
            .iter()
            .zip(bench_means.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let bitwise_equal = prod_means
            .iter()
            .zip(bench_means.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let mean_check = MeanCheck {
            prod_best: best_prod_mean,
            bench_best: best_bench_mean,
            max_abs_diff,
            bitwise_equal
        };
        let audit = audit_means(&full_cells, actions.len(), args.select_n, args.full_n)?;
        write_arm_csv(&raw_dir.join(format!("{}_full.csv", spec.id)), &full_cells)?;
        let mut arms = vec![summarize_arm(
            "full",
            args.full_n,
            LeafDepth::Full,
            &full,
            &full_cells,
            best_rf,
            ranked_rf
        )];
        println!(
            "  full  n={} rf实际={:.3}（上限 {}）选中 {} | rf=0 选中 {} | 均值选择器 选中 {} 逐位一致={} 最大差={:.3e}",
            args.full_n, rf_actual, args.radical_factor_max, best_rf, best_prod_mean, best_bench_mean, bitwise_equal, max_abs_diff
        );
        println!(
            "        请求 policy {} + leaf {} = {}（侧车服务 {}）终局 {} 叶 {} 波次 {} 墙钟 {:.1} s",
            arms[0].policy_requests,
            arms[0].leaf_requests,
            arms[0].total_requests,
            arms[0].served,
            arms[0].terminal_results,
            arms[0].leaf_results,
            arms[0].waves,
            arms[0].wall_s
        );

        // ---- 截断臂
        let mut losses = Vec::with_capacity(depths.len());
        for &d in depths {
            let tag = d.tag();
            let out = run_gpu_wave(
                &root,
                &actions,
                args.select_n,
                &seeds,
                nn,
                &mut sc,
                args.batch,
                false,
                args.policy_cache,
                d,
                args.leaf_diag
            )?;
            let cells = out.arm_cells();
            let (best, means) = mean_select(&cells, actions.len(), 0, args.select_n)?;
            write_arm_csv(&raw_dir.join(format!("{}_{}.csv", spec.id, tag)), &cells)?;
            let mut summary = summarize_arm(&tag, args.select_n, d, &out, &cells, best, means.clone());
            if args.leaf_diag {
                match diag_stats(&cells, &full_cells, actions.len())? {
                    None => println!(
                        "  {tag}诊断 无配对样本：本臂全部 rollout 走真实终局（H 超出剩余回合），不是错误"
                    ),
                    Some(ds) => {
                println!(
                    "  {tag}诊断 配对 {} 条 | 与 full 终局逐位一致 {} / 不一致 {} / full 中缺 {}",
                    ds.paired, ds.matches_full, ds.mismatches_full, ds.missing_in_full
                );
                println!(
                    "        叶边界上 policy 本来会 Resolved（单候选/守门）的条数 {}：这些条仍然发了叶请求",
                    summary.leaf_at_resolved
                );
                println!(
                    "        偏差(叶−真终局) 均值 {:+.1} 标准差 {:.1} | **去掉根共同偏移后**候选相对偏差最大 {:.1}",
                    ds.bias_mean, ds.bias_sd, ds.relative_bias_max_abs
                );
                match ds.oracle_best {
                    Some(o) => println!(
                        "        本臂选 {best} / 同批列真实终局最优（oracle）选 {o}{}",
                        if best == o { "（一致）" } else { "（❗不一致）" }
                    ),
                    None => println!(
                        "        部分候选没有配对样本（各候选配对数 {:?}），不给 oracle",
                        ds.pairs_by_candidate
                    )
                }
                summary.diag = Some(ds);
                    }
                }
            }
            let (d_rf, se_rf, _) = paired_diff(&full_cells, best_rf, best, args.select_n, args.full_n)?;
            let (d_mean, se_mean, _) = paired_diff(&full_cells, best_prod_mean, best, args.select_n, args.full_n)?;
            losses.push(AuditLoss {
                arm: tag.clone(),
                best,
                vs_full_rf: d_rf,
                vs_full_rf_se: se_rf,
                vs_full_mean: d_mean,
                vs_full_mean_se: se_mean,
                sel_regret_observed: bench_means[best_prod_mean] - bench_means[best],
                top1_same_as_full_rf: best == best_rf,
                top1_same_as_full_mean: best == best_prod_mean
            });
            println!(
                "  {tag}    选中 {best} | audit 配对差 vs full-rf {:+.1} ± {:.1} | vs full-mean {:+.1} ± {:.1} | 选择列观察 regret {:+.1}",
                d_rf, se_rf, d_mean, se_mean, losses[losses.len() - 1].sel_regret_observed
            );
            println!(
                "        请求 policy {} + leaf {} = {}（侧车服务 {}）终局 {} 叶 {} 波次 {} 墙钟 {:.1} s 顺延 {}",
                summary.policy_requests,
                summary.leaf_requests,
                summary.total_requests,
                summary.served,
                summary.terminal_results,
                summary.leaf_results,
                summary.waves,
                summary.wall_s,
                summary.deferrals
            );
            arms.push(summary);
        }

        summaries.push(RootSummary {
            spec: spec.clone(),
            uma: plan.uma,
            deck: plan.deck.to_vec(),
            shape: plan.shape.to_string(),
            turn: root.turn(),
            stage: format!("{:?}", root.stage),
            candidates: actions.len(),
            actions: actions.iter().map(action_repr).collect(),
            root_seed: seeds.root(),
            full_n: args.full_n,
            select_n: args.select_n,
            radical_factor_actual: rf_actual,
            radical_factor_max: args.radical_factor_max,
            mean_selector_check: mean_check,
            full_rf_best: best_rf,
            full_mean_best: best_prod_mean,
            audit_means: audit,
            arms,
            audit_losses: losses
        });
        // 每根写一次：中途失败也保得住已完成的部分
        let path = out_dir.join("summary.json");
        let text = json_to_string_pretty(&summaries).context("序列化摘要失败")?;
        fs::write(&path, text).with_context(|| format!("写 {} 失败", path.display()))?;
    }
    println!("\n完成 {} 个根，摘要落 {}", summaries.len(), out_dir.join("summary.json").display());
    Ok(())
}

fn main() -> Result<()> {
    let args = RootArgs::parse();
    // 守卫放在建局、模型加载与侧车启动之前：开关被静默忽略比报错更糟
    check_switch_support(
        args.mode,
        args.policy_cache,
        args.adaptive_batch,
        args.sidecar_graph,
        args.handwritten_rollout
    )?;
    let leaf_depths = check_leaf_support(&args)?;
    let game_route = check_game_route(&args, &leaf_depths)?;
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    let mut game_config = load_game_config()?;
    // 与 `ramen_space_bench` 同一前提：Y3 地区必须交回策略，否则测的不是同一个分布
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    // 评分口径在 `init_global_with_config` **之前**钉死，初始化后再回报生效值
    let scoring = ScoringOverride {
        selection: RamenSelection::Score,
        pt_favor_rate: args.pt_favor_rate
    };
    let eff_rate = scoring.apply(&mut game_config)?;
    init_global_with_config(&game_config)?;
    report_effective(
        &scoring,
        eff_rate,
        &EffectiveSearchFacts {
            search_n: args.search_n,
            use_ucb: false,
            radical_factor_max: args.radical_factor_max,
            stages: format!("{:?}", args.mode),
            rollout_policy: format!("NN rollout: {}", args.rollout_model.display()),
            region_strategy: "all（本入口强制）".to_string()
        }
    );

    if let Some(w) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(w)
            .build_global()
            .map_err(|e| anyhow!("设置 rayon worker 数失败: {e}"))?;
    }

    let space = build_space(&args)?;
    let inherit = gen1_inherit();
    // 这两种模式自己建根（可能要用 NN 策略推进），故在默认建根**之前**分发
    if matches!(args.mode, Mode::RootScan | Mode::LeafPilot | Mode::ValueCheck) {
        let nn_root = Arc::new(
            RamenNnTrainer::load(&args.rollout_model)?
                .with_special_mode(SpecialSelectMode::Canonical)
                .with_race_shield(true)
        );
        let guide = match args.root_policy.as_str() {
            "handwritten" => RootGuide::Handwritten(RecommendedRamenTrainer::for_rollout()),
            "nn" => RootGuide::Nn(Arc::clone(&nn_root)),
            other => bail!("未知的 --root-policy {other}（只支持 handwritten / nn）")
        };
        return match args.mode {
            Mode::RootScan => run_root_scan(&args, &space, &inherit, &guide),
            Mode::ValueCheck => run_value_check(&args, &nn_root, &space, &inherit, &guide),
            _ => run_leaf_pilot(&args, &nn_root, &leaf_depths, &space, &inherit, &guide)
        };
    }
    let plans = space.plans();
    let plan = plans
        .get(args.plan_index)
        .ok_or_else(|| anyhow!("计划下标 {} 越界（共 {} 个）", args.plan_index, plans.len()))?;
    let (root, mut rng) = build_root(&args, plan.uma, &plan.deck, &inherit, args.root_turn, args.root_stage.as_deref())?;
    let actions = root.list_actions()?;
    ensure!(!actions.is_empty(), "根上没有合法候选");
    let root_id = format!(
        "plan{}_seed{}_run{}_t{}_{:?}",
        args.plan_index,
        args.seed,
        args.run_idx,
        root.turn(),
        root.stage
    );
    println!(
        "固定根 {root_id}：候选 {} 个 search_n={} 模式 {:?} workers={:?}",
        actions.len(),
        args.search_n,
        args.mode,
        args.workers
    );
    // 计划身份必须落进日志：`plan_index` 只是枚举下标，换空间就指向别的组合
    println!(
        "  计划身份    plan{} uma={} deck={:?} shape={}",
        args.plan_index,
        plan.uma,
        plan.deck,
        plan.shape
    );

    // 模型加载单独计时：统一口径的性能窗口从「模型就绪之后」才开始
    let load_start = Instant::now();
    let nn = Arc::new(
        RamenNnTrainer::load(&args.rollout_model)?
            .with_special_mode(SpecialSelectMode::Canonical)
            .with_race_shield(true)
    );
    let cpu_model_load_s = load_start.elapsed().as_secs_f64();
    let config = SearchConfig {
        search_n: args.search_n,
        radical_factor_max: args.radical_factor_max,
        // ❗必须显式关：`SearchConfig::default()` 的 `use_ucb` 是 **true**，
        // `search_group_size=256`。`n<=256` 时首组恰好覆盖全部样本，看不出问题；
        // 到 n=512，官方汇总只会消费部分候选的部分备忘表结果，口径不再是均匀搜索。
        // 正式教师口径本身也是 `use_ucb=false`。
        use_ucb: false,
        ..SearchConfig::default()
    };
    // 预跑与官方汇总必须用同一张种子表：clone 出来派生，原 rng 留给 search_with
    let seeds = RolloutSeeds::from_rng(&mut rng.clone());

    let sink = match args.mode {
        Mode::Compare => Some(Arc::new(CompareSink {
            pending: Mutex::new(Vec::new()),
            sidecar: Mutex::new(Sidecar::start(&args)?),
            nn: Arc::clone(&nn),
            batch: args.batch,
            rows: Mutex::new(Vec::new()),
            resolved: Mutex::new(0)
        })),
        _ => None
    };
    // CPU 路径的逐决策记录端；波次路径自己记，不走 sink
    let log_sink: Option<Arc<DecisionLogSink>> = match (args.decision_csv.as_ref(), args.mode) {
        (Some(_), Mode::CpuFlat | Mode::CpuCandidate) => Some(Arc::new(DecisionLogSink::default())),
        _ => None
    };
    let record_ctx = sink.is_some() || log_sink.is_some();
    let rollout_trainer = {
        let t = RamenRolloutTrainer::handwritten().with_neural_net(Arc::clone(&nn), None);
        match (sink.as_ref(), log_sink.as_ref()) {
            (Some(s), _) => t.with_infer_sink(Arc::clone(s) as Arc<dyn RolloutInferSink>),
            (None, Some(s)) => t.with_infer_sink(Arc::clone(s) as Arc<dyn RolloutInferSink>),
            (None, None) => t
        }
    };
    let search = FlatSearch::<RamenGame>::new(config.clone())
        .with_rollout_trainer(rollout_trainer)
        .with_strict_rollout(true);

    // 这两种模式自己管侧车与建局，不走固定根的三种粒度
    match args.mode {
        Mode::SidecarReuse => return run_sidecar_reuse(&args, &search, &nn, plan.uma, &plan.deck, &inherit),
        Mode::GameSmoke => {
            return run_game_smoke(&args, &search, &nn, plan.uma, &plan.deck, &inherit, game_route);
        }
        Mode::TeacherConsistency => {
            return run_teacher_consistency(&args, &config, &nn, plan.uma, &plan.deck, &inherit);
        }
        Mode::TeacherGame => {
            return run_teacher_game(&args, &config, &nn, plan.uma, &plan.deck, &inherit, cpu_model_load_s);
        }
        _ => {}
    }

    let granularity = match args.mode {
        Mode::CpuCandidate => Granularity::Candidate,
        _ => Granularity::Flat
    };
    // 启动与预热在性能窗口之外：侧车在此处握手完成（模型已加载并预热）
    let mut sidecar = match args.mode {
        Mode::GpuWave => Some(Sidecar::start(&args)?),
        _ => None
    };
    // 本根的物理批档位：调度器活跃轨迹容量与 GPU 物理张量行数都取它
    let wave_batch = plan_batch(args.adaptive_batch, args.batch, actions.len() * args.search_n);
    if let Some(sc) = sidecar.as_mut() {
        // 换档（含该档位首次预热）放在计时窗口**之前**
        sc.ensure_batch(wave_batch)?;
    }

    // 统一口径：模型就绪之后，从建轨迹计到官方汇总完成。两种模式同一个窗口。
    let measure_start = Instant::now();
    let (cells, rollout_rows, wave_decisions, wave) = match sidecar.as_mut() {
        Some(sc) => {
            // gpu-wave 模式不接截断：`LeafDepth::Full` 保持既有行为逐字不变
            let out = run_gpu_wave(
                &root,
                &actions,
                args.search_n,
                &seeds,
                &nn,
                sc,
                wave_batch,
                args.decision_csv.is_some(),
                args.policy_cache,
                LeafDepth::Full,
                false
            )?;
            let cells = out.raw_cells()?;
            (cells, out.rollout_rows, out.decisions, Some(out.stats))
        }
        None => {
            let (cells, rows) = collect_rollouts(
                &search,
                &root,
                &actions,
                args.search_n,
                &seeds,
                granularity,
                record_ctx
            )?;
            (cells, rows, Vec::new(), None)
        }
    };
    if let Some(s) = sink.as_ref() {
        s.drain()?;
    }
    let (best, means) = official_result(&search, &root, &actions, &cells, &mut rng)?;
    let measured_s = measure_start.elapsed().as_secs_f64();

    println!("  启动与预热  模型加载 {cpu_model_load_s:.1} s（性能窗口之外）");
    if let Some(sc) = sidecar.as_ref() {
        println!("              侧车就绪 {:.1} s | {}", sc.startup_s, sc.banner);
    }
    if let Some(st) = wave.as_ref() {
        println!(
            "  物理批档位  B{}（上限 {}，自适应 {}）；换档与档位预热 {:.3} s（单列，不在耗时窗口内）",
            st.batch,
            args.batch,
            if args.adaptive_batch { "开" } else { "关" },
            st.setup_s
        );
        println!(
            "  policy 复用 {} | {}",
            if args.policy_cache { "开" } else { "关" },
            st.cache.line()
        );
        println!("  波次        {}", st.waves);
        println!(
            "  服务请求    {}（格位 {}，利用率 {}）",
            st.served,
            st.slots,
            fill_text(st.served, st.slots)
        );
        println!("    波次循环                     {:.1} s（不含官方汇总）", st.total_wall_s);
        println!("    ├ CPU 推进与恢复（并行段墙钟） {:.1} s", st.cpu_wall_s);
        println!("    ├ 侧车往返（含推理本身）       {:.1} s", st.gpu_wall_s);
        println!(
            "    └ 调度残余                     {:.1} s",
            st.total_wall_s - st.cpu_wall_s - st.gpu_wall_s
        );
        println!("  ❗各轨迹自报 CPU 之和 {:.1} s（跨 worker 累加，不可加进墙钟）", st.cpu_in_traj_s);
    }
    if sink.is_some() {
        println!("  统一口径耗时 {measured_s:.1} s（❗compare 含快照与同步开销，不作性能数字）");
    } else {
        println!(
            "  统一口径耗时 {measured_s:.1} s（模型就绪后：建轨迹 → 官方汇总完成；粒度 {}）",
            match args.mode {
                Mode::GpuWave => "波次驱动".to_string(),
                _ => format!("{granularity:?}")
            }
        );
    }
    println!("  最优候选    {best}（生产汇总：rank 加权均值 + radical factor）");
    for (i, m) in means.iter().enumerate() {
        println!("    候选 {i}: {m:.9}");
    }

    if let Some(path) = args.raw_csv.as_ref() {
        let mut f = std::fs::File::create(path).with_context(|| format!("创建 raw_csv 失败: {}", path.display()))?;
        writeln!(f, "candidate,j,seed,score,score_pt")?;
        for c in &cells {
            writeln!(f, "{},{},{:#018x},{:.17e},{:.17e}", c.candidate, c.j, c.seed, c.score, c.score_pt)?;
        }
        println!("  原始结果    {}", path.display());
    }
    if let Some(path) = args.decision_csv.as_ref() {
        let mut rows = match log_sink.as_ref() {
            Some(s) => std::mem::take(&mut *s.rows.lock().map_err(|_| anyhow!("决策记录锁被毒化"))?),
            None => wave_decisions
        };
        ensure!(!rows.is_empty(), "开启了 --decision-csv 却没有记录到任何网络决策");
        // 并行完成顺序不得影响写回顺序
        rows.sort_by_key(|r| (r.candidate, r.j, r.seq));
        let f = std::fs::File::create(path).with_context(|| format!("创建 decision_csv 失败: {}", path.display()))?;
        let mut f = std::io::BufWriter::new(f);
        writeln!(f, "candidate,j,seq,turn,stage,n_actions,actions,chosen,features")?;
        for r in &rows {
            let acts = r.actions.join("|");
            let feats = r
                .features
                .iter()
                .map(|v| format!("{v:.9e}"))
                .collect::<Vec<_>>()
                .join("|");
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{}",
                r.candidate,
                r.j,
                r.seq,
                r.turn,
                r.stage,
                r.actions.len(),
                acts,
                r.chosen,
                feats
            )?;
        }
        println!("  逐决策      {} 条 → {}", rows.len(), path.display());
    }
    if let Some(path) = args.rollout_csv.as_ref() {
        let mut f = std::fs::File::create(path).with_context(|| format!("创建 rollout_csv 失败: {}", path.display()))?;
        // `requests` 是实际发给侧车的请求数，`decisions` 是逻辑网络决策数：
        // 不开缓存时两列相等，开缓存后差值就是复用掉的请求
        writeln!(f, "candidate,j,requests,decisions")?;
        for r in &rollout_rows {
            writeln!(f, "{},{},{},{}", r.candidate, r.j, r.requests, r.decisions)?;
        }
        println!("  逐 rollout  {}", path.display());
    }
    if let Some(s) = sink.as_ref() {
        report_compare(s, &root_id, &args)?;
    }
    Ok(())
}

/// 打印对拍汇总并可选落盘
///
/// # 错误
///
/// 锁被毒化或写文件失败时报错。
fn report_compare(sink: &CompareSink, root_id: &str, args: &RootArgs) -> Result<()> {
    let rows = sink.rows.lock().map_err(|_| anyhow!("结果锁被毒化"))?;
    let resolved = *sink.resolved.lock().map_err(|_| anyhow!("计数锁被毒化"))?;
    let mismatched: Vec<&CompareRow> = rows.iter().filter(|r| r.cpu_winner != r.gpu_winner).collect();
    let max_delta = rows.iter().map(|r| r.action_delta).fold(0.0f32, f32::max);
    // 赢家由 top1−top2 决定，两端各自扰动最坏叠加约 2δ；**逐样本**比，不用全局最大值
    let risky = rows
        .iter()
        .filter(|r| r.cpu_margin.is_finite() && r.cpu_margin < 2.0 * r.action_delta)
        .count();
    println!("  推理决策    {} 个", rows.len());
    println!("  未经网络    {resolved} 个（守门 / 转交手写，不计入对拍覆盖）");
    println!("  赢家不一致  {} 个", mismatched.len());
    println!("  候选分数最大绝对差 {max_delta:.3e}（本批观测值，不是未来输入的上界）");
    println!("  分差 < 2×本样本扰动 的决策数 {risky}");
    if let Some(first) = mismatched.first() {
        println!(
            "  ❗首个分歧：候选 {} j={} 第 {} 次推理，回合 {} 阶段 {}，CPU 选 {} / GPU 选 {}，分差 {:.4e} 扰动 {:.4e}",
            first.ctx.candidate,
            first.ctx.j,
            first.ctx.seq,
            first.turn,
            first.stage,
            first.cpu_winner,
            first.gpu_winner,
            first.cpu_margin,
            first.action_delta
        );
    }
    if let Some(path) = args.trace.as_ref() {
        let mut f = std::fs::File::create(path).with_context(|| format!("创建 trace 失败: {}", path.display()))?;
        writeln!(
            f,
            "root,candidate,j,seq,turn,stage,n_actions,rng_probe,cpu_winner,gpu_winner,cpu_margin,action_delta,gap_us"
        )?;
        for r in rows.iter() {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:#018x},{},{},{:.6e},{:.6e},{:.3}",
                root_id,
                r.ctx.candidate,
                r.ctx.j,
                r.ctx.seq,
                r.turn,
                r.stage,
                r.n_actions,
                r.rng_probe,
                r.cpu_winner,
                r.gpu_winner,
                r.cpu_margin,
                r.action_delta,
                r.gap_us
            )?;
        }
        println!("  trace       {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env::set_current_dir, path::Path};

    use umasim::{
        search::{ActionResult, SearchOutput},
        trainer::RamenValueNorm
    };

    use super::*;

    /// 造一份长度为 [`INPUT_DIM`] 的输入
    fn feats(fill: f32) -> Vec<f32> {
        vec![fill; INPUT_DIM]
    }

    /// 造一份长度为 [`POLICY_DIM`] 的 policy
    fn policy(fill: f32) -> Vec<f32> {
        vec![fill; POLICY_DIM]
    }

    /// 本模块内的轻量检查收集器
    ///
    /// `umasim::utils::Checks` 只在 lib 自己的 test 配置下可见，bin 的集成测试取不到；
    /// 行为对齐本文件既有用例的写法：逐项 `println`，最后汇总，不用 `assert` 宏。
    #[derive(Default)]
    struct Chk {
        /// 不符项数
        bad: usize,
        /// 总项数
        total: usize
    }

    impl Chk {
        /// 记一项检查
        fn check(&mut self, ok: bool, what: &str) {
            self.total += 1;
            println!("{} {what}", if ok { "ok" } else { "❗" });
            if !ok {
                self.bad += 1;
            }
        }

        /// 汇总
        ///
        /// # 错误
        ///
        /// 存在不符项时报错。
        fn finish(self) -> Result<()> {
            println!("共 {} 项，不符 {} 项", self.total, self.bad);
            ensure!(self.bad == 0, "有 {} 项检查不通过", self.bad);
            Ok(())
        }
    }

    /// 造一批身份完整的截断实验结果
    ///
    /// 种子只与 `j` 有关（与 `RolloutSeeds::seed_at` 同样不吃候选下标），
    /// 这样默认造出来的就是 CRN 对齐的一批。
    fn grid_cells(values: &[Vec<f64>]) -> Vec<ArmCell> {
        let mut out = Vec::new();
        for (cand, row) in values.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                out.push(ArmCell {
                    candidate: cand,
                    j,
                    seed: 0x1000 + j as u64,
                    kind: "terminal",
                    value: *v,
                    score_pt: Some(*v + 1.0),
                    paired_terminal: None,
                    leaf_turn: None,
                    leaf_stage: None,
                    turn_delta: None,
                    deferrals: None
                });
            }
        }
        out
    }

    /// 均值选择器与生产 `SearchOutput`（rf=0）在**并列**上的取舍必须一致
    ///
    /// 生产用 `max_by(partial_cmp)`，Rust 的 `max_by` 在相等时保留**后一个**。
    /// 写成 `>` 比较会变成前一个胜出，这条用例就是那根绊线。
    ///
    /// # 错误
    ///
    /// 任一用例的选中下标或均值与生产路径不符时报错。
    #[test]
    fn mean_select_matches_production_rf_zero() -> Result<()> {
        // (用例名, 各候选的分数列)
        let cases: Vec<(&str, Vec<Vec<f64>>)> = vec![
            ("普通", vec![vec![10.0, 20.0, 30.0], vec![11.0, 21.0, 31.0], vec![9.0, 19.0, 29.0]]),
            ("三方完全并列", vec![vec![10.0, 20.0], vec![10.0, 20.0], vec![10.0, 20.0]]),
            ("末位与首位并列", vec![vec![50.0, 50.0], vec![10.0, 10.0], vec![50.0, 50.0]]),
            ("单候选", vec![vec![7.0, 8.0, 9.0, 10.0]]),
            ("负分与大跨度", vec![vec![-100.0, 100.0], vec![0.5, -0.5], vec![99999.0, -99999.0]])
        ];
        let mut bad = 0usize;
        for (name, values) in &cases {
            let n = values[0].len();
            // 生产路径：逐条 `add` 进 `ActionResult`，再交给 `SearchOutput` 以 rf=0 排序
            let results: Vec<(ActionResult, ActionResult)> = values
                .iter()
                .map(|row| {
                    let mut a = ActionResult::new();
                    let mut b = ActionResult::new();
                    for v in row {
                        a.add(*v);
                        b.add(*v);
                    }
                    (a, b)
                })
                .collect();
            let prod = SearchOutput::<usize, ()>::new((0..values.len()).collect(), results, 0.0);
            let prod_means: Vec<f64> = prod.action_results.iter().map(|(a, _)| a.mean()).collect();
            let cells = grid_cells(values);
            let (best, means) = mean_select(&cells, values.len(), 0, n)?;
            let same_best = best == prod.best_action_idx;
            let same_bits = prod_means
                .iter()
                .zip(means.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            let mark = if same_best && same_bits { "ok" } else { "❗" };
            println!(
                "{mark} [{name}] 生产选 {} / 本工具选 {}；均值逐位一致={same_bits}",
                prod.best_action_idx, best
            );
            if !(same_best && same_bits) {
                println!("    生产均值 {prod_means:?}");
                println!("    本工具   {means:?}");
                bad += 1;
            }
        }
        println!("并列与均值对拍：{} 条用例，不符 {bad} 条", cases.len());
        ensure!(bad == 0, "均值选择器与生产 rf=0 有 {bad} 条用例不一致");
        Ok(())
    }

    /// 身份完整性守卫的故障注入
    ///
    /// 只数条数挡不住「重复一个 `j` + 缺另一个 `j`」：两种错误条数相同，
    /// 必须逐 `(候选, j)` 核对。种子错位同理——条数全对，CRN 却已经不配对了。
    ///
    /// # 错误
    ///
    /// 任一注入的故障没有被拦下（或正常输入被误杀）时报错。
    #[test]
    fn validate_cells_rejects_injected_faults() -> Result<()> {
        let good = || grid_cells(&[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
        let mut cases: Vec<(&str, Vec<ArmCell>, bool)> = Vec::new();
        cases.push(("完好输入", good(), true));

        // 重复 j=1、缺 j=2：**条数仍然是 4**
        let mut dup = good();
        dup[2].j = 1;
        dup[2].seed = 0x1000 + 1;
        cases.push(("候选 0 重复 j=1 且缺 j=2（条数不变）", dup, false));

        // 少一条
        let mut missing = good();
        missing.pop();
        cases.push(("候选 1 缺最后一列", missing, false));

        // 种子错位：条数与列都对，只有一格的种子不同
        let mut seed_bad = good();
        seed_bad[5].seed ^= 1;
        cases.push(("候选 1 的 j=1 种子与候选 0 不同（CRN 错位）", seed_bad, false));

        // 非有限值
        let mut nan = good();
        nan[3].value = f64::NAN;
        cases.push(("含 NaN", nan, false));
        let mut inf = good();
        inf[0].value = f64::INFINITY;
        cases.push(("含 Inf", inf, false));

        // 候选下标越界
        let mut oob = good();
        oob[7].candidate = 9;
        cases.push(("候选下标越界", oob, false));

        let mut bad = 0usize;
        for (name, cells, want_ok) in &cases {
            let got = validate_cells(cells, 2, 0, 4);
            let ok = got.is_ok();
            let mark = if ok == *want_ok { "ok" } else { "❗" };
            match &got {
                Ok(_) => println!("{mark} [{name}] 通过（期望通过={want_ok}）"),
                Err(e) => println!("{mark} [{name}] 拦下：{e}（期望通过={want_ok}）")
            }
            if ok != *want_ok {
                bad += 1;
            }
        }
        println!("故障注入：{} 条用例，行为不符 {bad} 条", cases.len());
        ensure!(bad == 0, "身份完整性守卫有 {bad} 条用例行为不符");
        Ok(())
    }

    /// 评估列里混进叶结果必须被 `audit_means` 拦下
    ///
    /// # 错误
    ///
    /// 混入叶结果却通过，或纯终局输入被误杀时报错。
    #[test]
    fn audit_means_rejects_leaf_in_audit_bank() -> Result<()> {
        let mut c = Chk::default();
        let clean = grid_cells(&[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
        c.check(audit_means(&clean, 2, 2, 4).is_ok(), "纯终局的评估列通过");
        let mut dirty = clean.clone();
        dirty[3].kind = "leaf";
        dirty[3].score_pt = None;
        match audit_means(&dirty, 2, 2, 4) {
            Ok(_) => c.check(false, "评估列里的叶结果应当被拦下"),
            Err(e) => {
                println!("拦下：{e}");
                c.check(true, "评估列里的叶结果被拦下");
            }
        }
        // 叶落在**选择列**里则不该被 audit 检查影响
        let mut sel_leaf = clean.clone();
        sel_leaf[0].kind = "leaf";
        c.check(audit_means(&sel_leaf, 2, 2, 4).is_ok(), "叶落在选择列不影响评估列检查");
        c.finish()
    }

    /// value 反归一化：只解码一次，不重复加中心，不因成员尺度不同而错位
    ///
    /// 第二段复刻侧车的集成规则（各成员按**自身**常数反归一化 → 在实际分数空间平均
    /// → 再按成员 0 的常数重归一化），刻意让三个成员的 `center`/`scale` **互不相同**，
    /// 验证 Rust 侧按成员 0 解码一次恰好还原出「各成员实际分数的平均」。
    ///
    /// # 错误
    ///
    /// 任一环节与手算值不符时报错。
    #[test]
    fn value_decode_is_single_pass() -> Result<()> {
        let mut c = Chk::default();
        let norm = RamenValueNorm {
            center: [60000.0, 2000.0, 61000.0],
            scale: [4000.0, 800.0, 4500.0]
        };
        // 单路：center + scale * x，且 stdev 那一路截到非负
        let v = norm.denormalize(&[0.5f32, -3.0, 0.25])?;
        c.check(
            (v.mean - 62000.0).abs() < 1e-9,
            &format!("mean = 60000 + 4000*0.5 = 62000，实得 {}", v.mean)
        );
        c.check(v.stdev == 0.0, &format!("stdev 负值截到 0，实得 {}", v.stdev));
        c.check(
            (v.high - 62125.0).abs() < 1e-9,
            &format!("high = 61000 + 4500*0.25 = 62125，实得 {}", v.high)
        );
        // 没有「把当前分数再加一次」这种事：零输入必须恰好落在中心上
        let zero = norm.denormalize(&[0.0f32, 0.0, 0.0])?;
        c.check(
            zero.mean == norm.center[0],
            &format!("零归一化输入解码成中心 {}，实得 {}", norm.center[0], zero.mean)
        );

        // 三个成员尺度互不相同时，复刻侧车集成规则
        let centers = [60000.0f64, 58000.0, 62000.0];
        let scales = [4000.0f64, 3000.0, 5000.0];
        let raw = [0.5f64, -0.25, 0.10]; // 各成员自己的归一化输出
        let real: Vec<f64> = raw
            .iter()
            .zip(centers.iter().zip(scales.iter()))
            .map(|(x, (ct, sc))| ct + sc * x)
            .collect();
        let real_mean = real.iter().sum::<f64>() / real.len() as f64;
        // 侧车返回的是「按成员 0 重归一化」后的值
        let renorm = (real_mean - centers[0]) / scales[0];
        let member0 = RamenValueNorm {
            center: [centers[0], 2000.0, 61000.0],
            scale: [scales[0], 800.0, 4500.0]
        };
        let decoded = member0.denormalize(&[renorm as f32, 0.0, 0.0])?;
        println!(
            "各成员实际分数 {real:?} → 平均 {real_mean:.6} → 重归一化 {renorm:.6} → 解码 {:.6}",
            decoded.mean
        );
        c.check(
            (decoded.mean - real_mean).abs() < 1e-2,
            &format!("按成员 0 解码一次应还原成员平均 {real_mean:.6}，实得 {:.6}", decoded.mean)
        );
        // 直接平均**归一化**值是错的口径：本用例顺带记录两者确实不同
        let naive = raw.iter().sum::<f64>() / raw.len() as f64;
        let naive_score = centers[0] + scales[0] * naive;
        println!("❗直接平均归一化值会得到 {naive_score:.1}，与正确口径差 {:.1} 分", naive_score - real_mean);
        c.check(
            (naive_score - real_mean).abs() > 1.0,
            "用例的三组常数确实能区分两种口径（否则这条测试测不出东西）"
        );
        c.finish()
    }

    /// 叶响应的线级故障注入：错行长 / NaN / Inf 必须**失败**，且失败不消耗动作
    ///
    /// 这条测的是 `Traj::resume` 的 `ReqKind::LeafValue` 分支：坏行必须立刻报错，
    /// 不得填 0、不得丢列、不得转手写继续。另外核对「叶响应不执行一次动作」——
    /// 正常响应之后 `seq`（逻辑决策数）不增加、局面停在原回合原阶段。
    ///
    /// # 错误
    ///
    /// 任一坏行被放过、正常行被误杀，或叶响应执行了一次动作时报错。
    #[test]
    fn leaf_resume_rejects_bad_rows() -> Result<()> {
        set_current_dir(get_workspace_root()?)?;
        let model = "saved_models/arms/ens_NT4096_AllHistory_g123.onnx";
        ensure!(
            Path::new(model).exists(),
            "本轮冻结模型不存在：{model}（本测试依赖它，不另造假模型）"
        );
        let mut cfg = load_game_config()?;
        cfg.ramen_region_strategy = RamenRegionStrategy::All;
        init_global_with_config(&cfg)?;
        let args = RootArgs::try_parse_from([
            "ramen_root_bench",
            "--mode",
            "gpu-wave",
            "--rollout-model",
            model,
            "--search-n",
            "1",
            "--root-turn",
            "6",
            "--root-stage",
            "RamenSelect"
        ])?;
        let space = space_from_cli(None, &[])?;
        let plans = space.plans();
        let plan = plans.first().ok_or_else(|| anyhow!("采样空间没有计划"))?;
        let inherit = gen1_inherit();
        let (root, rng) =
            build_root(&args, plan.uma, &plan.deck, &inherit, args.root_turn, args.root_stage.as_deref())?;
        let actions = root.list_actions()?;
        ensure!(!actions.is_empty(), "根上没有合法候选");
        let nn = RamenNnTrainer::load(&args.rollout_model)?
            .with_special_mode(SpecialSelectMode::Canonical)
            .with_race_shield(true);
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let mut c = Chk::default();

        // 推到叶边界：H=2 ⇒ 目标回合 root_turn + 2
        let mk = || -> Result<Traj> {
            let mut t = Traj::start(0, 0, &root, &actions[0], seeds.seed_at(0), false, LeafDepth::Turns(2), false)?;
            loop {
                t.advance(&nn, false)?;
                if t.phase == Phase::Done {
                    bail!("H=2 的轨迹还没到叶边界就终局了，换一个根");
                }
                if t.req_kind == ReqKind::LeafValue {
                    return Ok(t);
                }
                // 还是 policy 请求：喂一次真实推理结果继续推
                let req = t.request.clone().ok_or_else(|| anyhow!("挂起却没有待推理输入"))?;
                let out = nn.infer_features(req)?;
                let mut row = out.policy;
                row.resize(OUTPUT_DIM, 0.0);
                t.resume(&nn, &row, false)?;
            }
        };

        let probe = mk()?;
        println!(
            "叶边界：t{} {:?}（根 t{}），待推理行长 {}",
            probe.game.turn(),
            probe.game.stage,
            root.turn(),
            probe.request.as_ref().map(Vec::len).unwrap_or(0)
        );
        c.check(
            probe.game.turn() >= root.turn() + 2,
            &format!("叶回合 {} 不早于根回合+2 = {}", probe.game.turn(), root.turn() + 2)
        );
        c.check(leaf_safe_stage(&probe.game.stage), "叶落在安全阶段");

        // 造一份合法的 245 行（value 段给一个正常的归一化值）
        let good_row = {
            let req = probe
                .request
                .clone()
                .ok_or_else(|| anyhow!("叶边界没有待推理输入"))?;
            let out = nn.infer_features(req)?;
            let mut row = out.policy;
            row.resize(OUTPUT_DIM, 0.0);
            row[VALUE_OFF] = 0.25;
            row
        };

        // (用例名, 造行, 期望成功)
        let cases: Vec<(&str, Box<dyn Fn(&Vec<f32>) -> Vec<f32>>, bool)> = vec![
            ("合法行", Box::new(|r: &Vec<f32>| r.clone()), true),
            ("行长少一个", Box::new(|r: &Vec<f32>| r[..OUTPUT_DIM - 1].to_vec()), false),
            ("行长多一个", Box::new(|r: &Vec<f32>| {
                let mut v = r.clone();
                v.push(0.0);
                v
            }), false),
            ("只有 policy 段（234）", Box::new(|r: &Vec<f32>| r[..POLICY_DIM].to_vec()), false),
            ("value[0] = NaN", Box::new(|r: &Vec<f32>| {
                let mut v = r.clone();
                v[VALUE_OFF] = f32::NAN;
                v
            }), false),
            ("value[0] = +Inf", Box::new(|r: &Vec<f32>| {
                let mut v = r.clone();
                v[VALUE_OFF] = f32::INFINITY;
                v
            }), false),
            ("value[0] = -Inf", Box::new(|r: &Vec<f32>| {
                let mut v = r.clone();
                v[VALUE_OFF] = f32::NEG_INFINITY;
                v
            }), false)
        ];
        for (name, build, want_ok) in &cases {
            let mut t = mk()?;
            let seq_before = t.seq;
            let turn_before = t.game.turn();
            let stage_before = format!("{:?}", t.game.stage);
            let row = build(&good_row);
            let got = t.resume(&nn, &row, false);
            match (&got, want_ok) {
                (Ok(()), true) => {
                    let no_action = t.seq == seq_before;
                    let same_place = t.game.turn() == turn_before && format!("{:?}", t.game.stage) == stage_before;
                    let done = t.phase == Phase::Done;
                    let value = match &t.outcome {
                        Some(TrajOutcome::Leaf(n)) => n.value,
                        _ => f64::NAN
                    };
                    println!(
                        "ok [{name}] 通过；叶估值 {value:.1}；决策数 {seq_before}→{} 局面 t{} {:?}",
                        t.seq,
                        t.game.turn(),
                        t.game.stage
                    );
                    c.check(no_action, "叶响应没有执行动作（逻辑决策数不变）");
                    c.check(same_place, "叶响应没有推进局面（回合与阶段不变）");
                    c.check(done, "叶响应后轨迹就地收尾");
                    c.check(
                        (value - (61439.01158101555 + 4240.3727274524235 * 0.25)).abs() < 1e-6,
                        &format!("叶估值等于 center[0] + scale[0] * 0.25，实得 {value}")
                    );
                }
                (Err(e), false) => {
                    println!("ok [{name}] 拦下：{e}");
                    c.check(t.seq == seq_before, "失败路径没有执行动作");
                }
                (Ok(()), false) => c.check(false, &format!("[{name}] 本应失败却通过了")),
                (Err(e), true) => c.check(false, &format!("[{name}] 本应通过却报错：{e}"))
            }
        }
        c.finish()
    }

    /// 路由判定走生产年份语义，且只认第三年地区
    ///
    /// 覆盖预登记 §3 的第 1 条与第 4 条的判定侧：扫一整局的全部决策点，
    /// 逐点核对 [`LeafRoute::HybridY3Full`] 与 [`LeafRoute::Uniform`] 的判定，
    /// 并核对 `RegionSelect` 在本剧本恰好只出现在 turn 2/23/47。
    /// **不用「候选数恰好 120」这种替代条件**，年份一律走
    /// [`RamenState::region_archive_year_idx`]。
    ///
    /// # 错误
    ///
    /// 任一决策点的路由与规则不符、地区回合集合不是 {2,23,47}，
    /// 或非地区回合的年份归档没有报错时报错。
    #[test]
    fn hybrid_route_decides_by_production_year_semantics() -> Result<()> {
        set_current_dir(get_workspace_root()?)?;
        let mut cfg = load_game_config()?;
        cfg.ramen_region_strategy = RamenRegionStrategy::All;
        init_global_with_config(&cfg)?;
        // 本测试只走手写策略建根与扫描，模型路径只为满足 CLI 必填项，不会被加载
        let model = "saved_models/arms/ens_NT4096_AllHistory_g123.onnx";
        let args = RootArgs::try_parse_from([
            "ramen_root_bench",
            "--mode",
            "root-scan",
            "--search-n",
            "1",
            "--rollout-model",
            model
        ])?;
        let space = space_from_cli(None, &[])?;
        let plans = space.plans();
        let plan = plans.first().ok_or_else(|| anyhow!("采样空间没有计划"))?;
        let inherit = gen1_inherit();
        let guide = RootGuide::Handwritten(RecommendedRamenTrainer::for_rollout());
        let pts = scan_decisions(&args, plan.uma, &plan.deck, &inherit, &guide, 279100)?;
        let mut c = Chk::default();
        println!("决策点 {} 个", pts.len());

        // 年份归档只认 2/23/47：其余回合必须报错，不得猜
        for t in [0, 1, 3, 22, 24, 46, 48, 71, 77] {
            c.check(
                RamenState::region_archive_year_idx(t).is_err(),
                &format!("非地区回合 turn={t} 的年份归档必须报错")
            );
        }

        let hybrid = LeafRoute::HybridY3Full(8);
        let uni_full = LeafRoute::Uniform(LeafDepth::Full);
        let uni_h8 = LeafRoute::Uniform(LeafDepth::Turns(8));
        let mut region_turns: Vec<i32> = Vec::new();
        let mut y3 = 0usize;
        let mut other = 0usize;
        // 逐点重建局面代价太高，这里直接用扫描出来的 (turn, stage) 判定：
        // `decide` 只读这两个量，等价且不引入新的推进路径。
        for p in &pts {
            let stage = p.stage.as_str();
            let want_y3 = stage == "RegionSelect" && p.turn == 47;
            if stage == "RegionSelect" {
                region_turns.push(p.turn);
            }
            // 年份判定
            if stage == "RegionSelect" {
                let idx = RamenState::region_archive_year_idx(p.turn)?;
                let expect = match p.turn {
                    2 => 0,
                    23 => 1,
                    47 => 2,
                    _ => usize::MAX
                };
                c.check(
                    idx == expect,
                    &format!("turn {} 的地区年份归档应为 {expect}，实得 {idx}", p.turn)
                );
            }
            if want_y3 {
                y3 += 1;
            } else {
                other += 1;
            }
        }
        region_turns.sort_unstable();
        println!("地区决策点回合 {region_turns:?}；第三年 {y3} 个、其余 {other} 个");
        c.check(region_turns == vec![2, 23, 47], "地区选择恰好出现在 turn 2/23/47");
        c.check(y3 == 1, &format!("第三年地区根恰好 1 个，实得 {y3}"));

        // 用真实局面再核一遍三个地区根与两个普通根的判定
        for (turn, stage, want_full, want_year) in [
            (2, "RegionSelect", false, Some(0usize)),
            (23, "RegionSelect", false, Some(1)),
            (47, "RegionSelect", true, Some(2)),
            (12, "Train", false, None),
            (71, "SuperRamenSelect", false, None)
        ] {
            let a = RootArgs::try_parse_from([
                "ramen_root_bench",
                "--mode",
                "root-scan",
                "--search-n",
                "1",
                "--rollout-model",
                model,
                "--run-idx",
                "279100",
                "--root-turn",
                &turn.to_string(),
                "--root-stage",
                stage
            ])?;
            let (root, _) = build_root(&a, plan.uma, &plan.deck, &inherit, turn, Some(stage))?;
            let d = hybrid.decide(&root)?;
            let uf = uni_full.decide(&root)?;
            let uh = uni_h8.decide(&root)?;
            println!(
                "  t{turn} {stage}: 混合→{} ({})，年份 {:?}；uniform-full→{}，uniform-h8→{}",
                d.depth.tag(),
                d.tag,
                d.region_year_idx,
                uf.depth.tag(),
                uh.depth.tag()
            );
            c.check(
                matches!(d.depth, LeafDepth::Full) == want_full,
                &format!("t{turn} {stage} 的混合路由深度应当是 {}", if want_full { "full" } else { "h8" })
            );
            c.check(d.region_year_idx == want_year, &format!("t{turn} {stage} 的年份归档应为 {want_year:?}"));
            c.check(d.tag == if want_full { "y3_region_full" } else { "other_trunc" }, "路由标签与深度一致");
            // Uniform 两支必须逐字保持既有行为
            c.check(matches!(uf.depth, LeafDepth::Full) && uf.tag == "uniform", "Uniform(full) 不受路由影响");
            c.check(
                matches!(uh.depth, LeafDepth::Turns(8)) && uh.tag == "uniform",
                "Uniform(h8) 不受路由影响"
            );
        }
        c.finish()
    }

    /// 混合路由在每类根上与对应的单一深度臂逐项一致（真跑侧车）
    ///
    /// 覆盖预登记 §3 的第 2/3/4/5/6 条：
    /// - Y3 地区根：混合 == `Uniform(full)`（推荐动作、实际 rf、请求与结局条数）
    /// - 普通根：混合 == `Uniform(h8)`（同上）
    /// - H 超出剩余回合的根：无叶请求，仍按 mean 选优
    /// - 同一个混合 trainer 连跑「普通根 → Y3 根」：rf 没有被上一根的 0 污染
    /// - 叶结果没有混进真实终局表（`raw_cells` 在 full 支路上必须成功）
    ///
    /// ❗小 `n` 只做正确性，**不是**性能结果。
    ///
    /// # 错误
    ///
    /// 模型或 checkpoint 缺失、侧车起不来，或任一项不一致时报错。
    #[test]
    fn hybrid_matches_uniform_arms_per_root() -> Result<()> {
        set_current_dir(get_workspace_root()?)?;
        let model = "saved_models/arms/ens_NT4096_AllHistory_g123.onnx";
        let sidecar = "scripts/ramen_nn/bench_sidecar.py";
        let cks = [
            "target/arm_NT4096_AllHistory_seed1/step_030000.pt",
            "target/arm_NT4096_AllHistory_seed2/step_030000.pt",
            "target/arm_NT4096_AllHistory_seed3/step_030000.pt"
        ];
        for p in [model, sidecar].into_iter().chain(cks) {
            ensure!(Path::new(p).exists(), "本轮冻结输入不存在：{p}（本测试不另造假输入）");
        }
        let mut cfg = load_game_config()?;
        cfg.ramen_region_strategy = RamenRegionStrategy::All;
        init_global_with_config(&cfg)?;
        let mut argv: Vec<String> = [
            "ramen_root_bench",
            "--mode",
            "game-smoke",
            "--search-n",
            "4",
            "--run-idx",
            "279100",
            "--rollout-model",
            model,
            "--sidecar",
            sidecar,
            "--policy-cache",
            "--sidecar-graph"
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        for ck in cks {
            argv.push("--checkpoint".to_string());
            argv.push(ck.to_string());
        }
        let args = RootArgs::try_parse_from(&argv)?;
        let space = space_from_cli(None, &[])?;
        let plans = space.plans();
        let plan = plans.first().ok_or_else(|| anyhow!("采样空间没有计划"))?;
        let inherit = gen1_inherit();
        let nn = Arc::new(
            RamenNnTrainer::load(&args.rollout_model)?
                .with_special_mode(SpecialSelectMode::Canonical)
                .with_race_shield(true)
        );
        let config = SearchConfig {
            search_n: args.search_n,
            radical_factor_max: args.radical_factor_max,
            use_ucb: false,
            ..SearchConfig::default()
        };
        let search = FlatSearch::<RamenGame>::new(config.clone())
            .with_rollout_trainer(RamenRolloutTrainer::handwritten().with_neural_net(Arc::clone(&nn), None))
            .with_strict_rollout(true);
        let mut c = Chk::default();

        // 三个根：Y3 地区、普通 Train、剩余回合不足 8 的晚期 Train
        let mut roots = Vec::new();
        for (turn, stage) in [(47, "RegionSelect"), (12, "Train"), (74, "Train")] {
            let (g, r) = build_root(&args, plan.uma, &plan.deck, &inherit, turn, Some(stage))?;
            let acts = g.list_actions()?;
            ensure!(!acts.is_empty(), "t{turn} {stage} 根上没有候选");
            println!("根 t{turn} {stage}：{} 个候选", acts.len());
            roots.push((format!("t{turn}_{stage}"), g, acts, r));
        }

        let mut sc = Sidecar::start(&args)?;
        println!("侧车就绪 {:.1} s | {}", sc.startup_s, sc.banner);

        // 跑一组「路由 × 根序列」，返回 (选中下标, 逐根路由记录)
        let mut run = |route: LeafRoute, idxs: &[usize], sc_in: Sidecar| -> Result<(Vec<usize>, Vec<RouteRow>, Sidecar)> {
            let trainer = WaveTrainer {
                nn: nn.as_ref(),
                search: &search,
                sidecar: Mutex::new(sc_in),
                max_batch: args.batch,
                adaptive: args.adaptive_batch,
                cache_on: args.policy_cache,
                n: args.search_n,
                fallback: RecommendedRamenTrainer::for_rollout(),
                route,
                stats: Mutex::new(GameStats::default())
            };
            let mut picks = Vec::new();
            for &i in idxs {
                let (_, g, acts, r) = &roots[i];
                picks.push(trainer.select_action(g, acts, &mut r.clone())?);
            }
            let rows = trainer
                .stats
                .into_inner()
                .map_err(|_| anyhow!("统计锁被毒化"))?
                .route_rows;
            let out = trainer.sidecar.into_inner().map_err(|_| anyhow!("侧车锁被毒化"))?;
            Ok((picks, rows, out))
        };

        // 三个臂各跑同一组三个根，同一份根 RNG（clone），CRN 因此相同
        let all = [0usize, 1, 2];
        let (p_full, r_full, s1) = run(LeafRoute::Uniform(LeafDepth::Full), &all, sc)?;
        let (p_h8, r_h8, s2) = run(LeafRoute::Uniform(LeafDepth::Turns(8)), &all, s1)?;
        let (p_hy, r_hy, s3) = run(LeafRoute::HybridY3Full(8), &all, s2)?;
        sc = s3;

        let names = ["Y3地区 t47", "普通 t12 Train", "晚期 t74 Train"];
        for (i, name) in names.iter().enumerate() {
            let (f, h, y) = (&r_full[i], &r_h8[i], &r_hy[i]);
            println!(
                "\n[{name}] 候选 {} | full 选 {} rf {:.6} | h8 选 {} rf {:.6} | 混合 选 {} rf {:.6} 路由 {} 深度 {}",
                f.candidates, p_full[i], f.rf_actual, p_h8[i], h.rf_actual, p_hy[i], y.rf_actual, y.route, y.depth
            );
            println!(
                "         请求 policy/leaf  full {}/{}  h8 {}/{}  混合 {}/{}；结局 terminal/leaf  full {}/{}  h8 {}/{}  混合 {}/{}",
                f.policy_requests,
                f.leaf_requests,
                h.policy_requests,
                h.leaf_requests,
                y.policy_requests,
                y.leaf_requests,
                f.terminal_results,
                f.leaf_results,
                h.terminal_results,
                h.leaf_results,
                y.terminal_results,
                y.leaf_results
            );
        }

        // ---- 第 2 条：Y3 根上混合 == Uniform(full)
        let (f, y) = (&r_full[0], &r_hy[0]);
        c.check(y.route == "y3_region_full", "Y3 地区根走 y3_region_full");
        c.check(y.depth == "full" && y.objective == "rf_weighted", "Y3 根实际深度 full、目标 rf 加权");
        c.check(p_hy[0] == p_full[0], &format!("Y3 根推荐动作一致：混合 {} vs full {}", p_hy[0], p_full[0]));
        c.check(
            y.rf_actual.to_bits() == f.rf_actual.to_bits(),
            &format!("Y3 根实际 rf 逐位一致：{} vs {}", y.rf_actual, f.rf_actual)
        );
        c.check(y.rf_actual > 0.0, &format!("Y3 根的 rf 不是 0（实得 {}）", y.rf_actual));
        c.check(
            y.policy_requests == f.policy_requests && y.leaf_requests == f.leaf_requests,
            "Y3 根两类请求数一致"
        );
        c.check(
            y.terminal_results == f.terminal_results && y.leaf_results == f.leaf_results,
            "Y3 根两类结局条数一致"
        );
        c.check(y.leaf_results == 0 && y.leaf_requests == 0, "Y3 根一条叶请求都没发");
        c.check(
            y.terminal_results == y.candidates * args.search_n,
            &format!("Y3 根真实终局条数 == 候选 × n（{} vs {}）", y.terminal_results, y.candidates * args.search_n)
        );

        // ---- 第 3 条：普通根上混合 == Uniform(h8)
        let (h, y) = (&r_h8[1], &r_hy[1]);
        c.check(y.route == "other_trunc" && y.depth == "h8", "普通根走截断");
        c.check(p_hy[1] == p_h8[1], &format!("普通根推荐动作一致：混合 {} vs h8 {}", p_hy[1], p_h8[1]));
        c.check(y.rf_actual == 0.0 && y.objective == "mean", "普通根实际聚合目标是 mean（rf 不参与）");
        c.check(
            y.policy_requests == h.policy_requests && y.leaf_requests == h.leaf_requests,
            &format!(
                "普通根两类请求数一致：{}/{} vs {}/{}",
                y.policy_requests, y.leaf_requests, h.policy_requests, h.leaf_requests
            )
        );
        c.check(
            y.terminal_results == h.terminal_results && y.leaf_results == h.leaf_results,
            "普通根两类结局条数一致"
        );
        c.check(y.leaf_requests > 0, "普通根确实发了叶请求");

        // ---- 第 4 条：H 超出剩余回合
        let y = &r_hy[2];
        c.check(y.route == "other_trunc" && y.depth == "h8", "晚期根仍走截断路由");
        c.check(y.leaf_requests == 0 && y.leaf_results == 0, "剩余回合不足时一条叶请求都没发");
        c.check(y.objective == "mean" && y.rf_actual == 0.0, "剩余回合不足仍按 mean 选优，不自动切回 rf");
        c.check(
            y.terminal_results == y.candidates * args.search_n,
            "剩余回合不足时全部走到真实终局"
        );
        c.check(p_hy[2] == p_h8[2], "晚期根与 uniform-h8 推荐动作一致");

        // ---- 第 5 条：同一个混合 trainer 连跑「普通根 → Y3 根」，rf 不被污染
        let (_, rows, s4) = run(LeafRoute::HybridY3Full(8), &[1usize, 0], sc)?;
        sc = s4;
        println!(
            "\n连跑顺序检查：第 1 根 {} rf {:.6}；第 2 根 {} rf {:.6}",
            rows[0].route, rows[0].rf_actual, rows[1].route, rows[1].rf_actual
        );
        c.check(rows[0].rf_actual == 0.0, "先跑的截断根 rf 记为 0");
        c.check(
            rows[1].rf_actual.to_bits() == f.rf_actual.to_bits(),
            "后跑的 Y3 根拿回原 rf，没有被上一根的 mean 口径污染"
        );
        c.check(
            config.radical_factor_max == args.radical_factor_max,
            "公共 SearchConfig 的 radical_factor_max 没有被写回"
        );
        drop(sc);
        c.finish()
    }

    /// 截断深度的目标说明与短名不会与 full 混淆
    ///
    /// # 错误
    ///
    /// 任一项与期望不符时报错。
    #[test]
    fn leaf_depth_labels_and_targets() -> Result<()> {
        let mut c = Chk::default();
        c.check(LeafDepth::Full.target_turn(12).is_none(), "full 没有目标回合");
        c.check(
            LeafDepth::Turns(4).target_turn(12) == Some(16),
            "H=4 自根回合 12 起目标回合为 16"
        );
        c.check(LeafDepth::Turns(8).tag() == "h8", "短名为 h8");
        c.check(
            LeafDepth::Turns(4).objective().contains("rf 不参与聚合"),
            "截断模式的目标说明写明 rf 不参与聚合"
        );
        c.check(
            LeafDepth::Full.objective().contains("radical_factor"),
            "full 模式的目标说明仍是原 rf 口径"
        );
        c.check(
            leaf_safe_stage(&RamenStage::Train)
                && leaf_safe_stage(&RamenStage::RamenSelect)
                && leaf_safe_stage(&RamenStage::RegionSelect)
                && leaf_safe_stage(&RamenStage::SuperRamenSelect),
            "四个决策阶段可截断"
        );
        c.check(
            !leaf_safe_stage(&RamenStage::SpecialSelect),
            "SpecialSelect 不截断（canonical 口径下与上一拍特征相同）"
        );
        c.check(
            !leaf_safe_stage(&RamenStage::Settlement)
                && !leaf_safe_stage(&RamenStage::Begin)
                && !leaf_safe_stage(&RamenStage::Distribute)
                && !leaf_safe_stage(&RamenStage::AfterTrain)
                && !leaf_safe_stage(&RamenStage::NextTurn)
                && !leaf_safe_stage(&RamenStage::BeginAfterRegionSelect),
            "非决策阶段与 Settlement 一律不截断"
        );
        c.finish()
    }

    /// 配对差按**列**求差，而不是拿两个候选各自的方差相加
    ///
    /// # 错误
    ///
    /// 与手算配对统计量不符，或缺列没有被拦下时报错。
    #[test]
    fn paired_diff_uses_column_pairing() -> Result<()> {
        let mut c = Chk::default();
        // 两个候选的绝对方差都很大，但**逐列差**恒为 10：配对方差应当是 0
        let a = vec![100.0, 200.0, 300.0, 400.0];
        let b: Vec<f64> = a.iter().map(|x| x - 10.0).collect();
        let cells = grid_cells(&[a.clone(), b.clone()]);
        let (mean, se, n) = paired_diff(&cells, 0, 1, 0, 4)?;
        println!("配对差均值 {mean} 标准误 {se} 列数 {n}");
        c.check((mean - 10.0).abs() < 1e-12, &format!("配对差均值应为 10，实得 {mean}"));
        c.check(se.abs() < 1e-12, &format!("逐列差恒定 ⇒ 配对标准误应为 0，实得 {se}"));
        c.check(n == 4, "配对列数为 4");
        // 同一个候选与自身的差恒为 0
        let (m0, s0, _) = paired_diff(&cells, 1, 1, 0, 4)?;
        c.check(m0 == 0.0 && s0 == 0.0, "与自身配对差为 0");
        // 缺列必须报错而不是静默少算
        let mut missing = cells.clone();
        missing.retain(|x| !(x.candidate == 1 && x.j == 2));
        match paired_diff(&missing, 0, 1, 0, 4) {
            Ok(_) => c.check(false, "缺列应当报错"),
            Err(e) => {
                println!("拦下：{e}");
                c.check(true, "缺列被拦下");
            }
        }
        c.finish()
    }

    /// 物理批档位规则的边界值
    ///
    /// # 错误
    ///
    /// 任一用例与期望不符时报错。
    #[test]
    fn batch_tier_boundaries() -> Result<()> {
        // (自适应, 侧车上限, 本根总 rollout 数, 期望档位)
        let cases = [
            (false, 512usize, 30720usize, 512usize),
            (false, 512, 1, 512),
            (false, 1024, 512, 1024),
            (true, 1024, 1, 256),
            (true, 1024, 255, 256),
            (true, 1024, 256, 256),
            (true, 1024, 511, 256),
            (true, 1024, 512, 512),
            (true, 1024, 1023, 512),
            (true, 1024, 1024, 1024),
            (true, 1024, 30720, 1024),
            (true, 512, 30720, 512),
            (true, 128, 30720, 128)
        ];
        let mut bad = 0usize;
        for (ad, max_b, total, want) in cases {
            let got = plan_batch(ad, max_b, total);
            let mark = if got == want { "ok" } else { "❗" };
            println!("{mark} 自适应={ad} 上限={max_b} 总 rollout={total} → 档位 {got}（期望 {want}）");
            if got != want {
                bad += 1;
            }
        }
        println!("档位规则：{} 条用例，不符 {bad} 条", cases.len());
        ensure!(bad == 0, "档位规则有 {bad} 条用例与期望不符");
        Ok(())
    }

    /// 缓存判定的全部分支
    ///
    /// ❗「输入含 NaN 但缓存是正常值」期望是 `FeatDiff` 而不是 `NonFinite`：
    /// 位模式先不同，非有限检查根本轮不到——照实写期望，不为了好看改判定顺序。
    ///
    /// # 错误
    ///
    /// 任一分支与期望不符时报错。
    #[test]
    fn cache_probe_branches() -> Result<()> {
        let base = feats(0.25);
        let good = PolicyCache {
            turn: 12,
            features: base.clone(),
            policy: policy(0.5)
        };
        // 只翻最后一个元素的最低位：数值上几乎相同，位模式不同 ⇒ 必须判未命中
        let mut one_bit = base.clone();
        let last = one_bit.len() - 1;
        one_bit[last] = f32::from_bits(base[last].to_bits() ^ 1);
        let short = base[..INPUT_DIM - 1].to_vec();
        let mut nan_in = base.clone();
        nan_in[0] = f32::NAN;
        let nan_cache = PolicyCache {
            turn: 12,
            features: nan_in.clone(),
            policy: policy(0.5)
        };
        let mut bad_policy = policy(0.5);
        bad_policy[3] = f32::INFINITY;
        let inf_cache = PolicyCache {
            turn: 12,
            features: base.clone(),
            policy: bad_policy
        };

        let cases: Vec<(&str, CacheProbe, CacheProbe)> = vec![
            ("正常命中", cache_probe(Some(&good), 12, true, &base), CacheProbe::Hit),
            (
                "当前不是 SpecialSelect",
                cache_probe(Some(&good), 12, false, &base),
                CacheProbe::NotEligible
            ),
            ("没有前一拍", cache_probe(None, 12, true, &base), CacheProbe::Miss(CacheMiss::NoPrev)),
            ("换回合", cache_probe(Some(&good), 13, true, &base), CacheProbe::Miss(CacheMiss::TurnDiff)),
            (
                "输入差一个位",
                cache_probe(Some(&good), 12, true, &one_bit),
                CacheProbe::Miss(CacheMiss::FeatDiff)
            ),
            (
                "输入长度不同",
                cache_probe(Some(&good), 12, true, &short),
                CacheProbe::Miss(CacheMiss::FeatDiff)
            ),
            (
                "输入含 NaN 而缓存是正常值",
                cache_probe(Some(&good), 12, true, &nan_in),
                CacheProbe::Miss(CacheMiss::FeatDiff)
            ),
            (
                "输入含 NaN 且与缓存位模式相同",
                cache_probe(Some(&nan_cache), 12, true, &nan_in),
                CacheProbe::Miss(CacheMiss::NonFinite)
            ),
            (
                "缓存 policy 含 Inf",
                cache_probe(Some(&inf_cache), 12, true, &base),
                CacheProbe::Miss(CacheMiss::NonFinite)
            )
        ];
        let mut bad = 0usize;
        for (name, got, want) in &cases {
            let mark = if got == want { "ok" } else { "❗" };
            println!("{mark} {name}：{got:?}（期望 {want:?}）");
            if got != want {
                bad += 1;
            }
        }
        println!("缓存判定：{} 条用例，不符 {bad} 条", cases.len());
        ensure!(bad == 0, "缓存判定有 {bad} 条用例与期望不符");
        Ok(())
    }

    /// 就绪行字段解析：读实际字段，缺字段返回 `None`
    ///
    /// # 错误
    ///
    /// 任一用例与期望不符时报错。
    #[test]
    fn banner_fields() -> Result<()> {
        let line = "[sidecar] ready device=cuda B=1024 members=3 tf32=off warmup=3 proto=2 maxB=1024";
        let cases: Vec<(&str, Option<u64>, Option<u64>)> = vec![
            ("proto=", banner_u64(line, "proto="), Some(2)),
            ("maxB=", banner_u64(line, "maxB="), Some(1024)),
            ("members=", banner_u64(line, "members="), Some(3)),
            ("缺失字段 foo=", banner_u64(line, "foo="), None),
            ("非整数 device=", banner_u64(line, "device="), None)
        ];
        let mut bad = 0usize;
        for (name, got, want) in &cases {
            let mark = if got == want { "ok" } else { "❗" };
            println!("{mark} {name} → {got:?}（期望 {want:?}）");
            if got != want {
                bad += 1;
            }
        }
        println!("就绪行解析：{} 条用例，不符 {bad} 条", cases.len());
        ensure!(bad == 0, "就绪行解析有 {bad} 条用例与期望不符");
        Ok(())
    }

    /// 开关支持矩阵：不消费开关的组合必须被拒
    ///
    /// ❗`game-smoke` 经 [`WaveTrainer`] **已接线**，不能错误拒绝。
    ///
    /// # 错误
    ///
    /// 任一组合的接受/拒绝与期望不符时报错。
    #[test]
    fn switch_support_matrix() -> Result<()> {
        let modes = [
            (Mode::Compare, false),
            (Mode::CpuCandidate, false),
            (Mode::CpuFlat, false),
            (Mode::GpuWave, true),
            (Mode::SidecarReuse, true),
            (Mode::GameSmoke, true),
            (Mode::TeacherConsistency, true),
            (Mode::TeacherGame, true)
        ];
        let mut bad = 0usize;
        let mut cases = 0usize;
        for (mode, supported) in modes {
            // 两个开关都不给：任何模式都必须放行
            let none = check_switch_support(mode, false, false, false, false).is_ok();
            cases += 1;
            if !none {
                bad += 1;
            }
            println!("{} {mode:?} 不给开关 → 放行 {none}（期望 true）", if none { "ok" } else { "❗" });
            for (cache_on, adaptive, graph, label) in [
                (true, false, false, "--policy-cache"),
                (false, true, false, "--adaptive-batch"),
                (false, false, true, "--sidecar-graph"),
                (true, false, true, "缓存 + 图"),
                (true, true, false, "缓存 + 自适应批")
            ] {
                let ok = check_switch_support(mode, cache_on, adaptive, graph, false).is_ok();
                cases += 1;
                if ok != supported {
                    bad += 1;
                }
                println!(
                    "{} {mode:?} + {label} → 放行 {ok}（期望 {supported}）",
                    if ok == supported { "ok" } else { "❗" }
                );
            }
            // 图与自适应批互斥：即便在支持的模式上也必须拒
            let both = check_switch_support(mode, false, true, true, false).is_ok();
            cases += 1;
            if both {
                bad += 1;
            }
            println!("{} {mode:?} + 图 + 自适应批 → 放行 {both}（期望 false）", if both { "❗" } else { "ok" });
        }
        // 手写臂不接网络、不起侧车：给了开关必须拒
        for (cache_on, adaptive, graph, label) in [
            (true, false, false, "--policy-cache"),
            (false, true, false, "--adaptive-batch"),
            (false, false, true, "--sidecar-graph")
        ] {
            let ok = check_switch_support(Mode::TeacherGame, cache_on, adaptive, graph, true).is_ok();
            cases += 1;
            if ok {
                bad += 1;
            }
            println!("{} teacher-game + --handwritten-rollout + {label} → 放行 {ok}（期望 false）", if ok { "❗" } else { "ok" });
        }
        // 手写臂不给开关仍要放行
        let ok = check_switch_support(Mode::TeacherGame, false, false, false, true).is_ok();
        cases += 1;
        if !ok {
            bad += 1;
        }
        println!("{} teacher-game + --handwritten-rollout 不给开关 → 放行 {ok}（期望 true）", if ok { "ok" } else { "❗" });
        println!("支持矩阵：{cases} 条用例，不符 {bad} 条");
        ensure!(bad == 0, "支持矩阵有 {bad} 条用例与期望不符");
        Ok(())
    }

    /// 零请求根不得污染最低批利用率
    ///
    /// # 错误
    ///
    /// 任一用例与期望不符时报错。
    #[test]
    fn zero_request_fill_stats() -> Result<()> {
        let mut bad = 0usize;

        // 用例 1：先零请求再有请求 —— 最低填充率只能来自有格位的那两个根
        let mut a = GameStats::default();
        a.note_fill(0, 0);
        a.note_fill(450, 512);
        a.note_fill(300, 512);
        let want_min = 300.0 / 512.0;
        let ok1 = a.zero_request_roots == 1 && a.fill_roots == 2 && (a.min_fill - want_min).abs() < 1e-12;
        println!(
            "{} 零请求→有请求：零请求根 {}（期望 1）有格位根 {}（期望 2）最低填充率 {:.4}（期望 {:.4}）",
            if ok1 { "ok" } else { "❗" },
            a.zero_request_roots,
            a.fill_roots,
            a.min_fill,
            want_min
        );
        println!("    文本：{}", a.min_fill_text());
        if !ok1 {
            bad += 1;
        }

        // 用例 2：有请求在前、零请求在后，同样不得被拉到 0
        let mut b = GameStats::default();
        b.note_fill(500, 512);
        b.note_fill(0, 0);
        let ok2 = b.fill_roots == 1 && b.zero_request_roots == 1 && (b.min_fill - 500.0 / 512.0).abs() < 1e-12;
        println!(
            "{} 有请求→零请求：最低填充率 {:.4}（期望 {:.4}），文本 {}",
            if ok2 { "ok" } else { "❗" },
            b.min_fill,
            500.0 / 512.0,
            b.min_fill_text()
        );
        if !ok2 {
            bad += 1;
        }

        // 用例 3：全是零请求根 —— 必须 N/A，不得显示 0%
        let mut c = GameStats::default();
        c.note_fill(0, 0);
        c.note_fill(0, 0);
        let txt = c.min_fill_text();
        let ok3 = c.fill_roots == 0 && c.zero_request_roots == 2 && txt.contains("N/A");
        println!(
            "{} 全零请求：有格位根 {}（期望 0）零请求根 {}（期望 2）文本 {txt}",
            if ok3 { "ok" } else { "❗" },
            c.fill_roots,
            c.zero_request_roots
        );
        if !ok3 {
            bad += 1;
        }

        // 用例 4：整局汇总文本同样不得把 0/0 写成 0%
        let ok4 = fill_text(0, 0).contains("N/A") && fill_text(450, 512).starts_with("87.9");
        println!(
            "{} 汇总文本：fill_text(0,0)={} fill_text(450,512)={}",
            if ok4 { "ok" } else { "❗" },
            fill_text(0, 0),
            fill_text(450, 512)
        );
        if !ok4 {
            bad += 1;
        }

        println!("填充率统计：4 条用例，不符 {bad} 条");
        ensure!(bad == 0, "填充率统计有 {bad} 条用例与期望不符");
        Ok(())
    }

    /// 测试用的缓存篡改方式
    ///
    /// ❗只篡改**被测代码写好的缓存**，不另建一套生命周期：写入、命中与清空仍全在
    /// [`Traj::advance`] / [`Traj::run_decision`] 里。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Poison {
        /// 不篡改
        NoTouch,
        /// 把缓存的回合改成下一回合
        Turn,
        /// 翻缓存输入最后一个 f32 的最低位
        Feature,
        /// 每次 resume 之后强行塞一个不可能的回合，用来看清空到底有没有发生
        Inject
    }

    /// 驱动一条 rollout 到终局时观测到的东西
    struct Observed {
        /// 终局评分
        score: SearchScore,
        /// 逻辑网络决策数
        seq: u32,
        /// 实际推理请求数
        infers: u32,
        /// 缓存计数
        cstat: CacheStats,
        /// 缓存被正常写入的次数
        writes: usize,
        /// resume 之后缓存状态与「刚执行的是不是 RamenSelect 推理决策」不符的次数
        bad_after_resume: usize,
        /// 挂起点上缓存还在、但回合已经不是当前回合的次数
        stale_at_suspend: usize,
        /// 挂起点上缓存还在的次数（不论回合）
        alive_at_suspend: usize,
        /// 注入的假缓存被清掉的次数
        inject_cleared: usize,
        /// 注入的假缓存活过了一次回合变化的次数（清空缺失才会发生）
        inject_survived_turn_change: usize
    }

    /// 用真实 [`Traj::advance`] / [`Traj::resume`] 驱动一条 rollout 到终局
    ///
    /// 只负责喂真实推理结果、在可观测点记录状态，并按 `poison` 篡改缓存。
    ///
    /// # 错误
    ///
    /// 建轨迹、推理或规则层报错时上抛。
    fn drive_one(
        nn: &RamenNnTrainer, root: &RamenGame, action: &RamenAction, seed: u64, cache_on: bool, poison: Poison
    ) -> Result<Observed> {
        /// 注入用的不可能回合
        const FAKE_TURN: i32 = -12345;
        // 本用例只驱动 policy 路径：深度固定 Full，不产生叶请求
        let mut t = Traj::start(0, 0, root, action, seed, cache_on, LeafDepth::Full, false)?;
        ensure!(t.cache.is_none(), "新建轨迹不应带缓存");
        let mut ob = Observed {
            score: SearchScore { score: 0.0, score_pt: 0.0 },
            seq: 0,
            infers: 0,
            cstat: CacheStats::default(),
            writes: 0,
            bad_after_resume: 0,
            stale_at_suspend: 0,
            alive_at_suspend: 0,
            inject_cleared: 0,
            inject_survived_turn_change: 0
        };
        let mut last_turn: Option<i32> = None;
        let mut injected = false;
        loop {
            t.advance(nn, false)?;
            if t.phase == Phase::Done {
                break;
            }
            let turn_now = t.game.turn();
            if let Some(c) = t.cache.as_ref() {
                ob.alive_at_suspend += 1;
                if c.turn != turn_now && c.turn != FAKE_TURN {
                    ob.stale_at_suspend += 1;
                }
            }
            if injected {
                let alive = t.cache.as_ref().map(|c| c.turn == FAKE_TURN).unwrap_or(false);
                if alive {
                    // 跨过回合边界必然执行过阶段；这时假缓存还在，就是清空代码漏了
                    if last_turn.is_some_and(|p| p != turn_now) {
                        ob.inject_survived_turn_change += 1;
                    }
                } else {
                    ob.inject_cleared += 1;
                }
                injected = false;
            }
            last_turn = Some(turn_now);
            let req = t
                .request
                .clone()
                .ok_or_else(|| anyhow!("挂起却没有待推理输入"))?;
            let was_ramen_select = t.game.stage == RamenStage::RamenSelect;
            let out = nn.infer_features(req.clone())?;
            // `resume` 现在收**整行**输出；本用例只走 policy 分支，
            // choice 与 value 段补零即可（`LeafDepth::Full` 下根本不会读到它们）
            let mut row = out.policy.clone();
            row.resize(OUTPUT_DIM, 0.0);
            t.resume(nn, &row, false)?;
            match (cache_on && was_ramen_select, t.cache.as_ref()) {
                (true, Some(c)) => {
                    ob.writes += 1;
                    if c.turn != turn_now || c.features != req || c.policy != out.policy {
                        ob.bad_after_resume += 1;
                    }
                }
                (false, None) => {}
                // 该写没写、或不该留却留着，两种都算不符
                _ => ob.bad_after_resume += 1
            }
            match poison {
                Poison::NoTouch => {}
                Poison::Turn => {
                    if let Some(c) = t.cache.as_mut() {
                        c.turn += 1;
                    }
                }
                Poison::Feature => {
                    if let Some(c) = t.cache.as_mut() {
                        let last = c.features.len() - 1;
                        c.features[last] = f32::from_bits(c.features[last].to_bits() ^ 1);
                    }
                }
                Poison::Inject => {
                    t.cache = Some(PolicyCache {
                        turn: FAKE_TURN,
                        features: vec![0.0f32; INPUT_DIM],
                        policy: vec![0.0f32; POLICY_DIM]
                    });
                    injected = true;
                }
            }
        }
        ob.score = match t.outcome.ok_or_else(|| anyhow!("终局轨迹缺少结局记录"))? {
            TrajOutcome::Terminal { score, .. } => score,
            TrajOutcome::Leaf { .. } => bail!("Full 深度下不应出现叶结果")
        };
        ob.seq = t.seq;
        ob.infers = t.infers;
        ob.cstat = t.cstat;
        Ok(ob)
    }

    /// 真实 `Traj` / 阶段执行路径上的缓存生命周期回归
    ///
    /// 覆盖：写入 → 命中 → 消费后清空；非目标阶段 / Resolved / 事件路径经过后失效；
    /// 篡改后必须落回真实推理且结果不变；新轨迹与槽位替换不继承缓存。
    ///
    /// # 错误
    ///
    /// 建局、推理或任一不变量被破坏时报错。
    #[test]
    fn cache_lifecycle_in_traj() -> Result<()> {
        // 测试工作目录统一到 workspace 根（项目约定）
        set_current_dir(get_workspace_root()?)?;
        let model = "saved_models/arms/ens_G2mix_g123.onnx";
        ensure!(
            Path::new(model).exists(),
            "本轮冻结模型不存在：{model}（本测试依赖它，不另造假模型）"
        );
        let mut cfg = load_game_config()?;
        // 与主流程同一前提
        cfg.ramen_region_strategy = RamenRegionStrategy::All;
        init_global_with_config(&cfg)?;
        let args = RootArgs::try_parse_from([
            "ramen_root_bench",
            "--mode",
            "gpu-wave",
            "--rollout-model",
            model,
            "--search-n",
            "1",
            "--root-turn",
            "6",
            "--root-stage",
            "RamenSelect"
        ])?;
        let space = space_from_cli(None, &[])?;
        let plans = space.plans();
        let plan = plans.first().ok_or_else(|| anyhow!("采样空间没有计划"))?;
        let inherit = gen1_inherit();
        let (root, mut rng) =
            build_root(&args, plan.uma, &plan.deck, &inherit, args.root_turn, args.root_stage.as_deref())?;
        let actions = root.list_actions()?;
        ensure!(!actions.is_empty(), "根上没有合法候选");
        println!("固定根 t{} {:?}：候选 {} 个", root.turn(), root.stage, actions.len());
        let nn = RamenNnTrainer::load(&args.rollout_model)?
            .with_special_mode(SpecialSelectMode::Canonical)
            .with_race_shield(true);
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());

        let mut bad = 0usize;
        let mut checks = 0usize;
        let mut total_writes = 0usize;
        let mut total_hits = 0usize;
        let mut total_cleared = 0usize;
        for j in 0..2usize {
            let seed = seeds.seed_at(j);
            let off = drive_one(&nn, &root, &actions[0], seed, false, Poison::NoTouch)?;
            let on = drive_one(&nn, &root, &actions[0], seed, true, Poison::NoTouch)?;
            let pt = drive_one(&nn, &root, &actions[0], seed, true, Poison::Turn)?;
            let pf = drive_one(&nn, &root, &actions[0], seed, true, Poison::Feature)?;
            let pi = drive_one(&nn, &root, &actions[0], seed, true, Poison::Inject)?;
            total_writes += on.writes;
            total_hits += on.cstat.hits;
            total_cleared += pi.inject_cleared;
            println!(
                "\n[j={j}] 关缓存：决策 {} 请求 {} 写入 {} | 开缓存：决策 {} 请求 {} 写入 {} 口径内 {} 命中 {}",
                off.seq, off.infers, off.writes, on.seq, on.infers, on.writes, on.cstat.eligible, on.cstat.hits
            );
            println!(
                "        投毒(回合)：命中 {} turn_diff {} 请求 {} | 投毒(一个位)：命中 {} feat_diff {} 请求 {}",
                pt.cstat.hits, pt.cstat.turn_diff, pt.infers, pf.cstat.hits, pf.cstat.feat_diff, pf.infers
            );
            println!(
                "        注入假缓存：被清掉 {} 次，活过回合变化 {} 次 | 挂起点缓存尚存 {} 次（其中过期 {} 次）",
                pi.inject_cleared, pi.inject_survived_turn_change, on.alive_at_suspend, on.stale_at_suspend
            );

            let mut expect = |name: &str, cond: bool| {
                checks += 1;
                if !cond {
                    bad += 1;
                    println!("        ❗{name}");
                } else {
                    println!("        ok {name}");
                }
            };
            expect("关缓存时从不写入、计数全零", off.writes == 0 && off.cstat.eligible == 0 && off.cstat.hits == 0);
            expect("关缓存时请求数等于逻辑决策数", off.seq == off.infers);
            expect("开缓存确实写入过（覆盖到写入→命中链）", on.writes > 0);
            expect("开缓存确实命中过", on.cstat.hits > 0);
            expect("每次 resume 后缓存状态都符合设计（写入内容逐字段相同 / 该清就清）", on.bad_after_resume == 0);
            expect("挂起点上没有过期缓存残留", on.stale_at_suspend == 0);
            expect("逻辑决策数不因缓存改变", on.seq == off.seq && pt.seq == off.seq && pf.seq == off.seq);
            expect(
                "终局评分不因缓存改变",
                on.score.score == off.score.score
                    && on.score.score_pt == off.score.score_pt
                    && pt.score.score == off.score.score
                    && pf.score.score == off.score.score
            );
            expect("命中几次就少发几次请求", on.infers + on.cstat.hits as u32 == off.infers);
            expect("回合被改过的缓存一律不命中", pt.cstat.hits == 0 && pt.cstat.turn_diff == pt.cstat.eligible);
            expect("输入差一个位的缓存一律不命中", pf.cstat.hits == 0 && pf.cstat.feat_diff == pf.cstat.eligible);
            expect("拒绝复用后老老实实落回真实推理", pt.infers == off.infers && pf.infers == off.infers);
            expect("注入的假缓存被真实清空路径清掉过", pi.inject_cleared > 0);
            expect("假缓存没有活过任何一次回合变化", pi.inject_survived_turn_change == 0);
            expect("假缓存永远不会被复用", pi.cstat.hits == 0);
        }

        // 不继承：新轨迹、以及 Vec 槽位被替换后的新轨迹，都不得带上一条的缓存
        let seed0 = seeds.seed_at(0);
        let mut slot: Vec<Traj> = vec![Traj::start(0, 0, &root, &actions[0], seed0, true, LeafDepth::Full, false)?];
        loop {
            slot[0].advance(&nn, false)?;
            if slot[0].phase == Phase::Done {
                break;
            }
            let req = slot[0]
                .request
                .clone()
                .ok_or_else(|| anyhow!("挂起却没有待推理输入"))?;
            let ramen = slot[0].game.stage == RamenStage::RamenSelect;
            let out = nn.infer_features(req)?;
            let mut row = out.policy;
            row.resize(OUTPUT_DIM, 0.0);
            slot[0].resume(&nn, &row, false)?;
            if ramen && slot[0].cache.is_some() {
                break;
            }
        }
        let had_cache = slot[0].cache.is_some();
        slot[0] = Traj::start(1, 1, &root, &actions[0], seeds.seed_at(1), true, LeafDepth::Full, false)?;
        checks += 2;
        if !had_cache {
            bad += 1;
            println!("❗替换前那条轨迹没能带上缓存，槽位复用这一项没覆盖到");
        } else {
            println!("ok 替换前轨迹带着缓存");
        }
        if slot[0].cache.is_some() || slot[0].cstat.hits != 0 {
            bad += 1;
            println!("❗槽位替换后的新轨迹继承了旧缓存");
        } else {
            println!("ok 槽位替换后的新轨迹不带缓存、计数清零");
        }

        println!(
            "\n缓存生命周期：{checks} 条断言，不符 {bad} 条；累计写入 {total_writes} 次、命中 {total_hits} 次、清掉注入缓存 {total_cleared} 次"
        );
        ensure!(bad == 0, "缓存生命周期有 {bad} 条断言不成立");
        Ok(())
    }
}
