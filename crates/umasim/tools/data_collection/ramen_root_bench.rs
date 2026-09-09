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
    collections::{HashMap, VecDeque},
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Instant
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use rand::rngs::StdRng;
use rayon::prelude::*;
use rand::{RngCore, SeedableRng};
use std::cell::Cell;
use umasim::{
    bench::seeded_rngs,
    game::{
        Game, InheritInfo, Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData, RamenRegionStrategy, init_global_with_config},
    sampler::{gen1_inherit, space_from_cli},
    search::{
        FlatSearch, FlatSearchGame, RamenBatchRollout, RamenBatchTable, RamenTerminal, RolloutOutcome, RolloutSeeds,
        SearchConfig, SearchScore
    },
    trainer::{
        DecisionPrep, DecisionSink, DecisionSnapshot, RamenMctsTrainer, RamenNnTrainer, RamenRolloutTrainer,
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
    TeacherGame
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
    #[arg(long, default_value_t = 512)]
    batch: usize,

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

    /// 逐决策等价性输出 CSV
    ///
    /// 落的是**实际值**：合法候选的原顺序、选中动作、完整输入特征。
    /// ❗体积随 `候选数 × n × 决策数` 线性增长，详细对拍请用小 `n`，与性能测量分开跑。
    #[arg(long)]
    decision_csv: Option<PathBuf>
}

/// 侧车就绪握手的标记行
///
/// 侧车必须在**模型加载与预热都完成之后**才写这一行：否则首个 `infer` 会把权重
/// 加载、CUDA 上下文建立与首次 kernel 编译一并算进推理耗时。
const SIDECAR_READY_MARK: &str = "[sidecar] ready";

/// 排空线程保留的诊断行数上限
const SIDECAR_DIAG_LINES: usize = 64;

/// 常驻侧车的客户端
struct Sidecar {
    /// 子进程句柄；`Drop` 时关闭 stdin 让侧车自行退出
    child: Child,
    /// 物理批尺寸
    batch: usize,
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

        Ok(Self {
            child,
            batch: args.batch,
            diag,
            drain: Some(drain),
            startup_s: t0.elapsed().as_secs_f64(),
            banner
        })
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
        ensure!(
            rows.len() <= self.batch,
            "一批 {} 行超过物理批尺寸 {}",
            rows.len(),
            self.batch
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
    /// 决策 RNG 指纹
    rng_probe: u64,
    /// 输入特征指纹
    feat_hash: u64,
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
    /// 该 rollout 实际发生的 NN 推理次数（可能为 0）
    requests: u32
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

impl DecisionSink for DecisionLogSink {
    fn on_inferred(&self, snap: DecisionSnapshot) -> Result<()> {
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
    pending: Mutex<Vec<(DecisionSnapshot, RolloutCtx, f64)>>,
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

/// FNV-1a 64（与仓库口径一致），用于特征指纹
fn fnv1a64_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl CompareSink {
    /// 对一批快照做 GPU 推理与赢家比较
    ///
    /// # 错误
    ///
    /// 侧车通信失败、返回行数不符或打分失败时报错。
    fn compare_batch(&self, batch: Vec<(DecisionSnapshot, RolloutCtx, f64)>) -> Result<()> {
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
                feat_hash: fnv1a64_f32(&snap.features),
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

impl DecisionSink for CompareSink {
    fn on_inferred(&self, snap: DecisionSnapshot) -> Result<()> {
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
    let (mut rng, rule_master) = seeded_rngs(args.seed, args.run_idx);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_rule_master(rule_master);
    let guide = RecommendedRamenTrainer::for_rollout();
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
        game.run_stage(&guide, &mut rng)?;
    }
    bail!("推进到终局仍未命中目标根（回合 >= {root_turn}，阶段 {want:?}）")
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
        let t0 = Instant::now();
        let (cells, _, _, _, st) = run_gpu_wave(game, actions, args.search_n, &seeds, nn, &mut sc, args.batch, false)?;
        let (best, _) = official_result(search, game, actions, &cells, &mut rng)?;
        println!(
            "  [{label}] {:.1} s 波次 {} 请求 {} 利用率 {:.1}% 最优候选 {best}",
            t0.elapsed().as_secs_f64(),
            st.waves,
            st.served,
            100.0 * st.served as f64 / st.slots as f64
        );
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
        batch: args.batch,
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
        return Ok(());
    }
    let sidecar = Sidecar::start(args)?;
    println!("  侧车就绪 {:.1} s | {}（整局复用，不计入整局墙钟）", sidecar.startup_s, sidecar.banner);
    let backend = Arc::new(WaveBackend {
        nn: Arc::clone(nn),
        sidecar: Mutex::new(sidecar),
        batch: args.batch,
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
        "  请求 / 格位 {} / {}（整局批利用率 {:.1}%）",
        st.served,
        st.slots,
        100.0 * st.served as f64 / st.slots.max(1) as f64
    );
    println!("  ❗单次搜索最低批利用率 {:.1}%", 100.0 * st.min_fill);
    println!("  波次        {}", st.waves);
    println!("  rollout 总数 {}", st.rollouts);
    println!(
        "  后端拆分    CPU 段 {:.1} s | 侧车往返 {:.1} s（轨迹自报 CPU 累加 {:.1} s，跨 worker，不可加进墙钟）",
        st.cpu_wall_s, st.gpu_wall_s, st.cpu_in_traj_s
    );
    println!("  ❗「侧车往返」是 infer() 整段：含序列化/解码、两次管道、主机↔设备拷贝与前向，未再拆分");
    println!("  终局评分    score={:.3} score_pt={:.3}", score.score, score.score_pt);
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
    inherit: &InheritInfo
) -> Result<()> {
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
        batch: args.batch,
        n: args.search_n,
        fallback: RecommendedRamenTrainer::for_rollout(),
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
    println!("  平均候选数  {:.1}", st.candidates as f64 / st.searched.max(1) as f64);
    println!(
        "  请求 / 格位 {} / {}（整局批利用率 {:.1}%）",
        st.served,
        st.slots,
        100.0 * st.served as f64 / st.slots.max(1) as f64
    );
    println!("  ❗单点最低批利用率 {:.1}%（小候选数 / 晚期根填不满批）", 100.0 * st.min_fill);
    println!("  波次        {}", st.waves);
    println!("  终局评分    score={:.3} score_pt={:.3}", score.score, score.score_pt);
    println!("  ❗冒烟只验证接入，回答不了 greedy(Q^NN) vs greedy(Q^手写)");
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
            RolloutRow { candidate, j, requests }
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
    /// 已发生的推理次数
    seq: u32,
    /// 推进阶段
    phase: Phase,
    /// 终局评分（`Done` 后才有）
    score: Option<SearchScore>,
    /// 终局多维记录（`Done` 后才有）
    terminal: Option<RamenTerminal>,
    /// 本次挂起待推理的输入（`RunStage` 且需网络时有值）
    request: Option<Vec<f32>>,
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
    /// 终局评分
    score: SearchScore,
    /// 该 rollout 实际发生的推理次数
    requests: u32,
    /// 该 rollout 自报的 CPU 推进耗时
    cpu_ns: u128,
    /// 终局多维记录
    terminal: RamenTerminal,
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
    fn start(candidate: usize, j: usize, game: &RamenGame, action: &RamenAction, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sim = game.fork_for_rollout(seed);
        sim.apply_root_action(action, &mut rng)?;
        Ok(Self {
            candidate,
            j,
            game: sim,
            rng,
            seq: 0,
            phase: Phase::NeedNext,
            score: None,
            terminal: None,
            request: None,
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
    fn advance(&mut self, nn: &RamenNnTrainer) -> Result<()> {
        let t0 = Instant::now();
        loop {
            match self.phase {
                Phase::Done => break,
                Phase::NeedNext => {
                    if self.game.next() {
                        self.phase = Phase::RunStage;
                    } else {
                        let t = PreselectTrainer::new(nn, None, None);
                        self.game.on_simulation_end(&t, &mut self.rng)?;
                        self.score = Some(self.game.search_score());
                        // 终局多维记录必须在这里取：`finish` 之后局面就没了
                        self.terminal = Some(RamenTerminal::from_game(&self.game));
                        self.phase = Phase::Done;
                        break;
                    }
                }
                Phase::RunStage => {
                    if is_decision_stage(&self.game.stage) {
                        let actions = self.game.list_actions()?;
                        // 用 RNG 的**克隆体**探测：探测不得改变生产随机流
                        let mut probe = self.rng.clone();
                        if let DecisionPrep::NeedsInference(f) =
                            nn.prepare_decision(&self.game, &actions, &mut probe)?
                        {
                            self.request = Some(f);
                            break;
                        }
                    }
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

    /// 用批量推理回来的 policy 执行挂起的那一个阶段
    ///
    /// # 错误
    ///
    /// 当前并非挂起在决策点、规则层报错，或答案未被恰好消费一次时报错。
    fn resume(&mut self, nn: &RamenNnTrainer, policy: Vec<f32>, record: bool) -> Result<()> {
        let expect = self
            .request
            .take()
            .ok_or_else(|| anyhow!("该 rollout 并未挂起在决策点"))?;
        ensure!(self.phase == Phase::RunStage, "恢复时阶段状态不是 RunStage");
        let t0 = Instant::now();
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
        self.phase = Phase::NeedNext;
        self.seq += 1;
        self.cpu_ns += t0.elapsed().as_nanos();
        Ok(())
    }

    /// 终局轨迹转成完成记录，**消费自身**以释放其持有的完整局面
    ///
    /// # 错误
    ///
    /// 尚未终局或缺终局评分时报错。
    fn finish(self, seed: u64) -> Result<TrajDone> {
        ensure!(self.phase == Phase::Done, "轨迹尚未终局，不能转成完成记录");
        let score = self.score.ok_or_else(|| anyhow!("终局轨迹缺少评分"))?;
        let terminal = self.terminal.ok_or_else(|| anyhow!("终局轨迹缺少多维记录"))?;
        Ok(TrajDone {
            candidate: self.candidate,
            j: self.j,
            seed,
            score,
            requests: self.seq,
            cpu_ns: self.cpu_ns,
            terminal,
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
    sidecar: &mut Sidecar, batch: usize, record: bool
) -> Result<(Vec<RawCell>, Vec<RolloutRow>, Vec<DecisionRow>, RamenBatchTable, WaveStats)> {
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
        active.push(Traj::start(c, j, game, &actions[c], seeds.seed_at(j))?);
    }

    while !active.is_empty() {
        // 1) CPU 并行推进到各自的下一个请求或终局
        let adv = Instant::now();
        active.par_iter_mut().try_for_each(|t| t.advance(nn))?;
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
            let mut t = Traj::start(c, j, game, &actions[c], seeds.seed_at(j))?;
            let adv = Instant::now();
            t.advance(nn)?;
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
        stats.waves += 1;
        stats.served += rows.len();
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
            .try_for_each(|(t, out)| t.resume(nn, out[..POLICY_DIM].to_vec(), record))?;
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
    let mut cells = Vec::with_capacity(done.len());
    let mut rollout_rows = Vec::with_capacity(done.len());
    let mut decisions = Vec::new();
    let mut table: RamenBatchTable = HashMap::with_capacity(cells.capacity());
    for t in done {
        cells.push(RawCell {
            candidate: t.candidate,
            j: t.j,
            seed: t.seed,
            score: t.score.score,
            score_pt: t.score.score_pt
        });
        rollout_rows.push(RolloutRow {
            candidate: t.candidate,
            j: t.j,
            requests: t.requests
        });
        stats.cpu_in_traj_s += t.cpu_ns as f64 / 1e9;
        decisions.extend(t.rows);
        table.insert(
            (t.candidate, t.seed),
            RolloutOutcome {
                score: t.score,
                terminal: t.terminal
            }
        );
    }
    // 查表键是 `(候选, seed)`：若同一候选内出现重复 seed，表项会被覆盖而条数变少，
    // 而 `cells` 仍然是满的。官方汇总走 `cells`、内核查表走 `table`，两边都要守。
    for (c, got) in per_candidate.iter().enumerate() {
        let in_table = table.keys().filter(|(cand, _)| *cand == c).count();
        ensure!(in_table == *got, "候选 {c} 查表项 {in_table} 条与完成记录 {got} 条不符（seed 重复？）");
    }
    stats.total_wall_s = total_start.elapsed().as_secs_f64();
    Ok((cells, rollout_rows, decisions, table, stats))
}

/// 把波次驱动接进生产教师的批量后端
///
/// 侧车由本结构持有并跨决策点复用；请求串行（整锁），不做跨局共享队列。
struct WaveBackend {
    /// 网络策略（rollout 基策）
    nn: Arc<RamenNnTrainer>,
    /// 常驻侧车
    sidecar: Mutex<Sidecar>,
    /// 物理批尺寸
    batch: usize,
    /// 累计统计
    stats: Mutex<GameStats>
}

impl RamenBatchRollout for WaveBackend {
    fn precompute(
        &self, game: &RamenGame, actions: &[RamenAction], seeds: &RolloutSeeds, n: usize
    ) -> Result<RamenBatchTable> {
        let t0 = Instant::now();
        let (_, _, _, table, wave) = {
            let mut sc = self.sidecar.lock().map_err(|_| anyhow!("侧车锁被毒化"))?;
            run_gpu_wave(game, actions, n, seeds, &self.nn, &mut sc, self.batch, false)?
        };
        let mut st = self.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
        let fill = wave.served as f64 / wave.slots.max(1) as f64;
        st.min_fill = if st.searched == 0 { fill } else { st.min_fill.min(fill) };
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
    cpu_in_traj_s: f64
}

/// 整局冒烟的累计统计
#[derive(Default)]
struct GameStats {
    /// 走了搜索的决策点数
    searched: usize,
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
    /// 单个决策点的最低批利用率
    min_fill: f64,
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
    cpu_in_traj_s: f64
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
    /// 物理批尺寸
    batch: usize,
    /// 每候选 rollout 条数
    n: usize,
    /// 非搜索出口
    fallback: RecommendedRamenTrainer,
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
        // 预跑与官方汇总共用同一张种子表：clone 派生，原 rng 留给 search_with
        let seeds = RolloutSeeds::from_rng(&mut rng.clone());
        let (cells, _rows, _decisions, _table, wave) = {
            let mut sc = self.sidecar.lock().map_err(|_| anyhow!("侧车锁被毒化"))?;
            run_gpu_wave(game, actions, self.n, &seeds, self.nn, &mut sc, self.batch, false)?
        };
        let (best, _) = official_result(self.search, game, actions, &cells, rng)?;
        let mut st = self.stats.lock().map_err(|_| anyhow!("统计锁被毒化"))?;
        let fill = wave.served as f64 / wave.slots as f64;
        st.min_fill = if st.searched == 0 { fill } else { st.min_fill.min(fill) };
        st.searched += 1;
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
/// 建局与随机流走 [`seeded_rngs`]，与 `ramen_space_bench` 同一条路径。
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
    Ok((out.best_action_idx, ranked))
}

fn main() -> Result<()> {
    let args = RootArgs::parse();
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    let mut game_config = load_game_config()?;
    // 与 `ramen_space_bench` 同一前提：Y3 地区必须交回策略，否则测的不是同一个分布
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    init_global_with_config(&game_config)?;

    if let Some(w) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(w)
            .build_global()
            .map_err(|e| anyhow!("设置 rayon worker 数失败: {e}"))?;
    }

    let space = space_from_cli(None, &[])?;
    let plans = space.plans();
    let plan = plans
        .get(args.plan_index)
        .ok_or_else(|| anyhow!("计划下标 {} 越界（共 {} 个）", args.plan_index, plans.len()))?;
    let inherit = gen1_inherit();
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
        "  计划身份    plan{} uma={} deck={:?} shape={} combo_key={}",
        args.plan_index,
        plan.uma,
        plan.deck,
        plan.shape,
        plan.combo_key()
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
            (Some(s), _) => t.with_decision_sink(Arc::clone(s) as Arc<dyn DecisionSink>),
            (None, Some(s)) => t.with_decision_sink(Arc::clone(s) as Arc<dyn DecisionSink>),
            (None, None) => t
        }
    };
    let search = FlatSearch::<RamenGame>::new(config.clone())
        .with_rollout_trainer(rollout_trainer)
        .with_strict_rollout(true);

    // 这两种模式自己管侧车与建局，不走固定根的三种粒度
    match args.mode {
        Mode::SidecarReuse => return run_sidecar_reuse(&args, &search, &nn, plan.uma, &plan.deck, &inherit),
        Mode::GameSmoke => return run_game_smoke(&args, &search, &nn, plan.uma, &plan.deck, &inherit),
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

    // 统一口径：模型就绪之后，从建轨迹计到官方汇总完成。两种模式同一个窗口。
    let measure_start = Instant::now();
    let (cells, rollout_rows, wave_decisions, wave) = match sidecar.as_mut() {
        Some(sc) => {
            let (cells, rows, decisions, _table, st) = run_gpu_wave(
                &root,
                &actions,
                args.search_n,
                &seeds,
                &nn,
                sc,
                args.batch,
                args.decision_csv.is_some()
            )?;
            (cells, rows, decisions, Some(st))
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
        println!("  波次        {}", st.waves);
        println!(
            "  服务请求    {}（格位 {}，利用率 {:.1}%）",
            st.served,
            st.slots,
            100.0 * st.served as f64 / st.slots as f64
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
        writeln!(f, "candidate,j,requests")?;
        for r in &rollout_rows {
            writeln!(f, "{},{},{}", r.candidate, r.j, r.requests)?;
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
            "root,candidate,j,seq,turn,stage,n_actions,rng_probe,feat_hash,cpu_winner,gpu_winner,cpu_margin,action_delta,gap_us"
        )?;
        for r in rows.iter() {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:#018x},{:#018x},{},{},{:.6e},{:.6e},{:.3}",
                root_id,
                r.ctx.candidate,
                r.ctx.j,
                r.ctx.seq,
                r.turn,
                r.stage,
                r.n_actions,
                r.rng_probe,
                r.feat_hash,
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
