//! 客户端配置下的整局 MCTS + 手写续跑基准（**只测量，不改搜索**）
//!
//! ## 用途
//!
//! 按 `umaai` 主程序（`crates/umaai/src/main.rs`）**同一份合并配置与同一种训练员装配**
//! 跑若干局完整拉面杯，产出固定基线：
//!
//! - 配置：`load_game_config()`（`gamedata/default_config.toml` + `game_config.toml` 合并）
//!   → `SearchConfig::new_game_config`，阶段门控走 `mcts.ramen_search_stages`。
//! - 线程：`rayon` 全局池按 `collector.threads`（可被 `[config_override] num_threads` 覆盖，
//!   或本工具 `--threads` 再覆盖）。
//! - 决策：由 `--trainer` 显式分派（见下），**标签与实际执行分支恒一致**。
//! - 整局入口：`umasim::bench::run_seeded`，与既有 bench 系列同一条世界派生协议。
//!
//! ## 策略分派 `--trainer`
//!
//! | 取值 | 实际执行 | 备注 |
//! |---|---|---|
//! | `mcts` | `RamenMctsTrainer`（`RamenSelection::Score`、原激进度计算） | 搜索内部手写续跑，不加载网络 |
//! | `nn` | `RamenNnTrainer`（需 `--model`，需 feature `onnx`） | 直接决策，不跑搜索 |
//! | `mcts+region_nn` | `RegionNnTrainer`（需 `--model`，需 feature `onnx`） | 搜索同 `mcts`，**只有外层实际对局的 `RegionSelect`** 换成网络 |
//!
//! 未知取值直接报错，**不会**出现「打印 nn 却跑 mcts」。`nn` 只在
//! `--features onnx` 的构建里可用；未开 feature 时同样报错而不是静默回退。
//!
//! ### `mcts+region_nn` 的隔离边界
//!
//! 唯一的策略差异是**外层实际对局**的三次 `RegionSelect`（turn 2 / 23 / 47）。
//! `Train` / `RamenSelect` 走与 `mcts` 完全相同的搜索；`SpecialSelect` 合并缓存、
//! 事件选项、自选比赛守门全部转发同一个 [`RamenMctsTrainer`] 实例；
//! **搜索内部模拟的地区选择仍是手写**（rollout 基策由 `RamenMctsTrainer` 自持，本壳不碰）。
//! 地区阶段候选数恒 >1（第 1/2 年 10，第 3 年 `all` 下 120），因此
//! **每局推理数恰好 = 3**——这就是「搜索内部没有推理」的直接证据。
//!
//! ❗**决策逻辑与主程序同源**：`mcts+region_nn` 用的就是主程序
//! （`game_config.toml` 里 `ramen_region_policy = "nn"`）那一份共享实现
//! [`umaai::region::RegionNnTrainer`]，本工具只额外挂一个观测钩子
//! （[`RegionObsRecorder`]）记录 `regions.csv`。两边**不存在**两份会逐渐分叉的接管逻辑。
//! 区别只在模型来源：本工具走 `--model` 命令行参数（保留一次跑多个模型的能力），
//! 主程序走配置项 `ramen_region_model_path`。
//!
//! `regions.csv` 逐次记录实际地区选择；`mcts+region_nn` 还用一个**独立的**
//! `RecommendedRamenTrainer` 在同一局面上算出「手写本来会选什么」作为观测，
//! 该调用用 `rng` 的克隆，既不推进外层随机流，也不触碰参与决策的训练员状态。
//!
//! `nn` 模式下模型**每进程只加载一次**，逐局共享同一份 `Arc`；
//! 推理次数取 `umasim::trainer::infer_request_count()` 的逐局增量，
//! 推理失败经 `Result` 直接终止本次测量，不会静默换成手写。
//!
//! ## ❗与 `ramen_space_bench` 的区别
//!
//! `ramen_space_bench` 的建局走 `SamplingSpace` + `gen1_inherit()`，马娘/卡组/因子
//! 由采样面板决定，**不是**用户 `game_config.toml` 的配置；它还会自动计算模型、源码、
//! 配置与游戏数据的 FNV 指纹。本工具两者都不做：建局直接取合并配置的
//! `uma` / `cards` / `blue_count` / `extra_count`，不计算任何哈希或指纹。
//!
//! ## 世界派生
//!
//! 第 `i` 局的世界由 `bench::seeded_rngs(base_seed, run_idx)` 决定，
//! `run_idx ∈ [run_offset, run_offset + runs)`。马娘、卡组、因子固定，只有世界变。
//! 同一 `(base_seed, run_idx)` 在别的工具里也给出同一个世界，可做后续配对比较。
//!
//! ## ❗`--hard-timeout-secs` 是硬超时
//!
//! 由独立看门狗线程实现：到点直接 `std::process::exit`，**会**中断进行中的对局。
//! 逐局 CSV 每局落盘并 flush，被中断时已完成的局仍在文件里；未完成的局不会补记。
//! 看门狗在**解析完参数后立刻起表**，即硬超时覆盖读配置、初始化全局数据与
//! **模型加载**在内的整个进程；模型加载耗时另行单独打印，不与逐局耗时混在一起。

//!
//! ## 配置覆盖
//!
//! `--cards` / `--extra-count` / `--threads` / `--search-n` / `--trainer` 只改**内存里的
//! 合并配置**，绝不写回 `game_config.toml`；生效值与文件值都会打印出来供人工核对。
//! `--search-n` 在 `SearchConfig::new_game_config` **之前**改 `mcts.search_n`，
//! 走的是与客户端逐字相同的那条装配路径；它是 UCB 停止阈值，**不是**整次搜索总预算。
//!
//! ## 用法
//!
//! ```text
//! cargo run --release --bin ramen_client_game_bench -- \
//!   --runs 10 --seed 61444 --run-offset 140000 --out logs/xxx \
//!   --cards 303124,303114,303044,303004,302894,303054 \
//!   --extra-count 0,20,20,40,40,40 --threads 16 --search-n 1024 --trainer mcts
//!
//! cargo run --release --features onnx --bin ramen_client_game_bench -- \
//!   --runs 10 --seed 61444 --run-offset 140000 --out logs/yyy \
//!   --cards 303124,303114,303044,303004,302894,303054 \
//!   --extra-count 0,20,20,40,40,40 --threads 16 --search-n 1024 \
//!   --trainer mcts+region_nn --model saved_models/arms/ens_R4_g123.onnx --special-mode canonical
//! ```
//!
//! 配对跑两臂用受版本控制的驱动 `scripts/bench/run_paired_game_bench.ps1`
//! （共享硬预算、逐臂留存退出码；不带参数只打印用法）。

use std::{
    env,
    fs,
    io::{self, Write},
    path::PathBuf,
    process,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant}
};

use anyhow::{Context, Result, anyhow, bail};
use rayon::{ThreadPoolBuilder, current_num_threads};
use serde::Deserialize;
use umasim::{
    bench,
    game::InheritInfo,
    gamedata::{GameConfig, init_global_with_config, ramen::RAMENDATA},
    search::{SearchConfig, SearchProbe},
    trainer::{LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RecommendedRamenTrainer},
    utils::{Array6, get_workspace_root, init_logger_stdout, load_game_config}
};
#[cfg(feature = "onnx")]
use rand::prelude::StdRng;
use umaai::region::check_region_nn_applicable;
#[cfg(feature = "onnx")]
use umaai::region::{RegionDecisionObserver, RegionNnTrainer};
#[cfg(feature = "onnx")]
use umasim::{
    game::{
        Game,
        Trainer,
        ramen::{RamenAction, RamenGame, RamenState}
    },
    trainer::{RamenNnTrainer, SpecialSelectMode, infer_request_count}
};

/// 本进程实际执行的决策策略
///
/// 与 `--trainer` 的取值一一对应：**标签就是执行分支**，不存在打印一套、跑另一套。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    /// `RamenMctsTrainer`：FlatSearch 搜索 + 手写续跑，不加载网络
    Mcts,
    /// `RamenNnTrainer`：ONNX 网络直接决策，不跑搜索
    Nn,
    /// `RamenMctsTrainer` + **仅外层实际对局的 `RegionSelect`** 交给 ONNX 网络
    ///
    /// 其余一切（Train / RamenSelect 搜索、SpecialSelect 缓存、事件选项、
    /// **搜索内部模拟的地区选择**）与 [`Policy::Mcts`] 完全相同。
    MctsRegionNn,
    /// `RecommendedRamenTrainer`：纯手写推荐策略，不跑搜索也不加载网络
    ///
    /// 它就是 [`Policy::Mcts`] 的 rollout 基策与地区 fallback 本体。存在的理由是
    /// **可行性筛查**：整局只有手写决策，一局几十毫秒，可以在正式实验前把
    /// 「这个马娘能不能完整育成」跑出来，而不必花 20 s/局的搜索预算。
    /// 它**不是**任何一臂的实验策略。
    Handwritten
}

impl Policy {
    /// 命令行标签（与 `--trainer` 取值、`game_config.trainer` 的登记值一致）
    fn label(self) -> &'static str {
        match self {
            Self::Mcts => "mcts",
            Self::Nn => "nn",
            Self::MctsRegionNn => "mcts+region_nn",
            Self::Handwritten => "handwritten"
        }
    }

    /// 一行说明实际执行分支，用于生效配置打印
    fn describe(self) -> &'static str {
        match self {
            Self::Mcts => "RamenMctsTrainer（FlatSearch 搜索，续跑用手写策略，不加载 NN）",
            Self::Nn => "RamenNnTrainer（ONNX 网络直接决策，不跑搜索）",
            Self::MctsRegionNn => {
                "RamenMctsTrainer + 仅外层 RegionSelect 由 RamenNnTrainer 决策（搜索内部地区仍手写）"
            }
            Self::Handwritten => "RecommendedRamenTrainer（纯手写推荐策略，不跑搜索、不加载 NN）"
        }
    }

    /// 本策略是否需要 `--model`
    fn needs_model(self) -> bool {
        matches!(self, Self::Nn | Self::MctsRegionNn)
    }

    /// 本策略是否跑 FlatSearch 搜索（决定要不要打印搜索字段）
    fn runs_search(self) -> bool {
        matches!(self, Self::Mcts | Self::MctsRegionNn)
    }
}

/// 按臂检查**生效配置**是否适用（地区 NN 臂与客户端共用同一套判断）
///
/// review#3：`mcts+region_nn` 此前直接构造 `RegionNnTrainer`，绕开了客户端的
/// `fixed` / 地区搜索开关冲突检查。结果是同一份 `game_config.toml`，客户端启动即
/// 拒绝、benchmark 却照跑，还会按「每局恒 3 次推理」的前提去读结果——而
/// `ramen_region_strategy = "fixed"` 下第 3 年只有 1 个候选，实际只会推理 2 次。
///
/// 只有 `mcts+region_nn` 受地区专用限制：`mcts` / `nn` / `handwritten` 三臂
/// 与地区接管无关，一律放行。
///
/// 模型路径**不在**这里检查：benchmark 的模型来自 `--model`（保留一次跑多个模型的
/// 能力），不要求用户去配 `ramen_region_model_path`；这正是本函数调
/// [`check_region_nn_applicable`] 而不是 `validate_region_policy` 的原因。
///
/// ❗`cfg` / `stages` 必须是**命令行覆盖之后**的生效值。
///
/// # 错误
///
/// 地区 NN 臂下配置不适用时返回带修复建议的错误。
fn check_arm_config(policy: Policy, cfg: &GameConfig, stages: RamenSearchStages) -> Result<()> {
    match policy {
        Policy::MctsRegionNn => check_region_nn_applicable(cfg, stages),
        Policy::Mcts | Policy::Nn | Policy::Handwritten => Ok(())
    }
}

/// 解析 `--trainer`
///
/// # 错误
///
/// 取值不是 `mcts` / `nn` / `mcts+region_nn` / `handwritten` 时报错——**不做任何默认回退**。
fn parse_policy(s: &str) -> Result<Policy> {
    match s {
        "mcts" => Ok(Policy::Mcts),
        "nn" => Ok(Policy::Nn),
        "mcts+region_nn" => Ok(Policy::MctsRegionNn),
        "handwritten" => Ok(Policy::Handwritten),
        other => bail!("未知 --trainer {other:?}（可选 mcts / nn / mcts+region_nn / handwritten）")
    }
}

/// 命令行参数
struct Args {
    /// 对局数
    runs: u64,
    /// 基种子
    seed: u64,
    /// 局号起点；本次跑 `run_idx ∈ [run_offset, run_offset + runs)`
    run_offset: u64,
    /// 证据目录（相对 workspace 根或绝对路径）
    out: PathBuf,
    /// 覆盖合并配置的 `extra_count`（种马额外属性，6 项）
    extra_count: Option<Array6>,
    /// 覆盖合并配置的 `cards`（6 张支援卡 idrank，末位友人卡）
    cards: Option<[u32; 6]>,
    /// 覆盖合并配置的 `uma`（马娘 gameId）
    uma: Option<u32>,
    /// 计划表文件：一行一个「马娘 + 六卡组 + 世界局号」，逐条各跑一局
    plans_file: Option<PathBuf>,
    /// 只跑计划表的 `[start, end)` 段（分块交替两臂顺序用）
    plan_range: Option<(usize, usize)>,
    /// 覆盖合并配置的线程数
    threads: Option<usize>,
    /// 覆盖合并配置的 `mcts.search_n`（UCB 停止阈值，**不是**整次搜索总预算）
    search_n: Option<usize>,
    /// 本次实际执行的策略（`--trainer`），同时覆盖登记用的 `game_config.trainer`
    policy: Policy,
    /// `--trainer nn` 的 ONNX 模型路径
    model: Option<PathBuf>,
    /// `--trainer nn` 的 `SpecialSelect` 推理口径（`canonical` / `raw` / `handwritten`）
    special_mode: String,
    /// 硬超时（秒）：到点直接终止进程
    hard_timeout_secs: u64
}

impl Default for Args {
    fn default() -> Self {
        Self {
            runs: 10,
            seed: 61444,
            run_offset: 140_000,
            out: PathBuf::from("logs/client_game_bench"),
            extra_count: None,
            cards: None,
            uma: None,
            plans_file: None,
            plan_range: None,
            threads: None,
            search_n: None,
            policy: Policy::Mcts,
            model: None,
            special_mode: "canonical".to_string(),
            hard_timeout_secs: 5400
        }
    }
}

/// 解析 `a,b,c,d,e,f` 为 [`Array6`]
fn parse_array6(text: &str) -> Result<Array6> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    if parts.len() != 6 {
        bail!("extra_count 需要 6 项，实际 {} 项：{text:?}", parts.len());
    }
    let mut out = Array6::default();
    for (slot, p) in out.iter_mut().zip(parts) {
        *slot = p.parse().with_context(|| format!("extra_count 项不是整数：{p:?}"))?;
    }
    Ok(out)
}

/// 解析 `a,b,c,d,e,f` 为 6 张支援卡的 idrank（6 位 = 5 位卡 ID + 1 位突破等级）
///
/// 只做形状与整数解析；卡是否存在、类型是否符合构成由 `gamedata` 与实验设计负责。
///
/// # 错误
///
/// 项数不是 6，或某项不是 `u32` 时报错。
fn parse_cards(text: &str) -> Result<[u32; 6]> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    if parts.len() != 6 {
        bail!("cards 需要 6 项，实际 {} 项：{text:?}", parts.len());
    }
    let mut out = [0u32; 6];
    for (slot, p) in out.iter_mut().zip(parts) {
        *slot = p.parse().with_context(|| format!("cards 项不是 u32：{p:?}"))?;
    }
    Ok(out)
}

/// 解析命令行
fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].clone();
        let val = raw.get(i + 1).cloned();
        let need = || val.clone().ok_or_else(|| anyhow!("{key} 需要取值"));
        match key.as_str() {
            "--runs" => {
                args.runs = need()?.parse()?;
                i += 2;
            }
            "--seed" => {
                args.seed = need()?.parse()?;
                i += 2;
            }
            "--run-offset" => {
                args.run_offset = need()?.parse()?;
                i += 2;
            }
            "--out" => {
                args.out = PathBuf::from(need()?);
                i += 2;
            }
            "--extra-count" => {
                args.extra_count = Some(parse_array6(&need()?)?);
                i += 2;
            }
            "--cards" => {
                args.cards = Some(parse_cards(&need()?)?);
                i += 2;
            }
            "--uma" => {
                args.uma = Some(need()?.parse()?);
                i += 2;
            }
            "--plans-file" => {
                args.plans_file = Some(PathBuf::from(need()?));
                i += 2;
            }
            "--plan-range" => {
                let start: usize = need()?.parse()?;
                let end: usize = raw
                    .get(i + 2)
                    .ok_or_else(|| anyhow!("--plan-range 需要两个取值 <start> <end>"))?
                    .parse()?;
                if start >= end {
                    bail!("--plan-range 需要 start < end，实得 {start} {end}");
                }
                args.plan_range = Some((start, end));
                i += 3;
            }
            "--threads" => {
                args.threads = Some(need()?.parse()?);
                i += 2;
            }
            "--search-n" => {
                args.search_n = Some(need()?.parse()?);
                i += 2;
            }
            "--trainer" => {
                args.policy = parse_policy(&need()?)?;
                i += 2;
            }
            "--model" => {
                args.model = Some(PathBuf::from(need()?));
                i += 2;
            }
            "--special-mode" => {
                args.special_mode = need()?;
                i += 2;
            }
            "--hard-timeout-secs" => {
                args.hard_timeout_secs = need()?.parse()?;
                i += 2;
            }
            other => bail!("未知参数 {other:?}")
        }
    }
    if args.runs == 0 {
        bail!("--runs 必须 > 0");
    }
    // 策略与其必需 / 禁止的参数：不一致就报错，避免「给了模型却跑搜索」这类静默错配
    if args.policy.needs_model() && args.model.is_none() {
        bail!("--trainer {} 需要同时给出 --model <onnx 路径>", args.policy.label());
    }
    if !args.policy.needs_model() && args.model.is_some() {
        bail!("--trainer {} 不加载任何网络，不应给 --model", args.policy.label());
    }
    if args.search_n.is_some() && !args.policy.runs_search() {
        bail!("--trainer {} 不跑搜索，不应给 --search-n", args.policy.label());
    }
    if args.search_n == Some(0) {
        bail!("--search-n 必须 > 0");
    }
    // 计划表模式与「单一固定组合 + 连续局号」模式互斥：两者都给会出现
    // 「打印一套、跑另一套」的隐性错配，直接拒绝。
    if args.plans_file.is_some() {
        for (flag, given) in [
            ("--cards", args.cards.is_some()),
            ("--uma", args.uma.is_some()),
            ("--runs", raw.iter().any(|a| a == "--runs")),
            ("--run-offset", raw.iter().any(|a| a == "--run-offset"))
        ] {
            if given {
                bail!("--plans-file 已给出每条计划的马娘/卡组/局号，不应再给 {flag}");
            }
        }
    } else if args.plan_range.is_some() {
        bail!("--plan-range 只在 --plans-file 模式下有意义");
    }
    Ok(args)
}

/// 逐局成本与结果记录
struct GameRecord {
    /// 计划编号（单一组合模式下即第几局，从 0 起）
    plan_id: usize,
    /// 本局马娘 gameId
    uma: u32,
    /// 本局六张卡（`a|b|c|d|e|f`）
    deck: String,
    /// 局号（世界标识的一半）
    run_idx: u64,
    /// 规则主种子（`bench::seeded_rngs` 派生，世界标识的另一半）
    rule_master: u64,
    /// 终局评分
    score: i32,
    /// 评分等级
    rank: String,
    /// 五维终值
    five_status: [i32; 5],
    /// 技能点
    skill_pt: i32,
    /// 整局墙钟（秒，本工具计时）
    wall_s: f64,
    /// 搜索次数（`SearchProbe` 条数）
    searches: usize,
    /// 搜索内核累计耗时（秒）
    search_s: f64,
    /// 计划续跑数
    planned: usize,
    /// 成功续跑数
    succeeded: usize,
    /// 失败续跑数
    failed: usize,
    /// 本局网络推理请求数（`mcts` 模式恒为 0）
    infers: u64,
    /// 自选比赛是否全部达标
    free_race_ok: bool,
    /// RMJ 成功年数
    rmj_ok: usize,
    /// 五次友人出行是否全部完成
    friend_all: bool
}

/// 已加载模型的槽位；未开 `onnx` feature 时没有可放的东西，恒为 `None`
#[cfg(feature = "onnx")]
type NnSlot = Option<RamenNnTrainer>;
/// 已加载模型的槽位；未开 `onnx` feature 时没有可放的东西，恒为 `None`
#[cfg(not(feature = "onnx"))]
type NnSlot = Option<()>;

/// NN 直接决策跑一局，并记录本局推理请求数
///
/// 模型由调用方加载一次后共享，本函数不重载模型。推理失败经 `Result` 向上传播，
/// **不会**静默退化成手写策略。
///
/// # 错误
///
/// 模型未加载、整局模拟报错（含网络推理失败）时报错。
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
fn nn_run(
    nn: &NnSlot, uma: u32, cards: &[u32; 6], inherit: &InheritInfo, seed: u64, run_idx: u64,
    rule_master: u64, cost: &mut CostStat
) -> Result<bench::GameOutcome> {
    let t = nn.as_ref().ok_or_else(|| anyhow!("NN 模式但模型未加载"))?;
    // 进程内累计推理数取逐局增量
    let before = infer_request_count();
    let mut trainer = LoggingTrainer::new(t.clone(), rule_master);
    trainer.set_logging(false);
    let out = bench::run_seeded(uma, cards, inherit, seed, run_idx, &trainer)?;
    cost.infers = infer_request_count() - before;
    Ok(out)
}

/// 未开 `onnx` feature 时的占位：直接报错
///
/// # 错误
///
/// 恒报错——该构建里没有任何网络策略可用。
#[cfg(not(feature = "onnx"))]
#[allow(clippy::too_many_arguments)]
fn nn_run(
    _nn: &NnSlot, _uma: u32, _cards: &[u32; 6], _inherit: &InheritInfo, _seed: u64, _run_idx: u64,
    _rule_master: u64, _cost: &mut CostStat
) -> Result<bench::GameOutcome> {
    bail!("--trainer nn 需要编译 feature onnx")
}

/// 「搜索 + 仅外层地区用网络」跑一局，并记录本局推理请求数与地区决策
///
/// 模型由调用方加载一次后共享；`mcts` 由调用方**逐局新建**（与纯 `mcts` 臂同样每局一个
/// 干净实例与独立探针）。推理失败经 `Result` 向上传播，**不会**静默退化成手写。
///
/// # 错误
///
/// 模型未加载、整局模拟报错（含网络推理失败、地区候选落格失败）时报错。
#[cfg(feature = "onnx")]
#[allow(clippy::too_many_arguments)]
fn region_nn_run(
    nn: &NnSlot, mcts: RamenMctsTrainer, uma: u32, cards: &[u32; 6], inherit: &InheritInfo, seed: u64,
    run_idx: u64, rule_master: u64, cost: &mut CostStat
) -> Result<RegionNnRun> {
    let t = nn.as_ref().ok_or_else(|| anyhow!("地区 NN 模式但模型未加载"))?;
    let recorder = Arc::new(RegionObsRecorder::new());
    // 决策逻辑走共享实现（主程序用的是同一份），benchmark 只额外挂观测钩子
    let hybrid = RegionNnTrainer::new(mcts, t.clone()).with_observer(Arc::clone(&recorder) as Arc<_>);
    let before = infer_request_count();
    let mut trainer = LoggingTrainer::new(hybrid, rule_master);
    trainer.set_logging(false);
    let out = bench::run_seeded(uma, cards, inherit, seed, run_idx, &trainer)?;
    cost.infers = infer_request_count() - before;
    // `trainer` 仍持有另一份 Arc，这里按值复制取出（`RegionObs` 是 `Copy`）
    let recs = recorder.snapshot()?;
    Ok(RegionNnRun {
        outcome: out,
        obs: recs
    })
}

/// 未开 `onnx` feature 时的占位：直接报错
///
/// # 错误
///
/// 恒报错——该构建里没有任何网络策略可用。
#[cfg(not(feature = "onnx"))]
#[allow(clippy::too_many_arguments)]
fn region_nn_run(
    _nn: &NnSlot, _mcts: RamenMctsTrainer, _uma: u32, _cards: &[u32; 6], _inherit: &InheritInfo,
    _seed: u64, _run_idx: u64, _rule_master: u64, _cost: &mut CostStat
) -> Result<RegionNnRun> {
    bail!("--trainer mcts+region_nn 需要编译 feature onnx")
}

/// 「搜索 + 仅外层地区用网络」一局的产物
///
/// 建新类型而非 `(GameOutcome, Vec<RegionObs>)`：两项都是本局结果，
/// 元组位置写反不会编译报错。
struct RegionNnRun {
    /// 整局终局结果
    outcome: bench::GameOutcome,
    /// 本局三次实际地区决策的观测（含同局面手写反事实）
    obs: Vec<RegionObs>
}

/// 单局成本统计（搜索侧与网络侧各占一半，另一半留 0）
#[derive(Default)]
struct CostStat {
    /// 搜索次数
    searches: usize,
    /// 搜索内核累计耗时（秒）
    search_s: f64,
    /// 计划续跑数
    planned: usize,
    /// 成功续跑数
    succeeded: usize,
    /// 失败续跑数
    failed: usize,
    /// 网络推理请求数
    infers: u64
}

/// 计划表文件里的一条计划（**外部输入**，字段名即 JSON 键名）
///
/// 一条计划 = 一个「马娘 + 六卡组 + 世界局号」，两臂逐字段共用同一份文件。
#[derive(Debug, Clone, Deserialize)]
struct PlanEntry {
    /// 计划编号，必须与数组下标逐值相同（防止删行后编号与位置错位）
    plan_id: usize,
    /// 马娘 gameId
    uma: u32,
    /// 六张支援卡 idrank（末位必须是友人卡，由建局自己校验）
    cards: [u32; 6],
    /// 世界局号，与 `--seed` 一起经 `bench::seeded_rngs` 派生世界
    run_idx: u64
}

/// 计划表文件的整体结构
#[derive(Debug, Clone, Deserialize)]
struct PlanFile {
    /// 基种子；必须与 `--seed` 逐值相同，否则两臂可能读同一份计划却跑不同世界
    base_seed: u64,
    /// 全部计划
    plans: Vec<PlanEntry>
}

/// 一局的完整身份（预登记用）：计划 + 世界
///
/// 建新类型而非元组：字段同类型，位置写反不会编译报错。
#[derive(Clone)]
struct PlanSpec {
    /// 计划编号
    plan_id: usize,
    /// 马娘 gameId
    uma: u32,
    /// 六张支援卡 idrank
    cards: [u32; 6],
    /// 局号
    run_idx: u64,
    /// `bench::seeded_rngs` 派生的规则主种子
    rule_master: u64
}

impl PlanSpec {
    /// 卡组渲染成 `a|b|c|d|e|f`（CSV 单元格内不含逗号）
    fn deck_cell(&self) -> String {
        self.cards
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join("|")
    }
}

/// 读取计划表文件并校验
///
/// 校验项：`base_seed` 与 `--seed` 逐值相同、`plan_id` 与下标逐值相同、
/// `run_idx` 互不重复（同一个世界被两条计划占用会让配对分析对不上）。
/// **不算任何指纹**，全部是字段的直接比较。
///
/// # 错误
///
/// 文件读不了、JSON 结构不符、上述任一校验不过，或 `--plan-range` 越界时报错。
fn load_plans(path: &PathBuf, seed: u64, range: Option<(usize, usize)>) -> Result<Vec<PlanEntry>> {
    let text = fs::read_to_string(path).with_context(|| format!("读不了计划表: {}", path.display()))?;
    let file: PlanFile =
        serde_json::from_str(&text).with_context(|| format!("计划表 JSON 解析失败: {}", path.display()))?;
    if file.base_seed != seed {
        bail!("计划表 base_seed={} 与 --seed {seed} 不一致", file.base_seed);
    }
    if file.plans.is_empty() {
        bail!("计划表为空: {}", path.display());
    }
    let mut seen: Vec<u64> = Vec::with_capacity(file.plans.len());
    for (idx, plan) in file.plans.iter().enumerate() {
        if plan.plan_id != idx {
            bail!("计划表第 {idx} 条的 plan_id={} 与下标不符", plan.plan_id);
        }
        if seen.contains(&plan.run_idx) {
            bail!("计划表 run_idx={} 出现多次，世界会被两条计划占用", plan.run_idx);
        }
        seen.push(plan.run_idx);
    }
    let picked = match range {
        None => file.plans,
        Some((start, end)) => {
            if end > file.plans.len() {
                bail!("--plan-range {start} {end} 越界：计划表只有 {} 条", file.plans.len());
            }
            file.plans[start..end].to_vec()
        }
    };
    Ok(picked)
}

/// 逐局 CSV 表头
const CSV_HEADER: &str = "plan_id,uma,deck,run_idx,rule_master,score,rank,speed,stamina,power,guts,wisdom,skill_pt,wall_s,searches,search_s,planned,succeeded,failed,infers,free_race_ok,rmj_ok,friend_all";

impl GameRecord {
    /// 转一行 CSV（各字段均无逗号，直接拼接）
    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{:.3},{},{},{},{},{},{},{}",
            self.plan_id,
            self.uma,
            self.deck,
            self.run_idx,
            self.rule_master,
            self.score,
            self.rank,
            self.five_status[0],
            self.five_status[1],
            self.five_status[2],
            self.five_status[3],
            self.five_status[4],
            self.skill_pt,
            self.wall_s,
            self.searches,
            self.search_s,
            self.planned,
            self.succeeded,
            self.failed,
            self.infers,
            u8::from(self.free_race_ok),
            self.rmj_ok,
            u8::from(self.friend_all)
        )
    }
}

/// 一次**实际对局**的地区选择记录（只观测，不参与决策）
#[derive(Clone, Copy)]
struct RegionObs {
    /// 决策发生的回合（2 / 23 / 47）
    turn: i32,
    /// 该次选择作用的年份（1 / 2 / 3）
    ///
    /// ❗由 `turn` 经 [`RamenState::region_archive_year_idx`] 推出，
    /// **不是** `current_year()`：turn 23 选的是第 2 年，而 `current_year()` 仍是 1。
    year: usize,
    /// 候选数（第 1/2 年 C(5,3)=10；第 3 年 `all` 下 C(10,3)=120）
    ///
    /// 纯 `mcts` 臂从 `GameOutcome.yearly_selected_regions` 事后还原，拿不到候选数，为 `None`。
    candidates: Option<usize>,
    /// 实际执行的三个地区下标
    picked: [usize; 3],
    /// 同一局面下手写策略**本来会选**的三个地区下标（仅 NN 臂有值）
    hand: Option<[usize; 3]>
}

/// 记录一次实际地区决策的旁观器（只观测，不参与决策）
///
/// 决策逻辑本身在共享实现 [`umaai::region::RegionNnTrainer`] 里，本结构只挂钩子，
/// 因此 benchmark 与主程序**不会**各自跑一份会逐渐分叉的地区接管逻辑。
///
/// `hand_ref` 是一个**独立的** [`RecommendedRamenTrainer`]，只用来观测「同一局面下
/// 手写会选什么」：它用共享实现传入的**推理前**随机流快照调用，既不推进外层随机流，
/// 也不触碰参与决策的搜索训练员的任何内部状态。
#[cfg(feature = "onnx")]
struct RegionObsRecorder {
    /// 观测用手写参照，与 [`RamenMctsTrainer`] 的 fallback 同为 `RecommendedRamenTrainer::new()`
    hand_ref: RecommendedRamenTrainer,
    /// 本局的地区决策记录
    obs: Mutex<Vec<RegionObs>>
}

#[cfg(feature = "onnx")]
impl RegionObsRecorder {
    /// 新建一个空记录器
    fn new() -> Self {
        Self {
            hand_ref: RecommendedRamenTrainer::new(),
            obs: Mutex::new(Vec::new())
        }
    }

    /// 按值取出已记录的观测（`RegionObs` 是 `Copy`）
    ///
    /// # 错误
    ///
    /// 记录锁被毒化时报错。
    fn snapshot(&self) -> Result<Vec<RegionObs>> {
        Ok(self.obs.lock().map_err(|_| anyhow!("地区观测锁被毒化"))?.clone())
    }
}

#[cfg(feature = "onnx")]
impl RegionDecisionObserver for RegionObsRecorder {
    /// 记录本次实际选择，并算出同局面下手写本来会选什么
    ///
    /// # 错误
    ///
    /// 回合不是地区回合、候选落格失败、手写参照报错或记录锁被毒化时报错。
    fn on_region_decided(
        &self, game: &RamenGame, actions: &[RamenAction], picked: usize, mut rng_before: StdRng
    ) -> Result<()> {
        let turn = game.turn() as i32;
        let year_idx = RamenState::region_archive_year_idx(turn)?;
        // 手写参照跑在**推理前**的随机流快照上：与「不接管时手写会看到的状态」一致
        let hand_idx = self.hand_ref.select_action(game, actions, &mut rng_before)?;
        let rec = RegionObs {
            turn,
            year: year_idx + 1,
            candidates: Some(actions.len()),
            picked: RegionNnTrainer::region_of(actions, picked)?,
            hand: Some(RegionNnTrainer::region_of(actions, hand_idx)?)
        };
        self.obs
            .lock()
            .map_err(|_| anyhow!("地区观测锁被毒化"))?
            .push(rec);
        Ok(())
    }
}

/// 地区决策记录 CSV 表头
const REGION_CSV_HEADER: &str =
    "plan_id,run_idx,turn,year,source,candidates,picked,picked_names,hand,hand_names,disagree";

/// 把三个地区下标渲染成名称串（取自 `RAMENDATA.ramen_region_effect`）
///
/// # 错误
///
/// 全局拉面数据未初始化，或某个下标越界时报错。
fn region_names(regions: [usize; 3]) -> Result<String> {
    let data = RAMENDATA.get().ok_or_else(|| anyhow!("RAMENDATA 未初始化"))?;
    let mut out = Vec::with_capacity(3);
    for idx in regions {
        let r = data
            .ramen_region_effect
            .get(idx)
            .ok_or_else(|| anyhow!("地区下标 {idx} 越界"))?;
        out.push(r.name.clone());
    }
    Ok(out.join("|"))
}

/// 渲染一行地区记录 CSV
///
/// # 错误
///
/// 地区名称查表失败时报错。
fn region_csv_line(plan_id: usize, run_idx: u64, source: &str, obs: &RegionObs) -> Result<String> {
    let picked = format!("{}|{}|{}", obs.picked[0], obs.picked[1], obs.picked[2]);
    let (hand, hand_names, disagree) = match obs.hand {
        Some(h) => (
            format!("{}|{}|{}", h[0], h[1], h[2]),
            region_names(h)?,
            if h == obs.picked { "0" } else { "1" }.to_string()
        ),
        None => (String::new(), String::new(), String::new())
    };
    let cand = obs.candidates.map(|n| n.to_string()).unwrap_or_default();
    Ok(format!(
        "{plan_id},{run_idx},{},{},{source},{cand},{picked},{},{hand},{hand_names},{disagree}",
        obs.turn,
        obs.year,
        region_names(obs.picked)?
    ))
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // 与 ramen_client_cost / ramen_manual 同约束：cwd 必须是 workspace 根
    env::set_current_dir(get_workspace_root()?)?;

    // ---- 硬超时看门狗：解析完参数立刻起表 ----
    // ❗必须在读配置、初始化全局数据、**加载模型**之前启动：硬超时要覆盖整个进程，
    // 否则模型加载卡住就永远不会被计时。到点直接终止进程（逐局 CSV 已每局落盘）。
    let deadline = Duration::from_secs(args.hard_timeout_secs);
    thread::spawn(move || {
        thread::sleep(deadline);
        eprintln!(
            "❗硬超时 {}s 到达，终止进程（已完成的局保留在逐局 CSV 中）",
            deadline.as_secs()
        );
        let _ = io::stderr().flush();
        process::exit(99);
    });

    // ---- 与 main.rs 同序：先读配置，再初始化日志 / 全局数据 / 线程池 ----
    let mut game_config = load_game_config()?;
    // 本次任务的显式覆盖：只改内存里的合并配置，不写用户的 game_config.toml
    let extra_before = game_config.extra_count;
    let cards_before = game_config.cards;
    let threads_before = game_config.collector.threads;
    let trainer_before = game_config.trainer.clone();
    if let Some(v) = args.extra_count {
        game_config.extra_count = v;
    }
    if let Some(v) = args.cards {
        game_config.cards = v;
    }
    let uma_before = game_config.uma;
    if let Some(v) = args.uma {
        game_config.uma = v;
    }
    if let Some(v) = args.threads {
        game_config.collector.threads = v;
    }
    // 覆盖发生在 SearchConfig::new_game_config 之前，走的是与客户端完全相同的那条装配路径
    let search_n_before = game_config.mcts.search_n;
    if let Some(v) = args.search_n {
        game_config.mcts.search_n = v;
    }
    // 登记用的 trainer 字段恒等于实际执行分支；与文件不一致时显式提示，
    // 不静默改写用户的 game_config.toml
    let want = args.policy.label();
    if game_config.trainer != want {
        println!(
            "❗trainer 覆盖：配置文件为 {trainer_before:?}，本次按 {want:?} 运行（{}）",
            args.policy.describe()
        );
    }
    game_config.trainer = want.to_string();

    let mcts_config = SearchConfig::new_game_config(&game_config);
    init_logger_stdout("ramen_client_game_bench", &game_config.log_level)?;
    init_global_with_config(&game_config)?;
    ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;

    let ramen_stages = RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?;
    // ❗地区 NN 臂与客户端共用同一套适用配置检查；此处 game_config / ramen_stages
    // 已吃过全部命令行覆盖，校验的是**实际生效值**而不是文件里的原值。
    check_arm_config(args.policy, &game_config, ramen_stages)?;

    // ---- 打印生效配置，供人工核对 ----
    println!("=== 生效配置（合并后 + 本次覆盖）===");
    println!(
        "uma={}（文件值 {uma_before}） cards={:?}（文件值 {cards_before:?}） blue_count={:?} extra_count={:?}（文件值 {extra_before:?}）",
        game_config.uma, game_config.cards, game_config.blue_count, game_config.extra_count
    );
    if args.plans_file.is_some() {
        println!(
            "❗上一行的 uma / cards 是**合并配置**里的值，本次不参与建局：             计划表逐条给出马娘与卡组，实际用的是下面「预登记计划与世界」里逐条打印的那一份"
        );
    }
    println!(
        "trainer={:?}（文件值 {trainer_before:?}；实际执行 {}）",
        game_config.trainer,
        args.policy.describe()
    );
    println!(
        "num_threads(配置)={}（文件值 {threads_before}） rayon 实际线程={}",
        game_config.collector.threads,
        current_num_threads()
    );
    if args.policy.runs_search() {
        {
            println!(
                "search_n={}（文件值 {search_n_before}） use_ucb={} search_group_size={} search_cpuct={} expected_search_stdev={}",
                mcts_config.search_n,
                mcts_config.use_ucb,
                mcts_config.search_group_size,
                mcts_config.search_cpuct,
                mcts_config.expected_search_stdev
            );
            println!(
                "radical_factor_max={} max_depth={} policy_delta={} rollout_evaluator={:?} crn_stage_reseed={}",
                mcts_config.radical_factor_max,
                mcts_config.max_depth,
                mcts_config.policy_delta,
                game_config.mcts.rollout_evaluator,
                game_config.mcts.crn_stage_reseed
            );
            println!(
                "ramen_search_stages={:?} → {ramen_stages:?}",
                game_config.mcts.ramen_search_stages
            );
        }
        if args.policy == Policy::MctsRegionNn {
            println!(
                "❗本次只把**外层实际对局**的 RegionSelect（turn 2 / 23 / 47）交给网络；\
                 Train / RamenSelect 仍是上面那套搜索，**搜索内部模拟的地区选择仍为手写**"
            );
            println!(
                "地区阶段候选数：第 1/2 年 C(5,3)=10，第 3 年 ramen_region_strategy=all 下 C(10,3)=120；\
                 均 >1，故每局恰好 3 次地区推理，整局推理数 = 3 即为「无搜索内部推理」的直接证据"
            );
        }
    } else if args.policy == Policy::Handwritten {
        println!(
            "❗本次不跑搜索、不加载网络：整局全部决策由 RecommendedRamenTrainer（手写推荐策略）做出。             本模式只用于可行性筛查，不是任何一臂的实验策略"
        );
    } else {
        // 搜索一次都不跑，上面那些搜索字段本次全部不参与决策，避免误读
        println!(
            "❗本次不跑搜索：search_n / use_ucb / search_group_size / ramen_search_stages 等搜索字段均不生效"
        );
        println!(
            "NN 决策阶段：Train / RamenSelect / SpecialSelect / RegionSelect / SuperRamenSelect（全部决策阶段）"
        );
        println!(
            "保留手写的部分：事件选项（select_choice / select_event_choice，choice 头未训练）、\
             自选比赛硬守门（Train 阶段截止局面直接选比赛，优先于 policy）、\
             单候选短路（唯一候选直接中选，不发推理，结果与推理后 argmax 相同）"
        );
    }
    println!(
        "ramen_region_strategy={:?} mcts_selected_onsen={} mcts_selection={:?}",
        game_config.ramen_region_strategy, game_config.mcts_selected_onsen, game_config.mcts_selection
    );
    match &args.plans_file {
        Some(path) => println!(
            "计划表={}{} 基种子={}；硬超时={}s",
            path.display(),
            match args.plan_range {
                Some((a, b)) => format!(" 段 [{a}, {b})"),
                None => String::new()
            },
            args.seed,
            args.hard_timeout_secs
        ),
        None => println!(
            "基种子={} run_idx ∈ [{}, {}) 共 {} 局；硬超时={}s",
            args.seed,
            args.run_offset,
            args.run_offset + args.runs,
            args.runs,
            args.hard_timeout_secs
        )
    }

    // ---- NN 模式：整进程只加载一次模型 ----
    #[cfg(feature = "onnx")]
    let nn: NnSlot = match args.policy {
        Policy::Mcts | Policy::Handwritten => None,
        Policy::Nn | Policy::MctsRegionNn => {
            let path = args
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("--trainer {} 需要 --model", args.policy.label()))?;
            let meta = fs::metadata(path).with_context(|| format!("模型不存在: {}", path.display()))?;
            let mode = match args.special_mode.as_str() {
                "canonical" => SpecialSelectMode::Canonical,
                "raw" => SpecialSelectMode::Raw,
                "handwritten" => SpecialSelectMode::Handwritten,
                other => bail!("未知 --special-mode {other:?}（可选 canonical / raw / handwritten）")
            };
            let t0 = Instant::now();
            let t = RamenNnTrainer::load(path)?
                .with_race_shield(true)
                .with_special_mode(mode);
            let load_s = t0.elapsed().as_secs_f64();
            println!(
                "模型={} 字节数={} 加载耗时={load_s:.2}s special_mode={:?} race_shield=true",
                path.display(),
                meta.len(),
                mode
            );
            println!(
                "推理后端=tract-onnx CPU，固定 batch=1、逐请求同步推理（无批量后端、无 GPU）；\
                 直接决策不并行，rayon 全局池本次不参与决策"
            );
            Some(t)
        }
    };
    #[cfg(not(feature = "onnx"))]
    let nn: NnSlot = {
        if args.policy.needs_model() {
            bail!(
                "--trainer {} 需要编译 feature onnx：cargo build --release --features onnx --bin ramen_client_game_bench",
                args.policy.label()
            );
        }
        None
    };

    // ---- 预登记世界 ----
    let inherit = InheritInfo {
        blue_count: game_config.blue_count,
        extra_count: game_config.extra_count
    };
    println!("=== 预登记计划与世界 ===");
    // 两种来源：计划表逐条给出「马娘 + 卡组 + 局号」；否则用合并配置的固定组合 + 连续局号。
    let entries: Vec<PlanEntry> = match &args.plans_file {
        Some(path) => load_plans(path, args.seed, args.plan_range)?,
        None => (0..args.runs)
            .map(|k| PlanEntry {
                plan_id: k as usize,
                uma: game_config.uma,
                cards: game_config.cards,
                run_idx: args.run_offset + k
            })
            .collect()
    };
    let mut worlds: Vec<PlanSpec> = Vec::with_capacity(entries.len());
    for e in entries {
        let (_, rule_master) = bench::seeded_rngs(args.seed, e.run_idx);
        println!(
            "  plan_id={} uma={} cards={:?} run_idx={} rule_master={rule_master}",
            e.plan_id, e.uma, e.cards, e.run_idx
        );
        worlds.push(PlanSpec {
            plan_id: e.plan_id,
            uma: e.uma,
            cards: e.cards,
            run_idx: e.run_idx,
            rule_master
        });
    }
    println!("共 {} 局", worlds.len());
    println!();

    // ---- 证据目录与逐局 CSV ----
    fs::create_dir_all(&args.out).with_context(|| format!("创建证据目录失败: {}", args.out.display()))?;
    let csv_path = args.out.join("games.csv");
    if csv_path.exists() {
        bail!("逐局 CSV 已存在，拒绝覆盖既有证据: {}", csv_path.display());
    }
    let mut csv = fs_err::File::create(&csv_path)?;
    writeln!(csv, "{CSV_HEADER}")?;
    csv.flush()?;
    println!("逐局 CSV: {}", csv_path.display());

    let region_path = args.out.join("regions.csv");
    if region_path.exists() {
        bail!("地区决策 CSV 已存在，拒绝覆盖既有证据: {}", region_path.display());
    }
    let mut region_csv = fs_err::File::create(&region_path)?;
    writeln!(region_csv, "{REGION_CSV_HEADER}")?;
    region_csv.flush()?;
    println!("地区决策 CSV: {}", region_path.display());

    let wall0 = Instant::now();
    let total_plans = worlds.len();
    let mut records: Vec<GameRecord> = Vec::with_capacity(total_plans);
    for plan in worlds {
        let PlanSpec { plan_id, uma, cards, run_idx, rule_master } = plan.clone();
        println!(
            "=== 第 {}/{total_plans} 局 plan_id={plan_id} uma={uma} cards={cards:?} run_idx={run_idx} rule_master={rule_master} ===",
            records.len() + 1
        );
        // 成本统计：两条分支各自填自己那一半，另一半留 0
        let mut cost = CostStat::default();
        let started = Instant::now();
        let mut region_obs: Vec<RegionObs> = Vec::new();
        let out = match args.policy {
            Policy::Mcts | Policy::MctsRegionNn => {
                // 每局独立探针：搜索记录不跨局累加
                let probe: Arc<Mutex<Vec<SearchProbe>>> = Arc::new(Mutex::new(Vec::new()));
                let mcts = RamenMctsTrainer::new(mcts_config.clone())
                    .with_stages(ramen_stages)
                    .verbose(false)
                    .with_search_probe(Arc::clone(&probe));
                let out = if args.policy == Policy::Mcts {
                    let mut trainer = LoggingTrainer::new(mcts, rule_master);
                    trainer.set_logging(false);
                    bench::run_seeded(uma, &cards, &inherit, args.seed, run_idx, &trainer)?
                } else {
                    let run = region_nn_run(
                        &nn, mcts, uma, &cards, &inherit, args.seed, run_idx, rule_master, &mut cost
                    )?;
                    region_obs = run.obs;
                    run.outcome
                };
                let recs = probe.lock().map_err(|_| anyhow!("成本探针锁被毒化"))?;
                cost.searches = recs.len();
                cost.search_s = recs.iter().map(|r| r.elapsed.as_secs_f64()).sum();
                cost.planned = recs.iter().map(SearchProbe::total_planned).sum();
                cost.succeeded = recs.iter().map(SearchProbe::total_succeeded).sum();
                cost.failed = recs.iter().map(SearchProbe::total_failed).sum();
                out
            }
            Policy::Nn => nn_run(&nn, uma, &cards, &inherit, args.seed, run_idx, rule_master, &mut cost)?,
            Policy::Handwritten => {
                // 纯手写：没有搜索探针也没有推理计数，两半成本统计都留 0
                let mut trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), rule_master);
                trainer.set_logging(false);
                bench::run_seeded(uma, &cards, &inherit, args.seed, run_idx, &trainer)?
            }
        };
        let wall_s = started.elapsed().as_secs_f64();

        if out.seed != rule_master {
            bail!("世界标识不一致：预登记 rule_master={rule_master}，实跑 {}", out.seed);
        }

        // ---- 地区决策落盘 ----
        // 地区 NN 臂由训练员逐次观测（含同局面手写参照）；其余臂没有反事实可记，
        // 直接从终局归档 `yearly_selected_regions` 还原实际选择。
        const REGION_TURNS: [i32; 3] = [2, 23, 47];
        if region_obs.is_empty() {
            for (y, picked) in out.yearly_selected_regions.iter().enumerate() {
                let turn = *REGION_TURNS
                    .get(y)
                    .ok_or_else(|| anyhow!("年度地区归档下标 {y} 越界"))?;
                region_obs.push(RegionObs {
                    turn,
                    year: y + 1,
                    candidates: None,
                    picked: *picked,
                    hand: None
                });
            }
        } else {
            // 训练员观测到的实际动作必须与终局归档逐字段一致，否则记录不可信
            if region_obs.len() != out.yearly_selected_regions.len() {
                bail!(
                    "地区观测 {} 条，与终局归档 {} 年不符",
                    region_obs.len(),
                    out.yearly_selected_regions.len()
                );
            }
            for o in &region_obs {
                let archived = out
                    .yearly_selected_regions
                    .get(o.year - 1)
                    .ok_or_else(|| anyhow!("地区观测年份 {} 越界", o.year))?;
                if *archived != o.picked {
                    bail!(
                        "第 {} 年地区观测 {:?} 与终局归档 {archived:?} 不一致",
                        o.year,
                        o.picked
                    );
                }
            }
        }
        let region_src = match args.policy {
            Policy::Mcts | Policy::Handwritten => "handwritten",
            Policy::Nn => "nn_all",
            Policy::MctsRegionNn => "nn_region"
        };
        let mut disagree = 0usize;
        for o in &region_obs {
            if o.hand.is_some_and(|h| h != o.picked) {
                disagree += 1;
            }
            writeln!(region_csv, "{}", region_csv_line(plan_id, run_idx, region_src, o)?)?;
        }
        region_csv.flush()?;
        println!(
            "  地区（来源 {region_src}）: {}；同局面与手写分歧 {disagree}/{} 次",
            region_obs
                .iter()
                .map(|o| format!("Y{}@t{}={:?}", o.year, o.turn, o.picked))
                .collect::<Vec<_>>()
                .join(" "),
            region_obs.len()
        );
        let rec = GameRecord {
            plan_id,
            uma,
            deck: plan.deck_cell(),
            run_idx,
            rule_master,
            score: out.score,
            rank: out.rank.clone(),
            five_status: out.five_status,
            skill_pt: out.skill_pt,
            wall_s,
            searches: cost.searches,
            search_s: cost.search_s,
            planned: cost.planned,
            succeeded: cost.succeeded,
            failed: cost.failed,
            infers: cost.infers,
            free_race_ok: out.free_race_ok,
            rmj_ok: out.rmj_ok,
            friend_all: out.friend_all
        };

        println!(
            "  得分={} 等级={} 五维={:?} 技能点={} 耗时={:.1}s 搜索={} 搜索耗时={:.1}s 续跑 计划={}/成功={}/失败={} 推理={} 自选比赛达标={} RMJ={} 友人全完成={}",
            rec.score,
            rec.rank,
            rec.five_status,
            rec.skill_pt,
            rec.wall_s,
            rec.searches,
            rec.search_s,
            rec.planned,
            rec.succeeded,
            rec.failed,
            rec.infers,
            rec.free_race_ok,
            rec.rmj_ok,
            rec.friend_all
        );
        // 先落盘再入内存：进程被杀时已完成的局必须留下来
        writeln!(csv, "{}", rec.to_csv())?;
        csv.flush()?;
        println!("  累计墙钟={:.1}s", wall0.elapsed().as_secs_f64());
        records.push(rec);
    }

    // ---- 汇总 ----
    let total = wall0.elapsed().as_secs_f64();
    let n = records.len();
    let scores: Vec<f64> = records.iter().map(|r| f64::from(r.score)).collect();
    let mean = scores.iter().sum::<f64>() / n as f64;
    let stdev = if n > 1 {
        (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let lo = records.iter().map(|r| r.score).min().unwrap_or_default();
    let hi = records.iter().map(|r| r.score).max().unwrap_or_default();
    let wall_mean = records.iter().map(|r| r.wall_s).sum::<f64>() / n as f64;
    let fails: usize = records.iter().map(|r| r.failed).sum();
    let race_fail = records.iter().filter(|r| !r.free_race_ok).count();

    println!("\n=== 汇总 ===");
    println!("完成 {n} 局 / 计划 {total_plans} 局；总墙钟={total:.1}s");
    println!(
        "均分={mean:.1} 标准差={stdev:.1} 标准误={:.1} 最低={lo} 最高={hi}",
        stdev / (n as f64).sqrt()
    );
    println!("每局平均耗时={wall_mean:.1}s");
    println!("失败续跑合计={fails}；自选比赛不达标局数={race_fail}");
    let infers: u64 = records.iter().map(|r| r.infers).sum();
    println!(
        "推理请求合计={infers}（每局平均 {:.1}）",
        infers as f64 / n as f64
    );
    if n != total_plans {
        println!("❗未完成 {} 局", total_plans - n);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use anyhow::{Result, bail};

    use umasim::{
        game::ramen::RamenState,
        gamedata::init_global,
        search::SearchConfig,
        utils::{get_workspace_root, load_game_config}
    };
    #[cfg(feature = "onnx")]
    use std::{
        path::Path,
        slice,
        sync::{Arc, Mutex}
    };
    #[cfg(feature = "onnx")]
    use rand::{SeedableRng, prelude::StdRng};
    #[cfg(feature = "onnx")]
    use umaai::region::RegionNnTrainer;
    #[cfg(feature = "onnx")]
    use umasim::output::decision::SOURCE_REGION_NN;
    #[cfg(feature = "onnx")]
    use umasim::{
        game::{
            Game,
            InheritInfo,
            Trainer,
            ramen::{Operation, RamenAction, RamenGame, RamenStage}
        },
        trainer::{
            RamenMctsTrainer, RamenNnTrainer, RamenSearchStages, SpecialSelectMode,
            infer_request_count, ramen_handwritten_trainer::ramen_effective_stage
        }
    };
    #[cfg(feature = "onnx")]
    use umasim::{bench, trainer::LoggingTrainer};

    #[cfg(feature = "onnx")]
    use super::RegionObsRecorder;
    use super::{
        Policy, REGION_CSV_HEADER, RegionObs, check_arm_config, load_plans, parse_array6, parse_cards,
        parse_policy, region_csv_line
    };
    use umasim::{
        gamedata::{GameConfig, RamenRegionStrategy},
        trainer::RamenSearchStages as Stages
    };

    /// 地区 NN 臂必须复用客户端那套适用配置检查；其余三臂不受地区限制
    ///
    /// review#3：拆分前 benchmark 直接构造接管器，`fixed` / 地区搜索未关这两种
    /// 客户端会拒绝的配置，在这里会被放行，并被当成「每局恒 3 次推理」去解读。
    ///
    /// 用 fixture 配置，不读用户的 `game_config.toml`。
    #[test]
    fn test_region_nn_arm_shares_client_config_checks() -> Result<()> {
        let mut fails: Vec<String> = Vec::new();
        let mut note = |ok: bool, what: &str| {
            println!("  [{}] {what}", if ok { "OK" } else { "NG" });
            if !ok {
                fails.push(what.to_string());
            }
        };

        let base = GameConfig::default_for_init();
        let legal = Stages::parse("train,ramen")?;
        let with_region = Stages::parse("train,ramen,region")?;
        let mut fixed_cfg = base.clone();
        fixed_cfg.ramen_region_strategy = RamenRegionStrategy::Fixed;
        fixed_cfg.ramen_region_fixed = Some(vec![[11, 14, 15]]);

        // 1) 地区搜索开关未关 → 拒绝
        let e1 = check_arm_config(Policy::MctsRegionNn, &base, with_region);
        println!("region 搜索未关 → {:?}", e1.as_ref().err().map(ToString::to_string));
        note(e1.is_err(), "mcts+region_nn：ramen_search_stages 仍含 region 时拒绝");

        // 2) fixed 候选 → 拒绝（第 3 年只剩 1 个候选，实际只会推理 2 次）
        let e2 = check_arm_config(Policy::MctsRegionNn, &fixed_cfg, legal);
        println!("fixed 候选 → {:?}", e2.as_ref().err().map(ToString::to_string));
        note(e2.is_err(), "mcts+region_nn：ramen_region_strategy=fixed 时拒绝");

        // 3) 合法配置 → 通过（且**不要求** ramen_region_model_path，模型走 --model）
        note(
            base.ramen_region_model_path.is_none(),
            "fixture 未设 ramen_region_model_path（benchmark 的模型来自 --model）"
        );
        let ok = check_arm_config(Policy::MctsRegionNn, &base, legal);
        println!("合法配置 → {:?}", ok.as_ref().err().map(ToString::to_string));
        note(ok.is_ok(), "mcts+region_nn：合法地区面板配置照常通过");

        // 4) 其余三臂不受地区专用限制（连 fixed + region 全开都不该被拦）
        for arm in [Policy::Mcts, Policy::Nn, Policy::Handwritten] {
            let r = check_arm_config(arm, &fixed_cfg, with_region);
            println!("{arm:?} + fixed + region 全开 → {:?}", r.as_ref().err().map(ToString::to_string));
            note(r.is_ok(), &format!("{arm:?} 不被误加地区专用限制"));
        }

        if fails.is_empty() {
            return Ok(());
        }
        bail!("{} 项观测未通过: {}", fails.len(), fails.join(" / "))
    }

    /// `--trainer` 的标签必须与执行分支一一对应，未知取值报错
    #[test]
    fn test_parse_policy() -> Result<()> {
        for (text, want) in [
            ("mcts", Policy::Mcts),
            ("nn", Policy::Nn),
            ("mcts+region_nn", Policy::MctsRegionNn),
            ("handwritten", Policy::Handwritten)
        ] {
            let got = parse_policy(text)?;
            println!("--trainer {text:?} → {got:?}（label={}）", got.label());
            if got != want || got.label() != text {
                bail!("策略分派错位：{text:?} → {got:?}");
            }
        }
        // ❗`handwritten` 现在是**合法**取值（纯手写臂，面板筛查用），
        // 因此不在非法清单里；下面这些才是未知取值。
        for bad in ["MCTS", "", "search", "region_nn", "mcts+nn"] {
            match parse_policy(bad) {
                Ok(p) => bail!("未知 --trainer {bad:?} 未报错，反而得到 {p:?}"),
                Err(e) => println!("--trainer {bad:?} 正确报错：{e}")
            }
        }
        // 需模型 / 跑搜索这两项元信息也必须与分支一一对应
        for (p, needs_model, runs_search) in [
            (Policy::Mcts, false, true),
            (Policy::Nn, true, false),
            (Policy::MctsRegionNn, true, true)
        ] {
            println!(
                "{:?}: needs_model={} runs_search={}",
                p,
                p.needs_model(),
                p.runs_search()
            );
            if p.needs_model() != needs_model || p.runs_search() != runs_search {
                bail!("{p:?} 的元信息与预期不符");
            }
        }
        Ok(())
    }

    /// `--search-n` 必须真正进入 `SearchConfig`，而不是只改打印
    ///
    /// 走的是与 `main` 相同的顺序：先改 `game_config.mcts.search_n`，
    /// 再 `SearchConfig::new_game_config`。
    #[test]
    fn test_search_n_reaches_search_config() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        let mut cfg = load_game_config()?;
        let file_value = cfg.mcts.search_n;
        cfg.mcts.search_n = 1024;
        let sc = SearchConfig::new_game_config(&cfg);
        println!("文件值 search_n={file_value} → 覆盖后 SearchConfig.search_n={}", sc.search_n);
        if sc.search_n != 1024 {
            bail!("search_n 覆盖未进入 SearchConfig：{}", sc.search_n);
        }
        // 其余搜索字段不应被这次覆盖连带改动
        println!(
            "use_ucb={} group={} cpuct={} stdev={} max_depth={} policy_delta={}",
            sc.use_ucb, sc.search_group_size, sc.search_cpuct, sc.expected_search_stdev, sc.max_depth,
            sc.policy_delta
        );
        Ok(())
    }

    /// 地区门控：只有地区候选集才被判成 `RegionSelect`，训练/比赛候选不会误入网络分支
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_stage_gate() -> Result<()> {
        use umasim::game::ramen::TrainingType;

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let game = RamenGame::default();

        let regions = [
            RamenAction::no_ramen(Operation::RegionSelect([0, 1, 4])),
            RamenAction::no_ramen(Operation::RegionSelect([0, 3, 4]))
        ];
        let stage = ramen_effective_stage(&game, &regions);
        println!("地区候选 → {stage:?}");
        if stage != RamenStage::RegionSelect {
            bail!("地区候选未被判成 RegionSelect：{stage:?}");
        }

        let others = [
            RamenAction::no_ramen(Operation::Train(TrainingType::Speed)),
            RamenAction::no_ramen(Operation::Race),
            RamenAction::no_ramen(Operation::Rest)
        ];
        for a in &others {
            let s = ramen_effective_stage(&game, slice::from_ref(a));
            println!("候选 {:?} → {s:?}", a.operation);
            if s == RamenStage::RegionSelect {
                bail!("非地区候选 {:?} 被误判成 RegionSelect", a.operation);
            }
        }
        Ok(())
    }

    /// 地区记录：年份由 turn 推出（turn 23 = 第 2 年），地区名逐个查表，分歧位正确
    #[test]
    fn test_region_csv_line() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        // turn 23 选的是第 2 年——`current_year()` 此时仍是 1，不能拿来标年
        let year = RamenState::region_archive_year_idx(23)? + 1;
        println!("turn 23 → 第 {year} 年");
        if year != 2 {
            bail!("turn 23 的年份标注错误：{year}");
        }
        let obs = RegionObs {
            turn: 23,
            year,
            candidates: Some(10),
            picked: [5, 8, 9],
            hand: Some([7, 8, 9])
        };
        let line = region_csv_line(0, 150000, "nn_region", &obs)?;
        println!("{REGION_CSV_HEADER}");
        println!("{line}");
        if !line.ends_with(",1") {
            bail!("分歧位应为 1：{line}");
        }
        let same = RegionObs {
            hand: Some([5, 8, 9]),
            ..obs
        };
        let line2 = region_csv_line(0, 150000, "nn_region", &same)?;
        println!("{line2}");
        if !line2.ends_with(",0") {
            bail!("同选时分歧位应为 0：{line2}");
        }
        // 非地区 NN 臂没有反事实，hand 三列留空
        let plain = RegionObs {
            candidates: None,
            hand: None,
            ..obs
        };
        let line3 = region_csv_line(0, 150000, "handwritten", &plain)?;
        println!("{line3}");
        if !line3.ends_with(",,,") {
            bail!("无反事实时 hand 三列应为空：{line3}");
        }
        Ok(())
    }

    /// 地区包装器：地区决策后不暴露内部 MCTS 的旧摘要，其余决策照常转发
    ///
    /// 「不暴露旧摘要」的判据是**摘要内容**，不是 `last_decision()` 是否为 `None`：
    /// 网络地区决策会给出**它自己**的一条摘要（无搜索评分、来源标 `region_nn`、
    /// `action_index` 就是这次选中的候选），渲染端据此既不挂旧理由也不误标手写
    /// （review#1）。真正要抓的是：摘要里带上了搜索评分 / 旧的 `decision_kind` /
    /// 对不上的 `action_index`——那才说明上一次 MCTS 的旧摘要漏了出来。
    ///
    /// 用极小搜索预算（`search_n=2`、均匀分配）跑到第一次地区决策（turn 2）为止，
    /// 不做任何性能测量，也不跑整局。
    ///
    /// ❗`saved_models/` 在 `.gitignore` 里，干净检出**不会**有权重，所以模型缺失时
    /// 只能跳过——但那是**零覆盖**，不是通过。两点防护：跳过时打显眼警告；
    /// 置 `UMAAI_REQUIRE_REGION_MODEL=1` 则模型缺失直接失败，
    /// 供「必须覆盖这条路径」的场合强制。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_wrapper_masks_stale_summary() -> Result<()> {
        // 本测试也会产生推理，与整局计数测试互斥（见 INFER_GUARD 文档）
        let _guard = INFER_GUARD.lock().map_err(|_| anyhow::anyhow!("推理计数锁被毒化"))?;
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let model = Path::new("saved_models/arms/ens_R4_g123.onnx");
        if !model.exists() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 这意味着地区屏蔽位逻辑本次**零覆盖**，绿色不代表它还正确。");
            println!("❗❗ 需要强制覆盖时置环境变量 UMAAI_REQUIRE_REGION_MODEL=1。");
            if env::var("UMAAI_REQUIRE_REGION_MODEL").is_ok_and(|v| v == "1") {
                bail!("UMAAI_REQUIRE_REGION_MODEL=1，但模型不存在：{}", model.display());
            }
            return Ok(());
        }
        let cfg = load_game_config()?;
        let recorder = Arc::new(RegionObsRecorder::new());
        let hybrid = RegionNnTrainer::new(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
                .with_stages(RamenSearchStages::parse("train,ramen")?)
                .verbose(false),
            RamenNnTrainer::load(model)?
                .with_race_shield(true)
                .with_special_mode(SpecialSelectMode::Canonical)
        )
        .with_observer(Arc::clone(&recorder) as Arc<_>);
        let inherit = InheritInfo {
            blue_count: cfg.blue_count,
            extra_count: cfg.extra_count
        };
        let mut game = RamenGame::newgame(cfg.uma, &cfg.cards, inherit)?;
        game.set_rule_master(20260913);
        let mut rng = StdRng::seed_from_u64(20260913);

        let mut region_seen = 0usize;
        let mut leaked = 0usize;
        let mut released = false;
        while game.next() {
            let turn = game.turn();
            let before = recorder.snapshot()?.len();
            game.run_stage(&hybrid, &mut rng)?;
            let after = recorder.snapshot()?.len();
            let info = hybrid.last_decision();
            let decision = info.is_some();
            let breakdown = hybrid.last_breakdown().is_some();
            if after > before {
                region_seen += 1;
                // 本次地区决策的选中下标，用来核对摘要说的就是这一次
                let picked = recorder
                    .snapshot()?
                    .last()
                    .map(|o| o.picked)
                    .ok_or_else(|| anyhow::anyhow!("观测钩子未记录本次地区决策"))?;
                let stale = match &info {
                    // 摘要缺失也算没挂旧理由，但要记下来（上层会退回中性文案）
                    None => false,
                    Some(d) => {
                        let named = d.source_label() == Some(SOURCE_REGION_NN);
                        let scored = !d.candidate_scores.is_empty() || !d.candidate_n.is_empty();
                        let kind_ok = d.decision_kind == "region_select";
                        // 第 1 年地区候选恒 10 个；摘要必须是**这一次**的候选表，
                        // 且选中下标落在表内（旧搜索摘要的候选表长度与此不同）
                        let idx_ok =
                            d.candidate_descriptions.len() == 10 && d.action_index < d.candidate_descriptions.len();
                        println!(
                            "turn {turn}: 地区决策（网络）→ 来源={:?} 评分项={} kind={:?} idx={} 候选={}",
                            d.source_label(),
                            d.candidate_scores.len(),
                            d.decision_kind,
                            d.action_index,
                            d.candidate_descriptions.len()
                        );
                        println!("  观测钩子记录的选中地区 = {picked:?}");
                        !named || scored || !kind_ok || !idx_ok
                    }
                };
                println!("turn {turn}: last_decision={decision} last_breakdown={breakdown} 旧摘要泄漏={stale}");
                if stale || breakdown {
                    leaked += 1;
                }
            } else if region_seen > 0 && decision {
                let d = info.as_ref().expect("已判定为 Some");
                println!(
                    "turn {turn}: 地区之后的转发决策 → 恢复为搜索侧摘要（kind={:?} 评分项={} 来源={:?}）",
                    d.decision_kind,
                    d.candidate_scores.len(),
                    d.source_label()
                );
                if d.source_label() == Some(SOURCE_REGION_NN) {
                    bail!("地区屏蔽位没有解除：后续决策仍被标成 region_nn");
                }
                released = true;
                break;
            }
            if turn > 14 {
                break;
            }
        }
        println!("地区决策 {region_seen} 次；泄漏旧摘要 {leaked} 次；屏蔽位解除={released}");
        if region_seen == 0 {
            bail!("跑到 turn 14 仍未捕获地区决策，测试未覆盖到目标路径");
        }
        if leaked > 0 {
            bail!("地区决策的摘要不是本次网络决策自己的（疑似泄漏旧搜索摘要）{leaked} 次");
        }
        if !released {
            bail!("地区之后的转发决策未恢复摘要，屏蔽位没有解除");
        }
        Ok(())
    }

    /// 会做推理的测试之间的互斥锁
    ///
    /// `infer_request_count()` 是**进程级**全局计数，两个 onnx 测试并行跑会互相
    /// 污染增量。凡是要读增量、或会产生推理的测试，都先拿这把锁。
    #[cfg(feature = "onnx")]
    static INFER_GUARD: Mutex<()> = Mutex::new(());

    /// 整局隔离性：NN 地区模式下**恰好 3 次**推理、**恰好 3 次**地区决策
    ///
    /// 地区阶段候选数第 1/2 年 10、第 3 年 `all` 下 120，均 > 1，不会被单候选短路。
    /// 因此「整局推理数 == 地区决策数 == 3」就是**搜索内部没有任何网络推理**的直接证据：
    /// 只要 rollout 里跑过一次网络，这个数就会远大于 3。
    ///
    /// 搜索压到 `search_n=2`（本测试只看隔离性，不看棋力）。模型不存在时跳过并显式
    /// 声明零覆盖。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_nn_full_game_infers_exactly_three() -> Result<()> {
        let _guard = INFER_GUARD.lock().map_err(|_| anyhow::anyhow!("推理计数锁被毒化"))?;
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let model = Path::new("saved_models/arms/ens_G2mix_g123.onnx");
        if !model.exists() {
            println!("❗❗ 本测试被跳过：模型不存在 {}", model.display());
            println!("❗❗ 这意味着「整局恰好 3 次推理」本次**零覆盖**，绿色不代表它还正确。");
            println!("❗❗ 需要强制覆盖时置环境变量 UMAAI_REQUIRE_REGION_MODEL=1。");
            if env::var("UMAAI_REQUIRE_REGION_MODEL").is_ok_and(|v| v == "1") {
                bail!("UMAAI_REQUIRE_REGION_MODEL=1，但模型不存在：{}", model.display());
            }
            return Ok(());
        }
        let cfg = load_game_config()?;
        let recorder = Arc::new(RegionObsRecorder::new());
        let hybrid = RegionNnTrainer::new(
            RamenMctsTrainer::new(SearchConfig::default().with_search_n(2).with_ucb(false))
                .with_stages(RamenSearchStages::parse("train,ramen")?)
                .verbose(false),
            RamenNnTrainer::load(model)?
                .with_race_shield(true)
                .with_special_mode(SpecialSelectMode::Canonical)
        )
        .with_observer(Arc::clone(&recorder) as Arc<_>);
        let inherit = InheritInfo {
            blue_count: cfg.blue_count,
            extra_count: cfg.extra_count
        };

        // 与生产 / 实验路径同一个入口：外面包一层 LoggingTrainer（关日志）
        let mut trainer = LoggingTrainer::new(hybrid, 20260914);
        trainer.set_logging(false);
        let before = infer_request_count();
        let out = bench::run_seeded(cfg.uma, &cfg.cards, &inherit, 61444, 250000, &trainer)?;
        let infers = infer_request_count() - before;
        let obs = recorder.snapshot()?;
        let turns: Vec<i32> = obs.iter().map(|o| o.turn).collect();
        let cands: Vec<Option<usize>> = obs.iter().map(|o| o.candidates).collect();
        println!(
            "整局完成：分数={} 推理={infers} 地区决策={} 回合={turns:?} 候选数={cands:?}",
            out.score,
            obs.len()
        );
        if obs.len() != 3 {
            bail!("地区决策应恰好 3 次，实际 {}", obs.len());
        }
        if turns != vec![2, 23, 47] {
            bail!("地区决策回合应为 [2, 23, 47]，实际 {turns:?}");
        }
        if infers != 3 {
            bail!("整局推理应恰好 3 次（>3 说明搜索内部也在推理），实际 {infers}");
        }
        Ok(())
    }

    /// `--cards` 只接受 6 项 u32，且解析结果保持输入顺序
    #[test]
    fn test_parse_cards() -> Result<()> {
        let v = parse_cards("303124,303114,303044,303004,302894,303054")?;
        println!("cards = {v:?}");
        if v != [303124, 303114, 303044, 303004, 302894, 303054] {
            bail!("cards 解析错误: {v:?}");
        }
        for bad in ["303124,303114,303044,303004,302894", "303124,303114,303044,303004,302894,303054,1", "303124,x,0,0,0,0", "-1,0,0,0,0,0"] {
            match parse_cards(bad) {
                Ok(x) => bail!("非法 cards {bad:?} 未报错，反而得到 {x:?}"),
                Err(e) => println!("非法 cards {bad:?} 正确报错：{e}")
            }
        }
        Ok(())
    }

    /// `--extra-count` 只接受 6 项整数
    #[test]
    fn test_parse_array6() -> Result<()> {
        let v = parse_array6("0,20,20,40,40,40")?;
        println!("extra_count = {v:?}");
        if v != [0, 20, 20, 40, 40, 40] {
            bail!("extra_count 解析错误: {v:?}");
        }
        for bad in ["0,20,20,40,40", "0,20,20,40,40,40,40", "0,x,0,0,0,0"] {
            match parse_array6(bad) {
                Ok(x) => bail!("非法 extra_count {bad:?} 未报错，反而得到 {x:?}"),
                Err(e) => println!("extra_count {bad:?} 正确报错：{e}")
            }
        }
        Ok(())
    }

    /// 计划表：正常读入、`--plan-range` 截段，以及三种非法输入必须报错
    ///
    /// 用临时文件写 JSON，只比较字段取值，不做任何指纹。
    #[test]
    fn test_plans_file_parse_and_guards() -> Result<()> {
        let dir = env::temp_dir().join("ramen_client_game_bench_plans_test");
        std::fs::create_dir_all(&dir)?;
        let good = dir.join("good.json");
        std::fs::write(
            &good,
            r#"{"base_seed":61444,"plans":[
                {"plan_id":0,"uma":100603,"cards":[1,2,3,4,5,6],"run_idx":250000},
                {"plan_id":1,"uma":113101,"cards":[7,8,9,10,11,12],"run_idx":250001},
                {"plan_id":2,"uma":114101,"cards":[13,14,15,16,17,18],"run_idx":250002}]}"#
        )?;
        let all = load_plans(&good, 61444, None)?;
        println!("整表读入 {} 条，首条 uma={} run_idx={}", all.len(), all[0].uma, all[0].run_idx);
        let seg = load_plans(&good, 61444, Some((1, 3)))?;
        println!(
            "段 [1,3) 读入 {} 条，plan_id={:?}",
            seg.len(),
            seg.iter().map(|p| p.plan_id).collect::<Vec<_>>()
        );
        if seg.len() != 2 || seg[0].plan_id != 1 || seg[1].run_idx != 250002 {
            bail!("--plan-range 截段结果不对：{seg:?}");
        }

        // 1) base_seed 与 --seed 不一致
        match load_plans(&good, 99999, None) {
            Ok(_) => bail!("base_seed 不一致未报错"),
            Err(e) => println!("base_seed 不一致正确报错：{e}")
        }
        // 2) plan_id 与下标错位
        let bad_id = dir.join("bad_id.json");
        std::fs::write(
            &bad_id,
            r#"{"base_seed":61444,"plans":[{"plan_id":5,"uma":1,"cards":[1,2,3,4,5,6],"run_idx":1}]}"#
        )?;
        match load_plans(&bad_id, 61444, None) {
            Ok(_) => bail!("plan_id 错位未报错"),
            Err(e) => println!("plan_id 错位正确报错：{e}")
        }
        // 3) run_idx 重复（两条计划抢同一个世界）
        let dup = dir.join("dup.json");
        std::fs::write(
            &dup,
            r#"{"base_seed":61444,"plans":[
                {"plan_id":0,"uma":1,"cards":[1,2,3,4,5,6],"run_idx":7},
                {"plan_id":1,"uma":2,"cards":[1,2,3,4,5,6],"run_idx":7}]}"#
        )?;
        match load_plans(&dup, 61444, None) {
            Ok(_) => bail!("run_idx 重复未报错"),
            Err(e) => println!("run_idx 重复正确报错：{e}")
        }
        // 4) --plan-range 越界
        match load_plans(&good, 61444, Some((0, 9))) {
            Ok(_) => bail!("--plan-range 越界未报错"),
            Err(e) => println!("--plan-range 越界正确报错：{e}")
        }
        Ok(())
    }
}
