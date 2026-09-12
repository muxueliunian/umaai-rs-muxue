//! 采样空间基准：在第一代教师数据的**同一分布**上测某个策略的均分
//!
//! # 为什么不能用 `bench_base`
//!
//! `bench_config.toml` 的马娘是 `102601` 美浦波旁，它**不在**
//! [`SamplingSpace::gen1`] 定的 7 马娘名单里，且 `freeRaces = []` 没有自选比赛要求。
//! 计划文档 §2.3 已记明：`51168 / 50833 / 50872 / 51001` 这一系列手写策略基线
//! 全部测自该马娘，**不能用作第一代网络的验收门槛**——网络是在 7 马娘 × 525 种
//! (马娘, 卡组) 上训练的，拿一个训练分布外的马娘去比，结论没有意义。
//!
//! 本 bin 直接遍历 [`SamplingSpace::gen1`] 的全部计划，每个计划跑若干整局，
//! 给出与教师数据同分布的均分。它同时是第一代网络的验收口径：把 `--trainer`
//! 换成网络策略、其余参数不动，两个数字才可比。
//!
//! ❗`ramen_region_strategy` 由本 bin 强制为 `all`（与 `bench_base`、
//! `ramen_teacher_collect` 一致），不跟随 `game_config.toml`。跟随 toml 会让
//! 有人为手动模式改成 `fixed` 时，基准静默换成另一个分布，此前记下的全部均分
//! 基线一并作废，而输出里看不出任何区别。
//!
//! # 分布外模式
//!
//! 给出 `--shape` 后切换到 [`SamplingSpace::custom`]：只枚举该构成，并可用
//! `--extra-card` 往卡池里补卡。用途是检验网络对**未训练卡组流派**的泛化，
//! 例如「2 速 1 耐 2 智」——池内只有一张智力卡，必须补一张才组得出来。
//!
//! ❗分布外模式的分数**不能**与默认口径的数字直接比较：卡组空间换了，
//! 计划数变了，逐计划的种子段也随之不同。要下结论必须在同一模式下
//! 跑手写基线做配对参照。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p umasim --bin ramen_space_bench -- \
//!     --trainer handwritten --runs-per-plan 8
//!
//! cargo run --release -p umasim --features onnx --bin ramen_space_bench -- \
//!     --shape 2,1,0,0,2 --extra-card 303064 --trainer nn --model model.onnx
//! ```

use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex
};

#[cfg(feature = "onnx")]
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use rayon::prelude::*;
use umasim::{
    bench,
    collector::{compute_file_signature, fnv1a64, try_get_git_commit},
    gamedata::{GameConfig, RamenRegionStrategy, init_global_with_config},
    sampler::{DeckPlan, SamplingSpace, gen1_inherit, space_from_cli},
    search::SearchConfig,
    trainer::{
        LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RamenSelection, RandomTrainer, RecommendedRamenTrainer
    },
    utils::{get_workspace_root, load_game_config}
};
#[cfg(feature = "onnx")]
use umasim::trainer::{RamenNnTrainer, SpecialSelectMode, infer_request_count};

/// 基准参数
#[derive(Parser, Debug)]
#[command(about = "在第一代采样空间（7 马娘 × 525 卡组组合）上测策略均分；--shape 可切到分布外空间")]
struct BenchArgs {
    /// 策略：`handwritten`（手写规则）/ `random`（随机基线）/ `nn`（ONNX 网络，需 `--model`）/
    /// `search`（扁平搜索，即教师本身；配置见 `--search-n`）
    #[arg(long, default_value = "handwritten")]
    trainer: String,

    /// ONNX 模型路径；`--trainer nn` 时必填
    #[arg(long)]
    model: Option<PathBuf>,

    /// 每个计划跑几局
    #[arg(long, default_value_t = 8)]
    runs_per_plan: u64,

    /// 基础种子。每局的随机世界由 `derive_seed(seed + plan * 1000003, [run_idx])` 决定
    ///
    /// ❗**不要用相邻的基种子跑多批当作独立样本**。`derive_seed` 是「XOR 后
    /// splitmix64」，而 `base ^ r == base + r` 在低位无进位时成立，所以
    /// `--seed 61444 --runs-per-plan 8` 与 `--seed 61445 --runs-per-plan 8`
    /// 会大面积撞上同一批世界：实测三个相邻基种子跑出的 12600 局里只有 5248 个
    /// 唯一世界（3152 个重复 3 次），重复局分数完全相同，白白虚增样本量、低估标准误。
    ///
    /// 正确做法是**固定一个基种子，用 [`BenchArgs::run_offset`] 切分世界空间**：
    /// 同一 `seed` 下不同 `run_idx` 必然给出不同世界（splitmix64 对不同输入单射），
    /// 且跨计划也不会撞（计划间基种子相差 1000003 的倍数，远大于 `run_idx` 的位宽）。
    #[arg(long, default_value_t = 61444)]
    seed: u64,

    /// 局号起点；本次跑 `run_idx ∈ [run_offset, run_offset + runs_per_plan)`
    ///
    /// 用来切出**互不重叠**的世界子集，实现「选择集 / 验收集分离」：
    /// 例如选择集用 `--run-offset 0 --runs-per-plan 1`（525 局，约 1 分钟），
    /// 验收集用 `--run-offset 8 --runs-per-plan 24`（12600 局）。
    /// 两者由构造保证零重叠，因此在选择集上挑 checkpoint 不会污染验收集的无偏性。
    #[arg(long, default_value_t = 0)]
    run_offset: u64,

    /// 只跑前 N 个计划（调试用，默认全跑）
    #[arg(long)]
    plans: Option<usize>,

    /// 每隔 N 个计划取一个（默认 1 = 全取）
    ///
    /// 计划表是按「马娘 → 卡组构成」分组连续排布的，所以 `--plans 8` 取到的
    /// 是同一个马娘同一种构成的 8 个计划，**不是**空间的代表性样本。
    /// 昂贵策略（`--trainer search`）做先导样本时必须用本参数跨组取，
    /// 否则量到的方差只是组内方差。与 `--plans` 同时给出时，`--plans` 指的是
    /// **抽样后**保留的计划数。
    #[arg(long, default_value_t = 1)]
    plan_stride: usize,

    /// 把逐局结果写成 CSV
    #[arg(long)]
    csv: Option<PathBuf>,

    /// 从 `<csv>.journal.csv` 续跑：跳过日志里已完成的局
    ///
    /// 不给本参数而日志已存在时直接报错，避免把两次不同配置的结果混进同一份日志。
    #[arg(long)]
    resume: bool,

    /// EXP-006c 手写变体 token 串（如 `wisf0-capd0`）；仅 trainer=handwritten 时生效
    #[arg(long)]
    variant: Option<String>,

    /// 关闭网络策略的自选比赛硬守门（纯网络，仅供研究守门能否移除；不作为验收口径）
    #[arg(long)]
    no_race_shield: bool,

    /// `SpecialSelect` 阶段的推理口径：`canonical`（还原到联合决策根，默认）/
    /// `raw`（历史行为，存在训练—部署语义错位）/ `handwritten`（该阶段交给手写，对照组）
    #[arg(long, default_value = "canonical")]
    special_mode: String,

    /// 卡组构成 `速,耐,力,根,智`，五项合计为 5（友人卡固定 1 张，不计入）
    ///
    /// 给出即切到**分布外**空间：只枚举这一种构成，用于检验网络对未训练卡组流派
    /// 的泛化。不给时用第一代的 3 种构成，与教师数据同分布。
    #[arg(long)]
    shape: Option<String>,

    /// 追加进卡池的支援卡 idrank（6 位 = 5 位卡 ID + 突破等级），可重复给出
    ///
    /// 只在有 `--shape` 时允许：默认口径必须锁死在训练分布的 11 张卡上，
    /// 否则「同分布均分」这个名字就不成立了。
    #[arg(long)]
    extra_card: Vec<u32>,

    /// `--trainer search` 的每候选搜索次数；教师数据用的是 512
    #[arg(long, default_value_t = 512)]
    search_n: usize,

    /// `--trainer search` 的激进度因子最大值
    ///
    /// 必须显式给出：`SearchConfig::default()` 是 50.0，而教师采集用的是 1.4，
    /// 取错会让「教师闭环分」量到另一个策略上。
    #[arg(long, default_value_t = 1.4)]
    radical_factor_max: f64,

    /// `--trainer search` 的 rollout 基策改用这个 ONNX 网络（`Q^手写` → `Q^NN`）
    ///
    /// 不给时 rollout 仍是推荐手写策略，即历史教师口径。给出后搜索排序依据改变，
    /// **与此前记录的教师闭环分不可比**。守门与 `SpecialSelect` 口径跟随
    /// `--no-race-shield` / `--special-mode`，与 `--trainer nn` 同一套。
    #[arg(long)]
    rollout_model: Option<PathBuf>,

    /// 网络 rollout 的生效回合上限（含）；不给 = 整局都用网络
    ///
    /// 混合 rollout 的成本旋钮：网络推理远贵于手写，前 k 回合用网络、之后交回
    /// 手写可把成本压回可跑范围。仅在给了 `--rollout-model` 时有意义。
    #[arg(long)]
    rollout_nn_max_turn: Option<i32>
}

/// 解析 `--special-mode`（仅 onnx 下有意义）
///
/// # 错误
///
/// 未知取值时报错——静默回退到默认会让对照组静静地变成实验组。
#[cfg(feature = "onnx")]
fn parse_special_mode(s: &str) -> Result<SpecialSelectMode> {
    match s {
        "canonical" => Ok(SpecialSelectMode::Canonical),
        "raw" => Ok(SpecialSelectMode::Raw),
        "handwritten" => Ok(SpecialSelectMode::Handwritten),
        other => bail!("未知 --special-mode: {other}（可选 canonical / raw / handwritten）")
    }
}

/// 一个 ONNX 模型的身份：路径 + 内容哈希
///
/// 新类型而非裸字符串：路径与哈希必须成对出现——只记路径无法分辨同名文件被
/// 覆盖过，只记哈希则事后找不回文件。
#[derive(Debug, Clone)]
struct ModelIdentity {
    /// 模型路径（原样保留命令行给的写法）
    path: String,
    /// 文件内容的 FNV-1a 64 指纹（全仓库统一口径，见 `collector::fnv1a64`）
    hash_fnv1a64: String
}

impl ModelIdentity {
    /// 读文件算指纹
    ///
    /// # 错误
    ///
    /// 文件不存在或读取失败时报错——静默跳过哈希会让实验身份看着完整、实则不可核。
    fn of(path: &Path) -> Result<Self> {
        let sig = compute_file_signature(path, true)?;
        let hash = sig
            .hash_fnv1a64
            .ok_or_else(|| anyhow::anyhow!("未算出模型哈希: {}", path.display()))?;
        Ok(Self {
            path: path.to_string_lossy().to_string(),
            hash_fnv1a64: hash
        })
    }

    /// 上屏用的短写法：文件名 + 哈希前 8 位
    fn short(&self) -> String {
        let name = Path::new(&self.path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| self.path.clone());
        format!("{name}@{}", &self.hash_fnv1a64[..8])
    }
}

/// 本次跑的完整实验身份
///
/// 审查抓到的 P2：上屏标签原先只写模型路径，漏掉了同样参与装配的
/// `special_mode` 与自选比赛守门开关，于是**不同策略可能打印出相同身份**。
/// 逐局 CSV 也没有任何教师配置。这里把「决定策略是什么」的每一项都收进来，
/// 既用于上屏，也随 CSV 落一份 sidecar JSON。
struct RunIdentity {
    /// 上屏用的一行短标签
    label: String,
    /// 落盘用的完整 JSON
    json: serde_json::Value
}

/// 组装本次跑的实验身份
///
/// # 错误
///
/// 给了模型路径但读不出内容哈希时报错。
fn run_identity(args: &BenchArgs, game_config: &GameConfig, plan_indices: &[usize]) -> Result<RunIdentity> {
    // 网络策略（--trainer nn / --rollout-model）共用同一套推理开关，
    // 故两处都要记：只记其中一处会让另一处的对照组变成静默的实验组
    let race_shield = !args.no_race_shield;
    let policy_model = match args.model.as_ref() {
        Some(p) => Some(ModelIdentity::of(p)?),
        None => None
    };
    let rollout_model = match args.rollout_model.as_ref() {
        Some(p) => Some(ModelIdentity::of(p)?),
        None => None
    };

    let nn_knobs = format!(
        "{}{}",
        args.special_mode,
        if race_shield { "" } else { ",无守门" }
    );
    let label = match args.trainer.as_str() {
        "nn" => match &policy_model {
            Some(m) => format!("nn[{nn_knobs}] {}", m.short()),
            None => format!("nn[{nn_knobs}] (缺 --model)")
        },
        "search" => {
            let rollout = match (&rollout_model, args.rollout_nn_max_turn) {
                (Some(m), Some(t)) => format!("nn<=t{t}[{nn_knobs}] {}", m.short()),
                (Some(m), None) => format!("nn[{nn_knobs}] {}", m.short()),
                (None, _) => "handwritten".to_string()
            };
            format!(
                "search[n={} rf={} rollout={rollout}]",
                args.search_n, args.radical_factor_max
            )
        }
        "handwritten" => match args.variant.as_deref() {
            Some(v) if v != "base" => format!("handwritten[{v}]"),
            _ => "handwritten".to_string()
        },
        other => other.to_string()
    };

    let model_json = |m: &Option<ModelIdentity>| match m {
        Some(m) => serde_json::json!({ "path": m.path, "hash_fnv1a64": m.hash_fnv1a64 }),
        None => serde_json::Value::Null
    };
    let json = serde_json::json!({
        "label": label,
        "trainer": args.trainer,
        "variant": args.variant,
        "policy_model": model_json(&policy_model),
        "nn_special_mode": args.special_mode,
        "nn_race_shield": race_shield,
        "search": {
            "search_n": args.search_n,
            "radical_factor_max": args.radical_factor_max,
            "use_ucb": false,
            "stages": "all",
            "selection": "score",
            "rollout_model": model_json(&rollout_model),
            "rollout_nn_max_turn": args.rollout_nn_max_turn,
            "strict_rollout": rollout_model.is_some()
        },
        "space": {
            "shape": args.shape,
            "extra_card": args.extra_card,
            "ramen_region_strategy": "all"
        },
        "worlds": {
            "seed": args.seed,
            "run_offset": args.run_offset,
            "runs_per_plan": args.runs_per_plan,
            "plan_stride": args.plan_stride,
            "plans": args.plans
        },
        "git_commit": try_get_git_commit(Path::new(".")),
        // 同一个 commit **不代表**同一个程序：工作树里未提交的改动同样改变结果。
        "source_inputs_fnv1a64": source_inputs_fingerprint()?,
        // 生效配置而非 toml 文件：main 里会把 `ramen_region_strategy` 强制改成 All，
        // 光记文件哈希会把「改过的配置」记成「文件里写的配置」。
        "game_config_fnv1a64": fingerprint_json(&serde_json::to_value(game_config)?),
        // 生效配置之外的游戏数据（卡库 / 事件库 / 剧本表）同样决定结果
        "game_data_fnv1a64": game_data_fingerprint()?,
        // 实际参与本次跑的计划：只记条数不够，抽样步长相同而空间不同就会撞车
        "plan_selection": {
            "count": plan_indices.len(),
            "original_indices_fnv1a64": fingerprint_json(&serde_json::json!(plan_indices))
        }
    });
    Ok(RunIdentity { label, json })
}

/// 对任意 JSON 取 FNV-1a 64 指纹
///
/// 走 `to_string`：`serde_json::Value` 的 map 是有序的（`BTreeMap` 后端），
/// 故同内容必得同串、同指纹。
fn fingerprint_json(v: &serde_json::Value) -> String {
    format!("{:016x}", fnv1a64(v.to_string().as_bytes()))
}

/// 游戏数据文件的内容指纹
///
/// `gamedata/` 下的 `.json` 与 `.toml`：卡库、事件库、剧本表、常量表。改这些数据
/// 不改任何 `.rs`，程序行为照样变——只靠源码指纹与生效配置指纹会漏掉。
///
/// # 错误
///
/// 目录不存在、遍历或读文件失败时报错。
fn game_data_fingerprint() -> Result<String> {
    let dir = Path::new("gamedata");
    let entries = std::fs::read_dir(dir).with_context(|| format!("遍历游戏数据目录失败: {}", dir.display()))?;
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("读取目录项失败: {}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "json" || e == "toml") {
            files.push(path);
        }
    }
    // 遍历顺序随文件系统而变，排序后再喂哈希
    files.sort();
    let mut input: Vec<u8> = Vec::new();
    for f in &files {
        input.extend_from_slice(f.to_string_lossy().replace('\\', "/").as_bytes());
        input.push(0);
        let bytes = std::fs::read(f).with_context(|| format!("读游戏数据失败: {}", f.display()))?;
        input.extend_from_slice(&bytes.len().to_le_bytes());
        input.extend_from_slice(&bytes);
        input.push(0);
    }
    Ok(format!("{:016x}", fnv1a64(&input)))
}

/// 影响二进制行为的源码输入指纹（路径 + 内容）
///
/// # 为什么不用 git
///
/// 起初这里取的是 `git status --porcelain` + `git diff HEAD` 的指纹，**有洞**：
/// 前者对未跟踪文件只记路径，后者根本不含未跟踪文件的内容。于是改动一个未跟踪
/// 的 `.rs`（当前 `ramen_rollout_trainer.rs` 正是如此）重新编译后，指纹可能一字不变，
/// `--resume` 就会把旧结果混进新配置——正是身份闸门要挡的事。
///
/// 反过来，git 口径还会把文档改动算进身份，造成无谓的拒绝；实验输出若落在未被
/// 忽略的位置，首跑甚至会改变自己的身份。故这里改为**显式限定输入范围**：
/// 只吃参与编译的源码与依赖声明，已跟踪与未跟踪一视同仁，文档与产物一律不吃。
///
/// # 错误
///
/// 目录遍历或读文件失败时报错——静默跳过会让身份看着完整、实则漏掉改动。
fn source_inputs_fingerprint() -> Result<String> {
    /// 参与编译的源码根：只取 `.rs` 与 `Cargo.toml`
    fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let entries =
            std::fs::read_dir(dir).with_context(|| format!("遍历源码目录失败: {}", dir.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("读取目录项失败: {}", dir.display()))?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                // 产物目录与版本库内部不是源码输入
                if name != "target" && name != ".git" {
                    collect(&path, out)?;
                }
            } else if path.extension().is_some_and(|e| e == "rs") || name == "Cargo.toml" {
                out.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect(Path::new("crates"), &mut files)?;
    for extra in ["Cargo.toml", "Cargo.lock"] {
        let p = PathBuf::from(extra);
        if p.is_file() {
            files.push(p);
        }
    }
    // 遍历顺序随文件系统而变，必须排序后再喂哈希，否则同一棵树能算出不同指纹
    files.sort();

    let mut hasher_input: Vec<u8> = Vec::new();
    for f in &files {
        // 路径也参与：仅哈希内容会让「重命名文件」变成无差别改动
        hasher_input.extend_from_slice(f.to_string_lossy().replace('\\', "/").as_bytes());
        hasher_input.push(0);
        let bytes = std::fs::read(f).with_context(|| format!("读源码失败: {}", f.display()))?;
        hasher_input.extend_from_slice(&bytes.len().to_le_bytes());
        hasher_input.extend_from_slice(&bytes);
        hasher_input.push(0);
    }
    Ok(format!("{:016x}", fnv1a64(&hasher_input)))
}

/// 实验身份的落盘与校验闸门
///
/// 缺陷背景：日志原先只按 `(plan_index, run_idx)` 恢复，**不核验影响结果的配置**，
/// 且 meta 到跑完才写。于是「同路径换配置 + `--resume`」会把旧结果当成新配置的
/// 已完成局，最后还写出一份看着自洽的新 meta。修法是把身份**在开跑前**落盘，
/// 续跑时逐字段比对。
struct IdentityGate;

impl IdentityGate {
    /// 身份文件路径 = `<csv>.identity.json`
    fn path_for(csv: &Path) -> PathBuf {
        let mut s = csv.as_os_str().to_os_string();
        s.push(".identity.json");
        PathBuf::from(s)
    }

    /// 首次跑则写入身份，续跑则校验；返回「身份文件此前是否已存在」
    ///
    /// `journal_exists` 是遗留日志的守门条件：日志在、身份不在，说明这份日志来自
    /// 身份机制之前，来源无法核验。此时必须**在写身份之前**就拒绝——否则这一次写下的
    /// 身份会给那份来路不明的日志盖上章，下一次 `--resume` 就畅通无阻了。
    ///
    /// # 错误
    ///
    /// 遗留日志缺身份，或已有身份与本次身份在任一字段上不一致时报错。
    fn save_or_verify(csv: &Path, identity: &RunIdentity, journal_exists: bool) -> Result<bool> {
        let path = Self::path_for(csv);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("创建输出目录失败: {}", parent.display()))?;
        }
        if !path.is_file() {
            ensure!(
                !journal_exists,
                "续跑日志已存在但没有配套的实验身份: {}；无法核验它是哪次配置跑出来的，请换输出路径重跑",
                Journal::path_for(csv).display()
            );
            std::fs::write(&path, serde_json::to_string_pretty(&identity.json)?)
                .with_context(|| format!("写实验身份失败: {}", path.display()))?;
            return Ok(false);
        }

        let text = std::fs::read_to_string(&path).with_context(|| format!("读实验身份失败: {}", path.display()))?;
        let prev: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("解析实验身份失败: {}", path.display()))?;
        let diffs = Self::diff_keys(&prev, &identity.json);
        ensure!(
            diffs.is_empty(),
            "实验身份与已有记录不一致，拒绝续跑（{}）：{}\n旧身份来自 {}；换配置请换输出路径，不要复用",
            diffs.len(),
            diffs.join("；"),
            path.display()
        );
        Ok(true)
    }

    /// 列出两份身份里取值不同的顶层字段
    ///
    /// 只比顶层：嵌套内容整体参与比较，报错信息里直接给出新旧两个值，
    /// 便于人眼确认到底改了什么。
    fn diff_keys(prev: &serde_json::Value, cur: &serde_json::Value) -> Vec<String> {
        let empty = serde_json::Map::new();
        let pm = prev.as_object().unwrap_or(&empty);
        let cm = cur.as_object().unwrap_or(&empty);
        let mut keys: Vec<&String> = pm.keys().chain(cm.keys()).collect();
        keys.sort_unstable();
        keys.dedup();
        keys.into_iter()
            .filter_map(|k| {
                let a = pm.get(k).unwrap_or(&serde_json::Value::Null);
                let b = cm.get(k).unwrap_or(&serde_json::Value::Null);
                if a == b {
                    None
                } else {
                    Some(format!("{k}: 旧={a} 新={b}"))
                }
            })
            .collect()
    }
}

/// 按命令行构造采样空间：默认第一代，给了 `--shape` 则走分布外
///
/// # 错误
///
/// 见 [`space_from_cli`]。
fn build_space(args: &BenchArgs) -> Result<SamplingSpace> {
    space_from_cli(args.shape.as_deref(), &args.extra_card)
}

/// 一组分数的汇总统计
#[derive(Debug, Clone, Copy)]
struct ScoreStats {
    /// 局数
    games: usize,
    /// 均分
    mean: f64,
    /// 样本标准差（`n-1` 分母）
    stdev: f64,
    /// 均值的标准误
    stderr: f64
}

impl ScoreStats {
    /// 从分数序列汇总
    ///
    /// # 错误
    ///
    /// 序列为空时报错——空集的均分没有意义，静默返回 0 会被误读成「跑了但很差」。
    fn from_scores(scores: &[f64]) -> Result<Self> {
        let games = scores.len();
        if games == 0 {
            bail!("没有可汇总的局");
        }
        let mean = scores.iter().sum::<f64>() / games as f64;
        let stdev = if games < 2 {
            0.0
        } else {
            let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (games - 1) as f64;
            var.sqrt()
        };
        Ok(Self {
            games,
            mean,
            stdev,
            stderr: if games == 0 { 0.0 } else { stdev / (games as f64).sqrt() }
        })
    }
}

/// 单个计划的全部对局结果
/// 一局的完整记录
///
/// 既是最终 CSV 的一行，也是续跑日志的一行。前两个字段是**定位键**：终局 CSV 的
/// `seed` 列是 `rule_master`，不同计划之间并不保证唯一（相邻基种子撞世界的问题
/// 在 `--run-offset` 之前出现过），拿它做续跑键会误跳。
struct GameRow {
    /// 计划下标
    plan_index: usize,
    /// 局号
    run_idx: u64,
    /// 与 [`bench::RESULTS_HEADER`] 同序的单元格
    cells: Vec<String>
}

/// 在 [`bench::RESULTS_HEADER`] 里定位某列
///
/// # 错误
///
/// 列名不存在（表头契约被改动）时报错。
fn column_index(name: &str) -> Result<usize> {
    bench::RESULTS_HEADER
        .iter()
        .position(|&c| c == name)
        .with_context(|| format!("CSV 表头缺少列 {name}"))
}

/// 逐局追加的续跑日志
///
/// bench 原来只在**全部跑完**时才写 CSV，中途停掉等于零产出——先导实验已经因此
/// 丢过一次结果。本类型在每局结束时立刻追加一行并 flush。
///
/// 独立文件而非直接改终局 CSV：终局 CSV 的表头是既有分析脚本的契约，日志要多带
/// `plan_index` / `run_idx` 两列做定位键，不能混进去。
struct Journal {
    /// 追加句柄；多个计划在 rayon 上并发完成，故加锁
    file: Mutex<std::fs::File>
}

impl Journal {
    /// 日志文件路径 = `<csv>.journal.csv`
    fn path_for(csv: &Path) -> PathBuf {
        let mut s = csv.as_os_str().to_os_string();
        s.push(".journal.csv");
        PathBuf::from(s)
    }

    /// 打开（必要时创建）日志，并在 `resume` 时读回已完成的局
    ///
    /// 「日志无身份」由 [`IdentityGate::save_or_verify`] 在更早一步守掉，
    /// 那里必须先于本函数调用。
    ///
    /// # 错误
    ///
    /// 日志已存在但未给 `--resume`、出现重复定位键、目录创建失败，
    /// 或已有日志解析失败时报错。
    fn open(csv: &Path, resume: bool) -> Result<(Self, BTreeMap<(usize, u64), Vec<String>>)> {
        let path = Self::path_for(csv);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("创建输出目录失败: {}", parent.display()))?;
        }
        let existed = path.is_file();
        ensure!(
            !existed || resume,
            "续跑日志已存在: {}；给 --resume 续跑，或先删除该文件重跑（不给就追加会把两次配置混在一起）",
            path.display()
        );

        let mut done: BTreeMap<(usize, u64), Vec<String>> = BTreeMap::new();
        if existed {
            let mut rdr =
                csv::Reader::from_path(&path).with_context(|| format!("读取续跑日志失败: {}", path.display()))?;
            for record in rdr.records() {
                let record = record.with_context(|| format!("解析续跑日志失败: {}", path.display()))?;
                ensure!(
                    record.len() == bench::RESULTS_HEADER.len() + 2,
                    "续跑日志列数 {} 与预期 {} 不符: {}",
                    record.len(),
                    bench::RESULTS_HEADER.len() + 2,
                    path.display()
                );
                let plan_index: usize = record[0].parse().context("续跑日志 plan_index 不是整数")?;
                let run_idx: u64 = record[1].parse().context("续跑日志 run_idx 不是整数")?;
                let cells: Vec<String> = record.iter().skip(2).map(str::to_string).collect();
                // 静默覆盖会让「同一局被写了两遍且结果不同」看不出来
                ensure!(
                    done.insert((plan_index, run_idx), cells).is_none(),
                    "续跑日志出现重复定位键 (计划 {plan_index}, 局 {run_idx}): {}",
                    path.display()
                );
            }
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("打开续跑日志失败: {}", path.display()))?;
        if !existed {
            let mut header: Vec<&str> = vec!["plan_index", "run_idx"];
            header.extend_from_slice(&bench::RESULTS_HEADER);
            writeln!(file, "{}", header.join(","))
                .with_context(|| format!("写续跑日志表头失败: {}", path.display()))?;
            file.flush().context("刷新续跑日志失败")?;
        }
        Ok((
            Self {
                file: Mutex::new(file)
            },
            done
        ))
    }

    /// 追加一局并立刻落盘
    ///
    /// # 错误
    ///
    /// 锁被毒化或写盘失败时报错。
    fn append(&self, row: &GameRow) -> Result<()> {
        let mut cells: Vec<String> = vec![row.plan_index.to_string(), row.run_idx.to_string()];
        cells.extend(row.cells.iter().cloned());
        let mut file = self.file.lock().map_err(|_| anyhow!("续跑日志锁被毒化"))?;
        writeln!(file, "{}", cells.join(","))?;
        file.flush()?;
        Ok(())
    }
}

struct PlanResult {
    /// 计划下标，用于稳定排序
    plan_index: usize,
    /// 该计划的逐局记录（含从续跑日志读回的历史局）
    rows: Vec<GameRow>
}

/// 本进程选定的策略；`nn` 变体持有已加载的模型（Arc 共享，不每局重载）
#[derive(Clone)]
enum SelectedTrainer {
    /// 随机基线
    Random,
    /// 手写规则
    Handwritten,
    /// 手写规则 EXP-006c token 变体（trainer 按 --variant 现场构造，无需 Clone）
    HandwrittenVariant,
    /// 神经网络策略
    #[cfg(feature = "onnx")]
    Nn(RamenNnTrainer),
    /// 扁平搜索（教师本身）；`RamenMctsTrainer` 内含 `Mutex`/原子量不可 `Clone`，
    /// 故这里只带配置，每局现场构造（构造代价只是几个空容器，可忽略）
    Search(SearchSetup)
}

/// `--trainer search` 的完整装配
///
/// 新类型而非在枚举变体里堆字段：rollout 基策与搜索配置合起来才唯一确定
/// 「这是哪个教师」，拆开放很容易只换其中一半。
#[derive(Clone)]
struct SearchSetup {
    /// 搜索配置（`search_n` / `radical_factor_max` 等）
    config: SearchConfig,
    /// rollout 基策用的网络；`None` = 历史口径的手写 rollout
    ///
    /// 模型只在进程启动时加载一次，`Arc` 跨局共享——每局重载会把加载代价乘以局数。
    #[cfg(feature = "onnx")]
    rollout_nn: Option<Arc<RamenNnTrainer>>,
    /// 网络 rollout 的生效回合上限（含）；`None` = 整局
    #[cfg(feature = "onnx")]
    rollout_nn_max_turn: Option<i32>
}

/// 按命令行构造策略；`nn` 在此处加载一次模型
///
/// # 错误
///
/// 未知策略名、缺少 `--model`、未启用 `onnx` feature，或模型加载失败时报错。
fn select_trainer(args: &BenchArgs) -> Result<SelectedTrainer> {
    match args.trainer.as_str() {
        "random" => Ok(SelectedTrainer::Random),
        "handwritten" => Ok(if args.variant.as_deref().is_some_and(|v| v != "base") {
            SelectedTrainer::HandwrittenVariant
        } else {
            SelectedTrainer::Handwritten
        }),
        "nn" => {
            #[cfg(feature = "onnx")]
            {
                let path = args
                    .model
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--trainer nn 需要同时给出 --model <onnx 路径>"))?;
                Ok(SelectedTrainer::Nn(
                    RamenNnTrainer::load(path)?
                        .with_race_shield(!args.no_race_shield)
                        .with_special_mode(parse_special_mode(&args.special_mode)?)
                ))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = (&args.model, &args.special_mode);
                bail!("--trainer nn 需要编译 feature onnx（cargo build --release --features onnx --bin ramen_space_bench）")
            }
        }
        "search" => {
            let config = SearchConfig::default()
                .with_search_n(args.search_n)
                .with_ucb(false)
                .with_radical_factor_max(args.radical_factor_max);
            #[cfg(feature = "onnx")]
            {
                let rollout_nn = match args.rollout_model.as_ref() {
                    Some(path) => Some(Arc::new(
                        RamenNnTrainer::load(path)?
                            .with_race_shield(!args.no_race_shield)
                            .with_special_mode(parse_special_mode(&args.special_mode)?)
                    )),
                    None => None
                };
                Ok(SelectedTrainer::Search(SearchSetup {
                    config,
                    rollout_nn,
                    rollout_nn_max_turn: args.rollout_nn_max_turn
                }))
            }
            #[cfg(not(feature = "onnx"))]
            {
                ensure!(
                    args.rollout_model.is_none(),
                    "--rollout-model 需要编译 feature onnx（cargo build --release --features onnx --bin ramen_space_bench）"
                );
                Ok(SelectedTrainer::Search(SearchSetup { config }))
            }
        }
        other => bail!("未知 trainer: {other}（可选 random / handwritten / nn / search）")
    }
}

/// 跑一个计划的全部对局
///
/// # 错误
///
/// 任一局报错时报错。
fn run_plan(
    plan: &DeckPlan, plan_index: usize, args: &BenchArgs, kind: &SelectedTrainer, journal: Option<&Journal>,
    done: &BTreeMap<(usize, u64), Vec<String>>
) -> Result<PlanResult> {
    let inherit = gen1_inherit();
    // 每个计划用互不重叠的种子段，避免不同计划共用同一批随机世界
    let base_seed = args.seed.wrapping_add((plan_index as u64).wrapping_mul(1_000_003));
    let mut rows = Vec::with_capacity(args.runs_per_plan as usize);
    let run_end = args
        .run_offset
        .checked_add(args.runs_per_plan)
        .context("run_offset + runs_per_plan 溢出 u64")?;
    for run_idx in args.run_offset..run_end {
        // 续跑：日志里已有的局直接复用，不重跑。各局种子互相独立，跳过不影响其余局。
        if let Some(cells) = done.get(&(plan_index, run_idx)) {
            rows.push(GameRow {
                plan_index,
                run_idx,
                cells: cells.clone()
            });
            continue;
        }
        let outcome = match kind {
            SelectedTrainer::Random => {
                let t = LoggingTrainer::new(RandomTrainer, base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            SelectedTrainer::Handwritten => {
                let t = LoggingTrainer::new(RecommendedRamenTrainer::new(), base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            SelectedTrainer::HandwrittenVariant => {
                let tokens = args.variant.as_deref().unwrap_or("base");
                let t = LoggingTrainer::new(RecommendedRamenTrainer::with_tokens(tokens)?, base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            #[cfg(feature = "onnx")]
            SelectedTrainer::Nn(nn) => {
                let t = LoggingTrainer::new(nn.clone(), base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
            SelectedTrainer::Search(setup) => {
                let mcts = RamenMctsTrainer::new(setup.config.clone())
                    .with_stages(RamenSearchStages::all())
                    .with_selection(RamenSelection::Score);
                #[cfg(feature = "onnx")]
                let mcts = match setup.rollout_nn.as_ref() {
                    Some(nn) => mcts.with_nn_rollout(Arc::clone(nn), setup.rollout_nn_max_turn),
                    None => mcts
                };
                let t = LoggingTrainer::new(mcts, base_seed + run_idx);
                bench::run_seeded(plan.uma, &plan.deck, &inherit, base_seed, run_idx, &t)?
            }
        };
        let row = GameRow {
            plan_index,
            run_idx,
            cells: bench::outcome_to_row(plan.shape, &outcome)
        };
        // 先落盘再入内存：进程被杀时已完成的局必须留下来
        if let Some(j) = journal {
            j.append(&row)?;
        }
        rows.push(row);
    }
    Ok(PlanResult { plan_index, rows })
}

/// 按某个键分组汇总并打印
///
/// # 错误
///
/// 任一组为空时报错。
fn print_grouped(title: &str, groups: &BTreeMap<String, Vec<f64>>) -> Result<()> {
    println!("\n{title}");
    println!("  {:<28} {:>6} {:>10} {:>9} {:>8}", "分组", "局数", "均分", "标准差", "标准误");
    for (key, scores) in groups {
        let s = ScoreStats::from_scores(scores)?;
        println!("  {:<28} {:>6} {:>10.0} {:>9.0} {:>8.1}", key, s.games, s.mean, s.stdev, s.stderr);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = BenchArgs::parse();

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    // Y3 地区选择必须交回策略（All 枚举），与 `bench_base` / `ramen_teacher_collect`
    // 同一条前提：教师数据就是在 all 下采的，基准若跟着 toml 走 fixed，测的不再是
    // 同一个分布，而此前记下的全部均分基线会被静默作废。
    //
    // 只改 strategy，不碰 `ramen_region_fixed`：后者仅在 `strategy == Fixed` 时被读
    // （`action.rs::region_select_combos` 与 `game.rs` 的 Y3 分支各一处，都带该守卫），
    // 在 All 下清空它是空操作，写出来只会让人误以为它参与了结果。
    let mut game_config = load_game_config()?;
    if game_config.ramen_region_strategy != RamenRegionStrategy::All {
        println!(
            "已将 ramen_region_strategy 从 {:?} 强制改为 All（基准须与教师数据同分布）",
            game_config.ramen_region_strategy
        );
        game_config.ramen_region_strategy = RamenRegionStrategy::All;
    }
    init_global_with_config(&game_config)?;
    let kind = select_trainer(&args)?;

    let space = build_space(&args)?;
    let all_plans = space.plans();
    ensure!(args.plan_stride >= 1, "--plan-stride 必须 >= 1（当前 {}）", args.plan_stride);
    // 保留**原始下标**：逐计划的基种子是 `seed + plan_index * 1000003`，
    // 抽样后重新编号会换掉整批随机世界，抽样跑与全量跑就不再可配对。
    let mut plans: Vec<(usize, &DeckPlan)> = all_plans
        .iter()
        .enumerate()
        .step_by(args.plan_stride)
        .collect();
    if let Some(n) = args.plans {
        plans.truncate(n);
    }
    // 身份要包含**实际**参与的计划下标，故必须在抽样之后构造
    let plan_indices: Vec<usize> = plans.iter().map(|(i, _)| *i).collect();
    let identity = run_identity(&args, &game_config, &plan_indices)?;
    println!(
        "采样空间基准：{} 个计划 × {} 局（局号 {}..{}）= {} 局，策略 = {}",
        plans.len(),
        args.runs_per_plan,
        args.run_offset,
        args.run_offset + args.runs_per_plan,
        plans.len() as u64 * args.runs_per_plan,
        identity.label
    );

    println!("  基种子 {}", args.seed);
    match &args.shape {
        None => println!("  空间   第一代（与教师数据同分布）"),
        Some(text) => println!(
            "  空间   ❗分布外：构成 {}，追加卡 {:?}——分数不可与默认口径直接比较",
            text, args.extra_card
        )
    }

    // 逐局落盘：只在给了 --csv 时启用（没有落盘目标就没有续跑的意义）。
    // 顺序是刻意的：**先**校验/写入身份，**再**打开日志追加——反过来会先污染日志。
    let (journal, done) = match args.csv.as_ref() {
        Some(path) => {
            IdentityGate::save_or_verify(path, &identity, Journal::path_for(path).is_file())?;
            println!("  实验身份    {}", IdentityGate::path_for(path).display());
            let (j, d) = Journal::open(path, args.resume)?;
            if !d.is_empty() {
                println!("  续跑        日志中已有 {} 局，将跳过", d.len());
            }
            (Some(j), d)
        }
        None => (None, BTreeMap::new())
    };

    let resumed = done.len();
    let start = std::time::Instant::now();
    let mut results: Vec<PlanResult> = plans
        .par_iter()
        .map(|(i, plan)| run_plan(plan, *i, &args, &kind, journal.as_ref(), &done))
        .collect::<Result<Vec<_>>>()?;
    results.sort_by_key(|r| r.plan_index);
    let elapsed = start.elapsed().as_secs_f64();

    let mut all: Vec<f64> = Vec::new();
    let mut by_shape: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut by_uma: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut free_race_fail = 0usize;
    let mut rows: Vec<Vec<String>> = Vec::new();
    // 逐局记录里取分数与守门结果：续跑读回的局没有 `GameOutcome`，只有单元格
    let score_col = column_index("score")?;
    let free_race_col = column_index("free_race_ok")?;
    for r in &results {
        // r.plan_index 是**原始**下标（抽样时刻意保留），故必须回 all_plans 取
        let plan = &all_plans[r.plan_index];
        for row in &r.rows {
            let cell = |i: usize| -> Result<&str> {
                row.cells
                    .get(i)
                    .map(String::as_str)
                    .with_context(|| format!("逐局记录缺少第 {i} 列（计划 {} 局 {}）", row.plan_index, row.run_idx))
            };
            let score: f64 = cell(score_col)?.parse().context("score 列不是数字")?;
            all.push(score);
            by_shape.entry(plan.shape.to_string()).or_default().push(score);
            by_uma.entry(format!("{}", plan.uma)).or_default().push(score);
            // ❗`bench::result_cells` 把这一列写成 `u8::from(bool)`，即 "0" / "1"。
            // 原先在这里比 "true"，导致每一局都被记成未达标、打印出的比例恒为 100%。
            if cell(free_race_col)? != "1" {
                free_race_fail += 1;
            }
            if args.csv.is_some() {
                rows.push(row.cells.clone());
            }
        }
    }

    let overall = ScoreStats::from_scores(&all)?;
    print_grouped("按卡组构成", &by_shape)?;
    print_grouped("按马娘", &by_uma)?;
    println!("\n合计");
    println!("  局数        {}", overall.games);
    println!("  均分        {:.0}", overall.mean);
    println!("  标准差      {:.0}", overall.stdev);
    println!("  均值标准误  {:.1}", overall.stderr);
    println!("  自选比赛未达标  {} 局（{:.2}%）", free_race_fail, 100.0 * free_race_fail as f64 / all.len() as f64);
    println!("  耗时        {elapsed:.1} s");
    // 推理吞吐：评估 GPU 批量收益的唯一合法基准量。搜索本来就在多个 rayon 线程上
    // 并行推理，拿 GPU 满批吞吐去比单线程 tract 微基准会系统性高估收益。
    #[cfg(feature = "onnx")]
    {
        let requests = infer_request_count();
        if requests > 0 {
            println!("  推理请求    {requests} 次");
            println!("  聚合吞吐    {:.1} 次/s", requests as f64 / elapsed);
        }
    }

    if let Some(path) = &args.csv {
        bench::write_csv(path, &bench::RESULTS_HEADER, &rows)?;
        println!("  逐局 CSV    {}", path.display());
        // 实验身份落成 sidecar 而不是 CSV 新列：逐局 CSV 的表头是既有分析脚本
        // 的契约，加列会改动它；而每局的教师身份本来就是常量，放列里也是冗余。
        let meta_path = {
            let mut s = path.as_os_str().to_os_string();
            s.push(".meta.json");
            PathBuf::from(s)
        };
        // 完整性写进身份：中断的阶段绝不能被后来的人当成完整结果读
        let planned = plans.len() as u64 * args.runs_per_plan;
        let mut meta = identity.json.clone();
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("games_planned".into(), serde_json::json!(planned));
            obj.insert("games_completed".into(), serde_json::json!(all.len()));
            obj.insert("games_resumed".into(), serde_json::json!(resumed));
            obj.insert("complete".into(), serde_json::json!(all.len() as u64 == planned));
        }
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .with_context(|| format!("写实验身份失败: {}", meta_path.display()))?;
        println!("  实验身份    {}", meta_path.display());
    }
    Ok(())
}
