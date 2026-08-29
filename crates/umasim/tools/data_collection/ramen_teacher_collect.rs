//! 拉面杯教师数据采集驱动
//!
//! 流程：`sampler → search → export_ramen_sample → 分片落盘 + manifest`。
//!
//! 四条运行时前提由本 bin **硬编码写入** `SearchConfig` / `GAMECONFIG`，并原样记进
//! `manifest.json`。漏任何一条，采出来的数据都不能当教师标签用：
//!
//! 1. `record_ordered_rollouts = true` —— 否则 `export_ramen_sample` 直接报错
//! 2. `use_ucb = false` —— `SearchConfig::default()` 是 true；UCB 会把
//!    `radical_factor` 经样本分配烘进原始数据
//! 3. `radical_factor_max` 必须显式设 —— `SearchConfig::default()` 是 50.0，
//!    游戏配置是 1.4
//! 4. `ramen_region_strategy = all` —— 否则第 3 年地区选择只有单候选
//!
//! 用法：
//! ```text
//! cargo run --release -p umasim --bin ramen_teacher_collect -- \
//!     --count 5 --search-n 8 --output-dir target/ramen_teacher_smoke
//! ```

use std::{
    io::{BufWriter, Write},
    path::{Path, PathBuf}
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use umasim::{
    collector::{FileSignature, compute_file_signature, compute_text_hash_fnv1a64, scan_part_files, try_get_git_commit},
    game::{
        InheritInfo,
        ramen::{
            RamenAction,
            RamenGame,
            RamenStage,
            features::INPUT_DIM,
            policy_schema::POLICY_DIM,
            training_sample::{RamenSampleBatch, RamenTrainingSample, SAMPLE_FORMAT_VERSION}
        }
    },
    gamedata::{GAMECONFIG, RamenRegionStrategy, init_global_with_config},
    sampler::{SampledPosition, SamplerConfig, SamplingSpace, sample_position},
    search::{FlatSearch, SearchConfig},
    utils::{get_workspace_root, init_logger, load_game_config}
};

/// manifest 文件名
const MANIFEST_NAME: &str = "manifest.json";

/// 复现基座里要记签名的数据文件（相对工作空间根）
const GAMEDATA_SIG_PATHS: &[&str] = &[
    "gamedata/constants.json",
    "gamedata/events.json",
    "gamedata/umaDB.json",
    "gamedata/cardDB.json",
    "gamedata/scenario_ramen.json",
    "gamedata/default_config.toml",
    "game_config.toml"
];

// ============================================================================
// CLI
// ============================================================================

/// 拉面杯教师数据采集命令行参数
#[derive(Parser, Debug)]
#[command(name = "ramen_teacher_collect")]
#[command(about = "采样局面 + 扁平搜索，导出拉面杯教师样本并分片落盘")]
struct CollectArgs {
    /// 采样序号总量，**从 `--start` 起算的累计目标**（含未捕获而跳过的）
    ///
    /// 续跑时不是「这次再跑多少」：本次实际区间是
    /// `[manifest.next_index, --start + --count)`。想在已有 6 条的目录上再采 4 条，
    /// 要写 `--count 10` 而不是 `--count 4`。
    #[arg(long)]
    count: u64,

    /// 起始采样序号
    #[arg(long, default_value_t = 0)]
    start: u64,

    /// 每个候选的搜索次数
    #[arg(long)]
    search_n: usize,

    /// 输出目录（相对工作空间根，或绝对路径）
    #[arg(long, default_value = "training_data/ramen_teacher")]
    output_dir: PathBuf,

    /// 每个分片的样本条数
    #[arg(long, default_value_t = 256)]
    shard_size: usize,

    /// 激进度因子最大值（必须显式写入 SearchConfig；游戏配置 1.4，SearchConfig::default 是 50.0）
    #[arg(long, default_value_t = 1.4)]
    radical_factor_max: f64,

    /// 第 2/3 年地区选择采样配额（千分之几），逗号分隔 `Y2,Y3`
    #[arg(long, value_delimiter = ',', num_args = 1, default_value = "20,30")]
    region_quota_permille: Vec<u32>
}

// ============================================================================
// 新类型
// ============================================================================

/// 第 2/3 年地区选择的采样配额（千分之几）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct RegionQuotaPermille {
    /// 第 2 年
    pub y2: u32,
    /// 第 3 年
    pub y3: u32
}

impl RegionQuotaPermille {
    /// 从恰好两个整数构造
    ///
    /// # 错误
    ///
    /// 不是恰好 2 个整数时报错。
    fn from_cli(values: &[u32]) -> Result<Self> {
        ensure!(
            values.len() == 2,
            "--region-quota-permille 需要恰好 2 个整数（Y2,Y3），实得 {}",
            values.len()
        );
        let y2 = *values.first().ok_or_else(|| anyhow!("缺少 Y2 配额"))?;
        let y3 = *values.get(1).ok_or_else(|| anyhow!("缺少 Y3 配额"))?;
        Ok(Self { y2, y3 })
    }

    /// 转成采样器要的 `[Y2, Y3]`
    fn as_array(self) -> [u32; 2] {
        [self.y2, self.y3]
    }
}

/// 本次要处理的采样序号半开区间 `[start, end)`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexSpan {
    /// 含
    pub start: u64,
    /// 不含
    pub end: u64
}

impl IndexSpan {
    /// 区间是否为空
    fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// 已有任务的进度（断点续跑用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexProgress {
    /// 该目录第一次开跑时的 `--start`
    pub index_start: u64,
    /// 下一个尚未处理的序号
    pub next_index: u64
}

/// 四条运行时前提的**实际取值**（不是「已设置」四个字）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TeacherPremises {
    /// 必须为 true，否则 `export_ramen_sample` 报错
    pub record_ordered_rollouts: bool,
    /// 必须为 false，避免 UCB 把 `radical_factor` 烘进样本分配
    pub use_ucb: bool,
    /// 必须显式设置；`SearchConfig::default` 是 50.0，游戏配置是 1.4
    pub radical_factor_max: f64,
    /// 必须为 `all`；否则第 3 年地区选择只有单候选
    pub ramen_region_strategy: RamenRegionStrategy
}

impl TeacherPremises {
    /// 校验四条前提都落在教师采集允许的取值上
    ///
    /// `radical_factor_max` 只要求有限且为正——具体数字由 CLI 显式给出，
    /// 不在这里写死 1.4，以免挡住有意的对照实验。
    ///
    /// # 错误
    ///
    /// 任一条不满足时报错。
    fn check(&self) -> Result<()> {
        ensure!(
            self.record_ordered_rollouts,
            "record_ordered_rollouts 必须为 true，实际 {}",
            self.record_ordered_rollouts
        );
        ensure!(!self.use_ucb, "use_ucb 必须为 false，实际 {}", self.use_ucb);
        ensure!(
            self.radical_factor_max.is_finite() && self.radical_factor_max > 0.0,
            "radical_factor_max 必须是正有限值，实际 {}",
            self.radical_factor_max
        );
        ensure!(
            self.ramen_region_strategy == RamenRegionStrategy::All,
            "ramen_region_strategy 必须为 all，实际 {:?}",
            self.ramen_region_strategy
        );
        Ok(())
    }
}

/// 写入 manifest 的采样器快照（`SamplerConfig` 本身没有 Serialize）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ManifestSamplerConfig {
    /// 轨迹扰动概率
    pub epsilon: f64,
    /// 根局面至少要有的候选动作数
    pub min_actions: usize,
    /// 继承因子
    pub inherit: InheritInfo,
    /// 截断回合上界（含）
    pub max_turn: i32,
    /// 种子基底
    pub seed_base: u64,
    /// 第 2/3 年地区选择配额（千分之几）
    pub region_quota_permille: [u32; 2]
}

impl ManifestSamplerConfig {
    /// 从运行中的采样器配置拍快照
    fn from_sampler(cfg: &SamplerConfig) -> Self {
        Self {
            epsilon: cfg.epsilon,
            min_actions: cfg.min_actions,
            inherit: cfg.inherit.clone(),
            max_turn: cfg.max_turn,
            seed_base: cfg.seed_base,
            region_quota_permille: cfg.region_quota_permille
        }
    }
}

/// 一个已落盘分片的记录
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TeacherPart {
    /// 文件名（如 `part_000000.bin`）
    pub name: String,
    /// 该分片内的样本条数
    pub samples: usize,
    /// 文件签名（复用 collector 的 FNV-1a）
    pub signature: FileSignature
}

/// 教师采集 manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TeacherManifest {
    /// 容器格式版本，对齐 [`SAMPLE_FORMAT_VERSION`]
    pub format_version: u32,
    /// 定长特征维度
    pub input_dim: usize,
    /// policy 格位数
    pub policy_dim: usize,
    /// 四条运行时前提的实际取值
    pub premises: TeacherPremises,
    /// 每个候选的搜索次数
    pub search_n: usize,
    /// 该目录第一次开跑时的起始序号
    pub index_start: u64,
    /// 计划处理到的序号（不含）
    pub index_end: u64,
    /// 下一个尚未处理的序号（断点续跑从这里接着）
    pub next_index: u64,
    /// 采样器配置快照
    pub sampler: ManifestSamplerConfig,
    /// 分片大小（新分片用当前值，续跑允许改）
    pub shard_size: usize,
    /// 已落盘分片
    pub parts: Vec<TeacherPart>,
    /// 首次开跑时间
    pub started_at: String,
    /// 最近一次写盘时间
    pub updated_at: String,
    /// 整段任务跑完的时间；未完成则为 `None`
    pub finished_at: Option<String>,
    /// 采样返回 `None`（未捕获）的次数，累计
    pub skipped_uncaptured: u64,
    /// 已落盘样本条数，累计
    pub accepted: u64,
    /// git HEAD，取不到则为 `None`
    pub git_commit: Option<String>,
    /// 复现基座文件签名
    pub gamedata_sig: Vec<FileSignature>,
    /// 生效配方（前提 + 采样器 + search_n + 维度）的 FNV-1a
    pub recipe_hash_fnv1a64: String
}

impl TeacherManifest {
    /// 从路径读取
    ///
    /// # 错误
    ///
    /// 读文件或 JSON 解析失败时报错。
    fn load(path: &Path) -> Result<Self> {
        let text = fs_err::read_to_string(path).with_context(|| format!("读取 manifest 失败: {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("解析 manifest 失败: {}", path.display()))
    }

    /// 原子替换写入（Windows 下先删再 rename，与 collector 同一套）
    ///
    /// # 错误
    ///
    /// 写临时文件、删旧文件或 rename 失败时报错。
    fn save_replace(&self, path: &Path) -> Result<()> {
        save_json_replace(path, self)
    }
}

/// 不可变配方：续跑时必须与目录里已有的一致
#[derive(Debug, Clone, PartialEq, Serialize)]
struct CollectRecipe {
    /// 容器格式版本
    format_version: u32,
    /// 特征维度
    input_dim: usize,
    /// policy 维度
    policy_dim: usize,
    /// 四条前提
    premises: TeacherPremises,
    /// 搜索次数
    search_n: usize,
    /// 采样器快照
    sampler: ManifestSamplerConfig
}

impl CollectRecipe {
    /// 配方哈希，便于一眼看出两批数据是否能拼
    fn hash(&self) -> Result<String> {
        let text = serde_json::to_string(self).context("序列化采集配方失败")?;
        Ok(compute_text_hash_fnv1a64(&text))
    }
}

// ============================================================================
// 搜索配置 / 动作表
// ============================================================================

/// 教师采集用的搜索配置：三条搜索侧前提全部显式写入，不依赖 `Default`
fn teacher_search_config(search_n: usize, radical_factor_max: f64) -> SearchConfig {
    SearchConfig::default()
        .with_search_n(search_n)
        .with_ucb(false)
        .with_record_ordered_rollouts(true)
        .with_radical_factor_max(radical_factor_max)
}

/// 本决策点交给搜索的动作表
///
/// `RamenSelect` 走合并决策路径（`list_combined_ramen_select_actions`），
/// 其余阶段用采样器捕获的 `pos.actions`。
fn actions_for_search(pos: &SampledPosition) -> Vec<RamenAction> {
    if pos.stage == RamenStage::RamenSelect {
        pos.game.list_combined_ramen_select_actions()
    } else {
        pos.actions.clone()
    }
}

/// 分片文件名，6 位数字，与 [`scan_part_files`] 的识别规则一致
fn part_file_name(index: usize) -> String {
    format!("part_{:06}.bin", index)
}

/// 根据 CLI 与已有进度计算本次要处理的 `[start, end)`
///
/// 已有任务时，`--start` 必须等于原任务起点或当前 `next_index`：
/// - 等于原起点：把 `--count` 当成「从原起点起的总长度」（同一条命令续跑 / 拉长）
/// - 等于 `next_index`：把 `--count` 当成「从断点再往前走多少」
///
/// # 错误
///
/// `--start` 对不上，或 `start + count` 溢出时报错。
fn plan_index_span(cli_start: u64, cli_count: u64, existing: Option<IndexProgress>) -> Result<IndexSpan> {
    let end = cli_start
        .checked_add(cli_count)
        .ok_or_else(|| anyhow!("start + count 溢出: {cli_start} + {cli_count}"))?;
    let Some(progress) = existing else {
        return Ok(IndexSpan {
            start: cli_start,
            end
        });
    };
    if cli_start != progress.index_start && cli_start != progress.next_index {
        bail!(
            "输出目录已有采集任务 index_start={} next_index={}，本次 --start {} 对不上。换目录，或把 --start 设为 {} 或 {}",
            progress.index_start,
            progress.next_index,
            cli_start,
            progress.index_start,
            progress.next_index
        );
    }
    Ok(IndexSpan {
        start: progress.next_index,
        end
    })
}

/// 续跑时核对配方没有被改掉
///
/// # 错误
///
/// 格式版本、维度、前提、search_n 或采样器配置不一致时报错。
fn ensure_resume_compatible(old: &TeacherManifest, recipe: &CollectRecipe) -> Result<()> {
    ensure!(
        old.format_version == recipe.format_version,
        "format_version 不一致: manifest {} vs 当前 {}",
        old.format_version,
        recipe.format_version
    );
    ensure!(
        old.input_dim == recipe.input_dim,
        "INPUT_DIM 不一致: manifest {} vs 当前 {}",
        old.input_dim,
        recipe.input_dim
    );
    ensure!(
        old.policy_dim == recipe.policy_dim,
        "POLICY_DIM 不一致: manifest {} vs 当前 {}",
        old.policy_dim,
        recipe.policy_dim
    );
    ensure!(
        old.premises == recipe.premises,
        "四条运行时前提不一致:\n  manifest {:?}\n  当前 {:?}",
        old.premises,
        recipe.premises
    );
    ensure!(
        old.search_n == recipe.search_n,
        "search_n 不一致: manifest {} vs 当前 {}",
        old.search_n,
        recipe.search_n
    );
    ensure!(
        old.sampler == recipe.sampler,
        "采样器配置不一致:\n  manifest {:?}\n  当前 {:?}",
        old.sampler,
        recipe.sampler
    );
    Ok(())
}

// ============================================================================
// 落盘
// ============================================================================

/// 原子替换写入 JSON（Windows 下先删再 rename）
///
/// # 错误
///
/// 创建临时文件、序列化、删除旧文件或 rename 失败时报错。
fn save_json_replace(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    let file = fs_err::File::create(&tmp_path)
        .with_context(|| format!("创建临时 manifest 失败: {}", tmp_path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).context("写入 manifest JSON 失败")?;
    writer.flush().context("flush manifest 失败")?;
    writer.get_ref().sync_all().ok();
    if path.exists() {
        fs_err::remove_file(path).with_context(|| format!("删除旧 manifest 失败: {}", path.display()))?;
    }
    fs_err::rename(&tmp_path, path)
        .with_context(|| format!("重命名 manifest 失败: {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

/// 采集复现基座文件的签名
///
/// # 错误
///
/// 文件存在但读元信息 / 内容失败时报错。缺失的文件跳过。
fn collect_gamedata_signatures() -> Result<Vec<FileSignature>> {
    let mut out = Vec::new();
    for rel in GAMEDATA_SIG_PATHS {
        let path = Path::new(rel);
        if !path.exists() {
            continue;
        }
        let hash = match fs_err::metadata(path) {
            Ok(m) => m.len() <= 32 * 1024 * 1024,
            Err(_) => false
        };
        out.push(compute_file_signature(path, hash)?);
    }
    Ok(out)
}

/// 把当前批次写成一个分片
///
/// # 错误
///
/// 批次为空、目标文件已存在、写盘或签名失败时报错。
fn flush_shard(
    batch: &mut RamenSampleBatch, output_dir: &Path, part_index: usize
) -> Result<TeacherPart> {
    ensure!(!batch.is_empty(), "不能写空分片");
    let name = part_file_name(part_index);
    let final_path = output_dir.join(&name);
    ensure!(
        !final_path.exists(),
        "part 文件已存在，疑似续跑下标算错: {}",
        final_path.display()
    );
    let tmp_path = PathBuf::from(format!("{}.tmp", final_path.display()));
    batch
        .save_binary(&tmp_path)
        .with_context(|| format!("写临时分片失败: {}", tmp_path.display()))?;
    fs_err::rename(&tmp_path, &final_path)
        .with_context(|| format!("重命名分片失败: {} -> {}", tmp_path.display(), final_path.display()))?;
    let samples = batch.len();
    *batch = RamenSampleBatch::new();
    let signature = compute_file_signature(&final_path, true)?;
    Ok(TeacherPart {
        name,
        samples,
        signature
    })
}

/// 读回已写分片，条数必须与 manifest 一致
///
/// # 错误
///
/// 文件缺失、反序列化失败、或条数对不上时报错。
fn verify_written_parts(output_dir: &Path, parts: &[TeacherPart]) -> Result<usize> {
    let mut total = 0usize;
    for part in parts {
        let path = output_dir.join(&part.name);
        let batch = RamenSampleBatch::load_binary(&path)
            .with_context(|| format!("读回分片失败: {}", path.display()))?;
        ensure!(
            batch.len() == part.samples,
            "分片 {} 条数对不上: 文件 {} vs manifest {}",
            part.name,
            batch.len(),
            part.samples
        );
        total += batch.len();
        println!(
            "  读回 {} : {} 条，{} 字节",
            part.name,
            batch.len(),
            part.signature.size
        );
    }
    Ok(total)
}

/// 确认磁盘上的 `part_*.bin` 与 manifest 登记的名单一致
///
/// # 错误
///
/// 多文件、少文件或名单对不上时报错。
fn ensure_parts_match_disk(output_dir: &Path, parts: &[TeacherPart]) -> Result<()> {
    let on_disk = scan_part_files(output_dir)?;
    let names_on_disk: Vec<String> = on_disk.iter().map(|(idx, _)| part_file_name(*idx)).collect();
    let names_in_manifest: Vec<String> = parts.iter().map(|p| p.name.clone()).collect();
    ensure!(
        names_on_disk == names_in_manifest,
        "磁盘分片与 manifest 不一致:\n  disk: {names_on_disk:?}\n  manifest: {names_in_manifest:?}"
    );
    Ok(())
}

// ============================================================================
// 主流程
// ============================================================================

/// 采一条已捕获局面：搜 → 导出
///
/// # 错误
///
/// 动作表为空、搜索失败或导出失败时报错。未捕获由调用方在进本函数之前跳过。
fn collect_one(
    search: &FlatSearch<RamenGame>, pos: &SampledPosition, index: u64
) -> Result<RamenTrainingSample> {
    let actions = actions_for_search(pos);
    ensure!(
        !actions.is_empty(),
        "index={index} stage={:?} 动作表为空，无法搜索",
        pos.stage
    );
    let mut rng = pos.decision_rng.clone();
    let output = search
        .search(&pos.game, &actions, &mut rng)
        .with_context(|| format!("index={index} stage={:?} 搜索失败", pos.stage))?;
    output
        .export_ramen_sample(&pos.game, &pos.stage, index)
        .with_context(|| format!("index={index} 导出教师样本失败"))
}

fn main() -> Result<()> {
    let args = CollectArgs::parse();
    ensure!(args.count > 0, "--count 必须 > 0");
    ensure!(args.search_n > 0, "--search-n 必须 > 0");
    ensure!(args.shard_size > 0, "--shard-size 必须 > 0");

    let quota = RegionQuotaPermille::from_cli(&args.region_quota_permille)?;
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    init_logger("ramen_teacher_collect", "error")?;

    let mut game_config = load_game_config()?;
    if game_config.ramen_region_strategy != RamenRegionStrategy::All {
        println!(
            "已将 ramen_region_strategy 从 {:?} 强制改为 All（教师采集第 3 年必须枚举全部组合）",
            game_config.ramen_region_strategy
        );
        game_config.ramen_region_strategy = RamenRegionStrategy::All;
    }
    init_global_with_config(&game_config)?;

    let strategy = GAMECONFIG
        .get()
        .ok_or_else(|| anyhow!("GAMECONFIG 未初始化"))?
        .ramen_region_strategy;
    let search_cfg = teacher_search_config(args.search_n, args.radical_factor_max);
    let premises = TeacherPremises {
        record_ordered_rollouts: search_cfg.record_ordered_rollouts,
        use_ucb: search_cfg.use_ucb,
        radical_factor_max: search_cfg.radical_factor_max,
        ramen_region_strategy: strategy
    };
    premises.check()?;
    if (premises.radical_factor_max - 50.0).abs() < 1e-12 {
        println!(
            "警告: radical_factor_max=50.0 是 SearchConfig::default，排名加权有效样本约 40。游戏配置是 1.4。"
        );
    }

    let mut sampler_cfg = SamplerConfig::default();
    sampler_cfg.region_quota_permille = quota.as_array();
    let sampler_snap = ManifestSamplerConfig::from_sampler(&sampler_cfg);
    let recipe = CollectRecipe {
        format_version: SAMPLE_FORMAT_VERSION,
        input_dim: INPUT_DIM,
        policy_dim: POLICY_DIM,
        premises: premises.clone(),
        search_n: args.search_n,
        sampler: sampler_snap.clone()
    };
    let recipe_hash = recipe.hash()?;

    let output_dir = args.output_dir.clone();
    if output_dir.exists() {
        ensure!(
            output_dir.is_dir(),
            "输出路径存在但不是目录: {}",
            output_dir.display()
        );
    } else {
        fs_err::create_dir_all(&output_dir)
            .with_context(|| format!("创建输出目录失败: {}", output_dir.display()))?;
    }
    let manifest_path = output_dir.join(MANIFEST_NAME);

    let now = Utc::now().to_rfc3339();
    let (mut manifest, span) = if manifest_path.exists() {
        let old = TeacherManifest::load(&manifest_path)?;
        ensure_resume_compatible(&old, &recipe)?;
        ensure_parts_match_disk(&output_dir, &old.parts)?;
        let progress = IndexProgress {
            index_start: old.index_start,
            next_index: old.next_index
        };
        let span = plan_index_span(args.start, args.count, Some(progress))?;
        println!(
            "断点续跑: index_start={} next_index={} → 本次 [{}, {})",
            old.index_start, old.next_index, span.start, span.end
        );
        // 空区间说明 --count 已被此前的运行跑完（它是累计目标，不是增量）。
        // 此时必须原样退出：继续往下会把一个已完成任务的 finished_at 抹成 null，
        // 让后续消费方误以为数据集只采了一半。
        if span.is_empty() {
            bail!(
                "本次区间为空：--start {} --count {} 表示累计跑到序号 {}，而该目录已经跑到 {}。\
                 想继续采就把 --count 调大（例如 --count {}），manifest 未改动。",
                args.start,
                args.count,
                span.end,
                old.next_index,
                old.next_index - args.start + args.count
            );
        }
        let mut m = old;
        if span.end > m.index_end {
            m.index_end = span.end;
        }
        m.shard_size = args.shard_size;
        m.updated_at = now;
        m.finished_at = None;
        (m, span)
    } else {
        let extra = scan_part_files(&output_dir)?;
        ensure!(
            extra.is_empty(),
            "输出目录没有 manifest 但已有分片 {:?}，拒绝覆盖。换目录或删掉这些文件。",
            extra.iter().map(|(_, p)| p.display().to_string()).collect::<Vec<_>>()
        );
        let span = plan_index_span(args.start, args.count, None)?;
        let manifest = TeacherManifest {
            format_version: SAMPLE_FORMAT_VERSION,
            input_dim: INPUT_DIM,
            policy_dim: POLICY_DIM,
            premises: premises.clone(),
            search_n: args.search_n,
            index_start: args.start,
            index_end: span.end,
            next_index: span.start,
            sampler: sampler_snap,
            shard_size: args.shard_size,
            parts: Vec::new(),
            started_at: now.clone(),
            updated_at: now,
            finished_at: None,
            skipped_uncaptured: 0,
            accepted: 0,
            git_commit: try_get_git_commit(&workspace_root),
            gamedata_sig: collect_gamedata_signatures()?,
            recipe_hash_fnv1a64: recipe_hash
        };
        (manifest, span)
    };
    manifest.save_replace(&manifest_path)?;

    println!("=== 拉面杯教师采集 ===");
    println!("  输出目录              : {}", output_dir.display());
    println!("  index 区间            : [{}, {})", span.start, span.end);
    println!("  search_n              : {}", args.search_n);
    println!("  shard_size            : {}", args.shard_size);
    println!("  record_ordered_rollouts = {}", premises.record_ordered_rollouts);
    println!("  use_ucb                 = {}", premises.use_ucb);
    println!("  radical_factor_max      = {}", premises.radical_factor_max);
    println!("  ramen_region_strategy   = {:?}", premises.ramen_region_strategy);
    println!(
        "  region_quota_permille    = {:?}",
        sampler_cfg.region_quota_permille
    );
    println!("  INPUT_DIM / POLICY_DIM  = {INPUT_DIM} / {POLICY_DIM}");
    println!("  format_version          = {SAMPLE_FORMAT_VERSION}");

    if span.is_empty() {
        println!("区间为空（可能已经采完），只做读回校验。");
        let total = verify_written_parts(&output_dir, &manifest.parts)?;
        println!("已有样本 {total} 条，跳过 {} 次。", manifest.skipped_uncaptured);
        return Ok(());
    }

    let space = SamplingSpace::gen1()?;
    let search: FlatSearch<RamenGame> = FlatSearch::new(search_cfg);
    let mut batch = RamenSampleBatch::new();
    let mut next_part_index = manifest.parts.len();

    for index in span.start..span.end {
        match sample_position(&space, &sampler_cfg, index)?.into_captured() {
            None => {
                manifest.skipped_uncaptured += 1;
                println!("  index={index} 跳过（未捕获）");
            }
            Some(pos) => {
                let sample = collect_one(&search, &pos, index)?;
                println!(
                    "  index={index} turn={} stage={:?} 候选 {}",
                    pos.turn,
                    pos.stage,
                    sample.candidates.len()
                );
                batch.push(sample);
                if batch.len() >= args.shard_size {
                    let part = flush_shard(&mut batch, &output_dir, next_part_index)?;
                    println!("  写入 {} ({} 条)", part.name, part.samples);
                    manifest.parts.push(part);
                    manifest.accepted = manifest.parts.iter().map(|p| p.samples as u64).sum();
                    next_part_index += 1;
                    manifest.next_index = index + 1;
                    manifest.updated_at = Utc::now().to_rfc3339();
                    manifest.save_replace(&manifest_path)?;
                }
            }
        }
        manifest.next_index = index + 1;
    }

    if !batch.is_empty() {
        let part = flush_shard(&mut batch, &output_dir, next_part_index)?;
        println!("  写入 {} ({} 条)", part.name, part.samples);
        manifest.parts.push(part);
        manifest.accepted = manifest.parts.iter().map(|p| p.samples as u64).sum();
        manifest.updated_at = Utc::now().to_rfc3339();
        manifest.save_replace(&manifest_path)?;
    }

    let finished = Utc::now().to_rfc3339();
    manifest.updated_at = finished.clone();
    if manifest.next_index >= manifest.index_end {
        manifest.finished_at = Some(finished);
    }
    manifest.save_replace(&manifest_path)?;

    println!("=== 读回校验 ===");
    let total = verify_written_parts(&output_dir, &manifest.parts)?;
    ensure!(
        total as u64 == manifest.accepted,
        "accepted ({}) 与读回条数 ({total}) 不一致",
        manifest.accepted
    );
    println!(
        "完成: 接受 {} 条，跳过 {} 次，分片 {} 个，next_index={}",
        manifest.accepted,
        manifest.skipped_uncaptured,
        manifest.parts.len(),
        manifest.next_index
    );
    println!("manifest: {}", manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 新类型原则：配额必须恰好两个数
    #[test]
    fn test_region_quota_from_cli() -> Result<()> {
        match RegionQuotaPermille::from_cli(&[20, 30]) {
            Ok(q) => {
                println!("  [OK] [20,30] → y2={} y3={} array={:?}", q.y2, q.y3, q.as_array());
                if q.as_array() != [20, 30] {
                    bail!("as_array 不是 [20, 30]");
                }
            }
            Err(e) => bail!("合法输入不该失败: {e}")
        }
        match RegionQuotaPermille::from_cli(&[20]) {
            Ok(q) => bail!("单元素不该成功: {q:?}"),
            Err(e) => println!("  [OK] 单元素报错: {e}")
        }
        match RegionQuotaPermille::from_cli(&[]) {
            Ok(q) => bail!("空切片不该成功: {q:?}"),
            Err(e) => println!("  [OK] 空切片报错: {e}")
        }
        Ok(())
    }

    /// 分片文件名必须是 6 位数字，才能被 collector::scan_part_files 认出来
    #[test]
    fn test_part_file_name() -> Result<()> {
        let a = part_file_name(0);
        let b = part_file_name(12);
        println!("  part 0 → {a}");
        println!("  part 12 → {b}");
        if a != "part_000000.bin" {
            bail!("part 0 应为 part_000000.bin，实得 {a}");
        }
        if b != "part_000012.bin" {
            bail!("part 12 应为 part_000012.bin，实得 {b}");
        }
        Ok(())
    }

    /// 新开跑 / 同命令续跑 / 从 next 接着 / start 对不上 / 已经采完
    #[test]
    fn test_plan_index_span() -> Result<()> {
        let fresh = plan_index_span(0, 5, None)?;
        println!("  新开跑 [0,5) → [{}, {})", fresh.start, fresh.end);
        if fresh != (IndexSpan { start: 0, end: 5 }) {
            bail!("新开跑区间不对");
        }

        let same_cmd = plan_index_span(
            0,
            5,
            Some(IndexProgress {
                index_start: 0,
                next_index: 3
            })
        )?;
        println!("  同命令续跑 next=3 → [{}, {})", same_cmd.start, same_cmd.end);
        if same_cmd != (IndexSpan { start: 3, end: 5 }) {
            bail!("同命令续跑应从 3 到 5");
        }

        let from_next = plan_index_span(
            5,
            5,
            Some(IndexProgress {
                index_start: 0,
                next_index: 5
            })
        )?;
        println!("  从 next 再走 5 → [{}, {})", from_next.start, from_next.end);
        if from_next != (IndexSpan { start: 5, end: 10 }) {
            bail!("从 next 续跑应从 5 到 10");
        }

        let done = plan_index_span(
            0,
            5,
            Some(IndexProgress {
                index_start: 0,
                next_index: 5
            })
        )?;
        println!("  已采完 → [{}, {}) empty={}", done.start, done.end, done.is_empty());
        if !done.is_empty() {
            bail!("已采完应得到空区间");
        }

        match plan_index_span(
            100,
            5,
            Some(IndexProgress {
                index_start: 0,
                next_index: 5
            })
        ) {
            Ok(s) => bail!("start 对不上不该成功: {s:?}"),
            Err(e) => println!("  [OK] start 对不上: {e}")
        }
        Ok(())
    }

    /// 搜索配置必须把三条搜索侧前提写成教师采集要求的值
    #[test]
    fn test_teacher_search_config_premises() -> Result<()> {
        let cfg = teacher_search_config(8, 1.4);
        println!(
            "  record_ordered_rollouts={} use_ucb={} radical_factor_max={} search_n={}",
            cfg.record_ordered_rollouts, cfg.use_ucb, cfg.radical_factor_max, cfg.search_n
        );
        if !cfg.record_ordered_rollouts {
            bail!("record_ordered_rollouts 应为 true");
        }
        if cfg.use_ucb {
            bail!("use_ucb 应为 false");
        }
        if (cfg.radical_factor_max - 1.4).abs() > 1e-12 {
            bail!("radical_factor_max 应为 1.4");
        }
        if cfg.search_n != 8 {
            bail!("search_n 应为 8");
        }
        // Default 对照：确认我们没有误用 Default 的 50.0 / true / false
        let def = SearchConfig::default();
        println!(
            "  Default: record_ordered_rollouts={} use_ucb={} radical_factor_max={}",
            def.record_ordered_rollouts, def.use_ucb, def.radical_factor_max
        );
        if def.record_ordered_rollouts || !def.use_ucb || (def.radical_factor_max - 50.0).abs() > 1e-12 {
            bail!("SearchConfig::default 的前提变了，教师采集的硬编码需要复查");
        }
        Ok(())
    }

    /// 前提校验拒绝错误取值；manifest JSON 必须露出实际数字 / all
    #[test]
    fn test_premises_check_and_json() -> Result<()> {
        let ok = TeacherPremises {
            record_ordered_rollouts: true,
            use_ucb: false,
            radical_factor_max: 1.4,
            ramen_region_strategy: RamenRegionStrategy::All
        };
        ok.check()?;
        let json = serde_json::to_string_pretty(&ok)?;
        println!("  premises JSON:\n{json}");
        if !json.contains("\"record_ordered_rollouts\": true") {
            bail!("JSON 里看不到 record_ordered_rollouts = true");
        }
        if !json.contains("\"use_ucb\": false") {
            bail!("JSON 里看不到 use_ucb = false");
        }
        if !json.contains("1.4") {
            bail!("JSON 里看不到 radical_factor_max 的实际值 1.4");
        }
        if !json.contains("\"all\"") {
            bail!("JSON 里看不到 ramen_region_strategy = all");
        }

        let bad_ucb = TeacherPremises {
            use_ucb: true,
            ..ok.clone()
        };
        match bad_ucb.check() {
            Ok(()) => bail!("use_ucb=true 应被拒绝"),
            Err(e) => println!("  [OK] use_ucb=true: {e}")
        }
        let bad_region = TeacherPremises {
            ramen_region_strategy: RamenRegionStrategy::Fixed,
            ..ok
        };
        match bad_region.check() {
            Ok(()) => bail!("strategy=fixed 应被拒绝"),
            Err(e) => println!("  [OK] strategy=fixed: {e}")
        }
        Ok(())
    }

    /// RamenSelect 才走合并动作，其余阶段用 pos.actions——用阶段枚举钉死分支条件
    #[test]
    fn test_combined_only_on_ramen_select() -> Result<()> {
        let combined = |s: &RamenStage| *s == RamenStage::RamenSelect;
        let stages = [
            RamenStage::RamenSelect,
            RamenStage::SpecialSelect,
            RamenStage::Train,
            RamenStage::SuperRamenSelect,
            RamenStage::RegionSelect
        ];
        for s in &stages {
            println!("  {s:?} → combined={}", combined(s));
        }
        if !combined(&RamenStage::RamenSelect) {
            bail!("RamenSelect 必须走合并动作");
        }
        if stages.iter().filter(|s| combined(s)).count() != 1 {
            bail!("只有 RamenSelect 走合并动作");
        }
        Ok(())
    }
}
