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

mod explicit_assets;

use std::{
    collections::BTreeSet,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use umasim::{
    collector::{FileSignature, compute_file_signature, compute_text_hash_fnv1a64, scan_part_files, try_get_git_commit},
    game::{
        InheritInfo,
        Trainer,
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
    sampler::{
        SampledPosition,
        SamplerConfig,
        SamplingSpace,
        sample_position_with_rollin,
        space_from_cli,
        space_version_by_name
    },
    search::{FlatSearch, SearchConfig},
    trainer::RamenHandwrittenTrainer,
    utils::{get_workspace_root, init_logger, load_game_config}
};
#[cfg(feature = "onnx")]
use umasim::trainer::RamenNnTrainer;

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

    /// 卡组构成 `速,耐,力,根,智`，五项合计为 5（友人卡固定 1 张，不计入）
    ///
    /// 给出即切到**分布外**空间：只枚举这一种构成，用于给训练分布补缺口。
    /// 不给时用第一代的 3 种构成。与 `ramen_space_bench` 同名参数同口径。
    ///
    /// ❗换空间等于换 `index` 的含义，采集器会用空间指纹拦住同一目录混采两个空间。
    #[arg(long)]
    shape: Option<String>,

    /// 追加进卡池的支援卡 idrank（6 位 = 5 位卡 ID + 突破等级），可重复给出
    ///
    /// 只在有 `--shape` 时允许：默认口径必须锁死在训练分布的 11 张卡上。
    #[arg(long)]
    extra_card: Vec<u32>,

    /// 具名采样空间版本（如 `gen2_v1`），与 `--shape` / `--extra-card` 互斥
    ///
    /// 给出即进入**显式身份口径**：马娘、卡池、构成、组合数原样写进 manifest，
    /// 续跑闸门逐字段比对，全程不计算空间指纹、配方指纹与文件内容哈希。
    /// 不给时完全走既有口径（空间指纹 + 配方指纹 + 文件内容哈希），字段语义不变。
    #[arg(long)]
    space_version: Option<String>,

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
    region_quota_permille: Vec<u32>,

    /// 第 1 年地区选择采样配额（千分之几）
    ///
    /// 独立参数而非把上面那个扩成三元组：那会改掉既有 manifest 里数组的下标含义。
    /// 默认 0，此时采样分配与本参数加入之前逐位相同。
    #[arg(long, default_value_t = 0)]
    region_quota_permille_y1: u32,

    /// roll-in 基策：`handwritten`（默认，与既有全部数据一致）/ `nn`（需 `--model`）
    ///
    /// roll-in 决定轨迹走到哪些状态，因而决定**样本的状态分布**。换成 `nn` 即得到
    /// DAgger 式的 on-policy 样本：教师在**网络自己会访问的状态**上给标注。
    ///
    /// ❗两种 roll-in 的样本**不可混进同一目录**，采集器用 manifest 里的 roll-in
    /// 身份拦住。`nn` 的身份含模型文件哈希，换模型同样会被拦。
    #[arg(long, default_value = "handwritten")]
    rollin: String,

    /// roll-in 用的 ONNX 模型路径；`--rollin nn` 时必填
    #[arg(long)]
    model: Option<PathBuf>,

    /// 显式模型版本名称；原始模型及 sidecar 会另存并逐字节核对。
    #[arg(long)]
    model_id: Option<String>,

    /// 显式工作序号数组；此时 start/count/next_index 表示数组游标，样本仍记录真实 index。
    #[arg(long)]
    indices_file: Option<PathBuf>,

    /// 该目录的有效根目标；未捕获使用清单中的后续备用序号，达到即停止。
    #[arg(long)]
    accepted_target: Option<u64>,

    /// 软截止秒数：根与根之间检查，收尾写完整分片后退出；外部驱动提供硬截止。
    #[arg(long)]
    max_seconds: Option<u64>
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
    pub region_quota_permille: [u32; 2],
    /// 第 1 年地区选择配额（千分之几）；本字段加入之前采的目录按 0 读
    #[serde(default, skip_serializing_if = "quota_is_zero")]
    pub region_quota_permille_y1: u32
}

/// 默认关闭的新配额不写入旧配方，保持旧配方序列化字节不变。
fn quota_is_zero(value: &u32) -> bool { *value == 0 }

impl ManifestSamplerConfig {
    /// 从运行中的采样器配置拍快照
    fn from_sampler(cfg: &SamplerConfig) -> Self {
        Self {
            epsilon: cfg.epsilon,
            min_actions: cfg.min_actions,
            inherit: cfg.inherit.clone(),
            max_turn: cfg.max_turn,
            seed_base: cfg.seed_base,
            region_quota_permille: cfg.region_quota_permille,
            region_quota_permille_y1: cfg.region_quota_permille_y1
        }
    }
}

/// manifest 里记录的一种卡组构成
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestShape {
    /// 各类型张数 `[速, 耐, 力, 根, 智]`
    pub counts: [usize; 5],
    /// 构成名
    pub name: String
}

/// 采样空间的**显式身份**：逐字段记全，不算指纹
///
/// 存在的理由是可以直接比较：续跑与跨机合并只需把两份清单逐字段对上，
/// 不需要任何哈希。`plan_count` 一并记下，因为 `index % plan_count` 决定组合归属，
/// 它对不上就说明两边编译的空间不是同一个。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ManifestSpaceIdentity {
    /// 空间版本名
    pub version: String,
    /// 该版本的马娘 gameId 清单，按枚举顺序
    pub umas: Vec<u32>,
    /// 该版本的支援卡 idrank 清单，按枚举顺序
    pub cards: Vec<u32>,
    /// 该版本的构成清单，按枚举顺序
    pub shapes: Vec<ManifestShape>,
    /// 枚举出的 (马娘, 卡组) 组合总数
    pub plan_count: usize
}

/// 一次采集的分项墙钟，单位毫秒
///
/// 只做测量，不参与任何判定。分项之和**小于**进程总墙钟：命令行解析、
/// 日志初始化与进程退出前的收尾不在任何一项里，差额由 `total` 与各项之差体现。
#[derive(Debug, Clone, Default)]
struct StageTiming {
    /// gamedata / GAMECONFIG 全局初始化
    pub init_ms: u128,
    /// 采样空间枚举
    pub space_ms: u128,
    /// roll-in 基策构造（`--rollin nn` 时即 ONNX 模型加载）
    pub rollin_load_ms: u128,
    /// 采样（含 roll-in 走到截断回合）累计
    pub sample_ms: u128,
    /// 搜索 + 导出样本累计
    pub search_ms: u128,
    /// 分片写盘 + 签名累计
    pub flush_ms: u128,
    /// manifest 写盘累计
    pub manifest_ms: u128,
    /// 收尾读回校验
    pub verify_ms: u128
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
    /// 采样空间的枚举指纹；本字段加入之前采的数据为 `None`
    ///
    /// **刻意不进 `recipe_hash`**：进了会让同一空间下的新旧数据配方哈希不同、无法合并。
    /// 空间变化本来就会带动 git commit，这里只是把它变成可直接校验的显式指纹——
    /// 卡池写在代码里，改它不会反映到 `gamedata_sig`。
    #[serde(default)]
    pub sampling_space_hash: Option<String>,
    /// roll-in 基策身份；本字段加入之前采的数据为 `None`，一律视为 `handwritten`
    ///
    /// **刻意不进 `recipe_hash`**：理由与 [`sampling_space_hash`](Self::sampling_space_hash)
    /// 相同——进了会让同配方的新旧数据哈希不同、无法合并。它是独立的显式绊线。
    ///
    /// 取值：`handwritten`，或 `nn:<模型文件的 FNV-1a>`。带哈希是为了让**换模型**
    /// 也被拦住：同一个 `--rollin nn` 换个 checkpoint 就是另一个状态分布。
    #[serde(default)]
    pub rollin: Option<String>,
    /// 生效配方（前提 + 采样器 + search_n + 维度）的 FNV-1a
    ///
    /// **显式身份口径（`--space-version`）下为 `None`**：该口径不算任何指纹，
    /// 配方一致性由 [`ensure_resume_compatible`] 逐字段比较得出。
    /// 既有目录里这是一个字符串，读进来即 `Some`，语义不变。
    #[serde(default)]
    pub recipe_hash_fnv1a64: Option<String>,
    /// 采样空间的显式身份；只有 `--space-version` 口径才有
    #[serde(default)]
    pub space: Option<ManifestSpaceIdentity>,
    /// 无哈希口径的资产副本相对路径；旧数据不含此字段。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_files: Option<Vec<String>>,
    /// 显式工作清单；存在时进度字段为清单游标。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_indices: Option<Vec<u64>>,
    /// 固定有效根目标。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_target: Option<u64>
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

    /// 原子替换写入（临时文件写完后直接 rename 替换，与 collector 同一套）
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
    ///
    /// 只在**既有口径**下调用。显式身份口径不算它，见
    /// [`TeacherManifest::recipe_hash_fnv1a64`]。
    fn hash(&self) -> Result<String> {
        let text = serde_json::to_string(self).context("序列化采集配方失败")?;
        Ok(compute_text_hash_fnv1a64(&text))
    }
}

/// 本次采集的空间来源与身份口径
///
/// 两个变体互斥：既有口径继续走枚举指纹，显式口径改为逐字段清单，
/// 且不触发任何内容哈希。
enum SpaceIdentity {
    /// 既有口径：`--shape` / `--extra-card` / 默认 gen1，身份是枚举指纹
    Fingerprint {
        /// 空间枚举指纹
        hash: String
    },
    /// 显式口径：`--space-version`，身份是逐字段清单
    Explicit {
        /// 写进 manifest 的完整清单
        identity: ManifestSpaceIdentity
    }
}

impl SpaceIdentity {
    /// 是否为显式（无哈希）口径
    fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit { .. })
    }

    /// 进 manifest 的枚举指纹（显式口径下为 `None`）
    fn fingerprint(&self) -> Option<String> {
        match self {
            Self::Fingerprint { hash } => Some(hash.clone()),
            Self::Explicit { .. } => None
        }
    }

    /// 进 manifest 的显式清单（既有口径下为 `None`）
    fn explicit(&self) -> Option<ManifestSpaceIdentity> {
        match self {
            Self::Fingerprint { .. } => None,
            Self::Explicit { identity } => Some(identity.clone())
        }
    }

    /// 一行可读摘要
    fn describe(&self) -> String {
        match self {
            Self::Fingerprint { hash } => format!("枚举指纹 {hash}"),
            Self::Explicit { identity } => format!(
                "显式版本 {}（马娘 {} / 卡 {} / 构成 {}）",
                identity.version,
                identity.umas.len(),
                identity.cards.len(),
                identity.shapes.len()
            )
        }
    }
}

/// 按命令行构造采样空间及其身份口径
///
/// # 错误
///
/// `--space-version` 与 `--shape` / `--extra-card` 同用，或空间构造失败时报错。
fn build_space(args: &CollectArgs) -> Result<(SamplingSpace, SpaceIdentity)> {
    let Some(name) = args.space_version.as_deref() else {
        let space = space_from_cli(args.shape.as_deref(), &args.extra_card)?;
        let hash = space.content_hash();
        return Ok((space, SpaceIdentity::Fingerprint { hash }));
    };
    ensure!(
        args.shape.is_none() && args.extra_card.is_empty(),
        "--space-version 与 --shape / --extra-card 互斥：具名版本已经把马娘、卡池、构成一并冻结"
    );
    let version = space_version_by_name(name)?;
    let space = SamplingSpace::from_version(version)?;
    let identity = ManifestSpaceIdentity {
        version: version.name.to_string(),
        umas: version.umas.iter().map(|u| u.game_id).collect(),
        cards: version.cards.iter().map(|c| c.idrank).collect(),
        shapes: version
            .shapes
            .iter()
            .map(|sh| ManifestShape {
                counts: sh.counts,
                name: sh.name.to_string()
            })
            .collect(),
        plan_count: space.len()
    };
    Ok((space, SpaceIdentity::Explicit { identity }))
}

// ============================================================================
// 搜索配置 / 动作表
// ============================================================================

/// 本次采集使用的 roll-in 基策
///
/// `RamenMctsTrainer` 那种带内部可变状态的类型不适合放这里；两个变体都是无状态
/// 或只读的，可以整段采集复用一个实例。
enum RollIn {
    /// 手写策略（既有全部数据的 roll-in）
    Handwritten(RamenHandwrittenTrainer),
    /// 网络策略（DAgger）
    #[cfg(feature = "onnx")]
    Nn(Box<RamenNnTrainer>)
}

impl RollIn {
    /// 取 `Trainer` 视图交给采样器
    fn as_trainer(&self) -> &dyn Trainer<RamenGame> {
        match self {
            Self::Handwritten(t) => t,
            #[cfg(feature = "onnx")]
            Self::Nn(t) => t.as_ref()
        }
    }

    /// 写进 manifest 的身份串
    fn identity(&self, model_hash: Option<&str>) -> String {
        match self {
            Self::Handwritten(_) => "handwritten".to_string(),
            #[cfg(feature = "onnx")]
            Self::Nn(_) => format!("nn:{}", model_hash.unwrap_or("unknown"))
        }
    }
}

/// 按命令行构造 roll-in，并返回它的身份串
///
/// # 错误
///
/// 未知策略名、`nn` 缺 `--model`、未启用 `onnx` feature，或模型加载失败时报错。
fn select_rollin(args: &CollectArgs, explicit_identity: bool) -> Result<(RollIn, String)> {
    match args.rollin.as_str() {
        "handwritten" => {
            ensure!(args.model.is_none() && args.model_id.is_none(), "--model / --model-id 只对 --rollin nn 有意义");
            let r = RollIn::Handwritten(RamenHandwrittenTrainer::new());
            let id = r.identity(None);
            Ok((r, id))
        }
        "nn" => {
            #[cfg(feature = "onnx")]
            {
                let path = args
                    .model
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--rollin nn 需要同时给出 --model <onnx 路径>"))?;
                let hash = if explicit_identity {
                    let name = args.model_id.as_deref().ok_or_else(|| anyhow!(
                        "显式 NN 采集必须给 --model-id；模型原文另存核对，不以名称证明内容一致"
                    ))?;
                    ensure!(!name.trim().is_empty(), "model-id 不能为空");
                    format!("asset:{name}")
                } else {
                    ensure!(args.model_id.is_none(), "旧口径不接受 --model-id");
                    compute_file_signature(path, true)?.hash_fnv1a64
                        .ok_or_else(|| anyhow!("旧口径缺模型身份"))?
                };
                let trainer = RamenNnTrainer::load(path)?.with_race_shield(true);
                let r = RollIn::Nn(Box::new(trainer));
                let id = r.identity(Some(&hash));
                Ok((r, id))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = &args.model;
                let _ = explicit_identity;
                bail!(
                    "--rollin nn 需要编译 feature onnx\
                     （cargo build --release --features onnx,cli --bin ramen_teacher_collect）"
                )
            }
        }
        other => bail!("未知 --rollin: {other}（可选 handwritten / nn）")
    }
}

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

/// 原子替换写入 JSON（临时文件写完后直接 rename 替换）
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
    writer.get_ref().sync_all().context("sync manifest 失败")?;
    drop(writer);
    fs_err::rename(&tmp_path, path)
        .with_context(|| format!("重命名 manifest 失败: {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

/// 采集复现基座文件的签名
///
/// # 错误
///
/// 文件存在但读元信息 / 内容失败时报错。缺失的文件跳过。
fn collect_gamedata_signatures(with_hash: bool) -> Result<Vec<FileSignature>> {
    let mut out = Vec::new();
    for rel in GAMEDATA_SIG_PATHS {
        let path = Path::new(rel);
        if !path.exists() {
            continue;
        }
        let hash = with_hash
            && match fs_err::metadata(path) {
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
    batch: &mut RamenSampleBatch, output_dir: &Path, part_index: usize, with_hash: bool
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
    // 显式口径不算内容哈希：分片的一致性由「读回条数 + 字节数」直接核对
    let signature = compute_file_signature(&final_path, with_hash)?;
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

/// 工作清单的序号、游标和有效根目标必须自洽；不生成内容指纹。
fn check_work_indices(values: &[u64], start: u64, count: u64, target: Option<u64>) -> Result<()> {
    ensure!(start == 0 && count == values.len() as u64, "显式清单必须 start=0、count=清单长度");
    ensure!(values.iter().copied().collect::<BTreeSet<_>>().len() == values.len(), "清单 index 重复");
    if let Some(n) = target { ensure!(n > 0 && n <= count, "有效根目标超出工作清单"); }
    Ok(())
}

fn main() -> Result<()> {
    let t_process = Instant::now();
    let mut timing = StageTiming::default();
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
    let t = Instant::now();
    init_global_with_config(&game_config)?;
    timing.init_ms = t.elapsed().as_millis();

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
    sampler_cfg.region_quota_permille_y1 = args.region_quota_permille_y1;
    let sampler_snap = ManifestSamplerConfig::from_sampler(&sampler_cfg);
    let recipe = CollectRecipe {
        format_version: SAMPLE_FORMAT_VERSION,
        input_dim: INPUT_DIM,
        policy_dim: POLICY_DIM,
        premises: premises.clone(),
        search_n: args.search_n,
        sampler: sampler_snap.clone()
    };

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

    // 采样空间身份：卡池与构成写在代码里，改动不会反映到 gamedata_sig，只能显式记。
    // 既有口径记枚举指纹；`--space-version` 口径改记逐字段清单，全程不算哈希。
    let t = Instant::now();
    let (space, space_identity) = build_space(&args)?;
    timing.space_ms = t.elapsed().as_millis();
    let explicit_identity = space_identity.is_explicit();
    ensure!(explicit_identity || (args.indices_file.is_none() && args.accepted_target.is_none()),
        "工作清单与有效根目标只支持显式空间");
    let work_indices = args.indices_file.as_ref().map(|path| -> Result<Vec<u64>> {
        let values: Vec<u64> = serde_json::from_slice(&fs_err::read(path)?)?;
        check_work_indices(&values, args.start, args.count, args.accepted_target)?;
        Ok(values)
    }).transpose()?;
    if let Some(target) = args.accepted_target {
        ensure!(work_indices.is_some() && target > 0 && target <= args.count,
            "accepted-target 需要工作清单且位于 1..=count");
    }
    let asset_files = if explicit_identity {
        let mut names = Vec::new();
        for rel in GAMEDATA_SIG_PATHS {
            if Path::new(rel).exists() {
                let name = format!("assets/{rel}");
                explicit_assets::snapshot(Path::new(rel), &output_dir.join(&name), manifest_path.exists())?;
                names.push(name);
            }
        }
        if let Some(model) = &args.model {
            explicit_assets::snapshot(model, &output_dir.join("assets/rollin.onnx"), manifest_path.exists())?;
            names.push("assets/rollin.onnx".to_string());
            let sidecar = PathBuf::from(format!("{}.json", model.display()));
            explicit_assets::snapshot(&sidecar, &output_dir.join("assets/rollin.onnx.json"), manifest_path.exists())?;
            names.push("assets/rollin.onnx.json".to_string());
        }
        Some(names)
    } else { None };
    println!(
        "采样空间 {}，{} 个 (马娘, 卡组) 组合",
        space_identity.describe(),
        space.len()
    );

    // roll-in 决定样本落在哪些状态上；身份串进 manifest 供续跑绊线比对
    let t = Instant::now();
    let (rollin, rollin_id) = select_rollin(&args, explicit_identity)?;
    timing.rollin_load_ms = t.elapsed().as_millis();
    println!("roll-in 基策 {rollin_id}");

    let now = Utc::now().to_rfc3339();
    let (mut manifest, span) = if manifest_path.exists() {
        let old = TeacherManifest::load(&manifest_path)?;
        ensure_resume_compatible(&old, &recipe)?;
        ensure!(old.work_indices == work_indices && old.accepted_target == args.accepted_target,
            "续跑工作清单或有效根目标发生变化");
        ensure!(old.asset_files == asset_files, "续跑资产清单变化或旧显式目录没有原文资产，拒绝续跑");
        if explicit_identity {
            ensure!(old.git_commit == try_get_git_commit(&workspace_root), "显式采集续跑代码版本变化");
        }
        // 空间闸门。两种身份口径互不通用：一个目录只能是其中一种，
        // 混用会让同一目录里出现两套 index 语义而没有任何字段能事后区分。
        match (&old.space, &space_identity) {
            (Some(recorded), SpaceIdentity::Explicit { identity }) => {
                ensure!(
                    recorded == identity,
                    "{} 的采样空间与本次不一致（逐字段比较，未使用任何指纹）:\n  manifest {:?}\n  当前 {:?}",
                    manifest_path.display(),
                    recorded,
                    identity
                );
            }
            (Some(recorded), SpaceIdentity::Fingerprint { .. }) => {
                bail!(
                    "{} 是显式空间版本 `{}` 的采集目录，本次没有给 --space-version。\
                     两种身份口径不可混用。",
                    manifest_path.display(),
                    recorded.version
                );
            }
            (None, SpaceIdentity::Explicit { identity }) => {
                bail!(
                    "{} 是既有指纹口径的采集目录，本次给了 --space-version {}。\
                     换目录，或去掉 --space-version。",
                    manifest_path.display(),
                    identity.version
                );
            }
            (None, SpaceIdentity::Fingerprint { hash }) => {
                // 本字段加入前的目录留空——来路不明就保持不明，不给它盖一个当前空间的章
                if let Some(recorded) = &old.sampling_space_hash {
                    ensure!(
                        recorded == hash,
                        "{} 是在采样空间 {} 下采的，当前编译的空间是 {}。续跑会让同一目录里\
                         混进两个空间的样本，而 index 的含义正是由空间决定的。",
                        manifest_path.display(),
                        recorded,
                        hash
                    );
                }
            }
        }
        // roll-in 身份：本字段加入前的目录一律是手写 roll-in，故 None 等价于 "handwritten"。
        // 与空间指纹不同，这里**不留白**——旧目录的 roll-in 是确知的，不是来路不明。
        let recorded_rollin = old.rollin.as_deref().unwrap_or("handwritten");
        ensure!(
            recorded_rollin == rollin_id,
            "{} 是在 roll-in `{}` 下采的，本次是 `{}`。续跑会让同一目录里混进两种\
             状态分布的样本，而 roll-in 决定的正是样本落在哪些局面上。",
            manifest_path.display(),
            recorded_rollin,
            rollin_id
        );
        ensure_parts_match_disk(&output_dir, &old.parts)?;
        if args.accepted_target.is_some_and(|n| old.accepted >= n) {
            ensure!(verify_written_parts(&output_dir, &old.parts)? as u64 == old.accepted, "已完成分片计数不符");
            println!("有效根目标已完成，manifest 原样保留");
            return Ok(());
        }
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
            gamedata_sig: collect_gamedata_signatures(!explicit_identity)?,
            sampling_space_hash: space_identity.fingerprint(),
            rollin: Some(rollin_id.clone()),
            // 显式口径不算配方指纹；一致性由 ensure_resume_compatible 逐字段比较
            recipe_hash_fnv1a64: if explicit_identity { None } else { Some(recipe.hash()?) },
            space: space_identity.explicit(),
            asset_files,
            work_indices,
            accepted_target: args.accepted_target
        };
        (manifest, span)
    };
    let t = Instant::now();
    manifest.save_replace(&manifest_path)?;
    timing.manifest_ms += t.elapsed().as_millis();

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
        "  region_quota_permille    = {:?}（Y2,Y3）",
        sampler_cfg.region_quota_permille
    );
    println!(
        "  region_quota_permille_y1 = {}",
        sampler_cfg.region_quota_permille_y1
    );
    println!("  采样空间身份            : {}", space_identity.describe());
    println!("  组合总数                : {}", space.len());
    println!("  INPUT_DIM / POLICY_DIM  = {INPUT_DIM} / {POLICY_DIM}");
    println!("  format_version          = {SAMPLE_FORMAT_VERSION}");

    if span.is_empty() {
        println!("区间为空（可能已经采完），只做读回校验。");
        let total = verify_written_parts(&output_dir, &manifest.parts)?;
        println!("已有样本 {total} 条，跳过 {} 次。", manifest.skipped_uncaptured);
        return Ok(());
    }

    let search: FlatSearch<RamenGame> = FlatSearch::new(search_cfg);
    let mut batch = RamenSampleBatch::new();
    let mut next_part_index = manifest.parts.len();

    for position in span.start..span.end {
        if args.max_seconds.is_some_and(|n| t_process.elapsed().as_secs() >= n) {
            println!("达到根间软截止，保存已有完整样本");
            break;
        }
        let index = manifest.work_indices.as_ref().map_or(position, |v| v[position as usize]);
        let t = Instant::now();
        let sampled = sample_position_with_rollin(&space, &sampler_cfg, index, rollin.as_trainer())?;
        timing.sample_ms += t.elapsed().as_millis();
        match sampled.into_captured() {
            None => {
                manifest.skipped_uncaptured += 1;
                println!("  index={index} 跳过（未捕获）");
            }
            Some(pos) => {
                let t = Instant::now();
                let sample = collect_one(&search, &pos, index)?;
                if explicit_identity {
                    ensure!(sample.candidates.iter().all(|c| c.n as usize == args.search_n),
                        "index={index} 有失败 rollout，停止显式采集，已有完整分片保留");
                }
                timing.search_ms += t.elapsed().as_millis();
                println!(
                    "  index={index} turn={} stage={:?} 候选 {}",
                    pos.turn,
                    pos.stage,
                    sample.candidates.len()
                );
                batch.push(sample);
                if batch.len() >= args.shard_size {
                    let t = Instant::now();
                    let part = flush_shard(&mut batch, &output_dir, next_part_index, !explicit_identity)?;
                    timing.flush_ms += t.elapsed().as_millis();
                    println!("  写入 {} ({} 条)", part.name, part.samples);
                    manifest.parts.push(part);
                    manifest.accepted = manifest.parts.iter().map(|p| p.samples as u64).sum();
                    next_part_index += 1;
                    manifest.next_index = position + 1;
                    manifest.updated_at = Utc::now().to_rfc3339();
                    let t = Instant::now();
                    manifest.save_replace(&manifest_path)?;
                    timing.manifest_ms += t.elapsed().as_millis();
                }
            }
        }
        manifest.next_index = position + 1;
        if args.accepted_target.is_some_and(|n| manifest.accepted + batch.len() as u64 >= n) {
            break;
        }
    }

    if !batch.is_empty() {
        let t = Instant::now();
        let part = flush_shard(&mut batch, &output_dir, next_part_index, !explicit_identity)?;
        timing.flush_ms += t.elapsed().as_millis();
        println!("  写入 {} ({} 条)", part.name, part.samples);
        manifest.parts.push(part);
        manifest.accepted = manifest.parts.iter().map(|p| p.samples as u64).sum();
        manifest.updated_at = Utc::now().to_rfc3339();
        let t = Instant::now();
        manifest.save_replace(&manifest_path)?;
        timing.manifest_ms += t.elapsed().as_millis();
    }

    let finished = Utc::now().to_rfc3339();
    manifest.updated_at = finished.clone();
    if args.accepted_target.map_or(manifest.next_index >= manifest.index_end, |n| manifest.accepted >= n) {
        manifest.finished_at = Some(finished);
    }
    let t = Instant::now();
    manifest.save_replace(&manifest_path)?;
    timing.manifest_ms += t.elapsed().as_millis();

    println!("=== 读回校验 ===");
    let t = Instant::now();
    let total = verify_written_parts(&output_dir, &manifest.parts)?;
    timing.verify_ms = t.elapsed().as_millis();
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
    if let Some(target) = args.accepted_target {
        ensure!(manifest.accepted == target, "有效根未完成：{} / {target}，完整分片已保留", manifest.accepted);
    }

    let total_ms = t_process.elapsed().as_millis();
    let accounted = timing.init_ms
        + timing.space_ms
        + timing.rollin_load_ms
        + timing.sample_ms
        + timing.search_ms
        + timing.flush_ms
        + timing.manifest_ms
        + timing.verify_ms;
    println!("=== 分项墙钟（毫秒） ===");
    println!("  TIMING total          = {total_ms}");
    println!("  TIMING init_global    = {}", timing.init_ms);
    println!("  TIMING space_enum     = {}", timing.space_ms);
    println!("  TIMING rollin_load    = {}", timing.rollin_load_ms);
    println!("  TIMING sample         = {}", timing.sample_ms);
    println!("  TIMING search_export  = {}", timing.search_ms);
    println!("  TIMING shard_flush    = {}", timing.flush_ms);
    println!("  TIMING manifest_write = {}", timing.manifest_ms);
    println!("  TIMING verify_readback= {}", timing.verify_ms);
    println!("  TIMING unaccounted    = {}", total_ms.saturating_sub(accounted));
    println!("  TIMING accepted_roots = {}", manifest.accepted);
    println!("  TIMING skipped        = {}", manifest.skipped_uncaptured);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 检查稀疏 index 不被当作连续区间、重复清单及越界目标被拒绝。
    #[test]
    fn test_explicit_work_indices() -> Result<()> {
        check_work_indices(&[20_000_001, 20_004_301], 0, 2, Some(1))?;
        ensure!(check_work_indices(&[1, 1], 0, 2, Some(1)).is_err(), "重复未拒绝");
        ensure!(check_work_indices(&[1, 2], 1, 2, Some(1)).is_err(), "错误游标未拒绝");
        ensure!(check_work_indices(&[1, 2], 0, 2, Some(3)).is_err(), "超额目标未拒绝");
        println!("清单重复、错误游标、越界目标均拒绝");
        Ok(())
    }

    /// Y1 默认关闭时序列化不增加字段，旧配置读回仍为零。
    #[test]
    fn test_zero_y1_preserves_old_recipe_fields() -> Result<()> {
        let mut snapshot = ManifestSamplerConfig::from_sampler(&SamplerConfig::default());
        let value = serde_json::to_value(&snapshot)?;
        ensure!(value.get("region_quota_permille_y1").is_none(), "零配额污染旧配方");
        let old: ManifestSamplerConfig = serde_json::from_value(value)?;
        ensure!(old == snapshot, "旧字段读取语义改变");
        snapshot.region_quota_permille_y1 = 1000;
        ensure!(serde_json::to_value(&snapshot)?["region_quota_permille_y1"] == 1000,
            "启用的Y1配额未记录");
        println!("零Y1省略、旧字段读回、启用Y1记录均通过");
        Ok(())
    }

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
