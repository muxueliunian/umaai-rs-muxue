//! 拉面杯教师数据导出：bincode 分片 → NumPy `.npy` 数组目录
//!
//! 训练侧是 Python + PyTorch，读不了 bincode，需要一层格式转换。本 bin 把
//! [`RamenSampleBatch`] 摊平成一组定长数组落在同一个目录里，Python 侧用
//! `np.load(dir / "x.npy")` 逐个读取（大数组可加 `mmap_mode="r"`）。
//!
//! **只导原始量，不导标签。** 「rollout 分数 → policy 软标签」的配方仍是待定项，
//! 留在 Python 侧改一次不用重跑 Rust。value 的归一化常数同理，由训练侧从数据标定。
//!
//! # 候选是变长的
//!
//! 每个样本的候选数不同（实测 1~120），故候选维用 CSR 摊平：样本 `i` 的候选是
//! `cand_*[cand_ptr[i] .. cand_ptr[i + 1]]`。
//!
//! # 产出数组
//!
//! | 名字 | 形状 | dtype | 含义 |
//! |---|---|---|---|
//! | `x` | `[N, 754]` | f32 | 局面特征 |
//! | `stage` | `[N]` | u8 | 决策阶段的稳定编码 |
//! | `turn` | `[N]` | i16 | 回合 |
//! | `index` | `[N]` | u64 | 样本唯一 id |
//! | `legal_mask` | `[N, 234]` | u8 | 该局面下合法的 policy 格位 |
//! | `cand_ptr` | `[N + 1]` | i64 | CSR 偏移 |
//! | `cand_slots` | `[C, 3]` | i32 | 候选占据的格位，`-1` 表示无 |
//! | `cand_n` | `[C]` | i32 | 有效 rollout 次数 |
//! | `cand_mean` | `[C]` | f32 | rollout 分数均值 |
//! | `cand_stdev` | `[C]` | f32 | 样本标准差（n-1 分母）|
//!
//! `--raw` 额外导出（体积大得多，用于设计标签配方与统计实验）：
//!
//! | 名字 | 形状 | dtype | 含义 |
//! |---|---|---|---|
//! | `cand_scores` | `[C, R]` | f32 | 每次 rollout 的原始分数，失败槽位为 0.0 |
//! | `cand_valid` | `[C, R]` | u8 | 槽位是否有效，读 `cand_scores` 必须配合它 |
//!
//! 同一列的所有候选共享 rollout 种子（CRN），即 `cand_scores[:, k]` 是同一个随机
//! 世界下的配对比较——这是标签设计可以利用的结构。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p umasim --bin ramen_export_npy -- \
//!     --input training_data/prod_a_0 --input training_data/prod_d_0 \
//!     --output-dir training_data/npy_v1 --raw
//! ```

use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{Seek, SeekFrom, Write},
    marker::PhantomData,
    path::{Path, PathBuf}
};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use umasim::{
    collector::scan_part_files,
    game::ramen::{
        features::INPUT_DIM,
        policy_schema::POLICY_DIM,
        training_sample::{RamenSampleBatch, RamenTrainingSample, SAMPLE_FORMAT_VERSION, stage_of_code}
    },
    gamedata::init_global_with_config,
    sampler::SamplingSpace,
    utils::{get_workspace_root, load_game_config}
};

/// manifest 文件名，与采集端一致
const MANIFEST_NAME: &str = "manifest.json";

/// `.npy` 头部固定长度（含 magic），必须是 64 的倍数
const NPY_HEADER_LEN: usize = 128;

/// 地区选择候选占据的格位数，也是 `cand_slots` 的列宽
const SLOTS_PER_CAND: usize = 3;

// ============================================================================
// 命令行
// ============================================================================

/// 导出参数
#[derive(Parser, Debug)]
#[command(about = "把拉面教师样本分片导出为 NumPy .npy 数组目录")]
struct ExportArgs {
    /// 采集输出目录，可重复指定；顺序不影响结果（内部按路径排序）
    #[arg(long = "input", required = true)]
    inputs: Vec<PathBuf>,

    /// 导出目录，不存在则创建
    #[arg(long)]
    output_dir: PathBuf,

    /// 额外导出 `cand_scores` / `cand_valid`（体积约为 reduced 的 150 倍）
    #[arg(long, default_value_t = false)]
    raw: bool
}

// ============================================================================
// 源 manifest（只反序列化本 bin 需要的字段）
// ============================================================================

/// 采集端 manifest 的子集
///
/// 采集端的完整结构定义在 `ramen_teacher_collect.rs` 里且是私有的。这里只取
/// 一致性校验需要的几个字段，多余字段被 serde 忽略。
#[derive(Debug, Clone, Deserialize)]
struct SourceManifest {
    /// 样本容器格式版本
    format_version: u32,
    /// 特征维度，必须与本次编译的 [`INPUT_DIM`] 一致
    input_dim: usize,
    /// 格位数，必须与本次编译的 [`POLICY_DIM`] 一致
    policy_dim: usize,
    /// 每候选 rollout 次数
    search_n: usize,
    /// 采集配方哈希，跨机合并的唯一有效判据
    recipe_hash_fnv1a64: String,
    /// 采集时的 git commit
    git_commit: String
}

/// 所有输入目录必须一致的那部分配方
#[derive(Debug, Clone, PartialEq, Eq)]
struct SharedRecipe {
    /// 采集配方哈希
    recipe_hash: String,
    /// git commit
    git_commit: String,
    /// 每候选 rollout 次数
    search_n: usize
}

impl SharedRecipe {
    /// 从 manifest 提取，并校验维度常数与本次编译一致
    ///
    /// # 错误
    ///
    /// `format_version` / `input_dim` / `policy_dim` 与本次编译不符时报错——
    /// 那说明数据是另一份代码采的，摊平出来的数组含义会静默错位。
    fn from_manifest(m: &SourceManifest, dir: &Path) -> Result<Self> {
        ensure!(
            m.format_version == SAMPLE_FORMAT_VERSION,
            "{} 的 format_version={} 与本次编译的 {SAMPLE_FORMAT_VERSION} 不符",
            dir.display(),
            m.format_version
        );
        ensure!(m.input_dim == INPUT_DIM, "{} 的 input_dim={} 与本次编译的 {INPUT_DIM} 不符", dir.display(), m.input_dim);
        ensure!(m.policy_dim == POLICY_DIM, "{} 的 policy_dim={} 与本次编译的 {POLICY_DIM} 不符", dir.display(), m.policy_dim);
        Ok(Self {
            recipe_hash: m.recipe_hash_fnv1a64.clone(),
            git_commit: m.git_commit.clone(),
            search_n: m.search_n
        })
    }
}

// ============================================================================
// .npy 写出
// ============================================================================

/// 可写进 `.npy` 的标量类型
///
/// `DESCR` 是 NumPy 的 dtype 描述串，小端固定。
trait NpyElem: Copy {
    /// NumPy dtype 描述串
    const DESCR: &'static str;

    /// 以小端追加到缓冲区
    fn push_le(self, out: &mut Vec<u8>);
}

/// 为整数/浮点实现 [`NpyElem`]，全部小端
macro_rules! impl_npy_elem {
    ($ty:ty, $descr:literal) => {
        impl NpyElem for $ty {
            const DESCR: &'static str = $descr;

            fn push_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl_npy_elem!(f32, "<f4");
impl_npy_elem!(i16, "<i2");
impl_npy_elem!(i32, "<i4");
impl_npy_elem!(i64, "<i8");
impl_npy_elem!(u64, "<u8");

impl NpyElem for u8 {
    const DESCR: &'static str = "|u1";

    fn push_le(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

/// 流式 `.npy` 写出器
///
/// 行数在写完之前是未知的，故先落一个定长占位头，收尾时 seek 回去重写。
/// 头部固定 [`NPY_HEADER_LEN`] 字节（NumPy 只要求总长是 64 的倍数），
/// 重写时长度不变，不会推移数据。
struct NpyWriter<T: NpyElem> {
    /// 目标文件
    file: File,
    /// 列宽；`None` 表示一维数组
    cols: Option<usize>,
    /// 已写行数
    rows: usize,
    /// 复用的行缓冲，避免每行分配
    buf: Vec<u8>,
    /// 元素类型标记
    _marker: PhantomData<T>
}

impl<T: NpyElem> NpyWriter<T> {
    /// 在 `dir/{name}.npy` 新建写出器
    ///
    /// `cols` 为 `None` 时产出形状 `(rows,)` 的一维数组。
    ///
    /// # 错误
    ///
    /// 文件创建失败或占位头写入失败时报错。
    fn create(dir: &Path, name: &str, cols: Option<usize>) -> Result<Self> {
        let path = dir.join(format!("{name}.npy"));
        let mut file = File::create(&path).with_context(|| format!("创建数组文件失败: {}", path.display()))?;
        file.write_all(&npy_header(T::DESCR, 0, cols)?)
            .with_context(|| format!("写入占位头失败: {}", path.display()))?;
        Ok(Self {
            file,
            cols,
            rows: 0,
            buf: Vec::new(),
            _marker: PhantomData
        })
    }

    /// 追加一行
    ///
    /// # 错误
    ///
    /// 行长与列宽不符，或写入失败时报错。
    fn push_row(&mut self, row: &[T]) -> Result<()> {
        let want = self.cols.unwrap_or(1);
        ensure!(row.len() == want, "行长 {} 与列宽 {want} 不符", row.len());
        self.buf.clear();
        for v in row {
            v.push_le(&mut self.buf);
        }
        self.file.write_all(&self.buf).context("写入数组数据失败")?;
        self.rows += 1;
        Ok(())
    }

    /// 追加一个标量行，仅一维数组可用
    ///
    /// # 错误
    ///
    /// 同 [`Self::push_row`]。
    fn push(&mut self, v: T) -> Result<()> {
        self.push_row(&[v])
    }

    /// 回填真实行数并落盘
    ///
    /// # 错误
    ///
    /// seek 或写入失败时报错。
    fn finish(mut self) -> Result<usize> {
        let header = npy_header(T::DESCR, self.rows, self.cols)?;
        self.file.seek(SeekFrom::Start(0)).context("回到文件头失败")?;
        self.file.write_all(&header).context("回填数组头失败")?;
        self.file.flush().context("刷新数组文件失败")?;
        Ok(self.rows)
    }
}

/// 构造定长 `.npy` 头部
///
/// 布局：magic `\x93NUMPY` + 版本 `1.0` + `u16` 头长 + 字典（空格补齐）+ `\n`。
///
/// # 错误
///
/// 字典超出 [`NPY_HEADER_LEN`] 时报错——只有 dtype 描述串异常长才可能发生。
fn npy_header(descr: &str, rows: usize, cols: Option<usize>) -> Result<Vec<u8>> {
    let shape = match cols {
        Some(c) => format!("{rows}, {c}"),
        // NumPy 的一元 tuple 必须带尾逗号
        None => format!("{rows},")
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape}), }}");
    // magic 6 + 版本 2 + 头长字段 2 = 10
    let dict_room = NPY_HEADER_LEN - 10;
    ensure!(dict.len() < dict_room, "npy 头字典过长: {} >= {dict_room}", dict.len());

    let mut out = Vec::with_capacity(NPY_HEADER_LEN);
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1u8, 0u8]);
    let dict_len = u16::try_from(dict_room).context("npy 头长溢出 u16")?;
    out.extend_from_slice(&dict_len.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(NPY_HEADER_LEN - 1, b' ');
    out.push(b'\n');
    Ok(out)
}

// ============================================================================
// 导出
// ============================================================================

/// 一组同时写出的数组
struct ArraySet {
    /// 局面特征
    x: NpyWriter<f32>,
    /// 阶段编码
    stage: NpyWriter<u8>,
    /// 回合
    turn: NpyWriter<i16>,
    /// 样本 id
    index: NpyWriter<u64>,
    /// 合法格位掩码
    legal_mask: NpyWriter<u8>,
    /// CSR 偏移
    cand_ptr: NpyWriter<i64>,
    /// 候选格位
    cand_slots: NpyWriter<i32>,
    /// 有效 rollout 次数
    cand_n: NpyWriter<i32>,
    /// 分数均值
    cand_mean: NpyWriter<f32>,
    /// 分数标准差
    cand_stdev: NpyWriter<f32>,
    /// 原始分数，仅 `--raw`
    cand_scores: Option<NpyWriter<f32>>,
    /// 槽位有效性，仅 `--raw`
    cand_valid: Option<NpyWriter<u8>>
}

impl ArraySet {
    /// 在 `dir` 下新建全套写出器
    ///
    /// `rollout_width` 仅在 `raw` 为真时用到，作为 `cand_scores` 的列宽。
    ///
    /// # 错误
    ///
    /// 任一文件创建失败时报错。
    fn create(dir: &Path, raw: bool, rollout_width: usize) -> Result<Self> {
        Ok(Self {
            x: NpyWriter::create(dir, "x", Some(INPUT_DIM))?,
            stage: NpyWriter::create(dir, "stage", None)?,
            turn: NpyWriter::create(dir, "turn", None)?,
            index: NpyWriter::create(dir, "index", None)?,
            legal_mask: NpyWriter::create(dir, "legal_mask", Some(POLICY_DIM))?,
            cand_ptr: NpyWriter::create(dir, "cand_ptr", None)?,
            cand_slots: NpyWriter::create(dir, "cand_slots", Some(SLOTS_PER_CAND))?,
            cand_n: NpyWriter::create(dir, "cand_n", None)?,
            cand_mean: NpyWriter::create(dir, "cand_mean", None)?,
            cand_stdev: NpyWriter::create(dir, "cand_stdev", None)?,
            cand_scores: if raw {
                Some(NpyWriter::create(dir, "cand_scores", Some(rollout_width))?)
            } else {
                None
            },
            cand_valid: if raw {
                Some(NpyWriter::create(dir, "cand_valid", Some(rollout_width))?)
            } else {
                None
            }
        })
    }

    /// 回填所有头部
    ///
    /// # 错误
    ///
    /// 任一回填失败时报错。
    fn finish(self) -> Result<()> {
        self.x.finish()?;
        self.stage.finish()?;
        self.turn.finish()?;
        self.index.finish()?;
        self.legal_mask.finish()?;
        self.cand_ptr.finish()?;
        self.cand_slots.finish()?;
        self.cand_n.finish()?;
        self.cand_mean.finish()?;
        self.cand_stdev.finish()?;
        if let Some(w) = self.cand_scores {
            w.finish()?;
        }
        if let Some(w) = self.cand_valid {
            w.finish()?;
        }
        Ok(())
    }
}

/// 导出过程中累计的统计量
#[derive(Debug, Default, Clone, Serialize)]
struct ExportStats {
    /// 样本数
    samples: usize,
    /// 候选总数
    candidates: usize,
    /// rollout 槽位宽度（全体候选必须一致）
    rollout_width: usize,
    /// 阶段编码 → 样本数
    stage_hist: BTreeMap<u8, usize>,
    /// 阶段名 → 样本数，便于人读
    stage_names: BTreeMap<String, usize>
}

/// 导出目录的元信息，与数组同目录落盘
#[derive(Debug, Clone, Serialize)]
struct ExportMeta {
    /// 本 bin 的导出格式版本
    export_version: u32,
    /// 样本容器格式版本
    format_version: u32,
    /// 特征维度
    input_dim: usize,
    /// 格位数
    policy_dim: usize,
    /// 每候选 rollout 次数
    search_n: usize,
    /// 采集配方哈希
    recipe_hash_fnv1a64: String,
    /// 采集时的 git commit
    git_commit: String,
    /// 采样空间的计划数（(马娘, 卡组) 组合数）
    ///
    /// 采样器按 `index % plan_count` 轮转分配，所以 `index % plan_count` 就是
    /// 卡组组合的标识。训练侧要按组合切留出集就得用它——随机切样本会让同一套
    /// 卡组同时进训练与验证，泛化指标虚高。
    plan_count: usize,
    /// 是否含 `cand_scores` / `cand_valid`
    raw: bool,
    /// 参与合并的源目录名（按序，样本按此序拼接）
    sources: Vec<String>,
    /// 统计量
    stats: ExportStats
}

/// 把一个样本的候选格位摊成 `[C, 3]` 的一行，不足补 `-1`
///
/// # 错误
///
/// 候选占据的格位超过 [`SLOTS_PER_CAND`]，或格位下标越界时报错。
fn slots_row(sample: &RamenTrainingSample, cand_idx: usize) -> Result<[i32; SLOTS_PER_CAND]> {
    let slots = sample.candidates[cand_idx].slots.as_slice();
    ensure!(slots.len() <= SLOTS_PER_CAND, "候选占据 {} 格，超出 {SLOTS_PER_CAND}", slots.len());
    let mut row = [-1i32; SLOTS_PER_CAND];
    for (i, &s) in slots.iter().enumerate() {
        ensure!(s < POLICY_DIM, "格位下标 {s} 越界（POLICY_DIM={POLICY_DIM}）");
        row[i] = i32::try_from(s).context("格位下标溢出 i32")?;
    }
    Ok(row)
}

/// 扫描一个输入目录，返回 manifest 与排序后的分片路径
///
/// # 错误
///
/// manifest 缺失、解析失败，或目录下没有分片时报错。
fn scan_input(dir: &Path) -> Result<(SourceManifest, Vec<PathBuf>)> {
    let mpath = dir.join(MANIFEST_NAME);
    let text = std::fs::read_to_string(&mpath).with_context(|| format!("读取 manifest 失败: {}", mpath.display()))?;
    let manifest: SourceManifest =
        serde_json::from_str(&text).with_context(|| format!("解析 manifest 失败: {}", mpath.display()))?;
    // `scan_part_files` 返回 (分片序号, 路径)，按序号升序取路径即为落盘顺序
    let mut indexed = scan_part_files(dir).with_context(|| format!("扫描分片失败: {}", dir.display()))?;
    indexed.sort_by_key(|(i, _)| *i);
    let parts: Vec<PathBuf> = indexed.into_iter().map(|(_, p)| p).collect();
    ensure!(!parts.is_empty(), "{} 下没有分片文件", dir.display());
    Ok((manifest, parts))
}

/// 主流程
///
/// # 错误
///
/// 输入不一致、样本 id 重复、rollout 宽度不齐，或任一 IO 失败时报错。
fn run(args: &ExportArgs) -> Result<()> {
    // 采样空间需要 gamedata；切到工作空间根以便按相对路径读取
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    init_global_with_config(&load_game_config()?)?;
    let plan_count = SamplingSpace::gen1()?.len();

    let mut inputs = args.inputs.clone();
    inputs.sort();
    inputs.dedup();

    // ---- 第一遍：校验配方一致，并确定 rollout 宽度 ----
    let mut shared: Option<SharedRecipe> = None;
    let mut scanned: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();
    for dir in &inputs {
        let (manifest, parts) = scan_input(dir)?;
        let recipe = SharedRecipe::from_manifest(&manifest, dir)?;
        match &shared {
            None => shared = Some(recipe),
            Some(first) => ensure!(
                first == &recipe,
                "{} 的采集配方与前面的目录不一致：{:?} vs {:?}。跨机合并只认 recipe_hash + git_commit，不要合并不同配方的数据",
                dir.display(),
                recipe,
                first
            )
        }
        println!("源 {:<40} {:3} 个分片", dir.display(), parts.len());
        scanned.push((dir.clone(), parts));
    }
    let shared = shared.context("没有可用的输入目录")?;

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("创建导出目录失败: {}", args.output_dir.display()))?;

    // rollout 宽度取第一个样本的槽位数，后续逐个校验。定长是 `cand_scores` 能摊成
    // 矩形数组的前提，不齐就必须报错而不是补零——补零会被训练侧当成真实分数。
    let rollout_width = first_rollout_width(&scanned)?;

    let mut arrays = ArraySet::create(&args.output_dir, args.raw, rollout_width)?;
    let mut stats = ExportStats {
        rollout_width,
        ..Default::default()
    };
    let mut seen: HashSet<u64> = HashSet::new();
    let mut cursor: i64 = 0;
    arrays.cand_ptr.push(cursor)?;

    // ---- 第二遍：摊平 ----
    for (dir, parts) in &scanned {
        let mut in_dir = 0usize;
        for part in parts {
            let batch = RamenSampleBatch::load_binary(part)?;
            for sample in &batch.samples {
                write_sample(&mut arrays, sample, rollout_width, args.raw, &mut cursor, &mut seen, &mut stats)?;
                in_dir += 1;
            }
        }
        println!("  {:<40} {in_dir:6} 条", dir.display());
    }

    arrays.finish()?;

    for (code, n) in &stats.stage_hist {
        let name = stage_of_code(*code).map(|s| format!("{s:?}")).unwrap_or_else(|_| format!("未知({code})"));
        stats.stage_names.insert(name, *n);
    }

    let meta = ExportMeta {
        export_version: 1,
        format_version: SAMPLE_FORMAT_VERSION,
        input_dim: INPUT_DIM,
        policy_dim: POLICY_DIM,
        search_n: shared.search_n,
        recipe_hash_fnv1a64: shared.recipe_hash.clone(),
        git_commit: shared.git_commit.clone(),
        plan_count,
        raw: args.raw,
        sources: scanned.iter().map(|(d, _)| d.display().to_string()).collect(),
        stats: stats.clone()
    };
    let mpath = args.output_dir.join("meta.json");
    let text = serde_json::to_string_pretty(&meta).context("序列化 meta 失败")?;
    std::fs::write(&mpath, text).with_context(|| format!("写入 meta 失败: {}", mpath.display()))?;

    println!();
    println!("导出完成 → {}", args.output_dir.display());
    println!("  样本 {}  候选 {}  rollout 宽 {}", stats.samples, stats.candidates, stats.rollout_width);
    println!("  阶段分布 {:?}", stats.stage_names);
    println!("  采样计划数 {plan_count}（index % {plan_count} 即卡组组合 id）");
    println!("  raw = {}", args.raw);
    Ok(())
}

/// 取第一个样本的 rollout 槽位宽度
///
/// # 错误
///
/// 所有分片都没有样本时报错。
fn first_rollout_width(scanned: &[(PathBuf, Vec<PathBuf>)]) -> Result<usize> {
    for (_, parts) in scanned {
        for part in parts {
            let batch = RamenSampleBatch::load_binary(part)?;
            if let Some(s) = batch.samples.first()
                && let Some(c) = s.candidates.first()
            {
                return Ok(c.rollouts());
            }
        }
    }
    bail!("所有输入分片都不含样本")
}

/// 摊平并写出单个样本
///
/// # 错误
///
/// 样本 id 重复、维度不符、rollout 宽度不齐，或写入失败时报错。
fn write_sample(
    arrays: &mut ArraySet,
    sample: &RamenTrainingSample,
    rollout_width: usize,
    raw: bool,
    cursor: &mut i64,
    seen: &mut HashSet<u64>,
    stats: &mut ExportStats
) -> Result<()> {
    ensure!(
        sample.format_version == SAMPLE_FORMAT_VERSION,
        "样本 index={} 的 format_version={} 与本次编译不符",
        sample.meta.index,
        sample.format_version
    );
    ensure!(
        sample.features.len() == INPUT_DIM,
        "样本 index={} 的特征维度 {} != {INPUT_DIM}",
        sample.meta.index,
        sample.features.len()
    );
    ensure!(seen.insert(sample.meta.index), "样本 id 重复: index={}。多半是两个目录的索引号段撞了", sample.meta.index);
    ensure!(!sample.candidates.is_empty(), "样本 index={} 没有候选", sample.meta.index);

    arrays.x.push_row(&sample.features)?;
    arrays.stage.push(sample.meta.stage)?;
    arrays.turn.push(i16::try_from(sample.meta.turn).context("回合号溢出 i16")?)?;
    arrays.index.push(sample.meta.index)?;

    let mut mask = vec![0u8; POLICY_DIM];
    let mut scores = vec![0f32; rollout_width];
    let mut valid = vec![0u8; rollout_width];
    for (ci, cand) in sample.candidates.iter().enumerate() {
        ensure!(
            cand.rollouts() == rollout_width,
            "样本 index={} 的候选 {ci} 有 {} 个 rollout 槽，与首个样本的 {rollout_width} 不一致",
            sample.meta.index,
            cand.rollouts()
        );
        let row = slots_row(sample, ci)?;
        for &s in &row {
            if s >= 0 {
                mask[s as usize] = 1;
            }
        }
        arrays.cand_slots.push_row(&row)?;
        arrays.cand_n.push(i32::try_from(cand.n).context("rollout 次数溢出 i32")?)?;
        arrays.cand_mean.push(cand.mean() as f32)?;
        arrays.cand_stdev.push(cand.stdev() as f32)?;

        if raw {
            for k in 0..rollout_width {
                scores[k] = cand.scores[k];
                valid[k] = u8::from(cand.valid.is_valid(k));
            }
            let w = arrays.cand_scores.as_mut().context("raw 模式缺少 cand_scores 写出器")?;
            w.push_row(&scores)?;
            let w = arrays.cand_valid.as_mut().context("raw 模式缺少 cand_valid 写出器")?;
            w.push_row(&valid)?;
        }
    }
    arrays.legal_mask.push_row(&mask)?;

    *cursor += i64::try_from(sample.candidates.len()).context("候选数溢出 i64")?;
    arrays.cand_ptr.push(*cursor)?;

    stats.samples += 1;
    stats.candidates += sample.candidates.len();
    *stats.stage_hist.entry(sample.meta.stage).or_insert(0) += 1;
    Ok(())
}

fn main() -> Result<()> {
    let args = ExportArgs::parse();
    run(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 逐条打印判定，失败时汇总报错（与本目录其他 bin 的测试风格一致）
    struct Checks {
        /// 失败的条目
        failed: Vec<String>
    }

    impl Checks {
        /// 新建
        fn new() -> Self {
            Self { failed: Vec::new() }
        }

        /// 记录一条判定
        fn check(&mut self, ok: bool, what: &str) {
            println!("  [{}] {what}", if ok { "OK" } else { "NG" });
            if !ok {
                self.failed.push(what.to_string());
            }
        }

        /// 汇总
        ///
        /// # 错误
        ///
        /// 有任一条目失败时报错。
        fn finish(self) -> Result<()> {
            if self.failed.is_empty() {
                Ok(())
            } else {
                bail!("{} 条判定未通过: {:?}", self.failed.len(), self.failed)
            }
        }
    }

    /// 头部长度、对齐与 shape 串
    #[test]
    fn test_npy_header() -> Result<()> {
        let mut c = Checks::new();

        let h2 = npy_header("<f4", 1234, Some(754))?;
        println!("二维头 {} 字节", h2.len());
        println!("{}", String::from_utf8_lossy(&h2[10..]).trim_end());
        c.check(h2.len() == NPY_HEADER_LEN, "头部定长 128");
        c.check(h2.len().is_multiple_of(64), "总长是 64 的倍数");
        c.check(&h2[0..6] == b"\x93NUMPY", "magic 正确");
        c.check(h2[6] == 1 && h2[7] == 0, "版本 1.0");
        c.check(u16::from_le_bytes([h2[8], h2[9]]) as usize == NPY_HEADER_LEN - 10, "头长字段 = 118");
        c.check(*h2.last().context("头部为空")? == b'\n', "以换行结尾");
        let txt = String::from_utf8_lossy(&h2[10..]).to_string();
        c.check(txt.contains("'shape': (1234, 754)"), "二维 shape 正确");
        c.check(txt.contains("'descr': '<f4'"), "dtype 正确");
        c.check(txt.contains("'fortran_order': False"), "C 序");

        let h1 = npy_header("<i8", 7, None)?;
        let txt1 = String::from_utf8_lossy(&h1[10..]).to_string();
        println!("一维头 {}", txt1.trim_end());
        c.check(txt1.contains("'shape': (7,)"), "一维 shape 带尾逗号");
        c.check(h1.len() == NPY_HEADER_LEN, "一维头也是定长");

        // 回填前后长度必须一致，否则 seek 重写会推移数据
        let h0 = npy_header("<f4", 0, Some(754))?;
        c.check(h0.len() == h2.len(), "占位头与回填头等长");

        c.finish()
    }

    /// 写出→按 npy 布局读回，校验行数、字节数与数值
    #[test]
    fn test_npy_writer_roundtrip() -> Result<()> {
        let mut c = Checks::new();
        let dir = umasim::utils::get_workspace_root()?.join("target").join("npy_writer_test");
        std::fs::create_dir_all(&dir)?;

        let mut w: NpyWriter<f32> = NpyWriter::create(&dir, "probe", Some(3))?;
        w.push_row(&[1.0, 2.0, 3.0])?;
        w.push_row(&[-1.5, 0.0, 1e6])?;
        let rows = w.finish()?;
        println!("写出 {rows} 行");
        c.check(rows == 2, "行数 2");

        let bytes = std::fs::read(dir.join("probe.npy"))?;
        println!("文件 {} 字节", bytes.len());
        c.check(bytes.len() == NPY_HEADER_LEN + 2 * 3 * 4, "字节数 = 头 + 2x3x4");
        let txt = String::from_utf8_lossy(&bytes[10..NPY_HEADER_LEN]).to_string();
        c.check(txt.contains("'shape': (2, 3)"), "回填后的 shape 是 (2, 3)");

        let mut got = Vec::new();
        for i in 0..6 {
            let off = NPY_HEADER_LEN + i * 4;
            got.push(f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]));
        }
        println!("读回 {got:?}");
        c.check(got == vec![1.0, 2.0, 3.0, -1.5, 0.0, 1e6], "数值逐个一致");

        // 行长不符必须报错，不能静默补零
        let mut w2: NpyWriter<u8> = NpyWriter::create(&dir, "bad", Some(4))?;
        c.check(w2.push_row(&[1, 2]).is_err(), "行长不符被拒绝");

        // 一维数组
        let mut w3: NpyWriter<i64> = NpyWriter::create(&dir, "flat", None)?;
        w3.push(10)?;
        w3.push(-20)?;
        w3.push(30)?;
        let n = w3.finish()?;
        let b3 = std::fs::read(dir.join("flat.npy"))?;
        c.check(n == 3 && b3.len() == NPY_HEADER_LEN + 3 * 8, "一维长度正确");

        std::fs::remove_dir_all(&dir)?;
        c.finish()
    }

    /// 格位摊平：单格补 -1，三格全填
    #[test]
    fn test_slots_row() -> Result<()> {
        use umasim::game::ramen::{
            policy_schema::PolicySlots,
            training_sample::{RamenCandidate, RamenSampleMeta}
        };
        let mut c = Checks::new();

        let one = RamenCandidate::from_rollouts(PolicySlots::One(7), &[Some(1.0), Some(2.0)])?;
        let three = RamenCandidate::from_rollouts(PolicySlots::Three([214, 220, 233]), &[Some(1.0), Some(2.0)])?;
        let sample = RamenTrainingSample {
            format_version: SAMPLE_FORMAT_VERSION,
            meta: RamenSampleMeta {
                index: 0,
                turn: 0,
                stage: 2,
                root_seed: 0
            },
            features: vec![0.0; INPUT_DIM],
            candidates: vec![one, three]
        };

        let r0 = slots_row(&sample, 0)?;
        let r1 = slots_row(&sample, 1)?;
        println!("单格 {r0:?}  三格 {r1:?}");
        c.check(r0 == [7, -1, -1], "单格右侧补 -1");
        c.check(r1 == [214, 220, 233], "三格全填");

        c.finish()
    }
}
