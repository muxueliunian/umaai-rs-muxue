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

mod explicit_assets;

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
use serde_json::{Value, json};
use umasim::{
    collector::scan_part_files,
    game::ramen::{
        features::INPUT_DIM,
        policy_schema::POLICY_DIM,
        training_sample::{RamenSampleBatch, RamenTrainingSample, SAMPLE_FORMAT_VERSION, stage_of_code}
    },
    gamedata::init_global_with_config,
    sampler::{GEN1_SPACE_HASH_V1, SamplingSpace, space_from_cli, space_version_by_name},
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
    raw: bool,

    /// 采集时用的卡组构成 `速,耐,力,根,智`；与 `ramen_teacher_collect` 同名参数必须一致
    ///
    /// 不给时按第一代空间导出。给错会被空间指纹校验挡下——`index` 的含义由空间决定，
    /// 用错空间导出会让组合键整体指向别的卡组。
    #[arg(long)]
    shape: Option<String>,

    /// 采集时追加进卡池的支援卡 idrank；与 `--shape` 同用
    #[arg(long)]
    extra_card: Vec<u32>,

    /// 采集时用的具名采样空间版本（如 `gen2_v1`），与 `--shape` / `--extra-card` 互斥
    ///
    /// 给出即按**显式身份口径**导出：源目录必须都记有同一份逐字段空间清单，
    /// 且必须没有枚举指纹。全程不计算任何指纹，一致性靠字段直接比较。
    #[arg(long)]
    space_version: Option<String>
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
    /// 采集配方哈希；显式身份口径的目录没有这一项
    #[serde(default)]
    recipe_hash_fnv1a64: Option<String>,
    /// 采集时的 git commit
    git_commit: String,
    /// 采样空间枚举指纹；本字段加入之前采的目录为 `None`，按 v1 空间处理
    #[serde(default)]
    sampling_space_hash: Option<String>,
    /// 采样空间的显式身份；只有 `--space-version` 口径的目录才有
    ///
    /// 用 `Value` 原样收下：结构由采集端定义，这里只需**逐字段相等**比较，
    /// 不需要在本 bin 里复制一份类型定义。
    #[serde(default)]
    space: Option<Value>,
    /// 四条运行时前提的实际取值；显式口径下参与逐字段比较
    #[serde(default)]
    premises: Option<Value>,
    /// 采样器配置快照；显式口径下参与逐字段比较
    #[serde(default)]
    sampler: Option<Value>,
    /// Roll-in 名称；显式口径还需比较原始资产副本。
    #[serde(default)]
    rollin: Option<String>,
    /// 数据目录内的原始资产相对路径。
    #[serde(default)]
    asset_files: Option<Vec<String>>
}

/// 所有输入目录必须一致的那部分配方
///
/// 两种身份口径共用本类型：既有口径比 `recipe_hash`，显式口径 `recipe_hash` 为空、
/// 改比 `space` / `premises` / `sampler` 三份**实际字段**。
#[derive(Debug, Clone, PartialEq)]
struct SharedRecipe {
    /// 采集配方哈希；显式口径下为 `None`
    recipe_hash: Option<String>,
    /// git commit
    git_commit: String,
    /// 每候选 rollout 次数
    search_n: usize,
    /// 采样空间指纹；显式口径下为 `None`，既有目录未记录时回落到 [`GEN1_SPACE_HASH_V1`]
    sampling_space_hash: Option<String>,
    /// 采样空间的显式身份；既有口径下为 `None`
    space: Option<Value>,
    /// 四条运行时前提
    premises: Option<Value>,
    /// 采样器配置快照
    sampler: Option<Value>,
    /// 同一模型名称不能代替内容核对，run 还会逐字节比较原文资产。
    rollin: Option<String>,
    /// 两侧必须携带同一组资产。
    asset_files: Option<Vec<String>>
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
        // 两种口径互斥，且各自必须自洽：显式目录不得带指纹，指纹目录必须有配方哈希。
        match (&m.space, &m.recipe_hash_fnv1a64) {
            (Some(_), Some(_)) => bail!(
                "{} 同时带显式空间清单与配方指纹，身份口径自相矛盾",
                dir.display()
            ),
            (Some(_), None) => {
                ensure!(
                    m.sampling_space_hash.is_none(),
                    "{} 是显式身份口径的目录，却记有空间指纹",
                    dir.display()
                );
                ensure!(
                    m.premises.is_some() && m.sampler.is_some(),
                    "{} 缺 premises / sampler，显式口径无法做逐字段配方比较",
                    dir.display()
                );
                Ok(Self {
                    recipe_hash: None,
                    git_commit: m.git_commit.clone(),
                    search_n: m.search_n,
                    sampling_space_hash: None,
                    space: m.space.clone(),
                    premises: m.premises.clone(),
                    sampler: m.sampler.clone(),
                    rollin: m.rollin.clone(),
                    asset_files: m.asset_files.clone()
                })
            }
            (None, Some(hash)) => Ok(Self {
                recipe_hash: Some(hash.clone()),
                git_commit: m.git_commit.clone(),
                search_n: m.search_n,
                // 本字段加入前采的目录一律产自 v1 空间——现存的教师数据全部如此
                sampling_space_hash: Some(
                    m.sampling_space_hash
                        .clone()
                        .unwrap_or_else(|| GEN1_SPACE_HASH_V1.to_string())
                ),
                space: None,
                premises: None,
                sampler: None,
                rollin: None,
                asset_files: None
            }),
            (None, None) => bail!(
                "{} 既无配方指纹也无显式空间清单，无法判定采集身份",
                dir.display()
            )
        }
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
    /// (马娘, 卡组) 组合键，见 `DeckPlan::combo_key`
    combo_key: Option<NpyWriter<u64>>,
    /// 显式口径保存完整元组：[马娘, 六张卡按 ID 排序]，不生成哈希键。
    combo_fields: Option<NpyWriter<u64>>,
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
    fn create(dir: &Path, raw: bool, rollout_width: usize, explicit: bool) -> Result<Self> {
        Ok(Self {
            x: NpyWriter::create(dir, "x", Some(INPUT_DIM))?,
            stage: NpyWriter::create(dir, "stage", None)?,
            turn: NpyWriter::create(dir, "turn", None)?,
            index: NpyWriter::create(dir, "index", None)?,
            combo_key: if explicit { None } else { Some(NpyWriter::create(dir, "combo_key", None)?) },
            combo_fields: if explicit { Some(NpyWriter::create(dir, "combo_fields", Some(7))?) } else { None },
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
        if let Some(writer) = self.combo_key { writer.finish()?; }
        if let Some(writer) = self.combo_fields { writer.finish()?; }
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
    /// 采集配方哈希；显式身份口径下为 `None`
    recipe_hash_fnv1a64: Option<String>,
    /// 采集时的 git commit
    git_commit: String,
    /// 采样空间的枚举指纹，与 `plan_count` 取自同一个 space；显式口径下为 `None`
    sampling_space_hash: Option<String>,
    /// 采样空间的显式身份（原样转录源 manifest）；既有口径下为 `None`
    space: Option<Value>,
    /// 显式配方与模型来源，供后续标签/训练核对，不以名称替代原文资产。
    #[serde(skip_serializing_if = "Option::is_none")]
    sampler: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    premises: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollin: Option<String>,
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

/// 本次导出要求的采样空间身份口径
///
/// 与采集端的两种口径一一对应；两者不可混用，混了就没有任何字段能事后区分
/// 同一批 `.npy` 里的 index 属于哪个空间。
enum ExpectedSpace {
    /// 既有口径：身份是枚举指纹
    Fingerprint {
        /// 本次编译出的空间指纹
        hash: String
    },
    /// 显式口径：身份是版本名与逐字段清单
    Explicit {
        /// 版本名
        version: String
    }
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
    // plan_count 与空间指纹必须成对取自同一个 space：训练侧按 `sample_id % plan_count`
    // 切留出组合，取错了会让切分静默错位。
    let (space, expected_space) = match args.space_version.as_deref() {
        None => {
            let space = space_from_cli(args.shape.as_deref(), &args.extra_card)?;
            let hash = space.content_hash();
            (space, ExpectedSpace::Fingerprint { hash })
        }
        Some(name) => {
            ensure!(
                args.shape.is_none() && args.extra_card.is_empty(),
                "--space-version 与 --shape / --extra-card 互斥"
            );
            let version = space_version_by_name(name)?;
            let space = SamplingSpace::from_version(version)?;
            (space, ExpectedSpace::Explicit {
                version: version.name.to_string()
            })
        }
    };
    let plan_count = space.len();

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
                "{} 的采集配方与前面的目录不一致：{:?} vs {:?}。既有口径认 recipe_hash + git_commit，\
                 显式口径逐字段认 space / premises / sampler，都不要合并不同配方的数据",
                dir.display(),
                recipe,
                first
            )
        }
        if manifest.space.is_some() {
            let names = manifest.asset_files.as_ref().ok_or_else(|| anyhow::anyhow!(
                "{} 缺原文资产，不接受 size/mtime 代替内容", dir.display()
            ))?;
            ensure!(!names.is_empty(), "资产清单为空");
            let reference = scanned.first().map_or(dir, |(path, _): &(PathBuf, Vec<PathBuf>)| path);
            explicit_assets::compare_sets(reference, dir, names)?;
            // 枚举依赖当前 gamedata，不能只比较两个源目录却忽略当前导出环境。
            for name in names.iter().filter(|n| n.starts_with("assets/gamedata/")) {
                explicit_assets::ensure_same_bytes(&dir.join(name), Path::new(&name[7..]))?;
            }
        }
        println!("源 {:<40} {:3} 个分片", dir.display(), parts.len());
        scanned.push((dir.clone(), parts));
    }
    let shared = shared.context("没有可用的输入目录")?;

    // ❗本次编译的采样空间必须就是数据被采时的那个空间。卡池与构成写在 sampler.rs 里，
    // 改动它既不会动 gamedata_sig 也不会动 recipe_hash——不做这个校验的话，扩空间之后
    // 重导旧目录会把 plan_count 静默改写成新值，训练侧的组合切分随之整体错位，
    // 而且没有任何一处会报错。
    let space_note = match (&expected_space, &shared.sampling_space_hash, &shared.space) {
        (ExpectedSpace::Fingerprint { hash }, Some(recorded), _) => {
            ensure!(
                recorded == hash,
                "数据采自采样空间 {}，本次编译的空间是 {}（{} 个组合）。\
                 index 的含义由空间决定，用当前空间导出旧数据会让 plan_count 与组合切分静默错位。\
                 请用采集时的代码版本导出，或为新空间新建独立的导出目录。",
                recorded,
                hash,
                plan_count
            );
            format!("枚举指纹 {hash}")
        }
        (ExpectedSpace::Fingerprint { .. }, None, _) => bail!(
            "源目录是显式身份口径采的，导出必须同样给 --space-version"
        ),
        (ExpectedSpace::Explicit { version }, _, Some(recorded)) => {
            let definition = space_version_by_name(version)?;
            let mut expected = json!({
                "version": definition.name,
                "umas": definition.umas.iter().map(|u| u.game_id).collect::<Vec<_>>(),
                "cards": definition.cards.iter().map(|c| c.idrank).collect::<Vec<_>>(),
                "shapes": definition.shapes.iter().map(|s| json!({
                    "counts": s.counts, "name": s.name
                })).collect::<Vec<_>>(),
                "plan_count": plan_count
            });
            // 与采集器同口径：必带卡为空时不写该字段，旧目录的比对逐字段不变
            if !definition.required.is_empty() {
                expected["required"] = json!(definition.required);
            }
            ensure!(recorded == &expected, "源空间完整字段与当前枚举定义不同");
            format!("显式版本 {version}")
        }
        (ExpectedSpace::Explicit { version }, _, None) => bail!(
            "本次给了 --space-version {version}，但源目录没有显式空间清单（是既有指纹口径采的）"
        )
    };
    println!("采样空间 {space_note}，{plan_count} 个 (马娘, 卡组) 组合");

    ensure!(!args.output_dir.exists() || args.output_dir.read_dir()?.next().is_none(),
        "导出目录非空，拒绝覆盖：{}", args.output_dir.display());
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("创建导出目录失败: {}", args.output_dir.display()))?;

    // rollout 宽度取第一个样本的槽位数，后续逐个校验。定长是 `cand_scores` 能摊成
    // 矩形数组的前提，不齐就必须报错而不是补零——补零会被训练侧当成真实分数。
    let rollout_width = first_rollout_width(&scanned)?;

    let explicit = shared.space.is_some();
    let mut arrays = ArraySet::create(&args.output_dir, args.raw, rollout_width, explicit)?;
    let mut stats = ExportStats {
        rollout_width,
        ..Default::default()
    };
    // 组合键表：每个采样计划一个，供样本按 `index % plan_count` 查表。
    // 它替代「训练侧自己算 index % plan_count」这一步——那个口径绑死在单一空间上，
    // 换空间后同一 index 指向别的组合，新旧数据因此无法合并按组合切分。
    let combo_keys: Vec<u64> = if explicit { Vec::new() }
        else { space.plans().iter().map(|plan| plan.combo_key()).collect() };
    let combo_fields: Vec<[u64; 7]> = space.plans().iter().map(|plan| {
        let mut row = [0u64; 7];
        row[0] = u64::from(plan.uma);
        let mut cards = plan.deck;
        cards.sort_unstable();
        for (slot, card) in row[1..].iter_mut().zip(cards) { *slot = u64::from(card); }
        row
    }).collect();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut cursor: i64 = 0;
    arrays.cand_ptr.push(cursor)?;

    // ---- 第二遍：摊平 ----
    for (dir, parts) in &scanned {
        let mut in_dir = 0usize;
        for part in parts {
            let batch = RamenSampleBatch::load_binary(part)?;
            for sample in &batch.samples {
                write_sample(
                    &mut arrays,
                    sample,
                    &combo_keys,
                    &combo_fields,
                    rollout_width,
                    args.raw,
                    &mut cursor,
                    &mut seen,
                    &mut stats
                )?;
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
        export_version: if explicit { 2 } else { 1 },
        format_version: SAMPLE_FORMAT_VERSION,
        input_dim: INPUT_DIM,
        policy_dim: POLICY_DIM,
        search_n: shared.search_n,
        recipe_hash_fnv1a64: shared.recipe_hash.clone(),
        git_commit: shared.git_commit.clone(),
        sampling_space_hash: shared.sampling_space_hash.clone(),
        space: shared.space.clone(),
        sampler: shared.sampler.clone(),
        premises: shared.premises.clone(),
        rollin: shared.rollin.clone(),
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
    combo_keys: &[u64],
    combo_fields: &[[u64; 7]],
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
    // 组合键取自空间的计划表，与 `SamplingSpace::spec_at` 的 `index % len` 分层同口径。
    // 空 `combo_keys` 在本函数被调用前已排除（空间非空是 SamplingSpace 的不变量）。
    let plan_index = (sample.meta.index % combo_fields.len() as u64) as usize;
    if let Some(writer) = &mut arrays.combo_key { writer.push(combo_keys[plan_index])?; }
    if let Some(writer) = &mut arrays.combo_fields { writer.push_row(&combo_fields[plan_index])?; }

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
        ensure!(cand.scores.iter().all(|v| v.is_finite()), "样本包含非有限 rollout 分数");
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

    /// 采样空间指纹：旧 manifest 回落到 v1，新 manifest 原样取用
    ///
    /// 这条护栏挡的是：扩了采样空间之后重新导出旧目录，`plan_count` 会被静默改写成
    /// 新值，而 `recipe_hash` 与 `gamedata_sig` 都察觉不到（卡池写在代码里）。
    #[test]
    fn test_sampling_space_hash_fallback() -> Result<()> {
        let mut c = Checks::new();
        let dir = Path::new("training_data/示例");
        let base = SourceManifest {
            format_version: SAMPLE_FORMAT_VERSION,
            input_dim: INPUT_DIM,
            policy_dim: POLICY_DIM,
            search_n: 512,
            recipe_hash_fnv1a64: Some("d80184067dad807f".into()),
            git_commit: "2f20806".into(),
            sampling_space_hash: None,
            space: None,
            premises: None,
            sampler: None,
            rollin: None,
            asset_files: None
        };

        let old = SharedRecipe::from_manifest(&base, dir)?;
        println!("旧 manifest（无字段）→ {:?}", old.sampling_space_hash);
        c.check(
            old.sampling_space_hash.as_deref() == Some(GEN1_SPACE_HASH_V1),
            "缺字段时回落到 GEN1_SPACE_HASH_V1"
        );

        let mut newer = base.clone();
        newer.sampling_space_hash = Some("0123456789abcdef".into());
        let recorded = SharedRecipe::from_manifest(&newer, dir)?;
        println!("新 manifest（有字段）→ {:?}", recorded.sampling_space_hash);
        c.check(
            recorded.sampling_space_hash.as_deref() == Some("0123456789abcdef"),
            "有字段时原样取用"
        );
        c.check(recorded != old, "两者不相等，跨目录一致性校验能分辨出来");

        c.finish()
    }

    /// 显式身份口径：源 manifest 带逐字段清单、不带任何指纹时能被正确识别
    ///
    /// 同时覆盖两种矛盾形态——既带清单又带配方指纹、以及两者都没有——都必须报错，
    /// 因为那说明该目录的身份口径无法判定，不能让它静默参与合并。
    #[test]
    fn test_explicit_space_identity_source() -> Result<()> {
        let mut c = Checks::new();
        let dir = Path::new("training_data/示例");
        let space = json!({
            "version": "gen2_v1",
            "umas": [100603],
            "cards": [302754],
            "shapes": [{"counts": [3, 1, 0, 0, 1], "name": "3速1耐1智1友"}],
            "plan_count": 4288
        });
        let explicit = SourceManifest {
            format_version: SAMPLE_FORMAT_VERSION,
            input_dim: INPUT_DIM,
            policy_dim: POLICY_DIM,
            search_n: 512,
            recipe_hash_fnv1a64: None,
            git_commit: "99d975f".into(),
            sampling_space_hash: None,
            space: Some(space.clone()),
            premises: Some(json!({"use_ucb": false})),
            sampler: Some(json!({"epsilon": 0.15})),
            rollin: Some("handwritten".into()),
            asset_files: Some(vec![])
        };
        let shared = SharedRecipe::from_manifest(&explicit, dir)?;
        println!("显式口径 → recipe_hash={:?} space={:?}", shared.recipe_hash, shared.space);
        c.check(shared.recipe_hash.is_none(), "显式口径不带配方指纹");
        c.check(shared.sampling_space_hash.is_none(), "显式口径不带空间指纹");
        c.check(shared.space.as_ref() == Some(&space), "空间清单原样转录");

        let mut both = explicit.clone();
        both.recipe_hash_fnv1a64 = Some("d80184067dad807f".into());
        match SharedRecipe::from_manifest(&both, dir) {
            Ok(_) => c.check(false, "同时带清单与指纹应当报错"),
            Err(e) => {
                println!("  [OK] 两种身份并存被拒: {e}");
                c.check(true, "同时带清单与指纹被拒");
            }
        }

        let mut neither = explicit.clone();
        neither.space = None;
        match SharedRecipe::from_manifest(&neither, dir) {
            Ok(_) => c.check(false, "两者都没有应当报错"),
            Err(e) => {
                println!("  [OK] 无身份被拒: {e}");
                c.check(true, "无身份被拒");
            }
        }

        c.finish()
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
