//! 记录手写策略在教师样本各局面上的选择，导出为 `.npy`
//!
//! # 为什么需要它
//!
//! 训练侧算出的「期望后悔值」是**相对教师**（512 次 rollout 的 argmax）的，
//! 它回答「网络离教师有多远」，**不回答「网络比手写策略强还是弱」**——后者才是
//! 第一代的验收命题。缺的是同一批局面、同一个 Q 口径下**手写策略自己的后悔值**。
//!
//! 采样器是确定性的（`index → 局面` 完全可复现），所以补这个数**不需要任何
//! rollout**：逐个 index 重放出局面，问一次手写策略选哪个，把它落在 policy
//! 格位上即可。与 `cand_slots` 比对就能定位到候选下标。
//!
//! 手写策略的后悔值同时就是策略改进定理里的**改进幅度**——手写离自己 Q 的贪心
//! 有多远，也就是搜索蒸馏这条路最多能赚多少分。
//!
//! # 为什么 `RamenSelect` 要两步走
//!
//! 采集时 `RamenSelect` 用的是**合并候选表**（吃哪碗面 × 万能风味用法），而手写
//! 策略的 `decide_ramen` 只看 `action.ramen`、完全不读 `special_targets`。直接把
//! 合并表喂给它，等于让它随机挑风味用法，会**高估**它的后悔值、让网络显得偏好。
//! 所以此处按真实对局的方式跑两个阶段：先选面，再选风味用法，最后合成。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p umasim --bin ramen_handwritten_choice -- \
//!     --input training_data/npy_v1 --output-dir training_data/handwritten_v1
//! ```

use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex
};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::Serialize;
use umasim::{
    game::{
        Game,
        ramen::{
            RamenAction,
            RamenGame,
            RamenStage,
            policy_schema::{EAT_NONE, PolicySlots, slots_of},
            training_sample::stage_of_code
        }
    },
    gamedata::{EventChoice, RamenRegionStrategy, init_global_with_config},
    sampler::{SampleOutcome, SamplerConfig, SamplingSpace, sample_position},
    trainer::RecommendedRamenTrainer,
    game::traits::Trainer,
    utils::{get_workspace_root, load_game_config}
};

/// `.npy` 头部固定长度，必须是 64 的倍数
const NPY_HEADER_LEN: usize = 128;

/// 格位列宽，与导出器的 `cand_slots` 一致
const SLOTS_PER_CAND: usize = 3;

/// 参数
#[derive(Parser, Debug)]
#[command(about = "记录手写策略在教师样本各局面上的选择")]
struct ChoiceArgs {
    /// 教师数据的 `.npy` 导出目录（读其中的 index / stage）
    #[arg(long)]
    input: PathBuf,

    /// 输出目录
    #[arg(long)]
    output_dir: PathBuf
}

/// 一个样本上手写策略的选择结果
#[derive(Debug, Clone, Copy)]
struct ChoiceRow {
    /// 样本在数据集中的顺序下标
    order: usize,
    /// 样本 id
    index: u64,
    /// 手写策略选择所落的格位，不足补 `-1`
    slots: [i32; SLOTS_PER_CAND],
    /// 重放出的阶段编码，用于与数据集核对
    stage: u8
}

/// 记录首次决策的包装训练员
///
/// 只在内层策略被问到时把 `(阶段, 选中动作)` 记下来，本身不改变任何决策。
struct RecordingTrainer<'a> {
    /// 真正做决策的手写策略
    inner: &'a RecommendedRamenTrainer,
    /// 按调用顺序记录的 `(阶段, 选中动作)`
    records: Mutex<Vec<(RamenStage, RamenAction)>>
}

impl<'a> RecordingTrainer<'a> {
    /// 包装一个手写策略
    fn new(inner: &'a RecommendedRamenTrainer) -> Self {
        Self {
            inner,
            records: Mutex::new(Vec::new())
        }
    }

    /// 取出记录
    ///
    /// # 错误
    ///
    /// 互斥锁被毒化时报错。
    fn take(&self) -> Result<Vec<(RamenStage, RamenAction)>> {
        let mut guard = self.records.lock().map_err(|_| anyhow::anyhow!("记录锁被毒化"))?;
        Ok(std::mem::take(&mut *guard))
    }
}

impl Trainer<RamenGame> for RecordingTrainer<'_> {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        let idx = self.inner.select_action(game, actions, rng)?;
        let action = actions.get(idx).ok_or_else(|| anyhow::anyhow!("手写策略返回越界下标 {idx}"))?;
        if let Ok(mut guard) = self.records.lock() {
            guard.push((game.stage.clone(), action.clone()));
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.inner.select_choice(game, choices, rng)
    }
}

/// 把格位摊成定长行，不足补 `-1`
///
/// # 错误
///
/// 格位数超过 [`SLOTS_PER_CAND`] 时报错。
fn slots_to_row(slots: &PolicySlots) -> Result<[i32; SLOTS_PER_CAND]> {
    let s = slots.as_slice();
    ensure!(s.len() <= SLOTS_PER_CAND, "格位数 {} 超出 {SLOTS_PER_CAND}", s.len());
    let mut row = [-1i32; SLOTS_PER_CAND];
    for (i, &v) in s.iter().enumerate() {
        row[i] = i32::try_from(v).context("格位下标溢出 i32")?;
    }
    Ok(row)
}

/// 重放一个 index 的局面并问手写策略选哪个
///
/// # 错误
///
/// 采样失败、阶段与数据集不符，或策略报错时报错。
fn choice_at(space: &SamplingSpace, config: &SamplerConfig, order: usize, index: u64, want_stage: u8) -> Result<ChoiceRow> {
    let pos = match sample_position(space, config, index)? {
        SampleOutcome::Captured(pos) => *pos,
        other => bail!("index={index} 未捕获到决策点（{other:?}），与数据集不一致")
    };
    let got = stage_of_code(want_stage)?;
    ensure!(pos.stage == got, "index={index} 阶段不符：数据集 {got:?} vs 重放 {:?}", pos.stage);

    let hand = RecommendedRamenTrainer::new();
    let recorder = RecordingTrainer::new(&hand);
    let mut rng = pos.decision_rng.clone();

    let slots = if pos.stage == RamenStage::RamenSelect {
        // 真实对局路径：先跑 RamenSelect，再跑 SpecialSelect，最后合成
        let mut game = pos.game.clone();
        game.run_stage(&recorder, &mut rng)?;
        if game.stage == RamenStage::SpecialSelect || (game.next() && game.stage == RamenStage::SpecialSelect) {
            game.run_stage(&recorder, &mut rng)?;
        }
        let records = recorder.take()?;
        let ramen = records
            .iter()
            .find(|(st, _)| *st == RamenStage::RamenSelect)
            .map(|(_, a)| a.ramen)
            .ok_or_else(|| anyhow::anyhow!("index={index} 未记录到 RamenSelect 决策"))?;
        match ramen {
            None => PolicySlots::One(EAT_NONE),
            Some(region) => {
                let targets = records
                    .iter()
                    .find(|(st, _)| *st == RamenStage::SpecialSelect)
                    .and_then(|(_, a)| a.special_targets)
                    .unwrap_or([0, 0, 0]);
                slots_of(RamenStage::RamenSelect, &RamenAction::combined_select(Some(region), targets))?
            }
        }
    } else {
        // 其余阶段的候选表就是采样时那份，问一次即可
        let idx = hand.select_action(&pos.game, &pos.actions, &mut rng)?;
        let action = pos.actions.get(idx).ok_or_else(|| anyhow::anyhow!("手写策略返回越界下标 {idx}"))?;
        slots_of(pos.stage.clone(), action)?
    };

    Ok(ChoiceRow {
        order,
        index,
        slots: slots_to_row(&slots)?,
        stage: want_stage
    })
}

/// 构造定长 `.npy` 头部
///
/// # 错误
///
/// 字典超出 [`NPY_HEADER_LEN`] 时报错。
fn npy_header(descr: &str, rows: usize, cols: Option<usize>) -> Result<Vec<u8>> {
    let shape = match cols {
        Some(c) => format!("{rows}, {c}"),
        None => format!("{rows},")
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape}), }}");
    let room = NPY_HEADER_LEN - 10;
    ensure!(dict.len() < room, "npy 头字典过长");
    let mut out = Vec::with_capacity(NPY_HEADER_LEN);
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1u8, 0u8]);
    out.extend_from_slice(&u16::try_from(room).context("头长溢出")?.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(NPY_HEADER_LEN - 1, b' ');
    out.push(b'\n');
    Ok(out)
}

/// 一次性写出一个 `.npy`
///
/// # 错误
///
/// IO 失败时报错。
fn write_npy(dir: &Path, name: &str, descr: &str, cols: Option<usize>, rows: usize, body: &[u8]) -> Result<()> {
    let path = dir.join(format!("{name}.npy"));
    let mut f = File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
    f.write_all(&npy_header(descr, rows, cols)?)?;
    f.write_all(body)?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&npy_header(descr, rows, cols)?)?;
    Ok(())
}

/// 输出目录的元信息
#[derive(Debug, Serialize)]
struct ChoiceMeta {
    /// 样本数
    samples: usize,
    /// 来源数据目录
    source: String,
    /// 策略名
    policy: &'static str
}

fn main() -> Result<()> {
    let args = ChoiceArgs::parse();

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)
        .with_context(|| format!("切换到工作空间根失败: {}", workspace_root.display()))?;
    let mut game_config = load_game_config()?;
    // 与采集端同一条前提：fixed 下第 3 年地区选择会绕过 trainer 直接落地
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    init_global_with_config(&game_config)?;

    let index: Vec<u64> = read_u64_npy(&args.input.join("index.npy"))?;
    let stage: Vec<u8> = read_u8_npy(&args.input.join("stage.npy"))?;
    ensure!(index.len() == stage.len(), "index 与 stage 长度不一致");
    println!("重放 {} 条局面…", index.len());

    let space = SamplingSpace::gen1()?;
    let config = SamplerConfig::default();
    let start = std::time::Instant::now();
    let mut rows: Vec<ChoiceRow> = (0..index.len())
        .into_par_iter()
        .map(|i| choice_at(&space, &config, i, index[i], stage[i]))
        .collect::<Result<Vec<_>>>()?;
    rows.sort_by_key(|r| r.order);
    let elapsed = start.elapsed().as_secs_f64();

    std::fs::create_dir_all(&args.output_dir)?;
    let mut slot_body = Vec::with_capacity(rows.len() * SLOTS_PER_CAND * 4);
    let mut idx_body = Vec::with_capacity(rows.len() * 8);
    for r in &rows {
        for v in r.slots {
            slot_body.extend_from_slice(&v.to_le_bytes());
        }
        idx_body.extend_from_slice(&r.index.to_le_bytes());
    }
    write_npy(&args.output_dir, "handwritten_slots", "<i4", Some(SLOTS_PER_CAND), rows.len(), &slot_body)?;
    write_npy(&args.output_dir, "index", "<u8", None, rows.len(), &idx_body)?;
    let meta = ChoiceMeta {
        samples: rows.len(),
        source: args.input.display().to_string(),
        policy: "RecommendedRamenTrainer"
    };
    std::fs::write(args.output_dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    println!("完成 → {}  {:.1} s", args.output_dir.display(), elapsed);
    Ok(())
}

/// 读一维 `u64` 的 `.npy`
///
/// # 错误
///
/// 文件缺失或头部不是预期 dtype 时报错。
fn read_u64_npy(path: &Path) -> Result<Vec<u64>> {
    let bytes = std::fs::read(path).with_context(|| format!("读取 {} 失败", path.display()))?;
    let head = String::from_utf8_lossy(&bytes[10..NPY_HEADER_LEN]).to_string();
    ensure!(head.contains("'<u8'"), "{} 不是 u64 数组", path.display());
    let body = &bytes[NPY_HEADER_LEN..];
    ensure!(body.len().is_multiple_of(8), "{} 长度不是 8 的倍数", path.display());
    Ok(body.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect())
}

/// 读一维 `u8` 的 `.npy`
///
/// # 错误
///
/// 文件缺失或头部不是预期 dtype 时报错。
fn read_u8_npy(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("读取 {} 失败", path.display()))?;
    let head = String::from_utf8_lossy(&bytes[10..NPY_HEADER_LEN]).to_string();
    if !head.contains("'|u1'") {
        bail!("{} 不是 u8 数组", path.display());
    }
    Ok(bytes[NPY_HEADER_LEN..].to_vec())
}
