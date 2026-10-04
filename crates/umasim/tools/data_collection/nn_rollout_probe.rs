//! NN-as-rollout 探针：把 MCTS 的 rollout 基策换成神经网络，跑整局并计时
//!
//! # 为什么必须把预算压到极小
//!
//! rollout 基策每**一个决策点**都要被调用一次：一次 rollout 跑到终局要经历
//! 约 171 个决策点（单局实测 Train 69 / RamenSelect 61 / SpecialSelect 25 /
//! Event 15 / RegionSelect 3 / SuperRamenSelect 1，见 `ramen_mcts_trainer`）。
//! 而一次搜索有「候选数 × `search_n`」条 rollout，一局又有上百个搜索点。
//! 所以网络档的推理次数 ≈ 决策点数 × rollout 总数，是乘积级：
//!
//! ```text
//! 推理次数 ≈ 171 × (候选数 × search_n) × 搜索点数
//! ```
//!
//! 生产 `search_n=8192` 下这个数字是上亿级，一局跑不完；本探针默认
//! `search_n=4` + 只搜 `train` 阶段，目的只是量出「每步网络推理 × 步数」的
//! 实际代价，再按线性外推。**输出不是策略验收口径**，分数无意义。
//!
//! # 与 `mcts_rollout_switch_verify` 的关系
//!
//! 那个 bin 对照「手写 rollout」的四档预算；本 bin 加的是第五档
//! 「rollout 换网络」。固定局面（马娘 / 卡组 / 继承因子）与它保持一致，
//! 便于把两条探针的数字放在一起看。
//!
//! # 环境变量
//!
//! | 变量 | 默认 | 含义 |
//! |---|---|---|
//! | `NN_ROLLOUT_MODEL` | 必填 | ONNX 模型路径（同目录需 `<模型>.onnx.json`） |
//! | `NN_ROLLOUT_RUNS` | 1 | 跑几局 |
//! | `NN_ROLLOUT_SEED` | 61444 | 基种子（第 i 局 = `derive_seed(seed, [i])`） |
//! | `NN_ROLLOUT_SEARCH_N` | 4 | 每个搜索点的 rollout 条数 |
//! | `NN_ROLLOUT_STAGES` | train | 搜索阶段：`train` / `all` / `none` |
//! | `NN_ROLLOUT_CONTROL` | 0 | 置 1 时先用同一份配置跑一局**手写 rollout** 作对照 |
//!
//! # 用法
//!
//! ```powershell
//! $env:NN_ROLLOUT_MODEL = "saved_models/<模型>.onnx"
//! cargo run --release -p umasim --features onnx --bin nn_rollout_probe
//! ```

use std::{env, path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use umasim::{
    bench,
    game::{InheritInfo, Trainer, ramen::RamenGame},
    gamedata::init_global_with_config,
    search::SearchConfig,
    trainer::{RamenMctsTrainer, RamenNnTrainer, RamenSearchStages, LoggingTrainer},
    utils::{get_workspace_root, load_game_config}
};

/// 固定局面（与 `mcts_rollout_switch_verify` 同款 speed build）
const UMA: u32 = 102_601;
/// 固定卡组（满破 speed build，含固定友人）
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
/// 固定继承因子（与 `mcts_rollout_switch_verify` 一致）
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};

/// 跑 `runs` 局并逐局打印分数与耗时
///
/// `make_trainer` 每局构造一次训练员：`LoggingTrainer` 按局号绑定日志种子，
/// 与 `mcts_rollout_switch_verify` 同款调用形态（`bench::run_seeded` 收的是
/// 包装后的 `LoggingTrainer`）。
///
/// # 错误
///
/// 任一局报错时原样返回——探针必须让失败可见，不能吞掉。
fn run_arm<T>(label: &str, runs: u64, seed: u64, make_trainer: impl Fn(u64) -> T) -> Result<()>
where
    T: Trainer<RamenGame> + Send + Sync
{
    for run_idx in 0..runs {
        let trainer = LoggingTrainer::new(make_trainer(run_idx), seed + run_idx);
        let start = Instant::now();
        let outcome = bench::run_seeded(UMA, &DECK, &INHERIT, seed, run_idx, &trainer)?;
        println!(
            "  {label:<20} run={run_idx} score={:>6} game_ms={:>9.0} wall_ms={:>9.0}",
            outcome.score,
            outcome.elapsed_ms,
            start.elapsed().as_millis()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let model = env::var("NN_ROLLOUT_MODEL").context("需要设置 NN_ROLLOUT_MODEL=<onnx 路径>")?;
    let runs: u64 = env::var("NN_ROLLOUT_RUNS").unwrap_or_else(|_| "1".into()).parse()?;
    let seed: u64 = env::var("NN_ROLLOUT_SEED").unwrap_or_else(|_| "61444".into()).parse()?;
    let search_n: usize = env::var("NN_ROLLOUT_SEARCH_N").unwrap_or_else(|_| "4".into()).parse()?;
    let stages_name = env::var("NN_ROLLOUT_STAGES").unwrap_or_else(|_| "train".into());
    let control = env::var("NN_ROLLOUT_CONTROL").is_ok_and(|v| v == "1");

    let stages = match stages_name.as_str() {
        "train" => RamenSearchStages::train_only(),
        "all" => RamenSearchStages::all(),
        "none" => RamenSearchStages::none(),
        other => bail!("未知 NN_ROLLOUT_STAGES: {other}（可选 train / all / none）")
    };

    std::env::set_current_dir(get_workspace_root()?)?;
    init_global_with_config(&load_game_config()?)?;

    let config = SearchConfig::default().with_search_n(search_n).with_ucb(false);
    println!(
        "NN-rollout 探针: model={model} search_n={search_n} stages={stages_name} runs={runs} seed={seed} threads={}",
        rayon::current_num_threads()
    );

    if control {
        run_arm("handwritten-rollout", runs, seed, |_| {
            RamenMctsTrainer::new(config.clone()).with_stages(stages)
        })?;
    }

    let nn = Arc::new(RamenNnTrainer::load(Path::new(&model))?);
    run_arm("nn-rollout", runs, seed, |_| {
        RamenMctsTrainer::new(config.clone())
            .with_stages(stages)
            .with_nn_rollout(Arc::clone(&nn), None)
    })?;

    Ok(())
}
