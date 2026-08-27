//! 基准测试 bin：固定种子批量跑批 + 决策日志 + 基线分布
//!
//! 对应手写策略计划 §8 主线第 1 步「先立地基」：没有基线无法量化改进。
//! 本 bin 产出 RandomTrainer 的基线分布（分数/PT/RMJ/耗时），
//! 并可选落盘每局决策轨迹（开发调参格式，见 `output::decision_log`）。
//!
//! 卡组来源：`bench_config.toml` 的 `[player_builds]`（玩家主流 build 预置），
//! 每个 build 用代表卡自动生成卡组后分组跑批；开头打印马娘名与各 build 卡组的卡名。
//! 运行设施（固定种子双 RNG、单局运行、统计、CSV）复用 [`umasim::bench`]。
//!
//! # 用法（Release）
//!
//! ```text
//! cargo run --release --bin bench_base -- [--runs N] [--seed S] [--log] [--out DIR]
//! ```
//!
//! 参数缺省时读取 workspace 根目录 `bench_config.toml`（不存在则用内置默认，与
//! `bench_config.toml` 一致）。固定种子下结果完全可复现：
//! 决策 RNG 与规则层 RNG（`RamenGame::set_internal_rng`）分别由 seed 派生。
//!
//! # 产出（默认 `logs/`）
//!
//! - `bench_base_results.csv`：每局一行（build、seed、分数、rank、五维、逐年 PT/吃面/地区、
//!   RMJ、自选比赛达标、耗时）
//! - `bench_base_decision_<build>_<seed>.csv`：仅 `--log` 时，每局一份决策轨迹
//! - 汇总打印：各 build 分组分数分布（mean/median/min/max/std）、RMJ 成功年数、自选比赛达标率、
//!   按阶段分组的决策耗时、吞吐

use anyhow::{Context, Result};
use lexopt::Arg;
use serde::Deserialize;
use umasim::{
    bench::{self, CardPickOpts, RESULTS_HEADER, load_player_builds, outcome_to_row},
    game::InheritInfo,
    gamedata::{GAMEDATA, RamenRegionStrategy, init_global_with_config},
    global,
    output::decision_log::DecisionLogRow,
    search::SearchConfig,
    trainer::{
        LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RamenSelection, RandomTrainer, RecommendedRamenTrainer
    },
    utils::{get_workspace_root, load_game_config}
};

/// bench_config.toml 的配置项（CLI 参数可覆盖同名项）
#[derive(Debug, Clone, Deserialize)]
struct BenchConfig {
    /// 马娘 ID
    uma: u32,
    /// 固定友人卡 idrank（build 卡组生成用）
    friend: u32,
    /// 种马蓝因子个数
    blue_count: [i32; 5],
    /// 种马额外属性
    extra_count: [i32; 6],
    /// 每个 build 的批量局数
    runs: usize,
    /// 基础种子（第 i 局 = seed + i）
    seed: u64,
    /// 输出目录（相对 workspace 根）
    out_dir: String,
    /// 是否落盘决策日志
    decision_log: bool,
    /// 训练员: "random"（基线）| "handwritten"（正式手写策略）| "mcts"（手写 + 扁平搜索）
    ///
    /// `handwritten` 档跑的是 **正式推荐手写策略**（[`RecommendedRamenTrainer`]：
    /// 三年分治 + 方案 E 残余折扣 + 动态属性平衡 + 吃面联动 + 体力门限等全机制）。
    /// 早期实现曾用纯策略核心 [`RamenHandwrittenTrainer`]（rollout 基策/组件对照用），
    /// 2026-08-26 手动局对比实测：核心 (47796) vs 推荐 (58380) 差 +10584，
    /// 基准语义修正为「当前最强手写策略」。
    trainer: String,
    /// mcts 专用：**每个候选**的 rollout 次数
    ///
    /// 不是每个决策点的总预算：均匀分配下一个决策点的实际 rollout 数是
    /// `候选数 × search_n`（`search_uniform` 对每个 action 各调一次
    /// `simulate_many(.., search_n, ..)`）。第 3 年地区选择有 C(10,3)=120 个
    /// 候选，`search_n=64` 在那一个点就是 7680 次 rollout。
    #[serde(default = "default_search_n")]
    search_n: usize,
    /// mcts 专用：搜哪些阶段（逗号分隔，见 `RamenSearchStages::parse`）
    #[serde(default = "default_search_stages")]
    search_stages: String,
    /// mcts 专用：是否用 UCB 分配预算（false 为均匀分配）
    #[serde(default = "default_search_ucb")]
    search_ucb: bool,
    /// mcts 专用：取分口径 "score" | "pt"
    #[serde(default = "default_search_selection")]
    search_selection: String,
    /// mcts 专用：激进度上限
    ///
    /// 缺省 **0.0**（取普通均值）而非 `SearchConfig::default()` 的 50.0：
    /// 非零时 `best_action_idx` 按 `weighted_mean(radical_factor)` 排序，偏向好运尾部，
    /// 那样测出来的就不是「搜索能否提高**均分**」。要复现 C++ 风格行为再手动调高。
    #[serde(default)]
    radical_factor_max: f64
}

/// `search_n` 缺省值：小预算档，够跑通又不至于把跑批时间拖爆
fn default_search_n() -> usize {
    64
}

/// `search_stages` 缺省值：只搜训练阶段（决策点最多、单点信息量最大）
fn default_search_stages() -> String {
    "train,ramen".to_string()
}

/// `search_ucb` 缺省值：均匀分配
///
/// 与 `SearchConfig::default()`（UCB 开）相反，原因不是「UCB 更差」，而是
/// bench 常用的小预算下 UCB 退化成均匀搜索、白付一层调度开销：
/// `search_ucb` 第一阶段给每个候选各跑一组，终止判据是 `max_planned >= search_n`，
/// 故 `search_group_size >= search_n` 时首组一结束就退出，自适应分配一次都不发生。
/// （首组本身已被 clamp 进 `search_n`，不再像早期那样超预算，见 `search_ucb`。）
/// 要让 UCB 真正起作用，必须把 `search_group_size` 调到远小于 `search_n`。
fn default_search_ucb() -> bool {
    false
}

/// `search_selection` 缺省值
fn default_search_selection() -> String {
    "score".to_string()
}

/// 内置默认值（与 bench_config.toml 保持一致；文件缺失时使用）
impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            uma: 102601,
            friend: 303054,
            blue_count: [12, 0, 0, 0, 6],
            extra_count: [10, 0, 0, 20, 20, 40],
            runs: 20,
            seed: 42,
            out_dir: "logs".to_string(),
            decision_log: false,
            trainer: "random".to_string(),
            search_n: default_search_n(),
            search_stages: default_search_stages(),
            search_ucb: default_search_ucb(),
            search_selection: default_search_selection(),
            radical_factor_max: 0.0
        }
    }
}

/// 解析 CLI 参数（`--key value` 或 `--key=value`），覆盖 bench 配置
fn apply_cli(mut cfg: BenchConfig) -> Result<BenchConfig> {
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("runs") => cfg.runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("seed") => cfg.seed = bench::parse_value(&mut parser, "seed")?,
            Arg::Long("log") => cfg.decision_log = true,
            Arg::Long("out") => cfg.out_dir = bench::parse_value(&mut parser, "out")?,
            Arg::Long("trainer") => cfg.trainer = bench::parse_value(&mut parser, "trainer")?,
            Arg::Long("search-n") => cfg.search_n = bench::parse_value(&mut parser, "search-n")?,
            Arg::Long("search-stages") => cfg.search_stages = bench::parse_value(&mut parser, "search-stages")?,
            Arg::Long("search-ucb") => cfg.search_ucb = bench::parse_value(&mut parser, "search-ucb")?,
            Arg::Long("search-selection") => {
                cfg.search_selection = bench::parse_value(&mut parser, "search-selection")?
            }
            Arg::Long("radical-factor") => {
                cfg.radical_factor_max = bench::parse_value(&mut parser, "radical-factor")?
            }
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: bench_base [--runs N] [--seed S] [--log] [--out DIR]
\n                     	[--trainer random|handwritten|mcts]
\n                     	mcts 专用: [--search-n N] [--search-stages train,ramen,...] [--search-ucb]
\n                     	           [--search-selection score|pt] [--radical-factor F] [--search-ucb true|false]\n\
                     缺省参数读取 workspace 根 bench_config.toml"
                );
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("未知参数: {other:?}（可用 --help 查看用法）");
            }
        }
    }
    Ok(cfg)
}

/// 读取 bench_config.toml（workspace 根）；缺失时用内置默认并提示
fn load_bench_config(workspace_root: &std::path::Path) -> Result<BenchConfig> {
    let path = workspace_root.join("bench_config.toml");
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取 bench_config.toml 失败: {}", path.display()))?;
        let cfg: BenchConfig =
            toml::from_str(&text).with_context(|| format!("解析 bench_config.toml 失败: {}", path.display()))?;
        Ok(cfg)
    } else {
        println!("提示: 未找到 bench_config.toml，使用内置默认参数");
        Ok(BenchConfig::default())
    }
}

/// 按决策阶段分组统计耗时（mean us / max us / 次数），按阶段名排序
fn summarize_decision_times(rows: &[DecisionLogRow]) -> Vec<(String, f64, u64, usize)> {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<String, (u128, u64, usize)> = BTreeMap::new();
    for r in rows {
        let e = acc.entry(r.stage.clone()).or_insert((0, 0, 0));
        e.0 += r.elapsed_us as u128;
        e.1 = e.1.max(r.elapsed_us);
        e.2 += 1;
    }
    acc.into_iter()
        .map(|(k, (sum, max, n))| (k, sum as f64 / n.max(1) as f64, max, n))
        .collect()
}

/// 单个 build 的分组跑批结果。
struct BuildResults {
    /// build 名。
    name: String,
    /// 每局结果。
    outcomes: Vec<bench::GameOutcome>
}

fn main() -> Result<()> {
    // 切换到 workspace 根（bench_config.toml / logs / gamedata 相对路径依赖）
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;

    let cfg = apply_cli(load_bench_config(&workspace_root)?)?;

    // 初始化全局数据（注入用户可调项：race_grades / mcts_turn_bonus 等）
    let mut game_config = load_game_config()?;
    // 基准测试模拟的是"策略决策"而非"玩家手动操作"——
    // Y3 地区选择必须交回策略（All 枚举），不受 game_config.toml / default_config.toml
    // 中为了手动模式而设的 `ramen_region_strategy="fixed"` 影响。
    // （手动模式 ramen_player / ramen_manual 才保留 fixed 配置。）
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;

    // 卡组来源：玩家 build 预置（每个 build 用代表卡自动生成卡组）
    let builds = load_player_builds()?;
    let data = global!(GAMEDATA);
    let uma_name = data.get_uma(cfg.uma)?.name.clone();

    let out_dir = workspace_root.join(&cfg.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let inherit = InheritInfo {
        blue_count: cfg.blue_count,
        extra_count: cfg.extra_count
    };

    println!(
        "===== bench_base: uma={} {} runs={} base_seed={} trainer={} builds={} =====",
        cfg.uma,
        uma_name,
        cfg.runs,
        cfg.seed,
        cfg.trainer,
        builds.len()
    );

    // mcts 参数提前解析：跑批循环里再报错等于跑了一半才发现参数拼错
    let search_stages = RamenSearchStages::parse(&cfg.search_stages)?;
    let search_selection = match cfg.search_selection.as_str() {
        "score" => RamenSelection::Score,
        "pt" => RamenSelection::Pt,
        other => anyhow::bail!("未知 search_selection: {other}（可选 score / pt）")
    };
    let search_config = SearchConfig::default()
        .with_search_n(cfg.search_n)
        .with_max_depth(0) // 拉面无 leaf 估值器，只能跑到终局
        .with_ucb(cfg.search_ucb)
        .with_radical_factor_max(cfg.radical_factor_max);
    if cfg.trainer == "mcts" {
        println!(
            "  mcts 参数: search_n={}/候选 stages={} ucb={} selection={} radical_factor_max={}",
            cfg.search_n, cfg.search_stages, cfg.search_ucb, cfg.search_selection, cfg.radical_factor_max
        );
    }

    let pick = CardPickOpts::default();
    let mut all_results: Vec<BuildResults> = Vec::with_capacity(builds.len());
    let mut all_rows: Vec<DecisionLogRow> = Vec::new();
    for (idx, build) in builds.iter().enumerate() {
        let deck = build.make_deck(&pick, cfg.friend)?;
        // 打印卡组信息（含卡名）
        let cards_desc = deck
            .iter()
            .map(|id| match data.get_card(id / 10) {
                Ok(card) => format!("{} {}", id, card.card_name),
                Err(_) => id.to_string()
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("[{}] {} 卡组: [{}]", idx + 1, build.name(), cards_desc);

        let mut outcomes = Vec::with_capacity(cfg.runs);
        for i in 0..cfg.runs {
            let run_idx = i as u64;
            let log_seed = cfg.seed + run_idx; // 决策日志标签（局号可读）
            // 构造训练员（LoggingTrainer 包装；决策日志默认开启，由 --log 决定是否落盘）
            let (outcome, log) = match cfg.trainer.as_str() {
                "random" => {
                    let trainer = LoggingTrainer::new(RandomTrainer, log_seed);
                    let outcome = bench::run_seeded(cfg.uma, &deck, &inherit, cfg.seed, run_idx, &trainer)?;
                    (outcome, trainer.take_records())
                }
                "handwritten" => {
                    let trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), log_seed);
                    let outcome = bench::run_seeded(cfg.uma, &deck, &inherit, cfg.seed, run_idx, &trainer)?;
                    (outcome, trainer.take_records())
                }
                "mcts" => {
                    let mcts = RamenMctsTrainer::new(search_config.clone())
                        .with_stages(search_stages)
                        .with_selection(search_selection);
                    let trainer = LoggingTrainer::new(mcts, log_seed);
                    let outcome = bench::run_seeded(cfg.uma, &deck, &inherit, cfg.seed, run_idx, &trainer)?;
                    (outcome, trainer.take_records())
                }
                other => anyhow::bail!("未知 trainer: {other}（可选 random / handwritten / mcts）")
            };
            println!(
                "  [#{:02}] seed={} score={} ({}) PT={}/{}/{} RMJ={}/3 自选比赛={} 耗时={:.3}ms",
                i + 1,
                outcome.seed,
                outcome.score,
                outcome.rank,
                outcome.yearly_scenario_pt[0],
                outcome.yearly_scenario_pt[1],
                outcome.yearly_scenario_pt[2],
                outcome.rmj_ok,
                if outcome.free_race_ok { "达标" } else { "未达标" },
                outcome.elapsed_ms,
            );
            if cfg.decision_log {
                log.save_to(&out_dir.join(format!("bench_base_decision_{}_{}.csv", build.name(), run_idx)))?;
            }
            all_rows.extend(log.rows);
            outcomes.push(outcome);
        }

        // 本 build 分组汇总
        let scores: Vec<f64> = outcomes.iter().map(|r| r.score as f64).collect();
        let stats = bench::summarize(&scores);
        let rmj_mean = outcomes.iter().map(|r| r.rmj_ok as f64).sum::<f64>() / outcomes.len().max(1) as f64;
        println!(
            "  {} 汇总: mean={:.0} median={:.0} min={:.0} max={:.0} std={:.0} RMJ={:.2}/3 自选比赛达标={:.0}%",
            build.name(),
            stats.mean,
            stats.median,
            stats.min,
            stats.max,
            stats.std,
            rmj_mean,
            free_race_rate(&outcomes) * 100.0,
        );
        all_results.push(BuildResults { name: build.name(), outcomes });
    }

    // ===== 落盘结果 CSV（合并单文件，build 列为第一列）=====
    let results_path = out_dir.join("bench_base_results.csv");
    let rows: Vec<Vec<String>> = all_results
        .iter()
        .flat_map(|r| r.outcomes.iter().map(|o| outcome_to_row(&r.name, o)))
        .collect();
    bench::write_csv(&results_path, &RESULTS_HEADER, &rows)?;
    println!("\n结果已写入: {}", results_path.display());

    // ===== 总览 =====
    println!("\n===== 总览 (各 build 分数分布) =====");
    for r in &all_results {
        let scores: Vec<f64> = r.outcomes.iter().map(|o| o.score as f64).collect();
        let stats = bench::summarize(&scores);
        let rmj_mean = r.outcomes.iter().map(|o| o.rmj_ok as f64).sum::<f64>() / r.outcomes.len().max(1) as f64;
        let elapsed_mean = r.outcomes.iter().map(|o| o.elapsed_ms).sum::<f64>() / r.outcomes.len().max(1) as f64;
        println!(
            "{:<14} mean={:>7.0} median={:>7.0} min={:>6.0} max={:>6.0} RMJ={:.2}/3 自选比赛={:>3.0}% 耗时={:.2}ms",
            r.name,
            stats.mean,
            stats.median,
            stats.min,
            stats.max,
            rmj_mean,
            free_race_rate(&r.outcomes) * 100.0,
            elapsed_mean,
        );
    }

    // ===== 全局决策耗时 =====
    let elapsed_ms: Vec<f64> = all_results
        .iter()
        .flat_map(|r| r.outcomes.iter().map(|o| o.elapsed_ms))
        .collect();
    let total_ms = elapsed_ms.iter().sum::<f64>();
    let total_runs = all_results.iter().map(|r| r.outcomes.len()).sum::<usize>();
    let throughput = total_runs as f64 / (total_ms / 1000.0).max(1e-9);
    println!("\n决策耗时 (mean/max us, 次数):");
    for (stage, m, max_us, n) in summarize_decision_times(&all_rows) {
        println!("  {stage:<14} {m:>8.1} {max_us:>8} {n:>6}");
    }
    println!(
        "整局耗时: mean {:.3}ms, 吞吐 {:.1} 局/s（共 {total_runs} 局）",
        elapsed_stats_mean(&elapsed_ms),
        throughput
    );
    Ok(())
}

/// 一组结果的自选比赛达标率（0.0-1.0），空序列返回 0。
///
/// 不达标即育成失败，故本值偏低时分数分布会被大量早停局拉垮——基线对比时先看这一项。
fn free_race_rate(outcomes: &[bench::GameOutcome]) -> f64 {
    let ok = outcomes.iter().filter(|o| o.free_race_ok).count();
    ok as f64 / outcomes.len().max(1) as f64
}

/// 计算一列耗时的均值（helper，避免 inline 过长）。
fn elapsed_stats_mean(elapsed_ms: &[f64]) -> f64 {
    elapsed_ms.iter().sum::<f64>() / elapsed_ms.len().max(1) as f64
}
