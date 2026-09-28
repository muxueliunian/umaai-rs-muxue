//! 客户端口径的地区对照最小验证入口（需要 `onnx` feature）
//!
//! # 为什么需要它
//!
//! `crates/umasim` 下的整局基准（`ramen_mcts_pair_bench`）覆盖不到本任务的客户端
//! 路径：依赖方向是 `umaai → umasim`，umasim 侧的 bin **不允许**反向依赖 umaai，
//! 因此拿不到 [`umaai::region::build_client_trainer`] 这条装配。本入口补的就是这一层，
//! 且**不复制装配**——训练员一律由 `build_client_trainer` 造，与主程序同一条路。
//!
//! # 它验证的三件事
//!
//! 1. **参照侧来源照实**：`ramen_search_stages` 含 `region` 时对照侧是真搜索
//!    （`kind = "search"`），不含时落在手写基策（`kind = "handwritten"`）。判据取自
//!    决策载荷本身，不靠猜配置。
//! 2. **对照不改变执行**：同一 `(种子, 局号)` 下两臂**逐决策相同**——比的是整局
//!    决策日志的每一行（回合 / 阶段 / 候选数 / 选中下标 / 动作描述 / 体力），
//!    不只是终局分（分数相同也可能是不同轨迹撞上同一个分）。
//!
//!    两臂的搜索阶段集**故意不同**：参照臂 `nn` 把 `--stages` 里的 `region` 去掉
//!    （完全接管与地区搜索互斥，这是既有校验的正确行为，不为测试放宽），对照臂
//!    `nn_compare` 用 `--stages` 原样。地区那一步本来就由网络执行、不进搜索，所以两臂
//!    唯一的差别就是**多出来的那次旁观地区搜索**。它若泄漏（推进了外层随机流、改了
//!    缓存或理由槽），逐决策对拍就会在后面某一行炸出来。
//! 3. **地区决策恰好 3 次**，来源标签与对照载荷（`executed_index` / `nn_index` /
//!    `candidates_total`）都与探针实际看到的一致。
//! 4. **`mcts_compare` 不改变既有装配**：`handwritten`（原装配）与 `mcts_compare`
//!    在同一 `(种子, 局号)`、同一 `--stages` 下逐决策相同。
//!
//!    这一组是冲着一个具体缺陷去的：`mcts_compare` 真正执行的是**搜索**那一条，而
//!    `FlatSearch::search` 会从传入的 rng 抽根种子。若搜索被跑在随机流副本上、网络
//!    反倒吃真实 rng，外层随机流的推进量就和原装配不同——本次动作可能还一样，后面
//!    的搜索却会分叉。只比终局分**看不出来**，必须逐决策比。
//!
//! # 用法（Release，从 workspace 根运行）
//!
//! ```text
//! # 参照侧 = 手写（本任务 §4 的示例配置：显式去掉 region）
//! cargo run --release -p umaai --features onnx --bin ramen_region_compare_check -- \
//!     --model saved_models/arms/ens_G2mix_g123.onnx --stages train,ramen --seeds 61444 --runs 1
//!
//! # 参照侧 = 真搜索（上游默认 stages）
//! cargo run --release -p umaai --features onnx --bin ramen_region_compare_check -- \
//!     --model saved_models/arms/ens_G2mix_g123.onnx --stages train,ramen,region --seeds 61444 --runs 1
//! ```
//!
//! ❗`--stages` 只作用于**本入口这一次运行**，不改任何配置文件，也不改默认值。

use std::{
    env,
    path::Path,
    process,
    sync::{Arc, Mutex}
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use lexopt::{Arg, ValueExt};
use rand::prelude::StdRng;
use serde::Deserialize;
use fs_err::read_to_string;
use serde_json::Value;
use umaai::region::{ClientTrainerParts, RamenClientTrainer, build_client_trainer};
use umasim::{
    bench::{self, CardPickOpts, GameOutcome, load_player_builds},
    game::{
        InheritInfo, Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{
        EventChoice, EventData, GameConfig, RamenRegionPolicy, RamenRegionStrategy, init_global_with_config
    },
    output::{
        DecisionInfo, DecisionLog, DecisionLogRow,
        decision::{SOURCE_REGION_HANDWRITTEN, SOURCE_REGION_NN, SOURCE_REGION_SEARCH}
    },
    search::SearchConfig,
    trainer::{LoggingTrainer, RamenMctsTrainer, RamenSearchStages},
    utils::{get_workspace_root, load_game_config}
};

/// `bench_config.toml` 里本入口用得到的基础字段（缺文件时用内置默认）
#[derive(Debug, Clone, Deserialize)]
struct BenchBase {
    /// 马娘 id
    uma: u32,
    /// 继承蓝因子
    blue_count: [i32; 5],
    /// 继承额外因子
    extra_count: [i32; 6],
    /// 友人卡 id（`make_deck` 用）
    friend: u32
}

impl Default for BenchBase {
    fn default() -> Self {
        Self {
            uma: 106402,
            blue_count: [15, 0, 0, 0, 3],
            extra_count: [0; 6],
            friend: 0
        }
    }
}

/// 命令行参数
#[derive(Debug, Clone)]
struct Args {
    /// 地区模型路径（含同名 `.json` 旁车）
    model: String,
    /// 基础种子列表
    seeds: Vec<u64>,
    /// 每个种子跑几局
    runs: u64,
    /// 本次运行生效的搜索阶段集（不改配置文件）
    stages: Option<String>,
    /// build 名过滤（默认取第一个）
    build: Option<String>,
    /// 覆盖 `search_n`（仅冒烟用）
    ///
    /// ❗覆盖后**不再代表生产档**，结果不得与生产档数据混比。留空 = 取
    /// `game_config.toml` 的 `[mcts] search_n`。
    search_n: Option<usize>
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: String::new(),
            seeds: vec![61444],
            runs: 1,
            stages: None,
            build: None,
            search_n: None
        }
    }
}

/// 一次地区决策留下的可观测物
#[derive(Debug, Clone)]
struct RegionObs {
    /// `select_action` 实际返回的下标（**探针自己看到的**，不是从载荷里读的）
    returned_idx: usize,
    /// 本决策点的候选数
    actions_len: usize,
    /// 决策来源标签（`region_nn` / `region_handwritten` / `region_search`）
    source: String,
    /// 对照载荷里参照侧的种类（`search` / `handwritten`；无对照时为空）
    baseline_kind: String,
    /// 两侧推荐是否一致（载荷里的 `agree`；无对照时为 `None`）
    agree: Option<bool>,
    /// 载荷里的 `executed_index`（用来和 [`Self::returned_idx`] 对账）
    executed_index: Option<usize>,
    /// 载荷里的 `nn_index`
    nn_index: Option<usize>,
    /// 载荷里的 `baseline_index`
    baseline_index: Option<usize>,
    /// 载荷里的 `candidates_total`
    candidates_total: Option<usize>,
    /// 是否带了对照载荷
    has_payload: bool
}

/// 包一层只读探针：转发全部决策，顺带抄走地区决策的摘要
///
/// 只读 [`Trainer::last_decision`] 这个既有公共出口，**不改变**任何决策、随机流或
/// 摘要内容；`last_decision` / `last_breakdown` 原样透传，`LoggingTrainer` 看到的
/// 东西与不包这一层时一致。
struct RegionProbe {
    /// 被包住的客户端训练员
    inner: RamenClientTrainer,
    /// 抄下来的地区决策
    seen: Arc<Mutex<Vec<RegionObs>>>
}

impl RegionProbe {
    /// 包住一个已装配好的客户端训练员
    fn new(inner: RamenClientTrainer, seen: Arc<Mutex<Vec<RegionObs>>>) -> Self {
        Self { inner, seen }
    }

    /// 从摘要里抽出来源与对照信息；不是地区决策时返回 `None`
    fn observe(info: &DecisionInfo, returned_idx: usize, actions_len: usize) -> Option<RegionObs> {
        let source = info.source_label()?.to_string();
        if !source.starts_with("region_") {
            return None;
        }
        let cmp = info
            .scenario_extra
            .as_ref()
            .and_then(|x| x.get("region_compare"));
        let num = |key: &str| -> Option<usize> {
            cmp.and_then(|c| c.get(key))
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        };
        Some(RegionObs {
            returned_idx,
            actions_len,
            source,
            baseline_kind: cmp
                .and_then(|c| c.get("baseline_kind"))
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            agree: cmp.and_then(|c| c.get("agree")).and_then(Value::as_bool),
            executed_index: num("executed_index"),
            nn_index: num("nn_index"),
            baseline_index: num("baseline_index"),
            candidates_total: num("candidates_total"),
            has_payload: cmp.is_some()
        })
    }
}

impl Trainer<RamenGame> for RegionProbe {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        let idx = self.inner.select_action(game, actions, rng)?;
        if let Some(info) = self.inner.last_decision() {
            if let Some(obs) = Self::observe(&info, idx, actions.len()) {
                self.seen
                    .lock()
                    .map_err(|e| anyhow!("地区观测锁中毒: {e}"))?
                    .push(obs);
            }
        }
        Ok(idx)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.inner.select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.inner.select_event_choice(game, event, choices, rng)
    }

    fn last_decision(&self) -> Option<DecisionInfo> {
        self.inner.last_decision()
    }

    fn last_breakdown(&self) -> Option<String> {
        self.inner.last_breakdown()
    }
}

/// 对一条对照臂的三次地区决策做完整校验
///
/// 两组配对（`nn` vs `nn_compare`、原装配 vs `mcts_compare`）**共用本函数**，免得
/// 一组查得细、另一组只看个 `all()` 就算过——空集合下 `all()` 恒真，那是个假绿。
///
/// `nn_primary` 为真表示执行的是网络那一条（`nn_compare`），为假表示执行的是参照
/// 那一条（`mcts_compare`）；执行侧的索引必须同时等于 `executed_index`，以及
/// `nn_index` / `baseline_index` 里对应的那一个。
///
/// 返回全部不符项；为空表示通过。
fn validate_regions(
    regions: &[RegionObs], want_source: &str, want_kind: &str, nn_primary: bool
) -> Vec<String> {
    let mut errs = Vec::new();
    // ❗先查条数：空集合会让下面所有的逐项检查**恒真**
    if regions.len() != 3 {
        errs.push(format!("地区决策 {} 次，应为 3 次", regions.len()));
    }
    for (i, r) in regions.iter().enumerate() {
        if r.source != want_source {
            errs.push(format!("#{i} 执行来源是 {} 而非 {want_source}", r.source));
        }
        if r.baseline_kind != want_kind {
            errs.push(format!("#{i} 参照种类是 {} 而非 {want_kind}", r.baseline_kind));
        }
        if !r.has_payload {
            errs.push(format!("#{i} 缺 region_compare 载荷"));
            continue;
        }
        if r.returned_idx >= r.actions_len {
            errs.push(format!(
                "#{i} 返回下标 {} 越界（候选数 {}）",
                r.returned_idx, r.actions_len
            ));
        }
        if r.candidates_total != Some(r.actions_len) {
            errs.push(format!(
                "#{i} 载荷 candidates_total={:?} 与实际候选数 {} 不一致",
                r.candidates_total, r.actions_len
            ));
        }
        if r.executed_index != Some(r.returned_idx) {
            errs.push(format!(
                "#{i} 载荷 executed_index={:?} 与实际返回下标 {} 不一致",
                r.executed_index, r.returned_idx
            ));
        }
        // 执行的是哪一侧，那一侧的索引就必须等于实际返回值
        let executed_side = match nn_primary {
            true => ("nn_index", r.nn_index),
            false => ("baseline_index", r.baseline_index)
        };
        if executed_side.1 != Some(r.returned_idx) {
            errs.push(format!(
                "#{i} 执行侧 {} = {:?}，应等于实际返回下标 {}",
                executed_side.0, executed_side.1, r.returned_idx
            ));
        }
        // 两侧索引都要在候选范围内
        for (name, v) in [("nn_index", r.nn_index), ("baseline_index", r.baseline_index)] {
            match v {
                None => errs.push(format!("#{i} 载荷缺 {name}")),
                Some(x) if x >= r.actions_len => {
                    errs.push(format!("#{i} {name}={x} 越界（候选数 {}）", r.actions_len))
                }
                Some(_) => {}
            }
        }
        // agree 必须与「两侧索引是否相同」自洽，不能是独立写进去的一个字段
        match (r.agree, r.nn_index, r.baseline_index) {
            (None, _, _) => errs.push(format!("#{i} 载荷缺 agree 字段")),
            (Some(a), Some(n), Some(b)) if a != (n == b) => errs.push(format!(
                "#{i} agree={a} 与 nn_index={n} / baseline_index={b} 不自洽"
            )),
            _ => {}
        }
    }
    errs
}

/// 一条臂的单局结果
struct ArmRun {
    /// 整局结果
    outcome: GameOutcome,
    /// 该局的三次地区决策
    regions: Vec<RegionObs>,
    /// 整局逐决策记录（用来做**动作级**对拍，不只是比终局分）
    log: DecisionLog
}

/// 逐决策对拍用的稳定键：去掉耗时与评分分解这类会抖的列
///
/// 两臂在同一 `(种子, 局号)` 下应当**逐行相同**。只比终局分不够：分数相同也可能是
/// 不同轨迹撞上同一个分。
fn stable_key(row: &DecisionLogRow) -> (i32, String, usize, usize, String, i32) {
    (
        row.turn,
        row.stage.clone(),
        row.candidates,
        row.action_index,
        row.action_desc.clone(),
        row.vital
    )
}

/// 解析命令行
///
/// # 错误
///
/// 参数缺失、取值非法或出现未知选项时报错。
fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("model") => args.model = parser.value()?.string()?,
            Arg::Long("seeds") => {
                let raw: String = parser.value()?.string()?;
                args.seeds = raw
                    .split(',')
                    .map(|s| s.trim().parse::<u64>().context("--seeds 需为逗号分隔的整数"))
                    .collect::<Result<Vec<_>>>()?;
            }
            Arg::Long("runs") => args.runs = parser.value()?.parse()?,
            Arg::Long("stages") => args.stages = Some(parser.value()?.string()?),
            Arg::Long("build") => args.build = Some(parser.value()?.string()?),
            Arg::Long("search-n") => args.search_n = Some(parser.value()?.parse()?),
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: ramen_region_compare_check --model <onnx> [--seeds a,b] [--runs N] \
                     [--stages train,ramen] [--build 名] [--search-n N]"
                );
                process::exit(0);
            }
            other => bail!("未知参数: {other:?}")
        }
    }
    ensure!(!args.model.trim().is_empty(), "--model 必填");
    ensure!(args.runs > 0, "--runs 必须为正");
    ensure!(!args.seeds.is_empty(), "--seeds 不能为空");
    Ok(args)
}

/// 按给定地区策略跑一局
///
/// # 错误
///
/// 装配失败（配置冲突 / 模型缺失）或整局推进报错时返回。
fn run_one(
    cfg: &GameConfig, stages: RamenSearchStages, search: &SearchConfig, policy: RamenRegionPolicy, uma: u32,
    deck: &[u32; 6], inherit: &InheritInfo, seed: u64, run_idx: u64
) -> Result<ArmRun> {
    let mut cfg = cfg.clone();
    cfg.ramen_region_policy = policy;
    let mcts = RamenMctsTrainer::new(search.clone())
        .with_stages(stages)
        .with_friend_complete_required(cfg.friend_complete_required)
        .verbose(false);
    let client = build_client_trainer(
        &cfg,
        stages,
        ClientTrainerParts {
            mcts,
            // 本入口不走屏幕对照行（它会打乱汇总输出）；对照结果从决策载荷里读
            human_mode: false,
            reason_gate: None
        }
    )?;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let probe = LoggingTrainer::new(RegionProbe::new(client, Arc::clone(&seen)), seed);
    let outcome = bench::run_seeded(uma, deck, inherit, seed, run_idx, &probe)?;
    let regions = seen
        .lock()
        .map_err(|e| anyhow!("地区观测锁中毒: {e}"))?
        .clone();
    Ok(ArmRun {
        outcome,
        regions,
        log: probe.take_records()
    })
}

fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    env::set_current_dir(&workspace_root).with_context(|| "切换到 workspace 根失败")?;
    let args = parse_args()?;
    ensure!(
        Path::new(&args.model).is_file(),
        "模型不存在: {}（本入口不伪造推理，缺模型直接停）",
        args.model
    );

    // 地区策略走 all：fixed 下第 3 年只剩 1 个候选，网络会被单候选短路
    let mut game_config = load_game_config()?;
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    game_config.ramen_region_model_path = Some(args.model.clone());
    init_global_with_config(&game_config)?;

    let mut search = SearchConfig::new_game_config(&game_config);
    if let Some(n) = args.search_n {
        search = search.with_search_n(n);
    }
    let stages = match &args.stages {
        Some(spec) => RamenSearchStages::parse(spec)?,
        None => RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?
    };
    // 参照臂把 region 摘掉：完全接管与地区搜索互斥（既有校验），不为测试放宽
    let plain_stages = RamenSearchStages {
        region_select: false,
        ..stages
    };
    let effective_stages = args
        .stages
        .clone()
        .unwrap_or_else(|| game_config.mcts.ramen_search_stages.clone());

    let bench_cfg = workspace_root.join("bench_config.toml");
    let base: BenchBase = match bench_cfg.exists() {
        true => toml::from_str(&read_to_string(&bench_cfg)?).context("解析 bench_config.toml 失败")?,
        false => {
            println!("提示: 未找到 bench_config.toml，使用内置默认参数");
            BenchBase::default()
        }
    };
    let builds = load_player_builds()?;
    let build = match &args.build {
        Some(want) => builds
            .into_iter()
            .find(|b| &b.name() == want)
            .with_context(|| format!("bench_config.toml 里没有 build: {want}"))?,
        None => builds
            .into_iter()
            .next()
            .context("bench_config.toml 的 [player_builds] 是空的")?
    };
    let deck = build.make_deck(&CardPickOpts::default(), base.friend)?;
    let inherit = InheritInfo {
        blue_count: base.blue_count,
        extra_count: base.extra_count
    };

    println!("模型          : {}", args.model);
    println!(
        "生效搜索阶段  : {effective_stages}（search_n={} ucb={}{}）",
        search.search_n,
        search.use_ucb,
        match args.search_n {
            Some(_) => "，❗search_n 被 --search-n 覆盖，非生产档",
            None => ""
        }
    );
    println!("build / uma   : {} / {}", build.name(), base.uma);
    println!("地区策略      : all（fixed 会被单候选短路，本入口强制 all）");
    println!();

    let mut mismatched = 0usize;
    for seed in &args.seeds {
        for run in 0..args.runs {
            // 参照臂：完全接管，阶段集去掉 region（互斥，不为测试放宽生产校验）
            let plain = run_one(
                &game_config,
                plain_stages,
                &search,
                RamenRegionPolicy::Nn,
                base.uma,
                &deck,
                &inherit,
                *seed,
                run
            )?;
            let compared = run_one(
                &game_config,
                stages,
                &search,
                RamenRegionPolicy::NnCompare,
                base.uma,
                &deck,
                &inherit,
                *seed,
                run
            )?;

            // 验收 1：地区决策恰好 3 次
            let n_cmp = compared.regions.len();
            let n_plain = plain.regions.len();
            // 验收 2：逐决策对拍（不只比终局）
            let a: Vec<_> = plain.log.rows.iter().map(stable_key).collect();
            let b: Vec<_> = compared.log.rows.iter().map(stable_key).collect();
            let first_diff = a.iter().zip(b.iter()).position(|(x, y)| x != y);
            let log_same = a.len() == b.len() && first_diff.is_none();
            let same_score = plain.outcome.score == compared.outcome.score;
            // 验收 3：来源标签与载荷对账（参与失败判定，不只是打印）
            let want_kind = match stages.region_select {
                true => "search",
                false => "handwritten"
            };
            let kinds: Vec<&str> = compared
                .regions
                .iter()
                .map(|r| r.baseline_kind.as_str())
                .collect();
            let sources: Vec<&str> = compared.regions.iter().map(|r| r.source.as_str()).collect();
            let agrees = compared
                .regions
                .iter()
                .filter(|r| r.agree == Some(true))
                .count();
            let label_errs = validate_regions(&compared.regions, SOURCE_REGION_NN, want_kind, true);

            let diff_desc = match first_diff {
                Some(i) => format!("首个分歧在第 {i} 行: {:?} vs {:?}", a[i], b[i]),
                None if a.len() == b.len() => "逐行相同".to_string(),
                None => "行数不同".to_string()
            };
            println!("seed={seed} run={run}");
            println!("  地区决策数     : nn={n_plain} nn_compare={n_cmp}（应均为 3）");
            println!(
                "  逐决策对拍     : {} 行 vs {} 行，{diff_desc}（终局评分 {} vs {}）",
                a.len(),
                b.len(),
                plain.outcome.score,
                compared.outcome.score
            );
            println!("  执行侧来源     : {sources:?}（期望全部 {SOURCE_REGION_NN}）");
            println!("  参照侧种类     : {kinds:?}（期望全部 {want_kind}；两侧推荐相同 {agrees}/{n_cmp} 次）");
            if label_errs.is_empty() {
                println!("  载荷对账       : 通过（executed_index / nn_index / candidates_total 均与实际一致）");
            } else {
                for e in &label_errs {
                    println!("  ❌ {e}");
                }
            }

            if n_plain != 3 || !log_same || !same_score || !label_errs.is_empty() {
                mismatched += 1;
            }

            // ---- 第二组：原装配 vs mcts_compare（两臂 stages 相同）----
            let plain_mcts = run_one(
                &game_config,
                stages,
                &search,
                RamenRegionPolicy::Handwritten,
                base.uma,
                &deck,
                &inherit,
                *seed,
                run
            )?;
            let mcts_cmp = run_one(
                &game_config,
                stages,
                &search,
                RamenRegionPolicy::MctsCompare,
                base.uma,
                &deck,
                &inherit,
                *seed,
                run
            )?;
            let c: Vec<_> = plain_mcts.log.rows.iter().map(stable_key).collect();
            let d: Vec<_> = mcts_cmp.log.rows.iter().map(stable_key).collect();
            let diff2 = c.iter().zip(d.iter()).position(|(x, y)| x != y);
            let log_same2 = c.len() == d.len() && diff2.is_none();
            let same_score2 = plain_mcts.outcome.score == mcts_cmp.outcome.score;
            let kinds2: Vec<&str> = mcts_cmp
                .regions
                .iter()
                .map(|r| r.baseline_kind.as_str())
                .collect();
            let sources2: Vec<&str> = mcts_cmp.regions.iter().map(|r| r.source.as_str()).collect();
            // mcts_compare 下执行侧是参照那一条：含 region 时是真搜索，否则是手写基策
            let want_src2 = match stages.region_select {
                true => SOURCE_REGION_SEARCH,
                false => SOURCE_REGION_HANDWRITTEN
            };
            let label_errs2 = validate_regions(&mcts_cmp.regions, want_src2, want_kind, false);
            let diff2_desc = match diff2 {
                Some(i) => format!("首个分歧在第 {i} 行: {:?} vs {:?}", c[i], d[i]),
                None if c.len() == d.len() => "逐行相同".to_string(),
                None => "行数不同".to_string()
            };
            println!(
                "  原装配 vs mcts_compare : {} 行 vs {} 行，{diff2_desc}（终局评分 {} vs {}）",
                c.len(),
                d.len(),
                plain_mcts.outcome.score,
                mcts_cmp.outcome.score
            );
            println!("  mcts_compare 执行来源  : {sources2:?}（期望全部 {want_src2}）");
            println!("  mcts_compare 参照种类  : {kinds2:?}（期望全部 {want_kind}）");
            if label_errs2.is_empty() {
                println!("  mcts_compare 载荷对账  : 通过（3 次决策、索引与 agree 均自洽）");
            } else {
                for e in &label_errs2 {
                    println!("  ❌ mcts_compare {e}");
                }
            }
            if !log_same2 || !same_score2 || !label_errs2.is_empty() {
                mismatched += 1;
            }
        }
    }

    println!();
    match mismatched {
        0 => {
            println!("全部通过：两组配对各自逐决策相同、地区决策数为 3、来源标签与对照载荷均与实际一致");
            println!("❗口径：「逐决策相同」比的是决策日志的稳定列，不是直接读 RNG 内部状态；");
            println!("  旁观侧若推进了随机流，会在后续某一行体现为分歧。");
            Ok(())
        }
        n => bail!("{n} 局不符合预期（详见上方逐局输出）")
    }
}
