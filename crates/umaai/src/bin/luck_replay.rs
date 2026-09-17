//! 拉面剧本运气分重放分析工具（方案 A：独立 replay binary）
//!
//! ## 用途
//!
//! 对 `logs/SendGameStatusPlugin/` 下**实际运行时收集**的回合快照（`game{chara}_turn{turn}[_{seq}].json`）
//! 按 (chara, turn, seq) 顺序重放，逐份复用在线 AI 的完整判定链路：
//!
//! - `parse_game_by_scenario` → `GameStatusRamen::into_game`：协议 stage dispatch（event /
//!   RMJ 结算 / 数据获取不全 / 超级拉面丢包等**不派发**的快照自动落 `Begin`，不进决策循环）；
//!   `command + active_effect 有` 写 `pending_ramen` 精准恢复「已吃面 → 选训练」中间态。
//! - `scenario::ramen::process_ramen`：切局检测 / 链式连续决策 / `emit_with_luck_decision`
//!   （T(n) 按局数加权 baseline + `LuckScoreTracker` 逐决策点更新运气分）。
//! - 捕获 sink：把每次 `sink.emit` 的 `DecisionInfo`（候选 top-5 描述 / 评分 / 局数、选中
//!   动作、决策种类、luck snapshot）记成 CSV，另出一份按局聚合的**波动统计** summary CSV。
//!
//! ## 与在线行为的一致性
//!
//! 判定规则**不在此复刻**——直接复用 `process_ramen`，什么快照计算 / 跳过 / 更新运气分
//! 与在线 AI 完全一致。CSV 的 `skip_reason` 只是事后按协议层同一顺序给「不派发」的快照
//! 标注原因（event / playing_state=5 / RMJ 46·48 / 数据获取不全 / 超级拉面丢包），供分析过滤。
//!
//! 玩家实际选择不存在于样本中：本工具输出的是 AI 在该快照局面的**推荐**（候选 + 选中 +
//! 选项后运气分）；回合/全局运气分只依赖局面序列自身的期望终局分演化（`T(n+1) − T(n)`），
//! 与玩家是否按推荐执行无关。
//!
//! ## 用法
//!
//! ```text
//! cargo run --release --bin luck-replay -- [--dir logs/SendGameStatusPlugin] [--out luck_replay]
//!     [--search-n 8192] [--seed 42] [--limit N]
//! ```
//!
//! 产物：`{out}.csv`（逐决策点明细）+ `{out}_summary.csv`（每局运气分波动统计）。
//! stdout 保持干净（只用 stderr 打进度）；MCTS 预算默认取 `game_config.toml` 的 `search_n`。

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
    time::Instant
};

use anyhow::{Result, anyhow};
use lexopt::{Arg, ValueExt};
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::Game,
    gamedata::{GAMECONSTANTS, init_global_with_config},
    global,
    output::{DecisionInfo, DecisionSink, GameView},
    search::SearchConfig,
    trainer::{RamenMctsTrainer, RamenSearchStages},
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

use umaai::{
    decision::{
        LastReasonSink,
        LuckScoreTracker,
        record::{build_row, classify_begin_reason, detail_header, num_col, raw_from_display, CsvSink, SnapRef}
    },
    protocol::{ParsedGame, parse_game_by_scenario},
    scenario::ramen::process_ramen
};

/// CLI 参数（lexopt，与项目主 bin 惯例一致）
#[derive(Debug)]
struct CliArgs {
    /// 快照目录（默认 `logs/SendGameStatusPlugin`）
    dir: PathBuf,
    /// CSV 输出前缀（默认 `luck_replay` → `luck_replay.csv` / `luck_replay_summary.csv`）
    out: PathBuf,
    /// MCTS 预算覆盖（默认取 `game_config.toml` 的 `search_n`）
    search_n: Option<usize>,
    /// 固定随机种子（默认 OS 熵，与在线 AI 口径一致）
    seed: Option<u64>,
    /// 只处理前 N 份快照（调试用）
    limit: Option<usize>,
    /// 只重放指定局（逗号分隔，如 `7076,7077`；默认全部）
    games: Option<String>
}

/// 解析 CLI（`--dir` / `--out` / `--search-n` / `--seed` / `--limit` / `-h`）
fn parse_cli() -> Result<CliArgs> {
    let mut args = CliArgs {
        dir: PathBuf::from("logs/SendGameStatusPlugin"),
        out: PathBuf::from("luck_replay"),
        search_n: None,
        seed: None,
        limit: None,
        games: None
    };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("dir") => args.dir = PathBuf::from(parser.value()?.string()?),
            Arg::Long("out") => args.out = PathBuf::from(parser.value()?.string()?),
            Arg::Long("search-n") => args.search_n = Some(parser.value()?.parse()?),
            Arg::Long("seed") => args.seed = Some(parser.value()?.parse()?),
            Arg::Long("limit") => args.limit = Some(parser.value()?.parse()?),
            Arg::Long("games") => args.games = Some(parser.value()?.string()?),
            Arg::Short('h') | Arg::Long("help") => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected().into())
        }
    }
    Ok(args)
}

/// 打印帮助（stdout，与 `-h` 语义一致）
fn print_help() {
    println!("luck-replay — 拉面运气分重放分析");
    println!();
    println!("用法: luck-replay [options]");
    println!();
    println!("选项:");
    println!("  --dir <path>      快照目录（默认 logs/SendGameStatusPlugin）");
    println!("  --out <prefix>    CSV 输出前缀（默认 luck_replay）");
    println!("  --search-n <n>    MCTS 预算覆盖（默认取 game_config.toml 的 search_n）");
    println!("  --seed <u64>      固定随机种子（默认 OS 熵）");
    println!("  --limit <n>       只处理前 N 份快照（调试）");
    println!("  --games <ids>     只重放指定局（逗号分隔，如 7076,7077）");
    println!("  -h, --help        打印本帮助");
}

/// 一份快照的定位信息（由文件名解析，`game7075_turn13_2.json` → `(7075, 13, 2)`）
#[derive(Debug)]
struct SnapshotRef {
    /// 育成局（文件名 game 前缀，与 `single_mode_chara_id` 一致）
    game: u64,
    /// 回合
    turn: u32,
    /// 同回合内写入序号（无后缀为 0；`_2`/`_3`… 为同回合多个决策点）
    seq: u32,
    /// 文件绝对路径
    path: PathBuf
}

/// 解析快照文件名 → 定位信息（`None` = 非快照文件，如 `thisTurn.json`）
///
/// 格式：`game{game}_turn{turn}.json` 或 `game{game}_turn{turn}_{seq}.json`。
fn parse_file_name(name: &str) -> Option<(u64, u32, u32)> {
    let stem = name.strip_suffix(".json")?;
    let body = stem.strip_prefix("game")?;
    let (game_s, rest) = body.split_once("_turn")?;
    let game = game_s.parse::<u64>().ok()?;
    let (turn_s, seq_s) = match rest.split_once('_') {
        Some((t, s)) => (t, Some(s)),
        None => (rest, None)
    };
    let turn = turn_s.parse::<u32>().ok()?;
    let seq = match seq_s {
        Some(s) => s.parse::<u32>().ok()?,
        None => 0
    };
    Some((game, turn, seq))
}

/// 扫描目录收集并按 (game, turn, seq) 排序全部快照
fn collect_snapshots(dir: &Path) -> Result<Vec<SnapshotRef>> {
    let mut snaps = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let Some((game, turn, seq)) = parse_file_name(&name) else {
            continue;
        };
        snaps.push(SnapshotRef { game, turn, seq, path: entry.path() });
    }
    snaps.sort_by_key(|s| (s.game, s.turn, s.seq));
    if snaps.is_empty() {
        return Err(anyhow!("目录 {} 下未找到 game*_turn*.json 快照", dir.display()));
    }
    Ok(snaps)
}

/// 捕获 sink：把每次 `sink.emit` 的决策原始数据按序存入内部缓冲
///
/// 每个快照处理前 `take()` 清空取走；`process_ramen` 连式决策可能产生多条
/// （中间项直接 emit 不触 luck，末项带 luck 挂载）。
struct CaptureSink {
    inner: Mutex<Vec<(DecisionInfo, GameView)>>
}

impl CaptureSink {
    fn new() -> Self {
        Self { inner: Mutex::new(Vec::new()) }
    }

    /// 取走并清空缓冲（每次快照处理前调用；处理结束读取本次 emit 的所有决策）
    fn take(&self) -> Vec<(DecisionInfo, GameView)> {
        std::mem::take(&mut *self.inner.lock().expect("capture sink"))
    }
}

impl DecisionSink for CaptureSink {
    fn emit(&self, info: &DecisionInfo, view: &GameView) {
        self.inner
            .lock()
            .expect("capture sink")
            .push((info.clone(), view.clone()));
    }
}

// 极简 CSV 转义 / CSV 写入器 / 明细列头 / raw 反推 / Begin 分类 / 行构造
// 已统一抽到 `umaai::decision::record`（与在线记录共用同一份 schema）——
// 本文件直接复用，保证在线 / 离线 CSV 逐行可比对。

/// 按局聚合的运气分波动统计（写明细行的同时累积）
#[derive(Default)]
struct GameStats {
    /// 处理的快照数
    snaps: u64,
    /// 触发计算（`process_ramen` 跑过 MCTS）的快照数
    calc_snaps: u64,
    /// 计算但无决策可出的快照数（NextTurn / Settlement / 无候选阶段）
    no_emit_snaps: u64,
    /// emit 决策行总数（含链式中间项与末项）
    emit_rows: u64,
    /// 各 skip 原因计数（key 均为 `&'static str`）
    skip: HashMap<&'static str, u64>,
    /// luck 更新行数（带 T(n) 的行）
    luck_rows: u64,
    /// 回合运气分（显示分口径）序列统计
    dd: DeltaStats,
    /// 回合运气分（raw 口径：显示分反推 T(n) 后的相邻差）序列统计
    dr: DeltaStats,
    /// 上一 luck 行的 raw T(n)（相邻 luck 行才能算 raw delta）
    prev_raw: Option<f64>,
    /// 终局全局运气分（最后一行 total_luck）
    total_luck_end: Option<f64>
}

/// 一列数值序列统计（单遍累积：count / mean / std / min / max / sum / 正负次数）
#[derive(Default)]
struct DeltaStats {
    n: u64,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
    neg: u64,
    pos: u64
}

impl DeltaStats {
    fn push(&mut self, v: f64) {
        self.n += 1;
        self.sum += v;
        self.sum_sq += v * v;
        if self.n == 1 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        if v < 0.0 {
            self.neg += 1;
        } else if v > 0.0 {
            self.pos += 1;
        }
    }

    /// (n, mean, std, min, max, sum, neg, pos)；空序列统计值为空
    fn report(&self) -> (u64, String, String, String, String, String, u64, u64) {
        let mean = self.sum / self.n.max(1) as f64;
        let var = (self.sum_sq / self.n.max(1) as f64 - mean * mean).max(0.0);
        (
            self.n,
            if self.n > 0 { format!("{mean:.2}") } else { String::new() },
            if self.n > 0 { format!("{:.2}", var.sqrt()) } else { String::new() },
            if self.n > 0 { format!("{:.2}", self.min) } else { String::new() },
            if self.n > 0 { format!("{:.2}", self.max) } else { String::new() },
            if self.n > 0 { format!("{:.2}", self.sum) } else { String::new() },
            self.neg,
            self.pos
        )
    }
}

/// 汇总 summary CSV 的列头
fn summary_header() -> Vec<String> {
    [
        "game",
        "snapshots",
        "calc_snaps",
        "no_emit_snaps",
        "emit_rows",
        "skip_snaps",
        "skip_event",
        "skip_playing_state5",
        "skip_rmj",
        "skip_data_incomplete",
        "skip_super_drop",
        "skip_parse_error",
        "skip_onsen",
        "skip_process_error",
        "skip_other",
        "luck_rows",
        "delta_display_n",
        "delta_display_mean",
        "delta_display_std",
        "delta_display_min",
        "delta_display_max",
        "delta_display_sum",
        "delta_display_neg",
        "delta_display_pos",
        "delta_raw_n",
        "delta_raw_mean",
        "delta_raw_std",
        "delta_raw_min",
        "delta_raw_max",
        "delta_raw_sum",
        "delta_raw_neg",
        "delta_raw_pos",
        "total_luck_end"
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// 主流程
fn run() -> Result<()> {
    let cli = parse_cli()?;

    // 1. 切到 workspace 根（load_game_config 相对路径）
    let ws_root = get_workspace_root()?;
    std::env::set_current_dir(&ws_root)?;

    // 2. 配置 / 日志 / 全局数据（必须早于 parse，与 ramen_turn_inspect 同序）
    let game_config = load_game_config()?;
    init_logger_stdout("luck_replay", "warn")?;
    init_global_with_config(&game_config)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;

    // 3. 训练器（与 main.rs 同构造；search_n 可被 CLI 覆盖）
    let cfg_search_n = SearchConfig::new_game_config(&game_config).search_n;
    let search_n = cli.search_n.unwrap_or(cfg_search_n);
    let mcts_config = SearchConfig::new_game_config(&game_config).with_search_n(search_n);
    let stages = RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?;
    let reason_slot = LastReasonSink::new();
    let trainer = RamenMctsTrainer::new(mcts_config)
        .with_stages(stages)
        .verbose(false)
        .with_reason_sink(reason_slot.clone());
    eprintln!("[luck-replay] search_n = {search_n}  stages = {}", game_config.mcts.ramen_search_stages);

    // 4. 收集并排序快照（可按 --games 过滤）
    let snaps = collect_snapshots(&cli.dir)?;
    let total_files = snaps.len();
    let snaps: Vec<_> = match &cli.games {
        Some(ids) => {
            let want: std::collections::HashSet<u64> =
                ids.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            snaps.into_iter().filter(|s| want.contains(&s.game)).collect()
        }
        None => snaps
    };
    let snaps: Vec<_> = match cli.limit {
        Some(n) => snaps.into_iter().take(n).collect(),
        None => snaps
    };
    let n_games = snaps.iter().map(|s| s.game).fold(0, |max, g| max.max(g));
    eprintln!("[luck-replay] 快照 {}/{} 份，最高局号 {n_games}", snaps.len(), total_files);

    // 5. 输出与状态（capture 同时充当 sink 与读取缓冲）
    let mut detail = CsvSink::new(&PathBuf::from(format!("{}.csv", cli.out.display())))?;
    detail.row(&detail_header())?;
    let mut summary = CsvSink::new(&PathBuf::from(format!("{}_summary.csv", cli.out.display())))?;
    let capture = Arc::new(CaptureSink::new());
    let sink: Arc<dyn DecisionSink> = capture.clone();
    let mut luck_tracker = LuckScoreTracker::new();
    let emit_info = |_: &str| {};
    let mut rng = match cli.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng()
    };
    let bonus = global!(GAMECONSTANTS).mcts_turn_bonus;

    // 6. 逐快照重放
    let mut stats: HashMap<u64, GameStats> = HashMap::new();
    let t0 = Instant::now();
    for snap in &snaps {
        let st = stats.entry(snap.game).or_default();
        st.snaps += 1;

        let contents = match fs::read_to_string(&snap.path) {
            Ok(c) => c,
            Err(e) => {
                detail.row(&build_row(&snap_ref(snap, "", 0, ""), "skip", &format!("read_error: {e}"), None, None, 0, 0))?;
                stats.entry(snap.game).or_default().skip.entry("skip_other").and_modify(|c| *c += 1).or_insert(1);
                continue;
            }
        };
        let json: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(v) => v,
            Err(e) => {
                detail.row(&build_row(&snap_ref(snap, "", 0, ""), "skip", &format!("parse_error(json): {e}"), None, None, 0, 0))?;
                stats.entry(snap.game).or_default().skip.entry("skip_parse_error").and_modify(|c| *c += 1).or_insert(1);
                continue;
            }
        };

        // baseGame 元信息（source / playing_state 供 CSV 标注与 skip reason 判定）
        let bg = json.get("baseGame").cloned().unwrap_or(serde_json::Value::Null);
        let source = bg.get("source").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let ps = bg.get("playing_state").and_then(|x| x.as_u64()).unwrap_or(0);

        let parsed = match parse_game_by_scenario(&contents) {
            Ok(p) => p,
            Err(e) => {
                detail.row(&build_row(&snap_ref(snap, &source, ps, ""), "skip", &format!("parse_error: {e}"), None, None, 0, 0))?;
                stats.entry(snap.game).or_default().skip.entry("skip_parse_error").and_modify(|c| *c += 1).or_insert(1);
                continue;
            }
        };
        let (stage_str, skip) = match &parsed {
            ParsedGame::Onsen(_) => {
                detail.row(&build_row(&snap_ref(snap, &source, ps, ""), "skip", "onsen_scenario", None, None, 0, 0))?;
                stats.entry(snap.game).or_default().skip.entry("skip_onsen").and_modify(|c| *c += 1).or_insert(1);
                continue;
            }
            ParsedGame::Ramen { game, .. } => (format!("{:?}", game.stage), classify_begin_reason(&json))
        };
        if stage_str == "Begin" {
            detail.row(&build_row(&snap_ref(snap, &source, ps, &stage_str), "skip", skip, None, None, 0, 0))?;
            stats.entry(snap.game).or_default().skip.entry(skip).and_modify(|c| *c += 1).or_insert(1);
            continue;
        }

        // 正常快照：先取走 capture 清空，跑在线链路完整处理，再读回 emit 结果
        let _ = capture.take();
        let gir = if let ParsedGame::Ramen { game, single_mode_chara_id } = parsed {
            let max_turn = game.max_turn();
            let res = process_ramen(
                game, single_mode_chara_id, &trainer, &reason_slot, &sink, &mut luck_tracker, &mut rng, true, &emit_info,
            );
            (res, stage_str.clone(), max_turn)
        } else {
            unreachable!("onsen 已在上方 continue")
        };
        let (res, stage_str, max_turn) = gir;
        let emits = capture.take();
        let st = stats.entry(snap.game).or_default();
        match res {
            Ok(()) if emits.is_empty() => {
                st.no_emit_snaps += 1;
                detail.row(&build_row(&snap_ref(snap, &source, ps, &stage_str), "no_emit", "no_decision", None, None, 0, 0))?;
            }
            Ok(()) => {
                st.calc_snaps += 1;
                st.emit_rows += emits.len() as u64;
                for (i, (info, view)) in emits.iter().enumerate() {
                    detail.row(&build_row(
                        &snap_ref(snap, &source, ps, &stage_str), "calc", "", Some((info, view)), Some((max_turn, bonus)),
                        emits.len(), i,
                    ))?;
                    // luck 更新行：scenario_extra 带 luck snapshot 的行（链式末项）
                    if let Some(extra) = &info.scenario_extra {
                        let display = extra.get("current_terminal_baseline").and_then(|x| x.as_f64());
                        let total = extra.get("total_luck_score").and_then(|x| x.as_f64());
                        let delta = extra.get("last_turn_delta").and_then(|x| x.as_f64());
                        if let (Some(disp), Some(tot)) = (display, total) {
                            let raw = raw_from_display(disp, view.turn as i32, max_turn, bonus);
                            let st = stats.entry(snap.game).or_default();
                            if let Some(prev) = st.prev_raw {
                                st.dr.push(raw - prev);
                            }
                            st.prev_raw = Some(raw);
                            if let Some(d) = delta {
                                st.dd.push(d);
                            }
                            st.luck_rows += 1;
                            st.total_luck_end = Some(tot);
                        }
                    }
                }
            }
            Err(e) => {
                detail.row(&build_row(
                    &snap_ref(snap, &source, ps, &stage_str), "skip", &format!("process_error: {e:?}"), None, None, 0, 0,
                ))?;
                stats.entry(snap.game).or_default().skip.entry("skip_process_error").and_modify(|c| *c += 1).or_insert(1);
            }
        }
    }
    eprintln!("[luck-replay] 完成 {} 份快照，耗时 {:.1}s", snaps.len(), t0.elapsed().as_secs_f64());

    // 7. 汇总 summary（每局一行）
    summary.row(&summary_header())?;
    let mut games: Vec<(u64, GameStats)> = stats.into_iter().collect();
    games.sort_by_key(|(g, _)| *g);
    for (g, st) in games {
        summary.row(&summarize_game(g, st))?;
    }
    eprintln!("[luck-replay] 输出 -> {}.csv + {}_summary.csv", cli.out.display(), cli.out.display());
    Ok(())
}

/// 把离线快照定位（文件名解析所得）转成共享 [`SnapRef`]（在线/离线共用行构造）
fn snap_ref(snap: &SnapshotRef, source: &str, playing_state: u64, stage: &str) -> SnapRef {
    SnapRef {
        game: snap.game,
        file: snap.path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        turn: snap.turn,
        seq: snap.seq,
        source: source.to_string(),
        playing_state,
        stage: stage.to_string()
    }
}

/// 生成一局的 summary 行（列序与 `summary_header` 一致）
fn summarize_game(game: u64, st: GameStats) -> Vec<String> {
    let (dn, dm, ds, dmin, dmax, dsum, dneg, dpos) = st.dd.report();
    let (rn, rm, rs, rmin, rmax, rsum, rneg, rpos) = st.dr.report();
    let skip_total: u64 = st.skip.values().sum();
    let get = |k: &str| st.skip.get(k).copied().unwrap_or(0).to_string();
    let rmj: u64 = st.skip.get("rmj_settle(46)").copied().unwrap_or(0)
        + st.skip.get("rmj_final(48)").copied().unwrap_or(0);
    vec![
        game.to_string(),
        st.snaps.to_string(),
        st.calc_snaps.to_string(),
        st.no_emit_snaps.to_string(),
        st.emit_rows.to_string(),
        skip_total.to_string(),
        get("event"),
        get("playing_state=5(event)"),
        rmj.to_string(),
        get("data_incomplete(selected_regions=0)"),
        get("super_ramen_drop(active_effect empty)"),
        get("skip_parse_error"),
        get("skip_onsen"),
        get("skip_process_error"),
        get("skip_other"),
        st.luck_rows.to_string(),
        dn.to_string(),
        dm,
        ds,
        dmin,
        dmax,
        dsum,
        dneg.to_string(),
        dpos.to_string(),
        rn.to_string(),
        rm,
        rs,
        rmin,
        rmax,
        rsum,
        rneg.to_string(),
        rpos.to_string(),
        num_col(st.total_luck_end)
    ]
}

/// 入口
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[luck-replay] 错误: {e:?}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 文件名解析：基本 / 带 seq / 非快照（thisTurn.json / zip / 其它）
    #[test]
    fn test_parse_file_name() {
        assert_eq!(parse_file_name("game7075_turn13.json"), Some((7075, 13, 0)));
        assert_eq!(parse_file_name("game7075_turn13_2.json"), Some((7075, 13, 2)));
        assert_eq!(parse_file_name("game7078_turn62_5.json"), Some((7078, 62, 5)));
        assert_eq!(parse_file_name("thisTurn.json"), None);
        assert_eq!(parse_file_name("archive.zip"), None);
        assert_eq!(parse_file_name("game7075.json"), None);
    }
}