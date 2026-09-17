//! 在线与离线共用的决策记录：明细 CSV 行构造 + 在线记录器
//!
//! ## 在线记录器（[`OnlineRecorder`]）
//!
//! umaai 实时运行时把「接收到的游戏数据」与「策略计算结果」按局落盘到
//! `logs/game{id}/` 目录（`id` = `single_mode_chara_id`，现有切局键）：
//!
//! - `game{id}_turn{turn}[_{seq}].json`：watch 收到的 `thisTurn.json` **原文**（每份一文件，写完即关）
//! - `decisions.csv`：逐决策点明细（与离线 `luck_replay` 同 schema，逐行写盘即时 flush）
//! - `meta.json`：局元信息（起止时间 / 起始回合 / 中途接入标记 / 终局运气分等），
//!   **末回合第 2 份快照（拉面 `turn77_2`，含决策行那份）处理完时写**（`end_reason=game_end`）；
//!   中途停止 / 未触发末回合的局由切局 / 退出兜底补写（`switch` / `process_exit`）
//! - `luck_trend.svg`：该局运气分趋势图（局末自动渲染；切局 / 退出兜底补渲）
//!
//! 末回合触发完 meta + SVG 后，**仅当 `end_reason=game_end` 时**自动把 `logs/game{id}/`
//! 打成 `logs/game{id}.zip` 并清理原目录（由 [`crate::decision::zip_export::zip_and_cleanup`] 实现）。
//! 切局 / 退出兜底不打包——中途停止的局保留 `logs/game{id}/` 方便人工排查 / 重打。
//!
//! 通过全局 [`RECORDER`]（`OnceLock<Mutex<Option<OnlineRecorder>>>`）访问：
//! `init` **之前所有入口为 no-op**——离线工具（`luck_replay` / `luck_probe` / bench）与
//! 单测复用同一份 lib 代码时绝不会误写 `logs/game*/`。
//!
//! ## 状态
//!
//! 只有两项：当前局上下文（`game`）与当前快照上下文（`snap`，供随后的决策行填定位字段）。
//! 决策行**即时落盘、不做每快照缓冲**——`step` / `chain_len` 在线留空（用户拍板，见
//! `issues.md`「运气分 SVG 趋势图 + 在线决策记录整合」）；本快照无任何决策时，补
//! `no_emit` 行在**下一条快照到达 / 收尾时**写出（此时才能确定"本快照没有决策"）。
//!
//! ## 共享 schema（与 `luck_replay` 离线明细严格同源）
//!
//! [`detail_header`] / [`build_row`] / [`classify_begin_reason`] / [`raw_from_display`] /
//! [`CsvSink`] 由离线重放与在线记录共用，保证两边 CSV 逐行可比对。

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use anyhow::{Result, bail};
use colored::Colorize;
use log::{info, warn};
use umasim::{
    gamedata::GAMECONSTANTS,
    output::{DecisionInfo, DecisionSink, GameView},
};

/// 候选列数（`reason_max_display` 截断后最多 5 个；与离线明细一致）
pub const MAX_CAND_COLS: usize = 5;

/// 全局在线记录器：`init` 前为 `None`（全部入口 no-op）
///
/// 用 `OnceLock<Mutex<Option<...>>>` 而非 `OnceLock<Mutex<...>>`：`OnceLock::get()`
/// 返回 `None` 表示未初始化（离线工具 / 单测），此时各入口直接返回；
/// `init` 只由 umaai 主二进制调用一次。
static RECORDER: OnceLock<Mutex<Option<OnlineRecorder>>> = OnceLock::new();

/// 初始化全局记录器（umaai 启动时按 `luck_record` 开关调用一次）
///
/// - `enabled == false`：不安装记录器，所有入口保持 no-op
/// - `logs_dir`：产物根目录（在线运行 = workspace 根的 `logs/`）
///
/// 重复调用返回 `Err`（全局只能初始化一次）。
pub fn init(enabled: bool, logs_dir: PathBuf) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    if RECORDER.set(Mutex::new(Some(OnlineRecorder::new(logs_dir)))).is_err() {
        bail!("OnlineRecorder 已初始化");
    }
    Ok(())
}

/// 收到一份游戏快照时的入口（main watch 循环 parse 后调用）
///
/// 内部完成：上一快照 `no_emit` 补行 → 切局检测与新本局游戏记录目录 → 原文落盘 → 快照上下文
/// 记录 → （Begin / 解析失败快照的 skip 行即时写出）。全程 best-effort，不返回错误。
pub fn on_snapshot(meta: &SnapMeta, raw: &str) {
    let Some(r) = RECORDER.get() else { return };
    let Ok(mut g) = r.lock() else { return };
    if let Some(rec) = g.as_mut() {
        rec.handle_snapshot(meta, raw);
    }
}

/// 一条决策 emit 时的入口（[`RecordingSink`] 调用）
///
/// 无当前快照上下文（未 init / 非拉面路径）时静默跳过；skip 快照不写决策行。
pub fn on_emit(info: &DecisionInfo, view: &GameView) {
    let Some(r) = RECORDER.get() else { return };
    let Ok(mut g) = r.lock() else { return };
    if let Some(rec) = g.as_mut() {
        rec.handle_emit(info, view);
    }
}

/// 一份快照处理完成后的入口（main 在 `process_ramen` 返回后调用）
///
/// 若刚处理的是**末回合第 2 份快照**（如拉面 `turn77_2`，含决策行的那份）——
/// 其决策行已全部落盘，这里立即写 `meta.json`（`end_reason=game_end`）并生成
/// `luck_trend.svg`，不必等切局 / 进程退出（那两者只作后续兜底）。
pub fn on_turn_done() {
    let Some(r) = RECORDER.get() else { return };
    let Ok(mut g) = r.lock() else { return };
    if let Some(rec) = g.as_mut() {
        rec.handle_turn_done();
    }
}

/// 进程退出 / 收尾入口：把当前局收尾（补 `no_emit` 行；末回合未出图的局补 meta + SVG）
pub fn finalize_shutdown() {
    let Some(r) = RECORDER.get() else { return };
    let Ok(mut g) = r.lock() else { return };
    if let Some(rec) = g.as_mut() {
        rec.finalize("process_exit");
    }
}

/// 一份快照的定位信息，供 [`build_row`] 生成明细行
#[derive(Debug, Clone, Default)]
pub struct SnapRef {
    /// 局号（`single_mode_chara_id`；`game_unknown` 局为 0）
    pub game: u64,
    /// 原始快照文件名（`game{id}_turn{turn}[_{seq}].json` 或 `_unparsed_` 形式）
    pub file: String,
    /// 回合（解析失败未取得时为 0，CSV 该列空）
    pub turn: u32,
    /// 同回合写入序号（0 = 无 `_{seq}` 后缀）
    pub seq: u32,
    /// `baseGame.source`（`command` / `event` / `special` 等）
    pub source: String,
    /// `baseGame.playing_state`
    pub playing_state: u64,
    /// 快照派发后的阶段（`Train` / `RamenSelect` / `Begin` 等）
    pub stage: String,
}

/// 快照定位 + 记录器本地状态（在线记录器使用）
#[derive(Debug, Clone)]
struct SnapCtx {
    /// 共享的 CSV 定位字段
    sref: SnapRef,
    /// 本快照是否已是 skip 行（skip 快照不再写 no_emit / 决策行）
    skip: Option<String>,
    /// 本快照是否已写过至少一条决策行（决定要不要补 no_emit）
    emitted: bool,
}

/// `on_snapshot` 入参：快照定位与 skip 判定
#[derive(Debug, Clone, Default)]
pub struct SnapMeta {
    /// 局号（`None` = 解析失败，归入当前局或 `game_unknown`）
    pub game: Option<u64>,
    /// 回合（`None` = 解析失败且原文不可读）
    pub turn: Option<u32>,
    /// 快照派发后的阶段字符串（`format!("{:?}", game.stage)`）
    pub stage: String,
    /// skip 原因（`Some` = 本快照不派发决策：Begin / 解析失败）
    pub skip: Option<String>,
    /// 剧本总回合数（拉面 = 77）——末回合判定用（`None` = 未知，不出图触发）
    pub max_turn: Option<u32>,
}

impl SnapMeta {
    /// 正常快照（parse 成功）
    pub fn normal(game: u64, turn: u32, stage: impl Into<String>) -> Self {
        Self {
            game: Some(game),
            turn: Some(turn),
            stage: stage.into(),
            skip: None,
            max_turn: None,
        }
    }

    /// 追加 skip 判定（Begin 快照由 main 用 [`classify_begin_reason`] 填 reason）
    pub fn with_skip(mut self, skip: Option<String>) -> Self {
        self.skip = skip;
        self
    }

    /// 追加剧本总回合数（`game.max_turn()`，拉面 = 77；末回合自动出图判定用）
    pub fn with_max_turn(mut self, max_turn: u32) -> Self {
        self.max_turn = Some(max_turn);
        self
    }

    /// 解析失败快照（`game`/`turn` 未知，skip reason = 错误信息）
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            game: None,
            turn: None,
            stage: String::new(),
            skip: Some(reason.into()),
            max_turn: None,
        }
    }
}

/// 在线记录器（每局一个目录，全部产物平铺）
///
/// 逻辑全部在本结构上（可脱离全局直接单测）；`RECORDER` 只是一层访问外壳。
pub struct OnlineRecorder {
    /// 产物根目录（`logs/`）
    logs_dir: PathBuf,
    /// 当前局上下文（切局时收尾换新）
    game: Option<GameCtx>,
    /// 当前快照上下文（供随后的决策行填定位字段）
    snap: Option<SnapCtx>,
    /// 上一条快照的回合（用于推 `_{seq}` 序号）
    last_turn: Option<u32>,
    /// 当前回合内已收份数
    seq: u32,
    /// 解析失败快照的全局序号（文件名 `_unparsed_{n}`）
    unparsed: u64,
}

/// 当前局上下文（一个 `logs/game{id}/` 目录）
struct GameCtx {
    /// 局号（`None` = `game_unknown`）
    game: Option<u64>,
    /// 目录路径
    dir: PathBuf,
    /// `decisions.csv` 写入器（列头已在建目录时写入）
    csv: CsvSink,
    /// `baseGame.umaId`（首个快照 best-effort 读取）
    uma_id: Option<u32>,
    /// 开局时间（RFC3339）
    started_at: String,
    /// 本局收到的第一个快照回合（> 0 = AI 中途接入）
    start_turn: Option<u32>,
    /// 本局快照数
    snaps: u64,
    /// 本局 `decisions.csv` 行数（含 skip / no_emit 行）
    csv_rows: u64,
    /// 本局决策行数（`outcome == calc`）
    decision_rows: u64,
    /// 本局最后一条 luck 行的 `total_luck_score`
    total_luck_end: Option<f64>,
    /// 是否已收到过末回合（`turn == max_turn`）的快照
    end_turn_seen: bool,
    /// 刚收到的这第 2 份末回合快照需在 `on_turn_done()`（决策行落盘后）触发收尾
    end_pending: bool,
    /// meta.json / luck_trend.svg 已在末回合生成（切局/退出收尾不再重复）
    end_done: bool,
}

impl OnlineRecorder {
    /// 新建记录器（单测直接驱动；在线运行经 [`init`] 安装到全局）
    pub fn new(logs_dir: PathBuf) -> Self {
        Self {
            logs_dir,
            game: None,
            snap: None,
            last_turn: None,
            seq: 0,
            unparsed: 0,
        }
    }

    /// 快照到达：见 [`on_snapshot`] 的模块文档
    fn handle_snapshot(&mut self, meta: &SnapMeta, raw: &str) {
        // 1) 上一条快照的 no_emit 补行（本快照到达时才确定"上一条没有决策"）
        if let Some(p) = self.snap.take()
            && p.skip.is_none()
            && !p.emitted
        {
            self.write_row(&p, "no_emit", "no_decision", None);
        }

        // 2) 定位字段：优先取 parse 结果，失败时从原文 best-effort 读 baseGame
        let bg = serde_json::from_str::<serde_json::Value>(raw).ok();
        let bg = bg.as_ref().and_then(|v| v.get("baseGame")).cloned();
        let raw_turn = meta.turn.or_else(|| bg_turn(&bg));
        let raw_source = bg_source(&bg);
        let raw_ps = bg_playing_state(&bg);
        let raw_uma = bg_uma_id(&bg);

        // 3) 切局检测 / 新本局游戏记录目录（解析失败未给 game 时沿用当前局）
        let game = meta.game.or_else(|| self.game.as_ref().and_then(|g| g.game));
        let cur_game = self.game.as_ref().and_then(|g| g.game);
        if self.game.is_none() || game != cur_game {
            self.finalize("switch");
            self.open(game, raw_turn, raw_uma);
        }

        // 4) 文件名与序号（与插件归档同口径：首份 seq=0 无后缀，同回合第 2 份 `_2`、第 3 份 `_3`…）
        let (fname, turn_cell, seq_cell) = match raw_turn {
            Some(t) => {
                if self.last_turn == Some(t) {
                    self.seq = if self.seq == 0 { 2 } else { self.seq + 1 };
                } else {
                    self.seq = 0;
                    self.last_turn = Some(t);
                }
                (snapshot_file_name(game, t, self.seq), t, self.seq)
            }
            None => {
                self.unparsed += 1;
                (unparsed_file_name(game, self.unparsed), 0, 0)
            }
        };

        // 5) 原始快照原文落盘（写完即关；失败仅告警不中断）
        if let Some(ctx) = self.game.as_mut() {
            let path = ctx.dir.join(&fname);
            if let Err(e) = fs::write(&path, raw) {
                warn!("记录原始快照失败 {}: {e:?}", path.display());
            }
            ctx.snaps += 1;
            if ctx.start_turn.is_none() {
                ctx.start_turn = raw_turn;
            }
        }

        // 6) 记录本快照上下文；skip 快照立即写行
        let mut snap = SnapCtx {
            sref: SnapRef {
                game: game.unwrap_or(0),
                file: fname,
                turn: turn_cell,
                seq: seq_cell,
                source: raw_source,
                playing_state: raw_ps,
                stage: meta.stage.clone(),
            },
            skip: meta.skip.clone(),
            emitted: false,
        };
        if let Some(reason) = &meta.skip {
            self.write_row(&snap, "skip", reason, None);
            snap.emitted = true;
        }
        self.snap = Some(snap);

        // 7) 末回合收尾触发：同一末回合（`turn >= max_turn`）出现第 2 份快照
        //    （实测 game6222：`turn77` 是 Begin skip、`turn77_2` 才是含决策的 calc）→
        //    本快照的决策行落盘后（`on_turn_done`）即生成 meta + SVG，不必等切局/退出。
        if let Some(max_turn) = meta.max_turn
            && let Some(t) = raw_turn
            && t >= max_turn
            && let Some(ctx) = self.game.as_mut()
        {
            if ctx.end_turn_seen {
                ctx.end_pending = true;
            } else {
                ctx.end_turn_seen = true;
            }
        }
    }

    /// 决策 emit：见 [`on_emit`] 的模块文档
    fn handle_emit(&mut self, info: &DecisionInfo, view: &GameView) {
        let Some(snap) = self.snap.clone().filter(|s| s.skip.is_none()) else {
            return;
        };
        if let Some(s) = self.snap.as_mut() {
            s.emitted = true;
        }
        self.write_row(&snap, "calc", "", Some((info, view)));
        if let Some(extra) = &info.scenario_extra
            && let Some(t) = extra.get("total_luck_score").and_then(|v| v.as_f64())
            && let Some(ctx) = self.game.as_mut()
        {
            ctx.total_luck_end = Some(t);
        }
    }

    /// 收尾当前局：补 no_emit 行 + （若末回合未出图）写 `meta.json` + 出 SVG + 关文件
    ///
    /// **末回合（77_2）已生成 meta/SVG 的局**（`end_done`）在此只关文件，不重复写；
    /// 其余情况（中途停止 / 未触发末回合的局）在此补写 `meta.json`（`reason` 为
    /// `switch` / `process_exit`）+ 自动出图——即切局/退出降级为「兜底」。
    fn finalize(&mut self, reason: &str) {
        let pending = self.snap.take();
        if let Some(s) = pending
            && s.skip.is_none()
            && !s.emitted
        {
            self.write_row(&s, "no_emit", "no_decision", None);
        }
        let Some(ctx) = self.game.take() else {
            self.last_turn = None;
            return;
        };
        if !ctx.end_done {
            self.game = Some(ctx);
            self.write_meta_and_plot(reason);
        }
        self.game = None;
        self.last_turn = None;
    }

    /// 末回合第 2 份快照的决策行落盘后由 main 调用：触发收尾（meta + SVG）
    fn handle_turn_done(&mut self) {
        let pending = self.game.as_ref().map(|c| c.end_pending).unwrap_or(false);
        if !pending {
            return;
        }
        self.write_meta_and_plot("game_end");
        if let Some(ctx) = self.game.as_mut() {
            ctx.end_done = true;
            ctx.end_pending = false;
        }
    }

    /// 写 `meta.json`（`end_reason` 指定）+ 生成 `luck_trend.svg` 并展示可跳转路径
    fn write_meta_and_plot(&mut self, reason: &str) {
        let Some(ctx) = self.game.as_mut() else {
            return;
        };
        let meta = serde_json::json!({
            "game": ctx.game,
            "uma_id": ctx.uma_id,
            "start_time": ctx.started_at,
            "end_time": chrono::Utc::now().to_rfc3339(),
            "start_turn": ctx.start_turn,
            "mid_entry": ctx.start_turn.map(|t| t > 0).unwrap_or(false),
            "end_reason": reason,
            "snapshots": ctx.snaps,
            "csv_rows": ctx.csv_rows,
            "decision_rows": ctx.decision_rows,
            "total_luck_end": ctx.total_luck_end,
        });
        let meta_path = ctx.dir.join("meta.json");
        match serde_json::to_string_pretty(&meta) {
            Ok(s) => {
                if let Err(e) = fs::write(&meta_path, s) {
                    warn!("写局元信息失败 {}: {e:?}", meta_path.display());
                }
            }
            Err(e) => warn!("局元信息序列化失败: {e:?}"),
        }
        let game = ctx.game.unwrap_or(0);
        let dir = ctx.dir.clone();
        // 自动生成 luck_trend.svg（best-effort：无有效行 / 渲染失败仅告警）
        match crate::plot::luck_trend::render_game(&dir, game) {
            Ok(path) => {
                // 展示绝对路径（Windows 上用 dunce 去掉 \\?\ 前缀）方便终端点击跳转；
                // 走 stderr：--json 模式下 stdout 必须保持严格 JSON 流
                let shown = dunce::canonicalize(&path).unwrap_or(path);
                eprintln!("{}", format!("运气分趋势图已生成: {}", shown.display()).bright_green());
            }
            Err(e) => warn!("该局自动出图失败: {e:?}"),
        }
        // 局末（game_end）自动打包：把 logs/game{id}/ 打成 logs/game{id}.zip 后清理原目录。
        // 仅 game_end 触发：切局 / 退出兜底不打包——中途停止的局保留 logs/game{id}/ 方便人工排查 / 重打。
        if reason == "game_end" {
            match crate::decision::zip_export::zip_and_cleanup(&dir) {
                Ok(zip_path) => {
                    let shown = dunce::canonicalize(&zip_path).unwrap_or(zip_path);
                    eprintln!("{}", format!("本局游戏记录已打包: {}", shown.display()).bright_green());
                    // json 模式下走 info 流：用户拍板「json 模式使用 info 消息类型输出」
                    info!("本局游戏记录已打包 zip: {}", shown.display());
                }
                Err(e) => warn!("本局游戏记录打包失败: {e:?}（原目录 {} 保留）", dir.display()),
            }
        }
    }

    /// 新开一局：建目录 + 建 `decisions.csv`（写列头）
    fn open(&mut self, game: Option<u64>, start_turn: Option<u32>, uma_id: Option<u32>) {
        let dir_name = match game {
            Some(id) => format!("game{id}"),
            None => "game_unknown".to_string(),
        };
        let dir = self.logs_dir.join(&dir_name);
        if let Err(e) = fs::create_dir_all(&dir) {
            warn!("创建本局游戏记录目录失败 {}: {e:?}", dir.display());
        }
        let csv = match CsvSink::new(&dir.join("decisions.csv")) {
            Ok(mut c) => {
                if let Err(e) = c.row(&detail_header()) {
                    warn!("写 decisions.csv 列头失败: {e:?}");
                }
                c
            }
            Err(e) => {
                warn!("创建 decisions.csv 失败 {}: {e:?}", dir.join("decisions.csv").display());
                CsvSink::discard()
            }
        };
        self.game = Some(GameCtx {
            game,
            dir,
            csv,
            uma_id,
            started_at: chrono::Utc::now().to_rfc3339(),
            start_turn,
            snaps: 0,
            csv_rows: 0,
            decision_rows: 0,
            total_luck_end: None,
            end_turn_seen: false,
            end_pending: false,
            end_done: false,
        });
    }

    /// 写一条明细行到当前局 CSV（best-effort）
    fn write_row(
        &mut self,
        snap: &SnapCtx,
        outcome: &str,
        reason: &str,
        emit: Option<(&DecisionInfo, &GameView)>,
    ) {
        let Some(ctx) = self.game.as_mut() else {
            return;
        };
        // t_raw 需要 (max_turn, bonus)；GAMECONSTANTS 未初始化（单测 / 异常）时留空
        let max_bonus = emit
            .map(|(_, v)| (v.max_turn as i32, GAMECONSTANTS.get().map(|g| g.mcts_turn_bonus).unwrap_or(0)));
        let row = build_row(&snap.sref, outcome, reason, emit, max_bonus, 0, 0);
        if let Err(e) = ctx.csv.row(&row) {
            warn!("写决策明细行失败: {e:?}");
        } else {
            ctx.csv_rows += 1;
            if outcome == "calc" {
                ctx.decision_rows += 1;
            }
        }
    }
}

/// 原始快照文件名：`game{id}_turn{turn}.json` / `game{id}_turn{turn}_{seq}.json`
///
/// 保留 `game{id}_` 前缀（与 SendGameStatusPlugin 归档同名、过 `luck_replay` 的
/// `parse_file_name` 前缀要求，用户拍板）。
fn snapshot_file_name(game: Option<u64>, turn: u32, seq: u32) -> String {
    let g = game_name(game);
    if seq > 0 {
        format!("{g}_turn{turn}_{seq}.json")
    } else {
        format!("{g}_turn{turn}.json")
    }
}

/// 解析失败快照文件名：`game{id}_unparsed_{n}.json`
fn unparsed_file_name(game: Option<u64>, n: u64) -> String {
    format!("{}_unparsed_{n}.json", game_name(game))
}

fn game_name(game: Option<u64>) -> String {
    match game {
        Some(id) => format!("game{id}"),
        None => "game_unknown".to_string(),
    }
}

/// 从 `baseGame` 段读回合（best-effort）
fn bg_turn(bg: &Option<serde_json::Value>) -> Option<u32> {
    bg.as_ref()
        .and_then(|v| v.get("turn"))
        .and_then(|v| v.as_u64())
        .map(|t| t as u32)
}

/// 从 `baseGame` 段读 `source`（best-effort）
fn bg_source(bg: &Option<serde_json::Value>) -> String {
    bg.as_ref()
        .and_then(|v| v.get("source"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// 从 `baseGame` 段读 `playing_state`（best-effort）
fn bg_playing_state(bg: &Option<serde_json::Value>) -> u64 {
    bg.as_ref().and_then(|v| v.get("playing_state")).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// 从 `baseGame` 段读 `umaId`（best-effort，写 meta.json）
fn bg_uma_id(bg: &Option<serde_json::Value>) -> Option<u32> {
    bg.as_ref().and_then(|v| v.get("umaId")).and_then(|v| v.as_u64()).map(|u| u as u32)
}

// ======================= 共享 CSV schema（离线/在线同源） =======================

/// CSV 写入器（基于现有 `csv` crate 的结构化 `Writer`，不手拼字符串）
///
/// - 行终止符固定 `\n`（与既有离线 CSV 逐字节一致；`csv` 默认 CRLF）
/// - 引号风格默认 `Necessary`（仅必要时加引号，与旧手拼语义一致）
/// - 每行 `write_record` 后立即 `flush`（不累积缓冲——断电最多丢当前行）
/// - `Write` 须 `Send`：记录器放在 `OnceLock<Mutex<_>>` 全局里（`Mutex<T>: Sync` 要求 `T: Send`）
pub struct CsvSink {
    w: csv::Writer<Box<dyn Write + Send>>,
}

/// 侧栏：不写入任何内容的 CSV 掉落（目录/文件创建失败时保持记录器可运行）
impl CsvSink {
    fn discard() -> Self {
        let w = csv::WriterBuilder::new()
            .terminator(csv::Terminator::Any(b'\n'))
            .from_writer(Box::new(std::io::sink()) as Box<dyn Write + Send>);
        Self { w }
    }
}

impl CsvSink {
    /// 以覆盖模式建文件（在线记录每局一个 `decisions.csv`）
    pub fn new(path: &Path) -> Result<Self> {
        let w = csv::WriterBuilder::new()
            .terminator(csv::Terminator::Any(b'\n'))
            .from_writer(Box::new(fs::File::create(path)?) as Box<dyn Write + Send>);
        Ok(Self { w })
    }

    /// 写一行（结构化 `write_record`，自动逃逸 / 加引号）
    pub fn row(&mut self, cols: &[String]) -> Result<()> {
        self.w.write_record(cols.iter().map(String::as_str))?;
        // 即时落盘：不依赖 Drop（全局记录器不会 Drop；退出 / 切局才收尾）
        self.w.flush()?;
        Ok(())
    }
}

/// 诊断明细 CSV 的列头（顺序即列序；与离线 `luck_replay` 完全一致）
pub fn detail_header() -> Vec<String> {
    let mut cols = vec![
        "game".into(),
        "file".into(),
        "turn".into(),
        "seq".into(),
        "source".into(),
        "playing_state".into(),
        "stage".into(),
        "outcome".into(),
        "reason".into(),
        "step".into(),
        "chain_len".into(),
        "decision_kind".into(),
        "n_actions".into(),
    ];
    for i in 1..=MAX_CAND_COLS {
        cols.push(format!("cand{i}_desc"));
    }
    for i in 1..=MAX_CAND_COLS {
        cols.push(format!("cand{i}_score"));
    }
    for i in 1..=MAX_CAND_COLS {
        cols.push(format!("cand{i}_n"));
    }
    cols.extend([
        "chosen_idx".into(),
        "chosen_desc".into(),
        "chosen_action_luck".into(),
        "t_n_raw".into(),
        "t_n_display".into(),
        "total_luck".into(),
        "turn_delta".into(),
    ]);
    cols
}

/// 数值 → CSV 列（`None`/NaN/Inf → 空串；f64 保留两位小数）
pub fn num_col(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.2}"),
        _ => String::new(),
    }
}

/// 把显示分 baseline 反推为 raw T(n)：`raw = display − (max_turn − turn) × bonus`
///
/// 与 `LuckScoreTracker::to_display` 互为逆运算；`bonus` 为全局 `mcts_turn_bonus`。
pub fn raw_from_display(display: f64, turn: i32, max_turn: i32, bonus: i32) -> f64 {
    display - (max_turn - turn) as f64 * bonus as f64
}

/// 生成一份诊断明细行（列序与 [`detail_header`] 一致；实现与离线 `luck_replay`
/// 的 `write_diag_row` 逐位等价）
///
/// - `emit` 为 `Some((info, view))` 且 `max_bonus = Some((max_turn, bonus))` 时是计算决策行
///   （链式末项带 luck 挂载）；`chain_len` / `step` 仅离线重放填写，在线传 0 留空（用户拍板）
#[allow(clippy::too_many_arguments)]
pub fn build_row(
    snap: &SnapRef,
    outcome: &str,
    reason: &str,
    emit: Option<(&DecisionInfo, &GameView)>,
    max_bonus: Option<(i32, i32)>,
    chain_len: usize,
    step: usize,
) -> Vec<String> {
    let mut cols: Vec<String> = vec![
        snap.game.to_string(),
        snap.file.clone(),
        snap.turn.to_string(),
        snap.seq.to_string(),
        snap.source.clone(),
        snap.playing_state.to_string(),
        snap.stage.clone(),
        outcome.to_string(),
        reason.to_string(),
    ];

    let mut cand_desc: Vec<String> = Vec::new();
    let mut cand_score: Vec<String> = Vec::new();
    let mut cand_n: Vec<String> = Vec::new();
    let mut decision_kind = String::new();
    let mut n_actions = String::new();
    let mut chosen_idx = String::new();
    let mut chosen_desc = String::new();
    let mut chosen_luck = String::new();
    let mut t_raw = String::new();
    let mut t_display = String::new();
    let mut total = String::new();
    let mut delta = String::new();

    if let Some((info, view)) = emit {
        decision_kind = info.decision_kind.clone();
        n_actions = info.candidate_descriptions.len().to_string();
        cand_desc = info.candidate_descriptions.iter().take(MAX_CAND_COLS).cloned().collect();
        cand_score = info.candidate_scores.iter().take(MAX_CAND_COLS).map(|s| format!("{s:.2}")).collect();
        cand_n = info.candidate_n.iter().take(MAX_CAND_COLS).map(|n| n.to_string()).collect();
        chosen_idx = info.action_index.to_string();
        chosen_desc = info.candidate_descriptions.get(info.action_index).cloned().unwrap_or_default();
        if let Some(extra) = &info.scenario_extra {
            if let Some(luck) = extra
                .get("action_luck")
                .and_then(|a| a.get(info.action_index.to_string()))
                .and_then(|v| v.as_f64())
            {
                chosen_luck = format!("{luck:.2}");
            }
            let display = extra.get("current_terminal_baseline").and_then(|x| x.as_f64());
            t_display = num_col(display);
            if let (Some(disp), Some((max_turn, bonus))) = (display, max_bonus) {
                t_raw = format!("{:.2}", raw_from_display(disp, view.turn as i32, max_turn, bonus));
            }
            total = num_col(extra.get("total_luck_score").and_then(|x| x.as_f64()));
            delta = num_col(extra.get("last_turn_delta").and_then(|x| x.as_f64()));
        }
    }

    cols.extend([
        if chain_len > 0 { (step + 1).to_string() } else { String::new() },
        if chain_len > 0 { chain_len.to_string() } else { String::new() },
        decision_kind,
        n_actions,
    ]);
    for i in 0..MAX_CAND_COLS {
        cols.push(cand_desc.get(i).cloned().unwrap_or_default());
    }
    for i in 0..MAX_CAND_COLS {
        cols.push(cand_score.get(i).cloned().unwrap_or_default());
    }
    for i in 0..MAX_CAND_COLS {
        cols.push(cand_n.get(i).cloned().unwrap_or_default());
    }
    cols.extend([
        chosen_idx,
        chosen_desc,
        chosen_luck,
        t_raw,
        t_display,
        total,
        delta,
    ]);
    cols
}

/// 快照层 skip 原因标注（仅当 dispatch 后 stage == `Begin` 时调用）
///
/// 判定顺序与 `GameStatusRamen::into_game` 的 stage dispatch 完全一致——
/// 只做标注，不参与任何计算决策。
pub fn classify_begin_reason(v: &serde_json::Value) -> &'static str {
    let bg = v.get("baseGame").unwrap_or(&serde_json::Value::Null);
    let ramen = v.get("ramen").unwrap_or(&serde_json::Value::Null);
    let turn = bg.get("turn").and_then(|x| x.as_u64()).unwrap_or(0);
    let source = bg.get("source").and_then(|x| x.as_str()).unwrap_or("");
    let ps = bg.get("playing_state").and_then(|x| x.as_u64()).unwrap_or(0);
    let selected = ramen
        .get("selected_regions")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().all(|r| r.as_u64().unwrap_or(0) == 0))
        .unwrap_or(false);
    let active_effect_empty = ramen
        .get("active_effect_array")
        .map(|a| a.as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap_or(true);

    if (2..=71).contains(&turn) && selected {
        "data_incomplete(selected_regions=0)"
    } else if source == "event" {
        "event"
    } else if ps == 5 {
        "playing_state=5(event)"
    } else if ps == 46 {
        "rmj_settle(46)"
    } else if ps == 48 {
        "rmj_final(48)"
    } else if turn >= 72 && active_effect_empty {
        "super_ramen_drop(active_effect empty)"
    } else {
        "begin_unclassified"
    }
}

/// 记录型 sink：转发内层 sink 的同时把每条决策 emit 记入在线记录器
///
/// 包装在输出 sink 外层（stdout JSON / 屏幕），**链式决策的中间项**（直接
/// `sink.emit`、不挂 luck）一样经此入口捕获，不漏行。无当前快照上下文时
/// [`on_emit`] 自动 no-op，非拉面路径不受影响。
pub struct RecordingSink<S: DecisionSink> {
    inner: S,
}

impl<S: DecisionSink> RecordingSink<S> {
    /// 包装一个内层 sink
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: DecisionSink> DecisionSink for RecordingSink<S> {
    fn emit(&self, info: &DecisionInfo, view: &GameView) {
        // 决策输出是主路径：先转发内层，再 best-effort 记录（失败不中断输出）
        self.inner.emit(info, view);
        on_emit(info, view);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use umasim::output::EmptySink;
    use zip::ZipArchive;

    use super::*;

    /// raw 反推与 display 正算互为逆运算
    #[test]
    fn test_raw_from_display_roundtrip() {
        let (raw, turn, max_turn, bonus) = (50150.0_f64, 5_i32, 77_i32, 2_i32);
        let display = raw + (max_turn - turn) as f64 * bonus as f64;
        assert_eq!(raw_from_display(display, turn, max_turn, bonus), raw);
    }

    /// classify_begin_reason：与协议 dispatch 的「不派发」分支逐一对应
    #[test]
    fn test_classify_begin_reason() {
        let mk = |source: &str, ps: u64, turn: u64, regions: &[u64], active_effect: usize| {
            serde_json::json!({
                "baseGame": {
                    "turn": turn,
                    "source": source,
                    "playing_state": ps,
                    "scenarioId": 14
                },
                "ramen": {
                    "selected_regions": regions,
                    "active_effect_array": vec![0u8; active_effect]
                }
            })
        };
        assert_eq!(classify_begin_reason(&mk("event", 1, 13, &[3, 7, 12], 0)), "event");
        assert_eq!(
            classify_begin_reason(&mk("event", 1, 13, &[0, 0, 0], 0)),
            "data_incomplete(selected_regions=0)"
        );
        assert_eq!(classify_begin_reason(&mk("command", 46, 13, &[3, 7, 12], 0)), "rmj_settle(46)");
        assert_eq!(
            classify_begin_reason(&mk("command", 1, 72, &[3, 7, 12], 0)),
            "super_ramen_drop(active_effect empty)"
        );
        assert_eq!(
            classify_begin_reason(&mk("command", 1, 72, &[3, 7, 12], 2)),
            "begin_unclassified"
        );
    }

    /// 列头：35 列（9 定位 + 4 附加 + 15 候选 + 7 选中/运气）
    #[test]
    fn test_detail_header_len() {
        let h = detail_header();
        println!("列头 {} 列: {h:?}", h.len());
        assert_eq!(h.len(), 35);
    }

    /// build_row：calc 行（emit 带 luck extra）与 skip 行列数一致、skip 行留空
    #[test]
    fn test_build_row_calc_and_skip() {
        let snap = SnapRef {
            game: 7075,
            file: "game7075_turn13.json".into(),
            turn: 13,
            seq: 0,
            source: "command".into(),
            playing_state: 1,
            stage: "Train".into(),
        };
        let info = DecisionInfo {
            action_index: 1,
            decision_kind: "train".into(),
            candidate_scores: vec![100.0, 200.0],
            candidate_descriptions: vec!["训练/速".into(), "训练/智".into()],
            candidate_n: vec![100, 50],
            scenario_extra: Some(serde_json::json!({
                "action_luck": {"0": -50.0, "1": 25.0},
                "current_terminal_baseline": 60410.5,
                "total_luck_score": 152.5,
                "last_turn_delta": -12.4
            })),
            ..Default::default()
        };
        let view = GameView { turn: 14, max_turn: 78, ..Default::default() };
        let row = build_row(&snap, "calc", "", Some((&info, &view)), Some((78, 2)), 0, 0);
        println!("calc 行: {row:?}");
        assert_eq!(row.len(), 35);
        assert_eq!(row[1], "game7075_turn13.json");
        assert_eq!(row[2], "13");
        assert_eq!(row[7], "calc");
        assert_eq!(row[11], "train"); // decision_kind
        assert_eq!(row[28], "1"); // chosen_idx（action_index=1）
        assert_eq!(row[29], "训练/智"); // chosen_desc
        assert_eq!(row[24], "50"); // cand4_n → 第 2 个候选局数
        assert!(!row[31].is_empty()); // t_n_raw
        assert!(!row[32].is_empty()); // t_n_display
        assert_eq!(row[33], "152.50"); // total_luck

        let skip = build_row(&snap, "skip", "event", None, None, 0, 0);
        println!("skip 行: {skip:?}");
        assert_eq!(skip.len(), 35);
        assert_eq!(skip[7], "skip");
        assert_eq!(skip[8], "event");
        assert_eq!(skip[28], ""); // chosen_idx 空
        assert_eq!(skip[30], ""); // chosen_action_luck 空
    }

    /// 端到端：临时目录内两局（切局）→ 原文文件 + decisions.csv + meta.json 齐备
    #[test]
    fn test_recorder_end_to_end() {
        let dir = std::env::temp_dir().join(format!("umaai_record_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut rec = OnlineRecorder::new(dir.clone());

        // 局 7075：turn 0 正常快照 + 一次决策；turn 1 又一份 + 两次决策（链式）
        let raw0 = r#"{"baseGame":{"scenarioId":14,"turn":0,"source":"command","playing_state":1,"umaId":100201},"ramen":{}}"#;
        rec.handle_snapshot(&SnapMeta::normal(7075, 0, "Train"), raw0);
        let info = decision_info("train");
        rec.handle_emit(&info, &GameView { turn: 1, max_turn: 78, ..Default::default() });

        let raw1 = r#"{"baseGame":{"scenarioId":14,"turn":1,"source":"command","playing_state":1,"umaId":100201},"ramen":{}}"#;
        rec.handle_snapshot(&SnapMeta::normal(7075, 1, "RamenSelect"), raw1);
        rec.handle_emit(&decision_info("ramen_select"), &GameView { turn: 2, max_turn: 78, ..Default::default() });
        rec.handle_emit(&decision_info("train"), &GameView { turn: 2, max_turn: 78, ..Default::default() });

        // 同回合第二份（覆盖写 / 事件再写）：seq 后缀 +1
        rec.handle_snapshot(&SnapMeta::normal(7075, 1, "Train"), raw1);
        rec.handle_emit(&decision_info("train"), &GameView { turn: 2, max_turn: 78, ..Default::default() });

        // 局 7076：切局收尾 7075；turn 0 Begin（skip）
        rec.handle_snapshot(
            &SnapMeta::normal(7076, 0, "Begin").with_skip(Some("event".into())),
            raw0,
        );
        rec.finalize("process_exit");

        // 7075 目录：3 份快照文件（turn0 / turn1 / turn1_2）+ decisions.csv（头 + 4 calc 行）+ meta.json（switch 收尾）
        let d7075 = dir.join("game7075");
        assert!(d7075.join("game7075_turn0.json").exists());
        assert!(d7075.join("game7075_turn1.json").exists());
        assert!(d7075.join("game7075_turn1_2.json").exists());
        let csv = fs::read_to_string(d7075.join("decisions.csv")).unwrap();
        println!("7075 decisions.csv:\n{csv}");
        assert_eq!(csv.lines().count(), 5); // 头 + 1 + 2 + 1
        assert!(csv.contains("7075,game7075_turn0.json,0,0,command,1,Train,calc"));
        assert!(csv.contains("7075,game7075_turn1.json,1,0,command,1,RamenSelect,calc"));
        assert!(csv.contains("7075,game7075_turn1_2.json,1,2,command,1,Train,calc"));

        // 7076 目录：原文 + skip 行 + meta（process_exit 收尾）
        let d7076 = dir.join("game7076");
        assert!(d7076.join("game7076_turn0.json").exists());
        let csv76 = fs::read_to_string(d7076.join("decisions.csv")).unwrap();
        println!("7076 decisions.csv:\n{csv76}");
        assert!(csv76.lines().any(|l| l.contains(",Begin,skip,event")));

        // meta.json：7075 由切局收尾（end_reason=switch），7076 由进程退出收尾
        let m75: serde_json::Value = serde_json::from_str(&fs::read_to_string(d7075.join("meta.json")).unwrap()).unwrap();
        let m76: serde_json::Value = serde_json::from_str(&fs::read_to_string(d7076.join("meta.json")).unwrap()).unwrap();
        println!("7075 meta: {m75}");
        println!("7076 meta: {m76}");
        assert_eq!(m75["decision_rows"], 4);
        assert_eq!(m75["csv_rows"], 4);
        assert_eq!(m75["snapshots"], 3);
        assert_eq!(m75["end_reason"], "switch");
        assert_eq!(m76["end_reason"], "process_exit");
        assert_eq!(m76["decision_rows"], 0);
        assert_eq!(m76["csv_rows"], 1);
        assert_eq!(m76["snapshots"], 1);

        let _ = fs::remove_dir_all(&dir);
    }

    /// 解析失败快照：原文落盘（unparsed 命名）+ skip(parse_error) 行
    #[test]
    fn test_recorder_parse_fail() {
        let dir = std::env::temp_dir().join(format!("umaai_record_fail_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut rec = OnlineRecorder::new(dir.clone());

        // 先一个正常快照建立当前局
        let raw = r#"{"baseGame":{"scenarioId":14,"turn":0,"source":"command","playing_state":1},"ramen":{}}"#;
        rec.handle_snapshot(&SnapMeta::normal(7075, 0, "Train"), raw);
        // 解析失败：归入当前局，原文落 unparsed 文件
        rec.handle_snapshot(&SnapMeta::failed("parse_error: bad struct"), "not-json");
        rec.finalize("process_exit");

        let d = dir.join("game7075");
        assert!(d.join("game7075_unparsed_1.json").exists());
        let csv = fs::read_to_string(d.join("decisions.csv")).unwrap();
        println!("parse_fail decisions.csv:\n{csv}");
        assert!(csv.lines().any(|l| l.contains(",skip,parse_error: bad struct")));
        // 第一快照无决策 → no_emit 行在收尾时补出
        assert!(csv.lines().any(|l| l.contains(",no_emit,no_decision")));

        let _ = fs::remove_dir_all(&dir);
    }

    /// RecordingSink 透传 + 记录 no-op（无全局时不影响内层输出）
    #[test]
    fn test_recording_sink_passthrough() {
        let sink = RecordingSink::new(EmptySink);
        sink.emit(&DecisionInfo::default(), &GameView::default());
        println!("RecordingSink 透传完成（未 init 全局时记录 no-op、内层不 panic）");
    }

    /// 末回合第 2 份快照（如拉面 turn77_2，含决策行那份）处理完 → `on_turn_done`
    /// 立即写 meta（`end_reason=game_end`）+ 出图；随后切局不再重复写 meta
    #[test]
    fn test_recorder_end_turn_triggers_plot() {
        // 路径白名单要求 logs_dir 末段为 logs：测试根在 <temp>/.../logs，
        // 模拟生产里 OnlineRecorder 用 workspace/logs 的形态
        let dir = std::env::temp_dir()
            .join(format!("umaai_record_end_{}", std::process::id()))
            .join("logs");
        let _ = fs::remove_dir_all(dir.parent().unwrap());
        let mut rec = OnlineRecorder::new(dir.clone());
        let raw = |t: &str| {
            format!(
                r#"{{"baseGame":{{"scenarioId":14,"turn":{t},"source":"command","playing_state":1,"umaId":100201}},"ramen":{{}}}}"#
            )
        };
        let view = |t: u32| GameView { turn: t, max_turn: 78, ..Default::default() };

        // turn76：正常快照 + 决策（非末回合，不触发）
        rec.handle_snapshot(&SnapMeta::normal(7075, 76, "Train").with_max_turn(77), &raw("76"));
        rec.handle_emit(&decision_info("train"), &view(77));
        assert!(!dir.join("game7075").join("luck_trend.svg").exists());

        // turn77 第 1 份（Begin skip，如超级拉面丢包）：仅标记"已见末回合"，不出图
        rec.handle_snapshot(
            &SnapMeta::normal(7075, 77, "Begin")
                .with_skip(Some("super_ramen_drop(active_effect empty)".into()))
                .with_max_turn(77),
            &raw("77"),
        );
        rec.handle_turn_done();
        assert!(!dir.join("game7075").join("luck_trend.svg").exists());

        // turn77_2（calc，决策行所在）：end_pending → 决策行落盘后 on_turn_done 出图
        rec.handle_snapshot(&SnapMeta::normal(7075, 77, "Train").with_max_turn(77), &raw("77"));
        rec.handle_emit(&decision_info("train"), &view(78));
        let d = dir.join("game7075");
        assert!(d.join("game7075_turn77_2.json").exists(), "末回合第 2 份应为 _2 命名");
        rec.handle_turn_done();
        // 77_2 处理完 → 出图 + 立即打包：原目录已被清理，svg 收纳进 zip
        assert!(!d.exists(), "game_end 后原目录应已被打包清理");

        // game_end → 自动打包成 game7075.zip
        let zip_path = dir.join("game7075.zip");
        assert!(zip_path.exists(), "game_end 应自动生成 game7075.zip");

        // 验证 zip 包内含 meta.json / luck_trend.svg / turn77_2.json（最小完整性检查）
        let f = std::fs::File::open(&zip_path).unwrap();
        let mut zip = ZipArchive::new(f).unwrap();
        let names: Vec<String> = (0..zip.len())
            .map(|i| zip.by_index(i).unwrap().name().to_string())
            .collect();
        println!("zip 包内条目: {names:?}");
        for required in &["meta.json", "luck_trend.svg", "decisions.csv"] {
            assert!(
                names.iter().any(|n| n == required),
                "zip 包内应含 {required}：{names:?}"
            );
        }

        // 切局到 7076：end_done → finalize 不再重写 meta（仍为 game_end）
        rec.handle_snapshot(&SnapMeta::normal(7076, 0, "Train"), &raw("0"));
        // 原目录已被打包删除；meta.json 内容只能从 zip 里读
        let f2 = std::fs::File::open(&zip_path).unwrap();
        let mut zip2 = ZipArchive::new(f2).unwrap();
        let mut meta_entry = zip2.by_name("meta.json").unwrap();
        let mut s = String::new();
        Read::read_to_string(&mut meta_entry, &mut s).unwrap();
        let m2: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(m2["end_reason"], "game_end", "切局不应覆盖末回合已写的 end_reason");

        // dir 父目录才是测试根（logs 的上一级）
        let _ = fs::remove_dir_all(dir.parent().unwrap());
    }

    /// 中途停止（未触发末回合）→ 切局 / 退出兜底写 meta（switch）+ 出图
    #[test]
    fn test_recorder_end_turn_fallback_on_switch() {
        let dir = std::env::temp_dir().join(format!("umaai_record_endfb_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut rec = OnlineRecorder::new(dir.clone());
        let raw = r#"{"baseGame":{"scenarioId":14,"turn":12,"source":"command","playing_state":1},"ramen":{}}"#;
        rec.handle_snapshot(&SnapMeta::normal(7075, 12, "Train").with_max_turn(77), raw);
        rec.handle_emit(&decision_info("train"), &GameView { turn: 13, max_turn: 78, ..Default::default() });

        // 没到 77 就切局（用户中途停止）
        rec.handle_snapshot(&SnapMeta::normal(7076, 0, "Train"), raw);
        let d = dir.join("game7075");
        let m: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(d.join("meta.json")).unwrap()).unwrap();
        println!("兜底 meta: {m}");
        assert_eq!(m["end_reason"], "switch");
        assert!(d.join("luck_trend.svg").exists(), "兜底也应出图");
        // switch 兜底不打包：原目录与原文件保留，zip 不应存在
        assert!(!dir.join("game7075.zip").exists(), "switch 兜底不应生成 zip");
        assert!(d.exists(), "switch 兜底原目录应保留");

        let _ = fs::remove_dir_all(&dir);
    }

    /// 构造一个最小带 luck extra 的决策（供端到端测试）
    fn decision_info(kind: &str) -> DecisionInfo {
        DecisionInfo {
            action_index: 0,
            decision_kind: kind.to_string(),
            candidate_scores: vec![100.0],
            candidate_descriptions: vec!["动作".to_string()],
            candidate_n: vec![10],
            scenario_extra: Some(serde_json::json!({
                "action_luck": {"0": 0.0},
                "current_terminal_baseline": 60000.0,
                "total_luck_score": 42.0,
                "last_turn_delta": 7.0
            })),
            ..Default::default()
        }
    }
}