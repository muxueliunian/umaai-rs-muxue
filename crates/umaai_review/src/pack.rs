//! 局包（`logs/game{id}.zip`）解包与文件角色识别
//!
//! 输入契约见 `.trae/documents/replay_review.md` §5.1：
//!
//! - 包内布局两种：带 `game{id}/` 外壳（旧样例）与不带外壳（当前 `zip_export`
//!   产出）→ 一律按 **basename** 匹配，忽略目录层级
//! - `game{game}_turn{turn}[_{seq}].json`：快照（`seq` 0 = 无 `_{seq}` 后缀，
//!   与在线记录器 `snapshot_file_name` 同口径）
//! - `game{game}_unparsed_{n}.json`：解析失败快照
//! - `decisions.csv` / `meta.json` / `luck_trend.svg`：固定角色
//! - `game_unknown` 前缀（解析失败局）归局号 0
//!
//! 角色识别失败的条目进 [`Pack::ignored`]（含原因），不阻断解析；
//! 局号不一致的快照不丢弃，仅在 [`Pack::self_check`] 里告警（防多局混包）。

use std::{collections::HashMap, fs, io::Read, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use zip::ZipArchive;

/// 单份快照（basename 定位 + 原文字节；后续按需反序列化为 `GameStatusRamen`）
#[derive(Debug, Clone)]
pub struct SnapEntry {
    /// 快照 basename（如 `game6234_turn10_3.json`）
    pub file: String,
    /// 局号（`game_unknown` 归 0；仅作一致性告警，不参与筛选）
    pub game: u64,
    /// 回合号
    pub turn: u32,
    /// 同回合写入序号（0 = 无 `_{seq}` 后缀）
    pub seq: u32,
    /// 原文字节
    pub bytes: Vec<u8>,
}

/// `meta.json` 局元信息（字段缺失容忍，`serde(default)`）
///
/// 与在线记录器 `finalize` 写出的字段一一对应（snake_case）。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PackMeta {
    /// 局号（`single_mode_chara_id`）
    pub game: u64,
    /// 马娘 ID
    pub uma_id: u32,
    /// 起始回合（中途接管时 > 0）
    pub start_turn: u32,
    /// 是否中途接管
    pub mid_entry: bool,
    /// 结束原因（`game_end` / `switch` / `process_exit` 等）
    pub end_reason: String,
    /// 快照份数（含 unparsed）
    pub snapshots: u64,
    /// 决策行数
    pub decision_rows: u64,
    /// CSV 总行数（含 skip 行）
    pub csv_rows: u64,
    /// 终局运气分
    pub total_luck_end: Option<f64>,
}

/// 解包产物（角色识别完成）
#[derive(Debug, Default)]
pub struct Pack {
    /// 局号（快照文件名多数派，`game_unknown` 不参与投票；全 unknown 时为 0）
    pub game: u64,
    /// 快照（按 (turn, seq) 升序）
    pub snaps: Vec<SnapEntry>,
    /// 解析失败快照 basename
    pub unparsed: Vec<String>,
    /// `decisions.csv` 原文（缺失时 `None`，后续降级推导）
    pub decisions_csv: Option<String>,
    /// `meta.json` 解析结果（解析失败不阻断，落 `None`）
    pub meta: Option<PackMeta>,
    /// `luck_trend.svg` 原文（交叉核对运气分口径用）
    pub luck_trend_svg: Option<String>,
    /// 未识别条目（basename + 原因）
    pub ignored: Vec<String>,
}

/// 文件名角色（按 basename 判定）
#[derive(Debug, PartialEq)]
enum Role {
    /// 快照
    Snap { game: u64, turn: u32, seq: u32 },
    /// 解析失败快照
    Unparsed { game: u64, n: u64 },
    /// `decisions.csv`
    Decisions,
    /// `meta.json`
    Meta,
    /// `luck_trend.svg`
    LuckTrend,
    /// 其他（外壳目录条目 / 无法识别文件名）
    Other,
}

/// 按 basename 识别角色（忽略目录层级；`game_unknown` 前缀归局号 0）
fn classify(basename: &str) -> Role {
    match basename {
        "decisions.csv" => return Role::Decisions,
        "meta.json" => return Role::Meta,
        "luck_trend.svg" => return Role::LuckTrend,
        _ => {}
    }
    let Some(stem) = basename.strip_suffix(".json") else {
        return Role::Other;
    };
    // 剥 `game` 前缀：`game{id}_…` 直接剩 `id_…`；`game_unknown_…` 剩 `_unknown_…`
    // 需再剥一个前导 `_`（两种形态都归一到「无前导下划线」）
    let rest = stem.strip_prefix("game").unwrap_or(stem);
    let rest = rest.strip_prefix('_').unwrap_or(rest);
    // 形态：{id}_turn{turn}[_{seq}] / {id}_unparsed_{n}（id 为数字或 unknown）
    let game = |s: &str| match s {
        "unknown" => Some(0u64),
        digits => digits.parse::<u64>().ok(),
    };
    let parts: Vec<&str> = rest.split('_').collect();
    match parts.as_slice() {
        [id, turn] => {
            if let (Some(g), Some(t)) = (game(id), turn.strip_prefix("turn").and_then(|t| t.parse::<u32>().ok())) {
                Role::Snap { game: g, turn: t, seq: 0 }
            } else {
                Role::Other
            }
        }
        [id, kind, tail] => {
            if kind == &"unparsed" {
                match (game(id), tail.parse::<u64>().ok()) {
                    (Some(g), Some(n)) => Role::Unparsed { game: g, n },
                    _ => Role::Other,
                }
            } else if let Some(t) = kind.strip_prefix("turn") {
                match (game(id), t.parse::<u32>().ok(), tail.parse::<u32>().ok()) {
                    (Some(g), Some(turn), Some(seq)) => Role::Snap { game: g, turn, seq },
                    _ => Role::Other,
                }
            } else {
                Role::Other
            }
        }
        _ => Role::Other,
    }
}

/// 打开局包并完成角色识别
///
/// - 条目按 basename 匹配，忽略目录层级（两种布局通吃）
/// - 快照按 (turn, seq) 升序排序
/// - `meta.json` 解析失败不阻断（落 `meta` = `None`，后续从文件名与 CSV 降级推导）
pub fn open_zip(path: &Path) -> Result<Pack> {
    let file = fs::File::open(path).with_context(|| format!("打开局包失败: {}", path.display()))?;
    let mut zip =
        ZipArchive::new(file).with_context(|| format!("读取局包失败: {}", path.display()))?;

    let mut snaps: Vec<SnapEntry> = Vec::new();
    let mut unparsed: Vec<String> = Vec::new();
    let mut decisions_csv: Option<String> = None;
    let mut meta_json: Option<String> = None;
    let mut luck_trend_svg: Option<String> = None;
    let mut ignored: Vec<String> = Vec::new();
    let mut game_votes: HashMap<u64, usize> = HashMap::new();

    for i in 0..zip.len() {
        let mut entry =
            zip.by_index(i).with_context(|| format!("读取局包条目 #{i} 失败"))?;
        if entry.is_dir() {
            continue; // 外壳目录（旧布局），忽略层级
        }
        let name = entry.name().replace('\\', "/");
        let basename = name.rsplit('/').next().unwrap_or_default().to_string();
        let mut bytes = Vec::new();
        Read::read_to_end(&mut entry, &mut bytes)
            .with_context(|| format!("读取条目内容失败: {basename}"))?;
        match classify(&basename) {
            Role::Snap { game, turn, seq } => {
                *game_votes.entry(game).or_default() += 1;
                snaps.push(SnapEntry { file: basename, game, turn, seq, bytes });
            }
            Role::Unparsed { game, .. } => {
                *game_votes.entry(game).or_default() += 1;
                unparsed.push(basename);
            }
            Role::Decisions => {
                decisions_csv = Some(String::from_utf8_lossy(&bytes).into_owned())
            }
            Role::Meta => {
                meta_json = Some(String::from_utf8_lossy(&bytes).into_owned())
            }
            Role::LuckTrend => {
                luck_trend_svg = Some(String::from_utf8_lossy(&bytes).into_owned())
            }
            Role::Other => {
                ignored.push(format!("{basename}（无法识别的文件名）"));
            }
        }
    }

    // 局号 = 多数派（game_unknown=0 不投票；全 unknown 时保持 0）
    let game = game_votes
        .iter()
        .filter(|(g, _)| **g != 0)
        .max_by_key(|(g, c)| (**c, *g))
        .map(|(g, _)| *g)
        .unwrap_or(0);

    // meta.json 解析失败不阻断（缺失字段走 default）
    let meta = meta_json.and_then(|s| serde_json::from_str::<PackMeta>(&s).ok());

    snaps.sort_by_key(|s| (s.turn, s.seq));

    Ok(Pack { game, snaps, unparsed, decisions_csv, meta, luck_trend_svg, ignored })
}

impl Pack {
    /// 快照回合覆盖范围（无快照时 `None`）
    pub fn turn_range(&self) -> Option<(u32, u32)> {
        let turns = self.snaps.iter().map(|s| s.turn);
        turn_range_of(turns)
    }

    /// 数据自检摘要（一行一条；口径见 replay_review.md §11 步骤 1）
    ///
    /// 内容：局号与份数 / 回合覆盖与缺失 / 同回合多快照与重复 (turn, seq) /
    /// meta.json 交叉核对（份数与行数）/ 固定角色产物存在性 / 未识别条目。
    pub fn self_check(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "局号: {}（快照 {} 份，unparsed {} 份，未识别 {} 条）",
            self.game,
            self.snaps.len(),
            self.unparsed.len(),
            self.ignored.len()
        ));

        // 局号一致性（混包告警，不丢数据）
        let mismatch = self.snaps.iter().filter(|s| s.game != 0 && s.game != self.game).count();
        if mismatch > 0 {
            lines.push(format!("⚠ 局号不一致快照: {mismatch} 份（疑似混包，仅告警不剔除）"));
        }

        // 回合覆盖
        if let Some((min, max)) = self.turn_range() {
            let present: HashMap<u32, usize> = {
                let mut m: HashMap<u32, usize> = HashMap::new();
                for s in &self.snaps {
                    *m.entry(s.turn).or_default() += 1;
                }
                m
            };
            let missing: Vec<String> = (min..=max)
                .filter(|t| !present.contains_key(t))
                .map(|t| t.to_string())
                .collect();
            lines.push(format!(
                "回合覆盖: {min}..={max}（缺回合: {}）",
                if missing.is_empty() { "无".to_string() } else { missing.join(",") }
            ));
            let multi: Vec<(u32, usize)> =
                present.iter().filter(|(_, c)| **c > 1).map(|(t, c)| (*t, *c)).collect();
            let max_seq = self.snaps.iter().map(|s| s.seq).max().unwrap_or(0);
            lines.push(format!(
                "同回合多份快照: {} 个回合（最大 seq = {max_seq}）",
                multi.len()
            ));
            let mut seen: HashMap<(u32, u32), usize> = HashMap::new();
            for s in &self.snaps {
                *seen.entry((s.turn, s.seq)).or_default() += 1;
            }
            let dup = seen.values().filter(|&&c| c > 1).count();
            if dup > 0 {
                lines.push(format!("⚠ 重复 (turn,seq) 定位: {dup} 组"));
            }
        } else {
            lines.push("回合覆盖: 无快照".to_string());
        }

        // meta.json 交叉核对
        match &self.meta {
            Some(m) => {
                let actual_snaps = (self.snaps.len() + self.unparsed.len()) as u64;
                let snap_mark = if m.snapshots == actual_snaps { "✓" } else { "⚠ 不一致" };
                let csv_part = self.decisions_csv.as_ref().map(|c| {
                    let data_rows =
                        c.lines().filter(|l| !l.trim().is_empty()).count() as u64;
                    let data_rows = data_rows.saturating_sub(1); // 去表头
                    let mark = if m.csv_rows == data_rows { "✓" } else { "⚠ 不一致" };
                    format!("（文件数据行 {data_rows} {mark}）")
                });
                let csv_part = csv_part.unwrap_or_default();
                let luck = m
                    .total_luck_end
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "缺失".to_string());
                lines.push(format!(
                    "meta.json: end_reason={}, start_turn={}, mid_entry={}, uma_id={}, \
                     snapshots={}（实际 {actual_snaps} {snap_mark}）, decision_rows={}, \
                     csv_rows={}{csv_part}, total_luck_end={luck}",
                    m.end_reason,
                    m.start_turn,
                    m.mid_entry,
                    m.uma_id,
                    m.snapshots,
                    m.decision_rows,
                    m.csv_rows
                ));
            }
            None => lines.push("meta.json: 缺失或解析失败（后续从文件名与 CSV 降级推导）".to_string()),
        }

        // 固定角色产物
        lines.push(format!(
            "decisions.csv: {}, luck_trend.svg: {}",
            if self.decisions_csv.is_some() { "存在" } else { "缺失" },
            if self.luck_trend_svg.is_some() { "存在" } else { "缺失" }
        ));
        if !self.ignored.is_empty() {
            let head: Vec<&str> = self.ignored.iter().take(10).map(String::as_str).collect();
            let more = self.ignored.len().saturating_sub(10);
            lines.push(format!(
                "未识别条目（{} 条）: {}{}",
                self.ignored.len(),
                head.join(", "),
                if more > 0 { format!(" …另有 {more} 条") } else { String::new() }
            ));
        }
        lines
    }
}

/// 数值迭代器的 (min, max)
fn turn_range_of<I: Iterator<Item = u32>>(turns: I) -> Option<(u32, u32)> {
    let mut min: Option<u32> = None;
    let mut max: Option<u32> = None;
    for t in turns {
        min = Some(min.map_or(t, |m: u32| m.min(t)));
        max = Some(max.map_or(t, |m: u32| m.max(t)));
    }
    match (min, max) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, path::PathBuf};

    /// 在临时目录构造一个测试局包（`with_shell` = 旧布局带 `game{id}/` 外壳）
    ///
    /// 包内容：2 快照（turn0 无后缀 / turn1_2）+ 1 unparsed + decisions.csv +
    /// meta.json + 1 个无法识别文件（noise.txt）。
    fn make_zip(path: &Path, with_shell: bool) {
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            let pfx = if with_shell { "game7/" } else { "" };
            for (name, data) in [
                ("game7_turn0.json", b"{}" as &[u8]),
                ("game7_turn1_2.json", b"{}"),
                ("game7_unparsed_1.json", b"{}"),
                ("decisions.csv", b"game,file\n7,x\n"),
                ("meta.json", br#"{"game":7,"end_reason":"game_end","snapshots":3}"#),
                ("noise.txt", b"x"),
            ] {
                w.start_file(format!("{pfx}{name}"), opts).unwrap();
                w.write_all(data).unwrap();
            }
            w.finish().unwrap();
        }
        fs::write(path, &buf).unwrap();
    }

    /// 测试根目录（按调用方命名，并行测试互不共享、互不互删）
    fn test_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("umaai_review_test_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 两种布局（带/不带外壳）的角色识别应一致
    #[test]
    fn test_open_zip_roles() {
        let root = test_root("roles");
        for with_shell in [false, true] {
            let zip_path = root.join(format!("t{shell}.zip", shell = with_shell as i32));
            make_zip(&zip_path, with_shell);
            let pack = open_zip(&zip_path).unwrap();
            println!("布局 shell={with_shell} 自检:\n{}", pack.self_check().join("\n"));
            assert_eq!(pack.game, 7, "局号应取多数派");
            assert_eq!(pack.snaps.len(), 2, "应识别 2 份快照");
            assert_eq!((pack.snaps[0].turn, pack.snaps[0].seq), (0, 0), "无后缀 seq=0");
            assert_eq!((pack.snaps[1].turn, pack.snaps[1].seq), (1, 2), "后缀 _2 → seq=2");
            assert_eq!(pack.unparsed.len(), 1, "应识别 1 份 unparsed");
            assert!(pack.decisions_csv.is_some(), "decisions.csv 应存在");
            assert!(pack.meta.is_some(), "meta.json 应存在");
            assert_eq!(pack.meta.as_ref().unwrap().end_reason, "game_end");
            assert_eq!(pack.ignored.len(), 1, "noise.txt 应未识别");
        }
        let _ = fs::remove_dir_all(&root);
    }

    /// `game_unknown` 前缀归局号 0，全 unknown 时 Pack.game = 0
    #[test]
    fn test_open_zip_unknown_game() {
        let root = test_root("unknown");
        let zip_path = root.join("unknown.zip");
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default();
            w.start_file("game_unknown_unparsed_1.json", opts).unwrap();
            w.write_all(b"{}").unwrap();
            w.finish().unwrap();
        }
        fs::write(&zip_path, &buf).unwrap();
        let pack = open_zip(&zip_path).unwrap();
        println!("unknown 局自检:\n{}", pack.self_check().join("\n"));
        assert_eq!(pack.game, 0, "全 unknown 时局号应为 0");
        assert_eq!(pack.unparsed.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    /// 文件名角色分类单元测试
    #[test]
    fn test_classify() {
        println!(
            "classify 结果: {:?} / {:?} / {:?} / {:?} / {:?} / {:?}",
            classify("game6234_turn0.json"),
            classify("game6234_turn10_3.json"),
            classify("game6234_unparsed_5.json"),
            classify("game_unknown_turn2.json"),
            classify("decisions.csv"),
            classify("nested/dir/meta.json")
        );
        assert_eq!(
            classify("game6234_turn0.json"),
            Role::Snap { game: 6234, turn: 0, seq: 0 }
        );
        assert_eq!(
            classify("game6234_turn10_3.json"),
            Role::Snap { game: 6234, turn: 10, seq: 3 }
        );
        assert_eq!(
            classify("game6234_unparsed_5.json"),
            Role::Unparsed { game: 6234, n: 5 }
        );
        assert_eq!(
            classify("game_unknown_turn2.json"),
            Role::Snap { game: 0, turn: 2, seq: 0 }
        );
        assert_eq!(classify("decisions.csv"), Role::Decisions);
        // classify 契约 = 按 basename 判定；目录层级在 open_zip 已剥掉
        assert_eq!(classify("nested/dir/meta.json"), Role::Other);
        assert_eq!(classify("nested/dir/luck_trend.svg"), Role::Other);
        assert_eq!(classify("game6234_turnX.json"), Role::Other);
        assert_eq!(classify("game6234_turn5_bad.json"), Role::Other);
        assert_eq!(classify("readme.md"), Role::Other);
    }
}
