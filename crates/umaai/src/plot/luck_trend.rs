//! 单局运气分趋势图（SVG）：期望评分 / 运气分 / 运气波动 3 子图
//!
//! 数据源：局目录内的 `decisions.csv`（在线记录与离线 `luck_replay` 同 schema）。
//! 口径与 `scripts/plot_luck_trend.py` 逐项对齐（用户多轮微调的样式基准）：
//!
//! - 按快照（`file`）聚合，**skip 快照剔除**；同快照链式多决策取末决策（主决策）做分类
//! - x 轴 = 非 skip 快照序号，刻度放在回合变化处、标签 = 回合数（过密抽样）
//! - 竖直底色带按 AI 决策类别着色（吃面/训练/出行/休息/比赛/地区选择）
//! - 子图[评分]：T(n) raw（实线）+ 显示口径（虚线）；[运气分]：显示累计 + raw 累计
//!   （以首行 raw 为 0 基准）；[运气波动]：raw 相邻差柱状，正绿负红
//! - 输出自包含单文件 `luck_trend.svg`（不内嵌字形，中文走渲染器回退）

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};

use super::svg::Svg;

/// AI 决策类别 → 色带颜色（浅色底、可透曲线，与 python 版一致）
const CAT_COLOR: [(&str, &str); 6] = [
    ("训练", "#ffffff"),
    ("出行", "#e08a3c"),
    ("休息", "#33a24d"),
    ("比赛", "#7b6cb5"),
    ("吃面", "#66c1f8"),
    ("地区选择", "#c8c8c8"),
];

/// 折线 / 图例色（与 python 版一致）
const RAW_CLR: &str = "#0e4fa0";
const DISP_CLR: &str = "#9cc3e5";
const LUCK_CLR: &str = "#c44e52";
// 原始运气分（raw 累计）颜色：按用户要求该系列不显示，颜色直接写在注释代码里
// const LUCK_RAW_CLR: &str = "#e8a3a6";
const BAR_POS: &str = "#2ca02c";
const BAR_NEG: &str = "#d62728";

/// 按选中动作描述分类 AI 决策（吃面优先于训练，避免「吃面/…」误分到训练）
fn classify_action(desc: &str) -> &'static str {
    if desc.is_empty() {
        return "地区选择";
    }
    if desc.contains("吃面") || desc.contains("不吃面") {
        return "吃面";
    }
    if desc.contains("训练") {
        return "训练";
    }
    if desc.contains("出行") {
        return "出行";
    }
    if desc.contains("休息") {
        return "休息";
    }
    if desc.contains("比赛") {
        return "比赛";
    }
    "地区选择"
}

/// 明细 CSV 行（只取绘图所需列）
struct Row {
    file: String,
    turn: u32,
    outcome: String,
    chosen_desc: String,
    raw: Option<f64>,
    disp: Option<f64>,
    total: Option<f64>,
}

/// 一个非 skip 快照（链式多决策取末决策的汇总）
struct Snap {
    turn: u32,
    cat: &'static str,
    raw: Option<f64>,
    disp: Option<f64>,
    total: Option<f64>,
}

/// 数值列：空串 / 非数值 → `None`（与 python `num()` 同语义）
fn num(v: &str) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.parse::<f64>().ok()
}

/// 读一局 `decisions.csv` → 按快照聚合（剔除 skip，链式取末决策）
fn load_game(dir: &Path) -> Result<Vec<Snap>> {
    let csv_path = dir.join("decisions.csv");
    if !csv_path.exists() {
        return Err(anyhow!("缺少 decisions.csv: {}", csv_path.display()));
    }
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_path(&csv_path)?;
    let headers = rdr.headers()?.clone();
    let idx = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow!("decisions.csv 缺列: {name}"))
    };
    let i_file = idx("file")?;
    let i_turn = idx("turn")?;
    let i_outcome = idx("outcome")?;
    let i_desc = idx("chosen_desc")?;
    let i_raw = idx("t_n_raw")?;
    let i_disp = idx("t_n_display")?;
    let i_total = idx("total_luck")?;

    let mut rows: Vec<Row> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() <= i_total {
            continue;
        }
        rows.push(Row {
            file: rec[i_file].to_string(),
            turn: rec[i_turn].parse::<u32>().unwrap_or(0),
            outcome: rec[i_outcome].to_string(),
            chosen_desc: rec[i_desc].to_string(),
            raw: num(&rec[i_raw]),
            disp: num(&rec[i_disp]),
            total: num(&rec[i_total]),
        });
    }

    // 按 file 保持出现顺序分组
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<Row>> = HashMap::new();
    for r in rows {
        if !groups.contains_key(&r.file) {
            order.push(r.file.clone());
        }
        groups.entry(r.file.clone()).or_default().push(r);
    }

    let mut snaps: Vec<Snap> = Vec::new();
    for file in &order {
        let rs = &groups[file];
        if rs.iter().all(|r| r.outcome == "skip") {
            continue; // skip 快照不列入图表
        }
        let main = rs.last().expect("非空分组");
        let luck = rs.iter().rev().find(|r| r.raw.is_some());
        snaps.push(Snap {
            turn: main.turn,
            cat: classify_action(&main.chosen_desc),
            raw: luck.and_then(|r| r.raw),
            disp: luck.and_then(|r| r.disp),
            total: luck.and_then(|r| r.total),
        });
    }
    Ok(snaps)
}

/// 生成一局 `luck_trend.svg`（落在局目录内），返回 SVG 路径
///
/// 只负责出图并返回路径（不打印）——路径展示由调用方决定（在线记录器在局收尾时
/// 以绿色绝对路径打到终端，方便点击跳转；`--json` 模式走 stderr 不污染 JSON 流）。
pub fn render_game(dir: &Path, game: u64) -> Result<PathBuf> {
    let snaps = load_game(dir)?;
    if snaps.is_empty() {
        return Err(anyhow!("{} 无可绘制的非 skip 快照", dir.display()));
    }
    let svg = draw(&snaps, game);
    let out = dir.join("luck_trend.svg");
    fs::write(&out, svg)?;
    Ok(out)
}

/// 布局常量（16:11 比例，右侧图例列）
const W: f64 = 1440.0;
const H: f64 = 1010.0;
/// 绘图区左边距（留出纵坐标刻度数字 + 纵轴标题）
const X0: f64 = 132.0;
const X1: f64 = 1130.0;
const LEG_X: f64 = 1160.0;
const TOP0: f64 = 90.0;
const PH: f64 = 225.0;
const GAP: f64 = 35.0;

/// 绘制（纯函数，便于单测）
fn draw(snaps: &[Snap], game: u64) -> String {
    let n = snaps.len().max(1);
    let plot_w = X1 - X0;
    // x 轴按「快照槽位」分格，并**多留 1 格**（用户要求覆盖到末回合 +1，即拉面 78 回合），
    // 使最后一条数据不贴右边界
    let slots = n + 1;
    let dx = plot_w / slots as f64;
    let px = |i: usize| X0 + i as f64 * dx;

    let mut svg = Svg::new(W, H);
    svg.text(W / 2.0, 45.0, &format!("game{game} 拉面运气分趋势"), 20.0, "middle");

    // 三个子图顶部 y
    let tops = [TOP0, TOP0 + PH + GAP, TOP0 + 2.0 * (PH + GAP)];

    // ---- 数据准备（仅含 luck 快照）----
    let lx: Vec<usize> = snaps.iter().enumerate().filter(|(_, s)| s.raw.is_some()).map(|(i, _)| i).collect();
    let raw: Vec<f64> = lx.iter().map(|&i| snaps[i].raw.unwrap()).collect();
    let disp: Vec<f64> = lx.iter().map(|&i| snaps[i].disp.unwrap_or(0.0)).collect();
    let total: Vec<f64> = lx.iter().map(|&i| snaps[i].total.unwrap_or(0.0)).collect();
    // 原始运气分（raw 累计，以首行 raw 为 0）：用户要求不显示 → 计算与绘制一并注释；
    // 恢复时取消本处注释 + 第二子图「原始运气分」绘制块（颜色 #e8a3a6）
    // let raw_first = *raw.first().unwrap_or(&0.0);
    // let raw_cum: Vec<f64> = raw.iter().map(|v| v - raw_first).collect();
    let raw_step: Vec<Option<f64>> =
        std::iter::once(None).chain(raw.windows(2).map(|w| Some(w[1] - w[0]))).collect();
    let steps: Vec<f64> = raw_step.iter().flatten().copied().collect();

    // ---- 三个子图的 y 范围 / y 映射 / 纵坐标刻度（先算，供网格与刻度标注共用）----
    let (lo1, hi1) = y_range(&[&raw, &disp], false);
    let (lo2, hi2) = y_range(&[&total], true);
    let (lo3, hi3) = y_range(&[&steps], true);
    let y1 = y_map(TOP0, lo1, hi1);
    let y2 = y_map(TOP0 + PH + GAP, lo2, hi2);
    let y3 = y_map(TOP0 + 2.0 * (PH + GAP), lo3, hi3);
    let t1 = nice_ticks(lo1, hi1);
    let t2 = nice_ticks(lo2, hi2);
    let t3 = nice_ticks(lo3, hi3);

    // ---- 底色带：每快照、每子图一条竖直色带（颜色 = AI 决策类别，与 python 对齐）----
    for (i, s) in snaps.iter().enumerate() {
        let color = CAT_COLOR.iter().find(|(c, _)| *c == s.cat).map(|(_, col)| *col).unwrap_or(CAT_COLOR[5].1);
        for &top in &tops {
            svg.rect(X0 + (i as f64 - 0.5) * dx, top, dx, PH, color, 0.38);
        }
    }

    // ---- 子图装饰：标题 / 纵轴标题 / 纵坐标刻度与网格 / 边框 ----
    let y_titles = ["估分", "运气分", "回合波动"];
    for (k, title) in ["期望评分", "运气分", "运气波动"].iter().enumerate() {
        let top = tops[k];
        let ymap = [&y1, &y2, &y3][k];
        let ticks = [&t1, &t2, &t3][k];
        // 子图标题（居中于绘图区上方）
        svg.text((X0 + X1) / 2.0, top - 12.0, title, 14.0, "middle");
        // 纵轴标题（自下而上竖排，位于刻度数字左侧）
        svg.text_rot(X0 - 62.0, top + PH / 2.0, y_titles[k], 12.0, "middle", -90.0);
        // 纵坐标刻度数字 + 刻度线 + 水平网格
        for &v in ticks {
            let y = ymap(v);
            svg.text(X0 - 8.0, y + 4.0, &fmt_tick(v), 10.5, "end");
            svg.line(X0 - 5.0, y, X0, y, "#333333", 1.0);
            svg.line(X0, y, X1, y, "#e0e0e0", 0.5);
        }
        // 子图边框：四边（底边即横坐标轴）
        svg.frame(X0, top, X1 - X0, PH, "#333333", 1.0);
    }

    // ---- [期望评分] 蒙特卡洛估分 + 显示估分 ----
    for &(pts, clr, dashed) in &[(raw.as_slice(), RAW_CLR, false), (disp.as_slice(), DISP_CLR, true)] {
        let poly: Vec<(f64, f64)> = lx.iter().zip(pts.iter()).map(|(&i, v)| (px(i), y1(*v))).collect();
        svg.polyline(&poly, clr, 1.7, dashed);
        for &(x, y) in &poly {
            svg.circle(x, y, 2.5, clr, 1.0);
        }
    }

    // ---- [运气分] 显示累计（原始运气分按用户要求不显示）----
    svg.line(X0, y2(0.0), X1, y2(0.0), "gray", 0.8);
    for &(pts, clr) in &[(total.as_slice(), LUCK_CLR)] {
        let poly: Vec<(f64, f64)> = lx.iter().zip(pts.iter()).map(|(&i, v)| (px(i), y2(*v))).collect();
        svg.polyline(&poly, clr, 1.7, false);
        for &(x, y) in &poly {
            svg.circle(x, y, 2.5, clr, 1.0);
        }
    }
    // 原始运气分（raw 累计）：用户要求不显示，取消注释即恢复（需先恢复上方 raw_cum 计算）
    // {
    //     let poly: Vec<(f64, f64)> = lx.iter().zip(raw_cum.iter()).map(|(&i, v)| (px(i), y2(*v))).collect();
    //     svg.polyline(&poly, "#e8a3a6", 1.7, true);
    //     for &(x, y) in &poly {
    //         svg.circle(x, y, 2.5, "#e8a3a6", 1.0);
    //     }
    // }

    // ---- [运气波动] raw 相邻差柱状（正绿负红）----
    svg.line(X0, y3(0.0), X1, y3(0.0), "gray", 0.8);
    let bar_w = (dx * 0.8).max(1.0);
    for (k, &i) in lx.iter().enumerate() {
        if let Some(dv) = raw_step[k] {
            let (y_top, h) = if dv >= 0.0 { (y3(dv), y3(0.0) - y3(dv)) } else { (y3(0.0), y3(dv) - y3(0.0)) };
            svg.rect(px(i) - bar_w / 2.0, y_top, bar_w, h, if dv >= 0.0 { BAR_POS } else { BAR_NEG }, 0.85);
        }
    }

    // ---- x 刻度：回合变化处，过密抽样（标签 = 回合数）；末尾补「末回合 +1」刻度 ----
    let mut pts: Vec<(usize, String)> = Vec::new();
    let mut last: Option<u32> = None;
    for (i, s) in snaps.iter().enumerate() {
        if last != Some(s.turn) {
            pts.push((i, s.turn.to_string()));
            last = Some(s.turn);
        }
    }
    // 多留的那一格（末回合 +1）也标刻度：x 轴覆盖到 78 回合
    if let Some(&t) = snaps.last().map(|s| &s.turn) {
        pts.push((n, (t + 1).to_string()));
    }
    const MAX_TICKS: usize = 24;
    if pts.len() > MAX_TICKS {
        let step = (pts.len() - 1) as f64 / (MAX_TICKS - 1) as f64;
        pts = (0..MAX_TICKS).map(|k| pts[(k as f64 * step).round() as usize].clone()).collect();
    }
    let last_idx = pts.last().map(|(i, _)| *i);
    for (i, label) in &pts {
        svg.line(px(*i), tops[2] + PH, px(*i), tops[2] + PH + 6.0, "#555", 0.8);
        // 末尾刻度贴右边框：标签右对齐避免出界
        let anchor = if Some(*i) == last_idx { "end" } else { "middle" };
        svg.text(px(*i), tops[2] + PH + 22.0, label, 11.0, anchor);
    }
    svg.text((X0 + X1) / 2.0, tops[2] + PH + 46.0, "回合数", 12.0, "middle");

    // ---- 图例（右侧独立列；署名「由 UmaAI-Ramen 生成」在图例下方）----
    draw_legend(&mut svg, n as usize);

    svg.render()
}

/// 图例：类别色块 + 4 条曲线说明 + 柱状正负
fn draw_legend(svg: &mut Svg, n_snaps: usize) {
    svg.text(LEG_X, 70.0, "图例", 13.0, "start");
    let mut y = 95.0;
    for (label, color) in CAT_COLOR {
        svg.rect(LEG_X, y, 14.0, 14.0, color, 0.38);
        svg.text(LEG_X + 22.0, y + 12.0, label, 11.5, "start");
        y += 22.0;
    }
    y += 8.0;
    for (label, clr, dashed) in [
        ("蒙特卡洛估分", RAW_CLR, false),
        ("显示估分", DISP_CLR, true),
        ("显示运气分", LUCK_CLR, false),
        // 原始运气分（raw 累计）不列图例；如需连带隐藏曲线，在下方第二子图绘制处一并注释
        // ("原始运气分", "#e8a3a6", true),
    ] {
        if dashed {
            svg.polyline(&[(LEG_X, y + 7.0), (LEG_X + 22.0, y + 7.0)], clr, 1.7, true);
        } else {
            svg.line(LEG_X, y + 7.0, LEG_X + 22.0, y + 7.0, clr, 1.7);
        }
        svg.text(LEG_X + 30.0, y + 12.0, label, 11.5, "start");
        y += 22.0;
    }
    y += 8.0;
    svg.rect(LEG_X, y, 14.0, 14.0, BAR_POS, 0.85);
    svg.text(LEG_X + 22.0, y + 12.0, "回合运气(raw 相邻差 正)", 11.5, "start");
    y += 22.0;
    svg.rect(LEG_X, y, 14.0, 14.0, BAR_NEG, 0.85);
    svg.text(LEG_X + 22.0, y + 12.0, "回合运气(raw 相邻差 负)", 11.5, "start");
    y += 34.0;
    svg.text(LEG_X, y, &format!("快照数: {n_snaps}"), 11.5, "start");
    // 署名：图例下侧（快照数下一行）
    svg.text(LEG_X, y + 20.0, "由 UmaAI-Ramen 生成", 11.5, "start");
}

/// 求 y 范围（多序列合并；`force_zero` 时把 0 纳入范围）
fn y_range(series: &[&[f64]], force_zero: bool) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for s in series {
        for &v in s.iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if force_zero {
        lo = lo.min(0.0);
        hi = hi.max(0.0);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    let pad = ((hi - lo) * 0.08).max(1.0);
    (lo - pad, hi + pad)
}

/// 数据值 → 像素 y（子图顶部 top，高 PH）
fn y_map(top: f64, lo: f64, hi: f64) -> impl Fn(f64) -> f64 {
    let span = (hi - lo).max(1e-9);
    move |v| top + (1.0 - (v - lo) / span) * PH
}

/// 纵坐标刻度：在 `[lo, hi]` 内取「好看」的等距刻度（1/2/5×10^n 步长，目标约 5 条）
fn nice_ticks(lo: f64, hi: f64) -> Vec<f64> {
    const TARGET: f64 = 5.0;
    let span = (hi - lo).max(1e-9);
    let raw = span / TARGET;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = mag
        * if norm <= 1.0 {
            1.0
        } else if norm <= 2.0 {
            2.0
        } else if norm <= 5.0 {
            5.0
        } else {
            10.0
        };
    let mut out = Vec::new();
    let mut v = (lo / step).ceil() * step;
    // 只保留严格落在绘图区内的刻度（浮点累加可能略微越界）
    while v <= hi + step * 1e-9 {
        if v >= lo {
            out.push(v);
        }
        v += step;
    }
    out
}

/// 刻度数字格式：大数值取整、小数值保留 1~2 位小数
fn fmt_tick(v: f64) -> String {
    let a = v.abs();
    if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else if a > 0.0 {
        format!("{v:.2}")
    } else {
        "0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 决策类别分类：吃面优先、中文关键词、空串回退
    #[test]
    fn test_classify_action() {
        assert_eq!(classify_action("吃面/中山-全(替换Ax1+Bx2)"), "吃面");
        assert_eq!(classify_action("不吃面"), "吃面");
        assert_eq!(classify_action("训练/速"), "训练");
        assert_eq!(classify_action("出行"), "出行");
        assert_eq!(classify_action("休息"), "休息");
        assert_eq!(classify_action("比赛/杏目 G2"), "比赛");
        assert_eq!(classify_action(""), "地区选择");
    }

    /// 渲染一局：临时目录内放一份最小 decisions.csv → 生成 luck_trend.svg
    #[test]
    fn test_render_game_smoke() {
        let dir = std::env::temp_dir().join(format!("umaai_plot_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // 手工构造与真实 schema 一致的明细（两快照：skip + calc 链式）
        let csv = "\
game,file,turn,seq,source,playing_state,stage,outcome,reason,step,chain_len,decision_kind,n_actions,cand1_desc,cand2_desc,cand3_desc,cand4_desc,cand5_desc,cand1_score,cand2_score,cand3_score,cand4_score,cand5_score,cand1_n,cand2_n,cand3_n,cand4_n,cand5_n,chosen_idx,chosen_desc,chosen_action_luck,t_n_raw,t_n_display,total_luck,turn_delta
6222,game6222_turn0.json,0,0,command,1,Train,calc,,,2,train,2,训练/速,训练/智,,,,,245.00,200.00,,,,,20,10,,,,0,训练/速,5.00,50000.00,50154.00,0.00,
6222,game6222_turn1.json,1,0,command,1,RamenSelect,calc,,,2,ramen_select,2,吃面/中山-全,不吃面,,,,,300.00,250.00,,,,,20,10,,,,1,吃面/中山-全,3.00,50100.00,50206.00,52.00,52.00
6222,game6222_turn2.json,2,0,event,1,Begin,skip,event,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,";
        fs::write(dir.join("decisions.csv"), csv).unwrap();
        let out = render_game(&dir, 6222).expect("render");
        let svg = fs::read_to_string(&out).unwrap();
        println!("SVG 头 400 字符:\n{}", &svg[..400.min(svg.len())]);
        assert!(out.file_name().unwrap() == "luck_trend.svg");
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("game6222 拉面运气分趋势"));
        assert!(svg.contains("期望评分"));
        assert!(svg.contains("运气分"));
        assert!(svg.contains("运气波动"));
        assert!(svg.contains("训练"));
        // skip 快照（event Begin）不产生色带之外的数据点——两 calc 行应绘制两条曲线折线
        assert!(svg.contains("<polyline"));
        let _ = fs::remove_dir_all(&dir);
    }
}