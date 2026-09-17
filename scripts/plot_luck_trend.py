#!/usr/bin/env python3
"""绘制 luck_replay 运气分趋势图：每局一张 JPG。

读取 luck_replay 明细 CSV（`--csv` 可多次指定；默认自动发现工作区前缀
`luck_replay*.csv`、跳过 `*summary*`），按 `game` 分组，**每局输出一张图**。

图结构（竖排 3 子图，横轴 = 该局**非 skip** 快照的处理序号）：
  [评分]    期望终局分 T(n)：raw（实线）与显示口径（虚线，含 mcts_turn_bonus 换算）
  [运气分]  全局运气分累计：显示口径 `total_luck`（实线）+ raw 口径累计（虚线，
            以首行 raw T(n) 为 0 基准，消除每回合 bonus 衰减的假趋势）
  [运气波动]每步回合运气分：raw 相邻差柱状（正绿负红）

规则：
- skip 快照（事件 / RMJ / 数据不全 / 超级拉面丢包等）**不列入图表**；
- 每份快照一个竖直底色带，颜色按该快照 **AI 选中决策的类别**区分：
  训练 / 出行 / 休息 / 比赛 / 吃面 / 地区选择（时效无决策为地区选择）；
- 同一快照链式多决策时取末决策（主决策）分类。

用法（workspace 根）:
    python3 scripts/plot_luck_trend.py [--csv luck_replay_7075.csv] [--out logs/luck_replay_trend]
                                       [--games 7075,7076] [--title "拉面运气分趋势"]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import OrderedDict
from pathlib import Path

# matplotlib 缓存目录可能落在只读家目录（容器），指到系统临时目录
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 注册系统 CJK 字体并设为默认 sans-serif（仅 addfont 不会改变默认字体族）
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            fm.fontManager.addfont(_p)
        except RuntimeError:
            pass
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Droid Sans Fallback", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# AI 决策类别 → 色带颜色（浅色底、可透曲线）
CAT_COLOR = {
    "训练": "#ffffff",
    "出行": "#e08a3c",
    "休息": "#33a24d",
    "比赛": "#7b6cb5",
    "吃面": "#66c1f8",
    "地区选择": "#c8c8c8",
}


def classify_action(desc: str) -> str:
    """按选中动作描述分类 AI 决策（吃面 优先于 训练，避免「吃面/…」误分到训练）。"""
    if not desc:
        return "地区选择"
    if "吃面" in desc or "不吃面" in desc:
        return "吃面"
    if "训练" in desc:
        return "训练"
    if "出行" in desc:
        return "出行"
    if "休息" in desc:
        return "休息"
    if "比赛" in desc:
        return "比赛"
    return "地区选择"


def load_rows(paths: list[Path]) -> OrderedDict[str, list[dict]]:
    """读入 1+ 个明细 CSV，按 game 分组（保持各 CSV 内部处理顺序）。"""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                groups.setdefault(row["game"], []).append(row)
    return groups


def num(v: str | None) -> float | None:
    """CSV 空串 / None → None，否则 float。"""
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def group_snapshots(rows: list[dict]) -> list[dict]:
    """按快照（file）聚合，剔除 skip 快照；同快照链式多决策取**末决策**。

    返回每个快照一行精简信息：turn / seq / 主决策行 / 类别 / luck 数值。
    """
    by_file: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)

    out: list[dict] = []
    for file, rs in by_file.items():
        if all(r["outcome"] == "skip" for r in rs):
            continue  # skip 快照不列入图表
        main = rs[-1]  # 链式多决策取末决策（主决策）
        luck = next((r for r in reversed(rs) if r["t_n_raw"]), None)
        out.append({
            "file": file,
            "turn": main["turn"],
            "seq": main["seq"],
            "main": main,
            "cat": classify_action(main.get("chosen_desc", "")),
            "raw": num(luck["t_n_raw"]) if luck else None,
            "disp": num(luck["t_n_display"]) if luck else None,
            "total": num(luck["total_luck"]) if luck else None,
            "tdelta": num(luck["turn_delta"]) if luck else None,
        })
    return out


def pick_xticks(snaps: list[dict], max_ticks: int = 24) -> tuple[list[int], list[str]]:
    """x=快照序号；刻度放在 turn 变化处，标签=回合数（过密则抽样）。"""
    pts: list[tuple[int, str]] = []
    last = None
    for i, s in enumerate(snaps):
        if s["turn"] != last:
            pts.append((i, s["turn"]))
            last = s["turn"]
    if len(pts) > max_ticks:
        step = (len(pts) - 1) / (max_ticks - 1)
        pts = [pts[round(k * step)] for k in range(max_ticks)]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_game(game: str, rows: list[dict], out_path: Path, title: str | None) -> None:
    """绘制一局：3 子图（评分 / 运气分 / 运气波动），保存单张 JPG。"""
    snaps = group_snapshots(rows)
    n = len(snaps)
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle(title or f"game{game} 拉面运气分趋势", fontsize=14)

    # ---- 底色：每快照一条竖直色带（颜色=AI 决策类别）----
    for i, s in enumerate(snaps):
        color = CAT_COLOR.get(s["cat"], CAT_COLOR["地区选择"])
        for ax in axes:
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.38, linewidth=0)

    # ---- 数据准备（只含 luck 快照）----
    lx = [i for i, s in enumerate(snaps) if s["raw"] is not None]
    raw = [snaps[i]["raw"] for i in lx]
    disp = [snaps[i]["disp"] for i in lx]
    total = [snaps[i]["total"] for i in lx]
    raw_first = raw[0] if raw else 0.0
    raw_cum = [v - raw_first for v in raw]
    raw_step = [None] + [b - a for a, b in zip(raw, raw[1:])]

    # ---- [评分] T(n) 期望终局分 ----
    ax = axes[0]
    ax.plot(lx, raw, "-o", color="#0e4fa0", lw=1.7, ms=3, label="T(n) raw 期望分")
    ax.plot(lx, disp, "--o", color="#9cc3e5", lw=1.4, ms=3, label="T(n) 显示口径")
    ax.set_title("期望评分", fontsize=12)
    ax.grid(alpha=0.3)

    # ---- [运气分] 全局运气分累计 ----
    ax = axes[1]
    ax.plot(lx, total, "-o", color="#c44e52", lw=1.7, ms=3, label="显示运气分")
    ax.plot(lx, raw_cum, "--o", color="#e8a3a6", lw=1.4, ms=3, label="原始运气分")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("运气分", fontsize=12)
    ax.grid(alpha=0.3)

    # ---- [运气波动] 每步回合运气分 ----
    ax = axes[2]
    xs_b, ys_b = [], []
    for i, dv in zip(lx, raw_step):
        if dv is not None:
            xs_b.append(i)
            ys_b.append(dv)
    ax.bar(xs_b, ys_b, width=0.8,
           color=["#2ca02c" if v >= 0 else "#d62728" for v in ys_b],
           alpha=0.85, label="回合运气(raw 相邻差)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("运气波动", fontsize=12)
    ax.grid(alpha=0.3)

    xt, xl = pick_xticks(snaps)
    # 每个子图都标出 x 轴回合数刻度
    for ax in axes:
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=9)
        ax.tick_params(labelbottom=True)
    axes[-1].set_xlabel("快照序号")

    # fig 级图例：放在图右侧独立列，避免遮挡曲线 / 柱状
    handles = [Patch(facecolor=CAT_COLOR[c], alpha=0.38, label=c) for c in CAT_COLOR]
    handles += [
        Line2D([], [], color="#0e4fa0", lw=1.7, label="T(n) raw 期望分"),
        Line2D([], [], color="#9cc3e5", lw=1.4, ls="--", label="T(n) 显示口径"),
        Line2D([], [], color="#c44e52", lw=1.7, label="显示运气分"),
        Line2D([], [], color="#e8a3a6", lw=1.4, ls="--", label="原始运气分"),
        Patch(facecolor="#2ca02c", alpha=0.85, label="回合运气(raw 相邻差 正)"),
        Patch(facecolor="#d62728", alpha=0.85, label="回合运气(raw 相邻差 负)"),
    ]
    fig.legend(handles=handles, loc="center right", fontsize=9, frameon=True)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已输出 {out_path}  (game{game}: {n} 快照, {len(raw)} 个 luck 更新点)")


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制 luck_replay 运气分趋势图（每局一张 JPG）")
    ap.add_argument("--csv", action="append", default=None,
                    help="明细 CSV（可多次指定；默认自动发现 luck_replay*.csv，跳过 summary）")
    ap.add_argument("--out", default="logs/luck_replay_trend",
                    help="输出前缀（默认 logs/luck_replay_trend）→ <前缀>_<game>.jpg")
    ap.add_argument("--games", default=None, help="只画指定局，逗号分隔（如 7075,7076）")
    ap.add_argument("--title", default=None, help="图标题（默认 game<id> 拉面运气分趋势）")
    args = ap.parse_args()

    if args.csv:
        paths = [Path(p) for p in args.csv]
    else:
        paths = [Path(p) for p in glob.glob("luck_replay*.csv")]
        paths = [p for p in paths if "summary" not in p.name]
    if not paths:
        print("未找到明细 CSV（可用 --csv 指定）")
        return

    groups = load_rows(paths)
    if args.games:
        want = set(args.games.split(","))
        groups = OrderedDict((g, r) for g, r in groups.items() if g in want)

    for game, rows in groups.items():
        out_path = Path(f"{args.out}_{game}.jpg")
        plot_game(game, rows, out_path, args.title)


if __name__ == "__main__":
    main()