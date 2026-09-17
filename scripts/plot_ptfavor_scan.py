#!/usr/bin/env python3
"""绘制 MCTS pt_favor_rate 扫参：系数 vs 评分 / 技能点 关系图。

读取 logs/ptfavor-<rate>/bench_base_results.csv（每档 7 build × N 局），
x 轴 = pt_favor_rate，左轴 = 实际评分（calc_score 均值），右轴 = skill_pt 均值，
绘制均值和 ±1σ 误差棒；另附 Δscore / Δpt 相对 1.0 的变化子图。

用法（workspace 根）:
    python3 scripts/plot_ptfavor_scan.py [--rates 1.0,2.0,3.0,4.0,6.0,8.0,12.0]
                                         [--out .trae/documents/ptfavor_scan_1_12.png]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# matplotlib 缓存目录可能落在只读家目录（容器），指到系统临时目录
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 注册系统 CJK 字体（容器内 matplotlib 缓存不可写、不自动索引系统字体）
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            fm.fontManager.addfont(_p)
        except RuntimeError:
            pass

_CJK_NAME = None
for _f in fm.fontManager.ttflist:
    if "CJK" in _f.name or "WenQuanYi" in _f.name or "UMing" in _f.name:
        _CJK_NAME = _f.name
        break
if _CJK_NAME is None:
    print("[warn] 未找到 CJK 字体，中文可能显示为方块", file=sys.stderr)
    _CJK_NAME = "DejaVu Sans"

plt.rcParams["font.sans-serif"] = [_CJK_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORKSPACE = Path(__file__).resolve().parent.parent


def load_means(rate: str) -> dict:
    csv_path = WORKSPACE / "logs" / f"ptfavor-{rate}" / "bench_base_results.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    mean = lambda key: sum(float(r[key]) for r in rows) / n
    std = lambda key: (sum((float(r[key]) - mean(key)) ** 2 for r in rows) / n) ** 0.5
    return {
        "n": n,
        "score": mean("score"),
        "score_std": std("score"),
        "skill_pt": mean("skill_pt"),
        "skill_pt_std": std("skill_pt"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="1.0,2.0,3.0,4.0,6.0,8.0,12.0")
    ap.add_argument(
        "--out",
        default=".trae/documents/ptfavor_scan_1_12.png",
        help="输出图片路径（相对 workspace 根）",
    )
    args = ap.parse_args()
    rates = args.rates.split(",")

    data = {}
    for r in rates:
        try:
            data[r] = load_means(r)
        except FileNotFoundError:
            print(f"[skip] 缺失 logs/ptfavor-{r}/bench_base_results.csv", file=sys.stderr)
    if not data:
        raise SystemExit("无任何档位数据")

    xs = [float(r) for r in rates if r in data]
    score = [data[r]["score"] for r in rates if r in data]
    score_std = [data[r]["score_std"] for r in rates if r in data]
    pt = [data[r]["skill_pt"] for r in rates if r in data]
    pt_std = [data[r]["skill_pt_std"] for r in rates if r in data]

    base = data["1.0"]["score"]
    base_pt = data["1.0"]["skill_pt"]
    d_score = [s - base for s in score]
    d_pt = [p - base_pt for p in pt]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.suptitle("MCTS pt_favor_rate 扫参：评分与技能点的得失", fontsize=14)

    # 上图：绝对值双轴
    c_score, c_pt = "#1f6fb2", "#d97b29"
    l1, = ax1.plot(xs, score, marker="o", color=c_score, label="实际评分 (score)")
    ax1.errorbar(xs, score, yerr=score_std, fmt="none", color=c_score, alpha=0.35, capsize=3)
    ax1.set_ylabel("实际评分（calc_score 均值）", color=c_score)
    ax1.tick_params(axis="y", labelcolor=c_score)
    ax1.set_ylim(min(score) - 2500, max(score) + 1500)

    ax1b = ax1.twinx()
    l2, = ax1b.plot(xs, pt, marker="s", color=c_pt, label="技能点 (skill_pt)")
    ax1b.errorbar(xs, pt, yerr=pt_std, fmt="none", color=c_pt, alpha=0.35, capsize=3)
    ax1b.set_ylabel("技能点（skill_pt 均值）", color=c_pt)
    ax1b.tick_params(axis="y", labelcolor=c_pt)
    ax1b.set_ylim(min(pt) - 250, max(pt) + 150)

    for x, s, p in zip(xs, score, pt):
        ax1.annotate(f"{s:.0f}", (x, s), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color=c_score)
        ax1b.annotate(f"{p:.0f}", (x, p), textcoords="offset points", xytext=(0, -14),
                      ha="center", fontsize=8, color=c_pt)

    ax1.legend(handles=[l1, l2], loc="lower left", fontsize=9)
    ax1.grid(alpha=0.25, linestyle="--")

    # 下图：相对 1.0 的变化
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.plot(xs, d_score, marker="o", color=c_score, label="Δscore vs 1.0")
    ax2.plot(xs, d_pt, marker="s", color=c_pt, label="Δskill_pt vs 1.0")
    for x, ds, dp in zip(xs, d_score, d_pt):
        if x == xs[0]:
            continue  # 基线点为 0，跳过避免标注重叠
        ax2.annotate(f"{ds:+.0f}", (x, ds), textcoords="offset points", xytext=(0, -13),
                     ha="center", fontsize=8, color=c_score)
        ax2.annotate(f"{dp:+.0f}", (x, dp), textcoords="offset points", xytext=(0, -26),
                     ha="center", fontsize=8, color=c_pt)
    ax2.set_xlabel("pt_favor_rate")
    ax2.set_ylabel("Δ vs 1.0")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.set_xticks(xs)

    out = WORKSPACE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"已保存: {out}  （档位 {rates}，n={data[rates[0]]['n']}/档）")


if __name__ == "__main__":
    main()