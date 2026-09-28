#!/usr/bin/env python3
"""新旧提交配对差分析（仅标准库）。

用法: analyze.py <results_dir>
读取 A_new/A_old/B_new/B_old 的 .journal.csv，按 (plan_index, run_idx) 配对，
输出 markdown 到 stdout。
"""
import csv, math, os, statistics as st, sys
from collections import defaultdict

# 双侧 95% t 临界值（df 1..30），df>30 用 1.96 近似
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
DIMS = ["score", "speed", "stamina", "power", "guts", "wisdom", "skill_pt"]


def tcrit(n):
    df = n - 1
    return T95[df - 1] if 1 <= df <= 30 else 1.96


def load(path):
    """journal -> {(plan_index, run_idx): row}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(int(r["plan_index"]), int(r["run_idx"]))] = r
    return out


def num(r, k):
    return float(r[k])


def ptsum(r):
    return num(r, "scenario_pt_y1") + num(r, "scenario_pt_y2") + num(r, "scenario_pt_y3")


def getter(k):
    return ptsum if k == "scenario_pt" else (lambda r, k=k: num(r, k))


def paired(new, old, key):
    """返回 (n, 均差, 半宽, 胜, 平, 负, sd)"""
    g = getter(key)
    ks = sorted(set(new) & set(old))
    d = [g(new[k]) - g(old[k]) for k in ks]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    hw = tcrit(n) * sd / math.sqrt(n) if n > 1 else float("nan")
    w = sum(1 for x in d if x > 0)
    t = sum(1 for x in d if x == 0)
    return n, m, hw, w, t, n - w - t, sd


def row(label, res, fmt="{:+.1f}"):
    n, m, hw, w, t, l, sd = res
    sig = "显著" if abs(m) > hw else "不显著"
    return f"| {label} | {n} | {fmt.format(m)} | [{fmt.format(m-hw)}, {fmt.format(m+hw)}] | {w}/{t}/{l} | {sig} |"


def table(title, new, old, extra_keys=True):
    lines = [f"#### {title}", "", "| 指标 | 配对局数 | new−old 均差 | 95% CI | 新胜/平/新负 | 结论 |", "|---|---:|---:|---|---|---|"]
    keys = DIMS + (["scenario_pt"] if extra_keys else [])
    for k in keys:
        lab = {"skill_pt": "skill_pt(技能PT)", "scenario_pt": "scenario_pt(剧本PT三年合计)"}.get(k, k)
        lines.append(row(lab, paired(new, old, k)))
    return "\n".join(lines)


def by_shape(new, old):
    ks = sorted(set(new) & set(old))
    groups = defaultdict(list)
    for k in ks:
        groups[new[k]["build"]].append(k)
    lines = ["| 构成(shape) | 局数 | score 均差 | 95% CI | 新胜/平/负 | 旧均分 | 新均分 |", "|---|---:|---:|---|---|---:|---:|"]
    for shp, kk in sorted(groups.items()):
        d = [num(new[k], "score") - num(old[k], "score") for k in kk]
        n = len(d)
        m = st.mean(d)
        sd = st.stdev(d) if n > 1 else 0
        hw = tcrit(n) * sd / math.sqrt(n) if n > 1 else float("nan")
        w = sum(1 for x in d if x > 0)
        t = sum(1 for x in d if x == 0)
        lines.append(f"| {shp} | {n} | {m:+.1f} | [{m-hw:+.1f}, {m+hw:+.1f}] | {w}/{t}/{n-w-t} | "
                     f"{st.mean(num(old[k],'score') for k in kk):.0f} | {st.mean(num(new[k],'score') for k in kk):.0f} |")
    return "\n".join(lines)


def completion(name, new, old):
    """友人出行走完率：仅新版 CSV 有 friend_out_*。"""
    lines = [f"#### {name}：友人出行走完率", ""]
    ks = sorted(new)
    outs = [sum(int(new[k][c]) for c in ("friend_out_y1", "friend_out_y2", "friend_out_y3")) for k in ks]
    n = len(outs)
    full = sum(1 for o in outs if o >= 5)
    lines.append(f"- 新版：{n} 局中出行次数 ≥5 的 **{full}** 局（{100*full/n:.1f}%）；平均出行 {st.mean(outs):.2f} 次")
    for y in ("y1", "y2", "y3"):
        v = [int(new[k][f"friend_out_{y}"]) for k in ks]
        lines.append(f"  - {y}：均值 {st.mean(v):.2f}，分布 " + str(dict(sorted(defaultdict(int, {x: v.count(x) for x in set(v)}).items()))))
    lines.append(f"- 新版 `friend_out_all` 标记（游戏内友人全走完标记）为 1 的比例："
                 f"{100*sum(int(new[k]['friend_out_all']) for k in ks)/n:.1f}%")
    if "friend_out_y1" in next(iter(old.values())):
        oo = [sum(int(old[k][c]) for c in ("friend_out_y1", "friend_out_y2", "friend_out_y3")) for k in sorted(old)]
        lines.append(f"- 旧版：{sum(1 for o in oo if o>=5)}/{len(oo)} 局 ≥5")
    else:
        lines.append("- 旧版：CSV 无 `friend_out_*` 列（该观测点是 0bcb79d 之后才有），**无法在不改源码的前提下取得旧版走完率**；"
                     "旧版仅有 `friend_turns_y*`（友人出现回合数），含义不同，不做比较。")
    return "\n".join(lines)


def pairing_check(name, new, old):
    ks = set(new) & set(old)
    lines = [f"#### {name}：世界配对核对", ""]
    lines.append(f"- 新 {len(new)} 局 / 旧 {len(old)} 局 / (plan_index, run_idx) 交集 {len(ks)}")
    same_seed = sum(1 for k in ks if new[k]["seed"] == old[k]["seed"])
    lines.append(f"- 逐局种子一致：{same_seed}/{len(ks)}")
    same_score = [k for k in ks if new[k]["score"] == old[k]["score"]]
    lines.append(f"- 分数完全相同的局：{len(same_score)}/{len(ks)}")
    common = [c for c in old[next(iter(ks))] if c in new[next(iter(ks))] and c not in ("build",)]
    ident = sum(1 for k in same_score if all(new[k][c] == old[k][c] for c in common if c != "elapsed_ms"))
    lines.append(f"- 其中所有公共列（除耗时）逐列相同：{ident}/{len(same_score)}（证明分数相同的局走了同一轨迹）")
    return "\n".join(lines)


def supplementary(d, old, new):
    """A2：新二进制 + --variant，拆分「配额」与「完成门限」的贡献。"""
    lab = {"fcap025": "配额还原为 [0,2,5]，门限关（= 旧口径）",
           "freq": "配额 [0,3,5]，门限开",
           "fcap025-freq": "配额 [0,2,5]，门限开"}
    lines = ["#### A 补充：新二进制 + `--variant` 拆分漂移来源（同种子同世界）", "",
             "| 新二进制配置 | 对 old 的 score 均差 | 95% CI | 新胜/平/负 | 与 old 全部公共列逐列相同的局 |", "|---|---:|---|---|---:|"]
    lines.append("| 默认（配额 [0,3,5]，门限关）＝ 上表 A 组 | %+.1f | [%+.1f, %+.1f] | %d/%d/%d | %d |" % (
        paired(new, old, "score")[1], paired(new, old, "score")[1]-paired(new, old, "score")[2],
        paired(new, old, "score")[1]+paired(new, old, "score")[2], *paired(new, old, "score")[3:6],
        sum(1 for k in new if all(new[k][c] == old[k][c] for c in old[k] if c in new[k] and c not in ("build", "elapsed_ms")))))
    for v in ("fcap025", "freq", "fcap025-freq"):
        p = os.path.join(d, f"A2_new_{v}.csv.journal.csv")
        if not os.path.exists(p):
            continue
        x = load(p)
        r = paired(x, old, "score")
        ident = sum(1 for k in x if all(x[k][c] == old[k][c] for c in old[k] if c in x[k] and c not in ("build", "elapsed_ms")))
        lines.append("| %s | %+.1f | [%+.1f, %+.1f] | %d/%d/%d | %d |" % (lab[v], r[1], r[1]-r[2], r[1]+r[2], r[3], r[4], r[5], ident))
    return "\n".join(lines)


def main(d):
    out = []
    for grp, title in (("A", "A 组：手写训练员（250 计划 × 4 局）"), ("B", "B 组：教师 search（n=1024, rf 1.4）")):
        pn, po = os.path.join(d, f"{grp}_new.csv.journal.csv"), os.path.join(d, f"{grp}_old.csv.journal.csv")
        if not (os.path.exists(pn) and os.path.exists(po)):
            continue
        new, old = load(pn), load(po)
        out += [f"### {title}", "", table("配对差（new − old）", new, old), "", pairing_check(grp, new, old), "",
                completion(grp, new, old), ""]
        if grp == "A":
            out += ["#### A 组：按构成拆分（score）", "", by_shape(new, old), "", supplementary(d, old, new), ""]
        else:
            out += ["#### B 组：逐局明细", "", "| plan_index | run_idx | 构成 | 旧 score | 新 score | 差 |", "|---:|---:|---|---:|---:|---:|"]
            for k in sorted(set(new) & set(old)):
                out.append(f"| {k[0]} | {k[1]} | {new[k]['build']} | {old[k]['score']} | {new[k]['score']} | {int(new[k]['score'])-int(old[k]['score']):+d} |")
            out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
