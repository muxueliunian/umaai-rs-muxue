"""以实际字段验收一个raw导出，分块读取大数组，不计算内容指纹。"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from run_formal_collect import read_json


def verify(directory, plans_path, width):
    """验证候选CSR、CRN列宽、有效计数、统计量、合法格位及组合身份。"""
    names = ("x", "stage", "turn", "index", "cand_ptr", "cand_slots", "cand_n",
             "cand_mean", "cand_stdev", "legal_mask", "cand_scores", "cand_valid", "combo_fields")
    a = {n: np.load(directory / (n + ".npy"), mmap_mode="r", allow_pickle=False) for n in names}
    n, c = len(a["x"]), len(a["cand_n"])
    expected = dict(x=(n, 754), stage=(n,), turn=(n,), index=(n,), cand_ptr=(n + 1,),
                    cand_slots=(c, 3), cand_n=(c,), cand_mean=(c,), cand_stdev=(c,),
                    legal_mask=(n, 234), cand_scores=(c, width), cand_valid=(c, width),
                    combo_fields=(n, 7))
    for key, shape in expected.items():
        if a[key].shape != shape:
            raise ValueError(f"{key} shape不符：{a[key].shape} != {shape}")
    if (not n or len(np.unique(a["index"])) != n or not np.isfinite(a["x"]).all()
            or a["cand_ptr"][0] != 0 or a["cand_ptr"][-1] != c
            or np.any(np.diff(a["cand_ptr"]) <= 0)):
        raise ValueError("空根、重复index、x非有限或CSR非法")
    mean_error = stdev_error = 0.0
    for start in range(0, c, 1024):
        end = min(c, start + 1024)
        valid = np.asarray(a["cand_valid"][start:end], dtype=bool)
        scores = np.asarray(a["cand_scores"][start:end], dtype=np.float64)
        counts = valid.sum(1)
        if (not np.array_equal(counts, a["cand_n"][start:end]) or (counts < 2).any()
                or not np.isfinite(scores[valid]).all()):
            raise ValueError("有效槽计数/分数非法")
        masked = np.where(valid, scores, 0)
        means = masked.sum(1) / counts
        diffs = np.where(valid, scores - means[:, None], 0)
        stdevs = np.sqrt((diffs * diffs).sum(1) / (counts - 1))
        for name, values in (("cand_mean", means), ("cand_stdev", stdevs)):
            if not np.allclose(values, a[name][start:end], rtol=1e-6, atol=1e-3):
                raise ValueError(f"{name}与raw重算不符")
        mean_error = max(mean_error, float(np.max(np.abs(means - a["cand_mean"][start:end]))))
        stdev_error = max(stdev_error, float(np.max(np.abs(stdevs - a["cand_stdev"][start:end]))))
    plans = read_json(plans_path)
    for i in range(n):
        start, end = a["cand_ptr"][i:i + 2]
        slots = np.asarray(a["cand_slots"][start:end])
        active = slots[slots >= 0]
        if (slots < -1).any() or (active >= 234).any():
            raise ValueError("候选格位越界")
        mask = np.zeros(234, dtype=bool)
        mask[active] = True
        if not np.array_equal(mask, a["legal_mask"][i]):
            raise ValueError("候选/合法掩码不符")
        if int(a["stage"][i]) == 4:
            turn = int(a["turn"][i])
            bounds = {2: (214, 219), 23: (219, 224), 47: (224, 234)}
            if turn not in bounds:
                raise ValueError("地区回合非法")
            lo, hi = bounds[turn]
            if sorted(map(tuple, np.sort(slots, axis=1).tolist())) != list(combinations(range(lo, hi), 3)):
                raise ValueError("地区候选不是完整组合集")
        elif (slots[:, 0] < 0).any() or (slots[:, 1:] != -1).any():
            raise ValueError("非地区候选必须恰好一格")
        fields = plans[int(a["index"][i]) % len(plans)]["fields"]
        if not np.array_equal(fields, a["combo_fields"][i]):
            raise ValueError("index与完整马娘卡组身份不符")
    report = dict(samples=n, candidates=c, width=width, mean_max_abs=mean_error,
                  stdev_max_abs=stdev_error, stage=a["stage"].tolist() if n < 10 else "large",
                  turn=a["turn"].tolist() if n < 10 else "large",
                  invalid_slots=int(c * width - np.sum(a["cand_n"], dtype=np.int64)))
    print(directory, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    verify(args.input, args.plans, args.width)
