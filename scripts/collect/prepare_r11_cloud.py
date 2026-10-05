"""R11 云端预检与参考资产冻结；不编译、不启动采集、不计算哈希。

与 R10 共用同一个 roll-in（Release `models-r8a-0920` 的 `ens_R8A_g123`）及资产冻结逻辑；
这里只换成 R11 的清单、号段与历史核对。
"""

import argparse
import subprocess
from pathlib import Path

from prepare_r10_cloud import freeze_assets, workspace_path
from prepare_r11_1005 import BATCH, COLLECT, MODEL_ID, RESERVATION, SEARCH_N, SMOKE_CLOUD
from run_formal_collect import ROOT, check_plan, read_json


def load_plan():
    """验证冻结清单、留出/排除、分轮配额与口径。"""
    folder = COLLECT / BATCH
    plan = read_json(folder / "manifest.json")
    check_plan(folder, plan, SEARCH_N, plan["target_valid"])
    if (plan["space"]["version"] != "gen2_v1" or plan["index_reservation"] != RESERVATION
            or plan["model_id"] != MODEL_ID or plan.get("friend_complete_required") is not True
            or not plan.get("rounds")):
        raise ValueError("空间、号段、模型、友人门限或分轮设置偏离 R11 配方")
    jobs = {job["name"]: read_json(folder / job["indices"]) for job in plan["jobs"]}
    return plan, jobs


def check_history(history_roots, plan, jobs, commit):
    """扫描云端全部历史 manifest；只允许同提交、同清单的本轮任务目录继续恢复。"""
    checked = 0
    for root in history_roots:
        root = workspace_path(root)
        if not root.is_dir():
            raise ValueError(f"历史数据目录不存在：{root}")
        for path in root.rglob("manifest.json"):
            workspace_path(path)
            old = read_json(path)
            checked += 1
            if path.resolve() == (COLLECT / BATCH / "manifest.json").resolve():
                continue
            indices = old.get("work_indices")
            spans = [(min(indices), max(indices) + 1)] if indices else []
            if old.get("index_reservation"):
                spans.append(tuple(old["index_reservation"]))
            if "index_start" in old and "index_end" in old:
                spans.append((int(old["index_start"]), int(old["index_end"])))
            hits = [s for s in spans for lo, hi in (RESERVATION, SMOKE_CLOUD) if s[0] < hi and s[1] > lo]
            if not hits:
                continue
            job = path.parent.name
            expected = ROOT / "training_data" / BATCH / "data" / job / "manifest.json"
            same_run = (indices and job in jobs and indices == jobs[job]
                        and path.resolve() == expected.resolve()
                        and old.get("git_commit") == commit
                        and old.get("search_n") == SEARCH_N
                        and old.get("rollin") == "nn:asset:" + MODEL_ID)
            in_smoke = all(SMOKE_CLOUD[0] <= s[0] and s[1] <= SMOKE_CLOUD[1] for s in hits)
            if not same_run and not (in_smoke and "smoke" in str(path.relative_to(ROOT))):
                raise ValueError(f"R11 号段与历史数据冲突：{path.relative_to(ROOT)}")
    print(f"历史清单核对 {checked} 份，R11 正式号段与冒烟号段无异批冲突")


def main():
    """要求指定已发布提交；先预检全部历史，再准备原文参考资产。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="本轮推送后确认的完整提交号")
    parser.add_argument("--history-root", type=Path, action="append", required=True,
                        help="所有历史采集目录，位于工作区内；可重复指定")
    parser.add_argument("--release-assets", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.commit:
        raise ValueError("当前提交与明确指定的发布提交不一致")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip():
        raise ValueError("已跟踪工作树有未提交修改，拒绝准备正式采集")
    plan, jobs = load_plan()
    check_history(args.history_root, plan, jobs, commit)
    freeze_assets(args.release_assets, args.reference_dir, commit)


if __name__ == "__main__":
    main()
