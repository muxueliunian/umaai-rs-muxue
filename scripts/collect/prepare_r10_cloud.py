"""R10 云端预检与参考资产冻结；不编译、不启动采集、不计算哈希。"""

import argparse
import json
import subprocess
from pathlib import Path

from prepare_r10_1004 import BATCHES, COLLECT, MODEL, MODEL_ID, ROOT, SEARCH_N
from run_formal_collect import check_plan, read_json

ASSETS = ["gamedata/constants.json", "gamedata/events.json", "gamedata/umaDB.json",
          "gamedata/cardDB.json", "gamedata/scenario_ramen.json",
          "gamedata/default_config.toml", "game_config.toml", MODEL, MODEL + ".json"]


def workspace_path(path):
    """解析工作区内路径，拒绝经软链接越界的输入和输出。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"路径必须位于当前工作区：{path}")
    return resolved


def load_plans():
    """逐批验证冻结配方、完整 index、留出和配额。"""
    plans = []
    for name, batch in BATCHES.items():
        folder = COLLECT / f"r10_{name}_1004"
        plan = read_json(folder / "manifest.json")
        target = sum(map(sum, batch["targets"]))
        check_plan(folder, plan, SEARCH_N, target)
        if (plan["space"]["version"] != batch["version"]
                or plan["index_reservation"] != batch["reservation"]
                or plan["model_id"] != MODEL_ID
                or plan.get("friend_complete_required") is not True):
            raise ValueError(f"{name}: 版本、号段、模型或友人门限偏离 R10 配方")
        jobs = {job["name"]: read_json(folder / job["indices"]) for job in plan["jobs"]}
        plans.append((name, plan, jobs))
    return plans


def check_history(history_roots, plans, commit):
    """检查云端已有采集清单；只允许同提交、同空间的本轮计划继续恢复。"""
    checked = 0
    for root in history_roots:
        root = workspace_path(root)
        if not root.is_dir():
            raise ValueError(f"历史数据目录不存在：{root}")
        for path in root.rglob("manifest.json"):
            workspace_path(path)
            old = read_json(path)
            indices = old.get("work_indices")
            checked += 1
            for name, plan, jobs in plans:
                start, end = plan["index_reservation"]
                overlap = (any(start <= i < end for i in indices) if indices else
                           int(old.get("index_start", 0)) < end
                           and int(old.get("index_end", 0)) > start)
                if not overlap:
                    continue
                job = path.parent.name
                expected = ROOT / "training_data" / f"r10_{name}_1004" / "data" / job / "manifest.json"
                same_run = (indices and job in jobs and indices == jobs[job]
                            and path.resolve() == expected.resolve()
                            and old.get("git_commit") == commit
                            and (old.get("space") or {}).get("version") == plan["space"]["version"]
                            and old.get("rollin") == "nn:asset:" + MODEL_ID
                            and old.get("search_n") == SEARCH_N)
                if not same_run:
                    raise ValueError(f"正式号段与历史数据冲突：{path.relative_to(ROOT)}")
    print(f"历史清单核对 {checked} 份，R10 正式号段无异批冲突")


def freeze_assets(release_assets, reference, commit):
    """与单独下载的 Release 文件逐字节比较，再冻结本提交的工作区资产。"""
    release_assets, reference = workspace_path(release_assets), workspace_path(reference)
    source_bytes = {}
    for relative in ASSETS:
        source = workspace_path(ROOT / relative)
        if relative in (MODEL, MODEL + ".json"):
            release = workspace_path(release_assets / Path(relative).name)
            if release == source:
                raise ValueError("Release 参考文件必须单独下载，不能拿模型自身作为参考")
            data = release.read_bytes()
            if source.exists() and source.read_bytes() != data:
                raise ValueError(f"当前模型与 Release 原文字节不同：{relative}")
        else:
            data = source.read_bytes()
        source_bytes[relative] = data
    provenance = dict(git_commit=commit, model_id=MODEL_ID,
                      model_release="models-r8a-0920", purchase_buffs=[],
                      source="clean_checkout_and_separately_downloaded_release")
    if reference.exists():
        if (read_json(reference / "files.json") != ASSETS
                or read_json(reference / "provenance.json") != provenance):
            raise ValueError("参考目录已存在但来源不同；保留旧目录，改用新的目录名")
        for relative, data in source_bytes.items():
            if workspace_path(reference / relative).read_bytes() != data:
                raise ValueError(f"冻结参考资产发生变化：{relative}")
    else:
        reference.mkdir(parents=True)
        for relative, data in source_bytes.items():
            dest = workspace_path(reference / relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        for name, value in [("files.json", ASSETS), ("provenance.json", provenance)]:
            (reference / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    for relative in (MODEL, MODEL + ".json"):
        dest = workspace_path(ROOT / relative)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source_bytes[relative])
    print(f"参考资产已冻结：{reference.relative_to(ROOT)}；未启动采集")


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
    plans = load_plans()
    check_history(args.history_root, plans, commit)
    freeze_assets(args.release_assets, args.reference_dir, commit)


if __name__ == "__main__":
    main()
