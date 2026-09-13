"""云端正式采集驱动：显式清单、累计有效目标、断点续跑、12 小时截止。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    """读取实际字段，兼容 PowerShell UTF-8 BOM。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """以临时文件替换状态；原始日志、数据分片不覆盖。"""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def terminate(proc):
    """仅终止本驱动启动的进程树，不按进程名称全杀。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=15)


def execute(command, evidence, deadline, threads):
    """日志直接流式写文件；退出码先保存；超时有界终止。"""
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / "args.json", command)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    started = time.time()
    if started >= deadline:
        raise TimeoutError("共享截止已到，未启动进程")
    flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else dict(start_new_session=True)
    with (evidence / "stdout.txt").open("wb", buffering=0) as out, (evidence / "stderr.txt").open("wb", buffering=0) as err:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, **flags)
        (evidence / "pid.txt").write_text(str(proc.pid), encoding="utf-8")
        timed_out = False
        try:
            code = proc.wait(timeout=max(1, deadline - time.time()) + 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate(proc)
            code = proc.returncode
        except BaseException:
            terminate(proc)
            raise
        (evidence / "exitcode.txt").write_text(str(code), encoding="utf-8")
    write_json(evidence / "timing.json", dict(seconds=time.time() - started, timed_out=timed_out))
    if timed_out or code:
        raise RuntimeError(f"进程未成功：exit={code} timeout={timed_out}，见 {evidence}")


def check_plan(plan_dir, plan):
    """开跑前验证全部清单、留出排除和配额，不执行采集。"""
    if plan["search_n"] != 2048 or plan["use_ucb"] or plan["target_valid"] != 22800:
        raise ValueError("正式配方必须为 2048 / uniform / 22800 有效根")
    definitions = read_json(plan_dir / "plans.json")
    held = {p["plan"] for p in read_json(plan_dir / "holdout.json")["plans"]}
    seen, targets = set(), {}
    for job in plan["jobs"]:
        name = job["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
            raise ValueError("非法任务名称")
        if name in targets:
            raise ValueError("重复任务名称")
        indices_path = (plan_dir / job["indices"]).resolve()
        if not indices_path.is_relative_to(plan_dir):
            raise ValueError("清单路径越界")
        values = read_json(indices_path)
        if len(values) != job["count"] or len(values) < job["target"]:
            raise ValueError("清单根数/备用数量不一致")
        targets[name] = job["target"]
        for index in values:
            p = index % len(definitions)
            if index in seen or p in held or definitions[p]["shape"] != job["shape"]:
                raise ValueError("清单重复、包含留出或构成不符")
            if not plan["index_reservation"][0] <= index < plan["index_reservation"][1]:
                raise ValueError("index 越界")
            seen.add(index)
    if sum(targets.values()) != 22800:
        raise ValueError("有效根总配额错误")
    return len(seen)


def run(args):
    """校验统一计划及资产后顺序采集，断点沿用最初截止而不自动续命。"""
    plan_dir = args.plan.resolve()
    plan = read_json(plan_dir / "manifest.json")
    check_plan(plan_dir, plan)
    if args.validate_only:
        print("正式清单校验通过；未启动采集")
        return
    if not args.exe or not args.export_exe or not args.output or not args.threads or not args.asset_reference:
        raise ValueError("实际开跑需要 --exe --export-exe --output --threads --asset-reference")
    exe, export_exe, output = args.exe.resolve(), args.export_exe.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT) or not exe.is_file() or not export_exe.is_file():
        raise ValueError("产物必须在工作区内，可执行文件必须存在")
    if args.threads <= 0:
        raise ValueError("threads 必须为正")
    model = ROOT / plan["model"]
    reference = args.asset_reference.resolve()
    # 参考目录由本机随模型传递；保存的是原始文件，不是大小或指纹证明。
    for relative in read_json(reference / "files.json"):
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("参考资产路径越界")
        if (ROOT / relative).read_bytes() != (reference / relative).read_bytes():
            raise ValueError(f"与本机冻结资产字节不同：{relative}")
    if not model.is_file() or plan["model"] not in read_json(reference / "files.json"):
        raise ValueError("参考资产中缺冻结模型")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip():
        raise ValueError("已跟踪代码有未提交修改，拒绝正式采集")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    identity = dict(commit=commit, plan=plan, threads=args.threads,
                    indices={j["name"]: read_json(plan_dir / j["indices"]) for j in plan["jobs"]})
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("续跑代码、配方、线程或完整 index 清单发生变化")
    else:
        if any(output.iterdir()):
            raise ValueError("输出目录非空却没有驱动状态，拒绝覆盖")
        state = dict(identity=identity, started=time.time(), deadline=time.time() + plan["seconds"], completed=[])
        write_json(state_path, state)
    deadline = state["deadline"]
    for job in plan["jobs"]:
        if job["name"] in state["completed"]:
            saved_meta = read_json(ROOT / state["exports"][job["name"]] / "meta.json")
            saved_manifest = read_json(output / "data" / job["name"] / "manifest.json")
            if (saved_meta["stats"]["samples"] != job["target"]
                    or saved_meta["stats"]["rollout_width"] != 2048
                    or saved_manifest["work_indices"] != identity["indices"][job["name"]]
                    or saved_manifest["accepted"] != job["target"]):
                raise ValueError("已完成任务的证据缺失或字段变化")
            continue
        data = output / "data" / job["name"]
        stamp = time.time_ns()
        manifest = data / "manifest.json"
        complete = manifest.exists() and read_json(manifest)["accepted"] == job["target"]
        if not complete:
            remain = int(deadline - time.time())
            if remain <= 0:
                raise TimeoutError("12 小时窗口结束，完整分片保留")
            cmd = [str(exe), "--space-version", "gen2_v1", "--indices-file", str(plan_dir / job["indices"]),
                   "--start", "0", "--count", str(job["count"]), "--accepted-target", str(job["target"]),
                   "--search-n", "2048", "--shard-size", "32", "--output-dir", str(data),
                   "--region-quota-permille-y1", str(job["quota_y1"]),
                   "--region-quota-permille", job["quota_y2_y3"], "--rollin", "nn", "--model", str(model),
                   "--model-id", plan["model_id"], "--max-seconds", str(remain)]
            execute(cmd, output / "evidence" / f"{job['name']}_{stamp}", deadline, args.threads)
        mf = read_json(manifest)
        if (mf["accepted"] != job["target"] or mf["search_n"] != 2048 or mf["premises"]["use_ucb"]
                or mf["work_indices"] != identity["indices"][job["name"]] or mf["git_commit"] != commit):
            raise ValueError("采集完成清单或有效根不符")
        sampler = mf["sampler"]
        if (mf["rollin"] != "nn:asset:" + plan["model_id"]
                or mf["premises"]["radical_factor_max"] != 1.4
                or mf["premises"]["ramen_region_strategy"] != "all"
                or not mf["premises"]["record_ordered_rollouts"]
                or sampler["epsilon"] != plan["epsilon"]
                or sampler["seed_base"] != plan["seed_base"]
                or sampler["inherit"]["blue_count"] != plan["inherit"]["blue_count"]
                or sampler["inherit"]["extra_count"] != plan["inherit"]["extra_count"]
                or sampler.get("region_quota_permille_y1", 0) != job["quota_y1"]
                or sampler["region_quota_permille"] != [int(v) for v in job["quota_y2_y3"].split(",")]):
            raise ValueError("实际生效的采集配方与正式清单不一致")
        for relative in read_json(reference / "files.json"):
            asset_name = ("assets/rollin.onnx.json" if relative == plan["model"] + ".json" else
                          "assets/rollin.onnx" if relative == plan["model"] else "assets/" + relative)
            if asset_name not in mf["asset_files"] or (data / asset_name).read_bytes() != (reference / relative).read_bytes():
                raise ValueError(f"采集资产与冻结原文字节不同：{relative}")
        # 每次导出到新目录，失败产物原样保留，不覆盖；采集完成但未导出时可以重试。
        export = output / "npy" / f"{job['name']}_{stamp}"
        execute([str(export_exe), "--space-version", "gen2_v1", "--input", str(data),
                 "--output-dir", str(export), "--raw"],
                output / "evidence" / f"export_{job['name']}_{stamp}", deadline, args.threads)
        meta = read_json(export / "meta.json")
        if meta["stats"]["samples"] != job["target"] or meta["stats"]["rollout_width"] != 2048:
            raise ValueError("导出有效根或列宽错误")
        state["completed"].append(job["name"])
        state.setdefault("exports", {})[job["name"]] = str(export.relative_to(ROOT))
        write_json(state_path, state)
        print(f"完成 {job['name']}: {job['target']} 有效根", flush=True)
    state["finished"] = time.time()
    write_json(state_path, state)
    print("22800 个有效根采集及逐任务 raw 导出完成；未启动训练", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "scripts/collect/formal2048_0914")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--export-exe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-reference", type=Path)
    parser.add_argument("--threads", type=int)
    run(parser.parse_args())
