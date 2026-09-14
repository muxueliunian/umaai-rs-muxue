"""显式清单采集：旧批保留截止语义，新批可显式无截止，均按有效目标恢复。"""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
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


def execute(command, evidence, deadline, threads, min_free_bytes=0):
    """日志直接流式写文件；退出码先保存；超时有界终止。"""
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / "args.json", command)
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    started = time.time()
    if deadline is not None and started >= deadline:
        raise TimeoutError("共享截止已到，未启动进程")
    flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else dict(start_new_session=True)
    with (evidence / "stdout.txt").open("wb", buffering=0) as out, (evidence / "stderr.txt").open("wb", buffering=0) as err:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, **flags)
        (evidence / "pid.txt").write_text(str(proc.pid), encoding="utf-8")
        timed_out = False
        try:
            while True:
                if min_free_bytes:
                    check_disk(evidence, min_free_bytes)
                if deadline is not None and time.time() >= deadline + 60:
                    timed_out = True
                    terminate(proc)
                    code = proc.returncode
                    break
                timeout = None if deadline is None else max(0.1, deadline + 60 - time.time())
                if min_free_bytes:
                    timeout = 30 if timeout is None else min(30, timeout)
                try:
                    code = proc.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            terminate(proc)
            (evidence / "exitcode.txt").write_text(str(proc.returncode), encoding="utf-8")
            raise
        (evidence / "exitcode.txt").write_text(str(code), encoding="utf-8")
    write_json(evidence / "timing.json", dict(seconds=time.time() - started, timed_out=timed_out))
    if timed_out or code:
        raise RuntimeError(f"进程未成功：exit={code} timeout={timed_out}，见 {evidence}")


def check_plan(plan_dir, plan):
    """开跑前验证全部清单、留出排除和配额，不执行采集。"""
    mode = plan.get("collection_profile", "legacy2048")
    if mode == "legacy2048":
        expected_n, expected_total = 2048, 22800
    elif mode == "newteacher4096":
        expected_n, expected_total = 4096, 50000
        if plan.get("seconds", "missing") is not None:
            raise ValueError("新批必须显式 seconds=null（无截止）")
        if plan.get("teacher_config") != {"ramen_pt_sacrifice_score": 0, "pt_favor_rate": 1.0}:
            raise ValueError("新教师参数不符")
    else:
        raise ValueError("未知采集配方")
    if plan["search_n"] != expected_n or plan["use_ucb"] or plan["target_valid"] != expected_total:
        raise ValueError("正式配方预算/总目标不符")
    if (plan.get("rollin") != "nn" or plan.get("epsilon") != .15
            or plan.get("radical_factor_max") != 1.4):
        raise ValueError("roll-in/epsilon/rf 不符")
    definitions = read_json(plan_dir / "plans.json")
    held = {p["plan"] for p in read_json(plan_dir / "holdout.json")["plans"]}
    if len(definitions) != 4288 or len(held) != 250:
        raise ValueError("空间/留出数量不符")
    if mode == "newteacher4096":
        original = ROOT / "scripts/collect/formal2048_0914"
        previous = read_json(original / "manifest.json")
        if any(plan.get(k) != previous[k] for k in
               ("space", "model", "model_id", "inherit", "seed_base", "shard_size")):
            raise ValueError("新批擅自改变冻结空间、roll-in模型或采样配方")
        if plan["index_reservation"][0] < 300000000:
            raise ValueError("不得复用旧正式预留号段")
        if (definitions != read_json(original / "plans.json")
                or read_json(plan_dir / "holdout.json") != read_json(original / "holdout.json")):
            raise ValueError("新批不得重排空间或重抽留出")
    seen, targets, cells = set(), {}, {}
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
        cell = (job["layer"], job["shape"])
        if cell in cells or job["target"] <= 0:
            raise ValueError("层构成重复或目标非法")
        cells[cell] = job["target"]
        quotas = {"general": (0, "0,0"), "region_y1": (1000, "0,0"),
                  "region_y2": (0, "1000,0"), "region_y3": (0, "0,1000")}
        if (job["quota_y1"], job["quota_y2_y3"]) != quotas.get(job["layer"]):
            raise ValueError("地区层配额不符")
        for index in values:
            p = index % len(definitions)
            if index in seen or p in held or definitions[p]["shape"] != job["shape"]:
                raise ValueError("清单重复、包含留出或构成不符")
            if not plan["index_reservation"][0] <= index < plan["index_reservation"][1]:
                raise ValueError("index 越界")
            seen.add(index)
    expected_cells = {}
    for shape in range(4):
        amounts = ([10000, 357, 1072 if shape < 2 else 1071, 1071 if shape < 2 else 1072]
                   if mode == "newteacher4096" else [5000, 100, 300, 300])
        expected_cells.update({(layer, shape): n for layer, n in
                               zip(("general", "region_y1", "region_y2", "region_y3"), amounts)})
    if cells != expected_cells or sum(targets.values()) != expected_total:
        raise ValueError("有效根总配额错误")
    return len(seen)


def check_disk(output, minimum):
    """保留明确的空闲磁盘安全余量；不足时不启动下一个子进程。"""
    if shutil.disk_usage(output).free < minimum:
        raise RuntimeError("磁盘余量低于安全阈值；完整产物保留，禁止自动删除旧数据")


def collect_command(exe, plan_dir, job, plan, data, model, remaining):
    """按冻结配方构造命令，无截止时完全不传 --max-seconds。"""
    cmd = [str(exe), "--space-version", "gen2_v1", "--indices-file", str(plan_dir / job["indices"]),
           "--start", "0", "--count", str(job["count"]), "--accepted-target", str(job["target"]),
           "--search-n", str(plan["search_n"]), "--shard-size", str(plan.get("shard_size", 32)),
           "--output-dir", str(data), "--region-quota-permille-y1", str(job["quota_y1"]),
           "--region-quota-permille", job["quota_y2_y3"], "--rollin", "nn", "--model", str(model),
           "--model-id", plan["model_id"]]
    if remaining is not None:
        cmd += ["--max-seconds", str(remaining)]
    if plan.get("collection_profile") == "newteacher4096":
        cmd += ["--require-newteacher-defaults"]
    return cmd


def verify_export(export, plan_dir, plan):
    """新批逐字段验收raw，完成标记不能先于验收。旧任务保持既有恢复语义。"""
    if plan.get("collection_profile") == "newteacher4096":
        from verify_raw_export import verify
        verify(export, plan_dir / "plans.json", plan["search_n"])


def run(args):
    """校验统一计划及资产后顺序采集，断点沿用最初截止而不自动续命。"""
    plan_dir = args.plan.resolve()
    plan = read_json(plan_dir / "manifest.json")
    check_plan(plan_dir, plan)
    if args.validate_only:
        print("正式清单校验通过；未启动采集")
        return
    if plan.get("collection_profile") == "newteacher4096" and not getattr(args, "reservation_confirmed", False):
        raise ValueError("新批必须先核对本机/云端历史与登记，再显式 --reservation-confirmed")
    if not args.exe or not args.export_exe or not args.output or not args.threads or not args.asset_reference:
        raise ValueError("实际开跑需要 --exe --export-exe --output --threads --asset-reference")
    exe, export_exe, output = args.exe.resolve(), args.export_exe.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT) or not exe.is_file() or not export_exe.is_file():
        raise ValueError("产物必须在工作区内，可执行文件必须存在")
    if args.threads <= 0:
        raise ValueError("threads 必须为正")
    if not 0 < getattr(args, "min_free_gib", 5) < 100000:
        raise ValueError("磁盘阈值必须为有限正数")
    model = ROOT / plan["model"]
    reference = args.asset_reference.resolve()
    if plan.get("collection_profile") == "newteacher4096":
        from prepare_collect_assets import FILES
        if read_json(reference / "files.json") != FILES:
            raise ValueError("新批资产参考必须包含完整模型、旁车、游戏基座及配置")
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
    if (plan.get("collection_profile") == "newteacher4096"
            and subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"],
                                        cwd=ROOT, text=True).strip()):
        raise ValueError("新批源码目录有未跟踪文件；先完成固定版本，不可用脏源码正式采集")
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
        state = dict(identity=identity, started=time.time(),
                     deadline=None if plan["seconds"] is None else time.time() + plan["seconds"], completed=[])
        write_json(state_path, state)
    deadline = state["deadline"]
    if (deadline is None) != (plan["seconds"] is None):
        raise ValueError("续跑截止模式改变")
    width = plan["search_n"]
    min_free = int(getattr(args, "min_free_gib", 5) * 1024 ** 3)
    for job in plan["jobs"]:
        if job["name"] in state["completed"]:
            saved_meta = read_json(ROOT / state["exports"][job["name"]] / "meta.json")
            saved_manifest = read_json(output / "data" / job["name"] / "manifest.json")
            if (saved_meta["stats"]["samples"] != job["target"]
                    or saved_meta["stats"]["rollout_width"] != width
                    or saved_manifest["work_indices"] != identity["indices"][job["name"]]
                    or saved_manifest["accepted"] != job["target"]):
                raise ValueError("已完成任务的证据缺失或字段变化")
            verify_export(ROOT / state["exports"][job["name"]], plan_dir, plan)
            continue
        data = output / "data" / job["name"]
        stamp = time.time_ns()
        manifest = data / "manifest.json"
        complete = manifest.exists() and read_json(manifest)["accepted"] == job["target"]
        if not complete:
            remain = None if deadline is None else int(deadline - time.time())
            if remain is not None and remain <= 0:
                raise TimeoutError("冻结时间窗口结束，完整分片保留")
            check_disk(output, min_free)
            cmd = collect_command(exe, plan_dir, job, plan, data, model, remain)
            execute(cmd, output / "evidence" / f"{job['name']}_{stamp}", deadline, args.threads, min_free)
        mf = read_json(manifest)
        if (mf["accepted"] != job["target"] or mf["search_n"] != width or mf["premises"]["use_ucb"]
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
        check_disk(output, min_free)
        execute([str(export_exe), "--space-version", "gen2_v1", "--input", str(data),
                 "--output-dir", str(export), "--raw"],
                output / "evidence" / f"export_{job['name']}_{stamp}", deadline, args.threads, min_free)
        meta = read_json(export / "meta.json")
        if meta["stats"]["samples"] != job["target"] or meta["stats"]["rollout_width"] != width:
            raise ValueError("导出有效根或列宽错误")
        verify_export(export, plan_dir, plan)
        state["completed"].append(job["name"])
        state.setdefault("exports", {})[job["name"]] = str(export.relative_to(ROOT))
        write_json(state_path, state)
        print(f"完成 {job['name']}: {job['target']} 有效根", flush=True)
    state["finished"] = time.time()
    write_json(state_path, state)
    print(f"{plan['target_valid']} 个有效根采集及逐任务 raw 导出完成；未启动训练", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "scripts/collect/formal2048_0914")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--export-exe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-reference", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--min-free-gib", type=float, default=5, help="子任务启动前最低磁盘余量")
    parser.add_argument("--reservation-confirmed", action="store_true", help="确认已核对所有预留及历史号段")
    run(parser.parse_args())
