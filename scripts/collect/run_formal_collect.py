"""云端正式采集驱动：显式清单、累计有效目标、断点续跑、清单自带截止。

截止秒数取自清单的 `seconds` 字段，不在本文件里写死；R6 是 72000 秒（20 小时），
0914 那轮是 43200 秒（12 小时）。
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
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


def execute(command, evidence, deadline, threads, ok_codes=(0,)):
    """日志直接流式写文件；退出码先保存；超时有界终止。返回退出码，`ok_codes` 以外一律报错。"""
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
    if timed_out or code not in ok_codes:
        raise RuntimeError(f"进程未成功：exit={code} timeout={timed_out}，见 {evidence}")
    return code


def check_plan(plan_dir, plan, expect_search_n, expect_target_valid):
    """开跑前验证全部清单、留出排除和配额，不执行采集。

    `expect_*` 是**操作者在命令行上显式声明的意图**，与 manifest 里的事实交叉核对。
    历史上这两个数字写死在本文件里（2048 / 22800）：能挡住手滑，但每换一轮配方就要改代码，
    而改代码本身又会被「已跟踪代码有未提交修改，拒绝正式采集」那条拦住。改成
    「manifest 提供事实、命令行提供意图、两者必须一致」后，
    **单改 manifest 仍然无法静默改变开跑口径**——这条防线原样保留。
    """
    if plan["search_n"] != expect_search_n or plan["target_valid"] != expect_target_valid:
        raise ValueError(
            f"清单与命令行声明不符：manifest 是 search_n={plan['search_n']} / "
            f"target_valid={plan['target_valid']}，命令行声明的是 "
            f"{expect_search_n} / {expect_target_valid}")
    if plan["use_ucb"] or plan["search_n"] <= 0 or plan["shard_size"] <= 0:
        raise ValueError("正式配方必须 uniform（use_ucb=false），且 search_n / shard_size 为正")
    definitions = read_json(plan_dir / "plans.json")
    held = {p["plan"] for p in read_json(plan_dir / "holdout.json")["plans"]}
    # 额外排除（开发验证、已登记评测面板）：完整字段必须与计划原文一致，数量与 manifest 声明一致
    excluded = read_json(plan_dir / "exclusions.json")["plans"] if (plan_dir / "exclusions.json").exists() else []
    if len(excluded) != plan.get("excluded_count", 0):
        raise ValueError("exclusions.json 条数与 manifest 声明不符")
    for row in excluded:
        if definitions[row["plan"]]["fields"] != row["fields"]:
            raise ValueError("exclusions.json 的组合字段与计划原文不符")
    held |= {row["plan"] for row in excluded}
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
    if sum(targets.values()) != plan["target_valid"]:
        raise ValueError("有效根总配额错误")
    if plan.get("rounds"):
        # 分轮清单：每轮配额相同，任务严格按轮排列，截断时已完成的整轮才是分层均衡的
        order = [job["round"] for job in plan["jobs"]]
        per_round = {r: sum(j["target"] for j in plan["jobs"] if j["round"] == r) for r in set(order)}
        if (order != sorted(order) or sorted(per_round) != list(range(1, plan["rounds"] + 1))
                or set(per_round.values()) != {plan["round_target"]}):
            raise ValueError("分轮清单的轮序或每轮配额不一致")
    return len(seen)


EXIT_EARLY_STOP = 3  # 采集器约定：根间软截止或停止文件触发的正常收尾（有效根未满）
EXPORT_GRACE = 1800  # 导出不受采集截止约束，另给半小时上限，避免截止前刚采完的任务无法导出


def check_job_manifest(mf, plan, job, identity, commit, reference, data):
    """核对采集器实际生效的配方、工作清单、提交与原文资产；不核对有效根数。"""
    if (mf["search_n"] != plan["search_n"] or mf["premises"]["use_ucb"]
            or mf["work_indices"] != identity["indices"][job["name"]] or mf["git_commit"] != commit):
        raise ValueError("采集清单、search_n 或提交不符")
    sampler = mf["sampler"]
    if (mf["rollin"] != "nn:asset:" + plan["model_id"]
            or mf["premises"]["radical_factor_max"] != 1.4
            or mf["premises"]["ramen_region_strategy"] != "all"
            or mf["premises"].get("friend_complete_required", False)
            != plan.get("friend_complete_required", False)
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


def export_raw(export_exe, plan, data, export, evidence, threads, samples):
    """导出到全新目录（失败产物原样保留），核对样本数与 rollout 列宽。"""
    execute([str(export_exe), "--space-version", plan["space"]["version"], "--input", str(data),
             "--output-dir", str(export), "--raw"],
            evidence, time.time() + EXPORT_GRACE, threads)
    meta = read_json(export / "meta.json")
    if meta["stats"]["samples"] != samples or meta["stats"]["rollout_width"] != plan["search_n"]:
        raise ValueError("导出有效根或列宽错误")


def finish_truncated(state, state_path, plan, partial=None):
    """分轮清单到达截止：已完成任务全部保留，未满任务的完整分片另行导出，正常结束。"""
    done = set(state["completed"])
    rounds = sorted({j["round"] for j in plan["jobs"]})
    full = [r for r in rounds if all(j["name"] in done for j in plan["jobs"] if j["round"] == r)]
    accepted = sum(j["target"] for j in plan["jobs"] if j["name"] in done)
    state["truncated"] = dict(at=time.time(), full_rounds=len(full), completed_jobs=len(done),
                              accepted_in_completed_jobs=accepted, partial=partial)
    state["finished"] = time.time()
    write_json(state_path, state)
    extra = f"；未满任务 {partial['job']} 另有 {partial['accepted']} 根已导出" if partial else ""
    print(f"采集窗口结束（清单声明 {plan['seconds']} 秒）：完整 {len(full)} 轮，"
          f"已完成任务 {len(done)} 个共 {accepted} 有效根{extra}；未启动训练", flush=True)


def run(args):
    """校验统一计划及资产后顺序采集，断点沿用最初截止而不自动续命。

    `--stop-file`：任务之间检查，并传给采集器在根之间检查；存在即安全停止（退出码 3），
    删掉文件后用同一命令续跑。分轮清单（manifest 含 `rounds`）到达截止视为正常结束。
    `--from-round`：分轮清单续采，跳过此前各轮；须配新输出目录，截止从新目录首次启动起算。
    """
    plan_dir = args.plan.resolve()
    plan = read_json(plan_dir / "manifest.json")
    check_plan(plan_dir, plan, args.expect_search_n, args.expect_target_valid)
    if args.validate_only:
        print("正式清单校验通过；未启动采集")
        return 0
    if not args.exe or not args.export_exe or not args.output or not args.threads or not args.asset_reference:
        raise ValueError("实际开跑需要 --exe --export-exe --output --threads --asset-reference")
    exe, export_exe, output = args.exe.resolve(), args.export_exe.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT) or not exe.is_file() or not export_exe.is_file():
        raise ValueError("产物必须在工作区内，可执行文件必须存在")
    if args.threads <= 0:
        raise ValueError("threads 必须为正")
    from_round = args.from_round
    if from_round != 1 and not (plan.get("rounds") and 1 < from_round <= plan["rounds"]):
        raise ValueError("--from-round 只用于分轮清单，且须在 2..rounds 之内")
    stop_file = args.stop_file.resolve() if args.stop_file else None
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
    if from_round > 1:
        # 只在续采时写入，旧输出目录的 run_state 身份保持不变、可照常续跑
        identity["from_round"] = from_round
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("续跑代码、配方、线程或完整 index 清单发生变化")
        if state.get("finished"):
            print("该输出目录已结束（见 run_state.json）；不再采集", flush=True)
            return 0
    else:
        if any(output.iterdir()):
            raise ValueError("输出目录非空却没有驱动状态，拒绝覆盖")
        state = dict(identity=identity, started=time.time(), deadline=time.time() + plan["seconds"], completed=[])
        write_json(state_path, state)
    deadline = state["deadline"]
    rounds_plan = bool(plan.get("rounds"))
    for job in plan["jobs"]:
        if rounds_plan and job["round"] < from_round:
            continue
        if job["name"] in state["completed"]:
            saved_meta = read_json(ROOT / state["exports"][job["name"]] / "meta.json")
            saved_manifest = read_json(output / "data" / job["name"] / "manifest.json")
            if (saved_meta["stats"]["samples"] != job["target"]
                    or saved_meta["stats"]["rollout_width"] != plan["search_n"]
                    or saved_manifest["work_indices"] != identity["indices"][job["name"]]
                    or saved_manifest["accepted"] != job["target"]):
                raise ValueError("已完成任务的证据缺失或字段变化")
            continue
        if stop_file and stop_file.exists():
            print(f"检测到停止文件，已在任务之间安全停止（下一个任务 {job['name']}）；"
                  "删除停止文件后用同一命令续跑", flush=True)
            return EXIT_EARLY_STOP
        data = output / "data" / job["name"]
        stamp = time.time_ns()
        manifest = data / "manifest.json"
        complete = manifest.exists() and read_json(manifest)["accepted"] == job["target"]
        if not complete:
            remain = int(deadline - time.time())
            if remain <= 0:
                if rounds_plan:
                    finish_truncated(state, state_path, plan)
                    return 0
                raise TimeoutError(
                    f"采集窗口结束（清单声明 {plan['seconds']} 秒），完整分片保留")
            cmd = [str(exe), "--space-version", plan["space"]["version"],
                   "--indices-file", str(plan_dir / job["indices"]),
                   "--start", "0", "--count", str(job["count"]), "--accepted-target", str(job["target"]),
                   "--search-n", str(plan["search_n"]),
                   "--shard-size", str(plan["shard_size"]), "--output-dir", str(data),
                   "--region-quota-permille-y1", str(job["quota_y1"]),
                   "--region-quota-permille", job["quota_y2_y3"], "--rollin", "nn", "--model", str(model),
                   "--model-id", plan["model_id"], "--max-seconds", str(remain)]
            if stop_file:
                cmd += ["--stop-file", str(stop_file)]
            code = execute(cmd, output / "evidence" / f"{job['name']}_{stamp}", deadline, args.threads,
                           ok_codes=(0, EXIT_EARLY_STOP))
            if code == EXIT_EARLY_STOP:
                mf = read_json(manifest)
                check_job_manifest(mf, plan, job, identity, commit, reference, data)
                if not mf["accepted"] < job["target"]:
                    raise ValueError("采集器报告提前停止，但有效根已满")
                if stop_file and stop_file.exists():
                    print(f"检测到停止文件，{job['name']} 已在根之间安全停止（{mf['accepted']} / "
                          f"{job['target']}）；删除停止文件后用同一命令续跑", flush=True)
                    return EXIT_EARLY_STOP
                if not rounds_plan:
                    raise TimeoutError(
                        f"采集窗口结束（清单声明 {plan['seconds']} 秒），完整分片保留")
                partial = dict(job=job["name"], accepted=mf["accepted"], target=job["target"])
                if mf["accepted"]:
                    export = output / "npy" / f"{job['name']}_partial_{stamp}"
                    export_raw(export_exe, plan, data, export,
                               output / "evidence" / f"export_{job['name']}_partial_{stamp}",
                               args.threads, mf["accepted"])
                    partial["export"] = str(export.relative_to(ROOT))
                finish_truncated(state, state_path, plan, partial)
                return 0
        mf = read_json(manifest)
        if mf["accepted"] != job["target"]:
            raise ValueError("采集完成清单的有效根不符")
        check_job_manifest(mf, plan, job, identity, commit, reference, data)
        # 每次导出到新目录，失败产物原样保留，不覆盖；采集完成但未导出时可以重试。
        export = output / "npy" / f"{job['name']}_{stamp}"
        export_raw(export_exe, plan, data, export, output / "evidence" / f"export_{job['name']}_{stamp}",
                   args.threads, job["target"])
        state["completed"].append(job["name"])
        state.setdefault("exports", {})[job["name"]] = str(export.relative_to(ROOT))
        write_json(state_path, state)
        print(f"完成 {job['name']}: {job['target']} 有效根", flush=True)
    state["finished"] = time.time()
    write_json(state_path, state)
    total = sum(j["target"] for j in plan["jobs"] if j.get("round", 1) >= from_round)
    print(f"{total} 个有效根采集及逐任务 raw 导出完成；未启动训练", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "scripts/collect/formal2048_0914")
    # 必填，且必须与 --plan 指向的 manifest 完全一致：让「这一轮到底按什么口径跑」
    # 必须在命令行上写出来，从而留在 shell 历史与 evidence/args.json 里。
    # 0914 那轮是 --expect-search-n 2048 --expect-target-valid 22800。
    parser.add_argument("--expect-search-n", type=int, required=True)
    parser.add_argument("--expect-target-valid", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--export-exe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-reference", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--stop-file", type=Path,
                        help="安全停止文件：存在即在任务或根之间停止（退出码 3），删除后同一命令续跑")
    parser.add_argument("--from-round", type=int, default=1,
                        help="分轮清单续采：跳过此前各轮，配新输出目录（截止重新起算）；"
                             "上一批被截断的那一轮整轮跳过，保证 index 不与上一批重叠")
    sys.exit(run(parser.parse_args()))
