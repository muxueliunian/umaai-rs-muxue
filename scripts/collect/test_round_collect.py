"""分轮清单、额外排除与安全停止/截止收尾的针对性测试；只用假进程，不启动真实采集。"""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_gen2_formal as gen  # noqa: E402
import run_formal_collect as formal  # noqa: E402

ROOT = gen.ROOT
RECIPE = HERE / "gen2_v1_recipe.json"
FAKE_COMMIT = "f" * 40

FAKE_COLLECT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
ctx = json.loads(Path(__file__).with_name("ctx.json").read_text(encoding="utf-8"))
out = Path(arg("--output-dir")); out.mkdir(parents=True, exist_ok=True)
work = json.loads(Path(arg("--indices-file")).read_text(encoding="utf-8"))
target = int(arg("--accepted-target")); job = out.name
ctrl = Path(ctx["control"])
accepted, code = target, 0
if (ctrl / f"stop_{job}").exists():
    Path(arg("--stop-file")).write_text("")
    accepted, code = target - 1, 3
elif (ctrl / f"deadline_{job}").exists():
    accepted, code = target - 1, 3
(out / "assets").mkdir(exist_ok=True)
(out / "assets/rollin.onnx").write_bytes(Path(ctx["model"]).read_bytes())
y23 = [int(v) for v in arg("--region-quota-permille").split(",")]
mf = dict(accepted=accepted, search_n=int(arg("--search-n")), work_indices=work, git_commit=ctx["commit"],
          rollin="nn:asset:" + arg("--model-id"), asset_files=["assets/rollin.onnx"],
          premises=dict(use_ucb=False, radical_factor_max=1.4, ramen_region_strategy="all",
                        friend_complete_required=True, record_ordered_rollouts=True),
          sampler=dict(epsilon=ctx["epsilon"], seed_base=ctx["seed_base"], inherit=ctx["inherit"],
                       region_quota_permille_y1=int(arg("--region-quota-permille-y1")),
                       region_quota_permille=y23))
(out / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
sys.exit(code)
'''

FAKE_EXPORT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
mf = json.loads((Path(arg("--input")) / "manifest.json").read_text(encoding="utf-8"))
out = Path(arg("--output-dir")); out.mkdir(parents=True)
(out / "meta.json").write_text(json.dumps(dict(stats=dict(samples=mf["accepted"], rollout_width=mf["search_n"]))))
'''


def fake_exe(folder, name, code):
    """写一个可被 Popen 直接执行的假程序（Windows 用 .cmd 包一层 Python）。"""
    script = folder / f"{name}.py"
    script.write_text(code, encoding="utf-8")
    if os.name == "nt":
        exe = folder / f"{name}.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe = folder / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


def make_plan(tmp, rounds=2, excluded=None):
    """用独立枚举合成 dump（只测 Python 生成器，不代替真实 Rust dump），生成极小分轮清单。"""
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    plans = gen.enumerate_plans(recipe)
    names = [s["name"] for s in recipe["space"]["shapes"]]
    dump = tmp / "gen2_v1.txt"
    dump.write_text("".join(f"GEN2PLAN {p['plan']} {p['uma']} {','.join(map(str, p['deck']))} {names[p['shape']]}\n"
                            for p in plans), encoding="utf-8")
    out = tmp / "plan"
    # 号段取远离一切登记的 [9e10, 9.01e10)；生成器照常扫描工作区历史 manifest
    gen.prepare(dump, out, recipe_path=RECIPE, recipe_id="test_rounds", model="game_config.toml",
                model_id="fake_model", search_n=8, layer_targets=[[2, 1, 1, 1]] * 4,
                index_start=90_000_000_000, index_end=90_100_000_000, seconds=3600,
                friend_gate=True, rounds=rounds, excluded=excluded)
    return out, plans


class RoundPlanTests(unittest.TestCase):
    """清单生成与校验。"""

    def test_rounds_and_exclusions(self):
        """分轮任务按轮排列、每轮配额相同；额外排除写入并被校验，篡改即拒绝。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="round_plan_") as tmp:
            tmp = Path(tmp)
            recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
            some = gen.enumerate_plans(recipe)[:3]
            excluded = {tuple(p["fields"]): {"src.json"} for p in some}
            excluded[(1, 2, 3, 4, 5, 6, 7)] = {"other_space.json"}  # 不在本空间：忽略
            out, plans = make_plan(tmp, excluded=excluded)
            plan = formal.read_json(out / "manifest.json")
            rows = formal.read_json(out / "exclusions.json")["plans"]
            print(f"轮数={plan['rounds']} 每轮={plan['round_target']} 任务={len(plan['jobs'])} 排除={len(rows)}")
            self.assertEqual((plan["rounds"], plan["round_target"], len(plan["jobs"])), (2, 20, 32))
            self.assertEqual(sorted(r["plan"] for r in rows), [p["plan"] for p in some])
            self.assertEqual(formal.check_plan(out, plan, 8, 40), 80)
            self.assertEqual([j["name"] for j in plan["jobs"][:2]], ["r01_region_y3_s1", "r01_region_y3_s2"])
            bad = copy.deepcopy(plan)
            bad["jobs"][0], bad["jobs"][-1] = bad["jobs"][-1], bad["jobs"][0]
            with self.assertRaises(ValueError):
                formal.check_plan(out, bad, 8, 40)
            (out / "exclusions.json").write_text(json.dumps(dict(plans=rows[:2])), encoding="utf-8")
            with self.assertRaises(ValueError):
                formal.check_plan(out, plan, 8, 40)
            # 把一个已采计划改写进排除：清单命中排除组合必须拒绝
            first = formal.read_json(out / plan["jobs"][0]["indices"])[0] % len(plans)
            rows.append(dict(plan=first, fields=plans[first]["fields"], sources=["x"]))
            (out / "exclusions.json").write_text(json.dumps(dict(plans=rows)), encoding="utf-8")
            plan["excluded_count"] = len(rows)
            with self.assertRaises(ValueError):
                formal.check_plan(out, plan, 8, 40)


class RoundDriverTests(unittest.TestCase):
    """驱动：停止文件安全停止与续跑、分轮截止的正常收尾。"""

    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="round_driver_")
        self.tmp = Path(self.tmp_ctx.name)
        self.plan_dir, _ = make_plan(self.tmp)
        self.plan = formal.read_json(self.plan_dir / "manifest.json")
        self.ctrl = self.tmp / "control"
        self.ctrl.mkdir()
        bins = self.tmp / "bin"
        bins.mkdir()
        (bins / "ctx.json").write_text(json.dumps(dict(
            control=str(self.ctrl), model=str(ROOT / "game_config.toml"), commit=FAKE_COMMIT,
            epsilon=self.plan["epsilon"], seed_base=self.plan["seed_base"], inherit=self.plan["inherit"])),
            encoding="utf-8")
        self.exe = fake_exe(bins, "collect", FAKE_COLLECT)
        self.export = fake_exe(bins, "export", FAKE_EXPORT)
        self.ref = self.tmp / "ref"
        self.ref.mkdir()
        shutil.copy(ROOT / "game_config.toml", self.ref / "game_config.toml")
        (self.ref / "files.json").write_text(json.dumps(["game_config.toml"]), encoding="utf-8")
        self.out = self.tmp / "out"
        self.stop = self.tmp / "STOP"

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def run_driver(self, out=None, from_round=1):
        """以假提交、干净工作树运行驱动（只替换 git 查询）。"""
        args = SimpleNamespace(plan=self.plan_dir, expect_search_n=8, expect_target_valid=40, validate_only=False,
                               exe=self.exe, export_exe=self.export, output=out or self.out, threads=1,
                               asset_reference=self.ref, stop_file=self.stop, from_round=from_round)
        git = lambda cmd, **kw: FAKE_COMMIT if "rev-parse" in cmd else ""
        with mock.patch.object(formal.subprocess, "check_output", side_effect=git):
            return formal.run(args)

    def state(self):
        return formal.read_json(self.out / "run_state.json")

    def test_stop_file_and_resume(self):
        """任务间与根间两种停止都返回 3 且可续跑；续跑后全部完成。"""
        self.stop.write_text("")
        self.assertEqual(self.run_driver(), formal.EXIT_EARLY_STOP)
        self.assertEqual(self.state()["completed"], [])
        self.stop.unlink()
        third = self.plan["jobs"][2]["name"]
        (self.ctrl / f"stop_{third}").write_text("")
        self.assertEqual(self.run_driver(), formal.EXIT_EARLY_STOP)
        done = self.state()["completed"]
        print(f"根间停止前完成 {len(done)} 个任务：{done}")
        self.assertEqual(done, [j["name"] for j in self.plan["jobs"][:2]])
        (self.ctrl / f"stop_{third}").unlink()
        self.stop.unlink()
        self.assertEqual(self.run_driver(), 0)
        state = self.state()
        self.assertEqual(len(state["completed"]), len(self.plan["jobs"]))
        self.assertTrue(state.get("finished") and not state.get("truncated"))

    def test_deadline_truncates_rounds(self):
        """分轮清单在第二轮中途到达截止：已完成任务保留，未满任务另行导出，正常结束且不再续采。"""
        mid = self.plan["jobs"][28]["name"]  # 第二轮的 general_s1，目标 2 根，截止时已采 1 根
        (self.ctrl / f"deadline_{mid}").write_text("")
        self.assertEqual(self.run_driver(), 0)
        state = self.state()
        cut = state["truncated"]
        print(f"截断：完整 {cut['full_rounds']} 轮，完成任务 {cut['completed_jobs']}，未满 {cut['partial']}")
        self.assertEqual((cut["full_rounds"], cut["completed_jobs"]), (1, 28))
        self.assertEqual(cut["partial"]["job"], mid)
        self.assertTrue((ROOT / cut["partial"]["export"] / "meta.json").exists())
        (self.ctrl / f"deadline_{mid}").unlink()
        self.assertEqual(self.run_driver(), 0)
        self.assertEqual(len(self.state()["completed"]), 28)

    def test_from_round_new_output(self):
        """截断后用 --from-round 配新目录续采：只跑起点及以后各轮，旧目录不受影响；越界或改起点即拒绝。"""
        mid = self.plan["jobs"][20]["name"]  # 第二轮中途截断，模拟上一批
        (self.ctrl / f"deadline_{mid}").write_text("")
        self.assertEqual(self.run_driver(), 0)
        (self.ctrl / f"deadline_{mid}").unlink()
        old = self.state()
        out2 = self.tmp / "out2"
        self.assertEqual(self.run_driver(out=out2, from_round=2), 0)
        state = formal.read_json(out2 / "run_state.json")
        round2 = [j["name"] for j in self.plan["jobs"] if j["round"] == 2]
        ran = sorted(p.name for p in (out2 / "data").iterdir())
        print(f"续采目录完成 {len(state['completed'])} 个任务，数据目录 {len(ran)} 个，起点 {state['identity']['from_round']}")
        self.assertEqual(state["completed"], round2)
        self.assertEqual(ran, sorted(round2))
        self.assertTrue(state.get("finished") and not state.get("truncated"))
        self.assertEqual(formal.read_json(self.out / "run_state.json"), old)
        with self.assertRaises(ValueError):
            self.run_driver(out=out2, from_round=1)
        with self.assertRaises(ValueError):
            self.run_driver(out=self.tmp / "out3", from_round=3)

    def test_deadline_without_rounds_still_fails(self):
        """非分轮清单（R10 等）到达截止仍按失败处理，保持旧语义。"""
        plan = formal.read_json(self.plan_dir / "manifest.json")
        for key in ("rounds", "round_target"):
            plan.pop(key)
        (self.plan_dir / "manifest.json").write_text(json.dumps(plan), encoding="utf-8")
        (self.ctrl / f"deadline_{plan['jobs'][0]['name']}").write_text("")
        with self.assertRaises(TimeoutError):
            self.run_driver()


if __name__ == "__main__":
    unittest.main()
