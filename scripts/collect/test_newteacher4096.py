"""4096配额、无截止与失败恢复测试；仅在工作区target使用假采集器。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_formal_collect as formal
import prepare_collect_assets as assets

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "scripts/collect/formal4096_0914"


class NewTeacherTests(unittest.TestCase):
    """配方校验与驱动状态机测试，不执行真实搜索。"""

    def test_recipe(self):
        """新旧配方都验证，预算/年份/号段/模型篡改均拒绝。"""
        plan = formal.read_json(PLAN_DIR / "manifest.json")
        self.assertEqual(formal.check_plan(PLAN_DIR, plan), 100000)
        for key, value in (("search_n", 2048), ("seconds", 43200),
                           ("target_valid", 22800), ("model_id", "wrong"),
                           ("index_reservation", [20000000, 800000000])):
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError, msg=key):
                formal.check_plan(PLAN_DIR, bad)
        bad = copy.deepcopy(plan)
        bad["jobs"][4]["quota_y1"] = 0
        with self.assertRaises(ValueError):
            formal.check_plan(PLAN_DIR, bad)

    def test_commands(self):
        """无截止不传max-seconds，旧批仍传剩余秒数，新批强制教师参数检查。"""
        plan = formal.read_json(PLAN_DIR / "manifest.json")
        command = formal.collect_command(Path("collector"), PLAN_DIR, plan["jobs"][0],
                                         plan, Path("data"), Path("model"), None)
        self.assertNotIn("--max-seconds", command)
        self.assertIn("--require-newteacher-defaults", command)
        self.assertEqual(command[command.index("--search-n") + 1], "4096")
        command = formal.collect_command(Path("collector"), PLAN_DIR, plan["jobs"][0],
                                         plan, Path("data"), Path("model"), 12)
        self.assertEqual(command[command.index("--max-seconds") + 1], "12")

    def test_process_without_deadline(self):
        """无截止支持成功退出和非零退出；PID只用于本驱动拥有的进程。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="t4096_proc_") as tmp:
            formal.execute([sys.executable, "-c", "print('no-deadline')"], Path(tmp) / "ok", None, 1)
            with self.assertRaises(RuntimeError):
                formal.execute([sys.executable, "-c", "raise SystemExit(7)"], Path(tmp) / "fail", None, 1)

    def test_resume_export_and_identity(self):
        """采完导出失败后只重试导出；重复启动无重复采集；改身份拒绝。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="t4096_resume_") as tmp:
            root = Path(tmp)
            plan_dir = root / "plan"
            plan_dir.mkdir()
            original = formal.read_json(PLAN_DIR / "manifest.json")
            plan = copy.deepcopy(original)
            job = copy.deepcopy(plan["jobs"][0])
            job.update(target=2, count=4, indices="indices.json")
            plan.update(jobs=[job], target_valid=2, model="model.onnx")
            formal.write_json(plan_dir / "manifest.json", plan)
            formal.write_json(plan_dir / "indices.json", [300001370, 300005658, 300009946, 300014234])
            (root / "model.onnx").write_bytes(b"fixture-model")
            reference = root / "reference"
            reference.mkdir()
            (reference / "model.onnx").write_bytes(b"fixture-model")
            formal.write_json(reference / "files.json", ["model.onnx"])
            for name in ("collector", "exporter"):
                (root / name).write_bytes(b"fixture")
            args = SimpleNamespace(plan=plan_dir, validate_only=False, reservation_confirmed=True,
                                   exe=root / "collector", export_exe=root / "exporter",
                                   output=root / "out", threads=1, asset_reference=reference, min_free_gib=0.001)
            calls = []
            export_fails = [True]

            def fake_execute(command, evidence, deadline, threads, min_free_bytes=0):
                self.assertIsNone(deadline)
                calls.append(Path(command[0]).name)
                output = Path(command[command.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                if calls[-1] == "exporter":
                    if export_fails[0]:
                        export_fails[0] = False
                        raise RuntimeError("fixture export failure")
                    formal.write_json(output / "meta.json", {"stats": {"samples": 2, "rollout_width": 4096}})
                    return
                (output / "assets").mkdir()
                (output / "assets/rollin.onnx").write_bytes(b"fixture-model")
                formal.write_json(output / "manifest.json", dict(
                    accepted=2, search_n=4096, git_commit="fixture-commit",
                    work_indices=formal.read_json(plan_dir / "indices.json"),
                    rollin="nn:asset:" + plan["model_id"], asset_files=["assets/rollin.onnx"],
                    premises=dict(use_ucb=False, radical_factor_max=1.4,
                                  ramen_region_strategy="all", record_ordered_rollouts=True),
                    sampler=dict(epsilon=.15, seed_base=plan["seed_base"], inherit=plan["inherit"],
                                 region_quota_permille_y1=0, region_quota_permille=[0, 0])))

            def fake_git(command, **kwargs):
                return "fixture-commit" if "rev-parse" in command else ""

            with patch.object(assets, "FILES", ["model.onnx"]), \
                    patch.object(formal, "ROOT", root), patch.object(formal, "check_plan", return_value=4), \
                    patch.object(formal, "verify_export"), \
                    patch.object(formal, "execute", side_effect=fake_execute), \
                    patch.object(formal.subprocess, "check_output", side_effect=fake_git):
                with self.assertRaises(RuntimeError):
                    formal.run(args)
                formal.run(args)
                formal.run(args)
                self.assertEqual(calls, ["collector", "exporter", "exporter"])
                args.threads = 2
                with self.assertRaises(ValueError):
                    formal.run(args)
            self.assertTrue((root / "out/run_state.json").is_file())


if __name__ == "__main__":
    unittest.main()
