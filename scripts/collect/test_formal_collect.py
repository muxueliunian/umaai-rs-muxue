"""显式配方驱动与完整字段读取的针对性测试；不启动采集或训练。"""

import copy
import importlib.util
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from ramen_nn.data import split_refs_by_combos, stable_split_refs

spec = importlib.util.spec_from_file_location("formal", Path(__file__).with_name("run_formal_collect.py"))
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


class FormalTests(unittest.TestCase):
    """只操作 target 下独立临时目录及假进程。"""

    def test_plan_and_duplicate(self):
        """正式计划总量和重复任务保护。"""
        directory = ROOT / "scripts/collect/formal2048_0914"
        plan = formal.read_json(directory / "manifest.json")
        self.assertEqual(formal.check_plan(directory, plan), 45600)
        bad = copy.deepcopy(plan)
        bad["jobs"][1] = bad["jobs"][0]
        with self.assertRaises(ValueError):
            formal.check_plan(directory, bad)

    def test_full_fields_split(self):
        """相同完整卡组跨目录仍落同一侧，且不回落到旧哈希切分。"""
        a, b = [1, 2, 3, 4, 5, 6, 7], [8, 2, 3, 4, 5, 6, 7]
        shards = [SimpleNamespace(combo_fields=np.array([a, b])), SimpleNamespace(combo_fields=np.array([a]))]
        train, valid = split_refs_by_combos(shards, [a])
        self.assertEqual(train.tolist(), [[0, 1]])
        self.assertEqual(valid.tolist(), [[0, 0], [1, 0]])
        with self.assertRaises(ValueError):
            stable_split_refs(shards, .1, 1)
        with self.assertRaises(ValueError):
            split_refs_by_combos([SimpleNamespace(combo_fields=None)], [a])

    def test_terminate_only_owned_process(self):
        """相同程序名的对照进程应存活，终止仅针对指定进程树。"""
        flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else dict(start_new_session=True)
        a = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], **flags)
        b = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], **flags)
        try:
            formal.terminate(a)
            self.assertIsNotNone(a.poll())
            self.assertIsNone(b.poll())
        finally:
            formal.terminate(a)
            formal.terminate(b)

    def test_streaming_logs_and_failure(self):
        """子进程仍存活时可读早期日志；失败退出码会保存并向上传播。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="formal_fake_") as tmp:
            root = Path(tmp)
            errors = []

            def task():
                try:
                    formal.execute([sys.executable, "-c", "import time; print('early',flush=True); time.sleep(2); print('late')"],
                                   root / "live", time.time() + 10, 1)
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=task)
            thread.start()
            output = root / "live/stdout.txt"
            for _ in range(30):
                if output.exists() and b"early" in output.read_bytes():
                    break
                time.sleep(.05)
            self.assertIn(b"early", output.read_bytes())
            self.assertNotIn(b"late", output.read_bytes())
            self.assertFalse((root / "live/exitcode.txt").exists())
            thread.join(8)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            with self.assertRaises(RuntimeError):
                formal.execute([sys.executable, "-c", "raise SystemExit(3)"], root / "fail", time.time() + 10, 1)
            self.assertEqual((root / "fail/exitcode.txt").read_text(), "3")


if __name__ == "__main__":
    unittest.main()
