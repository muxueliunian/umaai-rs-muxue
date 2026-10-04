"""R10 云端预检的拒绝与恢复测试；临时产物只放工作区 target，不删除历史数据。"""

import json
import time
import unittest
from unittest.mock import patch

import prepare_r10_cloud as cloud


class R10CloudTests(unittest.TestCase):
    """用实际字段及字节测试号段冲突、精确恢复和资产不可覆盖。"""

    def setUp(self):
        """建立工作区内独立测试目录，不读取用户云端数据。"""
        self.root = cloud.workspace_path(cloud.ROOT / "target" / f"r10_cloud_test_{time.time_ns()}")
        self.root.mkdir(parents=True)
        self.root_patch = patch.object(cloud, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.history = self.root / "training_data"
        self.history.mkdir()
        self.plans = [("brian", {"space": {"version": "gen4_brian_v1"},
                                 "index_reservation": [100, 200]}, {"general_s1": [101, 105]})]

    def put_manifest(self, relative, data):
        """写入隔离的历史清单，便于测试错误配置。"""
        path = self.history / relative / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def resume_manifest(self):
        """返回同提交、同任务的精确恢复信息。"""
        return dict(work_indices=[101, 105], git_commit="test-commit",
                    space={"version": "gen4_brian_v1"}, search_n=1024,
                    rollin="nn:asset:ens_R8A_g123")

    def test_history_collision_and_exact_resume(self):
        """同目录完整任务可恢复；重排或子集序号必须拒绝。"""
        path = self.put_manifest("r10_brian_1004/data/general_s1", self.resume_manifest())
        cloud.check_history([self.history], self.plans, "test-commit")
        for indices in ([105, 101], [101]):
            data = self.resume_manifest()
            data["work_indices"] = indices
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                cloud.check_history([self.history], self.plans, "test-commit")
        print("精确原目录恢复通过，重排和子集恢复已拒绝")

    def test_duplicate_output_and_old_ranges_rejected(self):
        """同号段换目录重复开跑和旧连续区间碰撞均拒绝。"""
        path = self.put_manifest("different_output/data/general_s1", self.resume_manifest())
        with self.assertRaises(ValueError):
            cloud.check_history([self.history], self.plans, "test-commit")
        path.write_text(json.dumps({"index_start": 150, "index_end": 210}), encoding="utf-8")
        with self.assertRaises(ValueError):
            cloud.check_history([self.history], self.plans, "test-commit")
        print("重复输出目录和历史连续号段碰撞已拒绝")

    def test_assets_are_frozen_by_bytes(self):
        """模型原文独立比对，来源或内容变化不能覆盖旧参考。"""
        release = self.root / "release"
        release.mkdir()
        for relative in cloud.ASSETS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original-asset")
        for relative in (cloud.MODEL, cloud.MODEL + ".json"):
            (release / cloud.Path(relative).name).write_bytes(b"original-asset")
        reference = self.root / "reference"
        cloud.freeze_assets(release, reference, "test-commit")
        cloud.freeze_assets(release, reference, "test-commit")
        (self.root / "gamedata/constants.json").write_bytes(b"changed--asset")
        with self.assertRaises(ValueError):
            cloud.freeze_assets(release, reference, "test-commit")
        self.assertEqual((reference / "gamedata/constants.json").read_bytes(), b"original-asset")
        with self.assertRaises(ValueError):
            cloud.freeze_assets(release, reference, "other-commit")
        print("资产内容和提交变化已拒绝，旧参考原文保留")

    def test_outside_workspace_rejected(self):
        """越界路径在任何读写前被拒绝。"""
        with self.assertRaises(ValueError):
            cloud.workspace_path(self.root.parent / "outside")
        print("工作区外路径已拒绝")


if __name__ == "__main__":
    unittest.main()
