"""R10 配额、空间成员及留出边界测试；不编译、不伪造 Rust dump、不采集。"""

import unittest

from prepare_r10_1004 import BATCHES, COLLECT, check_dump, check_recipes, read_recipe


def held_fields(recipe, plans):
    """按既有生成器的完整字段排序规则计算留出组合。"""
    held = set()
    all_unseen = recipe.get("holdout_rule") == "every_tenth_all"
    elsewhere = recipe.get("sampled_elsewhere_fields", [])
    for uma in recipe["space"]["umas"]:
        for shape in range(len(recipe["space"]["shapes"])):
            cell = sorted((plan for plan in plans
                           if plan["uma"] == uma and plan["shape"] == shape
                           and plan["fields"] not in elsewhere
                           and (all_unseen or uma == 114101
                                or any(card in plan["deck"] for card in (303124, 303114)))),
                          key=lambda plan: plan["fields"])
            held.update(tuple(plan["fields"]) for plan in cell[9::10])
    return held


class R10PrepareTests(unittest.TestCase):
    """只读枚举及输入拒绝测试，使用完整字段判断身份。"""

    @classmethod
    def setUpClass(cls):
        """共享独立枚举结果，减少重复读取卡表。"""
        cls.plans = check_recipes()

    def test_counts_and_targets(self):
        """检查实际枚举总数与每种构成的研究配额。"""
        expected = {"brian": (4606, [600, 600, 1200, 600, 3000]),
                    "admire": (4606, [600, 600, 1200, 600, 3000]),
                    "dualwis": (616, [3000]),
                    "newuma": (3560, [1200, 1200, 2400, 1200, 6000]),
                    "control": (4288, [750, 750, 750, 750])}
        for name, (count, targets) in expected.items():
            with self.subTest(batch=name):
                self.assertEqual(len(self.plans[name]), count)
                self.assertEqual(list(map(sum, BATCHES[name]["targets"])), targets)
                print(f"{name}: plans={count}, targets={targets}")

    def test_members_and_compositions(self):
        """确保单新智、双新智、新马空间成员符合独立配额含义。"""
        old_umas = set(read_recipe(BATCHES["control"])["space"]["umas"]) | {106402}
        shapes = [[3, 1, 0, 0, 1], [2, 2, 0, 0, 1], [2, 1, 1, 0, 1],
                  [2, 0, 1, 1, 1], [2, 1, 0, 0, 2]]
        inherit = read_recipe(BATCHES["control"])["inherit"]
        for name, plans in self.plans.items():
            recipe = read_recipe(BATCHES[name])
            actual_shapes = [shape["counts"] for shape in recipe["space"]["shapes"]]
            expected_shapes = shapes[-1:] if name == "dualwis" else shapes
            if name == "control":
                expected_shapes = shapes[:4]
            self.assertEqual(actual_shapes, expected_shapes)
            self.assertEqual(recipe["inherit"]["blue_count"], inherit["blue_count"])
            self.assertEqual(recipe["inherit"]["extra_count"], inherit["extra_count"])
            expected_umas = {101703, 101803} if name == "newuma" else old_umas
            if name == "control":
                expected_umas = old_umas - {106402}
            self.assertEqual({plan["uma"] for plan in plans}, expected_umas)
            for plan in plans:
                deck = plan["deck"]
                self.assertFalse({303184, 303214}.intersection(deck))
                if name == "brian":
                    self.assertIn(303194, deck)
                    self.assertNotIn(303204, deck)
                elif name == "admire":
                    self.assertIn(303204, deck)
                    self.assertNotIn(303194, deck)
                elif name == "dualwis":
                    self.assertTrue({303194, 303204}.issubset(deck))
                    self.assertEqual(plan["shape"], 0)
        print("四个新空间成员、必带卡与双智构成检查通过")

    def test_disjoint_holdouts(self):
        """完整组合跨空间互斥，任何空间的训练候选不能落入另一空间留出。"""
        training = set()
        held = set()
        for name, plans in self.plans.items():
            recipe = read_recipe(BATCHES[name])
            fields = {tuple(plan["fields"]) for plan in plans}
            own_held = held_fields(recipe, plans)
            self.assertTrue(own_held)
            self.assertTrue(own_held.issubset(fields))
            self.assertFalse(held.intersection(fields - own_held))
            self.assertFalse(training.intersection(own_held))
            training.update(fields - own_held)
            held.update(own_held)
            if name != "control":
                self.assertEqual(recipe["holdout_rule"], "every_tenth_all")
                self.assertEqual(recipe["purchase_buffs"], [])
            else:
                self.assertEqual(recipe.get("holdout_rule", "unseen_new_cards"), "unseen_new_cards")
            print(f"{name}: holdout={len(own_held)}, training_combos={len(fields - own_held)}")
        self.assertFalse(training.intersection(held))

    def test_missing_dump_rejected(self):
        """缺少真实 Rust dump 时必须失败，不用 Python 枚举伪造通过记录。"""
        path = COLLECT / "missing_r10_dump_for_test.txt"
        if path.exists():
            self.skipTest("缺失文件测试路径已存在")
        with self.assertRaises(FileNotFoundError):
            check_dump(path, self.plans["brian"], read_recipe(BATCHES["brian"]))
        print("缺少真实 Rust dump 时已拒绝准备")

    def test_non_dump_rejected(self):
        """普通配方文件不可冒充 Rust dump，逐字段校验不得自动退回 Python 数据。"""
        batch = BATCHES["brian"]
        with self.assertRaises(ValueError):
            check_dump(COLLECT / batch["recipe"], self.plans["brian"], read_recipe(batch))
        print("非 Rust dump 输入已拒绝")


if __name__ == "__main__":
    unittest.main()
