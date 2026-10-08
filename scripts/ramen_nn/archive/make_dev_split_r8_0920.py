"""R8 轮的开发验证组合清单：沿用 R67 那份 446 个，另并入三个第三代空间的闭环留出。

并入留出组合是**安全网**而不是新增验证样本：这三个空间的留出在采集时就已排除，
语料里本来就没有它们的根，所以并进来不会改变验证集构成（跨臂、跨轮仍可比），
只是万一以后有哪批旧数据碰巧含到，也会被划到验证侧而不是训练侧。

第三代空间的 `userdeck` 只有一个组合、配方里留出数为 0（1000 根全部入训练），
**不并入**——那是刻意要学的实战配置，不是盲测面板。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'target/train_r67_0919/dev_validation_combos.json'
OUT = ROOT / 'target/train_r8_0920/dev_validation_combos.json'
NEW_SPACES = ['r8_newuma_0920', 'r8_newcard_0920', 'r8_newboth_0920']

base = json.loads(BASE.read_text(encoding='utf-8'))
combos = [tuple(int(v) for v in row) for row in base['combos']]
seen = set(combos)
assert len(seen) == len(combos), '基础清单自身有重复'

added = {}
for space in NEW_SPACES:
    hold = json.loads((ROOT / 'scripts/collect' / space / 'holdout.json').read_text(encoding='utf-8'))
    rows = [tuple(int(v) for v in q['fields']) for q in hold['plans']]
    assert len(set(rows)) == len(rows), f'{space} 留出清单自身有重复'
    dup = [r for r in rows if r in seen]
    assert not dup, f'{space} 有 {len(dup)} 条与已有清单重复：{dup[:3]}'
    combos.extend(rows)
    seen.update(rows)
    added[space] = len(rows)
    print(f'并入 {space} 留出 {len(rows)} 条')

payload = {
    'method': 'R67 那份 446 条原样保留，另并入三个第三代空间的闭环留出组合（安全网，语料里本无这些根）',
    'base': {'path': 'target/train_r67_0919/dev_validation_combos.json', 'combos': len(base['combos'])},
    'base_method': base.get('method'),
    'added_gen3_holdout': added,
    'not_added': 'r8_userdeck_0920：留出数为 0，那一个组合的 1000 根刻意全部入训练，不是盲测面板',
    'validation_combos': len(combos),
    'combos': [list(row) for row in combos],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'写出 {OUT.relative_to(ROOT)}：{len(combos)} 条（{len(base["combos"])} + {sum(added.values())}）')
