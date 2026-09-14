"""按固定源码准备新资产参考：旧模型/游戏数据库必须逐字节一致，仅允许配置更新。"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "saved_models/arms/ens_R4_g123.onnx"
FILES = [MODEL, MODEL + ".json", "gamedata/constants.json", "gamedata/events.json",
         "gamedata/umaDB.json", "gamedata/cardDB.json", "gamedata/scenario_ramen.json",
         "gamedata/default_config.toml", "game_config.toml"]
CONFIGS = {"gamedata/default_config.toml", "game_config.toml"}


def prepare(reference, output):
    """先检查所有来源再写新目录；文本统一LF，模型及旁车保持原始字节。"""
    if output.exists():
        raise FileExistsError(output)
    data = {}
    for name in FILES:
        current = (ROOT / name).read_bytes()
        old = (reference / name).read_bytes()
        if name not in CONFIGS:
            # git在Windows可能转换游戏JSON换行；只接受换行差异，不接受资产字段改变。
            actual = current if name.startswith("saved_models/") else current.replace(b"\r\n", b"\n")
            expected = old if name.startswith("saved_models/") else old.replace(b"\r\n", b"\n")
            if actual != expected:
                raise ValueError(f"旧基座发生非预期变化：{name}")
        data[name] = current if name.startswith("saved_models/") else current.replace(b"\r\n", b"\n")
    for name, contents in data.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (output / "files.json").write_text(json.dumps(FILES, indent=2) + "\n", encoding="utf-8")
    print("新参考资产已准备；仅两份配置允许更新，旧模型及旁车原始字节一致。")
    print("正式运行前，运行目录文件必须与参考逐字节一致；Windows换行不同会被驱动拒绝。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="上一批已冻结的完整原文资产参考")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference.resolve(), args.output.resolve())
