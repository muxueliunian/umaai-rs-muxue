"""准备 R11d：沿用 R11 规则、配方、每轮配额与排除口径，换新号段再冻结 40 轮，按 22 小时截止采集。

R11 的 40 轮清单已由 R11、R11b、R11c 三批采完；本批只换批次名与独占号段，世界与 R11 全部不同。
不设冒烟段：采集代码与 R11 相同，云端沿用已验证的构建。用法同 `prepare_r11_1005.py`：
  python scripts/collect/prepare_r11d_1007.py --dump-dir <含 gen2_v1.txt 的目录>       --source-root <历史工作区> --history-root <历史工作区>/training_data       --history-root <历史工作区>/scripts/collect --history-root scripts/collect
"""

from prepare_r11_1005 import main

BATCH = "r11d_gen2n256_1007"
RESERVATION = [13000000000, 17000000000]

if __name__ == "__main__":
    main(batch=BATCH, reservation=RESERVATION, smoke_local=None, smoke_cloud=None)
