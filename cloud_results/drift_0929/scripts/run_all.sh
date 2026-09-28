#!/bin/bash
# 依次跑 A(手写) 与 B(教师)，新旧各用自己 worktree 的二进制与 cwd
export RAYON_NUM_THREADS=4
R=<WORK>/results; L=<WORK>/logs
COMMON="--space-version gen2_v1 --seed 61444 --no-fingerprint"
for d in new old; do
  cd <ROOT>/$d
  python3 <WORK>/timed.py . ./target/release/ramen_space_bench $COMMON \
    --plans-file scripts/collect/r8_gen2_0920/holdout.json \
    --trainer handwritten --runs-per-plan 4 --run-offset 400000 \
    --csv $R/A_$d.csv --resume > $L/A_$d.log 2>&1
done
for d in new old; do
  cd <ROOT>/$d
  python3 <WORK>/timed.py . ./target/release/ramen_space_bench $COMMON \
    --plans-file <WORK>/holdout_stride16.json \
    --trainer search --search-n 1024 --radical-factor-max 1.4 --runs-per-plan 1 --run-offset 402000 \
    --csv $R/B_$d.csv --resume > $L/B_$d.log 2>&1
done
echo ALLDONE > $L/done.flag
