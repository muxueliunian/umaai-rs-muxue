#!/bin/bash
# 容器重启后续跑 B 组（新旧各自 --resume）
export RAYON_NUM_THREADS=4
R=<WORK>/results; L=<WORK>/logs
COMMON="--space-version gen2_v1 --seed 61444 --no-fingerprint"
for d in new old; do
  cd <ROOT>/$d
  python3 <WORK>/timed.py . ./target/release/ramen_space_bench $COMMON \
    --plans-file <WORK>/holdout_stride16.json \
    --trainer search --search-n 1024 --radical-factor-max 1.4 --runs-per-plan 1 --run-offset 402000 \
    --csv $R/B_$d.csv --resume >> $L/B_${d}_resume.log 2>&1
done
echo ALLDONE > $L/done.flag
