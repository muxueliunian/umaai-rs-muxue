#!/usr/bin/env bash
# R8：8 万根 DAgger 采集的六份清单（roll-in = ens_R67mix60k_g123，三种子各 60k 步，每候选 1024，继承 gen1_inherit）。
# 配额按 codex 定稿（四层 50:1:3:3）：gen2 四构成 40000 + 2速1耐2智 16000 + 三个新空间各 8000
# （newboth 的 8000 中 1000 由 gen3_userdeck_v1 定额给实战卡组）。
# 每份清单只写全新目录；计划原文取自 sampler 单测的 GEN2PLAN 输出，与独立 Python 枚举逐字段比对。
set -e
DUMP=target/r8_plan_dumps_0920
MODEL=saved_models/arms/ens_R67mix60k_g123.onnx
ID=ens_R67mix60k_g123
mkdir -p $DUMP
dump() {  # $1=单测名 $2=输出文件
  cargo test --release --lib -j 8 -p umasim "$1" -- --exact --nocapture 2>/dev/null | grep "^GEN2PLAN" > "$2"
  echo "$1 → $(grep -c '^GEN2PLAN ' "$2") 行"
}
dump sampler::tests::test_gen2_v1_plan_dump         $DUMP/gen2_v1.txt
dump sampler::tests::test_gen2_2s1e2w_plan_dump     $DUMP/gen2_2s1e2w_v1.txt
dump sampler::tests::test_gen3_newuma_plan_dump     $DUMP/gen3_newuma_v1.txt
dump sampler::tests::test_gen3_newcard_plan_dump    $DUMP/gen3_newcard_v1.txt
dump sampler::tests::test_gen3_newboth_plan_dump    $DUMP/gen3_newboth_v1.txt
dump sampler::tests::test_gen3_userdeck_plan_dump   $DUMP/gen3_userdeck_v1.txt

prep() {  # $1=配方 $2=dump $3=输出目录 $4=配方名 $5=配额矩阵 $6=号段起 $7=号段止 $8=截止秒
  python scripts/collect/prepare_gen2_formal.py --rust-dump "$2" --output "$3" --recipe "$1" \
    --recipe-id "$4" --model $MODEL --model-id $ID --search-n 1024 \
    --shape-layer-targets "$5" --index-start "$6" --index-end "$7" --seconds "$8"
}
# gen2_v1 构成序：3速1耐1智 / 2速2耐1智 / 2速1耐1力1智 / 2速1力1根1智（2速1耐1力1智 加权到 13000）
prep scripts/collect/gen2_v1_recipe.json $DUMP/gen2_v1.txt scripts/collect/r8_gen2_0920 gen2_v1_r8dagger_0920 \
  "7900,160,470,470;7900,160,470,470;11400,220,690,690;7900,160,470,470" 3000000000 3400000000 36000
prep scripts/collect/gen2_2s1e2w_recipe.json $DUMP/gen2_2s1e2w_v1.txt scripts/collect/r8_2s1e2w_0920 gen2_2s1e2w_r8dagger_0920 \
  "14040,280,840,840" 3400000000 3420000000 18000
# 第三代构成序：3速1耐1智 / 2速2耐1智 / 2速1耐1力1智 / 2速1力1根1智 / 2速1耐2智（10/10/20/10/50%）
prep scripts/collect/gen3_newuma_recipe.json $DUMP/gen3_newuma_v1.txt scripts/collect/r8_newuma_0920 gen3_newuma_r8dagger_0920 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;3510,70,210,210" 3420000000 3440000000 10800
prep scripts/collect/gen3_newcard_recipe.json $DUMP/gen3_newcard_v1.txt scripts/collect/r8_newcard_0920 gen3_newcard_r8dagger_0920 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;3510,70,210,210" 3440000000 3480000000 10800
# newboth 的 2速1耐2智 扣掉实战卡组的 1000（880/20/50/50）
prep scripts/collect/gen3_newboth_recipe.json $DUMP/gen3_newboth_v1.txt scripts/collect/r8_newboth_0920 gen3_newboth_r8dagger_0920 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;2630,50,160,160" 3480000000 3490000000 10800
prep scripts/collect/gen3_userdeck_recipe.json $DUMP/gen3_userdeck_v1.txt scripts/collect/r8_userdeck_0920 gen3_userdeck_r8dagger_0920 \
  "880,20,50,50" 3490000000 3490100000 3600
echo R8_PREPARED
