#!/usr/bin/env bash
# R9：R8 配额原样复刻（八万根 @1024）+ gen2 高质量批（一万二千根 @4096），全部带友人完成门限。
# 与 R8 的差异只有三处：roll-in 换 ens_R8A_g123、教师 rollout 开「友人出行 5 次必须走完」门限、新号段。
# 卡池、构成、四层配额、留出、seed_base 全与 R8 相同 ⇒ 新旧两批可以逐层对照。
# 低内存机器：编译走 thin LTO、-j 2（默认 fat LTO 链接峰值约 13.5 GB）。
set -e
export CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_DEBUG=false
PY=${PYTHON:-python3}
DUMP=target/r9_plan_dumps_1002
MODEL=saved_models/arms/ens_R8A_g123.onnx
ID=ens_R8A_g123
mkdir -p $DUMP
dump() {  # $1=单测名 $2=输出文件
  cargo test --release --lib -j 2 -p umasim "$1" -- --exact --nocapture 2>/dev/null | grep "^GEN2PLAN" > "$2"
  echo "$1 → $(grep -c '^GEN2PLAN ' "$2") 行"
}
dump sampler::tests::test_gen2_v1_plan_dump         $DUMP/gen2_v1.txt
dump sampler::tests::test_gen2_2s1e2w_plan_dump     $DUMP/gen2_2s1e2w_v1.txt
dump sampler::tests::test_gen3_newuma_plan_dump     $DUMP/gen3_newuma_v1.txt
dump sampler::tests::test_gen3_newcard_plan_dump    $DUMP/gen3_newcard_v1.txt
dump sampler::tests::test_gen3_newboth_plan_dump    $DUMP/gen3_newboth_v1.txt
dump sampler::tests::test_gen3_userdeck_plan_dump   $DUMP/gen3_userdeck_v1.txt

prep() {  # $1=配方 $2=dump $3=输出目录 $4=配方名 $5=配额矩阵 $6=号段起 $7=号段止 $8=截止秒 $9=search_n
  $PY scripts/collect/prepare_gen2_formal.py --rust-dump "$2" --output "$3" --recipe "$1" \
    --recipe-id "$4" --model $MODEL --model-id $ID --search-n "$9" --friend-gate \
    --shape-layer-targets "$5" --index-start "$6" --index-end "$7" --seconds "$8"
}
# —— 高质量批：gen2_v1，按 R8 gen2 配额的 0.3 倍，@4096（构成序同 R8）——
prep scripts/collect/gen2_v1_recipe.json $DUMP/gen2_v1.txt scripts/collect/r9_gen2n4096_1002 gen2_v1_r9n4096_1002 \
  "2370,48,141,141;2370,48,141,141;3420,66,207,207;2370,48,141,141" 4500000000 4650000000 54000 4096
# —— 八万根 @1024：配额与 R8 逐项相同，号段 = R8 号段 + 1e9 ——
prep scripts/collect/gen2_v1_recipe.json $DUMP/gen2_v1.txt scripts/collect/r9_gen2_1002 gen2_v1_r9dagger_1002 \
  "7900,160,470,470;7900,160,470,470;11400,220,690,690;7900,160,470,470" 4000000000 4400000000 36000 1024
prep scripts/collect/gen2_2s1e2w_recipe.json $DUMP/gen2_2s1e2w_v1.txt scripts/collect/r9_2s1e2w_1002 gen2_2s1e2w_r9dagger_1002 \
  "14040,280,840,840" 4400000000 4420000000 18000 1024
prep scripts/collect/gen3_newuma_recipe.json $DUMP/gen3_newuma_v1.txt scripts/collect/r9_newuma_1002 gen3_newuma_r9dagger_1002 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;3510,70,210,210" 4420000000 4440000000 10800 1024
prep scripts/collect/gen3_newcard_recipe.json $DUMP/gen3_newcard_v1.txt scripts/collect/r9_newcard_1002 gen3_newcard_r9dagger_1002 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;3510,70,210,210" 4440000000 4480000000 10800 1024
prep scripts/collect/gen3_newboth_recipe.json $DUMP/gen3_newboth_v1.txt scripts/collect/r9_newboth_1002 gen3_newboth_r9dagger_1002 \
  "702,14,42,42;702,14,42,42;1404,28,84,84;702,14,42,42;2630,50,160,160" 4480000000 4490000000 10800 1024
prep scripts/collect/gen3_userdeck_recipe.json $DUMP/gen3_userdeck_v1.txt scripts/collect/r9_userdeck_1002 gen3_userdeck_r9dagger_1002 \
  "880,20,50,50" 4490000000 4490100000 3600 1024
echo R9_PREPARED
