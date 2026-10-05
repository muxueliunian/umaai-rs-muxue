# R9 新20k准备记录

仅准备并validate-only通过，尚未开始。必须等待当前10k整批Google Drive上传及完整核验通过，再由主worker启动。
工作目录：<ROOT>（<WORK>/repo）；所有命令路径相对<ROOT>。
新号段：[5100000000,5300000000)；40000候选；有效根20000；实际index最小5100001434、最大5271519801。
沿用4288计划、250留出、ens_R8A_g123、friend_complete_required=true、1024搜索、8线程与36000秒。
配额按当前10k每cell×2，保持其平衡取整结果，不重新取整。general/y1/y2/y3总量17550/350/1050/1050；四构成4500/4500/6500/4500。
已检查18份正式清单、34份实际manifest及登记/recipe预留段，交集为0；plans和holdout由cmp证实字节一致。
除recipe_id、目标、号段、indices及job target/count外，其余配方字段逐字段一致。
原候选[5100000000,5200000000)宽度不足，依据生成器公式预检改为2亿宽；未触发失败生成。

生成命令（已运行；目录已存在，不得直接重跑）：
python3 scripts/collect/prepare_gen2_formal.py --rust-dump target/r9_plan_dumps_1002/gen2_v1.txt --output scripts/collect/r9_gen2_1004_extra20k --recipe scripts/collect/gen2_v1_recipe.json --recipe-id gen2_v1_r9dagger_1004_extra20k --model saved_models/arms/ens_R8A_g123.onnx --model-id ens_R8A_g123 --search-n 1024 --friend-gate --shape-layer-targets '3950,80,236,234;3950,80,234,236;5700,110,346,344;3950,80,234,236' --index-start 5100000000 --index-end 5300000000 --seconds 36000

校验命令（已运行通过）：
python3 scripts/collect/run_formal_collect.py --plan scripts/collect/r9_gen2_1004_extra20k --expect-search-n 1024 --expect-target-valid 20000 --validate-only

后续启动命令（未运行；前置条件通过后运行）：
python3 scripts/collect/run_formal_collect.py --plan scripts/collect/r9_gen2_1004_extra20k --expect-search-n 1024 --expect-target-valid 20000 --exe target/release/ramen_teacher_collect --export-exe target/release/ramen_export_npy --output training_data/r9_gen2_1004_extra20k --asset-reference ../assets_r9 --threads 8

详细审计：range_audit.json；原字节快照与生成/校验stdout：<WORK>/cloud_results/r9_collect_1002/next20k_prepare/
未修改tracked文件、未手工改manifest/state、未编译、未计算指纹、未上传、未删除。
