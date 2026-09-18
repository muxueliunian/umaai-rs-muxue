# gen2_v1 正式采集

当前有效配方是 **`formal1024_0918/`（R6，2026-09-18）**。
`formal2048_0914/` 是上一轮已完成的历史清单，`gen2_v1_recipe.json` 与
`price0914_runs_*.json` 是更早的 512 定价资料——**两者都不得当成当前运行清单**。

## R6（当前）

目标 **68400 有效根**：general 每构成 15000（共 60000），地区 1200 / 3600 / 3600
（第 1/2/3 年，每年再按四构成均分）。云端单机承担，**`ens_NT4096_AllHistory_g123`** roll-in，
内部手写终局续跑，**每候选 1024**，uniform，rf 1.4，地区 All，分片 32。
留出的 250 个组合与 0914 **是同一批**，泛化面板保持可比。
截止 72000 秒（20 小时）；依据是 0914 那轮 22800 根 @2048 在 12 小时内完成，
本轮 1.5 倍工作量 ≤ 18 小时，留 2 小时余量。

❗`search_n` 从 2048 降到 1024 的依据是 `logs/gen2_budget_0914/report.md`：
2048−1024 的学生闭环配对差 +19.4 [−40.4, +79.2] 跨零，而扩覆盖面是 +207.5。
该结论有一个前提（当时新批只占语料 9.7%），R6 新批占比更高，
故另有一个并行的双 N 冒烟在标签层面复核，见 `logs/dualn_smoke_0918/`。

## 云端准备

1. 拉取用户已验收的最终提交。不得带未提交代码运行——驱动会在
   `git diff HEAD --name-only` 非空时直接拒绝。
2. 原样传递本机 `target/gen2_cloud_assets_0918/` 参考目录（不在 Git 内）。
   它含原始模型、sidecar、游戏基座和 `files.json`；是原文副本，不是指纹。
   云端工作区相应文件必须与参考目录**逐字节**一致。文件 mtime 和存放目录不参与一致性判断。
   缺资产时先补齐，不能换模型或只凭同大小放行——
   ❗`ens_NT4096_AllHistory_g123.onnx` 与 `ens_R4_g123.onnx` **字节数完全相同（8142565）**，
   大小不能用来确认身份。
3. 号段：本轮正式独占 `[301000000, 1500000000)`，未用备用序号也仍保留。
   双 N 冒烟占 `[300000000, 300200000)`，本机代码冒烟占 `[300200000, 300400000)`，
   两者都不入正式数据。云端冒烟使用前检查其云端历史占用；冲突则停止报告，不擅改正式清单。
4. Rust 只用 Release，不运行 `cargo fmt`，不计算哈希或替代指纹。

```text
cargo build --release -p umasim --features cli,onnx --bin ramen_teacher_collect --bin ramen_export_npy
python scripts/collect/run_formal_collect.py --plan scripts/collect/formal1024_0918     --expect-search-n 1024 --expect-target-valid 68400 --validate-only
```

❗`--expect-search-n` / `--expect-target-valid` 是**必填**的：它们必须与清单里的
`search_n` / `target_valid` 完全一致，否则直接拒绝。存在的理由是让「这一轮按什么口径跑」
必须写在命令行上、留在 shell 历史与 `evidence/args.json` 里，
**单改 manifest 无法静默改变开跑口径**。

云端先做独立小冒烟，核对模型、空间、1024 列宽和原文资产，然后运行：

```text
python scripts/collect/run_formal_collect.py --plan scripts/collect/formal1024_0918     --expect-search-n 1024 --expect-target-valid 68400     --exe target/release/ramen_teacher_collect --export-exe target/release/ramen_export_npy     --output training_data/gen2_formal1024_0918     --asset-reference <收到的 gen2_cloud_assets_0918 目录>     --threads <实例实际CPU配额对应线程数>
```

## 历史：0914 那轮（2048 / 22800 根，已完成）

清单在 `formal2048_0914/`，roll-in 是 `ens_R4_g123`，资产目录是
`target/gen2_cloud_assets_0914/`，号段 `[20000000, 300000000)`（实际消耗到 215535245）。
重放它要显式写出当轮口径：`--plan scripts/collect/formal2048_0914
--expect-search-n 2048 --expect-target-valid 22800`。

Windows可执行文件加`.exe`。Linux通过现有tmux/nohup等保持会话，不创建新付费机器。
线程数按实际CPU配额选定并记录，不根据宿主机虚报核心数设置。该命令是正式采集授权入口，
不是定价测试；驱动缺运行参数只报错，不自动开跑。

## 运行与恢复

- 16个分层任务交错覆盖马娘及卡组；清单含双倍候选序号作为同层备用。
  未捕获如实计数，达到有效根目标就停止，备用耗尽失败，不能换便宜根。
- `indices-file` 模式的manifest `index_start/index_end/next_index` 为**清单游标**，
  样本内的index仍是真实随机世界序号，`work_indices`原样保存，不能混淆。
- 同一命令恢复同一输出目录，必须保持代码、配方、线程和完整工作清单。
  恢复仍使用清单声明的原截止预算（R6 是 72000 秒）；不自动重置。根间检查软截止，外部进程最多额外60秒收尾，
  超过才终止本驱动进程树，完整分片保留。硬终止恰落在分片/manifest提交之间可能留下孤立分片，
  下一次会拒绝续跑，需人工核对，不能删除后盲跑。
- 日志直接流式写盘；每任务保存参数、PID、退出码、耗时。失败默认停止。
  `run_state.json`保存完成任务与对应raw导出目录。
- 每任务采完都核对实际配置、资产字节、有效数、完整index清单，并执行raw导出。
  采完但导出未完成不冒充交付完成；重试导出写新目录，保留旧失败证据。

## 数据兼容边界

- 新显式数据携带`assets/`原文副本；续跑逐字节对当前资产，合并逐字节对两侧资产。
  旧定价数据缺副本会被新导出拒绝，不可补写“已核对”身份来绕过。
- 显式导出版本为2，新增`combo_fields.npy [N,7]`：马娘加六张按ID排序的卡。
  不生成FNV combo_key。所有旧路径的旧字段与FNV语义保留，但本轮不执行旧哈希路径。
- `NpyShard`可读取完整字段；`split_refs_by_combos`按预登记完整组合划分。
  原默认划分遇到新字段会拒绝，防止静默回落到index或哈希。
  **旧combo_key数据与新字段数据混训尚需基于真实计划的字段映射，当前训练CLI不能直接混训。**
  这不影响本轮原始数据采集及raw导出；此次不生成标签或启动训练。
- 第三年保留完整120组合，原始候选槽位与有序随机世界分数保存，未改网络头。
- 留出250组合只来自已确认未见的两张新速卡或114101；按马娘×构成排序每十取一。
  这是保守子集，不代表所有新组合的10%。详细来源与边界见`holdout.json`。

## 验证脚本

`python scripts/collect/test_formal_collect.py -v` 使用假进程及完整字段测试，
不启动采集、训练。`prepare_gen2_formal.py`需要Rust完整计划dump，只允许生成新目录；
不要在云端重生成已冻结清单。正式号段与已用数据冲突时应报告，不能修改清单后续跑。
