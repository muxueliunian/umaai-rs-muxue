# gen2_v1 正式采集（2048）

`formal2048_0914/manifest.json` 是正式配方；`gen2_v1_recipe.json` 与
`price0914_runs_*.json` 是历史512定价资料，不得当成正式运行清单。

正式目标：20000通用有效根（四构成各5000），地区400/1200/1200
（每年再按四构成均分）。云端单机承担，R4 g123 roll-in，内部手写终局续跑，
每候选2048，uniform，rf1.4，地区All。四构成、马娘、卡池完整字段均在清单内。

## 云端准备

1. 拉取用户/GPT已验收的最终提交。不得使用99d975f开新空间采集，也不得带未提交代码运行。
2. 原样传递本机 `target/gen2_cloud_assets_0914/` 参考目录（不在Git内）。
   它含原始模型、sidecar、游戏基座和files.json；是原文副本，不是指纹。
   云端工作区相应文件必须与参考目录逐字节一致。文件mtime和存放目录不参与一致性判断。
   缺资产时先补齐，不能换模型或只凭同大小放行。
3. 检查本机/云端既有任务号段；本轮独占 `[20000000,300000000)`，未用备用序号也仍保留。
   本轮本地代码冒烟预留 `[19000000,20000000)`，不入正式数据。云端冒烟另分配
   `[18000000,19000000)`，使用前检查其云端历史占用；冲突则停止报告，不擅改正式清单。
4. Rust只用Release，不运行cargo fmt，不计算哈希或替代指纹。

```text
cargo build --release -p umasim --features cli,onnx --bin ramen_teacher_collect --bin ramen_export_npy
python scripts/collect/run_formal_collect.py --validate-only
```

云端先做独立小冒烟，核对模型、空间、2048列宽和原文资产，然后运行：

```text
python scripts/collect/run_formal_collect.py --exe target/release/ramen_teacher_collect --export-exe target/release/ramen_export_npy --output training_data/gen2_formal2048_0914 --asset-reference <收到的gen2_cloud_assets_0914目录> --threads <实例实际CPU配额对应线程数>
```

Windows可执行文件加`.exe`。Linux通过现有tmux/nohup等保持会话，不创建新付费机器。
线程数按实际CPU配额选定并记录，不根据宿主机虚报核心数设置。该命令是正式采集授权入口，
不是定价测试；驱动缺运行参数只报错，不自动开跑。

## 运行与恢复

- 16个分层任务交错覆盖马娘及卡组；清单含双倍候选序号作为同层备用。
  未捕获如实计数，达到有效根目标就停止，备用耗尽失败，不能换便宜根。
- `indices-file` 模式的manifest `index_start/index_end/next_index` 为**清单游标**，
  样本内的index仍是真实随机世界序号，`work_indices`原样保存，不能混淆。
- 同一命令恢复同一输出目录，必须保持代码、配方、线程和完整工作清单。
  恢复仍使用最初12小时截止；不自动重置预算。根间检查软截止，外部进程最多额外60秒收尾，
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
