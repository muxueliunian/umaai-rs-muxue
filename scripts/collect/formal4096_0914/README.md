# 新教师4096云端交接

本目录为50000有效根清单：40000通用 + 地区1428/4286/4286；四构成各12500。
每候选固定4096，非UCB，R4 g123 roll-in，epsilon=0.15，手写终局续跑，rf=1.4。
不设墙钟截止，不自动训练；出现数据错误或低磁盘时停止。

## 版本和隔离

交付分支：`codex/collect4096-newteacher`，基于本地采集版本602bff9整合上游8a23566。
只有提交和推送完成后它才是云端可拉取的版本，不能拿本文所在脏工作树正式开跑。
主工作树的地区NN接入与未提交训练修改不属于此分支。
上游手写新PT定价进入for_rollout，raw仍记录普通终局评分。没有独立复现上游声称的平均分提升。

原云端工作目录保留，新建同级隔离工作树；已存在则先核查，不覆盖：

```sh
git fetch origin codex/collect4096-newteacher
git worktree add --detach /workspace/umaai-rs-muxue-t4096 origin/codex/collect4096-newteacher
```

上述命令在旧云端仓库执行。记录新目录的实际HEAD，并与本机交付的提交号逐值相同。
不得直接追踪master或一边采一边更新版本。原R4模型/旁车从已有资产复制到新树同路径，不上传公开模型。

## 号段

- 正式候选预留 `[300000000,800000000)`；10万个含备用index实际落在300001370至728798169。
- 本机冒烟预留 `[800000000,800010000)`；实际使用每种层起点800000000/800001000/800002000/800003000。
- 云端冒烟候选预留 `[800010000,800020000)`。
- 旧正式 `[20000000,300000000)` 全部保持占用，包括未用备用。

本机已核查历史manifest；尚未核对云端全部历史和预留登记。
云agent必须核对后登记这些区间，才可传 `--reservation-confirmed`。有冲突即停止，
不得随手平移已经冻结的序号；重新分配须统一修改/验证完整清单并确认版本。
plans和holdout沿用旧原文字节：4288组合、250排除组合、4038可采组合。

## 构建和验证

```sh
cargo build --release --locked -p umasim --features cli,onnx --bin ramen_teacher_collect --bin ramen_export_npy
python scripts/collect/test_formal_collect.py -v
python scripts/collect/test_newteacher4096.py -v
python scripts/collect/run_formal_collect.py --plan scripts/collect/formal4096_0914 --validate-only
```

新驱动用 `--require-newteacher-defaults` 核对Rust实际配置：
`ramen_pt_sacrifice_score=0`、`pt_favor_rate=1`，不静默覆盖用户参数。
Linux/本机仍须分别冒烟，确认普通及三年地区4096列、地区候选10/10/120；
冒烟使用新独立index清单，`--start 0 --count <清单长度> --accepted-target 1`。
有accepted-target时必须传indices-file，不能把世界index直接当清单游标。

## 资产

模型与旁车必须是上一批R4原文件；无须重新核验已下载训练过的谷歌云旧数据备份。
新采集资产身份校验仍需执行，它防的是采错模型/配置，不是重复验收旧数据。
以旧参考资产目录为输入，生成新参考目录（输出必须不存在）：

```sh
python scripts/collect/prepare_collect_assets.py --reference <旧冻结资产参考目录> --output target/newteacher4096_assets_lf
```

配置取当前固定源码，文本统一LF；模型和旁车保持原始字节。游戏JSON只允许换行规范化，
其它差异拒绝。正式启动会再次要求运行目录与新参考逐字节一致。
Windows创建的参考包位于本机隔离树target/newteacher4096_assets_lf，未公开上传。
不得把旧资产包的default_config直接覆盖到新源码。

## 正式运行（前置检查完成、固定版本获确认之后）

线程数依据实例实际CPU配额确认，不照搬历史16线程。以下THREADS为替换占位，不是命令默认：

```sh
python scripts/collect/run_formal_collect.py \
  --plan scripts/collect/formal4096_0914 \
  --exe target/release/ramen_teacher_collect \
  --export-exe target/release/ramen_export_npy \
  --output training_data/newteacher4096_0914 \
  --asset-reference target/newteacher4096_assets_lf \
  --threads THREADS --reservation-confirmed --min-free-gib 5
```

使用既有tmux/nohup保持进程存活；不要开启新付费机器。正式模式拒绝未提交及未跟踪源码文件。
状态/log/args/exitcode保存在output内；命令恢复时保持完整配方、线程、源码、资产和index不变。
新批seconds=null不施加总时间截止；旧2048配方仍保留最初12小时截止。
每30秒检查磁盘余量，低于阈值停止本驱动拥有的子进程并报告，不能自动清理数据或无限重启。
异常中断可能留下孤立分片，应核对后人工处理，不能删掉后盲跑。
采完但导出失败时恢复只重试导出，失败导出目录不覆盖。每任务raw验收通过才标记完成。

## 存储和保护

用户已确认旧数据从谷歌云完整下载并用于训练，无须再验备份完整性。
只能清理明确的旧数据目录/传输分卷，不删除整个工作区。
先保存模型/旁车、manifest、清单、号段登记、资产参考、日志和验收报告等必要文件。
51G剩余空间不是开跑保证，先估算bincode与raw同时存在的峰值。

禁止SHA256及任何替代内容哈希/指纹；比较实际字节或字段。禁止cargo fmt。
不提交/推送其它分支，不创建PR，不训练、不自动部署。
