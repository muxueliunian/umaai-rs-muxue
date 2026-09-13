# 拉面 NN 模型、数据与评估世界登记

更新：2026-09-13。实验结论与证据边界见 [nn_pipeline_plan.md](nn_pipeline_plan.md) §15。
本表只登记已存在的文件与字段，所有路径相对仓库根目录；文件存在性与集成成员列表已逐项检查。

⚠ `saved_models/`、`target/`、`training_data/`、`logs/` 全部被 `.gitignore` 忽略。
下表产物**目前仅本地保存**，未发现异地备份记录；本次未上传。

## 1. 选定模型与实际入口

| 层面 | 当前状态 |
|---|---|
| 下一阶段对照 | R4@30k 两组：`ens_R4_g123`、`ens_R4_g456` |
| 后续实验候选 / roll-in（研究约定） | `ens_R4_g123`（沿用预先约定，不按事后成绩改选 g456） |
| 更早的研究约定 | 2026-09-11 起曾把 `ens_R3full30k_g123` 称作「默认模型」，只是约定，从未接入任何入口 |
| `umaai`（客户端决策入口） | 默认构建未开 `onnx`；拉面仍只构造 `RamenMctsTrainer`（手写 rollout 搜索），**不加载拉面 NN**。工作树新增的 opt-in onnx feature 供实验 bin 使用，不等于客户端接入 |
| `umasim` 主程序 | 配置 `trainer = "handwritten"` → 拉面走 `RecommendedRamenTrainer`；`nn` 分支未实现；`neuralnet_model_path` 属温泉 |
| bench / 采集工具 | 只通过命令行 `--model` 指定模型，没有默认拉面模型 |

**选定模型 ≠ 已部署模型。** 正式入口、配置与默认模型未改变。未提交实验 bin
`ramen_client_game_bench` 已支持 `mcts+region_nn`：仅外层地区用 R4 g123，搜索内部仍手写。
1024 档 20 对为 +1177.25 [+662.31,+1692.19]；详见计划 §15.11。
新增 20000 通用根及 2000–3000 地区专项根仍是计划，尚未采集或训练，新模型尚不存在。

## 2. 模型清单

共同项：三成员集成，policy 取成员平均 logit；成员为各种子的 `step_030000.pt`；输入 754 维、输出 245 维；
导出批大小 1 与 7 均核对过误差。g123 = 种子 1/2/3，g456 = 种子 4/5/6。

| 集成 | 状态 | 结构 / 每成员参数 | 训练 | 数据配方（total / train / val） | 评估世界 |
|---|---|---|---|---|---|
| `ens_R4_g123` | **保留对照；后续候选 / roll-in** | 96/4/2/256×2/0.08，671,797 | 30000 步 | R4 八组（211457 / 193371 / 18086） | 90000–90015、100000–100015、110000–110015、120000–120015 |
| `ens_R4_g456` | **保留对照** | 同上 | 同上 | 同上 | 同上 |
| `ens_R4wide_g123` | 实验归档（扩容暂停） | 128/4/2/384×2/0.08，1,306,357 | 30000 步 | 同 R4 | 100000–100015 |
| `ens_R4wide_g456` | 实验归档（扩容暂停） | 同上 | 同上 | 同上 | 同上 |
| `ens_R5_g123` | 实验归档 | 同 R4 结构 | 30000 步 | R4 + `npy_r4roll`（221621 / 202631 / 18990） | 110000–110015、120000–120015 |
| `ens_R5_g456` | 实验归档 | 同上 | 同上 | 同上 | 同上 |
| `ens_R5_20kdata_g123` | 实验归档 | 同 R4 结构 | 30000 步 | R4 + `npy_r4roll` + `npy_r4roll2`（231461 / 211592 / 19869） | 120000–120015 |
| `ens_R5_20kdata_g456` | 实验归档 | 同上 | 同上 | 同上 | 同上 |

注：`ens_R5_g123` 不是任何一批数据的 roll-in；两批 `r4roll` 的 roll-in 都是 `ens_R4_g123`。

### 2.1 文件路径

每套集成的文件模式如下（`<A>` = `R4` / `R4wide` / `R5` / `R5_20kdata`）：

- ONNX：`saved_models/arms/ens_<A>_g123.onnx`、`ens_<A>_g456.onnx`
  （原结构 8,142,565 字节；R4-wide 15,757,411 字节）
- 配套 JSON：同名加 `.json`，其 `ensemble_of` 已核对为本组三个成员
- 成员：g123 为 `target/arm_<A>_seed{1,2,3}/step_030000.pt`，g456 为 `target/arm_<A>_seed{4,5,6}/step_030000.pt`
- 训练记录：`target/arm_<A>_seed{1..6}/{run.json, metrics.jsonl}`

| 集成 | 评估报告 |
|---|---|
| R4 | `logs/arms/evaluation_r4/report.md`（另在 R4-wide、R5、R5-20kdata 报告中作对照） |
| R4-wide | `logs/r4wide/report.md` |
| R5 | `logs/arms/evaluation_r5/report.md`（另在 R5-20kdata 报告中作次要对照） |
| R5-20kdata | `logs/r5_20kdata/report.md` |

数据配方中「R4 八组」为 `training_data/` 下的 `npy_v6ck`、`npy_ood_half101`、`npy_dagger_c`、`npy_dagger_d`、
`npy_n4096_w1024`、`npy_n1024`、`npy_r2roll`、`npy_r3roll`，标签依次为 `labels_v6`、`labels_ood_half101`、
`labels_dagger_c`、`labels_dagger_d`、`labels_n4096_w1024`、`labels_n1024`、`labels_r2roll`、`labels_r3roll`。
冻结常量均取自 `target/dagger_d_seed1/run.json`。

## 3. 第五批数据登记

两批共同身份：roll-in `ens_R4_g123`（manifest `rollin` 为 `nn:741d02fc29a395d9`），采集代码 `07c4868`，
每候选 search_n 1024，有序 rollout，不开 UCB，radical 1.4，地区策略 all；标签参数见 [nn_pipeline_plan.md](nn_pipeline_plan.md) §15.0。

| 项 | 第一批 r4roll | 第二批 r4roll2 |
|---|---|---|
| index 号段 | 9600000–9610299 | 9610300–9620299 |
| 有效根 / 跳过 | 10164 / 136 | 9840 / 160 |
| 候选数 | 125539 | 123280 |
| 按 combo 规则 train / val | 9260 / 904 | 8961 / 879 |
| 原始分片 | `training_data/r4roll_s0{1,2,3}_p{0,1}` | `training_data/r4roll2_s0{1,2,3}_p{0,1}` |
| raw 导出（训练直接读取） | `training_data/npy_r4roll` | `training_data/npy_r4roll2`（由核验用导出原样移动而来，`meta.json` 的 `sources` 仍记暂存路径） |
| 标签 | `training_data/labels_r4roll` | `training_data/labels_r4roll2` |
| 收货暂存（未清理） | `training_data/incoming_r4roll_zip/` | `training_data/incoming_r4roll2_zip/` |
| 云机日志与配置 | `logs/cloud_r4roll/` | `logs/cloud_r4roll2/` |
| 冒烟 | 9589000–9589031，独立，不参与训练；只有 `logs/cloud_r4roll/r4roll/smoke.log`，数据目录未交付 | 无（复用第一批采集二进制） |
| 核验 | `logs/r5/verify_*` | `logs/r5_diagnosis/verify_*`、`logs/r5_20kdata/{prep_check.log, verify_labels_r4roll2.log}` |

实际使用：R5 用第一批；R5-20kdata 用两批。第二批未单独加入 R4 训练过。

## 4. 评估世界登记

全部为 seed 61444、gen1 空间 525 plan；不同行世界不同，绝对均分不可跨行比较。

| run_idx | 局 / plan | 用途 |
|---|---:|---|
| 10000–10001、20000–20003 | 2、4 | 教师预算截断：选择集、验收 |
| 30000–30001、40000–40003 | 2、4 | R2：选择集、验收 |
| 50000–50007 | 8 | R2-W3 |
| 60000–60007 | 8 | 种子组 4/5/6 复现 |
| 70000–70015 | 16 | R3 嵌套预算 |
| 80000–80015 | 16 | R3full 20k→30k |
| 90000–90015 | 16 | R4 |
| 91000–91007 | 8 | R4 行为面板 |
| 92000–92001 | 2 | 性能移植正确性 / 性能 |
| 100000–100015 | 16 | R4-wide（CPU 成本面板复用 100000–100003） |
| 110000–110015 | 16 | R5 |
| 120000–120015 | 16 | R5-20kdata（本表 gen1 面板最后一行；不是全部实验的最大号段） |

新验收须重新登记，并按实际世界种子检查与历史 CSV、采集世界的重叠，不只看 run_idx 数字。
这些世界都来自同一个固定配置面板，不证明真实客户端输入或分布外空间有效。

### 4.1 后续工具评估号段（2026-09-13）

以下也用基种子 61444，但配置与 plan 派生口径不同，不能与上表拼接绝对均分。
按已保存证据登记；后续分配仍须扫描实际 CSV/冻结记录，而非只查本表最大值。

| run_idx | 用途 / 状态 | 证据目录（仓库相对路径） |
|---|---|---|
| 129000 | 旧 gen1 plan 0 的 NN 完整续跑校准 | `logs/nn_rollout_price_0913/` |
| 130000–130003 | 旧 plan 0 效果测试；前三对完成，130003 预留未开始，仍记占用 | 同上 |
| 140000–140009 | 用户旧卡组：8192 手写搜索与 R4 直接决策配对 | `logs/handwritten8192_user_baseline_0913/`、`logs/r4g123_direct_user_baseline_0913/` |
| 149000 | 上述手写基线冒烟，不入统计 | `logs/handwritten8192_user_baseline_0913_smoke/` |
| 150000–150009 | 新 2速2耐1智1友：8192 手写搜索与 R4 直接决策配对 | `logs/newdeck_pair_0913/` |
| 160000–160019 | 1024 档：手写地区 vs 外层 R4 地区，20 对完成 | `logs/region_nn_1024_0913/` |
| 199000 | 地区混合两臂冒烟，不入统计 | `logs/_smoke_mcts/`、`logs/_smoke_region_nn/` |

## 5. 保存缺口

- 模型权重、checkpoint、训练数据、逐局 CSV、日志与实验脚本只在本地；无异地备份记录。
- 实验脚本（`logs/**/evaluate_*.py`、`register_worlds_*.py`、训练驱动 `.sh`）未进版本库，依赖本地目录结构。
- 更早轮次的训练驱动 `scratch_*.sh` 仍是根目录未跟踪文件。


## 6. gen2_v1 扩采号段预登记（2026-09-14）

本节为 sampler index，与整局benchmark的run_idx分开解释。范围端点按下表说明。

| 区间 | 状态与用途 |
|---|---|
| 9689000–9689807 | Claude 本机定价前冒烟占用，不入训练/评估 |
| 9700000–9720950 | Claude 本机512定价占用，不入训练/评估 |
| 9730000–9750966 | 云端历史定价清单预留，即使没跑也不复用 |
| [18000000,19000000) | 新版本云端冒烟预留，执行前另核对云端占用 |
| [19000000,20000000) | GPT本机2048修复冒烟预留，不入训练/评估 |
| [20000000,300000000) | 正式2048清单及备用独占预留，尚未开跑；实际稀疏index见formal2048_0914/indices |

正式计划4288，留出250；有效目标22800，清单含备用45600项。已扫描本机历史manifest无区间冲突。
冻结模型原文参考目录为 `target/gen2_cloud_assets_0914/`，已按字节核对本机副本，未上传、不在Git内。
云端需收到同一份原始资产目录，不能用大小/mtime当成内容一致证明。
