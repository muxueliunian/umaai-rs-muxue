# NN 实验工作树收尾审查（2026-09-13）

## 范围、状态与结论

审查需求来自本轮测量、地区局部接入任务单及用户的收尾请求；规范来自根 `AGENTS.md`。
本次只修改决策/登记/审查文档，未修改、回退或移动 Rust、配置、模型、脚本及 scratch。
未提交、推送、拉取合并、重跑实验或启动训练；没有运行 `cargo fmt` 或计算内容哈希。

- 审查基准为 `HEAD=ed56e9d` 的未提交差异（其前两个提交仅为文档，代码基线 `07c4868`）。
- 审查开始时：9 个已跟踪文件修改，增 830 / 删 53 行；另有 2 个未跟踪实验 bin，
  不在上述 diff 行数内；以及 8 个原有未跟踪 scratch。
- 远端只读核验 `git ls-remote upstream refs/heads/master` 返回 `e8f6f26`，
  与本地跟踪引用相同；`git merge-base --is-ancestor upstream/master HEAD` 成功。
  所以当前没有新上游提交待合入；**不等于未来不会冲突**。未修改远端或执行 fetch/merge。
- `origin/master` 本地跟踪引用为 `07c4868`，master 比它多 `93b6c8c`、`ed56e9d`
  两个文档提交；本次未在线核对 origin，也未推送。
- 主结论：地区实验隔离合理，不需要为了避免冲突整体撤销；**不建议把全部实验改动
  原样直接提交到 master**。先收敛旧工具扩展、修复证据写入边界，再整理提交。

## 文件处置建议（仅评估，未执行）

| 当前改动 | 用途与判断 | 建议 |
|---|---|---|
| `crates/umaai/src/bin/ramen_client_game_bench.rs`（新增） | 配置建局、mcts/nn/地区混合分派、逐局记录；地区包装器只在外层使用 NN | 保留为实验入口；不直接移植正式客户端，见 R3 |
| `crates/umaai/src/bin/ramen_client_cost.rs`（新增） | 复用 `calc_ramen_training` 测量模拟根客户端链路 | 保留；明确模拟根、软门限；补 CLI 根解析的直接测试 |
| `crates/umaai/Cargo.toml` | opt-in `onnx=["umasim/onnx"]`，未修改依赖项与默认 feature | 合理，保留；它是 crate 级开关，不是单 bin 专属开关 |
| `crates/umaai/src/lib.rs`、`src/decision/mod.rs` | 导出 scenario/decision 和 LastReasonSink 方法供新 bin 复用真实链路 | 有实际用途，保留最小接口；新公开方法补文档，不复制客户端算法 |
| `crates/umasim/src/search/flat_search.rs`、`search/mod.rs` | SearchProbe 及计数/计时出口、2 个中立性/失败计数测试 | 合理的共享测量接口；默认 None，但存在分支和字段开销；修正“零影响”措辞 |
| `crates/umasim/src/trainer/ramen_mcts_trainer.rs`、`trainer/mod.rs` | DecisionProbe 外壳、with_search_probe、2 个直接测试 | 基本合理；原 select_action 正文未改算法。后续不要继续往这里堆实验策略 |
| `crates/umasim/tools/data_collection/ramen_space_bench.rs` | 旧 NN 完整续跑实验增加 client-config、串行计划、成本导出并改身份字段 | **优先收敛**；见 R1/R2。新整局工具已承担当前配置测量，不再继续扩展此处 |
| `crates/umasim/tools/data_collection/ramen_root_bench.rs` | 扩展手写 CPU 臂、首次推理计时、摊薄单价 | 与旧定价有关，当前地区路径不依赖；应独立决定保留/移出，而非顺带合入 |

## 需求与行为审阅

### R1 — 旧空间工具成本采集不是 opt-in（P2）

`ramen_space_bench.rs` 的 `run_plan` 搜索分支无条件创建探针、调用 `with_search_probe`、
读取时钟及全局推理计数，并把所有局的成本保存在结果中。没有 `--cost-csv` 也执行这些操作。
这偏离「只测量、不开启时保留既有路径」的目标，会污染后续旧工具成本比较并增加维护面。

建议：优先将不再使用的测量扩展移出本轮待提交范围；若确需保留，则让明确的成本开关
控制完整采集过程。并行计划时逐局 infer 是全进程计数差，必须禁止当作单局推理数输出。
`--client-config` 只对齐搜索/线程，建局仍是 SamplingSpace + gen1_inherit，不能用作用户配置基线。

### R2 — 成本 CSV 的写入边界可破坏证据（P2）

`ramen_space_bench.rs` 到全部对局完成后才检查 `--cost-csv` 的适用性，并用 `File::create`
写出：中途失败没有成本 CSV；既有同名成本证据会被截断；若误把成本路径指向逐局 journal，
也没有事先保护。完全 resume 且无新局时还会因 costs 为空在尾部报错。

建议：若保留该扩展，开跑前校验策略和输出路径、拒绝覆盖/别名碰撞，逐局持久化，
并明确 resume 的成本缺失语义。不能靠人工记得使用新目录保证证据安全。

### R3 — 地区包装器仅满足实验评分，不能直接部署（P2，部署前）

`ramen_client_game_bench.rs` 的 `RegionNnTrainer::select_action` 在地区分支跳过内部 MCTS，
但 `last_decision()` / `last_breakdown()` 仍原样读取内部 MCTS。地区 NN 决策没有生成自己的
DecisionInfo，也没有清理此前 MCTS 摘要；复用到消费这些接口的客户端可能读到旧决策。
当前整局 benchmark 由显式地区记录与 GameOutcome 取结果，不据此否定已完成成绩。

建议：继续保留实验私有包装器。正式接入任务应单独实现并测试地区决策的来源、摘要、
reason/sink 输出与状态清理；不能把当前包装器直接接入客户端。

### R4 — 旧完整续跑工具的使用限制与证据基础设施（P2，下一轮前）

- `ramen_space_bench` 的身份链仍自动计算模型、源码、配置、数据的 FNV；这是既有实现，
  不是本轮新增，但本轮扩展并没有解除冲突。用户的无必要哈希约束下不得直接拿它开下一轮。
  身份 JSON 本次也改了字段，旧 journal 的身份闸门可能拒绝续跑；不能绕过闸门混入旧结果。
- `ramen_root_bench` 新加的首次 `nn.infer(&root)` 无条件发生，手写 CPU 臂也加载并推理 NN；
  它不属于“完全不依赖模型的手写进程”。其逐 rollout 记账窗口不能直接当纯模拟单价，
  已完成报告也撤回过该单价口径，不应再用该数给新任务定价。
- `logs/region_nn_1024_0913/run_both.ps1` 原驱动仍保留已知输出流故障；修正版 B 驱动
  把剩余预算写死为 2245 秒，并非可复用共享截止脚本。还需对非零 ExitCode 主动失败，
  而不只是写文件。下一轮应有一个受版本控制的最小可靠驱动，避免继续复制修补日志脚本。
- 当前正式证据、冻结清单与分析脚本在被忽略的 `logs/`；`git status` 不代表全部实验资产。
  不移动或删除原证据，但后续应把可复用驱动/分析逻辑与具体运行产物分开。

## 代码规范与测试边界

- 新 bin 和旧工具扩展仍多处直接使用 `std::env`、`std::fs`、`std::thread`、`rayon::...`、
  `fs_err::File`，违反项目先 `use` 的要求。新增固定记录仍有 tuple（如世界定位、
  region_nn_run 返回值），新公开 LastReasonSink 方法及部分新函数缺 rustdoc。
  这些应局部修正，禁止用 `cargo fmt` 扫全仓库。
- SearchProbe 注释同时存在“关闭非零开销”和“热路径零影响”，后者应改为不改变决策语义；
  搜索中立性测试比较 score 的 sum_sq，却漏了 score_pt 的 sum_sq，与描述不完全一致。
- 地区新 bin 的 6 项单测覆盖参数/配置、地区判别与 CSV，但未直接调用完整混合包装器
  验证非地区转发、事件、缓存和状态接口。已完成 20 对与 3 次/局推理记录支持本次隔离，
  不替代后续重构或部署时的直接回归测试。
- 本次不改 Rust，因此未重跑构建/测试或 benchmark；此前 Claude 报告探针相关
  58 passed / 3 ignored、umaai lib 18 passed、最新 bin 6 passed，分别属于当时版本，
  不合称为本次完整工作树重验通过。收敛实现后只跑受影响的 Release 检查。

## 上游合并成本与收尾顺序

1. **保留小型共享接口，隔离策略实验。** `flat_search.rs` 与 `ramen_mcts_trainer.rs`
   同时承载现有本地搜索/训练功能，是未来上游同步的主要风险点；探针新增确实增加交叉修改
   机会，但没有发现本轮改变 UCB 分配、选优或默认拉面策略。不要为规避潜在冲突复制整套搜索。
2. 新 bin 主要是本地文件新增，上游同名碰撞风险较小；Cargo feature 和 lib 导出改动较小。
   旧 bench 的大幅扩展虽可能是本地专属代码，仍会增加本地行为和续跑兼容性维护负担；
   **维护负担与上游文本冲突是两件事**。
3. 在用户确认后的收敛任务中，逐块处置旧完整续跑测量扩展、修 R1/R2 与必要规范问题，
   保留原始证据；不使用整文件 restore 覆盖此前已提交功能或用户修改。
4. 同一次收尾流程为不同用途准备可审阅变更组：共享探针与最小接口、实验入口与驱动、
   决策登记文档。提交前统一更新 changelog，按项目规则由用户确认；本次不创建提交或分支。
5. 保留所有原有 scratch：`scratch_arm.sh`、`scratch_arms_all.sh`、`scratch_r3full.sh`、
   `scratch_r3sub.sh`、`scratch_r3x30k.sh`、`scratch_round2.sh`、`scratch_s456.sh`、`scratch_w3.sh`。

扩采与地区头方案见 [nn_pipeline_plan.md](nn_pipeline_plan.md) §15.11。
完成当前代码收敛和采集任务单冻结后，再由用户派 Claude 开下一轮，避免边采集边改变协议。

---

## 收敛执行结果（2026-09-13 稍后，由 Claude 执行）

本节记录按上文逐项收敛之后的实际状态。仍未提交、未推送、未 merge/rebase、
未清理、未整文件回退用户改动；未训练、未采集、未跑任何 benchmark；
未改 `game_config.toml`、未换默认模型、未接正式客户端入口；未运行 `cargo fmt`；
未计算任何哈希或指纹，一致性核对一律直接比较字段与文件内容。
`logs/` 下的原有实验结果、脚本、stdout、CSV 全部原样保留。

### 已收回的旧工具扩展

| 文件 | 处置 | 依据 |
|---|---|---|
| `crates/umasim/tools/data_collection/ramen_space_bench.rs` | **本轮未提交差异已全部收回**，与 `HEAD` 内容一致 | 收回前逐块通读该文件的完整 diff：`--client-config` / `--serial-plans` / `--cost-csv` / `--search-group-size`、`search_n`/`radical_factor_max` 改 `Option`、`SearchSetup.stages*`、`GameCost`、`infer_count()`、`run_identity` 增 `kind` 参数与身份 JSON 字段变更、串行执行分支、成本打印与 CSV 落盘——**没有一块属于 HEAD 既有功能或本轮之外的改动**，故整文件回到 HEAD 与逐块撤销等价。HEAD 既有的 NN rollout（`--rollout-model` / `with_nn_rollout`）、续跑身份闸门与教师口径默认值（`search_n=512`、`use_ucb=false`、阶段 `all`、`radical_factor_max=1.4`）全部保留 |
| `crates/umasim/tools/data_collection/ramen_root_bench.rs` | **本轮未提交差异已全部收回**，与 `HEAD` 内容一致 | 同样逐块通读：无条件首次 `nn.infer(&root)` 与 `first_infer_ms`、手写 CPU 臂的 `ensure!` 与 `rollout_trainer` 分支、`infer_request_count` 导入与窗口净增量、rollout 条数/推理次数/摊薄单价三行打印——全部为本轮新增。HEAD 既有的 teacher-game 模式、批量后端、诊断与模型加载逻辑，以及 `--handwritten-rollout` 在 teacher-game 下的原有语义，均保留 |

核对方式：`git status` 对两文件无输出、`git diff` 为空（比较的是实际文件内容，不是哈希）。
未发现任何一块差异属于其他调用方依赖，因此没有需要保留的例外项。
两文件收回后在 `--features cli,onnx` 下 Release 构建通过，说明保留的共享探针接口是纯增量。

### 保留的能力

`ramen_client_game_bench`（用户配置整局对照入口，含 `mcts+region_nn` 实验私有模式）、
`ramen_client_cost`（模拟根决策链测量入口）、`SearchProbe` / `DecisionProbe` 与最小转发接口、
两个 bin 所需的 `umaai` lib 导出与 `LastReasonSink` 公共方法、`umaai` 的 opt-in `onnx` feature
（**不进 default**，正式客户端默认构建不变）。

### 逐条处置

| 编号 | 结论 | 实际文件与验证依据 |
|---|---|---|
| R1 旧空间工具成本采集不是 opt-in | **已解决（按收回处理）** | 成本采集整块随 `ramen_space_bench.rs` 一并收回，该工具回到 HEAD 的「只测分、不挂探针」行为；没有新增第二套通用测量框架 |
| R2 成本 CSV 写入边界可破坏证据 | **已解决（按收回处理）** | `--cost-csv` 已不存在。新入口 `ramen_client_game_bench` 的 `games.csv` / `regions.csv` 在**开跑前**检查同名文件并拒绝覆盖，逐局落盘并 flush |
| R3 地区包装器暴露旧 MCTS 摘要 | **已按实验所需的最小方案解决；正式接入仍待处理** | `ramen_client_game_bench.rs` 的 `RegionNnTrainer` 增 `region_last: AtomicBool`：地区决策成功后置位，`last_decision()` / `last_breakdown()` 返回 `None`；转发给内部 MCTS 的**任何一次**调用（动作、`select_choice`、`select_event_choice`）都会清位，
摘要照常可见——屏蔽只覆盖地区决策那一步，地区之后紧跟事件时事件自己的说明不会被连带屏蔽
（首版只在动作转发处清位，事件会被误屏蔽，已修）。**仍未**为地区决策生成自己的 `DecisionInfo`，也**未**接 `DecisionSink` / `AIRedirector`——正式部署前必须单独实现并测试。测试 `test_region_wrapper_masks_stale_summary`：turn 2 地区决策后泄漏 0 次，随后的转发决策屏蔽位解除 |
| R4 旧完整续跑工具的使用限制 | **部分解决，其余保留限制** | 自动 FNV 身份链是 `ramen_space_bench` 的既有实现，**本任务未改写身份系统**；收回后该工具的身份 JSON 字段回到 HEAD 形态，旧 journal 闸门不再因本轮字段变更而拒绝续跑。`ramen_root_bench` 的首次推理与摊薄单价打印已移除，不会再被误当纯模拟单价。下一轮用户配置实验继续走无哈希的新入口 |
| R4 驱动脚本 | **已解决** | 新增受版本控制的 `scripts/bench/run_paired_game_bench.ps1`：绝对截止时刻的共享预算（B 只拿真实剩余，含间隔与模型加载）、`WaitForExit` 后立刻保存 `ExitCode`、非零退出码明确失败且默认不启动下一臂、超时只 `Kill($true)` 本驱动启动的进程树、拒绝覆盖非空输出目录、留存命令/stdout/stderr/退出码/耗时、函数只返回 `[pscustomobject]` 且一律 `Write-Host` 打印、`UseShellExecute=$false` + `CreateNoWindow=$true` 不弹窗
（脚本用 `ProcessStartInfo` 而非 `Start-Process`，**没有** `-NoNewWindow` 这个参数，早前本节写错过机制名）、
不算任何哈希、必需参数缺失只打印用法并退出 2。**没有**照搬旧脚本里的硬编码世界、2245 秒剩余预算或历史输出目录；`logs/region_nn_1024_0913/` 里的历史脚本原样保留未改。

**GPT 静态复核后补修的三处边界**：(a) 两臂子目录名在 A 启动**之前**校验——必须互不相同、
只允许 `[A-Za-z0-9._+-]`、解析后必须是本次输出目录的直接子目录、各自不得已有内容，
堵掉「B 覆盖 A 的 command/stdout 之后才由 benchmark 报 CSV 已存在」与 `..` 逃逸；
(b) 改用 `ProcessStartInfo.ArgumentList` 逐个塞参数，由 .NET 负责转义，含空格的模型/输出
路径不再被拆开，另存 `command.args.json` 作为参数边界的权威记录（`command.txt` 降为人读形态）；
(c) 正式模式下 `-Arm?ExtraArgs` 出现受保护参数（`--runs --seed --run-offset --cards
--extra-count --threads --search-n --out --trainer --model --special-mode`）即拒绝启动，
杜绝「驱动打印相同条件、benchmark 按后出现的值跑成两臂不同」。
(d) stdout/stderr 由两条流各自 `CopyToAsync` 到**不带内部缓冲**（`bufferSize=1`）、
`FileShare.ReadWrite` 的 `FileStream`，**边跑边落盘、运行当中即可读**，驱动被中断也不丢已产生的日志；
顺序改为「等待退出 → 先存退出码 → 有界（5 s）等待日志收尾」，退出码不再排在日志之后。
（中间版本曾用 `ReadToEndAsync` 一次性落盘，是证据保存的退步，已改回流式） |
| 规范：先 `use` 再引用 | **已解决（限于本次保留的新增代码）** | `ramen_client_game_bench.rs` 与 `ramen_client_cost.rs` 的 `std::env` / `std::fs` / `std::io` / `std::process` / `std::thread` / `std::slice` 全部改为先 `use`；测试模块同理。未清扫历史全仓库同类问题 |
| 规范：固定结构用具名类型 | **已解决** | 新增 `WorldSpec { run_idx, rule_master }` 取代预登记世界的 `(u64, u64)`；新增 `RegionNnRun { outcome, obs }` 取代 `region_nn_run` 的 `(GameOutcome, Vec<RegionObs>)` |
| 规范：缺 rustdoc | **已解决（限于本次保留的新增代码）** | `LastReasonSink::new` / `take` 补文档；新增类型、字段与 `last_decision` / `last_breakdown` 覆写均有文档注释 |
| 规范：过期 CLI 示例 | **已解决** | `ramen_client_game_bench.rs` 模块文档里第一条示例原先把 `//!` 串进了命令行，且缺 `--search-n`；已改成两条可直接复制的示例（`mcts` 与 `mcts+region_nn`），并指向新驱动 |
| SearchProbe 注释「热路径零影响」 | **已解决** | `flat_search.rs` 的 `with_probe` 文档改为：默认不挂**不等于零开销**（多一个 `Option` 字段、每次搜索一次 `Option::map`、每组一次 `Option::is_some` 分支），它保证的是**不改变决策语义**，搜索结果与 RNG 消耗逐位不变 |
| 中立性测试漏比 `score_pt.sum_sq` | **已解决** | `test_probe_neutral_to_search_result` 增加 `score_pt` 平方和比较并打印两侧数值；`use_ucb` 开关两种情况各跑一次 |
| 未直接测试混合包装器的转发 | **部分解决，保留限制** | 新增 `test_region_wrapper_masks_stale_summary` 直接构造包装器跑到 turn 2，覆盖「地区走网络 + 摘要屏蔽 + 屏蔽位解除 + 非地区转发」。**仍未覆盖**：事件选项转发（`select_choice` / `select_event_choice` 的清位只经代码检视，测试在遇到第一个恢复摘要的转发决策时就 break，无法断定走的是哪条方法）、`SpecialSelect` 合并缓存行为、整局逐位对拍。❗**另一个天窗**：`saved_models/` 在 `.gitignore` 里，干净检出没有权重，该测试会**跳过并照样报 ok**；现已改为跳过时打三行显眼警告，并支持 `UMAAI_REQUIRE_REGION_MODEL=1` 强制失败（本机模型存在，强制开关下实跑仍通过；`bail` 那一支本身未被触发验证）。这些由已完成的 20 对实验与既有 `RamenMctsTrainer` 自身测试间接支撑，不替代正式接入时的直接回归测试 |

### 本次实际跑过的 Release 检查

| 命令 | 结果 |
|---|---|
| `cargo build --release -p umaai --bins` | 通过（仅既有 warning） |
| `cargo build --release -p umaai --features onnx --bins` | 通过（仅既有 warning） |
| `cargo build --release -p umasim --features "cli,onnx" --bin ramen_space_bench --bin ramen_root_bench` | 通过（收回后的两个旧工具） |
| `cargo test --release -p umasim --lib --features "cli,onnx" -j 8 probe -- --nocapture` | 4 passed / 0 failed（搜索探针中立性、失败计数、决策探针整局中立性、合并路径不暴露 `DecisionInfo`） |
| `cargo test --release -p umaai --features onnx --bin ramen_client_game_bench` | 7 passed / 0 failed |
| `cargo test --release -p umaai --bin ramen_client_game_bench`（默认 feature） | 5 passed / 0 failed（2 个 onnx 测试按 `cfg` 不编译） |
| `cargo test --release -p umaai --bin ramen_client_cost` | 2 passed / 0 failed（根规格解析正常 / 异常） |
| `pwsh -NoProfile -File logs/_driver_selftest_0913/selftest.ps1` | **11 个假进程用例、16 项断言全部 PASS，脚本退出码 0**。脚本本身已带机器判据（每项都有预期值，不符即 `exit 1`），且每次运行建带时间戳的新目录，**可原地重跑**——实测连跑两次都是 16/16、退出码 0（早前版本只打印不断言，且重跑时用例 1 会被「拒绝覆盖」抢答，已修）。逐项：无参数调用→2；两臂成功→0；A 退 3→B 未启动（B 目录都没建）、驱动 1；共享预算 8 s 中 A 占约 4.5 s、B 只拿到约 3.5 s 后被终止→驱动 1；重复输出目录→2；含空格与引号的参数原样到达子进程（`ARG=[path with spaces\model.onnx]`、`ARG=[has"quote]`）→0；两臂同名→2；臂名 `..`→2；臂名 `A` 与 `A.`→2；正式模式下 `-ArmBExtraArgs --seed 999`→2。四个拒绝用例**一臂都没启动**（输出目录下 0 个子目录）。日志实时性：A 臂先打一行再睡 8 s，驱动运行到约 1.2–1.5 s 时就已从 `run.stdout.txt` 读到 `EARLY-LINE`，**同一时刻** `LATE-LINE` 尚未出现、`run.exitcode.txt` 也尚未生成——8 s 的间隔远大于调度抖动，故能证伪「结束后一次性落盘」，同时证明退出码确实在进程结束后才写 |

未覆盖边界：没有重跑任何 benchmark 或整局测量；没有运行会自动计算模型/数据身份哈希的旧工具测试
（因此两个收回的旧工具只确认了「内容与 HEAD 一致 + 能构建」，**运行时行为未验证**）；
没有为未改动路径重复跑全仓库测试；`ramen_client_cost` 的整链测量本身未重跑（只测了参数解析）；
`UMAAI_REQUIRE_REGION_MODEL=1` 在模型缺失时的 `bail` 分支未被实际触发（本机有权重，不移动模型文件）；
`logs/` 被 `.gitignore`，历史证据只能靠文件清单与 mtime 比对，无版本对照，且按无哈希约束不做内容指纹。

### 仍未解决的风险

1. **正式客户端默认路径未变，也未验证。** `umaai` 主程序仍不加载拉面 NN，`onnx` 不在 default；
   本次没有改动主程序，也没有跑过它。
2. **地区摘要只做到「不暴露旧的」。** 地区决策没有自己的 `DecisionInfo` / breakdown / reason 输出；
   任何把该包装器往客户端搬的动作都必须先补这一层，不能沿用当前实现。
   `crates/umasim/src/game/ramen/action.rs` 把「手写逻辑」硬编码进地区日志标签的显示问题本次未改。
3. **驱动的共享预算是墙钟截止，不是 CPU 预算**；终止是进程树 `Kill`，被终止那一臂的逐局 CSV
   与日志都只保留到那一刻为止。受保护参数只拦「重复给共享字段」，
   不拦 benchmark 未来新增的其他影响可比性的开关；新增开关时要同步维护该清单。
   驱动只验证过假进程，**尚未**带真正的 benchmark 跑过一次完整配对——
   这一项**并入下一次已授权任务的开跑冒烟**即可，不必单独加跑。
4. 收敛后的工作树仍未提交；提交前需按项目规则统一更新 changelog 并由用户确认。
5. **不宜写成「代码阻塞项全部清零」，但以下三项都不是采集的阻塞项**，如实登记即可：
   - 地区决策的 `DecisionInfo` / reason / sink 输出仍缺 —— 属于**正式客户端部署前**的工作，
     与扩采无关；当前实验只需要「不暴露旧摘要」。
   - 驱动的真实 benchmark 冒烟 —— 并入下一次已授权任务的开跑步骤。
   - 混合包装器的事件转发、`SpecialSelect` 合并缓存无直接回归测试 —— 登记为已知未覆盖边界，
     本轮不扩成全面测试。
   **采集面板与配额设计可以继续推进。**
