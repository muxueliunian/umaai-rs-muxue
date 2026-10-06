# 拉面 NN rollout 研究工具链交接

本次移植基点为上游提交 2ec6b49。本文是完整交接入口；2026-09 的性能计划、实验任务书和结果只作为历史材料，不代表已在这个基点完成验证。

## 范围与行为

本交接提供 CPU 网络续跑、可选批量后端、观测与对拍、Python/CUDA 波次实验，以及已有的叶截断研究入口。默认客户端继续使用上游手写续跑与 PT 选择口径；研究工具显式使用 Score，记录实际生效的评分和搜索参数。

- RamenRolloutTrainer：默认包装推荐手写策略，可装载共享网络、设置网络生效的绝对回合上限，并保留友人完成要求与推理观测出口。
- FlatSearch：严格失败传播、RamenBatchRollout / RamenBatchTable、搜索成本观测。批量后端返回原始终局结果，完整续跑的汇总和选优仍复用搜索内核。
- RamenMctsTrainer：网络续跑、搜索及决策探针接线；启用网络续跑时同时启用严格失败传播。
- ramen_root_bench：CPU/GPU 固定根与整局接线实验、缓存和批量优化、叶值诊断。
- Python 侧车与比较脚本：常驻成员、严格 FP32、批量前向、CUDA Graph，以及实际字段比较。

本次不修改游戏规则、随机流派生或手写策略；不新增 Rust 原生 CUDA 后端，不纳入 Q1/Q2 驱动、具体采集批次清单或归档分支 archive/leaf-estimation-0922 的未完成 CPU 叶估值。Python 侧车是离线研究工具，不接入客户端运行时。

## 文件导航

| 内容 | 文件 |
|---|---|
| 续跑基策、窗口、推理快照 | crates/umasim/src/trainer/ramen_rollout_trainer.rs |
| 批量接口、严格错误传播、搜索探针 | crates/umasim/src/search/flat_search.rs |
| MCTS 接线与决策观测 | crates/umasim/src/trainer/ramen_mcts_trainer.rs |
| 网络预处理、合法候选打分、value 解码 | crates/umasim/src/trainer/ramen_nn_trainer.rs |
| 实验评分参数与实际口径报告 | crates/umasim/src/exp_config.rs |
| 固定根、波次调度与叶实验 | crates/umasim/tools/data_collection/ramen_root_bench.rs |
| 小预算 CPU 续跑探针 | crates/umasim/tools/data_collection/nn_rollout_probe.rs |
| GPU 侧车及其模型加载 | scripts/ramen_nn/bench_sidecar.py、model.py |
| 成员与集成 ONNX 导出 | scripts/ramen_nn/export_onnx.py、export_ensemble_onnx.py |
| 原始 rollout / 决策比较 | scripts/ramen_nn/compare_root_bench.py、tests/test_compare_root_bench.py |
| GPU 前向独立探针 | experiments/gpu_forward_probe_0915/probe.py、README.md |

search/mod.rs、trainer/mod.rs、lib.rs 的导出、Cargo 的 bin/feature 登记及采样空间解析是这些入口的必要依赖。移植时不能只复制实验 bin。

采样模块的交接仅覆盖具名空间、解析、枚举及其测试。采样执行保持上游行为；本次不移植 region_y1 配额、NN roll-in 采集改造或组合 key 哈希，也不需要修改 collector。集成 ONNX 导出脚本补入本次交接，使侧车成员与 CPU 集成有可复现的共同来源。

## 三个必须分清的口径

1. **窗口和截断不同。** max_turn 是网络生效的绝对游戏回合上限，含上限回合；超界后换回手写，但仍续跑到终局。它不是“从当前搜索根往后走 k 步”。叶实验的 --leaf-h H 才按根回合后的 H 个回合边界截断，也不是 H 次动作或推理。
2. **Score 与 PT 不同。** Score 的真实评分包含 Hint 折算；PT 轴的定义和聚合遵循上游。把 pt_favor_rate 设为 1 不能代替选择 Score。研究入口通过 exp_config 固定并打印参数；客户端默认口径不随实验改变。
3. **网络基策与叶 value 不同。** 网络基策决定 rollout 中的动作；现有 value 标签表示“教师先选一步、随后按采集基策续跑”的价值。把该 value 替换 NN 续跑尾部会同时改变未来策略语义、偏差和方差，不是单纯减少 Monte Carlo 方差。

输入/输出保持 754 → 245：policy 234、choice 8、value 3。三成员集成平均 logits；value 先分别反归一化到真实尺度再平均。CPU ONNX 和 GPU checkpoint 必须属于同一成员集合、结构与归一化配置。

## 最小复现

以下命令在仓库根目录用 PowerShell 7 执行。它们是复现模板，是否在本次移植基点运行成功以文末验收记录为准。先准备已有 CUDA 环境和 scripts/ramen_nn/requirements.txt 所列 Python 依赖；无 CUDA 的机器可运行 CPU 路径，不能运行 GPU 侧车。

### 构建与资产

~~~powershell
cargo --config 'profile.release.lto="thin"' build --release -j 2 -p umasim --features "cli onnx" --bin ramen_root_bench --bin nn_rollout_probe
~~~

除同目录的 ensemble.onnx.json 外，GPU 还需要导出该集成的三个 PyTorch checkpoint。只有 ONNX 文件不能启动此侧车。下列路径是占位示例，替换为接收到的相对路径；只运行 CPU 时仅核对 ONNX 和旁车 JSON，跳过 checkpoint 检查及后续 GPU 段。

~~~powershell
$nnBench = "./target/release/ramen_root_bench.exe"
$nnModel = "./saved_models/research/ensemble.onnx"
$nnCkpt1 = "./saved_models/research/member_1.pt"
$nnCkpt2 = "./saved_models/research/member_2.pt"
$nnCkpt3 = "./saved_models/research/member_3.pt"
$nnOut = "./logs/nn_rollout_handoff/run01"

foreach ($nnAsset in @($nnModel, "$nnModel.json", $nnCkpt1, $nnCkpt2, $nnCkpt3)) {
    if (-not (Test-Path -LiteralPath $nnAsset -PathType Leaf)) {
        throw "缺少模型资产：$nnAsset"
    }
}
if (Test-Path -LiteralPath $nnOut) { throw "输出目录已存在，请换一个新的运行编号：$nnOut" }
New-Item -ItemType Directory -Path $nnOut -ErrorAction Stop | Out-Null

$nnRoot = @(
    "--rollout-model", $nnModel,
    "--plan-index", "0", "--seed", "61444", "--run-idx", "1200",
    "--root-turn", "30", "--root-stage", "Train", "--root-policy", "handwritten",
    "--workers", "2", "--radical-factor-max", "1.4", "--pt-favor-rate", "2.0"
)
$nnGpu = @(
    "--sidecar", "scripts/ramen_nn/bench_sidecar.py", "--python", "python",
    "--checkpoint", $nnCkpt1, "--checkpoint", $nnCkpt2, "--checkpoint", $nnCkpt3,
    "--batch", "512"
)
~~~

这是历史性能根的诊断重放示例，不是新的盲测世界。旧模型在新规则下的表现也不能与旧日志绝对分直接比较。每条命令结束应检查退出码，任一失败就保留日志、停止该次验收；不要只看最后一个命令是否成功。

### 小预算正确性

~~~powershell
& $nnBench --mode cpu-flat @nnRoot --search-n 8 --raw-csv "$nnOut/cpu_raw.csv" --decision-csv "$nnOut/cpu_decisions.csv"
if ($LASTEXITCODE -ne 0) { throw "CPU 固定根失败" }

& $nnBench --mode gpu-wave @nnRoot @nnGpu --search-n 8 --raw-csv "$nnOut/gpu_raw.csv" --decision-csv "$nnOut/gpu_decisions.csv"
if ($LASTEXITCODE -ne 0) { throw "GPU 固定根失败" }

python scripts/ramen_nn/compare_root_bench.py raw "$nnOut/cpu_raw.csv" "$nnOut/gpu_raw.csv"
if ($LASTEXITCODE -ne 0) { throw "逐 rollout 结果不一致" }

python scripts/ramen_nn/compare_root_bench.py decisions "$nnOut/cpu_decisions.csv" "$nnOut/gpu_decisions.csv"
if ($LASTEXITCODE -ne 0) { throw "逐决策记录不一致" }
~~~

另以 --mode cpu-candidate 跑同一根，才能把 CPU 并行粒度收益与 GPU 收益分开。compare 模式会同时保留快照并同步对拍，适合正确性检查，不用其墙钟宣称性能。

缓存与 Graph 的组合示例：

~~~powershell
& $nnBench --mode gpu-wave @nnRoot @nnGpu --search-n 8 --policy-cache --sidecar-graph --raw-csv "$nnOut/optimized_raw.csv" --decision-csv "$nnOut/optimized_decisions.csv"
if ($LASTEXITCODE -ne 0) { throw "优化路径失败" }

python scripts/ramen_nn/compare_root_bench.py raw "$nnOut/gpu_raw.csv" "$nnOut/optimized_raw.csv"
if ($LASTEXITCODE -ne 0) { throw "优化前后逐 rollout 结果不一致" }

python scripts/ramen_nn/compare_root_bench.py decisions "$nnOut/gpu_decisions.csv" "$nnOut/optimized_decisions.csv"
if ($LASTEXITCODE -ne 0) { throw "优化前后逐决策记录不一致" }
~~~

--policy-cache、--adaptive-batch、--sidecar-graph 默认均关闭。Graph 与自适应 batch 互斥；按根自适应档位应单独比较。性能测量移除逐决策记录，并在相同负载下交错重复基线与优化臂，单列加载、侧车启动、预热和图捕获成本。

### 叶估值接线

~~~powershell
& $nnBench --mode value-check @nnRoot @nnGpu --value-rows 16
if ($LASTEXITCODE -ne 0) { throw "value 解码对拍失败" }

& $nnBench --mode leaf-pilot @nnRoot @nnGpu --leaf-h 4 --leaf-h 8 --full-n 16 --select-n 8 --leaf-objective mean --leaf-out "$nnOut/leaf"
if ($LASTEXITCODE -ne 0) { throw "叶截断固定根失败" }
~~~

叶试点以 --full-n / --select-n 定预算，完整参考列拆成选择列与独立审计列。上述小预算只检查接线，不足以判断排序精度或策略强弱。--leaf-diag 会在叶状态继续跑到真实终局以比较配对残差；带此开关的耗时不代表截断性能。

其他已交付模式：

| 模式 | 用途与边界 |
|---|---|
| sidecar-reuse | 同一侧车执行根 A → 根 B → 根 A，检查状态复用 |
| teacher-consistency | 生产教师装配下 CPU/GPU 逐步比较；整局接线比单根昂贵 |
| teacher-game | 完整续跑的整局成本；--handwritten-rollout 为手写对照，--steps-csv 记录外层决策 |
| root-scan | 用冻结策略扫描决策点，配 --leaf-out 保存清单 |
| game-smoke | 叶实验的整局接线，支持 --game-route uniform、hybrid-y3-full、hybrid-y3-full-mean 与 --route-csv |

独立的 nn_rollout_probe 通过原有 NN_ROLLOUT_MODEL / RUNS / SEED / SEARCH_N / STAGES / CONTROL 环境参数配置。本次新增 NN_ROLLOUT_MAX_TURN：不设置时为 None，网络可用于全局续跑；设置时必须是非负整数，表示包含上限回合的绝对游戏回合窗口。探针默认 SEARCH_N=4、STAGES=train，只用于小预算计价。环境参数属于使用者选定的运行配置，文中固定根示例不需要修改这些参数。

hybrid-y3-full 的第三年地区根使用完整续跑和原 rf，其余根截断并用 mean；hybrid-y3-full-mean 还把第三年地区根改为 mean。它们是不同算法，不可作为等价性能优化比较。生产拉面 max_depth > 0 仍受能力守卫限制；上述实验不等于把归档 CPU 叶估值接入了默认搜索。

## 历史证据与勘误

以下路径来自原研究工作区，通常被 Git 忽略；新克隆没有这些文件是预期情况。交接资产目录应另列可获得的附件，不能把不可获得的本地路径写成已随 PR 交付。

| 历史材料 | 支持的结论 |
|---|---|
| logs/perf_nn256_0915/bench_game.log | 旧配置 n256：21,037,772 有效推理行、45,407 波次；NN 续跑 706.6 s、手写 11.2 s，仅一对整局 |
| logs/perf_opt1_0915/07_game_arms.log | 缓存资格内 3,030,790 次，命中 3,029,766 次；整体请求减少约 14.4% |
| logs/perf_opt1_0915/review_fix/ | 缓存生命周期、参数支持矩阵、零请求统计修复及前后对拍 |
| logs/gpu_probe_0915/REPORT.md | 缓存开、B512条件下，Graph 的三根历史测量约 1.18–1.21 倍；稳定整局幅度未确定 |
| logs/leaf_value_allhistory_0916/、logs/leaf_hybrid_y3_0917/ | 旧模型、旧规则下的叶替换试点，不能证明当前版本效果 |

- **99.966% 的分母只包含缓存资格内的 SpecialSelect 推理决策。** 整体有效请求从 21,037,772 降至 18,008,006，减少约 14.4%；不是省掉 99.966% 的全部推理。
- **“约 2 亿次推理”是旧配置下的状态样本行数估算。** 有效行、补零后的物理格位、批前向调用次数、成员前向次数是不同量。波次后端每波调用一次侧车；三成员集成又包含各成员前向。
- 侧车往返包含序列化、管道、主机/设备拷贝及前向，不等于纯 GPU kernel 时间。旧整局墙钟漂移较大，不能把跨时段的最快值比成稳定加速倍数。
- 当前证据说明已测完整 NN 续跑成本高，尚未证明能满足客户端时延预算；不支持“任何后端都物理不可能实时运行”。
- 单局或少数固定根可证明接线和局部一致性，不能证明 greedy(Q^NN) 普遍强于手写续跑搜索。等 rollout 预算与等实际时间预算是两个研究问题。

当时的性能计划与叶估值任务书留在研究工作区，不随本交接提交；以本文区分已交接实现、历史实验与本次验证。

## 资产与证据交付

源码仓库应保存入口说明、比较脚本、可生成的小模型 fixture 与必要的小型预期结果。

ramen_root_bench 的单元测试默认不依赖任何本地权重：缓存生命周期与叶响应坏行拒绝用 testsupport/onnx_fixture.rs 现场生成常数模型。常数模型的输出与输入无关，因此只覆盖缓存的写入、命中、失效与计数，测不出「命中后用了别处写入的 policy」；命中内容的正确性由真实模型的 gpu-wave 开缓存对拍保证（见验收记录）。唯一需要真实资产的是混合路由对拍测试 hybrid_matches_uniform_arms_per_root，它默认忽略；准备好 CUDA、侧车 Python 依赖、saved_models/arms/ens_R8A_g123.onnx（含旁车 JSON）及 target/arm_R8A_seed{1,2,3}/step_060000.pt 后，用 `cargo test --release -p umasim --features "cli onnx" --bin ramen_root_bench -- --ignored` 运行。checkpoint 放在 target/ 下会被 `cargo clean` 删除，请另行保留原件。真实模型、checkpoint、大型 raw/trace 和完整数据集通过单独的版本化附件交付，不作为普通测试的隐式前提。

模型附件至少记录：模型 ID、成员列表、模型结构、输入/输出契约、归一化配置、导出代码版本，以及相互对应的 ONNX、旁车 JSON、三个 checkpoint。实验附件记录：规则与工具提交、配置原文、完整命令、实际根身份、退出码、有效行/物理格位/批调用计数和原始结果。

文件和实验结果按实际字节、数组或字段直接比较，报告具体差异；不新增 SHA256、MD5、FNV 或任何替代哈希/指纹流程。现有 Git revision 可用于记录代码版本，不能用文件大小或时间戳冒充内容一致。再次交接或生成下一会话任务时应保留这条约束。

## 本次基点验收记录

本节只记录上游 2ec6b49 基点上的新验证，不用上表旧日志代填。2026-10-05 在本机（RTX 4070 Laptop、PyTorch 2.6 + CUDA）执行；编译一律 release、thin LTO、`-j 2`。小预算运行日志在被忽略的 logs/nn_rollout_handoff/run01/，未随 PR 提交。

| 项目 | 状态与证据 |
|---|---|
| Release 构建：CPU默认路径、ONNX研究入口、feature边界 | 通过：默认 feature 的 umasim lib；`cli onnx` 下 ramen_root_bench、nn_rollout_probe、umasim 主程序；umaai `cargo check`。全部退出码 0 |
| 默认手写/PT 行为对拍，规则/RNG/手写实现无意外差异 | 通过：上游原有整局快照测试原样保留且通过（评分 61332、五维、SpecialSelect 调用 29 / 重搜 0），game/、RNG、手写策略文件相对 2ec6b49 无改动 |
| 窗口边界、严格失败、批表缺项、探针中性、参数拒绝测试 | umasim lib 439 通过、1 失败、7 忽略；唯一失败 output::reason::tests::test_color_thresholds 是 2ec6b49 上已有的失败，与本交接无关。ramen_root_bench bin 测试 16 通过、1 忽略（GPU 混合路由，见下行），含参数拒绝矩阵 root_policy_support / output_flags_support / switch_support_matrix |
| Python比较器与前向探针自检 | 通过：test_compare_root_bench 3/3；probe.py unit 合成回归 6/6；probe.py selftest（R8A 成员）注入 7 种错误全部拦住、基线干净 |
| 真实模型CPU/GPU小预算对拍、value反归一化 | 资产 ens_R8A_g123（ONNX + 三个 step_060000 成员 checkpoint）。固定根 plan0 / seed 61444 / run 1200 / t30 Train，9 候选 × search_n 8：cpu-flat、cpu-candidate、gpu-wave 逐 rollout 72 条 seed/score/score_pt 完全一致，逐决策 6098 条一致。value-check 16 个真实局面：归一化空间最大差 1.52e-6，反归一化后 0.0064 分，policy logits 2.86e-6。teacher-consistency（search_n 2）整局 181 步 CPU/GPU 逐步动作、候选顺序、RNG 探针与终局评分全同。nn_rollout_probe（窗口 40）跑通 |
| 缓存/Graph/自适应batch与叶实验接线 | 通过：gpu-wave 开 policy 缓存 + Graph（命中 977 次）、开自适应 batch，均与基线逐 rollout / 逐决策完全一致；leaf-pilot h4/h8 接线正常；sidecar-reuse A→B→A 两次 A 逐字段一致；hybrid_matches_uniform_arms_per_root 以 `--ignored` 运行 23 项全对。未运行：teacher-game、game-smoke、root-scan 的整局批量 |
| 对外模型及历史证据附件 | ens_R8A_g123 的 ONNX 与旁车 JSON 已在本 fork 的 Release models-r8a-0920 公开；三个成员 PyTorch checkpoint 与历史日志尚未上传，复现 GPU 路径需另行索取 |

研究工具的交付门槛是可复现、默认行为守恒和失败可见；不要求通过调参制造正向成绩。新的性能倍数或策略优势应另附固定预算、独立评测世界与事先确定判读规则的实验。
