# NN 与 MCTS 配合：老版 C++/CUDA 模拟器对照分析

> 适用范围：下文保留上游当时的实现与测量记录，不代表本分支当前能力。本分支继续使用
> `trainer/ramen_rollout_trainer.rs` 的训练器、`RamenBatchRollout` 批量后端及
> `ramen_root_bench` 的 GPU/CUDA 侧车入口；已支持网络回合窗口、严格失败传播与批量推理。
> 合并后的离线 main 和 `nn_rollout_probe` 通过本分支 `with_nn_rollout` 接入 CPU ONNX；
> 下文“没有批量推理”等缺口描述及成本数字仅属于原上游对照，不能直接套用到本分支。

核对于 2026-09-25。**外部参考工程**指本地旧版赛马娘模拟器（C++/CUDA，LARC 剧本，非本仓库）；
下文中它的路径以该工程根为基准（`UmaSimulator/...`）。本仓库路径以 workspace 根为基准。

---

## 技术摘要（TL;DR）

- **老版不是 MCTS**：没有树、没有节点、没有访问次数、没有先验（policy prior）、没有虚拟损失。
  它的真实形态是「**根候选穷举 × 每候选 N 次（NN policy 引导的有限深度 rollout + NN value 叶估值）
  × 激进度加权聚合 → argmax**」。
- 它跑得动的代价控制只有两条：**rollout 截断在 6 步**（不是跑到终局）＋ **batch=512 lockstep**
  （把同一局复制 512 份同步推进，一次前向算整批）。单次 rollout 的 NN 前向调用因此只有
  `7 次 / 512 局 ≈ 0.014 次/局`。
- 本仓库实测对照（同一条路）：把 MCTS 的 rollout 基策换成 NN（**终局 rollout + batch=1**）——
  13 888 ms/局 vs 手写 rollout 122 ms/局，**约 114×**；折算到生产 `search_n`/阶段是**小时级/局**。
- **结论：这条路可行与否取决于「截断深度 × 批摊薄」，与推理后端或 GPU 无关。** GPU 只改单次前向
  成本，改不了调用次数；不截断、不成批时它救不回来。
- 本仓库要复制它的可行性，缺三块拼图（互相依赖）：拉面**叶估值器 + 放开截断搜索**、**批量推理接线**、
  **批处理下仍逐位可复现的 CRN**。
- 最便宜的推进阶梯：①给 `infer` 加计时 → ②做「**rollout 前 k 步走 NN、之后切回手写**」的探针
  （不需要截断、不需要 leaf 估值器，rollout 仍跑到终局）→ ③才谈截断 + 批量。
- 可借鉴：NN 当**候选先验/剪枝**（每决策 1 次推理）、把搜索旋钮喂进网络、`rf` 按剩余回合缩放。
  不可照搬：无树/等预算/无 CRN/`mutex` 串行 GPU/每 kernel 同步/隐式「必须传满批」契约。

---

## 1. 老版实现（代码事实）

### 1.1 一次决策的数据流

```
Search::runSearch                          Search.cpp:111
 ├─ 枚举候选 (buyBuff 方案 × train 项)      Search.cpp:131-146  → allChoicesValue[4][10]
 ├─ 每候选 evaluateSingleAction             Search.cpp:165
 │   ├─ 起 threadNumInGame 个线程（每候选新建/销毁）  Search.cpp:190-210
 │   └─ evaluateSingleActionThread          Search.cpp:264-304
 │        for 每个 batch（batchSize 局 lockstep）
 │          assign(batchSize, game) + 施加候选动作   Search.cpp:273-279
 │          for depth in 0..maxDepth:               Search.cpp:281
 │            evaluateSelf(1) → policy 头 argmax 走一步  Search.cpp:283-291
 │          evaluateSelf(0) → value 头               Search.cpp:297
 │   └─ 分数直方图 + 激进加权                Search.cpp:225-262
 └─ 取 value 最大的候选                     Search.cpp:148-161
```

### 1.2 NN 介入的三个位置

| 位置 | 用什么 | 代码 |
|---|---|---|
| rollout 内**每一步**决策 | policy 头 argmax（带 `isLegal` 掩码） | `Search.cpp:281-296` + `Evaluator.cpp:93-141` |
| **叶估值**（到 maxDepth 或终局） | value 头；终局用真实 `finalScore()` | `Evaluator.cpp:51-64` |
| 根候选比较 | `value` 通道 argmax | `Search.cpp:148-161` |

**没有先验项**（policy 不写入任何"节点"），**没有访问次数分布**，因此没有 AlphaZero 式的
policy target 与树复用。

### 1.3 批与调度

- 「批」= 同一局复制 `batchSize` 份、同线程内 lockstep 推进（`Search.cpp:271-301`）。
- **无请求队列、无凑批等待、无超时发车、无虚拟损失、无独立推理线程**；搜索线程同步阻塞等 GPU。
- GPU 由共享单例 `Model` 的一把 `std::mutex` 串行化（`Model.h:170`、`Model_cuda.cpp:266`）。
- `searchN` 被向上取整到 `threadNumInGame × batchSize` 的整数倍（`Search.cpp:97-107`）。
- 在线默认（`ConfigTemplate/aiConfig.json`）：`searchN=2000`（→2048）、`batchSize=512`、
  `threadNum=4`、`searchDepth=6`、`radicalFactor=5`。自对弈另用 `batchsize=1024`、
  `threadNumInner=1`、外层 8 线程、`searchN` 按 lognormal 采样。

每个候选的前向次数 = `threadNum × batchNum × (depth+1)` = `4 × 1 × 7 = 28` 次（每次 batch 512）；
一回合 10–40 个候选 → **约 280–1120 次前向/决策**，约 14 万–57 万次样本求值/决策。

### 1.4 网络与训练口径

- `ems` 网络：输入 **2481**（global 357 + card 7×72 + person 20×81）→ 输出 **21**
  （policy 18 + value 3）；1 层 encoder（单头注意力、`relu(QKᵀ)` 无 softmax）+ 3 层 ResNet MLP；
  约 **0.7M 参数**。结构与超参在 C++ 侧硬编码（`Model.h:56-84`）并与 `training/model.py` 对齐。
- **value 头 3 路**：`scoreMean / scoreStdev / value`，反归一 `30000+200x / 100x / 30000+200x`
  ——与本仓库拉面模型的 `value[3]`（mean/stdev/high）同源思路。
- **输入含搜索参数**：3 个通道里只把 `radicalFactor` 真正喂进去（value 头按激进度条件化），
  `maxDepth`/`samplingNum` 两个通道写死为 0（`NNInput.cpp:351-356`）。
- **policy 标签 = 各候选 value 的 softmax**（温度 `policyDelta=50`，再按 train / buff 维度边缘化，
  `TrainingSample.cpp:23-104`），**不是访问次数分布**；value 标签 = 最优候选的 `(mean, stdev, value)`。
- 无 Dirichlet 噪声、无根节点温度采样；"探索"来自自对弈时随机化搜索参数（`SelfplayThread.cpp:41-77`）。
- 聚合：直方图 + `w = pow(累积概率 r, rf)` 加权平均，`rf` 按剩余回合缩放（`Search.cpp:33-39`、`:243-261`）。

### 1.5 已知缺陷（选择时留意）

- 无树 → 无访问次数、无树复用；subsequent depth 全靠策略 rollout 近似。
- `batchSize` 改变会改同批内随机数交错顺序 → 结果对 batchSize 敏感；种子来自 `random_device`
  → **不可复现**。
- `Model::evaluate` 忽略入参 `gameNum`，实际按成员 `batchSize` 处理 → 隐含"必须传满批"契约。
- 每个 kernel 后 `cudaDeviceSynchronize()`（一次前向 15+ 次全设备同步）；`ModelCudaBuf` 析构不
  `cudaFree`（显存泄漏）；无 CUDA stream / pinned memory。
- `BACKEND_EIGEN` 只声明无实现（选中链接失败）；libtorch 示例模型仍是旧 2341 维，与当前 2481 维不兼容。
- `runSearch` 在 `turn==40/42`（`buyBuffChoiceNum` 返回 -1）可能返回未初始化的 `Action`。
- `packages/` 下的 OnnxRuntime-DirectML 是**无人引用的死包**（代码中零使用点）。

---

## 2. 与当前版本的逐项对比

### 2.1 rollout 策略本身

| 维度 | 老版 | 本仓库（拉面） |
|---|---|---|
| rollout 决策器 | NN policy argmax；无网络退化手写 | 手写 `RecommendedRamenTrainer`；NN 档 = `RamenRolloutTrainer::Nn`（实验） |
| rollout 走多远 | `maxDepth=6` 截断（可配到 67） | 跑到终局（约 171 个决策点）；拉面硬拒 `max_depth>0` |
| 叶估值 | NN value 头（3 路全用） | 无 leaf eval；终局真实 `calc_score` + 25 维终端统计 |
| 候选预算分配 | 每候选等预算 | UCB 分配（`use_ucb` / `cpuct=1.0` / `group=512` / `expected_stdev=15000`） |
| 聚合 | 直方图 + `pow(r, rf)` 加权，`rf` 随剩余回合缩放 | radical 加权 mean（`radical_factor_max=1.4`）+ `score_pt` 轴分离 |
| CRN | 无 | 有（`RolloutSeeds` 共享第 j 轮种子 + 规则层 `rule_master`） |
| 可复现 | 不可复现 | 逐位可复现 + 测试守门 |
| 树 / 访问次数 | 无树、根一层穷举 | 扁平但带 UCB 预算分配 |

### 2.2 支撑它的工程与训练口径

| 维度 | 老版 | 本仓库 |
|---|---|---|
| 批 | batch=512 lockstep（同局复制） | batch=1 固定形状；`rollout_batch_size` 从未被读取 |
| 单次 rollout 的 NN 调用 | 7 次 / 512 局 ≈ 0.014 次/局 | 约 171 次/局（实验档） |
| 每决策前向次数 | 约 280–1120 次（batch 512） | 约 200 次（batch 1，仅 nn/hint 模式） |
| 并行/调度 | 候选 × 线程内切分；GPU 靠 mutex 串行 | rayon 候选间 + 候选项内双层并行 |
| 推理后端 | 自研 cuBLAS + 手写 kernel | tract-onnx CPU，固定 shape 编译 |
| 模型 I/O | 2481 → 21 | 754 → 245（policy 234 + choice 8 + value 3） |
| 搜索参数进模型 | `radicalFactor` 条件化 value 头 | 无搜索参数通道 |
| policy 标签 | 候选 value 的 softmax（温度 50） | Bayesian bootstrap「最优概率」 |
| value 语义 | 3 路绝对分 | 3 路 cross-fit（"先选一步、随后按采集策略走完"），决策侧未消费 |

**不可比因素（做任何数字对照前先记住）**：剧本不同（老版 67 回合、拉面 77 回合）、评分与 PT
口径不同、网络规模不同、硬件不同。两边只有**结构上的可比性**，没有绝对数字的可比性。

---

## 3. 对本仓库的启示

### 3.1 结论校准：不是"NN 进 rollout 不可行"，而是"没有截断和批量时不可行"

本仓库实测（`nn_rollout_probe`，`search_n=4` + 只搜 train、单局、24 线程）：

| 档 | 分数 | 单局耗时 |
|---|---:|---:|
| 手写 rollout（现状） | 53549 | 122 ms |
| NN rollout（实验） | 56430 | 13 888 ms |

差约 **114×**。差在哪里是纯算术：

```
单次 rollout 的 NN 前向调用数
  老版：7 次 / 512 局 ≈ 0.014 次/局
  本仓库：约 171 次 / 1 局 = 171 次/局      → 量级差约 1.2 万倍
```

按同一阶段集合把 `search_n` 由 4 外推到生产 8192（×2048）≈ **7.9 小时/局**；再算上
`ramen` / `region` 阶段只会更久。**这条路要成立，必须同时做「深度截断」与「批摊薄」两件事，
换推理后端或上 GPU 都只改单次前向成本，动不了调用次数。**

### 3.2 三块互相依赖的拼图

| 缺口 | 现状 | 依赖 |
|---|---|---|
| 拉面截断搜索 + leaf 估值器 | `SUPPORTS_TRUNCATED_LEAF = false`，`max_depth>0` 被搜索入口硬拒 | 需要 NN value 作叶值。value 的语义是"先选一步、随后按**采集 rollout 基策**走完"，而采集时的基策就是手写 REC，**与本仓库搜索的 rollout 基策同源，语义大体对齐**；但"先选一步"那一步仍有错位，必须实测而非假设（见 `scripts/ramen_nn/DESIGN.md` §2） |
| 批量推理 | `rollout_batch_size` 空转；模型按 batch=1 固定形状编译 | 需要搜索侧产生"同深度的并发批次"。老版靠 lockstep 复制同一局做到；本仓库各 rollout 独立且深度不同，要批就得改 rollout 调度 |
| 批下的逐位可复现 | CRN 是既有资产（等效样本 3.5–7×） | 任何批聚合顺序变化都可能破坏 CRN 配对与逐位一致性，必须守住现有测试 |

四件事（截断 → 叶值 → 网络 → 批量）是"先有鸡才有蛋"的环，老版是一次性设计在一起的，
这也是它代码简单（无队列、无虚拟损失）的原因。

### 3.3 低成本可借鉴

1. **NN 当候选先验/剪枝**：老版在 rollout 与根候选上都用 policy 做"合法 + argmax"过滤。
   本仓库可把 NN policy 用作候选剪枝——每决策 1 次推理（约 200 次/局，成本可忽略），
   按候选数比例降搜索成本；这是当前架构下性价比最高、且**不需要批量**的用法。
2. **把搜索旋钮喂进网络**：老版让 value 头按 `radicalFactor` 条件化。本仓库聚合有
   `radical_factor_max` / `pt_favor_rate` 等旋钮，一旦改动，value 头语义就与部署不一致。
   将来要用 value（leaf 或剪枝打分）之前，先决定"入特征"还是"锁死不调"。
3. **`rf` 按剩余回合缩放**：老版 `rf` 随剩余回合变化；本仓库目前是固定档 1.4。
   低风险、可直接做配对实验。

### 3.4 本仓库更强、不应退回去的地方

- **UCB 预算分配**：老版每候选等预算；本仓库能把预算压到更值得的候选上。
- **CRN + 逐位可复现**：老版无 CRN，结果对 `batchSize` 敏感且不可复现。
- **诚实终端观测**：25 维终端统计 + 真实 `calc_score`；老版叶值全部是网络估计，无真值兜底。

### 3.5 不要照搬

无树/无访问次数；`std::mutex` 串行 GPU + 每 kernel 同步；析构不释放显存；
`Model::evaluate` 忽略 batch 入参那种隐式契约（本仓库改批量时要把 batch 显式化并校验）。

---

## 4. 建议的验证阶梯

- **Step 1（必做，最便宜）**：给 `RamenNnTrainer::infer` 加计时，拿到 μs/次；所有后续推算的底座。
- **Step 2（新增，绕开硬约束）**：在 `nn_rollout_probe` 上做变体——**rollout 前 k 步走 NN、
  之后切回手写**。它**不需要截断搜索、不需要 leaf 估值器**（rollout 仍跑到终局），只是中途换基策，
  即可实测"NN 参与程度 vs 成本/分数"曲线，直接回答"NN 值不值得进搜索"。
- **Step 3（重工程，最后）**：只有当 Step 2 有明显收益，才动"截断 + leaf 估值 + 批量"三件套；
  GPU 排在批量之后（老版那 280–1120 次/决策的批量前向才是 GPU 的正当理由，本仓库量级不需要）。

### 一句话总结

老版证明的是「**批 + 截断**」的价值，不是「GPU + 树搜索」的价值。它的形态（无树、等预算、无 CRN）
不该学；它的成本控制手段（短深度、lockstep 大批、policy 渐进式引导）才是可迁移的部分。

---

## 5. 本次为量测新增的开关（默认行为不变）

| 文件 | 改动 |
|---|---|
| `crates/umasim/src/trainer/mod.rs` | 新增 `RamenRolloutTrainer`（`Handwritten` / `Nn`，转发 `Trainer<RamenGame>`；`Nn` 变体 `cfg(onnx)`） |
| `crates/umasim/src/search/searchable.rs` | `RamenGame::RolloutTrainer` 切到该枚举，默认仍 `handwritten()` |
| `crates/umasim/src/trainer/ramen_mcts_trainer.rs` | `with_friend_complete_required` 包一层枚举 |
| `crates/umasim/src/main.rs` | 离线 main：leaf-eval 段隔离到非拉面剧本；ramen 分支复用 `rollout_evaluator` 键解释为 rollout 基策 |
| `crates/umasim/tools/data_collection/nn_rollout_probe.rs` | 新增探针（env 驱动，跑整局并报分数/耗时） |
| `gamedata/default_config.toml` | 补 `rollout_evaluator` 在拉面分支的语义注释 |

- 离线切换方式：`[mcts] rollout_evaluator = "nn"` + `ramen_nn_model_path = "<模型路径>"`；
  `"handwritten"`（默认）= 现状不变；未知值报错。通道层（umaai）未接，只作用于离线侧。
- 探针用法：`NN_ROLLOUT_MODEL=<onnx>` `NN_ROLLOUT_SEARCH_N=4` `NN_ROLLOUT_STAGES=train`
  `NN_ROLLOUT_CONTROL=1`，`cargo run --release -p umasim --features onnx --bin nn_rollout_probe`。
- 实测口径提醒：探针的 `search_n` 极小，**输出的分数不是策略验收口径**；且两个 arm 的随机流消耗
  不同，不是逐位同世界，分数差不能当策略对比。
