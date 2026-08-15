# GPU 加速计划：CPU 局面生成 → 模型训练 → GPU 推理加速

## 概述

本项目是一个回合制游戏（温泉马娘）模拟器，当前使用 MCTS（蒙特卡洛树搜索）进行决策，搜索过程中需要产生大量随机局面用于模拟。核心思路是：

1. **保持 CPU 的局面生成逻辑**（游戏状态机 + 随机事件逻辑，难以 GPU 化）
2. **用 CPU 逻辑生成训练素材**（每回合状态 → 动作 → 终局价值）
3. **训练神经网络模型**（模仿学习 + 强化学习）
4. **用 GPU 推理加速搜索**（leaf eval 用神经网络估值替代 CPU 完整 rollout）

---

## 第一部分：代码库架构梳理

### 1.1 现有模块清单

| 模块 | 文件 | 职责 | 现状 |
|------|------|------|------|
| 游戏模拟层 | `crates/umasim/src/game/onsen/game.rs` | 温泉游戏状态机，包含训练/比赛/挖掘/事件/温泉券等逻辑 | ✅ 已实现 |
| 游戏动作枚举 | `crates/umasim/src/game/onsen/action.rs` | 动作类型定义（Train/Race/Dig/Upgrade/UseTicket/Choice 等） | ✅ 已实现 |
| 搜索核心 | `crates/umasim/src/search/flat_search.rs` | 扁平蒙特卡洛搜索（Rayon 并行化） | ✅ 已实现 |
| 搜索配置 | `crates/umasim/src/search/config.rs` | search_n / max_depth / UCB 参数 | ✅ 已实现 |
| 搜索结果 | `crates/umasim/src/search/result.rs` | 动作分数统计 / SearchOutput | ✅ 已实现 |
| MCTS 训练员 | `crates/umasim/src/trainer/mcts_trainer.rs` | 蒙特卡洛搜索决策 | ✅ 已实现 |
| 手写评估器 | `crates/umasim/src/neural/handwritten_evaluator.rs` | 启发式策略 + 估值 | ✅ 已实现 |
| 神经网络评估器 | `crates/umasim/src/neural/neural_net_evaluator.rs` | ONNX 模型推理（CPU, tract-onnx） | ✅ 已实现 |
| 评估器 Trait | `crates/umasim/src/neural/evaluator.rs` | `Evaluator<OnsenGame>` Trait 定义 | ✅ 已实现 |
| 样本收集器 | `crates/umasim/src/sample_collector.rs` | 回合数据记录 → 训练样本 | ✅ 已实现 |
| 数据收集训练员 | `crates/umasim/src/trainer/collector_trainer.rs` | HandwrittenEvaluator + SampleCollector | ✅ 已实现 |
| 神经网络训练员 | `crates/umasim/src/trainer/neural_net_trainer.rs` | ONNX 模型直接决策（无搜索） | ✅ 已实现 |
| 训练样本格式 | `crates/umasim/src/training_sample.rs` | 1121 输入 + 50 policy + 8 choice + 3 value | ✅ 已实现 |
| 训练数据生成器 | `crates/umasim/src/bin/generate_training_data.rs` | 跑大量模拟 → 筛选 Top N% 样本 | ✅ 已实现 |
| 特征提取 | `game.rs:extract_nn_features()` | 当前状态 → 1121 维 f32 向量 | ✅ 已实现 |

### 1.2 数据流图示

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       搜索时：MCTS 决策流程                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MctsTrainer::select_action()                                             │
│        │                                                                 │
│        ▼                                                                 │
│  FlatSearch::search()  ←─ Rayon 并行化 CPU rollout                        │
│        │                                                                 │
│        ├─ 对每个合法动作：game.clone() → simulate_n() → 跑到终局           │
│        │    (max_depth=0 时)                                                 │
│        │                                                                 │
│        └─ 或：simulate_until_leaf() → extract_nn_features()               │
│              (max_depth>0 时)                                               │
│              └─ ThreadLocalNeuralNetLeafEvaluator                          │
│                 └─ CPU (tract-onnx) 推理 score_mean                      │
│        │                                                                 │
│        ▼                                                                 │
│  SearchOutput（各动作分数分布 → 选最优）                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    训练数据收集：CollectorTrainer                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CollectorTrainer::select_action()                                        │
│        │                                                                 │
│        ├─ 提取特征: game.extract_nn_features() → [f32; 1121]              │
│        ├─ 决策: HandwrittenEvaluator::select_action() → 动作索引           │
│        └─ 记录: SampleCollector::record_turn()                            │
│                                                                          │
│  游戏结束后：                                                              │
│        ├─ set_final_score(score)                                          │
│        └─ finalize() → Vec<TrainingSample>                                │
│              ├─ policy_target: one-hot (50 维)                           │
│              ├─ choice_target: one-hot (8 维)                            │
│              └─ value_target: [final_score, 500, final_score]           │
│                                                                          │
│  generate_training_data.rs:                                                │
│        └─ 跑 N 局 → 按分数排序 → 取 Top M% → 保存为 .bin (bincode)          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 关键瓶颈分析

| 瓶颈 | 当前实现 | 问题 |
|------|----------|------|
| **搜索速度** | `max_depth=0` 时每次搜索 = search_n 次完整游戏模拟（~78 回合） | 每局搜索极其昂贵，Rayon 只能水平扩展 |
| **leaf eval 推理** | `max_depth>0` 时使用 CPU (tract-onnx) 推理 | 已实现，但 CPU 推理比 GPU 慢 10-50 倍 |
| **训练数据质量** | Policy target 是 one-hot 的手写策略动作 | 不够精确，没有搜索后验概率分布 |
| **Value target 质量** | 直接用 `final_score` 作为 value | 噪声大，没有考虑方差 / 搜索价值估计 |
| **缺失训练 pipeline** | 只有 Rust 端数据收集，没有训练脚本 | 需要 Python 训练脚本 + 模型导出 ONNX |

---

## 第二部分：详细改造计划

### Phase 0：现状分析与数据格式统一（不需代码修改，只需理解）

目标：确认当前训练数据格式与模型输入输出一致。

**现状确认清单：**

- [ ] `TrainingSample.nn_input` = 1121 维 f32（对齐 `extract_nn_features`）
- [ ] `TrainingSample.policy_target` = 50 维 f32（对齐 `POLICY_DIM`）
  - 当前是 one-hot（选中动作 = 1.0, 其他 = 0.0），质量不够好
  - **改进方向**：改为 MCTS 搜索后的 softmax 概率分布
- [ ] `TrainingSample.choice_target` = 8 维 f32（对齐 `CHOICE_DIM`）
  - 当前也是 one-hot，同上
- [ ] `TrainingSample.value_target` = 3 维 f32
  - `[0]` = score_mean（当前：final_score 的均值）
  - `[1]` = score_stdev（当前：固定 500）
  - `[2]` = value（当前：final_score）
  - **改进方向**：用多次搜索得到的均值 / 方差替代固定值

---

### Phase 1：训练数据生成流程升级（生成更高质量的训练数据）

**目标：** 用 MCTS 搜索来标注训练数据，而不是只用手写策略的 one-hot。

**修改文件清单：**

| 文件 | 改动内容 | 优先级 |
|------|----------|--------|
| `crates/umasim/src/bin/generate_mcts_training_data.rs` | **新建**。类似 `generate_training_data.rs`，但使用 MCTS 搜索做决策 | 高 |
| `crates/umasim/src/trainer/mcts_collector_trainer.rs` | **新建**。包装 MCTS 搜索 + 样本收集器，记录搜索后的 Policy/Value 分布 | 高 |
| `crates/umasim/src/training_sample.rs` | 增加 `with_policy_distribution()` / `with_value_stats()` 构造方法 | 中 |

**详细设计：**

#### 1.1 MctsCollectorTrainer（新建文件）

类似 `CollectorTrainer`，但用 `FlatSearch` 做决策：

```rust
// 伪代码，仅示意
pub struct MctsCollectorTrainer {
    search: FlatSearch,                  // MCTS 搜索器
    evaluator: HandwrittenEvaluator,      // 手写评估器（温泉/装备用）
    collector: RefCell<SampleCollector>, // 样本收集器
    // 额外保存每回合的搜索后验信息（用于生成 policy/value target）
    search_results: RefCell<Vec<MctsSearchSnapshot>>,
}

// 每回合搜索后保存的信息
struct MctsSearchSnapshot {
    features: Vec<f32>,                 // 1121 维输入
    action_scores: Vec<(f64, f64)>,     // 每个动作 (mean, weighted_mean)
    visited_action_idx: usize,          // 实际选中的动作
    final_score_mean: f64,              // 该状态下的平均终局分数（多次 rollout 均值）
    final_score_stdev: f64,             // 终局分数标准差
}
```

#### 1.2 Policy target 生成

**原方案**（质量差）：one-hot 编码 → `policy_target[action_idx] = 1.0`

**新方案**（质量高）：

```rust
// 对搜索后每个动作的 mean_score 做 softmax，作为 policy target
// policy_delta 控制温度（已有 SearchConfig.policy_delta）
let policy_target = softmax_with_temperature(
    &action_scores.iter().map(|s| s.weighted_mean).collect::<Vec<_>>(),
    search_config.policy_delta,
);
// 保存到 TrainingSample.policy_target
```

#### 1.3 Value target 生成

**原方案**：`[final_score, 500, final_score]`

**新方案**：

```rust
// score_mean = 所有 rollout 结果的均值（而非单局 final_score）
// score_stdev = 所有 rollout 结果的标准差（从搜索结果中直接提取）
// value = score_mean（对齐神经网络的 value head）
value_target = [score_mean, score_stdev, score_mean];
```

#### 1.4 新的二进制数据生成器

在 `crates/umasim/src/bin/generate_mcts_training_data.rs` 中：

- 用 `MctsCollectorTrainer` 跑 N 局游戏
- 每回合执行 MCTS 搜索（搜索次数可配置，例如 256-2048）
- 保存每回合的 `TrainingSample { features, policy_target, choice_target, value_target }`
- 按分数排序后取 Top M%
- 保存为 `.bin` (bincode) + 可选 `.jsonl`（调试用）

**新的命令行参数示例：**

```bash
cargo run --release --bin generate_mcts_training_data -- \
    --num-games 10000 \
    --search-n 512 \
    --top-percent 5 \
    --output training_data_mcts.bin \
    --save-jsonl true  # 调试用，保存可读格式
```

---

### Phase 2：Python 训练 Pipeline（外部 Python 项目）

**目标：** 从 Rust 生成的 `.bin` 数据训练神经网络，导出 ONNX 模型。

**技术选型：**

| 组件 | 选型 | 理由 |
|------|------|------|
| 深度学习框架 | **PyTorch 2.x** | 生态成熟，ONNX 导出完善 |
| 训练优化器 | AdamW + CosineLR | 标准配置 |
| 混合精度 | torch.cuda.amp | GPU 训练加速 |
| 分布式 | DDP（可选） | 多卡时使用 |
| 数据加载 | 自定义 `BincodeDataset` | 读取 Rust 端的 `.bin` |

**2.1 模型架构（同神经网络评估器对齐）**

```
Input: [batch, 1121] (f32)
  │
  ├─ MLP Backbone: Linear(1121) → LayerNorm → GELU
  │                       ↓
  │                  Linear(512) → LayerNorm → GELU
  │                       ↓
  │                  Linear(256) → LayerNorm → GELU
  │                       │
  │        ┌──────────────┼──────────────┐
  │        ▼              ▼              ▼
  │  Policy Head     Choice Head     Value Head
  │  Linear(50)     Linear(8)      Linear(3)
  │  (logits)       (logits)       (score_mean, score_stdev, value)
  │
Output: [batch, 61] = [policy(50), choice(8), value(3)]
```

**注**：输出维度 61 = 50 (POLICY_DIM) + 8 (CHOICE_DIM) + 3 (VALUE_DIM)，与当前 `NeuralNetEvaluator` 的推理逻辑完全对齐。

**2.2 损失函数**

| Head | Loss | 权重 |
|------|------|------|
| Policy | CrossEntropyLoss(policy_logits, policy_target) | 1.0 |
| Choice | CrossEntropyLoss(choice_logits, choice_target) | 0.5 |
| Value(mean) | MSELoss(mean_pred, score_mean_target) | 1.0 |
| Value(stdev) | MSELoss(stdev_pred, score_stdev_target) | 0.3 |

Value 的反归一化：
- **训练时**：输入原始 score_mean / score_stdev，先做归一化（`(x - 58000) / 300`），与 `neural_net_evaluator.rs` 中的常量 `VALUE_MEAN=58000`, `VALUE_SCALE=300`, `STDEV_SCALE=150` 对齐
- **推理时**：`score_mean = VALUE_MEAN + VALUE_SCALE * output[58]`

**2.3 训练脚本目录结构**

```
 umaai-rs/
   ├── crates/
   │   └── umasim/
   │         └── src/bin/... (Rust 数据收集)
   └── python/                    # 新建目录
        ├── training/
        │   ├── model.py           # PyTorch 模型定义
        │   ├── dataset.py         # BincodeDataset
        │   ├── train.py           # 训练主脚本
        │   ├── config.yaml        # 超参数配置
        │   └── export_onnx.py     # 导出 ONNX
        ├── evaluation/
        │   ├── eval_model.py      # 用 Rust 模拟器评估模型
        │   └── visualize.py       # 结果可视化
        └── requirements.txt
```

**2.4 训练流程**

```
Rust 端：generate_mcts_training_data
        │
        ▼
   training_data_mcts.bin (bincode: Vec<TrainingSample>)
        │
        ▼
Python 端：BincodeDataset
        │
        ├─ DataLoader (batch, shuffle, num_workers)
        │
        ├─ Forward pass: 1121 → 61
        │
        ├─ Loss = L_policy + 0.5 * L_choice + L_value_mean + 0.3 * L_value_stdev
        │
        ├─ Backward + Optimizer step
        │
        └─ 定期 checkpoint + 导出 ONNX
                │
                ▼
         model_epoch_N.onnx  →  Rust 端 NeuralNetEvaluator 加载
```

**2.5 ONNX 导出要点（与 Rust 端对齐）**

```python
# export_onnx.py 示意
dummy_input = torch.zeros(1, 1121, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # 动态 batch 维度
    opset_version=17,
)
```

**关键点**：`dynamic_axes` 必须开启，因为 `ThreadLocalNeuralNetLeafEvaluator::infer_batch()` 使用动态 batch 推理。

---

### Phase 3：GPU 推理引擎实现（Rust 端）

**目标：** 把当前 CPU 的 tract-onnx 推理替换为 GPU 推理，在 leaf eval 场景获得 10-50 倍加速。

**3.1 GPU 推理库选型**

| 方案 | 库 | 优点 | 缺点 |
|------|----|------|------|
| A. `tch-rs` (PyTorch C++ API) | `tch` crate | 直接加载 TorchScript，GPU 加速，动态 batch 天然支持 | 需要额外的 TorchScript 导出步骤 |
| B. `ort` (ONNX Runtime) | `ort` crate | 直接加载 `.onnx`，支持 CUDA/DirectML，API 简洁 | 依赖较大 |
| C. 继续使用 tract-onnx + wgpu | `tract-onnx` | 已有代码，改动最小 | tract 的 wgpu 后端支持不够完善 |

**推荐方案 B (ort)**：与现有 ONNX 流程无缝对接，支持动态 batch，部署简单。

**3.2 修改/新建文件**

| 文件 | 改动内容 |
|------|----------|
| `crates/umasim/src/neural/gpu_neural_net_evaluator.rs` | **新建**。基于 `ort` 的 GPU 推理评估器，实现 `Evaluator<OnsenGame>` |
| `crates/umasim/src/neural/mod.rs` | 导出 `GpuNeuralNetEvaluator` |
| `crates/umasim/src/search/flat_search.rs` | 在 `LeafEvaluator` 枚举中增加 `GpuNeuralNet` 变体，支持 GPU leaf eval |
| `crates/umasim/Cargo.toml` | 添加 `ort` 依赖（启用 cuda/directml 特征） |

**3.3 GpuNeuralNetEvaluator 设计要点**

```rust
// 伪代码示意
pub struct GpuNeuralNetEvaluator {
    session: Arc<ort::Session>,  // ONNX Runtime Session (GPU backend)
    input_dim: usize,             // 1121
    output_dim: usize,            // 61
}

impl GpuNeuralNetEvaluator {
    /// 加载 ONNX 模型到 GPU
    pub fn load_gpu(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([
                ExecutionProvider::CUDA(Default::default()),
                ExecutionProvider::DirectML(Default::default()),
                ExecutionProvider::CPU(Default::default()),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self { session: Arc::new(session), input_dim: 1121, output_dim: 61 })
    }

    /// GPU 批处理推理（核心加速点）
    pub fn infer_batch(&self, features_flat: &[f32], batch: usize) -> Result<Vec<f32>> {
        // features_flat: [batch * 1121]
        // 直接送入 GPU session