# 拉面杯神经网络训练侧设计

本目录只负责 `.npy` 教师数据生成标签、训练、评估和导出 ONNX；不修改模拟器。
第一代冻结输出为 `policy[234] + choice[8] + value[3]`，choice loss 恒为 0。

## 1. Policy 标签

默认标签不是 one-hot，也不是 `softmax(mean / T_stage)`，而是配对 Bayesian
bootstrap 下每个候选成为最优的概率：

1. 对 512 个 rollout 生成 `Dirichlet(1,...,1)` 权重；
2. 同一次抽样的所有候选共享同一列权重，保留 CRN 配对结构；
3. 计算每个候选的加权均值并给最优候选计票；
4. 精确并列时平分该票，512 次抽样后归一化。

这相当于用经验分布表达“有限 rollout 下谁真是最优”的不确定性。不同阶段候选差值
和配对噪声会自动进入概率，不需要四组人为温度。`--policy-temperature` 仅作为全局
校准实验参数，默认 1.0，不建议在没有留出集校准证据时改动。

默认优化候选的期望终局分（mean），因为最终验收是多局平均分。`radical_factor=1.4`
不参与动作选择，只用于 value[2]；如果未来目标改成凹单局上限，应生成另一份明确版本的
标签，不能静默把当前平均分 policy 改成高分位 policy。

普通候选的概率直接写入唯一格位。RegionSelect 先得到 120 个组合的概率，再边缘化：

```text
target[region] = sum(P(combination) for combination containing region) / 3
```

因此 20 格目标和仍为 1，训练与推理的 top-3 口径一致。它仍无法表达任意三元交互；
这是冻结的 20 格输出带来的表达上限，不是标签能修复的问题。

## 2. Winner's curse 与 value 口径

value 使用 leave-one-rollout-out cross-fitting。对每个随机世界 `k`：

- 用其余 511 列的候选均值选择动作；
- 只用第 `k` 列中该动作的分数估值；
- 遍历全部 512 列，得到 512 个样本外选择结果。

三路定义为：

- `value[0]`：上述 512 个结果的均值；
- `value[1]`：上述结果的样本标准差，分母 `n-1`；
- `value[2]`：与 Rust `weighted_mean` 相同的分组中点排名积分，默认 rf=1.4。

在当前 11957 条 pilot 上，`max(cand_mean) - crossfit_mean` 的平均值分别为：

| 阶段 | 乐观偏差 |
|---|---:|
| RamenSelect | 9.2 |
| Train | 16.1 |
| SuperRamenSelect | 4.7 |
| RegionSelect | 19.7 |
| 全部 | 13.7 |

中位数为 0，因为 CRN 后 leave-one-out 选择稳定率约 99%；但非零尾部仍会造成系统偏差，
所以保留 cross-fit，而不是因均值偏差“小”就退回同批选择同批估值。

这个 value 是“搜索改进策略先选一步、随后按采集 rollout 策略走完”的状态价值。
第二代若把它用作截断 leaf，Rust 侧必须保持这一语义；若 leaf 要的是另一个 rollout
policy 的 `V^pi`，则属于分布偏移，不能仅靠换反归一化常数解决。

## 3. Value 归一化

每次新训练只用稳定训练划分拟合三路 `(label - center) / scale`，并把常数保存到
checkpoint、`run.json` 和 `model.onnx.json`。当前 pilot 的训练划分实测为：

```text
center = [60988.10, 2140.95, 62465.22]
scale  = [ 4244.10,  791.90,  4678.82]
```

这些不是正式常数；8 万条数据到齐后会自动重算。Rust 必须使用模型旁的最终元数据。
特别是 stdev 也有非零 center，应按 `center + scale * output` 反归一化后截到非负，
不能沿用温泉代码的 `scale * abs(output)`。

value loss 是归一化后的三路 Huber，分量权重保持 `0.2 / 0.4 / 0.2`。

## 4. 等价候选

逐列完全相同的候选不合并、不删除合法格位：

- Bayesian bootstrap 每次都会判为并列，概率均分；
- RegionSelect 再按同一边缘化规则投影；
- cross-fit value 中选择任意一个都得到相同 held-out 分数。

这样不会让候选枚举顺序成为隐式监督信号。

## 5. 模型、容量与早停

模型把 global 作为一个 token，cards/persons 分别用共享 MLP 投影，再对最多 20 个
token 做多头注意力。card 有槽位 embedding；person 没有位置 embedding，且使用行内
第 15 维“已登场”标志同时屏蔽 key/value、残差和 pooling。汇总 global/card/person
三路后进入 ResMLP，最后只有一个 `Linear(...,245)`。

注意力实现可切换：

- `simple`（默认）：上游无 softmax 的 ReLU 相似度注意力；
- `softmax`：标准多头 softmax attention；
- `--encoder-blocks 0`：无注意力的 pooling 基线。

当前 11957 条数据、相同划分下跑 15 epoch、3 个种子的 pilot：

| 结构 | 最佳期望后悔值，三种子均值 | 最佳验证 KL，三种子均值 |
|---|---:|---:|
| simple | **148.8** | **0.897** |
| softmax | 153.8 | 0.901 |
| 无注意力 | 155.1 | 0.902 |

差距不够支持永久拍板，但足以否定“标准 attention 因为更新就应默认更好”。默认选算子
更少且 pilot 占优的 simple；8 万条数据必须用相同协议复验。

容量按训练样本数自动选择：

| 训练样本 | token dim / encoder blocks / MLP width / MLP blocks | dropout |
|---|---|---:|
| `<25k` | 64 / 1 / 192 / 2 | 0.15 |
| `25k..120k` | 96 / 2 / 256 / 2 | 0.08 |
| `>=120k` | 128 / 2 / 384 / 3 | 0.05 |

训练使用 Adam，trunk `lr=5e-4, wd=2e-5`，单一输出 Linear 用 `lr=1.25e-4`
且不做 weight decay，使无 loss 的 choice 八行保持初始化零。默认 batch 1024。

早停监控留出集的**期望后悔值**，不是训练 loss；小数据默认 patience 20，大数据 12，
同时用 ReduceLROnPlateau。日志写入 `metrics.jsonl`，是无需 TensorBoard 依赖的等价结构化日志。

## 6. 稀疏阶段

只对 policy KL 使用截断的逆平方根阶段权重：

```text
weight_s = min(sqrt(max_count / count_s), 4)，再归一到样本均值 1
```

不做过采样，也不对 value 重加权。当前训练划分中的实际权重约为
Ramen 1.00 / Train 0.82 / Super 3.29 / Region 2.35。42 条 Super 样本无论乘多大权重
都不足以可靠估计泛化误差；加权只能防止梯度完全被淹没，不能创造信息。正式数据应优先
增加该阶段覆盖，评估报告必须保留阶段样本数，不对 5 条验证样本的命中率作强结论。

## 使用

安装依赖：

```powershell
python -m pip install -r scripts/ramen_nn/requirements.txt
```

从 raw 目录生成标签（只需一次，可离线缓存）：

```powershell
python scripts/ramen_nn/labels.py `
  --input training_data/npy_v1_raw `
  --output training_data/labels_v1
```

训练；多份数据重复传入一一对应的参数：

```powershell
python scripts/ramen_nn/train.py `
  --data training_data/npy_v1 `
  --labels training_data/labels_v1 `
  --output-dir saved_models/ramen_v1
```

断点续训、独立评估与导出：

```powershell
python scripts/ramen_nn/train.py --data training_data/npy_v1 --labels training_data/labels_v1 `
  --output-dir saved_models/ramen_v1 --resume saved_models/ramen_v1/last.pt
python scripts/ramen_nn/eval.py --data training_data/npy_v1 --labels training_data/labels_v1 `
  --checkpoint saved_models/ramen_v1/best.pt
python scripts/ramen_nn/export_onnx.py --checkpoint saved_models/ramen_v1/best.pt `
  --output saved_models/ramen_v1/model.onnx
```

导出使用 opset 13、动态 batch，并强制检查算子白名单；随后对 batch 1 和 7 做
PyTorch/ONNX Runtime 逐元素对拍，最大误差必须小于 `1e-4`。最终是否超过手写策略仍只看
Rust 侧纯网络完整育成基准，Python 留出指标不能替代该验收。
