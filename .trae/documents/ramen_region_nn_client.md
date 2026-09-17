# 客户端可选的神经网络地区选择（拉面杯）

> 适用：`umaai` 主程序（在线模式）与 `ramen_client_game_bench` 实验入口
> 状态：默认关闭，opt-in；1024 档先导闭环有正向证据（见 §6）

## 1. 这是什么，边界在哪

`umaai` 支持显式选择拉面杯**地区选择**的决策来源：

| `ramen_region_policy` | 谁做地区决策 | 说明 |
|---|---|---|
| `"handwritten"`（默认） | 既有装配 | `[mcts] ramen_search_stages` 打开 `region` 时走搜索，否则落手写推荐策略。与本功能引入前逐字一致 |
| `"nn"` | ONNX 网络 | **只接管外层实际对局的三次 `RegionSelect`**（turn 2 / 23 / 47） |
| `"nn_compare"` | ONNX 网络 | 同上，但**额外**把既有装配在同一局面下的选择算出来一并显示（见 §4.2） |
| `"mcts_compare"` | 既有装配 | 两条推荐同样都显示，但**执行**的是既有装配那条，网络那条只作参考 |

`"nn"` 模式下**不变**的部分（全部原样转发同一个 `RamenMctsTrainer` 实例）：

- `Train` / `RamenSelect` / `SpecialSelect` / `SuperRamenSelect` 的决策；
- 事件选项、自选比赛硬守门；
- **搜索内部 rollout 的地区选择仍是手写基策**——模型不进 rollout evaluator，
  `[mcts] rollout_evaluator` 不受本功能影响。

因此正常一整局的推理请求**恰好 3 次**。地区阶段候选数恒 > 1（第 1/2 年 C(5,3)=10，
第 3 年 `all` 下 C(10,3)=120），不会被单候选短路吞掉——「每局恰好 3 次」就是
「搜索内部没有任何网络推理」的直接证据。

代码位置：`crates/umaai/src/region.rs`（装配与配置校验）+ `crates/umaai/src/region/nn.rs`
（接管实现）。主程序与 benchmark **共用同一份**接管实现，benchmark 额外需要的
「同局面下手写本来会选什么」通过 `RegionDecisionObserver` 钩子挂入，不另写一份决策逻辑。

## 2. 配置

三个字段都是 `game_config.toml` 的**顶层**字段（必须写在任何 `[xxx]` 段**之前**，
否则会被前一个段的未知字段检查吸收）：

```toml
# ---- 写在文件顶部、所有 [xxx] 段之前 ----
ramen_region_policy = "nn"                                   # 默认 "handwritten"
ramen_region_model_path = "saved_models/arms/ens_G2mix_g123.onnx"
ramen_region_strategy = "all"                                # 必须是 "all"

[config_override]
# ……

[mcts]
ramen_search_stages = "train,ramen"                          # 不能含 region
```

- `ramen_region_model_path` 与 `neuralnet_model_path`（温泉 leaf evaluator 模型）
  **是两个字段、两种模型**，不要混填；混填会在加载时因维度不符报错。
- ❗**相对路径按「进程当前工作目录」解析**，既不是「总是相对 exe」也不是「相对 workspace」。
  `umaai` 启动时只在**当前目录里找不到 `game_config.toml`** 的情况下才调
  `check_working_dir()` 切到 exe 所在目录；启动目录里有 `game_config.toml` 时就留在原地。
  所以：开发时从 workspace 根跑，相对路径是相对 workspace 根；发布包里从测试包目录双击
  或 `cd` 进去再跑，相对路径就是相对那个目录。**手测时请从测试包目录启动**，或者直接写绝对路径。

## 3. 构建

```bash
cargo build --release -p umaai --features onnx
```

默认构建（不带 `--features onnx`）**不含推理后端**，也不引入 `tract-onnx` 依赖链。
此时若配置了 `ramen_region_policy = "nn"`，启动直接报错退出并提示加 feature，
**不会**静默回退成手写继续跑。

## 4. 配置冲突诊断

启动时统一校验，任一条不满足就带修复建议报错退出（都是「打印一套、执行另一套」的隐患）：

| 冲突 | 为什么不允许 |
|---|---|
| `ramen_region_policy="nn"` + `ramen_search_stages` 含 `region` | 地区已被网络接管，搜索永远轮不到它，开关留着只会让日志与实际不符 |
| `ramen_region_policy="nn"` + `ramen_region_strategy="fixed"` | `fixed` 下第 3 年只枚举 1 个候选，网络会被单候选短路，实际只决策 2 次地区而非 3 次 |
| `ramen_region_policy="nn"` 但 `ramen_region_model_path` 缺失 / 空白 | 没有模型可加载 |
| `ramen_region_policy="nn"` 但构建未开 `onnx` feature | 二进制里没有推理后端 |
| 模型 / 旁车缺失、维度与冻结契约不符 | 由 `RamenNnTrainer::load` 报错 |
| 旁车声明正确，但 **ONNX 图本身**输出个数 ≠ 1 / 元素类型 ≠ f32 / 形状 ≠ `[1, 245]` | 旁车只是作者的**声明**，图导错时旧实现要到第一次真实决策才炸；现在加载阶段就校验图的 fact（不跑推理，不计入推理计数） |

`ramen_region_policy="handwritten"` 时即使填了 `ramen_region_model_path`，也只会打一条
warn 说明「本次不加载地区模型」，不会偷偷启用。

**实验入口 `ramen_client_game_bench --trainer mcts+region_nn` 复用同一套适用性检查**
（上表前两行）：同一份 `game_config.toml`，客户端拒绝启动的配置，benchmark 也会拒绝。
检查作用在**命令行覆盖之后**的生效配置上。模型路径那一行只对客户端成立——benchmark
的模型来自 `--model`。

## 4.1 屏幕输出的决策来源

无搜索评分的地区决策（手写 fallback 与网络接管**都**属于这一类）在屏幕上显示为
`选择<地区>（<来源>）`。来源取自决策的 `scenario_extra.decision_source`：

| 来源 | 屏幕文案 |
|---|---|
| `region_nn`（网络接管） | `（神经网络地区策略）` |
| `region_handwritten`（对照模式下执行手写基策那条） | `（手写地区策略）` |
| 未标注 | `（无搜索评分）` |

❗**不再把「没有搜索评分」等同于「手写」**。网络接管地区时同样没有搜索评分，
旧实现会把它印成「（手写逻辑）」，与实际执行分支不符。
来源标签**只标注来源，不伪造任何评分**：`score` / `candidate_scores` / `candidate_n`
保持原样，luck 挂载的路由口径（按 `candidate_scores` 是否为空）因此不变。

## 4.2 对照模式：同时显示两条推荐

`"nn_compare"` / `"mcts_compare"` 下，每次外层 `RegionSelect` 会把同一个局面算**两次**：

1. **既有装配**一次——`[mcts] ramen_search_stages` 含 `region` 时是**真搜索**（屏幕上
   标「MCTS搜索」），不含时是**手写基策**（标「手写策略」）。标签按「摘要里有没有候选
   评分」判定，不靠猜配置；
2. **网络**一次。

屏幕上多打一行（human 模式；`--json` 模式不打印，改由决策的
`scenario_extra.region_compare` 带出同样的信息）：

```
地区对照：手写策略 → 地区[米浴,大和赤骥,东海帝皇] ｜ 神经网络 → 地区[米浴,好歌剧,鲁道夫象征]（❗两者不一致；本次执行：神经网络）
```

两个取值只差在**哪条算执行推荐**（即屏幕主行与发给下游的那条 `decision`）：

| 取值 | 执行 | 另一条 |
|---|---|---|
| `"nn_compare"` | 网络 | 既有装配，仅显示 |
| `"mcts_compare"` | 既有装配 | 网络，仅显示 |

❗**对照那一侧跑在随机流的副本上**（`rng.clone()`），不推进外层 `rng`：开不开对照，
网络这一侧看到的随机状态完全一样。

❗**成本**：对照模式下 `ramen_search_stages` 含 `region` 时，第 3 年那一步要搜
C(10,3)=120 个候选，可能非常慢；不含 `region` 时对照侧只是手写基策，成本可忽略。
这是对照模式**允许** `region` 留在搜索阶段集里的原因——完全接管（`"nn"`）仍然拒绝
这种配置。

❗**别把预算算成「候选数 × `search_n`」**，那只适用于 `[mcts] use_ucb = false`：

| `use_ucb` | 每个候选的 rollout 次数 | 一次搜索的总次数 |
|---|---|---|
| `false`（均匀分配） | 恰好 `search_n` | 候选数 × `search_n` |
| `true`（**仓库默认**） | 由 UCB 动态分配，彼此悬殊 | 随局面变化，**下界**是候选数 × 首组大小 |

UCB 的停止判据是「**被搜得最多的那个候选**的计划次数 ≥ `search_n`」
（`FlatSearch::search_ucb`），不是每个候选都跑满 `search_n`。所以 8192 档那一步到底跑
多少次 rollout 只能实测，不能由 `120 × 8192` 推出来。

❗上表的「首组大小」是 `min(search_group_size, search_n).max(1)`，**不是**
`search_group_size` 本身：`search_ucb` 会把首组收进 `search_n`，否则
`search_group_size > search_n` 时每个候选跑完首组就已超预算，自适应轮次为零。
仓库默认 `search_group_size = 512`、`search_n = 12288`，此时两者相同；但把 `search_n`
调到 512 以下（调试用）时下界会跟着变小。

❗`mcts_compare` 且对照侧**有**搜索评分时，执行推荐就是那条搜索决策本身：候选评分、
luck 行与 `ramen_search_stages` 含 `region` 的手写装配逐字一致，来源标注
`region_search`。对照结果**照样随决策带出**——luck 挂载改成了「合并」而不是重建
`scenario_extra`，`region_compare` 与 `decision_source` 都保得住。
对照侧**没有**搜索评分时，执行推荐标注来源 `region_handwritten`，屏幕显示
`选择<地区>（手写地区策略）`。

❗`nn_compare` 且对照侧是**真搜索**时，参照那一跑的决策理由会被**挡在理由门外**
（`ReasonGate`，见 `umaai::decision`）：执行的是网络那条，不能拿参照搜索的「首选 /
对手候选」去解释它。门只静音那一跑，链式决策里其它步骤写好的理由一条不少。

### `scenario_extra.region_compare` 载荷

| 键 | 含义 |
|---|---|
| `index_space` | 恒为 `"actions"`——下面三个下标都是**候选全表**的下标 |
| `candidates_total` | 本次候选总数 |
| `nn_index` / `nn_choice` | 网络那条 |
| `baseline_index` / `baseline_choice` | 对照那条 |
| `baseline_kind` | `"search"`（真搜过）/ `"handwritten"`（落手写基策） |
| `agree` | 两侧是否一致 |
| `executed` | `"nn"` / `"baseline"` |
| `executed_index` / `executed_choice` | 本次实际执行的那条 |

❗`index_space` 存在的理由：执行侧是真搜索时，决策顶层的 `action_index` 指向的是
**截断后**的候选表（`reason_max_display`，默认 5），与这里的全表下标不是一个空间。
要对人显示就用 `*_choice` 文本。

## 4.3 整局无搜索 NN（`ramen_trainer_policy = "nn"`，**本地实验用**）

比地区那一层高一级：`ramen_trainer_policy` 决定**整局**归谁，`ramen_region_policy`
只在前者为 `"mcts"`（默认）时才有意义。

| `ramen_trainer_policy` | `ramen_region_policy` | 实际执行 |
|---|---|---|
| `"mcts"`（默认） | 任意 | 见 §1 / §4.2 |
| `"nn"` | **必须** `"handwritten"` | 整局所有动作决策（含地区）直接走网络，一次搜索都不跑 |
| `"nn"` | `"nn"` / `*_compare` | **启动报错**：两层都声称接管地区 |

```toml
# ---- 写在文件顶部、所有 [xxx] 段之前 ----
ramen_trainer_policy = "nn"
ramen_region_policy = "handwritten"                          # 中性默认值，必须是它
ramen_region_strategy = "all"                                # 必须是 "all"
ramen_region_model_path = "saved_models/arms/ens_G2mix_g123.onnx"   # 此模式下是**整局动作模型**
```

要点：

- **模型字段复用** `ramen_region_model_path`，不为本地实验新增第二套路径；启动横幅会
  说明它在本模式下是整局动作模型。
- `[mcts]` 的搜索参数（`search_n` / `ramen_search_stages` / `use_ucb` …）
  **不参与 NN 动作决策**，整局一次搜索都不跑。启动时打一条提示，但**不要求**用户删掉——
  默认配置本来就带着整段。
  ❗准确说法是「不参与决策」而不是「完全不读取」：主程序仍会解析 `ramen_search_stages`
  （写错照样在启动时报错），也仍会按既有路径构造搜索训练员，只是构造完就丢弃。
- ❗**不是「完全没有手写逻辑」**：事件选项与友人事件仍走手写策略（choice 头没训练），
  自选比赛硬守门保留（不达标直接育成失败，不是可权衡的价值项），单候选局面不跑推理。
  「纯 NN」指的是**动作决策**这条线。
- 来源照实标注，四个出口各有标签：

  | 出口 | 来源标签 | 屏幕文案 | 跑推理？ |
  |---|---|---|---|
  | 网络 argmax | `ramen_nn` | 神经网络（无搜索） | 是 |
  | 自选比赛硬守门 | `ramen_race_gate` | 自选比赛硬守门 | 否 |
  | 唯一候选 | `ramen_single_candidate` | 唯一候选 | 否 |
  | `SpecialSelect` 整阶段转手写 | `ramen_handwritten_stage` | 手写策略（该阶段） | 否 |

  标签来自 `RamenNnTrainer::prepare_decision_labeled`——**同一份**判定，不为了标来源
  再跑一遍守门判断，也不为了标来源多做一次推理。
- 每一步都**没有搜索评分**：`candidate_scores` / `candidate_n` 恒空、`score` 恒 0。
  因此这一路不挂 luck、不显示「期望评分」，也不会透传上一步的搜索理由。policy logits
  是相对量，不是终局分，不往评分字段里填。
- 链式决策照旧：不吃面之后给训练推荐、turn 1 之后给第 1 年地区推荐。

❗**这是实验取值，不是更强的配置**：闭环实测同族模型直接决策相对 `search_n=8192` 的
MCTS 基线，配对差 **−8947 分**（95% CI [−10208, −7686]，10/10 全负）。

## 5. 模型与旁车部署（❗发布待办）

- 需要**两个文件**：`<name>.onnx` 与同目录同名旁车 `<name>.onnx.json`
  （含 `input_dim` / `output_dim` / `value_normalization`，加载时逐项校验）。
- **模型不在版本库内**：`.gitignore` 第 7 行忽略整个 `saved_models/`，本仓库也没有
  LFS / release artifact 配置。本 PR **不上传模型、不改远端发布配置**。
- 因此当前只支持「用户自行把模型放到本机某处，再用 `ramen_region_model_path` 指过去」。
- ❗**待用户决定的发布待办**：地区模型走哪条分发渠道（GitHub Release 附件 / 单独下载包 /
  Git LFS）。渠道定下来之前，`"nn"` 模式只能由手上已有模型的人使用。

本轮实验使用的模型：`saved_models/arms/ens_G2mix_g123.onnx`（8142565 字节）。

## 6. 实验依据（能说什么 / 不能说什么）

先导闭环：随机抽取的 60 条「马娘 × 完整卡组 × 世界」配对面板，`search_n=1024`，
三种卡组构成各 20 对。预登记与完整报告在本机 `logs/region_panel_0914/`
（`preregistration.md` / `report.md`，该目录被 `.gitignore` 忽略，不进版本库）。

| 指标 | 值 |
|---|---|
| A（手写地区）均分 | 71750.5 |
| B（G2mix 地区）均分 | 72330.7 |
| 配对均差 B − A | **+580.2** |
| 95% CI（t, df=59） | **[+31.8, +1128.5]** |
| 胜 / 平 / 负 | 37 / 0 / 23 |
| 配对差 sd / SE | 2122.6 / 274.0 |
| B 的推理请求合计 | 180 = 60 局 × 3 次（`infers ≠ 3` 的计划一个也没有） |
| A 的推理请求合计 | 0 |

**能说的**：新增可选的 NN 地区选择，**1024 档先导闭环有正向证据**。

❗**不能说的**（区间下界只有 +31.8，离零不到点估计的 6%；配对差 sd 高达 2122）：

- 不能说 8192 档已验证——本轮只跑了 `search_n=1024`；
- 不能说全面优于手写——按卡组构成拆分时，`2速1耐1力1智1友` 那格点估计为负（−114.8，
  区间跨零），三格互不排斥，不能拆着念；
- 不能说稳定提升 580 分——换一份同样大小的面板，区间跨零是很有可能的；
- 不能说支持全部马娘机制已完全正确——本轮只核对了地区决策路径。

后续的 1 万–2 万地区专项根采集属于**下一轮任务**，配方未冻结，不在本次范围内。

## 7. 实验入口

`ramen_client_game_bench --trainer mcts+region_nn --model <path>` 走的是**同一条**
接管实现（`umaai::region::RegionNnTrainer`），只是额外挂了观测钩子记录
`regions.csv`（实际选择 + 同局面手写反事实）。它按 `--model` 命令行参数加载模型，
**不读** `ramen_region_model_path`——实验入口保持一次跑多个模型的能力。
