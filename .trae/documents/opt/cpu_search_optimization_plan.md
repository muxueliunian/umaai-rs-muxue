
# CPU 搜索性能优化计划

> 目标：在保持搜索质量不变的前提下，显著降低每次 MCTS 搜索的 CPU 时间开销。

## 现状分析

### 单次搜索的流程

当前 `MctsTrainer::select_action` → `FlatSearch::search` 的执行流程：

```
1. search_uniform / search_ucb 选择一个搜索模式
2. 对每个合法动作:
   a. 执行 search_n 次 rollout
   b. 每次 rollout = game.clone() + apply_action + 循环 run_stage 到终局 + calc_score
3. 统计各动作的分数分布，选择最优动作
```

### 关键性能热点

| # | 热点位置 | 代码 | 问题描述 |
|---|---------|------|---------|
| **1** | `OnsenGame::clone()` | [game.rs:37-68](file:///f:/UmaAI/umaai-rs/crates/umasim/src/game/onsen/game.rs#L37-L68) | 每次 rollout 都深拷贝整个大结构体，含 Vec、HashMap、多个嵌套结构体 |
| **2** | `run_stage` + `next()` 循环 | [game.rs:1377-1767](file:///f:/UmaAI/umaai-rs/crates/umasim/src/game/onsen/game.rs#L1377-L1767) | 每回合大量分支、随机数、事件生成，单次 rollout 需要跑 ~50-70 回合 |
| **3** | `HandwrittenEvaluator::select_action` | `handwritten_evaluator.rs` | rollout 中每回合都要做一次动作选择，包含启发式计算 |
| **4** | `search_uniform` 中 `from_os_rng()` | [flat_search.rs:195](file:///f:/UmaAI/umaai-rs/crates/umasim/src/search/flat_search.rs#L195) | 每个 action 闭包内都调用 `StdRng::from_os_rng()`，其实应该每个 worker 只初始化一次 RNG |
| **5** | 日志输出 | `info! / warn!` | rollout 过程中仍有日志输出（logger 内部有锁 + 格式化开销），虽然 disable_log 会关闭，但 formatter 等仍有开销 |

### OnsenGame 结构体内存分析

```
OnsenGame 包含:
  ├─ BaseGame
  │   ├─ uma: Uma            ← 属性/体力/技能点/flags 等 (~50 字段)
  │   ├─ deck: Vec<SupportCard> (6张，每张含 data/effect/is_locked 等)
  │   ├─ distribution: Vec<Vec<i32>>  ← 二维 Vec，每次 clone 要多次分配
  │   ├─ events: HashMap<u32, u32>    ← HashMap clone 成本很高
  │   ├─ unresolved_events: Vec<EventData>
  │   └─ ...
  ├─ persons: Vec<BasePerson>         ← 人物列表
  ├─ onsen_state: Vec<bool>
  ├─ dig_remain: Vec<[i32; 3]>
  ├─ dig_progress: Vec<[i32; 3]>
  └─ ... (其他 ~10 个字段)
```

每次 `clone()` 都会：
- 触发多次 `Vec::clone()` → 多次堆分配
- 触发 `HashMap::clone()` → 重新构建哈希表
- 触发所有嵌套结构体的 clone

## 优化方案（按优先级排序）

---

### P0：消除不必要的 `game.clone()` 开销

**修改文件**: `crates/umasim/src/search/flat_search.rs`

#### 方案 A：单次 clone + 种子化 RNG

**原理**：当前每次 rollout 都执行 `game.clone()` + 随机 rollout。
改为：在每个 worker 内部只 clone 一次 game，然后用不同的 RNG seed 来生成不同路径。

```
当前:
  for _ in 0..n:
    game.clone()
    apply_action
    run_stage to end
    calc_score

改为:
  let mut sim_game = game.clone();    // 只 clone 一次
  for i in 0..n:
    sim_game.reset_to_search_start(&game)?;   // 从搜索起点恢复（轻量 copy）
    apply_action
    run_stage to end
    calc_score
```

但 `reset_to_search_start` 本质上还是要把 game 的所有字段重新覆盖到 `sim_game`，和 clone 差不多。

**更优思路**：把 `simulate_many` 改为 batch 模式——在一个 worker 内部只 clone 一次 game，用不同的 RNG seed 来跑多个 rollout：

```rust
fn simulate_many_batch(
    &self,
    game: &OnsenGame,
    action: &OnsenAction,
    n: usize,
    rng: &mut StdRng,
    result: &mut ActionResult,
    result_pt: &mut ActionResult,
) -> Result<()> {
    // 在 worker 内部：
    // 思路：每次 rollout 前，从一个预存的 baseline 拷贝一份轻量副本
    // 而 baseline 在 worker 初始化时 clone 一次
    
    // 实际上更可行的是：不做任何改动，而是把整个 simulate_many 的内部实现改成
    // "每次 rollout 都从 game.clone()" → 但这本来就是这样做的
    // 真正的优化在于：让 game clone 变快
    
    // 替代方案：把 OnsenGame 里的大字段改成 "共享 + 回滚" 模式
    // 但这需要重构 game 结构，工作量大
}
```

#### 方案 B：拆分 OnsenGame 的 "不变量" 和 "可变量"

**原理**：将 `OnsenGame` 拆成两部分：
- `GameStatic`：包含所有在 rollout 中不需要修改或可以用 Copy-on-Write 共享的字段
  - 例如：`uma_id`、`deck_ids`（只读）、`inherit`（已经是 Arc）、`card_type_count`（已经是 Arc）
  - 温泉信息的静态部分（温泉数据用全局 `ONSENDATA`，不需要存）
- `GameState`：包含所有在 rollout 中会变化的字段
  - 例如：`turn`、`stage`、`uma`（当前属性值）、`friend`、`train_level_count`、`distribution`、`events`、`unresolved_events`、`persons`、`bathing`、`onsen_state`、`dig_*`、`pending_selection`

这样 clone 时只需要 clone `GameState`，`GameStatic` 用 `Arc` 共享。

**修改点**：
1. 重构 `OnsenGame`：`struct OnsenGame { static_data: Arc<GameStatic>, state: GameState }`
2. 所有访问 `self.uma` 改为 `self.state.uma`（或实现 Deref）
3. `GameStatic` 只在游戏初始化时构建一次

**评估**：
- ⚠️ 改动量较大（影响所有访问 `self.*` 的代码）
- 收益：clone 速度提升约 20-50%（取决于不变量的比例）
- 推荐度：中等

---

### P1：优化并行粒度与 RNG 初始化

**修改文件**: `crates/umasim/src/search/flat_search.rs`

#### 1.1 `search_uniform` 改为 `map_init`

**当前代码**（[flat_search.rs:185-206](file:///f:/UmaAI/umaai-rs/crates/umasim/src/search/flat_search.rs#L185-L206)）：
```rust
.par_iter()
.map(|action| {
    ...
    let mut thread_rng = StdRng::from_os_rng();  // 每个 action 都创建！
    let _ = self.simulate_many(game, action, self.config.search_n, &mut thread_rng, ...);
    ...
})
```

**问题**：每个 action 闭包执行时都创建一次 `StdRng::from_os_rng()`，而 from_os_rng 会调用系统随机源获取熵，成本不低。

**修复**：改为 `map_init`，每个 worker 线程只初始化一次 RNG：
```rust
.par_iter()
.map_init(|| StdRng::from_os_rng(), |rng, action| {
    // 同一个 worker 复用 RNG
    let mut result = ActionResult::new();
    let mut result_pt = ActionResult::new();
    let _ = self.simulate_many(game, action, self.config.search_n, rng, &mut result, &mut result_pt);
    (result, result_pt)
})
```

**收益**：
- 减少 `N_actions` 次 `from_os_rng()` 调用
- 如果 Rayon 复用线程，可能节省更多

#### 1.2 把并行粒度从 "按动作" 改为 "按 rollout"

**当前**（search_uniform）：每个 action 分到一个 worker，然后 worker 内部串行跑 `search_n` 次 rollout。

**问题**：如果 action 数量 < CPU 核心数（通常只有 5-8 个动作），CPU 利用率不足。

**改进**：把所有 rollout 摊平成一个大的迭代器，让 Rayon 自动分配：

```rust
// 伪代码
let all_rollouts: Vec<(action_index, rollout_index)> =
    (0..actions.len())
    .flat_map(|ai| (0..search_n).map(move |ri| (ai, ri)))
    .collect();

let results: Vec<Option<(f64, f64)>> = all_rollouts
    .par_iter()
    .map_init(
        || (StdRng::from_os_rng(), game.clone()),  // 每个 worker: 1个RNG + 1个game副本
        |(rng, sim_game), (action_idx, _rollout_idx)| {
            // 恢复 sim_game 到搜索起点（从 game 复制字段）
            *sim_game = game.clone();  // ← 这又变成 clone 了
            
            // 执行 rollout
            self.simulate(sim_game, &actions[*action_idx], rng).ok()
        }
    )
    .collect();
```

**问题**：每个 worker 内部还是要反复 clone game。

**更优的做法**：`simulate` 内部本来就要 clone game，所以把并行粒度调到 rollout 级别不会减少 clone 次数，只是更好地利用了多核。

**实际修改建议**：
- 如果动作数少（< CPU 核数），使用 rollout 级并行
- 保留当前实现，但在 `FlatSearch::search` 入口根据 `actions.len()` 选择并行策略

**收益评估**：
- 动作数少时，CPU 利用率从 ~20-40% 提升到 ~90%
- 但每次 rollout 的 clone 开销仍然存在

---

### P2：移除 rollout 中的日志开销

**修改文件**: `crates/umasim/src/search/flat_search.rs` + 所有在 rollout 中调用的代码

**问题**：
- `simulate` 内部调用 `run_stage` → `apply_action` → `do_train` 等，这些函数中有大量 `info! / warn!`
- 当前 `MctsTrainer` 在搜索前调用 `disable_log()`，搜索后 `enable_log()`
- 但即使 disable，logger 的格式化/调用栈仍有开销

**检查点**：搜索查看 `disable_log` 的实现是否真的完全关闭了日志

**优化建议**：
1. 在 `SimulationTrainer` 中，避免调用任何会产生日志的函数
2. 给 rollout 专用一个 "无日志" 的快速路径

**修改**：
- 为 `OnsenGame` 添加类似 `_no_log` 版本的核心函数，或者在 `HandwrittenEvaluator` 中避免调用 `explain*` 系列函数
- 检查 `HandwrittenEvaluator::select_action` 是否有日志输出

---

### P3：用 `max_depth` + leaf eval 减少 rollout 深度

**当前代码已经支持这个机制**（[flat_search.rs:460-498](file:///f:/UmaAI/umaai-rs/crates/umasim/src/search/flat_search.rs#L460-L498)），但可能没有开启：

```rust
if self.config.max_depth == 0 {
    // 跑到终局
    while sim_game.next() {
        sim_game.run_stage(&trainer_hw, rng)?;
    }
} else {
    // 跑到 max_depth 回合后，用 leaf eval 估分
    ...
}
```

**优化建议**：
1. **开启 `max_depth`**：设置 `search_config.max_depth = 20`（举例），在 20 回合后截断
2. **使用 `ThreadLocalNeuralNetLeafEvaluator` 做 leaf eval**：加载一个已经训练好的策略模型
3. **设置合适的 `rollout_batch_size`**：批处理 leaf 推理

**收益**：
- rollout 深度从 ~50-70 回合降到 ~20 回合，理论加速 2-3x
- 代价：需要一个训练好的模型来估分

**风险**：
- 如果 leaf eval 的估分不准，可能影响搜索质量
- 需要足够的训练数据（但正好可以用当前 MCTS 的数据来训练）

---

### P4：`simulate_many` 中的批量优化

**修改文件**: `crates/umasim/src/search/flat_search.rs`

#### 4.1 预分配 result 容器

当前 `ActionResult` 可能在循环中不断 push/累加。可以预先初始化好，或者用简单的累加变量，最后再转换成 `ActionResult`。

#### 4.2 `simulate_many` 内部的 `simulate_until_terminal_or_leaf` 优化

**当前逻辑**：
```rust
for _ in 0..n:
    match self.simulate_until_terminal_or_leaf(game, action, rng)? {
        SimOutcome::Terminal { score, score_pt } => { ... }
        SimOutcome::Leaf { features, pt_bias } => { ... }
    }
```

**改进**：把 `simulate` 改成返回裸分数，不通过 `anyhow::Result`，因为 rollout 中不应该有错误（游戏规则是确定的，只有 RNG 不同）：

```rust
fn simulate_fast(&self, game: &OnsenGame, action: &OnsenAction, rng: &mut StdRng) -> (f64, f64) {
    let mut sim_game = game.clone();
    // ... 省略 Result 处理，直接返回 (score, score_pt)
}
```

**收益**：
- 移除 `?` 的分支开销（虽然很小）
- 让代码更简洁，减少 unwrap/expect 的调用

**风险**：
- 如果 rollout 中真的出现错误，会 panic 或返回 0
- 但从游戏逻辑看，rollout 应该是确定性的，不太会出错

---

### P5：`HandwrittenEvaluator` rollout 简化

**修改文件**: `crates/umasim/src/neural/handwritten_evaluator.rs`

**问题**：当前 rollout 使用 `SimulationTrainer`，它调用 `HandwrittenEvaluator::select_action` 来选择动作。这个函数可能包含大量启发式计算，包括遍历所有训练、计算收益、选择最优等。

**优化**：
1. 创建一个 `FastRolloutEvaluator`，只做最简单的动作选择（例如：随机选择合法动作，或者只看体力选）
2. 在 `FlatSearch::simulate` 中使用 `FastRolloutEvaluator` 替代 `HandwrittenEvaluator`
3. 保持外层的搜索逻辑不变（搜索时还是选择动作，rollout 用简单策略）

**原理**：MCTS 的 rollout 阶段用简单策略即可，只要"无偏"就行。复杂的策略在搜索节点处已经做了。

**收益**：
- rollout 速度显著提升（每回合的动作选择从启发式变成 O(1) 随机）
- 搜索质量可能下降，但可以通过增大 search_n 补偿

---

### P6：搜索次数动态调整

**修改文件**: `crates/umasim/src/search/config.rs` + `crates/umasim/src/search/flat_search.rs`

**当前**：search_n = 1024，每个动作都跑 1024 次 rollout。

**优化**：
1. **早期终止**：如果某个动作的分数显著低于当前最优，可以提前停止对它的搜索（例如用 UCB 的置信上界已经低于其他动作的下界）
2. **回合依赖**：回合早期（turn < 20），rollout 到终局需要走 60+ 回合，很慢；回合后期（turn > 60），rollout 很短。可以对不同回合设置不同的 search_n。
3. **预搜索 + 精搜**：先用少量 rollout（如 100 次）筛选出 top-2 动作，然后把剩余预算全给这两个动作。

**实现思路**（不需要大改）：
```rust
// 在 FlatSearch::search 中
if radical_factor 很低 && actions.len() > 3 {
    // 先粗搜
    let quick_results = self.search_n(game, actions, 100, rng)?;
    // 筛选 top-2
    let top_indices = ...;
    // 精搜 top-2，分配更多预算
    let fine_results = self.search_n(game, &actions[top_indices], rest_budget, rng)?;
}
```

---

## 推荐实施顺序

| 优先级 | 方案 | 改动量 | 预期加速 | 风险 |
|--------|------|--------|---------|------|
| **1** | P1: `search_uniform` 改为 `map_init` | ✅ 很小 | 1-5% | 低 |
| **2** | P3: 开启 `max_depth` + 已有 leaf NN 推理 | ✅ 小 | 2-3x | 中（依赖模型质量） |
| **3** | P2: 移除 rollout 日志 | ✅ 小 | 5-10% | 低 |
| **4** | P4: `simulate_fast` 简化错误处理 | ✅ 小 | 2-5% | 低 |
| **5** | P5: `FastRolloutEvaluator` 简化 rollout 策略 | ⚠️ 中 | 30-50% | 中（可能影响搜索质量） |
| **6** | P6: 搜索次数动态调整 | ⚠️ 中 | 20-30% | 低 |
| **7** | P1.2: rollout 级并行 | ⚠️ 中 | 30-50%（动作少时）| 低 |
| **8** | P0.B: `OnsenGame` 拆分不变量/可变量 | ⚠️ 大 | 20-50% | 高（重构风险） |

## 风险点汇总

1. **leaf eval 估分精度**：如果 NN 模型不够好，截断 rollout 会导致搜索质量下降
   - 缓解：先在一组测试对局上对比 "完整 rollout" vs "max_depth=20+NN" 的分数分布
2. **`FastRolloutEvaluator` 的搜索质量**：简化 rollout 策略可能导致搜索偏向"短期最优"动作
   - 缓解：增大 search_n，或者用渐进式策略（前 20 回合用手写策略，后 50 回合用随机）
3. **`OnsenGame` 拆分不变量/可变量的重构风险**
   - 缓解：可以先做一个小实验——只把 `distribution` 和 `events` 改成更轻量的结构（例如 `Vec<i32>` 用固定大小数组）

## 下一步行动

1. **立即实施** P1（`map_init` 修复）+ P2（移除 rollout 日志）+ P4（简化错误处理）
2. **并行进行** P3（训练一个 leaf NN 模型，开启 max_depth）——需要先有训练数据
3. **评估** 实施后的性能提升，再决定是否需要更激进的优化（P5/P6/P0）
4. **长期**：训练足够的 RL 模型，之后可以直接用 NN 推理替代部分搜索（方案2的路径）
