# 日志与输出重构实施计划（Phase 3）

> 对应 ramen_refactor_development_plan.md Phase 3「结构化输出、日志和统计」。

## 1. 背景与目标

- 区分"游戏规则日志"（搜索时大量产生，需要可裁剪）与"决策层业务日志"（AI 推荐结果，必须始终可见）。
- 在 log crate + flexi_logger 之上叠加 `diag!` 宏，实现编译期裁剪规则层日志。
- 引入 `DecisionInfo` 作为 AI 决策输出标准格式，支持多下游传输（Android/MCP/WebSocket）。
- 渐进式：保留现有 log 基础设施不变，逐步迁移；每步可独立编译可提交。

## 2. 现状梳理

### 2.1 日志调用分布

| 层级 | 位置 | 调用数 |
|------|------|--------|
| **Game/Action 层** | `game/base/`、`game/ramen/`、`game/onsen/` | **435 处** |
| **Trainer/Search 层** | `trainer/*.rs`、`search/*.rs` | 35 处 |
| **上层** | `bin/`、`neural/`、`umaai/`、`analyzer/` | 90 处 |

### 2.2 现有机制

- `init_logger(app, spec)`：初始化 flexi_logger（utils.rs:57-131），支持屏幕+文件双输出或仅文件输出。
- `disable_log()` / `enable_log()`：通过 `push_temp_spec/pop_temp_spec` 运行时关闭（utils.rs:134-149）。
- 现有 Mutex 双重检查锁（`INIT_LOCK` + `LOGGER_INIT_DONE`）已解决并行测试下"log crate 全局状态只能初始化一次"的竞争。

### 2.3 输出格式化方法

`crates/umasim/src/explain.rs` 提供 7 个独立格式化函数；Game/Action 层各处又有 `pub fn explain()`：
- `BaseGame::explain()`、`OnsenGame::explain()`、`RamenGame::explain()`
- `Uma::explain()`、`FriendState::explain()`、`SupportCard::explain()`
- `BaseAction`/`BasicAction`/`OnsenAction`/`RamenAction` 的 `Display`
- `EventData::explain()`、`EventChoice::explain()`、`ActionValue::explain()`

**设计定位**：`explain()` 为开发者诊断快照（处理多义性结构如 `Array5`），**不是用户展示**。阶段 4 引入 `GameView::view()` 作为面向用户/AI 的结构化展示，两者并存。

## 3. 目标架构

### 3.1 三种输出形态

```
┌─────────────────────────────────────────────────────────────┐
│ 1. diag! 宏（可裁剪的诊断日志）                                │
│    Game/Action 层使用                                       │
│    #[cfg(feature = "diag")] 控制                             │
│    feature 关时：宏 = no-op，零运行时开销                     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ 2. log crate（业务日志）                                     │
│    决策层使用 log::info!/warn!/error!                       │
│    始终输出，永不裁剪                                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ 3. DecisionInfo（决策输出标准格式）                          │
│    Trainer 决策时产生                                       │
│    Serialize 派生，支持多下游传输                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 crate 级别编译期裁剪关系

```
┌─ 二进制：ramen_manual（手写策略玩家测试）──────────────┐
│ feature = "diag" 开启（required-features）             │
│   决策层日志（log::info!）               ✅ 输出         │
│   规则层日志（diag! 展开为 log::info!）  ✅ 输出         │
└──────────────────────────────────────────────────────────────┘
┌─ 二进制：umaai（AI 助手信息流）──────────────────────────┐
│ umasim 依赖 default-features = false                      │
│   决策层日志（log::info!）               ✅ 输出（推荐） │
│   规则层日志（diag! 宏 = no-op）         ❌ 不存在       │
└──────────────────────────────────────────────────────────────┘
```

## 4. output/ 模块设计

`crates/umasim/src/output/` 新模块（4 文件）：

### 4.1 `output/decision.rs` — DecisionInfo

```rust
/// AI 决策输出（与 Trainer trait 分离，Trainer 接口保持只输出 action_index）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionInfo {
    /// 选中的动作索引
    pub action_index: usize,
    /// 选中动作的评分
    pub score: f32,
    /// 所有候选动作的评分（按 actions 顺序）
    pub candidate_scores: Vec<f32>,
    /// 决策原因（手写逻辑说明 / MCTS 解释）
    pub reason: Option<String>,
    /// 决策耗时（毫秒）
    pub elapsed_ms: Option<u64>,
    /// 搜索相关（MCTS 适用）
    pub search_depth: Option<u32>,
    pub visit_count: Option<u32>,
    /// 评分细节（手写策略适用）
    pub score_breakdown: Option<HashMap<String, f32>>,
    /// 剧本相关扩展字段
    pub scenario_extra: Option<serde_json::Value>,
}
```

- `Serialize` 派生 → `to_json()` 直接 `serde_json::to_value(&self)` 即可
- 多个下游（Android/MCP/WebSocket）共享同一结构
- `scenario_extra` 用 `serde_json::Value` 扩展剧本特有字段

### 4.2 `output/diagnostic.rs` — diag! 宏

```rust
/// 可裁剪的诊断日志宏。feature = "diag" 关闭时为 no-op（编译期消除）。
#[macro_export]
macro_rules! diag {
    ($($arg:tt)*) => {
        #[cfg(feature = "diag")]
        ::log::info!(target: "diagnostic", $($arg)*);
    };
}
```

- 关闭 info 和 warn，保留 `log::error!`（永不裁剪）
- 关闭时 `format_args!` 也不执行，零运行时开销

### 4.3 `output/view.rs` — GameView 骨架

```rust
/// 用户/AI 视角的游戏状态展示（结构化，纯函数）
/// 详细字段定义留到阶段 4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameView {
    // 阶段 4 细化
}
```

### 4.4 `output/mod.rs` — 模块入口

```rust
pub mod decision;
pub mod diagnostic;
pub mod view;

pub use decision::DecisionInfo;
```

## 5. Trainer 可选扩展

Trainer trait **保持不变**（只输出 `action_index`）。各 Trainer 实现可选择性地提供 `last_decision()`：

```rust
impl MctsTrainer {
    last_decision: Mutex<Option<DecisionInfo>>,
    pub fn last_decision(&self) -> Option<DecisionInfo> { ... }
}
```

- **不强制** trait 要求
- 只有"面向用户"的 Trainer（MctsTrainer、HandwrittenEvaluator、MeanFilterCollectorTrainer）实现
- 上层代码：`trainer.select_action(...)?.last_decision()` 获取决策信息

## 6. Cargo features 改造

```toml
# crates/umasim/Cargo.toml
[features]
default = ["diag"]
diag = []

[[bin]]
name = "ramen_manual"
required-features = ["diag"]   # 手写策略必须能看诊断日志
```

```toml
# crates/umaai/Cargo.toml
[dependencies]
umasim = { path = "../umasim", default-features = false }   # 不带 diag

# crates/analyzer/Cargo.toml
[dependencies]
umasim = { path = "../umasim", default-features = false }   # 同上
```

## 7. 实施步骤

> 渐进式路线：先建骨架再迁移，每步独立可编译可提交。

### 7.0 阶段 0：测试日志简化（已完成）

新增 `init_test_logger(spec)`（`utils.rs`），只输出 stderr、不写文件；共享全局 LOGGER 单例，复用现有 Mutex 重入保护。测试代码 97 处 `init_logger("test", ...)` → `init_test_logger(...)`；删除测试中 3 处 `disable_log/enable_log` 调用。

### 7.1 阶段 1：骨架搭建（本次）

1. **新增 `output/decision.rs`** — DecisionInfo 结构
2. **新增 `output/diagnostic.rs`** — diag! 宏
3. **新增 `output/mod.rs`** — 模块入口
4. **新增 `output/view.rs`** — GameView 骨架
5. **`trainer/mcts_trainer.rs`** — 加 last_decision 字段 + 访问方法
6. **`trainer/handwritten_trainer.rs`** — 同上
7. **`trainer/mean_filter_collector_trainer.rs`** — 同上
8. **`umasim/Cargo.toml`** — diag feature + binary required-features
9. **`umaai/Cargo.toml`** — default-features = false
10. **`analyzer/Cargo.toml`** — default-features = false
11. **最小验证测试** — DecisionInfo 序列化往返、diag! 裁剪效果

**不动**：Game trait（不加 `type Event`、不加 `output` 字段）、现有 435 处日志、现有 `init_logger`/`disable_log`。

### 7.2 阶段 2：规则层迁移

- Game/Action 层 435 处 `info!`/`warn!` → `diag!`
- 按模块分组：`game/base/` → `game/ramen/` → `game/onsen/`（Phase 6 删除前）
- 每步 release 编译 + 测试通过

### 7.3 阶段 3：决策层日志梳理

后续任务：决策层 info 中混入 print 信息需重新分类。

### 7.4 阶段 4：GameView 完整定义

- Game trait 加 `fn view(&self) -> GameView`
- 各剧本实现默认版本
- `umaai`/`analyzer` 接入 `GameView`，替代当前的 `println!("{}", game.explain()?)`
- `explain()` 保留为开发者诊断快照

### 7.5 阶段 5：disable_log 优化

迁移完成后：
- 删除 `disable_log`/`enable_log` 公共函数
- 在 `mcts_trainer` 和测试中改用 `output.diagnostic.set_enabled(false)`（如果保留）或其他机制
- 简化 `LOGGER_INIT_DONE`/`INIT_LOCK` 过渡期锁

### 7.6 阶段 6：性能验证

- `cargo bloat` 对比 `umaai` 在 `diag` 开/关下 binary 大小
- MCTS rollout benchmark 对比
- 验证 `diag!` feature 关时零运行时开销

## 8. 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 日志库 | 保留 flexi_logger | 已支持屏幕+文件双输出 |
| 编译期裁剪 | `diag!` 宏 + `#[cfg(feature)]` | 关闭时连 `format_args!` 都不执行，零开销 |
| feature 默认 | 默认开启 | 开发体验好；`umaai` 显式 `default-features = false` |
| 关闭范围 | info 和 warn | 排查规则 bug 用；error 永不裁剪 |
| macro 命名 | `diag!` | 简化命名 |
| 决策层日志 | 保持 `log::info!/warn!/error!` | 业务输出必须可见 |
| DecisionInfo 与 Trainer | 独立类型，Trainer 接口不变 | 保持 trait 简洁；扩展信息通过 last_decision() 旁路提供 |
| DecisionInfo 序列化 | Serialize 派生 | to_json 直接 serde_json::to_value |
| 剧本特有字段 | `scenario_extra: Option<serde_json::Value>` | 灵活扩展，HashMap 备选 |
| explain() 去留 | 保留 | 多义性诊断快照，与 GameView 并存 |
| disable_log | 保留过渡期 | 阶段 5 删除 |
| tracing | 推迟到 Phase 5 | 不在当前阶段引入 |
| Game trait | 不变 | 不加 `type Event`、不加 output 字段 |
| EventCollector | 不引入 | 手动模拟场景不重要，决策输出用 DecisionInfo |

## 9. 验证

### 9.1 阶段 0（已完成）

- `cargo test --release -p umasim --lib` 全部通过
- `cargo build --release -p umaai -p umasim -p analyzer` 编译通过

### 9.2 阶段 1 验证

- 全 workspace release 编译通过
- `DecisionInfo` 序列化往返测试通过
- `diag!` 宏在 feature 开/关下编译产物对比
- `ramen_manual` 能正常启动并显示规则日志
- `umaai`/`analyzer` 默认 build 不带 `diag` feature，规则层日志完全不生成

### 9.3 阶段 2-6 验证

- 435 处规则层日志全部迁移后，release 编译 + 124 个 lib 测试通过
- `cargo bloat` 显示 `umaai` binary 中 `diag!` 调用数为 0
- MCTS rollout benchmark 吞吐量提升
- `explain()` 与 `GameView::view()` 测试均通过

## 10. 关联文档

- `ramen_refactor_development_plan.md` Phase 3：高层目标
- `config_refactor_plan.md`：本期实施的参考模板（结构/节奏）
- `project_context.md`：实施完成后更新相关章节
- `changelog.md`：实施完成后统一更新