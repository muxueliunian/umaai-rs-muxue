# 日志与统计重构实施计划（Phase 3）

> 对应 ramen_refactor_development_plan.md Phase 3「结构化输出、日志和统计」。
> 本文档为实施计划，详细方案与决策记录在 §6。

## 1. 背景与目标

- 区分"游戏规则日志"（搜索时大量产生，需要可裁剪）与"决策层业务日志"（AI 推荐结果，必须始终可见）。
- 在 log crate + flexi_logger 之上叠加 `diag!` 宏，表示"可裁剪的诊断 log"。
- 通过 crate 级别编译期裁剪（`feature = "rule_diagnostics"`），让 `umaai` 等生产场景在编译期就消除规则层日志调用（零运行时开销）。
- 引入结构化 `GameEvent` 与 `GameView` 数据通道，为下游（AI 助手、analyzer、回放分析）提供机器可解析的数据源。
- 渐进式：保留现有 log 基础设施不变，逐步迁移；每步可独立编译可提交。

## 2. 现状梳理

### 2.1 日志/输出调用分布（grep 统计）

| 层级 | 位置 | 调用数 | 性质 |
|------|------|--------|------|
| **Game/Action 层** | `game/base/`、`game/ramen/`、`game/onsen/` | **435 处** | `info!`/`warn!`/`println!`/`eprintln!` 混合（规则层） |
| **Trainer/Search 层** | `trainer/*.rs`、`search/*.rs` | 35 处 | 决策层业务日志 + 临时 `disable_log/enable_log` |
| **上层** | `bin/`、`neural/`、`umaai/`、`analyzer/` | 90 处 | 编排、汇总、UI 入口 |

### 2.2 现有"开关"机制

- `init_logger(app, spec)` / `init_logger_stdout(...)`：初始化 flexi_logger（utils.rs:57-131）。
- `init_logger_with(app, spec, duplicate_stderr)`：支持屏幕 + 文件（`Duplicate::All`）和仅文件模式（utils.rs:65-98，注释明确"TUI 兼容"）。
- `disable_log()` / `enable_log()`：通过 `push_temp_spec(LogSpecification::off())` 运行时关闭（utils.rs:134-149）。

调用位置：
- `mcts_trainer.rs:285/293` — MCTS rollout 期间关闭/恢复。
- `ramen/game.rs:1726/1728、2964/2966、3015/3033` — 批量测试期间关闭/恢复。

**问题**：
1. 开关粒度太粗——只能"开/关全部 log"，不能区分"用户信息 vs 程序诊断"。
2. 层级混乱——`mcts_trainer`（Trainer 层）控制 `apply_event`（Game 层）的日志，违反分层。
3. 全局可变状态——多线程/并发测试相互干扰。
4. 仅运行时关闭——`format_args!` 仍执行，浪费 CPU。
5. 多线程输出交错——`flexi_logger` 全局 Mutex 序列化但格式开销仍在每线程上发生。

### 2.3 输出格式化方法

`crates/umasim/src/explain.rs` 提供 7 个独立格式化函数。Game/Action 层各处又有 `pub fn explain()`：
- `BaseGame::explain()`、`OnsenGame::explain()`、`RamenGame::explain()`、`RamenGame::explain_ramen_info()`
- `Uma::explain()`、`FriendState::explain()`、`SupportCard::explain()`
- `BaseAction`/`BasicAction`/`OnsenAction`/`RamenAction` 的 `Display`
- `EventData::explain()`、`EventChoice::explain()`、`ActionValue::explain()`

**说明（用户澄清）**：`explain.rs` 的设计是为"存在多义性"或临时结构（如 `Array5` 既可表示五维属性也可表示训练加成）提供带语境的"诊断快照"，不是用户展示。重构时 `explain()` 保留为开发者诊断工具，新增 `GameView::view()` 作为面向用户/AI 的结构化展示。

### 2.4 已规划但未实施

Phase 3 规划中提到的类型 `GameEvent`、`GameView`、`TurnSnapshot`、`GameResult`、`DiagnosticLog` 当前不存在。

## 3. 目标架构

### 3.1 三层输出通道

```
┌─────────────────────────────────────────────────────────────┐
│ 通道 1：diag!（可裁剪的诊断日志）                            │
│   - 用途：排查规则 bug、确认分支走向（Game/Action 层使用）  │
│   - 实现：宏包装 log::info!/log::warn!，cfg feature 控制   │
│   - 默认：rule_diagnostics feature 开启（开发时可见）       │
│   - 关闭：编译期裁剪（feature 关时宏 = no-op，零开销）     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ 通道 2：log crate（业务日志，永不裁剪）                      │
│   - 用途：AI 推荐结果、决策进度、异常告警                    │
│   - 实现：log::info!/log::warn!/log::error!，target="decision"│
│   - 默认：始终输出                                          │
│   - 关闭：仅靠 RUST_LOG 级别过滤（运行时）                  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ 通道 3：GameEvent + GameView（结构化数据，按需收集）          │
│   - 用途：回放分析、UI 数据源、AI 助手信息流                  │
│   - 实现：GameEvent enum（Collector trait 收集）           │
│         GameView struct（Game::view() 构造）                │
│   - 默认：None（不收集，零开销）                             │
│   - 关闭：OutputConfig.statistics_level 控制                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 crate 级别编译期裁剪关系

```
┌─ 二进制：ramen_manual（手写策略玩家测试）──────────────┐
│ feature = "rule_diagnostics" 开启                          │
│   决策层日志（log::info! target="decision"） ✅ 输出     │
│   规则层日志（diag! 展开为 log::info!）         ✅ 输出     │
└──────────────────────────────────────────────────────────────┘
┌─ 二进制：umaai（AI 助手信息流）──────────────────────────┐
│ umasim 依赖 default-features = false                       │
│   决策层日志（log::info! target="decision"） ✅ 输出（推荐）│
│   规则层日志（diag! 宏 = no-op）           ❌ 不存在      │
└──────────────────────────────────────────────────────────────┘
```

## 4. 核心设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 日志库 | **保留 flexi_logger** | 已支持屏幕+文件双输出；仅缺编译期裁剪（用 `diag!` 补足） |
| 编译期裁剪方式 | **`diag!` 宏 + `#[cfg(feature)]`** | 关闭时连 `format_args!` 都不执行，真正零开销 |
| feature 默认 | **默认开启** | 开发者体验好；`umaai` 显式 `default-features = false` 表达意图 |
| macro 命名 | **`diag!`** | 简化命名 |
| 关闭范围 | **关闭 info 和 warn** | 排查规则 bug 用；error 是异常信号须保留 |
| 决策层日志 | **保持 log::info!/warn!/error!** | 业务输出，AI 推荐结果必须可见 |
| 决策层梳理 | **后续再做** | 当前决策层 info 混入 print 信息，先记录后梳理 |
| explain() 去留 | **保留** | 多义性诊断快照，与用户展示无关，不能删 |
| `GameView` 与 `explain` 关系 | **并存**（关注点不同） | `explain()` = 开发者诊断快照；`view()` = 用户/AI 视角 |
| `disable_log/enable_log` | **保留过渡期** | 迁移完成后删除 |
| tracing 引入 | **推迟到 Phase 5** | Phase 3 核心目标"按需开关"用 `diag!` 已覆盖 |
| `GameEvent` 同步路径 | **`EventCollector` trait** | 类型安全、可测试、可扩展 |
| `GameEvent` 多线程 | **`ChannelCollector` 作为可选 backend** | MCTS rollout 场景需要 worker 发事件到主线程 |
| `GameEvent` 默认收集级别 | **None** | 零开销；按需通过 OutputConfig 开启 |
| `GameEvent` Serialize | **是** | 支持回放文件、analyzer 持久化 |

## 5. 实施步骤

> 渐进式路线：先建骨架再迁移，每步独立可编译可提交。

### 5.0 阶段 0：测试日志简化（最终方案，本次先做）

**目标**：测试场景下日志只输出 stderr、不写文件，复用现有重入保护机制。

**背景**：
- 现有 `init_logger` 已通过 `INIT_LOCK` + `LOGGER_INIT_DONE`（双重检查锁）解决了并行测试下"log crate 全局状态只能初始化一次"的竞争问题（2026-08-19 修复）。
- 测试场景下不需写文件——cargo test 默认按测试名隔离捕获 stderr，天然不会输出交错。
- 正式环境 logger 是全局单例、单线程串行调用，不存在 logger 层重入；`disable_log/enable_log` 的 `push_temp_spec/pop_temp_spec` 栈式语义在生产环境 OK，按阶段 5 规划后续优化。
- 测试场景下的 `disable_log/enable_log` 调用（5 处）按用户确认在阶段 0 一并删除——cargo test 已隔离，不需要手动禁用。

**方案**：新增 `init_test_logger(spec)`，只输出到 stderr；测试代码把 `init_logger("test", info)` 改为 `init_test_logger(info)`。

**步骤**：

1. **新增 `init_test_logger(spec: &str)`**（`crates/umasim/src/utils.rs`）：
   ```rust
   /// 测试场景专用 logger 初始化：只输出 stderr、不写文件。
   /// 共享全局 LOGGER 单例；并行测试首次 init 串行化由现有 INIT_LOCK + LOGGER_INIT_DONE 保证。
   pub fn init_test_logger(spec: &str) -> Result<()> {
       // 复用现有双重检查锁（与 init_logger_with 相同的 fast path）
       if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
           return Ok(());
       }
       let lock = INIT_LOCK.get_or_init(|| Mutex::new(()));
       let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
       if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
           return Ok(());
       }
       let result: Result<()> = (|| {
           let logger = flexi_logger::Logger::try_with_str(spec)?
               .format_for_stderr(log_format)
               .log_to_stderr()      // 只 stderr，不写文件
               .start()?;
           let _ = LOGGER.set(Mutex::new(logger));
           Ok(())
       })();
       if result.is_ok() {
           LOGGER_INIT_DONE.store(true, Ordering::Release);
       }
       result
   }
   ```

2. **迁移测试调用点**：把 `init_logger("test", "info")` 改为 `init_test_logger("info")`（约 97 处，分布在 12 个文件中）。

4. **删除测试中的 `disable_log/enable_log` 调用**（5 处）：
   - `crates/umasim/src/trainer/mcts_trainer.rs:285/293`
   - `crates/umasim/src/game/ramen/game.rs:1726/1728、2964/2966、3015/3033`
   - 这些是测试场景（位于 `#[cfg(test)]` 模块），cargo test 已隔离，不需要手动禁用。

5. **不动**：
   - 业务 binary 的 `init_logger` / `init_logger_with` / `init_logger_stdout`（main.rs、ramen_manual 等）
   - `disable_log/enable_log` 函数实现本身（按阶段 5 规划后续优化）
   - 测试中的 `info!` / `warn!` 调用（保留走 logger 输出到 stderr）
   - 现有 `LOGGER` / `LOGGER_INIT_DONE` / `INIT_LOCK` 重入保护机制

**验证**：
- `cargo test --release -p umasim --lib` 全部通过。
- `cargo nextest run -p umasim --release` 并行运行无重入问题。
- 业务 binary `cargo build --release -p umaai -p ramen_manual` 正常编译。

### 5.1 阶段 1：骨架搭建

**目标**：建立 `output/` 模块 + `diag!` 宏 + Cargo features，不修改现有任何日志调用。

1. **新增 `crates/umasim/src/output/` 模块**（4 个文件）：
   - `output/mod.rs` —— `OutputConfig`、`GameOutput`（Game 持有的输出句柄集合）。
   - `output/diagnostic.rs` —— `diag!` 宏（`#[cfg(feature = "rule_diagnostics")]` 控制）+ `DiagnosticConfig`。
   - `output/event.rs` —— `GameEvent` enum + `EventCollector` trait + `ChannelCollector`（mpsc）+ `NullCollector`（默认）。
   - `output/view.rs` —— `GameView` 骨架（字段最小化，详细定义留到阶段 4）。
2. **Cargo features 改造**：
   - `umasim/Cargo.toml` 加 `rule_diagnostics` feature（默认开启）和 `[[bin]]` 的 `required-features = ["rule_diagnostics"]`。
   - `umaai/Cargo.toml` 显式 `default-features = false`。
   - `analyzer/Cargo.toml` 同上。
3. **最小验证测试**：
   - `diag!` 宏编译期裁剪效果测试（feature 开/关对比 binary 大小）。
   - `GameEvent` 收集基本集成测试。
   - `OutputConfig` 序列化往返测试。
4. **保留**：`disable_log/enable_log`、`utils.rs` 现有 flexi_logger 初始化代码。

### 5.2 阶段 2：规则层迁移（Game/Action 层 435 处）

**目标**：把规则层所有 `info!`/`warn!` 改为 `diag!`，按模块分组逐个 PR。

1. **迁移顺序**（由内向外）：
   - `game/base/mod.rs`（BaseGame 通用部分）
   - `game/base/action.rs`、`game/base/basic.rs`
   - `game/ramen/`（5 个文件）
   - `game/onsen/`（待 Phase 6 删除前再迁）
2. **每步验证**：release 编译 + 现有测试通过。
3. **保留**：`log::error!`（按设计永不裁剪）；`game/uma.rs::explain()`、`explain()` 链等诊断字符串方法。

### 5.3 阶段 3：决策层日志梳理（后续）

**目标**：用户提到决策层 info 混入 print 信息，需重新分类。

> 本阶段**不在当前任务范围**，留待用户后续单独梳理。本次仅记录到 `ramen_memo_cn.md` 或 `issues.md`。

### 5.4 阶段 4：GameView 完整定义

**目标**：把 `explain()` 中"用户可见部分"抽到 `view()`。

1. 在 `Game trait` 新增 `fn view(&self) -> GameView` 方法。
2. `BaseGame`/`RamenGame`/`OnsenGame` 实现默认版本。
3. `umaai`/`analyzer` 接入 `GameView`，替代当前的 `println!("{}", game.explain()?)`。
4. `explain()` 保留为开发者诊断快照（含所有 Array5 标签、所有字段）。

### 5.5 阶段 5：disable_log 优化

**目标**：迁移完成后删除 `disable_log/enable_log`，用 `OutputConfig.diagnostic_enabled` 替代。

1. 在 `mcts_trainer` 和 `ramen/game.rs` 测试中替换为 `output.diagnostic.set_enabled(false)`。
2. 替换完成后从 `utils.rs` 删除 `disable_log/enable_log` 公共函数。
3. 删除 `LOGGER_INIT_DONE`、`INIT_LOCK` 等过渡期锁。

### 5.6 阶段 6：性能验证

**目标**：验证编译期裁剪的实际收益。

1. `cargo bloat` 对比 `umaai` 在 `rule_diagnostics` 开/关下的 binary 大小。
2. MCTS rollout benchmark 对比（搜索速度、内存占用）。
3. 验证 `GameOutput::default()` 零开销（`GameEvent` 收集 None 时 `apply_event` 内部调用应该被分支优化掉）。

## 6. 决策记录

### 6.1 已确认（2026-08-20 用户确认）

| 决策 | 选项 | 备注 |
|------|------|------|
| 日志库 | 保留 flexi_logger | 不引入 tracing（Phase 5 再评估） |
| 编译期裁剪 | `diag!` 宏 + cfg feature | 在 log 基础设施上叠加层 |
| 语义 | `diag!` 表示"可裁剪的诊断 log" | 关闭 info 和 warn，保留 error |
| 决策层日志 | 保持 `log::info!/warn!/error!` | 不动；后续再梳理 |
| macro 命名 | `diag!` | 简化命名 |
| explain() 去留 | 保留 | 多义性诊断快照，与 GameView 并存 |
| GameView 引入 | 新增 | 用户/AI 视角结构化展示 |
| disable_log | 保留过渡期 | 迁移完成后删除（阶段 5） |
| tracing | 推迟到 Phase 5 | 不在当前阶段引入 |

### 6.2 仍待决策

| 决策点 | 选项 | 备注 |
|--------|------|------|
| GameOutput 在 Game 结构体中的形态 | A: `output: GameOutput` 字段（运行时携带，轻微克隆成本）<br>B: 线程局部变量（零克隆但与 trait 接口兼容性差） | 倾向 A |
| 统计级别粒度 | A: 4 级（None/Summary/Turn/Detailed）<br>B: 2 级（None/Turn） | 倾向 A（兼容 analyzer 等上层需求） |
| `GameOutput::diagnostic_enabled` 字段 | A: 保留（feature 关时无意义，feature 开时运行时切换）<br>B: 不保留（编译期决定就够了） | 倾向 A |
| mcts_trainer 日志归类（阶段 3） | A: 全部保留业务日志<br>B: 部分内部细节改 `diag!` | 倾向 A（现有都是推荐结果） |

## 7. 验证

### 7.1 阶段 1 验证

- 全 workspace release 编译通过。
- `umasim` 默认 build 包含 `rule_diagnostics` feature，所有 binary 通过编译。
- `umaai` 默认 build **不** 包含 `rule_diagnostics` feature，依赖 `umasim` 时显式 `default-features = false`。
- 新增测试：`diag!` 宏在 feature 开/关下 binary 大小差异；`GameEvent` 收集往返；`OutputConfig` 序列化。
- `ramen_manual`（手写策略玩家测试）能正常启动并显示规则日志。

### 7.2 阶段 2-6 验证

- 435 处规则层日志全部迁移后，release 编译 + 124 个 lib 测试通过。
- `cargo bloat` 显示 `umaai` binary 中 `diagnostic` target 调用数为 0。
- MCTS rollout benchmark 在 `umaai` 默认配置下吞吐量提升（相对 `disable_log/enable_log` 版本）。
- `explain()` 与 `view()` 测试均通过，互不影响。

## 8. 关联文档

- `ramen_refactor_development_plan.md` Phase 3：高层目标
- `config_refactor_plan.md`：本期实施的参考模板（结构/节奏）
- `project_context.md`：实施完成后更新相关章节
- `changelog.md`：实施完成后统一更新