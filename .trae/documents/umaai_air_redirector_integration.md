# UmaAI ↔ AIRedirector 集成方案

> 状态：方案定稿 v2（2026-09 用户多次拍板后整理）
> 目标：让 AIRedirector（C# 插件）能稳定接入 umaai（包括拉面杯剧本）
> 边界：仅修改 umaai-rs（Rust 项目）+ AIRedirector 端极小改动（不加 file watcher）

## 1. 当前流程概览

```text
URA (C#) 捕捉回合数据
    │
    ├── AIRedirector (C#)：按 scenarioId 拉起对应版本的 AI 子进程
    │
    └── SendGameStatusPlugin (C#)：写 thisTurn.json
            │
            ▼
UMA (Rust, umaai-rs)
    1. UraFileWatcher (notify / inotify) 监听文件变更
    2. parse_game 解析 baseGame + scenarioId 分发
    3. GameStatus::into_game 重建 RamenGame / OnsenGame
    4. MctsTrainer 计算最佳动作
    5. DecisionSink 输出（玩家模式屏幕 / AIRedirector 模式 stdout JSON）
            │
            ▼
AIRedirector (C#)：从 umaai 子进程 stdout 抓输出，识别 JSON 行并渲染回小黑板
```

## 2. 已识别的核心问题（用户原始反馈）

| # | 问题 | 严重度 | 处置 |
|---|---|---|---|
| P0-1 | **小黑板接不到回合数据**：路径错 / 未写入 / 解析失败等多种原因，导致 AI 没反应，排查困难 | 高 | **本方案核心**（§3.1） |
| P1-1 | JSON 解析效率：开销大头在 MCTS，不是 JSON | — | 砍掉（不优化） |
| P1-2 | AIRedirector vs 屏幕输出：两种消费方需要不同输出格式 | 中 | **本方案核心**（§3.2） |
| P1-3 | 异步搜索 + 玩家覆盖 | 中 | P2 之后再解决 |
| P1-4 | 搜索预算按年份差异化 | 低 | P2 之后再解决 |
| P1-5 | 配置热加载 | — | 砍掉（不做） |

## 3. 总体设计：CLI 模式分流 + 现有 output 基础设施复用

### 3.0 关键设计原则

1. **CLI 参数 `--json` 决定模式**：
   - 玩家直接启动 umaai → 默认人类屏幕模式
   - AIRedirector 启动子进程时传 `--json` → stdout 严格只 JSON
2. **复用 umasim 现有 output 基础设施**：
   - `DecisionInfo`（decision.rs）：协议格式结构体，已 Serialize/Deserialize
   - `DecisionReasonSink` trait + `NoopSink` / `LogJsonSink`（reason.rs）：已有 sink 模式
   - `RecordingTrainer` / `TurnDecision`（turn_flow.rs）：已记录每回合候选
   - 不重新发明 `OutputSink` / `JsonSink` / `HumanSink`
3. **AIRedirector 端几乎 0 改动**：不加 file watcher，仅在启动参数加 `--json` + HandleOutput 加 JSON 行识别
4. **崩溃防护范围收窄到 stdout 写一行 JSON 处**

### 3.1 P0-1：watcher 健壮性与可观测性

**现状**：`UraFileWatcher::do_poll` 只 listen `notify` 事件，初始化时读一次现存文件。失败时只 `warn!` 一句，没有结构化日志。

**方案**：

1. **失败分类**：把"AI 没反应"分解为以下 5 类，每类对应不同日志关键字：

   | 现象 | 根因 | 日志关键字 | AIRedirector 提示 |
   |---|---|---|---|
   | 持续无事件 | URA 路径错 / 子进程隔离目录权限不足 | `WATCH_NO_PATH` | "小黑板目录不存在" |
   | 持续无事件 | notify 后端初始化失败 | `WATCH_INIT_FAIL` | "监听初始化失败" |
   | 事件到但读不到文件 | 写入原子性问题（半截 JSON） | `WATCH_READ_EMPTY` | "小黑板写入异常" |
   | 文件读到但 JSON 坏 | C# 端序列化 bug | `WATCH_PARSE_FAIL` | "回合数据格式异常" |
   | JSON 好但 scenarioId 不匹配 | URA 端与 AI 版本不匹配 | `WATCH_SCENARIO_MISMATCH` | "剧本版本不匹配" |

2. **重试与背压**：
   ```text
   do_poll 读到空 → 短延时（50ms）重读，3 次失败才 warn
   连续 N 次（默认 10）空事件 → 输出"心跳"提示小黑板是否还在运行
   ```

**改动范围**：`crates/umaai/src/protocol/urafile.rs`（+100 行），无业务影响。

### 3.2 P1-2：输出格式切换（CLI `--json` + 复用 DecisionSink 模式）

#### 3.2.1 两种模式总览

| 模式 | 触发 | stdout | stderr | 文件 |
|---|---|---|---|---|
| **human**（默认，玩家手玩） | 直接双击 / cmd 启动 | 维持现状（println! + 彩色 + 启动横幅） | 日志 | 不写 |
| **json**（AIRedirector 模式） | 传 `--json` 参数 | **严格只 JSON 行**，每行一个决策 | 日志 + 启动横幅 + 彩色文字 | 不写 |

**关键**：**stdout/stderr 分流**代替前缀过滤（参考 `DecisionReasonSink` 现有模式）。AIRedirector 只消费 stdout。

#### 3.2.2 复用 DecisionInfo，不重新发明协议

**已存在的 `DecisionInfo`**（`umasim/output/decision.rs`，161 行，已 Serialize/Deserialize）：

```rust
pub struct DecisionInfo {
    pub action_index: usize,                              // 选中候选索引
    pub score: f32,                                       // 选中评分
    pub candidate_scores: Vec<f32>,                      // 全部候选评分（截断到 top-N）
    pub reason: Option<String>,                           // 决策理由
    pub elapsed_ms: Option<u64>,                          // 决策耗时
    pub search_depth: Option<u32>,                        // MCTS 搜索深度
    pub visit_count: Option<u32>,                         // 节点访问数
    pub score_breakdown: Option<HashMap<String, f32>>,    // 评分维度分解
    pub scenario_extra: Option<serde_json::Value>,        // 弹性挂载点（luck 等）
}
```

**AIRedirector 只关心这些字段**（用户确认）：
- `action_index` + `score` + `candidate_scores`（决策基本三件套）
- `reason`（人类可读理由，AIRedirector 可选渲染）
- `scenario_extra.luck_delta` / `total_luck_score`（§3.3 详细定义）

**AIRedirector 不关心**：
- 每回合五维变化（玩家操作后才发生，AI 推荐时未知）
- 训练即时预览（已经是旧 C++ AI 时代的产物，新设计不需要）

#### 3.2.3 候选评分截断：top-N（由 SearchConfig 控制，不引入新 CLI 选项）

**用户确认**：AIRedirector 收到的 `candidate_scores` 与屏幕显示一致，截断数由 `SearchConfig::reason_max_display: 5`（`search/config.rs:106`）控制。

**CLI 边界原则**：除 `--json` 模式开关外，**不引入新命令行参数**。所有可调项走 `game_config.toml` / `default_config.toml`（SearchConfig 等已有字段）。

```rust
// 在生成 DecisionInfo 时同步截断，N 由 SearchConfig 决定
let search_config = load_search_config(&game_config)?;
let max_n = search_config.reason_max_display;  // 默认 5

let mut sorted_scores: Vec<(usize, f32)> = candidate_scores.iter()
    .enumerate().map(|(i, &s)| (i, s)).collect();
sorted_scores.sort_by(|a, b| b.1.total_cmp(&a.1));  // 降序
let top_n: Vec<f32> = sorted_scores.iter().take(max_n).map(|(_, s)| *s).collect();
```

如果以后需要更多候选，改 `SearchConfig::reason_max_display` 配置即可（影响屏幕显示 + AIRedirector 候选数量）。

#### 3.2.4 新增 DecisionSink trait（模仿 DecisionReasonSink 模式）

**模仿** `umasim/output/reason.rs:187` 现有 trait 模式：

```rust
// 新增：umasim/output/sink.rs（与 DecisionReasonSink 同模式）
pub trait DecisionSink: Send + Sync {
    /// 发出一条决策原始数据
    fn emit(&self, info: &DecisionInfo, view: &GameView);
}

/// 默认实现：丢弃原始数据（玩家模式走 println!，不进 sink）
pub struct NoopSink;

impl DecisionSink for NoopSink {
    fn emit(&self, _info: &DecisionInfo, _view: &GameView) {}
}

/// 玩家屏幕 sink：渲染为人类文字 + println!
pub struct HumanReadableSink;

impl DecisionSink for HumanReadableSink {
    fn emit(&self, info: &DecisionInfo, view: &GameView) {
        // 用现有 RenderingTrainer / explain_distribution 等渲染
        println!("AI 选择: 第 {} 个动作（评分: {}）", info.action_index, info.score);
        println!("回合 {} · 阶段 {} · 候选评分：{:?}", view.turn, view.scenario, info.candidate_scores);
        // ...
    }
}

/// AIRedirector sink：序列化 JSON + println! 到 stdout
pub struct StdoutJsonSink;

impl DecisionSink for StdoutJsonSink {
    fn emit(&self, info: &DecisionInfo, view: &GameView) {
        let payload = serde_json::json!({
            "schema_version": 1,
            "turn": view.turn,
            "scenario": view.scenario,
            "action_index": info.action_index,
            "score": info.score,
            "candidate_scores": info.candidate_scores,
            "reason": info.reason,
            "elapsed_ms": info.elapsed_ms,
            "scenario_extra": info.scenario_extra,  // 含 luck_score（§3.3）
        });
        // 序列化返回 Result，失败时回退到占位 JSON（不 panic）
        match serde_json::to_string(&payload) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("[ERROR] decision serialize failed: {e}");
                println!("{}", serde_json::json!({
                    "schema_version": 1,
                    "error": "serialize_failed",
                    "turn": view.turn,
                }));
            }
        }
    }
}
```

#### 3.2.5 CLI 解析（lexopt，与项目惯例一致）

```rust
// crates/umaai/src/main.rs
use lexopt::prelude::*;

#[derive(Default)]
struct Args {
    json: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Long("json") => args.json = true,
            Short('h') | Long("help") => print_help_and_exit(),
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    
    // 根据模式选择 sink
    let sink: Arc<dyn DecisionSink> = if args.json {
        Arc::new(StdoutJsonSink)
    } else {
        Arc::new(HumanReadableSink)
    };
    
    // 启动横幅：两种模式都走 stderr（避免 stdout 污染 JSON 流）
    eprintln!("{}", to_art("UMAAI 0.26"...));
    
    // ... 启动逻辑、watcher、主循环 ...
}
```

**为什么用 lexopt 不是 clap**：
- 项目惯例：`umasim/src/bin/bench_base.rs` / `bench_compositions.rs` 主 bin 全部用 lexopt
- lexopt 编译产物更小，跨平台无兼容性问题
- umaai Cargo.toml 已声明 clap 但实际代码里**未使用**，跟随 umasim 主 bin 惯例用 lexopt 更一致

#### 3.2.6 emit 失败处理（JSON 模式专属）

**澄清**：`StdoutJsonSink::emit` 本身**不会 panic**——`serde_json::to_string` 返回 `Result`，`println!` 也不 panic。Emit 唯一可失败点是序列化（NaN/Inf 浮点等），用 `Result` 兜底为占位 JSON 即可。

JSON 模式下保持 stdio 干净的措施：

1. **序列化失败回退到占位 JSON**（见 §3.2.4 `StdoutJsonSink::emit` 代码）——占位 JSON 包含 `error: "serialize_failed"` 字段便于排查

2. **`text-to-ascii-art` 启动横幅必须 `eprintln!`**（避免 stdout 污染）

3. **彩色库 stdout 审计**：
   - `colored` 即使 `--no-color` 也可能输出 ANSI reset，**仅在 human 模式用**
   - JSON 模式**关闭 ANSI**（`colored::control::set_override(false)`）

4. **不引入自定义 panic hook**：默认 panic hook 已写 stderr，不污染 stdout。`emit` 路径无 panic 源，不需要兜底

#### 3.2.7 umaai-rs 改动范围汇总

| 改动点 | 行数 | 备注 |
|---|---|---|
| `DecisionInfo::candidate_n` 字段新增 | +1 字段 ~10 行 | 与 candidate_scores 同长同截断；UCB 下给 luck baseline 算局数加权 |
| `umasim/output/sink.rs` 新增（DecisionSink trait + 3 个 sink） | ~120 行 | 模仿 DecisionReasonSink（Step 3） |
| `MctsTrainer::last_decision()` override（仅 onsen） | ~50 行 | 候选评分按 `mcts_selection` 口径截断；search_output 缓存已有，+1 个 `last_action_idx` 哨兵字段 |
| `RamenMctsTrainer::last_decision()` override | ~120 行 | `LastSearchSummary` 缓存 + reason 文本 `summarize_ramen_reason` 辅助函数；合并搜索路径暂不覆盖（candidates 与 actions 下标不对应） |
| `RamenHandwrittenTrainer::last_decision()` override | ~70 行 | `LastDecisionSummary` 缓存 + `score_breakdown` 字段挂中选者 breakdown |
| `umaai/src/main.rs` CLI 解析 + sink 调用 | ~50 行 | lexopt + dispatch |
| `umaai/src/luck_score.rs` 新增 | ~80 行 | §3.3 详细（Step 5） |
| **合计** | **~500 行** | |

### 3.3 LuckScore 跟踪器（UmaAI 内部维护 + 发给 AIRedirector）

**用户原话（2026-09 拍板，三次迭代定稿）**：

> 1. 所有的运气分都是 **terminal score**（终局分数），不以局部评分作为判据
> 2. 时间线为：第 n 回合选择前分数计为 T(n)，选择某个选项后的运气分 = T(n, action) − T(n)
> 3. **T(n) baseline = mean(T(n, action) over all actions)** ——"不做任何事情时的 rollout 分数"等价于"做了每件事情后的 rollout 分数"的均值

#### 3.3.1 概念定义（terminal score 三层口径）

```text
T(n)             = 第 n 回合选择前的 baseline 终局分
                  = Σ (T(n, action_i) × n_i) / Σ n_i      ← 按局数加权（onsen update_score 历史口径）
                  = "什么都不做时 rollout 的预期终局分"

T(n, action_i)   = 第 n 回合选择 action_i 后的终局分
                  = MCTS candidate_scores[i]                  ← MCTS 直接输出

# 三个口径
回合运气分（实际发生）       = T(n+1) − T(n)
全局运气分（从开局/AI 启动起）  = T(n+1) − T(1)
选项后运气分（AI 推荐预测）    = T(n, action_i) − T(n)
```

**关键数学性质**（用户指出 + 2026-09 重新确认）：

> 不做任何事情时的 rollout 分数（按样本展开）= 做了每件事情后的 rollout 分数的均值（按局数加权）

UCB 下各候选 rollout 样本数 `n_i` 悬殊，**简单按 action 数等权的算术平均 `Σ mean_i / N` 与 onsen 历史 `update_score` 的 `sum / count = Σ (mean_i × n_i) / Σ n_i` 不等价**——后者按局数加权，让少跑的 action 自动降权，工程上更稳。两者数值可差 ~1 万分（见 §3.3 备注）。

**实现含义**：
- T(n) **不需要额外跑 baseline MCTS**——直接从 MCTS 已有的 `(mean_i, n_i)` 取**局数加权平均**
- T(n, action_i) **不需要 clone + apply**——直接是 MCTS 的 `candidate_scores[i]`
- 两个值都是 MCTS 单次搜索的副产品，**零额外开销**
- 候选评分与 `candidate_n` 必须**同步截断到 `reason_max_display`**（同一 i 配对），top-N 之外不参与 baseline 计算

#### 3.3.2 数据来源映射

| 概念 | 已有数据结构 | 来源 |
|---|---|---|
| T(n, action_i) | `candidate_scores[i]`（Vec<f32>） | MCTS `SearchOutput.terminal_stats[i].mean()` |
| T(n) | `Σ (candidate_scores[i] × candidate_n[i]) / Σ candidate_n[i]` | 按局数加权（onsen `update_score` 同口径） |
| T(1) | 首次 MCTS 的 T(1) | chara_id 切换 / AI 首次启动时记录 |
| T(n+1) | 下一回合 MCTS 的 T(n+1) | 每回合 thisTurn.json 到达时记录 |

**`DecisionInfo::candidate_scores` 字段同时承担两个职责**：
- 给 AIRedirector 看候选评分（已有）
- 给 UmaAI 算 luck score（新增复用）

#### 3.3.3 数据结构

```rust
// 新增：crates/umaai/src/luck_score.rs
#[derive(Default)]
pub struct LuckScoreTracker {
    /// T(1)：首次 MCTS 的 baseline terminal（一次性记录）
    initial_terminal_baseline: Option<f64>,
    /// T(n)：上一回合到达时的 baseline terminal（每回合 MCTS 更新）
    prev_turn_terminal_baseline: Option<f64>,
    /// T(n+1) - T(1)：全局运气分
    total_luck: f64,
    /// T(n+1) - T(n)：上一回合回合运气分
    last_turn_delta: Option<f64>,
    /// 上一次的 chara_id（用于检测新游戏重置）
    last_chara_id: Option<u64>,
}

#[derive(Serialize)]
pub struct LuckScoreSnapshot {
    pub initial_terminal_baseline: f64,  // T(1)
    pub current_terminal_baseline: f64,  // T(n+1)
    pub total_luck_score: f64,           // T(n+1) - T(1)
    pub last_turn_delta: Option<f64>,     // T(n+1) - T(n)
}
```

#### 3.3.4 接线

```rust
// crates/umaai/src/main.rs 主循环
loop {
    let contents = watcher.watch("thisTurn.json")?;
    let game = parse_game(&contents)?;
    let chara_id = game.uma().uma_id;
    
    // 1. AI 决策（已包含 MCTS，输出 candidate_scores = T(n, action_i) 数组）
    let action_idx = trainer.select_action(&game, &actions, rng)?;
    let info = trainer.last_decision()
        .expect("Trainer::last_decision must be implemented");
    
    // 2. T(n) baseline = Σ (candidate_scores[i] × candidate_n[i]) / Σ candidate_n[i]
    //    按局数加权（onsen update_score 历史口径 sum/count），零开销
    let t_n_baseline = if info.candidate_n.is_empty() {
        // 手写 / 随机等无局数概念的 trainer：退化为按候选数等权
        info.candidate_scores.iter().map(|&s| s as f64).sum::<f64>()
            / info.candidate_scores.len().max(1) as f64
    } else {
        let total_n: u32 = info.candidate_n.iter().sum();
        info.candidate_scores.iter().zip(info.candidate_n.iter())
            .map(|(&s, &n)| (s as f64) * (n as f64))
            .sum::<f64>()
            / total_n.max(1) as f64
    };
    
    // 3. 更新 luck tracker（计算回合运气分 + 全局运气分）
    let turn_delta = luck_tracker.on_new_turn(chara_id, t_n_baseline);
    
    // 4. 计算每个候选动作的"选项后运气分" = T(n, action_i) - T(n)
    //    AIRedirector 模式需要，player 模式跳过
    let action_luck: HashMap<usize, f64> = if args.json {
        info.candidate_scores.iter().enumerate()
            .map(|(i, &s)| (i, (s - t_n_baseline) as f64))
            .collect()
    } else {
        HashMap::new()
    };
    
    // 5. 挂载到 DecisionInfo.scenario_extra
    let mut extra = serde_json::to_value(luck_tracker.snapshot())?;
    if let Some(obj) = extra.as_object_mut() {
        obj.insert("action_luck".into(), serde_json::json!(action_luck));
    }
    info.scenario_extra = Some(extra);
    
    // 6. 发出
    let view = game.view();
    sink.emit(&info, &view);
}

// 旧的简单算术平均函数已删除——T(n) baseline 改按局数加权（见上方计算块）。
// 手写 / 随机等无 candidate_n 的 trainer 退化路径写在主循环计算块里。
```

#### 3.3.5 跨 chara_id 处理

```rust
impl LuckScoreTracker {
    /// AI 推荐后调用：传入当前回合 MCTS 给出的 T(n) baseline
    /// 返回：当前回合的回合运气分（None 表示首次）
    pub fn on_new_turn(&mut self, chara_id: u64, t_n_baseline: f64) -> Option<f64> {
        // 切局检测：chara_id 变了或 AI 第一次启动
        if self.last_chara_id != Some(chara_id) || self.initial_terminal_baseline.is_none() {
            self.initial_terminal_baseline = Some(t_n_baseline);  // T(1)
            self.prev_turn_terminal_baseline = Some(t_n_baseline); // T(n) ← T(1)
            self.total_luck = 0.0;
            self.last_turn_delta = None;
            self.last_chara_id = Some(chara_id);
            return None;
        }
        
        // T(n+1) - T(n) = t_n_baseline - prev_turn_terminal_baseline
        let delta = self.prev_turn_terminal_baseline.map(|p| t_n_baseline - p);
        if let Some(d) = delta {
            self.last_turn_delta = Some(d);
            // T(n+1) - T(1) = t_n_baseline - initial_terminal_baseline
            self.total_luck = t_n_baseline - self.initial_terminal_baseline.unwrap();
        }
        self.prev_turn_terminal_baseline = Some(t_n_baseline);  // 更新 T(n) ← T(n+1)
        delta
    }
}
```

#### 3.3.6 发给 AIRedirector 的字段

挂在 `DecisionInfo::scenario_extra` 下：

```json
{
  "action_index": 1,
  "score": 56712.3,
  "candidate_scores": [56712.3, 56100.5, 55234.1],   // top-N (MCTS terminal)
  "candidate_n": [1024, 800, 256],                   // 与 candidate_scores 同长同序同截断；AIRedirector 不消费
  "reason": "vs #2 智+180 PT-33",                     // ramen MCTS 走终局维度差值；onsen/手写为空
  "elapsed_ms": 128,
  "scenario_extra": {
    "initial_terminal_baseline": 50000.0,   // T(1)
    "current_terminal_baseline": 52450.0,   // T(n+1) = Σ (score × n) / Σ n（按局数加权）
    "total_luck_score": 2450.0,             // T(n+1) - T(1) 全局运气分
    "last_turn_delta": 125.5,               // T(n+1) - T(n) 回合运气分
    "action_luck": {                        // T(n, action_i) - T(n) 每个候选预测
      "0": -50.0,                            // action 0 相对 baseline -50
      "1": 125.5,                            // action 1 相对 baseline +125.5（AI 推荐）
      "2": 80.0
    }
  }
}
```

AIRedirector 解析示例：
```csharp
if (info.scenario_extra?.last_turn_delta is { } d)
    ui.UpdateTurnLuck(d);
if (info.scenario_extra?.total_luck_score is { } t)
    ui.UpdateTotalLuck(t);
if (info.scenario_extra?.action_luck?[info.action_index] is { } a)
    ui.UpdateActionLuckPrediction(a);
```

#### 3.3.7 口径汇总

| 字段 | 公式 | 数据来源 | 备注 |
|---|---|---|---|
| `initial_terminal_baseline` | T(1) | MCTS baseline 首次记录 | chara_id 切换 / AI 首次启动 |
| `current_terminal_baseline` | T(n+1) | `Σ (candidate_scores[i] × candidate_n[i]) / Σ candidate_n[i]` | 当前回合 baseline（按局数加权） |
| `total_luck_score` | T(n+1) − T(1) | 上述两个差值 | 全局运气分累计 |
| `last_turn_delta` | T(n+1) − T(n) | 两次 baseline 差值 | 回合运气分（首回合 None） |
| `action_luck[i]` | T(n, action_i) − T(n) | candidate_scores[i] − baseline | 选项后运气分 |
| `candidate_scores[i]` | T(n, action_i) | MCTS terminal_stats[i].mean() | 已在 `DecisionInfo` 字段 |
| `candidate_n[i]` | n_i | MCTS `ActionResult[i].count()` | **与 `candidate_scores` 同长同截断**；AIRedirector 不消费 |

#### 3.3.8 关键性质

1. **零额外 MCTS 开销**：T(n) = `Σ (mean_i × n_i) / Σ n_i`（按局数加权），候选 terminal 已经是 MCTS 输出
2. **无 clone + apply**：选项后运气分直接从 candidate_scores 数组计算，不修改 game state
3. **schema 兼容**：candidate_scores 字段在 AIRedirector 模式下已经存在；luck 字段新增但挂在 scenario_extra 下，不破坏协议；candidate_n 字段新增但 AIRedirector 不消费（C# 端 JsonDocument 默认宽容解析，可忽略未定义字段）
4. **chara_id 重置**：检测到切局时 T(1) 重置，total_luck 清零
5. **top-N 同步截断**：candidate_scores 与 candidate_n 必须按同一 i 同步排序截断到 `reason_max_display`，否则 T(n) baseline 计算时索引错位

### 3.4 P0-3：parse_game 按 scenarioId 分发（拉面协议定稿的后续）

**现状**：`main.rs:219` 硬编码 `parse_game::<GameStatusOnsen>`。

**方案**：

```rust
let contents = watcher.watch("thisTurn.json")?;
let value: Value = serde_json::from_str(&contents)?;
let scenario_id = value.get("baseGame")
    .and_then(|b| b.get("scenarioId"))
    .and_then(|s| s.as_u64())
    .ok_or_else(|| anyhow!("missing scenarioId"))?;

match scenario_id {
    12 => parse_game::<GameStatusOnsen>(&contents),
    14 => parse_game::<GameStatusRamen>(&contents),
    _ => Err(anyhow!("unsupported scenarioId: {}", scenario_id)),
}
```

**前置依赖**：
- `crates/umaai/src/protocol/ramen.rs`（新增，见 `ramen_protocol_v2.md` §5）
- `umasim` 核心层 `RamenGame::from_external_state`（新增）

**改动范围**：约 +30 行。

## 4. AIRedirector 端改动（极小）

### 4.1 UmaAiProcessStartInfo.cs 加 `--json` 参数

```csharp
internal static class UmaAiProcessStartInfo
{
    public static ProcessStartInfo Create(string executablePath, bool jsonMode = false)
    {
        var args = jsonMode ? "--json" : "";
        return new ProcessStartInfo
        {
            FileName = fullPath,
            WorkingDirectory = workingDirectory,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
    }
}
```

### 4.2 AIRedirectorConfig.cs 加拉面字段

```csharp
public class AIRedirectorConfig
{
    public bool UAF { get; set; }
    public string UAF_Path { get; set; }
    public bool Cook { get; set; }
    public string Cook_Path { get; set; }
    public bool Mecha { get; set; }
    public string Mecha_Path { get; set; }
    public bool Legend { get; set; }
    public string Legend_Path { get; set; }
    
    // 新增：拉面杯剧本支持
    public bool Ramen { get; set; }
    public string Ramen_Path { get; set; }
}
```

### 4.3 Class1.cs 加拉面分支 + JSON 行识别

```csharp
void StartProcess(ChildProcessManager manager, string name, string path,
                  bool applyToLegend = false)
{
    var process = new Process { StartInfo = UmaAiProcessStartInfo.Create(path, jsonMode: true) };
    process.OutputDataReceived += (_, e) => HandleOutput(e.Data, applyToLegend);
    // ...
}

void HandleOutput(string? line, bool applyToLegend)
{
    if (line is null) return;
    
    // 尝试解析 JSON 决策
    if (TryParseUmaAiDecision(line, out var decision))
    {
        ApplyDecision(decision);  // 路由到对应剧本渲染
        return;
    }
    
    // 非 JSON 行（理论上 JSON 模式不该有，但保留兜底）
    processOutput.Writer.TryWrite(new(line, applyToLegend));
}

bool TryParseUmaAiDecision(string line, out UmaAiDecision decision)
{
    decision = default;
    try
    {
        using var doc = JsonDocument.Parse(line);
        var root = doc.RootElement;
        if (!root.TryGetProperty("schema_version", out _)) return false;
        // ... 解析 action_index, score, candidate_scores, scenario_extra ...
        decision = ...;
        return true;
    }
    catch
    {
        return false;
    }
}
```

### 4.4 加拉面分支（Class1.cs）

```csharp
public void Initialize(IPluginContext context)
{
    // ...
    ValidateConfiguredPath("UAF", config.UAF, config.UAF_Path);
    ValidateConfiguredPath("Cook", config.Cook, config.Cook_Path);
    ValidateConfiguredPath("Mecha", config.Mecha, config.Mecha_Path);
    ValidateConfiguredPath("Legend", config.Legend, config.Legend_Path);
    ValidateConfiguredPath("Ramen", config.Ramen, config.Ramen_Path);  // 新增

    if (config.UAF) StartProcess(manager, "UAF", config.UAF_Path);
    if (config.Cook) StartProcess(manager, "Cook", config.Cook_Path);
    if (config.Mecha) StartProcess(manager, "Mecha", config.Mecha_Path);
    if (config.Legend) StartProcess(manager, "Legend", config.Legend_Path, applyToLegend: true);
    if (config.Ramen) StartProcess(manager, "Ramen", config.Ramen_Path);  // 新增
}
```

**AIRedirector 端总改动量**：约 +60 行，**无 file watcher / 无大幅重构**。

## 5. 暂不做（标记 P2）

| 项 | 触发条件 |
|---|---|
| 异步搜索 + 玩家操作覆盖处理 | AIRedirector 接拉面稳定后，玩家投诉"搜索到一半被覆盖"时再议 |
| 搜索预算按年份差异化 | 手写策略评分稳定后再优化；当前 handwritten 已按 turn 自动调整 |
| 配置热加载 | 无明确用户需求 |
| JSON 解析优化 | 用户明确大头在 MCTS，不优化 |

## 6. 实施顺序（推荐）

按依赖关系，从外围到核心：

1. **Step 1：watcher 健壮性**（§3.1）—— 独立、低风险、可单独验收
2. **Step 2：核心层 last_decision() override**（§3.2.7）—— MctsTrainer / RamenMctsTrainer / RamenHandwrittenTrainer override，**这是核心改动**
3. **Step 3：DecisionSink trait + 三种 sink**（§3.2.4）—— 模仿 DecisionReasonSink 模式
4. **Step 4：CLI `--json` 分流 + main.rs sink 调用**（§3.2.5）—— umaai 端最薄一层
5. **Step 5：LuckScoreTracker + 挂载 scenario_extra**（§3.3）—— 一次性接线
6. **Step 6：parse_game 按 scenarioId 分发**（§3.4）—— 拉面协议落地的前置
7. **Step 7：RamenGame::from_external_state + 测试**（§3.4 前置依赖）—— 用 151 份样本驱动
8. **Step 8：AIRedirector 端**（§4）—— 加拉面分支 + `--json` + JSON 行识别

## 7. 验证策略

- **Step 1**：人工模拟"路径错 / 未写入 / JSON 坏" 5 类失败，每类触发对应日志关键字
- **Step 2-5**：单测覆盖
  - `MctsTrainer::last_decision()`：跑一局取所有决策，断言 `score` / `candidate_scores` / `elapsed_ms` 非零
  - `DecisionSink` 各实现：单元测试 emit 不 panic / 输出格式正确
  - `LuckScoreTracker`：跨 chara_id 重置、累加正确
- **Step 6-7**：用 151 份样本驱动 turn import 测试（参考 `ramen_protocol_v2.md` §4）
- **Step 8**：实机联调
  - AIRedirector 拉起 `--json` 模式，stdout 能解析为合法 JSON
  - scenario_id=14 自动路由到 GameStatusRamen
  - 拉面剧本完整跑通 77 回合

## 8. 相关文档

- `ramen_protocol_v2.md`：拉面在线协议 V2 定稿
- `umaai-rs-上游重构需求与三层架构建议.md`：核心 / 通道 / 更新三层架构参考
- `changelog.md`：实施时按用户规范更新（每条一行，禁止多句）
