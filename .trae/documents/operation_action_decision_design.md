# Operation/Action/Decision 层次设计（最终版）

## 1. 核心约束

### 约束 1：分身分配是随机的

部分拉面带"分身"效果，吃面后在**特定训练位置随机增加一个人物**。这意味着：

- 选面前：不知道分身会落在哪个训练位置
- 面选择后、操作选择前：随机 resolve 分身分布
- 选操作必须在分布确定之后——最优训练取决于分身位置

### 约束 2：保留现有命名

不引入 `Policy`、`DecisionRequest`、`DecisionResponse` 等新抽象。在现有 `Trainer` trait 基础上扩展。

---

## 2. 现有结构已足够支撑

分析现有 `OnsenGame` 的代码模式：

```
list_actions() 按阶段分发：
  Bathing → [UseTicket(true/false)]
  Train   → [Train(0..4), Race, Sleep, NormalOuting, FriendOuting, Clinic, PR, ...]

run_stage() 按阶段推进：
  Begin → Distribute → Train → AfterTrain → (next turn)
  每个决策阶段调用 list_and_apply_action → Trainer::select_action

FlatSearch（搜索）对阶段无感知：
  simulate():
    克隆 → apply_action(当前候选) → 循环 run_stage 到终局
    run_stage 内部自然处理后续所有阶段
```

**关键结论：现有 `ActionEnum` + `Game::list_actions()` + `Trainer::select_action()` + 阶段驱动的模式已经覆盖了「跨多个阶段、异构动作类型、随机过渡」的全部需求。**

---

## 3. 拉面杯的映射

### 3.2 动作枚举

拉面杯使用一个统一的动作枚举，覆盖所有阶段。内层直接复用现有 `BaseAction`，不引入新类型：

```rust
/// 拉面杯完整动作枚举
enum RamenGameAction {
    /// 不吃面（RamenSelection 阶段）
    Nothing,
    /// 吃指定面，随机分配分身（RamenSelection 阶段）
    Eat(RamenType),
    /// 执行基础行动（MainAction 阶段），直接复用现有 BaseAction
    Act(BaseAction),
}
```

`BaseAction` 保持现有定义不变：

```rust
pub enum BaseAction {
    Train(i32),         // 属性训练
    Race,               // 比赛
    Sleep,              // 休息
    FriendOuting,       // 友人外出
    NormalOuting,       // 普通外出
    Clinic,             // 治病
}
```
### 3.3 随机分身处理

分身分配的随机性封装在 `Eat` 的 `apply()` 内部，对 `list_actions` 和 `Trainer` 完全透明：

```rust
fn apply(&self, game: &mut RamenGame, rng: &mut StdRng) -> Result<()> {
    match self {
        RamenGameAction::Nothing => Ok(()),
        RamenGameAction::Eat(ramen) => {
            game.consume_ingredients(ramen.recipe())?;
            game.apply_ramen_buff(ramen, rng);  // 含随机分身分配
            Ok(())
        }
        RamenGameAction::Act(base) => {
            // 通过 Deref<Target=BaseGame> 派发到 BaseAction 的方法
            base.apply(game, rng)
        }
    }
}
```


## 4. Trainer Trait

**完全不需要新增专用方法。** `Trainer::select_action` 签名保持不变：

```rust
pub trait Trainer<G: Game> {
    fn select_action(&self, game: &G, actions: &[G::Action], rng: &mut StdRng) -> Result<usize>;
    fn select_choice(&self, game: &G, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize>;
}
```

不同实现根据当前阶段在内部做不同策略选择，但对 trait 本身透明——Trainer 只需要返回一个索引。

---

## 5. 概念总结

| 概念 | 角色 | 说明 |
|---|---|---|
| `BaseAction` | 基础行动枚举 | `Train | Race | Sleep | FriendOuting | NormalOuting | Clinic`，现有类型 |
| `RamenGameAction` | 统一动作枚举 | `Nothing | Eat(RamenType) | Act(BaseAction)`，覆盖所有阶段 |
| `Trainer` | 决策器 | 从当前阶段的候选列表中选择一项，返回索引 |
| 阶段（stage） | 决策时机 | 每阶段有独立的 `list_actions()`，`apply()` 内部处理随机过渡 |


## 6. 与现有代码的差异

| 方面 | 温泉杯（现有） | 拉面杯（规划） |
|---|---|---|
| 回合阶段 | `Begin → Distribute → Bathing → Train → AfterTrain` | `Begin → Distribute → **RamenSelection** → **MainAction** → PostAction` |
| 动作枚举 | `OnsenAction`（含 Train/Race/Sleep/PR/UseTicket/Dig/Upgrade/Choice） | `RamenGameAction`（Nothing/Eat/Act） |
| 内层动作 | 直接硬编码进 `OnsenAction` | `Act(BaseAction)` 复用现有类型 |
| 跨阶段模式 | 全部在同一个枚举里 | 全部在同一个枚举里（模式相同） |
| 随机性处理 | 训练失败率、休息随机结果、事件随机结果 | 额外增加分身位置随机 |
| 搜索 | 对 Train 阶段的所有动作独立 rollout | 对 RamenSelection 的每种面 rollout，内部自然处理 MainAction |
| Trainer | `select_action` 返回索引 | 签名完全不变 |