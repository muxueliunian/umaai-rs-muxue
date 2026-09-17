//! 仅接管外层 `RegionSelect` 的网络决策器（`onnx` feature）
//!
//! 本文件是主程序与 benchmark **共用的唯一一份**地区接管实现；实验入口额外需要的
//! 「同局面下手写本来会选什么」等观测，通过 [`RegionDecisionObserver`] 钩子挂进来，
//! 不复制一份会逐渐分叉的决策逻辑。

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering}
};

use anyhow::{Result, anyhow, bail};
use colored::Colorize;
use rand::prelude::StdRng;
use serde_json::{Map, Value, json};
use umasim::{
    game::{
        Trainer,
        ramen::{Operation, RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, EventData},
    output::{
        DecisionInfo,
        decision::{SOURCE_REGION_HANDWRITTEN, SOURCE_REGION_NN, SOURCE_REGION_SEARCH}
    },
    trainer::{RamenMctsTrainer, RamenNnTrainer, ramen_handwritten_trainer::ramen_effective_stage}
};

use crate::decision::ReasonGate;

/// 地区决策的旁观钩子（只观测，不参与决策）
///
/// benchmark 用它记录「实际选了哪三个地区」与「同局面下手写本来会选什么」。
/// 主程序不挂钩子。
///
/// 钩子在**网络已经选定、动作尚未执行**时调用。`rng_before` 是**推理之前**
/// 外层随机流的快照副本：实现拿它跑手写参照，既与「不接管时手写会选什么」同一个
/// 随机状态，又不推进外层 `rng`，挂不挂钩子局面都不分叉。
pub trait RegionDecisionObserver: Send + Sync {
    /// 观测一次地区决策
    ///
    /// `picked` 是网络选中的候选下标；`rng_before` 是网络推理前的随机流快照，
    /// 实现可按值可变使用（它已经是副本）。
    ///
    /// # 错误
    ///
    /// 观测本身失败（如候选落格异常）时报错，会让整局中止——观测不该掩盖异常。
    fn on_region_decided(
        &self, game: &RamenGame, actions: &[RamenAction], picked: usize, rng_before: StdRng
    ) -> Result<()>;
}

/// 对照显示的装配（`ramen_region_policy` 的两种 `*_compare` 取值）
///
/// 对照模式下同一个 `RegionSelect` 局面会被**算两次**：既有装配一次、网络一次，
/// 屏幕上两条推荐都打印，玩家自己比。两个取值只差在「哪条算执行推荐」。
#[derive(Debug, Clone, Copy)]
pub struct CompareMode {
    /// 执行推荐是否取网络那一条（`nn_compare` 为真，`mcts_compare` 为假）
    pub nn_primary: bool,
    /// 是否把对照行直接打到 stdout
    ///
    /// ❗human 模式为真；`--json` 模式**必须**为假，否则会污染严格 JSON 流。
    pub print: bool
}

/// 对照那一侧（既有装配）在同一局面下的选择
#[derive(Debug, Clone)]
struct BaselinePick {
    /// 选中的候选下标（**在传入的 `actions` 全表里**的位置）
    idx: usize,
    /// 这一侧留下的协议摘要（真搜索时含完整候选评分；手写基策时为 `None`）
    ///
    /// ❗真搜索那条摘要的 `action_index` 指向的是**截断后的候选表**
    /// （见 `RamenMctsTrainer::last_decision` 的 `reason_max_display` 截断），
    /// 与 [`Self::idx`] 不在同一个下标空间，不能互相当作对方用。
    info: Option<DecisionInfo>
}

impl BaselinePick {
    /// 这条推荐是否来自**真搜索**（`ramen_search_stages` 含 `region`）
    ///
    /// 判据是「摘要里有没有候选评分」，不靠猜配置：为假表示落在手写基策上，
    /// 屏幕文案据此照实说，不含糊成「MCTS」。
    fn searched(&self) -> bool {
        self.info
            .as_ref()
            .is_some_and(|i| !i.candidate_scores.is_empty())
    }

    /// 屏幕上给这一侧用的名字
    fn label(&self) -> &'static str {
        match self.searched() {
            true => "MCTS搜索",
            false => "手写策略"
        }
    }

    /// 协议里给这一侧用的种类名
    fn kind(&self) -> &'static str {
        match self.searched() {
            true => "search",
            false => "handwritten"
        }
    }
}

/// 一次由本壳作出的地区决策留下的东西
struct RegionSlot {
    /// 本次决策自己的协议摘要
    info: DecisionInfo,
    /// [`Trainer::last_breakdown`] 是否透传内部 [`RamenMctsTrainer`] 的评分分解
    ///
    /// 只有「执行侧就是那次真搜索」时为真——此时分解说的正是这次执行的候选。
    /// 其余情况为假：网络与手写基策都没有搜索分解，透传上一次搜索的旧分解就是挂错理由。
    keep_breakdown: bool
}

/// `RamenMctsTrainer` + 仅外层 `RegionSelect` 交给网络的决策器
///
/// 除 `RegionSelect` 外的一切调用**原样转发**给同一个 [`RamenMctsTrainer`] 实例，
/// 因此 `SpecialSelect` 合并缓存、`last_decision` / `last_breakdown` 状态与不开
/// 网络时逐字一致。搜索内部的模拟决策器由 [`RamenMctsTrainer`] 自己持有
/// （手写 rollout 基策），本壳**不接触**，故搜索内部的地区选择仍是手写。
pub struct RegionNnTrainer {
    /// 真正做除地区外全部决策的搜索训练员
    mcts: RamenMctsTrainer,
    /// 只在外层 `RegionSelect` 使用的网络训练员
    nn: RamenNnTrainer,
    /// 可选旁观钩子（benchmark 用；主程序为 `None`）
    observer: Option<Arc<dyn RegionDecisionObserver>>,
    /// 可选对照显示（`None` = 只跑网络，与对照模式引入前逐字一致）
    compare: Option<CompareMode>,
    /// 可选的理由门（对照模式用；`None` = 不做理由隔离）
    ///
    /// 只在**参照侧不是执行侧**（`nn_compare`）时静音，`mcts_compare` 下参照侧就是
    /// 执行侧，它的理由该留着。见 [`ReasonGate`]。
    reason_gate: Option<Arc<ReasonGate>>,
    /// 最近一次决策是否为**本壳做出的地区决策**
    ///
    /// 地区决策（网络那条，或对照模式下的手写基策那条）不经内部 [`RamenMctsTrainer`]
    /// 的摘要通道，它的 `last_search_summary` 因此停在上一次搜索上。置位后
    /// `last_decision` 改读 [`Self::region_slot`]（本次地区决策自己的摘要），
    /// 避免把上一次 MCTS 的旧摘要当成本次地区决策的理由。
    ///
    /// 屏蔽**只覆盖地区决策本身这一步**：此后任何一次转发（动作、事件选项、
    /// 新接口事件选项）都会清位，之后的行为与不开网络时逐字一致。
    /// 尤其是地区之后紧跟事件时，事件自己的说明不会被连带屏蔽。
    /// 用 `AtomicBool` 是因为 `Trainer` 的方法都取 `&self`。
    region_last: AtomicBool,
    /// 最近一次**本壳做出的地区决策**留下的东西（`region_last` 为真时有效）
    ///
    /// 常态内容是「候选完整、评分为空、带来源标签」的一条 [`DecisionInfo`]：与手写
    /// fallback 合成的那条形状一致（下游 luck 路由按 `candidate_scores` 是否为空判断，
    /// 行为不变），但多带一个来源标签，渲染端因此不会把它印成「手写逻辑」。
    ///
    /// ❗**不伪造任何评分**：这条合成摘要的 `score` 恒 0、`candidate_scores` /
    /// `candidate_n` 恒空，不把 policy logits 当终局分。唯一的例外是
    /// `mcts_compare` 且执行侧**真搜过**——那时直接沿用搜索自己的摘要（评分是真的），
    /// 只额外挂上来源标签与对照结果。
    region_slot: Mutex<Option<RegionSlot>>
}

impl RegionNnTrainer {
    /// 装配一个不挂观测钩子的接管器（主程序用）
    pub fn new(mcts: RamenMctsTrainer, nn: RamenNnTrainer) -> Self {
        Self {
            mcts,
            nn,
            observer: None,
            compare: None,
            reason_gate: None,
            region_last: AtomicBool::new(false),
            region_slot: Mutex::new(None)
        }
    }

    /// 挂上旁观钩子（benchmark 用）
    pub fn with_observer(mut self, observer: Arc<dyn RegionDecisionObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 挂上理由门，用于隔离参照侧搜索写出的决策理由
    ///
    /// 必须传**内部 [`RamenMctsTrainer`] 实际挂着的那一个**门，否则静音不到它。
    /// 不挂门时对照模式照样能跑，只是 `nn_compare` + 真地区搜索下屏幕上的理由段
    /// 会是参照搜索的（那不是本次执行推荐的理由）。
    pub fn with_reason_gate(mut self, gate: Arc<ReasonGate>) -> Self {
        self.reason_gate = Some(gate);
        self
    }

    /// 打开对照显示（两种 `*_compare` 模式）
    ///
    /// 打开后每次外层 `RegionSelect` 会**多跑一次**既有装配：`ramen_search_stages`
    /// 含 `region` 时那是一次真搜索（第 3 年 `all` 下 120 个候选，很贵），否则只是
    /// 手写基策（成本可忽略）。
    ///
    /// ❗成本口径随 `[mcts] use_ucb` 变，**不能**一律按「候选数 × `search_n`」估：
    ///
    /// - `use_ucb = false`（均匀分配）：每个候选恰好 `search_n` 次 rollout，总数就是
    ///   候选数 × `search_n`；
    /// - `use_ucb = true`（**仓库默认**）：先给每个候选跑满一组，之后按 UCB 追加，直到
    ///   **被搜得最多的那个候选**的计划次数达到 `search_n` 才停。总次数随局面变化，
    ///   下界是候选数 × 首组大小，不是候选数 × `search_n`。
    ///   ❗首组大小是 `min(search_group_size, search_n).max(1)`，不是 `search_group_size`
    ///   本身——`FlatSearch::search_ucb` 会把它收进 `search_n`，否则 `group_size > search_n`
    ///   时每个候选跑完首组就已超预算，自适应轮次为零。
    pub fn with_compare(mut self, mode: CompareMode) -> Self {
        self.compare = Some(mode);
        self
    }

    /// 人类可读的模式标签（启动横幅用；**标签就是实际执行分支**）
    pub fn mode_label(&self) -> &'static str {
        match self.compare {
            None => "nn",
            Some(CompareMode { nn_primary: true, .. }) => "nn_compare",
            Some(CompareMode { nn_primary: false, .. }) => "mcts_compare"
        }
    }

    /// 屏幕上的对照行：两条推荐 + 是否一致 + 本次执行哪条
    fn compare_line(
        actions: &[RamenAction], nn_idx: usize, base: &BaselinePick, nn_primary: bool
    ) -> String {
        let nn_text = actions
            .get(nn_idx)
            .map(ToString::to_string)
            .unwrap_or_default();
        let base_text = actions
            .get(base.idx)
            .map(ToString::to_string)
            .unwrap_or_default();
        let verdict = match base.idx == nn_idx {
            true => "两者一致".to_string(),
            false => "❗两者不一致".to_string()
        };
        let exec = match nn_primary {
            true => "神经网络",
            false => base.label()
        };
        format!(
            "地区对照：{} → {base_text} ｜ 神经网络 → {nn_text}（{verdict}；本次执行：{exec}）",
            base.label()
        )
    }

    /// 把对照结果挂到决策的 `scenario_extra.region_compare`
    ///
    /// ❗只记录**两侧各选了什么**，不伪造任何评分：`score` / `candidate_scores` /
    /// `candidate_n` 保持原样，luck 路由口径因此不变。
    ///
    /// ❗`nn_index` / `baseline_index` / `executed_index` 一律是**传入 `actions` 全表**
    /// 的下标（载荷里的 `index_space` 明写了这一点）。它们与 [`DecisionInfo::action_index`]
    /// **不一定同空间**：执行侧是真搜索时，后者指向截断后的候选表。下游要对人显示就用
    /// `*_choice` 文本，要对下标就先看 `index_space`。
    fn attach_compare(
        mut info: DecisionInfo, actions: &[RamenAction], nn_idx: usize, base: &BaselinePick, nn_primary: bool
    ) -> DecisionInfo {
        let executed_index = match nn_primary {
            true => nn_idx,
            false => base.idx
        };
        let payload = json!({
            "index_space": "actions",
            "candidates_total": actions.len(),
            "nn_index": nn_idx,
            "nn_choice": actions.get(nn_idx).map(ToString::to_string),
            "baseline_index": base.idx,
            "baseline_choice": actions.get(base.idx).map(ToString::to_string),
            "baseline_kind": base.kind(),
            "agree": base.idx == nn_idx,
            "executed": match nn_primary { true => "nn", false => "baseline" },
            "executed_index": executed_index,
            "executed_choice": actions.get(executed_index).map(ToString::to_string)
        });
        match info.scenario_extra {
            Some(Value::Object(ref mut map)) => {
                map.insert("region_compare".to_string(), payload);
            }
            _ => {
                let mut map = Map::new();
                map.insert("region_compare".to_string(), payload);
                info.scenario_extra = Some(Value::Object(map));
            }
        }
        info
    }

    /// 记下本次地区决策，并把 `last_decision` 切到它上面
    ///
    /// 先写摘要再置位：置位后 `last_decision` 就会去读它，顺序反了会读到上一条。
    ///
    /// # 错误
    ///
    /// 摘要锁被毒化时报错——宁可让这一局停下，也不要把上一条摘要当成本次的结果发出去。
    fn store_region_slot(&self, slot: RegionSlot) -> Result<()> {
        *self
            .region_slot
            .lock()
            .map_err(|_| anyhow!("地区决策摘要锁被毒化"))? = Some(slot);
        self.region_last.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// 解除地区屏蔽位（转发给内部搜索训练员之前调用）
    fn clear_region_flag(&self) {
        self.region_last.store(false, Ordering::Relaxed);
    }

    /// 为一次网络地区决策合成协议摘要
    ///
    /// 形状与 `scenario::ramen::fallback_decision` 合成的那条对齐（候选描述完整、
    /// 评分为空），额外带上来源标签。`decision_kind` 先填 `region_select`；
    /// 主程序路径下 `calc_ramen_training` 会按 snapshot 的 stage 再填一次同样的值，
    /// benchmark 路径下没人填，所以这里必须自己填对。
    fn region_decision_info(actions: &[RamenAction], picked: usize, source: &str) -> DecisionInfo {
        DecisionInfo {
            action_index: picked,
            score: 0.0,
            decision_kind: "region_select".to_string(),
            candidate_scores: Vec::new(),
            candidate_descriptions: actions.iter().map(ToString::to_string).collect(),
            candidate_n: Vec::new(),
            scenario_extra: None
        }
        .with_source(source)
    }

    /// 取出候选 `i` 的三个地区下标
    ///
    /// # 错误
    ///
    /// 下标越界，或该候选不是 `RegionSelect` 时报错。
    pub fn region_of(actions: &[RamenAction], i: usize) -> Result<[usize; 3]> {
        match actions.get(i).map(|a| a.operation) {
            Some(Operation::RegionSelect(r)) => Ok(r),
            other => bail!("地区阶段候选 {i} 不是 RegionSelect：{other:?}")
        }
    }
}

impl Trainer<RamenGame> for RegionNnTrainer {
    /// 地区阶段走网络，其余阶段原样转发搜索训练员
    ///
    /// # 错误
    ///
    /// 网络推理失败、候选落格失败、观测钩子报错，或转发的搜索训练员报错时原样
    /// 返回——**任何一种都不会静默退回手写**。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        if ramen_effective_stage(game, actions) != RamenStage::RegionSelect {
            // 转发给 MCTS 的决策会自己刷新摘要，屏蔽位随之解除
            self.clear_region_flag();
            return self.mcts.select_action(game, actions, rng);
        }
        // 观测用的随机流快照必须取在**推理之前**：钩子里的手写参照要与「不接管时
        // 手写会看到的随机状态」一致。只在挂了钩子时克隆，主程序路径零开销。
        let rng_before = self.observer.as_ref().map(|_| rng.clone());
        // 执行推荐取哪条：不开对照恒取网络；开了对照按模式取
        let nn_primary = self.compare.map(|m| m.nn_primary).unwrap_or(true);
        // 对照模式：先在随机流的**副本**上跑一次既有装配（`ramen_search_stages` 含
        // `region` 时是真搜索，否则是手写基策）。跑在副本上，外层 `rng` 不被推进，
        // 因此开不开对照，网络这一侧看到的随机状态完全一样。
        let baseline = match self.compare {
            Some(_) => {
                let mut rng_copy = rng.clone();
                // ❗参照侧**不是执行侧**（`nn_compare`）时，把那一跑的两路输出副作用都挡住：
                //
                // 1. **理由数据**（`reason_sink` → 共用槽位 → 屏幕 / `scenario_extra.reason`）
                //    走 [`ReasonGate`]；
                // 2. **verbose 日志**（`[MCTS][回合 N] … 首选 #k`、终局维度差）走
                //    `RamenMctsTrainer::quiet_scope`。
                //
                // 两路都没有「这是参照侧」的标注，留着就会被当成本次执行推荐的理由。
                // 两个守卫都是 RAII，中途 `?` 早退照样恢复；它们只挡住这一跑，链式决策里
                // 其它步骤写好的理由与日志一条不少。
                let mute = match (nn_primary, self.reason_gate.as_ref()) {
                    (true, Some(gate)) => Some(gate.mute()),
                    _ => None
                };
                let hush = match nn_primary {
                    true => Some(self.mcts.quiet_scope()),
                    false => None
                };
                let base_idx = self.mcts.select_action(game, actions, &mut rng_copy)?;
                // 这一跑自己的摘要：有候选评分就说明它真搜过，没有就是落在手写基策上。
                // 判据是数据本身，不是猜配置。
                let base_info = self.mcts.last_decision();
                // 静音到此为止；上面任何一处 `?` 早退时守卫同样会析构，不会把整局静音掉。
                drop(hush);
                drop(mute);
                let _ = Self::region_of(actions, base_idx)?;
                Some(BaselinePick { idx: base_idx, info: base_info })
            }
            None => None
        };
        let nn_idx = self.nn.select_action(game, actions, rng)?;
        // 落格校验：即使没挂观测钩子也做，坏候选要在这里就炸而不是流到下游
        let _ = Self::region_of(actions, nn_idx)?;
        if let (Some(obs), Some(snapshot)) = (&self.observer, rng_before) {
            obs.on_region_decided(game, actions, nn_idx, snapshot)?;
        }

        // 不开对照：网络独跑，合成一条无评分、带来源标签的摘要
        let Some(base) = baseline else {
            let info = Self::region_decision_info(actions, nn_idx, SOURCE_REGION_NN);
            self.store_region_slot(RegionSlot { info, keep_breakdown: false })?;
            return Ok(nn_idx);
        };

        let picked = match nn_primary {
            true => nn_idx,
            false => base.idx
        };
        if self.compare.is_some_and(|m| m.print) {
            println!(
                "{}",
                Self::compare_line(actions, nn_idx, &base, nn_primary).bright_cyan()
            );
        }

        // 执行侧的摘要分两种：
        //
        // 1. 执行侧是**真搜索**（`mcts_compare` + `ramen_search_stages` 含 `region`）：
        //    沿用那次搜索**自己的**摘要——候选评分 / 次数 / 截断后的候选表全是真的，
        //    屏幕上的 luck 行与 `region` 开着的手写装配逐字一致。只额外挂来源标签与对照
        //    结果，一个评分都不改。
        //    ❗这里**不能**像旧实现那样提前 return 把摘要留给内部 MCTS：那样 `last_decision`
        //    走的是 MCTS 自己那条，对照结果与来源标签都挂不上去，JSON 里就丢了 `region_compare`。
        // 2. 其余情况（执行网络、或执行没搜过的手写基策）：合成无评分摘要 + 来源标签。
        let (info, keep_breakdown) = match (nn_primary, base.searched()) {
            (false, true) => {
                let mut searched_info = base
                    .info
                    .clone()
                    .ok_or_else(|| anyhow!("对照侧声称搜过却没有摘要，拒绝伪造"))?;
                // 搜索训练员不感知阶段，`decision_kind` 留空由发起方填；benchmark 路径
                // 没人填，所以这里自己填对（主程序随后会填成同一个值）。
                searched_info.decision_kind = "region_select".to_string();
                (searched_info.with_source(SOURCE_REGION_SEARCH), true)
            }
            _ => {
                let source = match nn_primary {
                    true => SOURCE_REGION_NN,
                    false => SOURCE_REGION_HANDWRITTEN
                };
                (Self::region_decision_info(actions, picked, source), false)
            }
        };
        // 只在**决策成功落定后**置位：中途报错时整局已经终止，不需要也不应改状态。
        let info = Self::attach_compare(info, actions, nn_idx, &base, nn_primary);
        self.store_region_slot(RegionSlot { info, keep_breakdown })?;
        Ok(picked)
    }

    /// 事件选项转发搜索训练员（与不开网络时逐字相同）
    ///
    /// 转发前清掉地区屏蔽位：事件由内部 MCTS 处理，它自己的说明不该被上一次
    /// 地区决策连带屏蔽。
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        self.clear_region_flag();
        self.mcts.select_choice(game, choices, rng)
    }

    /// 事件选项（新接口）转发搜索训练员
    ///
    /// 同 [`Self::select_choice`]：转发前清掉地区屏蔽位。
    ///
    /// # 错误
    ///
    /// 转发的训练员报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        self.clear_region_flag();
        self.mcts.select_event_choice(game, event, choices, rng)
    }

    /// 非地区决策返回内部 MCTS 的摘要；**地区决策后返回本次地区决策自己的摘要**
    ///
    /// 地区决策不经内部 MCTS 的摘要通道，它的 `last_search_summary` 停在上一次搜索上；
    /// 若原样透传，屏幕与协议里就会出现「上一步搜索的理由挂在这次地区决策上」。这里改为
    /// 返回 [`Self::region_slot`] 里那条**本次**决策的摘要，它一定带来源标签，渲染端
    /// 因此既不会挂旧理由，也不会误标手写。
    ///
    /// 摘要意外缺失（锁被毒化等）时退回 `None`——上层会走 `fallback_decision` 合成
    /// 一条无来源标签的决策行，渲染成中性文案，仍然不会误标手写。
    fn last_decision(&self) -> Option<DecisionInfo> {
        match self.region_last.load(Ordering::Relaxed) {
            true => self.region_slot.lock().ok()?.as_ref().map(|s| s.info.clone()),
            false => self.mcts.last_decision()
        }
    }

    /// 同 [`Self::last_decision`]：地区决策后不返回内部 MCTS 的旧 breakdown
    ///
    /// 唯一例外是执行侧就是那次真搜索（`mcts_compare` + 开着 `region` 搜索）：
    /// 此时分解说的正是这次执行的候选，透传是对的。
    fn last_breakdown(&self) -> Option<String> {
        if !self.region_last.load(Ordering::Relaxed) {
            return self.mcts.last_breakdown();
        }
        let keep = self
            .region_slot
            .lock()
            .ok()
            .and_then(|s| s.as_ref().map(|s| s.keep_breakdown))
            .unwrap_or(false);
        match keep {
            true => self.mcts.last_breakdown(),
            false => None
        }
    }
}
