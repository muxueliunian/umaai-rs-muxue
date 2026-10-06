//! 拉面杯 rollout 基策
//!
//! 扩展上游已有的手写 / NN rollout：网络可按绝对回合窗口生效，手写配置独立保留。
//! 更换基策会改变搜索估值的对象；是否提高闭环成绩需要另外验证。
//!
//! # 为什么是新类型而不是 trait object
//!
//! [`FlatSearchGame::RolloutTrainer`](crate::search::FlatSearchGame::RolloutTrainer)
//! 是关联类型，当初这样定是为了不掀开 `MctsTrainer` 与 `umaai` 的调用签名。
//! 改成 `FlatSearch<G, T>` 的第二个类型参数确实会把那些签名全部动一遍——
//! 但**在本转发器内部装一个 `Arc<dyn Trainer<G>>` 同样能保持外部签名不变**，
//! 那条路并没有被排除。这里保存手写与可选网络实例，以支持窗口外回退和配置叠加。
//! 保留 `nn(trainer)` / `Handwritten(trainer)` 构造写法；类型已不是枚举，旧模式匹配不兼容。
//!
//! # 默认行为逐位不变
//!
//! [`RamenRolloutTrainer::handwritten`] 内部持有的就是原先的
//! [`RecommendedRamenTrainer::for_rollout`]，未装载网络时三个 `select_*`
//! 全部原样转发，对 RNG 的消耗与改动前一致。

use anyhow::Result;
use rand::rngs::StdRng;

use crate::{
    game::{Game, Trainer, ramen::RamenGame},
    gamedata::{EventChoice, EventData},
    output::DecisionInfo
};

#[cfg(feature = "onnx")]
use crate::{game::ramen::RamenAction, trainer::DecisionPrep};

use super::RecommendedRamenTrainer;

#[cfg(feature = "onnx")]
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[cfg(feature = "onnx")]
use super::RamenNnTrainer;

/// 网络 rollout 的装载信息
///
/// 独立成类型而非在 [`RamenRolloutTrainer`] 里摊平两个字段：模型与它的生效
/// 回合上限是一组不可分割的配置，分开放很容易出现「换了模型忘了改上限」。
#[cfg(feature = "onnx")]
struct NnRollout {
    /// 已加载的网络策略；整进程共享一份，不每次搜索重载
    trainer: Arc<RamenNnTrainer>,
    /// 网络生效的回合上限（含）；`None` = 整局都用网络
    ///
    /// 这是游戏绝对回合，不是从搜索根开始计数的深度；超出窗口后仍由手写跑到终局。
    max_turn: Option<i32>
}

/// 一个走了网络的决策点的完整现场
///
/// 保留 `game` 与 `actions` 的**原件**，是为了让 GPU 侧的输出能用同一个
/// [`RamenNnTrainer::resolve_decision`](crate::trainer::RamenNnTrainer::resolve_decision)
/// 在同一个局面上打分——不需要另建一套「按记录重建局面、恢复 RNG、重走轨迹」的回放系统，
/// 也就没有重建不一致带来的排查。
///
/// RNG 探针取自**克隆体**（`rng.clone().next_u64()`），不消耗生产随机流——
/// 本文件既有测试已经在用这个手法。特征向量只覆盖被编码进模型的那部分状态，
/// **不能**代替随机流身份，故两者都记。
#[cfg(feature = "onnx")]
pub struct RolloutInferSnapshot {
    /// 决策发生时的局面原件（只读用）
    pub game: RamenGame,
    /// 当时的合法候选**及其顺序**（顺序参与 argmax 并列裁决，必须原样保留）
    pub actions: Vec<RamenAction>,
    /// 实际喂给模型的输入（SpecialSelect 下是 canonical 还原后的那一份）
    pub features: Vec<f32>,
    /// CPU 侧推理得到的 policy
    pub cpu_policy: Vec<f32>,
    /// CPU 侧按候选打分选出的赢家下标
    pub cpu_winner: usize,
    /// 决策发生的回合
    pub turn: i32,
    /// 决策发生的阶段
    pub stage: crate::game::ramen::RamenStage,
    /// 决策 RNG 探针（克隆体的下一个 `u64`，不消耗生产流）
    pub rng_probe: u64,
    /// 进入 `prepare_decision` 的时刻
    pub prep_start: std::time::Instant,
    /// CPU 侧推理返回的时刻
    ///
    /// 与**上一个**决策的 `prep_start` 相减，即「规则推进 + 编码 + 调度残余」，
    /// ❗不是纯规则推进耗时——真正拆开需要在 rollout 循环内埋点。
    pub infer_end: std::time::Instant
}

/// 决策录制的接收端
///
/// 装上后 rollout 的每个网络决策都会回调一次；不装（默认）时整条路径与改动前
/// 逐位一致，不额外克隆局面。
#[cfg(feature = "onnx")]
pub trait RolloutInferSink: Send + Sync {
    /// 一个真正走了网络的决策点
    ///
    /// # 错误
    ///
    /// 接收端自身失败时原样上抛——录制失败不能静默丢样本。
    fn on_inferred(&self, snap: RolloutInferSnapshot) -> Result<()>;

    /// 一个**没有**走网络就定案的决策点（守门命中 / 该阶段转交手写）
    ///
    /// 单独记：把它混进推理样本会虚增「GPU 对拍覆盖率」。
    ///
    /// # 错误
    ///
    /// 接收端自身失败时原样上抛。
    fn on_resolved(&self, turn: i32, stage: crate::game::ramen::RamenStage, winner: usize) -> Result<()>;
}

/// 拉面杯 rollout 用的决策器
///
/// 默认等价于原先写死的推荐手写策略；调用 [`Self::with_neural_net`] 后，
/// 落在生效窗口内的决策改由网络给出，窗口外与事件选项仍走手写。
pub struct RamenRolloutTrainer {
    /// 手写基策：默认路径本身，同时也是网络的窗口外回退与事件选项出口
    handwritten: RecommendedRamenTrainer,
    /// 可选的网络 rollout
    #[cfg(feature = "onnx")]
    nn: Option<NnRollout>,
    /// 可选的决策录制钩子（对拍与 trace 用；生产路径为 `None`）
    #[cfg(feature = "onnx")]
    sink: Option<Arc<dyn RolloutInferSink>>,
    /// 最近一次调用是否走 NN，用于把信息接口转发到实际使用的实例。
    #[cfg(feature = "onnx")]
    last_nn: AtomicBool,
    /// 测试专用：每第 N 次 `select_action` 报一次错
    ///
    /// 用来在**不依赖 ONNX 模型**的前提下构造「一部分 rollout 失败、另一部分
    /// 成功」，从而验证 [`FlatSearch::with_strict_rollout`] 真的把错误抛到了
    /// 搜索之外。按回合数注入做不到这件事：rollout 一律跑到终局，任何回合阈值
    /// 都会让**所有** rollout 失败，测不出「部分失败」这个正是要守的场景。
    #[cfg(test)]
    fail_every: Option<usize>,
    /// 测试专用：`select_action` 调用计数（rollout 并行，故用原子量）
    #[cfg(test)]
    calls: std::sync::atomic::AtomicUsize
}

impl RamenRolloutTrainer {
    /// 构造默认（纯手写）rollout 基策
    ///
    /// 用 [`RecommendedRamenTrainer::for_rollout`]：三份年策略的 breakdown 全关，
    /// 与改动前的 `default_rollout_trainer` 逐字一致。
    pub fn handwritten() -> Self {
        Self::Handwritten(RecommendedRamenTrainer::for_rollout())
    }

    /// 兼容上游的手写构造表达式；不兼容旧枚举的模式匹配。
    #[allow(non_snake_case)]
    pub fn Handwritten(trainer: RecommendedRamenTrainer) -> Self {
        Self {
            handwritten: trainer,
            #[cfg(feature = "onnx")]
            nn: None,
            #[cfg(feature = "onnx")]
            sink: None,
            #[cfg(feature = "onnx")]
            last_nn: AtomicBool::new(false),
            #[cfg(test)]
            fail_every: None,
            #[cfg(test)]
            calls: std::sync::atomic::AtomicUsize::new(0)
        }
    }

    /// 兼容上游的全程 NN 构造入口；模型内部共享 ONNX 图，不会重复加载。
    #[cfg(feature = "onnx")]
    pub fn nn(trainer: RamenNnTrainer) -> Self {
        Self::handwritten().with_neural_net(Arc::new(trainer), None)
    }

    /// 测试专用：构造一个每第 `n` 次决策报一次错的基策
    ///
    /// 生产代码里没有别的办法让 rollout 可控地失败——网络的推理错误依赖具体
    /// 模型与局面，做不成确定性测试。
    #[cfg(test)]
    pub(crate) fn failing_every(n: usize) -> Self {
        Self {
            fail_every: Some(n.max(1)),
            ..Self::handwritten()
        }
    }

    /// 设置手写基策的「友人出行必须走完 5 次」完成硬门限
    ///
    /// 只改手写那一份：它既是默认路径，也是网络窗口外的回退与事件选项出口。
    /// 窗口内 NN 动作不额外施加手写友人完成硬门限。
    pub fn with_friend_complete_required(mut self, required: bool) -> Self {
        self.handwritten = self.handwritten.with_friend_complete_required(required);
        self
    }

    /// 装载网络 rollout
    ///
    /// `max_turn` 为网络生效的回合上限（含），`None` 表示整局都用网络。
    /// 负数等价于网络从不生效（全程手写），入口参数应在调用前自行校验。
    /// 传入 `Arc` 而非路径：一次搜索会在多个 rayon 线程上跑成百上千条 rollout，
    /// 模型必须共享，按需加载会把加载代价乘以 rollout 数。
    #[cfg(feature = "onnx")]
    pub fn with_neural_net(mut self, trainer: Arc<RamenNnTrainer>, max_turn: Option<i32>) -> Self {
        self.nn = Some(NnRollout { trainer, max_turn });
        self
    }

    /// 本次决策是否该交给网络
    ///
    /// 未装载网络时恒为 `None`；装载后按回合窗口判定。
    #[cfg(feature = "onnx")]
    fn nn_for(&self, game: &RamenGame) -> Option<&RamenNnTrainer> {
        let nn = self.nn.as_ref()?;
        match nn.max_turn {
            Some(limit) if game.turn() > limit => None,
            _ => Some(nn.trainer.as_ref())
        }
    }

    /// 记录本次决策走了手写分支（`last_decision` / `last_breakdown` 据此转发）
    #[cfg(feature = "onnx")]
    fn mark_handwritten(&self) {
        // 未装网络时标记恒为 false，不写：默认手写路径上所有 rollout 线程共用这一个原子量
        if self.nn.is_some() {
            self.last_nn.store(false, Ordering::Relaxed);
        }
    }

    /// 取最近实际使用的 NN；窗口外决策与事件选项之后返回 `None`。
    #[cfg(feature = "onnx")]
    fn last_nn_trainer(&self) -> Option<&RamenNnTrainer> {
        if !self.last_nn.load(Ordering::Relaxed) {
            return None;
        }
        self.nn.as_ref().map(|nn| nn.trainer.as_ref())
    }

    /// 装上决策录制钩子
    ///
    /// 只影响**走网络**的决策：手写分支与事件选项不回调。录制会克隆局面与候选，
    /// 因而这条路径带额外开销，**只用于正确性对拍与逻辑长度分析，不用于性能测量**。
    #[cfg(feature = "onnx")]
    pub fn with_infer_sink(mut self, sink: Arc<dyn RolloutInferSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// 本 rollout 基策的简短标签，供日志与 CSV 区分对照组
    pub fn label(&self) -> String {
        #[cfg(feature = "onnx")]
        if let Some(nn) = self.nn.as_ref() {
            return match nn.max_turn {
                Some(limit) => format!("nn<=t{limit}"),
                None => "nn".to_string()
            };
        }
        "handwritten".to_string()
    }
}

impl Default for RamenRolloutTrainer {
    fn default() -> Self {
        Self::handwritten()
    }
}

impl Trainer<RamenGame> for RamenRolloutTrainer {
    /// 窗口内交给网络，其余转发手写
    ///
    /// # 错误
    ///
    /// 被选中的那一方报错时原样返回。**不做静默回退**：网络在 rollout 里悄悄
    /// 退化成手写，量到的就不再是 `Q^NN`，而分数上看不出来。
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
    ) -> Result<usize> {
        #[cfg(test)]
        if let Some(n) = self.fail_every {
            let seq = self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if seq % n == 0 {
                anyhow::bail!("测试注入的 rollout 失败（第 {seq} 次决策，turn={}）", game.turn());
            }
        }
        #[cfg(feature = "onnx")]
        if let Some(nn) = self.nn_for(game) {
            self.last_nn.store(true, Ordering::Relaxed);
            let Some(sink) = self.sink.as_ref() else {
                // 生产路径：不克隆任何东西，与加录制之前逐位一致
                return nn.select_action(game, actions, rng);
            };
            // 录制路径走的是 `select_action` 内部同一套三步接口，不是平行实现
            // 克隆后取样：探测随机流身份而不改变生产轨迹
            let rng_probe = {
                use rand::RngCore;
                rng.clone().next_u64()
            };
            let prep_start = std::time::Instant::now();
            return match nn.prepare_decision(game, actions, rng)? {
                DecisionPrep::Resolved(idx) => {
                    sink.on_resolved(game.turn(), game.stage.clone(), idx)?;
                    Ok(idx)
                }
                DecisionPrep::NeedsInference(features) => {
                    let out = nn.infer_features(features.clone())?;
                    let infer_end = std::time::Instant::now();
                    let cpu_winner = nn.resolve_decision(game, actions, &out.policy)?;
                    sink.on_inferred(RolloutInferSnapshot {
                        game: game.clone(),
                        actions: actions.to_vec(),
                        features,
                        cpu_policy: out.policy,
                        cpu_winner,
                        turn: game.turn(),
                        stage: game.stage.clone(),
                        rng_probe,
                        prep_start,
                        infer_end
                    })?;
                    Ok(cpu_winner)
                }
            };
        }
        #[cfg(feature = "onnx")]
        self.mark_handwritten();
        self.handwritten.select_action(game, actions, rng)
    }

    /// 事件选项一律走手写（网络的 choice 头未训练，它自己也是转发手写）
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        #[cfg(feature = "onnx")]
        self.mark_handwritten();
        self.handwritten.select_choice(game, choices, rng)
    }

    /// 事件选项（含友人事件特例）一律走手写
    ///
    /// # 错误
    ///
    /// 手写策略报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        #[cfg(feature = "onnx")]
        self.mark_handwritten();
        self.handwritten.select_event_choice(game, event, choices, rng)
    }

    /// 转发最近使用的决策器摘要，保留上游 wrapper 的信息接口。
    fn last_decision(&self) -> Option<DecisionInfo> {
        #[cfg(feature = "onnx")]
        if let Some(nn) = self.last_nn_trainer() {
            return nn.last_decision();
        }
        self.handwritten.last_decision()
    }

    /// 转发最近使用的决策器说明，避免窗口切换后显示另一分支的旧说明。
    fn last_breakdown(&self) -> Option<String> {
        #[cfg(feature = "onnx")]
        if let Some(nn) = self.last_nn_trainer() {
            return nn.last_breakdown();
        }
        self.handwritten.last_breakdown()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rand::RngCore;

    use super::*;
    use crate::{
        bench,
        game::ramen::RamenStage,
        gamedata::init_global,
        search::{FlatSearch, SearchConfig},
        trainer::LoggingTrainer,
        utils::{Checks, get_workspace_root, init_test_logger}
    };

    const TEST_UMA_ID: u32 = 102601;
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };

    /// 一次决策的完整记录
    ///
    /// 新类型而非裸元组：五个字段量纲各异（回合 / 阶段 / 出口 / 选中下标 /
    /// rng 探针），排错时靠位置区分极易读反。
    #[derive(Debug, PartialEq, Eq)]
    struct DecisionFingerprint {
        /// 决策发生的回合
        turn: i32,
        /// 决策发生的阶段
        stage: String,
        /// 走的是哪个 `Trainer` 出口（`action` / `choice` / `event_choice`）
        port: &'static str,
        /// 选中的下标
        chosen: usize,
        /// **决策前** rng 状态的探针取值：克隆一份取 `next_u64`，不消耗原 rng
        ///
        /// 这是「rng 消耗逐位一致」的直接证据。只比终局分数不够——不同的
        /// rng 消耗完全可能在同一局里重新汇合成相同终局。
        rng_probe: u64
    }

    /// 记录每次决策的转发包装
    ///
    /// 用 `Mutex` 而非 `RefCell`：`Trainer` 在搜索/并行场景要求 `Sync`。
    struct Recording<T> {
        /// 被包装的真实策略
        inner: T,
        /// 逐次决策记录
        ///
        /// 用 `Arc` 与调用方共享：`bench::run_seeded` 会拿走 trainer 的所有权，
        /// 跑完之后没有别的办法把记录取回来。
        log: Shared
    }

    /// 共享的决策记录句柄
    type Shared = Arc<Mutex<Vec<DecisionFingerprint>>>;

    impl<T> Recording<T> {
        /// 包装一个策略，同时返回可在跑完后读取的记录句柄
        fn new(inner: T) -> (Self, Shared) {
            let log: Shared = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    inner,
                    log: Arc::clone(&log)
                },
                log
            )
        }

        /// 记录一次决策
        #[allow(clippy::unwrap_used)]
        fn record(&self, turn: i32, stage: String, port: &'static str, chosen: usize, rng_probe: u64) {
            self.log.lock().unwrap().push(DecisionFingerprint {
                turn,
                stage,
                port,
                chosen,
                rng_probe
            });
        }
    }

    impl<T: Trainer<RamenGame>> Trainer<RamenGame> for Recording<T> {
        fn select_action(
            &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut StdRng
        ) -> Result<usize> {
            let probe = rng.clone().next_u64();
            let chosen = self.inner.select_action(game, actions, rng)?;
            self.record(game.turn(), format!("{:?}", game.stage), "action", chosen, probe);
            Ok(chosen)
        }

        fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
            let probe = rng.clone().next_u64();
            let chosen = self.inner.select_choice(game, choices, rng)?;
            self.record(game.turn(), format!("{:?}", game.stage), "choice", chosen, probe);
            Ok(chosen)
        }

        fn select_event_choice(
            &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
        ) -> Result<usize> {
            let probe = rng.clone().next_u64();
            let chosen = self.inner.select_event_choice(game, event, choices, rng)?;
            self.record(game.turn(), format!("{:?}", game.stage), "event_choice", chosen, probe);
            Ok(chosen)
        }
    }

    /// 准备一局固定种子的拉面局面
    fn setup(seed: u64) -> Result<(RamenGame, StdRng)> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        let (decision_rng, rule_master) = bench::seeded_rngs(seed, 0);
        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        game.set_rule_master(rule_master);
        Ok((game, decision_rng))
    }

    /// 跑一整局并取出决策轨迹
    #[allow(clippy::unwrap_used)]
    fn trajectory_of<T: Trainer<RamenGame>>(seed: u64, inner: T) -> Result<Vec<DecisionFingerprint>> {
        let (rec, log) = Recording::new(inner);
        bench::run_seeded(
            TEST_UMA_ID,
            &TEST_DECK,
            &TEST_INHERIT,
            seed,
            0,
            &LoggingTrainer::new(rec, seed)
        )?;
        let out = log.lock().unwrap().drain(..).collect();
        Ok(out)
    }

    /// 未装载网络时，转发器与它包住的手写策略**逐次决策**同轨
    ///
    /// 这是本次改动的守门：`RamenGame::RolloutTrainer` 从 `RecommendedRamenTrainer`
    /// 换成了本类型，若转发路径与原实现有任何差异（多消耗一次 rng、事件选项
    /// 走了别的分支），全部拉面搜索基线都会静默作废。
    ///
    /// 比的是**整条决策轨迹**：每次决策的回合、阶段、`Trainer` 出口、选中下标，
    /// 外加决策前 rng 状态的探针取值。只比终局四项指标不够——不同的 rng 消耗完全
    /// 可能在同一局里重新汇合成相同终局，那样的测试会漏掉真正的分叉。
    #[test]
    fn test_handwritten_wrapper_matches_inner() -> Result<()> {
        let _ = setup(0)?; // 只为切工作目录并初始化全局数据

        let mut c = Checks::new();
        for seed in [42u64, 1234u64] {
            let a = trajectory_of(seed, RamenRolloutTrainer::handwritten())?;
            let b = trajectory_of(seed, RecommendedRamenTrainer::for_rollout())?;
            println!("seed={seed} 决策次数 转发器={} 原实现={}", a.len(), b.len());
            c.check(!a.is_empty(), &format!("seed={seed} 记录到决策"));
            c.check(a.len() == b.len(), &format!("seed={seed} 决策次数一致"));
            let first_diff = a.iter().zip(b.iter()).position(|(x, y)| x != y);
            if let Some(i) = first_diff {
                println!("  首个分叉 #{i}: 转发器={:?} 原实现={:?}", a[i], b[i]);
            }
            c.check(
                first_diff.is_none(),
                &format!("seed={seed} 逐次决策（含决策前 rng 探针）完全一致")
            );
        }

        // 终局侧的独立佐证：轨迹一致 ⇒ 终局必然一致，反之不成立，故两者都查
        for seed in [42u64, 1234u64] {
            let wrapped = bench::run_seeded(
                TEST_UMA_ID,
                &TEST_DECK,
                &TEST_INHERIT,
                seed,
                0,
                &LoggingTrainer::new(RamenRolloutTrainer::handwritten(), seed)
            )?;
            let direct = bench::run_seeded(
                TEST_UMA_ID,
                &TEST_DECK,
                &TEST_INHERIT,
                seed,
                0,
                &LoggingTrainer::new(RecommendedRamenTrainer::for_rollout(), seed)
            )?;
            println!(
                "seed={seed} 转发器 score={} five={:?} pt={} / 原实现 score={} five={:?} pt={}",
                wrapped.score,
                wrapped.five_status,
                wrapped.skill_pt,
                direct.score,
                direct.five_status,
                direct.skill_pt
            );
            c.check(wrapped.score == direct.score, &format!("seed={seed} 分数一致"));
            c.check(wrapped.five_status == direct.five_status, &format!("seed={seed} 五维一致"));
            c.check(wrapped.skill_pt == direct.skill_pt, &format!("seed={seed} 技能点一致"));
            c.check(
                wrapped.yearly_eat_count == direct.yearly_eat_count,
                &format!("seed={seed} 逐年吃面次数一致")
            );
        }
        c.finish()
    }

    /// `strict_rollout` 打开时，**部分** rollout 失败即让整次搜索报错
    ///
    /// 这是审查抓到的 P1：`simulate_many` 默认把失败 rollout 计入 `failed` 后
    /// 继续统计，只要每个候选还剩样本，搜索就照常返回。若失败与局面相关
    /// （网络在某类状态上推理报错），剩下的样本就是「以推理成功为条件」的分布，
    /// 各候选的条件还互不相同，排序被系统性污染而分数上看不出来。
    ///
    /// 注入的是「每第 N 次决策失败一次」而非「某回合起全失败」：后者会让所有
    /// rollout 一起失败，测不到「部分失败仍被吞掉」这个正要守的场景。
    #[test]
    fn test_strict_rollout_propagates_partial_failure() -> Result<()> {
        let (mut game, mut rng) = setup(42)?;
        let hw = RecommendedRamenTrainer::new();
        // 推进到第一个 Train 根（多候选，且离终局远、每条 rollout 决策次数多）
        let mut reached = false;
        while game.next() {
            if game.stage == RamenStage::Train {
                reached = true;
                break;
            }
            game.run_stage(&hw, &mut rng)?;
        }
        let mut c = Checks::new();
        c.check(reached, "推进到 Train 根");
        let actions = game.list_actions()?;
        println!("根: turn={} stage={:?} 候选={}", game.turn(), game.stage, actions.len());
        c.check(actions.len() > 1, "根上多于一个候选");

        let config = SearchConfig::default().with_search_n(8).with_ucb(false);

        // 宽松模式（历史行为）：部分失败被吞掉，搜索照常返回
        let lenient = FlatSearch::<RamenGame>::new(config.clone())
            .with_rollout_trainer(RamenRolloutTrainer::failing_every(997));
        let lenient_out = lenient.search(&game, &actions, &mut rng.clone());
        println!("宽松模式 search -> ok={}", lenient_out.is_ok());
        c.check(lenient_out.is_ok(), "宽松模式下部分失败仍返回结果（历史行为）");

        // 严格模式：同样的失败注入必须把错误抛出来
        let strict = FlatSearch::<RamenGame>::new(config)
            .with_rollout_trainer(RamenRolloutTrainer::failing_every(997))
            .with_strict_rollout(true);
        let strict_out = strict.search(&game, &actions, &mut rng.clone());
        match &strict_out {
            Ok(_) => println!("严格模式 search -> 意外成功"),
            Err(e) => println!("严格模式 search -> 报错: {e}")
        }
        c.check(strict_out.is_err(), "严格模式下部分失败让整次搜索报错");
        c.finish()
    }

    /// 默认标签是 `handwritten`，装载网络后才会变
    ///
    /// 标签会进上屏行与实验身份，是事后分辨「这批数据是哪个教师跑的」的线索；
    /// 它若在默认路径下就报成别的值，对照组会被静静地记成实验组。
    #[test]
    fn test_default_label_is_handwritten() -> Result<()> {
        let mut c = Checks::new();
        let label = RamenRolloutTrainer::handwritten().label();
        println!("默认 rollout 基策标签 = {label}");
        c.check(label == "handwritten", "默认标签为 handwritten");
        c.finish()
    }

    /// 窗口测试的控制局面：末期、低体力、隐藏风味满，NN 偏好休息。
    #[cfg(feature = "onnx")]
    struct RoutingFixture {
        /// 固定状态，友人完成硬门限开启时应强制外出。
        game: RamenGame,
        /// 两路共用的初始随机流。
        rng: StdRng,
        /// 休息与友人外出两个候选。
        actions: Vec<RamenAction>,
        /// 当场生成、无需真实权重的 NN。
        nn: Arc<RamenNnTrainer>
    }

    /// 构造能区分手写友人门限与 NN 偏好的 fixture。
    #[cfg(feature = "onnx")]
    fn routing_fixture() -> Result<RoutingFixture> {
        use crate::game::ramen::{Operation, policy_schema::{POLICY_DIM, train_index}};

        let (mut game, rng) = setup(42)?;
        game.base.turn = 70;
        game.stage = RamenStage::Train;
        game.uma.vital = 0;
        game.ramen.special_feeling = 4;
        game.friend.out_used = vec![false; 5];
        let actions = vec![RamenAction::new(Operation::Rest), RamenAction::new(Operation::FriendOuting)];
        let mut policy = vec![0.0f32; POLICY_DIM];
        policy[train_index(&Operation::Rest)?] = 10.0;
        let nn = Arc::new(RamenNnTrainer::test_fixture(&policy)?.with_race_shield(false));
        Ok(RoutingFixture { game, rng, actions, nn })
    }

    /// 绝对闭区间与 None 都实际执行决策，并用每模型计数确认窗口外不推理。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_nn_turn_window_is_inclusive() -> Result<()> {
        let f = routing_fixture()?;
        let mut c = Checks::new();
        for limit in [None, Some(70), Some(71), Some(69)] {
            let trainer = RamenRolloutTrainer::handwritten()
                .with_friend_complete_required(true)
                .with_neural_net(Arc::clone(&f.nn), limit);
            let before = f.nn.test_infer_count();
            let pick = trainer.select_action(&f.game, &f.actions, &mut f.rng.clone())?;
            let calls = f.nn.test_infer_count() - before;
            let expected_nn = limit.is_none_or(|turn| f.game.turn() <= turn);
            println!("turn={} limit={limit:?}: pick={pick} NN请求={calls}", f.game.turn());
            c.check(pick == if expected_nn { 0 } else { 1 }, "窗口内选 NN 休息，窗口外选手写强制外出");
            c.check(calls == u64::from(expected_nn), "None/边界/窗口外的实际推理次数正确");
            c.check(trainer.last_nn_trainer().is_some() == expected_nn, "最近路由信息与实际决策器一致");
        }
        let legacy = RamenRolloutTrainer::nn(f.nn.as_ref().clone());
        c.check(legacy.label() == "nn", "旧 nn 构造入口保留全程网络语义");
        c.finish()
    }

    /// MCTS 上两种配置顺序均保留 NN，窗口外的手写完成门限可切换。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_nn_and_friend_configuration_order() -> Result<()> {
        use crate::trainer::RamenMctsTrainer;

        let f = routing_fixture()?;
        let mut c = Checks::new();
        for required in [false, true] {
            for limit in [Some(69), None] {
                let first_nn = RamenMctsTrainer::new(SearchConfig::default())
                    .with_nn_rollout(Arc::clone(&f.nn), limit)
                    .with_friend_complete_required(required);
                let first_friend = RamenMctsTrainer::new(SearchConfig::default())
                    .with_friend_complete_required(required)
                    .with_nn_rollout(Arc::clone(&f.nn), limit);
                let a = first_nn.search.rollout_trainer();
                let b = first_friend.search.rollout_trainer();
                c.check(a.nn.is_some() && b.nn.is_some(), "两种 setter 顺序都没有卸载 NN");
                let before = f.nn.test_infer_count();
                let pick_a = a.select_action(&f.game, &f.actions, &mut f.rng.clone())?;
                let pick_b = b.select_action(&f.game, &f.actions, &mut f.rng.clone())?;
                let expected = usize::from(required && limit.is_some());
                println!("friend_required={required} limit={limit:?}: NN先={pick_a} friend先={pick_b}");
                c.check(pick_a == expected && pick_b == expected, "手写门限按配置执行，NN 活跃分支仍选休息");
                c.check(f.nn.test_infer_count() - before == if limit.is_none() { 2 } else { 0 }, "两种顺序实际推理次数相同");
            }
        }
        c.finish()
    }

    /// 旧 Handwritten 构造保留传入实例及其说明，切换窗口时不读到旧分支说明。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_window_forwards_last_metadata() -> Result<()> {
        let f = routing_fixture()?;
        let trainer = RamenRolloutTrainer::Handwritten(RecommendedRamenTrainer::new())
            .with_neural_net(Arc::clone(&f.nn), Some(69));
        trainer.select_action(&f.game, &f.actions[..1], &mut f.rng.clone())?;
        let mut c = Checks::new();
        let details = trainer.last_breakdown();
        println!("窗口外手写说明: {details:?}");
        c.check(details.is_some(), "兼容构造没有丢掉原手写实例的说明收集设置");
        c.check(details == trainer.handwritten.last_breakdown(), "说明转发到实际手写实例");
        let mut earlier = f.game.clone();
        earlier.base.turn = 69;
        trainer.select_action(&earlier, &f.actions, &mut f.rng.clone())?;
        c.check(trainer.last_nn_trainer().is_some(), "回到窗口内后信息路由跟随 NN");
        c.check(trainer.last_breakdown() == f.nn.last_breakdown(), "NN 决策后不返回手写分支的旧说明");
        c.check(trainer.last_decision().is_none(), "NN 没有搜索摘要时不伪造摘要");
        c.finish()
    }
}
