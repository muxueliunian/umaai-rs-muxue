//! 客户端拉面决策器的装配：默认搜索、仅接管地区的网络、整局直接走网络
//!
//! 主程序（`main.rs` → `scenario::ramen`）与实验入口
//! （`bin/ramen_client_game_bench.rs`）**共用本模块这一条地区接管逻辑**，
//! 不各写一份。
//!
//! # 两层策略的优先级
//!
//! `ramen_trainer_policy`（整局）在上，`ramen_region_policy`（地区）在下：
//!
//! | `ramen_trainer_policy` | `ramen_region_policy` | 实际执行 |
//! |---|---|---|
//! | `mcts`（默认） | `handwritten`（默认） | 全部走搜索 / 手写，与本模块引入前逐字一致 |
//! | `mcts` | `nn` | 只有外层三次 `RegionSelect` 换成网络 |
//! | `mcts` | `nn_compare` / `mcts_compare` | 地区两条推荐都算，执行侧由取值决定 |
//! | `nn` | **必须是** `handwritten` | 整局所有动作决策直接走网络，一次搜索都不跑 |
//! | `nn` | `nn` / `*_compare` | **启动报错**（两层都声称接管地区） |
//!
//! # 隔离边界（`RamenRegionPolicy::Nn`）
//!
//! 唯一的策略差异是**外层实际对局**的三次 `RegionSelect`（turn 2 / 23 / 47）。
//! 除此之外的一切——`Train` / `RamenSelect` / `SpecialSelect` /
//! `SuperRamenSelect`、事件选项、自选比赛守门——全部**原样转发**同一个
//! [`RamenMctsTrainer`] 实例，因此合并缓存与 `last_decision` 状态与不开网络时逐字一致。
//!
//! **搜索内部模拟的地区选择仍是手写**：rollout 基策由 [`RamenMctsTrainer`]
//! 自己持有，本模块不接触，模型也不会进 rollout evaluator。地区阶段候选数恒 > 1
//! （第 1/2 年 C(5,3)=10，第 3 年 `all` 下 C(10,3)=120），因此正常整局
//! **推理请求恰好 3 次**——这就是「搜索内部没有推理」的直接证据。
//!
//! # 不静默回退
//!
//! 未开 `onnx` feature 却配置 `nn`、模型或旁车缺失、维度不符、与地区搜索开关
//! 或 `fixed` 候选冲突——以上全部在**启动时报错退出**，绝不降级成手写后继续跑。

use std::sync::Arc;

use anyhow::{Result, bail};
use rand::prelude::StdRng;
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData, GameConfig, RamenRegionPolicy, RamenRegionStrategy, RamenTrainerPolicy},
    output::DecisionInfo,
    trainer::{RamenMctsTrainer, RamenSearchStages}
};

use crate::decision::ReasonGate;

#[cfg(feature = "onnx")]
mod nn;
#[cfg(feature = "onnx")]
pub use nn::{CompareMode, RegionDecisionObserver, RegionNnTrainer};

#[cfg(feature = "onnx")]
use crate::ramen_nn::WholeGameNnTrainer;

/// 客户端实际对局使用的拉面决策器
///
/// 三个变体都实现 [`Trainer<RamenGame>`]，`scenario::ramen` 与 benchmark
/// 对它泛型调用，因此**同一条接管逻辑**服务两边。
pub enum RamenClientTrainer {
    /// 默认装配：地区随既有门控走搜索或手写 fallback
    Handwritten(RamenMctsTrainer),
    /// 仅外层三次 `RegionSelect` 交给网络，其余原样转发内部搜索训练员
    #[cfg(feature = "onnx")]
    RegionNn(Box<RegionNnTrainer>),
    /// **整局所有动作决策**直接走网络，完全不跑搜索（`ramen_trainer_policy = "nn"`）
    ///
    /// ❗本地实验用。本变体内部**没有** [`RamenMctsTrainer`]，`[mcts]` 的搜索参数
    /// **不参与动作决策**。但它们仍会被主程序解析（`ramen_search_stages` 写错照样报错，
    /// 搜索训练员也照常构造后丢弃）——「不参与决策」≠「没被读取」。
    #[cfg(feature = "onnx")]
    WholeNn(Box<WholeGameNnTrainer>)
}

/// 装配客户端拉面决策器所需的运行期部件
///
/// 不用裸元组：这三项的含义彼此无关，位置写反了编译器也不会报，出错时只能在屏幕上
/// 看到「JSON 模式却打印了人类文案」这种间接症状。
pub struct ClientTrainerParts {
    /// 已按既有路径构造好的搜索训练员（阶段门控 / reason sink / verbose 都已设好）
    ///
    /// 整局网络模式下**不使用**（那条路一次搜索都不跑）。
    pub mcts: RamenMctsTrainer,
    /// 是否为 human 模式
    ///
    /// ❗`--json` 下必须为假：那时 stdout 严格只有 JSON，对照行改挂
    /// `scenario_extra.region_compare`。
    pub human_mode: bool,
    /// 内部搜索训练员实际挂着的那个理由门
    ///
    /// 对照模式用它隔离参照侧搜索写出的理由。`None` 表示调用方没接理由门
    /// （benchmark 路径），对照模式照样能跑，只是不做理由隔离。
    pub reason_gate: Option<Arc<ReasonGate>>
}

impl RamenClientTrainer {
    /// 人类可读的策略标签（启动时打印；**标签就是实际执行分支**）
    ///
    /// 对照模式报的是 `nn_compare` / `mcts_compare`，与配置里的取值一致——不能
    /// 笼统印成 `nn`，否则「打印一套、执行另一套」。整局网络报 `whole_game_nn`，
    /// 与只接管地区的 `nn` 明确区分。
    pub fn label(&self) -> &'static str {
        match self {
            Self::Handwritten(_) => "handwritten",
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.mode_label(),
            #[cfg(feature = "onnx")]
            Self::WholeNn(_) => "whole_game_nn"
        }
    }

    /// 本装配是否**一次搜索都不跑**
    ///
    /// 启动时据此提示用户：残留的 `[mcts]` 搜索参数不参与本模式的动作决策，不必删。
    /// ❗说的是「不参与决策」，不是「没被读取」——主程序照旧解析它们、照旧构造搜索训练员。
    pub fn skips_search(&self) -> bool {
        match self {
            Self::Handwritten(_) => false,
            #[cfg(feature = "onnx")]
            Self::RegionNn(_) => false,
            #[cfg(feature = "onnx")]
            Self::WholeNn(_) => true
        }
    }
}

impl Trainer<RamenGame> for RamenClientTrainer {
    /// 按变体分派；`Handwritten` 与本枚举引入前逐字一致
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_action(game, actions, rng),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.select_action(game, actions, rng),
            #[cfg(feature = "onnx")]
            Self::WholeNn(t) => t.select_action(game, actions, rng)
        }
    }

    /// 事件选项（旧接口）——两个变体最终都落到同一个 [`RamenMctsTrainer`]
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_choice(game, choices, rng),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.select_choice(game, choices, rng),
            #[cfg(feature = "onnx")]
            Self::WholeNn(t) => t.select_choice(game, choices, rng)
        }
    }

    /// 事件选项（新接口）——同 [`Self::select_choice`]
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_event_choice(
        &self, game: &RamenGame, event: &EventData, choices: &[Vec<EventChoice>], rng: &mut StdRng
    ) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_event_choice(game, event, choices, rng),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.select_event_choice(game, event, choices, rng),
            #[cfg(feature = "onnx")]
            Self::WholeNn(t) => t.select_event_choice(game, event, choices, rng)
        }
    }

    /// 上一次决策的协议摘要
    ///
    /// 网络做出的地区决策返回**该次决策自己的**摘要（无搜索评分、带来源标签），
    /// 不是上一次搜索的旧摘要，见 [`RegionNnTrainer::last_decision`]。
    fn last_decision(&self) -> Option<DecisionInfo> {
        match self {
            Self::Handwritten(t) => t.last_decision(),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.last_decision(),
            #[cfg(feature = "onnx")]
            Self::WholeNn(t) => t.last_decision()
        }
    }

    /// 上一次决策的评分分解；网络做出的地区决策返回 `None`（它没有搜索分解可给）
    fn last_breakdown(&self) -> Option<String> {
        match self {
            Self::Handwritten(t) => t.last_breakdown(),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.last_breakdown(),
            #[cfg(feature = "onnx")]
            Self::WholeNn(t) => t.last_breakdown()
        }
    }
}

/// 校验「网络接管地区」这件事本身与相邻配置项的相容性（**不看模型来自哪里**）
///
/// 从 [`validate_region_policy`] 里拆出来，让**模型来源不同**的入口共用同一套适用性
/// 判断：客户端的模型来自 `ramen_region_model_path`，`bin/ramen_client_game_bench`
/// 的来自 `--model` 命令行参数。拆分之前 benchmark 绕开了这些检查，于是同一份
/// `game_config.toml` 客户端拒绝启动、benchmark 却照跑，还会按「每局恒 3 次推理」
/// 的前提去解读结果。
///
/// 下列两项必须同时成立，否则会出现「打印一套、执行另一套」：
///
/// 1. `ramen_search_stages` **不含** `region`：地区已被网络接管，搜索永远轮不到它，
///    留着开关只会让日志里的「地区参与搜索」与实际不符；
/// 2. `ramen_region_strategy` **不是** `fixed`：`fixed` 下第 3 年只剩 1 个候选
///    （`list_actions` 只给 1 个），网络会被单候选短路，实际只决策 2 次而非 3 次
///    ——「整局恰好 3 次推理」这条隔离证据随之失效。
///
/// ❗调用方必须传**命令行覆盖之后**的生效配置与生效阶段集，不能传文件里的原值。
///
/// # 错误
///
/// 上述任一条不成立时返回带修复建议的错误。
pub fn check_region_nn_applicable(cfg: &GameConfig, stages: RamenSearchStages) -> Result<()> {
    // 对照模式**要**的就是让既有装配也算一遍，`region` 留在搜索阶段集里是合法配置
    // （那一侧因此是真搜索而不是手写基策），只在**完全接管**时才是冲突。
    if stages.region_select && !cfg.ramen_region_policy.shows_compare() {
        bail!(
            "配置冲突：ramen_region_policy=\"nn\" 已接管地区决策，但**生效**的搜索阶段集 \
             {stages:?} 仍含 region（配置文件 [mcts] ramen_search_stages = {file:?}）。\
             地区不会再经过搜索，这个开关不会生效——请从 ramen_search_stages 里去掉 region，\
             或改用对照模式 \"nn_compare\" / \"mcts_compare\"（那两种**要**让搜索也算一遍）",
            stages = stages,
            file = cfg.mcts.ramen_search_stages
        );
    }
    if matches!(cfg.ramen_region_strategy, RamenRegionStrategy::Fixed) {
        bail!(
            "配置冲突：ramen_region_policy={policy:?} 与 ramen_region_strategy=\"fixed\" 不相容。\
             fixed 下第 3 年只枚举 1 个候选，网络会被单候选短路，实际只决策 2 次地区而非 3 次。\
             请把 ramen_region_strategy 改回 \"all\"，或把 ramen_region_policy 改回 \"handwritten\"",
            policy = cfg.ramen_region_policy
        );
    }
    Ok(())
}

/// 校验客户端的地区策略配置（适用性 + **模型路径**）
///
/// 只做**诊断**，不改配置、不降级。`Handwritten` 恒通过（模型路径即使写了也只是
/// 未使用，由调用方打印提示）。`Nn` 先过 [`check_region_nn_applicable`]，再要求
/// `ramen_region_model_path` 已设置且非空——这一条只对客户端成立，benchmark 的
/// 模型走 `--model`，故不在共用的适用性检查里。
///
/// # 错误
///
/// 适用性检查不过，或 `Nn` 下缺模型路径时返回带修复建议的错误。
pub fn validate_region_policy(cfg: &GameConfig, stages: RamenSearchStages) -> Result<()> {
    if !cfg.ramen_region_policy.needs_model() {
        return Ok(());
    }
    check_region_nn_applicable(cfg, stages)?;
    match cfg.ramen_region_model_path.as_deref() {
        Some(p) if !p.trim().is_empty() => Ok(()),
        _ => bail!(
            "配置缺失：ramen_region_policy={policy:?} 需要 ramen_region_model_path（含同名 .json 旁车）。\
             ❗不要填 neuralnet_model_path 的温泉模型，两者结构不同",
            policy = cfg.ramen_region_policy
        )
    }
}

/// 校验**整局网络模式**（`ramen_trainer_policy = "nn"`）与相邻配置项的相容性
///
/// 本模式接管面是整局所有动作决策（含地区），因此优先级最高：地区那一层的取值只能是
/// 中性的 `handwritten`，写 `nn` / `*_compare` 就是两层都声称接管地区，必须报冲突而
/// 不是悄悄让某一层赢。
///
/// 三条：
///
/// 1. `ramen_region_policy` 必须是 `handwritten`（中性默认值）——地区由整局网络接管；
/// 2. `ramen_region_strategy` 不是 `fixed`：`fixed` 下第 3 年只枚举 1 个候选，
///    整局网络在那一步会被单候选短路，等于没接管，实验口径就不成立；
/// 3. `ramen_region_model_path` 已设置且非空——本模式复用它作为**整局动作模型**，
///    不再为本地实验新增一套路径字段。
///
/// ❗`[mcts]` 下的搜索参数（含 `ramen_search_stages`）**不参与本模式的动作决策**，
/// 但本函数**不因此报错**：默认配置本来就带着整段 `[mcts]`，要求用户删掉才肯启动是
/// 没道理的。启动时由调用方打一条提示。
///
/// ❗准确说法是「不参与决策」而不是「不读取」：主程序仍会解析 `ramen_search_stages`
/// （写错照样在启动时报错），也仍会按既有路径构造搜索训练员，只是构造完就丢弃。
///
/// # 错误
///
/// 上述任一条不成立时返回带修复建议的错误。
pub fn validate_whole_nn_policy(cfg: &GameConfig) -> Result<()> {
    if cfg.ramen_region_policy != RamenRegionPolicy::Handwritten {
        bail!(
            "配置冲突：ramen_trainer_policy=\"nn\"（整局网络）已经接管**包括地区在内**的全部动作决策，\
             但 ramen_region_policy={policy:?} 也声称要接管地区。两层不能同时接管——\
             请把 ramen_region_policy 改回中性默认值 \"handwritten\"；\
             只想让网络管地区、其余仍走搜索的话，请把 ramen_trainer_policy 改回 \"mcts\"",
            policy = cfg.ramen_region_policy
        );
    }
    if matches!(cfg.ramen_region_strategy, RamenRegionStrategy::Fixed) {
        bail!(
            "配置冲突：ramen_trainer_policy=\"nn\" 与 ramen_region_strategy=\"fixed\" 不相容。\
             fixed 下第 3 年只枚举 1 个候选，网络在那一步会被单候选短路，地区实际没被接管。\
             请把 ramen_region_strategy 改成 \"all\""
        );
    }
    match cfg.ramen_region_model_path.as_deref() {
        Some(p) if !p.trim().is_empty() => Ok(()),
        _ => bail!(
            "配置缺失：ramen_trainer_policy=\"nn\" 需要 ramen_region_model_path（含同名 .json 旁车）。\
             本模式下它是**整局动作模型**，不只是地区模型。\
             ❗不要填 neuralnet_model_path 的温泉模型，两者结构不同"
        )
    }
}

/// 按配置装配客户端拉面决策器
///
/// `parts.mcts` 由调用方按既有路径构造好（阶段门控、reason sink、verbose 都已设好）。
/// 本函数先按 `ramen_trainer_policy` 决定「整局归谁」，再在默认的 `mcts` 那条里按
/// `ramen_region_policy` 决定「地区这一步由谁做」。模型**每进程加载一次**：本函数只被
/// 调用一次，加载出的训练员内部用 `Arc` 共享模型。
///
/// # 错误
///
/// 配置冲突（见 [`validate_region_policy`] / [`validate_whole_nn_policy`]）、未开
/// `onnx` feature 却配置网络、模型 / 旁车缺失或维度不符时报错——**不会**静默回退成手写。
pub fn build_client_trainer(
    cfg: &GameConfig, stages: RamenSearchStages, parts: ClientTrainerParts
) -> Result<RamenClientTrainer> {
    if cfg.ramen_trainer_policy == RamenTrainerPolicy::Nn {
        validate_whole_nn_policy(cfg)?;
        return build_whole_nn(cfg);
    }
    validate_region_policy(cfg, stages)?;
    match cfg.ramen_region_policy {
        RamenRegionPolicy::Handwritten => Ok(RamenClientTrainer::Handwritten(parts.mcts)),
        _ => build_region_nn(cfg, parts)
    }
}

/// 整局网络分支的实际装配（开了 `onnx` feature）
///
/// 与 [`build_region_nn`] 用同一套 `race_shield` / `special_mode` 取值，保证「地区那一层
/// 换成整局那一层」时这两个开关不漂。
///
/// # 错误
///
/// 模型或旁车缺失、维度与冻结契约不符、ONNX 图无法转为可运行图时报错。
#[cfg(feature = "onnx")]
fn build_whole_nn(cfg: &GameConfig) -> Result<RamenClientTrainer> {
    use std::path::Path;

    use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

    let path = cfg
        .ramen_region_model_path
        .as_deref()
        .unwrap_or_default();
    let nn = RamenNnTrainer::load(Path::new(path))?
        .with_race_shield(true)
        .with_special_mode(SpecialSelectMode::Canonical);
    Ok(RamenClientTrainer::WholeNn(Box::new(WholeGameNnTrainer::new(nn))))
}

/// 未开 `onnx` feature 时的整局网络分支：直接报错
///
/// # 错误
///
/// 恒报错——这个构建里根本没有推理后端，静默回退成搜索会让用户以为整局网络已生效。
#[cfg(not(feature = "onnx"))]
fn build_whole_nn(_cfg: &GameConfig) -> Result<RamenClientTrainer> {
    bail!(
        "ramen_trainer_policy=\"nn\" 需要启用 onnx feature 的构建：\
         cargo build --release --features onnx -p umaai。\
         当前二进制不含推理后端，**不会**回退成搜索继续跑"
    )
}

/// `nn` 分支的实际装配（开了 `onnx` feature）
///
/// # 错误
///
/// 模型或旁车缺失、维度与冻结契约不符、ONNX 图无法转为可运行图时报错。
#[cfg(feature = "onnx")]
fn build_region_nn(cfg: &GameConfig, parts: ClientTrainerParts) -> Result<RamenClientTrainer> {
    use std::path::Path;

    use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

    let path = cfg
        .ramen_region_model_path
        .as_deref()
        .unwrap_or_default();
    // race_shield / special_mode 对地区决策不产生任何影响（守门只在 Train 生效，
    // special_mode 只在 SpecialSelect 生效），这里取与实验臂相同的取值，
    // 保证「万一将来扩大接管面」时行为不漂。
    let nn = RamenNnTrainer::load(Path::new(path))?
        .with_race_shield(true)
        .with_special_mode(SpecialSelectMode::Canonical);
    let mut trainer = RegionNnTrainer::new(parts.mcts, nn);
    if cfg.ramen_region_policy.shows_compare() {
        // `print` 只在 human 模式为真：`--json` 下 stdout 必须严格只有 JSON，
        // 对照结果改由 `scenario_extra.region_compare` 随决策带出。
        trainer = trainer.with_compare(CompareMode {
            nn_primary: cfg.ramen_region_policy.nn_is_primary(),
            print: parts.human_mode
        });
        // 理由门：`nn_compare` 下参照侧那一跑的理由不得写进共用槽位（见
        // `RegionNnTrainer::with_reason_gate`）。没接门时对照照跑，只是不做隔离。
        if let Some(gate) = parts.reason_gate {
            trainer = trainer.with_reason_gate(gate);
        }
    }
    Ok(RamenClientTrainer::RegionNn(Box::new(trainer)))
}

/// 未开 `onnx` feature 时的 `nn` 分支：直接报错
///
/// # 错误
///
/// 恒报错——这个构建里根本没有推理后端，静默回退手写会让用户以为网络已生效。
#[cfg(not(feature = "onnx"))]
fn build_region_nn(_cfg: &GameConfig, _parts: ClientTrainerParts) -> Result<RamenClientTrainer> {
    bail!(
        "ramen_region_policy=\"nn\" 需要启用 onnx feature 的构建：\
         cargo build --release --features onnx -p umaai。\
         当前二进制不含推理后端，**不会**回退成手写继续跑"
    )
}

#[cfg(test)]
mod tests {
    use umasim::{gamedata::GameConfig, trainer::RamenSearchStages};

    use crate::utils::Checks;

    use super::*;

    /// 构造一份最小配置：默认手写地区
    fn base_cfg() -> GameConfig {
        GameConfig::default_for_init()
    }

    /// 默认（handwritten）恒通过校验，且不要求任何模型路径
    #[test]
    fn test_handwritten_policy_never_requires_model() -> Result<()> {
        let mut c = Checks::new();
        let cfg = base_cfg();
        c.check(
            cfg.ramen_region_policy == RamenRegionPolicy::Handwritten,
            "缺省地区策略是 handwritten"
        );
        c.check(
            validate_region_policy(&cfg, RamenSearchStages::all()).is_ok(),
            "handwritten + 地区搜索全开：不报冲突（这是既有合法配置）"
        );
        c.check(
            validate_region_policy(&cfg, RamenSearchStages::none()).is_ok(),
            "handwritten + 不搜任何阶段：不报冲突"
        );
        c.finish()
    }

    /// `nn` 的三类配置冲突各自被单独诊断出来
    #[test]
    fn test_nn_policy_conflicts_are_diagnosed() -> Result<()> {
        let mut c = Checks::new();

        // 1) 缺模型路径
        let mut cfg = base_cfg();
        cfg.ramen_region_policy = RamenRegionPolicy::Nn;
        let e = validate_region_policy(&cfg, RamenSearchStages::none()).unwrap_err().to_string();
        println!("缺模型路径 → {e}");
        c.check(e.contains("ramen_region_model_path"), "缺模型路径时错误指名该字段");

        // 2) 地区搜索开关仍开着
        let mut cfg2 = base_cfg();
        cfg2.ramen_region_policy = RamenRegionPolicy::Nn;
        cfg2.ramen_region_model_path = Some("some/model.onnx".to_string());
        let e2 = validate_region_policy(&cfg2, RamenSearchStages::all()).unwrap_err().to_string();
        println!("region 搜索未关 → {e2}");
        c.check(e2.contains("ramen_search_stages"), "地区搜索冲突时错误指名 ramen_search_stages");

        // 3) fixed 候选
        let mut cfg3 = cfg2.clone();
        cfg3.ramen_region_strategy = RamenRegionStrategy::Fixed;
        cfg3.ramen_region_fixed = Some(vec![[11, 14, 15]]);
        let e3 = validate_region_policy(&cfg3, RamenSearchStages::none()).unwrap_err().to_string();
        println!("fixed 候选 → {e3}");
        c.check(e3.contains("fixed"), "fixed 冲突时错误指名该取值");

        // 4) 三项都满足 → 通过（此处不加载模型，只测校验函数）
        c.check(
            validate_region_policy(&cfg2, RamenSearchStages::none()).is_ok(),
            "nn + 无地区搜索 + all + 有模型路径：校验通过"
        );
        c.finish()
    }

    /// 空白模型路径等同缺失（防止用户写成 `ramen_region_model_path = ""`）
    #[test]
    fn test_blank_model_path_rejected() -> Result<()> {
        let mut c = Checks::new();
        let mut cfg = base_cfg();
        cfg.ramen_region_policy = RamenRegionPolicy::Nn;
        cfg.ramen_region_model_path = Some("   ".to_string());
        let e = validate_region_policy(&cfg, RamenSearchStages::none());
        println!("空白路径 → {:?}", e.as_ref().err().map(ToString::to_string));
        c.check(e.is_err(), "纯空白模型路径被判为缺失");
        c.finish()
    }

    /// 把一段 `game_config.toml` 文本合进仓库的 `default_config.toml`
    ///
    /// ❗**不读用户的 `game_config.toml`**：那是用户自己的文件，按文档启用 `nn`
    /// 是完全合法的操作，测试不该因此变红，更不该反过来要求用户关掉网络。
    /// 这里只用受版本控制的 `gamedata/default_config.toml` 当底座，覆盖层由各用例
    /// 自带 fixture 文本给出。合并走的是与 `load_game_config` 完全相同的那条
    /// `OverrideGameConfig::merge` 路径。
    ///
    /// # 错误
    ///
    /// 定位工作区、读默认配置、解析任一侧 TOML 失败时报错（非法取值即由此返回）。
    fn merge_fixture(override_toml: &str) -> Result<GameConfig> {
        use umasim::{gamedata::OverrideGameConfig, utils::get_workspace_root};

        let def_path = get_workspace_root()?.join("gamedata").join("default_config.toml");
        let default_config: GameConfig = toml::from_str(&fs_err::read_to_string(&def_path)?)?;
        // ❗顶层字段必须排在任何 `[表]` 之前，否则会被前一个表吃掉（`game_config.toml`
        // 同样的坑）。`[config_override]` 是 `OverrideGameConfig` 的必填表，补一个空表即可。
        let text = format!("{override_toml}{NL}[config_override]{NL}", NL = '\n');
        let ov: OverrideGameConfig = toml::from_str(&text)?;
        Ok(ov.merge(&default_config))
    }

    /// 仓库**默认配置**（不含用户文件）必须是手写、且不要求任何模型
    ///
    /// 守住「默认不变、网络是 opt-in」这条线：本功能不得让默认构建因为多了两个
    /// 配置项就要求模型。
    #[test]
    fn test_default_config_file_is_handwritten() -> Result<()> {
        let mut c = Checks::new();
        let cfg = merge_fixture("")?;
        println!(
            "default_config.toml：ramen_region_policy={:?} ramen_region_model_path={:?} ramen_search_stages={:?}",
            cfg.ramen_region_policy, cfg.ramen_region_model_path, cfg.mcts.ramen_search_stages
        );
        c.check(
            cfg.ramen_region_policy == RamenRegionPolicy::Handwritten,
            "仓库默认地区决策来源是 handwritten"
        );
        c.check(cfg.ramen_region_model_path.is_none(), "仓库默认不指定地区模型路径");
        let stages = RamenSearchStages::parse(&cfg.mcts.ramen_search_stages)?;
        c.check(
            validate_region_policy(&cfg, stages).is_ok(),
            "默认配置通过地区策略校验（handwritten 不要求模型）"
        );
        c.check(
            cfg.ramen_region_policy == GameConfig::default_for_init().ramen_region_policy,
            "代码内默认值与配置文件默认值一致"
        );
        c.finish()
    }

    /// 覆盖层 fixture：显式 `nn`、单独给模型路径、非法取值三种情况
    #[test]
    fn test_override_fixtures_control_region_policy() -> Result<()> {
        let mut c = Checks::new();

        // 1) 显式切 nn + 模型路径
        let nn = merge_fixture(&format!(
            "ramen_region_policy = {Q}nn{Q}{NL}ramen_region_model_path = {Q}saved_models/arms/x.onnx{Q}{NL}",
            Q = '"',
            NL = '\n'
        ))?;
        println!("显式 nn → {:?} / {:?}", nn.ramen_region_policy, nn.ramen_region_model_path);
        c.check(nn.ramen_region_policy == RamenRegionPolicy::Nn, "覆盖层能把来源切成 nn");
        c.check(
            nn.ramen_region_model_path.as_deref() == Some("saved_models/arms/x.onnx"),
            "覆盖层能设定地区模型路径"
        );
        c.check(
            validate_region_policy(&nn, RamenSearchStages::none()).is_ok(),
            "nn + 无地区搜索 + 有模型路径：校验通过"
        );

        // 2) 只给模型路径、不切来源：仍是手写，且不因此要求加载模型
        let only_path =
            merge_fixture(&format!("ramen_region_model_path = {Q}saved_models/arms/x.onnx{Q}", Q = '"'))?;
        println!(
            "只给路径 → {:?} / {:?}",
            only_path.ramen_region_policy, only_path.ramen_region_model_path
        );
        c.check(
            only_path.ramen_region_policy == RamenRegionPolicy::Handwritten,
            "只写模型路径不会把来源切成 nn"
        );
        c.check(
            validate_region_policy(&only_path, RamenSearchStages::all()).is_ok(),
            "handwritten + 写了模型路径：不报错（路径只是未使用）"
        );

        // 3) 非法取值必须在解析阶段就失败，不得静默落回默认
        let bad = merge_fixture(&format!("ramen_region_policy = {Q}neural{Q}", Q = '"'));
        println!("非法取值 → {:?}", bad.as_ref().err().map(ToString::to_string));
        c.check(bad.is_err(), "未知的 ramen_region_policy 取值被拒绝（不静默落回 handwritten）");
        c.finish()
    }

    /// 对照模式的配置语义：**允许** region 留在搜索阶段集里，且仍要求模型
    ///
    /// 这是对照模式与完全接管（`nn`）唯一的适用性差别：对照那一侧要的正是「让既有
    /// 装配也算一遍」，`region` 开着时那一侧才是真搜索。其余（缺模型、`fixed` 候选）
    /// 与 `nn` 同样严格——两种 `*_compare` 都要加载模型。
    #[test]
    fn test_compare_modes_allow_region_search_but_still_need_model() -> Result<()> {
        let mut c = Checks::new();
        for policy in [RamenRegionPolicy::NnCompare, RamenRegionPolicy::MctsCompare] {
            let mut cfg = base_cfg();
            cfg.ramen_region_policy = policy;
            cfg.ramen_region_model_path = Some("saved_models/arms/x.onnx".to_string());

            let with_region = validate_region_policy(&cfg, RamenSearchStages::all());
            println!("{policy:?} + region 搜索 → {:?}", with_region.as_ref().err().map(ToString::to_string));
            c.check(with_region.is_ok(), &format!("{policy:?}：region 留在搜索阶段集里是合法配置"));

            let no_region = validate_region_policy(&cfg, RamenSearchStages::none());
            c.check(no_region.is_ok(), &format!("{policy:?}：不开 region 搜索同样合法（对照侧落手写基策）"));

            let mut no_model = cfg.clone();
            no_model.ramen_region_model_path = None;
            let e = no_model_err(&no_model);
            println!("{policy:?} 缺模型 → {e}");
            c.check(e.contains("ramen_region_model_path"), &format!("{policy:?}：缺模型仍然报错"));

            let mut fixed = cfg.clone();
            fixed.ramen_region_strategy = RamenRegionStrategy::Fixed;
            fixed.ramen_region_fixed = Some(vec![[11, 14, 15]]);
            let e2 = validate_region_policy(&fixed, RamenSearchStages::none())
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            println!("{policy:?} + fixed → {e2}");
            c.check(e2.contains("fixed"), &format!("{policy:?}：fixed 候选仍然被拒"));

            c.check(policy.needs_model(), &format!("{policy:?}.needs_model() 为真"));
            c.check(policy.shows_compare(), &format!("{policy:?}.shows_compare() 为真"));
        }

        // 完全接管仍然拒绝 region 搜索（本次改动不得放松这一条）
        let mut nn = base_cfg();
        nn.ramen_region_policy = RamenRegionPolicy::Nn;
        nn.ramen_region_model_path = Some("saved_models/arms/x.onnx".to_string());
        let e = validate_region_policy(&nn, RamenSearchStages::all())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        println!("nn + region 搜索 → {e}");
        c.check(e.contains("ramen_search_stages"), "nn（完全接管）仍然拒绝 region 搜索");

        // 执行推荐取哪条：nn / nn_compare 取网络，mcts_compare 取既有装配
        c.check(RamenRegionPolicy::Nn.nn_is_primary(), "nn 执行网络那条");
        c.check(RamenRegionPolicy::NnCompare.nn_is_primary(), "nn_compare 执行网络那条");
        c.check(!RamenRegionPolicy::MctsCompare.nn_is_primary(), "mcts_compare 执行既有装配那条");
        c.check(!RamenRegionPolicy::Handwritten.needs_model(), "handwritten 不需要模型");
        c.check(!RamenRegionPolicy::Handwritten.shows_compare(), "handwritten 不做对照");
        c.finish()
    }

    /// 取校验错误文本的小工具（测试内用）
    fn no_model_err(cfg: &GameConfig) -> String {
        validate_region_policy(cfg, RamenSearchStages::none())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    /// 整局网络模式的优先级与三类冲突
    ///
    /// 1. `ramen_region_policy` 只能是中性的 `handwritten`——`nn` / 两种 `*_compare`
    ///    都是「两层同时声称接管地区」，必须报错而不是悄悄让某一层赢；
    /// 2. `fixed` 候选被拒（第 3 年只剩 1 个候选，网络在那一步被单候选短路）；
    /// 3. 缺模型路径报错，且错误里说清它在本模式下是**整局动作模型**；
    /// 4. `ramen_search_stages` 含 `region` **不**报错——本模式一个搜索参数都不读，
    ///    要求用户删掉默认配置里的整段 `[mcts]` 是没道理的。
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_whole_nn_policy_conflicts_are_diagnosed() -> Result<()> {
        let mut c = Checks::new();
        let base = || {
            let mut cfg = base_cfg();
            cfg.ramen_trainer_policy = RamenTrainerPolicy::Nn;
            cfg.ramen_region_model_path = Some("saved_models/arms/x.onnx".to_string());
            cfg
        };

        // 1) 地区层必须中性
        for policy in [RamenRegionPolicy::Nn, RamenRegionPolicy::NnCompare, RamenRegionPolicy::MctsCompare] {
            let mut cfg = base();
            cfg.ramen_region_policy = policy;
            let e = validate_whole_nn_policy(&cfg)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            println!("整局 nn + 地区 {policy:?} → {e}");
            c.check(
                e.contains("ramen_region_policy"),
                &format!("整局 nn 与地区 {policy:?} 冲突时错误指名 ramen_region_policy")
            );
            c.check(e.contains("handwritten"), &format!("{policy:?}：错误里给出了修复取值"));
        }

        // 2) fixed 候选
        let mut fixed = base();
        fixed.ramen_region_strategy = RamenRegionStrategy::Fixed;
        fixed.ramen_region_fixed = Some(vec![[11, 14, 15]]);
        let e2 = validate_whole_nn_policy(&fixed)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        println!("整局 nn + fixed → {e2}");
        c.check(e2.contains("fixed"), "整局 nn 拒绝 fixed 地区候选");

        // 3) 缺模型路径
        let mut no_model = base();
        no_model.ramen_region_model_path = None;
        let e3 = validate_whole_nn_policy(&no_model)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        println!("整局 nn 缺模型 → {e3}");
        c.check(e3.contains("ramen_region_model_path"), "缺模型时错误指名复用的那个字段");
        c.check(e3.contains("整局"), "错误里说清它在本模式下是整局动作模型");
        let mut blank = base();
        blank.ramen_region_model_path = Some("  ".to_string());
        c.check(validate_whole_nn_policy(&blank).is_err(), "纯空白模型路径同样被判为缺失");

        // 4) 合法配置：默认地区 + all + 有模型，搜索阶段集怎么写都不影响
        let ok = base();
        println!("整局 nn 合法配置 → {:?}", validate_whole_nn_policy(&ok).is_ok());
        c.check(validate_whole_nn_policy(&ok).is_ok(), "整局 nn + 地区 handwritten + all + 有模型：通过");
        c.check(
            validate_whole_nn_policy(&ok).is_ok() && RamenSearchStages::all().region_select,
            "残留的 region 搜索开关不影响整局 nn 的校验（本模式不读搜索参数）"
        );

        // 默认（mcts）一侧不受影响
        let mut plain = base_cfg();
        plain.ramen_region_policy = RamenRegionPolicy::Nn;
        plain.ramen_region_model_path = Some("saved_models/arms/x.onnx".to_string());
        c.check(
            plain.ramen_trainer_policy == RamenTrainerPolicy::Mcts,
            "缺省整局策略是 mcts（默认不变）"
        );
        c.check(
            validate_region_policy(&plain, RamenSearchStages::none()).is_ok(),
            "整局策略为 mcts 时，地区 nn 仍按原有规则校验"
        );
        c.finish()
    }

    /// `ramen_trainer_policy` 能从 TOML 覆盖层解析进来，且默认是 `mcts`
    ///
    /// # 错误
    ///
    /// 任一观测未通过时返回错误。
    #[test]
    fn test_trainer_policy_parses_from_override() -> Result<()> {
        let mut c = Checks::new();
        let def = merge_fixture("")?;
        println!("默认 → {:?}", def.ramen_trainer_policy);
        c.check(def.ramen_trainer_policy == RamenTrainerPolicy::Mcts, "配置文件默认是 mcts");
        for (text, want) in [("mcts", RamenTrainerPolicy::Mcts), ("nn", RamenTrainerPolicy::Nn)] {
            let cfg = merge_fixture(&format!("ramen_trainer_policy = {Q}{text}{Q}", Q = '"'))?;
            println!("{text} → {:?}", cfg.ramen_trainer_policy);
            c.check(cfg.ramen_trainer_policy == want, &format!("{text} 解析为 {want:?}"));
        }
        let bad = merge_fixture(&format!("ramen_trainer_policy = {Q}neural{Q}", Q = '"'));
        println!("非法取值 → {:?}", bad.as_ref().err().map(ToString::to_string));
        c.check(bad.is_err(), "未知取值被拒绝（不静默落回 mcts）");
        c.finish()
    }

    /// 两种 `*_compare` 取值能从 TOML 覆盖层解析进来
    #[test]
    fn test_compare_policy_parses_from_override() -> Result<()> {
        let mut c = Checks::new();
        for (text, want) in [
            ("nn_compare", RamenRegionPolicy::NnCompare),
            ("mcts_compare", RamenRegionPolicy::MctsCompare)
        ] {
            let cfg = merge_fixture(&format!("ramen_region_policy = {Q}{text}{Q}", Q = '"'))?;
            println!("{text} → {:?}", cfg.ramen_region_policy);
            c.check(cfg.ramen_region_policy == want, &format!("{text} 解析为 {want:?}"));
        }
        c.finish()
    }

    /// 三年 `RegionSelect` 的候选必须全部解码成合法地区组合
    ///
    /// 候选数：第 1 年 C(5,3)=10、第 2 年 10、第 3 年 `all` 下 C(10,3)=120。
    /// 三个地区必须互不相同且落在该年的地区区间内——这就是「候选落格」的定义，
    /// 接管器在决策后会对选中项做同样的解码校验。
    ///
    /// ❗走**纯函数** `region_select_combos` 并把策略显式传成 `All`，不走
    /// `game.list_actions()`：后者从全局 `GAMECONFIG` 读 `ramen_region_strategy`，
    /// 而 `init_global*` 是幂等的（先到先得），用户把配置设成 `fixed` 时第 3 年只会
    /// 给 1 个候选，测试就会随用户配置变红；同一测试进程里还会和别的用例抢全局配置。
    /// 纯函数路径既不读用户配置，也不引入进程级全局竞争。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_candidates_decode_all_three_years() -> Result<()> {
        use std::env;

        use umasim::{
            game::ramen::{RamenAction, region_select_combos, rules::validate_region_selection},
            gamedata::init_global,
            utils::get_workspace_root
        };

        // `get_region_combinations` 读的是 RAMENDATA（**静态数据**，不是可调配置），
        // 所以这里只需要全局数据加载完成，与谁先初始化过 GAMECONFIG 无关。
        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        for (year_idx, want_n) in [(0usize, 10usize), (1, 10), (2, 120)] {
            let combos = region_select_combos(year_idx, RamenRegionStrategy::All, None)?;
            let actions: Vec<RamenAction> = combos
                .iter()
                .map(|&r| RamenAction::no_ramen(umasim::game::ramen::Operation::RegionSelect(r)))
                .collect();
            let mut bad = 0usize;
            for (i, a) in actions.iter().enumerate() {
                // 走接管器**自己那条**解码路径，而不是再写一遍 match
                match RegionNnTrainer::region_of(&actions, i) {
                    Ok(r) if validate_region_selection(year_idx, &r) => {}
                    Ok(r) => {
                        bad += 1;
                        println!("第 {} 年候选 {i} 非法组合 {r:?}", year_idx + 1);
                    }
                    Err(e) => {
                        bad += 1;
                        println!("第 {} 年候选 {i} 解码失败（{a:?}）: {e}", year_idx + 1);
                    }
                }
            }
            println!("第 {} 年候选数 {} 非法 {bad}", year_idx + 1, actions.len());
            c.check(actions.len() == want_n, &format!("第 {} 年候选数 = {want_n}", year_idx + 1));
            c.check(bad == 0, &format!("第 {} 年全部候选合法且可解码", year_idx + 1));
        }
        c.finish()
    }

    /// 模型加载的四条错误路径：文件缺失 / 旁车缺失 / 旁车维度不符 / 文件不是合法 ONNX
    ///
    /// 全部必须在 `build_client_trainer` 就报错，且不得回退成手写。
    ///
    /// ❗**不依赖未入库的正式权重**：`RamenNnTrainer::load` 的检查顺序是
    /// 「模型文件存在 → 旁车存在 → 旁车维度 → 编译 ONNX 图」，前三条在碰到 ONNX
    /// 解析之前就会命中，所以占位文件足够；第四条正是要一个**不是合法 ONNX** 的文件。
    /// 因此本测试在任何机器上都有完整覆盖，没有「缺模型 → 零覆盖」的洞。
    ///
    /// 临时文件建在工作区 `target/test-tmp/` 下的**每次唯一**目录里（见
    /// `crate::utils::unique_test_dir`），并发跑多个测试进程也不会互相覆盖；
    /// 清理前先核对绝对路径确实在该基准目录之下。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_model_error_paths() -> Result<()> {
        use std::{env, path::Path};

        use umasim::search::SearchConfig;

        use crate::utils::{cleanup_test_dir, unique_test_dir};

        let mut c = Checks::new();
        env::set_current_dir(umasim::utils::get_workspace_root()?)?;
        let mut cfg = GameConfig::default_for_init();
        cfg.ramen_region_policy = RamenRegionPolicy::Nn;
        let stages = RamenSearchStages::none();
        let mk = || RamenMctsTrainer::new(SearchConfig::new_game_config(&GameConfig::default_for_init()));
        let err_of = |cfg: &GameConfig| {
            let parts = ClientTrainerParts {
                mcts: mk(),
                human_mode: false,
                reason_gate: None
            };
            build_client_trainer(cfg, stages, parts)
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default()
        };

        // 1) 模型文件根本不存在
        cfg.ramen_region_model_path = Some("saved_models/arms/__definitely_missing__.onnx".to_string());
        let e1 = err_of(&cfg);
        println!("模型缺失 → {e1}");
        c.check(e1.contains("不存在"), "模型缺失时报错而非回退手写");

        let dir = unique_test_dir("region_model_error_paths")?;
        println!("fixture 目录: {}", dir.display());
        // 占位内容：前三条路径在碰到 ONNX 解析之前就会命中，内容是什么都无所谓
        let placeholder = b"not-a-real-onnx";

        // 2) 有模型文件、无旁车 json
        let no_meta = dir.join("no_meta.onnx");
        fs_err::write(&no_meta, placeholder)?;
        c.check(!Path::new(&dir.join("no_meta.onnx.json")).exists(), "确认旁车确实不存在");
        cfg.ramen_region_model_path = Some(no_meta.to_string_lossy().into_owned());
        let e2 = err_of(&cfg);
        println!("旁车缺失 → {e2}");
        c.check(e2.contains("元数据"), "旁车缺失时错误指明模型元数据");

        // 3) 旁车维度与冻结契约不符
        let bad_dim = dir.join("bad_dim.onnx");
        fs_err::write(&bad_dim, placeholder)?;
        fs_err::write(
            dir.join("bad_dim.onnx.json"),
            r#"{"input_dim":7,"output_dim":245,"value_normalization":{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}"#
        )?;
        cfg.ramen_region_model_path = Some(bad_dim.to_string_lossy().into_owned());
        let e3 = err_of(&cfg);
        println!("维度不符 → {e3}");
        c.check(e3.contains("input_dim"), "维度不符时错误指明 input_dim");

        // 4) 旁车声明正确，但模型文件不是合法 ONNX
        let not_onnx = dir.join("not_onnx.onnx");
        fs_err::write(&not_onnx, placeholder)?;
        fs_err::write(
            dir.join("not_onnx.onnx.json"),
            r#"{"input_dim":754,"output_dim":245,"value_normalization":{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}"#
        )?;
        cfg.ramen_region_model_path = Some(not_onnx.to_string_lossy().into_owned());
        let e4 = err_of(&cfg);
        println!("非法 ONNX → {e4}");
        c.check(!e4.is_empty(), "文件不是合法 ONNX 时启动即报错，不回退手写");

        cleanup_test_dir(&dir)?;
        c.check(!dir.exists(), "本次 fixture 目录已清理（清理前核对过在 target/test-tmp 之下）");
        c.finish()
    }

    /// 未开 `onnx` feature 的构建里，`nn` 装配必须报错而不是回退手写
    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_nn_without_onnx_feature_errors() -> Result<()> {
        use std::env;

        use umasim::{search::SearchConfig, utils::get_workspace_root};

        // 构造 RamenMctsTrainer 会读 gamedata/，工作目录必须是 workspace 根
        env::set_current_dir(get_workspace_root()?)?;
        let mut c = Checks::new();
        let mut cfg = base_cfg();
        cfg.ramen_region_policy = RamenRegionPolicy::Nn;
        cfg.ramen_region_model_path = Some("saved_models/arms/ens_G2mix_g123.onnx".to_string());
        let mcts = RamenMctsTrainer::new(SearchConfig::new_game_config(&cfg));
        let parts = ClientTrainerParts {
            mcts,
            human_mode: false,
            reason_gate: None
        };
        let e = build_client_trainer(&cfg, RamenSearchStages::none(), parts);
        let msg = e.err().map(|e| e.to_string()).unwrap_or_default();
        println!("未开 onnx → {msg}");
        c.check(msg.contains("onnx"), "错误信息指出需要 onnx feature");
        c.check(!msg.is_empty(), "没有静默回退成手写");
        c.finish()
    }
}
