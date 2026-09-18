//! 实验入口的**评分轴与 PT 倍率显式固定**，以及生效值回报
//!
//! # 为什么需要这一层
//!
//! 上游 `70550cd` 之后，拉面终局评分走 [`FlatSearchGame::search_score`](crate::search::FlatSearchGame::search_score)：
//!
//! - `score`   = `calc_score()`，PT 项用 `total_pt()`（**含 Hint 折算**）；
//! - `score_pt` = `skill + (skill_pt × pt_score_rate × pt_favor_rate) + Σfive_status`，
//!   PT 项用 `skill_pt`（**不含 Hint 折算**），且不再乘旧公式的 `×0.37`。
//!
//! 由此得到两条必须写死在代码里、不能靠默契的结论：
//!
//! 1. **`pt_favor_rate = 1.0` 不等价于 Score 轴**——两条公式的 PT 口径本身就不同
//!    （Hint 折算的有无），倍率取 1 只是让 PT 项不被放大，`total_hints > 0` 时两轴
//!    仍然给出不同的数。想要 Score 轴就显式选 [`RamenSelection::Score`]，
//!    不要用「倍率 1 的 PT 轴」冒充。
//! 2. `pt_favor_rate` 的**默认值**在上游从 `8.0` 改成了 `2.0`，且它同时被
//!    `gamedata/default_config.toml` 与用户 `game_config.toml` 覆盖。实验入口若不显式
//!    固定，同一条命令在不同机器 / 不同时间会读到不同倍率，而这一项**会进终局评分**。
//!
//! # 本模块的契约
//!
//! 实验入口在调用 [`init_global_with_config`](crate::gamedata::init_global_with_config)
//! **之前**用 [`ScoringOverride::apply`] 把倍率钉进 [`GameConfig`]，初始化之后用
//! [`report_effective`] 把实际生效值打到日志。缺失配置**不回落用户 toml**：
//!
//! - [`RamenSelection::Pt`] 且未显式给倍率 → **直接报错**，不猜；
//! - [`RamenSelection::Score`] 且未显式给倍率 → 固定为 [`PINNED_PT_FAVOR_RATE`]
//!   （代码常量，不是读 toml），并在回报里标明「由实验入口钉死」。

use anyhow::{Result, bail, ensure};

use crate::{gamedata::GameConfig, trainer::RamenSelection};

/// 实验入口在未显式指定时钉死的 `pt_favor_rate`
///
/// 数值与上游 `gamedata/default_config.toml` 的定档一致（`2.0`），但**来源是本常量**，
/// 不是运行时读到的 toml——把「默认值」变成代码里的一个可审计的定义，
/// 用户改 toml 不会静默改变实验口径。
pub const PINNED_PT_FAVOR_RATE: f32 = 2.0;

/// 一次实验运行的评分轴与 PT 倍率
#[derive(Debug, Clone, Copy)]
pub struct ScoringOverride {
    /// 选动作用哪条轴
    pub selection: RamenSelection,
    /// 显式指定的 `pt_favor_rate`；`None` = 未指定
    pub pt_favor_rate: Option<f32>
}

impl ScoringOverride {
    /// Score 轴 + 未指定倍率（最常见的本地实验口径）
    pub fn score_axis() -> Self {
        Self {
            selection: RamenSelection::Score,
            pt_favor_rate: None
        }
    }

    /// 解析出本次真正要用的倍率
    ///
    /// # 错误
    ///
    /// PT 轴未显式给倍率、或给了非有限 / 非正值时报错。
    pub fn resolve_rate(&self) -> Result<f32> {
        match (self.selection, self.pt_favor_rate) {
            (RamenSelection::Pt, None) => bail!(
                "选动作轴为 PT 时必须显式指定 pt_favor_rate：它直接进终局 score_pt，\
                 不接受回落到 default_config.toml / game_config.toml 的当前值"
            ),
            (_, Some(v)) => {
                ensure!(v.is_finite() && v > 0.0, "pt_favor_rate 必须是正有限值，实际 {v}");
                Ok(v)
            }
            (RamenSelection::Score, None) => Ok(PINNED_PT_FAVOR_RATE)
        }
    }

    /// 在 `init_global_with_config` **之前**把倍率钉进配置，返回生效值
    ///
    /// # 错误
    ///
    /// 见 [`Self::resolve_rate`]。
    pub fn apply(&self, cfg: &mut GameConfig) -> Result<f32> {
        let rate = self.resolve_rate()?;
        cfg.pt_favor_rate = rate;
        Ok(rate)
    }
}

/// 一次实验运行的其余口径（只作回报用，不改配置）
#[derive(Debug, Clone)]
pub struct EffectiveSearchFacts {
    /// 每候选 rollout 预算
    pub search_n: usize,
    /// 是否走 UCB 分配
    pub use_ucb: bool,
    /// 激进度上限（实际 rf 随回合缩放，本项只是上限）
    pub radical_factor_max: f64,
    /// 搜索覆盖的阶段（调用方自己格式化）
    pub stages: String,
    /// rollout 基策（手写 / NN / 混合，调用方自己格式化）
    pub rollout_policy: String,
    /// 地区候选枚举策略
    pub region_strategy: String
}

/// 把**实际生效**的评分与搜索口径打到日志
///
/// `rate_was_explicit` 区分「命令行显式给的」与「由 [`PINNED_PT_FAVOR_RATE`] 钉死的」，
/// 两者都不是「读用户 toml」，但来源不同，报告里要能看出来。
/// ❗用 `println!` 而不是 `info!`：`ramen_teacher_collect` 把日志级别设成 `error`，
/// `info!` 会被整条吞掉，生效口径就等于没打印。这行信息是**验收凭据**，
/// 不能随日志级别消失。
pub fn report_effective(ov: &ScoringOverride, rate: f32, facts: &EffectiveSearchFacts) {
    let src = if ov.pt_favor_rate.is_some() { "命令行显式" } else { "实验入口常量钉死" };
    let axis = match ov.selection {
        RamenSelection::Score => "Score（SearchScore::score = calc_score()，PT 项含 Hint 折算，统计量 mean）",
        RamenSelection::Pt => "PT（SearchScore::score_pt，PT 项不含 Hint 折算，统计量 weighted_mean(rf)）"
    };
    println!("[生效口径] 选动作轴 = {axis}");
    println!("[生效口径] pt_favor_rate = {rate}（来源：{src}；**未**回落 game_config.toml）");
    if matches!(ov.selection, RamenSelection::Score) {
        println!(
            "[生效口径] ❗Score 轴下 pt_favor_rate 不参与选动作，但仍随 GAMECONSTANTS 进 score_pt；\
             pt_favor_rate=1 也 ≠ Score（Hint 折算口径不同），故照样记录"
        );
    }
    println!(
        "[生效口径] search_n={} use_ucb={} radical_factor_max={} 阶段={} rollout基策={} 地区枚举={}",
        facts.search_n, facts.use_ucb, facts.radical_factor_max, facts.stages, facts.rollout_policy, facts.region_strategy
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Checks;

    /// PT 轴缺倍率必须报错；Score 轴缺倍率钉死到常量；显式值原样生效
    #[test]
    fn test_scoring_override_resolution() -> Result<()> {
        let mut c = Checks::new();

        let pt_missing = ScoringOverride {
            selection: RamenSelection::Pt,
            pt_favor_rate: None
        };
        let err = pt_missing.resolve_rate();
        println!("PT 轴缺倍率 -> {:?}", err.as_ref().err().map(ToString::to_string));
        c.check(err.is_err(), "PT 轴未显式指定倍率时拒绝");

        let score_default = ScoringOverride::score_axis().resolve_rate()?;
        println!("Score 轴缺倍率 -> {score_default}");
        c.check(score_default == PINNED_PT_FAVOR_RATE, "Score 轴缺倍率钉死到 PINNED_PT_FAVOR_RATE");

        let explicit = ScoringOverride {
            selection: RamenSelection::Pt,
            pt_favor_rate: Some(3.5)
        }
        .resolve_rate()?;
        println!("PT 轴显式 3.5 -> {explicit}");
        c.check(explicit == 3.5, "显式倍率原样生效");

        let bad = ScoringOverride {
            selection: RamenSelection::Score,
            pt_favor_rate: Some(0.0)
        }
        .resolve_rate();
        println!("倍率 0 -> {:?}", bad.as_ref().err().map(ToString::to_string));
        c.check(bad.is_err(), "非正倍率被拒绝");
        c.finish()
    }
}
