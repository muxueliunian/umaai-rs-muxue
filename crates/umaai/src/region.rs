//! 客户端拉面决策器的装配：按 `ramen_region_policy` 决定外层地区选择由谁负责
//!
//! | `ramen_region_policy` | 外层地区选择 | 其余决策 |
//! |---|---|---|
//! | `handwritten`（默认） | 既有逻辑（`ramen_search_stages` 含 `region` 时搜索，否则手写） | 既有逻辑 |
//! | `region_nn_hint` | 既有逻辑执行，另外显示网络推荐作参考 | 既有逻辑 |
//! | `region_nn` | 网络执行，不做地区搜索 | 既有逻辑 |
//!
//! 外层地区选择只有每年一次（共 3 次）。搜索内部模拟的地区选择始终是手写 rollout
//! 策略，网络不进搜索。
//!
//! 未开 `onnx` feature 却选了网络、模型或旁车缺失、维度不符时启动即报错，
//! 不会降级成手写继续跑。

use anyhow::{Result, bail};
use rand::prelude::StdRng;
use umasim::{
    game::{
        Trainer,
        ramen::{RamenAction, RamenGame}
    },
    gamedata::{EventChoice, EventData, GameConfig, RamenRegionPolicy},
    output::DecisionInfo,
    trainer::RamenMctsTrainer
};

#[cfg(feature = "onnx")]
mod nn;
#[cfg(feature = "onnx")]
pub use nn::{RegionNnRole, RegionNnTrainer};

/// 客户端实际对局使用的拉面决策器
pub enum RamenClientTrainer {
    /// 默认装配：与上游搜索训练员完全相同
    Handwritten(RamenMctsTrainer),
    /// 外层地区选择接入网络（执行或参考），其余原样转发内部搜索训练员
    #[cfg(feature = "onnx")]
    RegionNn(Box<RegionNnTrainer>)
}

impl RamenClientTrainer {
    /// 与配置取值一致的策略标签（启动时打印）
    pub fn label(&self) -> &'static str {
        match self {
            Self::Handwritten(_) => "handwritten",
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => match t.role() {
                RegionNnRole::Execute => "region_nn",
                RegionNnRole::Hint => "region_nn_hint"
            }
        }
    }
}

impl Trainer<RamenGame> for RamenClientTrainer {
    /// 按变体分派
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_action(game, actions, rng),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.select_action(game, actions, rng)
        }
    }

    /// 事件选项（旧接口），两个变体最终都落到同一个 [`RamenMctsTrainer`]
    ///
    /// # 错误
    ///
    /// 内部训练员报错时原样返回。
    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        match self {
            Self::Handwritten(t) => t.select_choice(game, choices, rng),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.select_choice(game, choices, rng)
        }
    }

    /// 事件选项（新接口），同 [`Self::select_choice`]
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
            Self::RegionNn(t) => t.select_event_choice(game, event, choices, rng)
        }
    }

    /// 上一次决策的协议摘要
    fn last_decision(&self) -> Option<DecisionInfo> {
        match self {
            Self::Handwritten(t) => t.last_decision(),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.last_decision()
        }
    }

    /// 上一次决策的评分分解
    fn last_breakdown(&self) -> Option<String> {
        match self {
            Self::Handwritten(t) => t.last_breakdown(),
            #[cfg(feature = "onnx")]
            Self::RegionNn(t) => t.last_breakdown()
        }
    }
}

/// 校验地区策略配置：需要网络的取值必须给出非空的模型路径
///
/// # 错误
///
/// `region_nn` / `region_nn_hint` 下缺模型路径（或路径为空白）时报错。
pub fn validate_region_policy(cfg: &GameConfig) -> Result<()> {
    if !cfg.ramen_region_policy.needs_model() {
        return Ok(());
    }
    match cfg.ramen_region_model_path.as_deref() {
        Some(p) if !p.trim().is_empty() => Ok(()),
        _ => bail!(
            "配置缺失：ramen_region_policy={policy:?} 需要 ramen_region_model_path（含同名 .json 旁车）。\
             不要填 neuralnet_model_path 的温泉模型，两者结构不同",
            policy = cfg.ramen_region_policy
        )
    }
}

/// 按配置装配客户端拉面决策器
///
/// `mcts` 由调用方按既有路径构造好（阶段门控、reason sink、verbose 都已设好）。
/// `handwritten` 下原样返回它；其余取值加载一次地区模型并包在它外面。
///
/// # 错误
///
/// 配置校验不过（见 [`validate_region_policy`]）、未开 `onnx` feature 却选了网络、
/// 模型或旁车缺失、维度不符时报错。
pub fn build_client_trainer(cfg: &GameConfig, mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    validate_region_policy(cfg)?;
    match cfg.ramen_region_policy {
        RamenRegionPolicy::Handwritten => Ok(RamenClientTrainer::Handwritten(mcts)),
        _ => build_region_nn(cfg, mcts)
    }
}

/// 地区网络的实际装配（开了 `onnx` feature）
///
/// # 错误
///
/// 模型或旁车缺失、维度与契约不符、ONNX 图无法转为可运行图时报错。
#[cfg(feature = "onnx")]
fn build_region_nn(cfg: &GameConfig, mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    use std::path::Path;

    use umasim::trainer::{RamenNnTrainer, SpecialSelectMode};

    let path = cfg
        .ramen_region_model_path
        .as_deref()
        .unwrap_or_default();
    // 守门与 special_mode 不影响地区决策，取与训练评测时相同的取值
    let nn = RamenNnTrainer::load(Path::new(path))?
        .with_race_shield(true)
        .with_special_mode(SpecialSelectMode::Canonical);
    let role = match cfg.ramen_region_policy {
        RamenRegionPolicy::RegionNnHint => RegionNnRole::Hint,
        _ => RegionNnRole::Execute
    };
    Ok(RamenClientTrainer::RegionNn(Box::new(RegionNnTrainer::new(mcts, nn, role))))
}

/// 未开 `onnx` feature 时选了网络：直接报错
///
/// # 错误
///
/// 恒报错：当前构建没有推理后端。
#[cfg(not(feature = "onnx"))]
fn build_region_nn(cfg: &GameConfig, _mcts: RamenMctsTrainer) -> Result<RamenClientTrainer> {
    bail!(
        "ramen_region_policy={policy:?} 需要启用 onnx feature 的构建：\
         cargo build --release --features onnx -p umaai",
        policy = cfg.ramen_region_policy
    )
}

#[cfg(test)]
mod tests {
    use std::env;

    use umasim::{
        gamedata::{GameConfig, OverrideGameConfig},
        search::SearchConfig,
        utils::get_workspace_root
    };

    use super::*;
    use crate::utils::Checks;

    /// 需要网络的两个取值
    const NN_POLICIES: [RamenRegionPolicy; 2] = [RamenRegionPolicy::RegionNnHint, RamenRegionPolicy::RegionNn];

    /// 取校验错误文本（通过时为空串）
    fn validate_err(cfg: &GameConfig) -> String {
        validate_region_policy(cfg)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    /// 把一段 `game_config.toml` 文本合进仓库的 `default_config.toml`
    ///
    /// 不读用户自己的 `game_config.toml`，合并走与 `load_game_config` 相同的
    /// `OverrideGameConfig::merge`。
    ///
    /// # 错误
    ///
    /// 定位工作区、读默认配置、解析任一侧 TOML 失败时报错（非法取值即由此返回）。
    fn merge_fixture(override_toml: &str) -> Result<GameConfig> {
        let def_path = get_workspace_root()?.join("gamedata").join("default_config.toml");
        let default_config: GameConfig = toml::from_str(&fs_err::read_to_string(&def_path)?)?;
        // 顶层字段必须排在任何 `[表]` 之前；`[config_override]` 是必填表，补一个空表
        let text = format!("{override_toml}\n[config_override]\n");
        let ov: OverrideGameConfig = toml::from_str(&text)?;
        Ok(ov.merge(&default_config))
    }

    /// `handwritten` 不要求模型；另外两个取值缺模型路径或路径为空白时报错
    #[test]
    fn test_validate_region_policy() -> Result<()> {
        let mut c = Checks::new();
        let base = GameConfig::default_for_init();
        c.check(base.ramen_region_policy == RamenRegionPolicy::Handwritten, "代码内默认是 handwritten");
        c.check(validate_err(&base).is_empty(), "handwritten 不要求模型路径");

        for policy in NN_POLICIES {
            let mut cfg = base.clone();
            cfg.ramen_region_policy = policy;
            let e = validate_err(&cfg);
            println!("{policy:?} 缺模型 → {e}");
            c.check(e.contains("ramen_region_model_path"), &format!("{policy:?} 缺模型时错误指名该字段"));

            cfg.ramen_region_model_path = Some("   ".to_string());
            c.check(!validate_err(&cfg).is_empty(), &format!("{policy:?} 空白路径等同缺失"));

            cfg.ramen_region_model_path = Some("saved_models/x.onnx".to_string());
            c.check(validate_err(&cfg).is_empty(), &format!("{policy:?} 有模型路径时通过"));
        }
        c.finish()
    }

    /// 仓库默认配置是 `handwritten` 且不指定模型
    #[test]
    fn test_default_config_file_is_handwritten() -> Result<()> {
        let mut c = Checks::new();
        let cfg = merge_fixture("")?;
        println!(
            "default_config.toml：ramen_region_policy={:?} ramen_region_model_path={:?}",
            cfg.ramen_region_policy, cfg.ramen_region_model_path
        );
        c.check(cfg.ramen_region_policy == RamenRegionPolicy::Handwritten, "默认地区策略是 handwritten");
        c.check(cfg.ramen_region_model_path.is_none(), "默认不指定地区模型路径");
        c.check(validate_err(&cfg).is_empty(), "默认配置通过校验");
        c.finish()
    }

    /// 覆盖层能解析三个合法取值，未知取值在解析阶段被拒绝
    #[test]
    fn test_override_parses_region_policy() -> Result<()> {
        let mut c = Checks::new();
        for (text, want) in [
            ("handwritten", RamenRegionPolicy::Handwritten),
            ("region_nn_hint", RamenRegionPolicy::RegionNnHint),
            ("region_nn", RamenRegionPolicy::RegionNn)
        ] {
            let cfg = merge_fixture(&format!("ramen_region_policy = \"{text}\""))?;
            println!("{text} → {:?}", cfg.ramen_region_policy);
            c.check(cfg.ramen_region_policy == want, &format!("{text} 解析为 {want:?}"));
        }

        let only_path = merge_fixture("ramen_region_model_path = \"saved_models/x.onnx\"")?;
        c.check(
            only_path.ramen_region_policy == RamenRegionPolicy::Handwritten,
            "只写模型路径不会切换地区策略"
        );

        for bad in ["nn", "neural"] {
            let r = merge_fixture(&format!("ramen_region_policy = \"{bad}\""));
            println!("{bad} → {:?}", r.as_ref().err().map(ToString::to_string));
            c.check(r.is_err(), &format!("未知取值 {bad} 被拒绝"));
        }
        c.finish()
    }

    /// 三年 `RegionSelect` 的候选全部能解码成合法地区组合
    ///
    /// 走纯函数 `region_select_combos` 并显式传 `All`，不读全局配置。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_candidates_decode_all_three_years() -> Result<()> {
        use umasim::{
            game::ramen::{Operation, region_select_combos, rules::validate_region_selection},
            gamedata::{RamenRegionStrategy, init_global}
        };

        env::set_current_dir(get_workspace_root()?)?;
        init_global()?;
        let mut c = Checks::new();
        for (year_idx, want_n) in [(0usize, 10usize), (1, 10), (2, 120)] {
            let actions: Vec<RamenAction> = region_select_combos(year_idx, RamenRegionStrategy::All, None)?
                .into_iter()
                .map(|r| RamenAction::no_ramen(Operation::RegionSelect(r)))
                .collect();
            let bad = (0..actions.len())
                .filter(|&i| {
                    !RegionNnTrainer::region_of(&actions, i).is_ok_and(|r| validate_region_selection(year_idx, &r))
                })
                .count();
            println!("第 {} 年候选数 {} 非法 {bad}", year_idx + 1, actions.len());
            c.check(actions.len() == want_n, &format!("第 {} 年候选数 = {want_n}", year_idx + 1));
            c.check(bad == 0, &format!("第 {} 年全部候选合法且可解码", year_idx + 1));
        }
        c.finish()
    }

    /// 模型加载的错误路径：文件缺失 / 旁车缺失 / 旁车维度不符 / 不是合法 ONNX
    ///
    /// 全部必须在装配时报错。前三条在解析 ONNX 之前命中，占位文件即可覆盖。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_region_model_error_paths() -> Result<()> {
        use crate::utils::{cleanup_test_dir, unique_test_dir};

        env::set_current_dir(get_workspace_root()?)?;
        let mut c = Checks::new();
        let mut cfg = GameConfig::default_for_init();
        cfg.ramen_region_policy = RamenRegionPolicy::RegionNn;
        let err_of = |cfg: &GameConfig| {
            let mcts = RamenMctsTrainer::new(SearchConfig::new_game_config(&GameConfig::default_for_init()));
            build_client_trainer(cfg, mcts)
                .err()
                .map(|e| format!("{e:#}"))
                .unwrap_or_default()
        };

        cfg.ramen_region_model_path = Some("saved_models/__definitely_missing__.onnx".to_string());
        let e1 = err_of(&cfg);
        println!("模型缺失 → {e1}");
        c.check(e1.contains("不存在"), "模型缺失时报错");

        let dir = unique_test_dir("region_model_error_paths")?;
        let placeholder = b"not-a-real-onnx";
        let meta = |input_dim: usize| {
            format!(
                r#"{{"input_dim":{input_dim},"output_dim":245,"value_normalization":{{"center":[0.0,0.0,0.0],"scale":[1.0,1.0,1.0]}}}}"#
            )
        };

        let no_meta = dir.join("no_meta.onnx");
        fs_err::write(&no_meta, placeholder)?;
        cfg.ramen_region_model_path = Some(no_meta.to_string_lossy().into_owned());
        let e2 = err_of(&cfg);
        println!("旁车缺失 → {e2}");
        c.check(e2.contains("元数据"), "旁车缺失时错误指明模型元数据");

        let bad_dim = dir.join("bad_dim.onnx");
        fs_err::write(&bad_dim, placeholder)?;
        fs_err::write(dir.join("bad_dim.onnx.json"), meta(7))?;
        cfg.ramen_region_model_path = Some(bad_dim.to_string_lossy().into_owned());
        let e3 = err_of(&cfg);
        println!("维度不符 → {e3}");
        c.check(e3.contains("input_dim"), "维度不符时错误指明 input_dim");

        let not_onnx = dir.join("not_onnx.onnx");
        fs_err::write(&not_onnx, placeholder)?;
        fs_err::write(dir.join("not_onnx.onnx.json"), meta(754))?;
        cfg.ramen_region_model_path = Some(not_onnx.to_string_lossy().into_owned());
        let e4 = err_of(&cfg);
        println!("非法 ONNX → {e4}");
        c.check(!e4.is_empty(), "文件不是合法 ONNX 时报错");

        cleanup_test_dir(&dir)?;
        c.check(!dir.exists(), "fixture 目录已清理");
        c.finish()
    }

    /// 未开 `onnx` feature 时，需要网络的取值装配即报错
    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_region_nn_without_onnx_feature_errors() -> Result<()> {
        env::set_current_dir(get_workspace_root()?)?;
        let mut c = Checks::new();
        for policy in NN_POLICIES {
            let mut cfg = GameConfig::default_for_init();
            cfg.ramen_region_policy = policy;
            cfg.ramen_region_model_path = Some("saved_models/x.onnx".to_string());
            let mcts = RamenMctsTrainer::new(SearchConfig::new_game_config(&cfg));
            let msg = build_client_trainer(&cfg, mcts)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            println!("{policy:?} 未开 onnx → {msg}");
            c.check(msg.contains("onnx"), &format!("{policy:?}：错误信息指出需要 onnx feature"));
        }
        c.finish()
    }
}
