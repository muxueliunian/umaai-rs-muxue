//! 客户端搜索预算与决策耗时核实工具（**只测量，不改搜索**）
//!
//! ## 用途
//!
//! 按 `umaai` 主程序（`crates/umaai/src/main.rs`）**同一条调用链与同一份合并配置**
//! 测量单个决策根的成本：
//!
//! - 配置：`load_game_config()`（`gamedata/default_config.toml` + `game_config.toml` 合并）
//!   → `SearchConfig::new_game_config`，阶段门控走 `mcts.ramen_search_stages`。
//! - 线程：`rayon` 全局池按 `collector.threads`（可被 `[config_override] num_threads` 覆盖）。
//! - 决策：`umaai::scenario::ramen::calc_ramen_training`，即客户端 `process_ramen`
//!   内部真正出推荐的那一段（含定向连续决策）。
//!
//! ## 三层计时口径（不要混用）
//!
//! | 层 | 来源 | 覆盖范围 |
//! |---|---|---|
//! | 搜索耗时 | `SearchProbe::elapsed` | `FlatSearch` 内核：候选分配 + 全部 rollout + 汇总排序 |
//! | 决策耗时 | `DecisionProbe::elapsed` | 一次完整 `select_action`：合并候选枚举 + 搜索 + 搜索后 `stash`/`reason` 收尾 |
//! | 整链耗时 | 本工具计时 | 一次 `calc_ramen_training`：链上**全部**决策 + 阶段推进 + 屏幕/理由渲染 |
//!
//! 未搜索的决策（门控关 / 单候选 / 合并缓存命中）没有搜索耗时，但**有**决策耗时。
//!
//! ## 两种「决策条数」
//!
//! - **执行决策次数** = `DecisionProbe` 条数 = `select_action` 实际被调用的次数。
//! - **输出决策记录数** = `calc_ramen_training` 返回的 `Vec` 长度 = 真正会被
//!   `DecisionSink` emit 的条数。合并 `RamenSelect` 路径搜了但不暴露
//!   `DecisionInfo`，两者因此会不相等——这是既有行为，本工具只负责把差额显示出来。
//!
//! ## ❗输入不是真实客户端快照
//!
//! 工作区内没有 `logs/GameStatusSend_Ramen/*.json`（协议单测引用的 151 份样本目录
//! 不存在），故本工具用 `RamenGame::newgame` + 手写推荐策略推进到目标
//! `(阶段, 回合)` 生成**模拟根**。模拟根与真实客户端快照在局面分布上并不等价，
//! 报告中必须按模拟根口径陈述。
//!
//! ## ❗`--budget-secs` 是软门限
//!
//! 它只在**启动下一次测量之前**检查，**不会中断**正在进行的建根或搜索。单次测量
//! 超长时总墙钟会越过该值；越界会在汇总里显式标出，但不能把它当作硬上限。
//!
//! ## 用法
//!
//! ```text
//! cargo run --release --bin ramen_client_cost -- --roots Train:1,Train:30 --repeats 2 --budget-secs 600
//! ```

use std::{
    env,
    sync::{Arc, Mutex},
    time::{Duration, Instant}
};

use anyhow::{Result, anyhow, bail};
use rand::{SeedableRng, rngs::StdRng};
use rayon::{ThreadPoolBuilder, current_num_threads};
use umasim::{
    game::{
        Game,
        InheritInfo,
        ramen::{RamenGame, RamenStage}
    },
    gamedata::init_global_with_config,
    search::{SearchConfig, SearchProbe},
    trainer::{DecisionPath, DecisionProbe, RamenMctsTrainer, RamenSearchStages, RecommendedRamenTrainer},
    utils::{get_workspace_root, init_logger_stdout, load_game_config}
};

use umaai::{decision::LastReasonSink, scenario::ramen::calc_ramen_training};

/// 一个待测根的规格
///
/// 建新类型而非 `(String, i32)`：两个字段都能从 CLI 串里解析出来，位置写反不会
/// 编译报错，只会静默量错根。
#[derive(Debug, Clone)]
struct RootSpec {
    /// 目标阶段名（与 `RamenStage` 的 `Debug` 形式大小写无关地比较）
    stage: String,
    /// 命中该阶段所需的最小回合
    min_turn: i32
}

impl RootSpec {
    /// 解析 `阶段:回合`
    ///
    /// # 错误
    ///
    /// 缺少 `:`、阶段名为空、回合数不是 `i32` 时报错——**不做**任何默认回退，
    /// 否则会静默量到另一个根上。
    fn parse(text: &str) -> Result<Self> {
        let (stage, turn) = text
            .split_once(':')
            .ok_or_else(|| anyhow!("根规格应为 阶段:回合，实际 {text:?}"))?;
        if stage.is_empty() {
            bail!("根规格的阶段名为空：{text:?}");
        }
        Ok(Self {
            stage: stage.to_string(),
            min_turn: turn.parse::<i32>()?
        })
    }

    /// 报告里用的短标签
    fn label(&self) -> String {
        format!("{}:{}", self.stage, self.min_turn)
    }
}

/// 命令行参数
struct Args {
    /// 根面板
    roots: Vec<RootSpec>,
    /// 每个根在首次运行之外的热态重复次数
    repeats: usize,
    /// 软墙钟门限（秒）：只在启动下一次测量前检查，不中断进行中的测量
    budget_secs: u64,
    /// 建根用的随机种子
    seed: u64
}

impl Default for Args {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            repeats: 2,
            budget_secs: 600,
            seed: 12648430
        }
    }
}

/// 解析 `--roots` / `--repeats` / `--budget-secs` / `--seed`
fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].as_str();
        let val = raw.get(i + 1).cloned();
        match key {
            "--roots" => {
                let v = val.ok_or_else(|| anyhow!("--roots 需要取值"))?;
                args.roots = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(RootSpec::parse)
                    .collect::<Result<Vec<_>>>()?;
                i += 2;
            }
            "--repeats" => {
                args.repeats = val.ok_or_else(|| anyhow!("--repeats 需要取值"))?.parse()?;
                i += 2;
            }
            "--budget-secs" => {
                args.budget_secs = val.ok_or_else(|| anyhow!("--budget-secs 需要取值"))?.parse()?;
                i += 2;
            }
            "--seed" => {
                args.seed = val.ok_or_else(|| anyhow!("--seed 需要取值"))?.parse()?;
                i += 2;
            }
            other => bail!("未知参数 {other:?}")
        }
    }
    if args.roots.is_empty() {
        bail!("必须用 --roots 指定根面板（例：--roots Train:1,RamenSelect:10）");
    }
    Ok(args)
}

/// `DecisionPath` → 报告用短名
fn path_name(path: DecisionPath) -> &'static str {
    match path {
        DecisionPath::Searched => "搜索",
        DecisionPath::CombinedCacheHit => "合并缓存命中",
        DecisionPath::FallbackSingleCandidate => "单候选转发手写",
        DecisionPath::FallbackGated => "门控关转发手写"
    }
}

/// 把一局推进到 `(阶段, 回合 >= min_turn)` 的第一个决策点，作为模拟根
///
/// 与 `ramen_root_bench::build_root` 同构：用手写推荐策略（rollout 实例，
/// 关掉 breakdown 输出）推进，不跑任何搜索。
fn build_root(uma: u32, cards: &[u32; 6], inherit: &InheritInfo, seed: u64, spec: &RootSpec) -> Result<RamenGame> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut game = RamenGame::newgame(uma, cards, inherit.clone())?;
    game.set_rule_master(seed ^ 0x5EED);
    let guide = RecommendedRamenTrainer::for_rollout();
    while game.next() {
        let is_decision = matches!(
            game.stage,
            RamenStage::Train
                | RamenStage::RamenSelect
                | RamenStage::SpecialSelect
                | RamenStage::RegionSelect
                | RamenStage::SuperRamenSelect
        );
        if is_decision
            && format!("{:?}", game.stage).eq_ignore_ascii_case(&spec.stage)
            && game.turn() >= spec.min_turn
        {
            return Ok(game);
        }
        game.run_stage(&guide, &mut rng)?;
    }
    bail!(
        "推进到终局仍未命中目标根（阶段 {}，回合 >= {}）",
        spec.stage,
        spec.min_turn
    )
}

/// 一次测量用到的两个探针出口
struct Probes {
    /// 搜索内核记录
    search: Arc<Mutex<Vec<SearchProbe>>>,
    /// 决策记录（含未搜索的决策）
    decision: Arc<Mutex<Vec<DecisionProbe>>>
}

impl Probes {
    /// 清空两个出口，准备下一次测量
    fn reset(&self) -> Result<()> {
        self.search.lock().map_err(|e| anyhow!("搜索探针锁中毒: {e}"))?.clear();
        self.decision
            .lock()
            .map_err(|e| anyhow!("决策探针锁中毒: {e}"))?
            .clear();
        Ok(())
    }
}

/// 跑一次完整决策链，打印逐决策的原始记录
fn measure_once(
    trainer: &RamenMctsTrainer, root: &RamenGame, rng_seed: u64, reason_slot: &Arc<LastReasonSink>, probes: &Probes
) -> Result<()> {
    probes.reset()?;
    let mut game = root.clone();
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let noop = |_: &str| {};
    let t0 = Instant::now();
    let chain_out = calc_ramen_training(trainer, &mut game, &mut rng, true, reason_slot, &noop)?;
    let chain = t0.elapsed();

    let decisions = probes
        .decision
        .lock()
        .map_err(|e| anyhow!("决策探针锁中毒: {e}"))?
        .clone();
    let searches = probes
        .search
        .lock()
        .map_err(|e| anyhow!("搜索探针锁中毒: {e}"))?
        .clone();

    // 搜索记录按发生顺序与「走了搜索的决策」一一对应
    let mut si = 0usize;
    for (di, d) in decisions.iter().enumerate() {
        println!(
            "    决策#{di}: 阶段={:?} 回合={} list_actions={} 路径={} 决策耗时={:.3}s 暴露DecisionInfo={}",
            d.stage,
            d.turn,
            d.actions_len,
            path_name(d.path),
            d.elapsed.as_secs_f64(),
            d.exposes_decision_info
        );
        if d.path != DecisionPath::Searched {
            continue;
        }
        let Some(p) = searches.get(si) else {
            bail!("决策#{di} 标为已搜索，但搜索探针只有 {} 条", searches.len());
        };
        si += 1;
        println!(
            "      搜索: 搜索候选={} radical={:.4} 中选#{} 搜索耗时={:.3}s（占决策 {:.1}%）",
            p.candidates,
            p.radical_factor,
            p.best_action_idx,
            p.elapsed.as_secs_f64(),
            100.0 * p.elapsed.as_secs_f64() / d.elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
        );
        println!(
            "      总续跑: 计划={} 成功={} 失败={}",
            p.total_planned(),
            p.total_succeeded(),
            p.total_failed()
        );
        for c in 0..p.candidates {
            println!(
                "      候选[{c}]: 计划={} 成功={} 失败={}",
                p.planned[c], p.succeeded[c], p.failed[c]
            );
        }
    }
    if si != searches.len() {
        bail!("搜索探针有 {} 条未配上决策记录", searches.len() - si);
    }
    let searched_decisions = decisions.iter().filter(|d| d.path == DecisionPath::Searched).count();
    println!(
        "    整链耗时={:.3}s 执行决策次数={} （其中搜索 {}）输出决策记录数={}",
        chain.as_secs_f64(),
        decisions.len(),
        searched_decisions,
        chain_out.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // 与 ramen_manual / ramen_turn_inspect 同约束：cwd 必须是 workspace 根
    env::set_current_dir(get_workspace_root()?)?;

    // ---- 与 main.rs 完全同序：先读配置，再初始化日志 / 全局数据 / 线程池 ----
    let game_config = load_game_config()?;
    let mcts_config = SearchConfig::new_game_config(&game_config);
    init_logger_stdout("ramen_client_cost", &game_config.log_level)?;
    init_global_with_config(&game_config)?;
    ThreadPoolBuilder::new()
        .num_threads(game_config.collector.threads)
        .build_global()?;

    // ---- 与 main.rs 完全一致地构造拉面训练员 ----
    let ramen_stages = RamenSearchStages::parse(&game_config.mcts.ramen_search_stages)?;
    let reason_slot = LastReasonSink::new();
    let probes = Probes {
        search: Arc::new(Mutex::new(Vec::new())),
        decision: Arc::new(Mutex::new(Vec::new()))
    };
    let mut trainer = RamenMctsTrainer::new(mcts_config.clone())
        .with_stages(ramen_stages)
        .verbose(true)
        .with_reason_sink(reason_slot.clone())
        // 与 main.rs 的唯一差异：挂上两个只读成本探针
        .with_decision_probe(probes.decision.clone());
    trainer.search = trainer.search.with_probe(probes.search.clone());

    println!("=== 生效配置（合并后）===");
    println!(
        "search_n={} use_ucb={} search_group_size={} search_cpuct={} expected_search_stdev={}",
        mcts_config.search_n,
        mcts_config.use_ucb,
        mcts_config.search_group_size,
        mcts_config.search_cpuct,
        mcts_config.expected_search_stdev
    );
    println!(
        "radical_factor_max={} max_depth={} policy_delta={} ramen_search_stages={:?}",
        mcts_config.radical_factor_max,
        mcts_config.max_depth,
        mcts_config.policy_delta,
        game_config.mcts.ramen_search_stages
    );
    println!(
        "selection={:?} use_combined_ramen_select={} stages={:?}",
        trainer.selection, trainer.use_combined_ramen_select, trainer.stages
    );
    println!(
        "ramen_region_strategy={:?} threads(rayon)={} uma={} cards={:?}",
        game_config.ramen_region_strategy,
        current_num_threads(),
        game_config.uma,
        game_config.cards
    );
    println!(
        "建根种子={} 热态重复={} 软门限={}s（只拦下一次测量的启动，不中断进行中的测量）",
        args.seed, args.repeats, args.budget_secs
    );
    println!(
        "根面板（{} 个）: {}",
        args.roots.len(),
        args.roots.iter().map(RootSpec::label).collect::<Vec<_>>().join(", ")
    );
    println!();

    let inherit = InheritInfo {
        blue_count: game_config.blue_count,
        extra_count: game_config.extra_count
    };
    let budget = Duration::from_secs(args.budget_secs);
    let wall0 = Instant::now();
    let mut skipped: Vec<String> = Vec::new();
    let mut done = 0usize;

    for spec in &args.roots {
        let label = spec.label();
        if wall0.elapsed() >= budget {
            skipped.push(format!("{label} 全部 {} 次（未开始：已过软门限）", args.repeats + 1));
            continue;
        }
        println!("=== 根 {label} ===");
        let t_build = Instant::now();
        let root = build_root(game_config.uma, &game_config.cards, &inherit, args.seed, spec)?;
        println!(
            "  建根耗时={:.3}s 实际 stage={:?} 回合={} list_actions 候选={}",
            t_build.elapsed().as_secs_f64(),
            root.stage,
            root.turn(),
            root.list_actions()?.len()
        );

        for run in 0..=args.repeats {
            if wall0.elapsed() >= budget {
                skipped.push(format!("{label} run#{run}（未开始：已过软门限）"));
                break;
            }
            let tag = if run == 0 { "首次" } else { "热态" };
            println!("  -- {tag} run#{run} --");
            // 每次重复用同一 rng 种子：链路上的随机消耗一致，量的是同一件事
            measure_once(&trainer, &root, args.seed ^ 0xABCD, &reason_slot, &probes)?;
            done += 1;
            println!("    累计墙钟={:.1}s", wall0.elapsed().as_secs_f64());
        }
        println!();
    }

    let total = wall0.elapsed();
    println!("=== 汇总 ===");
    println!(
        "完成测量 {done} 次 / 计划 {} 次；总墙钟={:.1}s，软门限={}s{}",
        args.roots.len() * (args.repeats + 1),
        total.as_secs_f64(),
        args.budget_secs,
        if total >= budget { "（❗已越过软门限）" } else { "" }
    );
    if skipped.is_empty() {
        println!("未完成项：无");
    } else {
        println!("未完成项（已过软门限；不降预算冒充实配结果）：");
        for s in &skipped {
            println!("  - {s}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, bail};

    use super::RootSpec;

    /// 合法根规格：阶段名原样保留，回合数解析成 `i32`，`label()` 能还原输入
    #[test]
    fn test_root_spec_parse_ok() -> Result<()> {
        for (text, stage, turn) in [
            ("Train:1", "Train", 1),
            ("RamenSelect:30", "RamenSelect", 30),
            ("regionselect:23", "regionselect", 23),
            ("Train:0", "Train", 0)
        ] {
            let spec = RootSpec::parse(text)?;
            println!("{text:?} → stage={:?} min_turn={}（label={}）", spec.stage, spec.min_turn, spec.label());
            if spec.stage != stage || spec.min_turn != turn {
                bail!("根规格解析错位：{text:?} → {spec:?}");
            }
            if spec.label() != text {
                bail!("label 未还原输入：{text:?} → {}", spec.label());
            }
        }
        Ok(())
    }

    /// 非法根规格必须报错，**不做**任何默认回退（否则会静默量错根）
    #[test]
    fn test_root_spec_parse_err() -> Result<()> {
        for bad in ["Train", "", "Train:", ":5", "Train:x", "Train:1:2", "Train 1"] {
            match RootSpec::parse(bad) {
                Ok(spec) => bail!("非法根规格 {bad:?} 未报错，反而得到 {spec:?}"),
                Err(e) => println!("非法根规格 {bad:?} 正确报错：{e}")
            }
        }
        Ok(())
    }
}
