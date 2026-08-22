//! 拉面杯全卡型构成基准。
//!
//! 枚举速/耐/力/根/智各 0..=3 张、普通卡合计 5 张的全部 101 种构成，
//! 再加入一张固定友人卡。每种类型使用 [`bench::select_representatives`] 选出的
//! 代表性满破 SSR（最新候选池 + 面板阈值过滤弱卡），也可用 `--cards-file` 手动指定；
//! 该选择只用于比较类型构成，不表示支援卡强度排名。
//!
//! 运行设施（固定种子双 RNG、单局运行、统计、CSV）复用 [`umasim::bench`]。
//!
//! ```text
//! cargo run --release --bin bench_compositions -- \
//!   [--runs N] [--seed S] [--friend IDRANK] \
//!   [--trainer random|handwritten] [--min-panel N] [--pool-size N] [--pick N] \
//!   [--cards-file FILE] [--out FILE]
//! ```

use std::{path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use lexopt::Arg;
use serde::Deserialize;
use umasim::{
    bench::{self, CardPickOpts, CardRep, DeckComposition},
    game::{InheritInfo, Trainer, ramen::RamenGame},
    gamedata::{GAMEDATA, init_global_with_config},
    global,
    trainer::{LoggingTrainer, RamenHandwrittenTrainer, RandomTrainer},
    utils::{get_workspace_root, load_game_config}
};

/// 默认测试马娘。
const DEFAULT_UMA: u32 = 102601;
/// 默认拉面杯友人卡（满破 idrank）。
const DEFAULT_FRIEND: u32 = 303054;
/// 默认继承因子，与 bench_base 保持一致。
const DEFAULT_INHERIT: InheritInfo = InheritInfo {
    blue_count: [12, 0, 0, 0, 6],
    extra_count: [10, 0, 0, 20, 20, 40]
};

/// 命令行配置。
#[derive(Debug, Clone)]
struct Config {
    /// 每种构成模拟局数。
    runs: usize,
    /// 基础种子。
    seed: u64,
    /// 固定友人卡 idrank。
    friend: u32,
    /// 训练员名称。
    trainer: String,
    /// 汇总 CSV 输出路径。
    out: String,
    /// 代表卡选择参数（pool_size/min_panel/pick 均可由 CLI 覆盖）。
    pick: CardPickOpts,
    /// 手动代表卡配置（toml），覆盖自动选择。
    cards_file: Option<String>
}

impl Default for Config {
    fn default() -> Self {
        Self {
            runs: 20,
            seed: 42,
            friend: DEFAULT_FRIEND,
            trainer: "handwritten".to_string(),
            out: "logs/bench_compositions.csv".to_string(),
            pick: CardPickOpts::default(),
            cards_file: None
        }
    }
}

/// 一种构成的聚合结果。
#[derive(Debug)]
struct Summary {
    /// 构成。
    composition: DeckComposition,
    /// 六张卡。
    deck: [u32; 6],
    /// 成功局数。
    completed: usize,
    /// 异常局数。
    failed: usize,
    /// 平均评分。
    score_mean: f64,
    /// 评分中位数。
    score_median: f64,
    /// 评分 P10。
    score_p10: f64,
    /// 五维均值。
    status_mean: [f64; 5],
    /// 训练技能 PT 均值。
    skill_pt_mean: f64,
    /// 三年 RMJ 全通率。
    rmj_all_rate: f64,
    /// 五次友人出行完成率。
    friend_all_rate: f64
}

impl Summary {
    /// 输出 CSV 行（不含表头）。
    fn csv_row(&self) -> Vec<String> {
        let mut row = vec![
            self.composition.name(),
            self.deck.iter().map(u32::to_string).collect::<Vec<_>>().join("/"),
            self.completed.to_string(),
            self.failed.to_string(),
            format!("{:.3}", self.score_mean),
            format!("{:.3}", self.score_median),
            format!("{:.3}", self.score_p10),
        ];
        row.extend(self.status_mean.iter().map(|mean| format!("{mean:.3}")));
        row.push(format!("{:.3}", self.skill_pt_mean));
        row.push(format!("{:.4}", self.rmj_all_rate));
        row.push(format!("{:.4}", self.friend_all_rate));
        row
    }
}

/// 解析命令行参数（lexopt，`--key value` 或 `--key=value`）。
fn parse_args() -> Result<Config> {
    let mut parser = lexopt::Parser::from_env();
    let mut cfg = Config::default();
    while let Some(arg) = parser.next()? {
        match arg {
            Arg::Long("runs") => cfg.runs = bench::parse_value(&mut parser, "runs")?,
            Arg::Long("seed") => cfg.seed = bench::parse_value(&mut parser, "seed")?,
            Arg::Long("friend") => cfg.friend = bench::parse_value(&mut parser, "friend")?,
            Arg::Long("trainer") => cfg.trainer = bench::parse_value(&mut parser, "trainer")?,
            Arg::Long("out") => cfg.out = bench::parse_value(&mut parser, "out")?,
            Arg::Long("min-panel") => cfg.pick.min_panel = bench::parse_value(&mut parser, "min-panel")?,
            Arg::Long("pool-size") => cfg.pick.pool_size = bench::parse_value(&mut parser, "pool-size")?,
            Arg::Long("pick") => cfg.pick.pick = bench::parse_value(&mut parser, "pick")?,
            Arg::Long("cards-file") => cfg.cards_file = Some(bench::parse_value(&mut parser, "cards-file")?),
            Arg::Long("help") | Arg::Short('h') => {
                println!(
                    "用法: bench_compositions [--runs N] [--seed S] [--friend IDRANK] \
                     [--trainer random|handwritten] [--min-panel N] [--pool-size N] [--pick N] \
                     [--cards-file FILE] [--out FILE]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("未知参数: {other:?}（可用 --help 查看用法）")
        }
    }
    ensure!(cfg.runs > 0, "--runs 必须大于 0");
    ensure!(
        matches!(cfg.trainer.as_str(), "random" | "handwritten"),
        "--trainer 只能为 random 或 handwritten"
    );
    Ok(cfg)
}

/// 枚举全部 101 种合法普通卡类型构成（复用 [`DeckComposition`]，无预设名）。
fn enumerate_compositions() -> Vec<DeckComposition> {
    let mut result = Vec::new();
    for speed in 0..=3 {
        for stamina in 0..=3 {
            for power in 0..=3 {
                for guts in 0..=3 {
                    for wisdom in 0..=3 {
                        let counts = [speed, stamina, power, guts, wisdom];
                        if counts.iter().sum::<usize>() == 5 {
                            result.push(DeckComposition { counts, name: String::new() });
                        }
                    }
                }
            }
        }
    }
    result
}

/// cards.toml 手动代表卡配置：每种类型的满破 idrank 列表（按序取前 pick 张）。
#[derive(Debug, Deserialize)]
struct ManualCards {
    /// 速。
    speed: Vec<u32>,
    /// 耐。
    stamina: Vec<u32>,
    /// 力。
    power: Vec<u32>,
    /// 根。
    guts: Vec<u32>,
    /// 智。
    wisdom: Vec<u32>
}

/// 读取手动代表卡配置（toml），转成与自动选择一致的 `RepresentativeSet`（无跳过卡）。
fn load_manual_cards(path: &str, pick: usize) -> Result<bench::RepresentativeSet> {
    let text = std::fs::read_to_string(path).with_context(|| format!("读取手动卡配置失败: {path}"))?;
    let manual: ManualCards = toml::from_str(&text).with_context(|| format!("解析手动卡配置失败: {path}"))?;
    let lists = [manual.speed, manual.stamina, manual.power, manual.guts, manual.wisdom];
    let data = global!(GAMEDATA);
    let mut result: [Vec<CardRep>; 5] = std::array::from_fn(|_| Vec::new());
    for (card_type, ids) in lists.into_iter().enumerate() {
        ensure!(
            ids.len() >= pick,
            "{} 类型手动卡不足 {pick} 张（当前 {}）",
            bench::type_name_zh(card_type),
            ids.len()
        );
        let mut reps = Vec::with_capacity(pick);
        for &idrank in ids.iter().take(pick) {
            let card = data
                .get_card(idrank / 10)
                .with_context(|| format!("{} 类型手动卡 idrank={idrank} 不存在", bench::type_name_zh(card_type)))?;
            reps.push(CardRep {
                idrank,
                name: card.card_name.clone()
            });
        }
        result[card_type] = reps;
    }
    Ok(bench::RepresentativeSet {
        picked: result,
        skipped: std::array::from_fn(|_| Vec::new())
    })
}

/// 使用指定训练员执行一种构成（每局构造 `LoggingTrainer`，与 bench_base 一致）。
fn run_composition<T: Trainer<RamenGame>>(
    cfg: &Config, composition: &DeckComposition, deck: [u32; 6], make_trainer: &dyn Fn(u64) -> LoggingTrainer<T>
) -> Summary {
    let mut scores = Vec::with_capacity(cfg.runs);
    let mut status_sum = [0_i64; 5];
    let mut skill_pt_sum = 0_i64;
    let mut rmj_all = 0_usize;
    let mut friend_all = 0_usize;
    let mut failed = 0_usize;

    for run_idx in 0..cfg.runs {
        let run_idx_u = run_idx as u64;
        let log_seed = cfg.seed + run_idx_u; // 决策日志标签（局号可读）
        let trainer = make_trainer(log_seed);
        match bench::run_seeded(DEFAULT_UMA, &deck, &DEFAULT_INHERIT, cfg.seed, run_idx_u, &trainer) {
            Ok(outcome) => {
                scores.push(outcome.score);
                for (idx, value) in outcome.five_status.iter().enumerate() {
                    status_sum[idx] += i64::from(*value);
                }
                skill_pt_sum += i64::from(outcome.skill_pt);
                rmj_all += usize::from(outcome.rmj_ok == 3);
                friend_all += usize::from(outcome.friend_all);
            }
            Err(error) => {
                failed += 1;
                eprintln!("构成 {} seed={} 模拟失败: {error:#}", composition.name_zh(), log_seed);
            }
        }
    }

    let completed = scores.len();
    let divisor = completed.max(1) as f64;
    let mut sorted: Vec<f64> = scores.iter().map(|score| f64::from(*score)).collect();
    sorted.sort_by(f64::total_cmp);
    let stats = bench::summarize(&sorted);
    Summary {
        composition: composition.clone(),
        deck,
        completed,
        failed,
        score_mean: stats.mean,
        score_median: stats.median,
        score_p10: bench::percentile(&sorted, 0.1),
        status_mean: std::array::from_fn(|idx| status_sum[idx] as f64 / divisor),
        skill_pt_mean: skill_pt_sum as f64 / divisor,
        rmj_all_rate: rmj_all as f64 / divisor,
        friend_all_rate: friend_all as f64 / divisor
    }
}

/// 将代表卡说明输出到终端（含被阈值跳过的弱卡）。
fn print_representatives(set: &bench::RepresentativeSet, opts: &CardPickOpts) {
    println!(
        "代表卡规则：各类型取最新 {} 张满破 SSR，友情+干劲+训练低于阈值{} 视为弱卡跳过；仅为确定性样本，不是强度排名",
        opts.pool_size, opts.min_panel
    );
    for (card_type, cards) in set.picked.iter().enumerate() {
        let detail = cards
            .iter()
            .map(|card| format!("{} {}", card.idrank, card.name))
            .collect::<Vec<_>>()
            .join(" / ");
        println!("  {}: {detail}", bench::type_name_zh(card_type));
        let skipped = &set.skipped[card_type];
        if !skipped.is_empty() {
            let detail = skipped
                .iter()
                .map(|card| format!("{} {}", card.idrank, card.name))
                .collect::<Vec<_>>()
                .join(" / ");
            println!("    跳过: {detail}");
        }
    }
}

/// 执行全部构成并返回汇总。
fn run_all(
    cfg: &Config, compositions: &[DeckComposition], representatives: &[Vec<CardRep>; 5]
) -> Result<Vec<Summary>> {
    let random = |seed: u64| LoggingTrainer::new(RandomTrainer, seed);
    let handwritten = |seed: u64| LoggingTrainer::new(RamenHandwrittenTrainer::new(), seed);
    let mut summaries = Vec::with_capacity(compositions.len());
    for (idx, composition) in compositions.iter().enumerate() {
        let deck = composition.build_deck(representatives, cfg.friend)?;
        eprintln!(
            "[{}/{}] {} {:?}",
            idx + 1,
            compositions.len(),
            composition.name_zh(),
            deck
        );
        let summary = match cfg.trainer.as_str() {
            "random" => run_composition(cfg, composition, deck, &random),
            "handwritten" => run_composition(cfg, composition, deck, &handwritten),
            _ => unreachable!("训练员已在参数解析时校验")
        };
        summaries.push(summary);
    }
    Ok(summaries)
}

/// 保存全部构成汇总 CSV。
fn save_csv(path: &str, summaries: &[Summary]) -> Result<()> {
    let header = [
        "composition",
        "deck",
        "completed",
        "failed",
        "score_mean",
        "score_median",
        "score_p10",
        "speed_mean",
        "stamina_mean",
        "power_mean",
        "guts_mean",
        "wisdom_mean",
        "skill_pt_mean",
        "rmj_all_rate",
        "friend_all_rate"
    ];
    let rows: Vec<Vec<String>> = summaries.iter().map(Summary::csv_row).collect();
    bench::write_csv(Path::new(path), &header, &rows)
}

/// 程序入口。
fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    let cfg = parse_args()?;
    let game_config = load_game_config()?;
    init_global_with_config(&game_config)?;

    let compositions = enumerate_compositions();
    ensure!(
        compositions.len() == 101,
        "合法构成应为 101 种，实际为 {}",
        compositions.len()
    );
    let representative_set = match &cfg.cards_file {
        Some(path) => load_manual_cards(path, cfg.pick.pick)?,
        None => bench::select_representatives(&cfg.pick)?
    };
    println!(
        "开始基准：trainer={} compositions={} runs_each={} total_runs={} base_seed={} friend={} min_panel={}",
        cfg.trainer,
        compositions.len(),
        cfg.runs,
        compositions.len() * cfg.runs,
        cfg.seed,
        cfg.friend,
        cfg.pick.min_panel,
    );

    let started = Instant::now();
    let summaries = run_all(&cfg, &compositions, &representative_set.picked)?;
    save_csv(&cfg.out, &summaries)?;
    // 代表卡列表放在所有组合跑批之后输出
    print_representatives(&representative_set, &cfg.pick);
    let failed = summaries.iter().map(|summary| summary.failed).sum::<usize>();
    println!(
        "完成：输出={} 耗时={:.2}s 异常局数={failed}",
        cfg.out,
        started.elapsed().as_secs_f64()
    );
    ensure!(failed == 0, "存在 {failed} 个异常局，基准结果不完整");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证完整枚举严格包含 101 种构成及所有边界约束。
    #[test]
    fn test_enumerate_all_101_compositions() -> Result<()> {
        let compositions = enumerate_compositions();
        println!("构成数量: {}", compositions.len());
        ensure!(compositions.len() == 101, "构成数量不是 101");
        ensure!(
            compositions.iter().all(|composition| {
                composition.counts.iter().sum::<usize>() == 5 && composition.counts.iter().all(|count| *count <= 3)
            }),
            "存在不满足合计五张或单类型最多三张的构成"
        );
        Ok(())
    }

    /// 验证每种构成都能生成五张普通卡加一张固定友人的卡组。
    #[test]
    fn test_build_all_composition_decks() -> Result<()> {
        let representatives: [Vec<CardRep>; 5] = std::array::from_fn(|card_type| {
            (0..3)
                .map(|idx| CardRep {
                    idrank: 100_000 + card_type as u32 * 100 + idx,
                    name: format!("type-{card_type}-{idx}")
                })
                .collect()
        });
        for composition in enumerate_compositions() {
            let deck = composition.build_deck(&representatives, DEFAULT_FRIEND)?;
            ensure!(deck[5] == DEFAULT_FRIEND, "最后一张不是固定友人");
            ensure!(deck[..5].iter().all(|id| *id != DEFAULT_FRIEND), "普通卡槽混入固定友人");
        }
        println!("全部 101 种卡组结构验证通过");
        Ok(())
    }
}
