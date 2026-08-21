//! 拉面杯全卡型构成基准。
//!
//! 枚举速/耐/力/根/智各 0..=3 张、普通卡合计 5 张的全部 101 种构成，
//! 再加入一张固定友人卡。每种类型使用 cardDB 中 card_id 最新的满破 SSR 作为
//! 确定性代表卡；该选择只用于比较类型构成，不表示支援卡强度排名。
//!
//! ```text
//! cargo run --release --bin bench_compositions -- \
//!   [--runs N] [--seed S] [--friend IDRANK] \
//!   [--trainer random|handwritten] [--out FILE]
//! ```

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng};
use umasim::game::ramen::RamenGame;
use umasim::game::{Game, InheritInfo, Trainer};
use umasim::gamedata::{GAMEDATA, init_global_with_config};
use umasim::global;
use umasim::trainer::{RamenHandwrittenTrainer, RandomTrainer};
use umasim::utils::{get_workspace_root, load_game_config};

/// 五种普通支援卡类型名称。
const TYPE_NAMES: [&str; 5] = ["speed", "stamina", "power", "guts", "wisdom"];
/// 默认测试马娘。
const DEFAULT_UMA: u32 = 102601;
/// 默认拉面杯友人卡（满破 idrank）。
const DEFAULT_FRIEND: u32 = 303054;
/// 默认继承因子，与 bench_base 保持一致。
const DEFAULT_INHERIT: InheritInfo = InheritInfo {
    blue_count: [12, 0, 0, 0, 6],
    extra_count: [10, 0, 0, 20, 20, 40],
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            runs: 20,
            seed: 42,
            friend: DEFAULT_FRIEND,
            trainer: "handwritten".to_string(),
            out: "logs/bench_compositions.csv".to_string(),
        }
    }
}

/// 一种普通卡类型数量构成。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Composition {
    /// 速、耐、力、根、智的卡片数量。
    counts: [usize; 5],
}

impl Composition {
    /// 返回便于 CSV 和日志阅读的构成名称。
    fn name(&self) -> String {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(idx, count)| format!("{count}{}", TYPE_NAMES[idx]))
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// 单个支援卡代表。
#[derive(Debug, Clone)]
struct CardRepresentative {
    /// 满破 idrank。
    idrank: u32,
    /// 显示名称。
    name: String,
}

/// 一种构成的聚合结果。
#[derive(Debug)]
struct Summary {
    /// 构成。
    composition: Composition,
    /// 六张卡。
    deck: [u32; 6],
    /// 成功局数。
    completed: usize,
    /// 异常局数。
    failed: usize,
    /// 平均评分。
    score_mean: f64,
    /// 评分中位数。
    score_median: i32,
    /// 评分 P10。
    score_p10: i32,
    /// 五维均值。
    status_mean: [f64; 5],
    /// 训练技能 PT 均值。
    skill_pt_mean: f64,
    /// 三年 RMJ 全通率。
    rmj_all_rate: f64,
    /// 五次友人出行完成率。
    friend_all_rate: f64,
}

impl Summary {
    /// 输出 CSV 行。
    fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4}",
            self.composition.name(),
            self.deck.iter().map(u32::to_string).collect::<Vec<_>>().join("/"),
            self.completed,
            self.failed,
            self.score_mean,
            self.score_median,
            self.score_p10,
            self.status_mean[0],
            self.status_mean[1],
            self.status_mean[2],
            self.status_mean[3],
            self.status_mean[4],
            self.skill_pt_mean,
            self.rmj_all_rate,
            self.friend_all_rate,
        )
    }
}

/// 解析命令行参数。
fn parse_args(args: &[String]) -> Result<Config> {
    let mut cfg = Config::default();
    let mut idx = 1;
    while idx < args.len() {
        let key = args[idx].as_str();
        match key {
            "--runs" => cfg.runs = parse_value(args, &mut idx, key)?,
            "--seed" => cfg.seed = parse_value(args, &mut idx, key)?,
            "--friend" => cfg.friend = parse_value(args, &mut idx, key)?,
            "--trainer" => cfg.trainer = parse_value(args, &mut idx, key)?,
            "--out" => cfg.out = parse_value(args, &mut idx, key)?,
            "--help" | "-h" => {
                println!(
                    "用法: bench_compositions [--runs N] [--seed S] \
                     [--friend IDRANK] [--trainer random|handwritten] [--out FILE]"
                );
                std::process::exit(0);
            }
            _ => anyhow::bail!("未知参数: {key}"),
        }
        idx += 1;
    }
    ensure!(cfg.runs > 0, "--runs 必须大于 0");
    ensure!(
        matches!(cfg.trainer.as_str(), "random" | "handwritten"),
        "--trainer 只能为 random 或 handwritten"
    );
    Ok(cfg)
}

/// 读取一个命令行参数值并推进索引。
fn parse_value<T: std::str::FromStr>(args: &[String], idx: &mut usize, key: &str) -> Result<T> {
    *idx += 1;
    let value = args.get(*idx).ok_or_else(|| anyhow::anyhow!("参数 {key} 缺少值"))?;
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("参数 {key} 的值无效: {value}"))
}

/// 枚举全部 101 种合法普通卡类型构成。
fn enumerate_compositions() -> Vec<Composition> {
    let mut result = Vec::new();
    for speed in 0..=3 {
        for stamina in 0..=3 {
            for power in 0..=3 {
                for guts in 0..=3 {
                    for wisdom in 0..=3 {
                        let counts = [speed, stamina, power, guts, wisdom];
                        if counts.iter().sum::<usize>() == 5 {
                            result.push(Composition { counts });
                        }
                    }
                }
            }
        }
    }
    result
}

/// 从 cardDB 为每种普通卡类型选取三个确定性代表。
///
/// 只考虑拥有满破面板的 SSR，按 card_id 从新到旧排列。此规则避免引入主观权重，
/// 但结果仅代表这些样本卡，不应解释为卡片强度排名。
fn select_representatives() -> Result<[Vec<CardRepresentative>; 5]> {
    let data = global!(GAMEDATA);
    let mut pools: [Vec<CardRepresentative>; 5] = std::array::from_fn(|_| Vec::new());
    for card in data.card.values() {
        if card.rarity == 3 && (0..5).contains(&card.card_type) && card.card_value.len() >= 5 {
            pools[card.card_type as usize].push(CardRepresentative {
                idrank: card.card_id * 10 + 4,
                name: card.card_name.clone(),
            });
        }
    }
    for (card_type, pool) in pools.iter_mut().enumerate() {
        pool.sort_by_key(|card| std::cmp::Reverse(card.idrank));
        ensure!(pool.len() >= 3, "{} 类型满破 SSR 不足三张", TYPE_NAMES[card_type]);
        pool.truncate(3);
    }
    Ok(pools)
}

/// 根据构成与固定友人生成六张卡组。
fn build_deck(
    composition: Composition, representatives: &[Vec<CardRepresentative>; 5], friend: u32,
) -> Result<[u32; 6]> {
    let mut deck = Vec::with_capacity(6);
    for (card_type, count) in composition.counts.iter().copied().enumerate() {
        ensure!(
            representatives[card_type].len() >= count,
            "{} 类型代表卡不足 {count} 张",
            TYPE_NAMES[card_type]
        );
        deck.extend(representatives[card_type].iter().take(count).map(|card| card.idrank));
    }
    deck.push(friend);
    deck.try_into()
        .map_err(|_| anyhow::anyhow!("卡组必须恰好包含五张普通卡和一张友人卡"))
}

/// 使用指定训练员执行一种构成。
fn run_composition<T: Trainer<RamenGame>>(
    cfg: &Config, composition: Composition, deck: [u32; 6], trainer: &T,
) -> Summary {
    let mut scores = Vec::with_capacity(cfg.runs);
    let mut status_sum = [0_i64; 5];
    let mut skill_pt_sum = 0_i64;
    let mut rmj_all = 0_usize;
    let mut friend_all = 0_usize;
    let mut failed = 0_usize;

    for run_idx in 0..cfg.runs {
        let seed = cfg.seed + run_idx as u64;
        let mut decision_rng = StdRng::seed_from_u64(seed);
        let rule_rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let game_result = RamenGame::newgame(DEFAULT_UMA, &deck, DEFAULT_INHERIT).and_then(|mut game| {
            game.set_internal_rng(rule_rng);
            game.run_full_game(trainer, &mut decision_rng)?;
            Ok(game)
        });
        match game_result {
            Ok(game) => {
                scores.push(game.uma.calc_score());
                for (idx, value) in game.uma.five_status.iter().enumerate() {
                    status_sum[idx] += i64::from(*value);
                }
                skill_pt_sum += i64::from(game.uma.skill_pt);
                rmj_all += usize::from(game.ramen.rmj_results.iter().take(3).all(|ok| *ok));
                friend_all += usize::from(game.friend.out_used.iter().all(|used| *used));
            }
            Err(error) => {
                failed += 1;
                eprintln!("构成 {} seed={} 模拟失败: {error:#}", composition.name(), seed);
            }
        }
    }

    scores.sort_unstable();
    let completed = scores.len();
    let divisor = completed.max(1) as f64;
    Summary {
        composition,
        deck,
        completed,
        failed,
        score_mean: scores.iter().map(|score| f64::from(*score)).sum::<f64>() / divisor,
        score_median: scores.get(completed / 2).copied().unwrap_or_default(),
        score_p10: scores
            .get(completed.saturating_sub(1) / 10)
            .copied()
            .unwrap_or_default(),
        status_mean: std::array::from_fn(|idx| status_sum[idx] as f64 / divisor),
        skill_pt_mean: skill_pt_sum as f64 / divisor,
        rmj_all_rate: rmj_all as f64 / divisor,
        friend_all_rate: friend_all as f64 / divisor,
    }
}

/// 将代表卡说明输出到终端。
fn print_representatives(representatives: &[Vec<CardRepresentative>; 5]) {
    println!("代表卡规则：各类型取 card_id 最新的三张满破 SSR（仅为确定性样本，不是强度排名）");
    for (card_type, cards) in representatives.iter().enumerate() {
        let detail = cards
            .iter()
            .map(|card| format!("{} {}", card.idrank, card.name))
            .collect::<Vec<_>>()
            .join(" / ");
        println!("  {}: {detail}", TYPE_NAMES[card_type]);
    }
}

/// 执行全部构成并返回汇总。
fn run_all(
    cfg: &Config, compositions: &[Composition], representatives: &[Vec<CardRepresentative>; 5],
) -> Result<Vec<Summary>> {
    let mut summaries = Vec::with_capacity(compositions.len());
    let random = RandomTrainer;
    let handwritten = RamenHandwrittenTrainer::new();
    for (idx, composition) in compositions.iter().copied().enumerate() {
        let deck = build_deck(composition, representatives, cfg.friend)?;
        eprintln!("[{}/{}] {} {:?}", idx + 1, compositions.len(), composition.name(), deck);
        let summary = match cfg.trainer.as_str() {
            "random" => run_composition(cfg, composition, deck, &random),
            "handwritten" => run_composition(cfg, composition, deck, &handwritten),
            _ => unreachable!("训练员已在参数解析时校验"),
        };
        summaries.push(summary);
    }
    Ok(summaries)
}

/// 保存全部构成汇总 CSV。
fn save_csv(path: &str, summaries: &[Summary]) -> Result<()> {
    let output_path = std::path::Path::new(path);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("创建输出目录失败: {}", parent.display()))?;
    }
    let mut csv = String::from(
        "composition,deck,completed,failed,score_mean,score_median,score_p10,speed_mean,stamina_mean,power_mean,guts_mean,wisdom_mean,skill_pt_mean,rmj_all_rate,friend_all_rate\n",
    );
    for summary in summaries {
        csv.push_str(&summary.csv_row());
        csv.push('\n');
    }
    std::fs::write(output_path, csv).with_context(|| format!("写入结果失败: {}", output_path.display()))
}

/// 程序入口。
fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    let cfg = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let game_config = load_game_config()?;
    init_global_with_config(&game_config)?;

    let compositions = enumerate_compositions();
    ensure!(
        compositions.len() == 101,
        "合法构成应为 101 种，实际为 {}",
        compositions.len()
    );
    let representatives = select_representatives()?;
    print_representatives(&representatives);
    println!(
        "开始基准：trainer={} compositions={} runs_each={} total_runs={} base_seed={} friend={}",
        cfg.trainer,
        compositions.len(),
        cfg.runs,
        compositions.len() * cfg.runs,
        cfg.seed,
        cfg.friend,
    );

    let started = Instant::now();
    let summaries = run_all(&cfg, &compositions, &representatives)?;
    save_csv(&cfg.out, &summaries)?;
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
        let representatives: [Vec<CardRepresentative>; 5] = std::array::from_fn(|card_type| {
            (0..3)
                .map(|idx| CardRepresentative {
                    idrank: 100_000 + card_type as u32 * 100 + idx,
                    name: format!("type-{card_type}-{idx}"),
                })
                .collect()
        });
        for composition in enumerate_compositions() {
            let deck = build_deck(composition, &representatives, DEFAULT_FRIEND)?;
            ensure!(deck[5] == DEFAULT_FRIEND, "最后一张不是固定友人");
            ensure!(deck[..5].iter().all(|id| *id != DEFAULT_FRIEND), "普通卡槽混入固定友人");
        }
        println!("全部 101 种卡组结构验证通过");
        Ok(())
    }
}
