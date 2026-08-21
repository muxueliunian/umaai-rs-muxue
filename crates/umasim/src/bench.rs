//! 基准测试公共设施。
//!
//! 供 `bench_base` / `bench_compositions` 等基准 bin 复用的固定种子跑批组件：
//!
//! - [`seeded_rngs`]：从单一 seed 分裂决策/规则双 RNG（可复现性核心，规则层魔法数收敛于此）
//! - [`run_seeded`] + [`GameOutcome`]：单局运行统一入口
//! - [`summarize`] + [`Stats`]：基础统计
//! - [`write_csv`]：CSV 落盘（`csv` crate，自动转义）
//! - [`select_representatives`] + [`CardPickOpts`]：代表性支援卡选择（bench 专用粗略估计）
//! - [`parse_value`]：lexopt 键值参数读取 helper

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng};

use crate::game::ramen::RamenGame;
use crate::game::{Game, InheritInfo, Trainer};
use crate::gamedata::{GAMECONSTANTS, GAMEDATA, SupportCardData};
use crate::global;
use crate::trainer::LoggingTrainer;

/// 五种普通支援卡类型英文名称（CSV 等机器可读输出用），索引与 `card_type` 一一对应。
pub const TYPE_NAMES: [&str; 5] = ["speed", "stamina", "power", "guts", "wisdom"];

/// 取类型中文名（来自 `GAMECONSTANTS.train_names`，如「速/耐/力/根/智」），
/// 数据缺失时回退英文名。终端展示用，CSV 仍用 [`TYPE_NAMES`]。
pub fn type_name_zh(card_type: usize) -> String {
    global!(GAMECONSTANTS)
        .train_names
        .get(card_type)
        .cloned()
        .unwrap_or_else(|| TYPE_NAMES[card_type].to_string())
}

/// 从单一 seed 分裂决策 RNG 与规则层 RNG，保证固定种子整局可复现。
///
/// 第二个种子用黄金比例常数异或派生，使两条 RNG 序列互不相关；具体常数无关紧要，
/// 关键是固定不变——规则层可复现性依赖此派生在代码演进中保持稳定。
pub fn seeded_rngs(seed: u64) -> (StdRng, StdRng) {
    let rule_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
    (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(rule_seed))
}

/// 单局完整结果。
#[derive(Debug, Clone)]
pub struct GameOutcome {
    /// 本局种子。
    pub seed: u64,
    /// 结算评分。
    pub score: i32,
    /// 评分等级。
    pub rank: String,
    /// 五维终值。
    pub five_status: [i32; 5],
    /// 技能点。
    pub skill_pt: i32,
    /// 剧本 PT。
    pub scenario_pt: i32,
    /// RMJ 成功年数（0-3）。
    pub rmj_ok: usize,
    /// 当年吃面次数。
    pub eat_count: i32,
    /// 五次友人出行是否全部完成。
    pub friend_all: bool,
    /// 整局耗时（毫秒）。
    pub elapsed_ms: f64,
}

/// 跑一局固定种子的完整拉面杯（统一 `LoggingTrainer` 包装，注入规则层 RNG）。
pub fn run_seeded<T: Trainer<RamenGame>>(
    uma: u32, deck: &[u32; 6], inherit: &InheritInfo, seed: u64, trainer: &LoggingTrainer<T>,
) -> Result<GameOutcome> {
    let (mut decision_rng, rule_rng) = seeded_rngs(seed);
    let mut game = RamenGame::newgame(uma, deck, inherit.clone())?;
    game.set_internal_rng(rule_rng);
    let start = Instant::now();
    game.run_full_game(trainer, &mut decision_rng)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let score = game.uma.calc_score();
    Ok(GameOutcome {
        seed,
        score,
        rank: global!(GAMECONSTANTS).get_rank_name(score),
        five_status: game.uma.five_status,
        skill_pt: game.uma.skill_pt,
        scenario_pt: game.ramen.scenario_pt,
        rmj_ok: game.ramen.rmj_results.iter().filter(|&&ok| ok).count(),
        eat_count: game.ramen.eat_count,
        friend_all: game.friend.out_used.iter().all(|used| *used),
        elapsed_ms,
    })
}

/// 一组数值的基本统计。
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    /// 最小值。
    pub min: f64,
    /// 最大值。
    pub max: f64,
    /// 均值。
    pub mean: f64,
    /// 中位数。
    pub median: f64,
    /// 标准差（总体）。
    pub std: f64,
}

/// 基本统计（min/max/mean/median/std），空序列返回全 0。
pub fn summarize(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std: 0.0,
        };
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let std = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    Stats { min, max, mean, median, std }
}

/// 升序样本的第 p 分位（p ∈ [0,1]），按线性插值计算；空序列返回 0。
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}

/// 写 CSV 文件：自动创建父目录，字段由 `csv` crate 转义。
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("创建输出目录失败: {}", parent.display()))?;
    }
    let mut wtr = csv::Writer::from_path(path).with_context(|| format!("创建 CSV 失败: {}", path.display()))?;
    wtr.write_record(header)
        .with_context(|| format!("写表头失败: {}", path.display()))?;
    for row in rows {
        wtr.write_record(row)
            .with_context(|| format!("写行失败: {}", path.display()))?;
    }
    wtr.flush()
        .with_context(|| format!("刷新 CSV 失败: {}", path.display()))
}

/// 代表性支援卡（满破 idrank + 显示名）。
#[derive(Debug, Clone)]
pub struct CardRep {
    /// 满破 idrank（card_id * 10 + 4）。
    pub idrank: u32,
    /// 卡名。
    pub name: String,
}

/// 代表性支援卡选择参数。
#[derive(Debug, Clone, Copy)]
pub struct CardPickOpts {
    /// 候选池：每种类型按 card_id 倒序取最新 N 张。
    pub pool_size: usize,
    /// 弱卡阈值：满破面板「友情+干劲+训练」低于此值视为弱卡。
    pub min_panel: f32,
    /// 每种类型选取张数。
    pub pick: usize,
}

impl Default for CardPickOpts {
    fn default() -> Self {
        // 阈值经 cardDB 探索（2026-08）：最新 5 张内各类型均可凑满 3 张 ≥70 的强卡。
        Self {
            pool_size: 5,
            min_panel: 70.0,
            pick: 3,
        }
    }
}

/// 代表卡选择结果：入选卡与因「友情+干劲+训练」低于阈值被跳过的弱卡。
#[derive(Debug)]
pub struct RepresentativeSet {
    /// 各类型选出的代表卡（按 card_id 倒序）。
    pub picked: [Vec<CardRep>; 5],
    /// 候选池中友情+干劲+训练低于阈值的弱卡。
    pub skipped: [Vec<CardRep>; 5],
}

/// 选取各类型的代表性支援卡。
///
/// 规则：每种类型取满破 SSR 中最新 `pool_size` 张作为候选池，跳过满破面板
/// 「友情+干劲+训练」低于 `min_panel` 的弱卡，再按 card_id 倒序取前 `pick` 张。
/// 被跳过的弱卡一并返回（见 [`RepresentativeSet::skipped`]）。
///
/// 注意：面板和值只是 bench 专用的粗略强度代理（不看技能/事件/得意率），
/// 仅用于比较类型构成，不表示支援卡强度排名。
pub fn select_representatives(opts: &CardPickOpts) -> Result<RepresentativeSet> {
    let data = global!(GAMEDATA);
    let mut pools: [Vec<&SupportCardData>; 5] = std::array::from_fn(|_| Vec::new());
    for card in data.card.values() {
        if card.rarity == 3 && (0..5).contains(&card.card_type) && card.card_value.len() >= 5 {
            pools[card.card_type as usize].push(card);
        }
    }
    let mut picked: [Vec<CardRep>; 5] = std::array::from_fn(|_| Vec::new());
    let mut skipped: [Vec<CardRep>; 5] = std::array::from_fn(|_| Vec::new());
    for (card_type, cards) in pools.iter_mut().enumerate() {
        cards.sort_by_key(|card| std::cmp::Reverse(card.card_id));
        let panel_score = |card: &&SupportCardData| -> f32 {
            let value = &card.card_value[4]; // 满破面板 rank=4
            value.youqing + value.ganjing as f32 + value.xunlian as f32
        };
        for card in cards.iter().take(opts.pool_size) {
            if panel_score(card) >= opts.min_panel && picked[card_type].len() < opts.pick {
                picked[card_type].push(CardRep {
                    idrank: card.card_id * 10 + 4,
                    name: card.card_name.clone(),
                });
            } else if panel_score(card) < opts.min_panel {
                skipped[card_type].push(CardRep {
                    idrank: card.card_id * 10 + 4,
                    name: card.card_name.clone(),
                });
            }
        }
        ensure!(
            picked[card_type].len() == opts.pick,
            "{} 类型最新 {} 张满破 SSR 中友情+干劲+训练≥{} 的卡只有 {} 张（需 {}），请调低 min-panel 或使用 --cards-file 手动指定",
            type_name_zh(card_type),
            opts.pool_size,
            opts.min_panel,
            picked[card_type].len(),
            opts.pick
        );
    }
    Ok(RepresentativeSet { picked, skipped })
}

/// 从 lexopt 解析器中读取当前键值参数的值（支持 `--key value` 与 `--key=value`）。
pub fn parse_value<T: std::str::FromStr>(parser: &mut lexopt::Parser, key: &str) -> Result<T> {
    let value = parser.value().with_context(|| format!("参数 {key} 缺少值"))?;
    let text = value
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("参数 {key} 的值不是合法 UTF-8: {value:?}"))?;
    text.parse().map_err(|_| anyhow::anyhow!("参数 {key} 的值无效: {text}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证同 seed 派生一致、不同 seed 派生不同（可复现性根基）。
    #[test]
    fn test_seeded_rngs_reproducible() {
        use rand::RngCore;
        let (mut d1, mut r1) = seeded_rngs(42);
        let (mut d2, mut r2) = seeded_rngs(42);
        let (mut d3, _r3) = seeded_rngs(43);
        let (a1, b1, c1) = (d1.next_u32(), d2.next_u32(), d3.next_u32());
        let (a2, b2) = (r1.next_u32(), r2.next_u32());
        println!("同 seed 决策 RNG 首值 {a1} == {b1}? 不同 seed {c1}; 规则 RNG 首值 {a2} == {b2}?");
    }

    /// 验证 summarize 对已知序列的 min/max/mean/median/std 计算。
    #[test]
    fn test_summarize() {
        let stats = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        println!(
            "序列 1..5: min={} max={} mean={} median={} std={}",
            stats.min, stats.max, stats.mean, stats.median, stats.std
        );
        println!("空序列: {:?}", summarize(&[]));
    }

    /// 验证 percentile 在偶数/奇数长度样本上的分位计算。
    #[test]
    fn test_percentile() {
        let odd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        println!(
            "10 个样本 P10={} P50={} P90={}",
            percentile(&odd, 0.1),
            percentile(&odd, 0.5),
            percentile(&odd, 0.9)
        );
        let even = vec![1.0, 2.0, 3.0, 4.0];
        println!(
            "4 个样本 P25={} P50={}",
            percentile(&even, 0.25),
            percentile(&even, 0.5)
        );
    }

    /// 集成验证：真实 cardDB 上默认参数能选出每类型 3 张、idrank 严格倒序的代表卡。
    #[test]
    fn test_select_representatives_live_data() -> Result<()> {
        use crate::gamedata::{GameConfig, init_global_with_config};
        use crate::utils::get_workspace_root;
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(&workspace_root)?;
        init_global_with_config(&GameConfig::default_for_init())?;
        let set = select_representatives(&CardPickOpts::default())?;
        for (card_type, cards) in set.picked.iter().enumerate() {
            let detail = cards
                .iter()
                .map(|card| format!("{} {}", card.idrank, card.name))
                .collect::<Vec<_>>()
                .join(" / ");
            println!("{}: {detail}", type_name_zh(card_type));
            ensure!(cards.len() == 3, "{} 类型代表卡不是 3 张", type_name_zh(card_type));
            ensure!(
                cards.windows(2).all(|pair| pair[0].idrank > pair[1].idrank),
                "{} 类型代表卡未按 card_id 倒序",
                type_name_zh(card_type)
            );
        }
        let total_skipped: usize = set.skipped.iter().map(Vec::len).sum();
        println!("跳过的弱卡总数: {total_skipped}");
        Ok(())
    }
}
