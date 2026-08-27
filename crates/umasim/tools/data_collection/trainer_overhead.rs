//! 决策器粗略性能对比：同一组种子下 `RamenHandwrittenTrainer` 与
//! `RecommendedRamenTrainer` 各跑 N 局，按 `LoggingTrainer` 的 `elapsed_us`
//! 累计「决策时间」，整局 `elapsed_ms` 减去决策累计为「局面生成时间」。
//!
//! 用法：`cargo run --release --bin trainer_overhead_diagnostic`
//! 可选环境变量：
//! - `TRAINER_OVERHEAD_RUNS`：跑批数（默认 100）
//! - `TRAINER_OVERHEAD_SEED`：基础种子（默认 61444）
//!
//! 用途：在切换 MCTS rollout / fallback 之前，先确认两个 trainer 单机的
//! 决策耗时量级——这是「把 rollout 切到 RecommendedRamenTrainer 后，单次
//! 搜索实际要多花多少 ms」的上界估计。

use std::{collections::BTreeMap, env, time::Instant as StdInstant};

use anyhow::Result;
use umasim::{
    bench, game::InheritInfo, gamedata::init_global_with_config, trainer::{
        LoggingTrainer, RamenHandwrittenTrainer, RecommendedRamenTrainer
    }, utils::{get_workspace_root, load_game_config}
};

/// 美浦波旁（memory 中"玩家手动基准"使用的同一匹）
const UMA: u32 = 102_601;
/// 标准 speed build 卡组
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
/// 种马继承（speed build）
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};

/// 一档 trainer 的统计聚合
struct TrainerStats {
    name: &'static str,
    /// 整局平均耗时（ms）
    avg_total_ms: f64,
    /// 单局平均决策耗时（ms），由 LoggingTrainer 的 elapsed_us 累加
    avg_decision_ms: f64,
    /// 单局平均局面生成耗时（ms）= 整局 - 决策
    avg_simulate_ms: f64,
    /// 决策占总耗时的比例
    decision_ratio: f64,
    /// 总决策点（select_action + select_choice + select_event_choice）数
    avg_decision_count: f64,
    /// 按 stage 分组的平均单次决策耗时（μs）
    per_stage_us: BTreeMap<String, f64>,
    /// 各候选数分布（用于评估决策成本 vs 候选数）
    avg_candidates: f64
}

/// 跑一档 trainer 在 `runs` 个种子上的整局耗时与决策分解
fn benchmark_trainer<T>(name: &'static str, runs: u64, base_seed: u64, make_trainer: impl Fn() -> T)
    -> Result<TrainerStats>
where
    T: umasim::game::Trainer<umasim::game::ramen::RamenGame> + Send + Sync
{
    let mut total_ms_sum = 0.0f64;
    let mut decision_us_sum = 0.0f64;
    let mut decision_count_sum = 0.0f64;
    let mut candidate_sum = 0.0f64;
    // 按 stage 累计 elapsed_us 与出现次数（→ 平均）
    let mut stage_us_sum: BTreeMap<String, f64> = BTreeMap::new();
    let mut stage_count: BTreeMap<String, u64> = BTreeMap::new();

    for run_idx in 0..runs {
        // 关掉 decision_log 以避免 24×100 行 CSV 写入放大耗时测量
        let mut trainer = LoggingTrainer::new(make_trainer(), run_idx);
        trainer.set_logging(false);
        let outcome = bench::run_seeded(UMA, &DECK, &INHERIT, base_seed, run_idx, &trainer)?;
        let total_ms = outcome.elapsed_ms;
        total_ms_sum += total_ms;

        // 决策日志未启用——为了拿到 elapsed_us，需要再跑一遍开启 logging 的版本。
        // 这两次跑耗时几乎相同（只多 DecisionLog 的少量 format + RefCell push），
        // 但为了避免再跑一遍对总耗时的污染，本测试只在最后一局开启 logging 取一次分阶段数据。
        if run_idx == 0 {
            let mut trainer_log = LoggingTrainer::new(make_trainer(), run_idx);
            trainer_log.set_logging(true);
            let _ = bench::run_seeded(UMA, &DECK, &INHERIT, base_seed, run_idx, &trainer_log)?;
            let log = trainer_log.take_records();
            decision_count_sum = log.rows.len() as f64;
            for row in &log.rows {
                decision_us_sum += row.elapsed_us as f64;
                candidate_sum += row.candidates as f64;
                *stage_us_sum.entry(row.stage.clone()).or_default() += row.elapsed_us as f64;
                *stage_count.entry(row.stage.clone()).or_default() += 1;
            }
        }
    }

    let avg_total_ms = total_ms_sum / runs as f64;
    // 决策耗时按"第 0 局"估，不代表全程——但 candidate 数 / stage 分布是确定的，
    // 决策耗时也可按"第一局耗时 × runs"外推。这里采用 first-game 外推，简单标注。
    let avg_decision_ms = decision_us_sum / 1000.0;
    let avg_simulate_ms = (avg_total_ms - avg_decision_ms).max(0.0);
    let decision_ratio = if avg_total_ms > 0.0 { avg_decision_ms / avg_total_ms } else { 0.0 };
    let avg_decision_count = decision_count_sum;
    let avg_candidates = if decision_count_sum > 0.0 {
        candidate_sum / decision_count_sum
    } else {
        0.0
    };
    let per_stage_us: BTreeMap<String, f64> = stage_us_sum
        .into_iter()
        .filter_map(|(stage, sum)| {
            let n = *stage_count.get(&stage)?;
            Some((stage, sum / n as f64))
        })
        .collect();

    Ok(TrainerStats {
        name,
        avg_total_ms,
        avg_decision_ms,
        avg_simulate_ms,
        decision_ratio,
        avg_decision_count,
        per_stage_us,
        avg_candidates
    })
}

fn print_stats(stats: &TrainerStats) {
    println!("\n========== {} ==========", stats.name);
    println!("  整局平均耗时:   {:8.1} ms", stats.avg_total_ms);
    println!("  决策累计耗时:   {:8.1} ms  (第 0 局外推，估读)", stats.avg_decision_ms);
    println!("  局面生成耗时:   {:8.1} ms", stats.avg_simulate_ms);
    println!("  决策 / 总耗时:  {:6.1}%", stats.decision_ratio * 100.0);
    println!("  平均决策点数:   {:.0}", stats.avg_decision_count);
    println!("  平均候选数:     {:.2}", stats.avg_candidates);
    println!("  按阶段平均决策耗时 (μs):");
    let mut entries: Vec<_> = stats.per_stage_us.iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (stage, us) in entries {
        println!("    {:30} {:8.1} μs", stage, us);
    }
}

fn main() -> Result<()> {
    let runs: u64 = env::var("TRAINER_OVERHEAD_RUNS").unwrap_or_else(|_| "100".into()).parse()?;
    let base_seed: u64 = env::var("TRAINER_OVERHEAD_SEED").unwrap_or_else(|_| "61444".into()).parse()?;

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(workspace_root)?;
    init_global_with_config(&load_game_config()?)?;

    println!(
        "Trainer overhead diagnostic: runs={runs} base_seed={base_seed} uma={UMA} deck={DECK:?}"
    );
    let wall = StdInstant::now();

    let hw_stats = benchmark_trainer("RamenHandwrittenTrainer", runs, base_seed, || {
        RamenHandwrittenTrainer::new()
    })?;
    print_stats(&hw_stats);

    let rec_stats = benchmark_trainer("RecommendedRamenTrainer", runs, base_seed, || {
        RecommendedRamenTrainer::new()
    })?;
    print_stats(&rec_stats);

    println!("\n========== 对比 ==========");
    let total_ratio = rec_stats.avg_total_ms / hw_stats.avg_total_ms;
    let decision_ratio = rec_stats.avg_decision_ms / hw_stats.avg_decision_ms.max(0.001);
    let simulate_ratio = rec_stats.avg_simulate_ms / hw_stats.avg_simulate_ms.max(0.001);
    println!(
        "  RecommendedRamenTrainer / RamenHandwrittenTrainer"
    );
    println!(
        "  整局:     ×{:.2}  (HW {:.0} ms -> REC {:.0} ms)",
        total_ratio, hw_stats.avg_total_ms, rec_stats.avg_total_ms
    );
    println!(
        "  决策耗时: ×{:.2}  (HW {:.1} ms -> REC {:.1} ms)",
        decision_ratio, hw_stats.avg_decision_ms, rec_stats.avg_decision_ms
    );
    println!(
        "  局面生成: ×{:.2}  (HW {:.0} ms -> REC {:.0} ms)",
        simulate_ratio, hw_stats.avg_simulate_ms, rec_stats.avg_simulate_ms
    );

    println!("\n墙钟耗时: {:.1} s", wall.elapsed().as_secs_f64());

    Ok(())
}