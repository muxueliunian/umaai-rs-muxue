//! 1000 组配对测试：现有手写基准与带保护上限的本地修正。

use std::{cmp::Ordering, fs};

use anyhow::{Result, ensure};
use umasim::{
    bench::{self, GameOutcome},
    game::{InheritInfo, ramen::rules::calc_ramen_pt_gain},
    gamedata::init_global_with_config,
    output::decision_log::{DecisionLog, DecisionLogRow},
    trainer::{LocalRamenTrainer, LoggingTrainer, RamenHandwrittenTrainer},
    utils::{get_workspace_root, load_game_config}
};

const RUNS: usize = 1000;
const BASE_SEED: u64 = 61444;
const UMA: u32 = 102601;
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [10, 10, 20, 20, 20, 40]
};

fn fill_ramen_metrics(outcome: &mut GameOutcome, rows: &[DecisionLogRow]) -> Result<()> {
    let mut yearly_eat_count = [0_i32; 3];
    let mut scenario_pt = 0;

    for row in rows {
        if row.stage != "RamenSelect" || !row.action_desc.starts_with("吃面/") {
            continue;
        }
        let year = if row.turn < 24 {
            0
        } else if row.turn < 48 {
            1
        } else {
            2
        };
        scenario_pt += calc_ramen_pt_gain(year, yearly_eat_count[year])?;
        yearly_eat_count[year] += 1;
    }

    let eat_count = yearly_eat_count.iter().sum();
    ensure!(eat_count > 0, "seed={} 无吃面", outcome.seed);
    outcome.scenario_pt = scenario_pt;
    outcome.eat_count = eat_count;
    Ok(())
}

fn compare_scores(a: i32, b: i32) -> String {
    match a.cmp(&b) {
        Ordering::Less => format!("A比B少{}分", b - a),
        Ordering::Greater => format!("B比A少{}分", a - b),
        Ordering::Equal => "A与B同分".into()
    }
}

fn print_summary(name: &str, outcomes: &[GameOutcome]) {
    let scores = outcomes.iter().map(|outcome| outcome.score as f64).collect::<Vec<_>>();
    let stats = bench::summarize(&scores);
    println!(
        "RESULT {name}: 局数={} 平均评分={:.0} 中位评分={:.0} 最低评分={:.0} 最高评分={:.0}",
        outcomes.len(),
        stats.mean,
        stats.median,
        stats.min,
        stats.max
    );
}

fn clean_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn main() -> Result<()> {
    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(workspace_root)?;
    init_global_with_config(&load_game_config()?)?;

    println!("A=master 现有手写基准；B=现有基准+受限本地修正（最多牺牲120点基础训练分）。");
    println!("开始 A/B：每套策略 {RUNS} 局，基础种子 {BASE_SEED}，局号 0..{}", RUNS - 1);

    let mut outcomes_a = Vec::with_capacity(RUNS);
    let mut outcomes_b = Vec::with_capacity(RUNS);
    let mut records_a = Vec::new();
    let mut records_b = Vec::new();
    let mut divergences = String::from(
        "run_idx\tfinal_comparison\tturn\tstage\tA_action\tB_action\tA_breakdown\tB_breakdown\n"
    );

    for run_idx in 0..RUNS as u64 {
        let trainer_a = LoggingTrainer::new(RamenHandwrittenTrainer::new(), run_idx);
        let trainer_b = LoggingTrainer::new(LocalRamenTrainer::new(), run_idx);

        let mut outcome_a = bench::run_seeded(UMA, &DECK, &INHERIT, BASE_SEED, run_idx, &trainer_a)?;
        let mut outcome_b = bench::run_seeded(UMA, &DECK, &INHERIT, BASE_SEED, run_idx, &trainer_b)?;
        let log_a = trainer_a.take_records();
        let log_b = trainer_b.take_records();

        fill_ramen_metrics(&mut outcome_a, &log_a.rows)?;
        fill_ramen_metrics(&mut outcome_b, &log_b.rows)?;
        let comparison = compare_scores(outcome_a.score, outcome_b.score);

        if let Some((row_a, row_b)) = log_a.rows.iter().zip(&log_b.rows).find(|(row_a, row_b)| {
            row_a.turn != row_b.turn
                || row_a.stage != row_b.stage
                || row_a.action_desc != row_b.action_desc
        }) {
            divergences.push_str(&format!(
                "{run_idx}\t{comparison}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                row_a.turn,
                clean_tsv_field(&row_a.stage),
                clean_tsv_field(&row_a.action_desc),
                clean_tsv_field(&row_b.action_desc),
                clean_tsv_field(row_a.score_breakdown.as_deref().unwrap_or("")),
                clean_tsv_field(row_b.score_breakdown.as_deref().unwrap_or(""))
            ));
        } else {
            divergences.push_str(&format!(
                "{run_idx}\t{comparison}\t-\t无分歧\t-\t-\t-\t-\n"
            ));
        }

        println!(
            "run={run_idx} | A[评分={}] | B[评分={}] | {comparison}",
            outcome_a.score, outcome_b.score
        );
        records_a.extend(log_a.rows);
        records_b.extend(log_b.rows);
        outcomes_a.push(outcome_a);
        outcomes_b.push(outcome_b);
    }

    fs::create_dir_all("logs")?;
    DecisionLog { rows: records_a }
        .save_to(std::path::Path::new("logs/A_existing_decisions.csv"))?;
    DecisionLog { rows: records_b }.save_to(std::path::Path::new("logs/B_local_decisions.csv"))?;
    fs::write("logs/first_divergence.tsv", divergences)?;

    print_summary("A(master 现有手写基准)", &outcomes_a);
    print_summary("B(现有基准+受限本地修正)", &outcomes_b);

    let deltas = outcomes_a
        .iter()
        .zip(&outcomes_b)
        .map(|(a, b)| b.score - a.score)
        .collect::<Vec<_>>();
    let wins_a = deltas.iter().filter(|&&delta| delta < 0).count();
    let wins_b = deltas.iter().filter(|&&delta| delta > 0).count();
    println!("PAIRED B胜={wins_b}局 A胜={wins_a}局 同分={}局", RUNS - wins_a - wins_b);

    let mean_delta = deltas.iter().sum::<i32>() as f64 / RUNS as f64;
    println!("DELTA 平均评分比较：{}", compare_scores(0, mean_delta.round() as i32));
    Ok(())
}
