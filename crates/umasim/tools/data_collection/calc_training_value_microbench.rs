//! `RamenGame::calc_training_value` 及其前置段的微基准（性能调优专用）
//!
//! 用 speed build + friendship 全 100 + 固定 turn / scenario_pt / 拉面 buff 的
//! 可复现场景，把"人头分配 → 卡 buff 聚合 → 训练数值计算"全链路拆成 4 段，
//! 分别测 ns/op，输出占比对照表。**不验证正确性、不覆盖边缘 case**——纯性能基线。
//!
//! 目的：手写策略 / MCTS rollout 中 `calc_training_value` 是 pprof Top5 热函数
//! （pprof 2.58% × MCTS N 局 × rollout 调用），但单条调用本身又嵌套多层
//! （卡 buff 聚合 + 拉面 buff 累乘 + 上下层约束）。单点 microbench 不能告诉
//! 优化者瓶颈在"buff 聚合"还是"拉面 buff 累乘"。本工具按段拆开，给出可独立
//! 对比的基线。
//!
//! 段定义（B/C/D 每轮都对所有 5 个训练位置各跑一遍，对齐
//! `LocalRamenTrainer::score_train_action` 的"一回合评估所有 train"语义）：
//!
//! - 段 A：`distribute_all`（reset + 按 PersonType 顺序遍历 + 单次 `distribute_person`）
//! - 段 B：5 train × `default_calc_training_buff`（卡 buff 聚合，含 `SupportCard::calc_training_effect`）
//! - 段 C：5 train × `calc_training_value`（含 `default_calc_training_value` 下层 + 拉面 buff 上层）
//! - 段 D：端到端一回合 = A + B + C
//! - 段 E：`RamenPolicy::score_train_action`（单 candidate 打分，含 buff+value+failure_rate+score 拆解；pprof 1.61%）
//! - 段 F：`LocalRamenTrainer::decide_train`（整回合打分：5 train + 修复路径；pprof 0.54% 含子项）
//! - 段 G：`calc_ramen_training_effect`（calc_training_value 内部嵌套的拉面 buff 累乘路径）
//!
//! 用法：
//! ```text
//! # 默认 1000 次 / 段，warmup 1000 次
//! cargo run --release --bin calc_training_value_microbench
//! ```
//!
//! 可选环境变量：
//! - `CT_MICROBENCH_RUNS`：每段测量次数（默认 1000）
//! - `CT_MICROBENCH_WARMUP`：warmup 次数（默认 1000）
//!
//! 与 `local_ramen_trainer.rs::tests::test_perf_top_functions` 的区别：后者只对
//! speed build turn=30 单点测速一次过，本工具按段分桶输出，更适合"逐段优化→逐段
//! 回归"的迭代节奏。
//!
//! 预设参数：
//! - 马娘：美浦波旁（102601）
//! - 卡组：speed build（速3耐1智1 + 友人 303054）
//! - friendship：deck + persons 全部 100（让 is_shining_at 在得意位置返回 true，
//!   即全程走"友情训练"路径，命中 calc_training_value 内 youqing buff 分支）
//! - 回合：固定 30（避开 0-1 边界、地区选择、第 1 年体力波动）
//! - scenario_pt：1500（PT tier 拉满，让 ramen pt_effect 取最高档）
//! - 地区：[5, 7, 9] = 中山-全 / 京都-耐根 / 小仓-智（中山覆盖所有 5 train）
//! - 当前吃面：`Some(5)`（中山-全，命中所有 5 train 的 region 路径）

use std::{env, hint::black_box, time::Instant};

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use umasim::{
    game::{
        CardTrainingEffect, Game, InheritInfo,
        ramen::{
            Operation, RamenAction, RamenGame, TrainingType,
            effects::calc_ramen_training_effect,
            policy::{RamenPolicy, RamenPolicyConfig}
        }
    },
    gamedata::init_global_with_config,
    trainer::LocalRamenTrainer,
    utils::{get_workspace_root, load_game_config}
};

/// 美浦波旁（speed build memory 基准）
const UMA: u32 = 102_601;
/// 标准 speed build（速3耐1智1 + 友人 303054）
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
/// 种马继承（与 sim_profiler / trainer_overhead 同源）
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};
/// 固定回合（避开边界 + 地区选择）
const TURN: i32 = 30;
/// 固定 scenario_pt（最高 tier，让 ramen pt_effect 拉满）
const SCENARIO_PT: i32 = 1500;
/// friendship 全 100（人手设置，让 is_shining_at 走真分支）
const FRIENDSHIP_FULL: i32 = 100;
/// train_level_count 设置（[12; 5] → train_level = 12/4+1 = 4，turn=30 玩家已练过几轮的典型值）
const TRAIN_LEVEL_COUNT_INIT: [i32; 5] = [12, 12, 12, 12, 12];
/// 当年已选地区拉面下标：[5, 7, 9] = 中山-全 / 京都-耐根 / 小仓-智
const SELECTED_REGIONS: [usize; 3] = [5, 7, 9];
/// 当前回合吃的面（中山-全，at_trains=[0..4] 全部命中，让 region buff 走完整路径）
const CURRENT_RAMEN: Option<usize> = Some(5);
/// 每轮每段要跑的训练位置数量（速耐力根智 = 5 个）
const TRAIN_NUM: usize = 5;

/// 一段 microbench 的统计结果
#[derive(Debug, Clone, Copy)]
struct BenchResult {
    /// 段名
    name: &'static str,
    /// 单轮最小总耗时（纳秒）
    min_total_ns: u128,
    /// 单轮平均总耗时（纳秒）
    mean_total_ns: f64,
    /// 每轮次数
    n: usize
}

impl BenchResult {
    fn min_ns_per_call(&self) -> f64 {
        self.min_total_ns as f64 / self.n as f64
    }
    /// `mean_total_ns` 字段已经按轮内每调用平均过均值，直接返回即「每次调用平均 ns」
    fn mean_ns_per_call(&self) -> f64 {
        self.mean_total_ns
    }
    fn calls_per_sec(&self) -> f64 {
        1e9 / self.mean_total_ns
    }
}

/// 构造一份已"冻结"到典型 turn=30 状态的 speed build game
///
/// 完成步骤：
/// 1. `newgame`：5 张支援卡 + 1 理事长（拉面布局）
/// 2. 友人卡 + 5 NPC + 记者（模拟 turn ≥ 12 时的人头构成）
/// 3. deck / persons 全部 friendship=100（走 is_shining 真分支）
/// 4. 设定 turn = 30、scenario_pt = 1500、train_level_count = [12; 5]（典型玩家进度）
/// 5. 设定拉面 buff：selected_regions = [5, 7, 9]、yearly 归档同样、current_ramen = Some(5)
///
/// 步骤 5 让 `calc_ramen_training_effect` 走完整两层：
/// - PT 常驻（PT tier 拉满）
/// - RMJ 常驻（y2 用 y1 结果，未设置走默认）
/// - 吃面基础（ramen_basic_effect[y2]）
/// - 地区效果（中山-全 at_trains=[0..4]，全部 train 命中）
///
/// is_shining=true → region/youqing 不归零。
///
/// 注：早期版本这里曾带 `CT_MICROBENCH_LOCKED=1` 对照实验 burn-in lock 路径——
/// `SupportCard::calc_training_effect` 改用 `CardTrainingEffect::from(&card_value)`
/// 作为起点后，lock 概念彻底退化为 NN feature 标记（`is_locked: bool` 字段保留），
/// 对照实验本身已无意义，相应 env var 与 burn-in 代码一并清理。
fn make_test_game() -> Result<RamenGame> {
    let mut game = RamenGame::newgame(UMA, &DECK, INHERIT)?;
    game.add_friend_and_npcs()?;
    game.add_reporter();
    for person in game.persons_mut() {
        person.friendship = FRIENDSHIP_FULL;
    }
    for card in &mut game.deck {
        card.friendship = FRIENDSHIP_FULL;
    }
    game.base.turn = TURN;
    game.ramen.scenario_pt = SCENARIO_PT;
    game.ramen.train_level_bonus = 0;
    game.base.train_level_count = TRAIN_LEVEL_COUNT_INIT;
    // 拉面 buff：让 calc_ramen_training_effect 走完整两层（吃面 + 地区）
    game.ramen.selected_regions = SELECTED_REGIONS;
    // yearly 归档是观测出口，但策略/搜索读 live；这里两份都设，行为对齐即可
    for year_idx in 0..3 {
        game.ramen.yearly_selected_regions[year_idx] = SELECTED_REGIONS;
    }
    game.ramen.current_ramen = CURRENT_RAMEN;
    Ok(game)
}

/// 跑一段 N 次的 microbench：先 warmup，后取 3 轮中 min / mean
///
/// 返回 `BenchResult`。闭包 `f` 是 FnMut，每次循环会重新构建状态（在前置
/// 闭包里手动 `reset_distribution` / 重新算 buffs 等）。
fn run_microbench<F: FnMut()>(name: &'static str, mut f: F, n: usize, warmup: usize) -> BenchResult {
    // Warmup：让分配器 / cache 稳定，与 local_ramen_trainer microbench 模板一致
    for _ in 0..warmup {
        black_box(f());
    }
    let mut min_total = u128::MAX;
    let mut mean_sum = 0.0f64;
    const ROUNDS: usize = 3;
    for round in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..n {
            black_box(f());
        }
        let total = start.elapsed().as_nanos();
        min_total = min_total.min(total);
        mean_sum += total as f64 / n as f64;
        println!(
            "  [{}] {} 轮 {}: total={} ns, mean={:.1} ns/iter",
            name, round + 1, ROUNDS, total, total as f64 / n as f64
        );
    }
    BenchResult {
        name,
        min_total_ns: min_total,
        mean_total_ns: mean_sum / ROUNDS as f64,
        n
    }
}

/// 对全部 5 个训练位置各算一次 buff，存到 `[CardTrainingEffect; 5]`
///
/// 与生产 `LocalRamenTrainer::score_train_action` 中循环 5 train 算 buff 的语义
/// 对齐；C 段与 D 段都需要这份缓存。
fn calc_buffs_for_all_trains(game: &RamenGame) -> [CardTrainingEffect; TRAIN_NUM] {
    std::array::from_fn(|i| game.default_calc_training_buff(i).expect("calc_training_buff"))
}

/// 构造与生产 `LocalRamenTrainer` 一致的 train 阶段候选动作列表：
/// 5 个训练位 + Race + Rest + NormalOuting + FriendOuting + Clinic。
///
/// 与 d10872a microbench（local_ramen_trainer.rs:2697）构造的"5 train only"相比，
/// 这里加 Rest/Race 等非训练候选覆盖真实 `decide_train` 内部对各类 `Operation`
/// 的分支（match 在 `score_train_action` 里 dispatch）。`current_ramen = Some(5)`
/// 触发「已吃面→5 train 必须存在」分支，但 5 train 都齐全所以不会触发 fail。
fn make_train_actions_with_extras() -> Vec<RamenAction> {
    let mut actions = Vec::with_capacity(7);
    for t in [
        TrainingType::Speed,
        TrainingType::Stamina,
        TrainingType::Power,
        TrainingType::Guts,
        TrainingType::Wisdom,
    ] {
        actions.push(RamenAction::new(Operation::Train(t)));
    }
    actions.push(RamenAction::new(Operation::Race));
    actions.push(RamenAction::new(Operation::Rest));
    actions
}

/// 单个 Train(Speed) action（段 E 专用，模拟"单 candidate 单 train 打分"）
fn make_single_train_action() -> RamenAction {
    RamenAction::new(Operation::Train(TrainingType::Speed))
}

/// 7 段 microbench：分别构造 game / trainer 实例以避免 FnMut 跨段借用冲突
fn bench_all(n: usize, warmup: usize) -> Result<[BenchResult; 7]> {
    // ----- 段 A: distribute_all（含 reset_distribution 前置；一回合 1 次） -----
    let mut game_a = make_test_game()?;
    let mut rng_a = StdRng::seed_from_u64(42);
    let a = run_microbench("A.distribute_all", move || {
        game_a.reset_distribution();
        game_a.distribute_all(&mut rng_a).unwrap();
    }, n, warmup);

    // ----- 段 B: 5 train × calc_training_buff（每轮算一遍全部 5 train 的 buff） -----
    let mut game_b = make_test_game()?;
    // 先填一次 distribution，让 buff 路径走"非空"分支（与 turn=30 真实状态对齐）
    {
        let mut rng_b = StdRng::seed_from_u64(42);
        game_b.distribute_all(&mut rng_b)?;
    }
    let b = run_microbench("B.calc_training_buff x5", move || {
        // 每轮算 5 train 全套 buff，存到 local 数组防止编译器消除
        let local: [CardTrainingEffect; TRAIN_NUM] = std::array::from_fn(|i| {
            black_box(game_b.default_calc_training_buff(i).unwrap())
        });
        black_box(local);
    }, n, warmup);

    // ----- 段 C: 5 train × calc_training_value（固定 5 份 buff，每轮做 5 train value） -----
    let mut game_c = make_test_game()?;
    {
        let mut rng_c = StdRng::seed_from_u64(42);
        game_c.distribute_all(&mut rng_c)?;
    }
    // 预先算 5 份 buff：测的是「value 主路径」而不是「buff 计算+value 混合」
    let c_buffs = calc_buffs_for_all_trains(&game_c);
    let c = run_microbench("C.calc_training_value x5", move || {
        // 每轮 5 train 各算一次 value（用固定的 5 份 buff，避免与 B 段重复）
        for t in 0..TRAIN_NUM {
            let _ = black_box(game_c.calc_training_value(&c_buffs[t], t).unwrap());
        }
    }, n, warmup);

    // ----- 段 D: 端到端一回合（reset + distribute_all + 5 train buff + 5 train value） -----
    let mut game_d = make_test_game()?;
    let mut rng_d = StdRng::seed_from_u64(42);
    let d = run_microbench("D.端到端一回合(5train)", move || {
        game_d.reset_distribution();
        game_d.distribute_all(&mut rng_d).unwrap();
        let d_buffs: [CardTrainingEffect; TRAIN_NUM] = std::array::from_fn(|i| {
            game_d.default_calc_training_buff(i).unwrap()
        });
        for t in 0..TRAIN_NUM {
            let _ = black_box(game_d.calc_training_value(&d_buffs[t], t).unwrap());
        }
    }, n, warmup);

    // ----- 段 E: RamenPolicy::score_train_action 单 candidate 打分（含 buff+value+score 拆解） -----
    let mut game_e = make_test_game()?;
    {
        let mut rng_e = StdRng::seed_from_u64(42);
        game_e.distribute_all(&mut rng_e)?;
    }
    let policy_e = RamenPolicy::new(RamenPolicyConfig::default());
    let action_e = make_single_train_action();
    let e = run_microbench("E.score_train_action x1", move || {
        // 每轮对同一个 Train(Speed) candidate 调一次打分（覆盖 buff+value+fail_rate+score）
        let out = black_box(policy_e.score_train_action(&game_e, &action_e).unwrap());
        black_box(out.score);
    }, n, warmup);

    // ----- 段 F: LocalRamenTrainer::decide_train 整回合打分（5 train + Rest/Race 等） -----
    let mut game_f = make_test_game()?;
    {
        let mut rng_f = StdRng::seed_from_u64(42);
        game_f.distribute_all(&mut rng_f)?;
    }
    let trainer_f = LocalRamenTrainer::new();
    let actions_f = make_train_actions_with_extras();
    let f = run_microbench("F.decide_train(整回合)", move || {
        // 每轮对全候选打分一次（含 5 train + Rest + Race 的 score + 选择）
        let (pick, _out) = black_box(trainer_f.decide_train(&game_f, &actions_f).unwrap());
        black_box(pick);
    }, n, warmup);

    // ----- 段 G: calc_ramen_training_effect 单 train（calc_training_value 内部嵌套的拉面 buff 路径） -----
    let mut game_g = make_test_game()?;
    {
        let mut rng_g = StdRng::seed_from_u64(42);
        game_g.distribute_all(&mut rng_g)?;
    }
    let g = run_microbench("G.calc_ramen_training_effect x1", move || {
        // 每轮对一个 train 算一次拉面 buff（不进 calc_training_value，独立测速）
        let effect = black_box(calc_ramen_training_effect(&game_g, 0, true));
        black_box(effect.xunlian);
    }, n, warmup);

    Ok([a, b, c, d, e, f, g])
}

/// 打印 7 段汇总表 + 段 D 时间拆解 + 段 F 时间拆解
///
/// 表头：[name | min ns/iter | mean ns/iter | iter/s]
/// - 段 A 的 iter = 1 次 distribute_all
/// - 段 B/C/D 的 iter = 1 次"训练评估循环"（5 train 各跑一遍 buff 或 value）
/// - 段 E/G 的 iter = 1 次单 candidate / 1 次单 train 操作（不循环）
/// - 段 F 的 iter = 1 次整回合打分（5 train + Rest + Race）
///
/// 拆解 D 是"段 D 时间由哪几部分贡献"的直观展示：
/// - 各段占 D 百分比（A/B/C 单独测得的耗时占 D 端到端时间的份额）
/// - Σ A+B+C vs D：理论上 D = A + B + C（B/C 都算 5 train），实际会略小，
///   因为 D 端到端让 LLVM 把内层函数 inline / cache 共享更充分，单段分开测
///   则各自承担闭包边界 + black_box 序列化开销
///
/// 拆解 F：策略层（不在 D 内）—— 段 F 包含 7 个 candidate 的 score_train_action
/// + fix-up 分支 + choose，所以 F ≈ 5×E（仅 score 单 candidate 维度）。
fn print_summary(r: [BenchResult; 7]) {
    let [a, b, c, d, e, f, g] = r;
    println!("\n================ microbench 汇总 ================");
    println!(
        "{:<28} {:>13} {:>13} {:>10}",
        "段", "min ns/iter", "mean ns/iter", "iter/s"
    );
    println!("{}", "-".repeat(68));
    for x in r.iter() {
        println!(
            "{:<28} {:>13.1} {:>13.1} {:>10.0}",
            x.name,
            x.min_ns_per_call(),
            x.mean_ns_per_call(),
            x.calls_per_sec()
        );
    }
    println!();
    let d_mean = d.mean_ns_per_call();
    let a_mean = a.mean_ns_per_call();
    let b_mean = b.mean_ns_per_call();
    let c_mean = c.mean_ns_per_call();
    let e_mean = e.mean_ns_per_call();
    let f_mean = f.mean_ns_per_call();
    let g_mean = g.mean_ns_per_call();
    let a_pct = a_mean / d_mean * 100.0;
    let b_pct = b_mean / d_mean * 100.0;
    let c_pct = c_mean / d_mean * 100.0;
    // 分桶测时 B/C 单独跑（前一次 distribute_all 已让 cache 热），
    // D 实测要经历完整的 cache-miss 链路，所以 D 通常 ≥ Σ A+B+C。
    let sum_mean = a_mean + b_mean + c_mean;
    let overhead_d = d_mean - sum_mean;
    let overhead_d_pct = overhead_d / d_mean * 100.0;
    println!("================ 段 D 时间拆解（端到端一回合）================");
    println!("  A.distribute_all           {:>7.1} ns  (D 的 {:>5.1}%)", a_mean, a_pct);
    println!("  B.calc_training_buff ×5    {:>7.1} ns  (D 的 {:>5.1}%)", b_mean, b_pct);
    println!("  C.calc_training_value ×5   {:>7.1} ns  (D 的 {:>5.1}%)", c_mean, c_pct);
    println!();
    println!("  Σ A+B+C（分桶独立测得）     {:>7.1} ns", sum_mean);
    println!("  D 实测（端到端一回合）       {:>7.1} ns", d_mean);
    println!(
        "  D − Σ（cache miss / 边界开销）{:>+7.1} ns  ({:+.1}% of D)",
        overhead_d, overhead_d_pct
    );
    println!("================================================\n");

    println!("================ 段 F 时间拆解（策略层整回合打分）================");
    println!("  E.score_train_action x1     {:>7.1} ns  (单 candidate)", e_mean);
    println!("  F.decide_train(7 候选)      {:>7.1} ns", f_mean);
    // F ≈ 7×E + fixup（约 5 train + Rest/Race 等非 train 的 cost），比纯 5×E 多
    // 出的是 choose / argmax / fix-up 分支 / decide_train 后续处理（reserve_penalty
    // / dynamic_status_adjustment 在 default config 下走早返回所以近似 0）
    let ideal_f = 5.0 * e_mean; // 5 train candidates，不计 Rest/Race/fixup
    println!(
        "  5×E（仅 5 train candidate 理想值）{:>7.1} ns   F / (5×E) = {:.2}",
        ideal_f, f_mean / ideal_f
    );
    println!("  G.calc_ramen_training_effect x1 {:>7.1} ns（calc_value 内部嵌套的拉面 buff 路径）", g_mean);
    println!("================================================\n");
}

fn main() -> Result<()> {
    let runs: usize = env::var("CT_MICROBENCH_RUNS")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000);
    let warmup: usize = env::var("CT_MICROBENCH_WARMUP")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000);

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    init_global_with_config(&load_game_config()?)?;

    println!("calc_training_value microbench: runs={runs}/段 warmup={warmup}");
    println!(
        "preset: speed build, friendship={FRIENDSHIP_FULL}, turn={TURN}, scenario_pt={SCENARIO_PT}"
    );
    println!(
        "ramen buff: selected_regions={SELECTED_REGIONS:?} current_ramen={:?}",
        CURRENT_RAMEN
    );
    println!(
        "uma={UMA} deck={DECK:?} inherit=blue{}/extra{}",
        INHERIT.blue_count.len(), INHERIT.extra_count.len()
    );
    println!();

    let wall = Instant::now();
    let results = bench_all(runs, warmup)?;
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;

    print_summary(results);
    println!("墙钟总耗时: {:.1} ms（{} 段 × {} iter + warmup × {} 段 × {} iter）",
        wall_ms, 7, runs, 7, warmup);

    Ok(())
}
