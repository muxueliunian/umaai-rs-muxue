//! 手写策略参数扫描（speed / stamina 两个主流配卡）
//!
//! 目标：在 `RecommendedRamenTrainer` 上做同 seed 配对的参数扫描，
//! 找回上限修复后失效的 `status_gap_strength` / `status_overflow_strength` 等最优值。
//!
//! 为什么要重扫：这些值是在五维上限有缺陷时（速度被压在 2958、从未摸到上限）
//! 调出来的，「近上限衰减」当时根本没机会生效。上限修复后速度顶格 3337，
//! 该机制第一次真正启动，旧最优值不再可信。
//!
//! 跑法：`HW_TUNE_RUNS=200 cargo run --release --bin handwritten_tuning`
//!
//! 环境变量：
//! - `HW_TUNE_RUNS`   每个 build 的局数（缺省 200）
//! - `HW_TUNE_SEED`   基础种子（缺省 61444，与 bench_config 一致）
//! - `HW_TUNE_GRID`   扫描哪一组网格（缺省 `gap_over`）

use std::env;

use anyhow::Result;
use rayon::prelude::*;
use umasim::{
    bench::{self, CardPickOpts, load_player_builds},
    game::InheritInfo,
    gamedata::{RamenRegionStrategy, init_global_with_config},
    trainer::{LoggingTrainer, RecommendedRamenTrainer},
    utils::{get_workspace_root, load_game_config}
};

/// 一个待测参数组合
#[derive(Clone, Debug)]
struct Variant {
    label: String,
    pt_rates: [f32; 3],
    gap: f32,
    over: f32,
    sacrifice: f32,
    window: f32,
    reserve: f32,
    early_bond: f32,
    hint: f32,
    weakboost: f32,
    region_weak: f32,
    covered: bool
}

impl Variant {
    /// preset 现值（对齐 `RecommendedRamenTrainer::new()`）。
    ///
    /// 每一项都能用 `HW_B_<名>` 环境变量覆盖，用于坐标下降：
    /// 上一轮找到的最优值写进环境变量，下一轮就在新基线上扫其余参数。
    fn baseline() -> Self {
        fn ev(key: &str, default: f32) -> f32 {
            env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        Self {
            label: "baseline".into(),
            pt_rates: [ev("HW_B_PT1", 16.0), ev("HW_B_PT2", 64.0), ev("HW_B_PT3", 64.0)],
            gap: ev("HW_B_GAP", 0.5),
            over: ev("HW_B_OVER", 0.5),
            sacrifice: ev("HW_B_SAC", 140.0),
            window: ev("HW_B_WINDOW", 0.10),
            reserve: ev("HW_B_RESERVE", 40.0),
            // LocalRamenConfig::default() 现值；preset 未覆盖这两项
            early_bond: ev("HW_B_BOND", 8.0),
            hint: ev("HW_B_HINT", 6.0),
            weakboost: ev("HW_B_WEAKBOOST", 0.0),
            region_weak: ev("HW_B_REGIONWEAK", 0.0),
            covered: ev("HW_B_COVERED", 1.0) > 0.5
        }
    }

    fn build(&self) -> RecommendedRamenTrainer {
        RecommendedRamenTrainer::with_experiment_overrides(
            self.pt_rates,
            self.gap,
            self.over,
            self.sacrifice,
            self.window,
            self.reserve,
            self.early_bond,
            self.hint,
            self.weakboost,
            self.region_weak,
            self.covered
        )
    }
}

fn main() -> Result<()> {
    let runs: u64 = env::var("HW_TUNE_RUNS").unwrap_or_else(|_| "200".into()).parse()?;
    let seed: u64 = env::var("HW_TUNE_SEED").unwrap_or_else(|_| "61444".into()).parse()?;
    let grid = env::var("HW_TUNE_GRID").unwrap_or_else(|_| "gap_over".into());

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(&workspace_root)?;
    let mut game_config = load_game_config()?;
    // 与 bench_base 对齐：基准测的是策略决策，Y3 地区必须交回策略枚举
    game_config.ramen_region_strategy = RamenRegionStrategy::All;
    game_config.ramen_region_fixed = None;
    init_global_with_config(&game_config)?;

    // 只取用户点名的两个主流配卡：3速1耐1智1友 / 2速2耐1智1友
    let all_builds = load_player_builds()?;
    let targets = ["speed", "stamina"];
    let builds: Vec<_> =
        all_builds.into_iter().filter(|b| targets.contains(&b.name().as_str())).collect();
    anyhow::ensure!(builds.len() == 2, "未找到 speed / stamina build，实得 {}", builds.len());

    let pick = CardPickOpts::default();
    let friend: u32 = 303054;
    let decks: Vec<[u32; 6]> =
        builds.iter().map(|b| b.make_deck(&pick, friend)).collect::<Result<_>>()?;
    let inherit = InheritInfo { blue_count: [15, 0, 0, 0, 3], extra_count: [10, 10, 20, 20, 20, 40] };

    let variants = make_grid(&grid);
    println!(
        "handwritten_tuning: grid={grid} variants={} builds={:?} runs={runs}/build seed={seed}",
        variants.len(),
        targets
    );

    // 每个 variant × build × run 的分数；同 (build, run_idx) 即同 seed，可直接配对
    let mut table: Vec<(String, Vec<Vec<i32>>)> = Vec::with_capacity(variants.len());
    for v in &variants {
        let per_build: Vec<Vec<i32>> = (0..builds.len())
            .map(|bi| {
                (0..runs)
                    .into_par_iter()
                    .map(|run_idx| {
                        let mut trainer = LoggingTrainer::new(v.build(), run_idx);
                        trainer.set_logging(false);
                        bench::run_seeded(102601, &decks[bi], &inherit, seed, run_idx, &trainer)
                            .map(|o| o.score)
                            .unwrap_or(0)
                    })
                    .collect()
            })
            .collect();
        let total: f64 = per_build.iter().flatten().map(|&s| s as f64).sum();
        let n: usize = per_build.iter().map(|v| v.len()).sum();
        println!("  {:<26} 两 build 合并均分 {:.0}", v.label, total / n as f64);
        table.push((v.label.clone(), per_build));
    }

    // 配对报表：以第一个 variant（baseline）为对照
    let (base_label, base) = table[0].clone();
    println!("\n=== 相对 {base_label} 的配对差（同 build 同 seed）===");
    println!("{:<26}{:>10}{:>10}{:>10}{:>8}{:>8}", "variant", "speed", "stamina", "合并", "SE", "t");
    let mut ranked: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    for (label, cur) in table.iter() {
        let mut diffs_all: Vec<f64> = Vec::new();
        let mut per_build_mean = Vec::new();
        for bi in 0..builds.len() {
            let d: Vec<f64> =
                cur[bi].iter().zip(&base[bi]).map(|(&a, &b)| (a - b) as f64).collect();
            per_build_mean.push(d.iter().sum::<f64>() / d.len() as f64);
            diffs_all.extend(d);
        }
        let n = diffs_all.len() as f64;
        let m = diffs_all.iter().sum::<f64>() / n;
        let var = diffs_all.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        let t = if se > 0.0 { m / se } else { 0.0 };
        println!(
            "{:<26}{:>+10.0}{:>+10.0}{:>+10.0}{:>8.0}{:>8.2}",
            label, per_build_mean[0], per_build_mean[1], m, se, t
        );
        ranked.push((label.clone(), per_build_mean[0], per_build_mean[1], m, t));
    }

    ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("\n=== 合并配对差 Top 5 ===");
    for (label, s, st, m, t) in ranked.iter().take(5) {
        println!("  {label:<26} speed{s:>+8.0}  stamina{st:>+8.0}  合并{m:>+8.0}  t={t:>+6.2}");
    }
    Ok(())
}

fn make_grid(name: &str) -> Vec<Variant> {
    let mut out = vec![Variant::baseline()];
    match name {
        // 上限修复后重扫短板追赶 / 近上限衰减
        "gap_over" => {
            for gap in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0] {
                for over in [0.0f32, 0.5, 1.0, 2.0] {
                    if (gap - 0.5).abs() < 1e-6 && (over - 0.5).abs() < 1e-6 {
                        continue; // 与 baseline 重复
                    }
                    let mut v = Variant::baseline();
                    v.label = format!("gap{gap:.1}-over{over:.1}");
                    v.gap = gap;
                    v.over = over;
                    out.push(v);
                }
            }
        }
        // 单参数敏感性：每个旋钮独立扫，先找杠杆最大的那个
        "sens" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            for x in [8.0f32, 32.0, 64.0, 128.0] {
                push(format!("ptY1={x:.0}"), &|v: &mut Variant| v.pt_rates[0] = x);
            }
            for x in [16.0f32, 32.0, 128.0, 256.0] {
                push(format!("ptY2={x:.0}"), &|v: &mut Variant| v.pt_rates[1] = x);
            }
            for x in [16.0f32, 32.0, 128.0, 256.0] {
                push(format!("ptY3={x:.0}"), &|v: &mut Variant| v.pt_rates[2] = x);
            }
            for x in [0.0f32, 70.0, 280.0, 560.0] {
                push(format!("sacrifice={x:.0}"), &|v: &mut Variant| v.sacrifice = x);
            }
            for x in [0.0f32, 0.05, 0.2, 0.4, 0.8] {
                push(format!("window={x:.2}"), &|v: &mut Variant| v.window = x);
            }
            for x in [0.0f32, 20.0, 80.0, 160.0] {
                push(format!("reserve={x:.0}"), &|v: &mut Variant| v.reserve = x);
            }
            for x in [0.0f32, 4.0, 16.0, 32.0] {
                push(format!("earlybond={x:.0}"), &|v: &mut Variant| v.early_bond = x);
            }
            for x in [0.0f32, 3.0, 12.0, 24.0] {
                push(format!("hint={x:.0}"), &|v: &mut Variant| v.hint = x);
            }
            for x in [1.0f32, 2.0, 5.0, 10.0] {
                push(format!("weakboost={x:.1}"), &|v: &mut Variant| v.weakboost = x);
            }
            for x in [0.5f32, 1.0, 2.0] {
                push(format!("regionweak={x:.1}"), &|v: &mut Variant| v.region_weak = x);
            }
            push("covered=false".into(), &|v: &mut Variant| v.covered = false);
            // gap 单调向上，补几个更大的
            for x in [4.0f32, 6.0, 10.0] {
                push(format!("gap={x:.0}"), &|v: &mut Variant| v.gap = x);
            }
        }
        // 找 sacrifice / ptY3 的拐点
        "knee" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            for x in [280.0f32, 560.0, 1000.0, 2000.0, 4000.0, 10000.0, 100000.0] {
                push(format!("sac={x:.0}"), &|v: &mut Variant| v.sacrifice = x);
            }
            for x in [0.0f32, 2.0, 4.0, 8.0, 16.0, 24.0, 48.0] {
                push(format!("ptY3={x:.0}"), &|v: &mut Variant| v.pt_rates[2] = x);
            }
            for x in [0.0f32, 4.0, 8.0, 16.0] {
                push(format!("ptY1={x:.0}"), &|v: &mut Variant| v.pt_rates[0] = x);
            }
            for x in [64.0f32, 96.0, 160.0] {
                push(format!("ptY2={x:.0}"), &|v: &mut Variant| v.pt_rates[1] = x);
            }
        }
        // 把找到的最佳值叠起来，看是否可加
        "combo" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            let sac = env::var("HW_SAC").unwrap_or_else(|_| "2000".into()).parse::<f32>().unwrap();
            let pt3 = env::var("HW_PT3").unwrap_or_else(|_| "8".into()).parse::<f32>().unwrap();
            push(format!("sac{sac:.0}"), &|v: &mut Variant| v.sacrifice = sac);
            push(format!("pt3={pt3:.0}"), &|v: &mut Variant| v.pt_rates[2] = pt3);
            push(format!("sac+pt3"), &|v: &mut Variant| {
                v.sacrifice = sac;
                v.pt_rates[2] = pt3;
            });
            push(format!("sac+pt3+gap4"), &|v: &mut Variant| {
                v.sacrifice = sac;
                v.pt_rates[2] = pt3;
                v.gap = 4.0;
            });
            push(format!("sac+pt3+gap4+nocov"), &|v: &mut Variant| {
                v.sacrifice = sac;
                v.pt_rates[2] = pt3;
                v.gap = 4.0;
                v.covered = false;
            });
            push(format!("sac+pt3+gap4+wb10"), &|v: &mut Variant| {
                v.sacrifice = sac;
                v.pt_rates[2] = pt3;
                v.gap = 4.0;
                v.weakboost = 10.0;
            });
            push(format!("sac+pt3+gap4+res160"), &|v: &mut Variant| {
                v.sacrifice = sac;
                v.pt_rates[2] = pt3;
                v.gap = 4.0;
                v.reserve = 160.0;
            });
        }
        // sac 峰值细扫（4000~50000 之间找顶）
        "sacfine" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            for x in [4000.0f32, 6000.0, 8000.0, 10000.0, 14000.0, 20000.0, 30000.0, 50000.0] {
                push(format!("sac={x:.0}"), &|v: &mut Variant| v.sacrifice = x);
            }
        }
        // gap 大范围（round2 显示到 10 还在涨）
        "gapbig" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            for x in [4.0f32, 10.0, 20.0, 40.0, 80.0, 160.0] {
                push(format!("gap={x:.0}"), &|v: &mut Variant| v.gap = x);
            }
            for x in [0.0f32, 1.0, 2.0, 4.0] {
                push(format!("over={x:.1}"), &|v: &mut Variant| v.over = x);
            }
            for x in [24.0f32, 48.0, 96.0] {
                push(format!("hint={x:.0}"), &|v: &mut Variant| v.hint = x);
            }
            for x in [60.0f32, 80.0, 120.0] {
                push(format!("reserve={x:.0}"), &|v: &mut Variant| v.reserve = x);
            }
        }
        // 累加验证：单参数最优是否可加
        "stack" => {
            let mut push = |label: String, f: &dyn Fn(&mut Variant)| {
                let mut v = Variant::baseline();
                v.label = label;
                f(&mut v);
                out.push(v);
            };
            let g = env::var("HW_GAP").unwrap_or_else(|_| "10".into()).parse::<f32>().unwrap();
            push("+gap".into(), &|v: &mut Variant| v.gap = g);
            push("+gap+hint".into(), &|v: &mut Variant| {
                v.gap = g;
                v.hint = 24.0;
            });
            push("+gap+hint+res".into(), &|v: &mut Variant| {
                v.gap = g;
                v.hint = 24.0;
                v.reserve = 80.0;
            });
            push("+gap+hint+res+wb10".into(), &|v: &mut Variant| {
                v.gap = g;
                v.hint = 24.0;
                v.reserve = 80.0;
                v.weakboost = 10.0;
            });
            push("ALL+nocov(speed偏向)".into(), &|v: &mut Variant| {
                v.gap = g;
                v.hint = 24.0;
                v.reserve = 80.0;
                v.weakboost = 10.0;
                v.covered = false;
            });
            push("ALL+ptY2=16(speed偏向)".into(), &|v: &mut Variant| {
                v.gap = g;
                v.hint = 24.0;
                v.reserve = 80.0;
                v.weakboost = 10.0;
                v.pt_rates[1] = 16.0;
            });
        }
        other => panic!("未知 grid: {other}")
    }
    out
}
