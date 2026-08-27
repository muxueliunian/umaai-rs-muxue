//! 单局拉面模拟的 CPU profile 采集工具
//!
//! 跑 N 局 `RecommendedRamenTrainer` 拉面整局（speed build, base_seed=61444），
//! 用 [`pprof`] 全程采样，输出 Google pprof protobuf 到 `logs/profile/<name>.pb.gz`，
//! 同时把 top 函数打到 stdout。`pprof-rs` 1kHz 采样，足够精度又不显著干扰测量。
//!
//! 用法：
//! ```text
//! cargo run --release --bin sim_profiler
//! go tool pprof -top logs/profile/baseline.pb.gz
//! go tool pprof -tree logs/profile/baseline.pb.gz
//! ```
//!
//! 可选环境变量：
//! - `SIM_PROFILER_RUNS`：跑批数（默认 50）
//! - `SIM_PROFILER_FREQ`：采样频率 Hz（默认 1000）
//! - `SIM_PROFILER_LABEL`：输出文件标签（默认 "baseline"）
//!
//! 目的：定位"手写逻辑下整局模拟"的热点函数，作为后续优化（吃面联动预演/打分
//! 流水线/数据结构缓存等）的输入。当前 rollout 已切到 REC trainer，搜索掉的
//! 分由这部分热点决定修复优先级。

use std::{env, path::PathBuf};

use anyhow::Result;
use pprof::{ProfilerGuardBuilder, protos::Message};
use umasim::{
    bench, game::InheritInfo, gamedata::init_global_with_config, trainer::{
        LoggingTrainer, RecommendedRamenTrainer
    }, utils::{get_workspace_root, load_game_config}
};

/// 美浦波旁 + speed build（memory 中"玩家手动基准"使用的同一组合）
const UMA: u32 = 102_601;
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};

fn main() -> Result<()> {
    let runs: u64 = env::var("SIM_PROFILER_RUNS").unwrap_or_else(|_| "50".into()).parse()?;
    let freq: i32 = env::var("SIM_PROFILER_FREQ").unwrap_or_else(|_| "1000".into()).parse()?;
    let label: String = env::var("SIM_PROFILER_LABEL").unwrap_or_else(|_| "baseline".into());

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(workspace_root.clone())?;
    init_global_with_config(&load_game_config()?)?;

    println!("sim_profiler: runs={runs} freq={freq}Hz label={label} build=speed");

    // 启动 CPU profiler（ITIMER_PROF）。blocklist 排除 libc/pthread 等系统库
    // ——这些是符号化/采样噪声来源，不在我们优化目标里。
    let guard = ProfilerGuardBuilder::default()
        .frequency(freq)
        .blocklist(&["libc", "pthread", "vdso", "rayon"])
        .build()?;

    let mut total_score = 0i64;
    for run_idx in 0..runs {
        let mut trainer = LoggingTrainer::new(RecommendedRamenTrainer::new(), run_idx);
        trainer.set_logging(false); // 关掉 logging 让采样聚焦在游戏/策略层
        let outcome = bench::run_seeded(UMA, &DECK, &INHERIT, 61444, run_idx, &trainer)?;
        total_score += outcome.score as i64;
        println!(
            "  #{:02} seed={} 五维={:?} skill_pt={} scenario_pt={:?} rmj={}/3 elapsed={:.1}ms",
            run_idx + 1,
            outcome.seed,
            outcome.five_status,
            outcome.skill_pt,
            outcome.yearly_scenario_pt,
            outcome.rmj_ok,
            outcome.elapsed_ms
        );
    }
    println!("完成 {runs} 局，平均分 {:.0}", total_score as f64 / runs as f64);

    // 构造 report
    let report = guard.report().build()?;
    let samples = report.data.len();
    let total_ticks: isize = report.data.values().sum();
    println!("\n========== Profile 概览 ==========");
    println!(
        "采样栈数: {} | 总 tick: {} | 频率 {}Hz | 实际 CPU 时间 ≈ {:.2}s",
        samples,
        total_ticks,
        freq,
        total_ticks as f64 / freq as f64
    );
    println!("(pprof-rs 用 HashMap<Frames, count> 表示 profile，每个 key 是若干次同栈采样的聚合)");
    println!("==================================\n");

    // 用 Debug 输出 top frames（Report 实现了 Debug，无 Display）
    // 输出太长会刷屏，只打印前 30 个非零 stack 的样本
    let mut sorted: Vec<_> = report.data.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("========== Top 30 热点栈（按采样数降序）==========");
    for (i, (frames, count)) in sorted.iter().take(30).enumerate() {
        println!("#{:02} count={:6}", i + 1, count);
        println!("{:?}", frames);
    }
    println!("===========================================\n");

    // Top-fold：按 self时间 聚合（等价于 `go tool pprof -top`）。
    // 关键：pprof 信号处理器触发时栈顶是 backtrace 自身，要跳过 backtrace/pprof/信号
    // 处理栈帧，找第一个用户态函数作为 self_time 归属。
    fn is_noise(name: &str) -> bool {
        name.starts_with("backtrace::")
            || name.starts_with("pprof::")
            || name.contains("perf_signal_handler")
            || name.contains("signal_handler")
            || name == "_start"
            || name == "__libc_start_call_main"
            || name == "__libc_start_main_impl"
    }
    let mut self_time: std::collections::HashMap<String, isize> = std::collections::HashMap::new();
    for (frames, count) in report.data.iter() {
        // pprof-rs 0.15 栈方向：frames[0]=leaf, frames.last()=root。
        // 在每个 Frame 内：symbols[0]=leaf（最近函数），最后=caller。
        // 跳过连续的噪音 Frame，找到第一个非噪音 Frame，再取该 Frame 的最深层 Symbol。
        let mut found_user = None;
        for frame in frames.frames.iter() {
            if let Some(sym) = frame.first() {
                let name = sym.name();
                if !is_noise(&name) {
                    found_user = Some(name);
                    break;
                }
            }
        }
        let key = found_user.unwrap_or_else(|| "<unknown>".to_string());
        *self_time.entry(key).or_insert(0) += count;
    }
    let mut top: Vec<_> = self_time.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1));
    println!("========== Top 函数（按 self time 聚合，等价于 `go tool pprof -top`）==========");
    println!("{:>6}  function", "ticks");
    for (name, count) in top.iter().take(40) {
        println!("{:6}  {}", count, name);
    }
    println!("=========================================================================\n");

    // 输出 protobuf 文件给 `go tool pprof` / `inferno-flamegraph`
    let profile = report.pprof()?;
    let out_dir = workspace_root.join("logs").join("profile");
    std::fs::create_dir_all(&out_dir)?;
    let out_path: PathBuf = out_dir.join(format!("{label}.pb"));
    let mut f = std::fs::File::create(&out_path)?;
    profile.write_to_writer(&mut f)?;
    println!(
        "pprof 输出: {} ({} samples, {} bytes)",
        out_path.display(),
        samples,
        std::fs::metadata(&out_path)?.len()
    );

    println!("\n解析命令:");
    println!("  go tool pprof -top {path}", path = out_path.display());
    println!("  go tool pprof -tree {path}", path = out_path.display());
    println!("  go tool pprof -list '<func>' {path}", path = out_path.display());
    println!("  inferno-flamegraph {path} > flame.svg", path = out_path.display());

    Ok(())
}