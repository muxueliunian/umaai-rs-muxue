//! MCTS 单局 pprof profiler（基于 d10872a `sim_profiler.rs` 模板）
//!
//! 跑 N 局 `RamenMctsTrainer` 拉面整局（speed build, base_seed=61444），
//! 用 [`pprof`] 全程采样，输出 Google pprof protobuf 到 `logs/profile/<label>.pb`，
//! 同时把 top 函数 / top 栈打到 stdout。`pprof-rs` 1kHz 采样，足够精度又不显著干扰测量。
//!
//! 用法（与 sim_profiler 同构）：
//! ```text
//! # pprof-rs 只支持 Linux/macOS，Windows 上无法运行
//! cargo run --release --features profiler --bin mcts_profiler
//! go tool pprof -top logs/profile/mcts.pb
//! go tool pprof -tree logs/profile/mcts.pb
//! inferno-flamegraph logs/profile/mcts.pb > flame.svg
//! ```
//!
//! 环境变量：
//! - `MCTS_PROFILER_RUNS`：跑批数（默认 1；MCTS 单局很慢，按需扩）
//! - `MCTS_PROFILER_FREQ`：采样频率 Hz（默认 1000）
//! - `MCTS_PROFILER_LABEL`：输出文件标签（默认 "mcts"）
//! - `MCTS_PROFILER_SEARCH_N`：每候选 rollout 数（默认 64）
//! - `MCTS_PROFILER_STAGES`：搜索阶段，逗号分隔（默认 "train,ramen,special"）
//! - `MCTS_PROFILER_NUM_THREADS`：rayon 线程数（默认 1，避免 par_iter 跨线程
//!   干扰 pprof 采样；多线程场景 pprof-rs 仍能跑但栈更碎）
//!
//! 目的：定位「MCTS rollout / 搜索 / 调度」真实 hot path，作为后续优化（rollout
//! 数据共享、ActionResult 池化、init 阶段剥离等）的输入。当前 cargo flamegraph
//! 多线程 + opt-level 任意档都会因为 `par_iter` + 闭包 inline 把栈压扁，看不清
//! MCTS 主循环（sim_profiler 同模板之所以可读，是因为单线程 handwritten）——
//! pprof-rs 用户态 backtrace + symbolization 不依赖 dwarf unwind tables，能在
// 任意 profile 下拿到清晰的 umasim 函数名。

#![cfg(feature = "profiler")]

use std::{collections::HashMap, env, path::PathBuf};

use anyhow::Result;
use pprof::{ProfilerGuardBuilder, protos::Message};
use umasim::{
    bench::run_seeded,
    game::InheritInfo,
    gamedata::init_global_with_config,
    search::SearchConfig,
    trainer::{LoggingTrainer, RamenMctsTrainer, RamenSearchStages, RamenSelection},
    utils::{get_workspace_root, load_game_config}
};

/// 美浦波旁 + speed build（与 `sim_profiler` 同款）
const UMA: u32 = 102_601;
const DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 0, 0, 0, 3],
    extra_count: [0, 10, 30, 10, 30, 40]
};

fn main() -> Result<()> {
    let runs: u64 = env::var("MCTS_PROFILER_RUNS").unwrap_or_else(|_| "1".into()).parse()?;
    let freq: i32 = env::var("MCTS_PROFILER_FREQ").unwrap_or_else(|_| "1000".into()).parse()?;
    let label: String = env::var("MCTS_PROFILER_LABEL").unwrap_or_else(|_| "mcts".into());
    let search_n: usize =
        env::var("MCTS_PROFILER_SEARCH_N").unwrap_or_else(|_| "64".into()).parse()?;
    let stages_str =
        env::var("MCTS_PROFILER_STAGES").unwrap_or_else(|_| "train,ramen,special".into());
    let num_threads: usize =
        env::var("MCTS_PROFILER_NUM_THREADS").unwrap_or_else(|_| "1".into()).parse()?;

    // 强制 rayon 线程数（默认 1：单线程跑局，pprof 栈最干净）
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    let workspace_root = get_workspace_root()?;
    std::env::set_current_dir(workspace_root.clone())?;
    init_global_with_config(&load_game_config()?)?;

    let search_config = SearchConfig::default()
        .with_search_n(search_n)
        .with_max_depth(0) // 拉面无 leaf 估值器，只能跑到终局
        .with_ucb(false)   // 默认均匀分配，小预算下 UCB 退化成均匀
        .with_radical_factor_max(0.0); // 取普通均值，measure 真实均分而不是好运尾部
    let stages = RamenSearchStages::parse(&stages_str)?;

    println!(
        "mcts_profiler: runs={runs} freq={freq}Hz search_n={search_n} \
         num_threads={num_threads} stages={stages_str} label={label}"
    );

    // 启动 CPU profiler（ITIMER_PROF）。blocklist 排除 libc/pthread 等系统库
    // ——这些是符号化/采样噪声来源，不在我们优化目标里。
    let guard = ProfilerGuardBuilder::default()
        .frequency(freq)
        .blocklist(&["libc", "pthread", "vdso", "rayon"])
        .build()?;

    // MCTS trainer 不实现 Clone / Copy，每局重新构造一次（与 sim_profiler 同款）
    let build_mcts = || {
        RamenMctsTrainer::new(search_config.clone())
            .with_stages(stages.clone())
            .with_selection(RamenSelection::Score)
    };

    let mut total_score = 0i64;
    for run_idx in 0..runs {
        let mcts = build_mcts();
        let mut trainer = LoggingTrainer::new(mcts, run_idx);
        // 关掉 logging 让采样聚焦在游戏/策略层
        trainer.set_logging(false);
        let outcome = run_seeded(UMA, &DECK, &INHERIT, 61444, run_idx, &trainer)?;
        total_score += outcome.score as i64;
        println!(
            "  #{:02} seed={} score={} elapsed={:.1}ms",
            run_idx + 1,
            outcome.seed,
            outcome.score,
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
        "采样栈数: {samples} | 总 tick: {total_ticks} | 频率 {freq}Hz | CPU 时间 ≈ {:.2}s",
        total_ticks as f64 / freq as f64
    );
    println!("(pprof-rs 用 HashMap<Frames, count> 表示 profile，每个 key 是若干次同栈采样的聚合)");
    println!("==================================\n");

    // Top 30 热点栈
    let mut sorted: Vec<_> = report.data.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("========== Top 30 热点栈（按采样数降序）==========");
    for (i, (frames, count)) in sorted.iter().take(30).enumerate() {
        println!("#{:02} count={:6}", i + 1, count);
        println!("{:?}", frames);
    }
    println!("===========================================\n");

    // Top-fold：按 self 时间 聚合（等价于 `go tool pprof -top`）。
    // 关键：pprof 信号处理器触发时栈顶是 backtrace 自身，要跳过 backtrace/pprof/
    // 信号处理栈帧，找第一个用户态函数作为 self_time 归属。
    fn is_noise(name: &str) -> bool {
        name.starts_with("backtrace::")
            || name.starts_with("pprof::")
            || name.contains("perf_signal_handler")
            || name.contains("signal_handler")
            || name == "_start"
            || name == "__libc_start_call_main"
            || name == "__libc_start_main_impl"
    }
    let mut self_time: HashMap<String, isize> = HashMap::new();
    for (frames, count) in report.data.iter() {
        // pprof-rs 0.15 栈方向：frames[0]=leaf, frames.last()=root。
        // 在每个 Frame 内：symbols[0]=leaf（最近函数），最后=caller。
        // 跳过连续的噪音 Frame，找到第一个非噪音 Frame，再取该 Frame 的最深层 Symbol。
        let mut found_user: Option<String> = None;
        for frame in frames.frames.iter() {
            if let Some(sym) = frame.first() {
                if !is_noise(&sym.name()) {
                    found_user = Some(sym.name());
                    break;
                }
            }
        }
        let key = found_user.unwrap_or_else(|| "<unknown>".to_string());
        *self_time.entry(key).or_insert(0) += count;
    }
    let mut top: Vec<_> = self_time.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1));
    println!("========== Top 函数（self time 聚合，等价于 `go tool pprof -top`）==========");
    println!("{:>6}  function", "ticks");
    for (name, count) in top.iter().take(40) {
        println!("{count:6}  {name}");
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