//! 固定/随机具体卡组的配对实验；输入 manifest 固定卡组、种子、策略，逐局保留失败。
use std::{collections::HashSet, env, fs, path::Path};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Deserialize;
use umasim::{
    bench,
    game::InheritInfo,
    gamedata::init_global_with_config,
    output::diagnostic::DiagGuard,
    trainer::{LoggingTrainer, RecommendedRamenTrainer},
    utils::load_game_config,
};

/// 可复现的整批实验定义，卡组生成与模拟随机性分离。
#[derive(Deserialize)]
struct Manifest {
    variants: Vec<String>,
    runs: u64,
    cases: Vec<Case>,
}

/// 一个卡组分层；新旧策略复用相同种子序列。
#[derive(Deserialize)]
struct Case {
    name: String,
    cohort: String,
    uma: u32,
    deck: [u32; 6],
    blue_count: [i32; 5],
    extra_count: [i32; 6],
    seed: u64,
}

/// 在开始模拟前拒绝空实验、重复配对键和非法策略，避免结果被重复样本加权。
fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(manifest.runs > 0 && !manifest.cases.is_empty() && !manifest.variants.is_empty(), "实验不能为空");
    let mut names = HashSet::new();
    for case in &manifest.cases {
        ensure!(!case.name.is_empty() && names.insert(&case.name), "卡组名称为空或重复: {}", case.name);
    }
    let mut variants = HashSet::new();
    for variant in &manifest.variants {
        ensure!(variants.insert(variant), "策略重复: {variant}");
        RecommendedRamenTrainer::with_tokens(variant)?;
    }
    Ok(())
}

/// 运行 manifest 并保存逐局结果；异常不丢弃、不补采。
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    ensure!(args.len() == 3, "policy_pair_bench MANIFEST.json OUTPUT.csv（从 workspace 根运行）");
    let manifest: Manifest = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    validate_manifest(&manifest)?;
    init_global_with_config(&load_game_config()?)?;
    let _diag = DiagGuard::suppress();
    let mut rows = Vec::new();
    let mut total_failures = 0;
    for case in &manifest.cases {
        let batch: Vec<Vec<String>> = (0..manifest.runs).into_par_iter().flat_map_iter(|run| {
            manifest.variants.iter().map(move |variant| {
                let result = (|| -> Result<bench::GameOutcome> {
                    let policy = RecommendedRamenTrainer::with_tokens(variant)?;
                    let mut trainer = LoggingTrainer::new(policy, run);
                    trainer.set_logging(false);
                    let inherit = InheritInfo { blue_count: case.blue_count, extra_count: case.extra_count };
                    bench::run_seeded(case.uma, &case.deck, &inherit, case.seed, run, &trainer)
                })();
                let mut row = vec![case.name.clone(), case.cohort.clone(), case.uma.to_string(),
                    case.deck.iter().map(u32::to_string).collect::<Vec<_>>().join("/"),
                    case.seed.to_string(), run.to_string(), variant.clone()];
                match result {
                    Ok(out) => {
                        row.extend([out.seed.to_string(), out.score.to_string(), out.skill_pt.to_string(),
                            out.rmj_ok.to_string(), out.free_race_ok.to_string(), out.friend_all.to_string(),
                            out.five_status.iter().map(i32::to_string).collect::<Vec<_>>().join("/"),
                            out.elapsed_ms.to_string(), String::new()]);
                    }
                    Err(err) => {
                        row.extend(vec![String::new(); 8]);
                        row.push(format!("{err:#}"));
                    }
                }
                row
            })
        }).collect();
        let failures = batch.iter().filter(|r| !r[15].is_empty()).count();
        total_failures += failures;
        println!("{}: {} results, {} errors", case.name, batch.len(), failures);
        rows.extend(batch);
        bench::write_csv(Path::new(&args[2]), &["case", "cohort", "uma", "deck", "base_seed", "run",
            "variant", "rule_seed", "score", "skill_pt", "rmj_ok", "free_race_ok", "friend_all", "five", "elapsed_ms", "error"], &rows)
            .context("保存配对实验结果失败")?;
    }
    ensure!(total_failures == 0, "{total_failures} 局失败，全部异常行已保留到 CSV");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Manifest, validate_manifest};
    use anyhow::{Result, ensure};

    /// 重复实验键必须在开跑前报错，保证统计样本不会静默重复或覆盖。
    #[test]
    fn rejects_ambiguous_pair_keys() -> Result<()> {
        let source = include_str!("../../../../experiments/validated_policy/final-check.json");
        let mut manifest: Manifest = serde_json::from_str(source)?;
        ensure!(validate_manifest(&manifest).is_ok(), "冻结清单合法");
        manifest.variants.push(manifest.variants[0].clone());
        ensure!(validate_manifest(&manifest).is_err(), "重复策略应报错");
        manifest.variants.clear();
        ensure!(validate_manifest(&manifest).is_err(), "空策略应报错");
        manifest = serde_json::from_str(source)?;
        manifest.cases[1].name = manifest.cases[0].name.clone();
        ensure!(validate_manifest(&manifest).is_err(), "重复卡组名称应报错");
        manifest = serde_json::from_str(source)?;
        manifest.runs = 0;
        ensure!(validate_manifest(&manifest).is_err(), "零局数应报错");
        println!("冻结清单合法；重复策略、空策略、重复名称和零局数均被拒绝");
        Ok(())
    }
}
