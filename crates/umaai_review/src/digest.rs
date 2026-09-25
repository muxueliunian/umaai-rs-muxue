//! digest.json 组装与落盘（文档 §3.5 schema）
//!
//! 强类型 struct 直接 `Serialize`（schema 与文档对齐）；`inherit` 块在 M4
//! 里程碑补齐（缺失字段直接不出现在 JSON 中）。

use std::{collections::BTreeMap, fs, path::{Path, PathBuf}};

use anyhow::{Context, Result};
use serde::Serialize;
use umasim::{gamedata::UmaData, global, gamedata::ramen::RAMENDATA, game::SupportCard};

use crate::{decisions::DecisionsResult, pack::Pack, schedule::Schedule, timeline::TimelineResult};
use umaai::protocol::GameStatusBase;
use umasim::gamedata::GAMEDATA;

/// digest 顶层
#[derive(Debug, Serialize)]
pub struct Digest {
    pub meta: Meta,
    pub timeline: Vec<crate::timeline::TimelineRow>,
    pub decisions: Vec<crate::decisions::DecRow>,
    pub execution: Vec<crate::execution::ExecRow>,
    pub luck: crate::decisions::LuckBlock,
    pub schedule: Schedule,
    /// 继承质量（§7；配置不可用 / 回合缺快照时缺席）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherit: Option<crate::inherit::InheritBlock>,
    /// 分身彩圈观测（§6.5；gamedata 缺失时缺席）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clones: Option<crate::clones::ClonesBlock>,
    pub coverage: crate::decisions::Coverage,
    pub findings: Vec<crate::execution::Finding>,
    pub context: DigestContext,
}

/// 局元信息（§3.5 meta；meta.json 缺失时从文件名与 CSV 降级推导）
#[derive(Debug, Serialize)]
pub struct Meta {
    pub game: u64,
    pub uma_id: u32,
    pub uma_name: String,
    pub deck: Vec<DeckCard>,
    pub start_turn: u32,
    pub mid_entry: bool,
    pub end_reason: String,
    pub snapshots: u64,
    pub decision_rows: u64,
    pub total_luck_end: Option<f64>,
    pub final_score: Option<i32>,
    pub rank: Option<String>,
}

/// 卡组条目（card_id = 协议 idrank = cardId×10 + 突破等级）
#[derive(Debug, Serialize)]
pub struct DeckCard {
    pub card_id: u32,
    pub name: String,
    /// 0速 1耐 2力 3根 4智 5友人 6团队
    pub card_type: i32,
    /// 突破等级（idrank % 10）
    pub limit_break: u32,
}

/// 归因环境（§3.5 context：换环境也能归因）
#[derive(Debug, Serialize)]
pub struct DigestContext {
    /// 地区 id → 名称（RAMENDATA.ramen_region_effect）
    pub region_names: BTreeMap<String, String>,
    /// 运气分口径说明
    pub luck_formula: String,
    /// 判据摘要 + 降级注记
    pub criteria: Vec<String>,
}

/// digest 组装入参
pub struct Inputs<'a> {
    pub pack: &'a Pack,
    pub timeline: &'a TimelineResult,
    pub decisions: &'a DecisionsResult,
    /// 实际执行推断（§11 步骤 4）
    pub execution: crate::execution::ExecutionResult,
    pub schedule: Schedule,
    /// gamedata 是否可用（names / score / region 表的降级开关）
    pub gamedata_ok: bool,
    /// 伪波动标记（§6.7；`checks::flagged_turns` 产出）
    pub flags: Vec<crate::decisions::FlaggedTurn>,
    /// 继承质量块（§7）
    pub inherit: Option<crate::inherit::InheritBlock>,
    /// 分身彩圈观测块（§6.5）
    pub clones: Option<crate::clones::ClonesBlock>,
    /// 检查项 findings（§6.1 / §6.6）
    pub extra_findings: Vec<crate::execution::Finding>,
    /// 自带 gamedata 的版本注记（`Some` = 用的是 skill 携带的旧版数据）
    pub gamedata_bundled: Option<String>,
}

/// 组装 digest（meta 降级推导 + deck / uma_name / final_score / context）
pub fn build(inputs: &Inputs) -> Digest {
    let pack = inputs.pack;
    let tl = inputs.timeline;
    let dec = inputs.decisions;

    // —— meta（meta.json 缺失时降级推导）——
    let meta_game = pack.meta.as_ref().map(|m| m.game).unwrap_or(pack.game);
    let uma_id = pack
        .meta
        .as_ref()
        .map(|m| m.uma_id)
        .or_else(|| tl.first_status.as_ref().map(|s| s.base_game.uma_id))
        .unwrap_or(0);
    let start_turn = pack
        .meta
        .as_ref()
        .map(|m| m.start_turn)
        .or_else(|| tl.rows.first().map(|r| r.turn))
        .unwrap_or(0);
    let end_reason = pack
        .meta
        .as_ref()
        .map(|m| m.end_reason.clone())
        .unwrap_or_else(|| "unknown(meta缺失)".to_string());
    let total_luck_end = pack
        .meta
        .as_ref()
        .and_then(|m| m.total_luck_end)
        .or_else(|| dec.luck.series.last().map(|p| p.total_luck));

    // deck（首份快照的 cardId = idrank；gamedata 缺失 → 纯 ID）
    let deck = deck_of(tl.first_status.as_ref().map(|s| &s.base_game), inputs.gamedata_ok);

    // uma_name（gamedata 缺失 → 纯 ID 展示，§9.2 第 7 条降级）
    let uma_name = if inputs.gamedata_ok {
        uma_data(uma_id)
            .map(|d| d.short_name().to_string())
            .unwrap_or_else(|| format!("unknown({uma_id})"))
    } else {
        format!("unknown({uma_id})")
    };

    // 终局评分 + 等级（末份快照；gamedata 缺失 → None + 注记）
    let (final_score, rank) = match (tl.last_status.as_ref().map(|s| &s.base_game), inputs.gamedata_ok) {
        (Some(base), true) => {
            let score = crate::score::final_score(base);
            (Some(score), Some(crate::score::rank_name(score)))
        }
        _ => (None, None),
    };

    let meta = Meta {
        game: meta_game,
        uma_id,
        uma_name,
        deck,
        start_turn,
        mid_entry: pack.meta.as_ref().map(|m| m.mid_entry).unwrap_or(false),
        end_reason,
        snapshots: pack
            .meta
            .as_ref()
            .map(|m| m.snapshots)
            .unwrap_or((pack.snaps.len() + pack.unparsed.len()) as u64),
        decision_rows: dec.rows.len() as u64,
        total_luck_end,
        final_score,
        rank,
    };

    // —— context ——
    let mut region_names = BTreeMap::new();
    if inputs.gamedata_ok {
        for re in &global!(RAMENDATA).ramen_region_effect {
            region_names.insert(re.id.to_string(), re.name.clone());
        }
    }
    let mut criteria = vec![
        "YEAR_BOUNDARIES=[24,48,72]（剧本年份边界，代码常量）".to_string(),
        "INHERIT_TURNS=[30,54]（两次继承回合，代码常量）".to_string(),
        "SUPER_RAMEN_TURNS=turn>=72（超级拉面期；实测分身自 72 起）".to_string(),
        "运气分读法（已知 bug 与固定波动区都会影响读数）：① 已知 bug——RMJ 结算时机\
         可能与实际数据不符（年界附近运气虚高）；支援卡连续事件进度未统计（模拟高估\
         好事件，整条曲线被系统性压低）→ total_luck_end 略低于实际运气，< -2000（≈\
         方差量级估计）才判「这局运气差」；② 固定波动区——年界 / 继承 / RMJ / 开局\
         2-3 回合第 1 年地区选择（选择带来的期望跳变，小赚或小亏均为程序性），\
         这些位置一增一减配对出现或为选择本身导致，读运气分时降级或跳过，\
         不要当真实损益".to_string(),
        "final_score 口径 = Uma::calc_score，仅供参考：略低于实际小黑板分数，且未计入「努力家」\
         等新状态".to_string(),
        "flagged_turns = 程序性波动标记（年界前2至后1回合 / 继承回合 / RMJ 结算 / 开局\
         2-3 回合第 1 年地区选择），归因时降级或跳过；turn 72 双属性：既标记为年界\
         波动、也算进超级拉面期统计（该回合份量实打实，正跳不是纯程序性回吐）"
            .to_string(),
        "检查项覆盖：已验证判据（目标赛未跑赢 / 关键资源过早耗尽 / 心情掉落未恢复）直接产\
         findings；训练失败出候选清单（info，人工复核）；其余判据（体力健康 / 吃面节奏 / 友人\
         完成度 / free_race / 状态健康 / 属性溢出）待实测，由 SKILL 层用 digest 数据判读"
            .to_string(),
        "彩圈判定 = 分身新增落位 == cardType（本体占的得意位不算）；有效增加彩圈来源二分：\
         luck=随机有效增加（吃面前本体在场 → 好运气）/ strategy=规则有效增加（吃面前本体\
         缺席，地区/超拉规则带入得意位 → 好策略）；A 类地区分身（turn<72）评判判据待定\
         义，clones 块为观测数据（game6234 实测分身真实落得意位仅 2/31，分身主要价值是\
         加人头）；B 类只统计训练卡，没吃到彩圈不算亏，只能用该期运气分判盈亏".to_string(),
        "运气极值归因（top_gain/top_loss 的叙事分类）由 SKILL 层结合 timeline + flagged_turns + \
         decisions 完成——注意 turn_delta 是「局面期望终局分」变化，不等于本回合属性增量"
            .to_string()
    ];
    if !inputs.gamedata_ok {
        criteria.push("gamedata 缺失：uma/卡名、地区名、赛程、终局评分与等级均已降级（纯 ID）".to_string());
    }
    if let Some(v) = &inputs.gamedata_bundled {
        criteria.push(format!(
            "gamedata 为 skill 自带旧版（{v}）：卡名 / 赛程 / 地区名可能与当前游戏版本不一致，\
             结论按旧版口径读"
        ));
    }
    if pack.meta.is_none() {
        criteria.push("meta.json 缺失：局元信息从文件名与 CSV 降级推导".to_string());
    }
    if pack.decisions_csv.is_none() {
        criteria.push("decisions.csv 缺失：决策明细 / 运气分 / coverage 不可用".to_string());
    }
    if !tl.parse_errors.is_empty() {
        criteria.push(format!("快照解析失败 {} 份（coverage.parse_error）", tl.parse_errors.len()));
    }

    // coverage（unparsed / parse_error 来自 pack 与 timeline）
    let mut coverage = dec.coverage.clone();
    coverage.unparsed = pack.unparsed.len() as u64;
    coverage.parse_error = tl.parse_errors.len() as u64;

    let context = DigestContext {
        region_names,
        luck_formula: "total_luck = T(n) − T(1)（T = 期望终局分；t_n_display = raw + (78 − turn) × \
                       mcts_turn_bonus，t_n_raw 为反推 raw）；turn_delta = T(n+1) − T(n) 是「局面期望\
                       终局分」变化，不等于本回合属性增量；同回合多段 Δ 以回合合计为基本观察单位"
            .to_string(),
        criteria,
    };

    // luck 块：填充伪波动标记（decisions 层无 timeline 信息，由 checks 产出）
    let mut luck = dec.luck.clone();
    luck.flagged_turns = inputs.flags.clone();

    // findings：execution 偏离 + 检查项命中
    let mut findings = inputs.execution.findings.clone();
    findings.extend(inputs.extra_findings.clone());

    Digest {
        meta,
        timeline: tl.rows.clone(),
        decisions: dec.rows.clone(),
        execution: inputs.execution.rows.clone(),
        luck,
        schedule: inputs.schedule.clone(),
        inherit: inputs.inherit.clone(),
        clones: inputs.clones.clone(),
        coverage,
        findings,
        context,
    }
}

/// deck 组装（idrank → SupportCard；gamedata 缺失 → 纯 ID 展示）
fn deck_of(first_base: Option<&GameStatusBase>, gamedata_ok: bool) -> Vec<DeckCard> {
    let Some(base) = first_base else {
        return Vec::new();
    };
    base.card_id
        .iter()
        .map(|&idrank| {
            if gamedata_ok {
                match SupportCard::new(idrank) {
                    Ok(card) => DeckCard {
                        card_id: idrank,
                        name: card.data.card_name.clone(),
                        card_type: card.card_type,
                        limit_break: card.rank,
                    },
                    Err(_) => DeckCard {
                        card_id: idrank,
                        name: format!("unknown({idrank})"),
                        card_type: -1,
                        limit_break: idrank % 10,
                    },
                }
            } else {
                DeckCard {
                    card_id: idrank,
                    name: format!("unknown({idrank})"),
                    card_type: -1,
                    limit_break: idrank % 10,
                }
            }
        })
        .collect()
}

/// 取马娘数据（gamedata 可用但 id 查不到时返回 None）
fn uma_data(uma_id: u32) -> Option<&'static UmaData> {
    GAMEDATA.get().and_then(|g| g.get_uma(uma_id).ok())
}

/// digest 落盘（紧凑 JSON，不换行缩进）→ 返回文件路径
pub fn write_json(digest: &Digest, out_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("创建输出目录失败: {}", out_dir.display()))?;
    let path = out_dir.join("digest.json");
    let f = fs::File::create(&path).with_context(|| format!("创建 digest.json 失败: {}", path.display()))?;
    // 紧凑序列化（不换行不缩进）：timeline/decisions 行数多，省空间优先于可读性
    serde_json::to_writer(f, digest).with_context(|| "序列化 digest 失败")?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decisions, timeline};

    /// 最小可用 digest 组装（不依赖 gamedata，走降级路径）+ 落盘回读
    #[test]
    fn test_build_and_write_digest() -> Result<()> {
        let json = r#"{
            "baseGame": {
                "scenarioId": 14, "umaId": 112402, "umaStar": 5, "turn": 3,
                "vital": 80, "maxVital": 100, "motivation": 4,
                "fiveStatus": [100, 200, 300, 400, 500],
                "fiveStatusLimit": [1200, 1200, 1200, 1200, 1200],
                "skillPt": 10, "skillScore": 0, "totalHints": 5,
                "trainLevelCount": [1, 2, 3, 4, 5],
                "ptScoreRate": 2.0, "failureRateBias": 0,
                "isIll": false, "isQieZhe": false, "isAiJiao": false,
                "isPositiveThinking": false, "isRefreshMind": false, "isLucky": false,
                "zhongMaBlueCount": [0, 0, 0, 0, 0], "isRacing": false,
                "cardId": [302424, 302894],
                "persons": [], "personDistribution": [[], [], [], [], []],
                "lockedTrainingId": -1,
                "friendship_noncard_yayoi": 0, "friendship_noncard_reporter": 0,
                "friend_stage": 0, "friend_outgoingUsed": 0,
                "playing_state": 1, "raceHistory": [], "story": null,
                "source": "command"
            },
            "ramen": {}
        }"#;
        let snaps = vec![crate::pack::SnapEntry {
            file: "g1_turn3.json".to_string(),
            game: 1,
            turn: 3,
            seq: 0,
            bytes: json.as_bytes().to_vec(),
        }];
        let tl = timeline::build(&snaps);
        let dec = decisions::parse(None).unwrap();
        let exec = crate::execution::build(&tl.rows, &dec.rows, &[]);
        let inputs = Inputs {
            pack: &Pack {
                game: 1,
                snaps: snaps.clone(),
                unparsed: vec![],
                decisions_csv: None,
                meta: None,
                luck_trend_svg: None,
                ignored: vec![],
            },
            timeline: &tl,
            decisions: &dec,
            execution: exec,
            schedule: Schedule::default(),
            gamedata_ok: false,
            flags: vec![],
            inherit: None,
            clones: None,
            extra_findings: vec![],
            gamedata_bundled: None,
        };
        let digest = build(&inputs);
        let out = std::env::temp_dir().join(format!("digest_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&out);
        let path = write_json(&digest, &out).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        println!("digest.json 回读:\n{}", serde_json::to_string_pretty(&v["meta"]).unwrap());
        println!("context.criteria: {:#?}", v["context"]["criteria"]);
        assert_eq!(v["meta"]["game"], serde_json::json!(1));
        assert_eq!(v["meta"]["uma_id"], serde_json::json!(112402));
        assert_eq!(v["meta"]["uma_name"], serde_json::json!("unknown(112402)"));
        assert_eq!(v["meta"]["deck"].as_array().unwrap().len(), 2, "deck 来自首快照 cardId");
        assert_eq!(v["meta"]["final_score"], serde_json::Value::Null, "gamedata 缺失 → 评分降级");
        assert_eq!(v["timeline"].as_array().unwrap().len(), 1);
        assert!(
            v.get("inherit").is_none(),
            "inherit 块 M4 才实现，应缺席"
        );
        assert_eq!(
            v["execution"].as_array().unwrap().len(),
            0,
            "无决策行 → execution 空数组"
        );
        assert_eq!(v["findings"].as_array().unwrap().len(), 0);
        let _ = fs::remove_dir_all(&out);
        Ok(())
    }
}
