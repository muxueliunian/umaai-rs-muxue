//! RNG 受控重构集成测试（rng_refactor_plan_v2 §6 层 2 / 层 3）
//!
//! - 层 2：跨策略逐回合一致——固定 rule_master 下 RandomTrainer 与手写策略
//!   看到的分布/角标/hint 逐位相同（事件 ID 增量输出对比）。
//! - 层 3：隔离性——回合重置（前 14 回合消耗不影响第 15 回合）、克隆隔离、
//!   流间不污染。
//!
//! 按项目规范以 println 输出对比结果，不 assert。

use std::{cell::RefCell, collections::BTreeMap};

use anyhow::Result;
use rand::RngCore;

use crate::{
    bench::seeded_rngs,
    game::{
        ActionEnum,
        Game,
        InheritInfo,
        ramen::{FeelingType, RamenGame},
        traits::Trainer
    },
    rng::SplitmixRng,
    trainer::{RamenHandwrittenTrainer, RandomTrainer},
    utils::{get_workspace_root, init_test_logger}
};

const TEST_UMA_ID: u32 = 102601;
const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
const TEST_INHERIT: InheritInfo = InheritInfo {
    blue_count: [15, 3, 0, 0, 0],
    extra_count: [0, 30, 0, 0, 30, 30]
};

/// 每回合一条的局面快照
#[derive(Clone)]
struct TurnSnap {
    /// 回合号
    turn: i32,
    /// 分布表快照
    dist: Vec<Vec<i32>>,
    /// 训练角标
    feeling: Option<[FeelingType; 5]>,
    /// 全体人头 hint 位
    hints: Vec<bool>,
    /// 本回合事件计数增量（ID -> 次数）
    event_delta: Vec<(u32, u32)>,
    /// 回合结束时固定流已消费次数（调试固定流序列用）
    fixed_counter: u64
}

/// 局面快照 Trainer：每次决策前记录当前回合的固定局面
struct SnapTrainer<T> {
    /// 内部真实训练员
    inner: T,
    /// 逐回合快照（每回合一条）
    snaps: RefCell<Vec<TurnSnap>>,
    /// 上一份事件计数（增量对比用）
    last_events: RefCell<BTreeMap<u32, u32>>,
    /// 当前回合是否已记录
    recorded_turn: RefCell<i32>,
    /// 事件增量暂存（记录快照时合并）
    pending_delta: RefCell<Vec<(u32, u32)>>
}

impl<T> SnapTrainer<T> {
    /// 构造快照训练员
    fn new(inner: T) -> Self {
        Self {
            inner,
            snaps: RefCell::new(Vec::new()),
            last_events: RefCell::new(BTreeMap::new()),
            recorded_turn: RefCell::new(-1),
            pending_delta: RefCell::new(Vec::new())
        }
    }

    /// 取快照（测试输出用）
    fn snaps(&self) -> Vec<TurnSnap> {
        self.snaps.borrow().iter().cloned().collect()
    }

    /// 决策前记录：事件增量 + 每回合首条固定局面快照
    fn observe(&self, game: &RamenGame) {
        let turn = game.turn();
        // 事件计数增量（与上次观测之差）
        let cur: BTreeMap<u32, u32> = game.base.events.iter().map(|(k, v)| (*k, *v)).collect();
        let mut delta: Vec<(u32, u32)> = Vec::new();
        for (k, v) in &cur {
            let prev = self.last_events.borrow().get(k).copied().unwrap_or(0);
            if *v > prev {
                delta.push((*k, *v - prev));
            }
        }
        *self.last_events.borrow_mut() = cur;

        // 每回合只记一条（回合内固定局面不变）
        if *self.recorded_turn.borrow() != turn {
            *self.recorded_turn.borrow_mut() = turn;
            let hints: Vec<bool> = game.persons.iter().map(|p| p.is_hint).collect();
            self.snaps.borrow_mut().push(TurnSnap {
                turn,
                dist: game.base.distribution.clone(),
                feeling: game.ramen.train_feeling_type,
                hints,
                event_delta: std::mem::take(&mut *self.pending_delta.borrow_mut()),
                fixed_counter: game.turn_fixed.as_ref().map(|r| r.counter()).unwrap_or(u64::MAX)
            });
        } else {
            // 同回合后续决策：事件增量并入待记录
            self.pending_delta.borrow_mut().extend(delta);
        }
    }
}

impl<T: Trainer<RamenGame>> Trainer<RamenGame> for SnapTrainer<T> {
    fn select_action(
        &self, game: &RamenGame, actions: &[<RamenGame as Game>::Action], rng: &mut rand::rngs::StdRng
    ) -> Result<usize> {
        self.observe(game);
        self.inner.select_action(game, actions, rng)
    }

    fn select_choice(
        &self, game: &RamenGame, choices: &[Vec<crate::gamedata::EventChoice>], rng: &mut rand::rngs::StdRng
    ) -> Result<usize> {
        self.observe(game);
        self.inner.select_choice(game, choices, rng)
    }

    fn select_event_choice(
        &self, game: &RamenGame, _event: &crate::gamedata::EventData, choices: &[Vec<crate::gamedata::EventChoice>],
        rng: &mut rand::rngs::StdRng
    ) -> Result<usize> {
        self.observe(game);
        self.inner.select_event_choice(game, _event, choices, rng)
    }
}

/// 按回合号索引快照（对比用）
fn by_turn_map(s: &[TurnSnap]) -> BTreeMap<i32, &TurnSnap> {
    s.iter().map(|x| (x.turn, x)).collect()
}

/// 用固定 rule_master 跑 `turns` 个完整回合（到 turn == turns 的 Begin 前停止）
///
/// 每回合 Distribute 前把支援卡羁绊锁定为 100：得意率（deyilv）随羁绊变化，
/// 会经分布权重影响人头分配——测试要验证的是「固定流随机序列与策略无关」，
/// 故需消除羁绊这一策略状态对分布的干扰（用户拍板：测试时设为最大羁绊）。
fn run_turns<T: Trainer<RamenGame>>(trainer: &T, master: u64, turns: i32) -> Result<()> {
    let (mut decision_rng, _) = seeded_rngs(master, 0);
    let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
    game.set_rule_master(master);
    while game.turn() < turns {
        if game.stage == crate::game::ramen::RamenStage::Distribute {
            for i in 0..6 {
                game.deck[i].friendship = 100;
                game.persons[i].friendship = 100;
            }
        }
        game.run_stage(trainer, &mut decision_rng)?;
        if !game.next() {
            break;
        }
    }
    Ok(())
}

/// 层 2：跨策略逐回合一致
///
/// 同一 rule_master 下 RandomTrainer 与手写策略各跑 20 回合，逐回合对比
/// 分布/角标/hint（应完全一致）并输出事件增量（固定事件应一致，策略事件
/// 可能不同属预期）。
#[test]
fn test_layer2_cross_strategy_consistency() -> Result<()> {
    let root = get_workspace_root()?;
    std::env::set_current_dir(root)?;
    let _ = init_test_logger("error");
    let _ = crate::gamedata::init_global();

    let master = 20260822u64;
    let rt = SnapTrainer::new(RandomTrainer);
    let ht = SnapTrainer::new(RamenHandwrittenTrainer::new());
    run_turns(&rt, master, 20)?;
    run_turns(&ht, master, 20)?;

    let rs = rt.snaps();
    let hs = ht.snaps();
    println!("===== 层 2：跨策略逐回合一致（rule_master={master}，各 20 回合）=====");
    println!("RandomTrainer 快照 {} 条，手写策略快照 {} 条", rs.len(), hs.len());

    let by_turn = by_turn_map;
    let rmap = by_turn(&rs);
    let hmap = by_turn(&hs);

    let mut mismatch = 0;
    let mut compared = 0;
    for (turn, r) in &rmap {
        if let Some(h) = hmap.get(turn) {
            compared += 1;
            // 一致性判定口径：角标 + 分布表 + 固定流消费量（hint 依赖 PT 档位，不参与）
            let same = r.feeling == h.feeling && r.dist == h.dist && r.fixed_counter == h.fixed_counter;
            if !same {
                mismatch += 1;
            }
            let fixed_only = |d: &[(u32, u32)]| -> Vec<(u32, u32)> {
                d.iter()
                    .copied()
                    .filter(|(id, _)| !(5007..=5011).contains(id) && *id != 830305102)
                    .collect()
            };
            let same_dist = r.dist == h.dist;
            let same_feel = r.feeling == h.feeling;
            // hint 位依赖剧本 PT 档位（scenario_pt 决定 hint 出现率，策略状态），
            // 属游戏机制而非随机错位——单独输出，不作为一致性的判定口径。
            let same_hint = r.hints == h.hints;
            println!(
                "回合 {}: 角标一致={same_feel} 分布一致={same_dist} | 固定流消费 随机={} 手写={} | hint一致={same_hint}(依赖PT档位) | 事件增量 随机={:?} 手写={:?}",
                turn,
                r.fixed_counter,
                h.fixed_counter,
                fixed_only(&r.event_delta),
                fixed_only(&h.event_delta)
            );
            if !same || !same_feel || !same_hint {
                println!("  随机: dist={:?}", r.dist);
                println!("  手写: dist={:?}", h.dist);
                println!("  随机: feel={:?} hints={:?}", r.feeling, r.hints);
                println!("  手写: feel={:?} hints={:?}", h.feeling, h.hints);
            }
        }
    }
    println!("对比 {} 个回合，分布/角标不一致 {mismatch} 个", compared);
    println!("结论: 角标/分布/固定流消费量应逐位一致（同一 (master, turn) 固定局面）；");
    println!("      hint 位依赖剧本 PT 档位（策略状态，游戏机制）不参与一致性判定");
    Ok(())
}

/// 层 3a：回合重置隔离——狂训练 vs 狂休息各 20 回合，回合 15 分布一致
///
/// 前 14 回合的策略随机消耗（策略流）不得影响第 15 回合的固定流。
#[test]
fn test_layer3_turn_reset_isolation() -> Result<()> {
    let root = get_workspace_root()?;
    std::env::set_current_dir(root)?;
    let _ = init_test_logger("error");
    let _ = crate::gamedata::init_global();

    // 狂训练：永远选第一个 Train 动作
    struct TrainAll;
    impl Trainer<RamenGame> for TrainAll {
        fn select_choice(
            &self, _game: &RamenGame, _choices: &[Vec<crate::gamedata::EventChoice>], _rng: &mut rand::rngs::StdRng
        ) -> Result<usize> {
            Ok(0)
        }

        fn select_action(
            &self, _game: &RamenGame, actions: &[<RamenGame as Game>::Action], _rng: &mut rand::rngs::StdRng
        ) -> Result<usize> {
            Ok(actions
                .iter()
                .position(|a| matches!(a.as_base_action(), Some(crate::game::BaseAction::Train(_))))
                .unwrap_or(0))
        }
    }
    // 狂休息
    struct RestAll;
    impl Trainer<RamenGame> for RestAll {
        fn select_choice(
            &self, _game: &RamenGame, _choices: &[Vec<crate::gamedata::EventChoice>], _rng: &mut rand::rngs::StdRng
        ) -> Result<usize> {
            Ok(0)
        }

        fn select_action(
            &self, _game: &RamenGame, actions: &[<RamenGame as Game>::Action], _rng: &mut rand::rngs::StdRng
        ) -> Result<usize> {
            Ok(actions
                .iter()
                .position(|a| a.as_base_action() == Some(crate::game::BaseAction::Sleep))
                .unwrap_or(0))
        }
    }

    let master = 777u64;
    let ta = SnapTrainer::new(TrainAll);
    let ra = SnapTrainer::new(RestAll);
    run_turns(&ta, master, 20)?;
    run_turns(&ra, master, 20)?;

    let tvec = ta.snaps();
    let rvec = ra.snaps();
    let tmap = by_turn_map(&tvec);
    let rmap = by_turn_map(&rvec);
    println!("===== 层 3a：回合重置隔离（狂训练 vs 狂休息，20 回合）=====");
    let mut mismatch = 0;
    for turn in 0..20 {
        match (tmap.get(&turn), rmap.get(&turn)) {
            (Some(t), Some(r)) => {
                // 口径：分布表 + 角标 + 固定流消费量逐位一致（hint 依赖 PT 档位，不参与）
                let same = t.dist == r.dist && t.feeling == r.feeling && t.fixed_counter == r.fixed_counter;
                if !same {
                    mismatch += 1;
                }
                if turn % 3 == 0 || turn == 15 {
                    println!("回合 {turn}: 分布/角标/固定流消费一致 = {same}");
                }
            }
            _ => println!("回合 {turn}: 一侧无决策（比赛/无决策回合），跳过")
        }
    }
    println!("不一致回合数: {mismatch}（应为 0）");
    Ok(())
}

/// 层 3b：克隆隔离——克隆局面的流消费不影响原局面（MCTS rollout 隔离原子验证）
#[test]
fn test_layer3_clone_isolation() -> Result<()> {
    let mut a = SplitmixRng::new(0xDEAD_BEEF);
    let mut b = a;
    let _ = b.next_u64();
    println!("===== 层 3b：克隆隔离 =====");
    println!(
        "原流 counter={}（消费克隆后应保持），克隆流 counter={}",
        a.counter(),
        b.counter()
    );
    println!("原流下一个值不受克隆消费影响: {:#018x}", a.next_u64());
    Ok(())
}

/// 层 3c：流间不污染——同回合内策略流消耗后，回合固定流下一值不变
#[test]
fn test_layer3_stream_isolation() -> Result<()> {
    let root = get_workspace_root()?;
    std::env::set_current_dir(root)?;
    let _ = crate::gamedata::init_global();

    let master = 999u64;
    let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
    game.set_rule_master(master);

    // 消费固定流 3 次，记录第 4 次前的状态
    let mut fixed = game.turn_fixed.take().expect("set_rule_master 后应有固定流");
    let _ = (fixed.next_u64(), fixed.next_u64(), fixed.next_u64());
    let before = fixed.next_u64();
    game.turn_fixed = Some(fixed);

    // 消费策略流 5 次
    let mut strat = game.strategy.take().expect("set_rule_master 后应有策略流");
    for _ in 0..5 {
        let _ = strat.next_u64();
    }
    game.strategy = Some(strat);

    // 再取固定流第 4 次值（应为同一值）
    let mut fixed2 = game.turn_fixed.take().expect("固定流应还在");
    let after = fixed2.next_u64();
    game.turn_fixed = Some(fixed2);

    println!("===== 层 3c：流间不污染 =====");
    println!("策略流消耗 5 次前后，固定流第 4 次值: {before:#018x} vs {after:#018x}");
    println!("相同（不污染）: {}", before == after);
    Ok(())
}
