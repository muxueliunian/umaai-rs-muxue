//! 拉面杯手写策略
//!
//! 两个层次：
//!
//! 1. **固定策略**（初期实现）：地区选择固定顺序、超级拉面固定选项二。
//! 2. **手写策略核心**（本轮交付）：参数化打分策略 [`RamenPolicy`]——确定性
//!    `f(局面, 候选) -> 索引` 纯函数，不依赖 RNG、平局按候选固定顺序，
//!    供 `RamenHandwrittenTrainer`（整局测试壳）与未来 MCTS rollout 基策使用。
//!
//! 设计要点（对应 `.trae/documents/handwritten_policy/handwritten_base_policy_plan.md`）：
//! - 策略形态：结构体存权重/阈值常量（[`RamenPolicyConfig`]）+ 预设构造器（default / speed_build）
//! - 尽量复用现有公式：训练数值直接调 `Game::calc_training_buff` / `calc_training_value` /
//!   `calc_training_failure_rate`；吃面 PT 调 `rules::calc_ramen_pt_gain`
//! - 评分折算：属性按 `GAMECONSTANTS.five_status_final_score` 差分（与 `Uma::calc_score` 一致，边际准确）

use anyhow::Result;

use super::rules::{calc_ramen_pt_gain, get_region_range, get_super_ramen_clone_train_options};
use crate::{
    game::{
        ramen::{Operation, RamenAction, RamenGame},
        traits::Game
    },
    gamedata::{EventChoice, FreeRaceData, GAMECONSTANTS, ramen::RAMENDATA},
    global
};

/// 手写策略参数化配置（权重/阈值常量）
///
/// 所有启发式分支的数字都集中在这里，调参只改常量表；
/// 每个常量的调参依据记录在 `log/决策日志 + 调参记录`（计划 §7）。
#[derive(Debug, Clone, PartialEq)]
pub struct RamenPolicyConfig {
    // ===== 守门（安全剪枝）=====
    /// 体力低于此值强制休息（经验：<45 训练失败率高）
    pub vital_rest: i32,
    /// 心情低于此值强制外出（经验：<3 训练数值损失大）
    pub motivation_outing: i32,
    /// 生病时治病（Clinic）优先级权重（守门直通，无需打分）
    // ===== Train 打分 =====
    /// 满足心情时属性差分每点折算倍率（通常 1.0）
    pub status_rate: f32,
    /// PT→评分折算（默认与 `pt_score_rate` 同量级）
    pub pt_rate: f32,
    /// 训练失败惩罚（期望值中被扣减的固定分）
    pub failure_penalty: f32,
    /// 彩圈（友情训练）加成：每个彩圈附加分
    pub shining_bonus: f32,
    /// 训练体力消耗的折算（负值即当前消耗的体力价值；纳入后训练会自减肥力成本）
    pub train_vital_value: f32,
    // ===== 休息 =====
    /// 休息基础价值（体力充足时也有的固定收益，避免频繁休息）
    pub rest_base: f32,
    /// 休息恢复体力每点价值（恢复越多、体力越低，价值越高）
    pub rest_vital_value: f32,
    /// 训练前保留的目标体力（低于此值更倾向休息；高于此值体力收益边际下降）
    pub rest_target_vital: i32,
    // ===== 比赛 =====
    /// 比赛等级（G2/G1 等）→ 分数折算
    ///
    /// **仅用于「自选比赛已达标 / 本马娘无自选比赛要求」的场合**：此时比赛纯粹是
    /// 「用一回合换属性与技能点」，实测（seed 42-81，40 局）该权重每提高一档都在
    /// 掉分（0→51168 / 300→49129 / 900→43168），故默认 0——不主动打无意义的比赛。
    pub race_grade_weight: f32,
    /// 自选比赛缺口的紧迫度权重
    ///
    /// 打分形态 `weight × 缺口场数 / 区间内剩余可比赛回合数`：区间宽裕时接近 0
    /// （不打扰训练），越接近截止回合分值越高，自然地在后段补齐比赛。
    /// 与 [`race_gate_slack`](Self::race_gate_slack) 的硬守门配合使用。
    pub race_free_urgency_weight: f32,
    /// 自选比赛硬守门的缓冲回合数
    ///
    /// 当「区间内剩余可比赛回合数 ≤ 缺口场数 + slack」时**强制比赛**，优先于一切打分。
    /// 依据：自选比赛不达标直接导致育成失败（`BaseGame::check_free_race`），
    /// 见 `.trae/documents/handwritten_policy/good_bad_labels_draft.md` §四.1。
    /// slack=1 表示留 1 个回合的余量。
    pub race_gate_slack: u32,
    // ===== 外出 =====
    /// 普通外出基础价值（随机事件期望）
    pub outing_base: f32,
    /// 友人出行额外价值（+2 隐藏风味 + 友人事件链进度）
    pub friend_outing_bonus: f32,
    // ===== RamenSelect（选面）=====
    /// 吃面 PT 增益→分数折算
    pub ramen_pt_weight: f32,
    /// 地区效果（xunlian 对训练增益）→分数折算
    pub ramen_effect_weight: f32,
    /// 消耗 1 个隐藏风味的成本（保留库存）
    pub ramen_special_cost: f32,
    /// 普通诀窍机会成本权重（吃面消耗 5 诀窍的折算）
    pub ramen_stock_cost: f32,
    // ===== RegionSelect（年度选面）=====
    /// 地区 xunlian 加成→分数折算
    pub region_xunlian_weight: f32,
    /// 地区 pt_bonus→分数折算
    pub region_pt_weight: f32,
    /// 地区 hint_count→分数折算
    pub region_hint_weight: f32,
    /// 地区 youqing 加成→分数折算（与 `region_xunlian_weight` 同族，作用于不同年份）
    ///
    /// 第 1 年地区只有 `xunlian`、第 2/3 年只有 `youqing`，两项不会同时非零；
    /// 且同一年内 `pt_bonus` / `hint_count` 恒定，故本权重的绝对值不影响 argmax，
    /// 只影响打印出来的分数量级。
    pub region_youqing_weight: f32,
    // ===== Event =====
    /// 事件体力每点折算
    pub event_vital_weight: f32,
    /// 事件干劲每点折算
    pub event_motivation_weight: f32,
    /// 事件获得 bad flag（ill/bad_trainer）的惩罚
    pub event_bad_flag_penalty: f32
}

impl Default for RamenPolicyConfig {
    /// 保守默认：先求稳定，再逐项调参
    fn default() -> Self {
        Self {
            vital_rest: 45,
            motivation_outing: 3,
            status_rate: 1.0,
            pt_rate: 8.0,
            failure_penalty: 60.0,
            shining_bonus: 60.0,
            train_vital_value: 1.8,
            rest_base: 20.0,
            rest_vital_value: 2.5,
            rest_target_vital: 55,
            race_grade_weight: 0.0,
            race_free_urgency_weight: 2000.0,
            race_gate_slack: 1,
            outing_base: 15.0,
            friend_outing_bonus: 45.0,
            ramen_pt_weight: 5.0,
            ramen_effect_weight: 3.0,
            ramen_special_cost: 12.0,
            ramen_stock_cost: 0.4,
            region_xunlian_weight: 40.0,
            region_pt_weight: 30.0,
            region_hint_weight: 15.0,
            region_youqing_weight: 1.0,
            event_vital_weight: 2.2,
            event_motivation_weight: 40.0,
            event_bad_flag_penalty: 300.0
        }
    }
}

impl RamenPolicyConfig {
    /// 速度/智力特化档（速卡多的卡组适用；先于 default 验证调参通路）
    pub fn speed_build() -> Self {
        Self {
            // 放速度/智力的训练倾向与地区选择权重（通过 train_bias 体现）
            ..Self::default()
        }
    }
}

/// 候选动作的打分结果（内部结构，随意演进；不直接用于协议）
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RamenPolicyOutput {
    /// 综合得分（越大越优）
    pub score: f32,
    /// 评分分解（调参用，进入决策日志 score_breakdown 列）
    pub breakdown: Vec<(String, f32)>,
    /// 决策原因（人类可读，调试用）
    pub reason: String
}

impl RamenPolicyOutput {
    /// 追加一个分解项
    pub fn add(&mut self, key: &str, value: f32) {
        self.breakdown.push((key.to_string(), value));
    }
}

/// 手写策略核心：各阶段确定性打分与选择
///
/// 纯策略层：不持有可变状态、不修改 game，相同局面必然给出相同索引。
/// 各阶段候选列表由规则层生成（`list_*_actions`），本层只排序选择。
#[derive(Debug, Clone)]
pub struct RamenPolicy {
    /// 参数化配置
    pub config: RamenPolicyConfig
}

impl Default for RamenPolicy {
    fn default() -> Self {
        Self::new(RamenPolicyConfig::default())
    }
}

impl RamenPolicy {
    /// 创建策略（指定配置）
    pub fn new(config: RamenPolicyConfig) -> Self {
        Self { config }
    }

    /// 速度特化预设
    pub fn speed_build() -> Self {
        Self::new(RamenPolicyConfig::speed_build())
    }

    // ========== 阶段选择入口（确定性 argmax）==========

    /// Train 阶段决策：守门（自选比赛/生病/体力/心情）→ 否则按收益打分选最优
    ///
    /// 返回 `(选中索引, 各候选评分分解)`——评分供决策日志 breakdown 列（调参用）；
    /// 守门触发时评分列表为单元素（记录守门原因），不重复打分。
    pub fn decide_train(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        if actions.is_empty() {
            anyhow::bail!("Train 阶段候选为空");
        }
        let is_xiahesu = game.is_xiahesu();
        let uma = &game.uma;

        // 守门 0：自选比赛达标（优先于一切——不达标直接育成失败）
        if let Some(idx) = self.free_race_gate(game, actions) {
            return Ok((
                idx,
                vec![RamenPolicyOutput {
                    score: f32::MAX,
                    reason: format!("守门: {}", self.free_race_gate_reason(game)),
                    ..Default::default()
                }],
            ));
        }
        // 守门 1：生病 → 治病（夏合宿无治病候选，休息自动治病）
        if uma.flags.ill || uma.flags.bad_trainer {
            if let Some(idx) = actions
                .iter()
                .position(|a| a.operation == Operation::Clinic && !is_xiahesu)
            {
                return Ok((idx, vec![RamenPolicyOutput {
                    score: f32::MAX,
                    reason: "守门: 生病治病".to_string(),
                    ..Default::default()
                }]));
            }
            if is_xiahesu {
                if let Some(idx) = actions.iter().position(|a| a.operation == Operation::Rest) {
                    return Ok((idx, vec![RamenPolicyOutput {
                        score: f32::MAX,
                        reason: "守门: 夏合宿休息(自动治病)".to_string(),
                        ..Default::default()
                    }]));
                }
            }
        }
        // 守门 2：体力低 → 休息（防失败率崩盘；优先于心情、训练）
        if uma.vital < self.config.vital_rest {
            if let Some(idx) = actions.iter().position(|a| a.operation == Operation::Rest) {
                return Ok((idx, vec![RamenPolicyOutput {
                    score: f32::MAX,
                    reason: format!("守门: 体力{}<{}休息", uma.vital, self.config.vital_rest),
                    ..Default::default()
                }]));
            }
        }
        // 守门 3：心情低 → 外出（回干劲）
        if uma.motivation < self.config.motivation_outing {
            if let Some(idx) = actions
                .iter()
                .position(|a| matches!(a.operation, Operation::NormalOuting | Operation::FriendOuting))
            {
                return Ok((idx, vec![RamenPolicyOutput {
                    score: f32::MAX,
                    reason: format!("守门: 心情{}<{}外出", uma.motivation, self.config.motivation_outing),
                    ..Default::default()
                }]));
            }
        }

        // 打分选择
        let scores = self.score_train_actions(game, actions)?;
        Ok((argmax_index(&scores), scores))
    }

    /// 对所有 Train 阶段候选打分（守门通过后调用）
    pub fn score_train_actions(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<Vec<RamenPolicyOutput>> {
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            scores.push(self.score_train_action(game, a)?);
        }
        Ok(scores)
    }

    /// Train 阶段：守门（生病/体力/心情）→ 否则按收益打分选最优（仅索引）
    pub fn select_train(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        Ok(self.decide_train(game, actions)?.0)
    }

    /// RamenSelect 阶段：吃面收益（PT + 地区效果）与不吃面比较，贪心
    ///
    /// 返回 `(选中索引, 各候选评分分解)`。
    pub fn decide_ramen(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        if actions.is_empty() {
            anyhow::bail!("RamenSelect 阶段候选为空");
        }
        // 每个候选：不吃面固定 0 分；吃面按收益 - 成本
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            scores.push(self.score_ramen_action(game, a)?);
        }
        Ok((argmax_index(&scores), scores))
    }

    /// RamenSelect 阶段（仅索引）
    pub fn select_ramen(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        Ok(self.decide_ramen(game, actions)?.0)
    }

    /// SpecialSelect 阶段：最省隐藏风味（保留库存）；候选已按 sum(t) 升序，
    /// 这里显式按 -sum(targets) 打分，不依赖排序保证
    pub fn decide_special(
        &self, _game: &RamenGame, actions: &[RamenAction]
    ) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        if actions.is_empty() {
            anyhow::bail!("SpecialSelect 阶段候选为空");
        }
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            let used = a.special_targets.map(|t| t.iter().sum::<i32>()).unwrap_or(0) as f32;
            let mut out = RamenPolicyOutput {
                score: -used * self.config.ramen_special_cost,
                ..Default::default()
            };
            out.add("hidden_used", -used * self.config.ramen_special_cost);
            out.reason = format!("隐藏风味消耗 {used}");
            scores.push(out);
        }
        Ok((argmax_index(&scores), scores))
    }

    /// SpecialSelect 阶段（仅索引）
    pub fn select_special(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        Ok(self.decide_special(game, actions)?.0)
    }

    /// RegionSelect 阶段：按地区静态价值打分选组合（含第 3 年 120 组合全枚举，O(360) 便宜）
    pub fn decide_region(
        &self, game: &RamenGame, _year_idx: usize, actions: &[RamenAction]
    ) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        if actions.is_empty() {
            anyhow::bail!("RegionSelect 阶段候选为空");
        }
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            let Operation::RegionSelect(combo) = a.operation else {
                anyhow::bail!("RegionSelect 候选应携带 RegionSelect 操作");
            };
            let mut out = RamenPolicyOutput::default();
            for &rid in combo.iter() {
                out.score += self.score_region(game, rid)?;
            }
            out.reason = format!("{combo:?}");
            scores.push(out);
        }
        Ok((argmax_index(&scores), scores))
    }

    /// RegionSelect 阶段（仅索引）
    pub fn select_region(&self, game: &RamenGame, year_idx: usize, actions: &[RamenAction]) -> Result<usize> {
        Ok(self.decide_region(game, year_idx, actions)?.0)
    }

    /// 事件选项打分（返回各候选评分分解）
    pub fn decide_event(&self, game: &RamenGame, choices: &[Vec<EventChoice>]) -> Result<(usize, Vec<RamenPolicyOutput>)> {
        if choices.is_empty() {
            anyhow::bail!("事件候选为空");
        }
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(choices.len());
        for c in choices {
            let mut out = RamenPolicyOutput::default();
            for choice in c {
                out.score += self.score_event_choice(game, choice)?;
            }
            scores.push(out);
        }
        Ok((argmax_index(&scores), scores))
    }

    /// 事件选项打分（仅索引）
    pub fn select_event(&self, game: &RamenGame, choices: &[Vec<EventChoice>]) -> Result<usize> {
        Ok(self.decide_event(game, choices)?.0)
    }

    // ========== 自选比赛（free_race）==========

    /// 自选比赛硬守门：区间内剩余可比赛回合已不够补齐缺口时，强制返回「比赛」候选索引
    ///
    /// 自选比赛不达标会在 `BaseGame::check_free_race` 判定育成失败，损失远大于任何一次训练，
    /// 因此该守门排在生病/体力/心情之前。返回 `None` 表示无需干预（无要求 / 已达标 / 仍宽裕 /
    /// 本回合等级不满足 / 剩余有效回合不足（摆烂））。
    ///
    /// 等级语义：`free.mask` 已按 `race_grades[i] <= grade` 过滤（grade=2 表示 G2 及以上，
    /// G3 不计数），见 [`FreeRaceData::update_turn_mask`](crate::gamedata::FreeRaceData::update_turn_mask)。
    /// 当前回合若不在 mask 内（等级不满足），打了也不计入达标——强制无意义，不干预；
    /// 剩余有效回合少于缺口（即使全部打完也补不齐，摆烂）时同样不强制，
    /// 由正常打分决策并记录原因（`free_race_gate_reason` 进决策日志 breakdown）。
    fn free_race_gate(&self, game: &RamenGame, actions: &[RamenAction]) -> Option<usize> {
        let free = game.uma.find_free_race(game.turn())?;
        let need = free.count.saturating_sub(game.uma.count_free_race(free));
        // 达标后直到区间结束不再干预（软倾向同理由 `score_race` 降级为普通比赛分）
        if need == 0 {
            return None;
        }
        let remain = remaining_race_slots(game.turn(), free);
        // 摆烂：剩余有效回合少于缺口，打完也补不齐 → 不再强制（原因进决策日志）
        if remain < need {
            return None;
        }
        if remain > need + self.config.race_gate_slack {
            return None;
        }
        // 本回合等级不满足：打了不计数，不强制（留给后续有效回合）
        if !race_turn_qualified(game.turn(), free) {
            return None;
        }
        actions.iter().position(|a| a.operation == Operation::Race)
    }

    /// 自选比赛守门的详细原因（供决策日志 breakdown 记录摆烂/强制情形）
    fn free_race_gate_reason(&self, game: &RamenGame) -> String {
        let Some(free) = game.uma.find_free_race(game.turn()) else {
            return "无自选比赛要求".to_string();
        };
        let need = free.count.saturating_sub(game.uma.count_free_race(free));
        let remain = remaining_race_slots(game.turn(), free);
        let grade_note = match free.grade {
            Some(g) => {
                let name = ["?", "G1", "G2", "G3", "OP"][g.min(4) as usize];
                format!("要求{name}及以上")
            }
            None => "无等级要求".to_string()
        };
        if need == 0 {
            format!("自选比赛已达标({grade_note}), 区间剩余回合不再干预")
        } else if remain < need {
            format!(
                "自选比赛缺{need}场({grade_note})但只剩{remain}个有效回合, 打完也不够(摆烂), 不再强制"
            )
        } else if !race_turn_qualified(game.turn(), free) {
            format!("自选比赛缺{need}场({grade_note}), 本回合等级不满足, 不白打")
        } else {
            format!(
                "自选比赛缺{need}场/剩{remain}回合({grade_note}), 剩余回合不足需强制补赛",
            )
        }
    }

    /// 自选比赛打分
    ///
    /// - 有未补齐缺口且**本回合等级满足**：`urgency × 缺口 / 剩余可比赛回合`——区间宽裕时
    ///   接近 0（不打扰训练），越接近截止越高，与硬守门形成「软倾向 + 硬兜底」两层。
    /// - 本回合等级不满足：打也不计数，不给 urgency（降级为普通比赛分，避免白打浪费回合）。
    /// - 剩余有效回合少于缺口（摆烂，打完也不够）：同样不给 urgency（打了也白打）。
    /// - 无要求或已达标：退化为按比赛等级折算（`race_grade_weight`，默认 0）。
    fn score_race(&self, game: &RamenGame) -> (f32, String) {
        if let Some(free) = game.uma.find_free_race(game.turn()) {
            let need = free.count.saturating_sub(game.uma.count_free_race(free));
            if need > 0 && race_turn_qualified(game.turn(), free) {
                let remain = remaining_race_slots(game.turn(), free);
                // 摆烂：剩余有效回合少于缺口，打完也不够 → 不再引导比赛
                if remain < need {
                    return (0.0, format!("自选比赛(缺{need}场/剩{remain}回合,摆烂)"));
                }
                let val = self.config.race_free_urgency_weight * need as f32 / remain as f32;
                return (val, format!("自选比赛(缺{need}场/剩{remain}回合)"));
            }
        }
        let grade = game_race_grade(game);
        (grade * self.config.race_grade_weight, format!("比赛(等级{grade})"))
    }

    // ========== Train 动作打分 ==========

    /// 对单个 Train 阶段动作打分
    fn score_train_action(&self, game: &RamenGame, a: &RamenAction) -> Result<RamenPolicyOutput> {
        let mut out = RamenPolicyOutput::default();
        match a.operation {
            Operation::Train(t) => {
                let train = t as usize;
                let buffs = game.calc_training_buff(train)?;
                let value = game.calc_training_value(&buffs, train)?;
                let fail_rate = game.calc_training_failure_rate(&buffs, train);
                // 属性增益（five_status_final_score 差分，与 calc_score 一致）
                let mut attr_gain = 0.0;
                for i in 0..5 {
                    attr_gain += self.status_gain(game, i, value.status_pt[i]);
                }
                let pt_gain = value.status_pt[5] as f32;
                // 注：`status_gain` 内部已乘 status_rate，此处不可再乘（否则成平方）
                let attr = attr_gain;
                let pt = pt_gain * self.config.pt_rate;
                // 体力成本（消耗按 train_vital_value 折算）
                let vital_cost = (-value.vital).max(0) as f32 * self.config.train_vital_value;
                let shining = game.shining_count(train) as f32 * self.config.shining_bonus;
                // 失败的期望损失：成功时才有的收益 × 失败率 + 固定失败惩罚 × 失败率
                let fail_p = fail_rate / 100.0;
                let gross = attr + pt - vital_cost + shining;
                let fail_adj = -(gross * fail_p + self.config.failure_penalty * fail_p);
                // breakdown 各项之和 == score（调参日志需自洽，见 test_breakdown_sums_to_score）
                out.add("attr", attr);
                out.add("pt", pt);
                out.add("vital_cost", -vital_cost);
                out.add("shining", shining);
                out.add("fail_adj", fail_adj);
                out.score = gross + fail_adj;
                out.reason = format!(
                    "{}训练 失败率{fail_rate:.0}% 属性+{attr_gain:.0} PT+{pt_gain:.0}",
                    global!(GAMECONSTANTS).train_names[train]
                );
            }
            Operation::Race => {
                let (val, reason) = self.score_race(game);
                out.add("race", val);
                out.score = val;
                out.reason = reason;
            }
            Operation::Rest => {
                // 休息价值：恢复体力×边际价值 + 基础值（体力越低越值）
                let need = (self.config.rest_target_vital - game.uma.vital).max(0) as f32;
                let val = self.config.rest_base + need * self.config.rest_vital_value;
                out.add("rest", val);
                out.score = val;
                out.reason = "休息".to_string();
            }
            Operation::NormalOuting => {
                out.add("outing", self.config.outing_base);
                out.score = self.config.outing_base;
                out.reason = "普通外出".to_string();
            }
            Operation::FriendOuting => {
                let val = self.config.outing_base + self.config.friend_outing_bonus;
                out.add("outing", self.config.outing_base);
                out.add("friend", self.config.friend_outing_bonus);
                out.score = val;
                out.reason = "友人出行".to_string();
            }
            Operation::Clinic => {
                // 健康时治病无收益（生病由守门规则直通，这里给 0 分避免误选）
                out.reason = "治病".to_string();
            }
            Operation::RegionSelect(_) | Operation::StageOnly => {
                anyhow::bail!("Train 阶段不应出现 RegionSelect/StageOnly 操作");
            }
        }
        Ok(out)
    }

    /// 单维属性增量的评分（按 five_status_final_score 差分）
    fn status_gain(&self, game: &RamenGame, i: usize, inc: i32) -> f32 {
        let cons = global!(GAMECONSTANTS);
        let cur = game.uma.five_status[i].min(game.uma.five_status_limit[i]).max(0) as usize;
        let next = (cur + inc as usize).min(game.uma.five_status_limit[i] as usize);
        let cur_score = cons.five_status_final_score.get(cur).copied().unwrap_or(0) as f32;
        let next_score = cons.five_status_final_score.get(next).copied().unwrap_or(0) as f32;
        (next_score - cur_score) * self.config.status_rate
    }

    // ========== RamenSelect 动作打分 ==========

    /// 对单个 RamenSelect 动作打分（不吃面 0 分；吃面按 PT + 效果 - 成本）
    fn score_ramen_action(&self, game: &RamenGame, a: &RamenAction) -> Result<RamenPolicyOutput> {
        let mut out = RamenPolicyOutput::default();
        let Some(region_id) = a.ramen else {
            out.reason = "不吃面".to_string();
            return Ok(out);
        };
        // PT 增益（当年已吃次数 eat_count 计入）
        let year_idx = (game.current_year() - 1) as usize;
        let pt_gain = calc_ramen_pt_gain(year_idx, game.ramen.eat_count)? as f32;
        out.add("pt_gain", pt_gain * self.config.ramen_pt_weight);
        // 地区效果（训练加成、PT 加成、hint）
        let region = RAMENDATA
            .get()
            .and_then(|d| d.ramen_region_effect.get(region_id))
            .ok_or_else(|| anyhow::anyhow!("地区效果缺失: region_id={region_id}"))?;
        let effect_val = region.xunlian as f32 * self.config.ramen_effect_weight
            + region.pt_bonus as f32 * self.config.ramen_pt_weight
            + region.hint_count as f32 * self.config.region_hint_weight;
        out.add("region_effect", effect_val);
        // 成本：隐藏风味消耗（由 targets 决定）+ 诀窍库存机会成本
        let hidden = a.special_targets.map(|t| t.iter().sum::<i32>()).unwrap_or(0) as f32;
        let stock_cost =
            self.config.ramen_stock_cost * 5.0 * (1.0 - (game.ramen.special_feeling as f32 + 4.0) / 12.0).max(0.2);
        out.add("hidden_cost", -hidden * self.config.ramen_special_cost);
        out.add("stock_cost", -stock_cost);
        out.score =
            pt_gain * self.config.ramen_pt_weight + effect_val - hidden * self.config.ramen_special_cost - stock_cost;
        out.reason = format!("吃面/{}", region.name);
        Ok(out)
    }

    // ========== RegionSelect 单地区价值 ==========

    /// 单个地区的静态价值（xunlian×训练倾向 + pt_bonus + hint）
    fn score_region(&self, game: &RamenGame, region_id: usize) -> Result<f32> {
        let region = RAMENDATA
            .get()
            .and_then(|d| d.ramen_region_effect.get(region_id))
            .ok_or_else(|| anyhow::anyhow!("地区效果缺失: region_id={region_id}"))?;
        // 训练倾向：卡组中每种训练类型的卡数量（卡组派生系数；友人/团队卡不计）
        let mut bias = [0.0f32; 5];
        for card in game.deck.iter() {
            let t = card.data.card_type;
            if (0..5).contains(&t) {
                bias[t as usize] += 1.0;
            }
        }
        // 该地区覆盖的训练位在卡组里的分量；0.5 下限避免「一张都没有」直接归零
        let bias_sum: f32 = region
            .at_trains
            .iter()
            .map(|&t| {
                let t = t as usize;
                if t < 5 { bias[t].max(0.5) } else { 0.0 }
            })
            .sum();
        // xunlian（第 1 年）与 youqing（第 2/3 年）都按 bias_sum 缩放：
        // 第 2/3 年地区的 xunlian 恒为 0，若只算 xunlian 则同年所有候选同分、
        // argmax 恒取第一个，卡组构成完全不参与决策。
        Ok(bias_sum
            * (region.xunlian as f32 * self.config.region_xunlian_weight
                + region.youqing as f32 * self.config.region_youqing_weight)
            + region.pt_bonus as f32 * self.config.region_pt_weight
            + region.hint_count as f32 * self.config.region_hint_weight)
    }

    // ========== Event 打分 ==========

    /// 单个事件选项的效果折算（含 flags 修正）
    fn score_event_choice(&self, game: &RamenGame, c: &EventChoice) -> Result<f32> {
        // prob=0 视为必触发（与规则层语义一致）
        let prob = if c.prob == 0 { 100.0 } else { c.prob as f32 };
        let mut val = 0.0;
        for i in 0..5 {
            val += self.status_gain(game, i, c.value.status_pt[i]);
        }
        val += c.value.status_pt[5] as f32 * self.config.pt_rate;
        val += c.value.vital as f32 * self.config.event_vital_weight;
        val += c.value.motivation as f32 * self.config.event_motivation_weight;
        // flags：ill/bad_trainer 是坏状态，获得惩罚、移除奖励
        if let Some(flags) = &c.add_flags {
            if flags.ill || flags.bad_trainer {
                val -= self.config.event_bad_flag_penalty;
            }
        }
        if let Some(flags) = &c.remove_flags {
            if flags.ill || flags.bad_trainer {
                val += self.config.event_bad_flag_penalty;
            }
        }
        Ok(val * prob / 100.0)
    }
}

/// 确定性 argmax：取最高分候选；平局取索引最小（候选固定顺序，不依赖 HashMap）
fn argmax_index(scores: &[RamenPolicyOutput]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.score.total_cmp(&b.score).then_with(|| ib.cmp(ia)))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// 当前回合是否满足自选比赛区间的等级要求（mask 含当前回合 bit）
///
/// `free.mask` 的 bit *b* 对应回合 *b+11*，且已按 `race_grades[i] <= grade` 过滤
/// （如 grade=2 → G1/G2 计入、G3 不计）。本回合不在 mask 内时打了也不计入
/// `count_free_race`，策略不应为达标目的打它。
fn race_turn_qualified(turn: i32, free: &FreeRaceData) -> bool {
    let bit = turn - 11;
    if bit < 0 || bit >= 64 {
        return false;
    }
    free.mask & (1u64 << bit) != 0
}

/// 自选比赛区间内「当前回合及以后」还剩多少个可比赛回合
///
/// `FreeRaceData::mask` 已按区间与等级要求预置（bit *b* 对应回合 *b+11*）。
/// 这里再叠加 `BaseGame::can_self_race` 的通用限制（回合 13-71 才可自选比赛），
/// 即去掉 bit 0-1（回合 11-12）与 bit ≥ 61（回合 ≥ 72）。
pub(crate) fn remaining_race_slots(turn: i32, free: &FreeRaceData) -> u32 {
    /// 回合 13 起才可自选比赛（bit 2）
    const LOW_CUT: u64 = !0b11;
    /// 回合 72 起进入 URA，不可自选比赛（bit 61 及以上）
    const HIGH_CUT: u64 = (1u64 << 61) - 1;

    let mut mask = free.mask & LOW_CUT & HIGH_CUT;
    let lo = (turn - 11).max(0);
    if lo >= 64 {
        return 0;
    }
    mask &= !0u64 << lo;
    mask.count_ones()
}

/// 当前回合比赛等级（0 = 无比赛；等级越高越好）
fn game_race_grade(game: &RamenGame) -> f32 {
    let turn = game.turn();
    let grades = &global!(GAMECONSTANTS).race_grades;
    if turn >= 72 {
        // URA 回合固定 G1（表外）
        return 4.0;
    }
    grades.get(turn as usize).copied().unwrap_or(0).max(0) as f32
}

/// 地区选择策略：固定顺序
///
/// 每年从可选地区中按固定顺序选择前 3 个。
/// - 第 1 年（year_idx=0）：选择 [0, 1, 2]（札幌、函馆、新潟）
/// - 第 2 年（year_idx=1）：选择 [5, 6, 7]（中山、中京、京都）
/// - 第 3 年（year_idx=2）：选择 [10, 11, 12]（札幌、函馆、新潟）
pub fn fixed_region_selection(year_idx: usize) -> Result<[usize; 3]> {
    let range = get_region_range(year_idx)?;
    if range.len() < 3 {
        anyhow::bail!("可选地区不足 3 个: year_idx={year_idx}, range={range:?}");
    }
    Ok([range[0], range[1], range[2]])
}

/// 超级拉面选择策略：固定选项二
///
/// 初期固定选择 `training_limit_options` 的第二个选项（索引 1）。
/// 返回选项对应的训练位置列表。
pub fn fixed_super_ramen_selection() -> Result<Vec<i32>> {
    let options = get_super_ramen_clone_train_options()?;
    if options.len() < 2 {
        anyhow::bail!("超级拉面选项不足 2 个");
    }
    Ok(options[1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::ramen::TrainingType,
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    #[test]
    fn test_fixed_region_selection() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 第 1 年：选择 [0, 1, 2]
        let sel = fixed_region_selection(0)?;
        println!("第1年固定选择: {sel:?}");
        assert_eq!(sel, [0, 1, 2]);

        // 第 2 年：选择 [5, 6, 7]
        let sel = fixed_region_selection(1)?;
        println!("第2年固定选择: {sel:?}");
        assert_eq!(sel, [5, 6, 7]);

        // 第 3 年：选择 [10, 11, 12]
        let sel = fixed_region_selection(2)?;
        println!("第3年固定选择: {sel:?}");
        assert_eq!(sel, [10, 11, 12]);

        // 无效 year_idx
        assert!(fixed_region_selection(3).is_err());
        println!("无效 year_idx 验证通过");

        Ok(())
    }

    #[test]
    fn test_fixed_super_ramen_selection() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let sel = fixed_super_ramen_selection()?;
        println!("超级拉面固定选择(选项二): {sel:?}");
        // 选项2: 速/耐/力/智 [0,1,2,4]
        assert_eq!(sel, vec![0, 1, 2, 4]);

        Ok(())
    }

    // ========== 手写策略核心测试 ==========

    /// 第 3 年地区选择必须随卡组构成变化（build 自适应）
    ///
    /// 速度向卡组（3速） vs 智力向卡组（3智），第 3 年 120 组合全枚举打分，
    /// 两者选中的组合必须不同。第 3 年地区的 `xunlian` 恒为 0，区分度只能来自
    /// `youqing × at_trains × 卡组 bias`——若 `score_region` 漏掉 youqing 项，
    /// 同年所有候选同分、argmax 恒取第一个，本测试即失败。
    #[test]
    fn test_region_build_sensitivity() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let combos = crate::game::ramen::rules::get_region_combinations(2)?;
        let actions: Vec<RamenAction> = combos
            .iter()
            .map(|&c| RamenAction::no_ramen(Operation::RegionSelect(c)))
            .collect();
        let policy = RamenPolicy::default();

        // 速度向卡组（现有测试卡组：3速1智1耐1友）
        let game_speed = make_game()?;
        let (idx_s, outs_s) = policy.decide_region(&game_speed, 2, &actions)?;
        println!(
            "速度向卡组 → 第3年选中: {:?} score={:.0}",
            combos[idx_s], outs_s[idx_s].score
        );

        // 智力向卡组：从卡表取 3 张智卡 + 2 张速卡 + 新友人卡（拉面杯要求 30305 系列）
        use crate::gamedata::GAMEDATA;
        let gd = global!(GAMEDATA);
        let pick = |card_type: i32, n: usize| -> Vec<u32> {
            gd.card
                .values()
                .filter(|c| c.card_type == card_type)
                .map(|c| c.card_id * 10 + c.rarity)
                .take(n)
                .collect()
        };
        let mut deck_w: Vec<u32> = pick(4, 3);
        deck_w.extend(pick(0, 2));
        deck_w.push(303054); // 新友人卡
        while deck_w.len() < 6 {
            deck_w.push(302424);
        }
        let deck_w: [u32; 6] = deck_w.try_into().expect("卡组长度 6");
        let game_wise = RamenGame::newgame(102601, &deck_w, crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        })?;
        let (idx_w, outs_w) = policy.decide_region(&game_wise, 2, &actions)?;
        println!(
            "智力向卡组 {:?} → 第3年选中: {:?} score={:.0}",
            deck_w, combos[idx_w], outs_w[idx_w].score
        );

        println!("两者选择是否不同: {}", combos[idx_s] != combos[idx_w]);
        assert_ne!(combos[idx_s], combos[idx_w], "不同 build 必须选出不同的第 3 年地区组合");
        Ok(())
    }

    /// 构造一个可用的 RamenGame（默认卡组 102601，train 阶段可打分）
    fn make_game() -> anyhow::Result<RamenGame> {
        use crate::game::ramen::RamenGame;
        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        Ok(RamenGame::newgame(102601, &deck, inherit)?)
    }

    fn train_actions() -> Vec<RamenAction> {
        vec![
            RamenAction::new(Operation::Train(TrainingType::Speed)),
            RamenAction::new(Operation::Train(TrainingType::Stamina)),
            RamenAction::new(Operation::Train(TrainingType::Power)),
            RamenAction::new(Operation::Rest),
            RamenAction::new(Operation::NormalOuting),
            RamenAction::new(Operation::Clinic),
        ]
    }

    /// 守门 1：生病时必须治病（优先于训练/休息）
    #[test]
    fn test_gate_ill_clinic() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.flags.ill = true;
        game.uma.vital = 100;
        let policy = RamenPolicy::default();
        let idx = policy.select_train(&game, &train_actions())?;
        println!("生病时选择: {}", train_actions()[idx]);
        assert_eq!(train_actions()[idx].operation, Operation::Clinic);
        Ok(())
    }

    /// 守门 2：体力低时必须休息（优先于训练）
    #[test]
    fn test_gate_vital_low_rest() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.vital = 30;
        let policy = RamenPolicy::default();
        let idx = policy.select_train(&game, &train_actions())?;
        println!("体力 30 时选择: {}", train_actions()[idx]);
        assert_eq!(train_actions()[idx].operation, Operation::Rest);
        Ok(())
    }

    /// 守门 3：心情低时必须外出
    #[test]
    fn test_gate_motivation_low_outing() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.motivation = 2;
        game.uma.vital = 100;
        let policy = RamenPolicy::default();
        let idx = policy.select_train(&game, &train_actions())?;
        println!("心情 2 时选择: {}", train_actions()[idx]);
        assert_eq!(train_actions()[idx].operation, Operation::NormalOuting);
        Ok(())
    }

    /// 正常局面：确定性 argmax，两次调用一致且不选治病/外出
    #[test]
    fn test_train_selector_deterministic() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.motivation = 4;
        game.uma.vital = 80;
        let policy = RamenPolicy::default();
        let actions = train_actions();
        let idx1 = policy.select_train(&game, &actions)?;
        let idx2 = policy.select_train(&game, &actions)?;
        println!("健康局面两次选择: {} / {}", actions[idx1], actions[idx2]);
        assert_eq!(idx1, idx2);
        assert!(matches!(actions[idx1].operation, Operation::Train(_)));
        Ok(())
    }

    /// SpecialSelect：选隐藏风味消耗最小的候选
    #[test]
    fn test_special_selector_min_hidden() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let policy = RamenPolicy::default();
        let actions = vec![
            RamenAction::special_select(0, [2, 0, 0]), // 用 2 个隐藏风味
            RamenAction::special_select(0, [0, 1, 0]), // 用 1 个
            RamenAction::special_select(0, [0, 0, 1]), // 用 1 个
            RamenAction::special_select(0, [0, 0, 0]), // 不用
        ];
        let idx = policy.select_special(&game, &actions)?;
        println!("SpecialSelect 选择: {:?}", actions[idx].special_targets);
        assert_eq!(actions[idx].special_targets, Some([0, 0, 0]));
        Ok(())
    }

    /// Event：选总分更高的选项（PT 100 vs PT 20）
    #[test]
    fn test_event_selector_higher_value() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let policy = RamenPolicy::default();
        use crate::gamedata::ActionValue;
        let low = EventChoice {
            prob: 100,
            value: ActionValue {
                status_pt: [0, 0, 0, 0, 0, 20],
                ..Default::default()
            },
            ..Default::default()
        };
        let high = EventChoice {
            prob: 100,
            value: ActionValue {
                status_pt: [0, 0, 0, 0, 0, 100],
                ..Default::default()
            },
            ..Default::default()
        };
        let choices = vec![vec![low.clone()], vec![high.clone()]];
        let idx = policy.select_event(&game, &choices)?;
        println!("事件选择: 第 {} 组", idx + 1);
        assert_eq!(idx, 1); // 高分组
        assert_ne!(low, high);
        Ok(())
    }

    /// RegionSelect：所有组合均可打分，返回合法索引；确定性
    #[test]
    fn test_region_selector_valid_and_deterministic() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_game()?;
        let policy = RamenPolicy::default();
        // 第 1 年 10 个组合
        let combos = crate::game::ramen::rules::get_region_combinations(0)?;
        let actions: Vec<RamenAction> = combos
            .iter()
            .map(|&c| RamenAction::no_ramen(Operation::RegionSelect(c)))
            .collect();
        let idx1 = policy.select_region(&game, 0, &actions)?;
        let idx2 = policy.select_region(&game, 0, &actions)?;
        println!("地区选择 idx={idx1} 组合={:?}", combos[idx1]);
        assert_eq!(idx1, idx2);
        assert!(idx1 < actions.len());
        Ok(())
    }

    // ========== 自选比赛守门 / 打分自洽性 ==========

    /// 构造带自选比赛要求的 RamenGame（无声铃鹿 100201：回合 12-26 需 1 场）
    fn make_free_race_game() -> anyhow::Result<RamenGame> {
        make_free_race_game_uma(100201)
    }

    /// 构造指定马娘的自选比赛测试局（默认卡组；102601 无要求 / 100201 无等级 /
    /// 101901 回合 46-59 需 3 场 G1 及以上）
    fn make_free_race_game_uma(uma_id: u32) -> anyhow::Result<RamenGame> {
        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30]
        };
        let deck = [302424, 302894, 303044, 302924, 303024, 303054];
        Ok(RamenGame::newgame(uma_id, &deck, inherit)?)
    }

    /// 带「比赛」候选的 Train 阶段候选表
    fn train_actions_with_race() -> Vec<RamenAction> {
        let mut acts = train_actions();
        acts.push(RamenAction::new(Operation::Race));
        acts
    }

    /// `remaining_race_slots`：按当前回合裁剪，并排除回合 11-12 与 URA 回合
    #[test]
    fn test_remaining_race_slots() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        // 区间 11-26 无等级限制：mask 覆盖 bit 0-15（回合 11-26）
        let mut free = FreeRaceData {
            start_turn: 11,
            end_turn: 26,
            count: 1,
            grade: None,
            mask: 0
        };
        free.update_turn_mask();
        for (turn, expect) in [(0, 14), (13, 14), (20, 7), (25, 2), (26, 1), (27, 0)] {
            let got = remaining_race_slots(turn, &free);
            println!("turn={turn} 剩余可比赛回合={got}（期望 {expect}）");
            assert_eq!(got, expect);
        }
        // 回合 11、12 被 can_self_race 排除：区间共 16 回合，可用只有 14
        Ok(())
    }

    /// 硬守门：缺口紧张时强制比赛，宽裕 / 已达标时不干预
    #[test]
    fn test_free_race_gate() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions_with_race();

        // 回合 20：剩余 7 个可比赛回合 > 缺口 1 + slack 1 → 不干预
        let mut game = make_free_race_game()?;
        game.base.turn = 20;
        let idx = policy.free_race_gate(&game, &actions);
        println!("回合 20 守门结果: {idx:?}");
        assert_eq!(idx, None);

        // 回合 25：只剩 2 个可比赛回合 ≤ 1 + 1 → 强制比赛
        game.base.turn = 25;
        let idx = policy
            .free_race_gate(&game, &actions)
            .ok_or_else(|| anyhow::anyhow!("回合 25 应触发自选比赛守门"))?;
        println!("回合 25 守门选择: {}", actions[idx]);
        assert_eq!(actions[idx].operation, Operation::Race);

        // 已打过 1 场（达标）→ 不再干预
        game.uma.set_race(20);
        let idx = policy.free_race_gate(&game, &actions);
        println!("达标后守门结果: {idx:?}");
        assert_eq!(idx, None);

        // 无自选比赛要求的马娘（102601）任何回合都不干预
        let mut plain = make_game()?;
        plain.base.turn = 25;
        println!("无要求马娘守门结果: {:?}", policy.free_race_gate(&plain, &actions));
        assert_eq!(policy.free_race_gate(&plain, &actions), None);
        Ok(())
    }

    /// 等级过滤：`race_turn_qualified` 正确判定本回合是否满足区间等级要求
    ///
    /// 语义：mask 按 `race_grades[i] <= grade` 过滤。grade=1 → 仅 G1 回合有效；
    /// grade=2 → G1/G2 有效（G3 及以下不算）。用 101901（回合 46-59 需 G1）验证。
    #[test]
    fn test_race_turn_qualified() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let game = make_free_race_game_uma(101901)?;
        let free = game.uma.find_free_race(50).expect("101901 回合 50 应在区间 46-59 内");
        println!("101901 free_race: {free:?}");
        assert_eq!(free.grade, Some(1)); // G1 及以上

        // 回合 46/48/49/52/54/57 为 G2/G3（race_grades > 1）→ 不满足
        for turn in [46, 48, 49, 52, 54, 57] {
            let ok = race_turn_qualified(turn, free);
            println!("turn={turn} (非G1) qualified={ok}");
            assert!(!ok, "turn={turn} 不应满足 G1 要求");
        }
        // 回合 47/50/51/53/55/56/58/59 为 G1 → 满足
        for turn in [47, 50, 51, 53, 55, 56, 58, 59] {
            let ok = race_turn_qualified(turn, free);
            println!("turn={turn} (G1) qualified={ok}");
            assert!(ok, "turn={turn} 应满足 G1 要求");
        }
        Ok(())
    }

    /// 等级不满足回合不强制：G2 回合即使剩余回合紧张也不强制补赛（打了不计数，白打）
    #[test]
    fn test_free_race_gate_skips_nonqualified_turn() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions_with_race();

        // 101901 回合 46-59 需 3 场 G1；已打 1 场有效（turn 47）→ 缺口 2。
        // 回合 57（G2，等级不满足）：剩余有效回合 58/59 共 2 ≤ 缺口2+slack1 → 若等级满足会强制
        let mut game = make_free_race_game_uma(101901)?;
        game.uma.set_race(47);
        game.base.turn = 57;
        let idx = policy.free_race_gate(&game, &actions);
        println!("回合 57 (G2) 守门结果: {idx:?}（应为 None，不打白打）");
        assert_eq!(idx, None);

        // 同一局回合 58（G1，等级满足）：剩余有效回合 58/59 共 2 ≤ 缺口2+slack1 → 强制
        game.base.turn = 58;
        let idx = policy
            .free_race_gate(&game, &actions)
            .ok_or_else(|| anyhow::anyhow!("回合 58 (G1) 应触发强制补赛"))?;
        println!("回合 58 (G1) 守门选择: {}", actions[idx]);
        assert_eq!(actions[idx].operation, Operation::Race);
        Ok(())
    }

    /// 达标后区间内不再干预：缺口清零后任何回合（含最后回合）都不强制
    #[test]
    fn test_free_race_gate_quiet_after_done() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions_with_race();

        // 101901：回合 46-59 需 3 场 G1。已打满 3 场有效比赛 → 达标
        let mut game = make_free_race_game_uma(101901)?;
        for turn in [47, 50, 51] {
            game.uma.set_race(turn);
        }
        for turn in 52..=59 {
            game.base.turn = turn;
            let idx = policy.free_race_gate(&game, &actions);
            println!("达标后回合 {turn} 守门结果: {idx:?}");
            assert_eq!(idx, None, "达标后回合 {turn} 不应再触发强制比赛");
        }
        Ok(())
    }

    /// 摆烂：有效回合打光仍未补齐时不强制，且守门原因完整记录（进决策日志 breakdown）
    #[test]
    fn test_free_race_gate_giveup_recorded() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions_with_race();

        // 101901：缺口 3，但已打到回合 59 仍未打满（只打了 1 场）→ 摆烂
        let mut game = make_free_race_game_uma(101901)?;
        game.uma.set_race(47); // 只打了 1 场有效比赛
        game.base.turn = 59;
        let idx = policy.free_race_gate(&game, &actions);
        println!("回合 59 缺口未补齐守门结果: {idx:?}（最后回合仍缺口 → 不强制摆烂）");
        assert_eq!(idx, None);

        // 守门原因文本完整（含缺口/等级要求/摆烂说明），供决策日志 breakdown 记录
        let reason = policy.free_race_gate_reason(&game);
        println!("守门原因: {reason}");
        assert!(reason.contains("缺2场"), "原因应含缺口: {reason}");
        assert!(reason.contains("摆烂"), "原因应说明摆烂: {reason}");

        // 等级不满足时的原因（回合 54 G2，还有有效回合）
        game.base.turn = 54;
        let reason = policy.free_race_gate_reason(&game);
        println!("等级不满足原因: {reason}");
        assert!(reason.contains("等级不满足"), "原因应说明等级不满足: {reason}");
        Ok(())
    }

    /// 小栗帽 100603 专项：两段区间 + 限 G1 的守门行为
    ///
    /// 该马娘是采样空间里最硬的用例——第二段要求回合 48-59 内打满 2 场 G1，
    /// 可比赛回合数远少于区间长度。逐回合扫描而非硬编码回合号，
    /// 使测试不随 `race_grades` 常量表调整而失效。
    #[test]
    fn test_free_race_gate_oguri_two_intervals() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions_with_race();
        let mut game = make_free_race_game_uma(100603)?;

        // 两段区间必须从 DB 正确读出
        let intervals: Vec<(u32, u32, u32, Option<u32>)> = game
            .uma
            .get_data()?
            .free_races
            .iter()
            .map(|f| (f.start_turn, f.end_turn, f.count, f.grade))
            .collect();
        println!("100603 自选比赛区间: {intervals:?}");
        assert_eq!(intervals.len(), 2);
        assert_eq!((intervals[0].0, intervals[0].1, intervals[0].2), (12, 23, 1));
        assert_eq!((intervals[1].0, intervals[1].1, intervals[1].2), (48, 59, 2));
        assert!(intervals[1].3.is_some(), "第二段应带等级限制");

        // 限 G1 使第二段的可比赛回合数显著少于区间长度（12 回合）
        let free2 = game
            .uma
            .find_free_race(48)
            .ok_or_else(|| anyhow::anyhow!("回合 48 应落在第二段区间内"))?;
        let slots2 = remaining_race_slots(48, free2);
        println!("第二段（48-59，限 G1）可比赛回合数: {slots2}");
        assert!(slots2 >= 2, "可比赛回合数不足以打满 2 场，规则或掩码有误");
        assert!(slots2 < 12, "限 G1 未生效：可比赛回合数不应等于区间长度");

        // 逐回合扫描第一段：找到守门首次触发的回合
        let first_gate = (12..=23).find(|&turn| {
            game.base.turn = turn;
            policy.free_race_gate(&game, &actions).is_some()
        });
        let first_gate = first_gate.ok_or_else(|| anyhow::anyhow!("第一段守门从未触发"))?;
        println!("第一段守门首次触发回合: {first_gate}");
        game.base.turn = first_gate;
        let idx = policy
            .free_race_gate(&game, &actions)
            .ok_or_else(|| anyhow::anyhow!("守门应返回候选下标"))?;
        assert_eq!(actions[idx].operation, Operation::Race);

        // 打满第一段后，第一段区间内不再干预
        game.uma.set_race(first_gate);
        game.base.turn = 23;
        println!("第一段达标后回合 23 守门: {:?}", policy.free_race_gate(&game, &actions));
        assert_eq!(policy.free_race_gate(&game, &actions), None);

        // 第二段缺 2 场：扫描触发回合，并确认返回比赛
        let second_gate = (48..=59).find(|&turn| {
            game.base.turn = turn;
            policy.free_race_gate(&game, &actions).is_some()
        });
        let second_gate = second_gate.ok_or_else(|| anyhow::anyhow!("第二段守门从未触发"))?;
        println!("第二段（缺 2 场）守门首次触发回合: {second_gate}");
        game.base.turn = second_gate;
        let idx = policy
            .free_race_gate(&game, &actions)
            .ok_or_else(|| anyhow::anyhow!("第二段守门应返回候选下标"))?;
        assert_eq!(actions[idx].operation, Operation::Race);
        Ok(())
    }

    /// 守门在候选表不含「比赛」时必须返回 None，不得 panic
    ///
    /// 生病 / 体力不足等情形下 `Operation::Race` 可能不在候选中，
    /// 此时守门只能放弃干预，交由规则层判定育成失败，而不是越界取下标。
    #[test]
    fn test_free_race_gate_without_race_candidate() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();
        let actions = train_actions(); // 不含 Operation::Race
        let mut game = make_free_race_game_uma(100603)?;
        // 推到第一段最后一个回合，缺口最紧张
        game.base.turn = 23;
        let got = policy.free_race_gate(&game, &actions);
        println!("无比赛候选时守门结果: {got:?}");
        assert_eq!(got, None);
        Ok(())
    }

    /// 软倾向：等级不满足回合不给 urgency 分（比赛降级为普通分，避免白打）
    #[test]
    fn test_score_race_skips_nonqualified_turn() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let policy = RamenPolicy::default();

        // 101901 回合 54（G2，等级不满足）：比赛分应为 0（race_grade_weight=0），reason 说明等级
        let mut game = make_free_race_game_uma(101901)?;
        game.base.turn = 54;
        let (val, reason) = policy.score_race(&game);
        println!("回合 54 (G2) score_race: val={val} reason={reason}");
        assert_eq!(val, 0.0, "等级不满足回合不应给 urgency 分");
        assert!(reason.starts_with("比赛"), "reason 应为普通比赛分: {reason}");

        // 回合 55（G1，等级满足，缺口 3、剩 4 有效回合）：给 urgency 分
        game.base.turn = 55;
        let (val, reason) = policy.score_race(&game);
        println!("回合 55 (G1) score_race: val={val} reason={reason}");
        assert!(val > 0.0, "等级满足回合应给 urgency 分");
        assert!(reason.contains("自选比赛"), "reason 应为自选比赛: {reason}");
        Ok(())
    }

    /// 打分自洽性：训练动作的 breakdown 各项之和 == score（调参日志不能撒谎）
    #[test]
    fn test_breakdown_sums_to_score() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.vital = 60;
        let policy = RamenPolicy::default();
        for a in train_actions_with_race() {
            let out = policy.score_train_action(&game, &a)?;
            let sum: f32 = out.breakdown.iter().map(|(_, v)| v).sum();
            println!(
                "{:<10} score={:>9.3} breakdown和={:>9.3} {:?}",
                a.to_string(),
                out.score,
                sum,
                out.breakdown
            );
            assert!((sum - out.score).abs() < 1e-2);
        }
        Ok(())
    }

    /// `status_rate` 必须线性生效（历史上被乘了两次，调参时表现为平方）
    #[test]
    fn test_status_rate_is_linear() -> anyhow::Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();

        let mut game = make_game()?;
        game.uma.vital = 100;
        let action = RamenAction::new(Operation::Train(TrainingType::Speed));
        let attr_of = |rate: f32| -> anyhow::Result<f32> {
            let mut cfg = RamenPolicyConfig::default();
            cfg.status_rate = rate;
            let out = RamenPolicy::new(cfg).score_train_action(&game, &action)?;
            Ok(out
                .breakdown
                .iter()
                .find(|(k, _)| k == "attr")
                .map(|(_, v)| *v)
                .unwrap_or(0.0))
        };
        let (a1, a2) = (attr_of(1.0)?, attr_of(2.0)?);
        println!("status_rate=1 → attr={a1:.3}；status_rate=2 → attr={a2:.3}（期望恰好 2 倍）");
        assert!((a2 - a1 * 2.0).abs() < 1e-2);
        Ok(())
    }
}
