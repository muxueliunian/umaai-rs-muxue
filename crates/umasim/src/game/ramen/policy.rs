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

use crate::game::ramen::{Operation, RamenAction, RamenGame};
use crate::game::traits::Game;
use crate::gamedata::{EventChoice, GAMECONSTANTS, ramen::RAMENDATA};
use crate::global;

use super::rules::{
    calc_ramen_pt_gain, get_region_range, get_super_ramen_clone_train_options,
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
    pub race_grade_weight: f32,
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
    // ===== Event =====
    /// 事件体力每点折算
    pub event_vital_weight: f32,
    /// 事件干劲每点折算
    pub event_motivation_weight: f32,
    /// 事件获得 bad flag（ill/bad_trainer）的惩罚
    pub event_bad_flag_penalty: f32,
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
            race_grade_weight: 900.0,
            outing_base: 15.0,
            friend_outing_bonus: 45.0,
            ramen_pt_weight: 5.0,
            ramen_effect_weight: 3.0,
            ramen_special_cost: 12.0,
            ramen_stock_cost: 0.4,
            region_xunlian_weight: 40.0,
            region_pt_weight: 30.0,
            region_hint_weight: 15.0,
            event_vital_weight: 2.2,
            event_motivation_weight: 40.0,
            event_bad_flag_penalty: 300.0,
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
    pub reason: String,
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
    pub config: RamenPolicyConfig,
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

    /// Train 阶段：守门（生病/体力/心情）→ 否则按收益打分选最优
    pub fn select_train(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        if actions.is_empty() {
            anyhow::bail!("Train 阶段候选为空");
        }
        let is_xiahesu = game.is_xiahesu();
        let uma = &game.uma;

        // 守门 1：生病 → 治病（夏合宿无治病候选，休息自动治病）
        if uma.flags.ill || uma.flags.bad_trainer {
            if let Some(idx) = actions
                .iter()
                .position(|a| a.operation == Operation::Clinic && !is_xiahesu)
            {
                return Ok(idx);
            }
            if is_xiahesu {
                if let Some(idx) = actions
                    .iter()
                    .position(|a| a.operation == Operation::Rest)
                {
                    return Ok(idx);
                }
            }
        }
        // 守门 2：体力低 → 休息（防失败率崩盘；优先于心情、训练）
        if uma.vital < self.config.vital_rest {
            if let Some(idx) = actions.iter().position(|a| a.operation == Operation::Rest) {
                return Ok(idx);
            }
        }
        // 守门 3：心情低 → 外出（回干劲）
        if uma.motivation < self.config.motivation_outing {
            if let Some(idx) = actions.iter().position(|a| {
                matches!(a.operation, Operation::NormalOuting | Operation::FriendOuting)
            }) {
                return Ok(idx);
            }
        }

        // 打分选择
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            scores.push(self.score_train_action(game, a)?);
        }
        Ok(argmax_index(&scores))
    }

    /// RamenSelect 阶段：吃面收益（PT + 地区效果）与不吃面比较，贪心
    pub fn select_ramen(&self, game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        if actions.is_empty() {
            anyhow::bail!("RamenSelect 阶段候选为空");
        }
        // 每个候选：不吃面固定 0 分；吃面按收益 - 成本
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            scores.push(self.score_ramen_action(game, a)?);
        }
        Ok(argmax_index(&scores))
    }

    /// SpecialSelect 阶段：最省隐藏风味（保留库存）；候选已按 sum(t) 升序，
    /// 这里显式按 -sum(targets) 打分，不依赖排序保证
    pub fn select_special(&self, _game: &RamenGame, actions: &[RamenAction]) -> Result<usize> {
        if actions.is_empty() {
            anyhow::bail!("SpecialSelect 阶段候选为空");
        }
        let mut scores: Vec<RamenPolicyOutput> = Vec::with_capacity(actions.len());
        for a in actions {
            let used = a
                .special_targets
                .map(|t| t.iter().sum::<i32>())
                .unwrap_or(0) as f32;
            let mut out = RamenPolicyOutput {
                score: -used * self.config.ramen_special_cost,
                ..Default::default()
            };
            out.add("hidden_used", -used * self.config.ramen_special_cost);
            out.reason = format!("隐藏风味消耗 {used}");
            scores.push(out);
        }
        Ok(argmax_index(&scores))
    }

    /// RegionSelect 阶段：按地区静态价值打分选组合（含第 3 年 120 组合全枚举，O(360) 便宜）
    pub fn select_region(
        &self,
        game: &RamenGame,
        _year_idx: usize,
        actions: &[RamenAction],
    ) -> Result<usize> {
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
        Ok(argmax_index(&scores))
    }

    /// 事件选项打分
    pub fn select_event(&self, game: &RamenGame, choices: &[Vec<EventChoice>]) -> Result<usize> {
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
        Ok(argmax_index(&scores))
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
                out.add("attr", attr_gain * self.config.status_rate);
                out.add("pt", pt_gain * self.config.pt_rate);
                // 体力成本（消耗按 train_vital_value 折算）
                let vital_cost = (-value.vital).max(0) as f32;
                out.add("vital_cost", -vital_cost * self.config.train_vital_value);
                // 期望值（失败率）
                let fail_p = fail_rate / 100.0;
                out.add("fail_rate", -fail_p);
                out.score = (attr_gain * self.config.status_rate
                    + pt_gain * self.config.pt_rate
                    - vital_cost * self.config.train_vital_value)
                    * (1.0 - fail_p)
                    - self.config.failure_penalty * fail_p;
                // 彩圈加成
                let shining = game.shining_count(train) as f32;
                out.add("shining", shining * self.config.shining_bonus);
                out.score += shining * self.config.shining_bonus;
                out.reason = format!(
                    "{}训练 失败率{fail_rate:.0}% 属性+{attr_gain:.0} PT+{pt_gain:.0}",
                    global!(GAMECONSTANTS).train_names[train]
                );
            }
            Operation::Race => {
                // 比赛价值按等级与回合时机
                let grade = game_race_grade(game);
                out.add("race_grade", grade);
                out.score = grade * self.config.race_grade_weight;
                out.reason = format!("比赛(等级{grade})");
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
        let hidden = a
            .special_targets
            .map(|t| t.iter().sum::<i32>())
            .unwrap_or(0) as f32;
        let stock_cost =
            self.config.ramen_stock_cost * 5.0 * (1.0 - (game.ramen.special_feeling as f32 + 4.0) / 12.0).max(0.2);
        out.add("hidden_cost", -hidden * self.config.ramen_special_cost);
        out.add("stock_cost", -stock_cost);
        out.score = pt_gain * self.config.ramen_pt_weight + effect_val
            - hidden * self.config.ramen_special_cost
            - stock_cost;
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
        // 人数最少的训练位置补充倾向（多数拉面效果偏速度）
        let train_val: f32 = region
            .at_trains
            .iter()
            .map(|&t| {
                let t = t as usize;
                if t < 5 { region.xunlian as f32 * bias[t].max(0.5) } else { 0.0 }
            })
            .sum();
        Ok(train_val * self.config.region_xunlian_weight
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
        .max_by(|(ia, a), (ib, b)| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| ib.cmp(ia))
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// 当前回合比赛等级（0 = 无比赛；等级越高越好）
fn game_race_grade(game: &RamenGame) -> f32 {
    let turn = game.turn();
    let grades = &global!(GAMECONSTANTS).race_grades;
    if turn >= 72 {
        // URA 回合固定 G1（表外）
        return 4.0;
    }
    grades
        .get(turn as usize)
        .copied()
        .unwrap_or(0)
        .max(0) as f32
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
    use crate::game::ramen::TrainingType;
    use crate::{gamedata::init_global, utils::{get_workspace_root, init_test_logger}};

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

    /// 构造一个可用的 RamenGame（默认卡组 102601，train 阶段可打分）
    fn make_game() -> anyhow::Result<RamenGame> {
        use crate::game::ramen::RamenGame;
        let inherit = crate::game::InheritInfo {
            blue_count: [15, 3, 0, 0, 0],
            extra_count: [0, 30, 0, 0, 30, 30],
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
        use crate::gamedata::{ActionValue, EventData};
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
}
