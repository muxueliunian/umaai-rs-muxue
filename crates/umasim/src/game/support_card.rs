use std::{collections::HashMap, default::Default, sync::Arc};

use anyhow::{Result, anyhow};
use log::debug;
use serde::{Deserialize, Serialize};

use crate::{
    diag,
    explain::Explain,
    game::Game,
    gamedata::{CardValue, GAMEDATA, SupportCardData},
    global,
    utils::*
};

/// 局中带入计算的支援卡属性. 其他不变的属性从data里取
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CardTrainingEffect {
    /// 友情
    pub youqing: f32,
    /// 干劲
    pub ganjing: i32,
    /// 训练
    pub xunlian: i32,
    /// 赛后
    pub saihou: i32,
    /// 得意率
    pub deyilv: f32,
    /// 副属性
    pub bonus: Array6,
    /// 智训练体力恢复
    pub wiz_vital_bonus: i32,
    /// 失败率下降
    pub fail_rate_drop: f32,
    /// 体力消费下降
    pub vital_cost_drop: f32,
    /// 事件效果提高，乘算
    pub event_effect_up: i32,
    /// 事件回复量提高，乘算
    pub event_recovery_amount_up: i32,
    /// Hint数量(由固有改变)
    pub hint_count_bonus: i32
}

impl CardTrainingEffect {
    pub fn explain(&self) -> String {
        let mut ret = String::new();
        if self.youqing > 0.0 {
            ret += &format!("友情{:.1} ", self.youqing);
        }
        if self.ganjing > 0 {
            ret += &format!("干劲{} ", self.ganjing);
        }
        if self.xunlian > 0 {
            ret += &format!("训练{} ", self.xunlian);
        }
        if self.saihou > 0 {
            ret += &format!("赛后{} ", self.saihou);
        }
        if self.deyilv > 0.0 {
            ret += &format!("得意{:.1} ", self.deyilv);
        }
        if self.bonus != [0; 6] {
            ret += &format!("副属{} ", Explain::status_with_pt(&self.bonus));
        }
        if self.wiz_vital_bonus > 0 {
            ret += &format!("回体+{} ", self.wiz_vital_bonus);
        }
        if self.fail_rate_drop > 0.0 {
            ret += &format!("失败-{:.1}% ", self.fail_rate_drop);
        }
        if self.vital_cost_drop > 0.0 {
            ret += &format!("耗体-{:.1}% ", self.vital_cost_drop);
        }
        if self.event_effect_up > 0 {
            ret += &format!("事件效果+{} ", self.event_effect_up);
        }
        if self.event_recovery_amount_up > 0 {
            ret += &format!("事件回复+{} ", self.event_recovery_amount_up);
        }
        ret
    }

    /// 根据公式与另一个效果叠加，不包含得意率
    pub fn add(&self, other: &CardTrainingEffect) -> Self {
        let mut new_bonus = self.bonus.clone();
        new_bonus.add_eq(&other.bonus);
        Self {
            youqing: (100.0 + self.youqing) * (100.0 + other.youqing) / 100.0 - 100.0,
            ganjing: self.ganjing + other.ganjing,
            xunlian: self.xunlian + other.xunlian,
            saihou: self.saihou + other.saihou,
            deyilv: self.deyilv, // 不计算
            bonus: new_bonus,
            wiz_vital_bonus: self.wiz_vital_bonus + other.wiz_vital_bonus,
            fail_rate_drop: 100.0 - (100.0 - self.fail_rate_drop) * (100.0 - other.fail_rate_drop) / 100.0,
            vital_cost_drop: 100.0 - (100.0 - self.vital_cost_drop) * (100.0 - other.vital_cost_drop) / 100.0,
            event_effect_up: self.event_effect_up + other.event_effect_up,
            event_recovery_amount_up: self.event_recovery_amount_up + other.event_recovery_amount_up,
            hint_count_bonus: self.hint_count_bonus // 不叠加，单独计算
        }
    }

    /// 修改单个词条
    pub fn add_effect_line(&mut self, effect_id: i32, value: i32) -> &mut Self {
        match effect_id {
            1 => {
                // 友情
                self.youqing = (100.0 + self.youqing) * (100.0 + value as f32) / 100.0 - 100.0;
            }
            2 => {
                // 干劲
                self.ganjing += value;
            }
            3..8 => {
                // 副属性
                self.bonus[(effect_id - 3) as usize] += value;
            }
            8 => {
                // 训练
                self.xunlian += value;
            }
            9..15 => {}
            15 => {
                // 赛后
                self.saihou += value;
            }
            19 => {
                // 得意率
                self.deyilv = (100.0 + self.deyilv) * (100.0 + value as f32) / 100.0 - 100.0;
            }
            27 => {
                // 失败率减乘
                self.fail_rate_drop = 100.0 - (100.0 - self.fail_rate_drop) * (100.0 - value as f32) / 100.0;
            }
            28 => {
                // 耗体减乘
                self.vital_cost_drop = 100.0 - (100.0 - self.vital_cost_drop) * (100.0 - value as f32) / 100.0;
            }
            30 => {
                // pt加成
                self.bonus[5] += value;
            }
            31 => {
                // 智力回体
                self.wiz_vital_bonus += value;
            }
            33 => {
                // Hint数量增加
                self.hint_count_bonus += value;
            }
            41 => {
                // ALL
                for i in 0..5 {
                    self.bonus[i] += 1;
                }
            }
            _ => {
                diag!("未知效果词条: {effect_id} = {value}");
            }
        }
        self
    }
}

impl From<&CardValue> for CardTrainingEffect {
    fn from(v: &CardValue) -> Self {
        Self {
            youqing: v.youqing,
            ganjing: v.ganjing,
            xunlian: v.xunlian,
            saihou: v.saihou,
            deyilv: v.deyilv,
            bonus: v.bonus.clone(),
            wiz_vital_bonus: v.wiz_vital_bonus,
            fail_rate_drop: v.fail_rate_drop,
            vital_cost_drop: v.vital_cost_drop,
            event_effect_up: v.event_effect_up,
            event_recovery_amount_up: v.event_recovery_amount_up,
            hint_count_bonus: 0 // 面板中没有此词条
        }
    }
}

/// 局中的支援卡信息，剧本通用
#[derive(Debug, Clone, PartialEq)]
pub struct SupportCard {
    /// 面板数据，避免查表
    pub data: Arc<SupportCardData>,
    /// 支援卡ID(5位)
    pub card_id: u32,
    /// 突破等级
    pub rank: u32,
    /// 支援卡类型
    pub card_type: i32,
    /// 羁绊
    pub friendship: i32,
    /// 是否剧本链接
    pub is_link_card: bool,
    /// 属性是否不会再更新
    pub is_locked: bool,
    /// 支援卡基础属性
    pub effect: CardTrainingEffect,
    /// 固有状态
    pub effect_state: HashMap<String, u32>,
    /// 已经获得的Hint等级
    pub total_hints: i32
}

impl SupportCard {
    pub fn short_name(&self) -> String {
        format!("{}◆ {}", self.data.short_name(), self.rank)
    }

    pub fn explain(&self) -> Result<String> {
        let mut ret = format!("{} 绊{} {}", self.short_name(), self.friendship, self.effect.explain());
        if !self.effect_state.is_empty() {
            ret += &format!(" {:?}", self.effect_state);
        }
        if self.is_locked {
            ret += " [锁定]";
        }
        Ok(ret)
    }

    /// 高频调用路径允许panic
    #[inline]
    pub fn card_value(&self) -> &CardValue {
        &self.data.card_value[self.rank as usize]
    }

    /// 初始属性加成
    pub fn initial_bonus(&self) -> &Array6 {
        &self.card_value().initial_bonus
    }
    /// idrank: 卡ID+突破等级，形如 301614
    pub fn new(idrank: u32) -> Result<Self> {
        let (id, rank) = (idrank / 10, idrank % 10);
        let gamedata = global!(GAMEDATA);
        let data = gamedata.get_card(id)?;
        if rank > 4 {
            return Err(anyhow!("Rank超出范围: {}", idrank));
        }
        let effect = CardTrainingEffect::from(&data.card_value[rank as usize]);
        let friendship = data.card_value[rank as usize].initial_jiban;
        Ok(Self {
            data: Arc::new(data.clone()),
            card_id: id,
            rank,
            card_type: data.card_type,
            effect,
            friendship,
            is_link_card: false,
            is_locked: false,
            effect_state: HashMap::new(),
            total_hints: 0
        })
    }
    /// 计算支援卡当前 buff（基础面板 + 已触发的 unique_effect 词条）。
    ///
    /// 简化历史：原本签名返回 `Result<(CardTrainingEffect, bool)>`，第二个元素
    /// "是否已锁定"标志位用于 Game impl override 回写 `is_locked = true`。
    /// 对照实验（见 `perf_profiling.md` / `calc_training_value_microbench`）表明这条
    /// 缓存路径 ROI 极低，且 `is_locked` 字段也因 `features.rs` NN 输入维度而保留，
    /// 这里直接去掉锁定分支并简化签名：
    ///
    /// - 函数体总是执行 unique_effect match（无 `if !self.is_locked` 短路）
    /// - 起点是**基础面板**（`data.card_value[rank]`），不读 `self.effect`
    ///   —— 保证每次返回独立的 fresh cumulative effect，避免锁回写后的叠加
    /// - 调用者拿到的 `CardTrainingEffect` 总包含已触发的词条
    /// - `is_locked: bool` 字段保留（Game impl override 在 deyilv 中仍写
    ///   `true`，features.rs 当 NN 输入读；语义从"已锁定计算"简化为"走过此函数"标记）
    ///
    /// 不传 `Result`：函数体内无 fail 路径（`add_effect_line` 的 `diag!` 不 panic，
    /// `shining_count` 返回 `usize`），与上层 `calc_training_value` 保留
    /// `Result`（防御 `train >= 5` 历史错误）形成清晰的错误边界划分：
    /// 「低层 effect 永不错；上层 value / buff 的 train 范围错误仍 Result」。
    pub fn calc_training_effect<G: Game>(&self, game: &G, train: i32) -> CardTrainingEffect {
        // 起点是基础面板，确保每次返回 fresh cumulative，
        // 不会被 Game impl override 的 lock 回写后多次累加 unique_effect 词条
        let mut ret = CardTrainingEffect::from(&self.data.card_value[self.rank as usize]);
        let param = &self.data.unique_effect_param;
        match self.data.unique_effect_type {
            0 => {}
            1 | 2 => {
                // 羁绊>args[1]时触发词条args[2] = args[3]
                if self.friendship >= param[1] {
                    debug!("{} 羁绊>{}, 触发固有: {:?}", self.data.short_name(), param[1], param);
                    if param[2] > 0 {
                        ret.add_effect_line(param[2], param[3]);
                    }
                    if param[4] > 0 {
                        ret.add_effect_line(param[4], param[5]);
                    }
                }
            }
            13 => {
                // b95 参与友情训练时触发
                if game.shining_count(train as usize) > 0 {
                    ret.add_effect_line(param[1], param[2]);
                }
            }
            20 => {
                // 巨匠: 羁绊>80时根据卡组决定加成
                if self.friendship >= param[2] {
                    let mut card_type_count = vec![0, 0, 0, 0, 0, 0];
                    for c in game.deck() {
                        if c.card_type >= 5 {
                            card_type_count[5] += 1;
                        } else {
                            card_type_count[c.card_type as usize] += 1;
                        }
                    }
                    debug!(
                        "{} 羁绊>{}, 根据卡组决定加成(最大+2): {:?}",
                        self.data.short_name(),
                        param[2],
                        card_type_count
                    );
                    for i in 0..5 {
                        if card_type_count[i] > 0 {
                            // 0-4对应副属性词条#3-7
                            ret.add_effect_line((i + 3) as i32, card_type_count[i].max(2));
                        }
                    }
                    if card_type_count[5] > 0 {
                        ret.add_effect_line(30, card_type_count[5].max(2));
                    }
                }
            }
            _ => {
                diag!(
                    "未实现固有逻辑: #{} - {}",
                    self.data.unique_effect_type,
                    self.short_name()
                );
            }
        } // match unique_effect_type
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    #[test]
    fn test_support() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;
        let card = SupportCard::new(302424)?;
        println!("{}", card.explain()?);
        let card2 = SupportCard::new(302464)?;
        println!("{}", card2.explain()?);
        println!("{}", (card.effect.add(&card2.effect)).explain());
        Ok(())
    }
}
