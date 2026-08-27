use std::default::Default;

use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::{
    diag,
    explain::Explain,
    gamedata::{ActionValue, EventChoice, FreeRaceData, GAMECONSTANTS, GAMEDATA, UmaData},
    global,
    utils::*
};

/// 训练中的马娘状态，剧本通用
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UmaFlags {
    /// 切者
    #[serde(default)]
    pub qiezhe: bool,
    /// 爱娇
    #[serde(default)]
    pub aijiao: bool,
    /// 擅长训练
    #[serde(default)]
    pub good_trainer: bool,
    /// 不擅长训练
    #[serde(default)]
    pub bad_trainer: bool,
    /// 正向思考
    #[serde(default)]
    pub positive_thinking: bool,
    /// 休息心得，表示持续了几回合
    #[serde(default)]
    pub refresh_mind: i32,
    /// 幸运体质
    #[serde(default)]
    pub lucky: bool,
    /// 是否抓过娃娃
    #[serde(default)]
    pub doll: bool,
    /// 是否生病
    #[serde(default)]
    pub ill: bool
}

impl UmaFlags {
    pub fn explain(&self) -> String {
        let mut s = String::new();
        if self.qiezhe {
            s += &format!("{}", "切者 ".bright_green());
        }
        if self.aijiao {
            s += "爱娇 ";
        }
        if self.good_trainer {
            s += "擅长训练 ";
        }
        if self.bad_trainer {
            s += "不擅长训练 ";
        }
        if self.positive_thinking {
            s += "正向思考 ";
        }
        if self.lucky {
            s += "幸运体质 ";
        }
        if self.doll {
            s += "抓过娃娃 ";
        }
        if self.ill {
            s += "*生病 ";
        }
        if self.refresh_mind > 0 {
            s += &format!("休息心得({}回合)", self.refresh_mind);
        }
        s
    }

    /// 添加状态
    pub fn add(&mut self, rhs: &UmaFlags) -> &mut Self {
        self.qiezhe |= rhs.qiezhe;
        self.aijiao |= rhs.aijiao;
        self.good_trainer |= rhs.good_trainer;
        self.bad_trainer |= rhs.bad_trainer;
        self.positive_thinking |= rhs.positive_thinking;
        self.refresh_mind += rhs.refresh_mind;
        self.lucky |= rhs.lucky;
        self.doll |= rhs.doll;
        self.ill |= rhs.ill;
        self
    }

    /// 减少状态
    pub fn remove(&mut self, rhs: &UmaFlags) -> &mut Self {
        self.qiezhe &= !rhs.qiezhe;
        self.aijiao &= !rhs.aijiao;
        self.good_trainer &= !rhs.good_trainer;
        self.bad_trainer &= !rhs.bad_trainer;
        self.positive_thinking &= !rhs.positive_thinking;
        self.refresh_mind -= rhs.refresh_mind;
        self.lucky &= !rhs.lucky;
        self.doll &= !rhs.doll;
        self.ill &= !rhs.ill;
        self
    }
}

/// 训练中的马娘信息，剧本通用（固定为5星）
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Uma {
    /// 马娘编号
    pub uma_id: u32,
    /// 体力
    pub vital: i32,
    /// 最大体力
    pub max_vital: i32,
    /// 干劲 [1, 5]
    pub motivation: i32,
    /// 当前属性。1200以上不减半
    pub five_status: Array5,
    /// 属性加成
    pub five_status_bonus: Array5,
    /// 属性上限
    pub five_status_limit: Array5,
    /// 剩余技能点
    pub skill_pt: i32,
    /// 已学技能评分
    pub skill_score: i32,
    /// 总共打折级数
    pub total_hints: i32,
    /// 比赛加成
    pub race_bonus: i32,
    /// Buff状态
    pub flags: UmaFlags,
    /// 生涯比赛bitset 低到高位对应11-71回合
    pub career_races: u64,
    /// 比赛场次 bitset 对应11-71回合
    pub win_races: u64
}

/// `calc_score()` 的可归因分量分解
///
/// 七个分量之和逐位等于 [`Uma::calc_score`]，用于搜索层的终局归因统计。
///
/// PT 项**不可**再拆成 skill_pt 与 hint 的独立贡献：`total_pt()` 内有一次 `floor()`、
/// 外面又有一次 `as i32`，两层截断使其数学上不可分。五维记的是查表后的分数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreParts {
    /// 技能分（`skill_score` 原值）
    pub skill: i32,
    /// PT 折算分：`(total_pt() as f32 * pt_score_rate) as i32`
    pub pt: i32,
    /// 五维各自的查表得分（速耐力根智），已按 limit 截断
    pub five_status: [i32; 5]
}

impl ScoreParts {
    /// 七个分量之和，逐位等于 [`Uma::calc_score`]
    pub fn total(&self) -> i32 {
        self.skill + self.pt + self.five_status.iter().copied().sum::<i32>()
    }
}

impl Uma {
    pub fn get_data(&self) -> Result<&UmaData> {
        global!(GAMEDATA).get_uma(self.uma_id)
    }

    /// 角色ID（高4位）
    pub fn chara_id(&self) -> u32 {
        self.uma_id / 100
    }

    pub fn explain(&self) -> Result<String> {
        let data = self.get_data()?;
        // 体力文字按档位着色：<35 红、<50 黄、其余亮绿（与整行风格一致）。
        // 各段独立上色而非整行包裹：内层 SGR 的 reset 码不会终止外层颜色。
        let vital_text = format!("{}/{}", self.vital, self.max_vital);
        let vital_colored = if self.vital < 35 {
            vital_text.red()
        } else if self.vital < 50 {
            vital_text.yellow()
        } else {
            vital_text.bright_green()
        };
        Ok(format!(
            "{} 体力 {} {} {} {}PT{} Hint{} 赛后{}",
            data.short_name().bright_green(),
            vital_colored,
            Explain::motivation(self.motivation).bright_green(),
            self.flags.explain().bright_green(),
            Explain::five_status_cutted(&self.five_status).bright_green(),
            self.skill_pt.to_string().bright_green(),
            self.total_hints.to_string().bright_green(),
            self.race_bonus.to_string().bright_green()
        ))
    }

    /// 建立马娘对象
    ///
    /// `limit_base` 是**所在剧本**的五维上限基值（不含继承）。每个剧本的基值都不同，
    /// 必须由调用方从对应的 `scenario_*.json` 取，不能在这里读全局常量——
    /// 早期版本先写全局值、再由各剧本事后修正，那种「打补丁」写法正是
    /// 「整体赋值擦掉继承增量」缺陷的来源。基值在构造时一次写对，之后只做加法。
    pub fn new(id: u32, limit_base: Array5) -> Result<Self> {
        let gamedata = global!(GAMEDATA);
        let data = gamedata.get_uma(id)?;
        Ok(Self {
            uma_id: id,
            vital: 100,
            max_vital: 100,
            motivation: 3,
            five_status: data.five_status_initial.clone(),
            five_status_bonus: data.five_status_bonus.clone(),
            five_status_limit: limit_base,
            skill_score: 510, // 固有按5星计算,
            total_hints: 21,  // 按全部初始技能3级打折计算
            career_races: data.zip_races(),
            ..Default::default()
        })
    }

    pub fn is_race_turn(&self, turn: i32) -> bool {
        if turn == 73 || turn == 75 || turn == 77 {
            true
        } else if turn < 11 || turn > 72 {
            false
        } else {
            (1u64 << (turn - 11)) & self.career_races != 0
        }
    }

    /// 设置第x回合为比赛状态，用于统计自选比赛
    pub fn set_race(&mut self, turn: i32) {
        if turn < 11 || turn > 72 {
            return;
        }
        self.win_races |= 1u64 << (turn - 11);
    }

    /// 计算技能点和总Hint等级换算得到的总pt数，不包括已学习的技能
    pub fn total_pt(&self) -> i32 {
        (self.skill_pt as f32 + self.total_hints as f32 * global!(GAMECONSTANTS).hint_pt_rate).floor() as i32
    }

    /// 把 [`Self::calc_score`] 分解成可归因分量
    ///
    /// 七个分量之和逐位等于 [`Self::calc_score`]。只在 3 项（`skill` / `pt` /
    /// `five_status` 之和）或 7 项粒度上保证逐位相等；PT 项已含 `total_pt()` 的
    /// `floor` 与 `as i32` 两层截断，不可再拆。
    pub fn score_parts(&self) -> ScoreParts {
        let cons = global!(GAMECONSTANTS);
        let mut five_status = [0i32; 5];
        for i in 0..5 {
            let status = self.five_status[i].min(self.five_status_limit[i]);
            five_status[i] = cons.status_final_score(status);
        }
        ScoreParts {
            skill: self.skill_score,
            pt: (self.total_pt() as f32 * cons.pt_score_rate) as i32,
            five_status
        }
    }

    /// 正常计算评分
    ///
    /// 等于 [`Self::score_parts`] 七个分量之和。
    pub fn calc_score(&self) -> i32 {
        self.score_parts().total()
    }

    pub fn calc_score_with_pt_favor(&self) -> i32 {
        let cons = global!(GAMECONSTANTS);
        // 技能点x8, 不计Hint，只考虑技能点
        let mut score = self.skill_score + (self.skill_pt as f32 * cons.pt_score_rate) as i32;
        score = (score as f32 * cons.pt_favor_rate) as i32;
        for i in 0..5 {
            let status = self.five_status[i].min(self.five_status_limit[i]);
            score += cons.status_final_score(status);
        }
        // 乘一个系数与原本评分数量级接近
        ((score as f64) * 0.37) as i32
    }

    pub fn add_value(&mut self, action: &ActionValue) -> &mut Self {
        diag!("{}", action.explain().bright_black());
        for i in 0..5 {
            self.five_status[i] = (self.five_status[i] + action.status_pt[i]).min(self.five_status_limit[i]);
        }
        self.skill_pt += action.status_pt[5];
        self.motivation = (self.motivation + action.motivation).max(1).min(5);
        self.max_vital += action.max_vital;
        self.vital = (self.vital + action.vital).min(self.max_vital).max(0);
        self.total_hints += action.hint_level;
        self
    }

    /// 根据事件选项更新Flag状态
    pub fn update_flags(&mut self, choice: &EventChoice) -> &mut Self {
        if let Some(flags) = &choice.add_flags {
            self.flags.add(&flags);
            diag!("获得状态: {}", flags.explain());
        }
        if let Some(flags) = &choice.remove_flags {
            self.flags.remove(&flags);
            diag!("失去状态: {}", flags.explain());
        }

        self
    }

    /// 返回自选比赛场数
    pub fn count_free_race(&self, free: &FreeRaceData) -> u32 {
        (self.win_races & free.mask).count_ones()
    }

    /// 自选比赛是否全部达标
    ///
    /// [`crate::game::BaseGame::check_free_race`] 只在各区间结束回合的下一回合判定，
    /// 且不达标会直接终止育成；本方法在任意时点重新比对各区间的完成场数，
    /// 供基准统计使用。无自选比赛要求的马娘恒为 `true`。
    pub fn all_free_races_done(&self) -> Result<bool> {
        Ok(self
            .get_data()?
            .free_races
            .iter()
            .all(|f| self.count_free_race(f) >= f.count))
    }

    /// 返回当前所处的自选比赛区间
    pub fn find_free_race(&self, turn: i32) -> Option<&FreeRaceData> {
        if let Ok(data) = self.get_data() {
            data.free_races
                .iter()
                .find(|f| f.start_turn <= turn as u32 && f.end_turn >= turn as u32)
        } else {
            None
        }
    }

    /// 从bitmap转为比赛回合Vec
    pub fn list_races(&self) -> Vec<i32> {
        let mut ret = vec![];
        for bit in 0..63 {
            if self.win_races & (1 << bit) != 0 {
                ret.push(bit + 11);
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        gamedata::{GAMECONSTANTS, init_global},
        global,
        utils::{get_workspace_root, init_test_logger}
    };

    #[test]
    fn test_uma() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let uma = Uma::new(101901, global!(GAMECONSTANTS).five_status_limit_base)?;
        println!("{}", uma.explain()?);
        Ok(())
    }

    #[test]
    fn test_win_races() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut uma = Uma::new(101901, global!(GAMECONSTANTS).five_status_limit_base)?;
        uma.win_races = 0b110000_000000_1;
        println!("{:?}", uma.list_races());
        Ok(())
    }

    /// 按文档公式独立重算七个分量，用于对照 `score_parts()`
    fn expected_score_parts(uma: &Uma) -> ScoreParts {
        let cons = global!(GAMECONSTANTS);
        let mut five_status = [0i32; 5];
        for i in 0..5 {
            // 刻意**不**调 `cons.status_final_score()`：这份是用来对照 `score_parts()` 的
            // 独立实现，两边共用同一个查表函数就不再是 oracle 了。这里自己按同一套语义
            // （先夹 0、再饱和到表末）另写一遍。
            let table = &cons.five_status_final_score;
            let status = uma.five_status[i].min(uma.five_status_limit[i]).max(0) as usize;
            five_status[i] = table[status.min(table.len() - 1)];
        }
        ScoreParts {
            skill: uma.skill_score,
            pt: (uma.total_pt() as f32 * cons.pt_score_rate) as i32,
            five_status
        }
    }

    /// 打印并断言 `score_parts().total() == calc_score()`，七个分量逐位相等
    fn check_score_parts_case(label: &str, uma: &Uma) {
        let parts = uma.score_parts();
        let expected = expected_score_parts(uma);
        let total = parts.total();
        let calc = uma.calc_score();
        println!(
            "{label}: skill={} pt={} five={:?} total={} calc_score={}",
            parts.skill, parts.pt, parts.five_status, total, calc
        );
        println!(
            "  expected: skill={} pt={} five={:?} sum={}",
            expected.skill,
            expected.pt,
            expected.five_status,
            expected.total()
        );
        assert_eq!(parts, expected, "{label}: 七个分量必须与公式逐位相等");
        // ⚠ 转发契约，**不是**公式 oracle：`calc_score()` 当前的实现就是
        // `score_parts().total()`，所以这一行在今天等价于 `x == x`。
        // 它唯一的作用是：将来有人把 `calc_score` 拆开重写时会红。
        // 真正校验公式的是上面对 `expected_score_parts()` 的断言——
        // 那是独立重写的一份原公式，改坏 `score_parts` 会被它抓住。
        assert_eq!(total, calc, "{label}: score_parts().total() 必须等于 calc_score()");
    }

    /// P0.3：`score_parts` 求和逐位等于 `calc_score`
    #[test]
    fn test_score_parts_matches_calc_score() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        // 1. 全零
        let uma = Uma::default();
        check_score_parts_case("全零", &uma);

        // 2. 五维触顶被 limit 截断
        let mut uma = Uma::default();
        uma.five_status = [5000, 4000, 3000, 2000, 1000];
        uma.five_status_limit = [100, 80, 60, 40, 20];
        uma.skill_score = 510;
        check_score_parts_case("五维触顶", &uma);

        // 3. 五维为负（按 0 查表）
        let mut uma = Uma::default();
        uma.five_status = [-10, -1, 0, 50, 100];
        uma.five_status_limit = [1200, 1200, 1200, 1200, 1200];
        check_score_parts_case("五维为负", &uma);

        // 4. skill_pt 与 total_hints 都非零，total_pt() 发生 floor
        //    hint_pt_rate=6.5 → 1 + 1*6.5 = 7.5，floor 后 7
        let mut uma = Uma::default();
        uma.skill_pt = 1;
        uma.total_hints = 1;
        uma.skill_score = 100;
        uma.five_status = [200, 180, 160, 140, 120];
        uma.five_status_limit = [1200, 1200, 1200, 1200, 1200];
        println!(
            "floor 边界: skill_pt={} total_hints={} total_pt()={}",
            uma.skill_pt,
            uma.total_hints,
            uma.total_pt()
        );
        check_score_parts_case("floor 边界", &uma);

        // 5. 另一组非整数：10 + 3*6.5 = 29.5 → floor 29
        let mut uma = Uma::default();
        uma.skill_pt = 10;
        uma.total_hints = 3;
        uma.skill_score = 2000;
        uma.five_status = [400, 350, 300, 250, 200];
        uma.five_status_limit = [1200, 1200, 1200, 1200, 1200];
        println!(
            "floor 边界2: skill_pt={} total_hints={} total_pt()={}",
            uma.skill_pt,
            uma.total_hints,
            uma.total_pt()
        );
        check_score_parts_case("floor 边界2", &uma);

        Ok(())
    }
}
