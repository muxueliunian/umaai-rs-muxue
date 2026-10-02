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
    /// 小切
    #[serde(default)]
    pub xiaoqie: bool,
    /// 爱娇
    #[serde(default)]
    pub aijiao: bool,
    /// 擅长训练
    #[serde(default)]
    pub good_trainer: bool,
    /// 不擅长训练
    #[serde(default)]
    pub bad_trainer: bool,
    /// 正向思考（心情盾）剩余次数
    #[serde(default)]
    pub positive_thinking_count: i32,
    /// 休息心得，表示持续了几回合
    #[serde(default)]
    pub refresh_mind: i32,
    /// 幸运体质次数
    #[serde(default)]
    pub lucky_count: i32,
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
        if self.xiaoqie {
            s += "小切 ";
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
        if self.positive_thinking_count > 0 {
            s += &format!("正向思考({}) ", self.positive_thinking_count);
        }
        if self.lucky_count > 0 {
            s += &format!("幸运体质({}) ", self.lucky_count);
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

    /// 局内获得【切者】：与【小切】互斥——获得切者时小切失效
    ///
    /// 小切为局外（育成开始前）获得，局内不会再次获得，故不需要反向处理
    /// （小切不会把切者清掉）。
    pub fn gain_qiezhe(&mut self) -> &mut Self {
        self.qiezhe = true;
        self.xiaoqie = false;
        self
    }

    /// PT 折算系数（`pt_score_rate`）的额外倍数：切者 ×1.1、小切 ×1.04
    ///
    /// 语义：切者/小切让技能点更"值钱"（技能价格折扣），故终局评分里 PT 项按此放大。
    /// 两者互斥（见 [`Self::gain_qiezhe`]），判断顺序无关。**只乘 PT 项**，不改变
    /// [`Uma::total_pt`] 的 PT 数量口径，也不乘五维分与已学技能分。
    pub fn pt_score_rate_factor(&self) -> f32 {
        if self.qiezhe {
            1.1
        } else if self.xiaoqie {
            1.04
        } else {
            1.0
        }
    }

    /// 添加状态
    pub fn add(&mut self, rhs: &UmaFlags) -> &mut Self {
        if rhs.qiezhe {
            // 互斥：获得切者会清掉小切
            self.gain_qiezhe();
        } else if rhs.xiaoqie {
            self.xiaoqie = true;
        }
        self.aijiao |= rhs.aijiao;
        self.good_trainer |= rhs.good_trainer;
        self.bad_trainer |= rhs.bad_trainer;
        self.positive_thinking_count += rhs.positive_thinking_count;
        self.refresh_mind += rhs.refresh_mind;
        self.lucky_count += rhs.lucky_count;
        self.doll |= rhs.doll;
        self.ill |= rhs.ill;
        self
    }

    /// 减少状态
    pub fn remove(&mut self, rhs: &UmaFlags) -> &mut Self {
        self.qiezhe &= !rhs.qiezhe;
        self.xiaoqie &= !rhs.xiaoqie;
        self.aijiao &= !rhs.aijiao;
        self.good_trainer &= !rhs.good_trainer;
        self.bad_trainer &= !rhs.bad_trainer;
        // 次数类状态按下限 0 扣减（心情盾被消耗、幸运体质失效）
        self.positive_thinking_count = (self.positive_thinking_count - rhs.positive_thinking_count).max(0);
        self.refresh_mind -= rhs.refresh_mind;
        self.lucky_count = (self.lucky_count - rhs.lucky_count).max(0);
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

    /// 【特殊生涯比赛】开局提示文案；该马娘没有 `raceNote` 时返回 `None`
    ///
    /// 取 `UmaDB.json` 的 `raceNote`（如「第三年打安田纪念」「事件不影响比赛回合」），
    /// 以**蓝底亮黄**醒目上色。`umaai` 在检测到新局时打印（走 stderr，不污染 stdout 的
    /// JSON 流）；`uma_id` 查不到数据（如测试 fixture）时静默返回 `None`。
    pub fn explain_race_note(&self) -> Option<String> {
        let note = self.get_data().ok()?.race_note.as_deref()?;
        if note.trim().is_empty() {
            return None;
        }
        Some(format!("该马娘有特殊生涯比赛: {note}").bright_yellow().on_blue().to_string())
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
    ///
    /// PT 项额外乘 [`UmaFlags::pt_score_rate_factor`]（切者 ×1.1 / 小切 ×1.04）：
    /// 这是**终局评分**口径，不改变 [`Self::total_pt`] 的 PT 数量，也不改搜索的
    /// `score_pt` 选择轴（拉面 `/温泉` 的选择轴各有独立 PT 公式）。
    pub fn score_parts(&self) -> ScoreParts {
        let cons = global!(GAMECONSTANTS);
        let mut five_status = [0i32; 5];
        for i in 0..5 {
            let status = self.five_status[i].min(self.five_status_limit[i]);
            five_status[i] = cons.status_final_score(status);
        }
        ScoreParts {
            skill: self.skill_score,
            pt: (self.total_pt() as f32
                * cons.pt_score_rate
                * self.flags.pt_score_rate_factor()) as i32,
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

    /// 增减干劲：**掉心情时优先消耗【心情盾】**（`positive_thinking_count`），盾为 0 才真掉
    ///
    /// 心情盾按「防一次掉心情」计：一次扣减（无论扣几级，如大失败的 -3）消耗一层盾，
    /// 心情不变；盾为 0 时按原规则 `max(1).min(5)` 夹取。心情上涨不消耗盾。
    pub fn add_motivation(&mut self, delta: i32) -> &mut Self {
        if delta < 0 && self.flags.positive_thinking_count > 0 {
            self.flags.positive_thinking_count -= 1;
            diag!(
                "  心情盾挡下一次掉心情 {}（剩 {} 层）",
                delta,
                self.flags.positive_thinking_count
            );
            return self;
        }
        self.motivation = (self.motivation + delta).max(1).min(5);
        self
    }

    pub fn add_value(&mut self, action: &ActionValue) -> &mut Self {
        diag!("{}", action.explain().bright_black());
        for i in 0..5 {
            self.five_status[i] = (self.five_status[i] + action.status_pt[i]).min(self.five_status_limit[i]);
        }
        self.skill_pt += action.status_pt[5];
        self.add_motivation(action.motivation);
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

    /// 次数类状态：心情盾（正向思考）/幸运体质按次数累加、扣减下限 0；小切为布尔 flag
    #[test]
    fn test_flag_counts_add_remove() {
        let mut flags = UmaFlags::default();
        flags.add(&UmaFlags {
            xiaoqie: true,
            positive_thinking_count: 3,
            lucky_count: 2,
            ..Default::default()
        });
        println!("累加后: {:?} / {}", flags, flags.explain());
        assert_eq!(flags.positive_thinking_count, 3, "心情盾次数应累加");
        assert_eq!(flags.lucky_count, 2, "幸运体质次数应累加");
        assert!(flags.xiaoqie, "小切应被置位");

        // 消耗一次心情盾 + 幸运体质失效
        flags.remove(&UmaFlags {
            positive_thinking_count: 1,
            lucky_count: 2,
            ..Default::default()
        });
        println!("扣减后: {:?} / {}", flags, flags.explain());
        assert_eq!(flags.positive_thinking_count, 2, "消耗一次心情盾应只扣 1");
        assert_eq!(flags.lucky_count, 0, "幸运体质应扣到 0");
        assert!(flags.explain().contains("正向思考(2)"), "状态描述应带剩余次数");
        assert!(!flags.explain().contains("幸运体质"), "次数归零后不应再显示");

        // 超额扣减不得出现负数
        flags.remove(&UmaFlags {
            positive_thinking_count: 5,
            ..Default::default()
        });
        println!("超额扣减后: {:?} / {}", flags, flags.explain());
        assert_eq!(flags.positive_thinking_count, 0, "次数不应扣成负数");
        assert!(!flags.explain().contains("正向思考"), "次数归零后不应再显示");

        flags.remove(&UmaFlags {
            xiaoqie: true,
            ..Default::default()
        });
        assert!(!flags.xiaoqie, "小切应可移除");
    }

    /// 心情盾：掉心情优先消耗 `positive_thinking_count`，盾为 0 才真掉心情
    #[test]
    fn test_motivation_shield() {
        // 一层盾挡下一次掉心情
        let mut uma = Uma::default();
        uma.motivation = 4;
        uma.flags.positive_thinking_count = 1;
        uma.add_motivation(-1);
        println!(
            "盾=1 掉1级: 心情={} 盾={}",
            uma.motivation, uma.flags.positive_thinking_count
        );
        assert_eq!(
            (uma.motivation, uma.flags.positive_thinking_count),
            (4, 0),
            "一层盾应挡下 -1 且心情不变"
        );

        // 大失败 -3 也只消耗一层盾（心情盾按「防一次掉心情」计）
        let mut uma = Uma::default();
        uma.motivation = 4;
        uma.flags.positive_thinking_count = 2;
        uma.add_motivation(-3);
        println!(
            "盾=2 掉3级: 心情={} 盾={}",
            uma.motivation, uma.flags.positive_thinking_count
        );
        assert_eq!(
            (uma.motivation, uma.flags.positive_thinking_count),
            (4, 1),
            "一次扣减只消耗一层盾"
        );

        // 盾为 0：按原规则真掉，下限仍为 1
        let mut uma = Uma::default();
        uma.motivation = 4;
        uma.add_motivation(-1);
        assert_eq!(uma.motivation, 3, "无盾时应真掉心情");
        uma.motivation = 1;
        uma.add_motivation(-2);
        println!("无盾 掉2级: 心情={}", uma.motivation);
        assert_eq!(uma.motivation, 1, "心情下限仍为 1");

        // 涨心情不消耗盾，上限仍为 5
        let mut uma = Uma::default();
        uma.motivation = 3;
        uma.flags.positive_thinking_count = 2;
        uma.add_motivation(2);
        println!(
            "上涨2级: 心情={} 盾={}",
            uma.motivation, uma.flags.positive_thinking_count
        );
        assert_eq!(
            (uma.motivation, uma.flags.positive_thinking_count),
            (5, 2),
            "上涨不消耗盾且上限为 5"
        );
    }

    /// 特殊生涯比赛提示：带 `raceNote` 的马娘返回醒目文案，无则 `None`
    #[test]
    fn test_explain_race_note() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        init_test_logger("info")?;
        init_global()?;

        let mut uma = Uma::default();
        uma.uma_id = 100501; // UmaDB 里带 raceNote（"选择英里路线"）
        let note = uma.explain_race_note().expect("100501 应带 raceNote");
        println!("有 raceNote: {note:?}");
        assert!(
            note.contains("该马娘有特殊生涯比赛: 选择英里路线"),
            "提示文案应含 raceNote 原文: {note:?}"
        );
        // 颜色码取决于终端是否支持彩色（CI / 重定向时 colored 会自动关闭），
        // 只在实际着色时校验「蓝底亮黄」= `\e[93;44m`，避免环境相关的红。
        if note.contains('\u{1b}') {
            assert!(note.contains("\u{1b}[93;44m"), "着色时应为蓝底亮黄: {note:?}");
        } else {
            println!("（当前环境 colored 未着色，跳过颜色码校验）");
        }

        uma.uma_id = 100101; // UmaDB 里无 raceNote
        println!("无 raceNote: {:?}", uma.explain_race_note());
        assert!(uma.explain_race_note().is_none(), "无 raceNote 应返回 None");
        Ok(())
    }

    /// 切者 / 小切的互斥与 PT 折算系数
    #[test]
    fn test_qiezhe_xiaoqie_exclusive_and_factor() {
        // 局内获得切者：清掉小切
        let mut flags = UmaFlags {
            xiaoqie: true,
            ..Default::default()
        };
        flags.gain_qiezhe();
        println!("gain_qiezhe() 后: {:?}", flags);
        assert!(flags.qiezhe && !flags.xiaoqie, "获得切者应清掉小切");

        // 事件 add 路径同样互斥
        let mut flags = UmaFlags {
            xiaoqie: true,
            ..Default::default()
        };
        flags.add(&UmaFlags {
            qiezhe: true,
            ..Default::default()
        });
        println!("add(切者) 后: {:?}", flags);
        assert!(flags.qiezhe && !flags.xiaoqie, "事件 add 路径应清掉小切");

        // PT 折算系数
        let factor = |qiezhe: bool, xiaoqie: bool| {
            UmaFlags {
                qiezhe,
                xiaoqie,
                ..Default::default()
            }
            .pt_score_rate_factor()
        };
        println!(
            "折算系数: 无={} 切者={} 小切={} 同时={}",
            factor(false, false),
            factor(true, false),
            factor(false, true),
            factor(true, true)
        );
        assert_eq!(factor(false, false), 1.0, "无状态不加成");
        assert_eq!(factor(true, false), 1.1, "切者 ×1.1");
        assert_eq!(factor(false, true), 1.04, "小切 ×1.04");
        assert_eq!(factor(true, true), 1.1, "两者同时存在时以切者为准");
    }

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
        // 切者/小切的 PT 折算加成：这里独立写死系数（不复用 `pt_score_rate_factor()`），
        // 保证这份 oracle 能抓住实现里的系数写错。
        let pt_factor = if uma.flags.qiezhe {
            1.1
        } else if uma.flags.xiaoqie {
            1.04
        } else {
            1.0
        };
        ScoreParts {
            skill: uma.skill_score,
            pt: (uma.total_pt() as f32 * cons.pt_score_rate * pt_factor) as i32,
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

        // 6. 切者 / 小切：终局评分的 PT 项按 1.1 / 1.04 放大（只乘 PT 项）
        for (label, qiezhe, factor) in [("切者", true, 1.1_f32), ("小切", false, 1.04_f32)] {
            let mut uma = Uma::default();
            uma.skill_pt = 1000;
            uma.five_status = [200, 180, 160, 140, 120];
            uma.five_status_limit = [1200, 1200, 1200, 1200, 1200];
            let base_pt = uma.score_parts().pt;
            uma.flags.qiezhe = qiezhe;
            uma.flags.xiaoqie = !qiezhe;
            let boosted = uma.score_parts().pt;
            check_score_parts_case(label, &uma);
            println!("{label}: PT 项 {base_pt} → {boosted}（系数 {factor}）");
            assert_eq!(
                boosted,
                (base_pt as f32 * factor) as i32,
                "{label}: 终局评分的 PT 项应按折算系数放大"
            );
        }

        Ok(())
    }
}
