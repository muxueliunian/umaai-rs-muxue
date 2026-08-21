use colored::Colorize;

use crate::{
    gamedata::{EventChoice, GAMECONSTANTS},
    global,
    utils::{Array5, Array6}
};

pub struct Explain;

/// 解释一些基础类型的值
impl Explain {
    pub fn motivation(m: i32) -> String {
        if m == 0 {
            "干劲错误".to_string()
        } else {
            global!(GAMECONSTANTS)
                .motivation_names
                .get((m - 1) as usize)
                .cloned()
                .unwrap_or("干劲错误".to_string())
        }
    }

    pub fn five_status(stats: &Array5) -> String {
        let mut s = String::new();
        for (i, stat) in stats.iter().enumerate() {
            if *stat != 0 {
                s += &format!("{}{} ", global!(GAMECONSTANTS).train_names[i], stat);
            }
        }
        s
    }

    /// 带裁剪上限 + 终端彩色高亮的五维状态。
    ///
    /// colored 无条件加载；启用 `no-color` feature 时输出纯文本（保留 cutted 行为）。
    pub fn five_status_cutted(stats: &Array5) -> String {
        let mut s = String::new();
        for (i, stat) in stats.iter().enumerate() {
            if *stat != 0 {
                if *stat <= 1200 {
                    s += &format!("{}{} ", global!(GAMECONSTANTS).train_names[i], stat);
                } else {
                    let cutted = 1200 + (stat - 1200) / 2;
                    s += &format!(
                        "{}",
                        format!("{}{} ", global!(GAMECONSTANTS).train_names[i], cutted).cyan()
                    );
                }
            }
        }
        s
    }

    pub fn status_with_pt(stats: &Array6) -> String {
        if stats[5] != 0 {
            format!(
                "{}{}pt",
                Self::five_status(&stats[..5].try_into().expect("status-pt")),
                stats[5]
            )
        } else {
            Self::five_status(&stats[..5].try_into().expect("status-pt"))
        }
    }

    pub fn train_level_count(train: &Array5) -> String {
        let levels: Vec<_> = train.iter().map(|x| (x / 4 + 1).min(5)).collect();
        format!("{levels:?}")
    }

    /// 显示带剧本加成的训练等级
    pub fn train_level_count_with_bonus(train: &Array5, bonus: i32) -> String {
        let levels: Vec<_> = train
            .iter()
            .map(|&x| {
                let base = (x / 4 + 1).min(5);
                (base + bonus).min(5).max(1)
            })
            .collect();
        format!("{levels:?}")
    }

    pub fn event_choice(choice: &[EventChoice]) -> String {
        choice.iter().map(|x| x.explain()).collect::<Vec<_>>().join(" | ")
    }
}
