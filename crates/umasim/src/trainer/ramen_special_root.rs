//! `SpecialSelect` 局面的联合决策根还原
//!
//! # 为什么需要它
//!
//! 教师在 `RamenSelect` 根上搜的是**联合动作**（地区 × 隐藏风味用法），policy 格位
//! `[1,201)` 也是联合格；训练集里 `SpecialSelect` 阶段的样本数为 **0**
//! （实测 `stage.npy`：RamenSelect 20998 / Train 30683 / SuperRamen 232 /
//! RegionSelect 3820，合计即全部 55733 条）。
//!
//! 但真实对局把这个决策拆成两拍：先选面，`Game::next()` 切到 `SpecialSelect`
//! 后再选用法。网络若在第二拍直接推理，读到的阶段 one-hot 在全部训练样本里恒为 0,
//! 输出属于外推——这是训练—部署语义错位（train-serve skew）。本模块把局面还原到
//! 联合决策根，使两拍读的是同一个状态下的同一组联合格。
//!
//! # 为什么放在 `trainer/` 而不是规则层
//!
//! 这是**只服务于神经网络策略**的推理口径修正，规则层本身不需要它。
//! 模块不带 `onnx` feature 门控：还原逻辑与模型无关，这样守门测试在默认 feature
//! 下就会运行——挂到 `onnx` 之后，平时的 `cargo test --release --lib` 不开该
//! feature，测试等于永不执行。
//!
//! # 与规则层的耦合风险
//!
//! 还原是对 `RamenSelect → SpecialSelect` 过渡的逆操作，依赖「该过渡只写这三处」
//! 这一事实。规则层若新增字段，本函数会**静默变错**。
//!
//! 守门手段是本文件的 `tests::test_canonical_special_root_matches_ramen_select`：
//! 它走真实的 `Game::next` 完成过渡，再断言还原后的 754 维特征与联合根**逐位相同**。
//!
//! ⚠ 守门范围有限：**只有进入特征编码的字段才会被抓住**。过渡若写脏一个不进特征的
//! 字段（例如 `combined_decision`），特征逐位比较不会变红。故测试另行直接比对
//! `stage` / `pending_ramen` / `pending_special_targets` / `combined_decision` 四项。

use anyhow::{Result, bail};

use crate::game::ramen::{RamenGame, RamenStage};

/// 把 `SpecialSelect` 局面还原成它所来自的 `RamenSelect` 联合决策根
///
/// # 为什么还原是精确的
///
/// `RamenSelect → SpecialSelect` 的过渡只写三处：`pending_ramen`、
/// `pending_special_targets`（恒置 `[0,0,0]`）与 `stage`，见 `RamenAction::apply`
/// 的 `RamenSelect` + `StageOnly` 分支与 `Game::next`。过程中**不消耗任何随机流**，
/// 拉面效果要到 `SpecialSelect → Train` 过渡才由 `ground_ramen_effects` 落地。
/// 故三个字段写回即逐位还原。
///
/// 返回副本；调用方用它做推理，候选合法性校验与动作执行仍应基于原局面。
///
/// # 错误
///
/// 当前不在 `SpecialSelect` 阶段，或 `pending_ramen` 为空时报错——两者都说明调用点
/// 接错了阶段，静默返回原局面会让错位以另一种形式留下来。
pub fn canonical_ramen_select_root(game: &RamenGame) -> Result<RamenGame> {
    if game.stage != RamenStage::SpecialSelect {
        bail!(
            "canonical_ramen_select_root: 仅在 SpecialSelect 阶段可调用，当前 stage={:?}",
            game.stage
        );
    }
    if game.ramen.pending_ramen.is_none() {
        bail!("canonical_ramen_select_root: SpecialSelect 阶段 pending_ramen 不应为空");
    }
    let mut root = game.clone();
    root.ramen.pending_ramen = None;
    root.ramen.pending_special_targets = [0, 0, 0];
    root.stage = RamenStage::RamenSelect;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        game::{Game, ramen::features},
        gamedata::init_global,
        utils::{get_workspace_root, init_test_logger}
    };

    // 与 game.rs 测试同一套夹具：[速]杏目 / [智]青春永驻 / [耐]名将怒涛 /
    // [速]洛林军歌 / [速]里见光钻 / [友]骏川手纲
    const TEST_DECK: [u32; 6] = [302424, 302894, 303044, 302924, 303024, 303054];
    const TEST_INHERIT: crate::game::InheritInfo = crate::game::InheritInfo {
        blue_count: [15, 3, 0, 0, 0],
        extra_count: [0, 30, 0, 0, 30, 30]
    };
    const TEST_UMA_ID: u32 = 102601;

    /// 打勾标记（与仓库其余测试同风格）
    fn check(ok: bool) -> &'static str {
        if ok { "OK" } else { "NG" }
    }

    /// 还原后特征必须与联合决策根**逐位相同**
    ///
    /// 同时断言**不还原时特征确实不同**——否则这个还原就是空操作，测试会在
    /// 错位被意外修好（或字段被移出特征）时静默变绿，失去守门作用。
    #[test]
    fn test_canonical_special_root_matches_ramen_select() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("info");
        let _ = init_global();

        let mut game = RamenGame::newgame(TEST_UMA_ID, &TEST_DECK, TEST_INHERIT)?;
        // 回合 13：turn >= 2 才有吃面选择；直接给足库存，跳过 RegionSelect 等阶段
        game.base.turn = 13;
        game.ramen.feeling_stock = [5, 5, 5];
        game.ramen.special_feeling = 2;
        game.ramen.selected_regions = [0, 1, 2];
        game.stage = RamenStage::RamenSelect;

        let v_root = features::encode(&game)?;
        let root_snapshot = game.clone();

        let actions = game.list_actions()?;
        let pick = actions
            .iter()
            .position(|a| a.ramen.is_some())
            .ok_or_else(|| anyhow!("回合 13 应至少有一个候选面"))?;
        let ramen_idx = actions[pick].ramen.ok_or_else(|| anyhow!("已判定为 Some"))?;
        game.apply_action(&actions[pick], &mut StdRng::seed_from_u64(20260831))?;
        // 走真实的 Game::next() 完成过渡，而不是手写 `game.stage = ...`——
        // 后者会让「next() 里多写了别的字段」这类回归静默溜过去
        assert!(game.next(), "apply 后 next() 应返回 true");
        assert_eq!(
            game.stage,
            RamenStage::SpecialSelect,
            "吃面后 next() 应推进到 SpecialSelect，实得 {:?}",
            game.stage
        );

        // 不还原：这就是网络第二拍实际读到的局面
        let v_skewed = features::encode(&game)?;
        let skew_diffs: Vec<usize> = v_root
            .iter()
            .zip(v_skewed.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!("选面 {ramen_idx} 后，未还原局面与联合根的特征差异下标: {skew_diffs:?}");
        println!(
            "[{}] 未还原时特征必须不同（否则本测试失去守门意义）",
            check(!skew_diffs.is_empty())
        );
        assert!(
            !skew_diffs.is_empty(),
            "未还原的 SpecialSelect 局面竟与联合根特征相同，说明阶段/pending 未进特征，测试前提失效"
        );

        // 还原后必须逐位相同
        let root = canonical_ramen_select_root(&game)?;
        let v_canon = features::encode(&root)?;
        let canon_diffs: Vec<usize> = v_root
            .iter()
            .zip(v_canon.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        println!("还原后与联合根的特征差异下标: {canon_diffs:?}");
        println!("[{}] 还原后特征必须逐位相同", check(canon_diffs.is_empty()));
        assert!(
            canon_diffs.is_empty(),
            "还原后仍有 {} 个特征位不同: {:?}",
            canon_diffs.len(),
            canon_diffs
        );

        // 不进特征的字段也要还原：特征逐位比较抓不到它们
        let same_fields = root.stage == root_snapshot.stage
            && root.ramen.pending_ramen == root_snapshot.ramen.pending_ramen
            && root.ramen.pending_special_targets == root_snapshot.ramen.pending_special_targets
            && root.ramen.combined_decision == root_snapshot.ramen.combined_decision;
        println!(
            "还原后 stage={:?} pending_ramen={:?} pending_targets={:?} combined={} | 联合根 stage={:?} pending_ramen={:?} pending_targets={:?} combined={}",
            root.stage,
            root.ramen.pending_ramen,
            root.ramen.pending_special_targets,
            root.ramen.combined_decision,
            root_snapshot.stage,
            root_snapshot.ramen.pending_ramen,
            root_snapshot.ramen.pending_special_targets,
            root_snapshot.ramen.combined_decision
        );
        println!("[{}] 过渡涉及的四个字段必须与联合根一致", check(same_fields));
        assert!(same_fields, "还原后过渡字段与联合根不一致");

        // 阶段不对时必须报错，不得静默返回原局面
        let mut wrong = game.clone();
        wrong.stage = RamenStage::Train;
        println!(
            "[{}] 非 SpecialSelect 阶段调用必须报错",
            check(canonical_ramen_select_root(&wrong).is_err())
        );
        assert!(
            canonical_ramen_select_root(&wrong).is_err(),
            "非 SpecialSelect 阶段应报错"
        );

        // SpecialSelect 但 pending_ramen 为空同样必须报错（该分支此前没测）
        let mut empty = game.clone();
        empty.ramen.pending_ramen = None;
        println!(
            "[{}] SpecialSelect 但 pending_ramen 为空必须报错",
            check(canonical_ramen_select_root(&empty).is_err())
        );
        assert!(
            canonical_ramen_select_root(&empty).is_err(),
            "pending_ramen 为空时应报错"
        );

        Ok(())
    }
}
