//! 拉面杯剧本通信状态（`scenarioId = 14`）
//!
//! 协议定稿见 `.trae/documents/ramen_protocol_v2.md`（v1，2026-09）。
//!
//! **Step 6 现状**：`GameStatusRamen` 结构 + 字段映射 + `GameStatus` impl 骨架已完成；
//! `into_game` 当前返回 `RamenGame::newgame` 占位实例（仅构造基础 `Uma` / `deck` /
//! `friend` 等），**ramen 段所有增量字段（last_ramen / feeling_stock /
//! scenario_pt 等）的覆写在 Step 7 实现**——`RamenGame::from_external_state` 完整
//! 实现覆盖时，本文件的 `into_game` 替换为调 `from_external_state(self)`。
//!
//! 分发判据：`baseGame.scenarioId == 14`（详见 `mod.rs::parse_game_by_scenario`）。

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::ops::Deref;

use crate::protocol::{GameStatus, GameStatusBase};
use umasim::game::ramen::RamenGame;

/// 拉面剧本通信状态
///
/// **字段映射基线**（与 `ramen_protocol_v2.md` §2 一一对应）：
/// - `base_game`：与温泉剧本同构（`GameStatusBase`）+ 拉面增量字段（见 §1）
/// - 拉面段字段（待 §2 字段映射表全量实现）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatusRamen {
    /// 基础段（与温泉剧本共用 `GameStatusBase`）
    pub base_game: GameStatusBase,
    /// 拉面剧本增量字段（待 Step 7 完整填充）
    ///
    /// Step 6 留空 `Default`——C# 端 2026-09 实测样本数 `feeling_guage_gains` /
    /// `active_effect_array` / `super_ramen` 等都存在，但 `serde(default)` 允许缺字段反序列化成功。
    /// 完整字段定义 + 反序列化在 Step 7 落实（伴随 `RamenGame::from_external_state`）。
    #[serde(default)]
    pub ramen: serde_json::Value
}

impl Deref for GameStatusRamen {
    type Target = GameStatusBase;
    fn deref(&self) -> &Self::Target {
        &self.base_game
    }
}

impl GameStatus for GameStatusRamen {
    type Game = RamenGame;

    fn scenario_id() -> u32 {
        14
    }

    /// **Step 6 占位**：构造基础 RamenGame（newgame），**不**映射 ramen 段增量字段。
    ///
    /// Step 7 替换为调 `RamenGame::from_external_state(self)`，那时所有字段覆写、
    /// playing_state → stage dispatch、`source` 路由等一并落地。
    ///
    /// 当前占位的语义是"拉面剧本能走到 AI 主循环，但 state 是空架子"——仅保证 main
    /// 路由能跑通，不保证 AI 推荐正确。
    fn into_game(self) -> Result<Self::Game> {
        // Step 6 占位：直接 `RamenGame::newgame` 用 baseGame 的 deck / inherit 起步。
        // 真实覆写（last_ramen / feeling_stock / scenario_pt 等）在 Step 7 的
        // `from_external_state` 实现。
        let base = &self.base_game;
        let inherit = base.parse_inherit()?;
        let uma_id = base.uma_id;
        // `card_id` 已是 idrank 形式（id*10+rank），与 SupportCard::new 一致；
        // 直接传 newgame 即可（不要再 /10）
        let deck_ids: [u32; 6] = base
            .card_id
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| anyhow!("拉面协议要求 6 张卡"))?;
        RamenGame::newgame(uma_id, &deck_ids, inherit)
    }
}

/// `GameStatusRamen` → 拉面协议 JSON（暂未实现反向转换，Step 7 一并）
impl From<&RamenGame> for GameStatusRamen {
    fn from(_game: &RamenGame) -> Self {
        Self::default()
    }
}