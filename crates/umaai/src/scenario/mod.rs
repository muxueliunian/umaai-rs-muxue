//! 场景逻辑：按剧本（温泉 / 拉面）组织「收到回合数据 → AI 出决策 → 下发给 sink」。
//!
//! 每个子模块负责一个 `scenarioId` 对应的完整回合处理：newgame 检测、切局、决策
//! 计算（`calc_*`）与结果 emit。主程序 `crate::main` 只做分发调度。

pub mod onsen;
pub mod ramen;