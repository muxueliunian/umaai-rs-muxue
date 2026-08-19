# UmaAI-RS 问题记录

本文件用于记载较复杂问题（需要用户协助解决的）的解决过程。

## 问题记录模板

```
## [问题标题]
- **日期**：YYYY-MM-DD
- **状态**：待解决 / 解决中 / 已解决
- **问题描述**：简要描述问题现象
- **排查过程**：记录排查步骤和发现
- **解决方案**：记录最终解决方法
- **备注**：其他相关信息
```

---

## distribute_person 中"不出现"判定受得意率影响

- **日期**：2026-08-19
- **状态**：待解决
- **问题描述**：当前 `Game::distribute_person`（`traits.rs`）将"不出现"判定和"训练位置分配"混在一起，不在率 = `absent_rate / (500 + absent_rate + deyilv)`，导致得意率会影响"不出现"概率。按剧本原始规则，"不出现"概率应不受得意率影响，得意率只影响训练位置的权重分配。
- **排查过程**：
  - 用户给出剧本原始算法：
    1. 用基础权重 [100,100,100,100,100,absent_rate] 判定"不出现"，概率 = `absent_rate / (500 + absent_rate)`（**不含得意率**）
    2. 判定为出现后，按 [100+deyilv, 100, 100, 100, 100]（不含"不出现"项）随机分配训练位置
  - 当前算法：
    - 不在率 = `absent_rate / (500 + absent_rate + deyilv)`（含得意率）
    - 训练位置按 [100+deyilv, 100, 100, 100, 100, absent_rate]（含"不出现"项）分配
  - 关键差异：得意率会拉高"不出现"判定概率（deyilv 越大，不出现概率越低）——这是错误的
- **解决方案**：将 `distribute_person` 改为两步算法：
  1. 先用基础权重（不含 deyilv）判定"是否不出现"
  2. 判定为出现后，按训练位置权重（含 deyilv）随机分配训练位置
- **备注**：2026-08-19 曾确认暂不动 absent_rate 相关逻辑（涉及 `absent_rate_drop` 等其他领域知识），当时只修复了 `RamenGame::deyilv` 缺剧本加成的问题；`distribute_person` 修正留待本 issue。RamenGame 当前未 override `distribute_person`，两步算法尚未实现。

---

## 第三年地区选择组合过多

- **日期**：2026-08-17
- **状态**：待解决
- **问题描述**：第3年可选地区为10-19共10个，C(10,3)=120种组合，动作空间过大，影响搜索效率
- **排查过程**：
  - 第1/2年各5个地区，C(5,3)=10种组合，可接受
  - 第3年120种组合导致 Trainer 需要评估120个动作，计算开销显著增加
- **解决方案**：待讨论，可能的方向：
  1. 基于当前卡组和诀窍库存，预筛选出合理的候选组合（如基于配方消耗匹配度）
  2. 分两步决策：先选主方向（偏速/偏耐/偏力等），再在子集中选具体组合
  3. 使用评估函数对120个组合快速打分排序，只让Trainer从top-K中选择
- **备注**：第3年地区还包含pt_bonus效果，选择策略需要同时考虑youqing/pt_bonus和配方匹配

---

## Ubuntu 下 umaai 二进制图标方案待定

- **日期**：2026-08-17
- **状态**：待解决（暂缓）
- **问题描述**：umaai 的 Windows 版通过 build.rs + winscribe 把 `.ico` 图标嵌入二进制；Ubuntu 下 ELF 二进制没有 Windows 资源节的图标嵌入机制，需要确定 Linux 版的图标落地方式
- **排查过程**：
  - Linux ELF 无原生图标资源节，无法直接嵌入 `.ico`
  - 候选方案一：`.desktop` 文件 + 外置 PNG/SVG 图标（桌面菜单展示，Linux 惯例）
  - 候选方案二：`include_bytes!` 把 PNG 数据嵌入二进制（自包含、单文件分发，需补充 PNG 资源）
  - 已顺带修复 umaai 的 Linux 构建：winscribe 改为仅在 `cfg(windows)` 目标下作为 build-dependency；build.rs 的 Windows 资源编译逻辑用 `#[cfg(windows)]` 包裹；`windows` crate 及其依赖链（windows-future 0.3.2 与 windows-core 0.62.2 不兼容，Linux 上编译失败）改为 `cfg(windows)` 限定依赖；补声明 Linux 专用的 `libc` 依赖；Linux 版 `get_stack_size` 补 `pub` 修饰
- **解决方案**：暂不处理，待用户明确图标使用场景（桌面菜单展示或二进制自包含）后再实施
- **备注**：umaai 为 CLI + TUI 程序（clap + ratatui），桌面图标应用场景有限

---

## constants.json 排名数据需人工更新

- **日期**：2026-08-19
- **状态**：待解决（待人工更新）
- **问题描述**：constants.json 中的 `rank_scores`、`rank_names`、`five_status_final_score` 为游戏排名相关数据，当前数值可能已过期，需要稍后按最新游戏版本人工核对更新
- **排查过程**：配置系统整理（Phase 2）甄别 constants.json 各项归属时确认：这三项属固定游戏数据、不随剧本变化，保留在 constants.json，由人工更新
- **解决方案**：待用户准备最新数据后更新三个数组
- **备注**：与配置整理方案一致，见 .trae/documents/config_refactor_plan.md
