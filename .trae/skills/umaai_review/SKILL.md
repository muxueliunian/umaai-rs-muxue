---
name: umaai_review
description: 单局复盘分析。对 logs/game{id}.zip 运行 umaai_review 分析引擎生成 digest.json 与 report.html，然后输出叙事性归因结论（总体表现、运气走势与极值回合、继承质量、坏手法、执行偏离）。用户说「帮我分析一下这一局 / 复盘一下这一局 / 打得怎么样 / 分析 gameXXXX」时使用。
---

# umaai_review · 单局复盘归因

> 计算下沉到 Rust（`umaai_review` bin），判断力留给本 skill。
> 口径速查见 `reference/metrics_glossary.md`；拟人化口吻见 `reference/persona.md`
> （可选覆盖层，可删可改；缺失时按默认平实专业口吻输出，流程与检查项不受影响）。

## 工作流程

1. **定位局包**：用户给的路径（如 `logs/game6234.zip`）或会话上传的文件；没指名时取 `logs/` 下最新 `game*.zip`。
2. **运行分析引擎**，产物落局包同级 `logs/game{id}/`：`digest.json`（喂归因）+ `report.html`（只给路径不粘贴）：

   ```powershell
   cargo run --release -p umaai_review -- --zip logs/game6234.zip
   # 或直接调用编译产物
   target\release\umaai_review.exe --zip logs/game6234.zip [--gamedata <path>]
   ```

   - gamedata 解析顺序：`--gamedata` > `UMAI_DATA_DIR` > 局包向上找 `gamedata/` > cwd。全找不到时 bin 降级（纯 ID、无评分），**询问用户一次**，确认后写入本 skill 目录 `local_paths.json`（`{"gamedata": "..."}`），下次先读它。
   - exe 不可用降级：`Expand-Archive` 解包，直接读 `decisions.csv` + `meta.json` 做决策链路 / 运气分 / 置信度分析（丢失依赖快照时序的检查项），结论需注明降级。
3. **读取 digest.json**，块导航（字段口径见 glossary）：

   > 读取踩坑（实测）：digest.json 是单行紧凑 JSON（200KB 级），**不要整读整贴**——
   > 用工具分块提取、摘要输出；`clones.a_per_turn` 等大结构全量 dump 会淹没
   > 排在前面的 `context` / `inherit` / `schedule` 小块；`luck.flagged_turns`
   > 是对象数组（`{turn, reason}`），不是数字数组，别用 join 类方式直接拼接。

   | 块 | 用途 |
   |---|---|
   | `meta` | 马娘 / 卡组 / 评分等级 / 终局运气分 / 结束方式 |
   | `timeline` | 每回合快照状态（五维、体力、干劲、剧本 PT、缺席人物） |
   | `decisions` | 每个决策点的候选 / 选中 / 期望分 / 回合Δ / 链长 |
   | `execution` | AI 建议 vs 实际执行（动作推断 + 状态差证据） |
   | `luck` | 运气分序列 / top_gain / top_loss / flagged_turns（伪波动） |
   | `schedule` | 必赛回合 / 自由比赛区间与实跑 |
   | `inherit` | 继承质量（贡献 vs 参考值） |
   | `clones` | 分身彩圈观测（A/B 类，luck/strategy 来源） |
   | `coverage` | skip 分原因 / unparsed / parse_error |
   | `findings` | bin 自动产出的检查项命中清单 |
   | `context` | 口径前提（必读，解读前先过一遍） |

4. **归因**：见下节框架与口径。
5. **输出**：见输出纪律，叙事结论只在对话输出不落盘。
6. **叙述回填 report.html**：模板留有 4 个占位标记，用编辑工具把 report.html 里的标记行**整行替换**为叙述 HTML 块（内容 = 复盘叙事对应段，数据与 digest 一致；非 HTML 转义文本，直接写标签）：

   | 标记 | 回填内容 | 位置 |
   |---|---|---|
   | `<!-- NARRATIVE:overview -->` | 总体描述（Q1 + 五维结构） | 概览卡之后、图 1 之前 |
   | `<!-- NARRATIVE:luck_trend -->` | 运气走势分段解释（Q2） | 图 2 之下 |
   | `<!-- NARRATIVE:findings -->` | 检查项与执行偏离的叙述性分条（Q5/Q6 摘要） | 图 3 之下 |
   | `<!-- NARRATIVE:summary -->` | 总结与下一局建议 | 页尾、口径速览之前 |

   骨架（复用模板现有样式类；卡片用 `details open` 可折叠、默认展开，标题写 `summary`；
   p / ul / table 均已有样式）：

   ```html
   <section>
     <details open>
     <summary>总体</summary>
     <p>终局评分 64589，评价 US1……</p>
     <ul>
       <li>开局：……</li>
       <li>中段：……</li>
     </ul>
     </details>
   </section>
   ```

   回填时注意：
   - **overview 分条**：评分口径 / 五维 / 一致率 / 运气分各一条，不放段落
   - **luck_trend 通俗**：口语化表述，如「账面跳变」说程序性波动、「回血」说回收、
     「一路小亏」说阴跌；判据细节归口径速览，不在走势里展开
   - **findings 只讲异常**：命中的检查项分项目符号；正常项（干劲正常恢复、
     检查项无命中等）不提；执行偏离用表格（回合 / 建议 / 实际执行 / 落地四列）
   - **数据块叙述**：继承质量与分身彩圈的文字叙述写在对应数据 section 的表格之后
     （不在占位标记内）——继承一句判断（偏优 / 正常 / 偏弱及归因），分身分条讲
     彩圈是否吃到（`used` 字段）
   - **叙述口吻与对话一致**：persona 启用时按人设书写（summary 块尾可署「理事长 秋川弥生」），
     网页版与对话版只是同一复盘的两个出口
   - **背景图（可选装饰）**：`reference/yayoi.png` 存在时复制到输出目录 `logs/game{id}/`，
     并在 report.html 的 `</style>` 前插入下面两行 CSS——第一行铺背景（20% 透明度垫在内容下），
     第二行把前景文字框调成 70% 不透明度、隐约透出背景；图片缺省时跳过，报告照常：
     ```css
     body::before { content:""; position:fixed; inset:0; z-index:-1;
                    background:url("yayoi.png") center/cover no-repeat; opacity:0.2; }
     :root { --card:rgba(255,255,255,0.7); }
     ```

## 归因框架

bin 已自动产 findings，读 `findings` + 对应数据复核即可。后续增补检查项按「判据 / 取数 / 输出 / 状态」四要素格式追加，不动框架。

| 检查项 | 判据 | 取数 | 状态 |
|---|---|---|---|
| 目标赛未跑赢 | 必赛回合不在 raceHistory（只记跑赢） | findings / schedule.notes | 已实现（已验证） |
| 关键资源过早耗尽 | 友人次数达上限距结束仍远 + 此后体力低点 | findings / timeline | 已实现（已验证） |
| 心情掉落后未及时恢复 | 干劲下降后 3 回合无出行/休息且未回升 | findings / timeline | 已实现（已验证） |
| 训练失败候选 | AI 建议训练 + 五维零增长 + 体力有消耗 | findings | 候选清单，人工复核事件影响 |
| 超级拉面期盈亏 | turn≥72 回合合计Δ 正负（B 类分身机制保证落位，没吃到不算亏） | findings / luck | 已实现（已验证） |
| 执行偏离 | ai_choice ≠ actual_action | findings / execution | AIRedirector 局仅作校验、不产生结论 |

**归因口径（用户拍板）**：

- 目标赛未跑赢、超拉期连亏、掉干劲本身都是**运气不是操作**——输赛随机、掉干劲由输赛带来、超拉期估值走低是落地差。唯一例外「掉干劲后未主动恢复」是已知偏差②（模拟高估事件回复），作观察项与改进方向。
- 运气极值：`flagged_turns`（年界 / 继承 / RMJ / 开局地区选择）**降级或跳过**，是程序性波动；turn_delta 是期望终局分变化，不等于属性增量，同回合多段以合计观察。
- 继承偏差为负 → 质量偏弱，是确实的损失，如实报数不打折。
- 分身：`rainbow_luck` 归好运、`rainbow_strategy` 归好策略；价值是提高训练效果（玩家已知，叙事不解释）；**A 类彩圈吃到与否直接读 `a_per_turn` 的 `used` 字段**（当回合训练命中彩圈位 = 吃到）；B 类超拉彩圈只报汇总数，不追具体回合。
- 五维叙事报**显示值**：直接读 `timeline.five_status_display`（真实值 > 1200 部分减半，bin 已换算）；评分与运气分不受此换算影响。
- total_luck_end 略低于实际运气（支援卡事件进度未统计），`< -2000` 才判「这局运气差」。
- 判据待实测项谨慎下结论：训练与体力健康、吃面节奏、友人完成度、free_race 次数与时机、状态健康、属性溢出。

**六问清单（复盘必答，顺序即叙事骨架）**：

| # | 问题 | 取数 | 要点 |
|---|---|---|---|
| 1 | 总体打得怎么样？ | `meta`、`execution` 一致率 | 一两句定调；评分仅供参考（略低于小黑板、未计「努力家」）；全胜不提战绩，输赛才展开 |
| 2 | 运气走势如何？ | `luck.series`、`flagged_turns`、`total_luck_end` | 分段叙事（开局 / 中段 / 末期 / 超拉期）；伪波动段降级读 |
| 3 | 极值回合发生了什么？ | `top_gain` / `top_loss` → 回该回合 `decisions`、`timeline`、`clones` | 每回合一段：具体操作 + 落地证据；flagged 回合注明程序性波动 |
| 4 | 继承质量如何？ | `inherit` | 小表格（回合 / 贡献 / 偏差）+ 一句判断 |
| 5 | 还有哪些波动来源？ | `luck.series` 转折、`clones`、`execution` 偏离、`schedule`、`timeline` | 归到具体操作或局面，区分「操作导致」vs「随机」；分身与干劲用小表格 |
| 6 | 建议被执行了吗？ | `execution` + findings 偏离 | 逐条给证据；persona 启用时称「我当时的建议」，AIRedirector 局只作校验 |

## 输出纪律

- **口吻**：马娘玩家术语优先——彩圈 / 友情训练 / 羁绊 / 干劲 / 体力 / 吃面 / 超级拉面 / 目标赛 / URA 决赛 / 继承 / 出行 / 友人卡；少用内部字段名（`turn_delta` 说成「运气波动」，`flagged_turns` 说成「程序性波动回合」）。
- **少括号少破折号**：口径与回合号融进句子（例：「终局 62500，评价 UA9。顺带一提，这个评分口径仅供参考，实际还要略高一点」），不要堆插入语。
- **语气基调**：LG 档（final_score ≥ 76000）放开夸，夸到回合与数字；坏局减弱负面、给希望；根因是继承弱时一句带过。
- **篇幅**：原因一两句点透，不长篇大论；继承 / 分身 / 干劲 / 自选比赛 / 执行偏离用小表格或分条，正文只留一两句解释；额外的判据、口径说明、待定项统一放结尾一小节**分条列出**，不穿插在分析中间。
- 运气数字带符号表示（如-3526, +800）。
- **叙事纪律（用户拍板）**：
  - 全胜不提：比赛全赢是基本操作，输赛才展开
  - 安抚不出口：「不深究」「不数落」「不打折扣」「一笔带过」不落正文——说出口就成了数落，弱化体现在措辞与详略上
  - 玩家已知口径不解释（分身价值 = 提高训练效果），只报观测数据
  - 自选比赛两类分开：`free_races`（区间内剧本要求的自选，含 required 次数与实跑）与 notes 的「区间外自选」（不在任何要求内自发跑的）分别列行；马娘没有自由比赛区间数据时如实说明
  - 干劲恢复一般是做了动作才回升（出行 / 休息 / 事件选项），不表述成自然回升，恢复列写具体动作
  - turn 72 兜底：超级拉面期开始自动加体力与心情（干劲），不建议 72 前专门补状态
- **persona**：启用时 `execution.ai_choice` 统一改称理事长自己的建议（第一人称「我当时的建议」），叙事不出现「AI」字样；无论有无 persona，数据与结论必须与 digest 一致，拟人化只改表达层。
- **结构** = 六问顺序 → 改进建议 → 末尾集中一小节口径备注（待定项、读数须知）→ `report.html` 路径收尾；叙事结论对话输出，叙述四块按工作流程第 6 步回填进 report.html。
