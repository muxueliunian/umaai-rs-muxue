# 探查踩坑清单（使用 skill 前必读）

> 实测累积的坑位。每次复盘发现新坑回填到这里；SKILL.md 只保留指针，详细坑位以本文件为准。
> 主代理派发子代理时，任务描述里必须带上相关坑位提醒，避免子代理重踩。

## digest.json 读取

1. 单行紧凑 JSON（200KB 级），**不要整读整贴**：分块取字段、摘要输出。**取数优先 Python**（临时 `.py` 一次加载多块，避免反复启动解释器）；PowerShell 仅作备选，需 `-Encoding UTF8` 读文件。两版片段见 `probe_prompts.md`。
2. **`timeline` 与 `execution` 块直接是行数组**，不是带 `rows` 字段的对象（game1458 实测）：Python `d['execution']['rows']` / PS `$d.execution.rows` 都取到空，取行要用 `d['execution']` / `$d.execution` 本身。
3. `luck.series` 每项字段是 `{turn, seq, total_luck}`，没有 `value` 字段。
4. `luck.flagged_turns` 是对象数组（`{turn, reason}`），不是数字数组，别用 join 类方式直接拼接。
5. `clones.a_per_turn` 大结构勿全量 dump：先按 `cards[].rainbow_positions` 非空过滤，再取 `used` / `origin`。
6. **digest.json 顶层没有 `raceHistory` 键**（game3098 实测，顶层仅 meta/timeline/decisions/execution/luck/schedule/inherit/clones/coverage/findings/context）：比赛要用 `timeline[].race_count` 的增量定位，胜负用 findings 有无「目标赛未跑赢」兜底。
7. **`ConvertFrom-Json` 可能在中途报解析失败**（多字节处）：读文件时显式指定 UTF-8（`Get-Content <path> -Raw -Encoding UTF8`），否则字段名可能变 mojibake 并破坏结构。
8. **会话级环境差异**：部分会话的 PowerShell 工具执行成功但不回显 stdout（取数全为空）。**这正是不用 PowerShell 取数、统一走 Python 的首要理由**；已遇到时直接换 `python`（临时 `.py` 或 `-c`）按同一字段口径取数，字段名与方法不变；子代理同样适用。
9. **A 类彩圈取数必须按 `rainbow_positions` 非空过滤，且同一回合可能有多张卡同时带彩圈**（game3098 t37 实测：`card0 pos=[0] used=False` 与 `card4 pos=[4] used=True` 同时存在，正好对上汇总的「彩圈 2 / 被训练 1」）。**汇总数与明细条数必须一致；对不上说明取数漏了**，不要当成数据缺口、更不要反推回合或宣称「明细无记录」。
   - **PowerShell 特有坑（game3101 t63 实测）**：`rainbow_positions` 只有 `[0]`（速位）时，PS 会把单元素数组**解包成标量 `0`**，而 `[bool]0 = False`——写成 `Where-Object { $_.rainbow_positions -and ... }` 会**静默漏掉速位彩圈**（该局唯一一个 A 类彩圈正好在速位，旧写法输出「无」而真相是「1 个且吃到」）。必须写 `@($_.rainbow_positions).Count -gt 0`。**Python 无此问题**，直接判 list 非空即可——这也是优先 Python 的理由之一。

## 产物与回填

10. **重复运行 bin 会整目录覆盖产物**：`umaai_review --zip` 每次都重写 `logs/game{id}/` 下的 `brief.md`、`digest.json` 与 `report.html`。**第 6 步回填叙述之后再跑引擎，回填内容会被抹回模板**（game3098 实测，跑第 2 次即丢）。回填后不要再跑引擎；若必须重跑，先备份 `report.html` 或重跑后重新回填。brief.md 是纯生成物，无需保留修改。
11. **回填要幂等**：把 7 处替换（4 个占位标记 + 继承表后 + 分身表后 + 背景图 CSS）写成一个 **Python 脚本**一次执行，比逐条编辑快得多，也便于覆盖后一键重做。锚点未命中要报错退出，不要静默通过。
12. **`reference/yayoi.png` 不会被 bin 清理**，但 CSS 插入语句会随 `report.html` 一起被覆盖，重跑后要一并补回。

## 游戏机制口径（用户拍板）

13. **输比赛不会掉干劲**：干劲掉落是别的事件导致的，叙事不得把干劲掉落归因于输赛。快照通常没有事件明细，具体原因列待查。
14. **turn 73/75/77 固定有 URA 目标比赛**，数据里可能没有标出：按默认胜利处理，末快照未记录 77 决赛结果不算输。
15. **只调查已知口径内的问题**：解释不了的现象一律列入待查，不编造机制解释、不硬下结论。
16. **执行偏离是估算**：actual_action 由状态差推断，训练上限溢出会干扰主增量维判读（耐到上限时根增量反超，被误判成根训练，game1458 t47/t61 实测）。可疑处（相邻维反超 / 涉及属性上限）标「存疑」。
17. **运气分阴跌的总口径**：display 运气分含 mcts_turn_bonus（MCTS 预期的每回合平均运气增长），实际游戏非线性增长；一句话：**运气增长慢于预期，不进则退**，叠加已知 bug 的估分偏差。不写成「小额落地差」这类含糊归因。

## 数据残留

18. `raceHistory` / `race_count` 在开局快照（t0-t2）可能残留上一局数据，t3 起才清零（game1458 实测 t0 显示 14 条）：比赛计数从年界稳定后读取。

## 性能（实测数据，game3098）

19. **瓶颈是模型往返次数，不是 JSON 解析**：200KB digest 的 `ConvertFrom-Json` 仅 24～61ms、`Get-Content -Raw` 11ms、python `json.load` 2～4ms、node `JSON.parse` <1ms，都不是瓶颈。真正的开销是每个工具调用/子代理各占一次模型往返。
20. **子代理有约 20 秒的固定成本**：一个只跑 2 步的最简子代理实测墙钟 21.3s（派发到首次工具调用 9.1s，第二次调用到最后答复 12.2s）。因此 7 路子代理**串行**派发 ≈ 6 分钟；**同一条消息并行**派发只需一个最慢子代理的量级。
21. **探查可以压成一次调用**：单进程脚本一次读 digest 直接产出全部 9 组紧凑结论，实测约 1.0s（含解释器启动）。9 组的取数逻辑彼此独立、无相互依赖，没有必要拆给多个代理。
22. **`cargo run --release` 比直接调 exe 慢约 1.8s**（3.04s vs 1.23s）：已编译过就直接用 `target/release/umaai_review.exe`，别每次走 cargo。
23. **brief.md 已落地（bin 侧 brief.rs + `templates/brief.md.j2`）**：常规复盘**不再需要分组探查**——bin 一次运行即产出 `brief.md`（约 11KB，digest.json 的 1/18），六问事实已预答（含五维显示值逐年与触顶维、数据健康度），SKILL.md 第 3 步一次 Read 即可。上面第 19-21 条的探查开销仅在 brief 缺失（旧版 exe / exe 不可用）时才会遇到。**排版与文案在外置模板里，改措辞不用重编译**；跨块关联与判定（分段边界、存疑判定、排序）仍在 brief.rs。

## 运气分读数：`turn_delta` 的语义（game3101 实测）

24. **`turn_delta` 是「变化到达本行」，不是「本回合内的变化」**：行的 `turn_delta` = 本行 `total_luck` − **上一行**（上一个决策点）的 `total_luck`。实测 t72 行 `turn_delta=-1204.10`，而 `total_luck(t72) − total_luck(t70) = -2496.35 − (-1292.26) = -1204.09` 完全吻合；t74、t76 同样吻合。因此：
    - 同回合多行求和 = 「上一回合末 → 本回合末」的变化，作为「回合合计」是对的；
    - 但**按 `turn >= N` 切片求和会多算一段**——`super_ramen_stats` 累加 `turn>=72` 的行，实际覆盖的是「**t70 → 结束**」(-1684.48)，而不是「t72 → 结束」(-480.38)。两数在 brief.md 的 §2 超拉期净变与 §7 super_ramen_luck 中必须同源，否则同一期间出现两个数字，LLM 会写歪。
    - 一句话：**切片边界处要么用「段前最后一个决策点」作起点，要么严格用 `turn > N`**，不要混用。
