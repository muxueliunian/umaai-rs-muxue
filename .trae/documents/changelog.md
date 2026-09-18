# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。记录应尽量精简，每条修改一行，不包含代码细节。

## 2026-09-18
- **采集自检用例修复**：`test_formal_collect.py` 仍按旧的两参数签名调用 `check_plan`，在驱动改成「命令行声明意图 + manifest 提供事实」之后一直报错；补上该轮的 2048 / 22800 声明，并新增一条「命令行与 manifest 不符必须拒绝」的守门断言
- **R6 教师数据采集配方冻结**：`scripts/collect/formal1024_0918/`——gen2_v1 空间、每候选 1024、68400 有效根（general 每构成 15000，地区 1200/3600/3600）、roll-in 换 `ens_NT4096_AllHistory_g123`、独占号段 `[301000000,1500000000)`、20 小时截止；留出的 250 个组合与 0914 是同一批，泛化面板保持可比；号段登记在 `nn_model_registry` 第 7 节，云端资产参考目录 `target/gen2_cloud_assets_0918/`（不在 Git 内）
- **采集清单生成器参数化**：`prepare_gen2_formal.py` 的搜索数/号段/模型/配方名/分层根数/截止全部改为命令行必填且无默认值（口径必须写在命令行上留痕）；新增号段宽度提前检查（index 按 `cycle×4288+plan` 生成，跨度由候选序号总数决定而非有效根数，不足时直接算出需要多宽）；任务发射顺序改为地区层在前——region_y3 每根 120 候选、约占总算力一半又最不可替代，原顺序把它压在截断风险上；用 0914 原参数重生成与既有清单**逐字节零差异**
- **采集驱动守门改造**：`run_formal_collect.py` 的 7 处 `2048`/`22800` 与 2 处写死空间名改为从清单读取，同时新增必填 `--expect-search-n` / `--expect-target-valid` 与清单交叉核对——「单改 manifest 无法静默改变开跑口径」这条防线原样保留，且本轮意图会落进 `evidence/args.json`
- **`exp_config` 实验口径模块移植进 master**：评分轴与 `pt_favor_rate` 在 `init_global_with_config` 之前钉死、初始化后回报生效值，接入教师采集 / 空间基准 / 根基准 / 客户端整局基准四个入口；新增可选 `--pt-favor-rate`，不传时钉死为代码常量 2.0 且**不回落** toml（与当前 `game_config.toml` 同值，采集结果逐位不变）；上游的 `--selection` 轴切换开关不采纳，本地仍固定 Score 轴
- **规则修复消融判定**：`d9374e8`（合宿诀窍全 MAX + URA 72-77 角标不落库）对**离线终局评分完全中性**——同种子 4200 局逐局分数全同，只有三个第 3 年观测计数器变（诀窍获得 −6.9 / 溢出 −5.9 / 友情回合 −3.0，因为末段拿到的诀窍已无处可用）；故新旧手写策略 +1573 的配对差全部归策略本身，且旧教师数据的标签在新规则下依然有效
- **`formal4096_newteacher_task.md` 标注为历史文档**：其对应的 50000 根 @4096 采集实际已完成（general 40000 + 地区 1428/4286/4286），但正文自述「尚未开跑」已陈旧，加横幅避免照此开跑
- **同步上游 xulai 至 04c739c 并纳入 PR31 / PR32**：首次以真正的 merge（含 ancestry）合入上游，默认卡组与马娘、PT 重视程度、地区弱位覆盖权重等跟随上游；刻意保留三处分歧——选动作仍走真实评分轴（教师标签与历史面板都在该轴上）、平局判据与选择轴同口径、地区字段保留定长表示与形状白名单（不把看不懂的帧补零成完整局面）。
- **受影响的冻结快照按本地评分轴重抓**：整局门控关闭用例的评分、五维与技能点，以及合并路径的特殊选择调用数；合并搜索路径按面聚合后已能暴露决策摘要，相应探针用例的期望随之翻转。
- **拉面杯第十二轮组合档进 preset（默认值）**：把放宽搜索界后经两个独立随机卡组池验收的组合设为 `RecommendedRamenTrainer::new()` 默认值（近上限连续定价窗口、已满位三类 PT 定价、状态缺口/溢出强度、预留上限与口径、超级拉面自适应范围、外出基准分、逐卡 Hint 倍率），随机组相对上一版 preset +130.4/+147.6、相对更早的 base 累计约 +517/+552，固定卡组不显著；同步重抓受影响的 6 处冻结快照（其中 2 处进入 preset 前就已过期）、把预留口径单测改为显式钉 token 以免随 preset 漂移，并按新默认值刷新 final-check 期望表
- **局末自动打包本局游戏记录**：umaai 末回合第 2 份快照（拉面 `turn77_2`）处理完写 meta + SVG 后，把 `logs/game{id}/` 打成 `logs/game{id}.zip`（包内条目相对原目录，去掉 `game{id}/` 外壳）并清理原目录——zip crate 依赖进 workspace；切局/退出兜底（`switch` / `process_exit`）不打包，中途停止局保留原目录方便人工排查；终端 stderr 绿色绝对路径（dunce 去 `\\?\`）+ `info!` 日志一份（json 模式友好）

## 2026-09-17
- **拉面杯逐卡 Hint 精确估值（可选）**：把固定 Hint 价值换成按卡面 Hint 等级与剩余可得的逐人头精确折算，默认关闭；一批全新 160 副随机卡组独立验收随机组 +174.0 [+153.0,+195.0]、预设 +77.7、固定卡组 −121.0 未证实，定位与 supermode3 同为“随机卡组可选”，关闭路径逐位不变。
- **拉面杯可选策略与验证**：新增近属性上限的连续技能PT估值和按终盘缺口选择超级拉面范围的可选开关，保留默认行为；提供固定/随机卡组配对工具、冻结复现配置、结果分析及边界回归，注明固定卡组和线上/MCTS适用限制。
- **ga_lab 最优策略合并（默认卡组）**：game_config 卡组切换 GA 通解骨架（速2耐1智2：待兼诗歌剧/千明代表/名将怒涛/优秀素质/青春永驻 + 骏川手纲），8 马 340 局同种子配对显著（Δ+516，t=3.36）
- **ga_lab 最优策略合并（9 旋钮组合档进 preset）**：已满位普通/超拉面 PT 定价上调（16→37 / 0→35）、弱位覆盖查表→35 直值、友情词条 1.5→0.4、掌握度 6→8、牺牲上限 140→200、拉面窗 0.10→0.15、剧本PT前瞻 0→0.15、Y1 pt_rate 16→56——4 马×2 种子块 420 局配对 Δ+1394 t=12.55（8/8 单元显著）；ramen_weak_train_boost 单测大负（−1017）明确不采纳
- **bench_base 新增 `--deck` 覆盖卡组入口**：显式 idrank 串指定卡组跑批（跳过 preset builds，标签 custom_deck），配卡对照实验用
- **重抓两个存量漂移基线**：`test_yearly_observability`（68118→70138，五维同步）与 `test_ramen_three_stage_action_unchanged`（7 组 rollout 均值）——GA 方向定稿进 preset 后当前行为
- **合宿训练诀窍填充修复**：`fill_feeling_gauge` 门控放宽为「合宿无条件三种槽全 MAX」——协议合宿回合角标全 0（None）时训练也能每种 +1 诀窍（此前在线对局合宿训练比休息/出行少 3 诀窍/回合，MCTS 合宿训练被系统性低估，是「高体力一选休息」的根因）；同步 URA 回合（72-77）角标照常抽签但不落库，训练不再产生诀窍槽（与在线一致）
- **新增合宿/URA 诀窍填充守门单测 ×2**
- **智力体力豁免语义修正为白名单**：`wisdom_vital_floor` 豁免带内只放行 智训练/休息/普通外出/治病（原实现=整个 40 门放开，速度/耐力等高位失败率训练在低体力下大亏；默认 MAX 仍不豁免，preset 行为逐位不变）
- **补齐已满位 PT 定价实验 token**：`trd/trdsh/trds`（N/100，文档早有命名从未实现）接入 `with_tokens` + 解析单测
- **重抓两个存量漂移基线**：`test_ramen_three_stage_action_unchanged`（7 组期望均值）与 `test_yearly_observability`（BASELINE_SCORE/FIVE）——最近策略调整与合宿诀窍修复后当前行为（干净 master 亦红，score/five 与本次改动逐位无关，纯同步）
- **混合叶估值路由（实验入口，默认关闭）**：根部可按候选数与局面段分别走整根完整估值或截断估值，逐根记录路由归属、推理请求数与耗时并导出对账表；默认路径、请求口径与随机流不变。

## 2026-09-16
- **第3年地区单/多点整局配对扫描**：新增 `region_y3_single_focus` 扫参入口；seed42×100 全 101 构成 + seed61444×50 复测——混合档显著负、纯单点档玩家真实 build 大亏（≈−1100，t≈−5）→ 不采纳，维持现有按 build 自适应公式
- **运气分重放分析工具 + 三项修复**：新增 `luck_replay`（快照重放 → 明细/波动 CSV）、`luck_probe` 探针、`scripts/plot_luck_trend.py` 趋势图；修复 `selected_regions` 空数组整份丢弃、只吃面回合计不到运气分、年度 RMJ 派生状态缺失（第 2/3 年期望虚降 ~2300）
- **在线决策记录（每局一目录）**：umaai 实时运行按 `single_mode_chara_id` 落 `logs/game{id}/`（`thisTurn.json` 原文 + `decisions.csv` + `meta.json`，`luck_record` 默认开可关）；明细 schema 抽到 lib 与 `luck_replay` 共用，CSV 改走 `csv` crate 结构化写入（离线输出逐字节不变）
- **局末自动出 SVG**：收到末回合第 2 份快照（拉面 `turn77_2`）处理完即写 meta（`game_end`）并生成 `logs/game{id}/luck_trend.svg`（3 子图：边框/回合数轴/纵刻度与竖排标题/图例「蒙特卡洛估分…」/署名；切局退出兜底；终端绿色绝对路径可跳转）
- **比赛回合手写 fallback 上屏修复**：`HumanReadableSink` 对空分 fallback 决策（地区选择 / 比赛回合单候选 / RamenSelect 单候选）统一打印「选择…（手写逻辑）」，比赛回合不再静默（JSON 模式不受影响）
- **叶价值截断试点（实验入口，默认关闭）**：新增按深度截断的叶估值臂与逐根诊断输出（按根与按候选两种加权、请求数与墙钟分别记账），支持按固定根清单续跑；仅实验工具可见。
- **新教师数据训练与闭环诊断记录**：登记新教师数据的训练配方、导出与闭环评估口径，以及分项诊断的结论与证据边界。

## 2026-09-15
- **地区弱位覆盖按智卡数查表（方案Ⅰ 固化）**：`region_weak_cover_weight` 改三态（0=按智卡数查表 智≤1→12/智≥2→0、<0=关闭、>0=固定值实验），与弱位训练偏好查表配套；全 101 种构成配对验证智≥2 零变化、智≤1 加权 +220
- **地区无卡位惩罚参数化**：`score_region` 硬编码 -10 提为 `region_waste_penalty`（默认 10 行为不变），扫描 5/15/20/30 均负收益，确认 10 为平衡点
- **地区 youqing 权重复核**：1.0~3.0 间为公共线性缩放无区分度（逐位不变），0.5 仅扰动 7 构成且方向不可靠，维持 1.5
- **主训位翻倍加分实验（未采纳）**：`region_main_bias_bonus` 让地区覆盖 build 卡最多位时 bias_sum 再 + 该位卡数，全 101 构成验证全档大负（3speed 类受伤最重），维持线性 bias_sum，字段保留可配
- **地区打分移除恒量项**：`score_region` 删 pt_bonus/hint_count 项（同年候选间恒定，不改变 argmax，验证逐位不变），删除孤儿字段 `region_pt_weight`；`region_hint_weight` 保留供吃面选择路径
- **地区权重扫描入口**：`bench_compositions` 新增 recommended 档（正式推荐 preset）与 `--region-weak-cover / --region-youqing-weight / --region-waste-penalty / --region-main-bias` 参数；`RecommendedRamenTrainer::with_region_weights` 覆盖入口
- **MCTS pt_favor_rate 扫参定档 2.0**：1.0~12.0 七档 N=1024 扫参（2.0~3.0 免费换 PT、4.0 起转亏、PT 增益 8.0 后饱和）+ 1.0/2.0/2.5 三档 N=4096 深测（5 速系卡组 10 局配对：2.0 评分损失不显著 t=−0.04、PT 增益显著 t=+6.3，2.5 边际仅多 +41pt）；正式默认 1.0→2.0，新增扫参汇总/绘图脚本
- **拉面 MCTS 决策候选分改真实评分**：`stash_last_summary` 缓存 `candidate_scores` 从 score_pt 轴（含 pt_favor_rate 缩放）改为 calc_score 轴——运气分 baseline、action_luck 与 AIRed 候选显示不再随 pt_favor_rate 虚增；选择仍走 score_pt 不变
- **urafile 读取改生产者-消费者模型（用户）**：thisTurn.json 监听从「收到事件后合并排空 + contents 去重」改为后台 producer 线程每次写事件读完整 JSON 进异步队列、主循环逐个消费——取消主观丢弃中间回合快照的路径（内容相同的冗余事件除外），notify 缓冲溢出错误事件改重读兜底
- **清理多余配置（用户）**：删除遗留 constants 快照文件；ramen_turn_inspect 展示同步 RamenMctsTrainer 字段变更（selection → use_combined_ramen_select）
- **推理侧车的图捕获与批量调度（默认关闭）**：侧车支持整段图捕获复用与按根自适应批量档位，两者互斥、默认关闭，开启前后输出逐位一致；同轨迹的重复推理改为复用上一拍结果（可开关）。
- **候选排序改用浮点全序比较**：避免边界取值下选择结果不确定。
- **整局网络直接决策（客户端，opt-in）**：客户端支持整局由网络决策的模式（默认仍走搜索）；育成身份登记与运气数值跟踪分离，不再在无搜索评分时把同一局的每一帧判成新局。
- **前向调度独立探针与性能基准侧车**：新增对照探针与本地基准脚本（严格 FP32、固定批量、仅传二进制结果与控制回执）。
- **实验文档**：新增 NN rollout 等价性能优化计划、叶估值实验交接任务书、新教师数据训练计划与云端采集任务书。
- **仓库卫生**：根目录一次性实验驱动脚本不再入库；修正客户端协议测试里把标准库解析函数当外部 crate 引用、导致该 crate 测试整体编译失败的既有问题。

## 2026-09-14
- **拉面手写策略评分换PT参数**：新增可调已满位训练 PT 折算价（`pt_tradeoff` 普通档 / `pt_tradeoff_shining` 有彩圈分级 / `pt_tradeoff_super` 超拉面档）——训练位主属性已满时属性收益为 0、只剩 PT，策略按独立价重估该训练候选，避免按 `pt_rate` 高估后终盘贪练已满位
- **评分换PT玩家配置**：新增顶层可调字段 `ramen_pt_sacrifice_score`（为多拿总 PT 最多愿意牺牲的总评分，默认 0 = 评分优先），按实测标定分段映射到已满位有彩圈定价（0→36/≤60→44/≤160→52/其余→64），`default_config.toml` 注明范围与对照表、`game_config.toml` 顶层可覆盖
- **实测调优结论**：彩圈分级实验确认 PT 产出由彩圈数主导（0圈≈40 / ≥1圈267-340）且与属性是否满无关（已满232 vs 未满225）；最优档=有彩圈定价36，100局 7 build 平均 +533 分、skill_pt -90；定价>64 会导致评分与 PT 双降
- **策略变体跑批入口**：`bench_base --tokens` 支持 `RecommendedRamenTrainer::with_tokens` 变体（trd/trds/trdsh/ptrate 等），实验不混入 preset
- **MCTS 评分换PT公式重设计**：`RamenGame::search_score()` 覆盖 trait 默认，`score_pt` 基于 `score_parts()` 使用干净公式 `skill_score + skill_pt × 2.0 × pt_favor_rate + five_status`（无 ×0.37 缩放），`pt_favor_rate=1.0` 等价 `calc_score()`；`pt_favor_rate` 代码默认 8.0→1.0
- **移除 RamenSelection 枚举**：ramen MCTS 统一走 `best_action_pt_idx()`（score_pt 口径），删除 `bench_base --search-selection` CLI、`RamenMctsTrainer::with_selection()`、`RamenSelection` re-export
- **友人卡 rank=0（未突破）合法化**：`newgame` 校验补 rank≤4 范围检查，修正测试断言（rank=0 应合法、rank=5-9 非法）
- **可选的神经网络地区选择**：客户端新增显式的地区决策来源配置（默认手写，网络为 opt-in）与独立的地区模型路径项，与温泉模型路径分开；网络只接管实际对局的三次地区选择，其余阶段、事件选项、比赛守门与搜索内部续跑全部保持原装配，模型不进搜索评估器。
- **地区接管共享实现**：主程序与整局评估工具改用同一份地区接管逻辑，评估工具额外的同局面手写对照改为旁观钩子挂入；决策路径改为按训练员类型泛型，不再绑定单一实现。
- **地区策略的冲突诊断**：未启用推理 feature 却配置网络、模型或旁车缺失、维度不符、地区搜索开关未关、地区候选为固定组合等情况一律启动即报错并给出修复建议，不静默回退手写、不出现「打印一套执行另一套」。
- **上游同步（xulai）**：吸收上游 09-13 的链式决策输出、计算完成信号与文件监听器修复（条目见下一节）；本地独有的链路成本工具跟随新增的输出接收方参数。
- **地区接入回归覆盖**：新增配置冲突与模型错误路径、三年地区候选解码、整局恰好三次地区决策与零搜索内推理、决策与计算完成信号次序、纯手写构建下拒绝启用网络等测试；修正评估工具中纯手写臂取值与其解析测试不一致的既有问题。
- **决策来源不再靠「有没有搜索评分」推断**：决策输出增加来源标注（走既有的剧本扩展字段，不新增顶层字段、不伪造评分、不改运气挂载口径），屏幕对无搜索评分的地区决策按来源显示，未标注时用中性文案；网络接管的地区决策改为给出本次决策自己的摘要，既不挂上一步搜索的理由，也不会被印成手写。
- **模型加载校验推理图本身的输出契约**：加载阶段核对图的输出个数、元素类型与形状，只读图的静态信息、不跑推理也不计入推理计数；此前只校验旁车声明，图导错要到第一次真实决策才暴露。
- **实验入口与客户端共用地区适用性检查**：整局评估工具的仅地区替换臂改为复用客户端那套检查（地区搜索开关、固定地区候选），作用于命令行覆盖后的生效配置；模型路径仍只由命令行给出，其余臂不受地区专用限制。
- **测试隔离**：地区默认值改用独立配置样例验证，不再读取用户的个人配置；地区候选测试改走不依赖全局配置的纯枚举接口；需要落盘的测试统一改用工作区构建目录下每次唯一的临时目录，清理前核对路径范围。
- **卡组合法性补齐**：采样空间枚举增加支援卡之间的同角色冲突判定（此前只查马娘与支援卡），友人卡一并纳入；旧空间卡池角色两两不同，枚举结果与顺序未变。
- **具名采样空间版本**：新增可注册的空间版本（马娘、卡池、构成一并冻结）与第二代空间；旧空间常量与既有导出口径原样保留，两者不共用身份字段。
- **显式身份采集口径**：采集与导出增加按版本名走的无指纹口径——完整空间与配方逐字段比对、模型和游戏基座保存原文副本并逐字节核对，导出完整马娘卡组字段代替新算组合哈希；未指定版本时完全走既有指纹口径。
- **第 1 年地区采样配额**：新增独立的第一年配额参数，默认关闭且关闭时分配逐位不变，原有第 2/3 年数组下标含义不改。
- **采集分项计时**：采集器输出初始化、空间枚举、模型加载、采样、搜索导出、写盘与读回校验的分项墙钟，仅测量不参与判定。
- **正式分层采集与恢复**：冻结新空间的构成和地区配额、留出组合及独立世界清单，支持按有效根目标使用同层备用序号；云端驱动记录原始资产、截止时刻和实时日志，逐任务验收并导出，失败不自动降预算。
- **完整组合字段读取**：新增完整字段读取及按预登记组合切分，旧键与新字段混训需显式映射，禁止静默回落；原始续跑截取工具透传完整字段。
- **扩采定价配方与驱动**：新增受版本管理的冻结测试清单、两机互不重叠的运行清单与顺序采集驱动（共享墙钟预算、实时落盘日志、峰值内存、拒绝覆盖证据、不算哈希）。
- **搜索测量接口**：增加可选搜索与决策探针，记录续跑计数和分层耗时，补充结果及随机流中立性验证，保持默认决策语义。
- **独立实验入口与配对驱动**：新增客户端链路成本和整局评估工具，支持可选 NN 直接决策及仅地区替换；配对驱动统一配置与预算、保留实时日志和退出码并拒绝覆盖证据，正式客户端默认行为不变。
- **NN 研究决策与工作树收敛**：登记地区局部接入实验、覆盖扩展方向和待定采集方案，记录验证边界；收回旧采集工具的临时测量扩展，保留独立实验工具与最小公共接口。
- **空间评估的具名版本与清单跑法**：空间评估工具支持点名采样空间版本、按完整马娘卡组字段的清单挑选计划并保留原始下标（逐计划世界与全量跑一致），并提供不计算任何内容指纹的身份口径（原样记录生效配置与计划清单）。
- **整局评估工具扩展**：新增纯手写对照臂，逐局地区记录补上计划编号与候选数，便于按计划配对核对。
- **旧导出组合身份补齐**：新增按实际采样计划补出完整马娘卡组字段的工具与对应的空间枚举复现测试；训练数据加载改为只认完整字段，旧的组合键不再参与切分，混训需显式映射、禁止静默回落。
- **训练与标签的口径加固**：标签目录记录实际对齐字段而非指纹并在训练侧逐值核对，多目录按位置配对加载并拒绝跨目录重复样本，断点续训以存档内的划分清单为准并逐项核对。
- **分析脚本**：新增开发验证组合清单生成、两种标签预算的逐样本描述性比较，以及两侧逐局日志的配对分析（按计划与世界配对并逐值核对世界种子）。
## 2026-09-13
- **拉面链式决策 JSON 输出顺序修正**：链式决策#1（不吃面 / 训练 turn1）在 `compute_next_step` 之前先 emit，下游先收决策结果再收"计算中"通知
- **拉面 RamenSelect 决策输出补齐**：吃面 / 不吃面在合并搜索路径下均合成 `ramen_select` 决策信息下发（不吃面同时作为链式决策#1 单独输出）
- **每轮计算完成信号**：拉面 / 温泉决策输出后 emit `compute_done`，无决策的 Begin 早退回合（事件 / 结算 / 数据不全）也补发，保证 `compute_start → compute_done` 成对
- **watcher 快速连续写入健壮性**：notify 错误事件不再退出进程，空 / 半截 / 文件锁读取带重试与两次一致校验，唤醒后排空事件队列合并为一次处理最新内容
- **Windows 构建脚本**：重新启用图标资源编译和栈链接参数输出（与 09-11 停用相反）
- **NN 实验记录与模型登记整理**：NN 管线文档补记近期数据增量、网络扩容与追加数据对照的结论、证据边界与下一阶段方向；新增模型、数据与评估世界登记表；训练侧设计文档说明容量自动选择不等于实验配方

## 2026-09-12
- **同步 xulai 上游至最新**：真实合并上游 rollout 手写基策与状态复制性能优化、候选内部并行、umaai 决策字段与输出调整、数据更新（新马娘与支援卡）及默认搜索量提升
- **合并取舍**：本地保留 fat LTO 与多代码生成单元的 Release 配置，暂不启用本机 CPU 指令集；隐藏风味替换上限继续使用统一常量；手写策略协议输出随上游删除评分明细字段
- **strict_rollout 与并行搜索**：候选内部并行后仍报出序号最小的失败；UCB 追加组改走同一执行路径，失败同样受该开关约束；补充多线程一致性、失败槽与追加组报错测试

## 2026-09-11
- **模拟状态与规则计算**：内联继承因子和卡组计数，支援卡面板借用全局只读卡表，共享事件采样分布，简化做面可行性计算，系统事件和概率查询仅在失败时构造错误，减少复制与分配。
- **选面与训练评分**：预演借用原局面并显式指定候选面，复用训练候选、评分空间、训练基础值、Hint 与羁绊估值、友人动态估值、体力结果和地区窗口分量，各面的实际加成与最终评分分别计算；rollout 省略评分明细，普通决策日志和协议输出保持完整。
- **性能验证**：补充状态隔离、逐碗计算、候选顺序、事件随机流、Hint 切换、必赛分支和普通策略日志的一致性检查，记录生产参数正反序配对测量及源码、配置和产物校验信息。
- **Windows 构建脚本**：停用构建脚本中的图标资源编译和栈链接参数输出。

## 2026-09-10
- **本机 Release 性能优化**：启用速度优先优化、ThinLTO 和本机 CPU 指令集，减少模拟、评分与搜索中的重复分配、复制和计算，复用事件队列与地区评分，候选内部模拟并行执行并按原序归并；rollout 跳过原因文本生成，保留数值评分分解、正常决策日志与协议输出。
- **性能基准与验证**：基准继承游戏线程和搜索配置，补充分配复用、评分、随机流、动作与事件轨迹、协议输出及跨线程失败槽一致性检查，记录生产参数整局与高预算搜索根的配对测量口径和复测步骤。
- **main.rs 按职责拆分重构**：主程序收敛为薄调度（CLI / 初始化 / watch 循环分发）；新增 `decision/` 目录（决策后处理：luck 计算与决策输出，`luck_score` 一并移入）与 `scenario/` 目录（温泉 / 拉面各一幕块：含 newgame 检测、切局、决策计算与 emit）；行为等价
- **连续决策中间状态输出时机修正**：中间决策的「计算后续动作」提示与 `compute_next_step` 通知从主循环移到决策循环内部、在真正执行下一步决策（可能耗时）之前发出
- **比赛回合策略输出修复**：拉面比赛回合仅一个固定动作、MCTS 不搜索导致无输出——为固定动作合成决策信息使其在屏幕 / JSON 上可见，不挂 luck
- **`--json` 开始接受数据时发 `connected`**：仅 json 模式、watcher 就绪进入监听时 `emit_info("connected")` 通知 AIRed 连接成功
- 修复空间评估工具误报自选比赛未达标的问题
- 支持教师数据预算对照与训练常量冻结，并增加数据、标签预算一致性检查
- 训练支持按数据来源设置抽样重复次数，只作用于划分后的训练集
- **`search_n` 预算记录订正**：1024 的选定来自教师闭环 sweep（被测量是整局闭环均分），此前误挂到一项计划中的 top-1 一致率实验；同时补记该 sweep 未实测 2048/4096、也未验证更大预算对学生的收益，以及实际采集批次全部为 512 而无改动决策记录
- **单候选决策点免推理**：`RamenNnTrainer::prepare_decision` 在候选只有一个时直接定案，省掉整次网络往返；只收敛「需要推理」这一种结果，不影响守门与手写转发分支，候选落格检查保留
- **改动前后对拍**：固定根上逐 rollout 终局评分与剩余网络决策均逐字段一致，请求数与墙钟均下降，批利用率同时下降；运行身份、数值与证据边界记于 `nn_pipeline_plan.md` 第 14 节
- **新增守门测试**：钉「多候选仍需推理 / 单候选直接定案 / 该路径不消耗随机流」；推理请求计数是进程级全局量，并行测试下只作观察不作断言
- **新增对拍工具 `compare_root_bench.py`**：比较 `ramen_root_bench` 的逐 rollout 与逐决策输出，直接比较实际字段不用哈希；逐决策按 rollout 顺序对齐，不依赖会随请求数变化的序号，并附缺组边界的单元测试
- **`ramen_root_bench` 种子口径注释**：补记它与 `ramen_space_bench` 的基种子口径不同，只有 `--plan-index 0` 两边对齐，跨工具比较前须先对齐有效基种子

## 2026-09-09
- **决策间 `decision_kind` 顶层字段 + `scenario_extra.ramen_action`**：partial decision 类型分发——main.rs 在 select_action 前按 `RamenStage` 填 `decision_kind`，onsen 填 `train`/`event`；`ramen_action` 由 `RamenAction.to_string()` 给出含吃面 + 隐藏诀窍 + 操作三阶段动作串（AIRed 端只显示不解析），合并路径一条 JSON 表达、三阶段路径按 chain 顺序分条
- **新增 `candidate_descriptions` 字段**：与 `candidate_scores` / `candidate_n` 严格同长同序同截断，供 AIRed 映射拉面组合动作名（onsen 取自 `SearchOutput.actions[i]`、拉面 MCTS / 手写策略分别缓存到 `LastSearchSummary` / `LastDecisionSummary`）
- **`DecisionInfo` 简化 + `scenario_extra.reason` 挂载**：删 5 个 stub 字段（reason / search_depth / visit_count / score_breakdown / elapsed_ms），保留 candidate_scores / candidate_n；拉面从 `LastReasonSink` 取 `DecisionReasonData` 挂到 `scenario_extra.reason` 完整透传 human reason 信息
- **`--json` 输出类型扩展**：stdout 顶层 `type` 区分三类消息（`decision` / `info` / `error`），去掉 `schema_version`；`info` 仅 event 取值、`error` 仅 message——在 watcher init、watch loop 入口、拉面链式决策中间、切局、失败五处按需发射
- **AI 不再推进游戏状态**：拉面 calc_ramen_training 与温泉 calc_onsen_training / calc_onsen_event 改为只调一次 select_action 出推荐、不再 apply / next；主循环每次 watch 收到新 JSON 后从零重建 game 重新计算，两次 JSON 间不互相依赖
- **温泉 / 拉面回合头部打印**：human mode 在每次计算后打印马娘状态 / 剧本信息 / 训练分布；json mode 跳过这些屏幕输出
- **JSON 模式 stdout 净化**：计算完成提示「计算完成，等待新数据...」、启动横幅走 stderr；`[按 F2 保存当前回合状态]` 等人类调试提示在 json mode 跳过
- **ctrl-s 热键功能临时停用**：tokio::spawn(hotkey_handler) 注释掉（crossterm 无限 poll 占用 worker 配额、AI 通道下无意义），后续重构时按 feature gate 恢复
- **版本号 / 横幅升级**：`umasim` / `umaai` Cargo.toml version 升 0.2.x → 0.14.0；启动横幅 "UMAAI 0.26" 改为 "UMAAI-Ramen"
- **拉面 / 温泉连续决策**：特定场景在前一决策基础上继续生成下一决策——应用上一决策并推进到下一阶段后再出推荐（第 1 回合训练后接地区选择、选"不吃面"后接训练选择），一回合内可连续下发多段决策
- **连续决策触发条件修正**：仅在第 1 回合训练时继续，避免其余回合误触发
- **拉面内部状态字段清理**：移除仅用于状态导入内部判断、无需持久化的字段及对应类型，协议不再写出
- **拉面当前生效面修正在下**：仅在效果列表非空时才透传，同步更新样本导入测试断言
- **启动日志精简**：去掉"载入用户配置 / 载入默认配置"的日志输出
- **human 输出改为运气行**：删去 AI 选择 / 理由两行，输出期望评分与运气分（本局 / 本回合），四舍五入为整数，本局运气按区间着色
- **期望评分叠加每回合加成**：期望评分与运气分统一计入"每回合比手写逻辑多的分数"，按剩余回合数加权，初始基线按第 0 回合计算
- **拉面决策理由输出解耦**：理由的原始数据始终下发供宿主使用，可读文字仅在诊断模式上屏，避免双打印
- **feeling_guage → feeling_gauge 改名**：上游已修复误拼，协议与 umaai 端 `RamenStatus` 字段随迁 `feeling_gauge` / `feeling_gauge_gains` / `feeling_gauge_gain_base`，文档同步更新
- **拉面 into_game 严格按协议重建 base**：弃 `RamenGame::newgame` 打补丁，改由 `parse_basegame` 重建（Uma / Friend 走 `parse_uma` / `parse_friend`、`five_status_limit` 取协议值、丢弃 `friend_event_ids`，友人事件 / 五维上限视为外部输入）；新增 `RamenGame::from_base_game`；`parse_basegame` 卡组循环补 persons 越界守卫
- **地区选择（手写 fallback）决策可输出**：`region` 门控关闭时 `last_decision()` 为 None 导致无结果——`decide` **仅对 `RegionSelect`** 合成决策信息入链 emit（其余 None 阶段保持旧行为不合成）、不挂 luck；human 紫色显示「选择地区[...]（手写逻辑）」
- **合入上游 AIRedirector 线与吃面 PT 规则变更**：上游 13 个提交并入本地 master；除 changelog 外无冲突，`game.rs` / `state.rs` / `bench.rs` / `flat_search.rs` / `ramen_mcts_trainer.rs` 五个双方都改的文件全部自动合并
- **rollout 推理录制钩子改名避开撞名**：`RamenRolloutTrainer` 的 `DecisionSink` / `DecisionSnapshot` / `with_decision_sink` 改为 `RolloutInferSink` / `RolloutInferSnapshot` / `with_infer_sink`，与上游 `output::DecisionSink`（决策主干输出）区分——两者语义无关，同名会读错


## 2026-09-08
- **新增 adapter_spec 文档**：整理 SendGameStatusPlugin 与 umaai 协议对接的易混淆点（feeling_guage 拼错 / persons/personDistribution 适配 / playing_state 含义 / 数据获取不全判定 / 超级拉面回合处理 / 阶段来源三态等）
- **Step 7 拉面剧本协议与主流程接入**：阶段派发按 source / active_effect / playing_state 三方联合；turn ≤ 1 直接进 Train；playing_state=45 进地区选择；超级拉面回合按 active_effect 区分丢包/决策；数据获取不全 warn + 不派发
- **拉面 persons layout 与协议对齐**：理事长 / 记者 / NPC 按 adapter_spec 排布；记者出现回合修正为 turn > 12；151 样本驱动测试同步更新
- **GameStatusBase 协议字段扩展**：新增 source（snake_case）、single_mode_chara_id（snake_case，单调递增切局键），兼容旧 JSON
- **拉面 AI 主循环接入**：main loop 拉面分支从占位升级为完整流程；切局检测改用 single_mode_chara_id，缺失时退化到 uma_id
- **拉面单回合诊断 binary**：新增 CLI 二进制，指定单个 ramen JSON 即可跑完整 into_game + MCTS，human-readable 输出
- **集成文档 §3.4 同步**：Step 7 实装方式（协议层 into_game + 单回合诊断工具）替代规划期描述

## 2026-09-07
- **Step 8 AIRedirector C# 端极小改动**（独立仓库 URA_Plugins/AIRedirector，已合 a8edca8）：`UmaAiProcessStartInfo.Create` 加 `jsonMode` 参数（`--json` 开关）；`AIRedirectorConfig` 加 `Ramen` / `Ramen_Path` 字段；`Class1.StartProcess` 加 `jsonMode` 形参 + 拉面分支；`HandleOutput` 试 `TryParseUmaAiDecision` 解析（`schema_version` 识别）后路由 `ApplyDecision`；`UmaAiDecision` record struct；配置文件 UI 加拉面分支（`ConfigAction.EditRamen`）；smoke test 加 4 个测试（拉面 / 温泉 / 非 JSON / 缺 schema_version）—— 实际 Windows 编译与运行测试延后到切回 Windows
- **Step 7 RamenGame::from_external_state 完整覆写**：`RamenState` 新增 `feeling_guage_gains` / `next_scenario_pt` / `feeling_guage_gain_base` / `active_effect_array: Vec<ActiveEffectEntry>`；`ActiveEffectEntry` 协议类型（category 语义搁置）；`protocol/ramen.rs::into_game` 完整实现（base 字段 + 5 人卡组 + 友人/理事长/记者 + events + 12 个 ramen 段字段 + stage dispatch 按 playing_state 1/5/45/46/48）；驱动 151 份 `logs/GameStatusSend_Ramen` 样本 round-trip 校验 scenario_pt / current_ramen / selected_regions / super_ramen 全透传，max_scenario_pt=7500 / stage 分布 Train 145 + Settlement 5 + SuperRamenSelect 1 与预期一致
- **Step 6 parse_game scenarioId 分发**：protocol/ramen.rs 加 GameStatusRamen 骨架（scenario_id=14） + mod.rs `ParsedGame` 枚举 + `parse_game_by_scenario`，main 按 12/14 分发（拉面侧 Step 7 接入 AI 主流程）；7 个 protocol 测试
- **Step 5 LuckScoreTracker + emit_with_luck 接线**：luck_score.rs 新增 tracker + 切局检测 + 按局数加权 baseline，main emit 走 `emit_with_luck` 挂 scenario_extra；移除 ratatui（utils 只用 crossterm）；5 个 luck_score 测试
- **健壮性 fix 三件套**：watcher 路径/env 缺失降级为 warn + 空字符串 + `release-pause` feature gate（发布版启用 `--features release-pause`）；注释 check_windows_terminal；延迟 spawn hotkey_handler（避免失败路径 runtime drop hang）
- **Step 4 CLI --json 分流 + sink 接线**：umaai 用 lexopt 替 clap 解析 `--json`/`-h`/`--help`；JSON 模式关 ANSI + 启动横幅 eprintln；按模式选 StdoutJsonSink / HumanReadableSink
- **Step 3 DecisionSink 三实现**：sink.rs 新增 trait + EmptySink / HumanReadableSink / StdoutJsonSink；reason::NoopSink 改名 DecisionReasonNoopSink 与 sink::EmptySink 区分
- **Step 2 last_decision override 三 trainer**：DecisionInfo 加 candidate_n（与 scores 同长同截断供 luck 按局数加权）；MctsTrainer / RamenMctsTrainer / RamenHandwrittenTrainer override last_decision()；集成文档 §3.3.1 改"按局数加权"

## 2026-09-06
- **`FlatSearch` 新增可选批量 rollout 后端**：`batch_rollout: Option<Arc<dyn RamenBatchRollout>>` + `with_batch_rollout`；置上后拉面根搜索先让后端一次算完该根全部 `(候选, rollout 种子)`，rollout 闭包退化为查表、缺项报错。候选分配、CRN 种子派生、rank 加权均值排序与终局多维统计仍全部走内核；种子由 `rng.clone()` 预派生、内核随后从原 rng 派生同一个根，RNG 消耗与不接后端时逐位一致
- **`RamenNnTrainer` 决策拆三步**：`prepare_decision`（守门 + 特例 + 编码）/ `infer_features` / `resolve_decision`，`select_action` 由三步组合而成；生产、对拍与跨 rollout 调度器因此共用同一份实现，不会各自漂移
- **`RamenRolloutTrainer` 可选决策录制钩子**：`RolloutInferSink` / `RolloutInferSnapshot` / `with_infer_sink`；未挂钩子时决策路径逐位不变
- **新工具 `ramen_root_bench`**：固定根与整局的实验入口，七种模式（CPU 按候选并行 / CPU 扁平并行 / CPU-GPU 对拍 / GPU 波次驱动 / 侧车复用 / 整局冒烟 / 生产教师两后端一致性与正式整局），改实验参数不触发重新编译
- **跨 rollout 波次驱动**：rollout 改为可暂停状态机，暂停点落在已有阶段边界（各决策阶段均为「先 `list_actions`、状态不变、再 `select_action`」，故进入 `run_stage` 前即可备好答案），配确定性补位与固定物理批尺寸；`game/` 未改动
- **终局轨迹即时释放局面**：已完成轨迹及时释放完整局面，避免局面内存随累计 rollout 数增长；完成记录、任务队列与结果数组仍随任务总数增长
- **侧车就绪握手与 stderr 排空**：推理侧车须在模型加载与预热完成后才写就绪标记，Rust 端阻塞等待；握手后由独立线程持续排空 stderr 并保留最近若干行诊断，避免管道填满导致子进程阻塞
- **实验身份指纹补齐**：`ramen_space_bench` 的运行身份改按源码路径与内容计算，并新增游戏数据指纹；`--resume` 在身份不符时拒绝续跑，日志重复键报错
- **`ramen_root_bench` 搜索配置显式关闭 UCB**：`SearchConfig::default()` 的 `use_ucb` 为真且分组尺寸 256，`n<=256` 时看不出差异、`n=512` 才会改变汇总口径；同时新增「每候选汇总计数等于 n」守门
- **本地实验配置（不进上游）**：release profile 改 `opt-level = 3` / `lto = "fat"` 以缩短整局实验耗时；`game_config.toml` 的 `config_override` 卡组、种马额外属性与 `trainer` 切到当前实验口径

## 2026-09-04
- **吃面 PT 增量延后到 NextTurn**：`ground_ramen_effects` 不再立即 `scenario_pt += pt_gain` / `eat_count += 1`，训练阶段 `calc_ramen_training_effect` 用吃面前 PT 算 `ramen_pt_effect` / `region_bonus` 档位；PT 增量与 eat_count 在 `next()` 的 `NextTurn` 阶段（清空 `current_ramen` 之前）统一处理，RMJ 归档与 `check_rmj` 行为不变
- **三处基线重抓 + 一条新守门**：`bench.rs` BASELINE_SCORE 64336→63870 / BASELINE_FIVE `[3337,2328,2200,1101,829]`→`[3337,2293,2200,1086,829]`、`flat_search.rs` 三阶段根搜索 7 候选 mean 重抓、`ramen_mcts_trainer` 两测试基线同步；新增 `test_eat_ramen_pt_gain_defers_to_next_turn` 钉「吃面 ground 后 scenario_pt 不变 / calc_ramen_training_effect 用吃面前 PT 算增量 / NextTurn 后累加」三条边界
- **吃面决策点埋点**：`RamenState` 新增 5 组逐年纯观测（决策点数 / 可做点数 / 有得做却不吃的点数 / 库存合计 / 型别偏斜合计），`rules.rs` 新增 `recipe_reachable` 与 `record_ramen_select`，三阶段与合并决策两条路径各挂一次；bench CSV 相应扩 15 列。用于把诀窍丢弃拆成「没料」「型别凑不齐」「有得做但不做」三种成因
- **组合身份改为直接量 `combo_key`**：`DeckPlan::combo_key()` 由 (马娘, 卡组) 直接算出，导出器新写 `combo_key.npy`，`data.py` 按组合切分时优先用它、缺失才回落 `index % plan_count`。旧口径绑死在单一采样空间上，换空间后同一 `index` 指向别的组合，新旧数据因此无法合并；新键与空间无关，故不同空间采的目录可以合并训练
- **采集器与导出器接通分布外空间**：`ramen_teacher_collect` / `ramen_export_npy` 新增 `--shape` / `--extra-card`，与 `ramen_space_bench` 同口径；三处共用的空间构造下沉为 `sampler::space_from_cli`，`parse_shape` / `format_shape_name` 一并移入
- **组合键守门**：第一代 525 个组合键互不碰撞；子空间的 240 个组合与 gen1 逐个相同（跨空间稳定）；分布外 190 个组合与 gen1 零重合
- **`SPECIAL_TARGET_LIMIT` 常量化**：隐藏风味单次用量上限 2 由字面量收口为常量
- **埋点守门**：`recipe_reachable` 与 `list_special_targets_for` 在 17150 组上逐位同真假；4200 局 A/B 原有列逐局一致、均分逐位相同、耗时无可测差异

## 2026-09-03
- **EXP-006h 复现合并**：handwritten token 入口接通 bench、闭环 Δ+67 t+4 显著，本地分支留档
- **`bench_compositions` 改加权口径**：满破面板「友情×2 + 干劲×0.5 + 训练」加权和，默认 pool_size=10 / min_panel=80（最新 10 张候选池留 5 张缓冲），池内加权降序、并列按 card_id 倒序取前 3
- **`ramen_space_bench` 固定地区策略为 `all`**：与 `bench_base`、`ramen_teacher_collect` 一致，不再跟随 `game_config.toml`，避免基线静默换分布
- **`ramen_handwritten_choice` 补转发 `select_event_choice`**：避免重放轨迹与采集时不一致
- **若干健壮性与文档修正**：补 `.npy` 长度检查、删 `ChoiceRow` 阶段字段 dead_code、NN 测试在 `saved_models/` 缺模型时跳过、修 `ramen_special_root` 死链等
- **训练评估单源化（B2）消除 policy↔local 双重计算**：`score_train_action` 拆 eval/other、`decide_train` 用 `TrainEvalCache` 同回合同 train 的 calc 链收口为 1 遍，decide_train 整回合 **-14%**、整局 **-10%**，平均分逐位一致
- **训练评估确定性守门**：`test_train_eval_deterministic_and_cached_consistent` 三条——`eval_train` 重复一致 / cached 与 uncached 决策逐项一致 / trait 双路径逐位等价
- **mcts_profiler bin 注册补 `required-features=["profiler"]`**：修复默认 `cargo check` `main not found` 硬错误
- **UCB search_group_size 2048→512**：`default_config.toml` 改值、断言同步

## 2026-09-02
- **`SupportCard::calc_training_effect` 简化签名 + 起点改基础面板**：去 `Result` 包裹与 `is_locked` 短路，起点改 `CardTrainingEffect::from` 保 fresh cumulative，`is_locked` 字段与 `effect = eff.clone()` 回写保留（NN feature 兼容）
- **deyilv 路径去掉 `eff.clone()` + microbench burn-in 清理**：三处 override `let deyilv = eff.deyilv; effect = eff;` move 而非 clone；microbench 删 `burn_in_lock_cards` 与 `CT_MICROBENCH_LOCKED`，数据基本不变
- **`Game::deyilv` trait 简化签名**：`Result<f32>` → `f32`（无 fail 路径），三处 impl override + 5 处 test caller + `distribute_person` 一处内部 caller 全部去 `?`；与 `calc_training_value`/`calc_training_buff` 保留 `Result` 的分工清晰
- **`calc_training_value` 微基准 bin**：新增 `calc_training_value_microbench.rs`（speed build + friendship 全 100 + 4 段：distribute_all / calc_training_buff / calc_training_value / 端到端），pprof 之外的"逐段基线"快速回归工具
- **`calc_training_value` microbench 扩到 7 段 + private 改 pub**：补 `score_train_action`/`decide_train`/`calc_ramen_training_effect` 覆盖 d10872a + perf_profiling pprof Top 20 缺口；`RamenPolicy::score_train_action`/`status_gain` 与 `LocalRamenTrainer::decide_train`/`dynamic_status_adjustment`/`reserve_penalty` 提 pub 供调优工具用
- **`perf_profiling.md` 第 7 章固化按段拆解基线**：新增 §7"按段拆解的最坏路径基线（calc_training_value_microbench）" + 与 pprof Top20 交叉校验 + 新优化优先级（distribute_all 缓存化、status_gain 两条新增）；附录 E 列 `CT_MICROBENCH_RUNS`/`CT_MICROBENCH_WARMUP`
- **`perf_profiling.md` 旧描述清理 / 重写**：§1 改"两类"→"三类"性能工具（加 microbench）；§3.1 改 `d10872a` commit hash + 实测数字；§3.3 旧优先级保留并加"原列表与 §7.6 互为补充"；§5 拆 §5.1 三 bin + §5.2 对比表；§6.4 加 microbench 复现 command
- **`perf_profiling.md` 多轮均值替换旧 pprof 单局**：§3 改"性能分析结论（当前方法论）"（microbench × N std ≤ 2.5%）；§3.2 旧"次要发现"标已弃用；附录 B/C 旧 pprof 单局 / cargo flamegraph 数据整体删除；§7.2/§7.3 7 段 × 3 轮（800.97/260.20/270.39/1398.90/449.82/4175.40/9.63 ns/iter）+ 附录 B 6 函数 × 3 轮；附录 D/E 上移
- **`perf_profiling.md` 重复内容整合**：原 §7 6 子节合并为 3（§7.1/§7.2/§7.3）；原 §7.5 复现命令删并引 §6.4；§7.6 优化优先级并入 §3.3 为 8 条（6 pprof + 2 microbench）
- **`distribute_person` 采样改零分配**：`WeightedIndex::new` 换 `Uniform::new + sample` + 5 元素线性分桶（新 `sample_bucket`），消 person clone 与负权重检查；7 组合 × 5 万次守门 + 全量数值逐位不变，每次采样 -31~-42%（A/D ≈ 53%）

## 2026-09-01
- **预声明 LR 日程**：train.py 新增 `--lr-schedule {plateau,cosine}`（默认 plateau）+ `--lr-warmup-steps`/`--lr-final-factor`，cosine 按 optimizer step 线性 warmup + 余弦退火；解决 ReduceLROnPlateau 按轮计数导致 LR 衰减次数随数据量漂移、对照差异无法归因
- **`--max-steps` 精确截断**：以 batch 边界精确截到指定步数取代"轮数向上取整"，`run_epoch` 增实跑步数、checkpoint 增 `global_step`，保证续跑对齐
- **采样空间指纹**：`SamplingSpace::content_hash()`（顺序敏感）+ `GEN1_SPACE_HASH_V1` 钉死，manifest 增 `sampling_space_hash`，导出器校验空间一致后才导出；防止扩空间后 plan_count 静默改写导致留出集切分整体错位；刻意不并入 `recipe_hash`（同空间新旧数据合并需要）
- **分布外采样空间**：`SamplingSpace::custom` + `--shape`/`--extra-card` 出口（不与 `--shape` 同用则不与教师数据同分布），输出显式标注分布外，用于检验网络未训练卡组流派泛化

## 2026-08-31
- **评估列窗口**：eval.py 的 `evaluate_model` 与命令行新增 `--eval-columns LO HI`，候选价值只用该段 rollout 列重算（data.py 的 NpyShard 相应惰性 mmap `cand_scores`/`cand_valid`，窗口内无有效列的样本整条跳过并计数）。原先只有全列 `cand_mean` 一个口径，而 `best.pt` 正是按它挑的，被结算的列因此参与了模型选择
- **eval.py 沿用 checkpoint 的切分粒度**：独立评估此前恒按默认的 combo 重切，按 sample 训练的模型会被静默换成另一套留出集
- **训练侧诊断设施**：train.py 新增 `--eval-columns`（每轮额外记一份限定列的留出指标到 `evaluation_a`，不参与早停与 LR 调度）、`--checkpoint-steps`（在指定 optimizer step 处存 `step_XXXXXX.pt`）、`--ema-halflife-steps`（按步维护带偏差校正的权重 EMA 并一并存盘）、`--no-early-stop`（训满轮数上限）。metrics 增记 `global_step`，run.json 增记 `diagnostics` 段。四项都不改变梯度，不给新参数时逐位复现旧结果
- **训练随机轴拆分**：train.py 新增 `--split-seed`（样本身份：划分与抽稀）与 `--init-seed`（优化路径：初始化、dropout、minibatch 顺序），未给出时都回落到 `--seed`，旧命令行为逐位不变。此前两者绑在一起，每换一个种子就换掉约 10% 训练样本；实测固定 split 后闭环 sd 从 2521 降到 156
- **早停按 optimizer step 计**：新增 `--patience-steps` 与 `--max-steps`，按每轮步数换算成轮。按轮计数时数据量翻倍会让同样的轮数变成两倍步数，学习曲线各点的训练时长口径不一致
- **闭环 bench 局号偏移**：ramen_space_bench 新增 `--run-offset`。相邻基种子会撞随机世界（`derive_seed` 是 XOR 后 splitmix，`base ^ r == base + r`），此前三个相邻种子的 12600 局实际只有 5248 个唯一世界，标准误被低估约 1.5 倍。改由固定基种子、按局号区间切分
- **选择集 / 验收集分离**：新增 scripts/ramen_nn/compare_bench.py，按世界去重做配对比较，并断言两集零重叠。同一批对局既挑 checkpoint 又报成绩会带 winner's curse
- **因子化吃面输出头**：model.py 新增 `factorized_eat_head`，把 `[1,201)` 联合格拆成「地区 + 用法 + 零初始化交互」。输出布局与 ONNX 算子集不变。三训练种子下无可测效果，默认关闭
- **SpecialSelect 联合决策根还原**：新增 trainer/ramen_special_root——把 `SpecialSelect` 局面写回 `pending_ramen` / `pending_special_targets` / `stage` 还原成它所来自的 `RamenSelect` 联合决策根，供网络在正确状态上读联合格位。教师在 `RamenSelect` 根上搜的是联合动作（地区 × 隐藏风味用法），policy 格 `[1,201)` 也是联合格，而训练集里 `SpecialSelect` 阶段样本数为 0；真实对局却把决策拆成两拍，第二拍的阶段 one-hot 在全部训练样本里恒为 0，网络输出属外推。模块不带 onnx 门控，否则守门测试在默认 feature 下不会运行。测试断言还原后特征逐位相同、且**不还原时必须不同**——后者保证规则层新增字段时测试会红而不是静默失效
- **NN 训练员 SpecialSelect 三档口径**：新增 `SpecialSelectMode::{Raw, Canonical, Handwritten}` 与 `with_special_mode`，默认 `Canonical`；`Raw` 保留作对照，`Handwritten` 把该阶段整个交给手写策略以给可恢复上限定界。`ramen_space_bench` 与 `ramen_advantage_probe` 都加 `--special-mode`。ramen_advantage_probe 的 required-features 补 onnx——无 onnx 时它没有任何可用功能，让 cargo 直接跳过该目标好过编一个只会报错的空壳；ramen_space_bench 仍须在默认 feature 下可构建（handwritten/random 是主用途），故其 special-mode 解析走 onnx 门控
- **拉面 NN 策略训练员**：新增 ramen_nn_trainer——把 ONNX 模型接到 `Trainer<RamenGame>`，编码定长特征后按冻结格位表给当前候选打分 argmax；choice 头未训练，事件选项委托推荐手写策略。仅在 onnx feature 下编译
- **自选比赛硬守门供网络复用**：`RamenPolicy::free_race_gate` 的判定本体抽成自由函数 `free_race_gate_index`，NN 训练员在 Train 阶段先过同一层守门再读网络输出。自选比赛不达标直接判育成失败，是硬性义务而非价值权衡，任何策略都要过；判定语义逐字保持不变，手写策略行为不变
- **采样空间基准接入网络**：ramen_space_bench 新增 `--trainer nn` 与 `--model`，模型在进程启动时加载一次由各局共享而非每局重载；`--no-race-shield` 可关掉守门，仅供研究守门能否移除，不作为验收口径。策略分派由字符串匹配改为预构造枚举，未知策略名在开跑前报错而不是每局重判
- **on-policy 配对 advantage 探针**：新增 ramen_advantage_probe bin——用指定策略跑完整局，在网络与手写选择不同的决策点上做配对 rollout（两动作共享同一张 CRN 种子表，rollout 基策为手写），按性能差分恒等式估计 `J(π) − J(H)`。两策略选同一动作的点贡献恒为 0，直接跳过不搜索；分歧点按蓄水池等概率抽样，单局估计按分歧点总数加权还原。`--rollin` 可把 roll-in 换成手写，用于量测占用分布错配。此前训练侧的 `expected_regret` 算在手写 roll-in 加扰动的分布上，与闭环结果反向，不能用来排序训练方案
- **训练集稳定抽稀**：data.py 新增 `subsample_train_refs`，train.py 新增 `--max-train-samples`——按 `splitmix64(sample_id, seed)` 排序取前 N 条，同一种子下各数据量点严格嵌套且验证集不变，曲线上的差异只来自数据量。checkpoint 的 split 段增记 `full_train_size` 与 `max_train_samples`
- **Train 阶段动作重加权（可选）**：train.py 新增 `--train-action-reweight`，按 policy 软标签主动作施加截断逆平方根样本权重（上限 4，归一到均值 1），只作用于 Train 阶段；权重与计数写入 run.json 与 checkpoint。默认关闭
- **onnx feature 编译修复**：neural_net_evaluator 补 `use rand::Rng`

## 2026-08-30
- **采样地区配额**：采样器新增地区配额与「只捕获指定阶段」开关，按工作项序号确定性分配、走独立随机频道，改配额不影响其余样本的截断回合。此前第 2/3 年的地区选择几乎采不到——它们在回合末，同回合的吃面/训练决策先命中采样白名单，实测 1200 次采样 turn 23 命中 0 条、turn 47 只有 9 条
- **拉面教师样本容器**：新增 training_sample 模块——定长特征 + 元信息 + 变长候选表，每候选按 rollout 序号存定长分数槽位并配有效性位图，失败的 rollout 留空而不是跳过，否则候选之间的 CRN 配对会整体错位；统计量由原始 f64 累加，均值与标准差和 ActionResult 同口径；附 pilot 用的 bincode 批次落盘。PolicySlots 补 serde 派生
- **搜索层保留有序 rollout**：SearchConfig 新增 `record_ordered_rollouts` 开关，默认关闭时不分配缓冲、不改变搜索结果；开启后按 rollout 序号定长记录 score 轴原始分，随 SearchOutput 一并输出根种子。失败的 rollout 留空而不是跳过，UCB 路径同样按序号写入，否则候选之间的 CRN 配对会整体错位
- **拉面版 export_sample**：搜索输出可直接导成教师样本——定长特征 + 元信息 + 按 rollout 序号对齐的候选分，不计算 policy/value 标签（标签是离线可再生的 sidecar）；未开启有序 rollout 记录时直接报错，不退化成用直方图回填
- **教师数据采集驱动**：新增 ramen_teacher_collect bin——采样局面、搜索、导出样本、分片落盘并写 manifest。四条运行时前提（记录有序 rollout / 关闭 UCB / 显式 radical_factor_max / 地区策略 all）由 bin 强制设置，manifest 记的是它们的实际取值，另存游戏数据签名与 git 提交以便复现；支持按 manifest 断点续跑，`--count` 是从 `--start` 起算的累计目标，区间为空时报错并保持 manifest 不变
- **教师数据 NumPy 导出**：新增 ramen_export_npy bin——把多个采集目录的 bincode 分片摊平成一组 .npy 数组供 Python 训练侧读取，候选维用 CSR 偏移表示变长。合并前校验各目录的采集配方哈希与 git 提交一致、样本 id 不重复，维度常数与本次编译不符时报错。只导原始量不导标签，软标签配方与 value 归一化留在训练侧。`--raw` 额外导出每次 rollout 的原始分数与槽位有效性。npy 头部定长占位、收尾回填行数，全程流式不驻留内存
- **采样空间基准**：新增 ramen_space_bench bin——遍历采样空间全部 (马娘, 卡组) 计划各跑若干整局，按构成与马娘分组给出均分、标准差与标准误。此前唯一的基准 bench_base 用的马娘不在采样空间内且无自选比赛要求，测出的手写基线不能当作网络验收门槛；本 bin 与教师数据同分布，两边数字才可比
- **手写策略选择记录**：新增 ramen_handwritten_choice bin——重放教师样本的每个局面、记录手写策略的选择并落到 policy 格位，供训练侧算出同一批局面同一 Q 口径下手写自己的后悔值。训练侧的后悔值是相对搜索教师的，不回答「网络比手写强还是弱」。RamenSelect 按真实对局分两阶段问再合成，因为手写的吃面决策不读万能风味用法，直接喂合并候选表等于让它随机挑
- **导出器记录采样计划数**：manifest 增记 plan_count，训练侧据此按 (马娘, 卡组) 组合切留出集，不再在 Python 侧硬编码组合数
- **Python 训练侧**：新增 scripts/ramen_nn——标签生成、模型、训练、评估、ONNX 导出与 mmap 多目录加载器。policy 标签取配对 Bayesian bootstrap 的最优概率而非温度化 softmax，value 走逐 rollout leave-one-out 的选择—估值以消除选择乐观偏差，地区选择按组合概率边缘化到三格，稀疏阶段用截断逆平方根加权。留出集默认按卡组组合切分而非按样本，避免同一套卡组同时进训练与验证；卡片 token 默认不加槽位 embedding——卡组顺序在游戏里没有含义，而训练数据的槽位与卡片类型完全相关，加了会让模型记顺序而非读属性

## 2026-08-29
- **拉面 NN policy 格位表**：新增 policy_schema 模块，把动作映射到 234 维固定格位并由单一入口分派；吃面按地区 ID 而非槽位编码、吃面与万能风味用法合成联合格、地区选择纳入第一代。**格位表冻结**
- **格位表与规则层的耦合回归**：原有测试只拿本文件常量自洽验证，规则层一改不会变红；补测试用采样器把真实候选（含合并决策形态）与规则层用法表全过一遍格位映射。顺带修合并决策的「不吃面」落格失败——原先只接受空 targets，实际每个吃面决策点都会漏掉该候选
- **NN 管线计划文档同步现状**：Phase 1/2 标完成；教师数据预算改按 search_n=1024 重算；拉面 CRN 机制更正为共享 rule_master（阶段重播种仅 onsen），配套作废一处无效测量；补 Phase 3 开跑前待办
- **MCTS pprof-rs profiler bin 固化**：sim_profiler 模板的 MCTS 版（pprof-rs 用户态采样，输出 .pb 给 go tool pprof / inferno-flamegraph）
- **性能分析指南文档**：新增 perf_profiling.md，记录 cargo flamegraph / pprof-rs 工具选择准则与 MCTS hot path 数据
- **决策理由输出按分排序 + 分差着色**：移除险胜门限触发，每回合都输出决策理由；"中选"改"首选"并固定亮绿色，其余按评分降序编号 `#2` 起，颜色按与**首选**差距分档（`<30` 亮绿 / `<100` 绿 / `<300` 黄 / 其余真彩色灰，与文本内 `±分差` 同源）；`reason_gap_threshold` 字段保留兼容但不再用作触发器；`test_color_thresholds` 在 `--features no-color` 下自动跳过
- **决策理由模块索引入项目文档**：`project_context.md` 新增"输出与决策理由"节，记录 `reason_color` 阈值调整位置（`reason.rs:107`）与 no-color feature 兼容性
- **拉面在线对接计划**：新增 `.trae/documents/ramen_online_integration_plan.md`——文件通道 thisTurn.json + scenarioId 分发、两阶段决策吃面前/吃面后、C# 端先行冻结协议再 Rust 接入
- **用户配置调整**：`game_config.toml` 切马娘 101901（stamina build）+ 卡组/蓝因子/extra_count 微调；`gamedata/default_config.toml` 同思路调 102601 + `ramen_region_strategy` 由 `"fixed"` 改 `"all"`

## 2026-08-28
- **MCTS rollout 诊断日志运行时屏蔽**：diagnostic 加进程级开关（DiagGuard 挂 `search_with_terminal`），`diag!` 双门控 + 8 处 explain 块补 `if enabled()`，rollout 搜索静默、业务日志不受影响，顺带拿回加速收益
- **险胜决策理由输出**：新增 output/reason——险胜回合（门限默认 150）显中选内容 + 未中选 top-N 分差与五维/PT 子项；参数走完整覆盖链，终局差异日志 info 降 debug
- **诊断出口整理**：basic.rs 回合分隔线 println 并入 diag!、地区选择 diag 补"手写逻辑"注记、state.rs 补五维上限初始化契约注释
- **合入 ramen_workbench 主干修改**：squash 单提交；实验脚本 / workflow / 实验采集 bin / 过程文档不合入
- **tests_overview 按 master 口径全量重写**：159→330 个测试逐条一行描述并按模块重组；旧「未来缩减参考」表随 159 口径移除

## 2026-08-27
- **五维属性上限剧本化**：上限基值改为随构造参数传入（`Uma::new` / `BaseGame::new` 新增 `limit_base`），顺序固定为"先写剧本基值、再加继承"，三个剧本各自从自己的 `scenario_*.json` 取值，`constants.json` 同名字段降级为 basic 与缺字段兜底。原先"先写全局值、再由各剧本事后修正"的打补丁式设计全部删除——拉面的整体赋值发生在累加开局继承之后，会把继承增量擦掉；温泉的 `min(2800)` 是速度基值 2600 时代的防御值，基值提高后变成硬截断，且在继承事件后还会再截一次。补丁写法本身就是这两个缺陷的来源，新剧本照抄必然复现。温泉基值补入 `scenario_onsen.json`（此前无该字段，一直吃全局值再被截断）。**改变拉面与温泉模拟数值，基线作废**
- **终局评分查表口径统一**：新增 `GameConstants::status_final_score`，越界一律饱和到表末。此前三处消费点行为各异——裸下标越界 panic、`unwrap_or(0)` 越界静默返回 0。后者最坏：属性增益按查表差分计算，返回 0 会让该维收益变成巨大负值，手写策略永久回避该维且不报错。评分表长度有限而上限＝剧本基值＋继承三次，蓝因子拉满即可越界。顺带修 `status_gain` 中负增量 `as usize` 回绕溢出（当前取值恒正打不到）
- **上限相关守门与契约测试**：新增跨三剧本的开局上限守门测试（期望值从各剧本 JSON 推导，故改代码会红、改数据不误报）、剧本基值字面量契约测试（守数据漂移，并锁两剧本基值必须不同——拉面与全局常量当前数值相同，误接全局的回归只有它能抓）、查表越界饱和测试。`expected_score_parts` 保持不调用生产查表函数，维持独立对照。修正 `eat_covered_train_gate_blocks_mismatched_ramen` 夹具写死旧上限当"满"的问题，改为从实际上限取值；三处硬守门快照基线随上限变化重抓
- **MCTS rollout 与 fallback 切到正式推荐策略**：原用机制残缺的策略核心评估局面；门控全关时逐位等价，rollout 档关掉观测开销
- **搜索掉分归因**：缺省 `radical_factor_max=50` 使有效样本量恒 3.9%，选择偏差压过搜索收益；rf=0 后方向反转，缺省值不动
- **硬守门快照重抓与收紧**：4 处基线随 trainer 切换重抓；合并搜索重搜断言改回逐位快照（先前放宽到搜 28/29 次也绿）；`for_rollout` 补决策等价守门
- **rollout 加速 −29% CPU**：编译期消掉 rollout 路径的屏幕输出，分数逐位一致；**仅关 diag 时生效**，umasim 自己的 bin 需显式关
- **perf 诊断工具与 Windows 可构建性**：新增 `sim_profiler`；pprof 编不过 Windows，收进可选 `profiler` feature；`microbench_top_fns` 改进程级 CWD，加 `#[ignore]`

## 2026-08-26
- **吃面后必训练 at_trains 覆盖位（C 方案）**：新增 `LocalRamenConfig.eat_requires_covered_train`（推荐 preset 开启）——`decide_ramen` 对每个吃面候选预演"落地后最优训练位"，不在该面 `at_trains` 内则否决，实现"吃面后必训练覆盖位、不训练就不吃面"。吃面训练覆盖实测 80%→99%，总分与技能点双升。**改变拉面模拟数值，基线作废**
- **弱位 boost 补"未满"条件**：`ramen_weak_train_boost` 与 `ramen_window_alignment` 的弱位放大仅在 `five_status < limit` 时生效——已满位只剩 PT 收益，放大只会虚高训练分。**改变拉面模拟数值**
- **地区选择弱位覆盖参数 + 配置覆盖修复**：`score_region` 新增 `region_weak_cover_weight`（默认 0.0，实验入口）；game_config.toml 顶层 `ramen_region_strategy/fixed` 覆盖修复（字段须写在所有 `[...]` 段之前，原注释位置被 `[mcts]` 段吸收导致不生效）

## 2026-08-26
- **搜索终局多维记录（P2）**：rollout 返回值扩为 `RolloutOutcome<T>`，新增 `search_with_terminal` 与 `MomentResult` 按候选累加终局观测量；`CandidateAccum` 收拢三条统计使其只在成功分支推进；UCB 失败计数统一末尾告警。**纯观测出口，模拟数值逐位不变**
- **拉面终局 25 维与诊断出口**：在 rollout 内部归约阈值类维度（PT 达成率等），避免均值丢信息；RMJ 直接读规则层；维度键名与顺序冻结（FROZEN_DIM_KEYS + 守门测试），合作伙伴用于手写策略前后对比
- **超级拉面纳入搜索**：补 `SuperRamenSelect` 阶段分支，新增 `Operation::SuperRamenSelect`；手写与 Local 同步补分支避免默认分支静默换选项。**门控默认关闭**
- **第 1 年地区纳入搜索**：拆出 `BeginAfterRegionSelect` 阶段边界，回合 2 走 `Begin → RegionSelect → BeginAfterRegionSelect → Distribute`；修 `encode_regions` 未选出时被编三份「地区 0」。**门控默认关闭；`all()` 语义变真，历史基线作废**
- **超级拉面搜索平局回退**：`deck_can_split == false` 时改为仅在确实平局时向选项二回退，判定跟随 `selection`
- **地区候选生成抽为纯函数**：`region_select_combos` 显式传参，守门测试直接调它，避免 `test_year1_2_always_all_regardless_of_strategy` 空转仍绿
- **补回 `test_combined_gate_off_full_game` 的 `#[test]`**：上次提交插入观察壳占用属性行导致该测试静默不运行，加静态扫描核对
- **拉面 MCTS 诊断出口接线**：主二进制单局开启 verbose，补 `#[ignore]` 整局观察壳；观察壳须自行设日志 info 级

## 2026-08-25
- **自由比赛收益真实衡量**：`race_grade_weight`（等级×常数）退役，改走训练同管线折算（真实收益 + 赛程压力叠加）；折扣经实测削弱至 0.3。**改变拉面模拟数值，基线作废**
- **bench handwritten 档切到正式推荐策略**：自动局表现失真，改为 `RecommendedRamenTrainer`；核心保留作 rollout 组件对照
- **方案 E 确认 PT 不打折**：残余折扣只作用于副属性，PT 独立计分；单点启发式无法观测的跨回合项留给 MCTS
- **拉面五维上限硬截断移除**：speed 恢复 3100，玩家高分档不再受 2800 截断拖累；bench 强制地区策略 All 不受手动模式影响
- **弱位训练偏好 + 按 build 自适应查表**：双层级（吃面前 / 吃面后）放大 at_trains 卡少位 raw；按智卡数查表（推荐 preset 默认启用），build 异质性极强
- **体力门限上调（30→40）**：300 局配对总加权 +397（7/7 build 正），失败率 1.5%→0.3%；y3 门禁改为每年评估，仅第三年吃面放掉硬门限
- **支援卡连续事件增强（用户手动）**：8001/8002 事件数值上调（体力 5→10、五维/PT/hint 增强）
- **地区权重重新评估**：当前策略下 300 局配对，`region_youqing_weight` 1.0→1.5（speed Y3 +387）
- **友人词条加成 + 主动使用**：词条 bonus（体力×1.6 / 属性×1.3），不溢出时主动用友人；失败率 2.4%→1.6%，友人 4.9/5
- **残余收益折扣（方案 E）**：主属性快满时副属性打折（PT 保留），300 局 +84
- **手写策略四项提分机制**：吃面联动 / 必成价值 / 友人饥饿 300 / 动态属性平衡，100 局 +749
- **地区选择修正公式 + 验证**：`bias×youqing - waste×10`；全 101 种验证：真实 build +99.9 / 残缺 -7.3
- **region_matrix 诊断工具 + test_region_selection_per_build**：按 build 打印三年选区 + 占比；7 build × 3 年人工审查
- **LocalRamenTrainer 补齐第 1 年地区选择打分**：不再恒选候选 0；基线作废
- **拉面动作空间不变量 + 终局分分解（MCTS P0 安全网）**
- **搜索层拉面合并动作落地（P1.1+P1.2）**：一次搜完 ramen×targets；拉面基线作废
- **拉面搜索阶段缺省补 `ramen`**：42 局配对 +2306
- **测试有效性审查修补**：缺省守门测试、结构恒等式、删无效测量壳
- **不在判定与得意率解耦**：distribute_person 两步算法，缺席名单入 RamenState
- **地区拉面分身缺席优先**：缺席卡优先补分身位；拉面基线作废

## 2026-08-24
- **训练人数加成按人头类型计数**：`1 + 0.05 × 人数` 乘区改按 `PersonType` 判定（替代硬编码下标），抽出 `count_training_persons`，负数与越界下标一并不计。**改变拉面模拟数值，基线作废；温泉与 base 逐位不变**
- **超级拉面分身补上友人卡**：候选收集改全扫全体人头（不再写死卡组下标范围），同时加「每训练一个友人」约束。**改变拉面模拟数值**
- **RecommendedTrainer 改进方案文档**：新增 `workbench_improve_1.md`，规划地区打分三指标、第三年体力门禁回合差异化、`matrix_variant` DSL 重构三件事。**文档规划，未实施代码**
- **配置层三处接线修复**：`[mcts]` 改全 Option + `deny_unknown_fields`；主二进制 onsen 改调既有 `SearchConfig::new_game_config`；`expected_search_stdev` 补注为 UCB 缩放标尺非实测统计量
- **搜索层 CRN 与 UCB 三处修正**：CRN 对照轴改按「候选间是否共享 `rule_master`」分臂（双种子 rollout 入口拆开决策流与规则主种子）；失败样本改按原始序号交集配对；UCB 首组步长收进 `search_n`。**生产语义与分数逐位不变**
- **拉面规则层四处数值修复**：分身分配改合法集直选（消除概率重试假失败）+ 按回合派生局部流使策略流消耗归零；训练人数加成改按人头类型计数；超级拉面分身补上友人卡与「每训练一个友人」约束。**改变拉面模拟数值，基线作废**
- **拉面杯逐年观测出口**：`scenario_pt` / `eat_count` / 地区选择改归零前按年归档，CSV 换逐年三列。**纯观测出口，模拟数值逐位不变**
- **第三方库引用规范化（续）**：bench 模块中 anyhow 宏的全名引用改 use 导入

## 2026-08-23
- **拉面杯 MCTS 训练员**：按阶段门控的搜索训练员，命中的决策点走扁平搜索、其余转发手写策略，门控全关时与纯手写逐位一致
- **拉面局面特征编码器**：新增 features 模块，把局面编码为定长向量（global / cards / persons 三段），较温泉版补齐成长率与属性上限并开启人头分支
- **人头下标与卡组槽位解耦**：拉面下人头顺序与卡组顺序不一致，原先按 person_index 直接当卡组下标的调用点全部改为按 card_id 反查。**改变拉面模拟数值，基线与落盘教师数据作废**
- **手写策略地区打分覆盖第 1 年 + build 自适应**：新增有效阶段判定使回合开始阶段内联触发的第 1 年地区选择也进入打分；`score_region` 纳入 youqing 项并按卡组 bias 统一缩放。**改变手写策略基线数值**
- **测试观测收集器**：新增 `utils::Checks`，测试全程 println 记 OK/NG、末尾汇总有失败才报错；既有裸断言与重复本地实现一并归拢

## 2026-08-22
- **基准新增自选比赛达标维度**：新增任意时点重比各区间完成场数的判定（原判定只在区间结束回合的下一回合执行，且不达标即终止育成），bench 结果与 CSV 加达标率并在每局 / 分组 / 总览打印；配套补两个守门测试（不改策略逻辑），逐回合扫描触发点以免随常量表调整失效
- **搜索层可复现 + 真 CRN + 泛型化（NN 管线 Phase 1，已完成）**：rollout 种子改为按序号确定性派生（候选索引不参与，否则协方差归零），移除全部随机播种，失败由静默丢弃改为计数告警；新增按阶段边界重播种的真 CRN（默认开启，可从 toml 关），实测朴素共享起始种子几乎无收益、按阶段重播种才显著；搜索结构泛型化并保留默认类型参数使活跃入口零改动，采用「公共内核 + rollout 闭包」规避泛型方法解析导致温泉特判静默失效；顺带修 NN leaf 微批路径漏重播种、UCB 终止判据用成功数会死循环两处缺陷，并把 rollout 基策的调试缓存改 Mutex 以满足跨线程共享
- **局面采样器（NN 管线 Phase 2 上半）**：为教师数据制造根局面——分层的采样空间、按工作项序号确定性导出采样任务（分片 / 续跑 / 改并行度均不变）、轨迹随机扰动、走真实决策路径截断捕获；根局面限定在阶段入口，回合开始阶段内联执行的决策点会破坏搜索的阶段推进契约
- **第三方库引用规范化**：搜索层与采样器中 anyhow 宏的全名引用改为 use 导入后直接调用
- **支援卡类型注释订正**：card_type 原注释与卡片数据实测相反（5 是友人、6 是团队）

- **RNG 受控重构（v3 三流，已实施）**：新增顶层 `rng.rs`（splitmix64 唯一实现 / 加法派生无状态流 SplitmixRng / 类型隔离三流 TurnFixedRng+EventRng+StrategyRng）；规则层随机改从 self 流取（run_distribute 独占局面流=角标/人头分布/hint 触发位，回合开始事件链走事件流，训练/分身/比赛走策略流），Trainer 决策流保持 StdRng；bench 局号进种子 `seeded_rngs(base,idx)→(StdRng,rule_master)`；拉面 CRN 由规则层接管（fork_for_rollout 注入 rule_master，simulate_common 退役阶段重播种），onsen 保留外挂 CRN；未注入 rule_master 时回退旧行为。验收：层 2/3 集成测试 `rng_consistency.rs`——跨策略 20 回合角标/分布/固定流消费量逐位一致（0 不一致），事件增量逐位一致；方案文档 `rng_refactor_plan.md` 更新为 v2/v3 并归档 v1，`rng_reply.md`（上游 CRN 评审意见）归档
- **umasim 主二进制接入拉面杯剧本**：main.rs 此前仅支持 onsen/basic（`scenario="ramen"` 时实际落 basic），新增 `run_ramen_once` 与 ramen 分发分支（random/handwritten/mcts 回退/默认 manual 均支持），handwritten 分支使用 RamenHandwrittenTrainer；`GameConfig::scenario` 注释补 ramen。实测主二进制跑通 77 回合拉面杯（UB2 49442 / PT 7941）
- **issues 更新**：第三年地区选择无 build 自适应（score_region 对第三年地区无区分度，实测各 build 同选一组合；方案已定待实施，含临时验证测试）
- **ramen_manual 屏幕输出整理（Agent 对话文本流风格）**：新增 turn_flow 渲染层与固定种子基线测试；候选内联预览（训练数值 / 吃面完整效果 / 诀窍配方）并分层着色；事件三段式、回合状态去重；ramen_manual 接入实时候选栏与选择确认；训练诊断输出暂屏蔽
- **第3年地区选择修复**：ramen_region 配置字段落错 TOML 段导致预设失效（恒枚举 120 组合），移回顶层后 fixed 预设生效
- **comfy-table custom_styling**：修复彩色表格 ANSI 宽度错乱
- **自选比赛守门 + 决策日志 breakdown**：等级过滤 / 摆烂判定 / 达标后停止，候选评分分解入决策日志
- **诀窍槽 NPC 按实际人数计算**、game_config.toml 加载修复、cargo-husky 撤销与 fmt 手动化、bench 玩家 build 外置与分组跑批
- **显示微调（用户）**：比赛加成信息亮品红；清理未使用 import
- **文档归档**：config_refactor_plan / log_refactor_plan 移入 archive

## 2026-08-21

- **bench 设施与全卡型基准**：新增 `umasim::bench` 公共设施（双 RNG 分裂 / 单局运行 / 统计 / CSV / 代表性选卡）+ `bench_compositions`（101 种卡组构成跑批），bench_base / bench_compositions 复用瘦身
- **手写策略规划文档**：新增 handwritten_policy 目录：定位（MCTS rollout 基策）、策略形态（参数化利于调参）、输出分层（决策日志 / DecisionInfo / GameView）、玩家经验标签
- **手写策略三步交付**：① 地基：bench_base + 决策日志 + 规则层可复现性修复（Random 基线 mean=30432）② 核心：RamenPolicy 各阶段打分 + RamenHandwrittenTrainer（较 Random +39%）③ 自选比赛守门 + 打分自洽性修正（实测 +18.5%）
- **rustfmt 规则固化 + AGENTS.md 微调（用户）**：明确 Nightly 格式、stable 禁跑 cargo fmt；需求澄清与安全注意事项表述精简

## 2026-08-20

- **注释精简**：umasim/Cargo.toml 注释 38→14 行；Rust 长注释压缩 6 处（文件头、重复的 1121 维清单去重），保留 13 处高价值文档（公式 / 索引映射 / 机制契约）
- **colored 无条件加载**：colored 从 cli feature 移出改为无条件依赖（非 Windows 纯 std 实现，Android / 嵌入式交叉编译无风险），消除 9 个文件约 20 处彩色双版本 cfg gate 重复代码；no-color 编译期无色语义不变
- **Phase 4 步骤1：依赖边界整理 + feature 拆分**：删除 analyzer crate；umasim feature 三层设计（default = cli + diag，新增 no-color / onnx）；15+ 文件 cfg gate 治理；nn 模块整体 cfg gate 到 onnx；umaai 依赖瘦身（去掉 tract-onnx）；四种编译组合通过；暂不抽 umasim-core
- **日志模块重构（Phase 3）**：新增 output 模块（diag! 宏 / GameView）；142 处规则层日志迁至 diag!；GameView 扩至 8 字段并删除 disable_log / enable_log；LOGGER 锁合并为 OnceLock，release 编译零 warning
- **测试日志简化**：新增 init_test_logger（只输出 stderr 不写文件），100+ 处测试迁移
- **友人事件词条生效修复**：apply_event 应用"事件效果提高 / 恢复量提高"词条，三剧本统一生效
- **排名数据补全**：rank_scores / rank_names 补齐至 LS24，速度档位上调
- **第3年地区选择默认 Fixed**：走固定组合 [[11,14,15]]，跳过 120 组合枚举
- **拉面杯回合规则收紧**：回合 0-12 无自选比赛；回合 0-1 与超级拉面回合跳过吃面阶段
- **其他**：友人高羁绊概率 0.3→0.25；ramen_manual 改密码学随机种子；新增 tests_overview.md

## 2026-08-19

- **吃面效果立即落地**：选完面与隐藏诀窍用法后立即消耗诀窍、效果生效并生成分身，玩家选训练前可见完整 buff
- **hint_special 全员触发**：第三年吃面且支援卡种类达标时，相关训练位置全部支援卡强制出 Hint
- **ManualTrainer 玩家测试**：支持真实终端交互与 mock 两种模式；新增完整 77 回合与 hint_special 路径的端到端测试
- **修复并发测试日志初始化竞争**
- **配置系统 Phase 2**：用户可调项迁至 default_config.toml（步骤1）；GameConfig 五子配置分组（步骤2+3）；配置加载集中化 + 统一校验（步骤4）；拉面杯第3年地区选择策略接入 PolicyConfig + TOML 精简（步骤5）；文档收尾（步骤7）
- **文档整理**：project_context 按实况更新，旧 issues 归档

## 2026-08-18

- **剧本 PT 每年归零**：RMJ 结算后归零重新累计，URA 阶段不再累计
- **RMJ 事件时机修正**：结算当回合立即触发；超级拉面基础效果 URA 回合自动生效（赛后加成仅首次）
- **事件补全**：RMJ 结算成功 / 失败事件 + 固定触发事件（登场 / 新年 / 抽签 / 结局），修复比赛回合事件漏触发
- **训练分布剧本得意率加成修复**（含 RMJ 效果）
- **夏合宿规则实现**：诀窍槽全 MAX、禁用普通 / 友人外出与治病、休息自动清除不良状态
- **决策重构**：新增"选面 + 吃法"一次性合并决策接口；动作阶段扩展为"选面 → 选诀窍用法 → 训练"三阶段

## 2026-08-17

- **umaai 跨平台构建支持**：可在 Ubuntu / Linux 下编译运行（Windows 专用依赖按平台限定）
- **拉面杯模块机制修正、显示改进与架构重构**：友人事件 / 分身系统 / 地区选择 / RMJ 结算 / 超级拉面 / 诀窍角标等
- **训练数值端到端观测测试**：固定回合打印吃面 / 不吃面场景的训练分布与数值

## 2026-08-16

- **拉面杯模块 1d 最小闭环**：回合 0-77 完整阶段流转、组合动作生成、事件处理、动态人头管理、回合边界处理
- **1b 核心游戏机制 + 1c 动作预览和手写策略**：诀窍 / 做面吃面 / RMJ 结算 / 地区选择 / 分身 / 隐藏风味 / 友人事件；"吃面选择 × 基础操作"分离决策模型
- **1a 核心类型定义 + 1b-1 诀窍系统**：拉面杯模块结构与核心类型；诀窍槽基础值分配、库存溢出、训练 / 友情加成
- **拉面重构计划调整**：Phase 合并为 1a-1d，归档旧规划文档、统一领域术语（食材→诀窍等）

## 2026-08-15

### 拉面剧本机制完善

- 补充友人解锁机制、诀窍槽算法、分身规则等核心机制文档
- 补充剧本机制初始化规则（第2回合开始时）
- 补充夏合宿规则（训练等级、事件触发）
- 补充超级拉面期间限制（不可吃其他面）
- 更新gamedata数据：调整事件概率、添加地域名称、完善超级拉面效果
- 更新AGENTS.md项目规则：完善提交规范和工作流程
- 添加ramen_story_flow.md拉面剧本流程文档
- 更新术语表：添加诀窍槽、友人解锁、复合宿等新术语
- 整理文档目录：将规划类文档移至opt子目录

## 2026-08-14

### 拉面剧本事件数据补充

- 在scenario_ramen.json中添加scenario_events和friend_events数据
- 更新RamenScenarioData结构体，添加对应的事件字段
- 添加单元测试验证事件数据加载

### EventData触发类型重构

- 新增TriggerType枚举：Random/Code/Fixed三种触发类型
- 移除EventData中的start_turn/end_turn/max_trigger_time字段
- 更新JSON数据文件和触发逻辑代码

## 2026-08-13

### 文档整理

- 创建了AGENTS.md项目规则总结文档
- 在.trae/documents/目录下整理相关文档

### 测试规范完善

- 在umasim::utils中新增get_workspace_root()函数，用于获取workspace根目录
- 修改了多个测试文件，在测试中使用get_workspace_root()切换到workspace根目录

### 拉面剧本数据完善

- 更新ramen_basic_effect：添加jiban/status_limit/hint_special字段，填充3年效果数据
- 添加finals_effect：定义超级拉面(含RMJ成功)的基础/额外/单独效果
- 添加ramen_region_effect：记录20条地域拉面效果数据
- 更新Rust结构体：添加RamenBasicEffect结构体
- 更新ramen_memo_cn.md文档：补充效果说明和字段定义
