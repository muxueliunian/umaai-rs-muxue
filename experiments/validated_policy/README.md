# 拉面杯可选策略与配对验证

主属性接近上限时，原手写策略在“未满/已满”之间突然切换技能 PT 价格；超级拉面默认固定选第二项。本改动增加连续 PT 定价和按终盘属性缺口选择超级拉面范围的可选模式，游戏规则、正式 preset 的字段语义与 game_config 均保持不变；默认 preset 的数值在第十二轮验收通过后更新为新组合（见下文「第十二轮」一节），改动前的默认值下文称 base。

## 使用

已有 bench_base --tokens 和 RecommendedRamenTrainer::with_tokens 可启用：

- ptblend200-capd0-rgn1：固定卡组优先参考此组合。
- ptblend200-capd0-rgn1-supermode3：随机卡组可选；不保证每副卡组提升。
- ptblendN：连续窗口为 N/100 次当前训练的主属性增量；0 关闭，合法范围0～1000。窗口外、主属性已满、主属性增量为零均保持原行为。
- supermode0：默认固定选项二；1/2：固定一/三的实验对照；3：根据当前属性缺口及卡型数估值，平局优先原选项二。

capd0/rgn1 是已有开关，本 PR 没有修改它们的公式。supermode3 以每个训练位6×(50+30×卡数)估算增量，按现有真实属性上限和终局评分表计算价值，再乘(1+卡数)后按候选范围累加；它是启发式，不是六回合真实预测。

这些选项仅接入已有 Rust 构造/bench 入口，未增加线上配置直连字段，也未更改 MCTS rollout 默认策略。

## 独立验收结果

精确对照基线：04c739ca450804d834b71ffa1d011517a74c1b8a。以下取同一批第五轮独立验收，指标为完整77回合模拟的最终 calc_score，而非中间策略估值或 score_pt。不能把此前上游 GA 收益计入本次提升。

| 可选组合 | 固定卡组300局 | 7副预设2100局 | 160副随机卡组×300局 |
|---|---:|---:|---:|
| ptblend200-capd0-rgn1 | +263.64 | +185.96 | +85.92 |
| 再加supermode3 | +274.26 | +238.91 | +204.43 |

supermode3 相对前一组合的增量及近似95%区间：固定+10.62 [-34.78, +56.03]；预设+52.95 [+32.22, +73.69]；随机+118.51 [+84.47, +152.54]。随机160副中40副相对前一组合退步，最差约-267分。

冻结后跨4位马娘复核：随机64副、共38,400配对局，supermode3额外+151.84 [+94.65, +209.02]；固定600局为-21.73 [-55.76, +12.30]。固定组未证实额外收益，模拟结果不能直接当作线上实战或 MCTS 决策收益。

## 追加候选：逐卡 Hint 精确估值（hintlv600）

训练成功且该训练位有带 Hint 人头时，模拟器必然推送 1 个 Hint 事件：25% 走属性事件，其余给 min(5, 1 + 卡面 hintLevel) 级 Hint（再受每卡上限截断），每级在终局折算 13 分（hint_pt_rate × pt_score_rate = 6.5 × 2.0）。原策略用固定 hint_bonus=8 覆盖整项，读不到卡面等级，1～6 级 Hint 一律同价。`hintlvW` 用逐人头精确期望替换该项，卡面 Hint 已满时只保留属性分支，默认关闭。

另一批全新卡组（160 副随机卡组与本文筛选、验收池零重叠，rule 种子 918700000 起）× 300 局同种子配对：

| 组合 | 固定卡组 | 7副预设 | 160副随机卡组×300局 |
|---|---:|---:|---:|
| ptblend200-capd0-rgn1-supermode3 | +269.3 | +300.2 | +232.0 [+190.5, +273.5] |
| 再加 hintlv600 | +148.3 [-45.4, +342.1] | +378.0 [+307.1, +448.8] | +406.0 [+357.4, +454.6] |

hintlv600 相对 supermode3 的增量：固定 −121.0 [-296.0, +54.0]（区间含零）、预设 +77.7 [+12.4, +143.0]、随机 +174.0 [+153.0, +195.0]（random_diverse +284.7、random_narrow +146.3）。160 副随机卡组中 8 副相对 base 退步，最差 −130.8 分，随机组技能 PT 平均 −46.3。固定卡组落后，是本项定位为“随机卡组可选”的原因，限制与 supermode3 相同。

复现（冻结清单 `hintlv.json`，base / supermode3 / hintlv600 三臂）：

~~~powershell
./target/release/policy_pair_bench.exe experiments/validated_policy/hintlv.json target/hintlv.csv
python experiments/validated_policy/analyze.py analyze experiments/validated_policy/hintlv.json target/hintlv.csv
~~~

同批的另一项候选（按该位历史点击频率估计未来点击次数的等级前瞻）在筛选阶段不显著，未纳入本 PR。


## 第十二轮：preset 采纳（本 PR 的新默认值）

本 PR 把第十二轮在两个独立池上通过验收的组合设为 `RecommendedRamenTrainer::new()` 的默认值，等价 token 串：

    ptblend800-capd0-hintlv600-trd4425-trds3450-trdsh2500-g1498-g2498-g3498-o1304-o2304-o3304-reserve157-rgn1-supermode3

相对已发布臂只动四处：已满位 PT 定价 trd/trds/trdsh、近上限连续定价窗口 ptblend、状态 gap/overflow/reserve、外出基准分 out；Hint 估值一处没碰。

两个互不重叠的独立池（各 160 副全新随机卡组 × 300 局，与筛选池及历史清单的卡组、种子零重叠）同种子配对，相对已发布臂（ptblend200-capd0-rgn1-supermode3-hintlv600）：

| 池 | 随机卡组（48000 对） | 预设组（2100 对） | 固定卡组（300 对） |
| --- | ---: | ---: | ---: |
| r10_holdout | +130.4 [+93.6, +167.2] | +92.7 [+33.8, +151.6] | +18.7 [-118.7, +156.1] |
| r11_holdout | +147.6 [+108.2, +187.0] | +57.2 [-1.2, +115.6] | -20.1 [-176.4, +136.2] |

同池的已发布臂相对 base 为 +386.7 [+343.4, +430.1] 与 +403.9 [+358.9, +448.9]，因此新默认值相对 base 合计约 +517 / +552。

消融（把每个旋钮组退回已发布值，两个池）：状态 gap/overflow/reserve 约 -97 / -99，PT 定价组 -24 / -35，ptblend -15 / -31，out -15 / -12。把 hintlv 从筛选期的 675 退回 600 是 +0.5 [-5.3, +6.3] 与 +4.0 [-1.2, +9.1]，区间含零，故默认值取 600。

局限与不适用范围：以上均为模拟器内、随机卡组 cohort 的结论；固定卡组只有 1 副、两池符号不一致，不作为结论；这批数据也说明本次提升不是靠加 Hint 权重换来的。**base 无法再用 token 复现**：所有 token 串都从 `new()` 出发，`new()` 改成新组合后「旧 base + token」的求值会随之前移，要复现上面的历史数字需在本文档对应的提交上重放。`final-check.expected.csv` 已按新默认值重新抓取，仅用于「同机同种子可复现」的校验，不再代表历史清单。


## 配对与抽样

每个候选与对照使用相同卡组、马娘、继承和 rule_master；异常保留、不补采。筛选与独立验收的普通卡身份分池，候选冻结后不再依据验收调参。

随机卡组从五种普通卡每类0～3张、合计5张的101种构成均匀采样；各类最新30张SSR按cardId排序奇偶分池、突破0～4均匀，固定满破友人303054，排除重复角色及育成角色。不代表全部玩家分布，不含SR/R及其他友人。random_diverse表示普通卡种类≥4，不等同规则中的deck_can_split。

随机置信区间按每卡组的配对均值聚类，跨马同卡组仍为一个聚类；固定和预设按配对局差分，均为正态近似95%区间。完整配置见冻结 JSON，重现时不要换用 bench_config 中可能不同的马娘/继承。

未纳入的实验也需说明：友人提前外出退步，弱位面板修正没有可靠收益，库存权重调整未通过验收，事件溢出修正没有改变实测选择，第六轮按后续隐藏用量统一成本使随机组退步。这些实现未加入本 PR。

## 复现

Rust 1.98.1、Release、锁定依赖；Python 3.11+，仅标准库。从仓库根执行：

~~~powershell
cargo build --release --locked -p umasim --no-default-features --bin policy_pair_bench
cargo test --release --locked -p umasim --no-default-features --lib pt_cap_blend
cargo test --release --locked -p umasim --no-default-features --lib test_round2
cargo test --release --locked -p umasim --no-default-features --lib super_choice
cargo test --release --locked -p umasim --no-default-features --lib test_yearly_observability
cargo test --release --locked -p umasim --no-default-features --bin policy_pair_bench
./target/release/policy_pair_bench.exe experiments/validated_policy/final-check.json target/policy-replay.csv
python experiments/validated_policy/analyze.py verify experiments/validated_policy/final-check.expected.csv target/policy-replay.csv
./target/release/policy_pair_bench.exe experiments/validated_policy/holdout.json target/policy-holdout.csv
python experiments/validated_policy/analyze.py analyze experiments/validated_policy/holdout.json target/policy-holdout.csv
./target/release/policy_pair_bench.exe experiments/validated_policy/cross.json target/policy-cross.csv
python experiments/validated_policy/analyze.py analyze experiments/validated_policy/cross.json target/policy-cross.csv
~~~

Linux/macOS去掉.exe。输出使用独立路径，以免覆盖冻结资料。默认配置含target-cpu=native，换机器需重新编译。

final-check.expected.csv保留8副×10种子×5臂共400条历史结果（仅去除耗时）；冻结JSON仅从历史清单去掉不再提供的evreal实验臂，卡组、种子和其他臂未改动。完整验收252,000条，跨马156,000条；分析器检查配对完整性、重复、异常及清单身份，并输出汇总JSON。原始大CSV、工具链和编译产物不纳入源码仓库，可由上述命令重新生成。
