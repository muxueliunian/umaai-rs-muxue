# 整理SendGameStatusPlugin -> umaai protocol.rs 中易混淆的地方

## feeling_guage 误拼

后续需要在上游插件 (SendGameStatusPlugin) 中修复. 目前按现状 feeling_guage 处理.

## last_ramen 需要额外判断

当前实现中，只在吃面回合更新 last_ramen. 这会导致在非吃面回合，last_ramen 保持为上次吃面的值，需要追加检查 active_effect_array 是否有内容。仅当有内容时表示当前状态当回合已经吃了面了。

## persons
只记录支援卡对应的人物，且顺序严格按照cardId对应卡的顺序，所以不需要记录charaId
注意这里剧本友人可能出现在任何一个位置，所以需要根据实际顺序决定friend_index等相关字段

### personType
0代表未加载（例如前两个回合的npc），1代表友人（R或SSR都行），2代表普通支援卡，3代表npc人头，4理事长，5记者，6不带卡的佐岳（拉面剧本没有）。暂不支持其他友人

## personDistribution 适配

C#端约定，-1为空，数字 0-5 支援卡，6理事长，7记者，8NPC（不区分NPC名字）。
但是，umaai-rs项目中，NPC必须为不同的charaId，否则无法正确均匀分配
在拉面杯中，会出现至多5个NPC，因此在读取时，要把personDistribution中出现的 8 数值，依次改为 8,9,10,11,12.

例如 [ 1, 8, 8, 8, -1 ] 可以转换为 [ 1, 8, 9, 10, -1 ]

## playing_state

1为正常训练，44,45,46为拉面剧本特殊状态

## source
影响游戏状态恢复到当前回合的哪个阶段
现版本应简单处理：
- command: 当回合已分配完人物，进行训练前。此时需要检查是否已有拉面Buff，也就是 active_effect_array 是否有内容。
如果没有，则决策状态为“吃面前”，对应 RamenStage::RamenSelect state；
如果有，则状态为“吃面后，训练前”，对应 Train state，需要根据当前的last_ramen 构造一个决策中间状态 RamenAction::ramen_select(Some(last_ramen))，让umaai决策接下来的训练。在模拟时要注意，当回合内拉面效果已经生效，分身位置已经在json里给出，需要计算已经生效的拉面效果数值，但不能重复调用分身分配方法。
- special: 进行下一次地区选择前. 目前这一状态判断有问题，还需要根据playing_state判断（参考下面样例），对应RegionSelect state
- event: 暂时忽略

## UmaAI需要在一些情况下给出复合决策
- 如果选择吃面，则需要给出combined_select，一次承载 ramen + targets 两个决策
- 目前暂时无法捕捉回合2开始时的第一次地区选择，所以要在回合1的训练决策给出的同时，继续给出回合2的地区选择决策。

## 理事长、记者、NPC生成
json中只有6张卡的person信息，into_game恢复局面时，要补上缺少的NPC：
- 理事长：固定为数组下标6，person_index = 6，从第0回合开始就有
- 记者：固定为数组下标7，person_index = 7，回合>12时才有（不包含12）
- NPC：固定为5个，分别对应数组下标8到12，person_index = 8~12，依次递增

## RamenGame反向转为RamenStatus输出
这个功能只用于玩家按下 ctrl+s 保存局面供检查时，不会反向输回给C#端，不用处理兼容性问题。

## 数据获取不完全的判断与处理
- 2 <= 回合数 <= 71 时，如果 selected_regions 全0，说明数据获取不全，需要给出警告信息，不进行搜索或计算
- 返回的json 格式要修改，需要包含成功或失败状态，失败时返回错误信息

## 超级拉面回合处理
- 回合数 >= 72 时，需要判断 active_effect_array 是否有内容(长度>0)。如果没有，则直接丢弃，等待下一条数据；下一条数据会有 active_effect_array，是超级拉面激活后的效果，此时需要直接给出训练决策，而且在模拟时不可重算超级拉面效果

## 解析例子

- game6204_turn3.json
active_effect_array 为空，所以当前没有吃面。
personDistribution 的一部分包含：       [1, 8, 8, 8, -1], [8, 8, -1, -1, -1]
根据NPC生成和person distribution适配规则，需要把出现的8依次改为8,9,10,11,12（不重复）.因此，修改后的值为：
[1, 8, 9, 10, -1], [11, 12, -1, -1, -1]
一回合内至多有5个NPC，且person_index都不同
决策状态为“吃面前”
如果决策为吃面，要给出combined_select，一次承载 ramen + targets 两个决策

- game6204_turn3_2.json
active_effect_array 有值，说明玩家在本回合内已经选择了吃面
此时诀窍槽的剩余数据已经在json里记录，所以不用管当前面已经消耗了哪些材料
last_ramen = 1，需要在Game中还原决策中间状态 RamenAction::ramen_select(Some(1))，让umaai决策接下来的训练选择
scenario_pt = 300 是【吃面后】的剧本pt，根据规则，计算拉面效果时，要按【吃面前】的剧本pt计算剧本加成。changelog里有相关订正记录。

- game6204_turn14_2.json
friend_stage = 2 说明友人已经解锁出行功能。当前版本C#端无法获取友人首次点击状态，在后面的版本会修复

- game6204_turn23_2.json
playing_state = 46，表示RMJ结算状态，不处理

- game6204_turn23_3.json
playing_state = 45，表示地区选择状态，需要把Game恢复到第二年（turn23）地区选择前，让umaai出地区选择决策

- game6204_turn71_4.json
playing_state = 48. 表示RMJ最终结算，不处理