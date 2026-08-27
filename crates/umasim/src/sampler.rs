//! 局面采样器 —— 为 NN 教师数据制造「根局面」
//!
//! # 这一层解决什么问题
//!
//! 教师数据是 `(局面, 搜索标签)` 配对。搜索标签由 [`crate::search`] 产出，
//! 本模块负责另一半：**局面从哪来**。
//!
//! 直觉做法是「跑一整局，78 个回合每个决策点收一条」。它有两个致命缺陷：
//!
//! 1. **样本高度相关**。相邻回合的局面差别只有体力、羁绊与几点面板，
//!    对网络而言近似同一个样本重复了 78 次，有效独立样本数远低于名义条数。
//! 2. **分布偏移**。数据集的边界就是手写策略走过的轨迹。手写策略从不做的事，
//!    网络永远见不到；一旦实战中遇到，输出等同瞎猜。这是自我迭代管线最典型的翻车点。
//!
//! 采样器的做法是：随机抽 `(马娘, 卡组, 截断回合 t)`，用手写策略跑到 `t`，
//! 途中以 ε 概率乱走，**只取最后那一帧**，前面全部丢弃。
//!
//! 看似浪费，但成本极不对称——跑一局到中途约 1 毫秒，而对一帧跑一次搜索需要
//! 十几到几十秒。采样成本相对搜索可忽略，因此值得不惜代价换取样本的独立性与覆盖度。
//!
//! # 可复现契约
//!
//! 一切从**工作项序号**导出，不从线程或迭代顺序导出——与 [`crate::search::seeds`]
//! 同一套纪律。因此分片、断点续跑、改并行度都不会改变任何一条样本。
//!
//! - 卡组：`index % 组合数`（分层，保证跑满整轮时每种 `(马娘, 卡组)` 出现次数完全相等）；
//! - 截断回合与局内种子：由 `index` 经 SplitMix64 分频道派生，与卡组不相关。
//!
//! 卡组周期 525 与回合数 78 的最大公约数是 3。若截断回合也写成 `index % 78`，
//! 每个卡组只能打到 26 个剩余类；走哈希派生正是为了拆开这两件事。
//!
//! # 两条使用约定
//!
//! **必须按 index 区间分片**，不能按「每个工作单元采满 N 条成功样本」分片——
//! `Exhausted` 的位置一变，成功样本集合就和分片方式绑死了。
//!
//! **复现基座包含 `gamedata` 与 `GameConfig`**，不只是 `SampleSpec`。
//! `run_region_select` 会读 `GAMECONFIG.ramen_region_strategy`（测试里 `init_global()`
//! 兜底为 `All`，而 `gamedata/default_config.toml` 是 `fixed`），`GAMECONSTANTS` 的
//! `race_grades` / `pt_favor_rate` 同理。同一条 spec 在两套配置下会走出不同轨迹。
//! Phase 3 落盘时必须把这两者的签名一并写进 manifest。

use std::cell::RefCell;

use anyhow::{Result, bail};
use rand::{Rng, rngs::StdRng};

use crate::{
    bench::seeded_rngs,
    game::{
        Game,
        InheritInfo,
        Trainer,
        ramen::{RamenAction, RamenGame, RamenStage}
    },
    gamedata::{EventChoice, GAMEDATA},
    global,
    rng::splitmix64,
    trainer::RamenHandwrittenTrainer
};

// ============================================================================
// 种子派生
// ============================================================================

/// SplitMix64 的 gamma 增量常数（黄金比例）
///
/// 与 [`crate::bench::seeded_rngs`]、[`crate::search::seeds`] 用的是同一个标准常数。
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// 采样器频道标签：与 rollout 种子流分开，避免两者在同一 index 上意外同步
const SAMPLER_STREAM_TAG: u64 = 0x5341_4D50_4C45_5230;

/// 截断回合频道标签
const TURN_STREAM_TAG: u64 = 0x5455_524E_5F44_5257;

/// 按 `(基底, 序号, 频道)` 派生一个独立种子
///
/// 终混合直接复用 [`crate::rng::splitmix64`] 的那一份（全仓库唯一权威实现），
/// 避免各模块各写一个同名但行为不同的函数。
fn derive_seed(base: u64, index: u64, tag: u64) -> u64 {
    splitmix64(base ^ tag ^ index.wrapping_add(1).wrapping_mul(GOLDEN_GAMMA))
}

// ============================================================================
// 采样空间：马娘 / 支援卡池 / 卡组构成
// ============================================================================

/// 采样空间中的一个马娘
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UmaEntry {
    /// 马娘 gameId
    pub game_id: u32,
    /// 通称（日志与测试可读性用，不参与任何计算）
    pub alias: &'static str
}

impl UmaEntry {
    /// 角色 ID
    ///
    /// 马娘 gameId 的高 4 位即角色 ID，低 2 位是同角色不同卡面的序号
    /// （例：`112901` → 角色 `1129`）。角色 ID 用于排除
    /// 「马娘与同角色支援卡不可共存」的非法卡组。
    pub fn chara_id(&self) -> u32 {
        self.game_id / 100
    }
}

/// 第一代采样空间的 7 个马娘
///
/// 覆盖成长率分布：速主（东海帝王 20）/ 耐主（摩耶 15）/ 根主（放声 20、小栗 14）/
/// 均衡（杏目 10-10-10）。自选比赛要求 4 无 3 有，其中小栗帽两段区间、
/// 第二段限 G1，是守门逻辑最硬的测试用例。
pub const GEN1_UMAS: [UmaEntry; 7] = [
    UmaEntry {
        game_id: 100603,
        alias: "小栗帽[芦毛灰姑娘]"
    },
    UmaEntry {
        game_id: 102403,
        alias: "摩耶重炮[Rock in MewMeow]"
    },
    UmaEntry {
        game_id: 112901,
        alias: "杏目[The Changer]"
    },
    UmaEntry {
        game_id: 110602,
        alias: "菱钻奇宝[快乐小音符]"
    },
    UmaEntry {
        game_id: 113101,
        alias: "放声欢呼"
    },
    UmaEntry {
        game_id: 108702,
        alias: "真弓快车[不融化的糖果]"
    },
    UmaEntry {
        game_id: 100301,
        alias: "东海帝王[无上喜悦]"
    }
];

/// 采样空间中的一张支援卡（满破）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardEntry {
    /// 6 位 idrank = 5 位卡 ID + 1 位突破等级（第一代锁定满破 ◆4）
    pub idrank: u32,
    /// 卡名（日志可读性用）
    pub alias: &'static str
}

impl CardEntry {
    /// 5 位卡 ID
    pub fn card_id(&self) -> u32 {
        self.idrank / 10
    }
}

/// 第一代支援卡池（出场率 Top 的 11 张 SSR，全部满破）
///
/// 类型分布 6 速 / 2 耐 / 1 力 / 1 智 / 1 友——**类型不在此硬编码**，
/// 一律由 `cardDB.json` 读出，见 [`SamplingSpace::gen1`]。
pub const GEN1_CARD_POOL: [CardEntry; 11] = [
    CardEntry {
        idrank: 302754,
        alias: "[天才的乌托邦]东海帝王"
    },
    CardEntry {
        idrank: 302984,
        alias: "[刀光迸发Clash！]跳舞城"
    },
    CardEntry {
        idrank: 302424,
        alias: "[改变世界的目光]杏目"
    },
    CardEntry {
        idrank: 302824,
        alias: "[铭记于心，京之华]气槽"
    },
    CardEntry {
        idrank: 303024,
        alias: "[永恒的誓言，永恒的光辉]里见光钻"
    },
    CardEntry {
        idrank: 302924,
        alias: "[响彻吧，两人的凯歌]洛林军歌"
    },
    CardEntry {
        idrank: 303044,
        alias: "[其执念如怒涛般汹涌]名将怒涛"
    },
    CardEntry {
        idrank: 303004,
        alias: "[载着热闹的未来奔驰吧！]樱花千代王"
    },
    CardEntry {
        idrank: 302834,
        alias: "[优雅，闪耀的旅途]美妙姿势"
    },
    CardEntry {
        idrank: 302894,
        alias: "[Innovator]青春永驻"
    },
    CardEntry {
        idrank: 303054,
        alias: "[一杯怀旧之味]骏川手纲"
    }
];

/// 卡组构成：5 张普通卡的类型分布，友人卡固定 1 张
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeckShape {
    /// 各类型张数 `[速, 耐, 力, 根, 智]`，合计恒为 5
    pub counts: [usize; 5],
    /// 构成名（进 manifest 与日志）
    pub name: &'static str
}

/// 第一代的 3 种卡组构成
pub const GEN1_SHAPES: [DeckShape; 3] = [
    DeckShape {
        counts: [3, 1, 0, 0, 1],
        name: "3速1耐1智1友"
    },
    DeckShape {
        counts: [2, 2, 0, 0, 1],
        name: "2速2耐1智1友"
    },
    DeckShape {
        counts: [2, 1, 1, 0, 1],
        name: "2速1耐1力1智1友"
    }
];

/// 该阶段的决策点能否作为搜索根局面
///
/// 白名单而非黑名单，因为「合法阶段」是可枚举的、而「嵌套决策」不是。
///
/// 第 1 年地区选择已是 turn 2 的 [`RamenStage::RegionSelect`] 阶段边界，
/// 本白名单会自动捕获它。`Begin` / `BeginAfterRegionSelect` 不是决策点，不收录。
fn is_capturable_stage(stage: &RamenStage) -> bool {
    matches!(
        stage,
        RamenStage::RamenSelect
            | RamenStage::SpecialSelect
            | RamenStage::Train
            | RamenStage::RegionSelect
            | RamenStage::SuperRamenSelect
    )
}

/// 友人卡在 `cardDB.json` 中的类型编号
///
/// 实际映射为 `0 速 / 1 耐 / 2 力 / 3 根 / 4 智 / 5 友人 / 6 团队`。
/// `SupportCardData::card_type` 原本的文档注释把 5 与 6 写反了，
/// 已一并订正——本模块最初照抄那句注释，被 `gen1()` 的数据校验当场抓出。
const CARD_TYPE_FRIEND: i32 = 5;

/// 一个具体的 `(马娘, 卡组)` 组合
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeckPlan {
    /// 马娘 gameId
    pub uma: u32,
    /// 卡组（6 位 idrank，前 5 张普通卡按类型序、末位友人卡）
    pub deck: [u32; 6],
    /// 所属构成名
    pub shape: &'static str
}

/// 第一代继承因子（沿用 `bench_config.toml` 现值，第一代固定不随机）
pub fn gen1_inherit() -> InheritInfo {
    InheritInfo {
        blue_count: [15, 0, 0, 0, 3],
        extra_count: [10, 10, 20, 20, 20, 40]
    }
}

/// 枚举好的采样空间
#[derive(Debug, Clone)]
pub struct SamplingSpace {
    /// 全部合法 `(马娘, 卡组)` 组合
    plans: Vec<DeckPlan>
}

impl SamplingSpace {
    /// 构造第一代采样空间
    ///
    /// 卡片类型一律从 `cardDB.json` 读取而非硬编码——数据更新导致类型变动时
    /// 应当报错，而不是静默产出错误卡组。同理，角色冲突由 `chara_id` 实际比对得出，
    /// 不写死「东海帝王撞 30275、杏目撞 30242」这两条已知结论。
    pub fn gen1() -> Result<Self> {
        let data = global!(GAMEDATA);

        // 按类型分桶；友人卡单独拎出
        let mut by_type: [Vec<u32>; 5] = Default::default();
        let mut friend: Option<u32> = None;
        for entry in GEN1_CARD_POOL.iter() {
            let card = data.get_card(entry.card_id())?;
            match card.card_type {
                CARD_TYPE_FRIEND => {
                    if friend.replace(entry.idrank).is_some() {
                        bail!("卡池含多张友人卡，第一代构成假定恰好 1 张");
                    }
                }
                t if (0..5).contains(&t) => by_type[t as usize].push(entry.idrank),
                other => bail!(
                    "卡池中 {} 的类型 {other} 不受支持（第一代只接受普通卡与友人卡）",
                    entry.alias
                )
            }
        }
        let Some(friend) = friend else {
            bail!("卡池未包含友人卡，拉面杯必须携带新友人卡");
        };

        let mut plans = Vec::new();
        for uma in GEN1_UMAS.iter() {
            // 马娘与同角色支援卡不可共存
            let mut usable: [Vec<u32>; 5] = Default::default();
            for (t, bucket) in by_type.iter().enumerate() {
                for &idrank in bucket {
                    if data.get_card(idrank / 10)?.chara_id != uma.chara_id() {
                        usable[t].push(idrank);
                    }
                }
            }
            for shape in GEN1_SHAPES.iter() {
                for normals in enumerate_decks(&usable, &shape.counts) {
                    let mut deck = [0u32; 6];
                    deck[..5].copy_from_slice(&normals);
                    deck[5] = friend;
                    plans.push(DeckPlan {
                        uma: uma.game_id,
                        deck,
                        shape: shape.name
                    });
                }
            }
        }
        if plans.is_empty() {
            bail!("采样空间为空：卡池与构成无法组出任何合法卡组");
        }
        Ok(Self { plans })
    }

    /// 组合总数
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// 是否为空（恒为 false，`gen1` 已拒绝空空间；仅为满足调用方习惯）
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// 全部组合
    pub fn plans(&self) -> &[DeckPlan] {
        &self.plans
    }

    /// 由工作项序号确定性导出一次采样任务
    ///
    /// `plans` 非空是本类型的不变量（唯一构造入口 [`Self::gen1`] 已拒绝空空间，
    /// 且字段私有），故此处的取模不会除零。新增构造入口时必须维持该不变量。
    ///
    /// 卡组用 `index % len` 分层而非哈希抽取：分层保证任意连续 index 区间内
    /// 各组合出现次数最多相差 1，随机抽取则服从泊松分布、小分片里会明显不均。
    /// 分片续跑正需要前者。截断回合与局内种子仍走哈希派生，与卡组不相关。
    pub fn spec_at(&self, config: &SamplerConfig, index: u64) -> SampleSpec {
        let plan = &self.plans[(index % self.plans.len() as u64) as usize];
        let turn_draw = derive_seed(config.seed_base, index, TURN_STREAM_TAG);
        let truncate_turn = (turn_draw % (config.max_turn.max(0) as u64 + 1)) as i32;
        SampleSpec {
            index,
            uma: plan.uma,
            deck: plan.deck,
            shape: plan.shape,
            inherit: config.inherit.clone(),
            truncate_turn,
            seed: derive_seed(config.seed_base, index, SAMPLER_STREAM_TAG),
            epsilon: config.epsilon,
            min_actions: config.min_actions
        }
    }
}

/// 从各类型可用卡中按张数要求枚举全部普通卡组合（结果长度恒为 5）
fn enumerate_decks(usable: &[Vec<u32>; 5], counts: &[usize; 5]) -> Vec<[u32; 5]> {
    // 逐类型做组合，再跨类型笛卡尔积
    let mut acc: Vec<Vec<u32>> = vec![Vec::new()];
    for (t, &need) in counts.iter().enumerate() {
        if need == 0 {
            continue;
        }
        let picks = combinations(&usable[t], need);
        if picks.is_empty() {
            return Vec::new();
        }
        let mut next = Vec::with_capacity(acc.len() * picks.len());
        for base in &acc {
            for pick in &picks {
                let mut merged = base.clone();
                merged.extend_from_slice(pick);
                next.push(merged);
            }
        }
        acc = next;
    }
    acc.into_iter()
        .filter_map(|cards| <[u32; 5]>::try_from(cards.as_slice()).ok())
        .collect()
}

/// 从 `pool` 中取 `k` 张的全部组合（保持输入顺序，按下标字典序输出）
fn combinations(pool: &[u32], k: usize) -> Vec<Vec<u32>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if pool.len() < k {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| pool[i]).collect::<Vec<_>>());
        // 找最右一个还能右移的下标位；都到顶则枚举结束
        let Some(i) = (0..k).rev().find(|&i| idx[i] != i + pool.len() - k) else {
            return out;
        };
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

// ============================================================================
// 采样配置与任务
// ============================================================================

/// 采样器配置
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 轨迹扰动概率：截断前每个多候选决策点以此概率改走随机动作
    ///
    /// 唯一目的是让轨迹偏离手写策略的舒适区，覆盖它平时不会踩的局面。
    /// 置 0 则数据集边界完全等于手写策略的轨迹，网络学不到策略外的任何东西。
    pub epsilon: f64,
    /// 根局面至少要有的候选动作数（少于此数的决策点没有搜索价值）
    pub min_actions: usize,
    /// 继承因子（第一代固定）
    pub inherit: InheritInfo,
    /// 截断回合的上界（含）
    pub max_turn: i32,
    /// 种子基底：换基底即得到一批全新但同样可复现的数据
    pub seed_base: u64
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.15,
            min_actions: 2,
            inherit: gen1_inherit(),
            max_turn: 77,
            seed_base: 0x5041_5254_5F31
        }
    }
}

/// 一次采样任务的完整输入
///
/// 全部字段由 `(SamplerConfig, index)` 确定性导出，故可分片、可续跑、可复现。
///
/// **自包含**：连 `epsilon` 与 `min_actions` 都固化进来，而不是执行时再从配置读。
/// 否则把任务写进 manifest 跨机器回放时，只要对端配置默认值不同，同一条任务
/// 就会复现出不同局面——而且不会有任何报错。
#[derive(Debug, Clone, PartialEq)]
pub struct SampleSpec {
    /// 工作项序号
    pub index: u64,
    /// 马娘 gameId
    pub uma: u32,
    /// 卡组（6 位 idrank）
    pub deck: [u32; 6],
    /// 卡组构成名
    pub shape: &'static str,
    /// 继承因子
    pub inherit: InheritInfo,
    /// 截断回合：跑到该回合及之后的首个合格决策点即停
    pub truncate_turn: i32,
    /// 本局主种子（决策流与规则流由它分裂而来）
    pub seed: u64,
    /// 轨迹扰动概率（随任务固化，不在执行时从配置读）
    pub epsilon: f64,
    /// 合格决策点的最小候选数（同上）
    pub min_actions: usize
}

/// 采样产出的根局面
///
/// `game` 是**执行该决策之前**的状态克隆，可直接作为搜索根节点。
///
/// `PartialEq` 覆盖 [`RamenGame`] 的全部字段（含 `internal_rng` 状态），
/// 因此比较两次采样结果即为逐位一致性检查，不会漏掉深层状态分叉。
#[derive(Debug, Clone, PartialEq)]
pub struct SampledPosition {
    /// 产生它的任务
    pub spec: SampleSpec,
    /// 根局面
    pub game: RamenGame,
    /// 该决策点的候选动作
    pub actions: Vec<RamenAction>,
    /// 捕获瞬间的决策 RNG 快照——后续搜索续用它即可保持整条链路可复现
    pub decision_rng: StdRng,
    /// 实际停下的阶段
    pub stage: RamenStage,
    /// 实际停下的回合（可能大于 `truncate_turn`：该回合起首个合格决策点才停）
    pub turn: i32
}

/// 一次采样的结果
#[derive(Debug, Clone, PartialEq)]
pub enum SampleOutcome {
    /// 成功捕获根局面
    ///
    /// `Box` 是因为 [`RamenGame`] 体积远大于另一变体，裸放会让整个枚举随之膨胀。
    Captured(Box<SampledPosition>),
    /// 直到育成结束都没遇到合格决策点
    ///
    /// 不当作错误，两种成因都合法：
    /// - 截断回合抽在末尾附近（URA 决赛前后已无可选动作）；
    /// - ε 扰动导致育成提前失败（乱走漏掉必须完成的比赛），育成流程直接终止。
    ///
    /// 两者靠 `final_turn` 区分：远小于 `truncate_turn` 即属后者。
    /// 调用方自行决定跳过还是换 index 重抽。
    Exhausted {
        /// 当时要求的截断回合
        truncate_turn: i32,
        /// 育成实际停在的回合
        final_turn: i32
    }
}

impl SampleOutcome {
    /// 是否成功捕获
    pub fn is_captured(&self) -> bool {
        matches!(self, Self::Captured(_))
    }

    /// 借出捕获到的局面
    pub fn captured(&self) -> Option<&SampledPosition> {
        match self {
            Self::Captured(pos) => Some(pos),
            Self::Exhausted { .. } => None
        }
    }

    /// 取走捕获到的局面
    pub fn into_captured(self) -> Option<SampledPosition> {
        match self {
            Self::Captured(pos) => Some(*pos),
            Self::Exhausted { .. } => None
        }
    }
}

// ============================================================================
// 采样执行
// ============================================================================

/// 捕获到的根局面（训练员内部暂存用）
struct CapturedRoot {
    /// 决策前的状态克隆
    game: RamenGame,
    /// 候选动作
    actions: Vec<RamenAction>,
    /// 决策 RNG 快照
    rng: StdRng,
    /// 阶段
    stage: RamenStage,
    /// 回合
    turn: i32
}

/// 采样用训练员：截断前做 ε 扰动，到达截断回合后捕获首个合格决策点
///
/// 只在单个工作项内部使用、不跨线程共享，故用 `RefCell` 而非 `Mutex`
/// （与搜索层的 rollout 决策器不同，那里必须 `Sync`）。
struct SamplingTrainer {
    /// 基策
    inner: RamenHandwrittenTrainer,
    /// 扰动概率
    epsilon: f64,
    /// 合格决策点的最小候选数
    min_actions: usize,
    /// 截断回合
    truncate_turn: i32,
    /// 捕获结果
    captured: RefCell<Option<CapturedRoot>>
}

impl SamplingTrainer {
    /// 是否已经捕获
    fn done(&self) -> bool {
        self.captured.borrow().is_some()
    }
}

impl Trainer<RamenGame> for SamplingTrainer {
    fn select_action(&self, game: &RamenGame, actions: &[RamenAction], rng: &mut StdRng) -> Result<usize> {
        if !self.done()
            && game.turn() >= self.truncate_turn
            && actions.len() >= self.min_actions
            && is_capturable_stage(&game.stage)
        {
            *self.captured.borrow_mut() = Some(CapturedRoot {
                game: game.clone(),
                actions: actions.to_vec(),
                rng: rng.clone(),
                stage: game.stage.clone(),
                turn: game.turn()
            });
            // 返回值不影响结果：外层在本 stage 结束后立即停止推进
            return Ok(0);
        }
        // ε 扰动：只在多候选点生效，单候选点扰动没有意义
        if self.epsilon > 0.0 && actions.len() > 1 && rng.random_bool(self.epsilon) {
            return Ok(rng.random_range(0..actions.len()));
        }
        self.inner.select_action(game, actions, rng)
    }

    fn select_choice(&self, game: &RamenGame, choices: &[Vec<EventChoice>], rng: &mut StdRng) -> Result<usize> {
        // 事件选项不做扰动：扰动动作是为了让局面**位置**分散，
        // 而乱选事件只会让面板分布偏离真实水平，把根局面推到实战中不会出现的区域。
        self.inner.select_choice(game, choices, rng)
    }
}

/// 执行一次采样
///
/// 必须走真实的 `run_stage → select_action` 路径取根局面。
/// 从外部空推进 `next()` 会跳过阶段初始化，得到的是**不合法状态**
/// （典型报错 `invalid bathing state`）——搜索层 Phase 1 已经踩过这个坑。
pub fn sample_position(space: &SamplingSpace, config: &SamplerConfig, index: u64) -> Result<SampleOutcome> {
    sample_from_spec(space.spec_at(config, index))
}

/// 按给定任务执行采样
///
/// 不再需要 `SamplerConfig`：任务本身已自包含，这样从 manifest 回放的任务
/// 不可能受执行端配置影响。
pub fn sample_from_spec(spec: SampleSpec) -> Result<SampleOutcome> {
    // `rand::Rng::random_bool` 对区间外的概率直接 panic，而这里是生产路径
    if !(0.0..=1.0).contains(&spec.epsilon) {
        bail!("epsilon 必须落在 [0, 1]，实际为 {}", spec.epsilon);
    }

    // 双流分裂沿用 bench 的既有契约：决策流与规则层内部流互不相关
    let (mut decision_rng, rule_master) = seeded_rngs(spec.seed, 0);
    let mut game = RamenGame::newgame(spec.uma, &spec.deck, spec.inherit.clone())?;
    game.set_rule_master(rule_master);

    let trainer = SamplingTrainer {
        inner: RamenHandwrittenTrainer::new(),
        epsilon: spec.epsilon,
        min_actions: spec.min_actions,
        truncate_turn: spec.truncate_turn,
        captured: RefCell::new(None)
    };

    game.run_stage(&trainer, &mut decision_rng)?;
    while !trainer.done() && game.next() {
        game.run_stage(&trainer, &mut decision_rng)?;
    }

    let Some(root) = trainer.captured.into_inner() else {
        return Ok(SampleOutcome::Exhausted {
            truncate_turn: spec.truncate_turn,
            final_turn: game.turn()
        });
    };
    Ok(SampleOutcome::Captured(Box::new(SampledPosition {
        spec,
        game: root.game,
        actions: root.actions,
        decision_rng: root.rng,
        stage: root.stage,
        turn: root.turn
    })))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use anyhow::{Result, anyhow, bail};

    use super::*;
    use crate::{
        gamedata::init_global,
        search::{FlatSearch, SearchConfig},
        utils::{get_workspace_root, init_test_logger}
    };

    /// 测试统一前置：切到 workspace 根并加载 gamedata
    fn setup() -> Result<()> {
        let workspace_root = get_workspace_root()?;
        std::env::set_current_dir(workspace_root)?;
        let _ = init_test_logger("error");
        let _ = init_global();
        Ok(())
    }

    /// 根局面的一行摘要（仅供打印，断言一律用 `SampledPosition` 的全量相等）
    ///
    /// 早先这里是个只比 7 个浅层字段的手写指纹，理由写的是「`RamenGame` 未实现
    /// `PartialEq`」——**该前提是错的**，`RamenGame` 与 `BaseGame` 都 derive 了
    /// `PartialEq`。浅层指纹会漏掉人头分布、羁绊、事件池、`internal_rng` 状态等
    /// 深层分叉，导致「测试通过但缺陷仍在」。现已改为直接比整个结构。
    fn summary(pos: &SampledPosition) -> String {
        format!(
            "回合 {} 阶段 {:?} 候选 {} 面板 {:?} 体力 {}",
            pos.turn,
            pos.stage,
            pos.actions.len(),
            pos.game.uma.five_status,
            pos.game.uma.vital
        )
    }

    /// 采一个局面，Exhausted 视为测试失败（调用方明确期望捕获成功）
    fn must_capture(space: &SamplingSpace, config: &SamplerConfig, index: u64) -> Result<SampledPosition> {
        match sample_position(space, config, index)? {
            SampleOutcome::Captured(pos) => Ok(*pos),
            SampleOutcome::Exhausted { truncate_turn, final_turn } => {
                bail!("index={index} 截断回合 {truncate_turn} 未能捕获决策点（育成停在回合 {final_turn}）")
            }
        }
    }

    /// 取前 `n` 个能成功捕获的 index
    ///
    /// 不能写死 index：ε 扰动可能让某局育成提前失败，
    /// 或截断回合恰好落在 URA 决赛之后，两种情况都合法地返回 `Exhausted`。
    fn first_captured(space: &SamplingSpace, config: &SamplerConfig, n: usize) -> Result<Vec<SampledPosition>> {
        let mut out = Vec::new();
        // 加上界：若捕获条件写错导致全部 Exhausted，无界循环会挂死测试而非报错
        for index in 0..(n as u64 * 20 + 100) {
            if out.len() >= n {
                return Ok(out);
            }
            if let SampleOutcome::Captured(pos) = sample_position(space, config, index)? {
                out.push(*pos);
            }
        }
        bail!("扫完上界仍未凑够 {n} 个可捕获局面，捕获条件可能过严")
    }

    // ========== 采样空间 ==========

    /// 采样空间规模与构成分布符合计划
    ///
    /// 期望 525 = 5 个无冲突马娘 × 85 + 2 个有冲突马娘 × 50。
    /// 无冲突：C(6,3)×C(2,1)=40 + C(6,2)×C(2,2)=15 + C(6,2)×C(2,1)=30 = 85；
    /// 有冲突（速池 6→5）：C(5,3)×2=20 + C(5,2)×1=10 + C(5,2)×2=20 = 50。
    #[test]
    fn test_gen1_space_size() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;

        let mut per_uma: HashMap<u32, usize> = HashMap::new();
        for plan in space.plans() {
            *per_uma.entry(plan.uma).or_default() += 1;
        }
        for uma in GEN1_UMAS.iter() {
            println!("{} ({}) -> {} 套卡组", uma.alias, uma.game_id, per_uma[&uma.game_id]);
        }
        println!("合计 {} 套 (马娘, 卡组)", space.len());

        assert_eq!(per_uma[&112901], 50, "杏目与 30242 同角色，速池应缩到 5");
        assert_eq!(per_uma[&100301], 50, "东海帝王与 30275 同角色，速池应缩到 5");
        for uma in GEN1_UMAS.iter().filter(|u| u.game_id != 112901 && u.game_id != 100301) {
            assert_eq!(per_uma[&uma.game_id], 85, "{} 无角色冲突，应为 85 套", uma.alias);
        }
        assert_eq!(space.len(), 5 * 85 + 2 * 50);
        Ok(())
    }

    /// 角色冲突卡确实不出现在对应马娘的任何卡组里
    ///
    /// 上一条只验证数量，本条直接查内容——数量对但排错卡的情况数量测试抓不到。
    #[test]
    fn test_gen1_space_excludes_chara_conflict() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        for plan in space.plans() {
            if plan.uma == 112901 {
                assert!(!plan.deck.contains(&302424), "杏目卡组混入了同角色卡 302424");
            }
            if plan.uma == 100301 {
                assert!(!plan.deck.contains(&302754), "东海帝王卡组混入了同角色卡 302754");
            }
        }
        Ok(())
    }

    /// 每套卡组都合法：6 张互不相同、恰好 1 张友人卡、类型分布匹配所属构成
    #[test]
    fn test_gen1_decks_wellformed() -> Result<()> {
        setup()?;
        let data = global!(GAMEDATA);
        let space = SamplingSpace::gen1()?;
        for plan in space.plans() {
            let unique: BTreeSet<u32> = plan.deck.iter().copied().collect();
            assert_eq!(unique.len(), 6, "卡组存在重复卡: {:?}", plan.deck);

            let mut counts = [0usize; 5];
            let mut friends = 0;
            for &idrank in plan.deck.iter() {
                let t = data.get_card(idrank / 10)?.card_type;
                if t == CARD_TYPE_FRIEND {
                    friends += 1;
                } else {
                    counts[t as usize] += 1;
                }
            }
            assert_eq!(friends, 1, "拉面杯必须恰好 1 张友人卡: {:?}", plan.deck);
            let shape = GEN1_SHAPES
                .iter()
                .find(|s| s.name == plan.shape)
                .ok_or_else(|| anyhow!("未知构成 {}", plan.shape))?;
            assert_eq!(counts, shape.counts, "卡组类型分布与所属构成不符: {:?}", plan.deck);
        }
        Ok(())
    }

    /// 组合枚举的边界行为
    ///
    /// `combinations` 是手写的字典序枚举，`gen1()` 的 525 只验证了「合起来对」，
    /// 抓不到单点边界错误，故单独覆盖。
    #[test]
    fn test_combinations_boundaries() {
        let pool = [10u32, 20, 30, 40];

        // k = 0：空集是唯一组合（C(n,0) = 1）
        assert_eq!(combinations(&pool, 0), vec![Vec::<u32>::new()]);
        // k > len：无解
        assert!(combinations(&pool, 5).is_empty());
        // k == len：只有全集
        assert_eq!(combinations(&pool, 4), vec![vec![10, 20, 30, 40]]);
        // k = 1：每个元素各成一组，保持输入顺序
        assert_eq!(combinations(&pool, 1), vec![vec![10], vec![20], vec![30], vec![40]]);
        // k = 2：C(4,2) = 6，按下标字典序
        assert_eq!(combinations(&pool, 2), vec![
            vec![10, 20],
            vec![10, 30],
            vec![10, 40],
            vec![20, 30],
            vec![20, 40],
            vec![30, 40]
        ]);
        // 空池：k = 0 有解，k > 0 无解
        let empty: [u32; 0] = [];
        assert_eq!(combinations(&empty, 0), vec![Vec::<u32>::new()]);
        assert!(combinations(&empty, 1).is_empty());
    }

    // ========== 任务导出 ==========

    /// 同一 index 必得同一任务；相差一整轮回到同一卡组但种子必须变化
    #[test]
    fn test_spec_deterministic() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();

        for index in [0u64, 1, 7, 524, 525, 99_999] {
            let a = space.spec_at(&config, index);
            let b = space.spec_at(&config, index);
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.truncate_turn, b.truncate_turn);
            assert_eq!(a.deck, b.deck);
            assert_eq!(a.uma, b.uma);
        }

        let a = space.spec_at(&config, 3);
        let b = space.spec_at(&config, 3 + space.len() as u64);
        println!("index=3    : uma={} turn={} seed={:#x}", a.uma, a.truncate_turn, a.seed);
        println!("index=3+N  : uma={} turn={} seed={:#x}", b.uma, b.truncate_turn, b.seed);
        assert_eq!(a.deck, b.deck, "分层设计下相差一整轮应回到同一卡组");
        assert_ne!(a.seed, b.seed, "种子必须随 index 变化");
        Ok(())
    }

    /// 截断回合覆盖 0..=77，卡组分层覆盖完全均衡
    #[test]
    fn test_spec_covers_turn_range_and_decks() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        let n = space.len() as u64 * 20;

        let mut turns: BTreeSet<i32> = BTreeSet::new();
        // 键必须是 (马娘, 卡组)：不同马娘可以共用同一副卡组，只按卡组计会把 525 折叠成 85
        let mut deck_hits: HashMap<(u32, [u32; 6]), usize> = HashMap::new();
        for index in 0..n {
            let spec = space.spec_at(&config, index);
            assert!(
                (0..=config.max_turn).contains(&spec.truncate_turn),
                "截断回合越界: {}",
                spec.truncate_turn
            );
            turns.insert(spec.truncate_turn);
            *deck_hits.entry((spec.uma, spec.deck)).or_default() += 1;
        }
        println!("{n} 个任务覆盖 {} / 78 个截断回合", turns.len());
        assert_eq!(turns.len(), 78, "截断回合应覆盖 0..=77 全部");

        let min = deck_hits.values().min().copied().unwrap_or(0);
        let max = deck_hits.values().max().copied().unwrap_or(0);
        println!("卡组命中次数 min={min} max={max}（分层设计下应恰好相等）");
        assert_eq!(deck_hits.len(), space.len(), "所有卡组都应被覆盖到");
        assert_eq!(min, max, "整轮 index 下分层必须完全均衡");
        Ok(())
    }

    // ========== 采样执行 ==========

    /// 采样可复现：同 index 两次得到逐位相同的根局面
    #[test]
    fn test_sample_reproducible() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        for index in [0u64, 11, 400, 1234] {
            let a = must_capture(&space, &config, index)?;
            let b = must_capture(&space, &config, index)?;
            println!("index={index}: {}", summary(&a));
            assert!(a == b, "index={index} 两次采样必须逐位一致");
        }
        Ok(())
    }

    /// 换 index 必须换局面——且要在**同一卡组**上验证
    ///
    /// 若只比 index=0 与 index=1，两者本来就是不同卡组，即使种子派生完全失效
    /// 该测试也会绿。故取相差一整轮的两个 index：卡组相同，只有种子与截断回合不同。
    #[test]
    fn test_sample_seed_actually_used() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        let period = space.len() as u64;
        let a = must_capture(&space, &config, 3)?;
        let b = must_capture(&space, &config, 3 + period)?;
        assert_eq!(a.spec.deck, b.spec.deck, "相差一整轮应是同一卡组");
        assert_eq!(a.spec.uma, b.spec.uma);
        println!("index=3        : {}", summary(&a));
        println!("index=3+{period}   : {}", summary(&b));
        assert!(a.game != b.game, "同卡组不同种子必须得到不同局面");
        Ok(())
    }

    /// ε 扰动确实改变轨迹
    ///
    /// 固定同一 index（同马娘同卡组同截断回合同种子），只改 ε。
    /// 单个 index 可能恰好在截断前没触发扰动，故统计一批。
    #[test]
    fn test_epsilon_perturbs_trajectory() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let plain = SamplerConfig {
            epsilon: 0.0,
            ..SamplerConfig::default()
        };
        let noisy = SamplerConfig {
            epsilon: 0.15,
            ..SamplerConfig::default()
        };

        let mut differ = 0;
        let mut compared = 0;
        for index in 0..40u64 {
            // 任一侧育成提前失败就跳过：那种情况下两侧不可比
            let (SampleOutcome::Captured(a), SampleOutcome::Captured(b)) = (
                sample_position(&space, &plain, index)?,
                sample_position(&space, &noisy, index)?
            ) else {
                continue;
            };
            compared += 1;
            if a != b {
                differ += 1;
            }
        }
        println!("{differ}/{compared} 个可比任务的轨迹被 ε=0.15 改变");
        assert!(compared >= 20, "可比任务太少（{compared}），样本不足以下结论");
        assert!(differ > 0, "ε 扰动完全没有生效");
        Ok(())
    }

    /// 越界的 epsilon 必须报错而不是 panic
    ///
    /// `rand::Rng::random_bool` 对区间外概率直接 panic，而采样是生产路径。
    #[test]
    fn test_epsilon_out_of_range_rejected() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        for bad in [-0.1, 1.5] {
            let config = SamplerConfig {
                epsilon: bad,
                ..SamplerConfig::default()
            };
            let err = sample_position(&space, &config, 0).expect_err("越界 epsilon 应当报错");
            println!("epsilon={bad} -> {err}");
        }
        Ok(())
    }

    /// 验收指标：采样产出的根局面覆盖绝大多数回合
    ///
    /// 有几个回合**结构上不可能**成为根局面，不是缺陷：
    /// - 回合 11 是出道赛，唯一动作，不构成决策点；
    /// - URA 决赛前后的末尾回合同样没有可选动作。
    ///
    /// 因此这里断言覆盖数而非「78 个全覆盖」，实测覆盖 74 个。
    #[test]
    fn test_sample_covers_all_turns() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();

        let mut reached: BTreeSet<i32> = BTreeSet::new();
        let mut stages: HashMap<String, usize> = HashMap::new();
        let mut exhausted = 0usize;
        let mut early_fail = 0usize;
        let total = 1200u64;
        for index in 0..total {
            match sample_position(&space, &config, index)? {
                SampleOutcome::Captured(pos) => {
                    assert!(pos.actions.len() >= config.min_actions);
                    assert!((0..=config.max_turn).contains(&pos.turn), "回合越界 {}", pos.turn);
                    assert!(pos.turn >= pos.spec.truncate_turn, "捕获点早于截断回合");
                    reached.insert(pos.turn);
                    *stages.entry(format!("{:?}", pos.stage)).or_default() += 1;
                }
                SampleOutcome::Exhausted { truncate_turn, final_turn } => {
                    exhausted += 1;
                    if final_turn < truncate_turn {
                        early_fail += 1;
                    }
                }
            }
        }
        let mut stage_list: Vec<_> = stages.iter().collect();
        stage_list.sort();
        // SuperRamenSelect 已是阶段入口，允许捕获。
        // 第 1 年地区选择现在也走 turn 2 的 RegionSelect 阶段，同样在白名单内。
        for (stage, count) in stages.iter() {
            assert!(
                matches!(
                    stage.as_str(),
                    "RamenSelect" | "SpecialSelect" | "Train" | "RegionSelect" | "SuperRamenSelect"
                ),
                "捕获到非阶段入口的根局面: {stage} x{count}"
            );
        }
        println!(
            "{total} 次采样：覆盖 {} 个回合，Exhausted {exhausted} 次（其中育成提前失败 {early_fail} 次）",
            reached.len()
        );
        println!("阶段分布: {stage_list:?}");
        let missing: Vec<i32> = (0..=config.max_turn).filter(|t| !reached.contains(t)).collect();
        println!("未覆盖回合: {missing:?}");
        assert!(reached.len() >= 70, "回合覆盖过窄，只有 {} 个", reached.len());
        Ok(())
    }

    /// 采出的根局面能直接推进到终局——状态合法性的硬证据
    ///
    /// Phase 1 踩过的空推进 `next()` 会造出非法状态，那种状态在这里会立刻报错。
    #[test]
    fn test_sampled_position_is_advanceable() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        let mut rows = Vec::new();
        for pos in first_captured(&space, &config, 30)? {
            let index = pos.spec.index;
            let mut game = pos.game.clone();
            let mut rng = pos.decision_rng.clone();
            let trainer = RamenHandwrittenTrainer::new();
            game.run_stage(&trainer, &mut rng)?;
            while game.next() {
                game.run_stage(&trainer, &mut rng)?;
            }
            game.on_simulation_end(&trainer, &mut rng)?;
            rows.push((index, pos.turn, game.uma.calc_score()));
        }
        println!("{} 个根局面全部推进到终局 (index, 起跑回合, 终局评分):", rows.len());
        for (index, turn, score) in rows.iter() {
            println!("  {index:>3} @ 回合 {turn:>2} -> {score}");
        }
        Ok(())
    }

    /// 端到端：采样器产出的局面能直接喂给搜索
    ///
    /// 这是 Phase 2 采样器与 Phase 1 搜索层的接缝，也是 Phase 3 的最小闭环。
    #[test]
    fn test_sampled_position_feeds_search() -> Result<()> {
        setup()?;
        let space = SamplingSpace::gen1()?;
        let config = SamplerConfig::default();
        let search: FlatSearch<RamenGame> = FlatSearch::new(SearchConfig::default().with_search_n(8).with_ucb(false));

        for index in [3u64, 42] {
            let pos = must_capture(&space, &config, index)?;
            let mut rng = pos.decision_rng.clone();
            let out = search.search(&pos.game, &pos.actions, &mut rng)?;
            println!(
                "index={index} 回合 {} 阶段 {:?} 候选 {}",
                pos.turn,
                pos.stage,
                pos.actions.len()
            );
            for (i, result) in out.action_results.iter().enumerate() {
                println!(
                    "  #{i} {} n={} mean={:.1}",
                    pos.actions[i],
                    result.0.count(),
                    result.0.mean()
                );
            }
            assert_eq!(out.action_results.len(), pos.actions.len(), "搜索输出应与候选表等长");
            assert!(
                out.action_results.iter().any(|r| r.0.count() > 0),
                "搜索没有产出任何有效样本"
            );
        }
        Ok(())
    }
}
