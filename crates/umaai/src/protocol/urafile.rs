use std::{
    env,
    fmt::Debug,
    path::Path,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use anyhow::{Result, anyhow};
use colored::Colorize;
use log::{info, warn};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::Value;

use crate::protocol::GameStatus;

pub fn format_err<E: Debug>(text: String, cause: E) -> anyhow::Error {
    anyhow!("{} ->\n{cause:?}", text.red())
}

pub struct UraFileWatcher {
    /// 保持 watcher 存活（被 drop 即停止监听）；事件实际在 producer 线程消费
    pub watcher: RecommendedWatcher,
    /// 处理队列：producer 线程把读到的完整 JSON 依次推入，watch() 逐个消费
    queue_rx: Receiver<String>,
    /// 当前监听的目标文件名（producer 线程按 basename 过滤事件）
    target_name: Arc<Mutex<String>>,
    /// producer 线程句柄，持有防止线程被提前终止
    _producer: thread::JoinHandle<()>,
}

impl UraFileWatcher {
    /// 定位小黑板数据目录，可能在当前目录下的 .portable 或者 appdata 下
    ///
    /// **健壮性**：所有失败路径（路径含中文 / `.portable` 不存在 / `LOCALAPPDATA`
    /// 缺失 / `UmamusumeResponseAnalyzer` 不存在）都改成 `warn!` + 返回空字符串，
    /// **不 bail**——让 caller（`init` / `main`）拿到空字符串后自行决定怎么处理。
    /// 原本 `bail!` 的语义是「缺配置就退出」，但小黑板配置不齐不应让 main panic：
    /// 实际游戏中路径是齐的；开发机 / CI / 移植到 Linux 时允许缺失并继续运行。
    ///
    /// `LOCALAPPDATA` 是 Windows 专用环境变量，Linux 上不存在。用
    /// `unwrap_or_default()` 而非 `?`：env var 缺失不抛 Err（污染成
    /// "environment variable not found"）。
    pub fn ura_root() -> Result<String> {
        // 先检查.portable
        if Path::new("./.portable").is_dir() {
            let ret = dunce::canonicalize(".portable")?;
            if let Some(s) = ret.to_str() {
                Ok(s.to_string())
            } else {
                warn!("路径中暂时不能包含中文: {}，小黑板数据目录降级为空", ret.to_string_lossy());
                Ok(String::new())
            }
        } else {
            // `?` 改成 `unwrap_or_default()`：见上方注释
            let local_app_path = env::var("LOCALAPPDATA").unwrap_or_default();
            let local_app_ura = format!("{local_app_path}/UmamusumeResponseAnalyzer");
            if Path::new(&local_app_ura).is_dir() {
                Ok(local_app_ura)
            } else {
                warn!(
                    "没有找到小黑板数据目录（Linux 需在 cwd 创建 .portable 子目录；Windows 需安装小黑板并设 LOCALAPPDATA）"
                );
                Ok(String::new())
            }
        }
    }
    /// 定位 SendGameStatusPlugin 输出数据的目录
    pub fn plugin_dir() -> Result<String> {
        let ura_root = Self::ura_root()?;
        Ok(format!("{ura_root}/PluginData/SendGameStatusPlugin"))
    }

    pub fn init() -> Result<Self> {
        let ura_dir = Self::plugin_dir()?;
        info!("小黑板数据目录: {}", ura_dir.cyan());

        // **健壮性**：路径为空说明 `ura_root` 没找到本地化目录。
        // 不 bail，让 main 走 match 路径优雅退出；这里只打 warn 提示。
        if ura_dir.is_empty() {
            warn!("小黑板数据目录为空字符串，watcher 不会工作（程序会退出而非 panic）");
            anyhow::bail!("小黑板数据目录无效（详见上方 warn）");
        }
        // 确保这个目录存在——缺失时 warn + bail（让 main match 优雅退出）
        if !fs_err::exists(&ura_dir).unwrap_or(false) {
            warn!("回合数据目录不存在: {}，请检查小黑板 SendGameStatusPlugin 插件是否正常工作", ura_dir);
            anyhow::bail!("回合数据目录不存在");
        }
        let ura_file = Path::new(&ura_dir).join("thisTurn.json");
        if !fs_err::exists(&ura_file).unwrap_or(false) {
            info!("{}", "开始接收游戏数据，请开始育成".green());
            warn!("如果开始育成后仍然显示此消息，请重启小黑板并检查 SendGameStatusPlugin 插件是否正确工作");
        }

        // 1) notify 事件通道：由 producer 线程独占消费
        let (event_tx, event_rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(event_tx)?;
        if let Err(e) = watcher.watch(&Path::new(&ura_dir), RecursiveMode::NonRecursive) {
            warn!("watcher.watch({ura_dir:?}) 失败: {e}");
            anyhow::bail!("watcher.watch 失败");
        }

        // 2) 处理队列：读到的完整 JSON 依次入队，主循环（watch）逐个消费
        let (queue_tx, queue_rx) = mpsc::channel::<String>();

        // 3) 后台 producer 线程：独立于主计算循环、随时消费事件，每次目标文件的
        //    写事件 → 读一份完整 JSON → 推入处理队列。主循环计算期间积压的事件
        //    不会像旧的"合并排空"那样被吞掉，而是一份份入队、一份份被消费。
        let target_name = Arc::new(Mutex::new(Self::URA_TARGET_DEFAULT.to_string()));
        let producer_target = Arc::clone(&target_name);
        let producer_dir = ura_dir.clone();
        let _producer = thread::Builder::new()
            .name("urafile-producer".to_string())
            .spawn(move || producer_loop(event_rx, queue_tx, producer_target, producer_dir))?;

        // watcher 保持存活（drop 即停止监听）；事件通道在 producer 线程里消费
        Ok(Self {
            watcher,
            queue_rx,
            target_name,
            _producer,
        })
    }

    /// 默认监听的回合数据文件名
    const URA_TARGET_DEFAULT: &str = "thisTurn.json";

    /// 从处理队列取下一份完整 JSON（消费端）。
    ///
    /// **架构（2026-09，生产者-消费者模型）**：目录监听和文件读取全部由后台
    /// `urafile-producer` 线程负责——每收到目标文件的一次 Create/Modify 事件就
    /// `read_stable` 读一份完整 JSON 推入处理队列；主循环在这里逐个消费。
    ///
    /// 相比旧的「收到事件后合并排空 + contents 去重」设计：只要 producer 能观察
    /// 到的写事件，其对应的完整 JSON 都会进队列（内容与上次完全相同的冗余事件
    /// 除外），主循环计算慢时队列里积压多份回合快照，一份份消费、一份不主动丢。
    /// notify 底层缓冲溢出丢的是事件（文件系统层面无法避免），但程序逻辑层面
    /// 不再有任何"吞掉中间回合"的路径。
    pub fn watch(&mut self, filename: &str) -> Result<String> {
        // 通知 producer 当前监听的目标文件名（当前只有 thisTurn.json，保留参数
        // 兼容主循环调用处；后台线程在下一次处理事件时读取，天然无竞争）
        if let Ok(mut guard) = self.target_name.lock() {
            *guard = filename.to_string();
        }
        // 队列里有 JSON 立即返回；没有则阻塞等待 producer 推入下一份
        self.queue_rx
            .recv()
            .map_err(|_| anyhow!("处理队列已关闭（producer 线程已退出）"))
    }

    /// 稳定读取：写入中可能读到空 / 半截内容或撞上文件锁，带重试 + 两次一致校验。
    ///
    /// 重试耗尽时返回最后一次成功读取的内容（比抛错让进程退出更宽容，最坏情况是
    /// 该事件被跳过、等下一个事件再试）；一次都没读到才报错。
    const READ_STABLE_ATTEMPTS: usize = 10;
    const READ_STABLE_INTERVAL: Duration = Duration::from_millis(50);

    fn read_stable(full_path: &Path) -> Result<String> {
        let mut last_ok: Option<String> = None;
        for _ in 0..Self::READ_STABLE_ATTEMPTS {
            match fs_err::read_to_string(full_path) {
                Ok(contents) => {
                    if contents.is_empty() {
                        // 空文件 = 正被 truncate 写入中
                        thread::sleep(Self::READ_STABLE_INTERVAL);
                        continue;
                    }
                    if let Some(prev) = &last_ok {
                        if prev == &contents {
                            // 两次读取一致 → 写入完成，快照稳定
                            return Ok(contents);
                        }
                    }
                    last_ok = Some(contents);
                    thread::sleep(Self::READ_STABLE_INTERVAL);
                }
                Err(_) => {
                    // 文件被独占打开等瞬时错误：重试
                    thread::sleep(Self::READ_STABLE_INTERVAL);
                }
            }
        }
        last_ok.ok_or_else(|| anyhow!("多次读取 {} 失败（文件可能持续被写入）", full_path.display()))
    }
}

/// 后台 producer 线程主体：持续消费 notify 事件，每次目标文件写事件 → 读一份完整
/// JSON → 推入处理队列。事件通道关闭（watcher 被 drop）时线程退出。
fn producer_loop(
    event_rx: Receiver<notify::Result<Event>>,
    queue_tx: Sender<String>,
    target_name: Arc<Mutex<String>>,
    ura_dir: String,
) {
    // 读目标文件的一份完整 JSON 并推入处理队列；内容与上一次已入队完全相同时跳过
    // （同一回合的覆盖写、或读得比写慢时多次读到同一份最新内容，无需重复入队——
    // 但任何「内容不同的回合」都会被保留，一份不丢）。读取失败仅 warn 跳过本事件。
    let read_and_push = |last_pushed: &mut Option<String>| {
        let target = Path::new(&ura_dir).join(current_target(&target_name));
        match UraFileWatcher::read_stable(&target) {
            Ok(contents) => {
                if last_pushed.as_deref() != Some(contents.as_str()) {
                    let _ = queue_tx.send(contents.clone());
                    *last_pushed = Some(contents);
                }
            }
            Err(e) => warn!("读取 {} 失败（跳过本事件，等待下一事件）: {e}", target.display()),
        }
    };

    let mut last_pushed: Option<String> = None;

    // 启动兜底：若目标文件已存在（umaai 中途重启 / 已进入育成），先推一份现存
    // 快照进队列，让主循环立即显示窗口并计算当前回合；否则要等下一次写事件才醒来。
    let target = Path::new(&ura_dir).join(current_target(&target_name));
    if fs_err::exists(&target).unwrap_or(false) {
        read_and_push(&mut last_pushed);
    }

    loop {
        match event_rx.recv() {
            Ok(Ok(event)) => {
                // 只处理目标文件的 Create/Modify 事件；目录里的 game*.json 等其它
                // 文件事件直接忽略——忽略时不碰队列，排队的本文件事件永远不被吞。
                if is_target_event(&event, &target_name) {
                    read_and_push(&mut last_pushed);
                }
            }
            Ok(Err(e)) => {
                // notify 错误（Windows 快速写入导致缓冲溢出时以错误事件上报）说明
                // 事件可能已丢失：主动重读一次兜底，内容变化才会入队。
                warn!("文件监听事件错误（可能丢失事件，重读兜底）: {e}");
                read_and_push(&mut last_pushed);
            }
            Err(_) => break, // 事件通道关闭 → producer 退出
        }
    }
}

/// 当前监听的目标文件名（锁中毒时回退到默认 thisTurn.json）
fn current_target(target_name: &Arc<Mutex<String>>) -> String {
    target_name
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|_| UraFileWatcher::URA_TARGET_DEFAULT.to_string())
}

/// 事件是否指向当前监听的目标文件（按 basename 匹配，避开 Windows 路径前缀差异）
fn is_target_event(event: &Event, target_name: &Arc<Mutex<String>>) -> bool {
    if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
        return false;
    }
    let name = current_target(target_name);
    event
        .paths
        .iter()
        .any(|p| p.file_name().map(|f| f.to_string_lossy().as_ref() == name).unwrap_or(false))
}

/// 载入小黑板数据并提供详细错误信息
pub fn parse_game<S: GameStatus>(contents: &str) -> Result<S::Game> {
    // 先解析json
    let value: Value = serde_json::from_str(contents).map_err(|e| format_err("Json格式错误".to_string(), e))?;
    // 解析baseGame.scenarioId
    if let Some(base) = value.get("baseGame") {
        let scenario = base.get("scenarioId").and_then(|x| x.as_i64());
        if scenario != Some(S::scenario_id() as i64) {
            return Err(anyhow!(
                "{}",
                format!("剧本错误: {scenario:?} != {}", S::scenario_id()).red()
            ));
        }
    } else {
        return Err(anyhow!(
            "{}",
            "缺少baseGame.scenarioId，请使用和AI配套发布的小黑板".red()
        ));
    }
    let status: S = serde_json::from_value(value).map_err(|e| format_err("回合数据出错".to_string(), e))?;
    status
        .into_game()
        .map_err(|e| format_err("载入回合出错".to_string(), e))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::RecvTimeoutError;

    use super::*;
    use notify::event::{CreateKind, ModifyKind};

    fn event(kind: EventKind, path: std::path::PathBuf) -> notify::Event {
        notify::Event {
            kind,
            paths: vec![path],
            attrs: notify::event::EventAttributes::default(),
        }
    }

    /// 起一个真实 temp 目录 + producer 线程（事件用合成通道喂入）。
    /// 先写目标文件再启动 producer，避免启动兜底与测试断言竞态。
    #[allow(clippy::type_complexity)]
    fn start_producer(
        test_dir: &str,
        initial: &str,
    ) -> (
        std::path::PathBuf,
        mpsc::Sender<notify::Result<Event>>,
        Receiver<String>,
        thread::JoinHandle<()>,
    ) {
        let dir = std::env::temp_dir().join(test_dir);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("thisTurn.json");
        std::fs::write(&target, initial).unwrap();

        let (event_tx, event_rx) = mpsc::channel();
        let (queue_tx, queue_rx) = mpsc::channel();
        let target_name = Arc::new(Mutex::new("thisTurn.json".to_string()));
        let producer_target = Arc::clone(&target_name);
        let producer_dir = dir.to_string_lossy().into_owned();
        let handle = thread::spawn(move || producer_loop(event_rx, queue_tx, producer_target, producer_dir));

        (target, event_tx, queue_rx, handle)
    }

    /// 每次相关写事件 → 队列里都有一份对应 JSON，一份不丢；不相关文件事件不入队。
    ///
    /// 这是本次重构的核心回归：旧的"排空合并"会把处理期间积压的多次写事件合并成
    /// 一次、只留最新，中间回合的 JSON 被主动丢弃；现在每个事件都读到并逐个入队。
    #[test]
    fn producer_pushes_every_related_event() {
        let (target, event_tx, queue_rx, handle) = start_producer("urafile_producer_test", r#"{"turn":1}"#);

        // 启动兜底：现存文件推入队列
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":1}"#);

        // 写入 + Modify 事件 → 第二份
        std::fs::write(&target, r#"{"turn":2}"#).unwrap();
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), target.clone()))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":2}"#);

        // 不相关文件（game*.json）事件 → 不入队，且绝不能吞掉队列里的本文件事件
        let unrelated = target.with_file_name("game6211_turn1.json");
        std::fs::write(&unrelated, "x").unwrap();
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), unrelated))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_millis(200)), Err(RecvTimeoutError::Timeout));

        // Create 事件同样触发读取入队
        std::fs::write(&target, r#"{"turn":3}"#).unwrap();
        event_tx.send(Ok(event(EventKind::Create(CreateKind::Any), target.clone()))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":3}"#);

        drop(event_tx);
        handle.join().unwrap();
    }

    /// 内容与上次完全相同的冗余事件只入队一次；内容变化（新回合）必须依次入队。
    #[test]
    fn producer_deduplicates_identical_content_keeps_distinct() {
        let (target, event_tx, queue_rx, handle) = start_producer("urafile_producer_dedup_test", r#"{"turn":1}"#);
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":1}"#);

        // 内容没变时连发两个 Modify 事件 → 都跳过（不重复入队）
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), target.clone()))).unwrap();
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), target.clone()))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_millis(300)), Err(RecvTimeoutError::Timeout));

        // 内容变化 → 每一份不同回合都入队（等上一份已入队后再写下一份，保证确定性）
        std::fs::write(&target, r#"{"turn":2}"#).unwrap();
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), target.clone()))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":2}"#);

        std::fs::write(&target, r#"{"turn":3}"#).unwrap();
        event_tx.send(Ok(event(EventKind::Modify(ModifyKind::Any), target.clone()))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":3}"#);

        drop(event_tx);
        handle.join().unwrap();
    }

    /// notify 报错（Windows 缓冲溢出等）→ producer 重读兜底，内容变化仍会入队
    #[test]
    fn producer_reeads_on_notify_error() {
        let (target, event_tx, queue_rx, handle) = start_producer("urafile_producer_err_test", r#"{"turn":1}"#);
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":1}"#);

        // 写入新回合后发一个 notify 错误事件 → 重读兜底入队
        std::fs::write(&target, r#"{"turn":9}"#).unwrap();
        event_tx.send(Err(notify::Error::generic("模拟缓冲溢出"))).unwrap();
        assert_eq!(queue_rx.recv_timeout(Duration::from_secs(1)).unwrap(), r#"{"turn":9}"#);

        drop(event_tx);
        handle.join().unwrap();
    }
}
