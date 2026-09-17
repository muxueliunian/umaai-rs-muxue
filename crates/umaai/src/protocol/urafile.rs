use std::{
    env,
    fmt::Debug,
    path::Path,
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration
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
    pub watcher: RecommendedWatcher,
    pub rx: Receiver<notify::Result<Event>>,
    /// 文件内容缓存, 用于判断是否修改
    pub contents: String
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

        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(tx)?;
        if let Err(e) = watcher.watch(&Path::new(&ura_dir), RecursiveMode::NonRecursive) {
            warn!("watcher.watch({ura_dir:?}) 失败: {e}");
            anyhow::bail!("watcher.watch 失败");
        }
        Ok(Self {
            watcher,
            rx,
            contents: String::new()
        })
    }

    /// 等待直到指定文件内容改变
    ///
    /// **健壮性（2026-09）**：修复快速连续写入（上一回合还没算完、下一回合数据已更新）
    /// 时 AI 收不到计算结果的两个问题：
    /// 1. notify 事件缓冲溢出（Windows 上快速写入常见）会上报错误事件——旧实现
    ///    把错误 `?` 出去直接让进程退出；现在当作"可能错过了事件"醒来重读文件校验内容；
    /// 2. 写入中读取可能拿到空 / 半截 JSON——旧实现拿到半截内容就返回，主循环解析失败
    ///    后事件已被消费，下一次 `recv()` 永久阻塞（C# 端表现为收不到 decision / compute_done，
    ///    重启 AI 才恢复）。现在带重试 + 两次一致校验，只返回稳定快照；
    /// 3. 唤醒后排空队列里积压的同类事件——处理期间多次写入合并为一次读取最新内容，
    ///    避免处理中间快照浪费 MCTS 计算。
    pub fn watch(&mut self, filename: &str) -> Result<String> {
        let full_path = Path::new(&Self::plugin_dir()?).join(filename);
        // 初始化时尝试直接读取文件内容
        if self.contents.is_empty() && full_path.exists() {
            if let Ok(contents) = self.read_stable(&full_path) {
                self.contents = contents.clone();
                return Ok(contents);
            }
            // 写入中拿不到稳定快照 → 落到下方事件等待循环
        }
        loop {
            // 等待本文件的写事件；notify 错误（缓冲溢出等）也当作"可能错过事件"醒来
            self.wait_event(&full_path)?;
            // 无论因何醒来都重新读文件：内容变了才返回（合并快速连续写入）
            let contents = self.read_stable(&full_path)?;
            if contents != self.contents {
                self.contents = contents.clone();
                return Ok(contents);
            }
            // 内容未变（事件是覆盖写回原值等）→ 继续等下一个事件
        }
    }

    /// 等待本文件的下一个写事件；随后排空队列中积压的同类事件。
    ///
    /// notify 的错误事件（Windows 下快速写入导致缓冲溢出时以错误形式上报）不抛错，
    /// 直接返回让上层重读文件校验内容——避免事件丢失后永久阻塞在 `recv()`。
    fn wait_event(&mut self, full_path: &Path) -> Result<()> {
        loop {
            match self.rx.recv() {
                Ok(Ok(event)) if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) => {
                    let relevant = event.paths.iter().any(|p| p == full_path);
                    if relevant {
                        // 仅确认是本文件事件时才排空：把处理期间积压的同类（快写）事件
                        // 合并成一次，上层重读文件只取最新内容。绝对不能在 relevant == false
                        // 时排空——目录里还有 SendGameStatusPlugin 写的 game*.json 等其它文件，
                        // 收到它们的事件可能 precede 本文件事件，排空会把后面排队的
                        // thisTurn.json 事件一起吞掉，导致 recv() 永久阻塞（表现为
                        // "每次重启只算一回合，之后不再响应"）。
                        while matches!(self.rx.try_recv(), Ok(Ok(_))) {}
                        return Ok(());
                    }
                    // 不相关路径的事件 → 忽略继续等（不动队列）
                }
                Ok(Ok(_)) => {} // 其他事件类型（Remove/Rename/Access 等）忽略
                Ok(Err(_)) => {
                    // 缓冲溢出等瞬时错误：事件可能已丢失，返回让上层重读文件
                    return Ok(());
                }
                Err(_) => return Err(anyhow!("文件监听通道已关闭")),
            }
        }
    }

    /// 稳定读取：写入中可能读到空 / 半截内容或撞上文件锁，带重试 + 两次一致校验。
    ///
    /// 重试耗尽时返回最后一次成功读取的内容（比抛错让进程退出更宽容，最坏情况是
    /// 主循环解析失败后等待下一个事件）；一次都没读到才报错。
    const READ_STABLE_ATTEMPTS: usize = 10;
    const READ_STABLE_INTERVAL: Duration = Duration::from_millis(50);

    fn read_stable(&self, full_path: &Path) -> Result<String> {
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
    use super::*;
    use notify::event::ModifyKind;

    /// 回归测试：`wait_event` 收到不相关文件（如 game*.json）的事件后，
    /// 绝不能把队列里排队的本文件（thisTurn.json）事件排空吞掉。
    ///
    /// 修复前：不相关事件 → `relevant == false` → 无条件排空 → 吞掉随后排队的
    /// 相关事件 → 继续 `recv()` 永久阻塞（表现为 AI"每次重启只算一回合"）。
    #[test]
    fn wait_event_does_not_swallow_related_event_after_unrelated() {
        let (tx, rx) = mpsc::channel();
        let local = notify::recommended_watcher(tx.clone()).unwrap();
        let mut watcher = UraFileWatcher {
            watcher: local,
            rx,
            contents: String::new()
        };

        let dir = std::env::temp_dir().join("wait_event_swallow_test");
        let this_turn_full = dir.join("thisTurn.json");
        let unrelated = dir.join("game6211_turn1.json");

        let event = |path| notify::Event {
            kind: EventKind::Modify(ModifyKind::Any),
            paths: vec![path],
            attrs: notify::event::EventAttributes::default()
        };
        // 实际游戏顺序：SendGameStatusPlugin 先写 game*.json（不相关），
        // 随后原子替换 thisTurn.json（相关）——两个事件几乎同时进队列。
        tx.send(Ok(event(unrelated))).unwrap();
        tx.send(Ok(event(this_turn_full.clone()))).unwrap();

        // 修复后应正常返回；若修复被回退（相关事件被吞），这里会永久阻塞。
        watcher.wait_event(&this_turn_full).expect("wait_event 应返回 Ok");
    }
}
