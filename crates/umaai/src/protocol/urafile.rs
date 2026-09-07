use std::{
    env,
    fmt::Debug,
    path::Path,
    sync::mpsc::{self, Receiver}
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

    /// 捕获指定文件修改时的内容
    pub fn do_poll(&mut self, filename: &str) -> Result<String> {
        let full_path = Path::new(&Self::plugin_dir()?).join(filename);
        loop {
            let event = self.rx.recv()??;
            if event.paths.contains(&full_path) && matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                if full_path.exists() {
                    // sanity check
                    let contents = fs_err::read_to_string(&full_path)?;
                    return Ok(contents);
                }
            }
        }
    }

    /// 等待直到指定文件内容改变
    pub fn watch(&mut self, filename: &str) -> Result<String> {
        let full_path = Path::new(&Self::plugin_dir()?).join(filename);
        // 初始化时尝试直接读取文件内容
        if self.contents.is_empty() && full_path.exists() {
            let contents = fs_err::read_to_string(&full_path)
                .map_err(|e| format_err(format!("读取 {filename} 出错，请检查小黑板通信"), e))?;
            self.contents = contents.clone();
            return Ok(contents);
        }
        loop {
            // 之后在变更时读取
            let contents = self
                .do_poll(filename)
                .map_err(|e| format_err(format!("监听 {filename} 出错，请检查小黑板通信"), e))?;
            if contents != self.contents {
                self.contents = contents.clone();
                return Ok(contents);
            }
        }
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
