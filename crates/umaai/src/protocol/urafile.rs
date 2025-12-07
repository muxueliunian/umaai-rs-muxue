use std::{
    env,
    fmt::Debug,
    path::Path,
    sync::mpsc::{self, Receiver}
};

use anyhow::{Result, anyhow};
use colored::Colorize;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher, event::ModifyKind};

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
    pub fn ura_dir() -> Result<String> {
        let local_app_path = env::var("LOCALAPPDATA")?;
        Ok(format!(
            "{local_app_path}/UmamusumeResponseAnalyzer/PluginData/SendGameStatusPlugin"
        ))
    }

    pub fn init() -> Result<Self> {
        let ura_dir = Self::ura_dir()?;
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(&Path::new(&ura_dir), RecursiveMode::NonRecursive)?;
        Ok(Self {
            watcher,
            rx,
            contents: String::new()
        })
    }

    /// 捕获指定文件修改时的内容
    pub fn do_poll(&mut self, filename: &str) -> Result<String> {
        let full_path = Path::new(&Self::ura_dir()?).join(filename);
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
        let full_path = Path::new(&Self::ura_dir()?).join(filename);
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
