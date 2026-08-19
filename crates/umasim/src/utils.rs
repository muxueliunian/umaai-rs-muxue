use std::{
    io::Write,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Result, anyhow};
use colored::Colorize;
use comfy_table::Table;
use flexi_logger::{DeferredNow, Duplicate, FileSpec, LogSpecification, style};
use log::{Record, error};
use serde::Serialize;

use crate::{
    gamedata::{EventCollection, EventData, GAMECONSTANTS, GAMEDATA, GameConfig, LOGGER, OverrideGameConfig},
    global
};

pub type Array5 = [i32; 5];
pub type Array6 = [i32; 6];

/// 记录 `flexi_logger::start()` 是否已成功调用过
///
/// 仅用于解决 `cargo test` 并行运行时的 TOCTOU 竞争问题：
/// 多个测试同时调用 `init_logger`，但 log crate 全局状态只能初始化一次。
/// 单纯依赖 `LOGGER.get().is_some()` 检查存在竞争窗口
/// （A 线程 `get()` 返回 None 后被 B 线程 `set()` 抢先，
/// 然后 A 调用 `start()` 会触发 "logger already initialized" 报错）。
///
/// 设置为 true 后，后续 `init_logger` 直接返回 Ok，不再调用 `start()`。
static LOGGER_INIT_DONE: AtomicBool = AtomicBool::new(false);

/// 串行化 `init_logger` 的初始化过程
///
/// 在持锁状态下检查 `LOGGER_INIT_DONE`/`LOGGER` 并执行 `start()`，
/// 保证 log crate 全局状态只被第一个线程初始化一次，
/// 其他线程观察到 `LOGGER_INIT_DONE = true` 后直接返回 Ok。
static INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn log_format(w: &mut dyn Write, _now: &mut DeferredNow, record: &Record) -> Result<(), std::io::Error> {
    let level = record.level();
    write!(
        w,
        "{} {}",
        style(level).paint(level.to_string()[..1].to_string()),
        style(level).paint(record.args().to_string())
    )
}

/// 初始化日志系统（默认：写文件 + stderr）
pub fn init_logger(app: &str, spec: &str) -> Result<()> {
    init_logger_with(app, spec, true)
}

/// 同 `init_logger`，但支持自定义是否输出到 stderr
///
/// - `duplicate_stderr=true`：写文件 + stderr（默认）
/// - `duplicate_stderr=false`：只写文件，不占用 stderr（TUI 兼容）
pub fn init_logger_with(app: &str, spec: &str, duplicate_stderr: bool) -> Result<()> {
    // 快速路径：已初始化过则直接返回
    if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    // 串行化初始化：避免并行测试同时调用 start() 导致 log crate 重复初始化
    let lock = INIT_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

    // 双重检查：持锁状态下再次检查
    if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }

    let result: Result<()> = (|| {
        let logger = flexi_logger::Logger::try_with_str(spec)?
            .format_for_stderr(log_format)
            .log_to_file(FileSpec::default().directory("logs").basename(app));
        let logger = if duplicate_stderr {
            logger.duplicate_to_stderr(Duplicate::All).start()?
        } else {
            // 只输出到文件，不干扰 stderr（TUI 玩家测试场景）
            logger.start()?
        };
        // LOGGER.set 可能失败（被其他线程抢先），但只要 start 成功，log crate 已被初始化
        let _ = LOGGER.set(Mutex::new(logger));
        Ok(())
    })();
    // start 成功则标记 LOG_CRATE 已初始化，后续调用直接 return Ok
    if result.is_ok() {
        LOGGER_INIT_DONE.store(true, Ordering::Release);
    }
    result
}

/// 初始化日志系统：只输出到 stdout，不写文件
///
/// 适用于 `ramen_manual` 等玩家测试场景：
/// - 日志与 `println!` 一起显示在 stdout，玩家可以直接看到训练/事件日志
/// - inquire 默认从 `/dev/tty` 读取，三者互不干扰
///
/// 注意：flexi_logger 的 `log_to_stdout` 与 `log_to_file` 互斥，
/// 所以 stdout 模式不写文件，调用方需自行处理日志持久化（如重定向 shell 输出）。
pub fn init_logger_stdout(app: &str, spec: &str) -> Result<()> {
    // 快速路径：已初始化过则直接返回
    if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }
    let lock = INIT_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

    if LOGGER.get().is_some() || LOGGER_INIT_DONE.load(Ordering::Acquire) {
        return Ok(());
    }

    let result: Result<()> = (|| {
        let logger = flexi_logger::Logger::try_with_str(spec)?
            .format_for_stdout(log_format)
            .log_to_stdout()
            .start()?;
        let _ = LOGGER.set(Mutex::new(logger));
        Ok(())
    })();
    if result.is_ok() {
        LOGGER_INIT_DONE.store(true, Ordering::Release);
    }
    result
}

pub fn disable_log() {
    // LOGGER 未初始化时直接返回（init_logger 之前/之后/被 reset 都可能触发）
    if let Some(logger) = LOGGER.get() {
        logger
            .lock()
            .expect("logger lock")
            .push_temp_spec(LogSpecification::off());
    }
}

pub fn enable_log() {
    // 与 disable_log 配对：仅在 LOGGER 已初始化时恢复
    if let Some(logger) = LOGGER.get() {
        logger.lock().expect("logger lock").pop_temp_spec();
    }
}

/// 把当前工作目录修改为exe所在目录
pub fn check_working_dir() -> Result<()> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().expect("parent");
    println!("正在进入UmaAI所在目录: {exe_dir:?}");
    std::env::set_current_dir(exe_dir)?;
    Ok(())
}

/// 获取workspace根目录路径
/// 
/// 通过CARGO_MANIFEST_DIR环境变量定位workspace根目录，
/// 适用于测试和需要访问workspace级别资源（如gamedata目录）的场景。
/// 
/// # 返回值
/// 返回workspace根目录的PathBuf，如果无法定位则返回错误。
/// 
/// # 示例
/// ```rust
/// use umasim::utils::get_workspace_root;
/// 
/// let workspace_root = get_workspace_root().expect("无法获取workspace根目录");
/// println!("Workspace根目录: {:?}", workspace_root);
/// ```
pub fn get_workspace_root() -> Result<std::path::PathBuf> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow!("无法定位workspace根目录，请确保在正确的crate中运行"))?;
    Ok(workspace_root.to_path_buf())
}

/// 检测终端类型
pub fn check_windows_terminal() -> Result<()> {
    if !std::env::var("WT_SESSION").is_ok() {
        println!(
            "{}",
            "警告: 当前终端不是Windows Terminal或版本太老，可能出现乱码或显示不全".yellow()
        );
        println!(
            "{}",
            "UmaAI推荐使用最新版Windows Terminal终端，以获得更好的体验".bright_green()
        );
        pause()?;
    }
    Ok(())
}

pub fn pause() -> Result<()> {
    println!("按任意键继续...");
    std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}

pub fn make_table<T: Serialize>(data: &[T]) -> Result<Table> {
    let mut table = Table::new();
    table.set_truncation_indicator("...");
    let mut has_headers = false;
    for row in data {
        if !has_headers {
            let header = serde_json::to_value(row)?;
            table.set_header(header.as_object().expect("map").keys());
            has_headers = true;
        }
        let row = serde_json::to_value(row)?;
        table.add_row(row.as_object().expect("row").values());
    }
    Ok(table)
}

pub fn format_luck(prefix: &str, luck: f64) -> String {
    let luck_str = if luck < 0.0 {
        format!("{luck:.0}")
    } else {
        format!("+{luck:.0}")
    };
    if luck < -1600.0 {
        format!("{prefix} {}", luck_str.red())
    } else if luck < -400.0 {
        format!("{prefix} {}", luck_str.yellow())
    } else if luck < 400.0 {
        format!("{prefix} {luck_str}")
    } else if luck < 1600.0 {
        format!("{prefix} {}", luck_str.green())
    } else {
        format!("{prefix} {}", luck_str.bright_green())
    }
}

#[macro_export]
macro_rules! global {
    ($name:ident) => {
        $name.get().expect(concat!(stringify!($name), " not initialized"))
    };
}

pub fn global_events() -> &'static EventCollection {
    &global!(GAMEDATA).events
}
/// 获得events.json里记载的指定system事件
pub fn system_event(key: &str) -> Result<&'static EventData> {
    global_events()
        .system_events
        .get(key)
        .ok_or(anyhow!("未知系统事件: {key}"))
}
/// 获得constants.json里记载的指定事件概率
pub fn system_event_prob(key: &str) -> Result<f64> {
    global!(GAMECONSTANTS)
        .event_probs
        .get(key)
        .map(|x| *x as f64)
        .ok_or(anyhow!("未知事件概率: {key}"))
}

pub trait AttributeArray {
    fn add_eq(&mut self, other: &Self) -> &mut Self;

    fn is_default(&self) -> bool;
}

impl<const N: usize> AttributeArray for [i32; N] {
    fn add_eq(&mut self, other: &Self) -> &mut Self {
        if self.len() != other.len() {
            error!("self: {self:?}, other: {other:?}");
            panic!("数组长度不匹配, 请检查调用代码");
        }
        for (i, x) in self.iter_mut().enumerate() {
            *x += other[i];
        }
        self
    }

    fn is_default(&self) -> bool {
        self.iter().all(|x| *x == 0)
    }
}

pub fn split_status(status_pt: &Array6) -> Result<(&Array5, i32)> {
    let left: &Array5 = status_pt[..5].try_into()?;
    let right = status_pt[5];
    Ok((left, right))
}

/// 载入 gamedata/default_config.toml, 和 game_config.toml 合并
pub fn load_game_config() -> Result<GameConfig> {
    let def_file = fs_err::read_to_string("gamedata/default_config.toml")?;
    let default_config: GameConfig = toml::from_str(&def_file)?;
    let cfg_file = fs_err::read_to_string("game_config.toml")?;
    let override_config: OverrideGameConfig = toml::from_str(&cfg_file)?;
    let ret = override_config.merge(&default_config);
    //println!("{ret:#?}");
    Ok(ret)
}
