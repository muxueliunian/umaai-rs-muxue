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
use log::{Record, error, info};
use serde::Serialize;

use crate::{
    gamedata::{
        EventCollection, EventData, GAMECONSTANTS, GAMEDATA, GameConfig, LOGGER,
        MctsConfig, OverrideConfig, OverrideGameConfig,
    },
    game::onsen::OnsenOrder,
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

// ========== 路径常量（Phase 2 步骤 4：加载集中化） ==========
//
// 路径解析优先级（从高到低）：
//   1. 环境变量 `UMAI_DATA_DIR`：data 根目录（含 gamedata/default_config.toml）
//   2. 工作目录下 `gamedata/default_config.toml`（默认）
//
// 用户配置 `game_config.toml` 始终位于工作目录根（Phase 6 可考虑移至 `UMAI_DATA_DIR`）。

/// 默认配置（开发者默认值）相对于 data 根目录的相对路径
pub const DEFAULT_CONFIG_REL_PATH: &str = "default_config.toml";
/// 用户配置（覆盖层）相对于工作目录的相对路径
pub const USER_CONFIG_REL_PATH: &str = "../game_config.toml";
/// data 根目录（gamedata/）相对于工作目录的相对路径
pub const DATA_DIR_REL_PATH: &str = "gamedata";
/// 环境变量名：覆盖 data 根目录的绝对路径
pub const ENV_DATA_DIR: &str = "UMAI_DATA_DIR";

/// 解析 data 根目录绝对路径：优先用环境变量 `UMAI_DATA_DIR`，否则用工作目录 + `gamedata/`
pub fn resolve_data_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var(ENV_DATA_DIR) {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from(DATA_DIR_REL_PATH)
    }
}

/// 解析默认配置绝对路径
pub fn resolve_default_config_path() -> std::path::PathBuf {
    resolve_data_dir().join(DEFAULT_CONFIG_REL_PATH)
}

/// 解析用户配置绝对路径
pub fn resolve_user_config_path() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_default().join(USER_CONFIG_REL_PATH)
}

/// 校验 GameConfig 关键字段（Phase 2 步骤 4：加载集中化）
///
/// 业务模块不应自行校验字段格式；统一在此处报错。当前覆盖：
/// - `scenario`：枚举合法性
/// - `trainer`：枚举合法性
/// - `cards`：长度 = 6
/// - `ramen_region_fixed`（fixed 策略时）：长度 = 1
pub fn validate_game_config(config: &GameConfig) -> Result<()> {
    match config.scenario.as_str() {
        "basic" | "onsen" | "ramen" => {}
        other => anyhow::bail!("未知 scenario={other:?}，应为 basic | onsen | ramen"),
    }
    match config.trainer.as_str() {
        "manual" | "random" | "handwritten" | "collector" | "neuralnet" | "mcts" => {}
        other => anyhow::bail!("未知 trainer={other:?}"),
    }
    if config.cards.len() != 6 {
        anyhow::bail!("cards 长度应为 6，实际 {}", config.cards.len());
    }
    if matches!(
        config.ramen_region_strategy,
        crate::gamedata::RamenRegionStrategy::Fixed
    ) {
        match &config.ramen_region_fixed {
            Some(fixed) if fixed.len() == 1 => {}
            Some(fixed) => anyhow::bail!(
                "ramen_region_strategy=fixed 但 ramen_region_fixed 长度 = {}（应为 1）",
                fixed.len()
            ),
            None => anyhow::bail!("ramen_region_strategy=fixed 但未设置 ramen_region_fixed"),
        }
    }
    Ok(())
}

/// 载入 gamedata/default_config.toml, 和 game_config.toml 合并
pub fn load_game_config() -> Result<GameConfig> {
    let def_path = resolve_default_config_path();
    info!("载入默认配置: {}", def_path.display());
    let def_file = fs_err::read_to_string(&def_path)?;
    let default_config: GameConfig = toml::from_str(&def_file)?;

    let cfg_path = resolve_user_config_path();
    let override_config: OverrideGameConfig = if cfg_path.exists() {
        info!("载入用户配置: {}", cfg_path.display());
        let cfg_file = fs_err::read_to_string(&cfg_path)?;
        toml::from_str(&cfg_file)?
    } else {
        info!(
            "用户配置不存在（{}），使用默认配置 + OverrideGameConfig 兜底",
            cfg_path.display()
        );
        OverrideGameConfig {
            onsen_order: OnsenOrder::default(),
            config_override: OverrideConfig {
                extra_count: [0; 6],
                mcts_selected_onsen: false,
                log_level: "info".to_string(), // 兜底，merge 后会被 default 覆盖
                num_threads: 0,
                mcts_turn_bonus: None,
                pt_favor_rate: None,
                race_grades: None
            },
            mcts: MctsConfig::default()
        }
    };

    let merged = override_config.merge(&default_config);
    validate_game_config(&merged)?;
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_game_config_scenario_enum() {
        let mut cfg = GameConfig::default_for_init();
        cfg.scenario = "ramen".to_string();
        assert!(validate_game_config(&cfg).is_ok());

        cfg.scenario = "bogus".to_string();
        assert!(validate_game_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_game_config_trainer_enum() {
        let mut cfg = GameConfig::default_for_init();
        cfg.trainer = "manual".to_string();
        assert!(validate_game_config(&cfg).is_ok());

        cfg.trainer = "unknown".to_string();
        assert!(validate_game_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_game_config_ramen_region_fixed_length() {
        use crate::gamedata::RamenRegionStrategy;
        let mut cfg = GameConfig::default_for_init();
        cfg.ramen_region_strategy = RamenRegionStrategy::Fixed;
        cfg.ramen_region_fixed = Some(vec![[0, 1, 2]]);
        assert!(validate_game_config(&cfg).is_ok());

        cfg.ramen_region_fixed = Some(vec![[0, 1, 2], [3, 4, 5]]); // 长度=2，应拒绝
        assert!(validate_game_config(&cfg).is_err());

        cfg.ramen_region_fixed = None;
        assert!(validate_game_config(&cfg).is_err());
    }

    #[test]
    fn test_resolve_default_config_path() {
        let p = resolve_default_config_path();
        // 默认相对路径应以 "gamedata/default_config.toml" 结尾
        assert!(p.ends_with("default_config.toml"));
    }
}
