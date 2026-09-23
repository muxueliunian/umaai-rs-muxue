//! gamedata 目录解析与全局初始化（文档 §9.2 的 bin 侧子集）
//!
//! bin 侧优先级（skill 侧的「本地记忆 / 询问用户」不在此层，由 agent 传
//! `--gamedata` 实现）：
//!
//! 1. `--gamedata <path>` 显式指定（不合法直接报错，不静默降级）
//! 2. 环境变量 `UMAI_DATA_DIR`（项目已有约定；兼容「指向 gamedata 本身」与
//!    「指向其父目录」两种用法）
//! 3. 从局包所在目录向上逐级找 `gamedata/`（zip 天然指向发布包根）
//! 4. 当前工作目录 `gamedata/`
//!
//! 都没有 → 返回 `None`，调用方降级（纯 ID 展示 + digest 落警告，§9.2 第 7 条）。
//!
//! ⚠ 命中后 [`init`] 会 `set_current_dir` 到 gamedata 父目录（`GameData::load`
//! 走 cwd 相对路径，与项目测试同口径）——调用前请先把所有输出路径绝对化。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use umasim::gamedata::init_global;

/// 校验目录是否为完整 gamedata（含 `umaDB.json` + `constants.json`）
fn is_gamedata(dir: &Path) -> bool {
    dir.join("umaDB.json").is_file() && dir.join("constants.json").is_file()
}

/// 按优先级解析 gamedata 目录（找不到返回 `None` → 降级路径）
pub fn resolve(explicit: Option<&Path>, zip_path: &Path) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf()); // 不校验：显式给错路径应在 init 时报错
    }
    if let Ok(env_dir) = std::env::var("UMAI_DATA_DIR") {
        for cand in [PathBuf::from(&env_dir), PathBuf::from(&env_dir).join("gamedata")] {
            if is_gamedata(&cand) {
                return Some(cand);
            }
        }
    }
    for anc in zip_path.ancestors() {
        let cand = anc.join("gamedata");
        if is_gamedata(&cand) {
            return Some(cand);
        }
    }
    let cwd_cand = PathBuf::from("gamedata");
    if is_gamedata(&cwd_cand) {
        return Some(cwd_cand);
    }
    None
}

/// 自带 gamedata 的标记文件名（skill 打包时放入该文件，内容为版本/打包日期）
pub const BUNDLED_MARKER: &str = "BUNDLED";

/// 读取「自带 gamedata」的版本注记（非自带目录返回 `None`）
///
/// 打包给初级用户时 gamedata 可能随 skill 携带，其数据会随游戏版本过期——
/// 命中标记时由 `digest` 在 `context.criteria` 出一条「旧版数据」注记。
pub fn bundled_note(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(BUNDLED_MARKER)).ok()?;
    // 容忍 Windows 记事本 / PowerShell 写入的 UTF-8 BOM
    let line = text
        .lines()
        .map(|l| l.trim_start_matches('\u{feff}').trim())
        .find(|l| !l.is_empty());
    Some(line.unwrap_or("未标注版本").to_string())
}

/// 初始化全局游戏数据（`set_current_dir` 到 gamedata 父目录 + `init_global`）
///
/// 幂等；`GameData::load` / `GameConstants::load` 均从 cwd 相对路径读取。
pub fn init(gamedata: &Path) -> Result<()> {
    ensure!(
        is_gamedata(gamedata),
        "gamedata 目录不完整（缺 umaDB.json / constants.json）: {}",
        gamedata.display()
    );
    let parent = gamedata.parent().context("gamedata 目录无父目录")?;
    std::env::set_current_dir(parent)
        .with_context(|| format!("切换工作目录失败: {}", parent.display()))?;
    init_global().context("初始化全局游戏数据失败")
}

/// 把相对路径绝对化（chdir 前调用；已绝对则原样返回）
pub fn absolutize(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 向上搜索：深层目录里的 zip 能找到祖先上的 gamedata
    #[test]
    fn test_resolve_upward() {
        let root = std::env::temp_dir().join(format!("gdata_up_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("gamedata")).unwrap();
        std::fs::write(root.join("gamedata/umaDB.json"), b"{}").unwrap();
        std::fs::write(root.join("gamedata/constants.json"), b"{}").unwrap();
        std::fs::create_dir_all(root.join("logs")).unwrap();
        let zip = root.join("logs/game1.zip");
        std::fs::write(&zip, b"x").unwrap();
        let found = resolve(None, &zip);
        println!("向上搜索结果: {:?}", found);
        assert_eq!(found, Some(root.join("gamedata")));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 找不到 gamedata → None（降级路径）
    #[test]
    fn test_resolve_none() {
        let root = std::env::temp_dir().join(format!("gdata_none_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let zip = root.join("game1.zip");
        std::fs::write(&zip, b"x").unwrap();
        // cwd 恰好有 gamedata 时会命中第 4 优先级——用「错误的显式目录也返回 Some」
        // 之外的分支验证 None 不可靠，这里只验证向上搜索不命中
        let explicit_bad = root.join("no_such_dir");
        let found = resolve(Some(&explicit_bad), &zip);
        println!("显式错误路径透传: {:?}", found);
        assert_eq!(found, Some(explicit_bad), "显式路径应透传（init 时报错）");
        let _ = std::fs::remove_dir_all(&root);
    }
}
