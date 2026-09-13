//! 无哈希资产快照：保存原始字节，续跑及合并直接比较内容。

use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};

/// 比较实际文件字节；路径与修改时间不参与内容判定。
pub fn ensure_same_bytes(left: &Path, right: &Path) -> Result<()> {
    let a = fs::read(left).with_context(|| format!("读取资产 {}", left.display()))?;
    let b = fs::read(right).with_context(|| format!("读取资产 {}", right.display()))?;
    ensure!(a == b, "资产内容不同：{} / {}", left.display(), right.display());
    Ok(())
}

/// 首次保存资产原文；续跑只允许与已保存字节完全相同的资产。
pub fn snapshot(source: &Path, target: &Path, resume: bool) -> Result<()> {
    if target.exists() {
        return ensure_same_bytes(source, target);
    }
    ensure!(!resume, "续跑缺少原始资产副本：{}", target.display());
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(source, target)?;
    ensure_same_bytes(source, target)
}

/// 核对两份数据携带的完整资产快照；不接受少文件或同大小不同内容。
pub fn compare_sets(left: &Path, right: &Path, names: &[String]) -> Result<()> {
    for name in names {
        ensure!(
            !name.is_empty() && name.split('/').all(|s| !s.is_empty() && s != "." && s != "..")
                && !name.contains('\\') && !name.contains(':'),
            "非法资产相对路径：{name}"
        );
        ensure_same_bytes(&left.join(name), &right.join(name))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use umasim::utils::get_workspace_root;

    /// 同大小内容变化必须被拒绝，复制到不同路径后的原文可以通过。
    #[test]
    fn test_asset_bytes_not_metadata() -> Result<()> {
        let dir = get_workspace_root()?.join("target").join(format!(
            "asset_bytes_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir_all(&dir)?;
        let source = dir.join("source");
        let saved = dir.join("saved");
        fs::write(&source, b"abcd")?;
        snapshot(&source, &saved, false)?;
        snapshot(&source, &saved, true)?;
        fs::write(&source, b"abce")?;
        let refused = snapshot(&source, &saved, true).is_err();
        println!("同大小不同字节拒绝={refused}");
        ensure!(refused, "同大小不同内容未被拦截");
        ensure!(snapshot(&source, &dir.join("missing"), true).is_err(), "缺资产续跑未被拦截");
        Ok(())
    }
}
