//! 本局游戏记录打包：把 `logs/game{id}/` 打成 `logs/game{id}.zip` 后清理原目录
//!
//! 触发点：在线记录器在末回合第 2 份快照（拉面 `turn77_2`，含决策行那份）
//! 处理完、写完 `meta.json` + 渲染完 `luck_trend.svg` 之后调用。打包**成功**
//! 才删原目录（失败保留原目录以便人工排查 / 重打）。
//!
//! zip 文件命名：与原目录同名换后缀 → `logs/game{id}.zip`（与 `logs/` 同级，
//! 不嵌套任何子目录）。
//!
//! 包内条目路径使用**相对于原目录**的路径（即 `decisions.csv` / `meta.json` /
//! `luck_trend.svg` / `game{id}_turn{N}_2.json` 等直接落在根），不保留 `game{id}/`
//! 外壳——解压即得全部局内产物。
//!
//! 错误处理：任一步失败（zip 创建 / 写入 / 重命名 / 删目录）都返回 `Err`，
//! 调用方记录 `warn!` 日志即可，不影响后续局、不抛错给 main。

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use dunce::canonicalize;
use log::warn;
use zip::write::SimpleFileOptions;

/// 把 `dir` 打成 `{dir_parent}/{dir_name}.zip` 后清理原目录
///
/// - `dir` 必须是一个**目录**（不存在 / 是文件都返回 `Err`）
/// - zip 包内条目使用 `Path::strip_prefix(dir).unwrap()` 后的相对路径
///   （与 zip 协议语义一致，去掉 `game{id}/` 外壳）
/// - 打包成功后才删原目录；任一步失败保留原目录
///
/// 返回：zip 文件的最终路径（绝对或相对，与入参 `dir` 语义一致）。
pub fn zip_and_cleanup(dir: &Path) -> Result<PathBuf> {
    if !dir.exists() {
        anyhow::bail!("本局游戏记录目录不存在: {}", dir.display());
    }
    if !dir.is_dir() {
        anyhow::bail!("局路径不是目录: {}", dir.display());
    }
    let Some(parent) = dir.parent() else {
        anyhow::bail!("本局游戏记录目录没有父目录: {}", dir.display());
    };
    let Some(dir_name) = dir.file_name() else {
        anyhow::bail!("本局游戏记录目录无法解析文件名: {}", dir.display());
    };

    // zip 临时名：先写到 `.partial` 后改名，避免打包过程中崩溃导致同名 zip 半成品。
    // 但 zip crate 的 write 也支持边写边落盘到最终路径——我们用「写临时 + 改名」
    // 保证只有完整 zip 才出现在 logs/ 下。
    let final_path = parent.join(format!("{}.zip", dir_name.to_string_lossy()));
    let tmp_path = parent.join(format!("{}.zip.partial", dir_name.to_string_lossy()));

    // 把目录内所有条目打包进 zip
    let zip_bytes = build_zip(dir).with_context(|| format!("打包 zip 失败: {}", dir.display()))?;

    // 写临时文件
    {
        let mut f = fs::File::create(&tmp_path)
            .with_context(|| format!("创建临时 zip 失败: {}", tmp_path.display()))?;
        f.write_all(&zip_bytes)
            .with_context(|| format!("写入临时 zip 失败: {}", tmp_path.display()))?;
        f.sync_all().ok();
    }

    // 原子改名到最终路径
    if let Err(e) = fs::rename(&tmp_path, &final_path) {
        // rename 在 Windows 上跨卷会失败——此时退化为 copy + remove
        warn!(
            "rename {} → {} 失败 ({}); 退化 copy + remove",
            tmp_path.display(),
            final_path.display(),
            e
        );
        fs::copy(&tmp_path, &final_path)
            .with_context(|| format!("copy 临时 zip 失败: {}", tmp_path.display()))?;
        let _ = fs::remove_file(&tmp_path);
    }

    // zip 完整落盘后才删原目录（删失败仅 warn，不让调用方误判）。
    // 删除前**路径白名单校验**：只允许删除 `logs/{game*}` 这类本局游戏记录目录，
    // 防止误传路径把无关目录干掉。
    verify_deletion_path(dir)?;
    if let Err(e) = fs::remove_dir_all(dir) {
        warn!(
            "zip 已生成但清理原目录失败 {}: {e:?}（zip 路径 {} 可正常使用）",
            dir.display(),
            final_path.display()
        );
    }

    Ok(final_path)
}

/// 路径白名单校验：仅允许删除 `logs/{game*}` 这类本局游戏记录目录
fn verify_deletion_path(dir: &Path) -> Result<()> {
    // canonicalize：dunce 在 Windows 上去掉 `\\?\` 前缀，跨平台稳定；
    // canonicalize 本身要求路径必须存在，解析失败 → 直接拒绝。
    let abs = canonicalize(dir).map_err(|_| anyhow::anyhow!("删除路径错误：{}", dir.display()))?;

    let parent_name = abs
        .parent()
        .and_then(|p| p.file_name())
        .ok_or_else(|| anyhow::anyhow!("删除路径错误：{}", dir.display()))?;
    if parent_name != "logs" {
        anyhow::bail!("删除路径错误：{}", dir.display());
    }
    let dir_name = abs
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("删除路径错误：{}", dir.display()))?
        .to_string_lossy();
    if !dir_name.starts_with("game") {
        anyhow::bail!("删除路径错误：{}", dir.display());
    }
    Ok(())
}

/// 收集 `dir` 下所有条目（递归）并构造一个内存里的 zip，返回字节流
///
/// **设计取舍**：拉面单局产物总大小 ~1MB（issues.md 已实测），全量内存构建
/// 安全且最简单；不上 streaming 是为了避免「中途异常 → zip 半成品 + 原目录
/// 已被删」的灾难组合。
fn build_zip(dir: &Path) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(64 * 1024);
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut writer = zip::ZipWriter::new(cursor);
        // 不存 mtime / 不压缩 zip 自身的 metadata；只压内容
        // （普通文本/JSON/SVG 都是 STORE 友好类型，DEFLATE 也几乎无差；这里选 DEFLATE 体积更小）
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        let entries = collect_entries(dir).with_context(|| format!("遍历目录失败: {}", dir.display()))?;
        for entry in entries {
            let rel = entry
                .strip_prefix(dir)
                .unwrap_or(&entry)
                .to_string_lossy()
                .replace('\\', "/"); // zip 协议统一用 /
            let name = rel.trim_start_matches('/');
            if entry.is_dir() {
                // 空目录也建一条（带斜杠）；拉面本局游戏记录目录里 meta/decisions/svg 等都非空，
                // 但保险起见保留空目录条目
                writer
                    .add_directory(name, opts)
                    .with_context(|| format!("添加目录条目失败: {name}"))?;
            } else {
                writer
                    .start_file(name, opts)
                    .with_context(|| format!("开始 zip 条目失败: {name}"))?;
                let data = fs::read(&entry)
                    .with_context(|| format!("读取文件失败: {}", entry.display()))?;
                writer
                    .write_all(&data)
                    .with_context(|| format!("写入 zip 条目失败: {name}"))?;
            }
        }

        writer
            .finish()
            .with_context(|| "关闭 zip writer 失败")?;
    }
    Ok(buf)
}

/// 递归收集目录下所有条目（目录 + 文件），不含 `.` / `..`
///
/// 排序：按字典序 → 同一目录两次打包字节稳定，便于调试 / diff
fn collect_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir 失败: {}", dir.display()))? {
        let entry = entry.with_context(|| format!("读目录项失败: {}", dir.display()))?;
        let path = entry.path();
        out.push(path.clone());
        if path.is_dir() {
            walk(&path, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// 构造一个临时本局游戏记录目录（含 3 个文件 + 1 个子目录/子文件）
    fn make_sample_dir(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("meta.json"), "{\"end_reason\":\"game_end\"}").unwrap();
        fs::write(dir.join("decisions.csv"), "col1,col2\n1,2\n").unwrap();
        fs::write(dir.join("luck_trend.svg"), "<svg></svg>").unwrap();
        let sub = dir.join("nested");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("game1_turn77_2.json"), "{}").unwrap();
        dir
    }

    #[test]
    fn test_zip_and_cleanup_happy_path() {
        // 路径白名单要求父目录名为 logs，所以测试根目录必须以 logs 收尾
        let root = std::env::temp_dir()
            .join(format!("umaai_zip_test_{}", std::process::id()))
            .join("logs");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let dir = make_sample_dir(&root, "game7075");

        let zip_path = zip_and_cleanup(&dir).expect("应成功打包");

        // zip 路径 = parent/game7075.zip
        assert_eq!(zip_path, root.join("game7075.zip"));
        assert!(zip_path.exists(), "zip 文件应存在");
        assert!(!dir.exists(), "原目录应已被清理");

        // 验证 zip 内容：4 个文件条目（meta.json / decisions.csv / luck_trend.svg /
        // nested/game1_turn77_2.json），无 `game7075/` 外壳
        let f = fs::File::open(&zip_path).unwrap();
        let mut zip = zip::ZipArchive::new(f).unwrap();
        let mut names: Vec<String> = (0..zip.len())
            .map(|i| zip.by_index(i).unwrap().name().to_string())
            .collect();
        names.sort();
        println!("zip 包内条目: {names:?}");
        assert!(names.contains(&"meta.json".to_string()));
        assert!(names.contains(&"decisions.csv".to_string()));
        assert!(names.contains(&"luck_trend.svg".to_string()));
        assert!(names.contains(&"nested/game1_turn77_2.json".to_string()));
        // 不应出现 game7075/ 前缀
        assert!(
            !names.iter().any(|n| n.starts_with("game7075/")),
            "zip 内条目不应带 game7075/ 外壳：{names:?}"
        );

        // 验证内容字节一致
        let mut meta_entry = zip.by_name("meta.json").unwrap();
        let mut s = String::new();
        std::io::Read::read_to_string(&mut meta_entry, &mut s).unwrap();
        assert_eq!(s, "{\"end_reason\":\"game_end\"}");

        // 清理测试根（root 的父目录）
        let _ = fs::remove_dir_all(root.parent().unwrap());
    }

    #[test]
    fn test_zip_and_cleanup_preserves_dir_on_failure() {
        // 路径白名单要求父目录名为 logs
        let root = std::env::temp_dir()
            .join(format!("umaai_zip_miss_{}", std::process::id()))
            .join("logs");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        // 不存在的目录 → 早返回 Err（在白名单校验之前）
        let missing = root.join("game_does_not_exist");
        let err = zip_and_cleanup(&missing);
        assert!(err.is_err(), "不存在目录应返回 Err");
        assert!(!missing.exists(), "原目录仍未创建");

        // 文件而非目录 → Err
        let file_path = root.join("a_file");
        fs::write(&file_path, b"x").unwrap();
        let err = zip_and_cleanup(&file_path);
        assert!(err.is_err(), "路径是文件应返回 Err");
        assert!(file_path.exists(), "原文件应未被删");

        // 清理测试根
        let _ = fs::remove_dir_all(root.parent().unwrap());
    }
}
