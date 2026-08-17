/// 构建脚本：Windows 目标下将 .ico 图标与 8MB 栈大小链接进二进制；
/// 其他目标（如 Linux）不执行任何资源编译。
#[cfg(windows)]
extern crate winscribe;

fn main() {
    #[cfg(windows)]
    {
        use winscribe::{ResBuilder, icon::Icon};

        println!("cargo:rustc-link-arg=/STACK:8388608");
        ResBuilder::from_env()
            .unwrap()
            .push(Icon::app("res/umaai-sm.ico"))
            .compile()
            .unwrap();
    }
}
