extern crate winscribe;

use winscribe::{ResBuilder, icon::Icon};
fn main() {
    println!("cargo:rustc-link-arg=/STACK:8388608");
    ResBuilder::from_env()
        .unwrap()
        .push(Icon::app("res/umaai-sm.ico"))
        .compile()
        .unwrap()
}
