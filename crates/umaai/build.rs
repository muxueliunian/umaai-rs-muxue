extern crate winscribe;

use winscribe::{ResBuilder, icon::Icon};
fn main() {
    ResBuilder::from_env()
        .unwrap()
        .push(Icon::app("res/umaai-sm.ico"))
        .compile()
        .unwrap()
}
