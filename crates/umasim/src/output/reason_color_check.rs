// 临时验证 colored 输出格式
use colored::Colorize;
#[test]
fn _tmp_color_dump() {
    colored::control::set_override(true);
    let s = format!("[回合 1] 首选: X").bright_yellow().on_red().to_string();
    println!("ANSI: {:?}", s);
    panic!("done");
}
