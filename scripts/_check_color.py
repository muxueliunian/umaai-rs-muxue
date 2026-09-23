import subprocess
# 复用 cargo test 加一行 print ANSI 字节
script = '''
#[test]
fn _check_color() {
    use colored::Colorize;
    let s = format!("[回合 1] 首选: 速训练").bright_yellow().on_red().to_string();
    println!("ANSI: {:?}", s);
}
'''
print("inline test:")
