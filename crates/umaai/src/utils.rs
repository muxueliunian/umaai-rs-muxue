use std::{
    sync::{Mutex, OnceLock},
    time::Duration
};

use anyhow::Result;
use log::warn;
use ratatui::crossterm::{
    self,
    event::{self, Event, KeyCode, KeyEventKind}
};
use umasim::game::onsen::game::OnsenGame;
#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::GetCurrentThreadStackLimits;

use crate::protocol::GameStatusOnsen;

pub static SAVED_GAME: OnceLock<Mutex<OnsenGame>> = OnceLock::new();
pub fn handle_f2() -> Result<()> {
    if let Some(mutex) = SAVED_GAME.get() {
        let game = mutex.lock().expect("saved game");
        let status = GameStatusOnsen::from(&*game);
        let filename = format!("logs/turn{}.json", game.turn);
        warn!("保存当前回合到 {filename}");
        fs_err::write(filename, serde_json::to_string_pretty(&status)?)?;
    } else {
        warn!("游戏未开始，无法保存游戏信息");
    }
    Ok(())
}

pub async fn hotkey_handler() {
    loop {
        if let Ok(true) = crossterm::event::poll(Duration::from_millis(100)) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.code == KeyCode::F(2) && k.kind == KeyEventKind::Release {
                    let _ = handle_f2().inspect_err(|e| log::error!("保存回合信息出错: {e:?}"));
                }
            }
        }
    }
}

/// 获取当前线程的栈大小，用于调试
#[cfg(target_os = "windows")]
pub fn get_stack_size() -> usize {
    unsafe {
        let mut low_limit = 0;
        let mut high_limit = 0;
        GetCurrentThreadStackLimits(&mut low_limit, &mut high_limit);
        high_limit - low_limit // 返回栈大小
    }
}

#[cfg(target_os = "linux")]
pub fn get_stack_size() -> usize {
    let mut limits = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut limits) };
    limits.rlim_cur as usize
}
