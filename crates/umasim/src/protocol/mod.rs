use std::{env, path::Path, sync::mpsc};

use anyhow::Result;
use notify::{Event, RecursiveMode, Watcher};

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_urafile() -> Result<()> {
        let local_app_path = env::var("LOCALAPPDATA")?;
        let urafile_path = format!("{local_app_path}/UmamusumeResponseAnalyzer/PluginData/SendGameStatusPlugin/");

        let (tx, rx) = mpsc::channel::<notify::Result<Event>>();
        let mut watcher = notify::recommended_watcher(tx)?;
        println!("{urafile_path}");
        watcher.watch(Path::new(&urafile_path), RecursiveMode::NonRecursive)?;
        loop {
            let event = rx.recv()??;
            println!("{event:?}");
        }
        Ok(())
    }
}
