use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn bin() -> String {
    env!("CARGO_BIN_EXE_llm-manager").to_string()
}

fn workspace_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut dir = std::env::temp_dir();
    dir.push(format!("{prefix}_{}_{}", std::process::id(), nanos));
    dir
}

#[test]
fn config_init_copies_profiles() {
    let home_dir = unique_temp_dir("llm_manager_home");
    fs::create_dir_all(&home_dir).unwrap();

    let status = Command::new(bin())
        .current_dir(workspace_root())
        .env("HOME", &home_dir)
        .args(["config", "init", "--force"])
        .status()
        .unwrap();

    assert!(status.success());

    let profile = home_dir
        .join(".llm-manager")
        .join("config")
        .join("llama_cpp_safe.toml");
    assert!(profile.exists());

    let _ = fs::remove_dir_all(&home_dir);
}

#[test]
fn config_show_verbose_runs() {
    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "show", "--verbose"])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn run_dry_run_with_profile() {
    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args([
            "run",
            "--profile",
            "llama_cpp_safe",
            "--model",
            "model.gguf",
            "--dry-run",
        ])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn kv_inspect_help_runs() {
    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args(["kv-inspect", "--help"])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn monitor_writes_log_file() {
    let temp_dir = unique_temp_dir("llm_manager_monitor");
    fs::create_dir_all(&temp_dir).unwrap();
    let log_path = temp_dir.join("monitor.log");

    let mut child = Command::new(bin())
        .current_dir(workspace_root())
        .args([
            "monitor",
            "--interval-ms",
            "50",
            "--warning",
            "1",
            "--critical",
            "2",
            "--log-file",
        ])
        .arg(&log_path)
        .spawn()
        .unwrap();

    std::thread::sleep(Duration::from_millis(200));
    let _ = child.kill();
    let _ = child.wait();

    let contents = fs::read_to_string(&log_path).unwrap_or_default();
    assert!(!contents.trim().is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}
