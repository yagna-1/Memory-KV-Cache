use std::fs;
use std::io::Write;
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
fn config_show_verbose_includes_cache_types() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "show", "--verbose"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cache_type_k="));
    assert!(stdout.contains("cache_type_v="));
    assert!(stdout.contains("warning_model="));
}

#[test]
fn config_validate_project_profiles() {
    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "validate", "--project"])
        .status()
        .unwrap();

    assert!(status.success());
}

#[test]
fn config_validate_dir_failure() {
    let temp_dir = unique_temp_dir("llm_manager_validate");
    fs::create_dir_all(&temp_dir).unwrap();
    let bad_path = temp_dir.join("bad.toml");
    let mut file = fs::File::create(&bad_path).unwrap();
    writeln!(file, "threads = 0").unwrap();

    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "validate", "--dir"])
        .arg(&temp_dir)
        .status()
        .unwrap();

    assert!(!status.success());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn config_validate_json_output() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "validate", "--project", "--json"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"results\""));
    assert!(stdout.contains("\"summary\""));
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
fn run_dry_run_includes_cache_types() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args([
            "run",
            "--model",
            "model.gguf",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q4_0",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--cache-type-k q8_0"));
    assert!(stdout.contains("--cache-type-v q4_0"));
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
    let lower = contents.to_lowercase();
    assert!(lower.contains("warning") || lower.contains("critical"));

    let _ = fs::remove_dir_all(&temp_dir);
}
