use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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
fn run_missing_runtime_executable_returns_error() {
    let temp_dir = unique_temp_dir("llm_manager_empty_path");
    fs::create_dir_all(&temp_dir).unwrap();

    let output = Command::new(bin())
        .current_dir(workspace_root())
        .env("PATH", &temp_dir)
        .args(["run", "--runtime", "llama.cpp", "--model", "model.gguf"])
        .output()
        .unwrap();

    assert!(!output.status.success());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn kv_inspect_missing_model_fails() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(["kv-inspect", "--model", "missing.gguf"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to open"));
}

#[test]
fn config_validate_invalid_profile_syntax_fails() {
    let temp_dir = unique_temp_dir("llm_manager_invalid_profile");
    fs::create_dir_all(&temp_dir).unwrap();
    let bad_path = temp_dir.join("bad.toml");
    fs::write(&bad_path, "threads 4").unwrap();

    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(["config", "validate", "--dir"])
        .arg(&temp_dir)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Invalid profile line") || stdout.contains("Profile validation failed")
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn monitor_rejects_critical_below_warning() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(["monitor", "--warning", "2", "--critical", "1"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Critical threshold must be >= warning threshold"));
}

#[test]
#[cfg(unix)]
fn config_init_fails_when_home_unwritable() {
    let temp_dir = unique_temp_dir("llm_manager_ro_home");
    fs::create_dir_all(&temp_dir).unwrap();
    let mut perms = fs::metadata(&temp_dir).unwrap().permissions();
    perms.set_mode(0o500);
    fs::set_permissions(&temp_dir, perms).unwrap();

    let output = Command::new(bin())
        .current_dir(workspace_root())
        .env("HOME", &temp_dir)
        .args(["config", "init"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to create"));

    let mut restore = fs::metadata(&temp_dir).unwrap().permissions();
    restore.set_mode(0o700);
    let _ = fs::set_permissions(&temp_dir, restore);
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
#[cfg(unix)]
fn monitor_fails_on_full_disk_log() {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args([
            "monitor",
            "--interval-ms",
            "10",
            "--warning",
            "1",
            "--critical",
            "2",
            "--log-file",
            "/dev/full",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Failed to write log") || stderr.contains("No space"));
}

#[test]
#[cfg(unix)]
fn run_with_mock_llama_cli_succeeds() {
    let temp_dir = unique_temp_dir("llm_manager_mock_llama");
    fs::create_dir_all(&temp_dir).unwrap();
    let script_path = temp_dir.join("llama-cli");
    fs::write(&script_path, "#!/bin/sh\necho ready\nexit 0\n").unwrap();
    let mut perms = fs::metadata(&script_path).unwrap().permissions();
    perms.set_mode(0o700);
    fs::set_permissions(&script_path, perms).unwrap();

    let output = Command::new(bin())
        .current_dir(workspace_root())
        .env("PATH", &temp_dir)
        .args([
            "run",
            "--runtime",
            "llama.cpp",
            "--model",
            "model.gguf",
            "--capture-output",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());

    let _ = fs::remove_dir_all(&temp_dir);
}
