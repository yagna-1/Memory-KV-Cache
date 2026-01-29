use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> String {
    env!("CARGO_BIN_EXE_llm-manager").to_string()
}

fn workspace_root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut path = std::env::temp_dir();
    path.push(format!("{prefix}_{}_{}.{}", std::process::id(), nanos, ext));
    path
}

fn run_cmd(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(args)
        .output()
        .unwrap();
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn write_u32(file: &mut File, value: u32) {
    file.write_all(&value.to_le_bytes()).unwrap();
}

fn write_u64(file: &mut File, value: u64) {
    file.write_all(&value.to_le_bytes()).unwrap();
}

fn write_string(file: &mut File, value: &str) {
    write_u32(file, value.len() as u32);
    file.write_all(value.as_bytes()).unwrap();
}

fn write_minimal_gguf(path: &PathBuf) {
    let mut file = File::create(path).unwrap();

    file.write_all(b"GGUF").unwrap();
    write_u32(&mut file, 2); // version
    write_u64(&mut file, 0); // tensor count
    write_u64(&mut file, 5); // kv count

    write_string(&mut file, "llama.n_layer");
    write_u32(&mut file, 4); // U32
    write_u32(&mut file, 32);

    write_string(&mut file, "llama.hidden_size");
    write_u32(&mut file, 4); // U32
    write_u32(&mut file, 4096);

    write_string(&mut file, "llama.n_head");
    write_u32(&mut file, 4); // U32
    write_u32(&mut file, 32);

    write_string(&mut file, "llama.n_head_kv");
    write_u32(&mut file, 4); // U32
    write_u32(&mut file, 8);

    write_string(&mut file, "general.architecture");
    write_u32(&mut file, 8); // string
    write_string(&mut file, "llama");
}

#[test]
fn run_context_length_zero_fails() {
    let (ok, _stdout, stderr) = run_cmd(&[
        "run",
        "--model",
        "model.gguf",
        "--context-length",
        "0",
        "--dry-run",
    ]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));
}

#[test]
fn run_threads_zero_fails() {
    let (ok, _stdout, stderr) =
        run_cmd(&["run", "--model", "model.gguf", "--threads", "0", "--dry-run"]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));
}

#[test]
fn run_monitor_interval_zero_fails() {
    let (ok, _stdout, stderr) = run_cmd(&[
        "run",
        "--model",
        "model.gguf",
        "--monitor-interval-ms",
        "0",
        "--dry-run",
    ]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));
}

#[test]
fn run_monitor_warning_zero_fails() {
    let (ok, _stdout, stderr) = run_cmd(&[
        "run",
        "--model",
        "model.gguf",
        "--monitor-warning",
        "0",
        "--dry-run",
    ]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));
}

#[test]
fn run_monitor_unknown_unit_fails() {
    let (ok, _stdout, stderr) = run_cmd(&[
        "run",
        "--model",
        "model.gguf",
        "--monitor-warning",
        "1xb",
        "--dry-run",
    ]);
    assert!(!ok);
    assert!(stderr.contains("Unknown unit"));
}

#[test]
fn monitor_interval_zero_fails() {
    let (ok, _stdout, stderr) = run_cmd(&["monitor", "--interval-ms", "0"]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));
}

#[test]
fn kv_inspect_context_step_zero_fails() {
    let model_path = unique_temp_path("kv_inspect_validation", "gguf");
    write_minimal_gguf(&model_path);

    let (ok, _stdout, stderr) = run_cmd(&[
        "kv-inspect",
        "--model",
        model_path.to_str().unwrap(),
        "--context-step",
        "0",
    ]);
    assert!(!ok);
    assert!(stderr.contains("must be > 0"));

    let _ = fs::remove_file(&model_path);
}

#[test]
fn kv_inspect_invalid_precision_fails() {
    let model_path = unique_temp_path("kv_inspect_precision_invalid", "gguf");
    write_minimal_gguf(&model_path);

    let (ok, _stdout, stderr) = run_cmd(&[
        "kv-inspect",
        "--model",
        model_path.to_str().unwrap(),
        "--precision",
        "fp64",
    ]);
    assert!(!ok);
    assert!(stderr.contains("Unsupported precision"));

    let _ = fs::remove_file(&model_path);
}
