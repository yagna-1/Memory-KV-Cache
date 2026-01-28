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

fn run_cli(args: &[&str]) -> String {
    let output = Command::new(bin())
        .current_dir(workspace_root())
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn kv_inspect_json_contains_expected_fields() {
    let model_path = unique_temp_path("kv_inspect", "gguf");
    write_minimal_gguf(&model_path);

    let stdout = run_cli(&[
        "kv-inspect",
        "--model",
        model_path.to_str().unwrap(),
        "--context-length",
        "128",
        "--precision",
        "fp16",
        "--available-ram",
        "1gb",
        "--json",
    ]);

    assert!(stdout.contains("\"hidden_size\":4096"));
    assert!(stdout.contains("\"num_layers\":32"));
    assert!(stdout.contains("\"head_count\":32"));
    assert!(stdout.contains("\"head_count_kv\":8"));
    assert!(stdout.contains("\"precision\":\"fp16\""));
    assert!(stdout.contains("\"available_ram_bytes\":1073741824"));
    assert!(stdout.contains("\"max_tokens_estimate\""));

    let _ = fs::remove_file(&model_path);
}

#[test]
fn kv_inspect_json_includes_charts() {
    let model_path = unique_temp_path("kv_inspect_chart", "gguf");
    write_minimal_gguf(&model_path);

    let stdout = run_cli(&[
        "kv-inspect",
        "--model",
        model_path.to_str().unwrap(),
        "--context-length",
        "128",
        "--precision",
        "fp16",
        "--available-ram",
        "1gb",
        "--context-chart",
        "--layer-chart",
        "--json",
    ]);

    assert!(stdout.contains("\"context_chart\""));
    assert!(stdout.contains("\"layer_chart\""));

    let _ = fs::remove_file(&model_path);
}

#[test]
fn run_dry_run_with_warning_flags() {
    let status = Command::new(bin())
        .current_dir(workspace_root())
        .args([
            "run",
            "--model",
            "model.gguf",
            "--warning-context-length",
            "1024",
            "--warning-context-step",
            "256",
            "--warning-context-min",
            "512",
            "--warning-max-tokens",
            "128",
            "--warning-threads",
            "4",
            "--warning-gpu-layers",
            "8",
            "--warning-model",
            "smaller.gguf",
            "--dry-run",
        ])
        .status()
        .unwrap();

    assert!(status.success());
}
