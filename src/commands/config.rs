use std::env;
use std::fmt::Display;
use std::fs;
use std::path::PathBuf;

use crate::profile::load_profile;

pub fn handle(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let mut iter = args.into_iter();
    let subcommand = iter.next().unwrap_or_else(|| "show".to_string());
    let rest: Vec<String> = iter.collect();
    match subcommand.as_str() {
        "show" => show_profiles(rest),
        "edit" => edit_profiles(),
        "init" => init_profiles(rest),
        "validate" => validate_profiles(rest),
        other => Err(format!("Unknown config subcommand: {other}")),
    }
}

fn show_profiles(args: Vec<String>) -> Result<(), String> {
    let mut verbose = false;
    for arg in args {
        match arg.as_str() {
            "--verbose" => verbose = true,
            other => return Err(format!("Unknown flag for config show: {other}")),
        }
    }

    let profile_dir = default_profile_dir()?;
    println!("User profile directory: {}", profile_dir.display());
    list_profile_files(&profile_dir, verbose)?;

    let project_dir = project_profile_dir()?;
    if project_dir.exists() {
        println!("\nProject default profiles: {}", project_dir.display());
        list_profile_files(&project_dir, verbose)?;
    }
    Ok(())
}

fn edit_profiles() -> Result<(), String> {
    let profile_dir = default_profile_dir()?;
    println!("Profile directory: {}", profile_dir.display());
    if !profile_dir.exists() {
        println!("Directory does not exist. Create it and add .toml profiles.");
    }
    if let Ok(editor) = env::var("EDITOR") {
        println!("Hint: {} {}", editor, profile_dir.display());
    }
    Ok(())
}

fn init_profiles(args: Vec<String>) -> Result<(), String> {
    let mut force = false;
    for arg in args {
        match arg.as_str() {
            "--force" => force = true,
            other => return Err(format!("Unknown flag for config init: {other}")),
        }
    }

    let project_dir = project_profile_dir()?;
    if !project_dir.exists() {
        return Err(format!(
            "Project profiles not found at {}",
            project_dir.display()
        ));
    }

    let user_dir = default_profile_dir()?;
    fs::create_dir_all(&user_dir)
        .map_err(|err| format!("Failed to create {}: {err}", user_dir.display()))?;

    let entries = fs::read_dir(&project_dir)
        .map_err(|err| format!("Failed to read {}: {err}", project_dir.display()))?;
    let mut copied = 0usize;
    let mut skipped = 0usize;

    for entry in entries {
        let entry = entry.map_err(|err| format!("Failed to read profile entry: {err}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let filename = path.file_name().ok_or_else(|| "Invalid filename".to_string())?;
        let dest = user_dir.join(filename);
        if dest.exists() && !force {
            skipped += 1;
            continue;
        }
        fs::copy(&path, &dest)
            .map_err(|err| format!("Failed to copy {}: {err}", dest.display()))?;
        copied += 1;
    }

    println!("Initialized profiles in {}", user_dir.display());
    println!("Copied: {copied}, Skipped: {skipped}");
    Ok(())
}

#[derive(Debug)]
struct ValidationItem {
    name: String,
    status: &'static str,
    error: Option<String>,
}

fn validate_profiles(args: Vec<String>) -> Result<(), String> {
    let mut json = false;
    let mut use_user = false;
    let mut use_project = false;
    let mut dir_override: Option<PathBuf> = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--user" => use_user = true,
            "--project" => use_project = true,
            "--dir" => {
                let dir = next_value("--dir", &mut iter)?;
                dir_override = Some(PathBuf::from(dir));
            }
            other => return Err(format!("Unknown flag for config validate: {other}")),
        }
    }

    if dir_override.is_some() && (use_user || use_project) {
        return Err("Do not combine --dir with --user/--project".to_string());
    }

    let dirs = if let Some(dir) = dir_override {
        vec![dir]
    } else {
        if !use_user && !use_project {
            use_user = true;
            use_project = true;
        }
        let mut dirs = Vec::new();
        if use_user {
            dirs.push(default_profile_dir()?);
        }
        if use_project {
            dirs.push(project_profile_dir()?);
        }
        dirs
    };

    let mut results = Vec::new();
    for dir in dirs {
        if !dir.exists() {
            continue;
        }
        let entries = fs::read_dir(&dir)
            .map_err(|err| format!("Failed to read {}: {err}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("Failed to read profile entry: {err}"))?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let result = match load_profile(&path.to_string_lossy(), None) {
                Ok(_) => ValidationItem {
                    name,
                    status: "ok",
                    error: None,
                },
                Err(err) => ValidationItem {
                    name,
                    status: "error",
                    error: Some(err),
                },
            };
            results.push(result);
        }
    }

    let total = results.len();
    let failures = results.iter().filter(|r| r.status == "error").count();
    let ok_count = total.saturating_sub(failures);

    if json {
        let mut out = String::new();
        out.push('{');
        out.push_str("\"results\":[");
        for (idx, item) in results.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push('{');
            push_json_field(&mut out, "name", &item.name, true);
            push_json_field(&mut out, "status", item.status, true);
            if let Some(err) = &item.error {
                push_json_field(&mut out, "error", err, true);
            }
            if out.ends_with(',') {
                out.pop();
            }
            out.push('}');
        }
        out.push_str("],");
        out.push_str("\"summary\":{");
        push_json_field(&mut out, "total", &total.to_string(), false);
        push_json_field(&mut out, "ok", &ok_count.to_string(), false);
        push_json_field(&mut out, "failed", &failures.to_string(), false);
        if out.ends_with(',') {
            out.pop();
        }
        out.push_str("}}");
        println!("{out}");
    } else {
        if total == 0 {
            println!("No profiles found to validate.");
        } else {
            println!("Validated {total} profiles (ok {ok_count}, failed {failures}).");
        }
        if failures > 0 {
            println!("Failures:");
            for item in results.iter().filter(|r| r.status == "error") {
                let err = item.error.as_deref().unwrap_or("unknown error");
                println!(" - {}: {err}", item.name);
            }
        }
    }

    if failures > 0 {
        return Err("Profile validation failed".to_string());
    }
    Ok(())
}

fn default_profile_dir() -> Result<PathBuf, String> {
    let home = env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    Ok(PathBuf::from(home).join(".llm-manager").join("config"))
}

fn project_profile_dir() -> Result<PathBuf, String> {
    env::current_dir()
        .map(|dir| dir.join("config").join("profiles"))
        .map_err(|err| format!("Failed to get current dir: {err}"))
}

fn list_profile_files(profile_dir: &PathBuf, verbose: bool) -> Result<(), String> {
    if !profile_dir.exists() {
        println!("  (not found)");
        return Ok(());
    }

    let entries = fs::read_dir(profile_dir)
        .map_err(|err| format!("Failed to read {}: {err}", profile_dir.display()))?;
    let mut found = false;
    for entry in entries {
        let entry = entry.map_err(|err| format!("Failed to read profile entry: {err}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if verbose {
                let (profile, _) = load_profile(&path.to_string_lossy(), None)?;
                println!(
                    " - {} (runtime={}, model={}, warning_model={}, context_length={}, max_tokens={}, threads={}, gpu_layers={}, cache_type_k={}, cache_type_v={}, max_model_len={}, max_num_seqs={}, quantization={}, precision={})",
                    name,
                    format_opt(profile.runtime.as_deref()),
                    format_opt(profile.model.as_deref()),
                    format_opt(profile.warning_model.as_deref()),
                    format_opt(profile.context_length),
                    format_opt(profile.max_tokens),
                    format_opt(profile.threads),
                    format_opt(profile.gpu_layers),
                    format_opt(profile.cache_type_k.as_deref()),
                    format_opt(profile.cache_type_v.as_deref()),
                    format_opt(profile.max_model_len),
                    format_opt(profile.max_num_seqs),
                    format_opt(profile.quantization.as_deref()),
                    format_opt(profile.precision.as_deref())
                );
            } else {
                println!(" - {}", name);
            }
            found = true;
        }
    }

    if !found {
        println!("  (no .toml profiles found)");
    }
    Ok(())
}

fn format_opt<T: Display>(value: Option<T>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string())
}

fn next_value(flag: &str, iter: &mut impl Iterator<Item = String>) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("Missing value for {flag}"))
}

fn push_json_field(out: &mut String, key: &str, value: &str, quoted: bool) {
    out.push('"');
    out.push_str(&escape_json(key));
    out.push_str("\":");
    if quoted {
        out.push('"');
        out.push_str(&escape_json(value));
        out.push('"');
    } else {
        out.push_str(value);
    }
    out.push(',');
}

fn escape_json(input: &str) -> String {
    input
        .chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            other => vec![other],
        })
        .collect()
}

fn print_help() {
    println!(
        "USAGE:
  llm-manager config [show|edit|init|validate]

SUBCOMMANDS:
  show   List profile files
  edit   Print profile directory and editor hint
  init   Copy default profiles to user config dir
  validate  Validate profiles for correctness

INIT OPTIONS:
  --force   Overwrite existing profiles

SHOW OPTIONS:
  --verbose   Print profile values

VALIDATE OPTIONS:
  --json      Output JSON results
  --user      Validate user profiles only
  --project   Validate project profiles only
  --dir <path> Validate profiles in a specific directory
"
    );
}
