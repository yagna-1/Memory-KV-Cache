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
                    " - {} (runtime={}, model={}, context_length={}, max_tokens={}, threads={}, gpu_layers={}, precision={})",
                    name,
                    format_opt(profile.runtime.as_deref()),
                    format_opt(profile.model.as_deref()),
                    format_opt(profile.context_length),
                    format_opt(profile.max_tokens),
                    format_opt(profile.threads),
                    format_opt(profile.gpu_layers),
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

fn print_help() {
    println!(
        "USAGE:
  llm-manager config [show|edit|init]

SUBCOMMANDS:
  show   List profile files
  edit   Print profile directory and editor hint
  init   Copy default profiles to user config dir

INIT OPTIONS:
  --force   Overwrite existing profiles

SHOW OPTIONS:
  --verbose   Print profile values
"
    );
}
