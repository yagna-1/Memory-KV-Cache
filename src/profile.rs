use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::llm_backend::RunConfig;

#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub runtime: Option<String>,
    pub model: Option<String>,
    pub context_length: Option<u64>,
    pub max_tokens: Option<u32>,
    pub threads: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub precision: Option<String>,
}

impl Profile {
    pub fn apply_to_run_config(&self, config: &mut RunConfig) -> Result<(), String> {
        if config.runtime.is_empty() {
            if let Some(runtime) = &self.runtime {
                config.runtime = runtime.to_string();
            }
        }

        if config.model.is_none() {
            if let Some(model) = &self.model {
                if !model.is_empty() {
                    config.model = Some(PathBuf::from(model));
                }
            }
        }

        if config.context_length.is_none() {
            if let Some(context_length) = self.context_length {
                let value: u32 = context_length
                    .try_into()
                    .map_err(|_| "context_length is too large for u32".to_string())?;
                config.context_length = Some(value);
            }
        }

        if config.max_tokens.is_none() {
            config.max_tokens = self.max_tokens;
        }

        if config.threads.is_none() {
            config.threads = self.threads;
        }

        if config.gpu_layers.is_none() {
            config.gpu_layers = self.gpu_layers;
        }

        Ok(())
    }

    pub fn validate(&mut self) -> Result<(), String> {
        if let Some(runtime) = &self.runtime {
            let norm = runtime.trim().to_lowercase();
            if norm.is_empty() {
                self.runtime = None;
            } else if !is_valid_runtime(&norm) {
                return Err(format!("Unsupported runtime in profile: {runtime}"));
            } else {
                self.runtime = Some(normalize_runtime(&norm));
            }
        }

        if let Some(model) = &self.model {
            let trimmed = model.trim();
            if trimmed.is_empty() {
                self.model = None;
            } else {
                self.model = Some(trimmed.to_string());
            }
        }

        if let Some(context_length) = self.context_length {
            if context_length == 0 {
                return Err("context_length must be > 0".to_string());
            }
        }

        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err("max_tokens must be > 0".to_string());
            }
        }

        if let Some(threads) = self.threads {
            if threads == 0 {
                return Err("threads must be > 0".to_string());
            }
        }

        if let Some(precision) = &self.precision {
            let norm = precision.trim().to_lowercase();
            if norm.is_empty() {
                self.precision = None;
            } else if !is_valid_precision(&norm) {
                return Err(format!("Unsupported precision in profile: {precision}"));
            } else {
                self.precision = Some(norm);
            }
        }

        Ok(())
    }
}

pub fn load_profile(profile: &str, profile_dir: Option<&Path>) -> Result<(Profile, PathBuf), String> {
    let path = resolve_profile_path(profile, profile_dir)?;
    let contents = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {err}", path.display()))?;
    let mut parsed = parse_profile(&contents)?;
    parsed.validate()?;
    Ok((parsed, path))
}

fn resolve_profile_path(profile: &str, profile_dir: Option<&Path>) -> Result<PathBuf, String> {
    let path = expand_path(profile)?;
    if path.is_absolute() || path.components().count() > 1 {
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("Profile file not found: {}", path.display()));
    }

    let filename = if profile.ends_with(".toml") {
        profile.to_string()
    } else {
        format!("{profile}.toml")
    };

    if let Some(dir) = profile_dir {
        let candidate = dir.join(&filename);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    if let Ok(default_dir) = default_profile_dir() {
        let candidate = default_dir.join(&filename);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    if let Ok(project_dir) = project_profile_dir() {
        let candidate = project_dir.join(&filename);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(format!("Profile '{profile}' not found in known directories"))
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

fn expand_path(input: &str) -> Result<PathBuf, String> {
    if let Some(stripped) = input.strip_prefix("~/") {
        let home = env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
        return Ok(PathBuf::from(home).join(stripped));
    }
    Ok(PathBuf::from(input))
}

fn parse_profile(contents: &str) -> Result<Profile, String> {
    let mut profile = Profile::default();
    for (idx, line) in contents.lines().enumerate() {
        let line = strip_comments(line).trim();
        if line.is_empty() {
            continue;
        }
        let (key, raw_value) = line
            .split_once('=')
            .ok_or_else(|| format!("Invalid profile line {}: {line}", idx + 1))?;
        let key = key.trim();
        let value = raw_value.trim();

        match key {
            "runtime" => profile.runtime = Some(parse_string(value)?),
            "model" => {
                let model = parse_string(value)?;
                profile.model = if model.is_empty() { None } else { Some(model) };
            }
            "context_length" => profile.context_length = Some(parse_u64(value)?),
            "max_tokens" => profile.max_tokens = Some(parse_u32(value)?),
            "threads" => profile.threads = Some(parse_u32(value)?),
            "gpu_layers" => profile.gpu_layers = Some(parse_u32(value)?),
            "precision" => profile.precision = Some(parse_string(value)?),
            _ => {}
        }
    }
    Ok(profile)
}

fn is_valid_runtime(runtime: &str) -> bool {
    matches!(runtime, "llama.cpp" | "llama" | "ollama" | "mlc" | "mlc-llm" | "vllm")
}

fn normalize_runtime(runtime: &str) -> String {
    match runtime {
        "llama" => "llama.cpp".to_string(),
        "mlc-llm" => "mlc".to_string(),
        other => other.to_string(),
    }
}

fn is_valid_precision(precision: &str) -> bool {
    matches!(precision, "fp16" | "fp32" | "int8")
}

fn strip_comments(line: &str) -> &str {
    line.split('#').next().unwrap_or("")
}

fn parse_string(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if let Some(value) = trimmed.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        Ok(unescape_string(value))
    } else {
        Ok(trimmed.to_string())
    }
}

fn unescape_string(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("Invalid numeric value: {value}"))
}

fn parse_u32(value: &str) -> Result<u32, String> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid numeric value: {value}"))
}

#[cfg(test)]
mod tests {
    use super::{parse_profile, Profile};

    #[test]
    fn parse_and_validate_profile() {
        let input = r#"
            runtime = "llama"
            model = ""
            context_length = 2048
            max_tokens = 128
            threads = 4
            gpu_layers = 2
            precision = "fp16"
        "#;
        let mut profile = parse_profile(input).unwrap();
        profile.validate().unwrap();

        assert_eq!(profile.runtime.as_deref(), Some("llama.cpp"));
        assert!(profile.model.is_none());
        assert_eq!(profile.context_length, Some(2048));
        assert_eq!(profile.max_tokens, Some(128));
        assert_eq!(profile.threads, Some(4));
        assert_eq!(profile.gpu_layers, Some(2));
        assert_eq!(profile.precision.as_deref(), Some("fp16"));
    }

    #[test]
    fn invalid_runtime_rejected() {
        let input = r#"runtime = "unknown""#;
        let mut profile = parse_profile(input).unwrap();
        assert!(profile.validate().is_err());
    }

    #[test]
    fn invalid_precision_rejected() {
        let input = r#"precision = "fp64""#;
        let mut profile = parse_profile(input).unwrap();
        assert!(profile.validate().is_err());
    }

    #[test]
    fn empty_values_cleared() {
        let input = r#"
            runtime = ""
            model = ""
            precision = ""
        "#;
        let mut profile = parse_profile(input).unwrap();
        profile.validate().unwrap();
        assert!(profile.runtime.is_none());
        assert!(profile.model.is_none());
        assert!(profile.precision.is_none());
    }
}
