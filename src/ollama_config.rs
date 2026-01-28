use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct OllamaModelConfig {
    base_model: Option<String>,
    parameters: HashMap<String, String>,
}

impl OllamaModelConfig {
    pub fn base_model(&self) -> Option<&str> {
        self.base_model.as_deref()
    }

    pub fn parameter(&self, key: &str) -> Option<&str> {
        self.parameters.get(key).map(|value| value.as_str())
    }

    pub fn context_length(&self) -> Option<u32> {
        self.parse_u32_param("num_ctx")
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.parse_u32_param("num_predict")
    }

    fn parse_u32_param(&self, key: &str) -> Option<u32> {
        self.parameter(key)
            .and_then(|value| value.trim().parse::<u32>().ok())
    }
}

pub fn parse_modelfile(contents: &str) -> OllamaModelConfig {
    let mut config = OllamaModelConfig::default();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("FROM ") {
            let base = rest.trim();
            if !base.is_empty() {
                config.base_model = Some(base.to_string());
            }
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("PARAMETER ") {
            let mut parts = rest.split_whitespace();
            let key = match parts.next() {
                Some(key) => key,
                None => continue,
            };
            let value = parts.collect::<Vec<_>>().join(" ");
            if !value.is_empty() {
                let cleaned = value.trim_matches('"').to_string();
                config.parameters.insert(key.to_string(), cleaned);
            }
        }
    }

    config
}

pub fn read_modelfile(path: &Path) -> Result<OllamaModelConfig, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read modelfile {}: {err}", path.display()))?;
    Ok(parse_modelfile(&contents))
}

#[cfg(test)]
mod tests {
    use super::{parse_modelfile, read_modelfile};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn parse_modelfile_extracts_base_and_params() {
        let contents = r#"
# comment
FROM llama3
PARAMETER num_ctx 4096
PARAMETER num_predict 256
"#;
        let config = parse_modelfile(contents);
        assert_eq!(config.base_model(), Some("llama3"));
        assert_eq!(config.parameter("num_ctx"), Some("4096"));
        assert_eq!(config.parameter("num_predict"), Some("256"));
        assert_eq!(config.context_length(), Some(4096));
        assert_eq!(config.max_tokens(), Some(256));
    }

    #[test]
    fn read_modelfile_ignores_unknown_lines() {
        let mut path = PathBuf::from(std::env::temp_dir());
        path.push(format!("ollama_modelfile_{}.txt", std::process::id()));
        fs::write(
            &path,
            "SYSTEM \"Be helpful\"\nPARAMETER num_ctx 2048\nTEMPLATE \"{{ .Prompt }}\"",
        )
        .unwrap();

        let config = read_modelfile(&path).unwrap();
        assert_eq!(config.context_length(), Some(2048));

        let _ = fs::remove_file(&path);
    }
}
