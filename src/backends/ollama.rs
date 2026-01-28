use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};
use crate::ollama_config::read_modelfile;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct OllamaBackend {
    pub binary: String,
}

impl Default for OllamaBackend {
    fn default() -> Self {
        Self {
            binary: "ollama".to_string(),
        }
    }
}

impl LLMRunner for OllamaBackend {
    fn build_command(&self, config: &RunConfig) -> LLMResult<CommandSpec> {
        let Some(model) = &config.model else {
            return Err(LLMError::MissingModel { runtime: "ollama" });
        };

        let mut args = vec!["run".to_string(), model.display().to_string()];
        args.extend(config.extra_args.clone());

        let mut env = Vec::new();
        let mut context_length = config.context_length;
        if context_length.is_none() {
            context_length = modelfile_context_length(model);
        }
        if let Some(context_length) = context_length {
            env.push((
                "OLLAMA_CONTEXT_LENGTH".to_string(),
                context_length.to_string(),
            ));
        }

        Ok(CommandSpec {
            program: self.binary.clone(),
            args,
            env,
        })
    }
}

fn modelfile_context_length(model: &PathBuf) -> Option<u32> {
    if !model.is_file() {
        return None;
    }
    read_modelfile(model).ok().and_then(|config| config.context_length())
}

#[cfg(test)]
mod tests {
    use super::OllamaBackend;
    use crate::llm_backend::{LLMRunner, RunConfig};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn builds_ollama_command() {
        let backend = OllamaBackend::default();
        let config = RunConfig {
            runtime: "ollama".to_string(),
            model: Some(PathBuf::from("mistral")),
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: vec!["--verbose".to_string()],
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(spec.program, "ollama");
        assert_eq!(spec.args, vec!["run", "mistral", "--verbose"]);
        assert!(spec.env.is_empty());
    }

    #[test]
    fn builds_ollama_command_with_context_env() {
        let backend = OllamaBackend::default();
        let config = RunConfig {
            runtime: "ollama".to_string(),
            model: Some(PathBuf::from("mistral")),
            context_length: Some(4096),
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(spec.program, "ollama");
        assert_eq!(spec.args, vec!["run", "mistral"]);
        assert_eq!(
            spec.env,
            vec![("OLLAMA_CONTEXT_LENGTH".to_string(), "4096".to_string())]
        );
    }

    #[test]
    fn uses_modelfile_context_length_when_missing_config() {
        let mut path = PathBuf::from(std::env::temp_dir());
        path.push(format!("ollama_modelfile_{}.txt", std::process::id()));
        fs::write(&path, "FROM llama3\nPARAMETER num_ctx 2048").unwrap();

        let backend = OllamaBackend::default();
        let config = RunConfig {
            runtime: "ollama".to_string(),
            model: Some(path.clone()),
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(
            spec.env,
            vec![("OLLAMA_CONTEXT_LENGTH".to_string(), "2048".to_string())]
        );

        let _ = fs::remove_file(&path);
    }
}
