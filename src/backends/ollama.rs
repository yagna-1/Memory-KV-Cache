use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};

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

        Ok(CommandSpec {
            program: self.binary.clone(),
            args,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::OllamaBackend;
    use crate::llm_backend::RunConfig;
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
            extra_args: vec!["--verbose".to_string()],
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(spec.program, "ollama");
        assert_eq!(spec.args, vec!["run", "mistral", "--verbose"]);
    }
}
