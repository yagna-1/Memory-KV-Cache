use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};

#[derive(Debug, Clone)]
pub struct VllmBackend {
    pub binary: String,
}

impl Default for VllmBackend {
    fn default() -> Self {
        Self {
            binary: "python".to_string(),
        }
    }
}

impl LLMRunner for VllmBackend {
    fn build_command(&self, config: &RunConfig) -> LLMResult<CommandSpec> {
        let Some(model) = &config.model else {
            return Err(LLMError::MissingModel { runtime: "vllm" });
        };

        let mut args = vec![
            "-m".to_string(),
            "vllm.entrypoints.openai.api_server".to_string(),
            "--model".to_string(),
            model.display().to_string(),
        ];

        let max_model_len = config.max_model_len.or(config.context_length);
        if let Some(value) = max_model_len {
            args.push("--max-model-len".to_string());
            args.push(value.to_string());
        }
        if let Some(value) = config.max_num_seqs {
            args.push("--max-num-seqs".to_string());
            args.push(value.to_string());
        }
        if let Some(value) = &config.quantization {
            if !value.trim().is_empty() {
                args.push("--quantization".to_string());
                args.push(value.to_string());
            }
        }

        args.extend(config.extra_args.clone());

        Ok(CommandSpec {
            program: self.binary.clone(),
            args,
            env: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::VllmBackend;
    use crate::llm_backend::{LLMRunner, RunConfig};

    #[test]
    fn vllm_backend_builds_command() {
        let backend = VllmBackend::default();
        let config = RunConfig {
            runtime: "vllm".to_string(),
            model: Some(std::path::PathBuf::from("mistral")),
            context_length: Some(4096),
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            max_model_len: None,
            max_num_seqs: Some(8),
            quantization: Some("awq".to_string()),
            extra_args: Vec::new(),
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(spec.program, "python");
        assert!(spec.args.contains(&"--model".to_string()));
        assert!(spec.args.contains(&"mistral".to_string()));
        assert!(spec.args.contains(&"--max-model-len".to_string()));
        assert!(spec.args.contains(&"4096".to_string()));
        assert!(spec.args.contains(&"--max-num-seqs".to_string()));
        assert!(spec.args.contains(&"8".to_string()));
        assert!(spec.args.contains(&"--quantization".to_string()));
        assert!(spec.args.contains(&"awq".to_string()));
    }
}
