use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};

#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    pub binary: String,
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self {
            binary: "llama-cli".to_string(),
        }
    }
}

impl LLMRunner for LlamaCppBackend {
    fn build_command(&self, config: &RunConfig) -> LLMResult<CommandSpec> {
        let mut args = Vec::new();

        let Some(model) = &config.model else {
            return Err(LLMError::MissingModel {
                runtime: "llama.cpp",
            });
        };

        args.push("-m".to_string());
        args.push(model.display().to_string());

        if let Some(max_tokens) = config.max_tokens {
            args.push("-n".to_string());
            args.push(max_tokens.to_string());
        }

        if let Some(context_length) = config.context_length {
            args.push("-c".to_string());
            args.push(context_length.to_string());
        }

        if let Some(threads) = config.threads {
            args.push("--threads".to_string());
            args.push(threads.to_string());
        }

        if let Some(gpu_layers) = config.gpu_layers {
            args.push("--gpu-layers".to_string());
            args.push(gpu_layers.to_string());
        }

        if let Some(cache_type_k) = &config.cache_type_k {
            args.push("--cache-type-k".to_string());
            args.push(cache_type_k.to_string());
        }

        if let Some(cache_type_v) = &config.cache_type_v {
            args.push("--cache-type-v".to_string());
            args.push(cache_type_v.to_string());
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
    use super::LlamaCppBackend;
    use crate::llm_backend::{LLMRunner, RunConfig};
    use std::path::PathBuf;

    #[test]
    fn builds_llama_command() {
        let backend = LlamaCppBackend::default();
        let config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: Some(1024),
            max_tokens: Some(128),
            threads: Some(4),
            gpu_layers: Some(10),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q4_0".to_string()),
            max_model_len: None,
            max_num_seqs: None,
            quantization: None,
            extra_args: vec!["--foo".to_string()],
        };

        let spec = backend.build_command(&config).unwrap();
        assert_eq!(spec.program, "llama-cli");
        assert_eq!(
            spec.args,
            vec![
                "-m",
                "model.gguf",
                "-n",
                "128",
                "-c",
                "1024",
                "--threads",
                "4",
                "--gpu-layers",
                "10",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q4_0",
                "--foo",
            ]
        );
    }

    #[test]
    fn omits_cache_type_flags_when_unset() {
        let backend = LlamaCppBackend::default();
        let config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            max_model_len: None,
            max_num_seqs: None,
            quantization: None,
            extra_args: Vec::new(),
        };

        let spec = backend.build_command(&config).unwrap();
        assert!(!spec.args.contains(&"--cache-type-k".to_string()));
        assert!(!spec.args.contains(&"--cache-type-v".to_string()));
    }
}
