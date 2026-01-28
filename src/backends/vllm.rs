use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};

#[derive(Debug, Clone)]
pub struct VllmBackend {
    pub binary: String,
}

impl Default for VllmBackend {
    fn default() -> Self {
        Self {
            binary: "vllm".to_string(),
        }
    }
}

impl LLMRunner for VllmBackend {
    fn build_command(&self, _config: &RunConfig) -> LLMResult<CommandSpec> {
        Err(LLMError::UnsupportedOperation(
            "vLLM backend is not implemented yet",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::VllmBackend;
    use crate::llm_backend::{LLMError, LLMRunner, RunConfig};

    #[test]
    fn vllm_backend_unimplemented() {
        let backend = VllmBackend::default();
        let config = RunConfig {
            runtime: "vllm".to_string(),
            model: None,
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            extra_args: Vec::new(),
        };

        let err = backend.build_command(&config).unwrap_err();
        assert!(matches!(err, LLMError::UnsupportedOperation(_)));
    }
}
