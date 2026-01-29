use crate::llm_backend::{CommandSpec, LLMError, LLMRunner, LLMResult, RunConfig};

#[derive(Debug, Clone)]
pub struct MlcBackend {
    pub binary: String,
}

impl Default for MlcBackend {
    fn default() -> Self {
        Self {
            binary: "mlc_llm".to_string(),
        }
    }
}

impl LLMRunner for MlcBackend {
    fn build_command(&self, _config: &RunConfig) -> LLMResult<CommandSpec> {
        Err(LLMError::UnsupportedOperation(
            "MLC-LLM backend is not implemented yet",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::MlcBackend;
    use crate::llm_backend::{LLMError, LLMRunner, RunConfig};

    #[test]
    fn mlc_backend_unimplemented() {
        let backend = MlcBackend::default();
        let config = RunConfig {
            runtime: "mlc".to_string(),
            model: None,
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

        let err = backend.build_command(&config).unwrap_err();
        assert!(matches!(err, LLMError::UnsupportedOperation(_)));
    }
}
