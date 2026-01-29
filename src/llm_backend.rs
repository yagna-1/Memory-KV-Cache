use std::path::PathBuf;
use std::process::Command;
use std::{error::Error, fmt};

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub runtime: String,
    pub model: Option<PathBuf>,
    pub context_length: Option<u32>,
    pub max_tokens: Option<u32>,
    pub threads: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub max_model_len: Option<u32>,
    pub max_num_seqs: Option<u32>,
    pub quantization: Option<String>,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AdjustmentParams {
    pub context_length: Option<u32>,
    pub max_tokens: Option<u32>,
    pub threads: Option<u32>,
    pub gpu_layers: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub enum LLMError {
    MissingModel { runtime: &'static str },
    LaunchFailed { program: String, source: String },
    ProcessExit { code: i32 },
    UnsupportedOperation(&'static str),
    InvalidConfig(String),
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::MissingModel { runtime } => {
                write!(f, "Missing --model for {runtime} runtime. Provide --model <path>")
            }
            LLMError::LaunchFailed { program, source } => {
                write!(f, "Failed to start {program}: {source}")
            }
            LLMError::ProcessExit { code } => write!(f, "Process exited with status {code}"),
            LLMError::UnsupportedOperation(msg) => write!(f, "{msg}"),
            LLMError::InvalidConfig(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for LLMError {}

pub type LLMResult<T> = Result<T, LLMError>;

pub trait LLMRunner {
    fn build_command(&self, config: &RunConfig) -> LLMResult<CommandSpec>;

    fn run(&self, config: &RunConfig) -> LLMResult<()> {
        let spec = self.build_command(config)?;
        let mut command = Command::new(&spec.program);
        command.args(&spec.args);
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        let status = command.status().map_err(|err| LLMError::LaunchFailed {
            program: spec.program.clone(),
            source: err.to_string(),
        })?;
        if !status.success() {
            return Err(LLMError::ProcessExit {
                code: status.code().unwrap_or(-1),
            });
        }
        Ok(())
    }

    fn stop(&self) -> LLMResult<()> {
        Err(LLMError::UnsupportedOperation(
            "Stop not implemented for this backend yet",
        ))
    }

    fn adjust_params(&self, _params: &AdjustmentParams) -> LLMResult<()> {
        Err(LLMError::UnsupportedOperation(
            "Dynamic adjustment not implemented for this backend yet",
        ))
    }
}
