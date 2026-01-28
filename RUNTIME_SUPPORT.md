# Runtime Support Notes

This document captures current runtime research and integration decisions.

## Ollama notes
- CLI usage centers on `ollama run <model>` for inference.
- Models can be inspected with `ollama show --json <model>` (used for metadata).
- Common CLI helpers: `ollama list`, `ollama pull`, and `ollama rm`.
- Local Modelfiles define base models and parameters, e.g.:
  - `FROM <model>`
  - `PARAMETER num_ctx <n>`
  - `PARAMETER num_predict <n>`
- Context length is applied via `OLLAMA_CONTEXT_LENGTH`.
- The HTTP API can be used for programmatic control, but is not integrated yet.
  - Keep CLI integration as the default until API semantics are stabilized.

### Decision
Continue using the CLI for now. Modelfile parsing provides context defaults when
`--context-length` is not supplied.

### RunConfig mapping
Ollama uses a small subset of the shared configuration:
- `model` -> `ollama run <model>`
- `context_length` -> `OLLAMA_CONTEXT_LENGTH`
- `max_tokens` is not wired yet (future `num_predict` mapping).
- `extra_args` are appended to the `ollama run` invocation.

### Open questions
- Confirm how `num_predict` interacts with `--max-tokens` for long prompts.
- Evaluate whether `ollama run` exposes a stable JSON progress stream.
- Determine the best way to detect model-specific defaults from `ollama show`.

## llama.cpp integration choices
The current approach uses the `llama-cli` subprocess rather than Rust bindings.

### Why subprocess for now
- Keeps the build dependency-free while `cargo` access is limited.
- CLI arguments map directly to tuning flags (`-c`, `-n`, `--threads`, etc.).
- Subprocess isolation makes restarts and monitoring straightforward.
- `llama-cli` output parsing already supports progress and log capture.

### Direct binding evaluation
We will revisit Rust bindings when:
- `cargo` dependency management is available in this environment.
- We can benchmark parity with the CLI path on macOS.
- The bindings expose stable APIs for KV cache tuning and sampling controls.

### Decision
Stick to subprocess execution for MVP. Re-evaluate after a stable release.

### RunConfig mapping
llama.cpp is configured via CLI flags:
- `model` -> `-m <path>`
- `context_length` -> `-c <n>`
- `max_tokens` -> `-n <n>`
- `threads` -> `--threads <n>`
- `gpu_layers` -> `--gpu-layers <n>`
- `cache_type_k` -> `--cache-type-k <type>`
- `cache_type_v` -> `--cache-type-v <type>`
- `extra_args` are appended at the end.

### Open questions
- Evaluate whether additional sampling flags should be surfaced in `RunConfig`.
- Track any breaking changes in `llama-cli` flag naming across releases.

## Model switching behavior
When memory pressure hits warning levels and `--warning-model` is set:
- The current process is stopped.
- The runtime restarts with the new model.
- When pressure returns to normal, the baseline config is restored.

### Constraints
- Switching is a full restart, not an in-process reload.
- Output streams are restarted; use `--capture-output` to track logs.
- Model-specific defaults should be validated before switching.

## MLC-LLM notes (stretch)
MLC-LLM integration is planned as a stretch goal with the following shape:
- Prefer a thin wrapper over the Python entrypoints or REST API.
- Ensure Metal acceleration paths are configurable for Apple silicon.
- Maintain parity with `RunConfig` fields where possible.

### Target parameters
- `max_seq_len` for context window control.
- Quantization options like W8A8 or W4A16 (if exposed).
- Precision settings such as bfloat16, INT8, or INT4.

### Decision
Track as a stretch item; no implementation committed yet.

### Open questions
- Identify the most stable Python entrypoint for local inference.
- Confirm if Metal backend supports dynamic context changes mid-run.
- Determine if process-level monitoring is enough or if API hooks are needed.

## vLLM status (future)
vLLM is CUDA-focused and best supported on Linux with NVIDIA GPUs. MPS support
is still evolving, so direct macOS integration is not targeted right now.

### Possible integration approaches
- Python wrapper that shells out to `python -m vllm.entrypoints` for local use.
- Docker-based workflow for CUDA environments.
- Future MPS path if vLLM stabilizes on Apple silicon.

### Decision
Document as future support only. No implementation planned in this phase.

### Open questions
- What is the minimal local workflow that works on macOS?
- Is there a stable vLLM MPS roadmap to depend on?
- Can a Docker workflow be documented without requiring GPU access?

## Memory monitoring notes
Memory monitoring is runtime-agnostic and uses the child process pid:
- `run --monitor` watches the spawned runtime process.
- `monitor --pid` can attach to a runtime launched separately.
- Warning and critical thresholds trigger the same adjustment logic for all runtimes.

### Known constraints
- Monitoring uses polling today; event-based pressure handling is macOS-only.
- Restarts may interrupt output streams unless `--capture-output` is used.

### Test coverage
- Unit tests cover thresholds, history limits, and child pid monitoring.
- CLI tests validate that `run --dry-run` includes runtime flags.
