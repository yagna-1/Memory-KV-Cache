# llm-manager

Memory-aware LLM manager for macOS with KV-cache inspection tools.

## Status
- Early scaffold based on `plan.md`
- CLI parsing is minimal and std-only
- macOS memory pressure hooks are stubbed (see `src/memory_monitor.rs`)

## Features
- Runtime launching with optional memory monitoring
- KV-cache inspection for GGUF models
- Profile-based defaults for runtime settings
- JSON output for programmatic inspection

## System requirements
- macOS (Intel or Apple Silicon)
- Rust toolchain (stable)

## Installation
```
git clone <repo-url>
cd Memory-KV-Cache
cargo build
```

## Quick start
1. Install Rust (stable toolchain).
2. Build:
   - `cargo build`
3. Run:
   - `cargo run -- kv-inspect --model /path/to/model.gguf`
   - `cargo run -- kv-inspect --profile llama_cpp_safe --model /path/to/model.gguf --auto-ram --context-chart --layer-chart`
   - `cargo run -- config init`
   - `cargo run -- config show --verbose`
   - `cargo run -- config validate --project --json`
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --dry-run`
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --monitor --monitor-warning 1gb --monitor-critical 2gb --abort-on-critical --warning-context-length 1024 --warning-context-step 256 --warning-context-min 512 --warning-max-tokens 256 --warning-threads 4 --warning-gpu-layers 8 --warning-model /path/to/smaller.gguf --pressure-events`
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --capture-output --progress`
   - `cargo run -- run --runtime vllm --model /path/to/model --max-model-len 4096 --max-num-seqs 8 --quantization awq`
   - `cargo run -- monitor --interval-ms 1000`

## Project layout
```
src/
  main.rs
  commands/
    run.rs
    kv_inspect.rs
    config.rs
  memory_monitor.rs
  llm_backend.rs
  gguf.rs
  profile.rs
  backends/
    llama_cpp.rs
    ollama.rs
    mlc.rs
    vllm.rs
config/
  profiles/
```

## Notes
- Dependency additions are deferred until `cargo` is available in this environment.
- Memory monitoring uses Mach `task_info` on macOS with a `ps` fallback.
- Memory pressure events are available via `monitor --pressure-events` on macOS.
- Ollama context length uses `OLLAMA_CONTEXT_LENGTH` when `--context-length` is provided and falls back to Modelfile `PARAMETER num_ctx`.
- Normal pressure restores the baseline run configuration after warning adjustments.
- llama.cpp cache types can be set with `--cache-type-k` and `--cache-type-v`.
- See `plan.md` for detailed roadmap.

## Docs
- `CLI.md` for command reference and examples
- `ARCHITECTURE.md` for system overview
- `DEVELOPER_GUIDE.md` for build/test and backend notes
- `RUNTIME_SUPPORT.md` for runtime research and decisions
- `flow.mdc` for the iteration workflow
