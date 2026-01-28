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
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --dry-run`
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --monitor --monitor-warning 1gb --monitor-critical 2gb --abort-on-critical --warning-context-length 1024 --pressure-events`
   - `cargo run -- run --profile llama_cpp_safe --model /path/to/model.gguf --capture-output --progress`
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
- See `plan.md` for detailed roadmap.

## Docs
- `CLI.md` for command reference and examples
- `ARCHITECTURE.md` for system overview
- `DEVELOPER_GUIDE.md` for build/test and backend notes
