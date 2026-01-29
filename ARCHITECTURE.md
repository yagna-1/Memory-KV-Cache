# Architecture

## System overview
`llm-manager` is a CLI that coordinates runtime processes, memory monitoring, and KV-cache analysis.
The current implementation uses lightweight polling on macOS and leaves hooks for future
DispatchSource + Mach integrations.

## Module layout
- `src/main.rs`: CLI entry point and command routing
- `src/commands/`: command handlers (`run`, `kv-inspect`, `config`, `monitor`)
- `src/llm_backend.rs`: runtime trait + shared types
- `src/backends/`: backend-specific command builders
- `src/memory_monitor.rs`: polling monitor, thresholds, callbacks, history
- `src/gguf.rs`: GGUF metadata parser
- `src/profile.rs`: profile parsing and validation

## GGUF parsing notes
- Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- A custom parser is used to avoid additional dependencies while still extracting
  core KV metadata (layers, hidden size, heads, architecture).

## Memory monitoring approach
The monitor polls process memory via Mach `task_info` on macOS with a `ps` fallback and maps RSS to
warning/critical thresholds. Each poll
produces a `MemoryEvent`, which is:
- emitted over a channel for consumers (`monitor` command, `run --monitor`)
- stored in a bounded in-memory history buffer
- optionally handled by callbacks (normal/warning/critical)

Future work replaces polling with DispatchSource memory pressure events and Mach `task_info`.

## Runtime integration
Backends currently build subprocess command lines:
- `llama.cpp`: `llama-cli` with context length, tokens, threads, GPU layers
- `ollama`: `ollama run <model>`
- `mlc` and `vllm`: placeholders

### KV cache types (llama.cpp)
The run configuration also supports KV cache quantization for llama.cpp:
- `cache_type_k` maps to `--cache-type-k`
- `cache_type_v` maps to `--cache-type-v`

These settings are applied only when a llama.cpp backend is selected. They can be supplied
via CLI flags or profile files and are preserved across warning-level adjustments.

## Data flow
```
User CLI
  |
  +--> run -------------------> backend command ----> child process
  |                                   |
  |                                   +--> stdout/stderr capture (optional)
  |                                   +--> memory monitor (optional)
  |
  +--> kv-inspect -----------> gguf parser -> kv math -> output
  |
  +--> monitor --------------> memory monitor -> event stream/log
  |
  +--> config ---------------> profiles -> load/validate
```
