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

## Memory monitoring approach
The monitor polls process memory via `ps` and maps RSS to warning/critical thresholds. Each poll
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
