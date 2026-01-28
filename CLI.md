# CLI Reference

## Global
```
llm-manager <command> [options]
```

Commands:
- `run`: launch a runtime with optional monitoring
- `kv-inspect`: compute KV-cache sizing for a GGUF model
- `config`: profile listing and initialization
- `monitor`: poll and report process memory
- `version`: print version

## run
```
llm-manager run --runtime <llm> [options] [-- extra args]
```

Options:
- `--runtime <llm>`: `llama.cpp`, `ollama`, `mlc`, `vllm`
- `--model <path>`: model path or name
- `--context-length <n>`: context length (llama.cpp `-c`)
- `--max-tokens <n>`: max tokens
- `--threads <n>`: threads count
- `--gpu-layers <n>`: llama.cpp GPU layers
- `--profile <name>`: load defaults from a profile
- `--profile-dir <path>`: additional profile search directory
- `--monitor`: poll memory while process runs
- `--monitor-interval-ms <n>`: monitor interval (ms)
- `--monitor-warning <size>`: warning threshold (e.g. `512mb`)
- `--monitor-critical <size>`: critical threshold (e.g. `1gb`)
- `--abort-on-critical`: stop runtime on critical pressure
- `--warning-context-length <n>`: restart with lower context length on warning
- `--warning-max-tokens <n>`: restart with lower max tokens on warning
- `--capture-output`: prefix stdout/stderr lines
- `--progress`: detect progress lines (implies capture)
- `--pressure-events`: use macOS memory pressure events
- `--dry-run`: print command without executing

Example:
```
llm-manager run --profile llama_cpp_safe --model /path/to/model.gguf --monitor --abort-on-critical
```

## kv-inspect
```
llm-manager kv-inspect --model model.gguf [options]
```

Options:
- `--context-length <n>`: context length (default 2048)
- `--precision <fp16|fp32|int8>`
- `--profile <name>`: load defaults from a profile
- `--profile-dir <path>`: additional profile search directory
- `--layers`: per-layer table
- `--chart`: cumulative layer chart
- `--layer-chart`: per-layer bar chart
- `--context-chart`: memory vs context length chart
- `--context-step <n>`: step size for context chart (default 512)
- `--available-ram <size>`: estimate max tokens
- `--auto-ram`: auto-detect RAM (macOS)
- `--json`: JSON output (includes charts when selected)

Example:
```
llm-manager kv-inspect --model model.gguf --auto-ram --context-chart --layer-chart
```

## config
```
llm-manager config [show|edit|init]
```

Options:
- `show`: list profile files (`--verbose` prints values)
- `edit`: print profile directory and editor hint
- `init`: copy default profiles into `~/.llm-manager/config` (`--force` overwrites)

Example:
```
llm-manager config init
llm-manager config show --verbose
```

## monitor
```
llm-manager monitor [options]
```

Options:
- `--pid <pid>`: monitor a specific process
- `--interval-ms <n>`: poll interval
- `--warning <size>`: warning threshold
- `--critical <size>`: critical threshold
- `--log-file <path>`: append events to a log file
- `--pressure-events`: use macOS memory pressure events

Example:
```
llm-manager monitor --pid 12345 --warning 1gb --critical 2gb --log-file monitor.log
```

Memory pressure events (macOS):
```
llm-manager monitor --pressure-events --log-file pressure.log
```

## Profile format
Profiles are simple key/value TOML files. Supported fields:
```
runtime = "llama.cpp"
model = "/path/to/model.gguf"
context_length = 2048
max_tokens = 512
threads = 4
gpu_layers = 0
precision = "fp16"
```
