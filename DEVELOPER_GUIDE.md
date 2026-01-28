# Developer Guide

## Build
```
cargo build
```

## Run
```
cargo run -- <command> [options]
```

## KV cache settings
For llama.cpp, KV cache quantization is controlled by:
- `--cache-type-k <type>` and `--cache-type-v <type>` CLI flags
- `cache_type_k` / `cache_type_v` keys in profile files

These values are passed directly to the llama.cpp CLI and should match supported cache types.

## Test
```
cargo test
```

## Adding a backend
1. Add a new module under `src/backends/`.
2. Implement `LLMRunner::build_command` in the backend.
3. Register the backend in `src/backends/mod.rs`.
4. Wire it in `src/commands/run.rs` for runtime selection.

## Testing guidelines
- Unit tests live next to modules (`#[cfg(test)]`).
- Prefer small, deterministic tests without external commands when possible.

## Contribution guidelines
- Keep changes focused and scoped to a single feature or fix.
- Add or update tests for new behavior.
- Update docs when CLI flags or outputs change.
