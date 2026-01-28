# Developer Guide

## Build
```
cargo build
```

## Run
```
cargo run -- <command> [options]
```

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
