# Memory-Aware LLM Manager for macOS - Development TODO

## Project Overview
A Rust-based CLI tool for managing LLM inference on memory-constrained macOS systems (8-16GB RAM), featuring real-time memory monitoring, dynamic parameter adjustment, and multi-runtime support.

---

## Tech Stack & Architecture Decisions

### Primary Stack
- **Language**: Rust (chosen over Go)
  - Rationale: Zero-cost abstractions, no garbage collector overhead, superior low-level memory control
  - Better for interfacing with C/C++ libraries (llama.cpp bindings)
  - Predictable performance and memory safety critical for 8-16GB systems
- **CLI Framework**: `clap` for argument parsing
- **Workspace Layout**: Standard Rust project structure

### Project Structure
```
llm-manager/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── commands/
│   │   ├── run.rs           # run subcommand
│   │   ├── kv_inspect.rs    # kv-inspect subcommand
│   │   └── config.rs        # config subcommand
│   ├── memory_monitor.rs    # macOS memory hooks
│   ├── llm_backend.rs       # LLMBackend trait
│   └── backends/
│       ├── llama_cpp.rs
│       ├── ollama.rs
│       ├── mlc.rs
│       └── vllm.rs
└── config/
    └── profiles/            # Runtime-specific configs
```

---

## CLI Design

### Subcommands Structure
```bash
# Main usage patterns
llm-manager run --runtime <llm> [options]
llm-manager kv-inspect --model model.gguf
llm-manager config [edit|show]
llm-manager version
llm-manager help
```

### Implementation Reference
```rust
let matches = clap::Command::new("llm-manager")
    .subcommand(SubCommand::with_name("run")…)
    .subcommand(SubCommand::with_name("kv-inspect")…)
    .get_matches();
```

---

## Phase 1: Setup & Research (Week 1)

### [ ] Environment Setup
- [x] Initialize Rust project (Cargo.toml + src scaffold)
- [ ] Add dependencies:
  - [ ] `clap` for CLI parsing
  - [ ] `dispatch` crate (or objc FFI bindings)
  - [ ] GGUF parsing library or implement custom parser
  - [ ] `llama_cpp` Rust bindings (optional)
- [x] Set up project structure (see above)
- [x] Create initial `.gitignore` and `README.md`

### [ ] macOS Memory API Research & Prototyping
- [ ] **DispatchSource Memory Pressure**
  - [ ] Study Grand Central Dispatch documentation
  - [ ] Implement prototype using `DispatchSourceMemoryPressure`
  - [ ] Reference Swift pattern:
    ```swift
    let source = DispatchSource.makeMemoryPressureSource(eventMask: .all, queue: .main)
    source.setEventHandler {
        let level = source.data
        switch level {
        case .warning: print("Memory pressure warning")
        case .critical: print("Memory pressure critical")
        default: break
        }
    }
    source.resume()
    ```
  - [ ] Implement Rust equivalent using `dispatch` crate or FFI
  - [ ] Test detection of `.normal`, `.warning`, `.critical` events

- [ ] **Mach API for Memory Stats**
  - [ ] Study `task_info()` API documentation
  - [ ] Implement wrapper for `task_info(TASK_BASIC_INFO_64)`
  - [ ] Reference pattern:
    ```c
    task_t target = mach_task_self();
    struct task_basic_info ti; // contains resident_size, etc.
    task_info(target, TASK_BASIC_INFO_64, (task_info_t)&ti, &count);
    ```
  - [ ] Get current process resident (RAM) usage in bytes
  - [ ] Test retrieving memory stats for child processes via `task_for_pid()`

### [ ] Configuration System Design
- [x] Design profile format (TOML/JSON)
- [x] Define schema for runtime-specific settings
- [x] Create default profiles for:
  - [x] llama.cpp (safe vs aggressive)
  - [x] Ollama
  - [x] MLC-LLM
  - [x] vLLM
- [x] Define config directory: `~/.llm-manager/config`

---

## Phase 2: Memory Monitor & Core Loop (Week 2)

### [ ] MemoryMonitor Module
- [x] Create `src/memory_monitor.rs`
- [ ] Implement DispatchSource subscription
  - [ ] Subscribe to memory pressure events
  - [x] Create callback handlers for:
    - [ ] `.normal` - reset to default settings
    - [ ] `.warning` - begin proactive mitigation
    - [ ] `.critical` - trigger aggressive measures
- [x] Implement logging system for memory events
- [x] Add threshold configuration (when to act on warnings)

### [ ] Process Memory Tracking
- [ ] Implement `task_info()` wrapper
- [x] Create function to get current process memory usage
- [x] Create function to get child process memory (for LLM runtimes)
- [x] Add periodic polling mechanism (configurable interval)
- [x] Implement memory usage history tracking

### [ ] Core Supervision Loop
- [ ] Design main event loop architecture
- [x] Implement process spawning mechanism
- [x] Add signal handling (SIGINT, SIGTERM)
- [ ] Test with simple subprocess (e.g., `llama-cli --help`)
- [x] Implement graceful shutdown logic
- [x] Add process cleanup on exit

### [ ] Dynamic Adjustment Logic (Initial)
- [x] Define intervention thresholds
- [x] Implement decision tree for memory mitigation:
  - [x] Warning level actions
  - [x] Critical level actions
- [x] Create state machine for tracking adjustment history

---

## Phase 3: LLM Runtime Integration (Week 3)

### [ ] LLMBackend Trait Definition
```rust
trait LLMRunner {
    fn run(&self, config: &RunConfig) -> Result<()>;
    fn stop(&self) -> Result<()>;
    fn adjust_params(&self, params: &AdjustmentParams) -> Result<()>;
}
```
- [x] Define `RunConfig` struct
- [x] Define `AdjustmentParams` struct
- [x] Add error types for runtime operations

### [ ] llama.cpp Integration (Primary Backend)
- [ ] **Basic Execution**
  - [x] Implement subprocess spawning for `llama-cli`
  - [x] Build command with flags:
    ```bash
    llama-cli -m model.gguf -n {max_tokens} --threads {N} --gpu-layers {L}
    ```
  - [x] Capture stdout/stderr streams
  - [x] Parse output for progress indicators
  - [x] Detect errors and completion

- [ ] **Dynamic Parameter Adjustment**
  - [ ] Implement context length reduction (`-n` flag)
  - [ ] Implement GPU layer adjustment (`--gpu-layers`)
  - [ ] Implement thread count adjustment (`--threads`)
  - [ ] Test restart with modified parameters
  - [ ] Implement state preservation during restart

- [ ] **Memory-Aware Features**
  - [x] Monitor llama-cli process memory
  - [x] Implement preemptive context reduction on warning
  - [x] Implement emergency stop on critical pressure
  - [ ] Add quantization flag support (if available)

- [ ] **Alternative: Direct Binding**
  - [ ] Evaluate `llama_cpp` Rust crate integration
  - [ ] Compare performance vs subprocess approach
  - [ ] Document decision and rationale

### [ ] Ollama Integration
- [ ] Study Ollama CLI and Python API
- [x] Implement wrapper for `ollama run <model>`
- [ ] Parse Ollama model configuration format
- [ ] Implement context length adjustment via config
- [ ] Add model switching capability (lighter models)
- [ ] Test memory monitoring of Ollama process

### [ ] MLC-LLM Integration (Stretch)
- [ ] Study MLC-LLM Python API
- [ ] Implement REST API client or Python subprocess wrapper
- [ ] Configure Apple Metal acceleration
- [ ] Implement parameters:
  - [ ] `max_seq_len` adjustment
  - [ ] Quantization options (W8A8, W4A16)
  - [ ] Precision settings (bfloat16, INT8, INT4)
- [ ] Test OpenAI-compatible interface

### [ ] vLLM Integration (Future/Optional)
- [ ] Document vLLM support status (CUDA-focused, MPS experimental)
- [ ] Design integration approach (Docker or native MPS)
- [ ] Implement Python API wrapper if supported
- [ ] Configure:
  - [ ] `max_model_len` for context limits
  - [ ] `max_num_seqs` for batch size
  - [ ] Quantization options
- [ ] Note as "future support" in docs

### [ ] Runtime Profile System
- [x] Create profile templates for each runtime
- [x] Define "safe" profile (conservative memory usage)
- [x] Define "aggressive" profile (maximize performance)
- [x] Implement profile switching logic
- [x] Add profile validation

---

## Phase 4: KV-Cache Tool & Enhancements (Week 4)

### [ ] GGUF Parser Implementation
- [ ] Research GGUF file format specification
- [ ] Evaluate existing Rust GGUF parsing libraries
- [x] Decision: Use library vs implement custom parser
- [ ] Implement metadata extraction:
  - [x] Model dimensions
  - [x] Layer count
  - [x] Head count
  - [x] Hidden size
  - [x] Architecture details

### [ ] KV-Cache Calculation Logic
- [x] **Per-Token, Per-Layer Calculation**
  - [x] Formula: Each layer stores K and V vectors per token
  - [x] Size = 2 × hidden_size elements per layer per token
  - [x] Account for precision (FP16 = 2 bytes, FP32 = 4 bytes)
  - [ ] Reference: "context size = sum of sizes of tensors created by llama_kv_cache_init"

- [x] **Total KV Memory Formula**
  ```
  total_kv_memory = (num_layers) × 2 × hidden_size × bytes_per_element × context_length
  ```
  - [x] Example: FP16 → 2 bytes per element
  - [x] Implement for various precision levels

- [x] **Per-Layer Breakdown**
  - [x] Calculate and store per-layer K size
  - [x] Calculate and store per-layer V size
  - [x] Sum for total per-layer KV size
  - [x] Multiply by context length for total usage

### [ ] kv-inspect Subcommand
- [x] Implement `commands/kv_inspect.rs`
- [x] CLI interface:
  ```bash
  llm-manager kv-inspect --model model.gguf [--context-length N] [--precision fp16|fp32]
  ```
- [x] Parse GGUF file
- [x] Compute KV-cache sizes
- [x] **Output Formats**:
  - [x] Table view (per-layer breakdown)
  - [x] JSON output (for programmatic use)
  - [x] Summary statistics
  - [x] ASCII chart/visualization

### [ ] Memory Projection Features
- [ ] **Context Window Exhaustion Estimator**
  - [x] Input: Available RAM (auto-detect or user-specified)
  - [x] Calculate: Max tokens given memory constraint
  - [x] Formula: `max_tokens = available_memory / (kv_size_per_token × num_layers)`
  - [x] Example: "With 4GB free RAM, you can handle ~N tokens"

- [ ] **Usage Extrapolation**
  - [x] Plot memory vs token count curve
  - [x] Identify when memory limit will be hit
  - [x] Warn about approaching limits

- [ ] **Visualization**
  - [x] Bar chart: Memory usage per layer
  - [x] Line graph: Total memory vs context length
  - [x] ASCII art plots for terminal output
  - [x] Optional: JSON export for external graphing

### [ ] Enhanced Runtime Adjustments
- [ ] **Quantization Support**
  - [ ] Implement switching between FP16/FP32/INT8
  - [ ] Test memory reduction from quantization
  - [ ] Document quantization impact on quality

- [ ] **Dynamic Context Resizing**
  - [ ] Implement gradual context reduction
  - [ ] Add context window monitoring
  - [ ] Implement "sliding window" strategy if supported

- [ ] **Batch Size Adjustment**
  - [ ] Reduce batch size on memory pressure (vLLM)
  - [ ] Monitor impact on throughput

- [ ] **Layer Offloading**
  - [ ] Adjust GPU vs CPU layer placement
  - [ ] Balance RAM vs GPU memory
  - [ ] Implement gradual migration strategy

---

## Phase 5: Testing & Polish (Week 5)

### [ ] Test Suite Development
- [ ] **Unit Tests**
  - [x] Memory monitor callbacks
  - [x] KV-cache calculations
  - [x] GGUF parsing
  - [x] Profile loading/validation
  - [x] Each backend trait implementation

- [ ] **Integration Tests**
  - [ ] End-to-end with mock LLM process
  - [ ] Config file loading
  - [ ] Subcommand execution
  - [ ] Memory pressure simulation

- [ ] **Stress Tests**
  - [ ] Simulate low-memory scenarios
  - [ ] Test with throttled memory limits
  - [ ] Verify graceful degradation
  - [ ] Test with various model sizes
  - [ ] Concurrent process handling

### [ ] Edge Case Handling
- [ ] OOM prevention verification
- [ ] Rapid memory pressure changes
- [ ] Runtime crashes during adjustment
- [ ] Invalid model files
- [ ] Missing runtime executables
- [ ] Corrupted config files
- [ ] Insufficient permissions
- [ ] Disk space exhaustion

### [ ] Documentation
- [x] **README.md**
  - [x] Project overview
  - [x] Installation instructions
  - [x] Quick start guide
  - [x] Feature list
  - [x] System requirements

- [x] **ARCHITECTURE.md**
  - [x] System design overview
  - [x] Memory monitoring approach
  - [x] Runtime integration details
  - [x] Data flow diagrams

- [x] **CLI Reference**
  - [x] All subcommands documented
  - [x] Flag descriptions
  - [x] Usage examples
  - [x] Configuration file format

- [x] **Developer Guide**
  - [x] Building from source
  - [x] Adding new backends
  - [x] Testing guidelines
  - [x] Contribution guidelines

### [ ] Packaging Preparation
- [ ] **Homebrew Formula**
  - [ ] Study Homebrew tap creation process
  - [ ] Create formula file
  - [ ] Reference: https://justin.searls.co/posts/how-to-distribute-your-own-scripts-via-homebrew/
  - [ ] Test installation on clean macOS system
  - [ ] Prepare for M-series Mac bottles
  - [ ] Set up GitHub releases integration

- [ ] **Binary Build**
  - [ ] Configure release builds
  - [ ] Test on Intel Mac
  - [ ] Test on Apple Silicon Mac
  - [ ] Create universal binary if needed
  - [ ] Strip symbols for size optimization

---

## Phase 6: MVP Release (Week 6)

### [ ] Pre-Release Checklist
- [ ] All critical tests passing
- [ ] Documentation complete
- [ ] Version number finalized (0.1.0)
- [ ] CHANGELOG.md created
- [ ] License file added (MIT/Apache-2.0)

### [ ] Performance Optimization
- [ ] Profile critical paths
- [ ] Optimize memory monitor overhead
- [ ] Reduce subprocess spawn latency
- [ ] Optimize GGUF parsing
- [ ] Minimize binary size

### [ ] Bug Fixes & Refinement
- [ ] Address all known bugs
- [ ] Code review
- [ ] Improve error messages
- [ ] Add helpful hints/suggestions
- [ ] Validate all user inputs

### [ ] Release Process
- [ ] **GitHub Release**
  - [ ] Tag v0.1.0
  - [ ] Create release notes
  - [ ] Upload binary artifacts
  - [ ] Generate checksums

- [ ] **Homebrew Publication**
  - [ ] Create personal tap repository
  - [ ] Write formula pointing to GitHub release
  - [ ] Test formula installation
  - [ ] Document installation steps

- [ ] **GitHub CLI Extension** (Optional)
  - [ ] Study: https://docs.github.com/en/github-cli/github-cli/creating-github-cli-extensions
  - [ ] Create extension structure
  - [ ] Package Rust binary
  - [ ] Test `gh extension install`
  - [ ] Publish extension

### [ ] Launch Activities
- [ ] Announce on relevant forums (Reddit, HN)
- [ ] Create demo video/GIF
- [ ] Set up issue templates
- [ ] Monitor initial user feedback
- [ ] Prepare for bug reports

---

## Stretch Goals (Post-MVP)

### [ ] GUI Frontend with Tauri
- [ ] **Setup**
  - [ ] Add Tauri dependencies
  - [ ] Initialize Tauri project
  - [ ] Design UI mockups

- [ ] **Features**
  - [ ] Real-time memory pressure display
  - [ ] Visual memory usage graphs
  - [ ] Runtime settings controls:
    - [ ] Context length slider
    - [ ] Quantization checkboxes
    - [ ] GPU layer slider
  - [ ] Model selection dropdown
  - [ ] Profile switcher
  - [ ] Log viewer

- [ ] **Integration**
  - [ ] Connect to Rust backend
  - [ ] Expose CLI functionality via IPC
  - [ ] Real-time event streaming
  - [ ] Settings persistence

- [ ] **Distribution**
  - [ ] Create macOS app bundle
  - [ ] Sign and notarize app
  - [ ] DMG installer creation
  - [ ] Auto-update mechanism

### [ ] Remote LLM Fallback
- [ ] **Cloud Routing**
  - [ ] Detect when local memory exhausted
  - [ ] Implement API fallback (OpenAI, Anthropic, etc.)
  - [ ] Cache remote responses locally
  - [ ] Seamless transition between local/remote

- [ ] **Smart Routing**
  - [ ] Small requests → local
  - [ ] Large contexts → remote
  - [ ] Cost tracking
  - [ ] User preferences for routing

### [ ] Additional Runtime Support
- [ ] Hugging Face Transformers integration
- [ ] WebLLM support
- [ ] LocalAI compatibility
- [ ] GPT4All integration
- [ ] Automatic runtime detection
- [ ] Multi-runtime load balancing

### [ ] Advanced Features
- [ ] **Memory Profiling Dashboard**
  - [ ] Detailed per-layer memory breakdown
  - [ ] Timeline view of memory usage
  - [ ] Export profiling data

- [ ] **Auto-Optimization**
  - [ ] Machine learning-based parameter tuning
  - [ ] Learn from usage patterns
  - [ ] Suggest optimal configurations

- [ ] **Multi-Model Management**
  - [ ] Switch between models automatically
  - [ ] Model download management
  - [ ] Storage optimization

- [ ] **Distributed Inference**
  - [ ] Split model across multiple machines
  - [ ] Network-based layer offloading

---

## Technical References & Resources

### macOS Memory APIs
- **DispatchSource Documentation**
  - GCD DispatchSource guide: https://shchukin-alex.medium.com/gcd-dispatchsource-and-target-queue-hierarchy-4c554ea10fd2
  - Memory pressure notifications via `.warning` and `.critical` events

- **Mach API**
  - `task_info()` for memory stats: https://stackoverflow.com/questions/5839626/how-is-top-able-to-see-memory-usage
  - Returns resident memory usage in bytes

### LLM Runtime Documentation
- **llama.cpp**
  - Main repo: https://github.com/ggml-org/llama.cpp
  - KV-cache discussion: https://github.com/ggml-org/llama.cpp/discussions/10068
  - Rust bindings: https://docs.rs/llama_cpp/latest/llama_cpp/

- **Ollama**
  - CLI reference: https://docs.ollama.com/cli

- **MLC-LLM**
  - GitHub: https://github.com/mlc-ai/mlc-llm
  - OpenAI-compatible interface with Apple Metal support

- **vLLM**
  - Memory conservation: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
  - Context limits, quantization, batch size tuning

### Distribution Resources
- **Homebrew**
  - Distribution guide: https://justin.searls.co/posts/how-to-distribute-your-own-scripts-via-homebrew/
  - Formula creation and tap setup

- **GitHub CLI Extensions**
  - Official docs: https://docs.github.com/en/github-cli/github-cli/creating-github-cli-extensions

### Technology Comparisons
- Rust vs Go: https://bitfieldconsulting.com/posts/rust-vs-go
  - Rust chosen for zero-cost abstractions and no GC overhead

---

## Key Formulas & Calculations

### KV-Cache Memory Formula
```
# Per-token, per-layer
kv_size_per_token_per_layer = 2 × hidden_size × bytes_per_element

# Total for all layers and tokens
total_kv_memory = num_layers × 2 × hidden_size × bytes_per_element × context_length

# Example (FP16):
# hidden_size = 4096, num_layers = 32, context_length = 2048, FP16 = 2 bytes
# total = 32 × 2 × 4096 × 2 × 2048 = 1,073,741,824 bytes (~1GB)
```

### Context Window Limit Estimation
```
max_context_tokens = available_ram_bytes / (num_layers × 2 × hidden_size × bytes_per_element)

# Example: 4GB available, model as above
# max_tokens = 4,294,967,296 / (32 × 2 × 4096 × 2) ≈ 8,192 tokens
```

---

## Success Criteria

### MVP Requirements
- [ ] Runs on macOS (Intel and Apple Silicon)
- [ ] Detects memory pressure events accurately
- [ ] Successfully adjusts at least one LLM runtime (llama.cpp)
- [ ] Prevents OOM crashes in test scenarios
- [ ] KV-cache inspection works for GGUF models
- [ ] Installable via Homebrew
- [ ] Documentation covers all features

### Quality Metrics
- [ ] <100ms overhead for memory monitoring
- [ ] <500ms response time to memory pressure events
- [ ] Binary size <20MB
- [ ] Memory footprint <50MB idle
- [ ] 80%+ test coverage
- [ ] Zero known critical bugs

---

## Notes & Decisions

### Why Rust over Go?
- No garbage collector overhead (critical for memory-constrained systems)
- Zero-cost abstractions for optimal performance
- Better FFI for C/C++ library integration (llama.cpp)
- Growing ecosystem for LLM tooling
- Predictable memory behavior

### Memory Pressure Thresholds
- **Normal**: Default operation
- **Warning**: Begin proactive mitigation (reduce context, lower batch size)
- **Critical**: Aggressive measures (quantize, offload, or abort)

### Runtime Priority Order
1. llama.cpp (most mature, best documented)
2. Ollama (popular, good UX)
3. MLC-LLM (Apple Metal optimized)
4. vLLM (future, requires more investigation)

### Config File Location
- `~/.llm-manager/config` for user profiles
- System-wide defaults in app bundle

---

## Current Status
- [ ] Not Started
- Project plan complete, ready to begin implementation

## Next Actions
1. Set up Rust project and dependencies
2. Begin Week 1 tasks (prototype memory monitoring)
3. Create initial GitHub repository