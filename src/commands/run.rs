use std::io::{BufRead, BufReader};
use std::os::raw::c_int;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{RecvTimeoutError, TryRecvError};
use std::sync::{atomic::{AtomicBool, Ordering}, Once};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::backends::{llama_cpp::LlamaCppBackend, mlc::MlcBackend, ollama::OllamaBackend, vllm::VllmBackend};
use crate::llm_backend::{CommandSpec, LLMRunner, RunConfig};
use crate::memory_monitor::{MemoryMonitor, MemoryPressureLevel, MemoryThresholds};
use crate::memory_pressure::start_memory_pressure_listener;
use crate::profile::load_profile;

static SIGNAL_INIT: Once = Once::new();
static SIGNAL_FLAG: AtomicBool = AtomicBool::new(false);

type SignalHandler = extern "C" fn(c_int);

extern "C" {
    fn signal(sig: c_int, handler: SignalHandler) -> SignalHandler;
}

const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

struct AdjustmentState {
    warning_applied: bool,
    critical_triggered: bool,
    adjusted: bool,
    baseline_config: Option<RunConfig>,
}

impl AdjustmentState {
    fn new() -> Self {
        Self {
            warning_applied: false,
            critical_triggered: false,
            adjusted: false,
            baseline_config: None,
        }
    }

    fn record_warning(&mut self) {
        self.warning_applied = true;
    }

    fn record_critical(&mut self) {
        self.critical_triggered = true;
    }

    fn record_baseline(&mut self, config: &RunConfig) {
        if self.baseline_config.is_none() {
            self.baseline_config = Some(config.clone());
        }
    }

    fn mark_adjusted(&mut self) {
        self.adjusted = true;
    }

    fn take_restore_config(&mut self) -> Option<RunConfig> {
        if !self.adjusted {
            return None;
        }
        self.adjusted = false;
        self.warning_applied = false;
        self.critical_triggered = false;
        self.baseline_config.take()
    }

    fn can_reduce_on_warning(&self, allow_repeat: bool) -> bool {
        allow_repeat || !self.warning_applied
    }
}

pub fn handle(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let mut runtime = None;
    let mut model = None;
    let mut context_length = None;
    let mut max_tokens = None;
    let mut threads = None;
    let mut gpu_layers = None;
    let mut cache_type_k = None;
    let mut cache_type_v = None;
    let mut profile_name = None;
    let mut profile_dir = None;
    let mut monitor = false;
    let mut monitor_interval_ms = 1000u64;
    let mut monitor_warning = None;
    let mut monitor_critical = None;
    let mut capture_output = false;
    let mut abort_on_critical = false;
    let mut warning_context_length = None;
    let mut warning_context_step = None;
    let mut warning_context_min = None;
    let mut warning_max_tokens = None;
    let mut warning_threads = None;
    let mut warning_gpu_layers = None;
    let mut warning_model = None;
    let mut progress = false;
    let mut pressure_events = false;
    let mut dry_run = false;
    let mut extra_args = Vec::new();

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--runtime" => runtime = Some(next_value("--runtime", &mut iter)?),
            "--model" => model = Some(PathBuf::from(next_value("--model", &mut iter)?)),
            "--context-length" => {
                context_length = Some(parse_u32("--context-length", &next_value("--context-length", &mut iter)?)?)
            }
            "--max-tokens" => {
                max_tokens = Some(parse_u32("--max-tokens", &next_value("--max-tokens", &mut iter)?)?)
            }
            "--threads" => {
                threads = Some(parse_u32("--threads", &next_value("--threads", &mut iter)?)?)
            }
            "--cache-type-k" => {
                cache_type_k = Some(next_value("--cache-type-k", &mut iter)?);
            }
            "--cache-type-v" => {
                cache_type_v = Some(next_value("--cache-type-v", &mut iter)?);
            }
            "--gpu-layers" => {
                gpu_layers = Some(parse_u32("--gpu-layers", &next_value("--gpu-layers", &mut iter)?)?)
            }
            "--profile" => profile_name = Some(next_value("--profile", &mut iter)?),
            "--profile-dir" => profile_dir = Some(PathBuf::from(next_value("--profile-dir", &mut iter)?)),
            "--monitor" => monitor = true,
            "--monitor-interval-ms" => {
                monitor_interval_ms = parse_u64("--monitor-interval-ms", &next_value("--monitor-interval-ms", &mut iter)?)?
            }
            "--monitor-warning" => {
                monitor_warning = Some(parse_bytes("--monitor-warning", &next_value("--monitor-warning", &mut iter)?)?)
            }
            "--monitor-critical" => {
                monitor_critical = Some(parse_bytes("--monitor-critical", &next_value("--monitor-critical", &mut iter)?)?)
            }
            "--abort-on-critical" => abort_on_critical = true,
            "--warning-context-length" => {
                warning_context_length =
                    Some(parse_u32("--warning-context-length", &next_value("--warning-context-length", &mut iter)?)?)
            }
            "--warning-context-step" => {
                warning_context_step =
                    Some(parse_u32("--warning-context-step", &next_value("--warning-context-step", &mut iter)?)?)
            }
            "--warning-context-min" => {
                warning_context_min =
                    Some(parse_u32("--warning-context-min", &next_value("--warning-context-min", &mut iter)?)?)
            }
            "--warning-max-tokens" => {
                warning_max_tokens =
                    Some(parse_u32("--warning-max-tokens", &next_value("--warning-max-tokens", &mut iter)?)?)
            }
            "--warning-threads" => {
                warning_threads =
                    Some(parse_u32("--warning-threads", &next_value("--warning-threads", &mut iter)?)?)
            }
            "--warning-gpu-layers" => {
                warning_gpu_layers =
                    Some(parse_u32("--warning-gpu-layers", &next_value("--warning-gpu-layers", &mut iter)?)?)
            }
            "--warning-model" => warning_model = Some(next_value("--warning-model", &mut iter)?),
            "--capture-output" => capture_output = true,
            "--progress" => progress = true,
            "--pressure-events" => pressure_events = true,
            "--dry-run" => dry_run = true,
            "--" => {
                extra_args.extend(iter);
                break;
            }
            other if other.starts_with('-') => {
                return Err(format!("Unknown flag for run: {other}"));
            }
            other => {
                extra_args.push(other.to_string());
            }
        }
    }

    let mut config = RunConfig {
        runtime: runtime.unwrap_or_default(),
        model,
        context_length,
        max_tokens,
        threads,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        extra_args,
    };

    if let Some(profile_name) = profile_name {
        let (profile, _path) = load_profile(&profile_name, profile_dir.as_deref())?;
        if warning_model.is_none() {
            warning_model = profile.warning_model.clone();
        }
        profile.apply_to_run_config(&mut config)?;
    }

    if config.runtime.is_empty() {
        config.runtime = "llama.cpp".to_string();
    }

    let runtime = config.runtime.clone();
    let backend: Box<dyn LLMRunner> = match runtime.as_str() {
        "llama.cpp" | "llama" => Box::new(LlamaCppBackend::default()),
        "ollama" => Box::new(OllamaBackend::default()),
        "mlc" | "mlc-llm" => Box::new(MlcBackend::default()),
        "vllm" => Box::new(VllmBackend::default()),
        other => return Err(format!("Unsupported runtime: {other}")),
    };

    if progress {
        capture_output = true;
    }
    if pressure_events {
        monitor = true;
    }

    if dry_run {
        let spec = backend.build_command(&config).map_err(|err| err.to_string())?;
        println!("{} {}", spec.program, spec.args.join(" "));
        return Ok(());
    }

    if monitor {
        run_with_monitor(
            backend,
            &config,
            monitor_interval_ms,
            monitor_warning,
            monitor_critical,
            capture_output,
            abort_on_critical,
            warning_context_length,
            warning_context_step,
            warning_context_min,
            warning_max_tokens,
            warning_threads,
            warning_gpu_layers,
            warning_model.as_deref(),
            progress,
            pressure_events,
        )
    } else if capture_output {
        run_with_capture(backend, &config, progress)
    } else {
        backend.run(&config).map_err(|err| err.to_string())
    }
}

fn next_value(flag: &str, iter: &mut impl Iterator<Item = String>) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("Missing value for {flag}"))
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("Invalid numeric value for {flag}: {value}"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("Invalid numeric value for {flag}: {value}"))
}

fn parse_bytes(flag: &str, value: &str) -> Result<u64, String> {
    let trimmed = value.trim().to_lowercase();
    let (number, unit) = trimmed
        .chars()
        .position(|c| !c.is_ascii_digit())
        .map(|idx| trimmed.split_at(idx))
        .unwrap_or((trimmed.as_str(), ""));
    let base: u64 = number
        .parse()
        .map_err(|_| format!("Invalid byte value for {flag}: {value}"))?;

    let multiplier = match unit {
        "" | "b" => 1,
        "k" | "kb" => 1024,
        "m" | "mb" => 1024u64.pow(2),
        "g" | "gb" => 1024u64.pow(3),
        "t" | "tb" => 1024u64.pow(4),
        _ => return Err(format!("Unknown unit for {flag}: {unit}")),
    };
    Ok(base * multiplier)
}

fn format_level(level: MemoryPressureLevel) -> &'static str {
    match level {
        MemoryPressureLevel::Normal => "normal",
        MemoryPressureLevel::Warning => "warning",
        MemoryPressureLevel::Critical => "critical",
    }
}

fn run_with_monitor(
    backend: Box<dyn LLMRunner>,
    config: &RunConfig,
    interval_ms: u64,
    warning: Option<u64>,
    critical: Option<u64>,
    capture_output: bool,
    abort_on_critical: bool,
    warning_context_length: Option<u32>,
    warning_context_step: Option<u32>,
    warning_context_min: Option<u32>,
    warning_max_tokens: Option<u32>,
    warning_threads: Option<u32>,
    warning_gpu_layers: Option<u32>,
    warning_model: Option<&str>,
    progress: bool,
    pressure_events: bool,
) -> Result<(), String> {
    install_signal_handlers();
    let mut active_config = config.clone();
    let spec = backend
        .build_command(&active_config)
        .map_err(|err| err.to_string())?;
    let (mut child, mut stdout_thread, mut stderr_thread) =
        spawn_child(&spec, capture_output, progress)?;

    let warning = warning.unwrap_or(512 * 1024 * 1024);
    let critical = critical.unwrap_or(1024 * 1024 * 1024);
    if critical < warning {
        return Err("Monitor critical threshold must be >= warning threshold".to_string());
    }

    let thresholds = MemoryThresholds {
        warning_bytes: warning,
        critical_bytes: critical,
    };
    let poll_interval = Duration::from_millis(interval_ms.max(100));
    let monitor = MemoryMonitor::new(poll_interval, thresholds).with_pid(child.id());
    let handle = monitor.start();
    let receiver = handle.receiver();
    let pressure_handle = if pressure_events {
        Some(start_memory_pressure_listener()?)
    } else {
        None
    };

    let mut state = AdjustmentState::new();
    loop {
        match receiver.recv_timeout(poll_interval) {
            Ok(event) => {
                println!(
                    "[monitor] {:<8} RSS {:>8} VSZ {:>8}",
                    format_level(event.level),
                    format_bytes(event.stats.resident_bytes),
                    format_bytes(event.stats.virtual_bytes)
                );
                if let Err(err) = handle_pressure_level(
                    event.level,
                    abort_on_critical,
                    warning_context_length,
                    warning_context_step,
                    warning_context_min,
                    warning_max_tokens,
                    warning_threads,
                    warning_gpu_layers,
                    warning_model,
                    &mut state,
                    &mut active_config,
                    backend.as_ref(),
                    capture_output,
                    progress,
                    &mut child,
                    &mut stdout_thread,
                    &mut stderr_thread,
                ) {
                    handle.stop();
                    join_output_threads(stdout_thread, stderr_thread);
                    return Err(err);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        if let Some(pressure_handle) = &pressure_handle {
            match pressure_handle.receiver().try_recv() {
                Ok(level) => {
                    println!("[pressure] {}", format_level(level));
                    if let Err(err) = handle_pressure_level(
                        level,
                        abort_on_critical,
                        warning_context_length,
                        warning_context_step,
                        warning_context_min,
                        warning_max_tokens,
                        warning_threads,
                        warning_gpu_layers,
                        warning_model,
                        &mut state,
                        &mut active_config,
                        backend.as_ref(),
                        capture_output,
                        progress,
                        &mut child,
                        &mut stdout_thread,
                        &mut stderr_thread,
                    ) {
                        handle.stop();
                        join_output_threads(stdout_thread, stderr_thread);
                        return Err(err);
                    }
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    handle.stop();
                    join_output_threads(stdout_thread, stderr_thread);
                    return Err("Pressure listener stopped".to_string());
                }
            }
        }

        if should_terminate() {
            let _ = child.kill();
            handle.stop();
            join_output_threads(stdout_thread, stderr_thread);
            return Err("Received termination signal".to_string());
        }

        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("Failed to poll child process: {err}"))?
        {
            handle.stop();
            join_output_threads(stdout_thread, stderr_thread);
            if status.success() {
                return Ok(());
            }
            return Err(format!(
                "Process exited with status {}",
                status.code().unwrap_or(-1)
            ));
        }
    }

    Ok(())
}

fn run_with_capture(
    backend: Box<dyn LLMRunner>,
    config: &RunConfig,
    progress: bool,
) -> Result<(), String> {
    install_signal_handlers();
    let spec = backend
        .build_command(config)
        .map_err(|err| err.to_string())?;
    let (mut child, stdout_thread, stderr_thread) = spawn_child(&spec, true, progress)?;
    loop {
        if should_terminate() {
            let _ = child.kill();
            join_output_threads(stdout_thread, stderr_thread);
            return Err("Received termination signal".to_string());
        }

        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("Failed to poll child process: {err}"))?
        {
            join_output_threads(stdout_thread, stderr_thread);
            if status.success() {
                return Ok(());
            }
            return Err(format!(
                "Process exited with status {}",
                status.code().unwrap_or(-1)
            ));
        }

        thread::sleep(Duration::from_millis(100));
    }
}

fn spawn_child(
    spec: &CommandSpec,
    capture_output: bool,
    progress: bool,
) -> Result<(Child, Option<JoinHandle<()>>, Option<JoinHandle<()>>), String> {
    let mut command = Command::new(&spec.program);
    command.args(&spec.args);
    for (key, value) in &spec.env {
        command.env(key, value);
    }
    if capture_output {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
    }
    let mut child = command
        .spawn()
        .map_err(|err| format!("Failed to start {}: {err}", spec.program))?;

    if capture_output {
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_thread = stdout.map(|stream| spawn_output_pump(stream, false, progress));
        let stderr_thread = stderr.map(|stream| spawn_output_pump(stream, true, progress));
        Ok((child, stdout_thread, stderr_thread))
    } else {
        Ok((child, None, None))
    }
}

fn spawn_output_pump<R: std::io::Read + Send + 'static>(
    reader: R,
    is_stderr: bool,
    progress: bool,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
                    if is_stderr {
                        eprintln!("[stderr] {trimmed}");
                    } else {
                        println!("[stdout] {trimmed}");
                    }
                    if progress {
                        if let Some(progress_line) = detect_progress(trimmed) {
                            println!("[progress] {progress_line}");
                        }
                    }
                }
                Err(_) => break,
            }
        }
    })
}

fn join_output_threads(stdout_thread: Option<JoinHandle<()>>, stderr_thread: Option<JoinHandle<()>>) {
    if let Some(thread) = stdout_thread {
        let _ = thread.join();
    }
    if let Some(thread) = stderr_thread {
        let _ = thread.join();
    }
}

fn handle_pressure_level(
    level: MemoryPressureLevel,
    abort_on_critical: bool,
    warning_context_length: Option<u32>,
    warning_context_step: Option<u32>,
    warning_context_min: Option<u32>,
    warning_max_tokens: Option<u32>,
    warning_threads: Option<u32>,
    warning_gpu_layers: Option<u32>,
    warning_model: Option<&str>,
    state: &mut AdjustmentState,
    active_config: &mut RunConfig,
    backend: &dyn LLMRunner,
    capture_output: bool,
    progress: bool,
    child: &mut Child,
    stdout_thread: &mut Option<JoinHandle<()>>,
    stderr_thread: &mut Option<JoinHandle<()>>,
) -> Result<(), String> {
    if abort_on_critical && matches!(level, MemoryPressureLevel::Critical) {
        state.record_critical();
        let _ = child.kill();
        return Err("Aborted on critical memory pressure".to_string());
    }
    if matches!(level, MemoryPressureLevel::Normal) {
        if let Some(baseline) = state.take_restore_config() {
            let _ = child.kill();
            join_output_threads(stdout_thread.take(), stderr_thread.take());
            *active_config = baseline;
            let spec = backend
                .build_command(active_config)
                .map_err(|err| err.to_string())?;
            let spawned = spawn_child(&spec, capture_output, progress)?;
            *child = spawned.0;
            *stdout_thread = spawned.1;
            *stderr_thread = spawned.2;
        }
        return Ok(());
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(false)
        && warning_context_length.is_some()
    {
        let new_length = warning_context_length.unwrap_or(0);
        if new_length > 0 && active_config.context_length != Some(new_length) {
            state.record_baseline(active_config);
            state.record_warning();
            state.mark_adjusted();
            let _ = child.kill();
            join_output_threads(stdout_thread.take(), stderr_thread.take());
            active_config.context_length = Some(new_length);
            let spec = backend
                .build_command(active_config)
                .map_err(|err| err.to_string())?;
            let spawned = spawn_child(&spec, capture_output, progress)?;
            *child = spawned.0;
            *stdout_thread = spawned.1;
            *stderr_thread = spawned.2;
            return Ok(());
        }
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(true)
        && warning_context_step.is_some()
    {
        let step = warning_context_step.unwrap_or(0);
        let min = warning_context_min.unwrap_or(1).max(1);
        if step > 0 {
            if let Some(current) = active_config.context_length {
                if current > min {
                    let target = current.saturating_sub(step).max(min);
                    if target != current {
                        state.record_baseline(active_config);
                        state.record_warning();
                        state.mark_adjusted();
                        let _ = child.kill();
                        join_output_threads(stdout_thread.take(), stderr_thread.take());
                        active_config.context_length = Some(target);
                        let spec = backend
                            .build_command(active_config)
                            .map_err(|err| err.to_string())?;
                        let spawned = spawn_child(&spec, capture_output, progress)?;
                        *child = spawned.0;
                        *stdout_thread = spawned.1;
                        *stderr_thread = spawned.2;
                        return Ok(());
                    }
                }
            }
        }
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(false)
        && warning_max_tokens.is_some()
    {
        let new_max = warning_max_tokens.unwrap_or(0);
        if new_max > 0 && active_config.max_tokens != Some(new_max) {
            state.record_baseline(active_config);
            state.record_warning();
            state.mark_adjusted();
            let _ = child.kill();
            join_output_threads(stdout_thread.take(), stderr_thread.take());
            active_config.max_tokens = Some(new_max);
            let spec = backend
                .build_command(active_config)
                .map_err(|err| err.to_string())?;
            let spawned = spawn_child(&spec, capture_output, progress)?;
            *child = spawned.0;
            *stdout_thread = spawned.1;
            *stderr_thread = spawned.2;
        }
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(false)
        && warning_threads.is_some()
    {
        let new_threads = warning_threads.unwrap_or(0);
        if new_threads > 0 && active_config.threads != Some(new_threads) {
            state.record_baseline(active_config);
            state.record_warning();
            state.mark_adjusted();
            let _ = child.kill();
            join_output_threads(stdout_thread.take(), stderr_thread.take());
            active_config.threads = Some(new_threads);
            let spec = backend
                .build_command(active_config)
                .map_err(|err| err.to_string())?;
            let spawned = spawn_child(&spec, capture_output, progress)?;
            *child = spawned.0;
            *stdout_thread = spawned.1;
            *stderr_thread = spawned.2;
        }
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(false)
        && warning_gpu_layers.is_some()
    {
        let new_layers = warning_gpu_layers.unwrap_or(0);
        if new_layers > 0 && active_config.gpu_layers != Some(new_layers) {
            state.record_baseline(active_config);
            state.record_warning();
            state.mark_adjusted();
            let _ = child.kill();
            join_output_threads(stdout_thread.take(), stderr_thread.take());
            active_config.gpu_layers = Some(new_layers);
            let spec = backend
                .build_command(active_config)
                .map_err(|err| err.to_string())?;
            let spawned = spawn_child(&spec, capture_output, progress)?;
            *child = spawned.0;
            *stdout_thread = spawned.1;
            *stderr_thread = spawned.2;
        }
    }
    if matches!(level, MemoryPressureLevel::Warning)
        && state.can_reduce_on_warning(false)
        && warning_model.is_some()
    {
        let new_model = warning_model.unwrap_or_default();
        if !new_model.is_empty() {
            let current = active_config
                .model
                .as_ref()
                .map(|m| m.display().to_string())
                .unwrap_or_default();
            if current != new_model {
                state.record_baseline(active_config);
                state.record_warning();
                state.mark_adjusted();
                let _ = child.kill();
                join_output_threads(stdout_thread.take(), stderr_thread.take());
                active_config.model = Some(std::path::PathBuf::from(new_model));
                let spec = backend
                    .build_command(active_config)
                    .map_err(|err| err.to_string())?;
                let spawned = spawn_child(&spec, capture_output, progress)?;
                *child = spawned.0;
                *stdout_thread = spawned.1;
                *stderr_thread = spawned.2;
            }
        }
    }
    Ok(())
}

fn install_signal_handlers() {
    SIGNAL_INIT.call_once(|| unsafe {
        let _ = signal(SIGINT, handle_signal as SignalHandler);
        let _ = signal(SIGTERM, handle_signal as SignalHandler);
    });
}

extern "C" fn handle_signal(_sig: c_int) {
    SIGNAL_FLAG.store(true, Ordering::SeqCst);
}

fn should_terminate() -> bool {
    SIGNAL_FLAG.load(Ordering::SeqCst)
}

fn detect_progress(line: &str) -> Option<String> {
    let lowered = line.to_lowercase();
    if lowered.contains("tok/s")
        || lowered.contains("tokens/s")
        || lowered.contains("ms/token")
    {
        Some(line.to_string())
    } else {
        None
    }
}

fn format_bytes(value: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = value as f64;
    let mut unit = 0usize;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value} {}", UNITS[unit])
    } else {
        format!("{:.2} {}", size, UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TestBackend {
        program: String,
    }

    impl LLMRunner for TestBackend {
        fn build_command(
            &self,
            _config: &RunConfig,
        ) -> crate::llm_backend::LLMResult<CommandSpec> {
            Ok(CommandSpec {
                program: self.program.clone(),
                args: vec!["0.2".to_string()],
                env: Vec::new(),
            })
        }
    }

    #[cfg(unix)]
    fn spawn_test_child(
        backend: &TestBackend,
        config: &RunConfig,
    ) -> (Child, Option<JoinHandle<()>>, Option<JoinHandle<()>>) {
        let spec = backend.build_command(config).unwrap();
        spawn_child(&spec, false, false).unwrap()
    }

    #[test]
    #[cfg(unix)]
    fn warning_context_length_updates_config() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: Some(1024),
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            Some(512),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();

        assert_eq!(config.context_length, Some(512));
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(unix)]
    fn warning_context_step_reduces_multiple_times() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: Some(768),
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            None,
            Some(128),
            Some(512),
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();
        assert_eq!(config.context_length, Some(640));

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            None,
            Some(128),
            Some(512),
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();
        assert_eq!(config.context_length, Some(512));

        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(unix)]
    fn warning_model_switch_updates_config() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("big.gguf")),
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("small.gguf"),
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();

        assert_eq!(
            config.model.as_ref().map(|m| m.display().to_string()),
            Some("small.gguf".to_string())
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(unix)]
    fn normal_restores_baseline_config() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: Some(1024),
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            Some(512),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();
        assert_eq!(config.context_length, Some(512));

        handle_pressure_level(
            MemoryPressureLevel::Normal,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();

        assert_eq!(config.context_length, Some(1024));
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(unix)]
    fn normal_restores_after_model_switch() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("big.gguf")),
            context_length: None,
            max_tokens: None,
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("small.gguf"),
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();
        assert_eq!(
            config.model.as_ref().map(|m| m.display().to_string()),
            Some("small.gguf".to_string())
        );

        handle_pressure_level(
            MemoryPressureLevel::Normal,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();

        assert_eq!(
            config.model.as_ref().map(|m| m.display().to_string()),
            Some("big.gguf".to_string())
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    #[cfg(unix)]
    fn normal_restores_after_max_tokens_change() {
        let backend = TestBackend {
            program: "sleep".to_string(),
        };
        let mut state = AdjustmentState::new();
        let mut config = RunConfig {
            runtime: "llama.cpp".to_string(),
            model: Some(PathBuf::from("model.gguf")),
            context_length: None,
            max_tokens: Some(256),
            threads: None,
            gpu_layers: None,
            cache_type_k: None,
            cache_type_v: None,
            extra_args: Vec::new(),
        };
        let (mut child, mut stdout_thread, mut stderr_thread) = spawn_test_child(&backend, &config);

        handle_pressure_level(
            MemoryPressureLevel::Warning,
            false,
            None,
            None,
            None,
            Some(128),
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();
        assert_eq!(config.max_tokens, Some(128));

        handle_pressure_level(
            MemoryPressureLevel::Normal,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &mut state,
            &mut config,
            &backend,
            false,
            false,
            &mut child,
            &mut stdout_thread,
            &mut stderr_thread,
        )
        .unwrap();

        assert_eq!(config.max_tokens, Some(256));
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn print_help() {
    println!(
        "USAGE:
  llm-manager run --runtime <llm> [options] [-- extra args]

OPTIONS:
  --runtime <llm>       Runtime to use (llama.cpp, ollama, mlc, vllm)
  --model <path>        Model path or name
  --context-length <n>  Context length (llama.cpp: -c)
  --max-tokens <n>      Max tokens to generate
  --threads <n>         Thread count
  --cache-type-k <type> KV cache type for K (llama.cpp)
  --cache-type-v <type> KV cache type for V (llama.cpp)
  --gpu-layers <n>      GPU layers for llama.cpp
  --profile <name>      Load profile from config directories
  --profile-dir <path>  Search this directory for profiles
  --monitor             Poll memory while process runs
  --monitor-interval-ms Poll interval for monitoring (default: 1000)
  --monitor-warning     Warning threshold (e.g. 512mb)
  --monitor-critical    Critical threshold (e.g. 1gb)
  --abort-on-critical   Stop the runtime on critical pressure
  --warning-context-length <n>
                       Restart with lower context length on warning
  --warning-context-step <n>
                       Reduce context length by step on warning
  --warning-context-min <n>
                       Minimum context length for step reduction
  --warning-max-tokens <n>
                       Restart with lower max tokens on warning
  --warning-threads <n>
                       Restart with fewer threads on warning
  --warning-gpu-layers <n>
                       Restart with fewer GPU layers on warning
  --warning-model <path>
                       Restart with a different model on warning
  --capture-output      Capture and prefix stdout/stderr
  --progress            Detect progress lines (implies capture)
  --pressure-events     Use macOS memory pressure events
  --dry-run             Print command without executing
  --help                Show this help
"
    );
}
