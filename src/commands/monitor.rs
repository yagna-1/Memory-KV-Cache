use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::memory_monitor::{MemoryMonitor, MemoryPressureLevel, MemoryThresholds};
use crate::memory_pressure::start_memory_pressure_listener;

pub fn handle(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let mut pid = None;
    let mut interval_ms = 1000u64;
    let mut warning = None;
    let mut critical = None;
    let mut log_file = None;
    let mut pressure_events = false;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--pid" => {
                pid = Some(parse_u32("--pid", &next_value("--pid", &mut iter)?)?);
            }
            "--interval-ms" => {
                interval_ms = parse_nonzero_u64(
                    "--interval-ms",
                    &next_value("--interval-ms", &mut iter)?,
                )?;
            }
            "--warning" => {
                warning = Some(parse_nonzero_bytes(
                    "--warning",
                    &next_value("--warning", &mut iter)?,
                )?);
            }
            "--critical" => {
                critical = Some(parse_nonzero_bytes(
                    "--critical",
                    &next_value("--critical", &mut iter)?,
                )?);
            }
            "--log-file" => log_file = Some(next_value("--log-file", &mut iter)?),
            "--pressure-events" => pressure_events = true,
            other if other.starts_with('-') => {
                return Err(format!(
                    "Unknown flag for monitor: {other}. Use --help for options."
                ));
            }
            _ => {}
        }
    }

    if pressure_events {
        if pid.is_some() {
            return Err("--pressure-events does not support --pid".to_string());
        }
        let mut log = if let Some(path) = log_file {
            Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|err| format!("Failed to open log file {path}: {err}"))?,
            )
        } else {
            None
        };
        let handle = start_memory_pressure_listener()?;
        println!("Listening for memory pressure events...");
        loop {
            let level = handle
                .receiver()
                .recv()
                .map_err(|err| format!("Pressure listener stopped: {err}"))?;
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default();
            println!("[{timestamp}] memory-pressure {level}", level = format_level(level));
            if let Some(file) = log.as_mut() {
                writeln!(file, "{timestamp} pressure={}", format_level(level))
                    .map_err(|err| format!("Failed to write log: {err}"))?;
            }
        }
    }

    let warning = warning.unwrap_or(512 * 1024 * 1024);
    let critical = critical.unwrap_or(1024 * 1024 * 1024);

    if critical < warning {
        return Err("Critical threshold must be >= warning threshold".to_string());
    }

    let mut log = if let Some(path) = log_file {
        Some(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|err| format!("Failed to open log file {path}: {err}"))?,
        )
    } else {
        None
    };

    let thresholds = MemoryThresholds {
        warning_bytes: warning,
        critical_bytes: critical,
    };
    let mut monitor = MemoryMonitor::new(Duration::from_millis(interval_ms), thresholds);
    if let Some(pid) = pid {
        monitor = monitor.with_pid(pid);
    }

    let handle = monitor.start();
    println!(
        "Monitoring memory (warning {}, critical {})",
        format_bytes(warning),
        format_bytes(critical)
    );
    loop {
        let event = handle
            .receiver()
            .recv()
            .map_err(|err| format!("Monitor stopped: {err}"))?;
        let timestamp = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        println!(
            "[{}] {:<8} RSS {:>8} VSZ {:>8}",
            timestamp,
            format_level(event.level),
            format_bytes(event.stats.resident_bytes),
            format_bytes(event.stats.virtual_bytes)
        );
        if let Some(file) = log.as_mut() {
            writeln!(
                file,
                "{timestamp} {} rss={} vsz={}",
                format_level(event.level),
                event.stats.resident_bytes,
                event.stats.virtual_bytes
            )
            .map_err(|err| format!("Failed to write log: {err}"))?;
        }
    }
}

fn format_level(level: MemoryPressureLevel) -> &'static str {
    match level {
        MemoryPressureLevel::Normal => "normal",
        MemoryPressureLevel::Warning => "warning",
        MemoryPressureLevel::Critical => "critical",
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

fn parse_nonzero_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = parse_u64(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} must be > 0"));
    }
    Ok(parsed)
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
        _ => {
            return Err(format!(
                "Unknown unit for {flag}: {unit} (use b, kb, mb, gb, tb)"
            ))
        }
    };
    Ok(base * multiplier)
}

fn parse_nonzero_bytes(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = parse_bytes(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} must be > 0"));
    }
    Ok(parsed)
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

fn print_help() {
    println!(
        "USAGE:
  llm-manager monitor [options]

OPTIONS:
  --pid <pid>             Monitor a specific process id
  --interval-ms <n>       Poll interval in milliseconds (default: 1000)
  --warning <size>        Warning threshold (e.g. 512mb)
  --critical <size>       Critical threshold (e.g. 1gb)
  --log-file <path>       Append events to a log file
  --pressure-events       Use macOS memory pressure events
  --help                  Show this help
"
    );
}
