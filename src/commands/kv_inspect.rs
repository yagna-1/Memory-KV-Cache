use std::path::PathBuf;
use std::process::Command;

use crate::gguf::{read_metadata, GgufValue};
use crate::profile::load_profile;

pub fn handle(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let mut model = None;
    let mut context_length = None;
    let mut precision = None;
    let mut json = false;
    let mut show_layers = false;
    let mut show_chart = false;
    let mut show_context_chart = false;
    let mut show_layer_chart = false;
    let mut available_ram = None;
    let mut auto_ram = false;
    let mut profile_name = None;
    let mut profile_dir = None;
    let mut context_step = 512u64;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(next_value("--model", &mut iter)?)),
            "--context-length" => {
                context_length =
                    Some(parse_u64("--context-length", &next_value("--context-length", &mut iter)?)?)
            }
            "--precision" => precision = Some(next_value("--precision", &mut iter)?),
            "--json" => json = true,
            "--layers" => show_layers = true,
            "--chart" => show_chart = true,
            "--context-chart" => show_context_chart = true,
            "--layer-chart" => show_layer_chart = true,
            "--profile" => profile_name = Some(next_value("--profile", &mut iter)?),
            "--profile-dir" => profile_dir = Some(PathBuf::from(next_value("--profile-dir", &mut iter)?)),
            "--context-step" => {
                context_step =
                    parse_u64("--context-step", &next_value("--context-step", &mut iter)?)?
            }
            "--available-ram" => {
                available_ram =
                    Some(parse_bytes("--available-ram", &next_value("--available-ram", &mut iter)?)?)
            }
            "--auto-ram" => auto_ram = true,
            other if other.starts_with('-') => {
                return Err(format!("Unknown flag for kv-inspect: {other}"));
            }
            _ => {}
        }
    }

    if let Some(profile_name) = profile_name {
        let (profile, _path) = load_profile(&profile_name, profile_dir.as_deref())?;
        if model.is_none() {
            if let Some(profile_model) = profile.model {
                if !profile_model.is_empty() {
                    model = Some(PathBuf::from(profile_model));
                }
            }
        }
        if context_length.is_none() {
            context_length = profile.context_length;
        }
        if precision.is_none() {
            precision = profile.precision;
        }
    }

    let Some(model_path) = model else {
        return Err("Missing --model for kv-inspect".to_string());
    };
    let context_length = context_length.unwrap_or(2048);
    let precision = precision.unwrap_or_else(|| "fp16".to_string()).to_lowercase();

    if auto_ram && available_ram.is_none() {
        available_ram = Some(
            detect_available_ram()
                .ok_or_else(|| "Failed to auto-detect available RAM".to_string())?,
        );
    }

    let metadata = read_metadata(&model_path)?;
    let architecture = find_string(
        &metadata,
        &["general.architecture", "llama.architecture", "model.architecture"],
    );
    let hidden_size = find_u64(
        &metadata,
        &[
            "llama.embedding_length",
            "llama.hidden_size",
            "llama.n_embd",
            "general.embedding_length",
        ],
    )
    .ok_or_else(|| "Missing hidden size in GGUF metadata".to_string())?;
    let num_layers = find_u64(
        &metadata,
        &[
            "llama.block_count",
            "llama.n_layer",
            "general.layer_count",
        ],
    )
    .ok_or_else(|| "Missing layer count in GGUF metadata".to_string())?;
    let head_count = find_u64(
        &metadata,
        &[
            "llama.attention.head_count",
            "llama.n_head",
            "general.head_count",
        ],
    );
    let head_count_kv = find_u64(
        &metadata,
        &[
            "llama.attention.head_count_kv",
            "llama.n_head_kv",
            "general.head_count_kv",
        ],
    );

    let bytes_per_element = match precision.as_str() {
        "fp16" => 2u64,
        "fp32" => 4u64,
        "int8" => 1u64,
        other => return Err(format!("Unsupported precision: {other}")),
    };

    let sizes = compute_kv_sizes(
        hidden_size,
        num_layers,
        context_length,
        bytes_per_element,
    )?;
    let kv_per_token_per_layer = sizes.kv_per_token_per_layer;
    let per_layer_k = sizes.per_layer_k;
    let per_layer_total = sizes.per_layer_total;
    let per_token_total = sizes.per_token_total;
    let total_kv = sizes.total_kv;

    if json {
        let mut out = String::new();
        out.push('{');
        push_json_field(&mut out, "model", &model_path.display().to_string(), true);
        if let Some(architecture) = &architecture {
            push_json_field(&mut out, "architecture", architecture, true);
        }
        push_json_field(&mut out, "hidden_size", &hidden_size.to_string(), false);
        if let Some(head_count) = head_count {
            push_json_field(&mut out, "head_count", &head_count.to_string(), false);
        }
        if let Some(head_count_kv) = head_count_kv {
            push_json_field(&mut out, "head_count_kv", &head_count_kv.to_string(), false);
        }
        push_json_field(&mut out, "num_layers", &num_layers.to_string(), false);
        push_json_field(&mut out, "context_length", &context_length.to_string(), false);
        push_json_field(&mut out, "precision", &precision, true);
        push_json_field(&mut out, "bytes_per_element", &bytes_per_element.to_string(), false);
        push_json_field(&mut out, "layer_k_bytes", &per_layer_k.to_string(), false);
        push_json_field(&mut out, "layer_v_bytes", &per_layer_k.to_string(), false);
        push_json_field(&mut out, "layer_total_bytes", &per_layer_total.to_string(), false);
        push_json_field(
            &mut out,
            "kv_per_token_per_layer",
            &kv_per_token_per_layer.to_string(),
            false,
        );
        push_json_field(&mut out, "kv_per_token_total", &per_token_total.to_string(), false);
        push_json_field(&mut out, "total_kv_bytes", &total_kv.to_string(), false);
        if let Some(available_ram) = available_ram {
            let max_tokens = available_ram / per_token_total;
            push_json_field(&mut out, "available_ram_bytes", &available_ram.to_string(), false);
            push_json_field(&mut out, "max_tokens_estimate", &max_tokens.to_string(), false);
        }
        if show_context_chart {
            push_json_context_chart(
                &mut out,
                context_length,
                per_token_total,
                available_ram,
                context_step,
            );
        }
        if show_layer_chart {
            push_json_layer_chart(&mut out, num_layers, per_layer_total);
        }
        if out.ends_with(',') {
            out.pop();
        }
        out.push('}');
        println!("{out}");
        return Ok(());
    }

    println!("Model: {}", model_path.display());
    if let Some(architecture) = &architecture {
        println!("Architecture: {architecture}");
    }
    println!("GGUF version: {}", metadata.version);
    println!("Hidden size: {hidden_size}");
    if let Some(head_count) = head_count {
        println!("Head count: {head_count}");
    }
    if let Some(head_count_kv) = head_count_kv {
        println!("Head count KV: {head_count_kv}");
    }
    println!("Layers: {num_layers}");
    println!("Context length: {context_length}");
    println!("Precision: {precision}");
    println!("Bytes per element: {bytes_per_element}");
    println!(
        "Per-layer KV per token: {}",
        format_bytes(kv_per_token_per_layer)
    );
    println!(
        "Per-layer KV (context length): {}",
        format_bytes(per_layer_total)
    );
    println!(
        "Total KV per token: {}",
        format_bytes(per_token_total)
    );
    println!("Total KV cache: {}", format_bytes(total_kv));

    if let Some(available_ram) = available_ram {
        let max_tokens = available_ram / per_token_total;
        println!(
            "Available RAM: {} (est. max tokens: {})",
            format_bytes(available_ram),
            max_tokens
        );
        println!(
            "With {} free RAM, you can handle ~{} tokens.",
            format_bytes(available_ram),
            max_tokens
        );
    }

    if show_layers {
        println!("\nPer-layer KV breakdown:");
        println!("{:<6} {:>12} {:>12} {:>12}", "Layer", "K", "V", "Total");
        let view = layer_view(num_layers);
        let omitted = omitted_layers(num_layers);
        for entry in view {
            match entry {
                Some(layer) => {
                    println!(
                        "{:<6} {:>12} {:>12} {:>12}",
                        layer,
                        format_bytes(per_layer_k),
                        format_bytes(per_layer_k),
                        format_bytes(per_layer_total)
                    );
                }
                None => {
                    println!("  ...");
                }
            }
        }
        if omitted > 0 {
            println!("(omitted {omitted} layers)");
        }
    }

    if show_layer_chart {
        println!("\nPer-layer KV bar chart:");
        print_layer_chart(num_layers, per_layer_total, total_kv);
    }

    if show_chart {
        println!("\nCumulative KV cache by layer:");
        let view = layer_view(num_layers);
        let omitted = omitted_layers(num_layers);
        for entry in view {
            if let Some(layer) = entry {
                let bar = cumulative_bar(layer, num_layers, 24);
                let pct = (layer as f64 / num_layers as f64) * 100.0;
                println!("{:<6} {} {:>5.1}%", layer, bar, pct);
            } else {
                println!("  ...");
            }
        }
        if omitted > 0 {
            println!("(omitted {omitted} layers)");
        }
    }

    if show_context_chart {
        println!("\nKV cache by context length:");
        print_context_chart(
            context_length,
            per_token_total,
            available_ram,
            context_step,
        );
    }

    Ok(())
}

fn find_u64(metadata: &crate::gguf::GgufMetadata, keys: &[&str]) -> Option<u64> {
    for key in keys {
        if let Some(value) = metadata.kv.get(*key) {
            if let Some(parsed) = value.as_u64() {
                return Some(parsed);
            }
        }
    }
    None
}

fn find_string(metadata: &crate::gguf::GgufMetadata, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = metadata.kv.get(*key) {
            if let GgufValue::String(parsed) = value {
                if !parsed.is_empty() {
                    return Some(parsed.clone());
                }
            }
        }
    }
    None
}

struct KvSizes {
    kv_per_token_per_layer: u64,
    per_layer_k: u64,
    per_layer_total: u64,
    per_token_total: u64,
    total_kv: u64,
}

fn compute_kv_sizes(
    hidden_size: u64,
    num_layers: u64,
    context_length: u64,
    bytes_per_element: u64,
) -> Result<KvSizes, String> {
    let kv_per_token_per_layer = 2u64
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(bytes_per_element))
        .ok_or_else(|| "Overflow computing per-token KV size".to_string())?;
    let per_layer_k = hidden_size
        .checked_mul(bytes_per_element)
        .and_then(|v| v.checked_mul(context_length))
        .ok_or_else(|| "Overflow computing per-layer K size".to_string())?;
    let per_layer_total = kv_per_token_per_layer
        .checked_mul(context_length)
        .ok_or_else(|| "Overflow computing per-layer total size".to_string())?;
    let per_token_total = num_layers
        .checked_mul(kv_per_token_per_layer)
        .ok_or_else(|| "Overflow computing per-token KV total".to_string())?;
    let total_kv = per_token_total
        .checked_mul(context_length)
        .ok_or_else(|| "Overflow computing total KV size".to_string())?;

    Ok(KvSizes {
        kv_per_token_per_layer,
        per_layer_k,
        per_layer_total,
        per_token_total,
        total_kv,
    })
}

fn next_value(flag: &str, iter: &mut impl Iterator<Item = String>) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("Missing value for {flag}"))
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

fn push_json_field(out: &mut String, key: &str, value: &str, quoted: bool) {
    out.push('"');
    out.push_str(&escape_json(key));
    out.push_str("\":");
    if quoted {
        out.push('"');
        out.push_str(&escape_json(value));
        out.push('"');
    } else {
        out.push_str(value);
    }
    out.push(',');
}

fn escape_json(input: &str) -> String {
    input
        .chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            other => vec![other],
        })
        .collect()
}

fn layer_view(num_layers: u64) -> Vec<Option<u64>> {
    let max_rows = 16usize;
    if num_layers as usize <= max_rows {
        return (1..=num_layers).map(Some).collect();
    }

    let head = 8u64;
    let tail = 4u64;
    let mut view = Vec::new();
    for i in 1..=head {
        view.push(Some(i));
    }
    view.push(None);
    for i in (num_layers - tail + 1)..=num_layers {
        view.push(Some(i));
    }
    view
}

fn omitted_layers(num_layers: u64) -> u64 {
    let max_rows = 16u64;
    if num_layers <= max_rows {
        0
    } else {
        num_layers.saturating_sub(12)
    }
}

fn cumulative_bar(layer: u64, total_layers: u64, width: usize) -> String {
    if total_layers == 0 {
        return String::from("|");
    }
    let ratio = layer as f64 / total_layers as f64;
    let filled = (ratio * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    bar.push_str(&"#".repeat(filled));
    bar.push_str(&".".repeat(empty));
    bar.push(']');
    bar
}

fn print_help() {
    println!(
        "USAGE:
  llm-manager kv-inspect --model <file.gguf> [options]

OPTIONS:
  --model <path>           GGUF model file
  --context-length <n>     Context length (default: 2048)
  --precision <fp16|fp32|int8>
  --profile <name>        Load profile for defaults
  --profile-dir <path>    Search this directory for profiles
  --layers                 Show per-layer table
  --chart                  Show cumulative layer chart
  --context-chart          Show memory vs context length chart
  --context-step <n>        Step size for context chart (default: 512)
  --layer-chart            Show per-layer memory bar chart
  --available-ram <size>   Estimate max tokens (e.g. 4gb)
  --auto-ram               Auto-detect available RAM (macOS)
  --json                   JSON output
  --help                   Show this help
"
    );
}

fn print_context_chart(
    context_length: u64,
    per_token_total: u64,
    available_ram: Option<u64>,
    step: u64,
) {
    let step = step.max(1);
    let max_context = context_length.max(1);
    let max_total = per_token_total.saturating_mul(max_context);
    let mut limit_hit = None;
    let mut warned = false;

    println!("{:<8} {:>12} {}", "Context", "KV Cache", "Usage");
    let mut ctx = step;
    while ctx <= max_context {
        let total = per_token_total.saturating_mul(ctx);
        let ratio = available_ram
            .map(|ram| total as f64 / ram as f64)
            .unwrap_or_else(|| total as f64 / max_total as f64);
        let line = ratio_line(ratio, 24);
        println!("{:<8} {:>12} {}", ctx, format_bytes(total), line);

        if let Some(ram) = available_ram {
            if total >= ram && limit_hit.is_none() {
                limit_hit = Some(ctx);
            }
            if !warned && ratio >= 0.9 {
                println!("(warning: approaching available RAM limit)");
                warned = true;
            }
        }

        ctx = ctx.saturating_add(step);
    }

    if let Some(ram) = available_ram {
        if let Some(hit) = limit_hit {
            println!(
                "Limit reached around {} tokens for {} RAM.",
                hit,
                format_bytes(ram)
            );
        }
    }
}

fn ratio_bar(ratio: f64, width: usize) -> String {
    let clamped = ratio.max(0.0).min(1.0);
    let filled = (clamped * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    bar.push_str(&"#".repeat(filled));
    bar.push_str(&".".repeat(empty));
    bar.push(']');
    bar
}

#[cfg(test)]
mod tests {
    use super::compute_kv_sizes;

    #[test]
    fn compute_kv_sizes_fp16_example() {
        let sizes = compute_kv_sizes(4096, 32, 2048, 2).unwrap();
        assert_eq!(sizes.total_kv, 1_073_741_824);
        assert_eq!(sizes.per_token_total, 524_288);
    }
}

fn ratio_line(ratio: f64, width: usize) -> String {
    let clamped = ratio.max(0.0).min(1.0);
    let pos = (clamped * (width.saturating_sub(1)) as f64).round() as usize;
    let mut line = String::with_capacity(width + 2);
    line.push('[');
    for i in 0..width {
        if i == pos {
            line.push('*');
        } else {
            line.push('-');
        }
    }
    line.push(']');
    line
}

fn print_layer_chart(num_layers: u64, per_layer_total: u64, total_kv: u64) {
    let view = layer_view(num_layers);
    let omitted = omitted_layers(num_layers);
    for entry in view {
        match entry {
            Some(layer) => {
                let ratio = if total_kv == 0 {
                    0.0
                } else {
                    per_layer_total as f64 / total_kv as f64
                };
                let bar = ratio_bar(ratio, 24);
                println!(
                    "{:<6} {} {:>12}",
                    layer,
                    bar,
                    format_bytes(per_layer_total)
                );
            }
            None => {
                println!("  ...");
            }
        }
    }
    if omitted > 0 {
        println!("(omitted {omitted} layers)");
    }
}

fn push_json_context_chart(
    out: &mut String,
    context_length: u64,
    per_token_total: u64,
    available_ram: Option<u64>,
    step: u64,
) {
    let step = step.max(1);
    let max_context = context_length.max(1);
    out.push_str("\"context_chart\":[");
    let mut first = true;
    let mut ctx = step;
    while ctx <= max_context {
        let total = per_token_total.saturating_mul(ctx);
        if !first {
            out.push(',');
        }
        first = false;
        out.push('{');
        out.push_str("\"context_length\":");
        out.push_str(&ctx.to_string());
        out.push_str(",\"total_kv_bytes\":");
        out.push_str(&total.to_string());
        if let Some(ram) = available_ram {
            let ratio = if ram == 0 {
                0.0
            } else {
                (total as f64 / ram as f64).min(1.0)
            };
            out.push_str(",\"ratio\":");
            out.push_str(&format!("{:.4}", ratio));
        }
        out.push('}');
        ctx = ctx.saturating_add(step);
    }
    out.push_str("],");
}

fn push_json_layer_chart(out: &mut String, num_layers: u64, per_layer_total: u64) {
    out.push_str("\"layer_chart\":[");
    let mut first = true;
    for layer in 1..=num_layers {
        if !first {
            out.push(',');
        }
        first = false;
        out.push('{');
        out.push_str("\"layer\":");
        out.push_str(&layer.to_string());
        out.push_str(",\"total_kv_bytes\":");
        out.push_str(&per_layer_total.to_string());
        out.push('}');
    }
    out.push_str("],");
}

fn detect_available_ram() -> Option<u64> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut page_size = None;
    let mut free_pages = 0u64;
    let mut inactive_pages = 0u64;
    let mut speculative_pages = 0u64;

    for line in stdout.lines() {
        if line.contains("page size of") {
            page_size = extract_first_number(line);
        } else if line.starts_with("Pages free") {
            free_pages = extract_first_number(line).unwrap_or(0);
        } else if line.starts_with("Pages inactive") {
            inactive_pages = extract_first_number(line).unwrap_or(0);
        } else if line.starts_with("Pages speculative") {
            speculative_pages = extract_first_number(line).unwrap_or(0);
        }
    }

    let page_size = page_size.unwrap_or(4096);
    let total_pages = free_pages + inactive_pages + speculative_pages;
    Some(total_pages.saturating_mul(page_size))
}

fn extract_first_number(line: &str) -> Option<u64> {
    let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u64>().ok()
    }
}
