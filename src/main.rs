mod commands;
mod llm_backend;
mod memory_monitor;
mod memory_pressure;
mod backends;
mod gguf;
mod ollama_config;
mod profile;

use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_usage();
        std::process::exit(1);
    };

    let cmd_args: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "run" => commands::run::handle(cmd_args),
        "kv-inspect" => commands::kv_inspect::handle(cmd_args),
        "config" => commands::config::handle(cmd_args),
        "monitor" => commands::monitor::handle(cmd_args),
        "version" => {
            println!("llm-manager {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("Unknown command: {other}")),
    };

    if let Err(err) = result {
        eprintln!("Error: {err}");
        std::process::exit(2);
    }
}

fn print_usage() {
    println!(
        "llm-manager

USAGE:
  llm-manager <command> [options]

COMMANDS:
  run         Run an LLM runtime with memory supervision
  kv-inspect  Inspect GGUF metadata and KV-cache usage
  config      Show or edit runtime profiles
  monitor     Poll and print memory usage
  version     Print version
  help        Show this help

For command help:
  llm-manager <command> --help
"
    );
}
