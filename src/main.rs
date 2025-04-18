use clap::Parser;
use dotenv::dotenv;
use reqwest::Client;
use std::env;
use std::error::Error;
use std::fs;
use colored::*;
use env_logger;

mod cli;
mod config;
mod history;
mod logging;
mod mcp;
mod model;
mod output;
mod app;
mod memory_broker;
mod auto_memory;

use crate::cli::Args;
use crate::config::{get_config_dir, get_config_file_path, 
                   load_config, handle_config_flags};
use crate::history::generate_session_id;
use crate::logging::{log_error, log_info};
use crate::mcp::config::load_mcp_servers;
use crate::mcp::host::McpHost;
use crate::output::print_usage_instructions;

/// Main function - handle command line args and talk to Gemini API
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logger
    env_logger::init();
    
    // Load environment variables
    dotenv().ok();
    
    // Parse command-line arguments
    let args = Args::parse();
    
    // Check if running as filesystem MCP server
    if args.filesystem_mcp {
        // Run the filesystem MCP server implementation
        // return run_filesystem_mcp_server().await;
        return crate::mcp::servers::filesystem::run().await;
    }
    
    // Check if running as command MCP server
    if args.command_mcp {
        // Run the command MCP server implementation
        // return run_command_mcp_server().await;
        return crate::mcp::servers::command::run().await;
    }

    // Check if running as memory store MCP server
    if args.memory_store_mcp {
        // Run the memory store MCP server implementation
        logging::log_info("Starting memory store MCP server...");
        return crate::mcp::servers::memory_store::run().await;
    }

    // Get the configuration directory
    let config_dir = get_config_dir()?;
    fs::create_dir_all(&config_dir)?;
    
    // Get/load config
    let config_path = get_config_file_path(&config_dir);
    let mut config = load_config(&config_path)?;
    
    // Handle config-related flags
    if handle_config_flags(&args, &mut config, &config_path)? {
        return Ok(());
    }
    
    // Validate API key
    let api_key = match config.api_key {
        Some(ref key) if !key.is_empty() => key.to_string(),
        _ => match env::var("GEMINI_API_KEY") {
            Ok(key) if !key.is_empty() => key,
            _ => {
                eprintln!("{}", "No API key found. Set it with --set-api-key or GEMINI_API_KEY env var.".red());
                return Ok(());
            }
        },
    };

    // Create HTTP client
    let client = Client::new();
    
    // Should we save history?
    let should_save_history = config.save_history.unwrap_or(true) && !args.disable_history;
    
    // Get system prompt (default or custom)
    let system_prompt = config.system_prompt.clone().unwrap_or_else(|| {
        "You are a helpful assistant that lives in the command line interface. You are friendly, and a professional programmer and developer.".to_string()
    });
    
    // Get/create chat history
    let session_id = generate_session_id();
    
    // Print the export command when running in debug mode or when session ID is day-based
    // This helps users maintain conversation continuity
    if (env::var("DEBUG").is_ok() || session_id.starts_with("day_")) && !args.disable_history && args.prompt.is_some() {
        println!("{}", "\nTo maintain chat history across commands, run:".cyan());
        println!("export GEMINI_SESSION_ID=\"{}\"", session_id);
        println!();
    }
    
    // Initialize MCP server host if any are configured
    let mut mcp_host: Option<crate::mcp::host::McpHost> = None;
    if let Ok(server_configs) = load_mcp_servers() {
        match McpHost::new(server_configs).await {
            Ok(host) => {
                // Demonstrate resource access through MCP
                if let Ok(system_info) = host.get_system_info().await {
                    log_info(&format!("Resource access test: {}", system_info));
                }
                
                // Send log message to servers
                host.log_to_servers("Gemini CLI started", 3).await;
                
                mcp_host = Some(host);
                log_info("MCP server host initialized successfully");
            },
            Err(e) => {
                log_error(&format!("Failed to create McpHost: {}", e));
            }
        }
    }
    
    // Check for interactive mode
    if args.interactive {
        if let Err(e) = crate::app::run_interactive_chat(
            &args,
            &config,
            &client,
            &mcp_host,
            &api_key,
            &system_prompt,
            &config_dir,
            &session_id,
            should_save_history,
        ).await {
            eprintln!("Error in interactive chat: {}", e);
        }
    } 
    // Check for task loop mode
    else if let Some(task) = &args.task {
        if let Err(e) = crate::app::run_task_loop(
            &args,
            &config,
            &client,
            &mcp_host,
            &api_key,
            &system_prompt,
            &config_dir,
            &session_id,
            should_save_history,
            task,
        ).await {
            eprintln!("Error in task loop: {}", e);
        }
    }
    // Single prompt mode
    else if let Some(prompt) = args.prompt.clone() {
        // Call the main processing logic from the app module
        if let Err(e) = crate::app::process_prompt(
            &args,
            &config,
            &client,
            &mcp_host,
            &api_key,
            &system_prompt,
            &config_dir,
            &session_id,
            should_save_history,
            &prompt,
        ).await {
            eprintln!("Error processing prompt: {}", e);
        }
    } else {
        // No prompt provided, show usage instructions
        print_usage_instructions();
    }
    
    // Clean up MCP resources if needed
    if let Some(host) = mcp_host {
        host.shutdown().await;
    }
    
    Ok(())
}

