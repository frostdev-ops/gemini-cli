use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::logging::log_debug;
use gemini_core::client::GeminiClient;
use gemini_core::types::{Content, GenerateContentRequest, GenerationConfig, Part};

// --- Constants --- //
pub const TOKEN_THRESHOLD: usize = 700000; // 700k tokens 
const CHARS_PER_TOKEN: usize = 4; // Estimate 4 chars per token as a heuristic

// --- Chat History Types --- //
pub type Role = String; // User-defined type for clarity

// Role constants
pub mod roles {
    pub const USER: &str = "user";
    pub const ASSISTANT: &str = "model"; // Gemini uses "model" for assistant
    pub const SYSTEM: &str = "system";
    pub const FUNCTION: &str = "function";
}

// History structure
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: Role, // "user" or "assistant"
    pub content: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ChatHistory {
    pub messages: Vec<ChatMessage>,
    pub session_id: String,
}

/// Generate a session ID unique to the current terminal session
pub fn generate_session_id() -> String {
    // First, try to get session ID from environment variable
    // This allows for explicit session control and persistence across commands
    if let Ok(session_id) = env::var("GEMINI_SESSION_ID") {
        if !session_id.is_empty() {
            return session_id;
        }
    }

    // Next, try to get a consistent terminal identifier
    // Use TERM_SESSION_ID on macOS/iTerm
    if let Ok(term_session) = env::var("TERM_SESSION_ID") {
        if !term_session.is_empty() {
            return format!("term_{}", term_session);
        }
    }

    // For Linux/other terminals, try using TTY
    if let Ok(output) = Command::new("sh").arg("-c").arg("tty").output() {
        if let Ok(tty) = String::from_utf8(output.stdout) {
            let tty = tty.trim();
            if !tty.is_empty() && tty != "not a tty" {
                // Normalize the TTY path to make it a valid filename component
                let normalized = tty.replace("/", "_").replace(" ", "_");
                return format!("tty_{}", normalized);
            }
        }
    }

    // Try using shell info as fallback
    let mut shell_info = String::new();

    // Try to get the shell's parent terminal PID
    if let Ok(output) = Command::new("sh")
        .arg("-c")
        .arg("ps -o ppid= -p $$")
        .output()
    {
        if let Ok(ppid) = String::from_utf8(output.stdout) {
            let ppid = ppid.trim();
            if !ppid.is_empty() {
                shell_info.push_str(&format!("term_{}", ppid));
            }
        }
    }

    // Add shell environment variables that should be consistent in a terminal session
    if let Ok(shlvl) = env::var("SHLVL") {
        if !shell_info.is_empty() {
            shell_info.push('_');
        }
        shell_info.push_str(&format!("lvl_{}", shlvl));
    }

    // Use the USER environment variable for additional stability
    if let Ok(user) = env::var("USER") {
        if !shell_info.is_empty() {
            shell_info.push('_');
        }
        shell_info.push_str(&format!("usr_{}", user));
    }

    // If we've gathered some shell info, use it
    if !shell_info.is_empty() {
        return shell_info;
    }

    // Ultimate fallback: use a stable identifier based on the day
    // This at least keeps conversations together within the same day
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert to day-based timestamp (seconds / (seconds per day))
    // 86400 = 60 * 60 * 24 = seconds in a day
    let day_timestamp = now / 86400;
    format!("day_{}", day_timestamp)
}

/// Get the path for a session's history file
pub fn get_history_file_path(config_dir: &Path, session_id: &str) -> PathBuf {
    // Sanitize session_id for filename
    let sanitized_session_id =
        session_id.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");
    config_dir.join(format!("history_{}.json", sanitized_session_id))
}

/// Load chat history from JSON file
pub fn load_chat_history(config_dir: &Path, session_id: &str) -> ChatHistory {
    let history_file_path = get_history_file_path(config_dir, session_id);

    match fs::read_to_string(&history_file_path) {
        Ok(json_str) => {
            match serde_json::from_str::<ChatHistory>(&json_str) {
                Ok(mut history) => {
                    // Ensure the loaded history has the correct session ID
                    history.session_id = session_id.to_string();
                    if env::var("GEMINI_DEBUG").is_ok() {
                        log_debug(&format!(
                            "Loaded history from: {}",
                            history_file_path.display()
                        ));
                    }
                    history
                }
                Err(e) => {
                    eprintln!(
                        "{}: {}. {} {}",
                        "Warning: Failed to parse history file".yellow(),
                        e,
                        "Creating new history for session:".yellow(),
                        session_id.cyan()
                    );
                    ChatHistory {
                        messages: Vec::new(),
                        session_id: session_id.to_string(),
                    }
                }
            }
        }
        Err(_) => {
            // File likely doesn't exist
            if env::var("GEMINI_DEBUG").is_ok() {
                log_debug(&format!(
                    "No history file found at: {} Starting new history.",
                    history_file_path.display()
                ));
            }
            ChatHistory {
                messages: Vec::new(),
                session_id: session_id.to_string(),
            }
        }
    }
}

/// Save chat history to JSON file
pub fn save_chat_history(config_dir: &Path, history: &ChatHistory) -> Result<(), Box<dyn Error>> {
    let history_file_path = get_history_file_path(config_dir, &history.session_id);
    let json_str = serde_json::to_string_pretty(history)?;
    fs::write(&history_file_path, json_str)?;
    if env::var("GEMINI_DEBUG").is_ok() {
        log_debug(&format!(
            "Saved history to: {}",
            history_file_path.display()
        ));
    }
    Ok(())
}

/// Start a new chat by removing existing history file
pub fn start_new_chat(config_dir: &Path, session_id: &str) {
    let history_file_path = get_history_file_path(config_dir, session_id);
    if history_file_path.exists() {
        if let Err(e) = fs::remove_file(&history_file_path) {
            eprintln!(
                "{}: {}: {}",
                "Warning: Failed to delete existing history file".yellow(),
                history_file_path.display(),
                e
            );
        } else {
            println!(
                "{} {}",
                "Started new chat (removed previous history file for session:".yellow(),
                session_id.cyan()
            );
        }
    }
}

/// Function to get the last N commands from shell history
#[allow(dead_code)]
pub fn get_recent_commands(count: usize) -> Vec<String> {
    let mut commands = Vec::new();

    // Try to determine shell type
    let shell = match env::var("SHELL") {
        Ok(s) => s,
        Err(_) => {
            if env::var("GEMINI_DEBUG").is_ok() {
                log_debug("Could not determine shell type from SHELL env var");
            }
            String::from("unknown")
        }
    };

    let is_zsh = shell.contains("zsh");

    if env::var("GEMINI_DEBUG").is_ok() {
        log_debug(&format!("Detected shell type: {}", shell));
    }

    // Read directly from history file - this provides access to the actual commands
    // rather than just command names from fc/history
    if env::var("GEMINI_DEBUG").is_ok() {
        log_debug("Reading command history directly from history file");
    }

    // Determine the history file path based on shell
    let history_path = if is_zsh {
        env::var("HISTFILE").unwrap_or_else(|_| {
            format!(
                "{}/.zsh_history",
                env::var("HOME").unwrap_or_else(|_| String::from("~"))
            )
        })
    } else {
        env::var("HISTFILE").unwrap_or_else(|_| {
            format!(
                "{}/.bash_history",
                env::var("HOME").unwrap_or_else(|_| String::from("~"))
            )
        })
    };

    if env::var("GEMINI_DEBUG").is_ok() {
        log_debug(&format!("Trying to read history from: {}", history_path));
    }

    if let Ok(file) = File::open(&history_path) {
        let reader = BufReader::new(file);
        // Use map_while to avoid potential infinite loop on read errors
        let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

        for line in lines.iter().rev() {
            if commands.len() >= count {
                break; // Stop once we have enough commands
            }
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let cmd = if is_zsh {
                // Zsh extended history format: ': <timestamp>:<duration>;<command>'
                trimmed_line
                    .split_once(';')
                    .map_or(trimmed_line, |(_meta, command)| command)
                    .trim()
            } else {
                // Bash history is simpler
                trimmed_line
            };

            commands.push(cmd.to_string());
        }

        if env::var("GEMINI_DEBUG").is_ok() {
            log_debug(&format!(
                "Found {} commands from history file",
                commands.len()
            ));
        }
    } else if env::var("GEMINI_DEBUG").is_ok() {
        log_debug(&format!("Could not open history file: {}", history_path));
    }

    // Filter out gemini commands and the command to run this tool
    commands.retain(|cmd| {
        !cmd.starts_with("gemini ")
            && !cmd.starts_with("./gemini ")
            && !cmd.contains("fc -l")
            && !cmd.contains("history")
            && !cmd.contains("gemini-cli-bin")
    });

    // Truncate to the requested count
    if commands.len() > count {
        commands.truncate(count);
    }

    if env::var("GEMINI_DEBUG").is_ok() && commands.is_empty() {
        log_debug("No recent commands found to add to system prompt");
    }

    commands
}

/// Create system prompt with recent command history context
#[allow(dead_code)]
pub fn create_system_prompt_with_history(base_prompt: &str) -> String {
    let recent_commands = get_recent_commands(5);

    if recent_commands.is_empty() {
        if env::var("GEMINI_DEBUG").is_ok() {
            log_debug("No recent commands found to add to system prompt");
        }
        return base_prompt.to_string();
    }

    let mut prompt = base_prompt.to_string();
    prompt.push_str("\n\nRecent command history:\n");

    for (i, cmd) in recent_commands.iter().enumerate() {
        prompt.push_str(&format!("{}. {}\n", i + 1, cmd));
        if env::var("GEMINI_DEBUG").is_ok() {
            log_debug(&format!("Adding command to system prompt: {}", cmd));
        }
    }

    prompt.push_str("\nPlease consider this command history when providing assistance.");

    if env::var("GEMINI_DEBUG").is_ok() {
        log_debug(&format!(
            "Added {} recent commands to system prompt",
            recent_commands.len()
        ));
        log_debug("SYSTEM PROMPT WITH HISTORY:");
        log_debug(&prompt);
    }

    prompt
}

/// Estimate the number of tokens in a string
/// This is a simple heuristic based on character count
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / CHARS_PER_TOKEN + 1 // +1 to avoid div by zero and round up
}

/// Estimate total tokens in a chat history
pub fn estimate_total_tokens(history: &ChatHistory, system_prompt: &str) -> usize {
    let mut total = estimate_tokens(system_prompt);

    for message in &history.messages {
        total += estimate_tokens(&message.content);
        // Add a small overhead for message formatting
        total += 4; // "role" field tokens
    }

    total
}

/// Summarize a long conversation history
pub async fn summarize_conversation(
    gemini_client: &GeminiClient, // Changed from client: &Client and removed api_key
    history: &ChatHistory,
) -> Result<ChatHistory, Box<dyn Error>> {
    log_debug("Summarizing conversation history...");
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
            .template("{spinner:.green} Summarizing history...")
            .unwrap(),
    );
    spinner.enable_steady_tick(Duration::from_millis(80));

    // Construct the summarization prompt
    let mut summarization_content = String::new();
    for msg in &history.messages {
        summarization_content.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }

    let prompt = format!(
        "Please summarize the following conversation concisely. The summary should retain the key information, decisions made, and the overall topic. Start the summary with 'This is a summarized version of a longer conversation:'\n\nCONVERSATION:\n\n{}",
        summarization_content
    );

    // Prepare request for Gemini API using GeminiClient
    let request = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![Part::text(prompt)],
            role: Some(roles::USER.to_string()), // Send prompt as user
        }],
        system_instruction: None, // No specific system prompt for summarization
        tools: None,
        generation_config: Some(GenerationConfig {
            temperature: Some(0.5), // Lower temp for factual summary
            ..Default::default()
        }),
    };

    match gemini_client.generate_content(request).await {
        Ok(response) => {
            spinner.finish_and_clear();
            let summary_text = gemini_client
                .extract_text_from_response(&response)
                .map_err(|e| format!("Failed to extract summary text: {}", e))?;

            // Create a new history with the summary
            let summary_message = ChatMessage {
                role: roles::SYSTEM.to_string(), // Store summary as system message
                content: summary_text,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let mut new_history = ChatHistory {
                messages: vec![summary_message],
                session_id: history.session_id.clone(),
            };

            // Keep the very last user message from the old history if it exists
            // This ensures the context for the immediate next turn isn't lost
            if let Some(last_user_message) = history
                .messages
                .iter()
                .rev()
                .find(|msg| msg.role == roles::USER)
            {
                if new_history.messages.last().map_or(true, |last_summary| {
                    last_summary.timestamp < last_user_message.timestamp
                }) {
                    new_history.messages.push(last_user_message.clone());
                }
            }
            // Keep the very last assistant message too, if it came after the last user message
            if let Some(last_assistant_message) = history
                .messages
                .iter()
                .rev()
                .find(|msg| msg.role == roles::ASSISTANT)
            {
                if new_history.messages.last().is_none_or(|last| {
                    last.timestamp < last_assistant_message.timestamp
                }) {
                    new_history.messages.push(last_assistant_message.clone());
                }
            }

            log_debug("Summarization successful.");
            Ok(new_history)
        }
        Err(e) => {
            spinner.finish_and_clear();
            Err(format!("Error during summarization API call: {}", e).into())
        }
    }
}

/// Log history debug information if GEMINI_DEBUG is set
#[allow(dead_code)]
pub fn log_history_debug(
    history: &ChatHistory,
    token_count: usize,
    was_summarized: bool,
    session_id: &str,
) {
    if std::env::var("GEMINI_DEBUG").is_err() {
        return;
    }

    println!("{}", "\n--- DEBUG INFO START ---".purple());

    println!("{} {}", "\nUsing session ID:".cyan(), session_id.yellow());

    if !history.messages.is_empty() {
        println!(
            "{}",
            format!(
                "\nConversation history (estimated tokens: {}):",
                token_count
            )
            .cyan()
        );
        if was_summarized {
            println!("{}", "(History was automatically summarized)".yellow());
        }
        println!("{}", "--------------------".purple());
        for (i, msg) in history.messages.iter().enumerate() {
            let role_colored = match msg.role.as_str() {
                "user" => msg.role.blue(),
                "assistant" => msg.role.green(),
                "system" => msg.role.yellow(),
                _ => msg.role.dimmed(),
            };
            println!("{}. [{}]: {}", i + 1, role_colored, msg.content);
        }
        println!("{}", "--------------------".purple());
    }
    println!("{}", "\n--- DEBUG INFO END ---".purple());
}
