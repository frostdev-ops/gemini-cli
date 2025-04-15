use crate::mcp::rpc::{Resource, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use log::debug;
use std::collections::HashMap;
use colored::Colorize;

/// Represents a Gemini function parameter
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionParameter {
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, FunctionParameter>>,
}

/// Represents a Gemini function definition
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

/// Represents a Gemini function call detected in the response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    #[serde(rename = "args")]
    pub arguments: Value,
}

/// Converts MCP Tool capabilities to Gemini function definitions
pub fn convert_mcp_tools_to_gemini_functions(tools: &[Tool]) -> Vec<FunctionDef> {
    let mut functions = Vec::new();
    
    for tool in tools {
        // Use the parameters field if available, otherwise create a minimal schema
        let parameters = if let Some(params) = &tool.parameters {
            params.clone()
        } else {
            // Default to an empty object schema
            json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        };
        
        let description = tool.description.clone().unwrap_or_else(|| 
            "No description provided".to_string()
        );
        
        // Format function name for Gemini compatibility
        // Replace slashes with dots to maintain namespacing without violating Gemini's naming rules
        let gemini_function_name = tool.name.clone().replace("/", ".");
        
        functions.push(FunctionDef {
            name: gemini_function_name,
            description: Some(description),
            parameters: parameters,
        });
    }
    
    functions
}

/// Builds a system prompt with MCP capabilities
pub fn build_mcp_system_prompt(tools: &[Tool], resources: &[Resource]) -> String {
    let mut prompt = String::from("\n\nYou have access to the following tools and resources through a Machine Capability Protocol (MCP) interface. Use the function calling capability to interact with these tools; DO NOT suggest or describe function calls in your text response.\n\n");
    
    // Add Tools Section
    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        for tool in tools {
            let description = tool.description.clone().unwrap_or_else(|| 
                "No description provided".to_string()
            );
            
            // Use the dot-notation for display to match what Gemini will use
            let display_name = tool.name.replace("/", ".");
            prompt.push_str(&format!("* **{}**: {}\n", display_name, description));
        }
        prompt.push_str("\n");
    }
    
    // Add Resources Section
    if !resources.is_empty() {
        prompt.push_str("## Available Resources\n\n");
        for resource in resources {
            let resource_desc = resource.description.clone().unwrap_or_else(|| 
                "No description provided".to_string()
            );
            prompt.push_str(&format!("* **{}**: {}\n", resource.name, resource_desc));
        }
        prompt.push_str("\n");
    }
    
    // Remove the old text-based instructions since we want structured function calls
    prompt.push_str("Important: Always use the function calling capability of the API, not text-based suggestions. DO NOT write code blocks with JSON in your response.");
    
    prompt
}

/// Legacy function call parsing from response text - only needed as a fallback when the model
/// doesn't return proper function call objects in the structured JSON response.
/// This function is deprecated and should only be used when the structured JSON parsing fails.
pub fn parse_function_calls(response_text: &str) -> Vec<FunctionCall> {
    // Gemini's function call format is in markdown code blocks like:
    // ```json
    // {
    //   "name": "function_name",
    //   "arguments": { ... }
    // }
    // ```
    
    let mut function_calls = Vec::new();
    let mut in_code_block = false;
    let mut current_block = String::new();
    
    for line in response_text.lines() {
        let trimmed = line.trim();
        
        if trimmed.starts_with("```") {
            if !in_code_block {
                // Start a new code block
                in_code_block = true;
                current_block.clear();
                // Skip this line if it contains the language identifier
                if trimmed.len() > 3 {
                    continue;
                }
            } else {
                // End of code block - try to parse it
                in_code_block = false;
                
                match serde_json::from_str::<FunctionCall>(&current_block) {
                    Ok(call) => {
                        function_calls.push(call);
                    },
                    Err(e) => {
                        // Not a valid function call JSON block
                        debug!("Failed to parse JSON as function call: {}", e);
                        // Could be just a normal code block, so we ignore the error
                    }
                }
                
                current_block.clear();
            }
        } else if in_code_block {
            // Accumulate the code block content
            current_block.push_str(line);
            current_block.push('\n');
        }
    }
    
    function_calls
}

/// Generate Gemini API function declarations 
pub fn generate_gemini_function_declarations(tools: &[Tool]) -> Option<Vec<FunctionDef>> {
    let functions = convert_mcp_tools_to_gemini_functions(tools);
    
    if functions.is_empty() {
        None
    } else {
        Some(functions)
    }
}

/// Process a detected function call from Gemini and ask for user confirmation
pub async fn process_function_call(
    function_call: &FunctionCall,
    mcp_host: &crate::mcp::host::McpHost
) -> Result<Value, String> {
    // Extract the server and tool name from the qualified name
    // Convert dots back to slashes for MCP host compatibility
    let qualified_name = &function_call.name.replace(".", "/");
    if std::env::var("GEMINI_DEBUG").is_ok() {
        println!("[DEBUG] Processing function call: original name='{}', converted name='{}'", 
            &function_call.name, qualified_name);
    }
    
    // Parse the qualified name into server and tool parts
    let parts: Vec<&str> = qualified_name.splitn(2, "/").collect();
    if parts.len() != 2 {
        return Err(format!("Invalid qualified tool name: {}", qualified_name));
    }
    let server_name = parts[0];
    let tool_name = parts[1];
    
    // Check if this tool is in the auto-execute list for this server
    let should_auto_execute = mcp_host.is_auto_execute(server_name, tool_name).await;
    
    if !should_auto_execute {
        // Ask for user confirmation
        println!("\n{} {}:", "Tool execution requested:".yellow().bold(), qualified_name.green());
        // Pretty print the arguments
        if let Ok(pretty_json) = serde_json::to_string_pretty(&function_call.arguments) {
            println!("{}", pretty_json);
        } else {
            println!("{:?}", function_call.arguments);
        }
        
        let mut confirmation_input = String::new();
        println!("{}", "Do you want to allow this tool execution? [Y/n/a(lways)]".cyan());
        std::io::stdin().read_line(&mut confirmation_input).map_err(|e| format!("Failed to read confirmation: {}", e))?;
        
        let input = confirmation_input.trim().to_lowercase();
        if input == "n" {
            return Err(format!("User denied execution of tool: {}", qualified_name));
        } else if input == "a" || input == "always" {
            // Add this tool to the auto-execute list
            mcp_host.add_to_auto_execute(server_name, tool_name).await?;
        }
    }
    
    if std::env::var("GEMINI_DEBUG").is_ok() {
        println!("[DEBUG] Executing tool: server='{}', tool='{}'", server_name, tool_name);
    }

    mcp_host.execute_tool(server_name, tool_name, function_call.arguments.clone()).await
} 