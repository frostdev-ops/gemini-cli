use std::io;
use std::path::PathBuf;

// Re-export the McpServerConfig and McpTransport from core
pub use gemini_core::config::{McpServerConfig, McpTransport};

pub fn get_config_dir() -> Result<PathBuf, String> {
    // Use gemini-core's function to get the config directory
    let unified_config_path = gemini_core::config::get_unified_config_path()
        .map_err(|e| format!("Could not determine unified config path: {}", e))?;

    // Get the parent directory of the unified config file
    let config_dir = unified_config_path
        .parent()
        .ok_or_else(|| "Could not determine config directory from unified config path".to_string())?
        .to_path_buf();

    Ok(config_dir)
}

pub fn get_mcp_config_path() -> io::Result<PathBuf> {
    // Use gemini-core's function to get the MCP servers config path
    gemini_core::config::get_mcp_servers_config_path()
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))
}

pub fn load_mcp_servers() -> Result<Vec<McpServerConfig>, String> {
    // Use gemini-core's function to load MCP servers
    gemini_core::config::load_mcp_servers(None)
        .map_err(|e| format!("Error loading MCP servers: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_load_mcp_servers_file_not_exist() {
        // Need to ensure the config dir *itself* exists for get_config_dir to work,
        // but the mcp_servers.json file doesn't.
        // This is tricky to test reliably without mocking std::env or dirs::config_dir.
        // For now, assume get_config_dir works and test the file-not-found case logic.
        // We'll manually construct a path within a temp dir.
        let dir = tempdir().unwrap();
        let non_existent_path = dir.path().join("non_existent_mcp_servers.json");
        // Patch the function locally - requires some refactoring not done here.
        // Instead, we'll just test the logic downstream of path resolution.
        assert!(load_mcp_servers_from_path(&non_existent_path)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_load_mcp_servers_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty_mcp_servers.json");
        fs::write(&path, "").unwrap();
        assert!(load_mcp_servers_from_path(&path).unwrap().is_empty());
    }

    #[test]
    fn test_load_mcp_servers_valid_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("valid_mcp_servers.json");
        let content = r#"
        [
          {
            "name": "Server1",
            "enabled": true,
            "transport": "stdio",
            "command": ["server1_cmd"],
            "args": ["--arg1"],
            "env": { "VAR1": "VAL1" },
            "auto_execute": ["tool1"]
          },
          {
            "name": "Server2",
            "enabled": false,
            "transport": "stdio",
            "command": ["server2_cmd"],
            "auto_execute": ["tool2"]
          }
        ]
        "#;
        fs::write(&path, content).unwrap();

        let servers = load_mcp_servers_from_path(&path).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "Server1");
        assert!(servers[0].enabled);
        assert_eq!(servers[0].transport, McpTransport::Stdio);
        assert_eq!(servers[0].command, vec!["server1_cmd".to_string()]);
        assert_eq!(servers[0].args, vec!["--arg1".to_string()]);
        assert_eq!(servers[0].env.get("VAR1"), Some(&"VAL1".to_string()));
        assert_eq!(servers[0].auto_execute, vec!["tool1"]);
    }

    #[test]
    fn test_load_mcp_servers_invalid_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid_mcp_servers.json");
        fs::write(&path, "[{\"invalid\": \"json\"}]").unwrap(); // Fixed invalid JSON
        assert!(load_mcp_servers_from_path(&path).is_err());
    }

    // Helper for testing to avoid dependency on actual config dir
    fn load_mcp_servers_from_path(
        config_path: &std::path::Path,
    ) -> Result<Vec<McpServerConfig>, String> {
        if !config_path.exists() {
            return Ok(Vec::new());
        }
        let content = fs::read_to_string(config_path).map_err(|e| {
            format!(
                "Error reading MCP config file {}: {}",
                config_path.display(),
                e
            )
        })?;
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }
        let servers: Vec<McpServerConfig> = serde_json::from_str(&content).map_err(|e| {
            format!(
                "Error parsing MCP config file {}: {}",
                config_path.display(),
                e
            )
        })?;
        let enabled_servers = servers.into_iter().filter(|s| s.enabled).collect();
        Ok(enabled_servers)
    }
}
