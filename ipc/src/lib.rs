// This crate centralizes Inter-Process Communication definitions and logic.

pub mod daemon_messages; // For CLI <-> mcp-host communication
pub mod happe_request;
pub mod internal_messages; // For HAPPE <-> IDA communication
