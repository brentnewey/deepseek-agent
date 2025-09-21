# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local coding agent powered by DeepSeek V2.5 that runs entirely via Ollama. It provides an interactive CLI for code generation, explanation, review, and file operations.

## Commands

### Development
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test file
pytest test_agent.py

# Format code
black src/
ruff src/

# Type checking
mypy src/
```

### Running the Agent
```bash
# Start interactive chat
deepseek-agent chat

# With specific workspace
deepseek-agent chat --workspace /path/to/project

# Windows batch script
run_agent.bat

# Run directly with Python
python src/deepseek_agent/cli.py chat
```

## Architecture

### Core Components

1. **DeepSeekClient** (`src/deepseek_agent/model/client.py`):
   - Handles communication with Ollama API
   - Manages model interactions and streaming responses
   - Implements model normalization and validation

2. **AgentCLI** (`src/deepseek_agent/cli.py`):
   - Main CLI interface and command parsing
   - Natural language command interpretation
   - Shell command execution support
   - Rich terminal formatting and display

3. **FileOperations** (`src/deepseek_agent/tools/file_ops.py`):
   - Safe file system operations within workspace
   - Gitignore-aware file filtering
   - Path safety validation

### Key Design Patterns

- **Workspace Isolation**: All file operations are restricted to a designated workspace directory
- **Streaming Responses**: Uses async generators for real-time model output
- **Natural Language Commands**: Supports both structured commands and natural language requests
- **Safety First**: Validates all paths and respects .gitignore patterns

### Test Structure

- `test_agent.py`: Main agent functionality tests
- `test_with_mock_ollama.py`: Tests with mocked Ollama responses
- `test_real_ollama.py`: Integration tests with actual Ollama
- `test_sample.py`: Basic sample tests

## Dependencies

- **ollama**: Local LLM interaction (requires Ollama service running on localhost:11434)
- **httpx**: Async HTTP client for API communication
- **rich**: Terminal formatting and UI
- **typer**: CLI framework
- **pathspec**: Gitignore pattern matching
- **pydantic**: Data validation and models

## Important Notes

- Default model is `deepseek-v2.5` (requires 120GB+ RAM)
- For memory-constrained systems, use smaller models with `--model` flag
- Shell commands are executed in workspace context
- The agent suggests shell commands for users to run rather than executing directly