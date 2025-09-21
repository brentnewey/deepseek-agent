# DeepSeek Agent

A local coding agent powered by DeepSeek V2.5 that runs entirely on your machine via Ollama.

## Features

- ðŸ¤– **Local AI**: Runs DeepSeek V2.5 locally via Ollama - no cloud dependencies
- ðŸ’» **Code Generation**: Generate code from natural language prompts
- ðŸ“– **Code Explanation**: Understand complex codebases with AI-powered explanations
- ðŸ” **Code Review**: Get detailed code reviews with suggestions for improvements
- ðŸ“ **File Operations**: Safe file system operations within your workspace
- ðŸ’¬ **Interactive Chat**: Natural conversation interface for coding assistance
- ðŸŽ¨ **Syntax Highlighting**: Beautiful code display with Rich terminal formatting

## Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Python 3.10+**: Make sure you have Python 3.10 or later installed

### Installation

1. **Clone and install**:
   ```bash
   cd deepseek-agent
   pip install -e .
   ```

2. **Pull DeepSeek V2.5** (will be done automatically on first run):
   ```bash
   ollama pull deepseek-v2.5
   ```
   *Tip*: DeepSeek V2.5 is a large model that needs over 120GB of RAM. If you are memory constrained, pull a smaller Ollama model (for example `deepseek-r1:1.5b`) and pass it with `--model` when starting the agent.

3. **Start the agent**:
   ```bash
   deepseek-agent chat
   ```

### Usage Examples

#### Interactive Chat Mode
```bash
deepseek-agent chat --workspace /path/to/your/project
```

#### Generate Code
```bash
deepseek-agent generate "Create a Python function to merge two sorted lists" --output merge.py
```

#### Review Code
```bash
deepseek-agent review src/main.py
```

## Interactive Commands

Once in chat mode, you can use these commands:

- `read <file>` - Read and display file content with syntax highlighting
- `ls [directory]` - List directory contents
- `find <pattern>` - Find files matching a pattern
- `generate <prompt>` - Generate code from a description
- `explain <file>` - Get an explanation of code in a file
- `review <file>` - Get a detailed code review\n- ! <command> or shell <command> - Run shell commands in the current workspace
- `help` - Show available commands
- `quit` - Exit the agent

You can also just chat naturally! Ask questions, request code changes, or discuss programming concepts.

## Configuration

### Workspace Safety
The agent operates within a designated workspace for security. All file operations are restricted to this workspace and respect `.gitignore` patterns.

### Model Configuration
- **Default Model**: `deepseek-v2.5`
- **Context Window**: 32K tokens
- **Temperature**: 0.7 (adjustable)

## Development

### Project Structure
```
deepseek-agent/
â”œâ”€â”€ src/deepseek_agent/
â”‚   â”œâ”€â”€ model/          # DeepSeek integration
â”‚   â”œâ”€â”€ tools/          # File operations and utilities  
â”‚   â”œâ”€â”€ cli.py          # Command line interface
â”‚   â””â”€â”€ __init__.py
â”œâ”€â”€ pyproject.toml      # Project configuration
â”œâ”€â”€ requirements.txt    # Dependencies
â””â”€â”€ README.md
```

### Development Setup
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff src/

# Type checking
mypy src/
```

## System Requirements

- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: 10GB for DeepSeek V2.5 model
- **OS**: Linux, macOS, or Windows
- **Ollama**: Version 0.3.0 or later

## Security

- All file operations are restricted to the specified workspace
- Respects `.gitignore` patterns to avoid sensitive files
- No network requests except to local Ollama instance
- Sandboxed execution environment (coming soon)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

