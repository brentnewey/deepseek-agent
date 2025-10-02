"""Command line interface for DeepSeek Agent with tool calling support"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner

from deepseek_agent.model.client import DeepSeekClient, Message
from deepseek_agent.tools.file_ops import FileOperations
from deepseek_agent.tools.tool_definitions import get_tool_definitions
from deepseek_agent.tools.tool_executor import ToolExecutor
from deepseek_agent.utils import ensure_ollama_running, get_ollama_install_instructions

app = typer.Typer(help="Local coding agent powered by DeepSeek models with tool calling")
console = Console()


class ToolAgentCLI:
    """Interactive CLI for DeepSeek Agent with tool calling"""

    def __init__(self, workspace: Optional[str] = None, model: str = "deepseek-v2.5", enable_logging: bool = True):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.model = model
        self.file_ops = FileOperations(str(self.workspace))
        self.tool_executor = ToolExecutor(self.workspace)
        self.conversation_history = []

        # Setup session logging
        self.enable_logging = enable_logging
        self.log_file = None
        if enable_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.workspace / f"agent_session_{timestamp}.log"
            self._log(f"=== DeepSeek Agent Session Started ===")
            self._log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log(f"Model: {model}")
            self._log(f"Workspace: {self.workspace}")
            self._log("=" * 50 + "\n")

        # For mistral, use only write_file and run_command tools (too many tools confuses it)
        all_tools = get_tool_definitions()
        if 'mistral' in model.lower():
            self.tools = [t for t in all_tools if t['function']['name'] in ['write_file', 'run_command']]
        else:
            self.tools = all_tools

        # Model-specific system prompts for better tool usage
        model_base = model.lower().split(':')[0]

        if 'mistral' in model_base:
            # Mistral needs simple, direct instructions - match what works in raw API
            import platform
            os_info = f" You are running on {platform.system()}." if platform.system() == "Windows" else ""
            self.system_prompt = (
                f"You are a coding assistant.{os_info} "
                "You have access to TWO tools: write_file and run_command. "
                "Use write_file to create files. Use run_command to execute shell commands. "
                "DO NOT describe what to do - actually use the tools! "
                "For run_command on Windows, use commands like: 'dir', 'type filename.txt', 'python script.py'"
            )
        elif 'llama' in model_base:
            # Llama works well with more detailed instructions
            self.system_prompt = (
                "You are DeepSeek Agent, an expert coding assistant with access to tools. "
                "CRITICAL: When asked to 'make and run' or 'create and run' a program:\n"
                "1. Call write_file tool to create the file\n"
                "2. IMMEDIATELY call run_command tool to execute it (e.g., 'python filename.py')\n"
                "3. Report the actual execution results\n\n"
                "WRONG: Creating file then saying what the result would be\n"
                "RIGHT: Creating file, running it, showing actual output\n\n"
                "You must VERIFY your code works by running it! Use run_command after write_file.\n"
                "Use simple relative paths like 'fibonacci.py'."
            )
        else:
            # Default prompt - strongest emphasis on tool usage
            self.system_prompt = (
                "You are DeepSeek Agent, an expert coding assistant with tool access. "
                "CRITICAL RULE: You must use tools to complete tasks, NOT describe what to do.\n\n"
                "When asked to 'create a program' or 'write code':\n"
                "1. Immediately use write_file tool with the complete code\n"
                "2. Immediately use run_command tool to execute it\n"
                "3. Report the results\n\n"
                "WRONG: Providing code in your text response or saying 'here is code...'\n"
                "RIGHT: Calling write_file tool immediately\n\n"
                "WRONG: Saying 'you can run python file.py'\n"
                "RIGHT: Calling run_command tool immediately\n\n"
                "Use simple filenames like 'fibonacci.py' or 'src/main.py'. "
                "Never use placeholder paths like '/path/to/file.py'."
            )

    def _log(self, message: str):
        """Write message to session log file"""
        if self.enable_logging and self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    async def setup_model(self) -> DeepSeekClient:
        """Setup and verify DeepSeek model"""
        # First ensure Ollama is running
        console.print("Checking Ollama service...", style="blue")
        success, message = await ensure_ollama_running()

        if not success:
            console.print(f"ERROR: {message}", style="red")
            console.print(get_ollama_install_instructions(), style="yellow")
            raise typer.Exit(code=1)

        console.print(f"Ollama service ready: {message}", style="green")

        client = DeepSeekClient(model=self.model)

        console.print(f"Checking {self.model} availability...", style="blue")

        if not await client.check_model_availability():
            console.print(f"WARNING: {self.model} not found. Pulling model...", style="yellow")
            console.print(f"Downloading {self.model}... This may take a few minutes.")

            try:
                async for update in client.pull_model():
                    status = str(update.get("status", "")).strip() if isinstance(update, dict) else ""
                    if status:
                        console.print(f"  {status}", style="blue")

                if not await client.check_model_availability():
                    raise RuntimeError(f"Model {self.model} is still unavailable after download.")

                console.print(f"[OK] {self.model} ready!", style="green")
            except Exception as e:
                console.print(f"ERROR: Failed to download {self.model}: {e}", style="red")
                await client.client.aclose()
                raise typer.Exit(code=1)
        else:
            console.print(f"[OK] {self.model} available!", style="green")

        return client

    def display_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Display the result of a tool execution"""
        if tool_name == "write_file":
            if result.get("success"):
                file_path = result.get("file_path", "")
                console.print(f"[+] Created file: {file_path}", style="green")

                # Always display the file content
                try:
                    content = self.file_ops.read_file(file_path)
                    # Auto-detect language from extension
                    ext = Path(file_path).suffix.lower()
                    lang_map = {
                        '.py': 'python', '.js': 'javascript', '.java': 'java',
                        '.cpp': 'cpp', '.c': 'c', '.rs': 'rust', '.go': 'go',
                        '.html': 'html', '.css': 'css', '.json': 'json',
                        '.sh': 'bash', '.bat': 'batch', '.ps1': 'powershell'
                    }
                    language = lang_map.get(ext, 'text')
                    # Show full content if under 2000 chars, otherwise truncate
                    display_content = content if len(content) < 2000 else content[:1900] + "\n... (truncated)"
                    syntax = Syntax(display_content, language, theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=f"Created: {file_path}", border_style="green"))
                except Exception as e:
                    console.print(f"Could not display file: {e}", style="yellow")
            else:
                console.print(f"[-] Failed to create file: {result.get('error')}", style="red")

        elif tool_name == "run_command":
            command = result.get("command", "")
            return_code = result.get("return_code", -1)
            if result.get("success"):
                status = "[OK]" if return_code == 0 else "[WARN]"
                style = "green" if return_code == 0 else "yellow"
                console.print(f"{status} Executed: {command} (exit: {return_code})", style=style)
                if result.get("stdout"):
                    console.print(Panel(result["stdout"], title="Output", border_style="blue"))
                if result.get("stderr"):
                    console.print(Panel(result["stderr"], title="Errors", border_style="red"))
            else:
                console.print(f"[-] Failed to execute: {result.get('error')}", style="red")

        elif tool_name == "read_file" and result.get("success"):
            file_path = result.get("file_path", "")
            console.print(f"Read file: {file_path}", style="blue")

        elif tool_name == "list_directory" and result.get("success"):
            files = result.get("files", [])
            directory = result.get("directory", ".")
            console.print(f"Contents of {directory}:", style="blue")
            for file_info in files[:20]:
                icon = "[DIR]" if file_info["is_dir"] else "[FILE]"
                size = f" ({file_info['size']} bytes)" if file_info.get("size") else ""
                console.print(f"  {icon} {file_info['path']}{size}")

    async def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Message]:
        """Process and execute tool calls from the model"""
        tool_responses = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except:
                    tool_args = {}

            console.print(f"[TOOL] Executing: {tool_name}", style="cyan")

            # Log tool execution
            self._log(f"[TOOL] {tool_name}")
            self._log(f"  Arguments: {json.dumps(tool_args, indent=2)}")

            # Execute the tool
            result = await self.tool_executor.execute(tool_name, tool_args)

            # Log tool result
            self._log(f"  Result: {json.dumps(result, indent=2)}\n")

            # Display the result
            self.display_tool_result(tool_name, result)

            # Debug output commented out for cleaner display
            # print(f"DEBUG: Tool result: {result}")

            # Add tool response to conversation
            # Format tool results as user message to avoid confusing models that don't support tool role
            tool_result_text = f"Tool {tool_name} returned: {json.dumps(result)}"
            tool_message = Message(
                role="user",
                content=tool_result_text
            )
            tool_responses.append(tool_message)

        return tool_responses

    async def process_command(self, command: str, client: DeepSeekClient) -> bool:
        """Process user command with tool support"""
        command = command.strip()
        if not command:
            return True

        if command.lower() in {"quit", "exit", "q"}:
            self._log("\n=== Session Ended ===")
            return False

        if command.lower() in {"help", "?"}:
            help_text = """
## DeepSeek Agent with Tool Calling

This agent can autonomously:
- Create and modify files
- Execute commands
- Read files and directories
- Search for files

Just describe what you want and the agent will use its tools to accomplish it!

Examples:
- "Create a fizzbuzz program"
- "Write a Python script that calculates prime numbers"
- "Build a simple web server"

Type 'quit' to exit.
            """
            console.print(Markdown(help_text))
            return True

        # Add user message to history
        self.conversation_history.append(Message(role="user", content=command))

        # Log user input
        self._log(f"\n[USER] {command}\n")

        console.print("DeepSeek is working...", style="blue")

        # Keep trying to get response and execute tools
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Send chat request with tools
            response_complete = False
            current_response = None

            async for response in client.chat(
                self.conversation_history[-10:],  # Keep last 10 messages
                system_prompt=self.system_prompt,
                tools=self.tools,
                stream=False  # Tool calls don't work well with streaming
            ):
                if response.done:
                    response_complete = True
                    current_response = response

            if not response_complete or not current_response:
                break

            # Check for tool calls first
            if current_response.tool_calls:
                # Execute tools
                tool_responses = await self.process_tool_calls(current_response.tool_calls)

                # Add tool responses to conversation
                for tool_response in tool_responses:
                    self.conversation_history.append(tool_response)

                # Check if there's also a message with the tool calls
                message = current_response.message
                if message and message.content:
                    # Add assistant's explanation to history
                    self.conversation_history.append(Message(role="assistant", content=message.content))
                    # Only show if it's substantial (not just thinking)
                    if len(message.content.strip()) > 10:
                        console.print(Panel(Markdown(message.content), title="Assistant Note", border_style="blue"))

                # Continue the loop to let the model see tool results and potentially call more tools
                continue

            # No tool calls in proper format, check if the message contains JSON tool calls
            message = current_response.message
            if message and message.content:
                content = message.content.strip()
                tool_calls_found = False

                # Try to detect and parse JSON tool calls in the text (Mistral sometimes does this)
                # Look for patterns like [{"name": "tool_name", "arguments": {...}}]
                if '[{"name"' in content or '{"name"' in content:
                    try:
                        all_tool_calls = []
                        import re

                        # Find all potential JSON arrays starting with [{"name"
                        # Use a smarter approach: find balanced brackets
                        start_positions = [m.start() for m in re.finditer(r'\[\s*\{\s*"name"\s*:', content)]

                        for start_pos in start_positions:
                            # Find the matching closing bracket
                            bracket_count = 0
                            end_pos = start_pos
                            for i in range(start_pos, len(content)):
                                if content[i] == '[':
                                    bracket_count += 1
                                elif content[i] == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_pos = i + 1
                                        break

                            if end_pos > start_pos:
                                json_str = content[start_pos:end_pos]
                                try:
                                    tool_data = json.loads(json_str)
                                    # Convert to proper tool call format
                                    valid_tools = ['write_file', 'run_command', 'read_file', 'list_directory', 'find_files']
                                    for item in (tool_data if isinstance(tool_data, list) else [tool_data]):
                                        if isinstance(item, dict) and 'name' in item and 'arguments' in item:
                                            tool_name = item['name']
                                            if tool_name not in valid_tools:
                                                console.print(f"[yellow]Model tried to call unknown tool '{tool_name}'. Valid tools: {', '.join(valid_tools)}[/yellow]")
                                                continue
                                            all_tool_calls.append({
                                                'function': {
                                                    'name': tool_name,
                                                    'arguments': item['arguments']
                                                }
                                            })
                                except json.JSONDecodeError:
                                    # Try without the outer array if it's just a single object
                                    pass

                        if all_tool_calls:
                            console.print(f"[cyan]Detected {len(all_tool_calls)} tool call(s) in text format, executing...[/cyan]")
                            tool_responses = await self.process_tool_calls(all_tool_calls)
                            for tool_response in tool_responses:
                                self.conversation_history.append(tool_response)
                            tool_calls_found = True
                            # Continue to let model see results
                            continue
                    except Exception as e:
                        console.print(f"[red]Could not parse tool calls: {e}[/red]")

                # If no tool calls were found/parsed, display as normal message
                if not tool_calls_found:
                    self.conversation_history.append(Message(role="assistant", content=message.content))

                    # Log model response
                    self._log(f"[ASSISTANT] {message.content}\n")

                    # Don't display if the model is just echoing tool results
                    if "Tool write_file returned:" not in message.content and "Tool run_command returned:" not in message.content:
                        console.print(Panel(Markdown(message.content), title="DeepSeek Response", border_style="green"))
                    else:
                        # Model is echoing tool results - extract and show only the final summary
                        lines = message.content.split('\n')
                        summary_lines = [line for line in lines if not line.strip().startswith('Tool ') and 'returned:' not in line]
                        if summary_lines:
                            summary = '\n'.join(summary_lines).strip()
                            if len(summary) > 20:  # Only show if there's substantial content
                                console.print(Panel(Markdown(summary), title="Summary", border_style="blue"))
            break

        return True

    async def run_interactive(self):
        """Run interactive CLI session"""
        info_text = (
            "DeepSeek Agent v0.2.0 - Tool Calling Edition\n"
            f"Model: {self.model}\n"
            f"Workspace: {self.workspace}\n"
        )
        if self.log_file:
            info_text += f"Session log: {self.log_file.name}\n"
        info_text += (
            "\n"
            "This agent can autonomously:\n"
            "  - Create and modify files\n"
            "  - Execute commands\n"
            "  - Read and search files\n\n"
            "Just describe what you want!"
        )
        console.print(Panel.fit(
            info_text,
            title="Welcome",
            border_style="green"
        ))

        client = await self.setup_model()

        try:
            while True:
                command = Prompt.ask("\n[bold blue]You[/bold blue]")

                if not await self.process_command(command, client):
                    break

        except KeyboardInterrupt:
            console.print("\nGoodbye!", style="green")
        finally:
            await client.client.aclose()


@app.command()
def chat(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    model: str = typer.Option("deepseek-v2.5", "--model", "-m", help="DeepSeek model to use")
):
    """Start interactive chat session with tool-enabled DeepSeek Agent"""

    # Check if model supports tools
    # Updated list based on actual Ollama tool support
    models_with_tools = [
        "llama3.1", "llama3.2", "llama3.3",  # Llama models
        "mistral", "mistral-nemo", "mistral-small", "mistral-large",  # Mistral models
        "qwen2.5", "qwen2.5-coder",  # Qwen models with tool support
        "firefunction-v2",  # FireFunction
        "command-r", "command-r-plus",  # Command R models
        "gemma2",  # Google Gemma 2
        "hermes3",  # Hermes 3
        "nemotron",  # NVIDIA Nemotron
    ]

    # Warn if model might not support tools
    model_base = model.lower().split(":")[0]
    if not any(supported in model_base for supported in models_with_tools):
        console.print(
            f"WARNING: {model} may not support tool calling.\n"
            "Models with tool support:\n"
            "  - llama3.1, llama3.2 (Meta's latest)\n"
            "  - qwen2.5-coder:7b (Great for coding)\n"
            "  - mistral, mistral-nemo (Fast and capable)\n"
            "  - command-r (Cohere's model)\n"
            "  - firefunction-v2 (Function calling specialist)\n\n"
            "For coding without tools, use the standard chat mode:\n"
            "  deepseek-agent chat --model deepseek-coder-v2:16b",
            style="yellow"
        )
        response = Prompt.ask("Continue anyway? [y/N]", default="n")
        if response.lower() != 'y':
            return

    cli = ToolAgentCLI(workspace, model=model)
    asyncio.run(cli.run_interactive())


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()