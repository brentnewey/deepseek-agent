"""Command line interface for DeepSeek Agent with tool calling support"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
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

app = typer.Typer(help="Local coding agent powered by DeepSeek models with tool calling")
console = Console()


class ToolAgentCLI:
    """Interactive CLI for DeepSeek Agent with tool calling"""

    def __init__(self, workspace: Optional[str] = None, model: str = "deepseek-v2.5"):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.model = model
        self.file_ops = FileOperations(str(self.workspace))
        self.tool_executor = ToolExecutor(self.workspace)
        self.conversation_history = []
        self.tools = get_tool_definitions()

        self.system_prompt = (
            "You are DeepSeek Agent, an expert coding assistant with access to tools for file operations and command execution. "
            "Use the provided tools to complete tasks effectively. When asked to create code, use the write_file tool. "
            "When you need to verify your work, use the run_command tool to execute the code. "
            "Always aim to create complete, working solutions."
        )

    async def setup_model(self) -> DeepSeekClient:
        """Setup and verify DeepSeek model"""
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

                console.print(f"✓ {self.model} ready!", style="green")
            except Exception as e:
                console.print(f"ERROR: Failed to download {self.model}: {e}", style="red")
                await client.client.aclose()
                raise typer.Exit(code=1)
        else:
            console.print(f"✓ {self.model} available!", style="green")

        return client

    def display_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Display the result of a tool execution"""
        if tool_name == "write_file" and result.get("success"):
            file_path = result.get("file_path", "")
            console.print(f"✓ Created file: {file_path}", style="green")

            # Offer to display the file
            if file_path.endswith(('.py', '.js', '.java', '.cpp', '.c', '.rs', '.go')):
                try:
                    content = self.file_ops.read_file(file_path)
                    # Auto-detect language from extension
                    ext = Path(file_path).suffix.lower()
                    lang_map = {
                        '.py': 'python', '.js': 'javascript', '.java': 'java',
                        '.cpp': 'cpp', '.c': 'c', '.rs': 'rust', '.go': 'go'
                    }
                    language = lang_map.get(ext, 'text')
                    syntax = Syntax(content[:1000], language, theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=f"File: {file_path}", border_style="blue"))
                except:
                    pass

        elif tool_name == "run_command" and result.get("success"):
            command = result.get("command", "")
            console.print(f"Executed: {command}", style="blue")

            if result.get("stdout"):
                console.print(Panel(result["stdout"][:500], title="Output", border_style="green"))
            if result.get("stderr"):
                console.print(Panel(result["stderr"][:500], title="Errors", border_style="red"))

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

            console.print(f"🔧 Executing tool: {tool_name}", style="cyan")

            # Execute the tool
            result = await self.tool_executor.execute(tool_name, tool_args)

            # Display the result
            self.display_tool_result(tool_name, result)

            # Add tool response to conversation
            tool_responses.append(Message(
                role="tool",
                content=json.dumps(result)
            ))

        return tool_responses

    async def process_command(self, command: str, client: DeepSeekClient) -> bool:
        """Process user command with tool support"""
        command = command.strip()
        if not command:
            return True

        if command.lower() in {"quit", "exit", "q"}:
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

        console.print("🤖 DeepSeek is working...", style="blue")

        # Keep trying to get response and execute tools
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Send chat request with tools
            response_complete = False
            current_response = None

            with Live(Spinner("dots", text="Thinking..."), console=console, refresh_per_second=4):
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

            # Check for tool calls
            message = current_response.message
            if message:
                # Check if there are tool calls
                tool_calls = None
                if hasattr(message, 'tool_calls'):
                    tool_calls = message.tool_calls
                elif isinstance(message, dict) and 'tool_calls' in message:
                    tool_calls = message['tool_calls']

                if tool_calls:
                    # Execute tools
                    tool_responses = await self.process_tool_calls(tool_calls)

                    # Add tool responses to conversation
                    for tool_response in tool_responses:
                        self.conversation_history.append(tool_response)

                    # Continue conversation to get final response
                    continue

                # No tool calls, display the response
                if message.content:
                    self.conversation_history.append(Message(role="assistant", content=message.content))
                    console.print(Panel(Markdown(message.content), title="🤖 DeepSeek Response", border_style="green"))
                break

        return True

    async def run_interactive(self):
        """Run interactive CLI session"""
        console.print(Panel.fit(
            "🤖 DeepSeek Agent v0.2.0 - Tool Calling Edition\n"
            f"Model: {self.model}\n"
            f"Workspace: {self.workspace}\n\n"
            "This agent can autonomously:\n"
            "  • Create and modify files\n"
            "  • Execute commands\n"
            "  • Read and search files\n\n"
            "Just describe what you want!",
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
            console.print("\n👋 Goodbye!", style="green")
        finally:
            await client.client.aclose()


@app.command()
def chat(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    model: str = typer.Option("deepseek-v2.5", "--model", "-m", help="DeepSeek model to use")
):
    """Start interactive chat session with tool-enabled DeepSeek Agent"""

    # Check if model supports tools
    models_with_tools = ["llama3.1", "llama3.2", "mistral-nemo", "firefunction-v2", "command-r", "command-r-plus"]

    # Warn if model might not support tools
    model_base = model.lower().split(":")[0]
    if not any(supported in model_base for supported in models_with_tools):
        console.print(
            f"⚠️  Warning: {model} may not support tool calling.\n"
            "Consider using: llama3.1, mistral-nemo, or firefunction-v2",
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