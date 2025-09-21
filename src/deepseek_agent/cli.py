"""Command line interface for DeepSeek Agent"""

import asyncio
import os
import re
from pathlib import Path
from typing import Optional
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

app = typer.Typer(help="Local coding agent powered by DeepSeek models")
console = Console()


class AgentCLI:
    """Interactive CLI for the DeepSeek Agent"""
    
    def __init__(self, workspace: Optional[str] = None, model: str = "deepseek-v2.5"):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.model = model
        self.file_ops = FileOperations(str(self.workspace))
        self.conversation_history = []
        self.system_prompt = (
            "You are DeepSeek Agent, a local coding assistant with file operation capabilities."
            "\n\nAVAILABLE COMMANDS YOU CAN SUGGEST:"
            "\n- `write <file_path>` - Create or overwrite a file (you provide the content)"
            "\n- `read <file_path>` - Read and display file content"
            "\n- `ls [directory]` - List directory contents"
            "\n- `find <pattern>` - Find files matching pattern"
            "\n- `! <command>` - Run shell commands (like python, npm, etc.)"
            "\n\nWHEN ASKED TO CREATE CODE:"
            "\n1. First suggest using `write <filename>` to create the file"
            "\n2. Provide the complete file content"
            "\n3. Then suggest running it with `! python <filename>` or appropriate command"
            "\n\nExample: If asked to create fizzbuzz, say:"
            "\n'I'll create a fizzbuzz program. Use `write fizzbuzz.py` and I'll provide the content.'"
            "\nThen provide the full code for the user to save."
        )

    def _resolve_directory_alias(self, query: str) -> str:
        """Interpret natural language directory references."""
        query = (query or "").strip().strip("\"").strip("\'").rstrip("?.!")
        if not query:
            return "."

        lowered = query.lower()
        common_aliases = {
            "this directory": ".",
            "current directory": ".",
            "here": ".",
            "workspace": ".",
            "project": ".",
            "project root": ".",
            "root directory": "."
        }
        for phrase, value in common_aliases.items():
            if phrase in lowered:
                return value

        if lowered.startswith("directory ") or lowered.startswith("folder "):
            query = query.split(maxsplit=1)[1].strip()
            return query or "."

        if " in " in lowered:
            match = re.search(r"\bin\s+(.+)", query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip("\"").strip("\'")
                lowered_candidate = candidate.lower()
                if lowered_candidate in common_aliases:
                    return common_aliases[lowered_candidate]
                return candidate
        return query

    def _maybe_handle_list_request(self, command: str) -> Optional[str]:
        """Detect natural language list commands and return directory if applicable."""
        command_lower = command.lower()
        if "files" not in command_lower:
            return None

        list_keywords = ["list", "show", "display", "what", "which"]
        if not any(keyword in command_lower for keyword in list_keywords):
            return None

        skip_keywords = ["read", "open", "create", "delete", "move", "copy"]
        if any(keyword in command_lower for keyword in skip_keywords):
            return None

        match = re.search(r"\bin\s+(?P<directory>.+)$", command, re.IGNORECASE)
        directory = match.group("directory") if match else ""
        return self._resolve_directory_alias(directory)

    def _list_directory_for_command(self, directory: str) -> str:
        """List directory contents and return textual summary."""
        display_dir = directory or "."
        try:
            entries = self.file_ops.list_directory(display_dir)
        except Exception as exc:
            console.print(f"ERROR: Error listing directory: {exc}", style="red")
            return f"[shell] ls {display_dir}\nERROR: {exc}"

        if not entries:
            console.print(f"Contents of {display_dir}:", style="blue")
            console.print("  (empty)", style="yellow")
            return f"[shell] ls {display_dir}\n(empty)"

        lines = []
        console.print(f"Contents of {display_dir}:", style="blue")
        for file_info in entries[:20]:
            icon = "[DIR]" if file_info.is_dir else "[FILE]"
            size = f" ({file_info.size} bytes)" if file_info.is_file else ""
            entry_line = f"  {icon} {file_info.path}{size}"
            console.print(entry_line)
            lines.append(entry_line.strip())

        if len(entries) > 20:
            more_line = f"  ... and {len(entries) - 20} more files"
            console.print(more_line)
            lines.append(more_line.strip())

        return f"[shell] ls {display_dir}\n" + "\n".join(lines)

    async def setup_model(self) -> DeepSeekClient:
        """Setup and verify DeepSeek model"""
        client = DeepSeekClient(model=self.model)
        
        console.print(f"Checking {self.model} availability...", style="blue")

        if not await client.check_model_availability():
            console.print(f"WARNING: {self.model} not found. Pulling model...", style="yellow")
            console.print(f"Downloading {self.model}... This may take a few minutes.")

            last_status: Optional[str] = None
            try:
                async for update in client.pull_model():
                    status = str(update.get("status", "")).strip() if isinstance(update, dict) else ""
                    if status and status != last_status:
                        normalized = status.lower()
                        style = "blue"
                        if normalized in {"success", "exists"}:
                            style = "green" if normalized == "success" else "yellow"
                        console.print(f"  {status}", style=style)
                        last_status = status

                if not await client.check_model_availability():
                    raise RuntimeError(f"Model download completed but {self.model} is still unavailable in Ollama.")

                console.print(f"SUCCESS: {self.model} ready!", style="green")
            except Exception as e:
                message = str(e)
                console.print(f"ERROR: Failed to download {self.model}: {message}", style="red")
                if "requires more system memory" in message.lower():
                    console.print("TIP: Try a smaller Ollama model (e.g. 'deepseek-r1:1.5b') with --model.", style="yellow")
                await client.client.aclose()
                raise typer.Exit(code=1)
        else:
            console.print(f"SUCCESS: {self.model} available!", style="green")

        return client
    
    def display_file_content(self, file_path: str, language: Optional[str] = None):
        """Display file content with syntax highlighting"""
        try:
            content = self.file_ops.read_file(file_path)
            
            # Auto-detect language from extension
            if not language:
                ext = Path(file_path).suffix.lower()
                lang_map = {
                    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                    '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                    '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
                    '.html': 'html', '.css': 'css', '.json': 'json', '.xml': 'xml',
                    '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml', '.md': 'markdown'
                }
                language = lang_map.get(ext, 'text')
            
            syntax = Syntax(content, language, theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"File: {file_path}", border_style="blue"))
            
        except Exception as e:
            console.print(f"ERROR: Error reading file: {e}", style="red")
    
    def list_files(self, directory: str = "."):
        """List files in directory"""
        try:
            files = self.file_ops.list_directory(directory)
            
            if not files:
                console.print("Empty directory", style="yellow")
                return
            
            console.print(f"Contents of {directory}:", style="blue")
            for file_info in files[:20]:  # Limit to 20 files
                icon = "[DIR]" if file_info.is_dir else "[FILE]"
                size = f"({file_info.size} bytes)" if file_info.is_file else ""
                console.print(f"  {icon} {file_info.path} {size}")
            
            if len(files) > 20:
                console.print(f"  ... and {len(files) - 20} more files")
                
        except Exception as e:
            console.print(f"ERROR: Error listing directory: {e}", style="red")
    

    async def run_shell_command(self, command: str) -> bool:
        """Execute a shell command within the workspace"""
        command = command.strip()
        if not command:
            console.print("Usage: ! <command> or shell <command>", style="yellow")
            return True

        console.print(f"Running command in {self.workspace}: [bold]{command}[/bold]", style="blue")
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env=os.environ.copy(),
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError as exc:
            console.print(f"ERROR: Command not found: {exc}", style="red")
            return True
        except Exception as exc:
            console.print(f"ERROR: Failed to run command: {exc}", style="red")
            return True

        def _print_stream(content: bytes, title: str, style: str) -> None:
            if content:
                text = content.decode(errors="replace").rstrip()
                if text:
                    console.print(Panel(text, title=title, border_style=style))

        _print_stream(stdout, "stdout", "green")
        _print_stream(stderr, "stderr", "red")

        if process.returncode == 0:
            console.print("Command completed successfully.", style="green")
        else:
            console.print(f"Command exited with code {process.returncode}", style="red")

        return True

    async def process_command(self, command: str, client: DeepSeekClient) -> bool:
        """Process user command"""
        command = command.strip()
        if not command:
            return True

        command_lower = command.lower()

        if command_lower in {"quit", "exit", "q"}:
            return False

        tool_messages = []
        directory_from_natural = self._maybe_handle_list_request(command)
        if directory_from_natural is not None:
            tool_output = self._list_directory_for_command(directory_from_natural)
            if tool_output:
                tool_messages.append(tool_output)

        # File operations
        if command_lower.startswith("write "):
            file_path = command[6:].strip()
            if not file_path:
                console.print("Usage: write <file_path>", style="yellow")
                return True

            # Ask for multiline content
            console.print(f"Creating file: {file_path}", style="blue")
            console.print("Enter the content (type 'EOF' on a new line when done):", style="yellow")

            lines = []
            while True:
                line = Prompt.ask("")
                if line == "EOF":
                    break
                lines.append(line)

            content = "\n".join(lines)

            try:
                self.file_ops.write_file(file_path, content)
                console.print(f"✓ File created: {file_path}", style="green")

                # Suggest running the file if it's a script
                if file_path.endswith(('.py', '.js', '.sh', '.bat')):
                    ext = Path(file_path).suffix
                    run_cmd = {
                        '.py': 'python',
                        '.js': 'node',
                        '.sh': 'bash',
                        '.bat': ''
                    }.get(ext, '')
                    if run_cmd:
                        console.print(f"\nTo run this file, use: `! {run_cmd} {file_path}`", style="cyan")
            except Exception as e:
                console.print(f"ERROR: Failed to write file: {e}", style="red")
            return True

        if command_lower.startswith("read "):
            file_path = command[5:].strip()
            if not file_path:
                console.print("Usage: read <file_path>", style="yellow")
                return True
            self.display_file_content(file_path)
            return True

        if command_lower.startswith("ls") or command_lower.startswith("list"):
            raw_directory = command.split(maxsplit=1)[1] if " " in command else ""
            directory = self._resolve_directory_alias(raw_directory)
            if directory_from_natural is None:
                tool_output = self._list_directory_for_command(directory)
                if tool_output:
                    self.conversation_history.append(Message(role="assistant", content=tool_output))
                return True

        if command_lower.startswith("find "):
            pattern = command[5:].strip()
            if not pattern:
                console.print("Usage: find <pattern>", style="yellow")
                return True
            try:
                matches = self.file_ops.find_files(pattern)
                if matches:
                    console.print(f"Found {len(matches)} files matching '{pattern}':")
                    for match in matches[:10]:
                        console.print(f"  {match}")
                    if len(matches) > 10:
                        console.print(f"  ... and {len(matches) - 10} more files")
                else:
                    console.print(f" No files found matching '{pattern}'", style="yellow")
            except Exception as e:
                console.print(f"ERROR: Error searching files: {e}", style="red")
            return True

        # Shell commands
        if command_lower.startswith("!") or command_lower.startswith("shell "):
            shell_command = command[1:].strip() if command.startswith("!") else command[6:].strip()
            return await self.run_shell_command(shell_command)

        # Help command
        if command_lower in {"help", "?"}:
            help_text = """
## Available Commands

**File Operations:**
- `write <file>` - Create or overwrite a file (agent will provide content)
- `read <file>` - Read and display file content
- `ls [dir]` - List directory contents
- `find <pattern>` - Find files matching pattern

**Code Operations:**
- `generate <prompt>` - Generate code based on prompt
- `explain <file>` - Explain code in a file
- `review <file>` - Review code for issues

**General:**
- `! <command>` or `shell <command>` - Run shell commands in the workspace
- `help` - Show this help
- `quit` - Exit the agent

**Chat Mode:**
Just type any question or request to chat with DeepSeek!
            """
            console.print(Markdown(help_text))
            return True

        # Code generation
        if command_lower.startswith("generate "):
            prompt = command[9:].strip()
            if not prompt:
                console.print("Usage: generate <prompt>", style="yellow")
                return True
            console.print(" Generating code...", style="blue")

            response_text = ""
            console.print("Processing...")
            async for chunk in client.generate_code(prompt):
                response_text += chunk

            console.print(Panel(Markdown(response_text), title=" Generated Code", border_style="green"))
            return True

        # Code explanation
        if command_lower.startswith("explain "):
            file_path = command[8:].strip()
            if not file_path:
                console.print("Usage: explain <file_path>", style="yellow")
                return True
            try:
                code = self.file_ops.read_file(file_path)
                console.print(" Analyzing code...", style="blue")

                console.print("Processing...")
                explanation = await client.explain_code(code, Path(file_path).suffix[1:])

                console.print(Panel(Markdown(explanation), title=f" Code Explanation: {file_path}", border_style="blue"))
            except Exception as e:
                console.print(f"ERROR: Error explaining code: {e}", style="red")
            return True

        # Code review
        if command_lower.startswith("review "):
            file_path = command[7:].strip()
            if not file_path:
                console.print("Usage: review <file_path>", style="yellow")
                return True
            try:
                code = self.file_ops.read_file(file_path)
                console.print(" Reviewing code...", style="blue")

                console.print("Processing...")
                review = await client.review_code(code, Path(file_path).suffix[1:])

                console.print(Panel(Markdown(review), title=f" Code Review: {file_path}", border_style="yellow"))
            except Exception as e:
                console.print(f"ERROR: Error reviewing code: {e}", style="red")
            return True

        # General chat
        # Check if this looks like a code generation request
        code_keywords = ['create', 'write', 'make', 'build', 'implement', 'generate', 'code', 'program', 'script', 'function', 'class', 'fizz', 'buzz']
        is_code_request = any(keyword in command_lower for keyword in code_keywords)

        # Enhance the prompt if it's a code request
        enhanced_command = command
        if is_code_request and 'file' not in command_lower:
            enhanced_command = (
                f"{command}\n\n"
                "IMPORTANT: Suggest using the `write <filename>` command to create a file, "
                "then provide the complete file content that can be saved and executed."
            )

        self.conversation_history.append(Message(role="user", content=enhanced_command))
        for tool_msg in tool_messages:
            self.conversation_history.append(Message(role="assistant", content=tool_msg))

        console.print(" DeepSeek is thinking...", style="blue")
        response_text = ""

        with Live(Spinner("dots", text="Processing..."), console=console, refresh_per_second=4):
            async for response in client.chat(self.conversation_history[-5:], system_prompt=self.system_prompt):  # Keep last 5 messages
                if response.message and response.message.content:
                    response_text += response.message.content

        self.conversation_history.append(Message(role="assistant", content=response_text))

        console.print(Panel(Markdown(response_text), title=" DeepSeek Response", border_style="green"))

        return True

    async def run_interactive(self):
        """Run interactive CLI session"""
        console.print(Panel.fit(
            " DeepSeek Agent v0.1.0\n"
            f"Your local coding assistant powered by {self.model}\n\n"
            f"Working directory: {self.workspace}\n\n"
            "Quick Commands:\n"
            "  • write <file> - Create a new file\n"
            "  • read <file> - View file contents\n"
            "  • ! <command> - Run shell commands\n"
            "  • help - Show all commands\n\n"
            "💡 Tip: Ask me to create any code and I'll help you save it to a file!",
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
            console.print("\n Goodbye!", style="green")
        finally:
            await client.client.aclose()


@app.command()
def chat(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    model: str = typer.Option("deepseek-v2.5", "--model", "-m", help="DeepSeek model to use"),
    tools: bool = typer.Option(False, "--tools", help="Enable tool calling (requires compatible model)")
):
    """Start interactive chat session with DeepSeek Agent"""
    if tools:
        from deepseek_agent.cli_with_tools import ToolAgentCLI
        cli = ToolAgentCLI(workspace, model=model)
        asyncio.run(cli.run_interactive())
    else:
        cli = AgentCLI(workspace, model=model)
        asyncio.run(cli.run_interactive())


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Code generation prompt"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Programming language"),
    model: str = typer.Option("deepseek-v2.5", "--model", "-m", help="DeepSeek model to use")
):
    """Generate code from prompt"""
    async def _generate():
        client = DeepSeekClient(model=model)

        console.print(" Generating code...", style="blue")
        response_text = ""

        async for chunk in client.generate_code(prompt, language):
            response_text += chunk

        if output:
            file_ops = FileOperations()
            file_ops.write_file(output, response_text)
            console.print(f"SUCCESS: Code saved to {output}", style="green")
        else:
            console.print(Panel(Markdown(response_text), title=" Generated Code", border_style="green"))

        await client.client.aclose()

    asyncio.run(_generate())


@app.command()
def review(
    file_path: str = typer.Argument(..., help="File to review"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save review to file"),
    model: str = typer.Option("deepseek-v2.5", "--model", "-m", help="DeepSeek model to use")
):
    """Review code file for issues and improvements"""
    async def _review():
        client = DeepSeekClient(model=model)
        file_ops = FileOperations()

        try:
            code = file_ops.read_file(file_path)
            console.print(" Reviewing code...", style="blue")

            review = await client.review_code(code, Path(file_path).suffix[1:])

            if output:
                file_ops.write_file(output, review)
                console.print(f"SUCCESS: Review saved to {output}", style="green")
            else:
                console.print(Panel(Markdown(review), title=f" Code Review: {file_path}", border_style="yellow"))

        except Exception as e:
            console.print(f"ERROR: Error: {e}", style="red")
        finally:
            await client.client.aclose()

    asyncio.run(_review())

def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()


















