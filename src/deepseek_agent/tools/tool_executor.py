"""Tool executor for handling LLM tool calls"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from deepseek_agent.tools.file_ops import FileOperations


class ToolExecutor:
    """Execute tools called by the LLM"""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.file_ops = FileOperations(str(workspace))

    def _clean_file_path(self, file_path: str) -> str:
        """Clean up common placeholder paths that LLMs might generate"""
        # Remove common placeholder patterns
        placeholder_patterns = [
            '/path/to/',
            '/your/path/',
            '/home/user/',
            'C:/path/to/',
            'C:\\path\\to\\',
            '~/'
        ]

        for pattern in placeholder_patterns:
            if file_path.startswith(pattern):
                # Extract just the filename
                file_path = file_path[len(pattern):]
                break

        # If path starts with / on Windows, remove it
        if os.name == 'nt' and file_path.startswith('/'):
            file_path = file_path[1:]

        # Remove any remaining leading slashes or backslashes
        file_path = file_path.lstrip('/\\')

        return file_path

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result"""

        if tool_name == "write_file":
            return await self.write_file(arguments)
        elif tool_name == "read_file":
            return await self.read_file(arguments)
        elif tool_name == "list_directory":
            return await self.list_directory(arguments)
        elif tool_name == "run_command":
            return await self.run_command(arguments)
        elif tool_name == "find_files":
            return await self.find_files(arguments)
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

    async def write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            file_path = args.get("file_path", "")
            content = args.get("content", "")

            if not file_path:
                return {"success": False, "error": "file_path is required"}

            # Clean up common placeholder paths
            file_path = self._clean_file_path(file_path)

            # Debug output commented out for cleaner display
            # print(f"DEBUG: Attempting to write file: {file_path}")
            # print(f"DEBUG: Content length: {len(content)}")
            # print(f"DEBUG: Workspace: {self.workspace}")

            # Call the file_ops write_file method
            result = self.file_ops.write_file(file_path, content)
            # print(f"DEBUG: write_file returned: {result}")

            return {
                "success": True,
                "message": f"File written successfully: {file_path}. You should now run this file to verify it works.",
                "file_path": file_path,
                "next_action": "Use run_command to execute this file"
            }
        except Exception as e:
            # print(f"DEBUG: Error in write_file: {e}")
            return {"success": False, "error": str(e)}

    async def read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read content from a file"""
        try:
            file_path = args.get("file_path", "")

            if not file_path:
                return {"success": False, "error": "file_path is required"}

            # Clean up common placeholder paths
            file_path = self._clean_file_path(file_path)

            content = self.file_ops.read_file(file_path)

            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_directory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents"""
        try:
            directory = args.get("directory", ".")
            files = self.file_ops.list_directory(directory)

            file_list = []
            for file_info in files[:50]:  # Limit to 50 entries
                file_list.append({
                    "path": file_info.path,
                    "is_dir": file_info.is_dir,
                    "size": file_info.size if file_info.is_file else None
                })

            return {
                "success": True,
                "files": file_list,
                "total_count": len(files),
                "directory": directory
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command"""
        try:
            command = args.get("command", "")
            timeout = args.get("timeout", 30)  # Default 30 second timeout

            if not command:
                return {"success": False, "error": "command is required"}

            # Determine shell based on OS
            if os.name == 'nt':  # Windows
                # Use cmd.exe for better compatibility
                shell_cmd = command
            else:  # Unix/Linux/Mac
                shell_cmd = command

            # Execute command with timeout
            try:
                process = await asyncio.create_subprocess_shell(
                    shell_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workspace),
                    env=os.environ.copy(),
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "command": command
                }

            return {
                "success": True,
                "stdout": stdout.decode(errors="replace") if stdout else "",
                "stderr": stderr.decode(errors="replace") if stderr else "",
                "return_code": process.returncode,
                "command": command
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def find_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find files matching a pattern"""
        try:
            pattern = args.get("pattern", "")

            if not pattern:
                return {"success": False, "error": "pattern is required"}

            matches = self.file_ops.find_files(pattern)

            return {
                "success": True,
                "matches": matches[:100],  # Limit to 100 results
                "total_count": len(matches),
                "pattern": pattern
            }
        except Exception as e:
            return {"success": False, "error": str(e)}