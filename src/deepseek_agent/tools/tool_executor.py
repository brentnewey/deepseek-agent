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

            self.file_ops.write_file(file_path, content)

            return {
                "success": True,
                "message": f"File written successfully: {file_path}",
                "file_path": file_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read content from a file"""
        try:
            file_path = args.get("file_path", "")

            if not file_path:
                return {"success": False, "error": "file_path is required"}

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

            if not command:
                return {"success": False, "error": "command is required"}

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env=os.environ.copy(),
            )
            stdout, stderr = await process.communicate()

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