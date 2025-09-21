#!/usr/bin/env python3
"""Unit tests for core DeepSeek Agent utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project package is importable during tests
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_agent.cli import AgentCLI
from deepseek_agent.model.client import DeepSeekClient
from deepseek_agent.tools.file_ops import FileOperations


def test_file_operations_list_read_and_find(tmp_path):
    """FileOperations should operate relative to the provided workspace."""
    workspace = tmp_path
    sample_file = workspace / "example.py"
    sample_file.write_text("print('hello')\n", encoding="utf-8")

    file_ops = FileOperations(str(workspace))

    entries = file_ops.list_directory(".")
    assert any(entry.path == "example.py" and entry.is_file for entry in entries)

    content = file_ops.read_file("example.py")
    assert content == "print('hello')\n"

    matches = file_ops.find_files("*.py")
    assert matches == ["example.py"]


@pytest.mark.asyncio
async def test_model_client_handles_unavailable_server():
    """check_model_availability should fail gracefully when Ollama is down."""
    client = DeepSeekClient(host="http://127.0.0.1:65535")
    try:
        available = await client.check_model_availability()
    finally:
        await client.client.aclose()
    assert available is False


@pytest.mark.asyncio
async def test_agent_cli_generate_command(tmp_path):
    """generate command should consume client output without errors."""
    cli = AgentCLI(str(tmp_path), model="mini-model")
    assert cli.model == "mini-model"
    assert "shell commands" in cli.system_prompt

    class StubClient:
        async def generate_code(self, prompt: str, language=None, context=None):
            assert prompt == "write hello"
            yield "```python\nprint('hello')\n```"

    result = await cli.process_command("generate write hello", StubClient())
    assert result is True


@pytest.mark.asyncio
async def test_agent_cli_list_natural_language(tmp_path):
    """Natural language list command should resolve to current directory."""
    workspace = tmp_path
    (workspace / "sample.txt").write_text("data", encoding="utf-8")
    cli = AgentCLI(str(workspace))

    class DummyClient:
        async def chat(self, messages, system_prompt=None):
            class Resp:
                def __init__(self):
                    self.message = type('Msg', (), {'content': ''})
            yield Resp()

    result = await cli.process_command("list the files in this directory", DummyClient())
    assert result is True
    assert any(msg.content.startswith("[shell] ls") for msg in cli.conversation_history if msg.role == 'assistant')

    result = await cli.process_command("What files are in this directory?", DummyClient())
    assert result is True


@pytest.mark.asyncio
async def test_agent_cli_shell_command(tmp_path):
    """Shell command helper should execute commands in the workspace."""
    cli = AgentCLI(str(tmp_path))
    command = f'"{sys.executable}" -c "print(\"shell-test\")"'
    result = await cli.run_shell_command(command)
    assert result is True
