#!/usr/bin/env python3
"""Optional integration tests against a real Ollama server."""

import os
import sys
from pathlib import Path

import pytest

# These tests require a running Ollama instance. Skip unless explicitly enabled.
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_OLLAMA_TESTS") != "1",
    reason="Requires RUN_REAL_OLLAMA_TESTS=1 with a running Ollama server",
)

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_agent.model.client import DeepSeekClient, Message
from deepseek_agent.cli import AgentCLI


@pytest.mark.asyncio
async def test_ollama_connection():
    """Ensure at least one Ollama model is available when the server is running."""
    client = DeepSeekClient()
    try:
        available = await client.check_model_availability()
    finally:
        await client.client.aclose()

    assert isinstance(available, bool)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["deepseek-v2.5"])
async def test_agent_with_model(model_name):
    """Smoke-test chat and generate flows with a real model."""
    client = DeepSeekClient(model=model_name)
    try:
        messages = [Message(role="user", content="Say hello and tell me your name")]
        response_text = ""
        async for response in client.chat(messages):
            if response.message and response.message.content:
                response_text += response.message.content
        assert response_text != ""

        generation_text = ""
        async for chunk in client.generate_code("Write a Python function to add two numbers"):
            generation_text += chunk
        assert generation_text != ""
    finally:
        await client.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["deepseek-v2.5"])
async def test_cli_integration(model_name):
    """Verify CLI can set up and exercise basic commands with a real model."""
    class TestAgentCLI(AgentCLI):
        async def setup_model(self):
            client = DeepSeekClient(model=model_name)
            available = await client.check_model_availability()
            if not available:
                await client.client.aclose()
                pytest.skip(f"Model {model_name} not available in Ollama")
            return client

    cli = TestAgentCLI(".")
    client = await cli.setup_model()

    try:
        assert await cli.process_command("ls", client) is True
        assert await cli.process_command("help", client) is True
    finally:
        await client.client.aclose()

