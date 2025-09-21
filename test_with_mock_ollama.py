#!/usr/bin/env python3
"""Test DeepSeek Agent with a mock Ollama server"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import httpx

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_agent.model.client import DeepSeekClient, Message
from deepseek_agent.cli import AgentCLI


class MockOllamaServer:
    """Mock Ollama server for testing"""

    def __init__(self):
        self.models = ["deepseek-v2.5"]
        self.response_chunks = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Sure! Here's a simple Python function to calculate the factorial of a number:\n\n```python\ndef factorial(n):\n    \"\"\"Calculate the factorial of a positive integer.\n    \n    Args:\n        n (int): A positive integer\n    \n    Returns:\n        int: The factorial of n\n    \"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        result = 1\n        for i in range(2, n + 1):\n            result *= i\n        return result\n\n# Example usage:\nprint(factorial(5))  # Output: 120\nprint(factorial(0))  # Output: 1\n```\n\nThis function:\n1. Handles edge cases (negative numbers, 0, and 1)\n2. Uses an iterative approach for efficiency\n3. Includes proper documentation\n4. Provides example usage",
                },
                "done": True,
            }
        ]

    def mock_get_request(self, url, **kwargs):
        """Mock GET request to Ollama"""
        if url.endswith("/api/tags"):
            return MockResponse(200, {"models": [{"name": "deepseek-v2.5"}]})
        return MockResponse(404, {"error": "Not found"})

    def mock_stream_request(self, method, url, **kwargs):
        """Mock streaming request to Ollama"""
        if url.endswith("/api/chat"):
            return MockStreamResponse(self.response_chunks)

        if url.endswith("/api/pull"):
            return MockStreamResponse(
                [
                    {"status": "downloading", "total": 100, "completed": 50},
                    {"status": "downloading", "total": 100, "completed": 100},
                    {"status": "success"},
                ]
            )

        return MockStreamResponse([{"error": "Not found"}])


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


class MockStreamResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def aiter_lines(self):
        for chunk in self.chunks:
            yield json.dumps(chunk)

    async def aread(self):
        return b"Mock error response"


@pytest.mark.asyncio
async def test_model_with_mock():
    """DeepSeekClient should communicate correctly with mocked Ollama APIs."""
    mock_server = MockOllamaServer()

    with patch.object(httpx.AsyncClient, "get") as mock_get, patch.object(
        httpx.AsyncClient, "stream"
    ) as mock_stream:
        mock_get.side_effect = mock_server.mock_get_request
        mock_stream.side_effect = mock_server.mock_stream_request

        client = DeepSeekClient()

        try:
            available = await client.check_model_availability()
            assert available is True

            generation_chunks = []
            async for chunk in client.generate_code("Create a factorial function"):
                generation_chunks.append(chunk)
            generated = "".join(generation_chunks)
            assert "factorial" in generated

            messages = [Message(role="user", content="What is 2+2?")]
            chat_responses = []
            async for response in client.chat(messages):
                if response.message and response.message.content:
                    chat_responses.append(response.message.content)
            assert chat_responses != []
        finally:
            await client.client.aclose()


@pytest.mark.asyncio
async def test_cli_with_mock(tmp_path):
    """AgentCLI should operate with mocked Ollama responses."""
    mock_server = MockOllamaServer()

    with patch.object(httpx.AsyncClient, "get") as mock_get, patch.object(
        httpx.AsyncClient, "stream"
    ) as mock_stream:
        mock_get.side_effect = mock_server.mock_get_request
        mock_stream.side_effect = mock_server.mock_stream_request

        workspace = tmp_path
        (workspace / "sample.py").write_text("print('hi')\n", encoding="utf-8")

        cli = AgentCLI(str(workspace))
        client = await cli.setup_model()

        try:
            assert await cli.process_command("ls", client) is True
            assert await cli.process_command("help", client) is True
            assert await cli.process_command(
                "generate Create a simple Hello World function", client
            ) is True
        finally:
            await client.client.aclose()
