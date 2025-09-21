#!/usr/bin/env python3
"""Test the tool-calling agent with various code generation tasks"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepseek_agent.model.client import DeepSeekClient, Message
from deepseek_agent.tools.tool_definitions import get_tool_definitions
from deepseek_agent.tools.tool_executor import ToolExecutor


async def test_tool_agent():
    """Test the agent with tool calling"""

    # Note: This test requires a model that supports tool calling
    # Good options: llama3.1, mistral-nemo, firefunction-v2
    model = "llama3.1"  # Change this to a model you have that supports tools

    client = DeepSeekClient(model=model)

    # Check model availability
    print(f"Checking if {model} is available...")
    if not await client.check_model_availability():
        print(f"Model {model} not available. Please pull it first with: ollama pull {model}")
        await client.client.aclose()
        return

    print(f"✓ Model {model} is available")

    # Setup tools
    tools = get_tool_definitions()
    tool_executor = ToolExecutor(Path.cwd())

    system_prompt = (
        "You are an expert coding assistant with access to tools for file operations. "
        "When asked to create code, use the write_file tool to save it. "
        "After creating code, use run_command to test it."
    )

    # Test prompts
    test_cases = [
        "Create a Python script called hello.py that prints 'Hello, World!'",
        "Write a fizzbuzz program in Python and save it as fizzbuzz.py",
        "Create a function to calculate factorial and save it as factorial.py"
    ]

    for test_prompt in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_prompt}")
        print('='*60)

        messages = [Message(role="user", content=test_prompt)]

        # Get response with tools
        response_text = ""
        tool_calls_made = []

        async for response in client.chat(
            messages,
            system_prompt=system_prompt,
            tools=tools,
            stream=False
        ):
            if response.done:
                # Check for tool calls
                if response.tool_calls:
                    print(f"\n🔧 Tool calls detected: {len(response.tool_calls)}")
                    for tool_call in response.tool_calls:
                        func_name = tool_call.get("function", {}).get("name")
                        print(f"  - {func_name}")
                        tool_calls_made.append(func_name)

                    # Execute tools (in real scenario)
                    for tool_call in response.tool_calls:
                        func_name = tool_call.get("function", {}).get("name")
                        func_args = tool_call.get("function", {}).get("arguments", {})

                        if isinstance(func_args, str):
                            import json
                            try:
                                func_args = json.loads(func_args)
                            except:
                                func_args = {}

                        print(f"\nExecuting: {func_name}")
                        result = await tool_executor.execute(func_name, func_args)
                        if result.get("success"):
                            print(f"  ✓ Success: {result.get('message', 'Tool executed')}")
                        else:
                            print(f"  ✗ Error: {result.get('error')}")

                elif response.message and response.message.content:
                    response_text = response.message.content
                    print(f"\nResponse: {response_text[:200]}...")

        if tool_calls_made:
            print(f"\n✅ SUCCESS: Agent used tools: {', '.join(tool_calls_made)}")
        else:
            print(f"\n⚠️  WARNING: Agent did not use any tools")

    await client.client.aclose()


if __name__ == "__main__":
    print("Testing Tool-Enabled DeepSeek Agent")
    print("Note: This requires a model that supports tool calling")
    print("Recommended models: llama3.1, mistral-nemo, firefunction-v2\n")

    asyncio.run(test_tool_agent())