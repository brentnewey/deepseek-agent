#!/usr/bin/env python3
"""Test script to verify the agent suggests file creation for code tasks"""

import asyncio
from src.deepseek_agent.model.client import DeepSeekClient, Message

async def test_code_suggestion():
    """Test that the agent suggests file creation for code tasks"""

    client = DeepSeekClient(model="deepseek-v2.5")

    system_prompt = (
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

    test_prompt = (
        "Create a fizzbuzz program\n\n"
        "IMPORTANT: Suggest using the `write <filename>` command to create a file, "
        "then provide the complete file content that can be saved and executed."
    )

    messages = [Message(role="user", content=test_prompt)]

    print("Testing agent response to code generation request...")
    print("=" * 60)

    response_text = ""
    async for response in client.chat(messages, system_prompt=system_prompt):
        if response.message and response.message.content:
            response_text += response.message.content

    print(response_text)
    print("=" * 60)

    # Check if the response includes file creation suggestion
    if "write" in response_text.lower() and ".py" in response_text:
        print("✅ SUCCESS: Agent suggested file creation!")
    else:
        print("❌ FAILED: Agent did not suggest file creation")

    await client.client.aclose()

if __name__ == "__main__":
    asyncio.run(test_code_suggestion())