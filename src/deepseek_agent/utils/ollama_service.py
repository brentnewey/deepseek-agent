"""Ollama service management utilities"""

import asyncio
import platform
import subprocess
import time
from typing import Optional
import httpx


async def is_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama service is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{host}/api/tags")
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def start_ollama_service() -> tuple[bool, str]:
    """
    Start Ollama service based on the operating system.
    Returns (success, message)
    """
    system = platform.system().lower()

    try:
        if system == "darwin":  # macOS
            # Try to start Ollama app
            subprocess.run(["open", "-a", "Ollama"], check=False, capture_output=True)
            time.sleep(3)  # Give it time to start

            # Check if it started
            result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                return True, "Ollama started successfully on macOS"
            else:
                return False, "Failed to start Ollama on macOS. Please start it manually."

        elif system == "linux":
            # Try systemctl first (for systems with systemd)
            result = subprocess.run(
                ["systemctl", "start", "ollama"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return True, "Ollama service started via systemctl"

            # Try direct ollama serve command
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(3)
            return True, "Ollama started in background"

        elif system == "windows":
            # On Windows, try to start ollama serve in background
            try:
                # First check if ollama.exe exists in PATH
                result = subprocess.run(
                    ["where", "ollama"],
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode != 0:
                    return False, "Ollama not found in PATH. Please install Ollama from https://ollama.ai"

                # Start ollama serve in a new window
                subprocess.Popen(
                    ["cmd", "/c", "start", "/min", "ollama", "serve"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(5)  # Windows might need more time
                return True, "Ollama service started in background on Windows"

            except Exception as e:
                return False, f"Failed to start Ollama on Windows: {e}"

        else:
            return False, f"Unsupported operating system: {system}"

    except FileNotFoundError:
        return False, "Ollama not found. Please install it from https://ollama.ai"
    except Exception as e:
        return False, f"Failed to start Ollama service: {e}"


async def ensure_ollama_running(host: str = "http://localhost:11434", max_retries: int = 3) -> tuple[bool, str]:
    """
    Ensure Ollama is running, starting it if necessary.
    Returns (success, message)
    """
    # First check if it's already running
    if await is_ollama_running(host):
        return True, "Ollama service is already running"

    # Try to start it
    for attempt in range(max_retries):
        success, message = start_ollama_service()

        if not success and attempt == max_retries - 1:
            return False, message

        # Wait a bit for the service to start
        await asyncio.sleep(3)

        # Check if it's running now
        for _ in range(10):  # Check for up to 10 seconds
            if await is_ollama_running(host):
                return True, f"Ollama service started successfully (attempt {attempt + 1})"
            await asyncio.sleep(1)

    return False, "Failed to start Ollama service after multiple attempts"


def get_ollama_install_instructions() -> str:
    """Get platform-specific installation instructions for Ollama"""
    system = platform.system().lower()

    if system == "darwin":
        return """
To install Ollama on macOS:
1. Visit https://ollama.ai
2. Download the macOS installer
3. Run the installer
4. Launch Ollama from Applications
"""
    elif system == "linux":
        return """
To install Ollama on Linux:
Run: curl -fsSL https://ollama.ai/install.sh | sh
"""
    elif system == "windows":
        return """
To install Ollama on Windows:
1. Visit https://ollama.ai
2. Download the Windows installer
3. Run the installer
4. Ollama will be added to your PATH
"""
    else:
        return "Visit https://ollama.ai for installation instructions"