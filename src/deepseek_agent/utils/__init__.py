"""Utility functions for DeepSeek Agent"""

from .ollama_service import (
    is_ollama_running,
    start_ollama_service,
    ensure_ollama_running,
    get_ollama_install_instructions
)

__all__ = [
    'is_ollama_running',
    'start_ollama_service',
    'ensure_ollama_running',
    'get_ollama_install_instructions'
]