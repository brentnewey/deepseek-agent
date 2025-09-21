"""Tool definitions for the DeepSeek Agent"""

from typing import Dict, List, Any

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return tool definitions in Ollama format"""
    return [
        {
            'type': 'function',
            'function': {
                'name': 'write_file',
                'description': 'Create or overwrite a file with the given content',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'file_path': {
                            'type': 'string',
                            'description': 'The path to the file to create or overwrite',
                        },
                        'content': {
                            'type': 'string',
                            'description': 'The content to write to the file',
                        },
                    },
                    'required': ['file_path', 'content'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'description': 'Read the contents of a file',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'file_path': {
                            'type': 'string',
                            'description': 'The path to the file to read',
                        },
                    },
                    'required': ['file_path'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'list_directory',
                'description': 'List files and directories in a given path',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'directory': {
                            'type': 'string',
                            'description': 'The directory path to list (default: current directory)',
                        },
                    },
                    'required': [],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'run_command',
                'description': 'Execute a shell command in the workspace',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'type': 'string',
                            'description': 'The shell command to execute',
                        },
                    },
                    'required': ['command'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'find_files',
                'description': 'Find files matching a pattern in the workspace',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'pattern': {
                            'type': 'string',
                            'description': 'The file pattern to search for (e.g., "*.py")',
                        },
                    },
                    'required': ['pattern'],
                },
            },
        },
    ]