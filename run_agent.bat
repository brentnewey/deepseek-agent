@echo off
cd /d "%~dp0"
set PYTHONPATH=src
python src/deepseek_agent/cli_with_tools.py --model llama3.1 %*
pause
