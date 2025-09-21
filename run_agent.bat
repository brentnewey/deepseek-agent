@echo off
cd /d "%~dp0"
set PYTHONPATH=src
python src/deepseek_agent/cli.py chat %*
pause
