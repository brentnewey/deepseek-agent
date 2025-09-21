from pathlib import Path

path = Path(r"C:\Users\brent\code\deepseek-agent\src\deepseek_agent\cli.py")
lines = path.read_text().splitlines()
for i, line in enumerate(lines):
    if line.strip().startswith('candidate = match.group(1)'):
        lines[i] = '                candidate = match.group(1).strip().strip("\"\'")'
path.write_text("\n".join(lines) + "\n")
