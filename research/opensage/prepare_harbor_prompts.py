"""Generate NeMo Gym-compatible JSONL from Harbor task directories.

Scans a Harbor tasks directory and produces JSONL with:
- System prompt (harbor_agent's coding agent prompt)
- User message (instruction.md)
- Tool definitions (run_terminal_command, view_file, str_replace_edit, finish_task)
- verifier_metadata (task_id, task_dir for verification)

Usage:
    python prepare_harbor_prompts.py --tasks swebench-verified -o data/harbor_prompts.jsonl
"""

import json
import shutil
import subprocess
from pathlib import Path

HARBOR_TASKS_CACHE = Path.home() / ".cache" / "harbor" / "tasks"

SYSTEM_PROMPT = """# System Prompt: Terminal Coding Agent

## Role
You are an expert coding assistant operating in a Linux terminal environment. Your role is to help users complete coding tasks efficiently and accurately.
Carefully read the task description and list the requirements provided by the user.

## Environment
- You are operating in a **sandboxed environment** where you have full freedom to experiment
- Use pip, npm, apt-get, or any other package manager as required
- Don't worry about breaking things - the sandbox is isolated and safe for experimentation

## Core Principles

### 1. Always Verify Your Work
Before considering any task complete:
- **Run the code** to ensure it executes without errors
- **Test with example inputs** to verify correct output
- **Check edge cases** where applicable
- If writing tests, **execute them** and confirm they pass

### 2. Review Task Requirements Before Finishing
Before marking any task as complete:
- **Re-read the original task description** carefully
- **Check each requirement** has been addressed
- **Verify all specified features** are implemented
- **Confirm the output format** matches what was requested
- Ask yourself: "Have I fully solved what was asked?"

### 3. Best Practices
- Show your working and explain your approach
- If you encounter errors, debug systematically
- Document your code with clear comments when helpful

Remember: Taking time to verify and review prevents mistakes and ensures quality results.

Before finishing, try to run existing tests or write new tests to validate your changes. But be careful not to break existing environments.

At last, state what you have done and how you finished the task."""

TOOLS = [
    {
        "name": "run_terminal_command",
        "type": "function",
        "description": "Execute a bash command in the sandboxed terminal environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "view_file",
        "type": "function",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to view",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "str_replace_edit",
        "type": "function",
        "description": "Replace the first occurrence of old_string with new_string in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "finish_task",
        "type": "function",
        "description": "Signal that you have completed the task. Call this when you are done.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def resolve_tasks_path(tasks: str) -> Path:
    local = Path(tasks)
    if local.is_dir():
        return local

    cached = HARBOR_TASKS_CACHE / tasks
    if cached.is_dir():
        return cached

    if shutil.which("harbor") is None:
        raise FileNotFoundError(
            f"'{tasks}' is not a local directory and `harbor` CLI is not installed."
        )

    print(f"Downloading harbor tasks: {tasks}")
    result = subprocess.run(
        ["harbor", "download", tasks, "-o", str(cached)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"harbor download failed:\n{result.stderr or result.stdout}")

    return cached


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Harbor tasks dir or registry name")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    args = parser.parse_args()

    tasks_path = resolve_tasks_path(args.tasks)
    count = 0

    with open(args.output, "w") as f:
        for task_dir in sorted(tasks_path.iterdir()):
            if not task_dir.is_dir():
                continue

            instruction_file = task_dir / "instruction.md"
            if not instruction_file.exists():
                continue

            instruction = instruction_file.read_text().strip()
            if not instruction:
                continue

            record = {
                "responses_create_params": {
                    "input": [
                        {"role": "developer", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                    ],
                    "tools": TOOLS,
                },
                "verifier_metadata": {
                    "task_id": task_dir.name,
                    "task_dir": str(task_dir),
                },
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
