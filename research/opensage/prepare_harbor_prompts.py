"""Generate prompts JSONL from Harbor task directories for NeMo RL.

Scans a Harbor tasks directory and extracts instruction.md from each task
into a JSONL file that NeMo RL's data loader can read.

Usage:
    # From harbor registry (auto-downloads)
    python dev/prepare_harbor_prompts.py --tasks swebench --output /root/harbor_prompts.jsonl

    # From local directory
    python dev/prepare_harbor_prompts.py --tasks /data/my_tasks --output /root/harbor_prompts.jsonl
"""

import json
import shutil
import subprocess
from pathlib import Path

HARBOR_TASKS_CACHE = Path.home() / ".cache" / "harbor" / "tasks"


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
                "prompt": instruction,
                "task_id": task_dir.name,
                "task_dir": str(task_dir),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
