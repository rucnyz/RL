"""Generate NeMo Gym JSONL for harbor_agent from Harbor task directories.

Harbor agent expects:
  {"instance_id": "swebench::task_name", "responses_create_params": {}, "agent_ref": {"name": "harbor_agent"}}

Usage:
    python prepare_harbor_data.py --tasks swebench-verified --alias swebench -o data/harbor_agent_data.jsonl
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
        raise FileNotFoundError(f"'{tasks}' not found and harbor CLI not installed")
    print(f"Downloading harbor tasks: {tasks}")
    result = subprocess.run(["harbor", "download", tasks, "-o", str(cached)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"harbor download failed: {result.stderr or result.stdout}")
    return cached


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Harbor tasks dir or registry name")
    parser.add_argument("--alias", required=True, help="Dataset alias for instance_id (e.g. swebench)")
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
                "instance_id": f"{args.alias}::{task_dir.name}",
                "responses_create_params": {},
                "agent_ref": {"name": "harbor_agent"},
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
