#!/usr/bin/env python3
# Build train/val JSONL files for the harbor + E2B recipe.
#
# A "task" in Nemotron-Terminal-Synthetic-Tasks is one subdirectory of a
# `skill_based/mixed/<subset>` folder, each containing its own task.toml +
# Dockerfile. The harbor agent expects rollout requests of the form:
#   {"instance_id": "<dataset_alias>::<task_dir_name>",
#    "responses_create_params": {"input": []},
#    "agent_ref": {"name": "harbor_agent"}}
# The dataset alias must match `harbor_datasets.<alias>` in the recipe yaml.
import argparse, json, random
from pathlib import Path


def discover(tasks_root: Path) -> list[str]:
    out = []
    for p in sorted(tasks_root.iterdir()):
        if p.is_dir() and (p / "task.toml").is_file():
            out.append(p.name)
    if not out:
        raise SystemExit(
            f"No tasks (directories with task.toml) found under {tasks_root}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, type=Path,
                    help="e.g. .../scientific_computing")
    ap.add_argument("--alias", required=True,
                    help="dataset alias as configured in harbor_agent.yaml "
                         "(e.g. 'scientific')")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = discover(args.tasks_root)
    random.Random(args.seed).shuffle(names)

    n_val = max(1, int(len(names) * args.val_frac))
    val_names, train_names = names[:n_val], names[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.alias}_train.jsonl"
    val_path = args.out_dir / f"{args.alias}_val.jsonl"

    def write(path: Path, names: list[str]) -> None:
        with path.open("w") as f:
            for name in names:
                f.write(json.dumps({
                    "instance_id": f"{args.alias}::{name}",
                    "responses_create_params": {"input": []},
                    "agent_ref": {"name": "harbor_agent"},
                }) + "\n")

    write(train_path, train_names)
    write(val_path, val_names)
    print(f"train={len(train_names)} val={len(val_names)}  ->  "
          f"{train_path}  {val_path}")


if __name__ == "__main__":
    main()
