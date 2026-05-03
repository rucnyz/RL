#!/usr/bin/env python3
# Remove `docker_image = "..."` from each task.toml under a given root so that
# harbor's E2BEnvironment falls back to building the template from the local
# `environment/Dockerfile`.
#
# The Nemotron tasks ship with `docker_image` pointing to NVIDIA's internal
# gitlab registry (gitlab-master.nvidia.com:5005/...) which is unreachable
# outside NVIDIA's network. The Dockerfile in each task is self-contained
# (FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest + apt + pip), so
# E2B can build a usable template from it directly.
import argparse, re
from pathlib import Path

LINE_RE = re.compile(r"^(\s*docker_image\s*=.*)$", re.MULTILINE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, type=Path,
                    help="e.g. .../scientific_computing")
    args = ap.parse_args()

    n_total = n_changed = 0
    for toml_path in sorted(args.tasks_root.rglob("task.toml")):
        n_total += 1
        text = toml_path.read_text()
        new_text, n_subs = LINE_RE.subn(r"# \1", text, count=1)
        if n_subs:
            toml_path.write_text(new_text)
            n_changed += 1

    print(f"checked={n_total}  changed={n_changed}")


if __name__ == "__main__":
    main()
