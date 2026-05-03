#!/usr/bin/env python3
# Pre-build E2B templates for every harbor task in a subset, before launching
# the trainer.
#
# Why: at training start, 8+ rollouts fire concurrently and each one calls
# harbor's `_create_template` for the SAME template name (template_name is
# derived from `task_name + dirhash(environment/)`, so all clones of the
# same task produce the same name). E2B then races those 8 build requests
# and cancels all but one with `400: build was cancelled`, the failed
# rollouts return empty, and the GRPO trainer crashes with the documented
# `IndexError: list index out of range` at `rollouts.py:1185`.
#
# Pre-building each (unique) template *once*, sequentially-ish, before the
# first rollout sidesteps the race entirely: every rollout afterward sees
# the alias already exists and reuses the cached image.
#
# Mirrors `harbor.environments.e2b.E2BEnvironment._create_template` exactly:
#   - same template_name format: `{task_name}__{dirhash(env_dir)[:8]}` with
#     `.` replaced by `-`
#   - same Template(file_context_path=env_dir).from_dockerfile(Dockerfile)
#   - same cpu_count / memory_mb from the task's `task.toml [environment]`
#
# Usage:
#   E2B_API_KEY=... python research/pgc_swe/scripts/prepare_e2b_templates.py \
#       --tasks-root <path>/scientific_computing
#
# Idempotent: re-running skips templates whose alias already exists in E2B.
import argparse
import asyncio
import os
import random
import sys
import tomllib
from pathlib import Path

from dirhash import dirhash
from e2b import AsyncTemplate, RateLimitException, Template


def _parse_memory(spec) -> int:
    """task.toml memory: '2G' / '512M' / 2048 → MiB."""
    if isinstance(spec, (int, float)):
        return int(spec)
    s = str(spec).strip().upper()
    if s.endswith("GB") or s.endswith("G"):
        return int(float(s.rstrip("BG")) * 1024)
    if s.endswith("MB") or s.endswith("M"):
        return int(float(s.rstrip("BM")))
    return int(s)


async def build_one(task_dir: Path) -> str:
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        return f"SKIP   {task_dir.name}: no environment/Dockerfile"

    task_name = task_dir.name
    template_name = f"{task_name}__{dirhash(str(env_dir), 'sha256')[:8]}".replace(".", "-")

    # NB: alias_exists returning True is not a guarantee the template can be
    # spawned. If a previous build was cancelled mid-flight (e.g. by a
    # RateLimitException) E2B can leave a "zombie" alias whose default tag
    # points to a build_id that never finalized, so every subsequent
    # `sandbox.create()` 404s with "tag 'default' does not exist for template
    # ...". get_tags doesn't catch this either (the tag IS in metadata; only
    # the underlying image is missing). For incident recovery, run
    # scripts/repair_broken_templates.py with skip_cache=True against the
    # affected tasks instead of paying a per-template probe here.
    if await AsyncTemplate.alias_exists(template_name):
        return f"CACHED {template_name}"

    with open(task_dir / "task.toml", "rb") as f:
        task_cfg = tomllib.load(f)
    env_cfg = task_cfg.get("environment", {})
    cpus = int(env_cfg.get("cpus", 1))
    # Two conventions in the wild: harbor's `memory = "2G"` (Nemotron) and the
    # newer `memory_mb = 4096` raw int (swebenchpro adapter, harbor 0.2+).
    # Prefer memory_mb if set; fall back to parsing the legacy memory string.
    if "memory_mb" in env_cfg:
        memory_mb = int(env_cfg["memory_mb"])
    else:
        memory_mb = _parse_memory(env_cfg.get("memory", "2G"))

    template = Template(file_context_path=str(env_dir)).from_dockerfile(
        dockerfile_content_or_path=str(dockerfile),
    )
    # E2B caps concurrent builds per org at 20. Even with --concurrency 4 on
    # our side, builds queue up server-side and `AsyncTemplate.build()` will
    # reject new submissions with `429: RateLimitException` when the org's
    # in-flight count >= 20. Backoff and retry rather than dropping the task.
    backoff = 30.0
    for attempt in range(8):
        try:
            await AsyncTemplate.build(
                template=template,
                alias=template_name,
                cpu_count=cpus,
                memory_mb=memory_mb,
            )
            return f"BUILT  {template_name}"
        except RateLimitException:
            if attempt == 7:
                raise
            jitter = random.uniform(0, backoff * 0.3)
            await asyncio.sleep(backoff + jitter)
            backoff = min(backoff * 1.5, 240.0)
    return f"BUILT  {template_name}"  # unreachable but mypy-friendly


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, type=Path, nargs="+",
                    help="one or more subset dirs (each containing per-task "
                         "subdirs with task.toml + environment/Dockerfile). "
                         "e.g. .../skill_based/mixed/scientific_computing "
                         "[.../skill_based/mixed/data_processing ...]")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="how many fresh builds to run in parallel (each "
                         "build is a *different* template, so no race; "
                         "E2B handles this fine — default 4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only build the first N tasks per subset (debug)")
    args = ap.parse_args()

    if not os.environ.get("E2B_API_KEY"):
        sys.exit("E2B_API_KEY not set")

    task_dirs: list[Path] = []
    for root in args.tasks_root:
        sub = sorted(p for p in root.iterdir()
                     if p.is_dir() and (p / "task.toml").is_file())
        if args.limit:
            sub = sub[: args.limit]
        print(f"discovered {len(sub)} tasks under {root}", flush=True)
        task_dirs.extend(sub)
    print(f"-> {len(task_dirs)} tasks total to build", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    n_done = [0]

    async def guarded(task: Path) -> None:
        async with sem:
            try:
                line = await build_one(task)
            except Exception as e:
                line = f"FAILED {task.name}: {type(e).__name__}: {e}"
            n_done[0] += 1
            print(f"[{n_done[0]:>4}/{len(task_dirs)}] {line}", flush=True)

    await asyncio.gather(*(guarded(t) for t in task_dirs))
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
