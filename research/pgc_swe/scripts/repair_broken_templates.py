"""One-shot: force-rebuild a manually-curated set of broken templates on E2B.

The current list is the 21 tasks in nemotron scientific_computing whose
Dockerfile contains `COPY files/ /app/` but ship without `environment/files/`.
Such tasks would be silently skipped by every harbor `_create_template`
attempt: the E2B SDK errors at `Template.from_dockerfile()` with
`ValueError: No files found in .../environment/files/` *before* the alias is
ever registered, so re-running prebuild won't help.

Fix is two-part:
  1. `mkdir environment/files` (touched in the dataset; only needs to happen
     once per task). Lets the E2B SDK accept the build.
  2. This script: force-rebuilds the templates with `skip_cache=True`.

Why skip_cache:
  * tasks 0250 and 0287 have a stale `default` tag in E2B metadata pointing to
    a cancelled build_id. `alias_exists` returns True and `get_tags` returns
    the tag, but every `sandbox.create()` 404s with
    `tag 'default' does not exist for template ...`. Without skip_cache the
    standard prebuild script's idempotence check (`alias_exists -> CACHED`)
    keeps the zombie alive.
  * The other 19 tasks have no alias at all (the SDK ValueError prevented
    registration) — skip_cache=True is a no-op for them.

Concurrency capped at 4 to stay under E2B's 20-concurrent-build org cap;
on top of that we have an 8-attempt exponential backoff for RateLimitException.
"""
import asyncio, os, sys, tomllib
from pathlib import Path
import httpx
from dirhash import dirhash
from e2b import AsyncTemplate, RateLimitException, Template

E2B_API = "https://api.e2b.dev"


def clear_waiting_builds() -> int:
    """Delete all templates currently in `waiting` buildStatus.

    These are zombie builds left by cancelled/interrupted `AsyncTemplate.build`
    calls (e.g. RateLimit during the upload phase). They count against E2B's
    20-concurrent-build org cap forever — every retry that hits RateLimit
    *adds another zombie*, so the situation snowballs into a permanent cap
    saturation that no amount of backoff can recover from. The only fix is
    DELETE-ing them via the templates REST API.

    Returns the number of zombies deleted.
    """
    key = os.environ.get("E2B_API_KEY")
    if not key:
        return 0
    headers = {"X-API-Key": key}
    try:
        r = httpx.get(f"{E2B_API}/templates", headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  (clear_waiting_builds: list failed: {e})")
        return 0
    waiting = [t for t in r.json() if t.get("buildStatus") == "waiting"]
    if not waiting:
        return 0
    print(f"  found {len(waiting)} zombie 'waiting' templates blocking the build cap; deleting them")
    n_deleted = 0
    for t in waiting:
        tid = t.get("templateID")
        if not tid:
            continue
        try:
            httpx.delete(f"{E2B_API}/templates/{tid}", headers=headers, timeout=15)
            n_deleted += 1
        except Exception:
            pass
    return n_deleted

TASKS_ROOT = Path("/scratch/yuzhou/projects/RL/3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks/skill_based/mixed/scientific_computing")
TASK_IDS = ["0020","0022","0038","0090","0250","0263","0287","0332","0350","0385",
            "0491","0539","0563","0647","0694","0755","0819","0860","0876","0882","0931"]


def parse_mem(spec):
    if isinstance(spec, (int, float)): return int(spec)
    s = str(spec).strip().upper()
    if s.endswith("GB") or s.endswith("G"): return int(float(s.rstrip("BG")) * 1024)
    if s.endswith("MB") or s.endswith("M"): return int(float(s.rstrip("BM")))
    return int(s)


async def rebuild_one(task_dir, idx, total, sem):
    async with sem:
        env = task_dir / "environment"
        df = env / "Dockerfile"
        if not df.exists():
            return f"[{idx}/{total}] SKIP   {task_dir.name}: no Dockerfile"
        alias = f"{task_dir.name}__{dirhash(str(env), 'sha256')[:8]}".replace(".", "-")
        with open(task_dir / "task.toml", "rb") as f:
            cfg = tomllib.load(f).get("environment", {})
        cpus = int(cfg.get("cpus", 1))
        mem = (int(cfg["memory_mb"]) if "memory_mb" in cfg
               else parse_mem(cfg.get("memory", "2G")))
        # was_zombie = alias existed before we started?
        existed = await AsyncTemplate.alias_exists(alias)
        tpl = Template(file_context_path=str(env)).from_dockerfile(str(df))
        backoff = 30.0
        for attempt in range(8):
            try:
                await AsyncTemplate.build(template=tpl, alias=alias,
                                          cpu_count=cpus, memory_mb=mem,
                                          skip_cache=True)
                return f"[{idx}/{total}] {'REBUILT' if existed else 'BUILT  '} {alias}"
            except RateLimitException:
                if attempt == 7: raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 240.0)


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel build slots (default 1 — serial). E2B's "
                         "20-build cap is org-wide so concurrent retries from "
                         "us tend to dogpile and re-fail at the same instant.")
    args = ap.parse_args()

    if not os.environ.get("E2B_API_KEY"):
        sys.exit("E2B_API_KEY not set")

    # Clear any zombie waiting builds before we start. Without this, every
    # subsequent build request will 429 even though no real build is running.
    n_cleared = clear_waiting_builds()
    if n_cleared:
        print(f"  cleared {n_cleared} zombie 'waiting' templates from E2B\n")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for i, tid in enumerate(TASK_IDS, 1):
        d = TASKS_ROOT / f"scientific_computing_task_{tid}"
        if not d.is_dir():
            print(f"[{i}/{len(TASK_IDS)}] MISSING {d}")
            continue
        tasks.append(rebuild_one(d, i, len(TASK_IDS), sem))
    for coro in asyncio.as_completed(tasks):
        try:
            print(await coro, flush=True)
        except Exception as e:
            print(f"FAILED {type(e).__name__}: {e}", flush=True)
    print("done")


asyncio.run(main())
