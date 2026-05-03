#!/usr/bin/env python3
"""Patch a harbor dataset for training-time use.

Two fixes, both idempotent (safe to re-run):

1. **Empty `environment/files/` for tasks whose Dockerfile copies it but the
   directory is missing.** Without this, the E2B SDK errors at
   `Template.from_dockerfile()` with `ValueError: No files found in ...
   /environment/files/` *before* the alias is registered, so harbor cannot
   spawn a sandbox for these tasks. We add a `.gitkeep` placeholder; Docker
   COPY of an empty directory succeeds (copies nothing).

2. **(optional, default on) Switch `tests/test.sh` from binary 0/1 reward to
   fractional `passed/total` reward.** The default harbor pytest template
   does `echo 0 if PYTEST_EXIT else 1 > /logs/verifier/reward.txt`, which is
   appropriate for eval (pass@1) but ~3× sparser than necessary for RL
   training: a trial that passes 5/6 tests gives the same gradient signal as
   one that passes 0/6. We replace the if/else with a one-line python3 call
   that reads the CTRF report pytest already writes.

Usage:
  python research/pgc_swe/scripts/patch_dataset.py \
      --tasks-root <path>/scientific_computing \
      [--fractional-reward]                       # default; --no-fractional-reward to keep binary
"""
import argparse
import re
from pathlib import Path

# The exact (verbatim) reward block we expect to see in upstream
# harbor/cli/template-task/pytest-tests/test.sh as it gets baked into each
# task's tests/test.sh by the dataset adapter.
BINARY_REWARD_RE = re.compile(
    r"# Write reward based on test result\nif \[ \$PYTEST_EXIT -eq 0 \]; then\n"
    r"  echo \"All tests passed!\"\n"
    r"  echo 1 > /logs/verifier/reward\.txt\nelse\n"
    r"  echo \"Some tests failed\.\"\n"
    r"  echo 0 > /logs/verifier/reward\.txt\nfi\n"
)
FRACTIONAL_REWARD_BLOCK = (
    "# Write fractional reward (passed / total) from pytest's CTRF report.\n"
    "# Falls back to 0 if ctrf.json is unreadable. We keep exit-status-driven\n"
    "# stats elsewhere; reward.txt is what GRPO consumes.\n"
    "python3 -c \"import json,sys; d=json.load(open('/logs/verifier/ctrf.json'))['results']['summary']; "
    "sys.stdout.write(f\\\"{d['passed']/d['tests'] if d['tests']>0 else 0:.6f}\\\")\" "
    "> /logs/verifier/reward.txt 2>/dev/null || echo 0 > /logs/verifier/reward.txt\n"
    "if [ $PYTEST_EXIT -eq 0 ]; then\n  echo \"All tests passed!\"\nelse\n  echo \"Some tests failed.\"\nfi\n"
)
FRACTIONAL_MARKER = "Write fractional reward"


def fix_missing_files_dir(env_dir: Path) -> bool:
    """If Dockerfile contains COPY files/ but environment/files/ doesn't exist,
    create it with a .gitkeep placeholder. Returns True if a fix was applied."""
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.is_file():
        return False
    txt = dockerfile.read_text()
    if not re.search(r"^\s*(COPY|ADD)\s+files/", txt, re.MULTILINE):
        return False
    files_dir = env_dir / "files"
    if files_dir.is_dir():
        return False
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / ".gitkeep").touch()
    return True


def patch_test_sh_to_fractional(test_sh: Path) -> str:
    """Returns 'patched', 'already_fractional', 'shape_changed', or 'no_test_sh'."""
    if not test_sh.is_file():
        return "no_test_sh"
    src = test_sh.read_text()
    if FRACTIONAL_MARKER in src:
        return "already_fractional"
    if not BINARY_REWARD_RE.search(src):
        return "shape_changed"
    test_sh.write_text(BINARY_REWARD_RE.sub(FRACTIONAL_REWARD_BLOCK, src))
    return "patched"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, type=Path,
                    help="dataset subset dir containing per-task subdirs")
    ap.add_argument("--fractional-reward", dest="fractional",
                    action="store_true", default=True,
                    help="rewrite tests/test.sh to write passed/total to reward.txt (default)")
    ap.add_argument("--no-fractional-reward", dest="fractional",
                    action="store_false",
                    help="leave tests/test.sh alone (keep binary 0/1 reward)")
    args = ap.parse_args()

    task_dirs = sorted(p for p in args.tasks_root.iterdir()
                       if p.is_dir() and (p / "task.toml").is_file())
    if not task_dirs:
        raise SystemExit(f"no task dirs found under {args.tasks_root}")
    print(f"discovered {len(task_dirs)} tasks under {args.tasks_root}")

    n_files_fix = n_already_files = 0
    n_test_patched = n_test_already = n_test_no = n_test_shape_changed = 0
    shape_changed_tasks = []

    for d in task_dirs:
        env = d / "environment"
        if env.is_dir():
            if fix_missing_files_dir(env):
                n_files_fix += 1
            elif (env / "files").is_dir():
                n_already_files += 1

        if args.fractional:
            r = patch_test_sh_to_fractional(d / "tests" / "test.sh")
            if r == "patched":
                n_test_patched += 1
            elif r == "already_fractional":
                n_test_already += 1
            elif r == "no_test_sh":
                n_test_no += 1
            elif r == "shape_changed":
                n_test_shape_changed += 1
                shape_changed_tasks.append(d.name)

    print(f"\nempty environment/files/ created: {n_files_fix} "
          f"(already had files/: {n_already_files})")
    if args.fractional:
        print(f"tests/test.sh patched to fractional reward: {n_test_patched} "
              f"(already fractional: {n_test_already}, "
              f"no test.sh: {n_test_no}, "
              f"shape changed — left alone: {n_test_shape_changed})")
        if shape_changed_tasks:
            print("  shape-changed tasks (test.sh doesn't match expected binary "
                  "block — manual review needed):")
            for t in shape_changed_tasks[:10]:
                print(f"    {t}")
            if len(shape_changed_tasks) > 10:
                print(f"    ... and {len(shape_changed_tasks)-10} more")


if __name__ == "__main__":
    main()
