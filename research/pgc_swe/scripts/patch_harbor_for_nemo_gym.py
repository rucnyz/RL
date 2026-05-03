#!/usr/bin/env python3
"""Patch a built harbor venv for nemo-gym compatibility.

Background
----------
We pin harbor to commit `9dddd79` (2025-08-04, the same SHA the upstream
`Gym` submodule's `responses_api_agents/harbor_agent/requirements.txt`
points at). That release is API-compatible with nemo-gym 0.3.0rc0
(`harbor.models.job.config.LocalDatasetConfig` etc. still exist) but is
missing one fix landed later in PR #1397 (`74c71f5`):

    `Template(file_context_path=...)` so COPY srcs in a task's Dockerfile
    resolve against the task's `environment/` dir instead of harbor's
    own install dir.

Without that fix, every E2B template build fails:

    ValueError: No files found in
    .../site-packages/harbor/environments/files/

(Newer harbor 0.6.x has the fix natively, but it also breaks nemo-gym in
multiple other ways — see harbor_agent_requirements.txt for the full
analysis. Easier to backport this one fix than maintain a growing pile
of compat shims.)

What we do
----------
Edit the venv's `harbor/environments/e2b.py` to thread
`file_context_path=str(self.environment_dir)` into the `Template(...)`
call. Idempotent — re-running detects the existing patch and is a no-op.

Usage
-----
    python scripts/patch_harbor_for_nemo_gym.py <harbor_venv_dir>
"""
import argparse
import sys
from pathlib import Path


# The exact source block as it ships in harbor 9dddd79
NEEDLE = (
    "        else:\n"
    "            template = Template().from_dockerfile(\n"
    "                dockerfile_content_or_path=str(self._environment_definition_path),\n"
    "            )\n"
)
PATCH = (
    "        else:\n"
    "            # Patched (research/pgc_swe/scripts/patch_harbor_for_nemo_gym.py):\n"
    "            # backport of upstream PR #1397 (74c71f5). Pass file_context_path\n"
    "            # so COPY srcs in the task's Dockerfile resolve against the task's\n"
    "            # environment/ dir, not against harbor library's install dir\n"
    "            # (the E2B SDK's default behavior).\n"
    "            template = Template(\n"
    "                file_context_path=str(self.environment_dir),\n"
    "            ).from_dockerfile(\n"
    "                dockerfile_content_or_path=str(self._environment_definition_path),\n"
    "            )\n"
)
PATCH_MARKER = "Patched (research/pgc_swe/scripts/patch_harbor_for_nemo_gym.py)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("harbor_venv", type=Path,
                    help="path to the harbor_agent venv "
                         "(e.g. 3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/.venv)")
    args = ap.parse_args()

    e2b_py = args.harbor_venv / "lib" / "python3.13" / "site-packages" / \
             "harbor" / "environments" / "e2b.py"
    if not e2b_py.is_file():
        print(f"ERROR: {e2b_py} not found — is the harbor venv built?",
              file=sys.stderr)
        return 2

    src = e2b_py.read_text()

    if PATCH_MARKER in src:
        print(f"already patched: {e2b_py}")
        return 0

    if "file_context_path=str(self.environment_dir)" in src:
        print(f"NOTE: harbor.environments.e2b already passes file_context_path "
              f"natively — patch is unnecessary, skipping.")
        return 0

    if NEEDLE not in src:
        print(f"ERROR: harbor/environments/e2b.py shape changed — needle not found. "
              f"Refresh this script for the harbor version pinned in "
              f"harbor_agent_requirements.txt.", file=sys.stderr)
        return 2

    e2b_py.write_text(src.replace(NEEDLE, PATCH))
    print(f"patched: {e2b_py}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
