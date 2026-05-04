#!/usr/bin/env bash
# Generic "register a harbor dataset" pipeline.
#
# Harbor's 80+ adapters (Nemotron-Terminal-Synthetic-Tasks, terminal-bench,
# aider_polyglot, swe-bench, ds1000, ...) all emit the SAME on-disk format:
#
#   <tasks_root>/
#     <task_name_1>/
#       task.toml                  # [environment] cpus, memory, [docker_image]
#       environment/
#         Dockerfile               # base image + apt + pip + COPY files/
#         files/                   # per-task input data (varies)
#       solution/                  # gold solution (used by verifier)
#       tests/                     # verifier scripts
#       instruction.md             # task prompt
#     <task_name_2>/...
#     ...
#
# This script takes one or more `<alias>=<tasks_root>` pairs and:
#   1. (optional, --strip-docker-image) comments out `docker_image = "..."` in
#      every task.toml — needed when the docker_image points at an
#      inaccessible registry (e.g. NVIDIA's internal gitlab for the Nemotron
#      adapter); harbor then falls back to building from the local Dockerfile
#   2. Generates `<alias>_{train,val}.jsonl` under research/pgc_swe/data/
#      (90/10 split by default)
#   3. Pre-builds every task's E2B template **sequentially-ish** so the first
#      training step's parallel rollouts don't all race on `_create_template`
#      for the same template name (harbor + E2B don't deduplicate concurrent
#      builds; without prebuild, 8 parallel rollouts of the same task
#      register-then-cancel each other and the trainer crashes on empty
#      rollouts with the documented IndexError at rollouts.py:1185)
#
# After this finishes, paste the printed yaml block into your recipe's
# `env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.harbor_datasets`
# map and update `data.train.data_path` / `data.validation.data_path` to point
# at the new jsonl files.
#
# Usage:
#   E2B_API_KEY=... bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
#       [--strip-docker-image] \
#       [--no-jsonl] [--no-prebuild] \
#       [--val-frac 0.1] [--prebuild-concurrency 4] [--prebuild-limit N] \
#       <alias>=<tasks_root> [<alias>=<tasks_root> ...]
#
# Examples:
#   # Nemotron scientific subset (needs docker_image strip)
#   bash prepare_harbor_dataset.sh --strip-docker-image \
#       scientific=$DATASET_ROOT/skill_based/mixed/scientific_computing
#
#   # Multiple Nemotron subsets at once
#   bash prepare_harbor_dataset.sh --strip-docker-image \
#       scientific=$DATASET_ROOT/skill_based/mixed/scientific_computing \
#       debugging=$DATASET_ROOT/skill_based/mixed/debugging \
#       file_ops=$DATASET_ROOT/skill_based/mixed/file_operations
#
#   # An adapter-generated dataset (its docker_image refs are public, no strip)
#   bash prepare_harbor_dataset.sh polyglot=$HOME/datasets/aider_polyglot
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PGC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PGC_DIR}/../.." && pwd)"

if [ -f "${PGC_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PGC_DIR}/.env"
    set +a
fi

STRIP_DOCKER_IMAGE=false
DO_JSONL=true
DO_PREBUILD=true
DO_PATCH=true
DO_FRACTIONAL_REWARD=true
VAL_FRAC="0.1"
PREBUILD_CONCURRENCY="4"
PREBUILD_LIMIT=""

PAIRS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --strip-docker-image)    STRIP_DOCKER_IMAGE=true; shift ;;
        --no-jsonl)              DO_JSONL=false; shift ;;
        --no-prebuild)           DO_PREBUILD=false; shift ;;
        --no-patch)              DO_PATCH=false; shift ;;
        --binary-reward)         DO_FRACTIONAL_REWARD=false; shift ;;
        --val-frac)              VAL_FRAC="$2"; shift 2 ;;
        --prebuild-concurrency)  PREBUILD_CONCURRENCY="$2"; shift 2 ;;
        --prebuild-limit)        PREBUILD_LIMIT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,60p' "$0"
            exit 0
            ;;
        --*)
            echo "unknown flag: $1" >&2; exit 2 ;;
        *)
            PAIRS+=("$1"); shift ;;
    esac
done

if [ "${#PAIRS[@]}" -eq 0 ]; then
    echo "usage: $0 [options] <alias>=<tasks_root> [<alias>=<tasks_root> ...]" >&2
    echo "  see top of file for examples." >&2
    exit 2
fi

if [ "${DO_PREBUILD}" = "true" ]; then
    : "${E2B_API_KEY:?E2B_API_KEY missing — put it in research/pgc_swe/.env or pass via env}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_OUT="${PGC_DIR}/data"
mkdir -p "${DATA_OUT}"

ROOTS=()
ALIASES=()
for pair in "${PAIRS[@]}"; do
    if [[ "${pair}" != *=* ]]; then
        echo "ERROR: argument must be <alias>=<tasks_root> (got: ${pair})" >&2
        exit 2
    fi
    alias_name="${pair%%=*}"
    tasks_root="${pair#*=}"
    if [ ! -d "${tasks_root}" ]; then
        echo "ERROR: ${tasks_root} is not a directory" >&2
        exit 2
    fi
    n_tasks=$(find "${tasks_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "[${alias_name}] ${tasks_root}  (${n_tasks} task dirs)"
    ROOTS+=("${tasks_root}")
    ALIASES+=("${alias_name}")
done
echo

set -x

# 1. strip docker_image (optional — only needed for adapters that ship
#    inaccessible registry refs, like Nemotron pointing at gitlab-master)
if [ "${STRIP_DOCKER_IMAGE}" = "true" ]; then
    for root in "${ROOTS[@]}"; do
        "${PYTHON}" "${SCRIPT_DIR}/strip_docker_image.py" --tasks-root "${root}"
    done
fi

# 1b. dataset bug fixes for known nemotron failure modes (idempotent)
#
# (a) Some tasks ship `Dockerfile` containing `COPY files/ /app/` but no
#     `environment/files/` directory. The E2B SDK errors at
#     `Template.from_dockerfile()` with `ValueError: No files found in ...
#     /environment/files/` *before* the alias is even registered, so harbor
#     can never spawn a sandbox for these tasks. Fix: create an empty
#     `files/` directory (Docker COPY of an empty dir succeeds and copies
#     nothing — agent writes everything itself based on instruction.md).
#
# (b) The default `tests/test.sh` shipped by harbor's pytest task template
#     writes binary 0/1 to `/logs/verifier/reward.txt` based on pytest's
#     exit status. Switching to fractional reward (`passed / total`) gives
#     ~3× denser gradient signal because partial-pass trials (often the
#     majority on hard tasks) stop being silently zeroed out. Override with
#     `--binary-reward` if you need the eval-style semantics.
if [ "${DO_PATCH}" = "true" ]; then
    for root in "${ROOTS[@]}"; do
        "${PYTHON}" "${SCRIPT_DIR}/patch_dataset.py" \
            --tasks-root "${root}" \
            $([ "${DO_FRACTIONAL_REWARD}" = "true" ] && echo "--fractional-reward")
    done
fi

# 2. generate JSONL per (alias, root)
if [ "${DO_JSONL}" = "true" ]; then
    for i in "${!ROOTS[@]}"; do
        "${PYTHON}" "${SCRIPT_DIR}/prep_harbor_jsonl.py" \
            --tasks-root "${ROOTS[$i]}" \
            --alias "${ALIASES[$i]}" \
            --out-dir "${DATA_OUT}" \
            --val-frac "${VAL_FRAC}"
    done
fi

# 3. prebuild templates across all roots in one pass (the prebuild script
#    accepts multiple --tasks-root args).
#
# Prefers the harbor_agent venv (whose `[e2b]` extra is what defined the SDK
# pull-in originally). Falls back to the main NeMo RL .venv if the harbor
# venv hasn't been built yet AND main .venv happens to already have the SDK
# (e.g. on H200 where main .venv was synced with the e2b dep before the
# harbor venv split was introduced). Errors out if neither has it.
HARBOR_AGENT_DIR="${REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent"
HARBOR_PYTHON="${HARBOR_AGENT_DIR}/.venv/bin/python"
MAIN_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ "${DO_PREBUILD}" = "true" ]; then
    if [ -x "${HARBOR_PYTHON}" ]; then
        PREBUILD_PYTHON="${HARBOR_PYTHON}"
    elif [ -x "${MAIN_PYTHON}" ] && "${MAIN_PYTHON}" -c "import e2b" 2>/dev/null; then
        echo "[prepare_harbor_dataset] harbor venv missing; falling back to main .venv (it has e2b)" >&2
        PREBUILD_PYTHON="${MAIN_PYTHON}"
    else
        echo "ERROR: no python with the e2b SDK available." >&2
        echo "       Build the harbor venv with:  bash research/pgc_swe/run_harbor_e2b.sh" >&2
        echo "       (or install e2b into ${MAIN_PYTHON%/python}/, then re-run this)." >&2
        exit 2
    fi
    prebuild_args=(--tasks-root "${ROOTS[@]}" --concurrency "${PREBUILD_CONCURRENCY}")
    if [ -n "${PREBUILD_LIMIT}" ]; then
        prebuild_args+=(--limit "${PREBUILD_LIMIT}")
    fi
    "${PREBUILD_PYTHON}" "${SCRIPT_DIR}/prepare_e2b_templates.py" "${prebuild_args[@]}"
fi

set +x

echo
echo "All done. Paste this into your recipe yaml's harbor_datasets map:"
echo
echo "      env:"
echo "        nemo_gym:"
echo "          harbor_agent:"
echo "            responses_api_agents:"
echo "              harbor_agent:"
echo "                harbor_agent_kwargs:"
echo "                  harbor_datasets:"
for i in "${!ROOTS[@]}"; do
    echo "                    ${ALIASES[$i]}:"
    echo "                      local_dataset_path: \"${ROOTS[$i]}\""
    echo "                      workdir: \"/app\""
done
echo
echo "And point your training command at the new jsonl files:"
for alias_name in "${ALIASES[@]}"; do
    echo "  data.train.data_path=${DATA_OUT}/${alias_name}_train.jsonl"
    echo "  data.validation.data_path=${DATA_OUT}/${alias_name}_val.jsonl"
done
