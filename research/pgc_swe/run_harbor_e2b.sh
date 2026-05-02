#!/usr/bin/env bash
# Launch the harbor + E2B PGC SWE recipe.
# Run from the NeMo RL repo root (`/scratch/yuzhou/projects/RL` here).
#
# Loads E2B_API_KEY + WANDB_API_KEY from research/pgc_swe/.env (gitignored)
# so the user does not have to remember to `export` them manually each session.
#
# Pass extra Hydra overrides on the command line, e.g.:
#   bash research/pgc_swe/run_harbor_e2b.sh \
#       data.train.data_path=$DATA_DIR/scientific_train.jsonl \
#       data.validation.data_path=$DATA_DIR/scientific_val.jsonl \
#       grpo.max_num_steps=20

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load secrets
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi
: "${E2B_API_KEY:?E2B_API_KEY missing — put it in research/pgc_swe/.env}"

cd "${REPO_ROOT}"

python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config research/pgc_swe/configs/grpo-qwen3.5-9b-1n8g-harbor-e2b.yaml \
    "$@"
