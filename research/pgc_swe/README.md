# PGC SWE — long-horizon RL on SWE-bench (NeMo RL + Qwen3.5-9B + 1 node)

Research project for our PGC paper: a 1-node Qwen3.5-9B + SWE-bench training
recipe built on NeMo RL + NeMo Gym, intended as the baseline we'll add
Proportional Gradient Clipping on top of.

## Why this project exists

Earlier we tried `rllm + verl + harbor + remote_runtime + fully_async` and
hit ~13 bugs end-to-end. The two most fundamental problems:

1. The combination is genuinely unsupported upstream. PR #495 ("harbor remote
   runtime") in rllm even admits "needs e2e training example". PR #493 ("fix:
   unified async trainer with verl backend") is still open and lists 5 bugs.
2. **Token-level data flow is broken**: mini-swe-agent's litellm calls don't
   request logprobs, so rllm-model-gateway captures empty traces, episodes
   contain 0 tokens, and verl's `karmarkar_karp` ends with `IndexError`.

NeMo Gym was explicitly designed for RL token capture — every responses_api
agent (harbor_agent, swe_agents/openhands, swe_agents/swe_agent, terminus,
mini_swe_agent, etc.) sets `collect_rollout_details=True` to surface
per-token IDs/logprobs back to NeMo RL's GRPO trainer. That's the missing
piece on the rllm side.

## Layout

```
research/pgc_swe/
├── README.md                                      # this file
├── pyproject.toml                                 # uv workspace member
├── pgc_swe/                                       # python module (PGC code goes here)
│   ├── __init__.py
│   ├── data_utils.py
│   └── worker_extension.py
└── configs/
    ├── grpo-qwen3.5-9b-1n8g-swe.yaml              # recipe A: real SWE-bench via OpenHands + Apptainer
    ├── grpo-qwen3.5-9b-1n8g-harbor-e2b.yaml       # recipe B: harbor terminal tasks via E2B (default)
    └── reference/                                 # untouched upstream files for diffing
        ├── grpo-qwen3.5-9b-1n8g-megatron.yaml     # NeMo RL main: Qwen3.5-9B math (4k ctx)
        ├── grpo-nemotron-nano-16n8g-async-1off-swe.yaml  # NeMo RL super-v3: Nemotron + SWE (16 nodes)
        ├── swebench_openhands_training.yaml       # NeMo Gym: OpenHands SWE-bench training agent
        └── harbor_agent.yaml                      # NeMo Gym: harbor terminal agent (RL-aware, alt option)
```

## Two recipes — pick one

| | A. `grpo-qwen3.5-9b-1n8g-swe.yaml` | B. `grpo-qwen3.5-9b-1n8g-harbor-e2b.yaml` ★ default |
|---|---|---|
| Dataset | SWE-Gym + SWE-bench Verified | `nvidia/Nemotron-Terminal-Synthetic-Tasks` (6 skill subsets) |
| Agent | NeMo Gym `swe_agents` (OpenHands CodexAgent / OpenCodeAgent / CodeActAgent) | NeMo Gym `harbor_agent` (Terminus-2 via NemoGymLLM) |
| Sandbox | Apptainer/Singularity (`.sif` files locally) | E2B cloud (uses `E2B_API_KEY`) |
| Setup cost | Pull ~100 `.sif` images (100-300 GB local disk) + install Apptainer | Just set `E2B_API_KEY`; templates auto-build on first use |
| Long horizon? | ✅ multi-turn, real repos | ✅ multi-turn shell tasks |
| "Real" SWE-bench? | ✅ yes | ❌ synthetic terminal tasks |

We default to **B** because the user has E2B Pro credits and the PGC paper's
"long-horizon agentic gradient instability" claim doesn't require real
SWE-bench specifically — synthetic terminal tasks exhibit the same dynamics.

## What our recipes combine

Both recipes share the same training stack; only the agent / sandbox differs.

| Layer            | Source                                           | Notes |
|------------------|--------------------------------------------------|-------|
| Training stack   | `grpo-qwen3.5-9b-1n8g-megatron.yaml`             | Qwen3.5-9B + Megatron + 1 node |
| SWE agent (A)    | `swebench_openhands_training.yaml`               | OpenHands CodexAgent / OpenCodeAgent / CodeActAgent variants |
| Harbor agent (B) | `harbor_agent.yaml`                              | Terminus-2 + harbor's pluggable env (we override `harbor_environment_type=e2b`) |
| Async loop       | `grpo-nemotron-nano-16n8g-async-1off-swe.yaml`   | async_grpo + in-flight weight updates |
| Loss             | `grpo-nemotron-nano-16n8g-async-1off-swe.yaml`   | DAPO clip 0.2/0.28 + token-level + TIS |
| Token capture    | NeMo Gym responses_api / NemoGymLLM              | `collect_rollout_details=True` (the bit rllm was missing) |

Scaled down from super-v3's 16-node Nemotron-Nano-30B-A3B to 1-node Qwen3.5-9B
dense:
- TP/EP both reduced from 8 to 4 (no MoE on Qwen3.5-9B-Base).
- max_total_sequence_length 131k → 50k for smoke tests (Qwen3.5-9B native
  supports 256k; bump back to 131k once stable).
- num_prompts_per_step 2 → 1, num_generations_per_prompt 32 → 8 (8 rollouts
  per step total, fits 1 node concurrency).
- gpu_memory_utilization 0.8 → 0.6.
- concurrency 768 → 16 (single-node Apptainer load).

## Prerequisites

1. **NeMo RL submodules initialized** (we already did this on `/scratch/yuzhou/projects/RL`):
   ```bash
   git submodule update --init --recursive
   ```
2. **uv environment**:
   ```bash
   uv sync --all-groups --extra nemo_gym
   ```
3. **Apptainer/Singularity** locally (NeMo Gym SWE agent runs each rollout in
   its own `.sif` container — same model as super-v3 SWE recipe).
4. **Pre-built SWE-bench `.sif` images** at a known path; pass via
   `env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter`.
5. **Training data JSONL** (SWE-Gym for train, SWE-bench Verified for val):
   ```bash
   python 3rdparty/Gym-workspace/Gym/scripts/ng_prepare_data.py \
     --config 3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml \
     --output-dir $DATA_DIR
   ```
6. **8× H200** (recipe defaults). For ≥80 GB GPUs other than H200, lower
   `policy.generation.vllm_cfg.gpu_memory_utilization`.

## Smoke test

### Recipe B — harbor + E2B (default)

```bash
# 1. one-time: download + unpack one subset of the dataset
hf download nvidia/Nemotron-Terminal-Synthetic-Tasks --repo-type dataset \
  --local-dir 3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks
tar -xzf 3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks/skill_based/mixed/scientific_computing.tar.gz \
  -C 3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks/skill_based/mixed/

# 2. apply the upstream patch documented in harbor_agent README §"Required patches to Gym"
#    (add "chat_template_kwargs" to the tokenize endpoint in
#     3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py)

# 3. set E2B_API_KEY in env
export E2B_API_KEY=...

# 4. run
python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config research/pgc_swe/configs/grpo-qwen3.5-9b-1n8g-harbor-e2b.yaml \
    data.train.data_path=$DATA_DIR/scientific_train.jsonl \
    data.validation.data_path=$DATA_DIR/scientific_val.jsonl
```

Expects `Tasks: 1/1` then training step logs in W&B (`nemo-rl/grpo-qwen3.5-9b-1n8g-harbor-e2b`).

### Recipe A — real SWE-bench via Apptainer

```bash
python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config research/pgc_swe/configs/grpo-qwen3.5-9b-1n8g-swe.yaml \
    data.train.data_path=$DATA_DIR/swe_train.jsonl \
    data.validation.data_path=$DATA_DIR/swe_val.jsonl \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=/path/to/swebench/{instance_id}.sif
```

Same trainer, just swap config — output goes to `nemo-rl/grpo-qwen3.5-9b-1n8g-swe`.

## TODO (PGC integration)

The recipe is the *baseline*. PGC will be added as:
- Either a `loss_fn.policy_loss` plugin registered via NeMo RL's loss registry
  (similar to verl's `@register_policy_loss`), or
- A `pgc_swe/` Python module hooked into `worker_extension.py`.

Decision point: where exactly NeMo RL's policy loss can be swapped in. Look at
`nemo_rl/algorithms/grpo.py` and `nemo_rl/loss_functions/` to find the hook
before adding PGC.
