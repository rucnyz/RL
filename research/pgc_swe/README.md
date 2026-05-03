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
- TP=4 + DP=2 (two model replicas across the 4 Megatron GPUs; no MoE on
  Qwen3.5-9B). vLLM gets the other 4 GPUs (non-colocated, TP=4).
- `max_total_sequence_length` 131k → **256k** (Qwen3.5-9B's native window;
  we hit p90≈105k single-trajectory in scientific_computing).
- `apply_rope_fusion: false` — Qwen3.5's `apply_rotary_pos_emb_absolute`
  asserts `not config.apply_rope_fusion`.
- `sequence_packing.enabled: false` — Qwen3.5's GDN attention layers raise
  `NotImplementedError: GDN does not support packed sequence`.
- `num_prompts_per_step` 2 → 8, `num_generations_per_prompt` 32 → 8 (64
  rollouts per step total — typical for 1-node async GRPO).
- `gpu_memory_utilization` 0.8 → 0.85 (non-colocated, vLLM owns its 4 GPUs).
- harbor `max_input_tokens=196608, max_output_tokens=32768` (sum kept below
  `max_total_sequence_length=262144` so harbor's vLLM context probe doesn't
  over-shoot).

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

One-time setup (already done on this machine — listed for reproducibility):

```bash
# 0. python toolchain — uv only ships up to 3.13.11 today; the upstream
#    pyproject pins 3.13.13 which fails to resolve. Patched in research/pgc_swe.
#
# 0a. CUDA libs in LD_LIBRARY_PATH segfault /usr/bin/git inside uv subprocesses
#     (it's only triggered through uv, manual git fetch works fine). Workaround:
#     `unset LD_LIBRARY_PATH` before `uv sync`. The launcher script
#     (`run_harbor_e2b.sh`) does this automatically.

# 1. install NeMo RL + NeMo Gym (CUDA wheels — takes ~10 min)
unset LD_LIBRARY_PATH
uv sync --all-groups --extra nemo_gym

# 2. one-time host packages (TE/deep-ep/mamba-ssm need to compile against
#    rdma-core headers + cmake/ninja + pybind11 in the mcore venv)
conda install -n base -c conda-forge rdma-core cmake ninja pybind11 -y

# 3. populate research/pgc_swe/.env from .env.example
#    (E2B_API_KEY + WANDB_API_KEY)
cp research/pgc_swe/.env.example research/pgc_swe/.env
$EDITOR research/pgc_swe/.env

# 4. download Nemotron-Terminal-Synthetic-Tasks
.venv/bin/hf download nvidia/Nemotron-Terminal-Synthetic-Tasks \
    --repo-type dataset \
    --local-dir 3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks
```

The upstream patch in harbor_agent README §"Required patches to Gym" (add
`chat_template_kwargs` to the tokenize endpoint in
`3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py`) is
**already applied upstream** in our checkout — verified line 397.

See "Repo footprint" at the bottom of this README for the full list of
files modified outside `research/pgc_swe/` and why each one is necessary.

### Adding a harbor dataset

Harbor's 80+ adapters all emit the same on-disk format:

```
<tasks_root>/
  <task_name>/
    task.toml             # [environment] cpus, memory, [docker_image]
    environment/
      Dockerfile          # base image + apt + pip + COPY files/
      files/              # per-task input data
    solution/             # gold solution (verifier ground truth)
    tests/                # verifier scripts
    instruction.md        # task prompt
```

`prepare_harbor_dataset.sh` accepts one or more `<alias>=<tasks_root>` pairs
and runs the same pipeline regardless of whether the tasks come from
Nemotron-Terminal-Synthetic-Tasks, terminal-bench, aider_polyglot,
swe-bench, an in-house adapter, etc.:

1. `strip_docker_image.py` (optional `--strip-docker-image`) — comment out
   `docker_image = "..."` in every `task.toml`. Only needed when the
   docker_image points at an inaccessible registry (Nemotron's adapter
   ships gitlab-master refs); harbor then falls back to building from the
   task-local Dockerfile.
2. `patch_dataset.py` — two **idempotent** dataset bug-fixes (skip with
   `--no-patch`):
   * **Empty `environment/files/`** for tasks whose Dockerfile contains
     `COPY files/ /app/` but ship without the `files/` directory. The E2B
     SDK errors at `Template.from_dockerfile()` with `ValueError: No files
     found in .../environment/files/` *before* registering the alias, so
     harbor cannot spawn a sandbox for these tasks. ~21 such tasks in
     Nemotron's `scientific_computing` subset.
   * **Fractional reward** (default; opt out with `--binary-reward`) —
     rewrites `tests/test.sh` to write `passed/total` (read from the CTRF
     report pytest already produces) instead of binary `0/1`. Sparse
     binary reward zeros out partial-pass trials (typically the majority
     on hard tasks); fractional gives ~3× denser gradient signal. Use
     `--binary-reward` if you specifically want `pass@1`-style eval
     semantics.
3. `prep_harbor_jsonl.py` — emit `<alias>_{train,val}.jsonl` under
   `research/pgc_swe/data/` (90/10 split by default). The recipe yaml's
   `harbor_datasets` map must list each alias you bring in.
4. `prepare_e2b_templates.py` — pre-build **every task's E2B template
   sequentially-ish** (concurrency 4 by default) so the first training
   step's parallel rollouts don't race on `_create_template` for the same
   template name. **This is critical** — without it, 8 simultaneous
   rollouts of the same task all try to build the same E2B alias, E2B
   cancels duplicates with `400: build was cancelled`, and the trainer
   crashes on empty rollouts (the documented `IndexError: list index out
   of range` at `rollouts.py:1185`). Pre-built templates are cached
   server-side forever, so this is a one-time cost per dataset (~20-30 min
   for 1000 tasks; subsequent shared layers cache).

   **Zombie `waiting` builds**: if `AsyncTemplate.build()` is interrupted
   between alias-registration and image-publish (RateLimit, network blip,
   KeyboardInterrupt), the alias is left in `buildStatus = "waiting"`
   server-side. These count against E2B's **20-concurrent-build org cap**
   forever — every retry that 429s registers another zombie before the 429
   fires, so retries actively make the cap WORSE. `alias_exists` and
   `get_tags` report them as healthy (the alias exists, the `default` tag
   is attached, just pointing at a never-finalized `build_id`). The 429
   error message ("contact support if you need more concurrent builds")
   misleadingly suggests the cap is held by other org members; it isn't,
   the zombies are ours.

   Both `prepare_e2b_templates.py` and `repair_broken_templates.py` call
   `clear_waiting_builds()` at startup — it lists everything our org owns
   via `GET /templates`, finds entries with `buildStatus == "waiting"`,
   and `DELETE`s each one (which frees the cap immediately). For ad-hoc
   recovery without re-running prebuild, you can also do it inline:

   ```python
   import httpx, os
   key = {"X-API-Key": os.environ["E2B_API_KEY"]}
   for t in httpx.get("https://api.e2b.dev/templates", headers=key).json():
       if t.get("buildStatus") == "waiting":
           httpx.delete(f"https://api.e2b.dev/templates/{t['templateID']}", headers=key)
   ```

Examples:

```bash
# Nemotron scientific subset (needs --strip-docker-image because the
# adapter ships unreachable gitlab refs)
DATASET_ROOT="3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks"
tar -xzf "${DATASET_ROOT}/skill_based/mixed/scientific_computing.tar.gz" \
    -C "${DATASET_ROOT}/skill_based/mixed/"

bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
    --strip-docker-image \
    scientific="${DATASET_ROOT}/skill_based/mixed/scientific_computing"

# Multiple subsets at once
bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
    --strip-docker-image \
    scientific="${DATASET_ROOT}/skill_based/mixed/scientific_computing" \
    debugging="${DATASET_ROOT}/skill_based/mixed/debugging" \
    file_ops="${DATASET_ROOT}/skill_based/mixed/file_operations"

# An adapter-generated dataset with public docker_image refs (no strip needed)
bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
    polyglot="$HOME/datasets/aider_polyglot"

# Skip prebuild (re-run JSONL generation only)
bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
    --no-prebuild scientific=...
```

#### Pre-built datasets (E2B templates ready to use)

These have already been run through `prepare_harbor_dataset.sh` against
this org's E2B account. Templates are cached server-side, so a fresh
training run picks them up with zero per-task build cost.

| Subset | Tasks | Source | yaml alias suggestion | Status |
|---|---:|---|---|---|
| `scientific_computing` | 1000 | Nemotron `skill_based/mixed/` | `scientific` | ✅ ready |
| `data_processing` | 1000 | Nemotron `skill_based/mixed/` | `data_processing` | ⏳ in progress |
| `data_science` | 1000 | Nemotron `skill_based/mixed/` | `data_science` | ⏳ in progress |
| `debugging` | 1000 | Nemotron `skill_based/mixed/` | `debugging` | ⏳ in progress |
| `file_operations` | 997 | Nemotron `skill_based/mixed/` | `file_ops` | ⏳ in progress |
| `security` | 987 | Nemotron `skill_based/mixed/` | `security` | ⏳ in progress |

To use any of these in a recipe, add an entry to the yaml's
`harbor_datasets` map (keyed by alias) and point
`data.{train,validation}.data_path` at the JSONL files under
`research/pgc_swe/data/`. E.g. to train on debugging instead of
scientific_computing:

```yaml
env:
  nemo_gym:
    harbor_agent:
      responses_api_agents:
        harbor_agent:
          harbor_agent_kwargs:
            harbor_datasets:
              debugging:
                local_dataset_path: "${...}/skill_based/mixed/debugging"
                workdir: "/app"
```
```bash
bash research/pgc_swe/run_harbor_e2b.sh \
    data.train.data_path=research/pgc_swe/data/debugging_train.jsonl \
    data.validation.data_path=research/pgc_swe/data/debugging_val.jsonl
```

Datasets we have on disk but **have not yet prebuilt** (different
categorization or larger scale; prebuild on demand):

| Subset | Tasks | Notes |
|---|---:|---|
| `easy_5000` | 5000 | Difficulty-curated; spans multiple skill types (data_science, model_training, ...) |
| `medium_20000` | 20000 | Same as above, harder difficulty bucket |

#### Example: SWE-bench Pro

Real-world software engineering benchmark, 731 tasks across Python/JS/TS/Go,
public docker images (`jefzda/sweap-images` on Docker Hub) — so no
`--strip-docker-image` needed. One-time setup to generate tasks via the
upstream harbor adapter:

```bash
git clone https://github.com/laude-institute/harbor.git ~/harbor   # if not cloned
cd ~/harbor/adapters/swebenchpro

# Generate all 731 tasks (or use --limit / --language to subset)
uv run swebenchpro --task-dir ~/datasets/swebenchpro
# Use --language python  / --language js,ts  / --language go  to restrict.

cd /scratch/yuzhou/projects/RL
bash research/pgc_swe/scripts/prepare_harbor_dataset.sh \
    swebench_pro="$HOME/datasets/swebenchpro"
```

This emits `data/swebench_pro_{train,val}.jsonl`, prebuilds 731 E2B
templates, and prints the yaml block to add `swebench_pro` to your recipe's
`harbor_datasets` map. The `instance_id` in each rollout looks like
`swebench_pro::instance_protonmail__webclients-e65cc5f...`.

Note: SWE-bench Pro Dockerfiles `FROM` the prebuilt `jefzda/sweap-images:*`
which are large (~5-10 GB each); E2B's first build per instance pulls the
base image, so the prebuild step is bandwidth-bound rather than CPU-bound
and can take 1-2 hours for the full 731-task set. Subset with `--language`
or `--limit` for a faster smoke run.

After the script finishes it prints the yaml block to paste into your
recipe's `harbor_datasets` map, and the `data.train.data_path` /
`data.validation.data_path` overrides for your run command.

The default recipe `grpo-qwen3.5-9b-1n8g-harbor-e2b.yaml` ships configured
for the single `scientific` alias on the Nemotron `scientific_computing`
subset; bring in more aliases by extending the yaml's `harbor_datasets`
block and editing `run_harbor_e2b.sh` (or passing CLI overrides) to point
at the combined train/val jsonl.

Run:

```bash
bash research/pgc_swe/run_harbor_e2b.sh
# Override defaults on the command line, e.g.
#   bash research/pgc_swe/run_harbor_e2b.sh grpo.max_num_steps=20
```

Expects `Tasks: 1/1` then training step logs in W&B
(`nemo-rl/grpo-qwen3.5-9b-1n8g-harbor-e2b`).

### Recipe A — real SWE-bench via Apptainer

```bash
python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config research/pgc_swe/configs/grpo-qwen3.5-9b-1n8g-swe.yaml \
    data.train.data_path=$DATA_DIR/swe_train.jsonl \
    data.validation.data_path=$DATA_DIR/swe_val.jsonl \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=/path/to/swebench/{instance_id}.sif
```

Same trainer, just swap config — output goes to `nemo-rl/grpo-qwen3.5-9b-1n8g-swe`.

## Repo footprint — what changes outside `research/pgc_swe/`?

To keep this project PR-able, we keep all work inside `research/pgc_swe/`
wherever possible. The full list of unavoidable changes elsewhere:

### Forced by uv (cannot be moved)
| File | Change | Why |
|---|---|---|
| `.python-version` | `3.13.13` → `3.13.11` | uv only ships interpreters up through 3.13.11 today. NeMo RL upstream bumped to 3.13.13 in PR #2243 but uv hasn't shipped that build yet. |
| `pyproject.toml` (top of file) | `requires-python = ">=3.13.13"` → `">=3.13.11"` | Same reason. Without this, `uv sync` refuses to resolve the workspace. |
| `research/template_project/.python-version` + `pyproject.toml` | Same 3.13.13 → 3.13.11 bump | uv workspace requires every member to declare a consistent Python pin. `template_project` is an unrelated NeMo RL research starter; we don't touch its code, only its Python version pin. |
| `uv.lock` | Cascades from the above | Auto-regenerated whenever `pyproject.toml` is changed; commit it so the lockfile is reproducible. |

These should be reverted upstream once uv ships 3.13.13.

### Ported from upstream PR #2332 (CUDA 13 migration)
We're running on a B300 (Blackwell Ultra, sm_103) which only has CUDA 13.2
in `/usr/local/cuda` — the cu129 stack pinned in NeMo RL `main` won't load
(`libcudart.so.12` is not present, and ptxas 12.x can't target sm_103).

NeMo RL upstream is mid-migration in **issue #2111 / PR #2332**
("chore: Enable cuda-13 build", currently OPEN draft). We've imported PR
#2332's pyproject + workspace setup.py changes ahead of merge so we can
build on B300 now. The diff is mechanical (cu129 → cu130 indexes, cu12 →
cu13 nvidia wheels, transformer-engine `core_cu12` → `core_cu13`,
flash-attn / vllm switched to cu13 GitHub release wheels). When PR #2332
lands upstream, our diff against `main` for this section drops to zero.

| File | Change | Source |
|---|---|---|
| `pyproject.toml` (deps + sources + override-dependencies) | cu129 → cu130, cu12 → cu13 across torch/nvidia/TE/flash-attn/vllm pins | verbatim from PR #2332 |
| `3rdparty/Megatron-LM-workspace/setup.py` | `core_cu12` → `core_cu13`, drop `_normalize_te_cuda` consistency-check shim | verbatim from PR #2332 |
| `3rdparty/Megatron-Bridge-workspace/setup.py` | same | verbatim from PR #2332 |
| `uv.lock` | Regenerated from above | local |

Skipped from PR #2332 (not relevant to bare-metal runs): `docker/Dockerfile`
base image bump and SGLang build-parallelism tweak.

### Submodules
**No modifications.** We previously had a one-line pin bump in
`3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/requirements.txt`,
but the harbor pin + openai override now live in
`research/pgc_swe/configs/harbor_agent_{requirements,overrides}.txt` and the
launcher passes them to `uv pip install` directly, leaving the Gym checkout
clean. Verify with `git -C 3rdparty/Gym-workspace/Gym status`.

### System-level prerequisites (not files in the repo)
| Artifact | Why |
|---|---|
| `/scratch/yuzhou/cuda_shim/libcudart.so.13` (zero-byte) | cuDNN frontend dlopens both libcudart.so.12 and .13 and throws `RuntimeError: Multiple libcudart libraries found` if both succeed. The system has CUDA 13.2 in ld.so.cache so the .13 dlopen always succeeds. A broken (zero-byte) `libcudart.so.13` in a high-priority `LD_LIBRARY_PATH` dir makes that dlopen fail, so cuDNN sees only the .12 from the torch wheel. The launcher creates this file on first run if missing. |
| `conda install -n base -c conda-forge rdma-core cmake ninja pybind11` | `transformer_engine` + `deep-ep` + `mamba-ssm` + `causal-conv1d` need to compile from CUDA sources into the per-worker mcore venv; they need rdma-core headers (`infiniband/mlx5dv.h` for deep-ep) and the build toolchain. Once installed, the launcher exports `CPATH`/`LIBRARY_PATH` to point at the conda prefix during the venv build. |
| Dataset under `3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/data/nemotron_terminal_synthetic_tasks/` | Downloaded once via `hf download`; modified in place by `scripts/patch_dataset.py` (empty `environment/files/` + fractional reward). The dataset is gitignored inside the Gym submodule, so these mutations do not show up in `git status`. |

## TODO (PGC integration)

The recipe is the *baseline*. PGC will be added as:
- Either a `loss_fn.policy_loss` plugin registered via NeMo RL's loss registry
  (similar to verl's `@register_policy_loss`), or
- A `pgc_swe/` Python module hooked into `worker_extension.py`.

Decision point: where exactly NeMo RL's policy loss can be swapped in. Look at
`nemo_rl/algorithms/grpo.py` and `nemo_rl/loss_functions/` to find the hook
before adding PGC.
