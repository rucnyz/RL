# OpenSage: RL Training on Harbor Benchmarks

Train agents on [Harbor](https://github.com/harbor-ai/harbor)-format benchmarks
(SWE-bench, CompileBench, GAIA, etc.) using NeMo RL's GRPO pipeline +
[OpenSage](https://github.com/opensage-agent/opensage-adk)'s Docker sandbox execution.

## Architecture

```
NeMo RL                                     OpenSageEnvironment
+-------------------------+
| Generation (vLLM/SGLang) |
|   model outputs response |
|   with tool calls        +------------>  Parse tool calls
|                          |              Execute in Docker sandbox
| Multi-turn rollout loop  |              Return observation
|   <- observation         |<------------
|   -> next generation     |
|   ...                    |
|                          +------------>  On termination:
| Policy update            |              Run tests/test.sh
| (Megatron/FSDP2)         |<------------  Return reward (0 or 1)
+-------------------------+
```

## Quick Start

All commands are run from `research/opensage/`.

```bash
cd research/opensage

# 1. Prepare prompts from Harbor tasks
uv run python prepare_harbor_prompts.py \
  --tasks swebench-verified -o data/harbor_prompts.jsonl

# 2. Run GRPO training (1 node, 8 GPUs, Megatron backend)
uv run python run_grpo.py \
  --config configs/grpo-qwen3.5-35ba3b-1n8g-opensage-harbor.yaml
```

## How It Works

The driver script `run_grpo.py` registers opensage-specific components into
NeMo RL's registries at import time -- no core library changes needed:

- `OpenSageDataset` into `DATASET_REGISTRY`
- `opensage_data_processor` into `PROCESSOR_REGISTRY`
- `OpenSageEnvironment` into `ENV_REGISTRY` and `ACTOR_ENVIRONMENT_REGISTRY`

Then it delegates to the standard GRPO training loop.

## Running with Docker

NeMo RL's worker venvs are pre-cached inside the official Docker image, avoiding
build-time GPU issues (e.g. `nv-grouped-gemm` for automodel). If you use a
Megatron-based config (like the one in this project), bare-metal `uv run` works
fine. Docker is still useful for reproducibility or cluster deployments.

```bash
# 1. Build the container from this fork (includes research/opensage in workspace)
docker buildx build --build-context nemo-rl=. \
  --target release -f docker/Dockerfile \
  --tag nemo-rl-opensage:latest .

# 2. Launch with GPU access
docker run --gpus all --rm -it \
  --network host --ipc host \
  -v $PWD:$PWD -w $PWD/research/opensage \
  nemo-rl-opensage:latest \
  bash

# 3. Inside the container: install opensage (not in base image)
uv pip install --python /opt/nemo_rl_venv/bin/python \
  "opensage @ git+https://github.com/opensage-agent/opensage-adk-dev.git"

# 4. Prepare data & run
python prepare_harbor_prompts.py --tasks swebench-verified -o data/harbor_prompts.jsonl
python run_grpo.py --config configs/grpo-qwen3.5-35ba3b-1n8g-opensage-harbor.yaml
```

To use a local opensage checkout instead, add a mount:

```bash
docker run --gpus all --rm -it \
  --network host --ipc host \
  -v $PWD:$PWD -w $PWD/research/opensage \
  -v /path/to/opensage-adk-dev:/path/to/opensage-adk-dev \
  nemo-rl-opensage:latest bash

# then inside:
uv pip install --python /opt/nemo_rl_venv/bin/python -e /path/to/opensage-adk-dev
```

## Using Local OpenSage (for development)

The `pyproject.toml` points to a local editable checkout by default.
To switch to the GitHub version, edit `[tool.uv.sources]`:

```toml
opensage = { git = "https://github.com/opensage-agent/opensage-adk-dev.git" }
```

## Viewing Trajectories

NeMo RL logs training and validation trajectories as JSONL files in the `logs/`
directory during training (`train_data_step{N}.jsonl`, `val_data_step{N}.jsonl`).

Use [nemotron-rl-viewer](https://github.com/rucnyz/nemotron-rl-viewer) to
browse them interactively:

```bash
# Clone and install (requires Node.js >= 20)
git clone https://github.com/rucnyz/nemotron-rl-viewer.git /data/yuzhou/projects/nemotron-rl-viewer
cd /data/yuzhou/projects/nemotron-rl-viewer
npm install

# Start the viewer
npx next dev --port 3000
```

Then open `http://localhost:3000` and enter the logs directory path
(e.g. `/data/yuzhou/projects/RL/research/opensage/logs`).

Features:
- File browser for all JSONL log files
- Conversation view with role-colored messages (user/assistant/system/tool)
- Reward statistics (accuracy, mean, range)
- Sort by index, reward, or advantage
- Filter by reward (all / positive / zero)
- Pagination for large files

To also see samples printed in the terminal during validation, set:

```yaml
logger:
  num_val_samples_to_print: 5
```

## Updating Dependencies

```bash
uv lock
```
