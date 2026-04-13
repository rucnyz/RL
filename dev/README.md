# Dev: OpenSage Integration

Fork of [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
with [OpenSage](https://github.com/opensage-agent/opensage-adk) environment support.

## What's Changed

Registered `opensage` environment in `nemo_rl/environments/utils.py`, pointing to
`opensage.evaluation.rl_adapters.nemo_rl_env.OpenSageEnvironment`.

This enables training agents on any Harbor-format benchmark (SWE-bench, CompileBench, GAIA, etc.) using NeMo RL's
training pipeline (GRPO/DPO/DAPO) + OpenSage's Docker sandbox execution.

## Architecture

```
NeMo RL                                     OpenSageEnvironment
┌─────────────────────────┐
│ Generation (vLLM/SGLang) │
│   model outputs response │
│   with tool calls        ├───────────►  Parse tool calls
│                          │              Execute in Docker sandbox
│ Multi-turn rollout loop  │              Return observation
│   ← observation          │◄────────────
│   → next generation      │
│   ...                    │
│                          ├───────────►  On termination:
│ Policy update            │              Run tests/test.sh
│ (Megatron/FSDP2)         │◄────────────  Return reward (0 or 1)
└─────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Clone this fork
git clone https://github.com/rucnyz/RL.git && cd RL
git submodule update --init --recursive
```

### 1. Build the Docker Container

NeMo RL requires a Docker container to run training. The container pre-caches all worker
virtual environments (vLLM, automodel, mcore, etc.) during build, so they don't need to be
compiled at runtime. See [dependency-management.md](../docs/design-docs/dependency-management.md)
for details.

```bash
# Build from this fork's source (includes opensage registration)
docker buildx build --build-context nemo-rl=. \
  --target release -f docker/Dockerfile \
  --tag nemo-rl-opensage:latest .
```

### 2. Prepare Prompts (outside container)

This step is plain file I/O and can run on the host:

```bash
# Install harbor CLI (subprocess only, avoids dep conflicts)
pip install "harbor>=0.3.0"

# Prepare prompts from Harbor tasks (auto-downloads swebench-verified from registry)
uv run --extra opensage python dev/prepare_harbor_prompts.py \
  --tasks swebench-verified -o dev/data/harbor_prompts.jsonl
```

### 3. Run Training (inside container)

```bash
# Launch container with GPU access, mounting source code
docker run --gpus all --rm -it \
  --network host --ipc host \
  -v $PWD:$PWD -w $PWD \
  nemo-rl-opensage:latest \
  bash

# Inside the container: install opensage (not pre-cached in base image)
uv pip install --python /opt/nemo_rl_venv/bin/python "opensage @ git+https://github.com/opensage-agent/opensage-adk-dev.git"

# Run training
uv run examples/run_grpo.py \
  --config dev/grpo-qwen3.5-35ba3b-2n8g-opensage-harbor.yaml
```

### Using Local OpenSage (for development)

Mount your local opensage checkout into the container:

```bash
docker run --gpus all --rm -it \
  --network host --ipc host \
  -v $PWD:$PWD -w $PWD \
  -v /data/yuzhou/projects/opensage-adk-dev:/data/yuzhou/projects/opensage-adk-dev \
  nemo-rl-opensage:latest \
  bash

# Inside container: install local opensage in editable mode
uv pip install --python /opt/nemo_rl_venv/bin/python -e /data/yuzhou/projects/opensage-adk-dev
```

## Syncing with Upstream

```bash
git fetch upstream
git merge upstream/main
```
