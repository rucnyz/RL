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

See the upstream [NeMo RL README](https://github.com/NVIDIA-NeMo/RL#prerequisites) for full setup, or:

```bash
# Clone this fork
git clone https://github.com/rucnyz/RL.git && cd RL
git submodule update --init --recursive
# Create venv
uv venv --seed
```

### Example: Qwen3.5-35B + SWE-bench (Harbor)

A ready-to-use config is at [
`grpo-qwen3.5-35ba3b-2n8g-opensage-harbor.yaml`](grpo-qwen3.5-35ba3b-2n8g-opensage-harbor.yaml), inheriting from the
upstream Qwen3.5-35B recipe with OpenSage + Harbor environment:

```bash
# 1. Sync dependencies (installs opensage + vllm)
uv sync --extra vllm --extra opensage

# 2. Install harbor CLI separately (subprocess only, avoids dep conflicts)
pip install "harbor>=0.3.0"

# 3. Prepare prompts from Harbor tasks (auto-downloads swebench from registry)
uv run --extra opensage python dev/prepare_harbor_prompts.py \
  --tasks swebench -o dev/data/harbor_prompts.jsonl

# 4. Run training
uv run --extra vllm --extra opensage python examples/run_grpo.py \
  --config dev/grpo-qwen3.5-35ba3b-2n8g-opensage-harbor.yaml
```

### Using Local OpenSage (for development)

By default, `--extra opensage` installs from GitHub. To switch to a local checkout:

```bash
uv add --editable "/data/yuzhou/projects/opensage-adk-dev"
```

To switch back to GitHub:

```bash
uv add opensage --git https://github.com/opensage-agent/opensage-adk-dev.git
```

### Template

```bash
# GRPO with vLLM backend + OpenSage environment
uv run --extra vllm --extra opensage python examples/run_grpo.py \
  --config examples/configs/my_config.yaml

# GRPO with Megatron backend + OpenSage environment
uv run --extra mcore --extra opensage python examples/run_grpo.py \
  --config examples/configs/my_config.yaml
```

## Syncing with Upstream

```bash
git fetch upstream
git merge upstream/main
```
