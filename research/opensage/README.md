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

## Using Local OpenSage (for development)

The `pyproject.toml` points to a local editable checkout by default.
To switch to the GitHub version, edit `[tool.uv.sources]`:

```toml
opensage = { git = "https://github.com/opensage-agent/opensage-adk-dev.git" }
```

## Updating Dependencies

```bash
uv lock
```
