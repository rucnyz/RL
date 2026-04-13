# Dev: OpenSage Integration

Fork of [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) with [OpenSage](https://github.com/opensage-agent/opensage-adk) environment support.

## What's Changed

Registered `opensage` environment in `nemo_rl/environments/utils.py`, pointing to `opensage.evaluation.rl_adapters.nemo_rl_env.OpenSageEnvironment`.

This enables training agents on any Harbor-format benchmark (SWE-bench, CompileBench, GAIA, etc.) using NeMo RL's training pipeline (GRPO/DPO/DAPO) + OpenSage's Docker sandbox execution.

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

# Create venv (NeMo RL uses uv)
uv venv

# Install opensage
pip install -e /path/to/opensage-adk-dev
```

### Run training

```bash
# GRPO training with OpenSage environment
uv run python examples/run_grpo.py --config examples/configs/my_config.yaml
```

### Example Config

```yaml
env:
  opensage:
    tasks_dir: swebench          # auto-downloads from harbor registry
    max_turns: 30
    test_timeout: 120

data:
  train:
    default:
      env_name: opensage
      file_path: /path/to/prompts.jsonl

policy:
  model_name: Qwen/Qwen3-8B
  dtensor_cfg:
    enabled: true

generation:
  backend: vllm
  vllm:
    gpu_memory_utilization: 0.7

algorithm:
  grpo:
    n_samples_per_prompt: 4
    kl_coeff: 0.01
```

## Syncing with Upstream

```bash
git fetch upstream
git merge upstream/main
```
