# OpenSage: RL Training on Harbor Benchmarks

Train agents on [Harbor](https://github.com/harbor-ai/harbor)-format benchmarks
(SWE-bench, CompileBench, GAIA, etc.) using NeMo RL's GRPO pipeline +
[OpenSage](https://github.com/opensage-agent/opensage-adk)'s Docker sandbox execution.

## Architecture

Two modes available:

**NemoGym mode (recommended)** — NemoGym handles the full multi-turn agent loop:

```
NeMo RL (training)                    NemoGym
+-------------------+       +-------------------------------+
| Policy update     |       | vLLM HTTP server (generation) |
| (Megatron/FSDP2)  |       |   + chat template (tools)     |
|                   |       |   + logprob tracking           |
| Async GRPO        |       |                               |
| (per-sample       | <---> | Agent (simple_agent)          |
|  parallel)        |       |   + multi-turn loop            |
+-------------------+       |                               |
                            | OpenSage Resources Server     |
                            |   + Docker sandbox exec       |
                            |   + Harbor test verification  |
                            +-------------------------------+
```

**Legacy mode** — NeMo RL controls the generation loop directly:

```
NeMo RL (vLLM generate) → OpenSageEnvironment (manual tool parse + Docker exec)
```

## Quick Start (NemoGym mode)

```bash
# 1. Link OpenSage agent server into NemoGym (one-time setup)
ln -s ../../../../research/opensage/opensage_agent_server \
  3rdparty/Gym-workspace/Gym/responses_api_agents/opensage_agent_server

# 2. Prepare data (from repo root or research/opensage/)
cd research/opensage
uv run python prepare_harbor_prompts.py \
  --tasks swebench-verified -o data/harbor_prompts.jsonl

# 3. Run training
uv run python run_grpo_gym.py \
  --config configs/grpo-qwen3.5-35ba3b-1n8g-opensage-harbor-gym.yaml
```

### Quick Start (legacy mode)

```bash
cd research/opensage
uv run python prepare_harbor_prompts.py \
  --tasks swebench-verified -o data/harbor_prompts.jsonl

uv run python run_grpo.py \
  --config configs/grpo-qwen3.5-35ba3b-1n8g-opensage-harbor.yaml
```

## How It Works

### NemoGym mode

`run_grpo_gym.py` uses NemoGym's microservice architecture:

- **OpenSage Resources Server** (`opensage_resources_server/app.py`): a NemoGym
  `SimpleResourcesServer` providing Docker sandbox tool execution
  (`/run_terminal_command`, `/view_file`, `/str_replace_edit`) and Harbor test
  verification (`/verify`). Symlinked into NemoGym's `resources_servers/` directory.
- **NemoGym simple_agent**: handles the multi-turn agent loop (LLM call → tool call
  → LLM call → ...) with proper system prompt and tool definitions from the chat
  template.
- **vLLM HTTP server**: exposed by NeMo RL with `expose_http_server: true`,
  provides generation with logprob tracking via OpenAI-compatible API.

Data format (`prepare_harbor_prompts.py` generates NemoGym-compatible JSONL):
```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "system prompt..."},
      {"role": "user", "content": "instruction from Harbor task"}
    ],
    "tools": [{"name": "run_terminal_command", ...}, ...]
  },
  "verifier_metadata": {"task_id": "...", "task_dir": "..."}
}
```

### Legacy mode

`run_grpo.py` registers opensage-specific components into NeMo RL's registries
at import time (dataset, processor, environment). NeMo RL controls generation
and calls `OpenSageEnvironment.step()` per turn, which manually parses tool
calls and executes them in Docker.

## Running with Docker

```bash
# Build container
docker buildx build --build-context nemo-rl=. \
  --target release -f docker/Dockerfile \
  --tag nemo-rl-opensage:latest .

# Launch
docker run --gpus all --rm -it \
  --network host --ipc host \
  -v $PWD:$PWD -w $PWD/research/opensage \
  nemo-rl-opensage:latest bash

# Inside container
uv pip install --python /opt/nemo_rl_venv/bin/python \
  "opensage @ git+https://github.com/opensage-agent/opensage-adk-dev.git"
python run_grpo_gym.py
```

## Viewing Trajectories

Use [nemotron-rl-viewer](https://github.com/rucnyz/nemotron-rl-viewer) to
browse training logs interactively:

```bash
git clone https://github.com/rucnyz/nemotron-rl-viewer.git
cd nemotron-rl-viewer && npm install
npx next dev --port 3000
```

Open `http://localhost:3000` and enter the logs directory path.

Features: file browser, conversation viewer with role coloring, reward stats,
rollout turn-by-turn view with tool call details, log viewer with ANSI colors.

## Updating Dependencies

```bash
uv lock
```
