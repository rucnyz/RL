# OpenSage: RL Training on Harbor Benchmarks

Train agents on [Harbor](https://github.com/harbor-ai/harbor)-format benchmarks
(SWE-bench, CompileBench, GAIA, etc.) using NeMo RL's GRPO pipeline +
[OpenSage](https://github.com/opensage-agent/opensage-adk)'s Docker sandbox execution.

## Architecture

```
NeMo RL (training)                      NemoGym + OpenSage
+-------------------+       +------------------------------------------+
| Policy update     |       | vLLM HTTP server (generation + logprobs) |
| (Megatron/FSDP2)  |       |                                          |
|                   |       | OpenSage Agent Server (/run)              |
| Async GRPO        | <---> |   └── HarborEvaluation._generate_one()   |
| (per-sample       |       |         ├── OpenSage session + sandbox   |
|  parallel)        |       |         ├── ADK agent (harbor_agent)     |
+-------------------+       |         ├── Multi-turn tool execution    |
                            |         └── Harbor test verification     |
                            +------------------------------------------+
```

NemoGym provides the vLLM HTTP server with logprob tracking. OpenSage's
`_generate_one()` runs the full agent pipeline — same code path as
standalone evaluation, with all hooks, plugins, and sandbox features.

## Quick Start

```bash
# 1. Link OpenSage agent server into NemoGym (one-time setup)
ln -s ../../../../research/opensage/opensage_agent_server \
  3rdparty/Gym-workspace/Gym/responses_api_agents/opensage_agent_server

# 2. Prepare data
cd research/opensage
uv run python prepare_harbor_prompts.py \
  --tasks swebench-verified -o data/harbor_prompts.jsonl

# 3. Run training
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python run_grpo_gym.py \
  --config configs/grpo-qwen3.5-35ba3b-1n8g-opensage-harbor-gym.yaml
```

## How It Works

`run_grpo_gym.py` uses NemoGym's microservice architecture:

- **OpenSage Agent Server** (`opensage_agent_server/app.py`): a NemoGym agent
  server whose `/run` endpoint calls `HarborEvaluation._generate_one()`. This
  runs the full OpenSage ADK agent with sandbox, tools, and verification — the
  same pipeline used in standalone OpenSage evaluation.
- **vLLM HTTP server**: exposed by NeMo RL with `expose_http_server: true`,
  provides generation with logprob tracking. The agent server creates a
  `NemoLlm(base_url=vllm_url)` that calls the Responses API directly via aiohttp.
- **StepTrackingLogger**: wraps NeMo RL's Logger to write the current training
  step to `jobs/.step`. The agent server reads this to group rollouts by step
  (`jobs/step_0000/`, `jobs/step_0001/`, ...).

Data format (`prepare_harbor_prompts.py` generates NemoGym-compatible JSONL):
```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "system prompt..."},
      {"role": "user", "content": "instruction from Harbor task"}
    ],
    "tools": [{"name": "run_terminal_command", ...}, ...]
  },
  "verifier_metadata": {"task_id": "...", "task_dir": "..."},
  "agent_ref": {"name": "opensage_agent"}
}
```

## Viewing Trajectories

Use [nemotron-rl-viewer](https://github.com/rucnyz/nemotron-rl-viewer) to
browse training logs interactively:

```bash
git clone https://github.com/rucnyz/nemotron-rl-viewer.git
cd nemotron-rl-viewer && npm install
npx next dev --port 3000
```

Open `http://localhost:3000`. The default logs path points to
`research/opensage/logs`. Sessions tab shows experiments, steps, and tasks
with live auto-refresh for in-progress rollouts.

## Updating Dependencies

```bash
uv lock
```
