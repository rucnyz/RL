"""
OpenSage Agent Server for NeMo Gym.

Follows the same architecture as NemoGym's harbor_agent:
- Receives /run requests from NemoGym rollout collection
- Runs OpenSage's _generate_one() via Ray remote for parallelism
- Extracts reward, trajectory, and logprobs from results
- Returns NeMo Gym-compatible response with training data

Key differences from harbor_agent:
- Uses OpenSage's HarborEvaluation + ADK agent instead of Harbor's Job framework
- Uses NemoLlm (Responses API) for logprob capture instead of Harbor's NemoGymLLM
"""

import asyncio
import json
import logging
import os
import sys
import time
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import ray
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================


class OpenSageAgentConfig(BaseResponsesAPIAgentConfig):
    concurrency: int = 32
    model_server: ModelServerRef
    # OpenSage agent settings
    agent_name: str = "harbor_agent"
    tasks_dir: str = "swebench-verified"
    test_timeout: int = 120
    max_llm_calls: int = 200
    # Output directory for results/trajectories
    jobs_dir: str = "jobs"
    # OpenSage history config overrides (injected into session.config at runtime,
    # so we don't have to fork opensage's harbor_agent/config.toml for RL tuning)
    max_history_summary_length: Optional[int] = None
    max_tool_response_length: Optional[int] = None
    # If trajectory token count exceeds this, return degenerate sample (reward=0)
    # to give the model a learning signal instead of letting NeMo RL drop it silently.
    max_trajectory_tokens: Optional[int] = None


# ============================================================
# Request/Response
# ============================================================


class OpenSageRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class OpenSageVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    metadata: Optional[Dict[str, Any]] = None
    context_length_exceeded_error: int = 0
    memory_limit_exceeded_error: int = 0
    agent_timeout_error: int = 0


# ============================================================
# Ray remote runner
# ============================================================


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
# Cache HarborEvaluation per Ray worker process (avoid recreating per request)
_CACHED_EVALUATION = None
_CACHED_EVAL_KEY = None


def _run_opensage_job_sync(
    agent_name: str,
    tasks_dir: str,
    test_timeout: int,
    max_llm_calls: int,
    task_id: str,
    task_dir: str,
    user_message: str,
    model_name: str,
    api_base: str,
    output_dir: str,
    history_overrides: Dict[str, Any],
) -> dict:
    """Run a single OpenSage task synchronously (for Ray remote)."""
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(
        _run_opensage_job_async(
            agent_name, tasks_dir, test_timeout, max_llm_calls,
            task_id, task_dir, user_message, model_name, api_base, output_dir,
            history_overrides,
        )
    )


def _trajectory_total_tokens(output_items: list) -> int:
    """Total tokens NeMo RL would see for this trajectory.

    NeMo RL's contiguity check requires each item's prompt_token_ids to extend
    the prior. The full trajectory length = last item's prompt_token_ids +
    its generation_token_ids.
    """
    for item in reversed(output_items):
        if not isinstance(item, dict):
            continue
        if "generation_token_ids" not in item:
            continue
        prompt = item.get("prompt_token_ids") or []
        gen = item.get("generation_token_ids") or []
        return len(prompt) + len(gen)
    return 0


def _make_degenerate_output_items() -> list:
    """Single dummy output item; NeMo RL needs >=1 item with token ids."""
    return [{
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "", "annotations": []}],
        "prompt_token_ids": [0],
        "generation_token_ids": [0],
        "generation_log_probs": [0.0],
    }]


def _make_config_patching_evaluation_class(base_cls, history_overrides: Dict[str, Any]):
    """Build a subclass of HarborEvaluation that patches session.config after creation.

    This lets us control compaction budgets etc. from the RL side without forking
    opensage's harbor_agent/config.toml.

    Notably we disable history_summarizer_plugin: NeMo RL's trajectory collector
    requires each turn's prompt_token_ids to be a contiguous extension of the
    previous turn (seen + tool_response). Compaction REPLACES old events with a
    summary, breaking that contiguity, so the trajectory gets dropped at the
    contiguity check. Tool response truncation (tool_response_summarizer_plugin)
    is fine — it shortens individual tool outputs before they're added to history.
    """

    class _RLHarborEvaluation(base_cls):
        def _register_opensage_session(self, task):
            super()._register_opensage_session(task)
            session = getattr(task, "opensage_session", None)
            if session is None or session.config is None:
                return

            # Disable history_summarizer_plugin (breaks trajectory contiguity for RL)
            plugins_cfg = getattr(session.config, "plugins", None)
            if plugins_cfg is not None and getattr(plugins_cfg, "enabled", None):
                plugins_cfg.enabled = [
                    p for p in plugins_cfg.enabled if p != "history_summarizer_plugin"
                ]

            history = getattr(session.config, "history", None)
            if history is None:
                return
            if history_overrides.get("max_tool_response_length") is not None:
                history.max_tool_response_length = history_overrides["max_tool_response_length"]
            if history_overrides.get("max_history_summary_length") is not None:
                comp = getattr(history, "events_compaction", None)
                if comp is not None:
                    comp.max_history_summary_length = history_overrides["max_history_summary_length"]

    return _RLHarborEvaluation


def _get_or_create_evaluation(
    agent_name, tasks_dir, test_timeout, max_llm_calls, output_dir, history_overrides,
):
    """Cache HarborEvaluation per Ray worker process."""
    global _CACHED_EVALUATION, _CACHED_EVAL_KEY
    key = (agent_name, tasks_dir, test_timeout, max_llm_calls, tuple(sorted(history_overrides.items())))
    if _CACHED_EVALUATION is not None and _CACHED_EVAL_KEY == key:
        # Update output_dir for this request (may differ per task)
        _CACHED_EVALUATION.output_dir = output_dir
        return _CACHED_EVALUATION

    from opensage.evaluation.rl_adapters.benchmark_interface import BenchmarkInterface
    benchmark = BenchmarkInterface.load("harbor")
    eval_cls = benchmark.evaluation_class
    if any(v is not None for v in history_overrides.values()):
        eval_cls = _make_config_patching_evaluation_class(eval_cls, history_overrides)
    evaluation = eval_cls(
        dataset_path=tasks_dir,
        agent_dir=_resolve_agent_dir(agent_name),
        max_llm_calls=max_llm_calls,
        test_timeout=test_timeout,
        output_dir=output_dir,
        non_interactive=True,
    )
    _CACHED_EVALUATION = evaluation
    _CACHED_EVAL_KEY = key
    return evaluation


async def _run_opensage_job_async(
    agent_name: str,
    tasks_dir: str,
    test_timeout: int,
    max_llm_calls: int,
    task_id: str,
    task_dir: str,
    user_message: str,
    model_name: str,
    api_base: str,
    output_dir: str,
    history_overrides: Dict[str, Any],
) -> dict:
    """Run OpenSage _generate_one() and return result + logprobs + error flags."""
    from opensage.evaluation.rl_adapters.nemo_llm import NemoLlm

    # Create NemoLlm pointing to NemoGym's model server
    model_str = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    nemo_llm = NemoLlm(
        model=model_str,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=api_base,
    )

    # Get or create cached HarborEvaluation
    evaluation = _get_or_create_evaluation(
        agent_name, tasks_dir, test_timeout, max_llm_calls, output_dir,
        history_overrides,
    )

    # Build task
    sample_dict = {
        "task_id": task_id,
        "description": user_message,
        "task_dir": task_dir,
        "test_timeout_sec": test_timeout,
    }
    task = evaluation._create_task(sample_dict)
    task.model = nemo_llm

    # Run full OpenSage agent loop with error tracking
    error_flags = {
        "context_length_exceeded": False,
        "memory_limit_exceeded": False,
        "agent_timeout": False,
    }
    result = None

    try:
        result = await evaluation._generate_one(task)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.warning(f"Task {task_id} failed: {error_type}: {error_msg}")

        # Detect error types
        if "context length" in error_msg.lower() or "max_tokens" in error_msg.lower():
            error_flags["context_length_exceeded"] = True
        elif "memory" in error_msg.lower() or "OOM" in error_msg:
            error_flags["memory_limit_exceeded"] = True
        elif "timeout" in error_type.lower() or "timeout" in error_msg.lower():
            error_flags["agent_timeout"] = True

        # Try to recover partial result from disk
        result_path = Path(output_dir) / task_id / "result.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    result = json.load(f)
            except Exception:
                pass

        if result is None:
            result = {"test_result": {"passed": False, "output": f"{error_type}: {error_msg}"}}

    # Collect logprobs from NemoLlm
    logprob_turns = [
        {
            "prompt_token_ids": t.prompt_token_ids,
            "generation_token_ids": t.generation_token_ids,
            "generation_log_probs": t.generation_log_probs,
        }
        for t in nemo_llm.get_logprob_turns()
    ]

    # Read session trace from disk and convert to NeMo Gym format
    from opensage.evaluation.rl_adapters.nemo_gym_utils import (
        events_to_nemo_gym_output,
        extract_input_from_events,
        extract_usage_from_events,
    )

    session_events = []
    session_trace_path = Path(output_dir) / task_id / "session_trace.json"
    if session_trace_path.exists():
        try:
            with open(session_trace_path) as f:
                session_data = json.load(f)
            session_events = session_data.get("events", [])
        except Exception as e:
            logger.warning(f"Failed to read session trace: {e}")

    output_items = events_to_nemo_gym_output(session_events, logprob_turns)
    input_messages = extract_input_from_events(session_events)
    usage = extract_usage_from_events(session_events)

    return {
        "result": result,
        "output_items": output_items,
        "input_messages": input_messages,
        "usage": usage,
        "error_flags": error_flags,
    }


def _resolve_agent_dir(agent_name: str) -> str:
    try:
        from opensage.utils.project_info import find_path
        resolved = find_path("examples", "agents", agent_name)
        if resolved.exists():
            return str(resolved.resolve())
    except Exception:
        pass
    return ""


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


# ============================================================
# Agent Server
# ============================================================


class OpenSageAgentServer(SimpleResponsesAPIAgent):
    config: OpenSageAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("OpenSageAgentServer uses /run, not /v1/responses")

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict, server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}"

    def _read_current_step(self) -> int:
        """Read current training step from the step file written by the logger wrapper."""
        step_file = Path(self.config.jobs_dir) / ".step"
        try:
            return int(step_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return 0

    async def run(self, body: OpenSageRunRequest) -> OpenSageVerifyResponse:
        async with self.sem:
            global_config_dict = get_global_config_dict()
            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)

            metadata = body.verifier_metadata or {}
            task_id = metadata.get("task_id", "unknown")
            task_dir = metadata.get("task_dir", "")

            # Extract user message
            user_message = ""
            responses_create_params = body.responses_create_params
            if hasattr(responses_create_params, 'model_dump'):
                params_dict = responses_create_params.model_dump(exclude_unset=True, exclude_none=True)
            else:
                params_dict = dict(responses_create_params) if responses_create_params else {}

            for msg in params_dict.get("input", []):
                if isinstance(msg, dict) and msg.get("role") in ("user",):
                    user_message = msg.get("content", "")
                    break

            step = self._read_current_step()
            output_dir = str(Path(self.config.jobs_dir) / f"step_{step:04d}" / f"{task_id}_{uuid4().hex[:8]}")

            # Write metadata early so the viewer can show system_prompt/user_message for live sessions.
            # OpenSage's HarborEvaluation is constructed with non_interactive=True so it skips the
            # "output_dir already exists, continue?" prompt.
            system_prompt = next(
                (m.get("content", "") for m in params_dict.get("input", [])
                 if isinstance(m, dict) and m.get("role") in ("system", "developer")),
                "",
            )
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "result.json", "w") as f:
                json.dump({
                    "task_id": task_id,
                    "user_message": user_message,
                    "system_prompt": system_prompt,
                    "reward": None,
                    "result": None,
                }, f, indent=2, default=str)

            try:
                # Run via Ray remote (non-blocking)
                params = dict(
                    agent_name=self.config.agent_name,
                    tasks_dir=self.config.tasks_dir,
                    test_timeout=self.config.test_timeout,
                    max_llm_calls=self.config.max_llm_calls,
                    task_id=task_id,
                    task_dir=task_dir,
                    user_message=user_message,
                    model_name=policy_model_name,
                    api_base=base_url,
                    output_dir=output_dir,
                    history_overrides={
                        "max_history_summary_length": self.config.max_history_summary_length,
                        "max_tool_response_length": self.config.max_tool_response_length,
                    },
                )
                future = runner_ray_remote.remote(_run_opensage_job_sync, params)
                job_result = await asyncio.to_thread(ray.get, future)

                result = job_result["result"]
                output_items = job_result["output_items"]
                input_messages = job_result["input_messages"]
                usage = job_result["usage"]
                error_flags = job_result.get("error_flags", {})

                # Extract reward
                test_result = (result or {}).get("test_result", {})
                passed = test_result.get("passed", False)
                reward = 1.0 if passed else 0.0

                # Trajectory-too-long check: NeMo RL's logprob call concatenates
                # all turns; if total > vLLM's max_seq_len it fails and the
                # sample is silently dropped. Replace with a degenerate sample
                # (reward=0) so the model still gets a learning signal.
                if self.config.max_trajectory_tokens is not None and output_items:
                    total_tokens = _trajectory_total_tokens(output_items)
                    if total_tokens > self.config.max_trajectory_tokens:
                        logger.warning(
                            f"Task {task_id}: trajectory {total_tokens} tokens > "
                            f"max_trajectory_tokens {self.config.max_trajectory_tokens}; "
                            f"replacing with degenerate sample (reward=0)"
                        )
                        output_items = _make_degenerate_output_items()
                        reward = 0.0
                        error_flags["context_length_exceeded"] = True

                logger.info(
                    f"Task {task_id}: reward={reward}, passed={passed}, "
                    f"output_items={len(output_items)}"
                )

            except Exception as e:
                logger.exception(f"OpenSage agent error for task {task_id}: {e}")
                result = None
                # Return a degenerate output item with dummy token data so NeMo RL
                # doesn't crash on empty output (it raises ValueError otherwise).
                # The sample gets reward=0 and minimal gradient contribution.
                output_items = _make_degenerate_output_items()
                input_messages = []
                usage = None
                error_flags = {}
                reward = 0.0

            # Build response (same as harbor_agent's get_default_response_object)
            responses_create_params = body.responses_create_params
            if hasattr(responses_create_params, 'model_dump'):
                params_dict = responses_create_params.model_dump(exclude_unset=True, exclude_none=True)
            else:
                params_dict = dict(responses_create_params) if responses_create_params else {}

            response = {
                "id": f"resp_{uuid4().hex}",
                "created_at": int(time.time()),
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "background": False,
                "max_output_tokens": None,
                "max_tool_calls": None,
                "previous_response_id": None,
                "prompt": None,
                "reasoning": {"effort": None, "generate_summary": None, "summary": None},
                "service_tier": "default",
                "status": "completed",
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "top_logprobs": 0,
                "truncation": "disabled",
                "user": None,
                "prompt_cache_key": None,
                "safety_identifier": None,
                "store": True,
                "model": policy_model_name,
                "temperature": params_dict.get("temperature"),
                "top_p": params_dict.get("top_p"),
                "output": output_items,
                "usage": usage,
            }

            # Update responses_create_params with actual input
            updated_params = body.responses_create_params
            if input_messages:
                updated_params = body.responses_create_params.model_copy(
                    update={"input": input_messages}
                ) if hasattr(body.responses_create_params, 'model_copy') else body.responses_create_params

            # Save result to disk with full context for viewer
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "result.json", "w") as f:
                json.dump({
                    "reward": reward,
                    "task_id": task_id,
                    "user_message": user_message,
                    "system_prompt": next(
                        (m.get("content", "") for m in params_dict.get("input", [])
                         if m.get("role") in ("system", "developer")),
                        "",
                    ),
                    "result": result,
                }, f, indent=2, default=str)

            return OpenSageVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                response=response,
                metadata=result if result else {},
                context_length_exceeded_error=int(error_flags.get("context_length_exceeded", False)),
                memory_limit_exceeded_error=int(error_flags.get("memory_limit_exceeded", False)),
                agent_timeout_error=int(error_flags.get("agent_timeout", False)),
            )


if __name__ == "__main__":
    OpenSageAgentServer.run_webserver()
