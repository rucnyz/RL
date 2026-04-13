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


# ============================================================
# Request/Response
# ============================================================


class OpenSageRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class OpenSageVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ============================================================
# Ray remote runner
# ============================================================


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


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
        )
    )


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
) -> dict:
    """Run OpenSage _generate_one() and return result + logprobs."""
    from opensage.evaluation.rl_adapters.benchmark_interface import BenchmarkInterface
    from opensage.evaluation.rl_adapters.nemo_llm import NemoLlm

    # Create NemoLlm pointing to NemoGym's model server
    model_str = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    nemo_llm = NemoLlm(
        model=model_str,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=api_base,
    )

    # Load HarborEvaluation
    benchmark = BenchmarkInterface.load("harbor")
    evaluation = benchmark.evaluation_class(
        dataset_path=tasks_dir,
        agent_dir=_resolve_agent_dir(agent_name),
        agent_id=f"nemo_gym_{uuid4().hex[:8]}",
        max_llm_calls=max_llm_calls,
        test_timeout=test_timeout,
        output_dir=output_dir,
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

    # Run full OpenSage agent loop
    result = await evaluation._generate_one(task)

    # Collect logprobs from NemoLlm
    logprob_turns = [
        {
            "prompt_token_ids": t.prompt_token_ids,
            "generation_token_ids": t.generation_token_ids,
            "generation_log_probs": t.generation_log_probs,
        }
        for t in nemo_llm.get_logprob_turns()
    ]

    # Read session trace from disk (written by _generate_one → _export_session_trace)
    # and convert to NeMo Gym format
    from opensage.evaluation.rl_adapters.nemo_gym_utils import (
        events_to_nemo_gym_output,
        extract_input_from_events,
        extract_usage_from_events,
    )

    session_events = []
    session_trace_path = Path(output_dir) / task_id / "session_trace.json"
    if session_trace_path.exists():
        try:
            import json as _json
            with open(session_trace_path) as f:
                session_data = _json.load(f)
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
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

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

            output_dir = str(Path(self.config.jobs_dir) / task_id)

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
                )
                future = runner_ray_remote.remote(_run_opensage_job_sync, params)
                job_result = await asyncio.to_thread(ray.get, future)

                result = job_result["result"]
                output_items = job_result["output_items"]
                input_messages = job_result["input_messages"]
                usage = job_result["usage"]

                # Extract reward
                test_result = result.get("test_result", {})
                passed = test_result.get("passed", False)
                reward = 1.0 if passed else 0.0

                logger.info(
                    f"Task {task_id}: reward={reward}, passed={passed}, "
                    f"output_items={len(output_items)}"
                )

            except Exception as e:
                logger.exception(f"OpenSage agent error for task {task_id}: {e}")
                result = None
                output_items = []
                input_messages = []
                usage = {"input_tokens": 0, "output_tokens": 0,
                         "input_tokens_details": {"cached_tokens": 0},
                         "output_tokens_details": {"reasoning_tokens": 0},
                         "total_tokens": 0}
                reward = 0.0
                passed = False

            # Build response (same pattern as harbor_agent)
            response = {
                "id": f"resp_{uuid4().hex}",
                "created_at": int(time.time()),
                "object": "response",
                "model": policy_model_name,
                "output": output_items,
                "status": "completed",
                "usage": usage,
            }

            # Update responses_create_params with actual input
            updated_params = body.responses_create_params
            if input_messages:
                updated_params = body.responses_create_params.model_copy(
                    update={"input": input_messages}
                ) if hasattr(body.responses_create_params, 'model_copy') else body.responses_create_params

            # Save result to disk
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "result.json", "w") as f:
                json.dump({"reward": reward, "result": result}, f, indent=2, default=str)

            return OpenSageVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                response=response,
            )


if __name__ == "__main__":
    OpenSageAgentServer.run_webserver()
