"""
OpenSage Agent Server for NeMo Gym.

Replaces NemoGym's simple_agent with OpenSage's full agent pipeline.
When NemoGym calls /run, this server:
1. Creates a LiteLlm pointing to NemoGym's vLLM HTTP server
2. Calls HarborEvaluation._generate_one() with OpenSage's ADK agent
3. The agent runs the full multi-turn loop with all OpenSage features
   (sandbox, hooks, plugins, tool execution, verification)
4. Returns reward + response data back to NemoGym

This is the same pattern as Miles adapter — inject external LLM into
OpenSage's Evaluation, let OpenSage handle everything else.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Body, Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_aggregate_metrics

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================


class OpenSageAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    # Harbor task settings
    tasks_dir: str = "swebench-verified"
    test_timeout: int = 120
    # OpenSage agent settings
    agent_name: str = "harbor_agent"
    max_llm_calls: int = 200


# ============================================================
# Request/Response
# ============================================================


class OpenSageRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class OpenSageRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    test_passed: bool = False
    test_output: str = ""


# ============================================================
# Agent Server
# ============================================================


class OpenSageAgentServer(SimpleResponsesAPIAgent):
    config: OpenSageAgentConfig

    def model_post_init(self, context):
        self._evaluation = None
        self._initialized = False

    def _ensure_evaluation(self):
        """Lazy-init HarborEvaluation."""
        if self._initialized:
            return

        from opensage.evaluation.rl_adapters.benchmark_interface import BenchmarkInterface

        benchmark = BenchmarkInterface.load("harbor")
        if benchmark.evaluation_class is not None:
            import uuid
            self._evaluation = benchmark.evaluation_class(
                dataset_path=self.config.tasks_dir,
                agent_dir=self._resolve_agent_dir(),
                agent_id=f"nemo_gym_{uuid.uuid4().hex[:8]}",
                max_llm_calls=self.config.max_llm_calls,
                test_timeout=self.config.test_timeout,
            )
            logger.info(f"HarborEvaluation initialized: tasks_dir={self.config.tasks_dir}")
        else:
            logger.warning("Could not load HarborEvaluation, will run without opensage features")

        self._initialized = True

    def _resolve_agent_dir(self) -> str:
        """Find the agent directory."""
        from opensage.utils.project_info import find_path
        resolved = find_path("examples", "agents", self.config.agent_name)
        if resolved.exists():
            return str(resolved.resolve())
        # Fallback
        return ""

    def _create_litellm(self, model_name: str = "") -> Any:
        """Create LiteLlm pointing to NemoGym's vLLM HTTP server."""
        from google.adk.models.lite_llm import LiteLlm

        # Get the model server URL from NemoGym's server client
        model_server_url = self.server_client.get_server_url(self.config.model_server.name)
        # vLLM HTTP server exposes OpenAI-compatible API
        api_base = f"{model_server_url}/v1"

        model_str = model_name or os.getenv("AGENT_MODEL_NAME", "model")
        if not model_str.startswith("openai/"):
            model_str = f"openai/{model_str}"

        model = LiteLlm(
            model=model_str,
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=api_base,
        )
        logger.info(f"Created LiteLlm: model={model_str}, base_url={api_base}")
        return model

    # ---- /v1/responses — not used, but required by base class ----

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Not used — /run handles the full pipeline."""
        raise NotImplementedError(
            "OpenSageAgentServer uses /run, not /v1/responses"
        )

    # ---- /run — the main entry point ----

    async def run(self, request: Request, body: OpenSageRunRequest) -> OpenSageRunResponse:
        """Run OpenSage agent for a single task.

        NemoGym calls this for each sample. We:
        1. Create LiteLlm pointing to NemoGym's vLLM
        2. Build a task from the sample data
        3. Call _generate_one() — OpenSage runs the full agent loop
        4. Return reward
        """
        self._ensure_evaluation()

        metadata = body.verifier_metadata or {}
        task_id = metadata.get("task_id", "unknown")

        if self._evaluation is None:
            logger.error("No evaluation available, returning 0 reward")
            return OpenSageRunResponse(
                **body.model_dump(),
                response=NeMoGymResponse(output=[], output_text=""),
                reward=0.0,
                test_passed=False,
                test_output="HarborEvaluation not available",
            )

        try:
            # 1. Create LiteLlm pointing to NemoGym's vLLM
            litellm_model = self._create_litellm()

            # 2. Build sample dict for Evaluation._create_task()
            # Extract user message from the input
            user_message = ""
            for msg in body.responses_create_params.get("input", []):
                if msg.get("role") in ("user",):
                    user_message = msg.get("content", "")
                    break

            sample_dict = {
                "task_id": task_id,
                "description": user_message,
                "task_dir": metadata.get("task_dir", ""),
                "test_timeout_sec": metadata.get("test_timeout_sec", self.config.test_timeout),
            }

            task = self._evaluation._create_task(sample_dict)

            # 3. Inject LiteLlm as the model
            task.model = litellm_model

            # 4. Run full OpenSage agent loop
            logger.info(f"Running OpenSage agent for task {task_id}")
            result = await self._evaluation._generate_one(task)

            # 5. Extract reward
            test_result = result.get("test_result", {})
            passed = test_result.get("passed", False)
            reward = 1.0 if passed else 0.0

            logger.info(f"Task {task_id}: reward={reward}, passed={passed}")

            return OpenSageRunResponse(
                **body.model_dump(),
                response=NeMoGymResponse(
                    output=[],
                    output_text=json.dumps(result, default=str),
                ),
                reward=reward,
                test_passed=passed,
                test_output=test_result.get("output", ""),
            )

        except Exception as e:
            logger.exception(f"OpenSage agent error for task {task_id}: {e}")
            return OpenSageRunResponse(
                **body.model_dump(),
                response=NeMoGymResponse(output=[], output_text=f"Error: {e}"),
                reward=0.0,
                test_passed=False,
                test_output=str(e),
            )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )


if __name__ == "__main__":
    OpenSageAgentServer.run_webserver()
