"""
Harbor Resources Server for NeMo Gym.

Provides tool execution in Docker sandboxes and Harbor test verification
as a NeMo Gym resources server. Compatible with NeMo Gym's multi-turn
agent loop.

Endpoints:
    POST /seed_session — Build Docker image, create container for a task
    POST /run_terminal_command — Execute bash command in container
    POST /view_file — Read file from container
    POST /str_replace_edit — Edit file in container
    POST /verify — Run Harbor tests/test.sh, return reward
"""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import docker
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

logger = logging.getLogger(__name__)

HARBOR_TASKS_CACHE = Path.home() / ".cache" / "harbor" / "tasks"


def _resolve_tasks_path(dataset_path: str) -> Path:
    local = Path(dataset_path)
    if local.is_dir():
        return local
    cached = HARBOR_TASKS_CACHE / dataset_path
    if cached.is_dir():
        return cached
    raise FileNotFoundError(f"'{dataset_path}' not found locally or in harbor cache")


# ============================================================
# Config
# ============================================================


class HarborResourcesServerConfig(BaseResourcesServerConfig):
    tasks_dir: str = "swebench-verified"
    test_timeout: int = 120
    docker_image_prefix: str = "opensage_gym"


# ============================================================
# Request/Response Models
# ============================================================


class HarborSeedSessionRequest(BaseSeedSessionRequest):
    task_id: str
    task_dir: str


class HarborSeedSessionResponse(BaseSeedSessionResponse):
    container_id: str
    task_id: str


class ToolRequest(BaseModel):
    command: Optional[str] = None
    path: Optional[str] = None
    file_path: Optional[str] = None
    old_string: Optional[str] = None
    new_string: Optional[str] = None


class HarborVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None


class HarborVerifyResponse(BaseVerifyResponse):
    test_passed: bool = False
    test_output: str = ""


# ============================================================
# Server
# ============================================================


class HarborResourcesServer(SimpleResourcesServer):
    config: HarborResourcesServerConfig

    def model_post_init(self, context):
        self._tasks_path = _resolve_tasks_path(self.config.tasks_dir)
        self._docker = docker.from_env()
        # Map session_id -> container_id
        self._containers: dict[str, str] = {}
        logger.info(f"HarborResourcesServer initialized: tasks_dir={self._tasks_path}")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Tool execution endpoints
        app.post("/run_terminal_command")(self.run_terminal_command)
        app.post("/view_file")(self.view_file)
        app.post("/str_replace_edit")(self.str_replace_edit)
        app.post("/finish_task")(self.finish_task)

        return app

    # ---- Session Management ----

    async def seed_session(self, body: HarborSeedSessionRequest) -> HarborSeedSessionResponse:
        """Build Docker image and create container for a Harbor task."""
        task_id = body.task_id
        task_dir = Path(body.task_dir)
        dockerfile_dir = task_dir / "environment"
        image_tag = f"{self.config.docker_image_prefix}_{task_id}"

        # Build image if needed
        try:
            self._docker.images.get(image_tag)
            logger.info(f"Image {image_tag} exists, reusing")
        except docker.errors.ImageNotFound:
            logger.info(f"Building image {image_tag}...")
            self._docker.images.build(path=str(dockerfile_dir), tag=image_tag, rm=True)
            logger.info(f"Image {image_tag} built")

        # Create container
        container = self._docker.containers.run(
            image_tag, command="sleep infinity", detach=True, working_dir="/app"
        )
        logger.info(f"Container {container.short_id} created for task {task_id}")

        return HarborSeedSessionResponse(container_id=container.id, task_id=task_id)

    # ---- Tool Execution ----

    def _get_container(self, request: Request) -> Any:
        """Get the Docker container for the current session."""
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id or session_id not in self._containers:
            raise ValueError(f"No container for session {session_id}")
        return self._docker.containers.get(self._containers[session_id])

    def _exec_in_container(self, container, command: str, timeout: int = 60) -> str:
        """Execute command in container, return output."""
        try:
            exit_code, output = container.exec_run(["bash", "-c", command], demux=True)
            stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
            stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""
            result = stdout
            if stderr:
                result += f"\n[stderr]\n{stderr}"
            if exit_code != 0:
                result += f"\n[exit code: {exit_code}]"
            return result[:10000]
        except Exception as e:
            return f"[execution error: {e}]"

    async def run_terminal_command(self, body: ToolRequest, request: Request) -> PlainTextResponse:
        """Execute a bash command in the task's Docker container."""
        container = self._get_container(request)
        result = self._exec_in_container(container, body.command or "")
        return PlainTextResponse(result)

    async def view_file(self, body: ToolRequest, request: Request) -> PlainTextResponse:
        """Read a file from the task's Docker container."""
        container = self._get_container(request)
        path = body.path or body.file_path or ""
        result = self._exec_in_container(container, f"cat {path}")
        return PlainTextResponse(result)

    async def str_replace_edit(self, body: ToolRequest, request: Request) -> PlainTextResponse:
        """Edit a file in the task's Docker container."""
        container = self._get_container(request)
        file_path = body.path or body.file_path or ""
        old = body.old_string or ""
        new = body.new_string or ""
        if old and new:
            # Use python for reliable string replacement
            py_cmd = (
                f"python3 -c \""
                f"p='{file_path}';"
                f"t=open(p).read();"
                f"open(p,'w').write(t.replace('''{old}''','''{new}''',1))\""
            )
            result = self._exec_in_container(container, py_cmd)
            return PlainTextResponse(f"Edited {file_path}\n{result}")
        return PlainTextResponse(f"Invalid edit args for {file_path}")

    async def finish_task(self, body: ToolRequest, request: Request) -> PlainTextResponse:
        """Agent signals task completion."""
        return PlainTextResponse("Task marked as finished.")

    # ---- Verification ----

    async def verify(self, body: HarborVerifyRequest) -> HarborVerifyResponse:
        """Run Harbor tests/test.sh in the container and return reward."""
        metadata = body.verifier_metadata or {}
        task_id = metadata.get("task_id", "unknown")
        task_dir = Path(metadata.get("task_dir", ""))
        tests_dir = task_dir / "tests"

        # Find container for this session
        # In NemoGym flow, the container was created in seed_session
        # and tracked by session cookie
        image_tag = f"{self.config.docker_image_prefix}_{task_id}"
        containers = self._docker.containers.list(filters={"ancestor": image_tag, "status": "running"})

        if not containers:
            return HarborVerifyResponse(
                **body.model_dump(), reward=0.0,
                test_passed=False, test_output="No running container found"
            )

        container = containers[0]

        if not tests_dir.exists():
            return HarborVerifyResponse(
                **body.model_dump(), reward=0.0,
                test_passed=False, test_output="No tests/ directory"
            )

        # Copy tests into container
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for f in tests_dir.rglob("*"):
                if f.is_file():
                    tar.add(str(f), arcname=f"tests/{f.relative_to(tests_dir)}")
        tar_stream.seek(0)
        container.put_archive("/", tar_stream)

        # Run tests
        test_sh = tests_dir / "test.sh"
        cmd = "chmod +x /tests/test.sh && /tests/test.sh" if test_sh.exists() else "cd /tests && python -m pytest -v --tb=short 2>&1"

        test_output = self._exec_in_container(container, cmd, timeout=self.config.test_timeout)
        passed = "[exit code:" not in test_output or "[exit code: 0]" in test_output

        # Cleanup container
        try:
            container.remove(force=True)
        except Exception:
            pass

        logger.info(f"Task {task_id}: {'PASSED' if passed else 'FAILED'}")

        return HarborVerifyResponse(
            **body.model_dump(),
            reward=1.0 if passed else 0.0,
            test_passed=passed,
            test_output=test_output,
        )


if __name__ == "__main__":
    HarborResourcesServer.run_webserver()
