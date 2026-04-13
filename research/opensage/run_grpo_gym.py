"""GRPO training with OpenSage Harbor environment via NemoGym.

Uses NemoGym's microservice architecture:
- HarborResourcesServer: Docker sandbox tool execution + Harbor test verification
- NemoGym simple_agent: multi-turn agent loop with tool calling
- vLLM: generation with logprob tracking, exposed as HTTP server

This gets us:
- Proper system prompt + tool definitions via chat template
- Per-sample async multi-turn rollouts
- vLLM logprob tracking through NemoGym's pipeline
- No manual tool parsing or Docker management
"""

import argparse
import os
import pprint
import sys

import wandb.util
wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    _should_use_nemo_gym,
    grpo_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
from nemo_rl.environments.utils import create_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO with OpenSage via NemoGym")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo-qwen3.5-35ba3b-1n8g-opensage-harbor-gym.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    log_dir = config["logger"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    print(f"Using log directory: {log_dir}")

    # Tee stdout/stderr
    class Tee:
        def __init__(self, stream, filepath):
            self.stream = stream
            self.file = open(filepath, "a")
        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()
        def fileno(self):
            return self.stream.fileno()
        def isatty(self):
            return self.stream.isatty()

    sys.stdout = Tee(sys.stdout, os.path.join(log_dir, "stdout.log"))
    sys.stderr = Tee(sys.stderr, os.path.join(log_dir, "stderr.log"))

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # NemoGym config setup
    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config), "NemoGym should be enabled for this config"

    # Setup data (no env_configs — NemoGym handles environments)
    train_dataset, val_dataset = setup_response_data(tokenizer, config["data"], env_configs=None)

    if val_dataset is not None:
        config["grpo"]["max_val_samples"] = len(val_dataset)
        config["grpo"]["val_batch_size"] = config["grpo"]["max_val_samples"]

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # Setup NemoGym environment
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    ray.get(nemo_gym.health_check.remote())
    print("NemoGym environment ready")

    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    # Run training
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        from nemo_rl.algorithms.grpo import async_grpo_train

        print("Running async GRPO training with NemoGym")
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=config["grpo"]["async_grpo"]["max_trajectory_age_steps"],
        )
    else:
        print("Running synchronous GRPO training with NemoGym")
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
