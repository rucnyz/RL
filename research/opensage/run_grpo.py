# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO training with OpenSage Harbor environment.

This driver registers opensage-specific components (dataset, processor,
environment) into NeMo RL's registries at import time, then delegates
to the standard GRPO training loop.
"""

import argparse
import os
import pprint

from omegaconf import OmegaConf

# --- Runtime registration (before any NeMo RL setup calls) ---
from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
from nemo_rl.data.processors import PROCESSOR_REGISTRY
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import ENV_REGISTRY

from opensage_rl.dataset import OpenSageDataset
from opensage_rl.processor import opensage_data_processor

DATASET_REGISTRY["OpenSageDataset"] = OpenSageDataset
PROCESSOR_REGISTRY["opensage_data_processor"] = opensage_data_processor
ENV_REGISTRY["opensage"] = {
    "actor_class_fqn": "opensage.evaluation.rl_adapters.nemo_rl_env.OpenSageEnvironment",
}
ACTOR_ENVIRONMENT_REGISTRY[
    "opensage.evaluation.rl_adapters.nemo_rl_env.OpenSageEnvironment"
] = PY_EXECUTABLES.SYSTEM

# --- Standard GRPO imports ---
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run GRPO training with OpenSage")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo-qwen3.5-35ba3b-1n8g-opensage-harbor.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    has_refit_draft_weights = bool(config["policy"]["draft"]["enabled"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_response_data(tokenizer, config["data"], config["env"])

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
    ) = setup(config, tokenizer, dataset, val_dataset)

    print("Running synchronous GRPO training")

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
