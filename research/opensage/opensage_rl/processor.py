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

from typing import Any

from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataSpec,
)
from nemo_rl.data.llm_message_utils import get_formatted_message_log

TokenizerType = PreTrainedTokenizerBase


def opensage_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process a datum for OpenSage Harbor tasks.

    Expects datum_dict with 'messages', 'extra_env_info' (containing
    'task_id' and 'task_dir'), and 'task_name' from OpenSageDataset.
    """
    messages = datum_dict["messages"]
    if task_data_spec.system_prompt:
        messages = [{"role": "system", "content": task_data_spec.system_prompt}] + messages

    message_log: LLMMessageLogType = get_formatted_message_log(
        messages,
        tokenizer,
        task_data_spec,
    )

    length = sum(len(m.get("token_ids", [])) for m in message_log)
    loss_multiplier = 1.0 if length < (max_seq_length or float("inf")) else 0.0

    extra_env_info: dict[str, Any] = datum_dict["extra_env_info"]

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output
