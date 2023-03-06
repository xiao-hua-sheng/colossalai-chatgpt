from typing import Optional

import torch.nn as nn
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Model
from .reward_model import RewardModel


class T5RM(RewardModel):
    """
    GPT Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[T5Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:

        if pretrained is not None:
            model = T5Model.from_pretrained(pretrained)
        elif config is not None:
            model = T5Model(config)
        else:
            model = T5Model(T5Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias)
