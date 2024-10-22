from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_custom_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use QA attention loss for training"},
    )
    custom_loss_weight: float = field(
        default=0.1,
        metadata={
            "help": "If you are using a custom loss, this parameter assigns a weight to the custom loss component "
            "in the total loss function, controlling its influence on model training."
        },
    )
