from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# batch size와 epoch 및 다른 hyper parameter를 조정할 수 있습니다
@dataclass
class CustomTrainingArguments(TrainingArguments):
    per_device_train_batch_size: float = field(
        default=8, metadata={"help": "Total number of training epochs to perform."}
    )
    num_train_epochs: float = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    save_steps: int = field(
        default=300, metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
