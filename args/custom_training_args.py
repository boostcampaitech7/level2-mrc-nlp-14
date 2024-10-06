from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


# batch size와 epoch 및 다른 hyper parameter를 조정할 수 있습니다
@dataclass
class CustomTrainingArguments(TrainingArguments):
    per_device_train_batch_size: float = field(
        default=8, metadata={"help": "Total number of training epochs to perform."}
    )
    num_train_epochs: float = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
