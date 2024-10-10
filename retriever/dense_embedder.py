import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


# BertEncoder를 직접 수정하지 말고, 상속 받아서 쓰세요
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output
