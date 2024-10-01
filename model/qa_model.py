from base import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
)


class QuestionAnsweringModel(BaseModel):
    def __init__(self, model_args):
        super().__init__()
        config = AutoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name is not None
                else model_args.model_name_or_path
            ),
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        """
        모델의 forward 패스. 입력을 받아서 질문에 대한 답변의 시작과 끝을 예측합니다.

        Args:
            kwargs: 모델에 필요한 모든 인자들 (input_ids, attention_mask, token_type_ids, start_positions, end_positions 등).

        Returns:
            모델의 예측값 출력
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
