# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    is_datasets_available,
    is_torch_tpu_available,
)
from datasets import load_metric
from utils import postprocess_qa_predictions

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QuestionAnsweringTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_examples=None,
        max_answer_length=512,
        answer_column_name=None,
        use_no_answer: bool = False,
        use_custom_loss: bool = False,
        custom_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.max_answer_length = max_answer_length
        self.answer_column_name = answer_column_name
        self.use_no_answer = use_no_answer
        self.use_custom_loss = use_custom_loss
        self.custom_loss_weight = custom_loss_weight
        self.metric = load_metric("squad")
        self.setup()

    def setup(self):
        """
        Data Collator를 설정합니다.

        flag가 True이면 데이터가 이미 max_length로 padding되어 있습니다.
        그렇지 않으면 Data Collator가 패딩을 수행합니다.
        Data Collator는 제공된 토크나이저를 사용하며, fp16이 활성화된 경우 8의 배수로 padding합니다.
        """
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=8 if self.args.fp16 else None,
        )
        self.compute_metrics = self.compute_metrics_fn

    def compute_metrics_fn(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_process_function(
        self, examples, features, predictions: Tuple[np.ndarray, np.ndarray]
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=self.max_answer_length,
            output_dir=self.args.output_dir,
            version_2_with_negative=self.use_no_answer,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if self.args.do_predict:
            return formatted_predictions
        elif self.args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]}
                for ex in examples
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions
        )
        return predictions

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.use_custom_loss and self.tokenizer is not None:
            if model.training:
                outputs = model(**inputs)
                total_loss = outputs.loss

                # 어텐션 가중치 추출
                attentions = outputs.attentions[
                    -1
                ]  # (batch_size, num_heads, seq_len, seq_len)
                attention_weights = attentions.mean(
                    dim=1
                )  # (batch_size, seq_len, seq_len)

                batch_size, seq_len, _ = attention_weights.size()
                device = attention_weights.device

                # token_type_ids 생성 (Question과 Context 구분) token_type_ids를 사용하지 않는 roberta 모델이므로 직접 id부여
                input_ids = inputs["input_ids"]
                token_type_ids = torch.zeros_like(
                    input_ids, dtype=torch.long, device=device
                )
                sep_token_id = self.tokenizer.sep_token_id

                for i in range(batch_size):
                    sep_indices = (
                        (input_ids[i] == sep_token_id).nonzero(as_tuple=False).squeeze()
                    )
                    if sep_indices.numel() >= 1:
                        first_sep_index = sep_indices[0].item()
                        token_type_ids[i, first_sep_index + 1 :] = (
                            1  # Context 부분을 1로 설정
                        )

                # 마스크 생성
                question_mask = token_type_ids == 0  # (batch_size, seq_len)
                context_mask = token_type_ids == 1  # (batch_size, seq_len)

                # Answer 토큰의 위치 추출
                start_positions = inputs["start_positions"]  # (batch_size)
                end_positions = inputs["end_positions"]  # (batch_size)

                # Answer 토큰의 마스크 생성
                answer_mask = torch.zeros_like(
                    input_ids, dtype=torch.bool, device=device
                )
                for i in range(batch_size):
                    answer_mask[i, start_positions[i] : end_positions[i] + 1] = True

                # Question과 Answer와 사이의 어텐션 점수를 계산해서 가중치 부여 (간단하긴 하지만 최선이라고는 장담못함)

                # answer의 어텐션 점수분포를 question이 가지는 attention 점수분포와 일치시켜서 가중치로 사용
                answer_to_question_attention = attention_weights.masked_fill(
                    ~answer_mask.unsqueeze(2), 0.0
                )
                answer_to_question_attention = answer_to_question_attention.masked_fill(
                    ~question_mask.unsqueeze(1), 0.0
                )
                # Answer에서 Question으로의 어텐션 분포 (batch_size, seq_len)
                answer_to_question = answer_to_question_attention.sum(
                    dim=1
                )  # (batch_size, seq_len)
                answer_to_question = answer_to_question * question_mask.float()
                answer_to_question = answer_to_question + 1e-12
                # 정규화
                answer_to_question_weight = answer_to_question / answer_to_question.sum(
                    dim=1, keepdim=True
                )

                """
                # question 어텐션 점수분포를 answer이 가지는 attention 점수분포와 일치시켜서 가중치로 사용 
                question_to_answer_attention = attention_weights.masked_fill(
                    ~question_mask.unsqueeze(2), 0.0
                )
                question_to_answer_attention = question_to_answer_attention.masked_fill(
                    ~answer_mask.unsqueeze(1), 0.0
                )
                # Question에서 Answer으로의 어텐션 분포 (batch_size, seq_len)
                question_to_answer = question_to_answer_attention.sum(dim=1)  # (batch_size, seq_len)
                question_to_answer = question_to_answer * question_mask.float()
                question_to_answer = question_to_answer + 1e-12
                # 정규화
                question_to_answer_weight = question_to_answer / question_to_answer.sum(dim=1, keepdim=True)
                """

                importance_weights = (
                    answer_to_question_weight  # question_to_answer_weight
                )

                # 어텐션 가중치에서 Question에서 Context로의 어텐션 분포 계산
                # Query: Question 토큰, Key: Context 토큰
                question_to_context_attention = attention_weights.masked_fill(
                    ~question_mask.unsqueeze(2), 0.0
                )
                question_to_context_attention = (
                    question_to_context_attention.masked_fill(
                        ~context_mask.unsqueeze(1), 0.0
                    )
                )
                # 중요도 가중치 적용 (가중 평균 적용x)
                # question_to_context_attention = question_to_context_attention * importance_weights.unsqueeze(2)
                question_context_attention = question_to_context_attention.sum(
                    dim=1
                )  # (batch_size, seq_len)
                question_context_attention = (
                    question_context_attention * context_mask.float()
                )
                question_context_probs = question_context_attention + 1e-12
                # 정규화
                question_context_probs = (
                    question_context_probs
                    / question_context_probs.sum(dim=1, keepdim=True)
                )

                # 어텐션 가중치에서 Answer에서 Context로의 어텐션 분포 계산
                # Query: Answer 토큰, Key: Context 토큰
                answer_to_context_attention = attention_weights.masked_fill(
                    ~answer_mask.unsqueeze(2), 0.0
                )
                answer_to_context_attention = answer_to_context_attention.masked_fill(
                    ~context_mask.unsqueeze(1), 0.0
                )
                # 중요도 가중치 적용 (가중 평균 적용o)
                answer_to_context_attention = (
                    answer_to_context_attention * importance_weights.unsqueeze(2)
                )
                answer_context_attention = answer_to_context_attention.sum(
                    dim=1
                )  # (batch_size, seq_len)
                answer_context_attention = (
                    answer_context_attention * context_mask.float()
                )
                answer_context_probs = answer_context_attention + 1e-12
                # 정규화
                answer_context_probs = answer_context_probs / answer_context_probs.sum(
                    dim=1, keepdim=True
                )

                # KL Divergence 계산
                kl_loss = F.kl_div(
                    answer_context_probs.log(),  # question_context_probs
                    question_context_probs,  # answer_context_probs
                    reduction="batchmean",
                )

                # 총 손실에 어텐션 손실 추가
                total_loss = total_loss + (
                    self.custom_loss_weight * kl_loss
                )  # 가중치는 필요에 따라 조정
            else:
                outputs = model(**inputs)
                total_loss = outputs.loss

            if return_outputs:
                return total_loss, outputs
            else:
                return total_loss
        else:
            return super().compute_loss(model, inputs, return_outputs)
