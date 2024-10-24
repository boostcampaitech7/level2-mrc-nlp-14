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

    def compute_loss(self, model, inputs):
        if self.use_custom_loss and self.tokenizer is not None:
            if model.training:
                outputs = model(**inputs)
                total_loss = outputs.loss

                # Extract attention weights
                attention_weights = self.get_attention_weights(outputs.attentions)

                batch_size, seq_len = attention_weights.size(0), attention_weights.size(
                    1
                )
                device = attention_weights.device

                # Generate token_type_ids
                input_ids = inputs["input_ids"]
                token_type_ids = self.get_token_type_ids(
                    input_ids, self.tokenizer.sep_token_id
                )

                # Create masks
                question_mask, context_mask = self.create_masks(token_type_ids)

                # Create answer mask
                start_positions = inputs["start_positions"]
                end_positions = inputs["end_positions"]
                answer_mask = self.create_answer_mask(
                    start_positions, end_positions, seq_len, batch_size, device
                )

                # Compute importance weights
                importance_weights = self.compute_importance_weights(
                    attention_weights, answer_mask, question_mask
                )

                # Compute context probabilities
                question_context_probs = self.compute_context_probs(
                    attention_weights, question_mask, context_mask
                )

                answer_context_probs = self.compute_context_probs(
                    attention_weights, answer_mask, context_mask, importance_weights
                )

                # Compute KL Divergence
                kl_loss = F.kl_div(
                    answer_context_probs.log(),
                    question_context_probs,
                    reduction="batchmean",
                )

                # Add attention loss to total loss
                total_loss += self.custom_loss_weight * kl_loss
            else:
                outputs = model(**inputs)
                total_loss = outputs.loss

            return total_loss
        else:
            return super().compute_loss(model, inputs)

    def get_attention_weights(self, attentions):
        """
        Extracts and averages attention weights from the last layer over all heads.

        Args:
            attentions (Tuple[Tensor]): Tuple of attention tensors from each layer.

        Returns:
            Tensor: Averaged attention weights from the last layer.
        """
        # Extract attention weights from the last layer and average over heads
        last_layer_attentions = attentions[
            -1
        ]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = last_layer_attentions.mean(
            dim=1
        )  # Shape: (batch_size, seq_len, seq_len)
        return attention_weights

    def get_token_type_ids(self, input_ids, sep_token_id):
        """
        Generates token type IDs to distinguish between question and context tokens.

        Args:
            input_ids (Tensor): Input token IDs.
            sep_token_id (int): Separator token ID.

        Returns:
            Tensor: Token type IDs with 0 for question and 1 for context.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

        for i in range(batch_size):
            sep_indices = (
                (input_ids[i] == sep_token_id).nonzero(as_tuple=False).squeeze()
            )
            if sep_indices.numel() >= 1:
                first_sep_index = sep_indices[0].item()
                token_type_ids[i, first_sep_index + 1 :] = (
                    1  # Mark context tokens with 1
                )

        return token_type_ids

    def create_masks(self, token_type_ids):
        """
        Creates masks for question and context tokens.

        Args:
            token_type_ids (Tensor): Token type IDs distinguishing question and context.

        Returns:
            Tuple[Tensor, Tensor]: Question and context masks.
        """
        question_mask = token_type_ids == 0  # Shape: (batch_size, seq_len)
        context_mask = token_type_ids == 1  # Shape: (batch_size, seq_len)
        return question_mask, context_mask

    def create_answer_mask(
        self, start_positions, end_positions, seq_len, batch_size, device
    ):
        """
        Creates a mask for the answer tokens.

        Args:
            start_positions (Tensor): Start positions of the answers.
            end_positions (Tensor): End positions of the answers.
            seq_len (int): Sequence length.
            batch_size (int): Batch size.
            device (torch.device): Device to create the tensor on.

        Returns:
            Tensor: Answer mask.
        """
        answer_mask = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            answer_mask[i, start_positions[i] : end_positions[i] + 1] = True
        return answer_mask

    def compute_importance_weights(self, attention_weights, answer_mask, question_mask):
        """
        Computes importance weights based on attention from answer to question tokens.

        Args:
            attention_weights (Tensor): Attention weights.
            answer_mask (Tensor): Mask for answer tokens.
            question_mask (Tensor): Mask for question tokens.

        Returns:
            Tensor: Normalized importance weights.
        """
        # Attention from answer tokens to question tokens
        answer_to_question_attention = attention_weights.masked_fill(
            ~answer_mask.unsqueeze(2), 0.0
        )
        answer_to_question_attention = answer_to_question_attention.masked_fill(
            ~question_mask.unsqueeze(1), 0.0
        )

        # Sum over query dimension
        answer_to_question = answer_to_question_attention.sum(
            dim=1
        )  # Shape: (batch_size, seq_len)
        answer_to_question = answer_to_question * question_mask.float()
        answer_to_question += 1e-12  # Prevent division by zero

        # Normalize to get importance weights
        importance_weights = answer_to_question / answer_to_question.sum(
            dim=1, keepdim=True
        )
        return importance_weights

    def compute_context_probs(
        self, attention_weights, source_mask, target_mask, importance_weights=None
    ):
        """
        Computes normalized attention probabilities from source to target tokens.

        Args:
            attention_weights (Tensor): Attention weights.
            source_mask (Tensor): Mask for source tokens.
            target_mask (Tensor): Mask for target tokens.
            importance_weights (Tensor, optional): Importance weights for weighting attention.

        Returns:
            Tensor: Normalized attention probabilities.
        """
        # Attention from source tokens to target tokens
        attention = attention_weights.masked_fill(~source_mask.unsqueeze(2), 0.0)
        attention = attention.masked_fill(~target_mask.unsqueeze(1), 0.0)

        # Apply importance weights if provided
        if importance_weights is not None:
            attention = attention * importance_weights.unsqueeze(2)

        # Sum over source tokens
        attention = attention.sum(dim=1)  # Shape: (batch_size, seq_len)
        attention = attention * target_mask.float()
        attention += 1e-12  # Prevent division by zero

        # Normalize to get probabilities
        context_probs = attention / attention.sum(dim=1, keepdim=True)
        return context_probs
