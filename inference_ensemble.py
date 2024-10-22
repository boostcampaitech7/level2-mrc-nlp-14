import logging
import sys
from collections import Counter
import numpy as np

from args import (
    DataTrainingArguments,
    ModelArguments,
    CustomTrainingArguments,
    RetrieverArguments,
)
from model import QuestionAnsweringModelLoader
from data_loader import TextDataLoader
from datasets import load_from_disk, DatasetDict
from trainer import QuestionAnsweringTrainer
from transformers import HfArgumentParser, set_seed
from utils import check_no_error


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
            RetrieverArguments,
        )
    )
    model_args, data_args, training_args, retriever_args = (
        parser.parse_args_into_dataclasses()
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    model_loader = QuestionAnsweringModelLoader(model_args)
    model, tokenizer = model_loader.get_model_tokenizer()

    # 모델 리스트 정의 (하드 보팅 및 소프트 보팅에 사용할 모델들)
    models = [
        model_1,
        model_2,
        model_3,
    ]  # model_1, model_2, model_3은 로드된 모델 객체들로 치환

    # eval or predict
    if training_args.do_eval or training_args.do_predict:
        run_ensemble_mrc(data_args, training_args, datasets, tokenizer, models)


def run_ensemble_mrc(
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    models,
):
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    data_loader = TextDataLoader(tokenizer, datasets, max_seq_length, data_args)
    eval_dataset = data_loader.get_validation_dataset()

    predictions_list = []

    for model in models:
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=datasets["validation"],
            max_answer_length=data_args.max_answer_length,
            answer_column_name=data_loader.answer_column_name,
            tokenizer=tokenizer,
            use_no_answer=data_args.use_no_answer,
        )
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        predictions_list.append(predictions)  # 각 모델의 예측값을 리스트에 추가
        print(predictions_list)

    # 하드 보팅 적용
    hard_voting_predictions = hard_voting(predictions_list)
    save_predictions(hard_voting_predictions, "hard_voting_predictions.json")

    # 소프트 보팅 적용
    soft_voting_predictions = soft_voting(predictions_list)
    save_predictions(soft_voting_predictions, "soft_voting_predictions.json")


def hard_voting(predictions_list):
    """
    하드 보팅 함수: 각 모델의 예측 결과 중 가장 많이 나온 답변을 선택합니다.
    """
    final_predictions = {}
    for i, example in enumerate(predictions_list[0]["test_examples"]):
        # 모델별로 예측된 텍스트를 모아 Counter 사용
        all_predictions = [pred["prediction_text"] for pred in predictions_list]
        most_common_answer = Counter(all_predictions).most_common(1)[0][0]
        final_predictions[example["id"]] = most_common_answer
    return final_predictions


def soft_voting(predictions_list):
    """
    소프트 보팅 함수: 각 모델의 예측 확률을 평균 내어 가장 높은 확률의 답변을 선택합니다.
    """
    final_predictions = {}
    for i, example in enumerate(predictions_list[0]["test_examples"]):
        all_logits = np.array([pred["logits"] for pred in predictions_list])
        averaged_logits = np.mean(all_logits, axis=0)
        best_answer_idx = np.argmax(averaged_logits)
        final_predictions[example["id"]] = predictions_list[0]["test_examples"][
            best_answer_idx
        ]["prediction_text"]
    return final_predictions


def save_predictions(predictions, file_name):
    """
    예측 결과를 JSON 파일로 저장하는 함수.
    """
    import json

    with open(file_name, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to {file_name}")


if __name__ == "__main__":
    main()
