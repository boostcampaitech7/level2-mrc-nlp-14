"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
from typing import NoReturn

from args import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from model import QuestionAnsweringModelLoader
from data_loader import TextDataLoader
from datasets import (
    DatasetDict,
    load_from_disk,
)
from retriever import run_sparse_retrieval
from trainer import QuestionAnsweringTrainer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from utils import check_no_error

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 args/ 내부의 클래스에서 확인 가능합니다.
    # 또는 --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    model_loader = QuestionAnsweringModelLoader(model_args)
    model, tokenizer = model_loader.get_model_tokenizer()

    # True일 경우 : run passage retrieval
    # 지금 run_sparse_retrieval에
    # training_args랑 data_args가 들어가서
    # run_sparse_retrieval 내부의 여러 함수에서 사용되는 중인데
    # 이걸 전부 retriever_args로 대체하고

    # model_args를 추가적으로 넣어줘서
    # tokenizer와 model에 대한 정보를 주는 방식이 괜찮아보임
    # sparse는 tokenizer 통일할 수 있고
    # dense는 encoder와 모델/토크나이저 같은지 다른지 비교 가능
    # 같으면 같은대로, 다르면 다른대로 기록해서 실험 결과 정리할 수 있음
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            datasets,
            model_args,
            training_args,
            data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    data_loader = TextDataLoader(tokenizer, datasets, max_seq_length, data_args)

    eval_dataset = data_loader.get_validation_dataset()

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        max_answer_length=data_args.max_answer_length,
        answer_column_name=data_loader.answer_column_name,
        tokenizer=tokenizer,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
