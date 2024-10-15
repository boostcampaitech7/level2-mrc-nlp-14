"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
from typing import NoReturn

from args import (
    DataTrainingArguments,
    ModelArguments,
    CustomTrainingArguments,
    RetrieverArguments,
)
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

    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            datasets,
            training_args,
            data_args,
            retriever_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
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
        use_no_answer=data_args.use_no_answer,
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
