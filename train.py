import logging
import os
import sys
from typing import NoReturn
import wandb

from args import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from model import QuestionAnsweringModelLoader
from data_loader import TextDataLoader
from datasets import DatasetDict, load_from_disk
from trainer import QuestionAnsweringTrainer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from utils import check_no_error

wandb.init(project="MRC", entity="word-maestro")

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 args/ 내부의 클래스에서 확인 가능합니다.
    # 또는 --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # 3개의 dataclass를 parser에 받아서
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    # 각각의 변수에 unpacking
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
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

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
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

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        train_dataset = data_loader.get_train_dataset()

    # Validation preprocessing

    if training_args.do_eval:
        eval_dataset = data_loader.get_validation_dataset()
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        max_answer_length=data_args.max_answer_length,
        answer_column_name=data_loader.answer_column_name,
        tokenizer=tokenizer,
        use_no_answer=data_args.use_no_answer,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
