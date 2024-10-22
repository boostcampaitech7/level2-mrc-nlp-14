import random

import numpy as np
from datasets import concatenate_datasets, load_from_disk

from retriever import create_retriever
from utils import timer
from transformers import HfArgumentParser
from args import ModelArguments, DataTrainingArguments, RetrieverArguments


seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


if __name__ == "__main__":

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            RetrieverArguments,
        )
    )
    args: tuple[ModelArguments, DataTrainingArguments, RetrieverArguments] = (
        parser.parse_args_into_dataclasses()
    )
    model_args, training_args, retriever_args = args
    # Test sparse
    org_dataset = load_from_disk(training_args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = create_retriever(retriever_args)

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if training_args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)

        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )
