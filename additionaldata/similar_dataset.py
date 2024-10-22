from args import DataTrainingArguments
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    concatenate_datasets,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
import os


class UseSimilarData:
    def __init__(self, data_args: DataTrainingArguments):
        self.data_args = data_args

    def load_dataset_custom(self, dataset_name: str) -> Dataset:
        try:
            if os.path.isdir(dataset_name):
                print(f"로컬 디렉토리에서 데이터셋을 로드합니다: {dataset_name}")
                dataset = load_from_disk(dataset_name)
            else:
                print(f"Hugging Face Hub에서 데이터셋을 로드합니다: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
            return dataset
        except Exception as e:
            print(f"데이터셋 로딩 중 오류 발생: {e}")
            raise

    def get_processed_data(self) -> Dataset:
        original_dataset = self.load_dataset_custom(self.data_args.dataset_name)
        for split in ["train", "validation"]:
            columns_to_remove_my = [
                col
                for col in ["document_id", "__index_level_0__"]
                if col in original_dataset[split].column_names
            ]
            if columns_to_remove_my:
                original_dataset[split] = original_dataset[split].remove_columns(
                    columns_to_remove_my
                )
            original_dataset[split] = (
                original_dataset[split]
                .map(
                    add_question_mark,
                )
                .cast(features)
            )

        if self.data_args.use_sim_data and self.data_args.sim_dataset_names:
            similar_datasets = []
            for sim_name in self.data_args.sim_dataset_names:
                sim_dataset = self.load_dataset_custom(sim_name)
                sim_dataset = sim_dataset.map(
                    add_question_mark,
                ).cast(features)
                similar_datasets.append(sim_dataset)

            if similar_datasets:
                concatenated_similar = concatenate_datasets(similar_datasets)
            else:
                raise ValueError(
                    "`use_sim_data`가 True로 설정되었지만 유사한 데이터셋이 로드되지 않았습니다."
                )

            combined_dataset = concatenate_datasets(
                [original_dataset["train"], concatenated_similar]
            )
            concat_dataset = DatasetDict(
                {
                    "train": combined_dataset,
                    "validation": original_dataset["validation"],
                }
            )
            return concat_dataset
        else:
            print("기존의 데이터만 사용합니다.")
            return original_dataset


def add_question_mark(example):
    if not example["question"].endswith("?"):
        example["question"] += "?"
    return example


features = Features(
    {
        "id": Value("string"),
        "title": Value("string"),
        "context": Value("string"),
        "question": Value("string"),
        "answers": {
            "text": Sequence(Value("string")),
            "answer_start": Sequence(Value("int64")),
        },
    }
)
