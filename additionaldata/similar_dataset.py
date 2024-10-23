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


class UseSimilarData:
    def __init__(self, data_args: DataTrainingArguments):
        self.data_args = data_args
        self.features = Features(
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

    def get_processed_data(self) -> Dataset:
        original_dataset = load_from_disk(self.data_args.dataset_name)
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
                .cast(self.features)
            )

        if self.data_args.use_sim_data and self.data_args.sim_dataset_names:
            print("추가적인 데이터를 사용합니다.")
            similar_datasets = []
            for sim_name in self.data_args.sim_dataset_names:
                sim_dataset = load_dataset(sim_name, split="train")
                sim_dataset = sim_dataset.map(
                    add_question_mark,
                ).cast(self.features)
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
