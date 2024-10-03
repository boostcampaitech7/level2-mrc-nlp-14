from .tfidf_retriever import TFIDFRetrieval
from typing import Optional, Callable, List
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from args import DataTrainingArguments, CustomTrainingArguments


def create_retriever(
    tokenize_fn,
    data_path: Optional[str] = "./data/",
    context_path: Optional[str] = "wikipedia_documents.json",
    type="tfidf",
):
    if type == "tfidf":
        return TFIDFRetrieval(tokenize_fn, data_path, context_path)
    elif type == "bm25":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid retriever type: {type}")


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: CustomTrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = create_retriever(
        tokenize_fn=tokenize_fn,
        data_path=data_path,
        context_path=context_path,
        type=data_args.retrieval_type,
    )

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        df = df.drop(columns=["original_context"])
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets
