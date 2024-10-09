from .sparse_retriever import SparseRetriever
from .bm25_retriever import BM25Retriever
from typing import Optional, Callable, List
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from args import DataTrainingArguments, CustomTrainingArguments
from base import BaseRetriever


def create_retriever(
    tokenize_fn,
    data_path: Optional[str] = "./data/",
    context_path: Optional[str] = "wikipedia_documents.json",
    retrieval_type="sparse",
    embedding_type="tfidf",
    model_name_or_path="bert-base-multilingual-cased",  # bm25 bin 이름에 모델 명을 포함시켜 식별하기 쉽게 하기 위해 작성
) -> BaseRetriever:
    if retrieval_type == "sparse":
        if embedding_type == "bm25":
            return BM25Retriever(
                tokenize_fn, data_path, context_path, model_name_or_path
            )
        else:
            return SparseRetriever(embedding_type, tokenize_fn, data_path, context_path)
    elif retrieval_type == "dense":
        # TODO: 나중에 DenseRetriever 클래스 만들면, retrieval_type == "dense" 부분 추가하면 된다
        # return DenseRetriever(embedding_type, tokenize_fn, data_path, context_path)
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
    model_name_or_path: str = "bert-base-multilingual-cased",  # bm25 bin 이름에 모델 명을 포함시켜 식별하기 쉽게 하기 위해 작성
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = create_retriever(
        tokenize_fn=tokenize_fn,
        data_path=data_path,
        context_path=context_path,
        retrieval_type=data_args.retrieval_type,
        embedding_type=data_args.embedding_type,
        model_name_or_path=model_name_or_path,  # bm25 bin 이름에 모델 명을 포함시켜 식별하기 쉽게 하기 위해 작성
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
