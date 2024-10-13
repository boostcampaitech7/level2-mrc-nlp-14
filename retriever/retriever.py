from retriever.sparse import BM25Retriever, SparseRetriever, SparseRetrieverArguments
from retriever.dense import DenseRetriever, DenseRetrieverArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from args import DataTrainingArguments, CustomTrainingArguments, RetrieverArguments
from base import BaseRetriever


def create_retriever(retriever_args: RetrieverArguments) -> BaseRetriever:
    if retriever_args.retrieval_type == "sparse":
        args = SparseRetrieverArguments(retriever_args)

        if retriever_args.sparse_embedding_type == "bm25":
            return BM25Retriever(args)
        else:
            return SparseRetriever(args)
    elif retriever_args.retrieval_type == "dense":
        return DenseRetriever(DenseRetrieverArguments(retriever_args))
    else:
        raise ValueError(f"Invalid retriever type: {type}")


def run_sparse_retrieval(
    datasets: DatasetDict,
    training_args: CustomTrainingArguments,
    data_args: DataTrainingArguments,
    retriever_args: RetrieverArguments,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = create_retriever(retriever_args)

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
