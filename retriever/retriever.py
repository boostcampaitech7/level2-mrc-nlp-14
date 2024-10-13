from retriever.sparse import BM25Retriever, SparseRetriever, SparseRetrieverArguments
from retriever.dense import DenseRetriever, DenseRetrieverArguments
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
    retrieval_type: str,
    retriever_args: SparseRetrieverArguments | DenseRetrieverArguments,
) -> BaseRetriever:
    if retrieval_type == "sparse":
        if retriever_args.embedding_type == "bm25":
            return BM25Retriever(retriever_args)
        else:
            return SparseRetriever(retriever_args)
    elif retrieval_type == "dense":
        return DenseRetriever(retriever_args)
    else:
        raise ValueError(f"Invalid retriever type: {type}")


def run_sparse_retrieval(
    datasets: DatasetDict,
    training_args: CustomTrainingArguments,
    data_args: DataTrainingArguments,
) -> DatasetDict:

    if data_args.retrieval_type == "sparse":
        retriever_args = SparseRetrieverArguments
    elif data_args.retrieval_type == "dense":
        retriever_args = DenseRetrieverArguments

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = create_retriever(
        retrieval_type=data_args.retrieval_type,
        retriever_args=retriever_args,
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
