from .sparse import BM25Retriever, SparseRetriever, SparseRetrieverArguments
from .dense import DenseRetriever, DenseRetrieverArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from args import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from base import BaseRetriever


def create_retriever(
    model_args,
    retrieval_type,
    embedding_type,
    data_path: str,
    context_path: str,
) -> BaseRetriever:
    if retrieval_type == "sparse":
        if embedding_type == "bm25":
            return BM25Retriever(model_args, data_path, context_path)
        else:
            return SparseRetriever(embedding_type, model_args, data_path, context_path)
    elif retrieval_type == "dense":
        return DenseRetriever(
            embedding_type, data_path, context_path, use_siamese=False
        )
    else:
        raise ValueError(f"Invalid retriever type: {type}")


def run_sparse_retrieval(
    datasets: DatasetDict,
    model_args: ModelArguments,
    training_args: CustomTrainingArguments,
    data_args: DataTrainingArguments,
) -> DatasetDict:

    retrieval_type = data_args.retrieval_type
    if retrieval_type == "sparse":
        retriever_args = SparseRetrieverArguments
    elif retrieval_type == "dense":
        retriever_args = DenseRetrieverArguments

    embedding_type = retriever_args.embedding_type
    data_path = retriever_args.data_path
    context_path = retriever_args.context_path

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = create_retriever(
        model_args=model_args,
        retrieval_type=data_args.retrieval_type,
        embedding_type=embedding_type,
        data_path=data_path,
        context_path=context_path,
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
