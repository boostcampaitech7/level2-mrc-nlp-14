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

from sentence_transformers import CrossEncoder

### 하드코딩된 부분 ###
# line 129 : cross_encoder_model_name


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


def run_1stage_retrieval(
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


def cross_encoder_rerank(cross_encoder_model, queries, passages):

    # Cross-Encoder로 query와 passage 쌍에 대한 relevance score 계산
    input_pairs = [
        (queries[idx], passage)
        for idx, one_passages in enumerate(passages)
        for passage in one_passages
    ]  # query와 passage를 하나의 쌍으로 묶음
    scores = cross_encoder_model.predict(
        input_pairs
    )  # Cross-Encoder로 예측 (relevance scores)

    # Relevance score를 원래 데이터프레임에 추가 (이때, 전체 개별 passage들의 relevance score가 반환됨)
    num_passages_per_query = len(passages[0])  # top k 만큼
    relevance_scores_per_query = [
        scores[i : i + num_passages_per_query]
        for i in range(0, len(scores), num_passages_per_query)
    ]

    return relevance_scores_per_query


def run_2stage_retrieval(
    datasets: DatasetDict,
    training_args: CustomTrainingArguments,
    data_args: DataTrainingArguments,
    retriever_args: RetrieverArguments,
) -> DatasetDict:

    # 첫 번째 단계: 첫 번째 리트리버로 Passage Retrieval
    first_retriever = create_retriever(retriever_args)
    df = first_retriever.retrieve(
        datasets["validation"], topk=data_args.top_k_retrieval
    )
    print("First Retrieval Done")

    # Query와 Passage들을 리스트로 변환
    queries = df["question"].tolist()

    # passage_separator을 기준으로 passage들을 나눔
    passages = (
        df["context"]
        .apply(lambda x: x.split(first_retriever.passage_seperator))
        .tolist()
    )

    # 두 번째 단계: Cross-Encoder로 재랭킹
    cross_encoder_model_name = "models/2nd_embedder/beomi_kcbert-base_cross-encoder"
    cross_encoder_model = CrossEncoder(
        cross_encoder_model_name
    )  # Cross-Encoder 모델 로드
    relevance_scores_per_query = cross_encoder_rerank(
        cross_encoder_model, queries, passages
    )
    print("Second Retrieval Done")

    # Query마다 top-n (예: top-5) passage들만 선택해서 다시 묶음
    selected_passages = [
        first_retriever.passage_seperator.join(
            [
                passage
                for _, passage in sorted(zip(scores, passage_set), reverse=True)[
                    : data_args.top_n_retrieval
                ]
            ]
        )
        for scores, passage_set in zip(relevance_scores_per_query, passages)
    ]

    # df["context"]에 새로 묶인 passage들 저장
    df["context"] = selected_passages

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
