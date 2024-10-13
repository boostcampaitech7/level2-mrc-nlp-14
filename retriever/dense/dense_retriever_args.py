from dataclasses import dataclass
from args import RetrieverArguments


@dataclass
class DenseRetrieverArguments:
    """
    DenseRetriever 클래스를 생성할 때 전달할 인자값들
    """

    data_path = "./data/"
    context_path = "wikipedia_documents.json"

    # 임베딩 종류 설정(dense에서는 모델 종류 = 임베딩 종류)
    embedding_type = "klue/bert-base"

    use_siamese = True

    def __init__(self, retriever_args: RetrieverArguments):
        self.embedding_type = retriever_args.dense_embedding_type
        self.use_siamese = retriever_args.dense_use_siamese
