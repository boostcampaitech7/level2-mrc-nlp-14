from dataclasses import dataclass
from args import RetrieverArguments


@dataclass
class SparseRetrieverArguments:
    """
    DenseRetriever 클래스를 생성할 때 전달할 인자값들
    """

    # Retrieve할 데이터 경로
    data_path = "./data/"
    context_path = "wikipedia_documents.json"

    # 임베딩 종류 설정 (tfidf, count, hash, bm25)
    embedding_type = "bm25"

    tokenizer_name: str = "klue/bert-base"

    def __init__(self, retriever_args: RetrieverArguments):
        self.embedding_type = retriever_args.sparse_embedding_type
        self.tokenizer_name = retriever_args.sparse_tokenizer_name
