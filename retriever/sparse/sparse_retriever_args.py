from dataclasses import dataclass
from transformers import AutoTokenizer
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

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=True,  # rust version tokenizer 사용 여부(좀 더 빠름)
        )
