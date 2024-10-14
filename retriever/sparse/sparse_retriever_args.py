from transformers import AutoTokenizer
from args import RetrieverArguments


class SparseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        # Retrieve할 데이터 경로
        self.data_path = "./data/"
        self.context_path = "wikipedia_documents.json"

        # 임베딩 종류 설정 (tfidf, count, hash, bm25)
        self.embedding_type = retriever_args.sparse_embedding_type
        self.tokenizer_name = retriever_args.sparse_tokenizer_name

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=True,  # rust version tokenizer 사용 여부(좀 더 빠름)
        )
