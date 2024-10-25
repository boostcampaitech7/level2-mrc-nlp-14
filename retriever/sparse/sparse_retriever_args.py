from transformers import AutoTokenizer
from args import RetrieverArguments


class SparseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        self.data_path = retriever_args.data_path
        self.context_path = retriever_args.context_path

        self.embedding_type = retriever_args.sparse_embedding_type
        self.tokenizer_name = retriever_args.sparse_tokenizer_name

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=True,  # rust version tokenizer 사용 여부(좀 더 빠름)
        )
