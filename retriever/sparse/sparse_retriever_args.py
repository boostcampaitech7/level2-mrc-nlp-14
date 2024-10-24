from transformers import AutoTokenizer
from args import RetrieverArguments
from konlpy.tag import Mecab


class SparseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        self.data_path = retriever_args.data_path
        self.context_path = retriever_args.context_path

        self.embedding_type = retriever_args.sparse_embedding_type
        self.tokenizer_name = retriever_args.sparse_tokenizer_name
        self.use_morph_analyzer = retriever_args.sparse_use_morph_analyzer
        self.mecab = Mecab()

    def tokenize_with_mecab(self, text):
        return self.mecab.morphs(text)

    def get_tokenize_fn(self):
        if self.use_morph_analyzer:

            return self.tokenize_with_mecab

        return AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=True,  # rust version tokenizer 사용 여부(좀 더 빠름)
        ).tokenize
