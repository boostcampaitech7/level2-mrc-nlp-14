from args import RetrieverArguments


class DenseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        self.data_path = "./data/"
        self.context_path = "wikipedia_documents.json"
        self.embedding_type = retriever_args.dense_embedding_type
        self.use_siamese = retriever_args.dense_use_siamese
