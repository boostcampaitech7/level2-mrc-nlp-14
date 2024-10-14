from args import RetrieverArguments


class DenseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        self.data_path = retriever_args.data_path
        self.context_path = retriever_args.context_path

        self.embedding_type = retriever_args.dense_embedding_type
        self.use_siamese = retriever_args.dense_use_siamese
        self.local_model_path = "./retriever/embedding"
