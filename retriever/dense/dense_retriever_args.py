from args import RetrieverArguments


class DenseRetrieverArguments:
    def __init__(self, retriever_args: RetrieverArguments):
        self.data_path = retriever_args.data_path
        self.context_path = retriever_args.context_path

        self.p_embedder_name = retriever_args.dense_p_embedder
        self.q_embedder_name = retriever_args.dense_q_embedder
        self.use_siamese = retriever_args.dense_use_siamese
        self.model_path = "./models/embedder"
        self.embedding_path = "./data"
