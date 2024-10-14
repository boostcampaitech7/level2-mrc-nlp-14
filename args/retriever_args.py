from dataclasses import dataclass, field


@dataclass
class RetrieverArguments:
    data_path: str = field(
        default="./data/",
        metadata={"help": "Define the path to the data."},
    )

    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "Define the path to the context."},
    )

    retrieval_type: str = field(
        default="sparse",
        metadata={"help": "Define which retrieval method to use. (sparse, dense)"},
    )

    sparse_embedding_type: str = field(
        default="bm25",
        metadata={
            "help": "Define which embedding method to use. (tfidf, count, hash, bm25)"
        },
    )

    sparse_tokenizer_name: str = field(
        default="klue/bert-base",
        metadata={"help": "Define which tokenizer to use."},
    )

    dense_embedding_type: str = field(
        default="klue/bert-base",
        metadata={"help": "Define which embedding method to use."},
    )

    dense_use_siamese: bool = field(
        default=True,
        metadata={"help": "Define whether to use siamese network for dense retrieval."},
    )
