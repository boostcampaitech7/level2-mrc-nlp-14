class SparseRetrieverArguments:
    """
    DenseRetriever 클래스를 생성할 때 전달할 인자값들
    """

    # Retrieve할 데이터 경로
    data_path = "./data/"
    context_path = "wikipedia_documents.json"

    # 임베딩 종류 설정 (tfidf, count, hash, bm25)
    embedding_type = "bm25"

    #
    #
    #
