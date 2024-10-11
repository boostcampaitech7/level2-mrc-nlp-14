class DenseRetrieverArguments:
    """
    DenseRetriever 클래스를 생성할 때 전달할 인자값들
    """

    data_path = "./data/"
    context_path = "wikipedia_documents.json"

    # 임베딩 종류 설정(dense에서는 모델 종류 = 임베딩 종류)
    embedding_type = "klue/bert-base"

    use_siamese = True

    # p_model_path =
    # q_model_path =
    # embedding_type =
