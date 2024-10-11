# from dataclasses import dataclass

# DenseRetriever class 생성할 때
# self만 빼고, 나머지 정보는 전부 이 파일에서 받을 수 있게
# 그래서 나머지 정보에 대한 @dataclass를 여기에 만들거다


# @dataclass
# 생각해보니 dataclass 데코레이터 없어도 될것 같음
# 일단 주석 처리해보고 구현해보자


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
