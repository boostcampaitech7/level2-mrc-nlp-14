# from dataclasses import dataclass

# DenseRetriever class 생성할 때
# self만 빼고, 나머지 정보는 전부 이 파일에서 받을 수 있게
# 그래서 나머지 정보에 대한 @dataclass를 여기에 만들거다


# @dataclass
# 생각해보니 dataclass 데코레이터 없어도 될것 같음
# 일단 주석 처리해보고 구현해보자


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
