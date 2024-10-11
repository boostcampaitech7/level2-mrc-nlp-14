import json
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
)

from base import BaseRetriever

# from .dense_embedder import BertEmbedder


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_type: str,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        use_siamese=False,
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        # BaseRetriever의 생성자를 호출하여 data_path를 초기화
        super().__init__(data_path)

        self.model_name = embedding_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 로컬에서 불러오는 부분 추가해야함

        if use_siamese:
            # Siamese 방식: 동일한 인코더 사용
            self.encoder = BertEmbedder.from_pretrained(self.model_name)
            self.p_encoder = self.encoder
            self.q_encoder = self.encoder
        else:
            # Asymmetric 방식: 다른 인코더 사용
            self.p_encoder = BertEmbedder.from_pretrained(self.model_name)
            self.q_encoder = BertEmbedder.from_pretrained(self.model_name)

        self.p_encoder.cuda()
        self.q_encoder.cuda()

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

    def get_dense_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            Embedding과 Encoder를 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        raise NotImplementedError

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        주어진 하나의 쿼리 또는 여러 개의 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        raise NotImplementedError

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        FAISS 인덱스를 사용하여 주어진 쿼리 또는 여러 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        raise NotImplementedError
