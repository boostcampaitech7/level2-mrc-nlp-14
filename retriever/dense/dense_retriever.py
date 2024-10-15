import json
import os
from typing import List, NoReturn, Tuple, Union
import pickle

import pandas as pd
import numpy as np
from datasets import Dataset

import torch, tqdm
from transformers import AutoTokenizer

from base import BaseRetriever
from .dense_retriever_args import DenseRetrieverArguments
from .dense_embedder import BertEmbedder


class DenseRetriever(BaseRetriever):
    def __init__(self, args: DenseRetrieverArguments) -> NoReturn:
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
        super().__init__(args.data_path)

        self.model_name = args.embedding_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Siamese (SDE) 또는 Asymmetric (ADE) 구분
        self.encoder_type = "SDE" if args.use_siamese else "ADE"

        with open(
            os.path.join(args.data_path, args.context_path), "r", encoding="utf-8"
        ) as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # p_encoder, q_encoder 로드 또는 초기화
        self.load_or_initialize_encoders(args)

        # 임베딩 생성 함수 호출
        self.get_dense_embedding()

    def load_or_initialize_encoders(self, args: DenseRetrieverArguments) -> NoReturn:
        """
        Summary:
            p_encoder와 q_encoder를 로드하거나 새로 초기화하는 함수
        """

        encoder_pickle_path = os.path.join(
            args.local_model_path, f"{self.model_name}_encoder_{self.encoder_type}.bin"
        )

        if os.path.isfile(encoder_pickle_path):
            # 기존 저장된 q_encoder와 p_encoder 불러오기
            with open(encoder_pickle_path, "rb") as f:
                self.p_encoder, self.q_encoder = pickle.load(f)
            print("Loaded existing encoder from local storage.")
        else:
            # 로컬에 없으면 새로 로드하고 저장
            if args.use_siamese:
                self.encoder = BertEmbedder.from_pretrained(self.model_name)
                self.p_encoder = self.encoder
                self.q_encoder = self.encoder
            else:
                self.p_encoder = BertEmbedder.from_pretrained(self.model_name)
                self.q_encoder = BertEmbedder.from_pretrained(self.model_name)

            # 모델을 pickle로 저장
            with open(encoder_pickle_path, "wb") as f:
                pickle.dump((self.p_encoder, self.q_encoder), f)
            print("Saved new encoder to local storage.")

        self.p_encoder.cuda()
        self.q_encoder.cuda()

    def get_dense_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            Embedding과 Encoder를 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle 파일 이름 설정
        pickle_name = f"{self.model_name}_dense_embedding_{self.encoder_type}.bin"
        embedding_path = os.path.join(self.local_model_path, pickle_name)

        # 기존 임베딩이 있으면 불러오기
        if os.path.isfile(embedding_path):
            with open(embedding_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Passage embedding loaded from pickle.")
        else:
            # 임베딩이 없으면 새로 생성
            print(f"Building passage embedding with {self.model_name} encoder...")

            # Passage를 batch로 나눠서 인코딩 (메모리 효율을 위해)
            batch_size = 64  # 배치 크기는 상황에 맞게 조절 가능
            all_embeddings = []

            self.p_encoder.eval()  # 평가 모드로 변경 (dropout off)
            with torch.no_grad():  # Gradient 계산 비활성화
                for i in tqdm(
                    range(0, len(self.contexts), batch_size), desc="Encoding passages"
                ):
                    batch_texts = self.contexts[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512,
                    ).to(self.p_encoder.device)

                    embeddings = self.p_encoder(**inputs).last_hidden_state[
                        :, 0, :
                    ]  # [CLS] 토큰 사용
                    all_embeddings.append(
                        embeddings.cpu().numpy()
                    )  # 메모리 절약을 위해 CPU로 옮김

            # 전체 임베딩을 numpy array로 변환
            self.p_embedding = np.concatenate(all_embeddings, axis=0)
            print(f"Passage embedding shape: {self.p_embedding.shape}")

            # 생성한 임베딩을 pickle로 저장
            with open(embedding_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Passage embedding saved to pickle.")

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
