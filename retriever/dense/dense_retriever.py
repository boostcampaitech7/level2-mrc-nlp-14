import json
import os
from typing import List, NoReturn, Optional, Tuple, Union
import pickle

import pandas as pd
import numpy as np
from datasets import Dataset

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

import torch.nn.functional as F

from utils import timer

from base import BaseRetriever
from .dense_retriever_args import DenseRetrieverArguments
from .dense_embedder import BertEmbedder


class DenseRetriever(BaseRetriever):
    def __init__(self, args: DenseRetrieverArguments) -> NoReturn:

        # BaseRetriever의 생성자를 호출하여 data_path를 초기화
        super().__init__(args.data_path)

        self.args = args

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
        self.load_encoders()

        # self.train_embedder()

        # 임베딩 생성 함수 호출
        self.get_dense_embedding()

    def load_encoders(self) -> NoReturn:
        """
        Summary:
            p_encoder와 q_encoder를 로드하거나 새로 초기화하는 함수
        """
        if self.args.use_siamese:
            encoder = BertEmbedder.from_pretrained(self.args.p_embedder_name)
            self.p_encoder = encoder
            self.q_encoder = encoder
            tokenizer = AutoTokenizer.from_pretrained(self.args.p_embedder_name)
            self.p_tokenizer = tokenizer
            self.q_tokenizer = tokenizer
        else:
            self.p_encoder = BertEmbedder.from_pretrained(self.args.p_embedder_name)
            self.p_tokenizer = AutoTokenizer.from_pretrained(self.args.p_embedder_name)

            self.q_encoder = BertEmbedder.from_pretrained(self.args.q_embedder_name)
            self.q_tokenizer = AutoTokenizer.from_pretrained(self.args.q_embedder_name)

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
        safe_embedder_name = self.args.p_embedder_name.replace("/", "_")
        pickle_name = f"dense_embedding_{safe_embedder_name}.bin"
        embedding_path = os.path.join(self.args.embedding_path, pickle_name)

        # 기존 임베딩이 있으면 불러오기
        if os.path.isfile(embedding_path):
            with open(embedding_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Passage embedding loaded from pickle.")
        else:
            # 임베딩이 없으면 새로 생성
            print(
                f"Building passage embedding with {self.args.p_embedder_name} encoder..."
            )

            # Passage를 batch로 나눠서 인코딩 (메모리 효율을 위해)
            batch_size = 32  # 배치 크기는 상황에 맞게 조절 가능
            all_embeddings = []

            self.p_encoder.eval()  # 평가 모드로 변경 (dropout off)
            with torch.no_grad():  # Gradient 계산 비활성화
                for i in tqdm(
                    range(0, len(self.contexts), batch_size), desc="Encoding passages"
                ):
                    batch_texts = self.contexts[i : i + batch_size]
                    inputs = self.p_tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512,
                    ).to(self.p_encoder.device)

                    # BertEmbedder는 pooled_output을 반환하니까 이를 사용
                    embeddings = self.p_encoder(**inputs)  # pooled_output이 반환됨
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
        Arguments:
            query_or_dataset (Union[str, Dataset]): str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): 상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우 -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame
        """
        assert (
            self.p_embedding is not None
        ), "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(queries, k=topk)

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        FAISS 인덱스를 사용하여 주어진 쿼리 또는 여러 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        raise NotImplementedError

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str): 하나의 Query를 받습니다.
            k (Optional[int]): 상위 몇 개의 Passage를 반환할지 정합니다.

        Returns:
            doc_scores, doc_indices: 상위 k개의 Passage와 그 유사도 점수 리스트
        """

        # Query를 q_encoder로 임베딩
        self.q_encoder.eval()
        with torch.no_grad():
            inputs = self.q_tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.q_encoder.device)

            # BertEmbedder는 pooled_output을 반환하므로 이를 사용
            query_vec = self.q_encoder(**inputs)  # pooled_output이 반환됨

        # 코사인 유사도 계산 (p_embedding과 query_vec 간의 유사도)
        query_vec = query_vec.cpu().numpy()
        similarities = np.dot(self.p_embedding, query_vec.T).squeeze()  # (N,)

        # 유사도가 높은 순으로 상위 k개의 passage 선택
        doc_indices = np.argsort(similarities)[::-1][:k]
        doc_scores = similarities[doc_indices].tolist()

        return doc_scores, doc_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List[str]): 여러 개의 Query 리스트를 받습니다.
            k (Optional[int]): 상위 몇 개의 Passage를 반환할지 정합니다.

        Returns:
            doc_scores, doc_indices: 각 query에 대해 상위 k개의 passage와 그 유사도 점수 리스트
        """

        self.q_encoder.eval()
        all_query_vecs = []

        # Query를 q_encoder로 임베딩
        with torch.no_grad():
            for query in tqdm(queries, desc="Encoding queries"):
                inputs = self.q_tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.q_encoder.device)

                # BertEmbedder는 pooled_output을 반환하므로 이를 사용
                query_vec = self.q_encoder(**inputs)  # pooled_output이 반환됨
                all_query_vecs.append(query_vec.cpu().numpy())

        # 전체 query 임베딩을 numpy array로 변환
        all_query_vecs = np.concatenate(all_query_vecs, axis=0)

        # 코사인 유사도 계산 (p_embedding과 모든 query 임베딩 간의 유사도)
        similarities = np.dot(
            self.p_embedding, all_query_vecs.T
        ).T  # (num_queries, num_passages)

        # 각 query마다 유사도가 높은 상위 k개의 passage 선택
        doc_scores = []
        doc_indices = []
        for i in range(similarities.shape[0]):
            sorted_result = np.argsort(similarities[i, :])[::-1][:k]
            doc_scores.append(similarities[i, sorted_result].tolist())
            doc_indices.append(sorted_result.tolist())

        return doc_scores, doc_indices
