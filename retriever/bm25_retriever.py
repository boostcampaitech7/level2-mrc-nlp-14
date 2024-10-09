import json
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from base import BaseRetriever
from utils import timer


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
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
            Passage 파일을 불러오고 BM25Vectorizer를 선언하는 기능을 합니다.
        """
        self.tokenize_fn = tokenize_fn

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.get_bm25()

    def get_bm25(self) -> None:

        self.embedder_path = os.path.join(self.data_path, "sparse_embedder_bm25.bin")

        if os.path.isfile(self.embedder_path):
            with open(self.embedder_path, "rb") as file:
                self.BM25 = pickle.load(file)

            print("BM25 embeddings loaded from pickle files.")
        else:
            print("Building BM25 embedder...")

            self.BM25 = BM25Okapi([self.tokenize_fn(text) for text in self.contexts])
            with open(self.embedder_path, "wb") as file:
                pickle.dump(self.BM25, file)

            print("BM25 embeddings saved to pickle files.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.BM25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        top_k_scores = doc_scores[top_k_indices]
        return top_k_scores.tolist(), top_k_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List, List]:
        doc_scores_list = []
        doc_indices_list = []
        for query in queries:
            tokenized_query = self.tokenize_fn(query)
            doc_scores = self.BM25.get_scores(tokenized_query)
            top_k_indices = np.argsort(doc_scores)[::-1][:k]
            top_k_scores = doc_scores[top_k_indices]
            doc_scores_list.append(top_k_scores.tolist())
            doc_indices_list.append(top_k_indices.tolist())
        return doc_scores_list, doc_indices_list

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset)):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        FAISS 인덱스를 사용하여 주어진 쿼리 또는 여러 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        # TODO : retrieve_faiss 부분 작성해야함..
        pass
