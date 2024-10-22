from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional
import pandas as pd
from datasets import Dataset
import os
import faiss
import numpy as np


class BaseRetriever(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(self, data_path: Optional[str]):
        self.passage_seperator = " [SEP] "
        self.data_path = data_path
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def split_passage(self, passage: str) -> List[str]:
        """
        Summary:
            Passage를 self.passage_seperator를 기준으로 나눕니다.
        """
        return passage.split(self.passage_seperator)

    def build_faiss(self, num_clusters=64):
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    @abstractmethod
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        주어진 하나의 쿼리 또는 여러 개의 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        pass

    @abstractmethod
    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        FAISS 인덱스를 사용하여 주어진 쿼리 또는 여러 쿼리로부터 관련된 문서를 검색합니다.
        단일 쿼리는 리스트 형태로, 여러 쿼리는 DataFrame 형태로 반환합니다.
        """
        pass
