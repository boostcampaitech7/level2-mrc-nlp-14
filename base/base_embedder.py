from typing import List
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def fit_transform(self, contexts: List[str]):
        pass

    @abstractmethod
    def transform(self, query: str):
        pass
