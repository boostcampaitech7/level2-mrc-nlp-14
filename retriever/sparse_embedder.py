from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)
from base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    def __init__(self, tokenize_fn, max_features=50000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=max_features
        )
        self.p_embedding = None

    def fit_transform(self, raw_documents):
        # TfidfVectorizer는 iterable을 받으므로 타입을 명시할 필요 없음
        self.p_embedding = self.vectorizer.fit_transform(raw_documents)
        return self.p_embedding

    def transform(self, query):
        # query가 str이나 다른 iterable일 수 있음, 유연하게 처리
        if isinstance(query, str):
            return self.vectorizer.transform([query])
        return self.vectorizer.transform(query)


class CountEmbedder(BaseEmbedder):
    def __init__(self, tokenize_fn, max_features=50000):
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=max_features
        )
        self.p_embedding = None

    def fit_transform(self, raw_documents):
        self.p_embedding = self.vectorizer.fit_transform(raw_documents)
        return self.p_embedding

    def transform(self, query):
        if isinstance(query, str):
            return self.vectorizer.transform([query])
        return self.vectorizer.transform(query)


class HashEmbedder(BaseEmbedder):
    def __init__(self, tokenize_fn, n_features=50000):
        self.vectorizer = HashingVectorizer(
            tokenizer=tokenize_fn, n_features=n_features
        )

    def fit_transform(self, raw_documents):
        self.p_embedding = self.vectorizer.fit_transform(raw_documents)
        return self.p_embedding

    def transform(self, query):
        if isinstance(query, str):
            return self.vectorizer.transform([query])
        return self.vectorizer.transform(query)
