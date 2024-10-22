import pandas as pd


class RetrieverMetrics:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def eval(self):
        print("recall@5: ", self.recall_at_k(5))
        print("MRR: ", self.mean_reciprocal_rank())

    def recall_at_k(self, k: int) -> float:
        total_relevant_docs = 0
        relevant_retrieved = 0
        for _, row in self.df.iterrows():
            retrieved_docs = row["context_list"][:k]
            relevant_doc = row["original_context"]  # original_context가 단일 문서일 때
            relevant_retrieved += sum(
                [1 for doc in retrieved_docs if doc == relevant_doc]
            )
            total_relevant_docs += 1

        recall = relevant_retrieved / total_relevant_docs
        return recall

    def mean_reciprocal_rank(self) -> float:
        reciprocal_ranks = []
        for _, row in self.df.iterrows():
            retrieved_docs = row["context_list"]
            relevant_doc = row["original_context"]

            # 정답 문서가 검색 결과에서 몇 번째에 있는지 찾기
            if relevant_doc in retrieved_docs:
                rank = (
                    retrieved_docs.index(relevant_doc) + 1
                )  # 인덱스는 0부터 시작하므로 +1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mrr
