import pandas as pd
from dataclasses import dataclass


@dataclass
class RetrieverMetrics:

    df: pd.DataFrame
    retrieved_documents_label: str
    original_document_label: str

    def eval(self):
        result = {}
        result["recall@1"] = self.recall_at_k(1)
        result["recall@5"] = self.recall_at_k(5)
        result["recall@10"] = self.recall_at_k(10)
        result["recall@30"] = self.recall_at_k(30)
        result["recall@50"] = self.recall_at_k(50)
        result["MRR"] = self.mean_reciprocal_rank()

        # Print the results in a tabular format
        print(f"{'Metric':<15} {'Value':<10}")
        print("-" * 25)
        for key, value in result.items():
            print(f"{key:<15} {value:<10.4f}")

        return result

    def recall_at_k(self, k: int) -> float:
        total_relevant_docs = 0
        relevant_retrieved = 0
        for _, row in self.df.iterrows():
            retrieved_docs = row[self.retrieved_documents_label][:k]
            relevant_doc = row[self.original_document_label]
            relevant_retrieved += sum(
                [1 for doc in retrieved_docs if doc == relevant_doc]
            )
            total_relevant_docs += 1

        recall = relevant_retrieved / total_relevant_docs
        return recall

    def mean_reciprocal_rank(self) -> float:
        reciprocal_ranks = []
        for _, row in self.df.iterrows():
            retrieved_docs = row[self.retrieved_documents_label]
            relevant_doc = row[self.original_document_label]

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
