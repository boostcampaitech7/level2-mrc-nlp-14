import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_from_disk
import os
from transformers import HfArgumentParser, AutoTokenizer, set_seed
from args import RetrieverArguments, DataTrainingArguments
from retriever.dense.dense_embedder import BertEmbedder
from retriever.dense.dense_retriever_args import DenseRetrieverArguments
import random
from rank_bm25 import BM25Okapi


# BM25을 사용하여 하드 네거티브 준비
def prepare_hard_negatives(questions, contexts, num_negatives=5):
    tokenized_contexts = [context.split() for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    hard_negatives = []

    for question in questions:
        tokenized_question = question.split()
        scores = bm25.get_scores(tokenized_question)
        top_neg_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[1 : num_negatives + 1]
        hard_negatives.append([contexts[idx] for idx in top_neg_indices])

    return hard_negatives


# 학습 함수 정의
def train_embedder(
    data_args: DataTrainingArguments, retriever_args: RetrieverArguments
):
    # dense_args 초기화
    dense_args = DenseRetrieverArguments(retriever_args=retriever_args)

    # MRC 데이터셋 불러오기
    train_data = load_from_disk(data_args.dataset_name)["train"]

    # question과 context 준비
    questions = [item["question"] for item in train_data]
    contexts = [item["context"] for item in train_data]

    hard_negatives = prepare_hard_negatives(questions, contexts)

    if dense_args.use_siamese:
        encoder = BertEmbedder.from_pretrained(dense_args.p_embedder_name)
        p_encoder = encoder
        q_encoder = encoder
    else:
        p_encoder = BertEmbedder.from_pretrained(dense_args.p_embedder_name)
        q_encoder = BertEmbedder.from_pretrained(dense_args.q_embedder_name)

    # 토크나이저 초기화 (p_encoder 이름에 기반하여)
    tokenizer = AutoTokenizer.from_pretrained(dense_args.p_embedder_name)

    # GPU로 옮기기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_encoder = p_encoder.to(device)
    q_encoder = q_encoder.to(device)

    # Optimizer 설정 (AdamW)
    optimizer = torch.optim.AdamW(
        list(p_encoder.parameters()) + list(q_encoder.parameters()),
        lr=3e-5,
    )

    p_encoder.train()
    q_encoder.train()

    batch_size = 16  # 기본 배치 크기
    epochs = 5  # 학습 에포크 수를 args에서 받아옴

    # 학습 루프 시작
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        combined_data = list(zip(questions, contexts, hard_negatives))
        random.shuffle(combined_data)
        questions, contexts, hard_negatives = zip(*combined_data)

        for i in tqdm(range(0, len(questions), batch_size), desc="Training progress"):
            batch_questions = questions[i : i + batch_size]
            batch_contexts = list(contexts[i : i + batch_size])  # 리스트로 변환
            batch_negatives = [
                random.choice(hn) for hn in hard_negatives[i : i + batch_size]
            ]

            # 토크나이저로 입력 변환
            q_inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            p_inputs = tokenizer(
                batch_contexts + batch_negatives,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # 인코더로 임베딩 추출
            q_embedding = q_encoder(**q_inputs)
            p_embedding = p_encoder(**p_inputs)

            # 유사도 계산
            similarity_matrix = torch.matmul(q_embedding, p_embedding.T)
            target = torch.arange(batch_size).to(similarity_matrix.device)

            # Loss 계산 (CrossEntropy 사용)
            loss = F.cross_entropy(similarity_matrix, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} completed, Average Loss: {total_loss / len(questions)}"
        )

    # 모델 저장
    p_embedder_path = dense_args.p_embedder_name.replace("/", "_")
    q_embedder_path = dense_args.q_embedder_name.replace("/", "_")
    if dense_args.use_siamese:
        model_save_path = os.path.join(dense_args.model_path, f"{p_embedder_path}_SDE")
        p_encoder.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
    else:
        p_save_path = os.path.join(
            dense_args.model_path, f"{p_embedder_path}_ADE_passage"
        )
        q_save_path = os.path.join(
            dense_args.model_path, f"{q_embedder_path}_ADE_query"
        )
        p_encoder.save_pretrained(p_save_path)
        q_encoder.save_pretrained(q_save_path)
        tokenizer.save_pretrained(p_save_path)
        tokenizer.save_pretrained(q_save_path)

    print("Models and Tokenizers Saved")


# main 함수
if __name__ == "__main__":
    # DenseRetrieverArguments를 불러오고 필요 시 인자 설정

    parser = HfArgumentParser((DataTrainingArguments, RetrieverArguments))

    data_args, retriever_args = parser.parse_args_into_dataclasses()

    # 시드 설정 (재현성을 위해)
    set_seed(42)

    # 학습 함수 실행
    train_embedder(data_args, retriever_args)
