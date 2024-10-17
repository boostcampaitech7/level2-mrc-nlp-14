import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_from_disk
import os
from transformers import AutoTokenizer, set_seed
from args import RetrieverArguments
from retriever.dense.dense_embedder import BertEmbedder  # BertEmbedder 가져오기
from retriever.dense.dense_retriever_args import (
    DenseRetrieverArguments,
)  # Arguments 가져오기
from transformers import set_seed


# 학습 함수 정의
def train_embedder(args: DenseRetrieverArguments):

    # MRC 데이터셋 불러오기
    train_data = load_from_disk("./data/train_dataset")["train"]

    # question과 context 준비
    questions = [item["question"] for item in train_data]
    contexts = [item["context"] for item in train_data]

    # p_encoder, q_encoder 초기화
    if args.use_siamese:
        encoder = BertEmbedder.from_pretrained(args.p_embedder_name)
        p_encoder = encoder
        q_encoder = encoder  # 동일한 인코더를 사용
    else:
        p_encoder = BertEmbedder.from_pretrained(args.p_embedder_name)
        q_encoder = BertEmbedder.from_pretrained(args.q_embedder_name)

    # 토크나이저 초기화 (p_encoder 이름에 기반하여)
    tokenizer = AutoTokenizer.from_pretrained(args.p_embedder_name)

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
    epochs = 3  # 학습 에포크 수를 args에서 받아옴

    # 학습 루프 시작
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        for i in tqdm(range(0, len(questions), batch_size), desc="Training progress"):
            batch_questions = questions[i : i + batch_size]
            batch_contexts = contexts[i : i + batch_size]

            # 토크나이저로 입력 변환
            q_inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            p_inputs = tokenizer(
                batch_contexts,
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
    p_embedder_path = args.p_embedder_name.replace("/", "_")
    q_embedder_path = args.q_embedder_name.replace("/", "_")
    if args.use_siamese:
        model_save_path = os.path.join(args.model_path, f"{p_embedder_path}_SDE")
        p_encoder.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
    else:
        p_save_path = os.path.join(args.model_path, f"{p_embedder_path}_ADE_passage")
        q_save_path = os.path.join(args.model_path, f"{q_embedder_path}_ADE_query")
        p_encoder.save_pretrained(p_save_path)
        q_encoder.save_pretrained(q_save_path)
        tokenizer.save_pretrained(p_save_path)
        tokenizer.save_pretrained(q_save_path)

    print("Models and Tokenizers Saved")


# main 함수
if __name__ == "__main__":
    # DenseRetrieverArguments를 불러오고 필요 시 인자 설정
    args = DenseRetrieverArguments(retriever_args=RetrieverArguments)

    # 시드 설정 (재현성을 위해)
    set_seed(42)

    # 학습 함수 실행
    train_embedder(args)
