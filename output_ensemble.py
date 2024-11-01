from collections import Counter, defaultdict
import numpy as np
import json

# 앙상블할 모델들의 예측값 파일(nbest_predictions.json)의 경로와 
# 해당 모델들의 train시 평가 점수 (all_result.json)의 경로가 각각 필요
# 또한 경로간 인덱스를 잘 맞춰주세요
# 만약 특정 모델의 nbest_predictions.json의 경로를 nbest_files[0]에 넣으셨다면
# 해당 모델의 평가점수 all_result.json의 경로를 score_files[0]에 넣어야지 정확한 가중치 계산이 됩니다.

def load_nbest_predictions(file_paths):
    """
    각 모델의 nbest_predictions.json 파일을 로드하는 함수.
    """
    predictions_list = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            predictions = json.load(f)
            predictions_list.append(predictions)
    return predictions_list

def load_model_scores(file_paths):
    """
    각 모델의 all_results.json 파일에서 평가 점수를 로드하는 함수.
    """
    scores = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            result = json.load(f)
            scores.append(result["exact_match"])  # exact_match 점수 사용
    return scores


def hard_voting_ensemble(predictions_list):
    """
    하드 보팅: 가장 많이 등장한 답변을 선택합니다.
    """
    final_predictions = {}
    for qid in predictions_list[0].keys():
        all_texts = []
        for predictions in predictions_list:
            # 각 모델의 가장 상위 답변을 모음
            all_texts.append(predictions[qid][0]["text"])
        # 최빈값(가장 많이 등장한 답변) 선택
        most_common_answer = Counter(all_texts).most_common(1)[0][0].strip().strip("\n") # 앞뒤 공백 제거
        final_predictions[qid] = most_common_answer
    return final_predictions


def weighted_soft_voting_ensemble(predictions_list, scores):
    """
    소프트 보팅: 각 답변의 확률에 모델별 exact_match 점수를 가중치로 곱하여 결합한 뒤, 
    가장 높은 가중 확률을 가진 답변을 선택합니다.
    """
    final_predictions = {}

    for qid in predictions_list[0].keys():
        weighted_probabilities = defaultdict(float)

        for i, predictions in enumerate(predictions_list):
            weight = scores[i] / 100.0  # exact_match 점수를 가중치로 사용
            for prediction in predictions[qid]:
                text = prediction["text"]
                probability = prediction["probability"] * weight  # 가중 확률 계산
                weighted_probabilities[text] += probability
                
        # 가장 높은 가중 확률을 가진 답변 선택
        best_answer = max(weighted_probabilities, key=weighted_probabilities.get).strip().strip("\n") # 앞뒤 공백 제거
        final_predictions[qid] = best_answer

    return final_predictions


def save_predictions(predictions, file_name):
    """
    예측 결과를 저장하는 함수.
    """
    with open(file_name, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to {file_name}")


if __name__ == "__main__":
    # 각 모델의 nbest_predictions.json 파일 경로와 EM score 경로
    # 인덱스 순서 주의해주셔야 합니다
    # inference.py의 결과값 중 nbest_predictions의 파일경로를 nbest_files에 넣고
    # score_files는 inference.py 실행할때 경로로 지정했던 모델의 학습시 valid로 나온 결과값이므로 유의하시길 바랍니다.
    nbest_files = [
        "./models/train_dataset/deberta-squad_ep3_batch8/nbest_predictions.json",
        "./models/train_dataset/klue_bert_ep3_batch8/nbest_predictions.json",
        "./models/train_dataset/klue_roberta_finetune_ep3_batch8/nbest_predictions.json"
    ]
    score_files = [
        "./models/train_dataset/deberta-squad_ep3_batch8/all_results.json",
        "./models/train_dataset/klue_bert_ep3_batch8/all_results.json",
        "./models/train_dataset/klue_roberta_finetune_ep3_batch8/all_results.json"
    ]

    # nbest_predictions 로드
    predictions_list = load_nbest_predictions(nbest_files)

    scores = load_model_scores(score_files)

    # 하드 보팅 적용
    hard_voting_preds = hard_voting_ensemble(predictions_list)
    save_predictions(hard_voting_preds, "hard_voting_predictions.json")

    # 소프트 보팅 적용
    soft_voting_preds = weighted_soft_voting_ensemble(predictions_list, scores)
    save_predictions(soft_voting_preds, "soft_voting_predictions.json")
