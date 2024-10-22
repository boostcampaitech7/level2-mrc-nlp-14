from collections import Counter
import numpy as np
import json


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
        most_common_answer = Counter(all_texts).most_common(1)[0][0]
        final_predictions[qid] = most_common_answer
    return final_predictions


def soft_voting_ensemble(predictions_list):
    """
    소프트 보팅: 각 답변의 확률을 평균 내어 가장 높은 확률을 가진 답변을 선택합니다.
    """
    final_predictions = {}
    for qid in predictions_list[0].keys():
        all_probabilities = []
        all_texts = []
        for predictions in predictions_list:
            for prediction in predictions[qid]:
                all_texts.append(prediction["text"])
                all_probabilities.append(prediction["probability"])

        # 확률을 평균 내고 가장 높은 확률을 가진 답변을 선택
        best_idx = np.argmax(np.mean(all_probabilities, axis=0))
        final_predictions[qid] = all_texts[best_idx]
    return final_predictions


def save_predictions(predictions, file_name):
    """
    예측 결과를 저장하는 함수.
    """
    with open(file_name, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to {file_name}")


if __name__ == "__main__":
    # 각 모델의 nbest_predictions.json 파일 경로
    nbest_files = [
        "nbest_predictions_model1.json",
        "nbest_predictions_model2.json",
        "nbest_predictions_model3.json",
    ]

    # nbest_predictions 로드
    predictions_list = load_nbest_predictions(nbest_files)

    # 하드 보팅 적용
    hard_voting_preds = hard_voting_ensemble(predictions_list)
    save_predictions(hard_voting_preds, "hard_voting_predictions.json")

    # 소프트 보팅 적용
    soft_voting_preds = soft_voting_ensemble(predictions_list)
    save_predictions(soft_voting_preds, "soft_voting_predictions.json")
